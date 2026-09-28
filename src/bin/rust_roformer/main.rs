//! The `rust-roformer` command line.
//!
//! # Why this binary exists
//!
//! Three of this crate's behaviours are only visible by running it: a window that
//! cannot fit is refused *before* anything is allocated, a long track is streamed
//! so that peak memory is a function of the window rather than of the track, and a
//! run that was interrupted continues instead of starting over. None of those are
//! observable from a `cargo add`, and all three are the reason the crate is worth
//! depending on, so they get a command.
//!
//! # What it deliberately does not have
//!
//! No `--gpu`: the ONNX arm is CPU-bound and the MLX arm already *is* the Apple
//! Silicon path, so a `--gpu` flag would promise a device this binary cannot
//! pick. No conversion: this reads WAV and points at `ffmpeg` for anything else,
//! because a decoder nothing here tests is a defect nobody could triage. No
//! output-format choice, no batch mode, no config file. And no timing table:
//! measurement is `examples/bench.rs`, which prints TSV and pairs runs inside one
//! session, because absolute wall clock across sessions is not an instrument here.

mod args;

use std::io::IsTerminal;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use args::{Args, Engine, Parsed};
use rust_roformer::audio::SAMPLE_RATE;
use rust_roformer::config::{Progress, ResumeMode, SeparationOptions, SeparationReport, StemPaths};
use rust_roformer::engine::SeparationEngine;
use rust_roformer::error::Error;
use rust_roformer::graph::HOP;
use rust_roformer::stream::WavSource;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match args::parse(&argv) {
        Parsed::Help => {
            print!("{}", args::HELP);
            ExitCode::SUCCESS
        }
        Parsed::Version => {
            println!(
                "rust-roformer {} ({})",
                env!("CARGO_PKG_VERSION"),
                built_engines()
            );
            ExitCode::SUCCESS
        }
        Parsed::Bad(why) => {
            eprintln!("rust-roformer: {why}");
            ExitCode::from(2)
        }
        Parsed::Run(a) => match run(a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("rust-roformer: {message}");
                ExitCode::FAILURE
            }
        },
    }
}

/// Which engines are in *this* binary, the one fact a `cargo install` with
/// unusual features cannot be guessed at from the outside.
fn built_engines() -> String {
    let mut v = Vec::new();
    if cfg!(feature = "onnx") {
        v.push("onnx");
    }
    if cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        v.push("mlx");
    }
    if v.is_empty() {
        "no engine: rebuild with --features onnx or --features mlx".to_string()
    } else {
        format!("engines: {}", v.join(", "))
    }
}

fn run(a: Args) -> Result<(), String> {
    // Checked before the model is opened, because parsing a 953 MB graph to find
    // out that the input was misspelled is a bad minute of anyone's day.
    if !a.model.is_file() {
        return Err(format!(
            "no model file at '{}' — this command downloads nothing; see 'Getting a \
             model' in the README, and check the file's sha256 before the first run",
            a.model.display()
        ));
    }
    if !a.input.exists() {
        return Err(format!("no input at '{}'", a.input.display()));
    }
    let mut engine = build(&a)?;
    // Read the grid back off the engine rather than repeating the flag: what gets
    // printed is then a statement about what will actually run, which is the point
    // of a tool whose headline behaviour is refusing windows.
    let window = engine.window_samples().map_err(|e| explain(&e))?;
    let source = WavSource::open(&a.input).map_err(|e| explain(&e))?;
    let frames = source.frames();

    let stems = StemPaths::new(
        a.out_dir.join("vocals.wav"),
        a.out_dir.join("background.wav"),
    );
    if !a.quiet {
        println!(
            "{} · window {} samples ({:.3} s, {} hops) · {} threads",
            engine.name(),
            window,
            window as f64 / f64::from(SAMPLE_RATE),
            window / HOP,
            a.threads
        );
        println!(
            "input {} · {} frames ({:.3} s at 44.1 kHz) · source {} Hz",
            a.input.display(),
            comma(frames as u64),
            frames as f64 / f64::from(SAMPLE_RATE),
            source.source_rate()
        );
    }

    let opts = SeparationOptions {
        intra_threads: a.threads,
        inter_threads: a.threads.clamp(1, 2),
        progress: if a.quiet { None } else { Some(progress()) },
        // The window is handed to the engine twice: once as the graph to build and
        // once as the statement to check against it. A session that ended up on a
        // different grid than asked for is a different job, and its checkpoint must
        // not be described as matching this one.
        window_samples: Some(window),
        resume: if a.fresh {
            ResumeMode::Fresh
        } else {
            ResumeMode::Auto
        },
        ..Default::default()
    };
    let t0 = Instant::now();
    let result = engine.separate(&a.input, &stems, &opts);
    if !a.quiet {
        // Whatever happened, close the line the progress callback was redrawing:
        // an error printed onto an open progress line is unreadable.
        end_progress_line();
    }
    match result {
        Ok(r) => {
            summary(&r, &stems, t0.elapsed().as_secs_f64());
            Ok(())
        }
        Err(e) => Err(explain(&e)),
    }
}

/// Build the engine the flags ask for, if this binary can build it at all.
fn build(a: &Args) -> Result<Box<dyn SeparationEngine>, String> {
    /// The flag's third spelling, resolved before the match: an `Auto` still in
    /// scope there would need an arm for a case that cannot happen.
    enum Chosen {
        Onnx,
        Mlx,
    }
    let want = match a.engine {
        Engine::Onnx => Chosen::Onnx,
        Engine::Mlx => Chosen::Mlx,
        Engine::Auto => {
            if cfg!(feature = "onnx") {
                Chosen::Onnx
            } else {
                Chosen::Mlx
            }
        }
    };
    match want {
        #[cfg(feature = "onnx")]
        Chosen::Onnx => {
            use rust_roformer::engine::onnx::OnnxEngine;
            // A window here is a graph edit done at load time, from the file the
            // user already has: nothing is written to disk.
            let built = match a.window {
                Some(n) => OnnxEngine::with_window(&a.model, n),
                None => OnnxEngine::load(&a.model),
            };
            let engine = built.map_err(|e| explain(&e))?;
            Ok(Box::new(
                engine.with_threads(a.threads, a.threads.clamp(1, 2)),
            ))
        }
        #[cfg(not(feature = "onnx"))]
        Chosen::Onnx => Err(
            "this binary was built without the onnx engine; rebuild with --features onnx \
             (--version lists what it has)"
                .to_string(),
        ),
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        Chosen::Mlx => {
            use rust_roformer::engine::mlx::MlxEngine;
            if a.window.is_some() {
                return Err(
                    "--window applies to the onnx engine only: the MLX arm implements the \
                     architecture and runs its checkpoint's own window"
                        .to_string(),
                );
            }
            let engine = MlxEngine::load(&a.model).map_err(|e| explain(&e))?;
            Ok(Box::new(engine))
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        Chosen::Mlx => Err(
            "the mlx engine is only built on Apple Silicon with --features mlx; this binary \
             has no mlx in it (--version lists what it has)"
                .to_string(),
        ),
    }
}

/// A progress callback that redraws one line on a terminal and prints one line a
/// decile at a time into a pipe, so a CI log of a four-hour track is eleven lines
/// rather than four thousand.
///
/// The engine's own message carries the percentage and the window count, so it is
/// printed as given rather than prefixed with ours: two numbers for one quantity
/// is one too many, and the engine's is the one that knows what it counted.
fn progress() -> Progress {
    let last = Arc::new(AtomicUsize::new(usize::MAX));
    let term = std::io::stdout().is_terminal();
    Arc::new(move |percent: i32, msg: &str| {
        let percent = percent.max(0) as usize;
        let prev = last.swap(percent, Ordering::Relaxed);
        if term {
            if percent == prev {
                return;
            }
            // Padded to a fixed width: a shorter message must not leave the tail of
            // the longer one it replaced on screen.
            print!("\r{:60}", msg);
        } else if percent / 10 != prev / 10 || percent >= 100 {
            println!("{}", msg);
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    })
}

/// End a redrawn progress line, if one is open.
fn end_progress_line() {
    if std::io::stdout().is_terminal() {
        println!();
    }
}

fn summary(r: &SeparationReport, stems: &StemPaths, elapsed_s: f64) {
    let bytes = |p: &Path| match std::fs::metadata(p) {
        Ok(m) => comma(m.len()),
        Err(_) => "?".to_string(),
    };
    println!(
        "wrote {} ({} B) and {} ({} B)",
        stems.vocals.display(),
        bytes(&stems.vocals),
        stems.background.display(),
        bytes(&stems.background)
    );
    println!(
        "  {} frames at {} Hz · {} windows inferred, {} skipped",
        comma(r.frames as u64),
        r.sample_rate,
        r.windows_inferred,
        r.windows_resumed
    );
    if let Some(from) = r.resumed_from_frames.filter(|n| *n > 0) {
        // The line that says a resume was real: "skipped" can legitimately be 0 on
        // a seam that was re-inferred without being rewritten, and that is still
        // work an earlier run did. A zero here means the same thing as `None`
        // (started from sample zero), so it is not printed as a continuation.
        println!(
            "  continued from frame {} — everything before it was already on disk",
            comma(from as u64)
        );
    }
    if let Some(mb) = r.peak_mb {
        println!(
            "  peak {} MB in this process (lifetime high-water, not per window)",
            comma(mb)
        );
    }
    println!("  {:.1} s elapsed here", elapsed_s);
}

/// Turn a crate error into a message that says what to do next.
///
/// The memory case is the one worth the words: on a 16 GB laptop the stock 8 s
/// graph cannot run at all, and a user who does not know that will re-download the
/// model and try again until the copy is worn out. Both engines already answer with
/// numbers, so the only thing added here is the knob to turn.
fn explain(e: &Error) -> String {
    let mut s = format!("{e}");
    let mut source = std::error::Error::source(e);
    while let Some(c) = source {
        s.push_str(&format!(" | caused by: {c}"));
        source = c.source();
    }
    if e.looks_like_allocation_failure() {
        let four = 4 * SAMPLE_RATE as usize;
        s.push_str(&format!(
            "\n  this machine cannot fund that window. A shorter window is a smaller graph, \
             not a smaller model: try --window 4s ({four} samples, {} hops), or run the stock \
             graph on a machine that can hold it.",
            four / HOP
        ));
    } else if matches!(e, Error::Wav { .. } | Error::Resample { .. }) {
        s.push_str(
            "\n  this command reads PCM WAV at 44.1 kHz stereo (other rates are resampled for \
             you); convert anything else with `ffmpeg -i IN -ar 44100 -ac 2 OUT.wav` — see \
             scripts/demo.sh for a worked example.",
        );
    }
    s
}

/// Digit-group a count: `5292000` and `5,292,000` are not the same thing to read
/// at the end of a four-hour job.
fn comma(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_groups_every_three_digits_from_the_right() {
        assert_eq!(comma(0), "0");
        assert_eq!(comma(999), "999");
        assert_eq!(comma(1000), "1,000");
        assert_eq!(comma(5_292_000), "5,292,000");
        assert_eq!(comma(1_000_000_000), "1,000,000,000");
    }

    #[test]
    fn a_memory_refusal_points_at_the_window_rather_than_the_file() {
        let e = Error::Memory {
            need_mb: 19_400,
            avail_mb: Some(14_281),
        };
        let m = explain(&e);
        assert!(
            m.contains("19400") || m.contains("19,400"),
            "numbers lost: {m}"
        );
        assert!(m.contains("--window 4s"), "no knob offered: {m}");
        assert!(
            !m.to_lowercase().contains("re-download"),
            "blames the file: {m}"
        );
    }

    #[test]
    fn an_allocation_failure_from_the_runtime_gets_the_same_hint() {
        // ONNX Runtime's own text, not our variant: the hint has to follow the
        // symptom wherever it comes from, or the worst machine gets no advice.
        let e = Error::Session {
            detail: "failed to allocate a buffer of 19398 MB".to_string(),
        };
        assert!(explain(&e).contains("--window 4s"));
    }

    #[test]
    fn a_wav_refusal_offers_the_conversion_and_keeps_the_engine_detail() {
        let e = Error::Wav {
            path: "song.mp3".into(),
            detail: "not a RIFF file".to_string(),
        };
        let m = explain(&e);
        assert!(m.contains("ffmpeg"), "{m}");
        assert!(m.contains("not a RIFF file"), "detail dropped from: {m}");
    }

    #[test]
    fn the_engines_built_into_this_binary_are_named() {
        let s = built_engines();
        // Whatever features cargo was handed for this test, the string must report
        // something true about it rather than something hardcoded.
        let onnx = cfg!(feature = "onnx");
        let mlx = cfg!(all(
            feature = "mlx",
            target_os = "macos",
            target_arch = "aarch64"
        ));
        if onnx {
            assert!(s.contains("onnx"), "{s}");
        }
        if mlx {
            assert!(s.contains("mlx"), "{s}");
        }
        if !onnx && !mlx {
            assert!(s.contains("no engine"), "{s}");
        }
    }
}
