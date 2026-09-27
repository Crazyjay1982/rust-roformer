//! Measurement harness for this crate.
//!
//!     cargo run --release --example bench -- synth --seconds 90 --out track.wav
//!     cargo run --release --example bench -- run --model m.onnx --input track.wav --repeat 3
//!     cargo run --release --example bench -- run --model m.onnx --input track.wav --window 176400
//!     cargo run --release --example bench -- resume-check --model m.onnx --input track.wav
//!
//! `synth` writes a deterministic two-"speaker" test track, so you can measure
//! the I/O path without downloading or shipping any audio. Size the workload
//! with `--seconds`; there is no window cap here on purpose, because a truncated
//! run is a different job than the one you are trying to time.
//!
//! `resume-check` exercises the claim this crate is really making: that a
//! cancelled run, continued, yields a file **byte-identical** to an
//! uninterrupted one. It fails loudly, with the offset of the first difference,
//! when that is not true. Its middle row is the interrupted pass read back off
//! the disk — a `Cancelled` return carries no report, so the surviving prefix,
//! its sidecar and the resume plan are what say how far it got.
//!
//! Nothing here is a benchmark by itself. Two sessions of this program are not
//! comparable on any machine we have measured — run each configuration both
//! first and second within one session, which is what `scripts/bench_pair.sh`
//! does for you.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rust_roformer::config::{CancelFlag, SeparationOptions, SeparationReport, StemPaths};
use rust_roformer::engine::SeparationEngine;
use rust_roformer::stream::{
    chunk_starts, discard_staging, job_path, part_path, resume_plan, StemJob, WavSource,
};
use rust_roformer::{mem, Error};

#[cfg(feature = "onnx")]
use rust_roformer::engine::onnx::OnnxEngine;

const SR: u32 = 44_100;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    let rest: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "synth" => cmd_synth(&rest),
        "run" => cmd_run(&rest),
        "resume-check" => cmd_resume_check(&rest),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(None)
        }
        other => Err(format!("unknown command '{other}' (try: bench help)")),
    };
    match result {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(code)) => ExitCode::from(code),
        Err(e) => {
            eprintln!("bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n  \
         bench synth       --seconds N --out FILE\n  \
         bench run         --model PATH --input PATH [--engine onnx|mlx]\n    \
         [--threads T] [--repeat K] [--window N] [--out DIR] [--label NAME]\n  \
         bench resume-check --model PATH --input PATH [--engine onnx|mlx]\n    \
         [--cancel-at PCT] [--window N] [--out DIR]\n\n\
         `run` prints TSV, one row per repetition:\n  \
         label  seconds  windows  wall_s  rtf  peak_mb  vocals_bytes  window_samples\n  \
         resumed  resumed_from\n\n  \
         window_samples is read back off the session, not taken from the command\n  \
         line: the row states the grid it timed. resumed_from is the frames already\n  \
         flushed when that call started, so it is the column that says whether a\n  \
         resume really had something to resume; resumed counts the windows skipped\n  \
         outright, which a one-window seam replay can legitimately leave at 0.\n\n\
         --window N reshapes the ONNX graph to an N-sample window at load time,\n  \
         in memory, from the stock export — no reduced-window file is written. N\n  \
         must be a whole number of hops (441) below the window the export already\n  \
         declares: shrinking is exact, growing is refused because the iSTFT\n  \
         normalisation table would have to be regenerated. The MLX engine refuses\n  \
         any window that is not the checkpoint's own 352800.\n\n\
         About peak_mb: on macOS it is the process's whole-lifetime high-water\n  \
         mark of phys_footprint, so a light arm measured after a heavy one in the\n  \
         same process cannot report lower. That is the instrument, not the code —\n  \
         which is why `run` takes --repeat and the shell wrapper takes the pair."
    );
}

// ---------------------------------------------------------------- arg parsing

#[derive(Default)]
struct Args {
    kv: Vec<(String, String)>,
}

fn parse(args: &[String]) -> Args {
    let mut out = Args::default();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].clone();
        if !key.starts_with("--") {
            i += 1;
            continue;
        }
        match args.get(i + 1) {
            Some(v) if !v.starts_with("--") => {
                out.kv.push((key, v.clone()));
                i += 2;
            }
            _ => {
                out.kv.push((key, String::new()));
                i += 1;
            }
        }
    }
    out
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.kv
            .iter()
            .find(|(a, _)| a == k)
            .map(|(_, b)| b.as_str())
            .filter(|s| !s.is_empty())
    }
    fn need(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("missing required {k}"))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
}

/// `--window`, as a request rather than a statement: `None` means "whatever the
/// graph declares", `Some(n)` means "reshape it to n at load time". A value that
/// does not parse is the user's typo, not a default to fall back on, so it is an
/// error here instead of a silent 8 s run.
fn window_arg(a: &Args) -> Result<Option<usize>, String> {
    match a.get("--window") {
        None => Ok(None),
        Some(s) => s
            .parse::<usize>()
            .map(Some)
            .map_err(|_| format!("--window wants a whole number of samples, got '{s}'")),
    }
}

// ------------------------------------------------------------------- synth

/// Two alternating tones panned apart over light noise, on a 1.7 s cycle: not a
/// voice, but the shape the loop cares about — energy moving between channels,
/// hard onsets, never silent. Fully deterministic, so two runs of `run` on the
/// same file see the same work.
fn cmd_synth(args: &[String]) -> Result<Option<u8>, String> {
    let a = parse(args);
    let seconds: f64 = a.num("--seconds", 90.0);
    if !(0.1..=86_400.0).contains(&seconds) {
        return Err(format!("--seconds out of range: {seconds}"));
    }
    let path = PathBuf::from(a.need("--out")?);
    let frames = (seconds * SR as f64) as usize;

    let file = fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = hound::WavWriter::new(
        BufWriter::new(file),
        hound::WavSpec {
            channels: 2,
            sample_rate: SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(|e| e.to_string())?;

    let two_pi = 2.0 * std::f64::consts::PI;
    for n in 0..frames {
        let t = n as f64 / SR as f64;
        let who = (t / 1.7) as usize % 2;
        let cycle = (t % 1.7) / 1.7;
        let env = (std::f64::consts::PI * cycle).sin();
        let tone = (two_pi * if who == 0 { 196.0 } else { 980.0 } * t).sin() * env;
        let harm = (two_pi * if who == 0 { 392.0 } else { 1960.0 } * t).sin() * env * 0.35;
        // Deterministic hash-noise: enough to keep every window non-silent.
        let h = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let noise = ((h >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.02;
        let (l, r) = if who == 0 {
            (tone * 0.72 + noise, harm * 0.10)
        } else {
            (harm * 0.10, tone * 0.72 + noise)
        };
        for s in [l, r] {
            w.write_sample((s * i16::MAX as f64) as i16)
                .map_err(|e| e.to_string())?;
        }
    }
    w.finalize().map_err(|e| e.to_string())?;
    eprintln!(
        "bench: wrote {} frames ({:.1} s stereo 44.1 kHz) to {}",
        frames,
        seconds,
        path.display()
    );
    Ok(None)
}

// -------------------------------------------------------------------- run

struct Arm {
    label: String,
    seconds: f64,
    windows: usize,
    wall_s: f64,
    peak_mb: Option<u64>,
    vocals_bytes: u64,
    window_samples: usize,
    /// Windows this call neither inferred nor wrote. Can be 0 on a run that still
    /// reused a checkpoint — one window re-inferred as seam priming is the normal
    /// case — which is why the row also carries [`Arm::resumed_frames`].
    resumed: usize,
    /// Frames that were already flushed when this call started. The direct answer
    /// to "did a resume have anything to resume".
    resumed_frames: usize,
}

impl std::fmt::Display for Arm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rtf = if self.seconds > 0.0 {
            self.wall_s / self.seconds
        } else {
            0.0
        };
        write!(
            f,
            "{}\t{:.2}\t{}\t{:.2}\t{:.3}\t{}\t{}\t{}\t{}\t{}",
            self.label,
            self.seconds,
            self.windows,
            self.wall_s,
            rtf,
            self.peak_mb
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            self.vocals_bytes,
            self.window_samples,
            self.resumed,
            self.resumed_frames
        )
    }
}

const TSV_HEADER: &str = "label\tseconds\twindows\twall_s\trtf\tpeak_mb\tvocals_bytes\t\
                          window_samples\tresumed\tresumed_from";

/// One configured way of putting the engine over a track: which arm, with what
/// geometry, whether to stop partway, and whether to wipe what a previous pass
/// left on disk first.
///
/// A bundle rather than six positional parameters because the three passes of
/// `resume-check` differ only in one field at a time — spreading the arguments
/// across the call sites is how a "continued" pass silently ends up on a different
/// window than the control it is being byte-compared against.
#[derive(Clone, Copy)]
struct Pass<'a> {
    kind: &'a str,
    model: &'a Path,
    threads: usize,
    window: Option<usize>,
    cancel_at_pct: Option<i32>,
    /// `true` = clear the output paths (published pair, staging pair, sidecar)
    /// before running. `false` = leave whatever is there, which is the only way to
    /// ask the engine to resume — and the thing `resume-check` used to get wrong by
    /// wiping the checkpoint it was about to continue from.
    fresh: bool,
}

fn cmd_run(args: &[String]) -> Result<Option<u8>, String> {
    let a = parse(args);
    let model = PathBuf::from(a.need("--model")?);
    let input = PathBuf::from(a.need("--input")?);
    let kind = a.get("--engine").unwrap_or("onnx").to_string();
    let repeat: usize = a.num("--repeat", 1).max(1);
    let threads: usize = a.num("--threads", 4).max(1);
    let window = window_arg(&a)?;
    let base = out_dir(a.get("--out"))?;
    let name = a.get("--label").unwrap_or("run").to_string();

    let pass = Pass {
        kind: &kind,
        model: &model,
        threads,
        window,
        cancel_at_pct: None,
        // Every repetition is its own job: a second arm that resumed the first
        // would time a partly-finished track and report it as a full one.
        fresh: true,
    };
    println!("{TSV_HEADER}");
    for i in 0..repeat {
        let label = format!("{name}#{i}");
        let arm = run_once(&label, &input, &base, pass)?;
        println!("{arm}");
    }
    Ok(None)
}

/// One pass over `input`, writing `<dir>/<label>/{vocals,background}.wav`.
///
/// The cancellation lands on the next window boundary, so where a pass stops is
/// a window, not a percentage. The requested `window` goes to the engine twice —
/// as the shape to build and as the statement to check against it — so a run
/// fails if the session ends up on a different grid than asked for rather than
/// quietly timing a different job.
fn run_once(label: &str, input: &Path, dir: &Path, pass: Pass) -> Result<Arm, String> {
    let mut engine = build(pass.kind, pass.model, pass.window)?;
    // Read the grid back off the engine rather than echoing `--window`: the row is
    // evidence of what actually ran, which is the one thing a timing table must not
    // take from the command line.
    let window_samples = engine
        .window_samples()
        .map_err(|e| format!("{e}{}", cause_chain(&e)))?;
    let out = dir.join(label);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let stems = StemPaths::new(out.join("vocals.wav"), out.join("background.wav"));
    if pass.fresh {
        // The whole checkpoint goes, not just the pair: a sidecar left behind
        // without the `.part` files it describes is how a later pass starts from
        // zero while reporting that it resumed.
        discard_staging(&stems.vocals, &stems.background);
        for p in [&stems.vocals, &stems.background] {
            let _ = fs::remove_file(p);
        }
    }

    let mut opts = SeparationOptions {
        intra_threads: pass.threads,
        inter_threads: pass.threads.clamp(1, 2),
        window_samples: pass.window,
        ..Default::default()
    };
    if let Some(pct) = pass.cancel_at_pct {
        let flag = CancelFlag::new();
        let hit = Arc::new(AtomicBool::new(false));
        let (f2, h2) = (flag.clone(), hit);
        opts.progress = Some(Arc::new(move |p: i32, _msg: &str| {
            // Stop at the first progress report past the mark; the engine only
            // looks at the flag between windows, so this lands on a boundary.
            if p >= pct && !h2.swap(true, Ordering::Relaxed) {
                f2.cancel();
            }
        }));
        opts.cancel = Some(flag);
    }

    let t0 = Instant::now();
    let report: SeparationReport = match engine.separate(input, &stems, &opts) {
        Ok(r) => r,
        // A pass that was *told* to stop and stopped is not a harness failure.
        // `Cancelled` carries no report, so the row is rebuilt from what the
        // engine kept on disk — which is the whole subject of `resume-check`.
        Err(Error::Cancelled) if pass.cancel_at_pct.is_some() => {
            return cancelled_arm(label, &stems, window_samples, t0.elapsed().as_secs_f64());
        }
        Err(e) => return Err(format!("{e}{}", cause_chain(&e))),
    };
    let wall_s = t0.elapsed().as_secs_f64();
    let vocals_bytes = fs::metadata(&stems.vocals).map(|m| m.len()).unwrap_or(0);

    Ok(Arm {
        label: label.to_string(),
        seconds: report.frames as f64 / report.sample_rate as f64,
        windows: report.windows_inferred,
        wall_s,
        peak_mb: report.peak_mb,
        vocals_bytes,
        window_samples,
        resumed: report.windows_resumed,
        resumed_frames: report.resumed_from_frames.unwrap_or(0),
    })
}

/// The row for a pass that ended in [`Error::Cancelled`].
///
/// A cancelled run keeps its `.part` pair and sidecar, so "how far did it get" is
/// a fact about the disk rather than a guess: the prefix length comes from the
/// staged WAV's own header, the geometry from the sidecar, and the window count
/// from the resume plan the *next* call will run — the windows it will not have to
/// infer again.
fn cancelled_arm(
    label: &str,
    stems: &StemPaths,
    window_samples: usize,
    wall_s: f64,
) -> Result<Arm, String> {
    let staged = {
        let part = part_path(&stems.vocals);
        if part.exists() {
            part
        } else {
            stems.vocals.clone()
        }
    };
    let (frames, rate) = {
        let src = WavSource::open(&staged).map_err(|e| format!("{}: {e}", staged.display()))?;
        (src.frames(), src.source_rate() as f64)
    };
    let sidecar = job_path(&stems.vocals);
    let text = fs::read_to_string(&sidecar).map_err(|e| {
        format!(
            "{}: {e} — a cancelled run is meant to leave its checkpoint behind, and \
             without it nothing about resuming can be proved",
            sidecar.display()
        )
    })?;
    let job = StemJob::decode(&text).map_err(|why| format!("unreadable resume sidecar: {why}"))?;
    let (total, win, overlap) = (
        job.total_frames as usize,
        job.window as usize,
        job.overlap as usize,
    );
    let starts = chunk_starts(total, win, overlap);
    let plan = resume_plan(&starts, total, win, frames);
    Ok(Arm {
        label: label.to_string(),
        seconds: frames as f64 / rate,
        // Committed windows, not inferences performed: the priming re-runs belong
        // to the continuing pass, and that is the row below this one.
        windows: plan.first,
        wall_s,
        peak_mb: mem::snapshot().and_then(|s| s.proc_peak_rss_mb),
        vocals_bytes: fs::metadata(&staged).map(|m| m.len()).unwrap_or(0),
        window_samples,
        // It wrote the checkpoint rather than reading one.
        resumed: 0,
        resumed_frames: 0,
    })
}

fn out_dir(explicit: Option<&str>) -> Result<PathBuf, String> {
    Ok(match explicit {
        Some(p) => PathBuf::from(p),
        None => std::env::temp_dir().join(format!("rust-roformer-bench-{}", std::process::id())),
    })
}
/// Build the ONNX arm, reshaped if `window` names something other than what the
/// export declares.
///
/// A refused reshape is fatal here rather than a quiet fallback to the native
/// graph: timing an 8 s window because `--window 176400` did not take would print
/// a row that looks like the answer you asked for.
#[cfg(feature = "onnx")]
fn onnx_engine(model: &Path, window: Option<usize>) -> Result<OnnxEngine, String> {
    let built = match window {
        Some(n) => OnnxEngine::with_window(model, n),
        None => OnnxEngine::load(model),
    };
    built.map_err(|e| format!("{e}{}", cause_chain(&e)))
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn build(
    kind: &str,
    model: &Path,
    window: Option<usize>,
) -> Result<Box<dyn SeparationEngine>, String> {
    use rust_roformer::engine::mlx::MlxEngine;
    match kind {
        #[cfg(feature = "onnx")]
        "onnx" => Ok(Box::new(onnx_engine(model, window)?)),
        "mlx" => {
            if window.is_some() {
                return Err(
                    "--window applies to the onnx engine only: the MLX arm implements the \
                     architecture and runs the checkpoint's native window"
                        .to_string(),
                );
            }
            Ok(Box::new(MlxEngine::load(model).map_err(|e| e.to_string())?))
        }
        other => Err(unknown_engine(other)),
    }
}

#[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn build(
    kind: &str,
    model: &Path,
    window: Option<usize>,
) -> Result<Box<dyn SeparationEngine>, String> {
    match kind {
        #[cfg(feature = "onnx")]
        "onnx" => Ok(Box::new(onnx_engine(model, window)?)),
        "mlx" => Err(
            "the `mlx` engine is only built on macOS/Apple Silicon with \
                      --features mlx"
                .to_string(),
        ),
        other => Err(unknown_engine(other)),
    }
}

fn unknown_engine(kind: &str) -> String {
    format!("unknown engine '{kind}': rebuild with --features onnx and/or mlx")
}

fn cause_chain(e: &dyn std::error::Error) -> String {
    let mut s = String::new();
    let mut cur = e.source();
    while let Some(c) = cur {
        s.push_str(" | caused by: ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}

// ------------------------------------------------------------ resume-check

fn cmd_resume_check(args: &[String]) -> Result<Option<u8>, String> {
    let a = parse(args);
    let model = PathBuf::from(a.need("--model")?);
    let input = PathBuf::from(a.need("--input")?);
    let kind = a.get("--engine").unwrap_or("onnx").to_string();
    let cancel_at = a.num("--cancel-at", 40i32);
    let window = window_arg(&a)?;
    let base = out_dir(a.get("--out"))?;

    let pass = Pass {
        kind: &kind,
        model: &model,
        threads: 4,
        window,
        cancel_at_pct: None,
        fresh: true,
    };
    // Control: uninterrupted.
    let ctrl = run_once("control", &input, &base, pass)?;
    // Arm: same output paths, but this pass stops partway…
    let cut = run_once(
        "cut",
        &input,
        &base,
        Pass {
            cancel_at_pct: Some(cancel_at),
            ..pass
        },
    )?;
    // …and this one continues from what survived. `fresh: false` is the whole
    // difference between a resume and a second full run.
    let cont = run_once(
        "cut",
        &input,
        &base,
        Pass {
            fresh: false,
            ..pass
        },
    )?;

    let cv = fs::read(base.join("control/vocals.wav")).map_err(|e| e.to_string())?;
    let cb = fs::read(base.join("control/background.wav")).map_err(|e| e.to_string())?;
    let rv = fs::read(base.join("cut/vocals.wav")).map_err(|e| e.to_string())?;
    let rb = fs::read(base.join("cut/background.wav")).map_err(|e| e.to_string())?;

    println!("{TSV_HEADER}");
    println!("{ctrl}\n{cut}\n{cont}");

    let mut ok = true;
    if cut.windows == 0 {
        ok = false;
        eprintln!(
            "FAIL: the cancelled pass committed 0 windows, so there was nothing to \
             resume from — lower --cancel-at"
        );
    }
    if cont.resumed_frames == 0 {
        ok = false;
        eprintln!(
            "FAIL: the continuing pass started from frame 0 — it re-inferred the whole \
             track, so whatever this compared was a full run against a full run. Either \
             the checkpoint was deleted, or it disagreed with this input, model or \
             geometry"
        );
    }
    for (name, expected, got) in [("vocals", &cv, &rv), ("background", &cb, &rb)] {
        match first_difference(expected, got) {
            Diff::Same => println!("{name}: identical ({} B)", expected.len()),
            Diff::At(i) => {
                ok = false;
                eprintln!(
                    "FAIL {name}: first difference at byte {i} of {} B",
                    expected.len()
                );
            }
            Diff::Len(a, b) => {
                ok = false;
                eprintln!("FAIL {name}: length differs — control {a} B, resumed {b} B");
            }
        }
    }
    println!(
        "resume-check: {}",
        if ok {
            "OK (resumed output byte-identical to uninterrupted)"
        } else {
            "FAILED"
        }
    );
    Ok(Some(if ok { 0 } else { 1 }))
}

enum Diff {
    Same,
    At(usize),
    Len(usize, usize),
}

fn first_difference(a: &[u8], b: &[u8]) -> Diff {
    if a.len() != b.len() {
        return Diff::Len(a.len(), b.len());
    }
    match (0..a.len()).find(|i| a[*i] != b[*i]) {
        Some(i) => Diff::At(i),
        None => Diff::Same,
    }
}
