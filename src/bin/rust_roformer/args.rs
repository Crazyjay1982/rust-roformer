//! Argument parsing for the `rust-roformer` binary.
//!
//! Separate from `main` because it is the part with the most ways to be wrong and
//! the only part that can be tested without a model file. The surface is
//! deliberately small: this tool exists so that the three behaviours worth seeing
//! in the crate — a window that cannot fit is refused before anything is
//! allocated, a long track is streamed rather than loaded, and an interrupted run
//! continues — can be observed without writing Rust. Anything else belongs in the
//! library API, where it can be reviewed by the code that calls it.

use std::path::PathBuf;

use rust_roformer::audio::SAMPLE_RATE;

/// Which engine to build. `Auto` picks one that is actually compiled in, so a
/// binary built with only `--features mlx` still runs with no flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    #[default]
    Auto,
    Onnx,
    Mlx,
}

/// A command line that made sense.
#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub model: PathBuf,
    pub input: PathBuf,
    pub out_dir: PathBuf,
    pub engine: Engine,
    /// Samples per forward. `None` means "use the window the graph declares".
    pub window: Option<usize>,
    pub threads: usize,
    pub fresh: bool,
    pub quiet: bool,
}

/// What `main` acts on.
#[derive(Debug, PartialEq, Eq)]
pub enum Parsed {
    Run(Args),
    Help,
    Version,
    /// A command line we can explain but not honour. Exit code 2.
    Bad(String),
}

pub const HELP: &str = "\
rust-roformer — Mel-Band RoFormer vocal/background separation, streamed to disk

usage:
  rust-roformer --model PATH --out DIR [OPTIONS] INPUT.wav

  --model PATH      the .onnx export for the onnx engine, or the .safetensors
                    extracted from that same export for the mlx engine. This
                    command downloads nothing: see 'Getting a model' in the
                    README, and check the file's sha256 before the first run.
  -o, --out DIR     where vocals.wav and background.wav go; created if missing.
      --window N    samples per forward, or seconds with an 's' suffix
                    (4s = 176400 = 400 hops of 441). The stock export declares
                    352800 (8 s); asking for a shorter one reshapes the graph at
                    load time and costs proportionally less memory. Only the
                    onnx engine can do that — mlx refuses any window that is
                    not its checkpoint's own.
      --engine E    onnx (portable CPU) or mlx (Apple Silicon, --features mlx).
                    Default: whichever of the two this binary was built with.
  -t, --threads N   backend inference threads (default 4)
      --fresh       ignore a checkpoint left by an earlier run and start over
  -q, --quiet       no progress line; the summary and errors still print
  -h, --help        this text
  -V, --version     version and the engines compiled into this binary

Interrupting a run is not wasting it. Both stems are staged as .part files with
valid headers and the checkpoint is rewritten at every window boundary, so
Ctrl-C costs at most the window in flight and re-running the same command
continues from the last flushed frame. If the checkpoint does not describe this
job — other input bytes, other model bytes, other window — the run starts over
instead of appending to the wrong track.

exit codes:
  0  both stems were written
  1  the run failed; the message says whether the machine was too small for the
     window, the file was not what we can read, or the model is wrong
  2  the command line was wrong (this help has the shape we expect)
";

/// Parse `argv` without the program name.
pub fn parse(argv: &[String]) -> Parsed {
    let mut model: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut engine = Engine::Auto;
    let mut window: Option<usize> = None;
    let mut threads: Option<usize> = None;
    let mut fresh = false;
    let mut quiet = false;
    let mut i = 0;

    while i < argv.len() {
        // A `--flag=value` spelling is folded onto the same path as
        // `--flag value`, so neither one surprises a hand typing from habit.
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (argv[i].clone(), None),
        };
        match flag.as_str() {
            "-h" | "--help" => return Parsed::Help,
            "-V" | "--version" => return Parsed::Version,
            "--model" | "-m" => match value(&flag, &inline, argv, &mut i) {
                Ok(v) if !v.is_empty() => model = Some(PathBuf::from(v)),
                Ok(_) => return Parsed::Bad("--model needs a path".to_string()),
                Err(e) => return Parsed::Bad(e),
            },
            "--out" | "-o" => match value(&flag, &inline, argv, &mut i) {
                Ok(v) if !v.is_empty() => out_dir = Some(PathBuf::from(v)),
                Ok(_) => return Parsed::Bad("--out needs a directory".to_string()),
                Err(e) => return Parsed::Bad(e),
            },
            "--window" => match value(&flag, &inline, argv, &mut i) {
                Ok(v) => match window_from(&v) {
                    Ok(n) => window = Some(n),
                    Err(e) => return Parsed::Bad(format!("--window: {e}")),
                },
                Err(e) => return Parsed::Bad(e),
            },
            "--engine" => match value(&flag, &inline, argv, &mut i) {
                Ok(v) => match v.as_str() {
                    "onnx" => engine = Engine::Onnx,
                    "mlx" => engine = Engine::Mlx,
                    other => {
                        return Parsed::Bad(format!("--engine: '{other}' is neither onnx nor mlx"))
                    }
                },
                Err(e) => return Parsed::Bad(e),
            },
            "--threads" | "-t" => match value(&flag, &inline, argv, &mut i) {
                Ok(v) => match v.parse::<usize>() {
                    Ok(0) => return Parsed::Bad("--threads must be at least 1".to_string()),
                    Ok(n) => threads = Some(n),
                    Err(_) => return Parsed::Bad(format!("--threads: not a number: '{v}'")),
                },
                Err(e) => return Parsed::Bad(e),
            },
            "--fresh" => {
                if inline.is_some() {
                    return Parsed::Bad("--fresh takes no value".to_string());
                }
                i += 1;
                fresh = true;
            }
            "--quiet" | "-q" => {
                if inline.is_some() {
                    return Parsed::Bad("--quiet takes no value".to_string());
                }
                i += 1;
                quiet = true;
            }
            other => {
                if other.starts_with('-') && other != "-" {
                    return Parsed::Bad(format!("unknown option '{other}' (try --help)"));
                }
                if input.is_some() {
                    return Parsed::Bad(format!(
                        "two inputs given; one WAV is expected ('{other}' is the second)"
                    ));
                }
                input = Some(PathBuf::from(other));
                i += 1;
            }
        }
    }

    let mut missing = Vec::new();
    if model.is_none() {
        missing.push("--model PATH");
    }
    if out_dir.is_none() {
        missing.push("--out DIR");
    }
    if input.is_none() {
        missing.push("INPUT.wav");
    }
    if !missing.is_empty() {
        return Parsed::Bad(format!(
            "missing {}: run --help for the shape",
            missing.join(", ")
        ));
    }
    Parsed::Run(Args {
        model: model.unwrap(),
        input: input.unwrap(),
        out_dir: out_dir.unwrap(),
        engine,
        window,
        threads: threads.unwrap_or(4),
        fresh,
        quiet,
    })
}

/// Read a flag's value from `--flag=v` or from the next argument, advancing `i`
/// past whatever was consumed.
///
/// A flag whose next token starts with `-` is treated as a missing value rather
/// than as "the user meant the flag to be empty": `--window --fresh` must not
/// quietly run the stock 8 s graph and report success. The exception is the
/// positional input, where a bare `-` is a filename a pipe habit produces and is
/// accepted by the caller below.
fn value(
    flag: &str,
    inline: &Option<String>,
    argv: &[String],
    i: &mut usize,
) -> Result<String, String> {
    if let Some(v) = inline {
        *i += 1;
        return Ok(v.clone());
    }
    match argv.get(*i + 1) {
        Some(next) if next != "-" && !next.starts_with('-') => {
            *i += 2;
            Ok(next.clone())
        }
        _ => Err(format!("{flag} needs a value")),
    }
}

/// `--window` accepts a sample count or a duration with an `s` suffix.
///
/// The suffix form exists because every memory figure in this crate is per
/// forward, and what a caller actually knows is how long a window they can
/// afford. One decimal place is accepted because at 44.1 kHz `0.1 s` is exactly
/// ten hops, so a rounded duration cannot fall off the hop grid for arithmetic
/// reasons — whether a window fits the graph stays the engine's judgement, not
/// something this parser duplicates.
pub fn window_from(text: &str) -> Result<usize, String> {
    let text = text.trim();
    let samples = if let Some(sec) = text.strip_suffix('s').or_else(|| text.strip_suffix('S')) {
        // Plain decimal only. Rust's `f64::from_str` also accepts `1e9`, `inf` and
        // `NaN`, and a window of 4.4e13 samples is not a duration anyone meant —
        // it fails somewhere far less legible than the flag that asked for it.
        if sec.is_empty()
            || sec == "."
            || !sec.bytes().all(|b| b.is_ascii_digit() || b == b'.')
            || sec.matches('.').count() > 1
        {
            return Err(format!("'{text}' is not a duration like 4s or 3.5s"));
        }
        let seconds: f64 = sec
            .parse()
            .map_err(|_| format!("'{text}' is not a duration like 4s or 3.5s"))?;
        if !(seconds.is_finite() && seconds > 0.0) {
            return Err(format!("'{text}' is not a positive number of seconds"));
        }
        (seconds * f64::from(SAMPLE_RATE)).round() as usize
    } else {
        text.parse::<usize>()
            .map_err(|_| format!("'{text}' is not a sample count or a duration like 4s"))?
    };
    if samples == 0 {
        return Err("a window of 0 samples is not a window".to_string());
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    /// The shortest command line that runs, plus whatever the test is about.
    fn run(args: &[&str]) -> Args {
        match parse(&v(args)) {
            Parsed::Run(a) => a,
            other => panic!("expected a runnable command line, got {other:?}"),
        }
    }

    #[test]
    fn the_minimum_command_line_fills_the_rest_with_defaults() {
        let a = run(&["--model", "m.onnx", "--out", "o", "song.wav"]);
        assert_eq!(a.model, PathBuf::from("m.onnx"));
        assert_eq!(a.input, PathBuf::from("song.wav"));
        assert_eq!(a.out_dir, PathBuf::from("o"));
        assert_eq!(a.engine, Engine::Auto);
        assert_eq!(a.window, None, "no --window means the graph's own");
        assert_eq!(a.threads, 4);
        assert!(!a.fresh);
        assert!(!a.quiet);
    }

    #[test]
    fn flag_order_is_free_and_the_input_can_come_first() {
        let a = run(&["song.wav", "--out", "o", "--model", "m.onnx", "--quiet"]);
        assert_eq!(a.input, PathBuf::from("song.wav"));
        assert!(a.quiet);
    }

    #[test]
    fn equals_and_space_spellings_agree() {
        assert_eq!(
            run(&["--model=m.onnx", "--out=o", "--window=176400", "song.wav"]),
            run(&["--model", "m.onnx", "--out", "o", "--window", "176400", "song.wav"])
        );
    }

    #[test]
    fn a_missing_required_part_names_itself() {
        for (args, wants) in [
            (v(&["--out", "o", "song.wav"]), "--model PATH"),
            (v(&["--model", "m.onnx", "song.wav"]), "--out DIR"),
            (v(&["--model", "m.onnx", "--out", "o"]), "INPUT.wav"),
        ] {
            match parse(&args) {
                Parsed::Bad(m) => assert!(m.contains(wants), "{m} does not mention {wants}"),
                other => panic!("expected Bad for {args:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_flag_losing_its_value_says_so_rather_than_defaulting() {
        // `--window --fresh` is the dangerous typo: honouring it would run the
        // stock 8 s graph, refuse on a small machine, and read as a false claim
        // about the memory gate. The assertion is on "needs a value" specifically —
        // the parse failure below would also be an error, so merely expecting a
        // non-zero exit would let a mutation that swallows the flag through.
        for args in [
            v(&["--model", "m", "--out", "o", "--window", "--fresh", "s.wav"]),
            v(&["--model", "m", "--out", "o", "--window"]),
        ] {
            match parse(&args) {
                Parsed::Bad(m) => assert_eq!(m, "--window needs a value", "wrong diagnosis: {m}"),
                other => panic!("expected Bad, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_unknown_option_is_an_error_but_a_dash_file_is_input() {
        match parse(&v(&["--gpu", "song.wav"])) {
            Parsed::Bad(m) => assert!(m.contains("unknown option '--gpu'"), "{m}"),
            other => panic!("expected Bad, got {other:?}"),
        }
        let a = run(&["--model", "m", "--out", "o", "-"]);
        assert_eq!(a.input, PathBuf::from("-"));
    }

    #[test]
    fn two_positional_inputs_are_refused_not_merged() {
        match parse(&v(&["--model", "m", "--out", "o", "a.wav", "b.wav"])) {
            Parsed::Bad(m) => assert!(m.contains("two inputs"), "{m}"),
            other => panic!("expected Bad, got {other:?}"),
        }
    }

    #[test]
    fn seconds_and_samples_reach_the_same_window() {
        assert_eq!(window_from("4s"), Ok(176_400));
        assert_eq!(window_from("4S"), Ok(176_400));
        assert_eq!(window_from("176400"), Ok(176_400));
        assert_eq!(window_from("8s"), Ok(352_800));
        assert_eq!(window_from(" 2.5s "), Ok(110_250));
        assert_eq!(window_from("4.0s"), Ok(176_400));
        for bad in [
            "0", "0s", "-4", "s", "4 x", "abc", "1e9s", "inf s", "infs", "nans", "1.2.3s", ".s",
        ] {
            assert!(window_from(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn one_decimal_place_never_leaves_the_hop_grid() {
        // The claim in window_from's doc comment, checked rather than asserted in
        // prose: 0.1 s is ten hops, so tenths land on multiples of 441.
        for tenths in 1..=80u32 {
            let text = format!("{}.{tenths}s", tenths / 10);
            let n = window_from(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(n % 441, 0, "{text} = {n} samples is off the hop grid");
        }
    }

    #[test]
    fn fresh_and_engine_switches_are_independent_of_position() {
        let a = run(&[
            "--fresh", "song.wav", "--engine", "mlx", "--model", "m", "-o", "o",
        ]);
        assert!(a.fresh);
        assert_eq!(a.engine, Engine::Mlx);
        match parse(&v(&[
            "--engine", "cuda", "--model", "m", "-o", "o", "s.wav",
        ])) {
            Parsed::Bad(m) => assert!(m.contains("neither onnx nor mlx"), "{m}"),
            other => panic!("expected Bad, got {other:?}"),
        }
    }

    #[test]
    fn valueless_flags_reject_the_equals_spelling() {
        match parse(&v(&["--fresh=yes", "--model", "m", "-o", "o", "s.wav"])) {
            Parsed::Bad(m) => assert!(m.contains("--fresh takes no value"), "{m}"),
            other => panic!("expected Bad, got {other:?}"),
        }
    }

    #[test]
    fn help_and_version_win_over_an_incomplete_line() {
        // Someone reaching for --help has not finished typing; making them satisfy
        // the required flags first teaches them to pipe --help away.
        assert_eq!(parse(&v(&["--help"])), Parsed::Help);
        assert_eq!(parse(&v(&["--model", "m", "-h"])), Parsed::Help);
        assert_eq!(parse(&v(&["-V"])), Parsed::Version);
    }

    #[test]
    fn threads_zero_is_refused_because_it_would_stall_the_run() {
        match parse(&v(&["--threads", "0", "--model", "m", "-o", "o", "s.wav"])) {
            Parsed::Bad(m) => assert!(m.contains("at least 1"), "{m}"),
            other => panic!("expected Bad, got {other:?}"),
        }
    }

    #[test]
    fn help_names_every_option_the_parser_takes() {
        // The parser and the text drift apart first and loudest. Enumerating them
        // here is cheaper than a user finding out by being ignored.
        let a = run(&[
            "--model",
            "m",
            "--out",
            "o",
            "--window",
            "4s",
            "--engine",
            "onnx",
            "--threads",
            "8",
            "--fresh",
            "--quiet",
            "s.wav",
        ]);
        assert_eq!(a.threads, 8);
        for flag in [
            "--model",
            "--out",
            "--window",
            "--engine",
            "--threads",
            "--fresh",
            "--quiet",
            "--help",
            "--version",
        ] {
            assert!(HELP.contains(flag), "HELP never mentions {flag}");
        }
        assert!(HELP.contains("INPUT.wav"), "HELP never shows the input");
    }
}
