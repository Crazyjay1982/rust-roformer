//! # rust-roformer
//!
//! Mel-Band RoFormer vocal/background separation in Rust.
//!
//! The model itself is someone else's (see the `NOTICE` file); what is ours is
//! everything around it: two inference engines and the I/O layer that lets a
//! four-hour track be separated on a 16 GB laptop without ever holding the
//! track, the intermediate buffers, or the result in memory at the same time.
//!
//! Maintained at [DeepForgeHub](https://deepforgehub.com), extracted from the
//! separation stage of their video-translation application DeepVideo and stood
//! up on its own:
//! nothing in this crate talks to that application or to any network at run time,
//! and no model file arrives except one you downloaded and named yourself.
//!
//! "RoFormer" here is the *Mel-Band RoFormer* separation architecture: not the
//! NLP model the rotary-embedding paper is named after, and not BS-RoFormer —
//! the two group the frequency axis differently, so their weights do not
//! interchange. Projects that already do parts of what is listed below are
//! linked from the README's "Related work" section, and the delta claimed here is
//! deliberately narrow.
//!
//! ## What you get
//!
//! * **Streaming in.** [`audio::WavSource`] decodes a window at a time and
//!   resamples in chunks, so peak memory does not scale with track length. The
//!   resampler runs per chunk with the same filter, and the resulting
//!   last-bit differences at 44.1 kHz inputs are documented in
//!   [`audio::resample_chunked`] rather than hidden.
//! * **Streaming out.** [`stream::StemWriter`] writes both stems as one
//!   interleaved pass, rewrites its own 44-byte WAV header at every window
//!   boundary, and therefore leaves a playable, complete prefix on disk at all
//!   times.
//! * **Resume.** A long job that dies — OOM, power loss, a user hitting stop —
//!   can be continued from the last flushed window instead of starting over,
//!   with the job identity (input bytes and mtime, model bytes and mtime,
//!   window, overlap) checked field by field first. Mismatch means start over,
//!   not "error out with a broken file".
//! * **A memory gate that refuses before it OOMs.** [`mem::snapshot`] reports
//!   what the OS will give you, and the window you asked for costs a measured
//!   per-forward amount; the 8-second graph is ~19 GB per forward on the
//!   machines this crate targets, which is exactly the case that should fail
//!   in microseconds with a number in the message instead of after an hour of
//!   partial work.
//! * **Cancellation that is actually observed.** Both engines check between
//!   windows, so stop costs you at most one window of latency.
//!
//! ## Engines
//!
//! * `onnx` (default): ONNX Runtime, portable CPU. `mlx` (feature `mlx`, Apple
//!   Silicon): a from-scratch Rust implementation of the architecture — its own
//!   STFT, band-split, alternating time/freq transformer, mask estimator —
//!   loading weights extracted from the exported checkpoint.
//!
//! ## From a shell
//!
//! The `rust-roformer` binary exposes exactly the list above — window, threads,
//! checkpoint, cancellation — behind seven options, so the behaviours that are
//! otherwise only visible from Rust (a refusal before allocation, a run that
//! continues where an interrupted one stopped) can be seen without a `main`.
//! `rust-roformer --help` is the reference, and the README's Quick start shows a
//! resumed run verbatim.
//!
//! ## What is not in this repository
//!
//! No model weights and no audio. Weights are licensed separately from this
//! code, and the checkpoints were fitted to recorded music; see
//! [`docs/LICENSES.md`] and the `NOTICE` file
//! for what that does and does not let you do. `tools/` turns a stock export you
//! download yourself into the reduced-window graph and into the weight file the
//! MLX engine loads. `scripts/demo.sh` is the other half of the same rule: it
//! fetches freely licensed audio from where its rights holder published it, so
//! that a listening example exists without any of it being *in* the repository.
//!
//! ## Minimal use
//!
//! ```ignore
//! use std::path::Path;
//! use rust_roformer::engine::{onnx::OnnxEngine, SeparationEngine};
//! use rust_roformer::config::StemPaths;
//!
//! # fn run() -> rust_roformer::error::Result<()> {
//! let mut engine = OnnxEngine::load(Path::new("melband_roformer_vocals.onnx"))?;
//! let out = StemPaths::new("vocals.wav", "background.wav");
//! let report = engine.separate(Path::new("song.wav"), &out, &Default::default())?;
//! println!("{} frames, {} windows, peak {:?} MB",
//!     report.frames, report.windows_inferred, report.peak_mb);
//! # Ok(())
//! # }
//! ```
//!
//! [`docs/LICENSES.md`]: ../docs/LICENSES.md

pub mod config;
pub mod engine;
pub mod error;
pub mod graph;

pub mod audio;
pub mod mem;
pub mod stream;

pub use config::{
    CancelFlag, Progress, ResumeMode, SeparationOptions, SeparationReport, StemPaths,
};
pub use engine::SeparationEngine;
pub use error::{Error, Result};
