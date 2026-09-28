//! The engine trait both backends implement.
//!
//! Two engines, one shape of work: read the track without materialising it,
//! run the model window by window, crossfade, and stream the two stems to disk
//! so that a four-hour input costs the same memory as a four-second one.
//!
//! * `onnx`: runs an exported graph through ONNX Runtime. Portable, and the
//!   path you want on a machine without Apple Silicon.
//! * `mlx` (feature `mlx`, Apple Silicon only): a from-scratch Rust
//!   implementation of the Mel-Band RoFormer architecture on MLX tensors. It
//!   loads weights extracted from the exported checkpoint, and its acceptance
//!   bar is a PyTorch FFT reference rather than the ONNX graph's own output.

use std::path::Path;

use crate::config::{SeparationOptions, SeparationReport, StemPaths};
use crate::error::Result;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;

pub trait SeparationEngine {
    /// Short identifier used in logs and in the resume sidecar.
    ///
    /// A method rather than an associated constant: engines are handed around as
    /// `Box<dyn SeparationEngine>`, and a `const` makes the trait not
    /// dyn-compatible.
    fn name(&self) -> &'static str;

    /// Window length, in samples per channel, that this loaded model runs one
    /// forward on. Read from the graph rather than assumed: the same code path
    /// serves a stock export and a reduced-window one.
    fn window_samples(&self) -> Result<usize>;

    /// Sample rate the model expects (44100 for these checkpoints).
    fn sample_rate(&self) -> u32;

    /// Separate `input` into `out`, streaming.
    ///
    /// Every failure leaves the `.part` pair and its sidecar in place, including
    /// [`crate::error::Error::Cancelled`], so a later call with the same paths and
    /// [`crate::config::ResumeMode::Auto`] picks up at the last window that
    /// finished rather than at second zero. Deciding that a partial result is
    /// worthless is not this trait's to make: that is
    /// [`crate::stream::discard_staging`], and it cannot be undone.
    fn separate(
        &mut self,
        input: &Path,
        out: &StemPaths,
        opts: &SeparationOptions,
    ) -> Result<SeparationReport>;
}
