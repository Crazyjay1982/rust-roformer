//! Caller-facing knobs shared by both engines.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Progress reporting: `(percent 0..=100, message)`.
///
/// Called from the inference thread. Implementations must not block — the
/// separation loop checks cancellation and flushes state between windows and
/// treats a slow callback as wall-clock cost, not as a bug.
pub type Progress = Arc<dyn Fn(i32, &str) + Send + Sync>;

/// Cooperative cancellation.
///
/// Checked **between windows**, never inside a forward pass: both engines drive
/// a runtime (ONNX Runtime / MLX) whose decode loop we do not own. A 4 s window
/// is the finest granularity you can observe, which is the point of keeping the
/// window small.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// What to do when a previous run left staging output behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResumeMode {
    /// Continue from the last complete window boundary already on disk, if the
    /// recorded job identity still matches. Otherwise start over.
    #[default]
    Auto,
    /// Always start over (truncating any staging files).
    Fresh,
}

/// Where the two stems go.
///
/// `background` is the model's second source, i.e. `mix − vocals` for the
/// vocals-only checkpoints: the residual comes out of the graph, it is not
/// computed by subtracting float buffers after the fact.
#[derive(Debug, Clone)]
pub struct StemPaths {
    pub vocals: PathBuf,
    pub background: PathBuf,
}

impl StemPaths {
    pub fn new(vocals: impl Into<PathBuf>, background: impl Into<PathBuf>) -> Self {
        Self {
            vocals: vocals.into(),
            background: background.into(),
        }
    }
}

/// Tuning for one `separate()` call.
#[derive(Clone)]
pub struct SeparationOptions {
    /// Intra-op threads for the inference backend.
    pub intra_threads: usize,
    /// Inter-op threads. 2 is what the long-audio measurements were taken at;
    /// the loop is sequential per window, so raising this buys little.
    pub inter_threads: usize,
    pub progress: Option<Progress>,
    pub cancel: Option<CancelFlag>,
    pub resume: ResumeMode,
    /// `None` = consult the OS for an available-memory figure and refuse to
    /// start a window that clearly does not fit. `Some(mb)` = use that budget
    /// instead (what an embedding host already knows about its own limits).
    pub memory_budget_mb: Option<u64>,
    /// Override the model's own window length in samples.
    ///
    /// `None` (the default) reads it from the loaded graph, so a stock export
    /// runs at the window it was exported for. Setting this is how you ask for
    /// a reduced-window graph produced by `tools/reduce_window.py` — the value
    /// must agree with the file on disk, because the window is baked into the
    /// graph as constants, not taken from the input shape.
    pub window_samples: Option<usize>,
    /// Overlap between windows, for the linear crossfade. `None` = the default
    /// for this crate (1.5 s at 44.1 kHz), which is what the flush proofs are
    /// stated against. Must be ≤ window/2.
    pub overlap_samples: Option<usize>,
}

impl std::fmt::Debug for SeparationOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeparationOptions")
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field("progress", &self.progress.as_ref().map(|_| "<closure>"))
            .field("cancel", &self.cancel)
            .field("resume", &self.resume)
            .field("memory_budget_mb", &self.memory_budget_mb)
            .field("window_samples", &self.window_samples)
            .field("overlap_samples", &self.overlap_samples)
            .finish()
    }
}

impl Default for SeparationOptions {
    fn default() -> Self {
        Self {
            intra_threads: 4,
            inter_threads: 2,
            progress: None,
            cancel: None,
            resume: ResumeMode::Auto,
            memory_budget_mb: None,
            window_samples: None,
            overlap_samples: None,
        }
    }
}

impl SeparationOptions {
    pub fn with_progress(mut self, p: Progress) -> Self {
        self.progress = Some(p);
        self
    }

    pub fn with_cancel(mut self, c: CancelFlag) -> Self {
        self.cancel = Some(c);
        self
    }

    /// Set a hard per-forward budget instead of asking the OS.
    pub fn with_memory_budget_mb(mut self, mb: u64) -> Self {
        self.memory_budget_mb = Some(mb);
        self
    }
}

/// What one finished `separate()` call reports.
#[derive(Debug, Clone, Default)]
pub struct SeparationReport {
    /// Sample rate of the written WAVs (44100; the model is fixed at it).
    pub sample_rate: u32,
    /// Frames per channel in each output file.
    pub frames: usize,
    /// Windows actually inferred this call (excludes the ones resumed from disk).
    pub windows_inferred: usize,
    /// Windows skipped because they were already complete on disk.
    pub windows_resumed: usize,
    /// Frames that were already flushed when this call started, if any.
    pub resumed_from_frames: Option<usize>,
    /// Peak resident footprint observed during this call, where the platform
    /// can report it. See `mem` for what this number is on each OS.
    pub peak_mb: Option<u64>,
    /// Wall-clock milliseconds spent inside `separate()`.
    pub wall_ms: u128,
}
