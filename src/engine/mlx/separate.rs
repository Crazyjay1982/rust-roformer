//! Streaming separation on MLX: one native window in, both stems out, a
//! checkpoint at every window boundary.
//!
//! [`MlxEngine::separate`] is the whole arm: decode one 8 s window of the mix
//! with [`WavSource`], run the model on it, crossfade its contribution into a
//! bounded pending region, append the samples that can no longer change to the
//! `.part` pair, and patch that pair's headers. Nothing in the loop knows how
//! long the track is except the schedule.
//!
//! ## The window is the checkpoint's, and it stays that way
//!
//! `WIN = 352_800` samples (8 s at 44.1 kHz) with `OVERLAP = 110_250` (2.5 s) and
//! a linear crossfade. This engine implements the architecture, so it runs the
//! window the model was trained on with no graph surgery: shortening it is not a
//! constant to edit but a change to the time-axis attention length the weights
//! were fitted for, which makes it a quality question needing re-validation. A
//! caller asking for another window through
//! [`SeparationOptions::window_samples`](crate::config::SeparationOptions::window_samples)
//! gets [`resolve_window`]'s refusal, which names the figure it wants and points
//! at the ONNX arm, where a rebuilt graph is the right answer.
//!
//! The overlap *is* a crossfade parameter rather than a model parameter — it
//! changes where the seams are, not what the network sees — so
//! [`resolve_overlap`] honours a caller's value as long as it keeps the grid
//! legal (`overlap ≤ win / 2`, the invariant [`chunk_starts`] asserts: otherwise
//! a sample can be touched by three or more windows and a resumed run cannot
//! reproduce the weights an uninterrupted one applied).
//!
//! ## What one window costs
//!
//! Five `WIN`-long `f32` accumulators (both stems, both channels, plus the weight
//! sum), the window in hand and its output: about 13 MB of host buffers, which is
//! the whole host-side footprint whatever the track length. The pre-streaming
//! version of this path held the mix, both stems and the weight sum at full track
//! length — ~4.4 GB of host memory per hour of audio — and shipped a measured
//! 3,405 MB host peak on a 45.8-minute track down to 72 MB, with the
//! *device*-side figures unchanged value for value. Those are the shipped app's
//! numbers, taken before this port. This crate does not re-measure them; every
//! run reports its own [`SeparationReport::peak_mb`] from
//! [`mem::peak_mb_while`](crate::mem::peak_mb_while) instead of quoting them.
//!
//! The device side is why [`mem::window_fits`] is called before the first
//! forward: the tensor library's allocator cache dominates, and the largest
//! transient in the graph is the time-axis attention's score matrix,
//! `bands × heads × frames × frames` ≈ 1.2 GB per copy at this window.
//! [`per_forward_mb`] is an analytic bound from those shapes, not a measurement,
//! and it is the figure the gate refuses against.
//!
//! ## Resume, cancellation, failures
//!
//! The `.part` pair is a checkpoint: [`StemWriter::checkpoint`] runs at every
//! window boundary, so a process that dies loses the window it was in and not the
//! track. A restart with [`ResumeMode::Auto`](crate::config::ResumeMode::Auto)
//! re-derives the pending crossfade by re-inferring the one or two windows that
//! still reach past the seam ([`resume_plan`](crate::stream::resume_plan)) and
//! then writes the bytes an uninterrupted run would have written. The pair's
//! identity is [`StemJob::for_run`], binding the input file, the **weight file**
//! and the window geometry, so a re-exported input or a different checkpoint
//! starts over instead of splicing two separations into one stem.
//!
//! Cancellation is observed between windows — never inside a forward, because the
//! runtime's own decode loop is not ours to interrupt. It keeps the staging, like
//! every other failure does: [`SeparationEngine::separate`]'s contract is that the
//! next call with the same paths continues from the last flushed window, and the
//! caller who pressed stop is the one most likely to want that. Deleting a
//! checkpoint is irreversible, so it stays the caller's explicit
//! [`discard_staging`](crate::stream::discard_staging) rather than an engine
//! policy — the production arm this was ported from *does* throw the pair away on
//! stop, because in that product "stop" means "I am not coming back to this one";
//! a library cannot know which of the two a caller meant.
//!
//! ## What is not ported
//!
//! The shipped arm's wall-clock hang guard is not here: this crate's contract has
//! no duration-scaled timeout, and a caller that wants one has
//! [`CancelFlag`](crate::config::CancelFlag) and a thread to set it from.

use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::Array;
use ndarray::Array2;

use crate::audio::SAMPLE_RATE;
use crate::config::{ResumeMode, SeparationOptions, SeparationReport, StemPaths};
use crate::engine::SeparationEngine;
use crate::error::{Error, Result};
use crate::mem;
use crate::stream::{chunk_starts, resume_plan, StemJob, StemWriter, WavSource};

use super::model::{RoFormerModel, DIM, DIM_INNER, HEADS, NUM_BANDS};
use super::stft::HOP;
use super::weights::load_weights;

/// Window length in samples: the checkpoint's native 8 s at 44.1 kHz.
pub const WIN: usize = 352_800;
/// Overlap between consecutive windows: 2.5 s, crossfaded linearly.
pub const OVERLAP: usize = 110_250;
/// STFT frames in one window: `WIN / hop + 1`, the sequence length the time-axis
/// attention runs on.
pub const FRAMES: usize = WIN / HOP as usize + 1;

/// The engine's name, as it appears in logs and in the resume sidecar.
pub const ENGINE_NAME: &str = "mlx";

// ─────────────────────── window and grid resolution ───────────────────────

/// Resolve a caller's window request against the native window.
///
/// `None` and `Some(WIN)` both mean "the checkpoint's own 8 s". Anything else is
/// refused rather than honoured: this engine builds the architecture, so the
/// window is the length the weights were trained at, and shortening it changes
/// the time-axis attention rather than just the buffer size. That is a quality
/// change that needs re-validation, not a constant to overwrite — and the ONNX
/// arm, which runs an exported graph, is where a reduced-window graph belongs.
pub fn resolve_window(requested: Option<usize>) -> Result<usize> {
    match requested {
        None => Ok(WIN),
        Some(w) if w == WIN => Ok(WIN),
        Some(w) => Err(Error::Model {
            detail: format!(
                "the MLX engine runs the checkpoint's native {WIN}-sample window (8 s at \
                 {SAMPLE_RATE} Hz) and cannot be given another: it implements the \
                 architecture directly, so {w} samples would change the time-axis attention \
                 length the weights were trained on, which is a quality change rather than a \
                 buffer resize. Leave SeparationOptions::window_samples unset for the native \
                 window, or use the ONNX engine with a graph rebuilt for that length."
            ),
        }),
    }
}

/// Resolve a caller's overlap request.
///
/// `None` gives the shipped 2.5 s. Any value is honoured while it keeps the grid
/// legal — `overlap ≤ WIN / 2`, i.e. one hop still covers the fade region, so a
/// sample is touched by at most two windows and a resumed run reproduces the
/// weights of an uninterrupted one.
pub fn resolve_overlap(requested: Option<usize>) -> Result<usize> {
    let overlap = requested.unwrap_or(OVERLAP);
    if overlap > WIN / 2 {
        return Err(Error::Model {
            detail: format!(
                "overlap of {overlap} samples exceeds half the {WIN}-sample window: the hop \
                 would be shorter than the fade region, a sample could be touched by three or \
                 more windows, and a resumed run could not reproduce the crossfade weights of \
                 an uninterrupted one"
            ),
        });
    }
    Ok(overlap)
}

/// Linear ramp over the fade region: `ramp[i] = i / (overlap - 1)`, so 0 → 1.
pub fn linear_ramp(overlap: usize) -> Vec<f32> {
    let denom = overlap.saturating_sub(1).max(1) as f32;
    (0..overlap).map(|i| i as f32 / denom).collect()
}

/// Per-sample crossfade weight for one window: 1 everywhere except a linear
/// fade-in over its first `overlap` samples (skipped for the first window) and a
/// linear fade-out over its last `overlap` (skipped for the last window).
///
/// The two ramps are the same ramp read in opposite directions, so where two
/// windows meet their weights sum to exactly 1 and the weight normalisation in
/// the flush step is a no-op there — it exists for the samples only one window
/// reaches.
pub fn fade_weights(ramp: &[f32], is_first: bool, is_last: bool) -> Vec<f32> {
    let overlap = ramp.len();
    assert!(
        overlap * 2 <= WIN,
        "the fade region has to fit the window twice"
    );
    let mut w = vec![1.0f32; WIN];
    if !is_first {
        w[..overlap].copy_from_slice(ramp);
    }
    if !is_last {
        for k in 0..overlap {
            w[WIN - 1 - k] = ramp[k];
        }
    }
    w
}

// ─────────────────────────── per-forward sizing ───────────────────────────

const fn mib(bytes: usize) -> u64 {
    (bytes / (1024 * 1024)) as u64
}

/// One copy of the time-axis attention's score matrix, in bytes.
///
/// The batch is the 60 bands, so this is the largest tensor the graph
/// materialises and the one the memory gate is really about.
pub const ATTENTION_SCORE_BYTES: usize = NUM_BANDS * HEADS * FRAMES * FRAMES * 4;

/// Additional memory one forward pass needs, in MiB: the analytic bound the
/// pre-flight gate compares with [`mem::window_fits`].
///
/// Summed from the graph's own shapes — the score matrix and its softmax copy,
/// the q/k/v projection, the block activations and the host buffers this loop
/// holds. It is *not* a measurement: the tensor library's allocator cache holds
/// freed regions, so the device-side peak sits above this figure rather than at
/// it. The window cannot be configured, so this is a constant, and what it
/// refuses is a machine that cannot hold the native window at all.
pub const fn per_forward_mb() -> u64 {
    let scores = 2 * mib(ATTENTION_SCORE_BYTES);
    let qkv = mib(NUM_BANDS * FRAMES * 3 * DIM_INNER * 4);
    let activations = mib(4 * FRAMES * NUM_BANDS * DIM * 4);
    let host = mib(7 * WIN * 4);
    scores + qkv + activations + host
}

// ──────────────────────────── the engine ────────────────────────────

/// Mel-Band RoFormer on MLX: the architecture, the checkpoint's weights, the
/// native 8 s window.
///
/// Load once with [`MlxEngine::load`], then call
/// [`separate`](SeparationEngine::separate) per track: the weights stay resident
/// and each track costs one window at a time.
pub struct MlxEngine {
    model: RoFormerModel,
    weights: PathBuf,
}

impl MlxEngine {
    /// Map the weight file and load all 672 tensors.
    ///
    /// `weights` is a `.safetensors` file you produced with
    /// `tools/extract_onnx_weights.py` from the stock fp32 export. This crate has
    /// no model registry and no default location for it — a host already knows
    /// where it keeps its models, and guessing a path here would only turn a
    /// missing file into a confusing one.
    pub fn load(weights: &Path) -> Result<Self> {
        if !weights.is_file() {
            return Err(Error::Model {
                detail: format!("MLX weight file not found: {}", weights.display()),
            });
        }
        let t0 = Instant::now();
        let mut model = RoFormerModel::new();
        load_weights(&mut model, weights)?;
        log::info!(
            "[mlx] weights loaded from {} in {:.1}s",
            weights.display(),
            t0.elapsed().as_secs_f64()
        );
        Ok(MlxEngine {
            model,
            weights: weights.to_path_buf(),
        })
    }

    /// The weight file this engine's resume identity is bound to.
    pub fn weights_path(&self) -> &Path {
        &self.weights
    }

    /// One window through the model: `(2, WIN)` f32 in, `(2, WIN)` vocals out.
    ///
    /// The input is always a full window; a caller with less audio left zero-pads
    /// the tail and keeps its own samples, which is what
    /// [`separate`](SeparationEngine::separate) does at the end of a track. The
    /// background stem is the residual `mix − vocals` taken from this same window,
    /// because the checkpoint is single-stem.
    pub fn separate_window(&self, chunk: &Array2<f32>) -> Result<Array2<f32>> {
        let samples = chunk.ncols();
        let data = chunk.as_slice().ok_or_else(|| Error::Output {
            detail: "window is not contiguous".to_string(),
        })?;
        let input = Array::from_slice(data, &[1, 2, samples as i32]);

        // The graph reports failures by panicking on the runtime's error status —
        // that is what `expect` on every op means. Unwinding into `Error::Session`
        // keeps the two things a caller needs: its staging, and a message
        // `mem::is_allocation_failure` can classify as "smaller machine" rather
        // than "broken file".
        let out = panic::catch_unwind(AssertUnwindSafe(|| self.model.forward(&input))).map_err(
            |payload| {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown failure inside the model".to_string());
                let summary = mem::snapshot()
                    .map(|s| s.summary())
                    .unwrap_or_else(|| "unavailable".to_string());
                Error::Session {
                    detail: format!("{msg} [at failure: {summary}]"),
                }
            },
        )?;

        // MLX buffers are not always row-major — an FFT on a non-last axis in
        // particular is not — and the host read below indexes raw memory in
        // C-order, so flatten to a contiguous copy first.
        let out = super::ensure_contiguous(&out);
        let flat = out.as_slice::<f32>();
        if flat.len() < 2 * samples {
            return Err(Error::Output {
                detail: format!(
                    "model returned {} samples, expected {}",
                    flat.len(),
                    2 * samples
                ),
            });
        }
        let mut result = Array2::<f32>::zeros((2, samples));
        result
            .row_mut(0)
            .assign(&ndarray::ArrayView1::from(&flat[0..samples]));
        result
            .row_mut(1)
            .assign(&ndarray::ArrayView1::from(&flat[samples..2 * samples]));
        Ok(result)
    }

    /// The loop: schedule, resume, one window at a time, checkpoints, publish.
    fn separate_inner(
        &self,
        input: &Path,
        out: &StemPaths,
        opts: &SeparationOptions,
        win: usize,
        overlap: usize,
    ) -> Result<SeparationReport> {
        debug_assert_eq!(
            win, WIN,
            "the native window is the only one this engine accepts"
        );
        // Thread counts are the ONNX Runtime arm's knobs. MLX dispatches to Metal
        // with no intra-op pool for us to size, so they are ignored out loud
        // rather than silently half-applied.
        log::debug!(
            "[mlx] intra_threads={} / inter_threads={} are not used by this engine",
            opts.intra_threads,
            opts.inter_threads
        );

        // Header only: the mix is decoded one window at a time below.
        let mut source = WavSource::open(input)?;
        let total_len = source.frames();
        let starts = chunk_starts(total_len, win, overlap);
        let n_chunks = starts.len();
        let hop = win - overlap;
        log::info!(
            "[mlx] {} samples ({:.1} s) in {n_chunks} window(s) of {win}, hop {hop}, overlap \
             {overlap}, {FRAMES} STFT frames per window",
            total_len,
            total_len as f64 / SAMPLE_RATE as f64,
        );

        // Resume identity: this run's geometry and both files that define it.
        let job = StemJob::for_run(total_len, &starts, win, overlap, input, &self.weights);
        let mut stems = match opts.resume {
            ResumeMode::Auto => StemWriter::open_or_resume(&out.vocals, &out.background, &job)?,
            ResumeMode::Fresh => StemWriter::create(&out.vocals, &out.background)?,
        };
        let plan = resume_plan(&starts, total_len, win, stems.frames());
        stems.rewind_to(plan.keep_frames)?;
        // Windows below `prime_from` are skipped outright; the ones between there
        // and `first` are re-inferred for the pending crossfade only.
        let windows_resumed = plan.prime_from;
        if plan.keep_frames > 0 {
            log::info!(
                "[mlx] resuming from the checkpoint: {:.1} of {:.1} s separated, re-deriving \
                 window(s) {}..{} and continuing at {}/{}",
                plan.keep_frames as f64 / SAMPLE_RATE as f64,
                total_len as f64 / SAMPLE_RATE as f64,
                plan.prime_from,
                plan.first,
                plan.first,
                n_chunks
            );
        }

        let ramp = linear_ramp(overlap);

        // Pending, not-yet-finalized region [flushed_upto, pend_end). Contributions
        // from processed windows land here; flushed samples shift out of the
        // buffers. Both stems keep their own accumulator rather than being rebuilt
        // as `mix − vocals` at flush time, so a resumed run adds the same terms in
        // the same order as an uninterrupted one.
        let mut flushed_upto = plan.keep_frames;
        let mut pend_end = plan.keep_frames;
        let mut pv0 = vec![0.0f32; win];
        let mut pv1 = vec![0.0f32; win];
        let mut pb0 = vec![0.0f32; win];
        let mut pb1 = vec![0.0f32; win];
        let mut pw = vec![0.0f32; win];

        // The window being inferred; also the `mix` the residual is taken from, so
        // it stays alive until the window is merged.
        let mut chunk = Array2::<f32>::zeros((2, win));
        let mut windows_inferred = 0usize;

        let (result, peak_mb) = mem::peak_mb_while(|| -> Result<()> {
            for i in 0..n_chunks {
                // The window(s) just below the seam are re-inferred to rebuild the
                // pending crossfade: they contribute to the buffers and finalize
                // nothing.
                let priming = i < plan.first;
                if i < plan.prime_from {
                    continue;
                }
                if opts.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                    return Err(Error::Cancelled);
                }
                let start = starts[i];
                if let Some(cb) = &opts.progress {
                    // Price the samples actually on disk, not the window index: on
                    // a resumed run the first report starts where the work stands.
                    let done = stems.frames().min(total_len) as f64 / total_len.max(1) as f64;
                    cb((100.0 * done) as i32, "separating (MLX)");
                }

                let take = (total_len - start).min(win);
                chunk.fill(0.0);
                {
                    let mut window = chunk.slice_mut(ndarray::s![.., 0..take]);
                    source.read(start, &mut window)?;
                }
                if i == 0 || i % 25 == 0 {
                    mem::log_point(i, &format!("mlx: window {i}/{n_chunks} decoded"));
                }

                let voc = self.separate_window(&chunk).map_err(|e| match e {
                    Error::Session { detail } => Error::Session {
                        detail: format!("at window {i}/{n_chunks}: {detail}"),
                    },
                    other => other,
                })?;
                windows_inferred += 1;

                // The fade is a property of the *schedule*, not of whether this run
                // inferred the window: a priming window must contribute with exactly
                // the weight an uninterrupted run gave it.
                let w = fade_weights(&ramp, i == 0, i == n_chunks - 1);

                // Merge the weighted window into the pending region. On an
                // uninterrupted run the flush boundary is the next window's start,
                // so `acc_from == start` and this is the same accumulation, in the
                // same per-sample order, as the whole-buffer version it has to match
                // bit for bit. Only a priming window takes the other branch: it
                // starts *behind* the flush point, and the part of it already on disk
                // must not be accumulated twice.
                debug_assert!(
                    priming || start >= flushed_upto,
                    "window start behind the flush point"
                );
                if start > pend_end {
                    // A forward gap is only reachable by a schedule change; fill it
                    // with zeros so its weight stays 0, i.e. silence.
                    let off = pend_end - flushed_upto;
                    let len = start - pend_end;
                    for buf in [&mut pv0, &mut pv1, &mut pb0, &mut pb1, &mut pw] {
                        buf[off..off + len].fill(0.0);
                    }
                }
                pend_end = pend_end.max(start + take);
                let acc_from = start.max(flushed_upto);
                let off = acc_from - flushed_upto;
                let len = start + take - acc_from;
                let k0 = acc_from - start;
                for k in 0..len {
                    let wv = w[k0 + k];
                    let v0 = voc[[0, k0 + k]];
                    let v1 = voc[[1, k0 + k]];
                    pv0[off + k] += v0 * wv;
                    pv1[off + k] += v1 * wv;
                    // Background = this window's mix minus its vocals, weighted the
                    // same way; the zero-padded tail contributes nothing.
                    pb0[off + k] += (chunk[[0, k0 + k]] - v0) * wv;
                    pb1[off + k] += (chunk[[1, k0 + k]] - v1) * wv;
                    pw[off + k] += wv;
                }
                if priming {
                    // The seam's earlier contribution is now in the buffers; the
                    // samples it overlaps are still on disk, unmodified.
                    continue;
                }

                // Flush every sample strictly before the next window's start: those
                // can no longer receive contributions.
                let flush_end = if i < n_chunks - 1 {
                    starts[i + 1].min(total_len)
                } else {
                    total_len
                };
                for x in flushed_upto..flush_end {
                    let j = x - flushed_upto;
                    let wk = pw[j];
                    let frame = if wk > 1e-8 {
                        [[pv0[j] / wk, pv1[j] / wk], [pb0[j] / wk, pb1[j] / wk]]
                    } else {
                        // Covered by no window: silence in both stems.
                        [[0.0, 0.0], [0.0, 0.0]]
                    };
                    stems.write_frame(frame[0], frame[1])?;
                }
                // Shift the residual [flush_end, pend_end) to the front.
                let keep = pend_end - flush_end;
                let shift = flush_end - flushed_upto;
                for buf in [&mut pv0, &mut pv1, &mut pb0, &mut pb1, &mut pw] {
                    buf.copy_within(shift..shift + keep, 0);
                    for slot in buf[keep..keep + shift].iter_mut() {
                        *slot = 0.0;
                    }
                }
                flushed_upto = flush_end;
                pend_end = flush_end + keep;

                // Everything up to the flush point is on disk, so patch both headers
                // to that prefix *before* spending another window of inference: a
                // process killed mid-window loses the window, not the track.
                debug_assert_eq!(
                    stems.frames(),
                    flushed_upto,
                    "written frames drifted from the flush point"
                );
                stems.checkpoint()?;

                #[cfg(test)]
                if CRASH_AFTER.with(|c| c.get()) == i {
                    return Err(Error::Session {
                        detail: format!("test crash after window {i}"),
                    });
                }
            }
            Ok(())
        });
        result?;

        if let Some(cb) = &opts.progress {
            cb(98, "publishing stems");
        }
        if stems.frames() != total_len {
            return Err(Error::Output {
                detail: format!(
                    "flush incomplete: wrote {}/{} frames",
                    stems.frames(),
                    total_len
                ),
            });
        }
        // Publishes the `.part` pair, so `vocals.wav` — the file a caller's
        // skip-if-exists check gates on — only appears once both stems are whole.
        stems.finish()?;
        log::info!(
            "[mlx] done: {} + {} ({:.1} s audio, {} window(s) inferred, {} resumed)",
            out.vocals.display(),
            out.background.display(),
            total_len as f64 / SAMPLE_RATE as f64,
            windows_inferred,
            windows_resumed,
        );

        Ok(SeparationReport {
            sample_rate: SAMPLE_RATE,
            frames: total_len,
            windows_inferred,
            windows_resumed,
            resumed_from_frames: (plan.keep_frames > 0).then_some(plan.keep_frames),
            peak_mb: Some(peak_mb),
            // The caller of `separate` owns the whole-call clock and overwrites
            // this; nothing here can see the time spent loading weights.
            wall_ms: 0,
        })
    }
}

impl SeparationEngine for MlxEngine {
    fn name(&self) -> &'static str {
        ENGINE_NAME
    }

    fn window_samples(&self) -> Result<usize> {
        Ok(WIN)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn separate(
        &mut self,
        input: &Path,
        out: &StemPaths,
        opts: &SeparationOptions,
    ) -> Result<SeparationReport> {
        let t0 = Instant::now();
        let win = resolve_window(opts.window_samples)?;
        let overlap = resolve_overlap(opts.overlap_samples)?;

        // A pre-cancelled run must not pay for the memory gate, must not map
        // anything, and must not create a single file.
        if opts.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
            return Err(Error::Cancelled);
        }
        // Refuse before the first forward rather than being killed mid-track.
        mem::window_fits(per_forward_mb(), opts.memory_budget_mb)?;

        let report = self.separate_inner(input, out, opts, win, overlap);
        // And every failure, cancellation included, leaves the `.part` pair and its
        // sidecar in place: the trait's contract is that the next call with the same
        // paths continues rather than starting at zero, and a user who hit "stop" is
        // the caller most likely to want that. Throwing a checkpoint away is the
        // caller's own, reversible-until-asked act —
        // [`discard_staging`](crate::stream::discard_staging) — not something this
        // engine does on their behalf.
        report.map(|mut r| {
            r.wall_ms = t0.elapsed().as_millis();
            r
        })
    }
}

// Window index after which the loop gives up as if the process had been killed,
// checkpoint left in place — the only way to get a half-written track here without
// actually exhausting memory. These are plain comments rather than doc comments
// because `thread_local!` expands to a module, so a doc comment would not attach.
#[cfg(test)]
thread_local! {
    static CRASH_AFTER: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CancelFlag;
    use crate::engine::mlx::weights::{expected_keys, load_safetensors, EXPECTED_TENSOR_COUNT};
    use crate::stream::{job_path, part_path, spec16, STEM_HEADER_BYTES};
    use hound::WavWriter;
    use std::path::PathBuf;

    // ─────────────────────── pure: window policy ───────────────────────

    /// The headline claim of this arm: original weights, original shape. The
    /// knob exists in the contract, and the honest answer to a request that
    /// disagrees with the checkpoint is a refusal that says why.
    #[test]
    fn the_window_must_be_the_checkpoints_native_window() {
        assert_eq!(resolve_window(None).unwrap(), WIN);
        assert_eq!(resolve_window(Some(WIN)).unwrap(), WIN);
        assert_eq!(WIN, 352_800, "8 s at 44.1 kHz");
        assert_eq!(WIN, 8 * 44_100);

        for req in [176_400usize, 4_096, WIN + 1, WIN * 2] {
            let err = resolve_window(Some(req)).expect_err(
                "{req}: a shorter window must not \
                be honoured by an engine that implements the architecture",
            );
            let detail = err.to_string();
            assert!(matches!(err, Error::Model { .. }), "{err}");
            assert!(detail.contains("native"), "{detail}");
            assert!(detail.contains("352800"), "{detail}");
            assert!(
                detail.contains("time-axis attention"),
                "the refusal must name what changes: {detail}"
            );
            // And it must offer the way out rather than just saying no.
            assert!(detail.contains("ONNX"), "{detail}");
        }
    }

    /// The overlap is a crossfade parameter, so a caller may change it — but not
    /// past the point where a sample gets three contributors, which is what makes
    /// a resumed stem differ from an uninterrupted one.
    #[test]
    fn the_overlap_must_keep_the_grid_legal() {
        assert_eq!(resolve_overlap(None).unwrap(), OVERLAP);
        assert_eq!(resolve_overlap(Some(0)).unwrap(), 0, "no overlap is legal");
        assert_eq!(resolve_overlap(Some(WIN / 2)).unwrap(), WIN / 2);
        let err = resolve_overlap(Some(WIN / 2 + 1)).expect_err("half + 1 must be refused");
        assert!(err.to_string().contains("three or more"), "{err}");
        // The shipped grid satisfies the invariant the schedule asserts.
        assert!(WIN - OVERLAP >= OVERLAP);
        assert!(2 * (WIN - OVERLAP) >= WIN, "at most two windows per sample");
    }

    /// The window/hop/frame bindings everything else is stated against, and the
    /// fact that the native window divides exactly by the STFT hop.
    #[test]
    fn the_grid_constants_are_the_architectures_own() {
        assert_eq!(HOP, 441);
        assert_eq!(WIN % HOP as usize, 0);
        assert_eq!(FRAMES, 801);
        assert_eq!(OVERLAP, 110_250, "2.5 s");
        assert_eq!(WIN - OVERLAP, 242_550, "hop = 5.5 s");
        // The re-read factor this grid implies: 352800 / 242550 ≈ 1.45.
        let ratio = WIN as f64 / (WIN - OVERLAP) as f64;
        assert!((ratio - 1.4545).abs() < 1e-3, "{ratio}");

        // Coverage, on the shipped grid, for the shapes an engine can meet.
        for total in [
            0usize,
            1,
            WIN / 3,
            WIN,
            WIN + 1,
            2 * WIN,
            850_000,
            44_100 * 45 * 60,
        ] {
            let starts = chunk_starts(total, WIN, OVERLAP);
            assert!(!starts.is_empty());
            assert_eq!(starts[0], 0);
            for w in starts.windows(2) {
                assert!(w[1] > w[0] && w[1] - w[0] <= WIN, "{starts:?}");
            }
            let last = *starts.last().unwrap();
            assert!(last + (total - last).min(WIN) >= total, "tail uncovered");
        }
    }

    /// Crossfade weights: 0 → 1 in, 1 → 0 out, and where two windows meet their
    /// weights sum to exactly one — which is what makes the flush step's
    /// normalisation a no-op inside a seam.
    #[test]
    fn the_crossfade_weights_add_up_to_one() {
        let ramp = linear_ramp(OVERLAP);
        assert_eq!(ramp.len(), OVERLAP);
        assert_eq!(ramp[0], 0.0);
        assert_eq!(*ramp.last().unwrap(), 1.0);
        // Monotonic, and linear by construction.
        for k in 1..OVERLAP {
            assert!(ramp[k] > ramp[k - 1]);
            let want = k as f32 / (OVERLAP - 1) as f32;
            assert!((ramp[k] - want).abs() < 1e-7, "{k}");
        }

        let interior = fade_weights(&ramp, false, false);
        assert_eq!(interior.len(), WIN);
        assert_eq!(interior[0], 0.0);
        assert_eq!(interior[OVERLAP], 1.0);
        assert_eq!(interior[WIN - OVERLAP - 1], 1.0);
        assert_eq!(interior[WIN - 1], 0.0);
        assert!(interior[OVERLAP..=WIN - OVERLAP - 1]
            .iter()
            .all(|&v| v == 1.0));

        // The seam itself. Window `i` covers `[s, s+WIN)` and window `i+1` covers
        // `[s+hop, s+hop+WIN)`, so they meet over `hop .. WIN` in window `i`'s
        // coordinates and over `0 .. overlap` in window `i+1`'s. At every position
        // in there the two weights have to sum to exactly one — that is what makes
        // the flush step's division by the weight sum a no-op across a seam, and
        // what a resumed run has to reproduce sample for sample.
        let hop = WIN - OVERLAP;
        // Any two interior windows carry the same weights, so one array serves for
        // both sides of the seam — in window `i`'s coordinates and in window
        // `i+1`'s, which are offset by exactly `hop`.
        let weights = fade_weights(&ramp, false, false);
        for j in 0..OVERLAP {
            let out_w = weights[hop + j];
            let in_w = weights[j];
            assert!(
                (out_w + in_w - 1.0).abs() < 1e-6,
                "seam position {j}: {out_w} + {in_w} != 1"
            );
            // Out of curiosity, in the other direction: the fade-out is the same
            // ramp read backwards.
            assert!(
                (out_w - ramp[OVERLAP - 1 - j]).abs() < 1e-7,
                "fade-out at {j}"
            );
        }
        // Beyond the seam region a window stands alone at weight 1, so no sample is
        // ever touched by three windows: `2 · hop ≥ WIN` above is what guarantees
        // it, and this is the same fact read off the weights.
        assert!(weights[OVERLAP..hop].iter().all(|&v| v == 1.0));

        // Ends: the first window does not fade in and the last does not fade out, so the
        // track's first and last samples are the model's own, undimmed.
        let first = fade_weights(&ramp, true, false);
        assert_eq!(first[0], 1.0);
        assert_eq!(first[WIN - 1], 0.0);
        let last = fade_weights(&ramp, false, true);
        assert_eq!(last[0], 0.0);
        assert_eq!(last[WIN - 1], 1.0);
        let only = fade_weights(&ramp, true, true);
        assert!(
            only.iter().all(|&v| v == 1.0),
            "a lone window must not be faded"
        );
    }

    /// The figure the memory gate refuses against, and the arithmetic behind it:
    /// the score matrix alone is over a gigabyte, so a machine that cannot hold it
    /// is refused in microseconds rather than an hour into the track.
    #[test]
    fn the_per_forward_sizing_is_the_graphs_arithmetic() {
        assert_eq!(
            ATTENTION_SCORE_BYTES,
            60 * 8 * 801 * 801 * 4,
            "bands × heads × frames × frames"
        );
        assert!(ATTENTION_SCORE_BYTES > 1_200_000_000);
        let total = per_forward_mb();
        // Named terms, so a change to any of them shows up here rather than in a
        // magic number.
        let scores = 2 * (ATTENTION_SCORE_BYTES / (1024 * 1024)) as u64;
        let host = (7 * WIN * 4 / (1024 * 1024)) as u64;
        assert_eq!(host, 9, "five accumulators + mix + vocals ≈ 9 MiB");
        assert!(total > scores, "scores must not be the whole estimate");
        assert!(
            total < scores * 2,
            "{total} MiB is looser than the graph warrants"
        );
        assert!(
            (2800..3200).contains(&total),
            "per-forward figure is {total} MiB"
        );
    }

    // ─────────────────── gated: weights and real audio ───────────────────

    /// A path from an env var, or `[SKIP]` — the arms below need the weight file
    /// and/or a real track, and neither ships with or is downloaded into this
    /// repository.
    fn fixture(var: &str) -> Option<PathBuf> {
        let raw = std::env::var(var).ok()?;
        let p = PathBuf::from(&raw);
        if p.is_file() {
            return Some(p);
        }
        println!("[SKIP] {var} points at nothing ({raw})");
        None
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rr_mlx_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// Deterministic pseudo-random stereo content: an LCG mixed with a tone, so
    /// every window has non-silent, non-clipping material everywhere.
    fn write_test_wav(path: &Path, len: usize) {
        let mut w = WavWriter::create(path, spec16()).expect("create");
        let mut state: u32 = 0x1234_5678;
        let mut rng = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as i16) / 4 // well below full scale
        };
        for i in 0..len as i64 {
            let tone = (((i as f64 * 0.11).sin()) * 6000.0) as i16;
            w.write_sample(tone.wrapping_add(rng())).expect("write");
            w.write_sample(tone.wrapping_mul(3).wrapping_add(rng()))
                .expect("write");
        }
        w.finalize().expect("finalize");
    }

    fn read_i16_stereo(path: &Path) -> Vec<i16> {
        let mut r = hound::WavReader::open(path).expect("read stem");
        assert_eq!(r.spec().channels, 2);
        r.samples::<i16>().map(|s| s.expect("sample")).collect()
    }

    /// `(samples differing, largest absolute per-sample difference)`, in the
    /// i16 domain the stems are published in.
    fn stem_diff(a: &[i16], b: &[i16]) -> (usize, i32) {
        let n_diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap_or(0);
        (n_diff, worst)
    }

    /// Minimal `.npy` (v1.0, f32, C-order) reader, for the PyTorch reference the
    /// acceptance bar is stated against.
    fn load_npy_f32(path: &Path) -> Vec<f32> {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("npy read {}: {e}", path.display()));
        assert_eq!(&bytes[0..6], b"\x93NUMPY", "not an npy file");
        assert_eq!(bytes[6], 1, "only npy v1.0 is supported here");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = String::from_utf8_lossy(&bytes[10..10 + header_len]).to_string();
        assert!(header.contains("'<f4'"), "only <f4 is supported: {header}");
        assert!(
            header.contains("fortran_order': False"),
            "only C order is supported: {header}"
        );
        let shape_part = header
            .split("'shape':")
            .nth(1)
            .expect("shape in the npy header")
            .split('}')
            .next()
            .expect("shape before the closing brace")
            .trim();
        let shape: Vec<usize> = shape_part
            .trim_start_matches('(')
            .trim_end_matches(|c: char| c == ')' || c == ',')
            .split(',')
            .map(|p| p.trim().parse().expect("shape dim"))
            .collect();
        let floats: Vec<f32> = bytes[10 + header_len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats.len(), shape.iter().product::<usize>());
        floats
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        let ma = a[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let mb = b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let (mut num, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let da = a[i] as f64 - ma;
            let db = b[i] as f64 - mb;
            num += da * db;
            na += da * da;
            nb += db * db;
        }
        num / (na.sqrt() * nb.sqrt())
    }

    /// The weight file's keys must be exactly the set this loader asks for. Run
    /// against the real file this is the only arm that checks the naming
    /// convention against what the exporter actually wrote; without it the
    /// convention lives in comments.
    ///
    /// `RR_MLX_WEIGHTS=/path/to/mlx_roformer_vocals_fp32.safetensors cargo test
    /// --features mlx weight_file`
    #[test]
    fn weight_file_keys_are_exactly_the_ones_the_loader_expects() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!("[SKIP] weight_file_keys_are_exactly_the_ones_the_loader_expects: no RR_MLX_WEIGHTS");
            return;
        };
        let loaded = load_safetensors(&weights).expect("load");
        let want: std::collections::HashSet<String> = expected_keys().into_iter().collect();
        let have: std::collections::HashSet<String> = loaded.keys().cloned().collect();
        assert_eq!(loaded.len(), EXPECTED_TENSOR_COUNT, "tensor count");
        let missing: Vec<&String> = want.difference(&have).take(10).collect();
        let extra: Vec<&String> = have.difference(&want).take(10).collect();
        assert!(missing.is_empty(), "weight file is missing {missing:?}");
        assert!(extra.is_empty(), "weight file has unexpected {extra:?}");
    }

    /// The acceptance bar: one native window of a real recording, against a
    /// PyTorch FFT reference. It is compared to PyTorch and *not* to the ONNX
    /// graph's output on purpose — the export computes the STFT as a
    /// convolution, and its accumulated error in quiet high-frequency bands is
    /// amplified by the per-band L2Norm and cascaded through the stack, so the
    /// graph's own output is the thing under suspicion rather than the yardstick.
    ///
    /// `RR_MLX_WEIGHTS=… RR_TEST_WAV=… RR_TORCH_REF=… cargo test --features mlx
    /// --lib -- --ignored --test-threads=1 one_window_matches`
    ///
    /// Ignored by default, and not merely because the artifact is ~1 GB: an engine
    /// that is resident while `mem`'s own footprint probe runs makes that probe
    /// under-read (the allocator answers a 512 MiB request out of a region a
    /// sibling already freed), which is the interaction
    /// `mem::footprint_tracks_a_known_allocation` documents from the other side.
    /// `--test-threads=1` keeps the two apart.
    #[test]
    #[ignore = "maps the RoFormer MLX weight file (~1 GB); run with --ignored"]
    fn one_window_matches_the_torch_fft_reference() {
        let (Some(weights), Some(wav)) = (fixture("RR_MLX_WEIGHTS"), fixture("RR_TEST_WAV")) else {
            println!("[SKIP] one_window_matches_the_torch_fft_reference: needs RR_MLX_WEIGHTS + RR_TEST_WAV");
            return;
        };
        let Some(reference) = fixture("RR_TORCH_REF") else {
            println!(
                "[SKIP] one_window_matches_the_torch_fft_reference: needs RR_TORCH_REF \
                       (a float32 f-array of the reference vocals for this window; this \
                       repository does not ship one)"
            );
            return;
        };
        let mut source = WavSource::open(&wav).expect("open test wav");
        let take = WIN.min(source.frames());
        let mut chunk = Array2::<f32>::zeros((2, WIN));
        {
            let mut view = chunk.slice_mut(ndarray::s![.., 0..take]);
            source.read(0, &mut view).expect("read window");
        }
        let engine = MlxEngine::load(&weights).expect("load engine");
        let t0 = Instant::now();
        let out = engine.separate_window(&chunk).expect("forward");
        println!(
            "[mlx] one window of {} in {:.2}s",
            wav.display(),
            t0.elapsed().as_secs_f64()
        );

        let reference = load_npy_f32(&reference);
        assert_eq!(reference.len(), 2 * take, "reference is not this window");
        let mut flat = Vec::with_capacity(2 * take);
        flat.extend_from_slice(out.row(0).as_slice().expect("contiguous"));
        flat.extend_from_slice(out.row(1).as_slice().expect("contiguous"));
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "the window contains non-finite samples"
        );
        let corr = correlation(&flat, &reference);
        println!("[mlx] corr vs the PyTorch FFT reference = {corr:.7}");
        assert!(corr > 0.999, "corr vs the PyTorch reference = {corr}");
    }

    /// Streaming equivalence: the bounded-buffer loop and a whole-track
    /// accumulation must publish *identical* i16 samples. The engine runs the
    /// same graph either way, so anything but zero differing samples means the
    /// I/O path moved a number. The lengths are the awkward ones: shorter than a
    /// window, exactly one, a grid plus an appended tail window, and a ragged
    /// three-window track.
    ///
    /// `RR_MLX_WEIGHTS=/path/to.safetensors cargo test --features mlx --lib --
    /// --ignored --test-threads=1 streaming_matches`
    #[test]
    #[ignore = "maps the RoFormer MLX weight file and runs minutes of forwards"]
    fn streaming_matches_a_whole_track_accumulation() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!("[SKIP] streaming_matches_a_whole_track_accumulation: no RR_MLX_WEIGHTS");
            return;
        };
        let mut engine = MlxEngine::load(&weights).expect("load engine");
        let dir = scratch("equiv");
        for (k, &len) in [WIN / 3, WIN, WIN + (WIN - OVERLAP) + 4321, 2 * WIN + 7]
            .iter()
            .enumerate()
        {
            let input = dir.join(format!("in_{k}.wav"));
            write_test_wav(&input, len);

            let streamed = dir.join(format!("v{k}_stream.wav"));
            let streamed_b = dir.join(format!("b{k}_stream.wav"));
            engine
                .separate(
                    &input,
                    &StemPaths::new(&streamed, &streamed_b),
                    &Default::default(),
                )
                .expect("streaming failed");

            let whole = dir.join(format!("v{k}_ref.wav"));
            let whole_b = dir.join(format!("b{k}_ref.wav"));
            whole_track_reference(&engine, &input, &whole, &whole_b).expect("reference failed");

            assert!(
                !part_path(&streamed).exists() && !job_path(&streamed).exists(),
                "len={len}: staging left behind"
            );
            for (a, b, name) in [
                (&streamed, &whole, "vocals"),
                (&streamed_b, &whole_b, "background"),
            ] {
                let (sa, sb) = (read_i16_stereo(a), read_i16_stereo(b));
                assert_eq!(sa.len(), sb.len(), "{name} length changed at len={len}");
                let (n_diff, worst) = stem_diff(&sa, &sb);
                println!(
                    "[mlx] len={len} {name}: {n_diff}/{} samples differ, max {worst} LSB",
                    sa.len()
                );
                assert_eq!(worst, 0, "{name} is not bit-identical at len={len}");
            }
            for f in [&input, &streamed, &streamed_b, &whole, &whole_b] {
                let _ = std::fs::remove_file(f);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pre-streaming implementation, kept as the oracle: hold both stems and
    /// the weight sum for the whole track, then write once. Exactly the code the
    /// streaming loop replaced, and the reason its memory model is a claim
    /// rather than an assumption.
    fn whole_track_reference(
        engine: &MlxEngine,
        input: &Path,
        vocals: &Path,
        background: &Path,
    ) -> Result<()> {
        let mut source = WavSource::open(input)?;
        let total = source.frames();
        let starts = chunk_starts(total, WIN, OVERLAP);
        let ramp = linear_ramp(OVERLAP);
        let mut voc_acc = Array2::<f32>::zeros((2, total));
        let mut bak_acc = Array2::<f32>::zeros((2, total));
        let mut weight = vec![0.0f32; total];
        let mut chunk = Array2::<f32>::zeros((2, WIN));
        for (i, &start) in starts.iter().enumerate() {
            let take = (total - start).min(WIN);
            chunk.fill(0.0);
            {
                let mut view = chunk.slice_mut(ndarray::s![.., 0..take]);
                source.read(start, &mut view)?;
            }
            let voc = engine.separate_window(&chunk)?;
            let w = fade_weights(&ramp, i == 0, i == starts.len() - 1);
            for ch in 0..2 {
                for k in 0..take {
                    let g = start + k;
                    let v = voc[[ch, k]];
                    voc_acc[[ch, g]] += v * w[k];
                    bak_acc[[ch, g]] += (chunk[[ch, k]] - v) * w[k];
                }
            }
            for k in 0..take {
                weight[start + k] += w[k];
            }
        }
        let mut writer = StemWriter::create(vocals, background)?;
        for g in 0..total {
            let wk = weight[g];
            if wk > 1e-8 {
                writer.write_frame(
                    [voc_acc[[0, g]] / wk, voc_acc[[1, g]] / wk],
                    [bak_acc[[0, g]] / wk, bak_acc[[1, g]] / wk],
                )?;
            } else {
                writer.write_frame([0.0, 0.0], [0.0, 0.0])?;
            }
        }
        writer.finish()?;
        Ok(())
    }

    /// A run that dies after window K and is restarted must deliver exactly the
    /// bytes a run that never died would have written — including the background
    /// stem, which is the residual and therefore the sensitive one. Two tracks:
    /// one landing exactly on the grid, one whose appended final window starts
    /// inside an earlier window's fade-out, so that seam has two contributors and
    /// the retry replays both.
    ///
    /// `RR_MLX_WEIGHTS=/path/to.safetensors cargo test --features mlx --lib --
    /// --ignored --test-threads=1 resuming_a_killed`
    #[test]
    #[ignore = "maps the RoFormer MLX weight file and runs minutes of forwards"]
    fn resuming_a_killed_run_is_bit_identical() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!("[SKIP] resuming_a_killed_run_is_bit_identical: no RR_MLX_WEIGHTS");
            return;
        };
        let dir = scratch("resume");
        for (tag, len) in [("grid", 4 * WIN), ("tail", 850_000)] {
            let input = dir.join(format!("{tag}_in.wav"));
            write_test_wav(&input, len);
            let starts = chunk_starts(len, WIN, OVERLAP);
            let n = starts.len();
            assert!(n >= 3, "{tag}: {n} windows is not enough to test a seam");
            if tag == "tail" {
                let p = resume_plan(&starts, len, WIN, starts[n - 1]);
                assert_eq!((p.first, p.prime_from), (n - 1, n - 3), "{tag}: {starts:?}");
            }

            let (v0, b0) = (
                dir.join(format!("{tag}_once_v.wav")),
                dir.join(format!("{tag}_once_b.wav")),
            );
            {
                let mut e = MlxEngine::load(&weights).expect("load");
                let r = e
                    .separate(&input, &StemPaths::new(&v0, &b0), &Default::default())
                    .expect("uninterrupted run failed");
                assert_eq!(r.windows_inferred, n, "{tag}: every window should run");
                assert_eq!(r.windows_resumed, 0);
                assert_eq!(r.frames, len);
                assert_eq!(r.sample_rate, SAMPLE_RATE);
            }

            for &k in &[0usize, n - 2, n - 1] {
                let (v, b) = (
                    dir.join(format!("{tag}_k{k}_v.wav")),
                    dir.join(format!("{tag}_k{k}_b.wav")),
                );
                let paths = StemPaths::new(&v, &b);
                CRASH_AFTER.with(|c| c.set(k));
                let err = {
                    let mut e = MlxEngine::load(&weights).expect("load");
                    e.separate(&input, &paths, &Default::default())
                        .expect_err("the crash hook did not fire")
                };
                CRASH_AFTER.with(|c| c.set(usize::MAX));
                assert!(err.to_string().contains("test crash"), "{tag} k={k}: {err}");

                // A failure that is not cancellation keeps the checkpoint — that
                // is the whole point of checkpointing.
                let finalized = if k + 1 < n { starts[k + 1] } else { len };
                assert_eq!(
                    std::fs::metadata(part_path(&v)).expect("pair gone").len(),
                    STEM_HEADER_BYTES + finalized as u64 * 4,
                    "{tag}: the checkpoint after window {k} holds the wrong prefix"
                );
                assert!(
                    part_path(&b).exists() && job_path(&v).exists(),
                    "{tag}: sidecar lost"
                );

                let t0 = Instant::now();
                let r = {
                    let mut e = MlxEngine::load(&weights).expect("load");
                    e.separate(&input, &paths, &Default::default())
                        .unwrap_or_else(|e| panic!("{tag}: resume after window {k} failed: {e}"))
                };
                let expected = if k + 1 < n {
                    n - resume_plan(&starts, len, WIN, finalized).prime_from
                } else {
                    0
                };
                assert_eq!(
                    r.windows_inferred, expected,
                    "{tag} k={k}: the resume re-inferred {} windows, expected {expected}",
                    r.windows_inferred
                );
                println!(
                    "[mlx] {tag} killed after window {k}: re-inferred {}/{} windows in {:.1}s",
                    r.windows_inferred,
                    n,
                    t0.elapsed().as_secs_f64()
                );
                assert_eq!(r.windows_resumed, n - expected, "{tag} k={k}");
                assert!(
                    !part_path(&v).exists() && !job_path(&v).exists(),
                    "{tag}: leftovers"
                );

                for (a, bb, name) in [(&v0, &v, "vocals"), (&b0, &b, "background")] {
                    let (sa, sb) = (read_i16_stereo(a), read_i16_stereo(bb));
                    assert_eq!(sa.len(), sb.len(), "{tag} k={k}: {name} length changed");
                    let (n_diff, worst) = stem_diff(&sa, &sb);
                    println!(
                        "[mlx] {tag} k={k} {name}: {n_diff}/{} samples differ, max {worst} LSB",
                        sa.len()
                    );
                    assert_eq!(
                        worst, 0,
                        "{tag} k={k}: the resumed {name} is not bit-identical"
                    );
                }
            }
            for f in [&input, &v0, &b0] {
                let _ = std::fs::remove_file(f);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cancellation between windows: the run stops, returns `Cancelled`, and keeps
    /// the checkpoint the trait promises — and a later call that continues it
    /// publishes the bytes an uninterrupted run would have.
    #[test]
    #[ignore = "maps the RoFormer MLX weight file (~1 GB)"]
    fn a_cancelled_run_leaves_a_resumable_checkpoint() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!("[SKIP] a_cancelled_run_leaves_a_resumable_checkpoint: no RR_MLX_WEIGHTS");
            return;
        };
        let dir = scratch("cancel");
        let input = dir.join("in.wav");
        write_test_wav(&input, 2 * WIN);
        let paths = StemPaths::new(dir.join("v.wav"), dir.join("b.wav"));

        // Pre-cancelled: nothing is created at all, not even the `.part` pair.
        let flag = CancelFlag::new();
        flag.cancel();
        let mut e = MlxEngine::load(&weights).expect("load");
        let err = e
            .separate(
                &input,
                &paths,
                &SeparationOptions::default().with_cancel(flag.clone()),
            )
            .expect_err("a pre-cancelled run must not produce output");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(!paths.vocals.exists() && !part_path(&paths.vocals).exists());
        assert!(!job_path(&paths.vocals).exists());

        // Cancelled after the first checkpoint: the windows already flushed were
        // paid-for work, so they stay on disk for whoever comes next.
        let mut e = MlxEngine::load(&weights).expect("load");
        let flag = CancelFlag::new();
        // The callback fires once per window before its forward, so this stops the
        // run at the first boundary after a checkpoint has been written — the
        // state a user hitting "stop" actually produces.
        let cb: crate::config::Progress = {
            let flag = flag.clone();
            std::sync::Arc::new(move |_pct, _msg| flag.cancel())
        };
        let err = e
            .separate(
                &input,
                &paths,
                &SeparationOptions::default()
                    .with_cancel(flag)
                    .with_progress(cb),
            )
            .expect_err("the flag was set by the progress callback");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(
            part_path(&paths.vocals).exists() && job_path(&paths.vocals).exists(),
            "a cancelled run threw away the checkpoint the next one resumes from"
        );
        assert!(
            !paths.vocals.exists(),
            "the published name appears only when the whole track is written"
        );

        // Continue it. This is the claim the crate sells, stated for the one
        // failure a user causes on purpose.
        let mut e = MlxEngine::load(&weights).expect("load");
        let r = e
            .separate(&input, &paths, &SeparationOptions::default())
            .expect("a cancelled run must be resumable");
        // The cancelled pass flushed exactly one hop (`starts[1]`), so that is the
        // seam the continuing pass must report it started from. `windows_resumed`
        // is deliberately *not* asserted non-zero here: one flushed window is
        // re-inferred as seam priming rather than skipped, and a count of skipped
        // windows says nothing about reused bytes.
        assert_eq!(
            r.resumed_from_frames,
            Some(WIN - OVERLAP),
            "the continuing pass did not start from the cancelled run's flush point: {r:?}"
        );
        assert_eq!(
            r.windows_inferred + r.windows_resumed,
            chunk_starts(2 * WIN, WIN, OVERLAP).len(),
            "the two counts must partition the schedule: {r:?}"
        );

        let control = scratch("cancel-control");
        let control_input = control.join("in.wav");
        write_test_wav(&control_input, 2 * WIN);
        let control_paths = StemPaths::new(control.join("v.wav"), control.join("b.wav"));
        let mut e = MlxEngine::load(&weights).expect("load");
        e.separate(
            &control_input,
            &control_paths,
            &SeparationOptions::default(),
        )
        .expect("control run");

        for (got, want, name) in [
            (&paths.vocals, &control_paths.vocals, "vocals"),
            (&paths.background, &control_paths.background, "background"),
        ] {
            let (a, b) = (read_i16_stereo(got), read_i16_stereo(want));
            let (n_diff, worst) = stem_diff(&a, &b);
            assert_eq!(a.len(), b.len(), "{name}: length changed");
            assert_eq!(
                n_diff,
                0,
                "{name}: {n_diff}/{} samples differ (max {worst} LSB) after resuming a \
                 cancellation",
                a.len()
            );
        }
        let _ = std::fs::remove_dir_all(&control);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The gate refuses *before* the first forward, so the refusal costs
    /// microseconds, names the figure it refused against, and creates nothing on
    /// disk — the state a caller can act on ("smaller window / other engine")
    /// without having lost an hour of work or left a `.part` pair behind.
    #[test]
    #[ignore = "maps the RoFormer MLX weight file (~1 GB)"]
    fn the_memory_gate_refuses_before_it_touches_the_track() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!(
                "[SKIP] the_memory_gate_refuses_before_it_touches_the_track: no RR_MLX_WEIGHTS"
            );
            return;
        };
        let dir = scratch("gate");
        let input = dir.join("in.wav");
        write_test_wav(&input, WIN / 3);
        let paths = StemPaths::new(dir.join("v.wav"), dir.join("b.wav"));

        let mut e = MlxEngine::load(&weights).expect("load");
        // A budget no window can fit inside. The engine is loaded, because the
        // crate's contract puts weight loading outside `separate`; what must not
        // happen is the run starting.
        let err = e
            .separate(
                &input,
                &paths,
                &SeparationOptions::default().with_memory_budget_mb(1),
            )
            .expect_err("a 1 MB budget cannot hold a native window");
        match &err {
            Error::Memory { need_mb, avail_mb } => {
                assert_eq!(*need_mb, per_forward_mb());
                assert_eq!(*avail_mb, Some(1), "{err}");
            }
            other => panic!("wrong variant: {other:?}"),
        }
        // The refusal is the memory answer, not the broken-file answer.
        assert!(err.looks_like_allocation_failure(), "{err}");
        assert!(
            !paths.vocals.exists()
                && !paths.background.exists()
                && !part_path(&paths.vocals).exists()
                && !job_path(&paths.vocals).exists(),
            "a refused run must not create any output"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `Fresh` run ignores a checkpoint; an `Auto` run of the same job after it
    /// starts from zero, and the report says so rather than hiding it.
    #[test]
    #[ignore = "maps the RoFormer MLX weight file (~1 GB)"]
    fn a_fresh_run_reports_that_it_threw_the_checkpoint_away() {
        let Some(weights) = fixture("RR_MLX_WEIGHTS") else {
            println!(
                "[SKIP] a_fresh_run_reports_that_it_threw_the_checkpoint_away: no RR_MLX_WEIGHTS"
            );
            return;
        };
        let dir = scratch("fresh");
        let input = dir.join("in.wav");
        write_test_wav(&input, WIN + (WIN - OVERLAP) + 4321);
        let starts = chunk_starts(WIN + (WIN - OVERLAP) + 4321, WIN, OVERLAP);
        let paths = StemPaths::new(dir.join("v.wav"), dir.join("b.wav"));

        CRASH_AFTER.with(|c| c.set(0));
        let mut e = MlxEngine::load(&weights).expect("load");
        e.separate(&input, &paths, &Default::default())
            .expect_err("crash after window 0");
        CRASH_AFTER.with(|c| c.set(usize::MAX));

        let mut e = MlxEngine::load(&weights).expect("load");
        let r = e
            .separate(
                &input,
                &paths,
                &SeparationOptions {
                    resume: ResumeMode::Fresh,
                    ..Default::default()
                },
            )
            .expect("fresh run");
        assert_eq!(r.windows_resumed, 0, "a fresh start resumed nothing");
        assert_eq!(r.windows_inferred, starts.len());
        assert_eq!(r.resumed_from_frames, None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
