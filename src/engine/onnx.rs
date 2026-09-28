//! ONNX Runtime engine: run a Mel-Band RoFormer export one window at a time.
//!
//! # What this file is responsible for
//!
//! The graph does the maths — conv STFT, band split, alternating transformer,
//! mask, iSTFT are all inside the exported model, and the input it wants is a
//! plain `[1, 2, window]` f32 block of stereo at 44.1 kHz. Everything that is
//! *not* the model lives here: the schedule that tiles a track with windows, the
//! crossfade that stitches the outputs back together, the streaming writer, the
//! resume identity, the cancellation check, and the memory gate that refuses an
//! impossible window before it spends an hour discovering it.
//!
//! ## The graph's two sources
//!
//! `sources` is `[1, source, channel, length]` with the **source axis first**:
//! source 0 is vocals, source 1 is the graph's own residual. In these exports
//! that residual is produced inside the graph as `Sub(mix, source0)` — it is not
//! recomputed here, because a graph that separates a different way (three
//! sources, an accompaniment stem, a learned residual) would be silently mangled
//! by a subtraction we imposed on it. Both stems are accumulated through the same
//! crossfade and written by the same pass.
//!
//! ## Window: a load-time parameter, not a file format
//!
//! The window is baked into the graph as `int64` constants plus the input and
//! output dimensions ([`crate::graph`]), so [`OnnxEngine::with_window`] patches
//! those bytes in memory and hands the result to
//! `ort`'s `Session::builder().commit_from_memory(..)`. Nothing derived is ever written
//! to disk: the file the user downloaded stays the file on disk. Loading a
//! session from a buffer — and editing a graph in memory, which `ort` also
//! exposes — is the runtime's capability, not this crate's; what this crate adds
//! is the map of which bytes in a Mel-Band RoFormer export carry the window, and
//! the argument for why rewriting them is exact. The transient
//! cost is one buffer holding the whole model (≈ the file size — 259 MiB for the
//! 271,832,758-byte int8 vocals export), released as soon as the session has
//! parsed it; it coexists with the runtime's own copy of the weights, so building
//! a patched session is the one moment the process holds the model twice.
//!
//! ## Stale `value_info`: measured, not assumed
//!
//! A patched graph keeps the cached shape-inference records it was exported with
//! (`value_info`), and they describe the *old* window. Deleting them would change the
//! message length and force a full re-serialisation, so they stay. What ONNX Runtime
//! makes of that is a fact about the runtime, and it was measured on the 271,832,758
//! byte int8 vocals export (7,908 nodes, **6,945** cached records), patched in memory
//! from its declared 352,800 down to 176,400:
//!
//! **The session is built; the runtime does not refuse.** Of the 6,945 records it
//! complains about four — the ones whose length actually contradicts the patched
//! shape — and it complains in the form of a warning, then proceeds:
//!
//! ```text
//! [W:onnxruntime:, graph.cc:123 MergeShapeInfo] Error merging shape info for output.
//!   '/Reshape_output_0' source:{2,1,176400} target:{2,1,352800}. Falling back to lenient merge.
//!   '/Sub_1_output_0' source:{-1,2,176400} target:{-1,2,352800}. ...
//!   'sources' source:{-1,2,2,352800} target:{1,2,2,176400}. ...
//!   '/Unsqueeze_17_output_0' source:{-1,1,2,176400} target:{-1,1,2,352800}. ...
//! ```
//!
//! "Falling back to lenient merge" is the story: ORT re-runs shape inference, and where
//! its own result disagrees with a cached record it does not treat the graph as broken.
//! The line worth reading is the third one — the `sources` output is involved in the
//! disagreement (the two shapes it names are `{1,2,2,176400}` and `{-1,2,2,352800}`),
//! which is precisely the tensor this engine consumes. So the behaviour is tolerated
//! *because it was measured*, not because a stale cache is assumed harmless:
//! [`OnnxEngine::with_window`] asserts the session's declared input dimension, every
//! forward re-checks the output length against the window, and a real 12-second
//! separation through a patched session produced a full-length pair whose resumed bytes
//! equalled the uninterrupted ones.
//!
//! ## Threads and the CPU arena
//!
//! The arena is **off by default**. Measured on the same entry point, same window,
//! with nothing else changed: **8.83 GB peak with the arena on, 3,440 MB with it
//! off, and not one output sample differing**. The reason is that this loop calls
//! a session once per window, so the arena's only job is to park a
//! gigabytes-sized set of dead activations between calls — a resident ceiling
//! that turns a long job into an allocation failure on a 16 GB machine, for no
//! reuse benefit at all. [`OnnxEngine::with_arena`] turns it back on for callers
//! who want it; expect the peak to follow.
//!
//! ## Estimating what a window costs
//!
//! [`OnnxEngine::estimate_forward_mb`] is built from **two measured anchors and
//! nothing else**, both taken on a 16 GB-class Windows machine and both
//! independent of track length:
//!
//! | window | seconds | measured per-forward commit |
//! |---|---|---|
//! | 176,400 | 4 | ≈ 5,100 MB |
//! | 352,800 | 8 | ≈ 19,400 MB |
//!
//! The time-axis attention in this architecture is quadratic in the window, so
//! the two anchors are fitted as `c + q·T²` and interpolated with that law. That
//! is an *estimate*: it is used to refuse a window that cannot possibly fit and to
//! put a number in the log, never to promise that a window it approves will run.
//! Between the anchors is interpolation; outside them the figure is an
//! extrapolation of the same curve and the log says which it is.
//!
//! ## Input: rates and channels
//!
//! WAV only, through [`WavSource`], which decodes one window at a time and
//! normalises to stereo 44.1 kHz: 16-, 24- and 32-bit integer PCM and 32-bit
//! float. A **mono** file is *replicated* onto both channels (not summed to one,
//! not mid/side upmixed), which is what that reader does and what keeps a mono
//! input and its stereo result comparable sample for sample. Any other rate is
//! resampled by that same reader — **linear interpolation**, deliberately: it
//! happens inside the windowed read, so no pass over the whole track is needed.
//! [`crate::audio::resample_chunked`] is the windowed-sinc alternative for a
//! caller willing to produce a 44.1 kHz file first; feeding that output back in
//! here is the higher-quality route, and it is a caller-side choice rather than a
//! default this engine can make silently.
//!
//! *WAV only.* Decoding an MP3 or a video track is not this engine's job, and a
//! `hound` failure on a compressed input is reported as [`Error::Wav`] rather than
//! papered over.
//!
//! ## What a cancelled run leaves behind
//!
//! [`crate::error::Error::Cancelled`] and every other failure both leave the `.part`
//! pair, its sidecar and its bytes in place, because both mean "the next attempt should
//! not start at zero" — the trait says as much, and a checkpoint you throw away is the
//! work you just paid for. Deleting a checkpoint is a *separate*, explicit act:
//! [`crate::stream::discard_staging`] (or [`crate::stream::StemWriter::cleanup`]) on the
//! caller's side. The published names only ever appear together, at the end, so a
//! skip-if-exists check cannot mistake a partial run for a finished one.
//!
//! ## `peak_mb` in the report
//!
//! Sampled *inside* the call by [`mem::peak_mb_while`], which is what
//! [`crate::mem`]'s own documentation says to use: the OS-side peak counters are
//! high-water marks **since the process started**, so on macOS a second `separate`
//! call in the same process would otherwise report the first call's peak forever
//! (and on Windows it would report an evictable working set). The residual caveat
//! is the sampler's interval: 20 ms, so a transient shorter than that can be
//! missed and `peak_mb` should be read as "at least this".

use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{s, Array2};
use ort::ep::CPU;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::TensorRef;
use ort::value::ValueType;

use crate::audio::SAMPLE_RATE;
use crate::config::{ResumeMode, SeparationOptions, SeparationReport, StemPaths};
use crate::engine::SeparationEngine;
use crate::error::{Error, Result};
use crate::graph;
use crate::mem;
use crate::stream::{chunk_starts, resume_plan, StemJob, StemWriter, WavSource};

/// Overlap for the linear crossfade when the caller does not choose one: 1.5 s at
/// 44.1 kHz, which is the geometry the flush proofs in [`crate::stream`] are
/// stated against — clamped to half the window where the model is shorter,
/// because the grid requires `overlap <= window / 2` (see
/// [`crate::stream::chunk_starts`]).
pub const DEFAULT_OVERLAP: usize = 66_150;

/// The two measured per-forward anchors behind [`OnnxEngine::estimate_forward_mb`]
/// — `(window samples, MB)`. See the module docs: they are the whole dataset, and
/// a third measurement that disagrees is a correction to this file, not a detail
/// to average away.
const SHORT_ANCHOR: (usize, f64) = (176_400, 5_100.0);
const LONG_ANCHOR: (usize, f64) = (352_800, 19_400.0);

/// How the session is (re)built. Both arms keep only the *path*, so a rebuild
/// after a thread-count change re-reads and re-patches rather than holding a
/// model-sized buffer for the life of the engine.
#[derive(Debug, Clone)]
enum Source {
    /// The file exactly as it is on disk, committed as the bytes read from it.
    Native,
    /// The same file with its baked window rewritten to `target` in memory.
    Patched { target: usize },
}

/// The three knobs that invalidate a live session when changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    intra: usize,
    inter: usize,
    arena: bool,
}

/// ONNX Runtime engine.
///
/// Build one with [`load`](Self::load) (the graph runs at the window it was
/// exported for) or [`with_window`](Self::with_window) (the same file, reshaped in
/// memory). Both are cheap to construct and heavy enough to hold that one engine
/// is normally reused for a whole batch.
pub struct OnnxEngine {
    path: PathBuf,
    source: Source,
    config: Config,
    /// Window this engine forwards: `target` when patched, the graph's own
    /// otherwise.
    window: usize,
    /// Window the file on disk declares, before any patch, so a log can name both.
    declared: usize,
    /// Cached shape-inference records in the graph — stale after a patch by
    /// construction, and left in place. See the module note on what the runtime
    /// makes of them.
    value_info_records: usize,
    /// The graph's own name, when it carries one.
    graph_name: Option<String>,
    /// Live session, together with the config it was built under.
    session: Option<(Session, Config)>,
}

impl OnnxEngine {
    /// Load a graph and read its window off the graph.
    ///
    /// The file is read **once** and the same buffer serves both steps: it is what
    /// [`graph::inspect`] walks (declaring the window, the table capacity and the
    /// number of `value_info` records) and what the runtime parses. The transient is
    /// therefore one model-sized buffer, not two, and it is dropped as soon as the
    /// session exists.
    ///
    /// There is no path-taking alternative on this dependency setup: `ort`'s
    /// `commit_from_file` is gated behind its `std` feature, which this crate does
    /// not enable, so every session here is built from bytes.
    pub fn load(path: &Path) -> Result<Self> {
        Self::build(path, None)
    }

    /// Load a graph and rewrite its window to `target_samples`, in memory.
    ///
    /// Refusals come straight from [`graph::patch_window`] and are surfaced
    /// verbatim: it will not grow the window past the exported capacity, will not
    /// touch a graph that carries the window inside a weight, and will not rewrite
    /// a dimension varint whose encoded width would change (which would shift every
    /// byte after it). A refusal leaves the buffer untouched, so there is no
    /// half-patched graph to load.
    ///
    /// After the session exists, its **own** declared input dimension is compared
    /// with `target_samples`. That is not redundant with the patch: the patch
    /// proves we rewrote the bytes we found, the session proves the runtime agrees
    /// about what those bytes mean. If the runtime reports a different number — a
    /// dimension we missed, or an input whose length is dynamic — this is
    /// [`Error::Model`] and nothing runs.
    pub fn with_window(path: &Path, target_samples: usize) -> Result<Self> {
        Self::build(path, Some(target_samples))
    }

    fn build(path: &Path, target: Option<usize>) -> Result<Self> {
        let mut bytes = std::fs::read(path).map_err(|e| Error::io_at(path, e))?;
        let graph_name = graph::graph_name(&bytes);
        let report = graph::inspect(&bytes)?;
        let declared = report.declared_window;
        if declared <= 0 {
            return Err(Error::Model {
                detail: format!(
                    "the graph input declares a window of {declared}; a non-positive or dynamic \
                     dimension cannot be separated against"
                ),
            });
        }
        let mut window = declared as usize;
        let mut value_info_records = report.value_info_records;
        let mut patched_to = None;
        if let Some(target) = target {
            // Every refusal in `patch_window` — growing past the exported capacity,
            // a window carried by a weight, a dimension varint that would change
            // width — leaves `bytes` untouched, so there is no half-patched graph
            // that could be committed and quietly compute the wrong thing.
            let report = graph::patch_window(&mut bytes, target)?;
            window = report.declared_window as usize;
            value_info_records = report.value_info_records;
            patched_to = Some(target);
        }
        let mut engine = Self {
            path: path.to_path_buf(),
            source: match patched_to {
                Some(t) => Source::Patched { target: t },
                None => Source::Native,
            },
            config: Config {
                // Matches `SeparationOptions::default`, so a caller who sets one
                // and not the other gets the same answer either way.
                intra: 4,
                inter: 2,
                arena: false,
            },
            window,
            declared: declared as usize,
            value_info_records,
            graph_name,
            session: None,
        };
        log::info!(
            "[onnx] {}: graph {:?}, declares {declared} sample(s){}, {} cached \
             shape-inference record(s); per-forward estimate ~{} MB",
            path.display(),
            engine.graph_name,
            match patched_to {
                Some(t) => format!(", running {t} after an in-memory patch"),
                None => String::new(),
            },
            engine.value_info_records,
            Self::estimate_forward_mb(window),
        );
        // Building the session here is the measurement, not a warm-up: if the
        // runtime will not accept patched bytes that still describe the old shape
        // in its cached inference records, this is where the answer arrives.
        let session = engine.commit_bytes(&bytes)?;
        // The runtime has parsed them; our copy is dead weight from here on.
        drop(bytes);
        engine.assert_declared_window(&session, window)?;
        engine.session = Some((session, engine.config));
        Ok(engine)
    }

    /// Read the model file, patching it first if this engine was asked to.
    ///
    /// The buffer holds the whole file — ≈ 259 MiB for the 271,832,758-byte int8
    /// vocals export — and is dropped by the caller as soon as the runtime has
    /// parsed it. It is the model's second copy, not a third, and it never
    /// outlives the build.
    fn load_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes =
            std::fs::read(&self.path).map_err(|e| Error::io_at(self.path.as_path(), e))?;
        if let Source::Patched { target } = self.source {
            graph::patch_window(&mut bytes, target)?;
        }
        Ok(bytes)
    }

    /// Hand model bytes to the runtime.
    ///
    /// Even an unpatched graph arrives as bytes: `commit_from_file` is gated behind
    /// `ort`'s `std` feature, which this crate does not enable, so the file path is
    /// not an option the runtime offers. The cost is the one model-sized transient
    /// described in [`load_bytes`](Self::load_bytes).
    fn commit_bytes(&self, bytes: &[u8]) -> Result<Session> {
        let what = match self.source {
            Source::Native => "unmodified".to_string(),
            Source::Patched { target } => format!(
                "window-patched to {target} in memory ({} cached shape-inference record(s) \
                 still describe the pre-patch shape)",
                self.value_info_records
            ),
        };
        let mut builder = self.builder()?;
        builder
            .commit_from_memory(bytes)
            .map_err(|e| Error::Session {
                detail: format!(
                    "could not build a session from {} bytes of the {what} graph: {e}",
                    bytes.len(),
                ),
            })
    }

    /// Intra-op / inter-op thread counts for the session.
    ///
    /// Takes effect on the next session build — which is the first forward of the
    /// next `separate()` call if a session is already live under different
    /// settings.
    #[must_use]
    pub fn with_threads(mut self, intra: usize, inter: usize) -> Self {
        self.config.intra = intra;
        self.config.inter = inter;
        self
    }

    /// Turn ONNX Runtime's CPU memory arena on or off. Off by default; see the
    /// module docs for the 8.83 GB vs 3,440 MB reading behind that.
    #[must_use]
    pub fn with_arena(mut self, use_arena: bool) -> Self {
        self.config.arena = use_arena;
        self
    }

    /// True when this engine runs a window rewritten in memory rather than the one
    /// the file declares.
    pub fn is_patched(&self) -> bool {
        matches!(self.source, Source::Patched { .. })
    }

    /// Window the file on disk declares, which for a patched engine differs from
    /// [`SeparationEngine::window_samples`].
    pub fn declared_window(&self) -> usize {
        self.declared
    }

    /// Cached shape-inference records found in the graph.
    pub fn value_info_records(&self) -> usize {
        self.value_info_records
    }

    /// Per-forward commit (MiB) a window of this length is *estimated* to cost.
    ///
    /// Two measured anchors, fitted by the `c + q·T²` law the module docs state:
    /// 176 400 samples (4 s) ≈ 5 100 MB and 352 800 samples (8 s) ≈ 19 400 MB of
    /// commit per forward, both on a 16 GB-class Windows machine and both
    /// independent of track length. Between the anchors this is an interpolation;
    /// outside them it is the same curve extended, so the answer is a shape rather
    /// than a measurement. The pre-flight log line names which of those two cases
    /// applied, so a refusal says whether its number was measured or extended.
    pub fn estimate_forward_mb(window: usize) -> u64 {
        if window == 0 {
            return 0;
        }
        let (t0, m0) = SHORT_ANCHOR;
        let (t1, m1) = LONG_ANCHOR;
        // Both windows are far too small for `T²` to overflow a `usize`.
        let q = (m1 - m0) / (t1 * t1 - t0 * t0) as f64;
        let c = m0 - q * (t0 * t0) as f64;
        (c + q * (window * window) as f64).max(0.0).round() as u64
    }

    /// The builder with this engine's thread and arena settings.
    fn builder(&self) -> Result<SessionBuilder> {
        let builder = Session::builder().map_err(|e| Error::Session {
            detail: format!("could not create a session builder: {e}"),
        })?;
        // Each setter returns its builder back inside an error on failure, so the
        // chain has to be broken up to keep the `?` shape.
        let builder =
            builder
                .with_intra_threads(self.config.intra)
                .map_err(|e| Error::Session {
                    detail: format!("could not set intra threads ({}): {e}", self.config.intra),
                })?;
        let builder =
            builder
                .with_inter_threads(self.config.inter)
                .map_err(|e| Error::Session {
                    detail: format!("could not set inter threads ({}): {e}", self.config.inter),
                })?;
        // Naming the CPU provider is what makes "CPU only" true. This graph mixes
        // convolutions and attention; an accelerator that recompiles per chunk is a
        // failure mode a windowed loop does not need, and the memory numbers quoted
        // in this module were all taken on this path.
        let builder = builder
            .with_execution_providers([CPU::default()
                .with_arena_allocator(self.config.arena)
                .build()])
            .map_err(|e| Error::Session {
                detail: format!("could not configure the CPU execution provider: {e}"),
            })?;
        Ok(builder)
    }

    /// The session's own declared input dimension must be `target`.
    ///
    /// This is the check that turns "we patched bytes" into "the runtime will run
    /// that shape", and it is the reason a patched engine can be trusted to
    /// produce a full-length window on every forward.
    fn assert_declared_window(&self, session: &Session, target: usize) -> Result<()> {
        let inputs = session.inputs();
        let Some(input) = inputs.first() else {
            return Err(Error::Model {
                detail: "the patched graph reports no inputs at all".to_string(),
            });
        };
        let actual = match input.dtype() {
            ValueType::Tensor { shape, .. } => shape.last().copied(),
            other => {
                return Err(Error::Model {
                    detail: format!(
                        "graph input `{}` is not a tensor ({other:?}); cannot read a window off it",
                        input.name()
                    ),
                })
            }
        };
        match actual {
            Some(v) if v == target as i64 => Ok(()),
            other => Err(Error::Model {
                detail: format!(
                    "patched the window to {target} but the session declares {:?} on input `{}`; \
                     the graph's shape does not agree with the patch, so nothing was run",
                    other,
                    input.name()
                ),
            }),
        }
    }

    /// Session for this call, rebuilt if the config moved since it was made.
    ///
    /// Rebuilding a patched session costs a re-read and a re-patch of the file
    /// (see [`load_bytes`](Self::load_bytes)): the alternative would be to keep a
    /// model-sized buffer alive between runs to serve a rebuild that happens at
    /// most once per process.
    fn session_mut(&mut self) -> Result<&mut Session> {
        let want = self.config;
        let fresh = match self.session.as_ref() {
            Some((_, built)) => *built != want,
            None => true,
        };
        if fresh {
            let bytes = self.load_bytes()?;
            let built = self.commit_bytes(&bytes)?;
            drop(bytes);
            self.session = Some((built, want));
        }
        Ok(&mut self.session.as_mut().expect("session was just built").0)
    }

    /// One forward: `[2, window]` in, the graph's `sources` out.
    ///
    /// `mix` is always the *full* window — the caller zero-pads a short tail and
    /// trims on output, so the runtime never sees a shape the graph was not built
    /// for. The result is **copied** out (≈ 1.4 MB for a 4 s window, against a
    /// forward that costs gigabytes and seconds) rather than borrowed: a borrow
    /// would keep a runtime output buffer alive across the whole merge-and-write
    /// step, which is exactly the memory this engine exists to avoid, and it would
    /// have to be released before the next `run` (which takes `&mut Session`)
    /// anyway.
    fn forward(&mut self, mix: &Array2<f32>) -> Result<Sources> {
        let window = self.window;
        if mix.dim() != (2, window) {
            return Err(Error::Output {
                detail: format!(
                    "internal shape error: the window buffer is {:?}, expected (2, {window})",
                    mix.dim()
                ),
            });
        }
        let started = Instant::now();
        // `ort`'s `ndarray` feature is deliberately off in this crate, so the
        // tensor is described by an explicit shape over the window's own slice.
        // `Array2::zeros` is C-contiguous, and `[1, 2, window]` is the shape the
        // graph's input declares.
        let data = mix.as_slice().ok_or_else(|| Error::Output {
            detail: "the input window is not contiguous".to_string(),
        })?;
        let input =
            TensorRef::from_array_view(([1usize, 2, window], data)).map_err(|e| Error::Output {
                detail: format!("could not wrap the input window as a tensor: {e}"),
            })?;
        let outputs = self.session_mut()?.run(ort::inputs![input]).map_err(|e| {
            // Sampled *here*, while the failure is still resident. By the time the
            // caller takes a snapshot of its own the session has been dropped, so
            // its numbers describe a machine that already got the commit back.
            let mem = mem::snapshot()
                .map(|s| s.summary())
                .unwrap_or_else(|| "unavailable".to_string());
            Error::Session {
                detail: format!("{e} [at failure: {mem}]"),
            }
        })?;
        let value = match outputs.get("sources") {
            Some(v) => v,
            None if outputs.len() != 0 => &outputs[0],
            None => {
                return Err(Error::Output {
                    detail: "the graph ran and produced no outputs".to_string(),
                })
            }
        };
        let (shape, flat) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Output {
                detail: format!("could not read the `sources` tensor as f32: {e}"),
            })?;
        if shape.len() != 4 || shape.iter().any(|d| *d < 0) {
            return Err(Error::Output {
                detail: format!(
                    "`sources` is {:?}; expected 4 dimensions ([1, source, channel, length]) \
                     with all of them known",
                    &shape[..]
                ),
            });
        }
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let expected = dims.iter().product::<usize>();
        if expected != flat.len() {
            return Err(Error::Output {
                detail: format!(
                    "`sources` declares {dims:?} ({} elements) but its buffer holds {} \
                     samples; the runtime's output does not match its own shape",
                    expected,
                    flat.len()
                ),
            });
        }
        if dims[0] != 1 || dims[1] < 2 || dims[2] < 2 || dims[3] != window {
            return Err(Error::Output {
                detail: format!(
                    "`sources` is {dims:?}; expected batch 1, at least 2 sources, at least 2 \
                     channels and a length equal to the window ({window})"
                ),
            });
        }
        if flat.iter().any(|v| !v.is_finite()) {
            return Err(Error::Output {
                detail: "the graph emitted a non-finite sample (NaN or inf)".to_string(),
            });
        }
        log::trace!(
            "[onnx] forward of a {window}-sample window took {:.3} s",
            started.elapsed().as_secs_f64()
        );
        Ok(Sources::new(flat.to_vec(), dims[1], dims[2], dims[3]))
    }
}

impl SeparationEngine for OnnxEngine {
    fn name(&self) -> &'static str {
        "onnx"
    }

    fn window_samples(&self) -> Result<usize> {
        Ok(self.window)
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
        let started = Instant::now();
        let window = self.window;
        let overlap = geometry(window, opts)?;

        self.preflight(window, opts)?;

        // `peak_mb_while` is the per-call figure: the OS-side peaks are maxima since
        // the process started, so a second run in the same process would otherwise
        // report the first one's high-water. See the module docs and `crate::mem`.
        let (result, peak_mb) =
            mem::peak_mb_while(|| self.run_windows(input, out, opts, window, overlap));
        let done = result?;
        log::info!(
            "[onnx] done: {} frames in {} window(s) ({} resumed), {:.1} s, peak {peak_mb} MB",
            done.frames,
            done.windows_inferred,
            done.windows_resumed,
            started.elapsed().as_secs_f64(),
        );
        Ok(SeparationReport {
            sample_rate: SAMPLE_RATE,
            frames: done.frames,
            windows_inferred: done.windows_inferred,
            windows_resumed: done.windows_resumed,
            resumed_from_frames: done.resumed_from_frames,
            peak_mb: Some(peak_mb).filter(|p| *p > 0),
            wall_ms: started.elapsed().as_millis(),
        })
    }
}

/// Resolve this call's window/overlap pair against the loaded graph.
///
/// Both checks are properties of the caller's options, and both are cheaper to state
/// as a function than to discover after a forward:
///
/// * [`SeparationOptions::window_samples`] is a *statement about the file*, not a
///   request the runtime can honour — the window is baked into the graph's constants,
///   so a disagreement means the wrong engine was loaded, and
///   [`OnnxEngine::with_window`] is what the caller actually wants.
/// * `overlap <= window / 2` is what keeps the not-yet-finalized region at one window
///   or less — the memory bound on the merge buffers, and the reason a resume replays
///   at most two windows to rebuild a seam. `chunk_starts` asserts it in a debug build;
///   in a release build an outside value would splice a resumed stem differently from an
///   uninterrupted one, which is silent and wrong. It does *not* promise two
///   contributors per frame: the appended tail window can land inside two earlier spans
///   and give a frame three, which the weight-sum division absorbs (see
///   `tests::the_grid_bounds_the_pending_region_and_characterizes_its_seams`).
fn geometry(window: usize, opts: &SeparationOptions) -> Result<usize> {
    if let Some(requested) = opts.window_samples {
        if requested != window {
            return Err(Error::Model {
                detail: format!(
                    "the loaded graph runs a {window}-sample window but the caller asked for \
                     {requested}; the window is baked into the graph's constants, so use \
                     OnnxEngine::with_window to reshape it in memory (or a file that already \
                     declares {requested})"
                ),
            });
        }
    }
    let overlap = opts
        .overlap_samples
        .unwrap_or_else(|| DEFAULT_OVERLAP.min(window / 2));
    if overlap == 0 || overlap > window / 2 {
        return Err(Error::Model {
            detail: format!(
                "the crossfade grid needs 0 < overlap <= window/2 ({} for this graph), got \
                 {overlap}; outside it a resumed stem is not the stem an uninterrupted run wrote",
                window / 2
            ),
        });
    }
    Ok(overlap)
}

/// What one finished loop did, before it is dressed as a [`SeparationReport`].
struct RunStats {
    frames: usize,
    windows_inferred: usize,
    windows_resumed: usize,
    resumed_from_frames: Option<usize>,
}

impl OnnxEngine {
    /// Refuse a window this machine cannot forward, before any work is spent.
    ///
    /// The estimate is exactly what its name says — see
    /// [`estimate_forward_mb`](Self::estimate_forward_mb) — and the log line says
    /// whether the window sits between the two measured anchors or outside them,
    /// because those two cases deserve different amounts of trust.
    fn preflight(&self, window: usize, opts: &SeparationOptions) -> Result<()> {
        let need = Self::estimate_forward_mb(window);
        let (t0, m0) = SHORT_ANCHOR;
        let (t1, m1) = LONG_ANCHOR;
        // Three different amounts of trust, and the log has to say which one the
        // number earned: between the anchors is interpolation, at one of them is the
        // measurement itself, and beyond either is the same curve extended into a
        // region nothing was ever measured in.
        let band = if window == t0 || window == t1 {
            "MEASURED at this window"
        } else if window < t0 {
            "EXTRAPOLATED below the short anchor"
        } else if window > t1 {
            "EXTRAPOLATED above the long anchor; expect the real figure to differ"
        } else {
            "interpolated between the two measured anchors"
        };
        log::info!(
            "[onnx] memory estimate for a {window}-sample forward: ~{need} MB commit [{band}; \
             measured: {t0} samples ≈ {m0:.0} MB and {t1} samples ≈ {m1:.0} MB, on a 16 GB-class \
             Windows machine, independent of track length]",
        );
        if self.is_patched() {
            log::info!(
                "[onnx] window {window} was rewritten in memory from the graph's own {}; the \
                 estimate above is the curve through the two measured anchors at this length, \
                 not a measurement of this patched graph",
                self.declared
            );
        }
        mem::window_fits(need, opts.memory_budget_mb)
    }

    /// The window loop: decode, forward, crossfade, write, checkpoint.
    fn run_windows(
        &mut self,
        input: &Path,
        out: &StemPaths,
        opts: &SeparationOptions,
        window: usize,
        overlap: usize,
    ) -> Result<RunStats> {
        let mut source = WavSource::open(input)?;
        let total = source.frames();
        let source_rate = source.source_rate();
        if source_rate != SAMPLE_RATE {
            log::info!(
                "[onnx] input is {source_rate} Hz; the windowed read normalises it to \
                 {SAMPLE_RATE} Hz by linear interpolation (see `WavSource`)"
            );
        }
        let starts = chunk_starts(total, window, overlap);
        let n = starts.len();
        log::info!(
            "[onnx] {} frames ({:.1} s) -> {n} window(s) of {window}, overlap {overlap}, hop {}",
            total,
            total as f64 / SAMPLE_RATE as f64,
            window - overlap,
        );

        // Identity of *this* run: geometry plus the two files' size and mtime.
        // A pair on disk that does not match it is truncated, not continued.
        let job = StemJob::for_run(total, &starts, window, overlap, input, &self.path);
        let mut stems = match opts.resume {
            ResumeMode::Auto => StemWriter::open_or_resume(&out.vocals, &out.background, &job)?,
            ResumeMode::Fresh => StemWriter::create(&out.vocals, &out.background)?,
        };
        let plan = resume_plan(&starts, total, window, stems.frames());
        stems.rewind_to(plan.keep_frames)?;
        if plan.keep_frames > 0 {
            log::info!(
                "[onnx] resuming: {} frames ({:.1} of {:.1} s) already finalized, re-inferring \
                 window(s) {}..{} for the crossfade and continuing at {}/{}",
                plan.keep_frames,
                plan.keep_frames as f64 / SAMPLE_RATE as f64,
                total as f64 / SAMPLE_RATE as f64,
                plan.prime_from,
                plan.first,
                plan.first,
                n,
            );
            mem::log_point(1, "onnx: resumed from checkpoint");
        }

        // Linear crossfade ramp: `w[k] = k / (overlap - 1)`. The fade-in uses it
        // forward and the fade-out uses it reversed, so a frame inside one seam gets
        // `ramp[k]` from the later window and `1 - ramp[k]` from the earlier one and the
        // two add to exactly 1. Where a third window also contributes — the appended
        // tail chunk — see
        // `tests::the_grid_bounds_the_pending_region_and_characterizes_its_seams`. The merge
        // is still a weighted mean because the weights are accumulated and
        // divided, not assumed.
        let denom = (overlap - 1).max(1) as f32;
        let ramp: Vec<f32> = (0..overlap).map(|i| i as f32 / denom).collect();

        // The pending, not-yet-finalized region [flushed_upto, pend_end). It never
        // exceeds one window, because the hop covers the overlap.
        let mut flushed_upto = plan.keep_frames;
        let mut pend_end = plan.keep_frames;
        let mut pv0 = vec![0.0f32; window];
        let mut pv1 = vec![0.0f32; window];
        let mut pb0 = vec![0.0f32; window];
        let mut pb1 = vec![0.0f32; window];
        let mut pw = vec![0.0f32; window];
        let mut mix = Array2::<f32>::zeros((2, window));
        let mut w = vec![1.0f32; window];

        let mut inferred = 0usize;
        for i in 0..n {
            // Windows below `prime_from` contribute nothing to what is left to do.
            if i < plan.prime_from {
                continue;
            }
            // The window(s) just below the seam are re-inferred to rebuild the
            // pending crossfade: they feed the buffers and finalize nothing.
            let priming = i < plan.first;
            let start = starts[i];

            if let Some(flag) = &opts.cancel {
                if flag.is_cancelled() {
                    // Staging stays exactly as the last `checkpoint()` left it, so a
                    // later run with the same geometry continues from here. Only
                    // `stream::discard_staging` removes it, and only a caller who
                    // means "throw this away" should call that.
                    log::info!(
                        "[onnx] cancelled at window {i}/{n}; {} frames are on disk and kept",
                        stems.frames()
                    );
                    return Err(Error::Cancelled);
                }
            }
            self.report_progress(opts, stems.frames(), total, i, n, "separating");

            // The full window goes to the graph; a short tail is zero-padded and its
            // padding is trimmed again at the flush (`take` bounds the merge), which
            // is what keeps every forward at the shape the graph was built for.
            let take = (total - start).min(window);
            mix.fill(0.0);
            {
                let mut view = mix.slice_mut(s![.., 0..take]);
                source.read(start, &mut view)?;
            }
            // One sample per window is enough to keep a long job's curve readable in
            // a log, and this is the only place a footprint reading is meaningful.
            if i == 0 || i % 32 == 0 {
                mem::log_point(i + 2, &format!("onnx: window {i}/{n} decoded"));
            }

            let sources = self.forward(&mix)?;

            fade_weights(&ramp, i, n, &mut w);

            debug_assert!(
                priming || start >= flushed_upto,
                "window {i} starts behind the flush point"
            );
            if start > pend_end {
                // A gap (currently unreachable from `chunk_starts`) is zero-filled so
                // its weight stays 0 and the sample comes out silent in both stems.
                let len = start - pend_end;
                let off = pend_end - flushed_upto;
                for buf in [&mut pv0, &mut pv1, &mut pb0, &mut pb1] {
                    buf[off..off + len].fill(0.0);
                }
                pw[off..off + len].fill(0.0);
            }
            pend_end = pend_end.max(start + take);
            let acc_from = start.max(flushed_upto);
            let off = acc_from - flushed_upto;
            let len = start + take - acc_from;
            let k0 = acc_from - start;
            let wv = &w[k0..k0 + len];
            // `sources` is [batch, source, channel, time]: source 0 is vocals, source 1
            // is the graph's own residual. Both are accumulated weighted; neither is
            // derived from the other here.
            for (ch, dst) in [(0usize, &mut pv0), (1, &mut pv1)] {
                let src = sources.row(0, ch).ok_or_else(|| missing_source(0, ch))?;
                accumulate(dst, off, &src[k0..k0 + len], wv);
            }
            for (ch, dst) in [(0usize, &mut pb0), (1, &mut pb1)] {
                let src = sources.row(1, ch).ok_or_else(|| missing_source(1, ch))?;
                accumulate(dst, off, &src[k0..k0 + len], wv);
            }
            for (d, wv) in pw[off..off + len].iter_mut().zip(wv.iter()) {
                *d += *wv;
            }
            inferred += 1;
            if priming {
                // The seam's earlier contribution is in the buffers now; the samples
                // it overlaps are already on disk and are not rewritten.
                continue;
            }

            // Everything strictly before the next window's start can no longer receive
            // a contribution, so it is final and goes to disk.
            let flush_end = if i < n - 1 {
                starts[i + 1].min(total)
            } else {
                total
            };
            debug_assert!(
                flushed_upto >= start && flush_end <= start + take,
                "flush range [{flushed_upto},{flush_end}) escaped window [{start},{})",
                start + take
            );
            for x in flushed_upto..flush_end {
                let j = x - flushed_upto;
                let wk = pw[j];
                let (v0, v1, b0, b1) = if wk > 1e-8 {
                    (pv0[j] / wk, pv1[j] / wk, pb0[j] / wk, pb1[j] / wk)
                } else {
                    // Uncovered sample: silence in both stems, never a partial sum.
                    (0.0, 0.0, 0.0, 0.0)
                };
                stems.write_frame([v0, v1], [b0, b1])?;
            }
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

            debug_assert_eq!(
                stems.frames(),
                flushed_upto,
                "written frames drifted from the flush point"
            );
            // Headers before the next forward: a process killed during inference
            // loses the window it was in, not the track.
            stems.checkpoint()?;
        }

        if stems.frames() != total {
            return Err(Error::Output {
                detail: format!(
                    "the flush loop finished with {}/{} frames written",
                    stems.frames(),
                    total
                ),
            });
        }
        self.report_progress(opts, total, total, n, n, "publishing");
        mem::log_point(n + 3, "onnx: all windows done, publishing");
        // Renames the `.part` pair into place, background first, so the vocals path
        // only ever appears next to a complete background stem.
        stems.finish()?;

        Ok(RunStats {
            frames: total,
            // Forwards actually run: the whole schedule minus the windows below
            // `prime_from`, which were neither inferred nor flushed here. Note that
            // the priming windows *are* counted — they really did run.
            windows_inferred: inferred,
            // Windows neither inferred nor written by this call — the ones below
            // `prime_from`, whose whole contribution was already flushed. This is
            // the figure that keeps `windows_inferred + windows_resumed == n`;
            // `plan.first` is *not* it, because the seam windows between there and
            // here ran again to rebuild the pending crossfade even though their
            // bytes came off disk. For "how much was already done" use
            // `resumed_from_frames`, which is non-zero whenever the resume reused
            // anything at all — including the case where it skipped nothing.
            windows_resumed: plan.prime_from,
            resumed_from_frames: (plan.keep_frames > 0).then_some(plan.keep_frames),
        })
    }

    /// `progress` is a percentage of *frames finalized*, which is the quantity a
    /// resumed run can actually pick up from; the window index rides along in the
    /// message because it is the thing a stuck run is described by.
    fn report_progress(
        &self,
        opts: &SeparationOptions,
        done: usize,
        total: usize,
        window: usize,
        windows: usize,
        phase: &str,
    ) {
        let Some(cb) = opts.progress.as_ref() else {
            return;
        };
        let pct = if total == 0 {
            100
        } else {
            (done.min(total) as u64 * 100 / total as u64) as i32
        };
        cb(
            pct,
            &format!("[onnx] {phase}: window {window}/{windows}, {pct}%"),
        );
    }
}

/// One window's worth of `sources`, in the graph's own layout.
///
/// `[batch, source, channel, length]` with `batch == 1`, so a row is one
/// contiguous run of `length` samples at `(source * channels + channel) * length`
/// in the flat buffer. Keeping the buffer flat rather than an `Array4` is what lets
/// the merge loop hand out plain slices — the shape checks happen once, when a
/// forward's output is adopted, and indexing after that cannot be wrong about the axis
/// order without being wrong here too.
pub struct Sources {
    data: Vec<f32>,
    sources: usize,
    channels: usize,
    length: usize,
}

impl Sources {
    fn new(data: Vec<f32>, sources: usize, channels: usize, length: usize) -> Sources {
        Sources {
            data,
            sources,
            channels,
            length,
        }
    }

    /// Samples for one source and channel across the whole window.
    pub fn row(&self, source: usize, channel: usize) -> Option<&[f32]> {
        if source >= self.sources || channel >= self.channels {
            return None;
        }
        let base = (source * self.channels + channel) * self.length;
        Some(&self.data[base..base + self.length])
    }

    /// How many sources the graph emitted (2 for a vocals checkpoint: vocals and
    /// its residual).
    pub fn source_count(&self) -> usize {
        self.sources
    }

    /// Samples per row, which equals the window the forward was given.
    pub fn length(&self) -> usize {
        self.length
    }
}

/// This window's crossfade envelope, written into `w` (which the loop reuses).
///
/// Fade in over the head, fade out over the tail, 1.0 in between — and *no* fade in
/// on the first window and *no* fade out on the last, because those edges are the
/// ends of the track: tapering them would silently attenuate the first and last
/// `overlap` frames of the stem with nothing to crossfade against.
///
/// With `hop >= overlap` a frame has at most two contributors, and where it has two
/// they are one window's tail and the next window's head, so the weights sum to
/// exactly 1 — which is why the accumulation divides by the weight sum rather than
/// normalising some other way. `adjacent_windows_partition_unity` pins it.
fn fade_weights(ramp: &[f32], i: usize, n: usize, w: &mut [f32]) {
    let overlap = ramp.len();
    let window = w.len();
    w.fill(1.0);
    if i > 0 {
        w[..overlap].copy_from_slice(ramp);
    }
    if i < n - 1 {
        // Mirrored, not copied: the tail has to *descend* from 1.0 to 0.0 so that it
        // and the next window's ascending head sum to exactly 1 at every frame of the
        // overlap. An ascending tail would make the normalised merge a weighted mean
        // dominated by the later window — a seam artefact at every hop, and silent in
        // any test that only compares two runs of the same code.
        for k in 0..overlap {
            w[window - 1 - k] = ramp[k];
        }
    }
    debug_assert!(
        overlap * 2 <= window,
        "the two fades would overlap each other"
    );
}

/// The graph promised at least two sources and handed us fewer.
fn missing_source(source: usize, channel: usize) -> Error {
    Error::Output {
        detail: format!(
            "no row for source {source} channel {channel}; the graph emitted fewer \
             sources or channels than the shape check accepted"
        ),
    }
}

/// `dst[off..off+len] += src * w`, elementwise.
///
/// A free function rather than a closure because the four calls borrow different
/// `pv*`/`pb*` buffers, and the borrow checker is happier with a function than with
/// a closure capturing all five.
fn accumulate(dst: &mut [f32], off: usize, src: &[f32], w: &[f32]) {
    let dst = &mut dst[off..off + w.len()];
    for ((d, s), w) in dst.iter_mut().zip(src.iter()).zip(w.iter()) {
        *d += *s * *w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model file is never shipped with the crate; a real-model arm runs only
    /// when the caller points at one, and says so when they do not.
    fn model_or_skip() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("RR_MODEL_ONNX").ok()?);
        if p.is_file() {
            Some(p)
        } else {
            eprintln!("[SKIP] RR_MODEL_ONNX points at no file: {}", p.display());
            None
        }
    }

    /// The `[onnx]` lines *are* the measurement in the real-model arms, so they
    /// have to reach the terminal rather than a dropped `log` sink. Installed once
    /// per test binary; a second call is the harmless case.
    fn install_logger() {
        struct Print;
        impl log::Log for Print {
            fn enabled(&self, _: &log::Metadata) -> bool {
                true
            }
            fn log(&self, r: &log::Record) {
                eprintln!("[{}] {}", r.level(), r.args());
            }
            fn flush(&self) {}
        }
        static PRINT: Print = Print;
        if log::set_logger(&PRINT).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    }

    /// The open question, measured.
    ///
    /// A patched graph keeps the `value_info` records it was exported with, and
    /// they describe the *old* window; deleting them would change the message
    /// length. Whether the runtime objects is a fact about ONNX Runtime, not a
    /// property of this code, so it is recorded here rather than assumed in the
    /// module docs.
    #[test]
    fn patched_graph_with_stale_value_info_is_accepted_by_the_runtime() {
        install_logger();
        let Some(path) = model_or_skip() else {
            eprintln!("[SKIP] RR_MODEL_ONNX is not set; nothing to patch");
            return;
        };
        let native = OnnxEngine::load(&path).expect("load the unpatched graph");
        let declared = native.declared_window();
        let records = native.value_info_records();
        let target = 4 * 44_100;
        if declared <= target {
            eprintln!("[SKIP] the file already declares {declared}; {target} would be growth");
            return;
        }
        let engine = OnnxEngine::with_window(&path, target)
            .expect("a shrunk window must build a session despite the stale records");
        assert_eq!(engine.window_samples().unwrap(), target);
        assert!(engine.is_patched());
        eprintln!(
            "[probe] VERDICT (measured): {records} stale value_info record(s) were TOLERATED — \
             the session built and declares a {target}-sample input"
        );
    }

    /// The pre-flight gate is the engine's only defence against an hour of partial
    /// work, so its own refusal has to be observable, not just documented.
    #[test]
    fn the_memory_gate_refuses_the_eight_second_window_on_a_sixteen_gigabyte_box() {
        install_logger();
        let Some(path) = model_or_skip() else {
            eprintln!("[SKIP] RR_MODEL_ONNX is not set; no window to price");
            return;
        };
        let mut engine = OnnxEngine::load(&path).expect("load");
        let window = engine.window_samples().unwrap();
        let need = OnnxEngine::estimate_forward_mb(window);
        let dir = scratch("gate");
        let out = StemPaths::new(dir.join("v.wav"), dir.join("b.wav"));
        let res = engine.separate(
            &dir.join("no_such_input.wav"),
            &out,
            &SeparationOptions::default(),
        );
        let err = match res {
            Ok(r) => panic!("an unestimated {need} MB forward returned {r:?}"),
            Err(e) => e,
        };
        let refused_for_memory = matches!(err, Error::Memory { .. });
        let missing_input_first = matches!(err, Error::Io { .. } | Error::Wav { .. });
        assert!(
            refused_for_memory || missing_input_first,
            "expected the gate or the missing input, got {err:?}"
        );
        eprintln!(
            "[probe] gate for a {window}-sample window (~{need} MB): {}",
            if refused_for_memory {
                "REFUSED before opening the input"
            } else {
                "allowed on this machine; the input was missing, which is the next check"
            }
        );
        assert!(
            !crate::stream::part_path(&out.vocals).exists(),
            "a refused run must not leave staging behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The engine end to end on a real graph and a real track: both stems, then a
    /// cancel in the middle and a resume that has to publish the same bytes an
    /// uninterrupted run wrote. Nothing here is skipped when both variables are set.
    #[test]
    fn real_model_separates_then_resumes_from_its_own_checkpoint() {
        install_logger();
        let Some(model) = model_or_skip() else {
            eprintln!("[SKIP] RR_MODEL_ONNX is not set; no model to separate with");
            return;
        };
        let dir = scratch("separate");
        let Some(track) = real_track(&dir, 12) else {
            eprintln!("[SKIP] RR_TEST_WAV is not set; no track to separate");
            return;
        };
        let frames = crate::stream::WavSource::open(&track)
            .expect("open the sliced track")
            .frames();

        // Prefer a patched-down window: it exercises the patch, the runtime's
        // tolerance of it, and a run whose memory this machine can actually cover.
        let target = 4 * 44_100;
        let mut engine = if declared_of(&model) > target {
            OnnxEngine::with_window(&model, target).expect("patched engine")
        } else {
            OnnxEngine::load(&model).expect("native engine")
        };
        let win = engine.window_samples().unwrap();
        let overlap = DEFAULT_OVERLAP.min(win / 2);
        let starts = chunk_starts(frames, win, overlap);
        eprintln!(
            "[probe] {} frames ({:.1} s) over {} window(s) of {win}/{overlap}",
            frames,
            frames as f64 / 44_100.0,
            starts.len()
        );

        // This machine is not the 16 GB Windows box the anchors came from, so the
        // gate is told what to allow explicitly; `Default::default()` is the arm the
        // gate test above exercises instead.
        let budget = || SeparationOptions {
            memory_budget_mb: Some(OnnxEngine::estimate_forward_mb(win)),
            ..Default::default()
        };

        let refs = StemPaths::new(dir.join("ref_v.wav"), dir.join("ref_b.wav"));
        let report = engine
            .separate(
                &track,
                &refs,
                &SeparationOptions {
                    resume: ResumeMode::Fresh,
                    ..budget()
                },
            )
            .expect("separate");
        assert_eq!(report.frames, frames, "a stem must be the track's length");
        assert_eq!(report.sample_rate, SAMPLE_RATE);
        assert_eq!(report.windows_inferred, starts.len());
        assert_eq!(report.windows_resumed, 0);
        assert_eq!(report.resumed_from_frames, None);
        assert!(report.wall_ms > 0);
        eprintln!(
            "[probe] report: {} frames, {} window(s), peak {:?} MB, {} ms",
            report.frames, report.windows_inferred, report.peak_mb, report.wall_ms
        );
        for path in [&refs.vocals, &refs.background] {
            let r =
                hound::WavReader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(r.spec().channels, 2, "stems are stereo");
            assert_eq!(r.spec().sample_rate, SAMPLE_RATE);
            assert_eq!(r.len() as usize, frames * 2, "two values per frame");
        }
        // Source 1 has to be the residual the graph emits, not something else in
        // the tensor: `vocals + background` must reproduce the mix.
        let mut mix = crate::stream::WavSource::open(&track).expect("re-open the mix");
        let mut v = crate::stream::WavSource::open(&refs.vocals).expect("re-open vocals");
        let mut b = crate::stream::WavSource::open(&refs.background).expect("re-open background");
        let probe = 8_192usize;
        let (mut m, mut vx, mut bx) = (
            Array2::<f32>::zeros((2, probe)),
            Array2::<f32>::zeros((2, probe)),
            Array2::<f32>::zeros((2, probe)),
        );
        // Past the first window's fade-in, where a single window dominates.
        let at = win;
        for (src, buf) in [(&mut mix, &mut m), (&mut v, &mut vx), (&mut b, &mut bx)] {
            let mut view = buf.view_mut();
            src.read(at, &mut view)
                .unwrap_or_else(|e| panic!("read at {at}: {e}"));
        }
        let mut worst = 0.0f32;
        for k in 0..probe {
            for ch in 0..2 {
                // Both stems are 16-bit on disk, so the floor on the agreement is
                // the quantisation of each, not the model's arithmetic.
                let d = (vx[[ch, k]] + bx[[ch, k]] - m[[ch, k]]).abs();
                worst = worst.max(d);
            }
        }
        eprintln!("[probe] |vocals + background - mix| worst in {probe} frames: {worst}");
        assert!(
            worst < 0.02,
            "source 1 is not the mix's residual: {worst} of full scale"
        );

        // ── cancel in the middle, then resume ──
        let flag = crate::config::CancelFlag::new();
        let seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cb: crate::config::Progress = {
            let flag = flag.clone();
            let seen = std::sync::Arc::clone(&seen);
            std::sync::Arc::new(move |_, _| {
                if seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 2 {
                    flag.cancel();
                }
            })
        };
        let stems = StemPaths::new(dir.join("v.wav"), dir.join("b.wav"));
        let err = engine
            .separate(
                &track,
                &stems,
                &SeparationOptions {
                    progress: Some(cb),
                    cancel: Some(flag),
                    ..budget()
                },
            )
            .expect_err("the flag is set, so the run must stop");
        assert!(
            matches!(err, Error::Cancelled),
            "expected Cancelled, got {err:?}"
        );
        assert!(
            crate::stream::part_path(&stems.vocals).exists(),
            "cancellation must leave the staging pair for the next run"
        );
        assert!(
            crate::stream::job_path(&stems.vocals).exists(),
            "and its sidecar"
        );
        assert!(
            !stems.vocals.exists(),
            "the final names stay absent until a run completes"
        );

        let resumed = engine
            .separate(&track, &stems, &budget())
            .expect("the resumed run must finish");
        assert_eq!(resumed.frames, frames);
        // The reuse is stated in frames and as a partition, not as "skipped more
        // than nothing": a run cancelled at its first boundary has one window of
        // output on disk and zero windows *skipped*, because that window is
        // re-inferred as seam priming — bytes reused, count unmoved.
        assert!(
            resumed.resumed_from_frames.is_some(),
            "the continuing pass wrote the track from sample zero: {resumed:?}"
        );
        assert_eq!(
            resumed.windows_inferred + resumed.windows_resumed,
            starts.len(),
            "the two counts must partition the schedule: {resumed:?} over {} window(s)",
            starts.len()
        );
        for (a, b) in [
            (&refs.vocals, &stems.vocals),
            (&refs.background, &stems.background),
        ] {
            assert_eq!(
                std::fs::read(a).unwrap(),
                std::fs::read(b).unwrap(),
                "{} != {}: a resume must reproduce the bytes an uninterrupted run wrote",
                a.display(),
                b.display()
            );
        }
        eprintln!(
            "[probe] resume: {} frames off disk, {} window(s) re-inferred of {}",
            resumed.resumed_from_frames.unwrap(),
            resumed.windows_inferred,
            starts.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─────────────────────────── no model required ───────────────────────────
    //
    // The geometry, the price curve and the output layout are all decided by code in
    // this file, so they are pinned here rather than left to a run that needs a
    // 259 MB download. The real-model arms below are the only tests that can say
    // anything about ONNX Runtime.

    #[test]
    fn the_price_curve_reproduces_both_anchors_and_is_quadratic() {
        // The anchors are measurements; anything else is the curve through them.
        assert_eq!(OnnxEngine::estimate_forward_mb(SHORT_ANCHOR.0), 5_100);
        assert_eq!(OnnxEngine::estimate_forward_mb(LONG_ANCHOR.0), 19_400);
        assert_eq!(OnnxEngine::estimate_forward_mb(0), 0, "no window, no cost");

        // Doubling the window quadruples the attention term, so the *estimate* more
        // than quadruples below 2x because of the constant, and lands on the second
        // anchor exactly at 2x.
        let four = OnnxEngine::estimate_forward_mb(176_400) as f64;
        let eight = OnnxEngine::estimate_forward_mb(352_800) as f64;
        let twelve = OnnxEngine::estimate_forward_mb(529_200) as f64;
        assert!(
            (eight / four - 3.8).abs() < 0.2,
            "8 s / 4 s = {:.2}, expected ~3.8 from c + q·T²",
            eight / four
        );
        assert!(
            twelve > eight * 2.0,
            "12 s should cost more than twice 8 s on a quadratic: {eight} -> {twelve}"
        );
        // Monotone, because the gate is only useful if a bigger window is never
        // cheaper than the one that already fitted.
        let mut last = 0;
        for win in (1..=12).map(|k| k * 44_100) {
            let mb = OnnxEngine::estimate_forward_mb(win);
            assert!(mb > last, "{win} samples priced at {mb} after {last}");
            last = mb;
        }
        // Outside the anchors the number is an extrapolation; the gate must still
        // refuse an absurd one when it is told the ceiling.
        let huge = OnnxEngine::estimate_forward_mb(4 * 352_800);
        assert!(
            mem::window_fits(huge, Some(16_000)).is_err(),
            "a ~{huge} MB forward must not pass a 16 GB budget"
        );
        assert!(
            mem::window_fits(OnnxEngine::estimate_forward_mb(176_400), Some(16_000)).is_ok(),
            "the 4 s window must pass a 16 GB budget"
        );
    }

    #[test]
    fn the_caller_s_grid_is_refused_before_any_forward() {
        let win = 176_400;
        // The default: 1.5 s, which is the geometry the flush proofs are stated at.
        assert_eq!(
            geometry(win, &SeparationOptions::default()).unwrap(),
            66_150
        );
        // A window shorter than the default overlap still gets a legal grid.
        assert_eq!(
            geometry(44_100, &SeparationOptions::default()).unwrap(),
            22_050,
            "half the window, not a third of a second"
        );
        // An override that disagrees with the graph is a wrong file, not a request.
        let e = geometry(
            win,
            &SeparationOptions {
                window_samples: Some(352_800),
                ..Default::default()
            },
        )
        .expect_err("the graph cannot run a window it does not declare");
        assert!(matches!(e, Error::Model { .. }), "{e:?}");
        assert!(e.to_string().contains("with_window"), "{e}");
        assert_eq!(
            geometry(
                win,
                &SeparationOptions {
                    window_samples: Some(win),
                    ..Default::default()
                }
            )
            .unwrap(),
            66_150,
            "agreeing with the graph is the no-op it should be"
        );
        // Overlap past half the window breaks the resume invariant.
        for bad in [0usize, 100_000, win / 2 + 1] {
            let e = geometry(
                win,
                &SeparationOptions {
                    overlap_samples: Some(bad),
                    ..Default::default()
                },
            )
            .expect_err("{bad} must be refused");
            assert!(matches!(e, Error::Model { .. }), "{e:?}");
        }
        assert_eq!(
            geometry(
                win,
                &SeparationOptions {
                    overlap_samples: Some(win / 2),
                    ..Default::default()
                }
            )
            .unwrap(),
            win / 2,
            "exactly half is the boundary the grid allows"
        );
    }

    /// The envelope: linear in, linear out, flat in between, and *no* taper on the
    /// two edges that are the ends of the track.
    #[test]
    fn fade_envelope_is_linear_and_leaves_the_track_ends_untouched() {
        for &(win, overlap) in &[
            (176_400usize, 66_150usize),
            (352_800, 110_250),
            (44_100, 22_050),
        ] {
            let denom = (overlap - 1).max(1) as f32;
            let ramp: Vec<f32> = (0..overlap).map(|i| i as f32 / denom).collect();
            let mut w = vec![1.0f32; win];

            fade_weights(&ramp, 0, 1, &mut w);
            assert!(
                w.iter().all(|&x| x == 1.0),
                "a one-window track is not tapered"
            );

            // The first window of the track has nothing before it, so its head is
            // not tapered — but its tail is, because the next window crosses into it.
            fade_weights(&ramp, 0, 4, &mut w);
            assert!(
                w[..overlap].iter().all(|&x| x == 1.0),
                "the first window must not fade in"
            );
            assert_eq!(w[win - 1], 0.0, "the fade-out reaches 0 at the window edge");
            assert_eq!(w[win - overlap], 1.0, "and starts from full weight");
            // The flat region between the two fades exists only where the window is
            // more than twice the overlap; at `overlap == win/2` they meet exactly.
            for k in 0..(win - 2 * overlap) {
                assert_eq!(w[overlap + k], 1.0, "interior is flat at {k}");
            }

            // A mid-list window fades both ways, and the two ramps meet without
            // overlapping each other.
            fade_weights(&ramp, 1, 4, &mut w);
            assert_eq!(w[0], 0.0);
            assert_eq!(
                w[win - overlap],
                1.0,
                "the flat region starts where the fade-in ends"
            );
            assert_eq!(w[win - overlap - 1], 1.0);
            assert_eq!(w[win - 1], 0.0);
        }
    }

    /// Neighbouring windows add to exactly 1 across the seam, which is what lets the
    /// merge divide by an accumulated weight and stay a *linear* crossfade rather than a
    /// normalisation that quietly reshapes the envelope. Checked at every offset of the
    /// overlap, for both grid bindings: reversing the fade-out wrongly fails at the ends
    /// of the range rather than in the middle, where a spot check would not see it.
    #[test]
    fn adjacent_windows_partition_unity() {
        for &(win, overlap) in &[
            (176_400usize, 66_150usize),
            (352_800, 110_250),
            (44_100, 22_050),
        ] {
            let hop = win - overlap;
            let denom = (overlap - 1).max(1) as f32;
            let ramp: Vec<f32> = (0..overlap).map(|i| i as f32 / denom).collect();
            let mut a = vec![1.0f32; win];
            let mut b = vec![1.0f32; win];
            fade_weights(&ramp, 1, 4, &mut a);
            fade_weights(&ramp, 2, 4, &mut b);
            // Frame `hop + k` of window 1 is frame `k` of window 2.
            for k in 0..overlap {
                let sum = a[hop + k] + b[k];
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "win={win}: at seam offset {k}, {} + {} = {sum}",
                    a[hop + k],
                    b[k]
                );
            }
            // And the envelope really is linear: consecutive samples differ by one step.
            for k in 1..overlap {
                let d = b[k] - b[k - 1];
                assert!(
                    (d - 1.0 / denom).abs() < 1e-6,
                    "win={win}: step at {k} is {d}"
                );
            }
            assert_eq!(ramp[0], 0.0);
            assert_eq!(ramp[overlap - 1], 1.0);
        }
    }

    /// What the grid has to guarantee for the merge loop, characterized rather than
    /// asserted loosely, over both bindings and the awkward totals.
    ///
    /// * What a window leaves behind its flush point is a subset of that window — one
    ///   window of buffers, never a track of them — and on a regular grid (every gap
    ///   exactly one hop) it tightens to `overlap`. An appended final window lands
    ///   closer than a hop, and the region it leaves is correspondingly wider; that is
    ///   still bounded by the window, which is the only thing the buffers promise.
    /// * Every frame is covered at least once and at most three times. Two is the
    ///   regular grid; a third contributor appears only where an appended final window
    ///   lands inside *two* earlier spans, i.e. only at the end of the track. The scan
    ///   requires such a total to show up, so the case the weight-sum division exists
    ///   for stays covered instead of quietly vacuous.
    #[test]
    fn the_grid_bounds_the_pending_region_and_characterizes_its_seams() {
        for &(win, overlap) in &[
            (176_400usize, 66_150usize),
            (352_800, 110_250),
            (44_100, 22_050),
        ] {
            let hop = win - overlap;
            assert!(hop >= overlap, "grid {win}/{overlap}");
            let mut any_three = false;
            for total in [
                0usize,
                1,
                win / 2,
                win,
                win + 1,
                win + hop,
                win + hop + 1,
                2 * win,
                3 * win,
                win + win / 3,
                5 * hop + win / 2,
                44_100 * 600,
            ] {
                let starts = chunk_starts(total, win, overlap);
                assert_eq!(starts[0], 0, "win={win} total={total}");
                let last = *starts.last().unwrap();

                let mut touched = vec![0u16; total.max(1)];
                let mut leftover_max = 0usize;
                let regular = starts.windows(2).all(|w| w[1] - w[0] == hop);
                for (i, &st) in starts.iter().enumerate() {
                    let take = (total - st).min(win);
                    // What survives the flush this window performs: the part of its
                    // window at or beyond the next window's start. The last window
                    // finalizes to the end of the track, so it leaves nothing behind.
                    let flush_end = starts.get(i + 1).copied().unwrap_or(st + take).min(total);
                    leftover_max = leftover_max.max((st + take).saturating_sub(flush_end));
                    for k in 0..take {
                        touched[st + k] += 1;
                    }
                }
                // The buffers are `window` wide, and that is the bound that holds for
                // every schedule: the region left behind is a subset of the window that
                // was just merged. On a *regular* grid it tightens to `overlap`, because
                // then the next start is exactly one hop away.
                assert!(
                    leftover_max <= win,
                    "win={win} total={total}: {leftover_max} frames left behind the flush point \
                     — wider than the window, so the merge buffers would have to grow with it"
                );
                if regular {
                    assert!(
                        leftover_max <= overlap,
                        "win={win} total={total}: a regular grid left {leftover_max} > {overlap} \
                         frames pending"
                    );
                }
                let mut three = 0;
                for (f, &n) in touched.iter().enumerate().take(total) {
                    assert!(n >= 1, "win={win} total={total}: frame {f} is in no window");
                    assert!(
                        n <= 3,
                        "win={win} total={total}: frame {f} has {n} contributors"
                    );
                    if n == 3 {
                        three += 1;
                        assert!(
                            f >= last,
                            "win={win} total={total}: frame {f} has three contributors before \
                             the final window even starts at {last}"
                        );
                    }
                }
                any_three |= three > 0;
            }
            assert!(
                any_three,
                "grid {win}/{overlap}: none of the scanned totals produced a three-way seam, \
                 so the case the weight-sum division exists for stopped being covered"
            );
        }
    }

    /// The boundary cases the schedule has to get right, restated for this engine's
    /// default grid: a short track is one window, a window-plus-one appends a second
    /// that ends flush with the track, and an exact multiple does not append.
    #[test]
    fn schedule_boundaries_on_the_default_grid() {
        let win = 176_400;
        let overlap = DEFAULT_OVERLAP.min(win / 2);
        let hop = win - overlap;
        assert_eq!(chunk_starts(0, win, overlap), vec![0]);
        assert_eq!(chunk_starts(1, win, overlap), vec![0]);
        assert_eq!(chunk_starts(win - 1, win, overlap), vec![0]);
        assert_eq!(chunk_starts(win, win, overlap), vec![0]);
        assert_eq!(chunk_starts(win + 1, win, overlap), vec![0, 1]);
        assert_eq!(chunk_starts(hop, win, overlap), vec![0]);
        assert_eq!(chunk_starts(win + hop, win, overlap), vec![0, hop]);
        let n = 4 * win;
        let exact = chunk_starts(n, win, overlap);
        assert_eq!(exact.last().copied().unwrap(), n - win);
        assert_eq!(exact.len(), 1 + (n - win) / hop + 1, "{exact:?}");
    }

    /// The row layout is the one place where a wrong axis order still produces audio:
    /// it would just be the wrong audio, at the right length. So the index arithmetic
    /// is pinned against a buffer whose contents say where they are.
    #[test]
    fn sources_rows_follow_the_declared_axis_order() {
        // [1, 2, 2, 3]: source-major, then channel, then three samples.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let s = Sources::new(data, 2, 2, 3);
        assert_eq!(s.source_count(), 2);
        assert_eq!(s.length(), 3);
        assert_eq!(s.row(0, 0).unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(s.row(0, 1).unwrap(), &[3.0, 4.0, 5.0]);
        assert_eq!(s.row(1, 0).unwrap(), &[6.0, 7.0, 8.0]);
        assert_eq!(s.row(1, 1).unwrap(), &[9.0, 10.0, 11.0]);
        assert!(s.row(2, 0).is_none(), "a third source was not emitted");
        assert!(s.row(0, 2).is_none(), "a third channel was not emitted");

        // The accumulator writes where the schedule says, scaled by the envelope.
        let mut dst = vec![0.0f32; 5];
        accumulate(&mut dst, 1, &[10.0, 20.0], &[0.25, 0.5]);
        assert_eq!(dst, vec![0.0, 2.5, 10.0, 0.0, 0.0]);
    }

    /// The trait exists in this shape so an engine can be handed around as a
    /// `Box<dyn SeparationEngine>`; that is a compile-time property, so assert it
    /// at compile time rather than trusting a comment. Building the box is what
    /// fails if someone adds an associated constant or a generic method.
    #[test]
    fn the_engine_is_dyn_compatible() {
        fn as_trait_object(engine: OnnxEngine) -> Box<dyn SeparationEngine> {
            Box::new(engine)
        }
        fn name_of(b: &dyn SeparationEngine) -> &'static str {
            b.name()
        }
        // Nothing here touches a model file: the coercion is the assertion.
        let build: fn(OnnxEngine) -> Box<dyn SeparationEngine> = as_trait_object;
        let read: fn(&dyn SeparationEngine) -> &'static str = name_of;
        let _ = (build, read);
    }

    /// What the file on disk declares, so a test can ask for a window the graph can
    /// actually be shrunk to.
    fn declared_of(model: &Path) -> usize {
        let bytes = std::fs::read(model).expect("read the model");
        crate::graph::inspect(&bytes)
            .map(|r| r.declared_window as usize)
            .unwrap_or(0)
    }

    /// A caller-supplied track, sliced to `seconds` of real audio.
    ///
    /// The slice is a prefix copied sample for sample — not a synthetic stand-in.
    /// A whole song is simply not what a window-loop test needs.
    fn real_track(dir: &Path, seconds: usize) -> Option<PathBuf> {
        let src = PathBuf::from(std::env::var("RR_TEST_WAV").ok()?);
        if !src.is_file() {
            eprintln!("[SKIP] RR_TEST_WAV points at no file: {}", src.display());
            return None;
        }
        let out = dir.join("track.wav");
        let mut r =
            hound::WavReader::open(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
        let spec = r.spec();
        let depth = (spec.sample_format, spec.bits_per_sample);
        let want = (seconds * spec.sample_rate as usize)
            .min(r.len() as usize / spec.channels as usize)
            .max(1);
        let mut w = hound::WavWriter::create(&out, spec).expect("create the sliced track");
        match depth {
            (hound::SampleFormat::Int, 16) => {
                for s in r.samples::<i16>().take(want * spec.channels as usize) {
                    w.write_sample(s.expect("read a sample")).unwrap();
                }
            }
            (hound::SampleFormat::Float, 32) => {
                for s in r.samples::<f32>().take(want * spec.channels as usize) {
                    w.write_sample(s.expect("read a sample")).unwrap();
                }
            }
            other => {
                eprintln!(
                    "[SKIP] RR_TEST_WAV is {other:?}; this slicer reads 16-bit PCM and f32 only"
                );
                return None;
            }
        }
        w.finalize().unwrap();
        Some(out)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rust_roformer_onnx_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }
}
