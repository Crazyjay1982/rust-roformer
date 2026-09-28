# rust-roformer

Mel-Band RoFormer vocal/background separation in Rust.

The model is not ours — the architecture is [ZFTurbo's](https://github.com/ZFTurbo/Music-Source-Separation-Training)
and the checkpoint is [KimberleyJSN's](https://huggingface.co/KimberleyJSN/melbandroformer).
What is ours is everything around it: two inference engines and an I/O layer that
separates a four-hour track on a 16 GB laptop without ever holding the track, the
intermediate activations, or the result in memory at the same time. If you want the
model, download it; if you want a Rust separation step that does not OOM at 02:00
on a laptop, this is the interesting part. Where that claim overlaps something
that already exists, [Related work](#related-work) links it.

**Which RoFormer.** "RoFormer" is an overloaded name. Here it means *Mel-Band
RoFormer*, the band-split separation model above — not the NLP RoFormer that the
rotary-embedding paper is named after, and not **BS-RoFormer**. The two audio
models share an idea and differ in how the frequency axis is grouped (fixed bin
groups vs a mel filterbank), so **their weights are not interchangeable**: a
`melband_*` checkpoint will not load into a BS-RoFormer implementation, whatever
the name in the README says.

```
cargo add rust-roformer            # or: git = "…"
```

## The one number that explains the design

A Mel-Band RoFormer forward pass is a static-shape graph whose time-axis
attention is `batch × heads × T × T`. Peak memory of **one window** is therefore
enormous and **independent of track length** — measured on the stock 8-second
export:

| Window | Commit per forward | Resident per forward | Time term |
| --- | --- | --- | --- |
| 8 s (stock export) | ~19.4 GB | ~10.7 GB | ~20 MB/min |
| 4 s (window shortened at load time) | ~5.1 GB | ~3.7 GB | ~20 MB/min |

*Windows, 16 GB-class machine, product harness; the ONNX Runtime arena was later
disabled, which moved the product path's own peak to 3,440 MB with a 349 MB
plateau. See [docs/benchmarks.md](docs/benchmarks.md) for every number's
provenance — several are Apple Silicon and do not transfer.*

That table is why this crate exists. A 19.4 GB single allocation on a 16 GB
machine does not fail politely, and it fails after you have paid for decoding and
resampling. So the crate's job is: **decide before allocating, stream both
directions, and make the work resumable.**

## What it does

* **Refuses instead of OOM-ing.** `mem::snapshot()` asks the OS what it has;
  the engine compares that against the window it is about to run and returns
  `Error::Memory { need_mb, avail_mb }` in microseconds. On macOS the figure is
  `phys_footprint`; on Windows it is available *commit* — different quantities,
  and the code says which one it read.
* **Streams in.** `audio::WavSource` decodes one window at a time; the resampler
  (`rubato`, chunked with one filter configuration) never builds a
  source-rate copy of the whole track. Reading 45.8 minutes used to peak at
  1,094 MB and now at 169 MB.
* **Streams out.** `stream::StemWriter` writes both stems in one pass and
  rewrites its own 44-byte WAV header at every window boundary — so at any
  instant the files on disk are a *playable, complete prefix*, not a headless
  blob. `Drop` cannot leak a half-written final file.
* **Resumes.** Kill it, lose power, hit stop: the next call with the same paths
  continues from the last flushed window. The job identity (input size+mtime,
  model size+mtime, window, overlap) is compared field by field first, and a
  mismatch means *start over* rather than "silently continue with a stale
  prefix" or "fail while leaving junk". A resumed run is **byte-identical** to an
  uninterrupted one — that is a test, not a claim.
* **Observes cancellation between windows.** No runtime gives us its decode
  loop, so stop costs at most one window. Keeping windows small is what makes
  that latency acceptable. Stopping does not throw the work away: no failure this
  crate can produce deletes a checkpoint, because a caller who pressed stop might
  have meant "later", and `stream::discard_staging` is there for the ones who did
  not.
* **No whole-track buffers anywhere in the pipeline.** The MLX engine's host
  peak on a 45.8-minute track went 3,405 MB → 72 MB with device-side figures
  unchanged value for value; separation quality was accepted at
  *identical-to-the-bit*, not "correlated enough".

## Engines

| | `onnx` (default) | `mlx` (feature `mlx`) |
| --- | --- | --- |
| What runs | exported graph, via ONNX Runtime | architecture re-implemented in Rust, via MLX |
| Platforms | Windows / Linux / macOS, CPU (CUDA/CoreML EPs behind features) | macOS on Apple Silicon |
| Builds from a clean clone | yes (`--features onnx`, the default) | **not without provisioning MLX** — see below |
| Weights | `.onnx` you download | `.safetensors` produced by `tools/extract_onnx_weights.py` from the same `.onnx` |
| Window | read from the graph; shorten it at load time with `OnnxEngine::with_window(path, samples)` | 8 s (`352800` samples @ 44.1 kHz), 2.5 s crossfade |
| Acceptance | vs the Python implementation on the same schedule: corr 0.99997 | vs a PyTorch FFT reference: corr 0.9999997 |

The MLX engine is the one to read if you came for "Rust port". It has its own
STFT, its own band-split, alternating time/freq transformer blocks and the mask
estimator, and it carries comments about the handful of numerical conventions
where a "reasonable" implementation is silently wrong — L2Norm scaling with
`sqrt(dim)` rather than RMS, exact erf GELU, `traditional` RoPE with
`base=10000`, manual softmax attention instead of the fused path.

Before you try `--features mlx`: `mlx-rs` pulls `mlx-sys`, whose build script runs
CMake and fetches the MLX C++ library from source. That command therefore needs a
C++ toolchain, CMake, network access and a long compile — or an MLX install to link
against. Measured here: the default build is green, `--features mlx` fails inside
the C++ build before any code in this crate is reached. The engine itself was
compiled and tested through a scratch manifest pointed at an MLX install that
already existed on the machine, and its results are reported beside the default
build rather than folded into it. The feature is never quietly disabled to make a
build look green.

Its acceptance target is deliberately **not** the ONNX graph's output: that
export computes its STFT as a convolution, and its fp32 accumulation error is
amplified by per-band L2Norm and cascaded through the transformer stack.
Arbitrated with a float64 FFT, the MLX path matched PyTorch (0.9999997) while the
ONNX path sat around 0.70. Comparing your port against the thing you are porting
away from would have hidden that.

## Quick start

```rust
use rust_roformer::config::{SeparationOptions, StemPaths};
use rust_roformer::engine::{onnx::OnnxEngine, SeparationEngine};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let mut engine = OnnxEngine::load(Path::new("melband_roformer_vocals.onnx"))?;
    println!("model window: {} samples", engine.window_samples()?);

    let report = engine.separate(
        Path::new("song.wav"),
        &StemPaths::new("vocals.wav", "background.wav"),
        &SeparationOptions::default(),
    )?;
    println!(
        "{} frames, {} windows, peak {:?}",
        report.frames, report.windows_inferred, report.peak_mb
    );
    Ok(())
}
```

The same thing through the shipped harness, which also reports windows, wall
clock and peak memory — and, with `--window`, does the load-time reshape below on
the command line so you can measure the two windows against the same file:

```sh
cargo run --release --example bench -- run --model melband_roformer_vocals.onnx \
    --input song.wav --out bench-out
cargo run --release --example bench -- run --model melband_roformer_vocals.onnx \
    --input song.wav --out bench-out --window 176400
```

Every row of that TSV ends with the window read back off the live session, not
echoed from the flag: a timing table should state the grid it timed.

## Getting a model

Nothing in this repository is a model file, and none will be accepted in a pull
request. The reference export is

* `smank/mel-band-roformer-vocals-onnx` → `melband_roformer_vocals.onnx`
  — 953,292,899 bytes,
  sha256 `64a4f3bee48fbe7d971b23875adc924ed004c3533f49672592641dddc0f6f561`

Download it, check the hash, and use it as it is. Nothing has to be converted
first, because the export's stock window is the one it was trained with:

```rust
// still inside `main`, still the same file on disk:
// the portable engine, at whatever window this file declares
let full = OnnxEngine::load(Path::new("melband_roformer_vocals.onnx"))?;
// or run the same weights on a 16 GB machine, without writing a new file
let small = OnnxEngine::with_window(Path::new("melband_roformer_vocals.onnx"), 176_400)?;
```

`with_window` is the library doing window surgery on a buffer at load time. A
Mel-Band RoFormer export fixes its window in *shape* — four `int64` `Constant`
nodes plus the input/output dimensions, which is why ONNX Runtime exposes no knob
for it — and those six numbers are rewritten in place and handed to
`commit_from_memory`. No weight byte is involved; the buffer is dropped once the
session owns what it needs, so the transient is one copy of the file. Two other
steps do need a file of their own — the MLX engine loads extracted tensors, and
you may want the shortened graph on disk to inspect it or to skip re-parsing on
every start:

```sh
# the 672 tensors the MLX engine loads, from the same stock export
python3 tools/extract_onnx_weights.py melband_roformer_vocals.onnx \
    mlx_roformer_vocals_fp32.safetensors

# the same window edit, written out; --report shows how a file bakes its window
python3 tools/reduce_window.py melband_roformer_vocals.onnx shortened_4s.onnx 176400
python3 tools/reduce_window.py --report melband_roformer_vocals.onnx
```

**Shrinking is exact; growing is refused.** The iSTFT overlap-add normalisation
initializer is `baked_window + 2048` samples long and is sliced at runtime, so
reusing its prefix is provably identical to a native export at that window (the
first sample where a discarded frame could contribute sits one hop *past* where
the graph trims), while growing would need that table regenerated — [`graph`]
refuses, and says so. What it does not refuse is a *stale* shape cache: patched
graphs keep the cached `value_info` records they were exported with, and ONNX
Runtime was measured to accept the session anyway — four `MergeShapeInfo …
Falling back to lenient merge` warnings naming the tensors we consume, then
correct output. That is why the engine asserts the session's own declared
dimensions instead of trusting that.

`tools/reduce_window.py` shares the rules above and adds two of its own: it is
shrink-only for the same table reason (`--force` overrides, and you should not),
and it strips the now-stale cached shapes so the output is byte-comparable with
the reference artifact — `--shape-metadata patch` keeps them consistent instead.

A derivative of a weight file carries the *weights'* terms, not this
repository's Apache-2.0 — and see [docs/LICENSES.md](docs/LICENSES.md) for the
one question about these checkpoints that no MIT tag in the chain answers.

## Measuring it

```sh
# two files
./scripts/bench_pair.sh --model-a melband_roformer_vocals.onnx \
    --model-b shortened_4s.onnx --input track.wav
# one file, two windows — the same claim, measured without a second artifact
./scripts/bench_pair.sh --model-a melband_roformer_vocals.onnx \
    --model-b melband_roformer_vocals.onnx --window-b 176400 --input track.wav
```

Read `docs/benchmarks.md` before you trust any timing number, including the ones
in this file. The short version, learned the expensive way on this crate's own
MLX engine: the **same build, same input, same machine** measured 88 s in one
session and 158 s in another. Absolute wall clock from two sessions is not an
instrument; only same-session, order-swapped pairs are, and
`scripts/bench_pair.sh` enforces that shape (it runs each configuration first and
second, drops contaminated arms, and reports medians rather than a winner).

## Related work

This is not a lonely field, and the useful comparison is with projects, not with
an empty list. What follows was checked against each project's own README and
source on 2026-09-28, which is also the date after which the statements below may
have gone stale.

| Project | What it already does |
| --- | --- |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | The de-facto tool for these checkpoints: runs RoFormer from `.ckpt` in PyTorch, with bounded overlap-add buffers and whole-track working buffers that spill to CPU on long inputs. |
| [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp) | C++/GGML engine for BS- **and** Mel-Band RoFormer on GGUF weights, where the window is a runtime flag (`--chunk-size`, `--overlap`). |
| [UVR-rs](https://github.com/IronHpc/UVR-rs) | Rust BS-RoFormer 1296 inference from the original `.ckpt` (no ONNX, no conversion), Burn with optional OpenVINO, exposing `--window-frames` and window-level parallelism. |
| [charon-audio](https://github.com/Valkyra-Labs/charon-audio) | Rust + `ort` separation of HTDemucs; its streaming path is documented bit-identical to whole-buffer separation, with progress and cancellation. RoFormer is explicitly out of scope there. |
| [OpenKara](https://github.com/thedavidweng/OpenKara) | Rust/Tauri app whose per-chunk checkpointing lets an interrupted separation resume. |
| [windowed-roformer](https://github.com/smulelabs/windowed-roformer) | Solves the same quadratic attention in the *architecture* — windowed sink attention, fine-tuned from the vocal checkpoint — at a fraction of the FLOPs. Needs its own weights. |
| [MSS_ONNX_TensorRT](https://github.com/ZFTurbo/MSS_ONNX_TensorRT), [port-bs-roformer-onnx](https://github.com/santiquiroz/port-bs-roformer-onnx) | The other answer to "shorter window": re-export the graph, and chunk outside it in the driver (8 s blocks, 50 % overlap, reflect padding). |
| [soufmer](https://github.com/Chaceon-albus/soufmer), [melband-roformer-mlx](https://github.com/Da1sypetals/melband-roformer-mlx), [mel-roformer-mlx-swift](https://github.com/xocialize/mel-roformer-mlx-swift) | A Tauri desktop front end for this model, and Rust / Swift MLX implementations of it. |

Also out there and not examined closely enough to describe: [demucs-rs](https://github.com/nikhilunni/demucs-rs),
[stem-splitter-core](https://github.com/gentij/stem-splitter-core),
[portable-vocal-remover](https://github.com/Nikaidou-Shinku/portable-vocal-remover),
[melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer).

So the delta this crate is claiming is narrow, and stated as three things rather
than as a superlative:

* **A survey of this export.** Which six integers carry the window in a Mel-Band
  RoFormer ONNX graph, why shrinking it is provably exact while growing it needs
  the iSTFT normalisation table regenerated, and an assertion that the session's
  own declared dimensions agree with what we think we built. The mechanism of
  loading an edited graph is not ours — ONNX Runtime does it, and
  [`ort`](https://github.com/pykeio/ort) wraps it (`commit_from_memory`, and
  `edit_from_memory` for editing). What is ours is knowing which bytes in *this*
  graph to change, and why that is sound.
* **Byte-identical resume across a crossfade grid** — including the seam-replay
  arithmetic — and a harness column (`resumed_from`) that fails the check when a
  "resume" silently re-ran the whole track. Resumable separation exists elsewhere;
  a resumed run whose bytes are indistinguishable from an uninterrupted one is a
  stronger statement, and it is a test here.
* **The window as a load-time parameter of the file you already downloaded**,
  with no derived artifact. Re-exporting produces a second model file, GGUF
  produces a converted one, and a from-scratch engine produces its own weights;
  none of those is wrong, they just cost a file.

What this crate is *not*: the first RoFormer outside Python, the first Rust
separator, the first to bound memory by windowing, or the first to refuse when
there isn't enough of it — inference servers have been printing "requires more
memory than is available" for years, and this one just does it before the first
forward instead of after a failed allocation.

## Non-goals

* No full-band multi-stem separator, no speech enhancement, no diarisation. This
  crate takes a mix and gives you vocals plus the residual.
* No `--gpu` heroics. The CPU/MLX paths are what 16 GB machines actually have.
* No bundled weights, no downloads from inside the library. A caller chooses the
  file; `sha256` checking is yours to wire up.
* No re-implementation of the ONNX Runtime tensor pipeline. If you want to run a
  *different* Mel-Band RoFormer checkpoint (drums, bass, 6-stem), the shapes and
  band counts are in the graph and in `model.rs`; only the vocals checkpoint is
  exercised here.

## Status

Version 0.1: the port of our production engine, laid down as a standalone crate.
The API may move once, on the strength of someone actually using it. Tests that
need a model file print `[SKIP]` and pass — CI has no weights, and it will not
get any; if you add a test that needs a 953 MB download, gate it behind an
environment variable naming a local file.

## Licence

Apache-2.0 for the code in this repository. [`NOTICE`](NOTICE) for the chain of
upstream artefacts, [`docs/LICENSES.md`](docs/LICENSES.md) for what the MIT tags
cover and what they do not.
