# rust-roformer

<!-- The crates.io and docs.rs badges go here once the first release exists; a
     badge that 404s is worse than a badge that is missing. The CI badge is here
     now because the remote it points at is real. -->
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
[![build](https://github.com/Crazyjay1982/rust-roformer/actions/workflows/ci.yml/badge.svg)](https://github.com/Crazyjay1982/rust-roformer/actions/workflows/ci.yml)
![MSRV](https://img.shields.io/badge/MSRV-1.88-lightgrey)
![weights included](https://img.shields.io/badge/weights%20included-none-green)
[![home](https://img.shields.io/badge/home-deepforgehub.com-informational)](https://deepforgehub.com)

Mel-Band RoFormer vocal/background separation in Rust, maintained at
DeepForgeHub. This code is the separation stage of
[DeepVideo](https://deepforgehub.com), a desktop application that translates and
dubs video, which is why the parts below (window sizing, memory, resume) were
built out rather than merely ported. The crate itself is Apache-2.0, stands on its
own, and works offline once built.

The model is not ours: the architecture is [ZFTurbo's](https://github.com/ZFTurbo/Music-Source-Separation-Training)
and the checkpoint is [KimberleyJSN's](https://huggingface.co/KimberleyJSN/melbandroformer).
What is ours is everything around it: two inference engines and an I/O layer that
separates a four-hour track on a 16 GB laptop without ever holding the track, the
intermediate activations, or the result in memory at the same time. If you want the
model, download it; if you want a Rust separation step that does not OOM at 02:00
on a laptop, this is the interesting part. Where that claim overlaps something
that already exists, [Related work](#related-work) links it.

**Which RoFormer.** "RoFormer" is an overloaded name. Here it means *Mel-Band
RoFormer*, the band-split separation model above, not the NLP RoFormer that the
rotary-embedding paper is named after, and not **BS-RoFormer**. The two audio
models share an idea and differ in how the frequency axis is grouped (fixed bin
groups vs a mel filterbank), so **their weights are not interchangeable**: a
`melband_*` checkpoint will not load into a BS-RoFormer implementation, whatever
the name in the README says.

```sh
# crates.io has no release yet, so the git form below is the one that works today:
cargo install --git https://github.com/Crazyjay1982/rust-roformer
# once the first release is out, these two are the same thing:
cargo install rust-roformer                              # the command line
cargo add rust-roformer                                  # or the library
```

**Use it if** your application embeds a separation step and runs somewhere memory
is the binding constraint: a static-shape graph you must not OOM on, jobs long
enough to be killed halfway through, and a build that has to work on Windows. Or
if you want one command that separates a track on a laptop without a Python
environment.

**Skip it if** you want the best possible separation regardless of footprint (run
the Python toolchain on a GPU), you need drums/bass/6-stem, or you need CUDA:
the ONNX arm is CPU and the accelerated arm is Apple Silicon.

## The one number that explains the design

A Mel-Band RoFormer forward pass is a static-shape graph whose time-axis
attention is `batch × heads × T × T`. Peak memory of **one window** is therefore
enormous and **independent of track length**, measured on the stock 8-second
export:

| Window | Commit per forward | Resident per forward | Time term |
| --- | --- | --- | --- |
| 8 s (stock export) | ~19.4 GB | ~10.7 GB | ~20 MB/min |
| 4 s (window shortened at load time) | ~5.1 GB | ~3.7 GB | ~20 MB/min |

*Windows, 16 GB-class machine, product harness; the ONNX Runtime arena was later
disabled, which moved the product path's own peak to 3,440 MB with a 349 MB
plateau. See [docs/benchmarks.md](docs/benchmarks.md) for every number's
provenance; several are Apple Silicon and do not transfer.*

Those two rows are also the whole range of the memory gate's trust: they are the
anchors `OnnxEngine::estimate_forward_mb` is fitted through, so asking for
`--window 2s` sits *below* the short anchor and the pre-flight line says so
(`EXTRAPOLATED below the short anchor`). What a 2 s window actually costs on this
crate's own binary is measured on two machines in
[docs/benchmarks.md](docs/benchmarks.md): 4,567 MB of whole-process
`phys_footprint` on Apple Silicon, 2,242 MB of Windows commit on a 13.86 GB box.
Both are whole-process figures carrying the 909 MiB of weights, so neither is
a third row of this table. On the Windows box the two windows disagree with the
anchors in opposite directions: the 2 s peak of 2,242 MB sits *above* the 1,525 MB
the gate asks for, and the 4 s peak of 3,968 MB sits *below* the 5,100 MB anchor
it is priced at. The `mlx` engine gets no shorter-window row at all: it refuses
every window but the native 8 s one, because there the window is an attention
length rather than a buffer size.

A 19.4 GB single allocation on a 16 GB
machine does not fail politely, and it fails after you have paid for decoding and
resampling. So the crate's job is: **decide before allocating, stream both
directions, and make the work resumable.**

## What it does

* **Refuses instead of OOM-ing.** `mem::snapshot()` asks the OS what it has;
  the engine compares that against the window it is about to run and returns
  `Error::Memory { need_mb, avail_mb }` in microseconds. On macOS the figure is
  `phys_footprint`; on Windows it is available *commit*: different quantities,
  and the code says which one it read.
* **Streams in.** `audio::WavSource` decodes one window at a time; the resampler
  (`rubato`, chunked with one filter configuration) never builds a
  source-rate copy of the whole track. Reading 45.8 minutes used to peak at
  1,094 MB and now at 169 MB.
* **Streams out.** `stream::StemWriter` writes both stems in one pass and
  rewrites its own 44-byte WAV header at every window boundary, so at any
  instant the files on disk are a *playable, complete prefix*, not a headless
  blob. `Drop` cannot leak a half-written final file.
* **Resumes.** Kill it, lose power, hit stop: the next call with the same paths
  continues from the last flushed window. The job identity (input size+mtime,
  model size+mtime, window, overlap) is compared field by field first, and a
  mismatch means *start over* rather than "silently continue with a stale
  prefix" or "fail while leaving junk". A resumed run is **byte-identical** to an
  uninterrupted one; that is a test, not a claim.
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
| Builds from a clean clone | yes (`--features onnx`, the default) | **not without provisioning MLX**; see below |
| Weights | `.onnx` you download | `.safetensors` produced by `tools/extract_onnx_weights.py` from the same `.onnx` |
| Window | read from the graph; shorten it at load time with `OnnxEngine::with_window(path, samples)` | 8 s (`352800` samples @ 44.1 kHz), 2.5 s crossfade |
| Acceptance | vs the Python implementation on the same schedule: corr 0.99997 | vs a PyTorch FFT reference: corr 0.9999997 |

The MLX engine is the one to read if you came for "Rust port". It has its own
STFT, its own band-split, alternating time/freq transformer blocks and the mask
estimator, and it carries comments about the handful of numerical conventions
where a "reasonable" implementation is silently wrong: L2Norm scaling with
`sqrt(dim)` rather than RMS, exact erf GELU, `traditional` RoPE with
`base=10000`, manual softmax attention instead of the fused path.

Before you try `--features mlx`: `mlx-rs` pulls `mlx-sys`, whose build script runs
CMake and fetches the MLX C++ library from source. That command therefore needs a
C++ toolchain, CMake, network access and a long compile, or an MLX install to link
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

### One command

`aria_12s.wav` below is the twelve-second excerpt `scripts/demo.sh` downloads and
names; the model is the stock export you fetched yourself. This is the *second*
run of the identical command line (the first was stopped with Ctrl-C eight
seconds in, with two of five windows already flushed), and the output is verbatim
except that the model path is shortened to its filename:

```console
$ rust-roformer --model melband_roformer_vocals.onnx --window 4s -o out aria_12s.wav
onnx · window 176400 samples (4.000 s, 400 hops) · 4 threads
input aria_12s.wav · 529,200 frames (12.000 s at 44.1 kHz) · source 44100 Hz
[onnx] separating: window 1/5, 41%
[onnx] separating: window 3/5, 62%
[onnx] publishing: window 5/5, 100%
wrote out/vocals.wav (2,116,844 B) and out/background.wav (2,116,844 B)
  529,200 frames at 44100 Hz · 4 windows inferred, 1 skipped
  continued from frame 220,500 — everything before it was already on disk
  peak 7,827 MB in this process (lifetime high-water, not per window)
  10.4 s elapsed here
```

There is no `--resume` flag: a checkpoint that describes this exact job (same
input bytes and mtime, same model bytes and mtime, same window and overlap) is
used, and anything else starts over rather than appending to the wrong track.
`--fresh` says discard it. The two numbers at the end are the two halves of that
statement: 4 + 1 = the 5 windows of the schedule, and 220,500 frames were on disk
before this call began. A resumed run's output was byte-identical to the same
command run without interruption, which is the only property worth having.

`--window 4s` is not a different model file: the stock export declares 352800
samples (8 s) and this reshapes that graph at load time, which is what the memory
gate below prices.

That is the whole surface: `--model`, `--out`, `--window`, `--engine`,
`--threads`, `--fresh`, `--quiet`, and `rust-roformer --help` for what each one
means, including what Ctrl-C costs. No `--gpu` (the ONNX arm is CPU, the MLX arm
is the Apple Silicon path), and no format conversion (WAV in; `ffmpeg` for the
rest).

### As a library

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

### Through the harness

The same thing through the measurement harness (`bench` below is
`cargo run --release --example bench --`), which is where the interesting part
is: what the crate *refuses*, in its own words. This is a real session on a 16 GB
Apple Silicon laptop against the unmodified stock export, output verbatim except
that the model path has been shortened to its filename:

```text
$ bench synth --seconds 12 --out track.wav
bench: wrote 529200 frames (12.0 s stereo 44.1 kHz) to track.wav

$ bench run --model melband_roformer_vocals.onnx --input track.wav --out o
bench: memory gate: this window needs ~19400 MB per forward, 14281 MB available
(exit 1 — the stock 8 s window, refused before anything was allocated)

$ bench run --model melband_roformer_vocals.onnx --input track.wav --out o --window 200000
bench: model error: window 200000 is not a positive multiple of the hop 441
(exit 1)

$ bench run --model melband_roformer_vocals.onnx --input track.wav --out o --window 705600
bench: model error: growing the window from 352800 to 705600 needs the iSTFT
        normalisation table regenerated (baked capacity 352800 samples); only
        shrinking is a shape edit
(exit 1)

$ bench run --model melband_roformer_vocals.onnx --input track.wav --out o --window 176400
label   seconds  windows  wall_s  rtf    peak_mb  vocals_bytes  window_samples  resumed  resumed_from
run#0   12.00    5        15.70   1.308    6730      2116844        176400        0          0
(exit 0 — same file on disk, reshaped in memory at load time)

$ bench resume-check --model melband_roformer_vocals.onnx --input track.wav \
      --out r --window 176400
label    seconds  windows  wall_s  rtf    peak_mb  vocals_bytes  window_samples  resumed  resumed_from
control  12.00    5        15.48   1.290    7829      2116844        176400        0          0
cut       7.50    3         9.53   1.271    7829      1323044        176400        0          0
cut      12.00    3        10.12   0.843    7800      2116844        176400        2      330750
vocals: identical (2116844 B)
background: identical (2116844 B)
resume-check: OK (resumed output byte-identical to uninterrupted)
(exit 0)
```

Three things that block are in there on purpose: the memory gate, the hop rule,
and the shrink-only rule. (Two of the error strings are wrapped for line width;
the numbers and wording are the program's.) And two things in the last block are
worth reading before you trust any timing table: `window_samples` is read back off
the live session rather than echoed from the flag, and `resumed_from` is the frame
count the continuing pass started from; without that column, a "resume" that
re-inferred the whole track would still have printed `identical`, because an
identical result is not evidence that the checkpoint was used.

What the numbers do *not* include: `peak_mb` is this process's lifetime
high-water `phys_footprint` (it contains the 953 MB graph the session holds, so
it is not a per-forward figure), and `wall_s` is not a speed claim: the same
12 s/176400 configuration measured 14.0 s to 53.4 s across six sittings on this
one machine, which is [why no wall clock is published as a
result](docs/benchmarks.md).

## Getting a model

Nothing in this repository is a model file, and none will be accepted in a pull
request. The reference export is

* `smank/mel-band-roformer-vocals-onnx` → `melband_roformer_vocals.onnx`,
  953,292,899 bytes,
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
Mel-Band RoFormer export fixes its window in *shape*: four `int64` `Constant`
nodes plus the input/output dimensions, which is why ONNX Runtime exposes no knob
for it. Those six numbers are rewritten in place and handed to
`commit_from_memory`. No weight byte is involved; the buffer is dropped once the
session owns what it needs, so the transient is one copy of the file. Two other
steps do need a file of their own: the MLX engine loads extracted tensors, and
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
the graph trims), while growing would need that table regenerated, so
[`graph`] refuses and says so. What it does not refuse is a *stale* shape
cache: patched
graphs keep the cached `value_info` records they were exported with, and ONNX
Runtime was measured to accept the session anyway: four `MergeShapeInfo …
Falling back to lenient merge` warnings naming the tensors we consume, then
correct output. That is why the engine asserts the session's own declared
dimensions instead of trusting that.

`tools/reduce_window.py` shares the rules above and adds two of its own: it is
shrink-only for the same table reason (`--force` overrides, and you should not),
and it strips the now-stale cached shapes so the output is byte-comparable with
the reference artifact; `--shape-metadata patch` keeps them consistent instead.

A derivative of a weight file carries the *weights'* terms, not this
repository's Apache-2.0, and see [docs/LICENSES.md](docs/LICENSES.md) for the
one question about these checkpoints that no MIT tag in the chain answers.

## Listening

Everything above is about memory, windows and resumption. To hear it, you need a
recording you are allowed to run a remover on, so `scripts/demo.sh` brings one:

```sh
./scripts/demo.sh --model melband_roformer_vocals.onnx              # Mozart: soprano over an orchestra
./scripts/demo.sh --model melband_roformer_vocals.onnx --source bright   # a cappella, the control
```

Both sources are on Wikimedia Commons under attribution-only licences (CC BY 2.5 /
CC BY 3.0), pinned by SHA-256, downloaded to your working directory and never
into this repository: the crate ships no audio and no weights, and that is not a
line it will cross. [docs/demo.md](docs/demo.md) has the credit lines, the exact
command, and what the resulting stems measure:

* On the aria the separation is by *content*, not by level: the vocal stem gives
  up **63.3 dB of everything under 140 Hz** (which stays at −0.0 dB in the
  background stem), while keeping the 200 Hz–4 kHz voice band to within 0.1 dB.
* On the a cappella there is nothing to remove, and the model does not invent a
  hole to fill: the background stem sits **39.8 dB below the input**, 53 dB down
  in the voice band.
* `vocals + background` reproduces the input to **75 dB**, so the mask is a
  partition of the mixture rather than an attenuated copy of it.

Once the script has fetched and cut the excerpt (it prints both paths), the file
through the command line is one more command, which is also how to compare the
two windows against each other by ear:

```sh
rust-roformer --model melband_roformer_vocals.onnx --window 4s -o stems demo-audio/aria_12s.wav
rust-roformer --model melband_roformer_vocals.onnx --window 2s -o stems-2s demo-audio/aria_12s.wav
```

Those are descriptions, not an SI-SDR claim: a commercial track has no
ground-truth stems, and [docs/demo.md](docs/demo.md) says so where the numbers
are. The same page explains why `bench synth`, the fixture the transcript above
runs on, produces a vocal stem whose samples are **all exactly zero**, and why
that is a fact about the checkpoint rather than a bug in this crate.

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

This is not a lonely field. What follows was checked against each project's
own README and source on 2026-09-28, which is also the date after which the
statements below may
have gone stale.

| Project | What it already does |
| --- | --- |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | The de-facto tool for these checkpoints: runs RoFormer from `.ckpt` in PyTorch, with bounded overlap-add buffers and whole-track working buffers that spill to CPU on long inputs. |
| [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp) | C++/GGML engine for BS- **and** Mel-Band RoFormer on GGUF weights, where the window is a runtime flag (`--chunk-size`, `--overlap`). |
| [UVR-rs](https://github.com/IronHpc/UVR-rs) | Rust BS-RoFormer 1296 inference from the original `.ckpt` (no ONNX, no conversion), Burn with optional OpenVINO, exposing `--window-frames` and window-level parallelism. |
| [charon-audio](https://github.com/Valkyra-Labs/charon-audio) | Rust + `ort` separation of HTDemucs; its streaming path is documented bit-identical to whole-buffer separation, with progress and cancellation. RoFormer is explicitly out of scope there. |
| [OpenKara](https://github.com/thedavidweng/OpenKara) | Rust/Tauri app whose per-chunk checkpointing lets an interrupted separation resume. |
| [windowed-roformer](https://github.com/smulelabs/windowed-roformer) | Solves the same quadratic attention in the *architecture*: windowed sink attention, fine-tuned from the vocal checkpoint, at a fraction of the FLOPs. Needs its own weights. |
| [MSS_ONNX_TensorRT](https://github.com/ZFTurbo/MSS_ONNX_TensorRT), [port-bs-roformer-onnx](https://github.com/santiquiroz/port-bs-roformer-onnx) | The other answer to "shorter window": re-export the graph, and chunk outside it in the driver (8 s blocks, 50 % overlap, reflect padding). |
| [soufmer](https://github.com/Chaceon-albus/soufmer), [melband-roformer-mlx](https://github.com/Da1sypetals/melband-roformer-mlx), [mel-roformer-mlx-swift](https://github.com/xocialize/mel-roformer-mlx-swift) | A Tauri desktop front end for this model, and Rust / Swift MLX implementations of it. |

Also out there and not examined closely enough to describe: [demucs-rs](https://github.com/nikhilunni/demucs-rs),
[stem-splitter-core](https://github.com/gentij/stem-splitter-core),
[portable-vocal-remover](https://github.com/Nikaidou-Shinku/portable-vocal-remover),
[melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer).

So the delta this crate claims is narrow, and it is these three:

* **A survey of this export.** Which six integers carry the window in a Mel-Band
  RoFormer ONNX graph, why shrinking it is provably exact while growing it needs
  the iSTFT normalisation table regenerated, and an assertion that the session's
  own declared dimensions agree with what we think we built. The mechanism of
  loading an edited graph is not ours: ONNX Runtime does it, and
  [`ort`](https://github.com/pykeio/ort) wraps it (`commit_from_memory`, and
  `edit_from_memory` for editing). What is ours is knowing which bytes in *this*
  graph to change, and why that is sound.
* **Byte-identical resume across a crossfade grid**, including the seam-replay
  arithmetic, and a harness column (`resumed_from`) that fails the check when a
  "resume" silently re-ran the whole track. Resumable separation exists elsewhere;
  a resumed run whose bytes are indistinguishable from an uninterrupted one is a
  stronger statement, and it is a test here.
* **The window as a load-time parameter of the file you already downloaded**,
  with no derived artifact. Re-exporting produces a second model file, GGUF
  produces a converted one, and a from-scratch engine produces its own weights;
  none of those is wrong, they just cost a file.

What this crate is *not*: the first RoFormer outside Python, the first Rust
separator, the first to bound memory by windowing, or the first to refuse when
there isn't enough of it. Inference servers have been printing "requires more
memory than is available" for years, and this one just does it before the first
forward instead of after a failed allocation.

## Non-goals

* No full-band multi-stem separator, no speech enhancement, no diarisation. This
  crate takes a mix and gives you vocals plus the residual.
* No GPU requirement. The default engine is portable CPU; the `mlx`, `cuda` and
  `coreml` features are opt-in accelerators, and the 16 GB laptop case, which is
  what this crate is designed around, is served by the CPU path.
* No bundled weights, no downloads from inside the library. A caller chooses the
  file; `sha256` checking is yours to wire up.
* No claim of checkpoint generality. If you want a *different* Mel-Band RoFormer
  (drums, bass, 6-stem), the shapes and band counts are in the graph and in
  `model.rs`, and the engine reads its window from the file rather than hardcoding
  it, but only the vocals checkpoint has been run here, so treat the rest as
  untested rather than as supported.

## Status

Version 0.1: the port of our production engine, laid down as a standalone crate.
The API may move once, on the strength of someone actually using it. Tests that
need a model file print `[SKIP]` and pass: CI has no weights, and it will not
get any; if you add a test that needs a 953 MB download, gate it behind an
environment variable naming a local file.

## Licence

Apache-2.0 for the code in this repository, copyright DeepForgeHub 2026.
[`NOTICE`](NOTICE) for the chain of
upstream artefacts, [`docs/LICENSES.md`](docs/LICENSES.md) for what the MIT tags
cover and what they do not.

## Who maintains this

This crate came out of [DeepVideo](https://deepforgehub.com), a desktop
application that translates and dubs video, maintained at the account behind it,
DeepForgeHub. The constraint it was built against is the machine its users
actually have: a 45-minute track on a business laptop with 16 GB of RAM, no GPU,
and no server to fall back on. That is why the memory gate, the streaming I/O and
the resume record exist, and why none of them needs a network.

Three things follow from that:

* **Every figure carries its machine, its sampling point and its failure mode.**
  Where a number came from the application running a 45-minute job and where it
  came from this crate's own harness on a twelve-second excerpt in a fresh clone,
  [docs/benchmarks.md](docs/benchmarks.md) says which, and the same page explains
  why no absolute wall clock is published as a result at all.
* **Nothing phones home at run time.** No telemetry, no licence check, no fetch
  from inside the library: the crate makes no network connection while it works,
  and `sha256` checking of the model file is yours to wire up. The one download in
  this project is `ort`'s build-time fetch of ONNX Runtime under the default
  features, which is exactly what the `ort-load-dynamic` feature exists to avoid.
* **The crate stands on its own.** Apache-2.0, no product dependency, no weights,
  no audio. You can use it without ever visiting the site, and the site is where
  DeepVideo itself is documented, including what this separation stage feeds into,
  which models were compared to get here, and what the rest of the pipeline does.
  This page repeats none of that.
