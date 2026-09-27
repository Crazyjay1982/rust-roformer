# Benchmarks, and how not to be fooled by them

Every number below carries the machine it came from and the instrument that read
it. A number without those is not a measurement, and several of the traps here
cost a day each to learn — they are written down so you do not pay again.

## The instruments

| Quantity | What reads it | What it actually means |
| --- | --- | --- |
| `peak_mb` on macOS | `proc_pid_rusage(RUSAGE_INFO_V4)`, `phys_footprint` | **high-water mark for the whole process lifetime.** An arm measured after a heavier arm in the same process can never report lower. One process, one number. |
| `peak_mb` on Windows | process memory counters + `GlobalMemoryStatusEx` | the useful figure is available *commit*, not resident set — these differ on both platforms and are not interchangeable |
| wall clock | `Instant` around `separate()` | comparable **only** within a session, and only paired (see below) |
| separation quality | correlation against a reference, and listening | see "the SI-SDR trap" |

Two calibration facts about the macOS instrument, measured rather than assumed:

* `ri_lifetime_max_phys_footprint` lags `ri_phys_footprint` inside the same
  struct while memory is climbing: a C probe saw **30,389 inversions in 835,455
  consecutive reads (3.6%), maximum skew 147,456 bytes**, and **zero inversions
  in 200,000 reads while idle**. The code takes `max(current)` where it reads,
  which holds for any lag, and a regression test feeds memory while reading in
  pairs to keep it honest.
* After a cache-clearing call, the kernel takes ~0.7 s to deduct
  `phys_footprint`. Reading immediately gives a mid-flight number; three reads of
  the same arm gave 25 / 26 / 233 MB, which looks like nondeterminism and is
  sampling time.

## Memory

| Case | Before | After | Machine |
| --- | --- | --- | --- |
| Whole-track audio read (45.8 min) | 1,094 MB | **169 MB** | Apple Silicon, debug build, 20 ms sampling |
| Separation host peak, 45.8 min (`mlx`) | 3,405 MB | **72 MB** | Apple Silicon; device-side figures unchanged value for value (6,017 → 6,016 MB) |
| One forward pass, stock 8 s window | — | ~19.4 GB commit / ~10.7 GB resident | Windows, 16 GB class |
| One forward pass, 4 s window | — | ~5.1 GB commit / ~3.7 GB resident | Windows, 16 GB class |
| Product path, arena disabled | 8.83 GB peak | **3,440 MB peak, 349 MB plateau** | Windows, 16 GB class |

The track-length term for the ONNX engine is ~20 MB/min — the per-window cost is
what you pay for. That asymmetry is the design: memory scales with the window you
choose, so nothing needs to scale with the track.

Note what "arena disabled" means. An arena allocator that keeps its blocks is a
*plateau*, not a peak: with the arena open the same entry point reported 8.83 GB,
and closing it moved the peak to 3.44 GB without changing a sample of output. If
you measure a memory "improvement" that turned out to be an allocator setting,
check which of the two you were reporting.

## Wall clock: we publish none

On the machine this crate was developed on, the **same build, same model and same
input** measured 88 s in one session and 158 s in another. Cross-session
comparisons made from those numbers once produced a confident "streaming is 44%
slower" conclusion, which was an artifact. Put both versions in one session and
swap the order and the effect vanishes; what remains is a position term of ≈ +5 s
(6%) for whichever configuration runs first.

So this document contains no absolute timing claim, and
`scripts/bench_pair.sh` is built to keep it that way: it runs each configuration
first *and* second in one session, reports the position-cancelled difference, and
prints `inconclusive` when the difference is smaller than the position term or
the scatter, or when the two orders disagree in sign.

## Quality, and the trap in the obvious metric

* The `mlx` engine vs a PyTorch FFT reference on one window: **corr 0.9999997**.
* The Rust port vs the Python reference on the same chunk schedule:
  **corr 0.99997**.
* Streaming vs non-streaming, and resumed vs uninterrupted: **byte-identical**
  output, enforced by tests rather than argued. `bench resume-check` re-proves it
  on your input.
* 4 s window vs the 8 s reference: transcript identical on continuous dialogue,
  no vocal leakage into the background stem, no audible difference in blind
  listening; SI-SDR against the 8 s output 16.4–16.7 dB.

**The SI-SDR trap.** That 16 dB looks like a regression and is not one. Moving
*only* the overlap of the same 4 s model from 1.5 s to 2.0 s moves the output
10.5 dB away from another 4 s configuration — the chunk grid's alignment
perturbs the waveform by the same order of magnitude as the change you are trying
to see. A per-sample difference metric against a different-schedule reference
cannot separate "worse" from "differently positioned". To judge a window or
overlap change you need either a self-mix reference (mix = vocals + background,
so each stem has a ground truth) or your ears. Correlation against a reference
computed on the *same* grid is legitimate; across grids it is not.

The same lesson chose the `mlx` acceptance target: comparing it against the ONNX
graph's output would have hidden a real deviation (the export's conv-STFT sits
near 0.70 against a float64 FFT arbitration while MLX sits at 0.9999997),
because the thing being compared against was the one carrying the error.

## Reproducing

```sh
cargo run --release --example bench -- synth --seconds 120 --out track.wav
cargo run --release --example bench -- run --model <file> --input track.wav --repeat 3
cargo run --release --example bench -- run --model <file> --input track.wav --window 176400
cargo run --release --example bench -- resume-check --model <file> --input track.wav
./scripts/bench_pair.sh --model-a big.onnx --model-b small.onnx --input track.wav
# one file, two windows: the same model on both arms, B reshaped at load time
./scripts/bench_pair.sh --model-a melband_roformer_vocals.onnx \
    --model-b melband_roformer_vocals.onnx --window-b 176400 --input track.wav
```

`bench synth` needs no model and no audio: it writes a deterministic two-channel
test signal, so you can measure the I/O path (which is where the interesting
memory behaviour lives) before touching anything licensed.

### Walked once on the stock export, in a fresh clone

Against `smank/mel-band-roformer-vocals-onnx` as downloaded — 953,292,899 bytes,
sha256 `64a4f3be…f561`, re-hashed on this machine before the run — and the 12 s
`bench synth` track, from a `git clone` of this repository with no other state.
What follows are *decisions*, not timings: the identical configuration (12 s, 5
windows of 176400, one machine, one day, one binary) measured 14.0 s, 15.7 s,
26.8 s, 26.8 s, 29.3 s and 53.4 s across six sittings. That 3.8× spread is the
point of the section above, and no timing in this table is quoted as a result.

| Requested | Outcome |
| --- | --- |
| (no `--window`, native 352800) | refused before any allocation: `needs ~19400 MB per forward, 15476 MB available` |
| `--window 200000` | refused: not a multiple of the 441-sample hop |
| `--window 705600` | refused: growing needs the iSTFT normalisation table regenerated |
| `--window 352800` | refused — reshaping to the declared window is a no-op, and the native 8 s price does not fit |
| `--window 176400` | accepted; the session reported `window_samples = 176400`, 5 windows, both stems 2,116,844 B |

The accepted runs carried a process-lifetime peak of 6.0–7.0 GB
(`phys_footprint`, macOS, whole harness process — it includes the 953 MB FP32 graph
the session holds, so it is not a per-forward figure and not comparable with the
Windows per-forward anchors).

`resume-check --window 176400` then ran three passes: the control inferred all 5
windows; the cancelled pass committed 3 (330,750 frames, 1,323,044 B of `.part`
with its sidecar beside it); the continuing pass inferred 3, skipped 2 outright and
started from 330,750 frames already on disk, and both stems came out
**byte-identical** to the control. Those last two figures are why `resumed` and
`resumed_from` are columns: a pass that quietly re-ran the whole track would still
have produced identical bytes, and nothing in the timing distinguishes the two. The
engine's own end-to-end test on a real 12 s excerpt prints the same partition
(`529200 frames in 3 window(s) (2 resumed)`, `3 + 2 = 5`).

One caveat that is easy to read backwards: on this synthetic track the *vocals*
stem is silent (peak 0 of int16, the model finding no voice in two alternating
tones) and the residual carries everything. That is the expected output for
non-vocal input, and it is why the harness prints byte counts rather than claiming
it separated anything — separation quality is measured on real audio in the
sections above, not here.

For the graph surgery there is a harder check than any timing number:
`tools/reduce_window.py` derives the short-window file from the long-window one,
and the result is **byte-identical** to the artifact this work was built on
(sha256 `2a83f2fe…`, 271,306,261 bytes). All 1,276 initializers — every weight,
quantisation scale and zero point — are unchanged between the two files, which is
what makes "shape constants only" a verified statement rather than a claim. The
one thing that does not survive is cached shape metadata (6,945 `value_info`
records), so a derived file is smaller than its source and a round trip back to
the original window is *not* bit-identical at file level.

## What is not measured

* No fresh Windows wall-clock or peak figures are published here; the Windows
  memory numbers above are from the field/lab runs that motivated the 4 s window,
  not from a re-run on a clean tree.
* Linux peak/commit behaviour of both engines: unmeasured.
* The int8 quantisation step that produced the smaller ONNX file is **not** in
  this repository — `tools/` changes the window of whatever file you feed it, and
  quantising is a separate decision you should not take from a README.
* Multi-stem or non-vocals Mel-Band RoFormer checkpoints: untested here. The
  shapes and band counts come from the file, but only the vocals checkpoint was
  ever run.
* The `mlx` engine is compiled on Apple Silicon only, CI does not exercise it
  (see `.github/workflows/ci.yml`), and `--features mlx` does not build from a
  clean checkout without an MLX install — the failure is `mlx-sys`'s CMake fetch of
  the C++ library, before any code here is reached. Every MLX number in this file
  was measured through a scratch manifest wired to a provisioned MLX, which is why
  they are reported beside the default build and not inside it.

## Reproduced by the port itself

Two claims worth re-measuring after extraction, taken from this crate's own test
arms rather than quoted from the application they came from:

* one 8-second window against a PyTorch float32 FFT reference: **corr 0.9999997**
  (that single window took 4.06 s of wall clock — one sample of a quantity this
  document elsewhere refuses to compare across sessions);
* resume: six kill points (three on a six-window grid, three on a four-window
  tail), each re-inferring fewer windows and finishing with a **bit-identical**
  pair of stems. The last kill point re-inferred 0 windows, which is the
  "already finished" case a buggy resume would turn into a silently truncated
  file reported as success.
