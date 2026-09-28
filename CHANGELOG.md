# Changelog

All notable changes to this crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — unreleased

First extraction from a shipping desktop application. Not yet published: no
remote exists, so `Cargo.toml` deliberately carries no `repository`.

### Added

- `graph` — locates and rewrites the input window baked into a Mel-Band RoFormer
  export, in memory: four `int64` `Constant` payloads and the input/output last
  dimensions. Refuses to grow past the iSTFT normalisation table's capacity, to
  touch a graph that carries the window inside a weight, and to rewrite a
  dimension varint that would change width (that case belongs to the offline
  tool, which re-serialises).
- `audio` — chunk-at-a-time WAV decode and a chunked resampler, so peak memory
  does not scale with track length. The exact-vs-last-bit behaviour across sample
  rates is asserted in tests (48 kHz→16 kHz is bit-identical; 44.1 kHz→16 kHz is
  not, by arithmetic, not by neglect).
- `stream` — `WavSource` (windowed reads) and `StemWriter` (both stems written in
  one pass, 44-byte header rewritten at each window boundary so the files on disk
  are always a playable prefix), plus a resume record checked field by field:
  mismatch means start over, never "continue from a stale prefix" and never
  "fail with a broken file behind you". Every failure, cancellation included, keeps
  that checkpoint; [`stream::discard_staging`] is the caller's own, explicit way to
  say the partial result is worthless, and no engine calls it for you — one arm of
  this crate used to throw the pair away on `Cancelled`, which contradicted the
  trait and the other arm, and a user who pressed stop is the last person who
  wants an hour of inference deleted.
- `mem` — per-platform memory figures (macOS `phys_footprint`, Windows available
  commit, Linux `/proc`), the refusal text classifier, and the pre-flight gate
  that prices a window before the first forward.
- `engine::onnx` — ONNX Runtime engine: windowed forward, linear crossfade,
  streaming output, checkpoint/resume, cancellation between windows, the gate.
  The arena is off by default; that choice is measured (8.83 GB peak with it on,
  3,440 MB with it off, identical output).
- `engine::mlx` (`--features mlx`, Apple Silicon) — the architecture
  re-implemented in Rust: own STFT, mel band split, alternating time/frequency
  transformer blocks, mask estimator. Accepted against a PyTorch FFT reference
  rather than the ONNX graph's output, for the reason recorded in its module
  docs. `--features mlx` needs a provisioned MLX; see README.
- `tools/reduce_window.py` — the same window edit, offline and byte-exact;
  `tools/extract_onnx_weights.py` — the 672-tensor `.safetensors` the MLX engine
  loads, derived from the stock export. Neither redistributes weights, and this
  repository contains no model file and no audio.
- `examples/bench.rs` — `synth` (deterministic test track, no model needed),
  `run`, and `resume-check` (cancelled-then-continued output must be
  byte-identical to uninterrupted, or it fails with the offset of the first
  difference). `--window` reshapes the stock export at load time, and every row
  ends with the window read back off the live session plus the count of windows
  that came off disk, so a pass that quietly re-ran everything cannot be mistaken
  for a resume. `scripts/bench_pair.sh` + `scripts/pair_summary.awk` do
  same-session, order-swapped A/B and print `inconclusive` when the difference is
  inside the position term or the scatter; `--window-a/--window-b` let one file be
  measured at two windows.
- `scripts/audit_public.sh` — refuses to let machine paths, account-looking
  strings, credentials or model/audio binaries into the tree. Its own banned list
  is split in two, because a gate that is published cannot carry the identifiers
  it guards: generic shapes (home directories, AWS key patterns, private-key
  headers, email domains, URL credentials) live in the script, the naming-specific
  patterns live in a gitignored `scripts/audit-local.txt` that a public clone does
  not have. The script fails if that file is ever tracked, and each class of leak —
  path, private name, weight file, tracked local list — was planted and proved to
  turn it red.
- `scripts/demo.sh` + `docs/demo.md` — a listening example that costs the
  repository nothing it does not already refuse to carry: two recordings from
  Wikimedia Commons (an aria over an orchestra, and an a cappella control), named
  by URL, pinned by SHA-256 at both the source file and the 12-second excerpt,
  fetched and decoded on the user's machine under their attribution-only licences
  (CC BY 2.5 / CC BY 3.0). The page reports what those exact bytes measure —
  63.3 dB of sub-140 Hz given up by the vocal stem while the voice band holds to
  0.1 dB, the background stem 39.8 dB under the input on material that has nothing
  to remove — and states plainly that none of it is an SI-SDR figure, because
  there is no ground truth to compute one against. It also records the failure
  mode a synthesized test signal runs into: `bench synth` yields a vocal stem of
  exactly zero, and a hand-built "voice-like" signal scores *worse* than not
  running the model at all (−10.35 dB against +3.89 dB), which is the checkpoint
  rejecting an input it was never trained on rather than a port defect.
- Documentation with the claims checked rather than phrased: `docs/benchmarks.md`
  gives every number its machine, its sampling point and its failure mode;
  `docs/LICENSES.md` states what the upstream MIT tags do and do not answer; and
  the README's "Related work" section links the Rust, C++ and Python projects
  already doing parts of this, then limits the delta to what was actually
  measured here. No superlatives, and no claim on the model.
- CI: fmt, clippy with warnings denied, tests on Linux/macOS/Windows with and
  without default features, and an MSRV job so `rust-version` is a claim someone
  checks rather than decor.
