# Changelog

All notable changes to this crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — unreleased

First extraction from a shipping desktop application. The public repository is
`https://github.com/Crazyjay1982/rust-roformer`; `cargo install --git` against it
works today, and `repository` is set in `Cargo.toml` accordingly. Crates.io is still
unpublished, so `cargo install rust-roformer` and the crates.io/docs.rs badges wait
for that first release rather than pointing at something that 404s.

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
- `rust-roformer`, the command line: `--model`, `--out`, `--window` (a sample
  count, or seconds with an `s` suffix), `--engine`, `--threads`, `--fresh`,
  `--quiet`. Seven options, each one a thing the library can already be asked to
  do and none a promise the crate does not make — hence no `--gpu` (the ONNX arm
  is CPU, the MLX arm is the Apple Silicon path) and no decoder beyond WAV. It
  builds with either engine or neither, and `--version` says which it got; exit
  codes are 0 done, 1 failed, 2 mistyped; a failure that means "this machine is
  too small for that window" says so and names the knob to turn. Stopping it is
  safe by construction rather than by a signal handler, because the pair on disk
  *is* the checkpoint: measured on a 12-second track interrupted eight seconds in
  (`SIGINT`, two of five windows flushed), the identical command resumed from
  frame 220,500 and wrote stems byte-identical to the uninterrupted run.
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
  turn it red. The scan has a second subject as well: **commit metadata**, because
  that is the one thing `git push` publishes which a file walk cannot see — the walk
  excludes `.git`, which is right for weights and wrong for identities. The same two
  pattern lists are run over `git log`'s author and committer lines; the check was
  motivated by eleven commits in this repository's own pre-rewrite history carrying a
  personal address that every file in the tree had been held clean of, and the
  history that is pushed now carries none. Being exact about the ordering: an earlier
  entry here described this sweep, and the commit that carried it touched only
  `CHANGELOG.md` — so the description sat in a published file with no code behind it.
  That is the failure mode this gate exists to catch in someone else's work, so it is
  named rather than smoothed over: the sweep exists as of this entry, and its probe
  is an empty commit authored by a banned address, which turns the run red on one
  generic domain pattern and one local name pattern. Commits reachable only from
  other refs (a rewrite's backup branch, filter-branch's own `refs/original`) produce
  a note carrying a *count*, not the identities: a plain push does not send them,
  failing on them would block the push that lands the fix, and printing them would
  write into the transcript the strings this gate keeps out of the repository.
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

### Changed

- Copyright and attribution now name the maintainer: `LICENSE` carries
  `Copyright 2026 DeepForgeHub (https://deepforgehub.com)` in place of the Apache
  appendix's `[yyyy]`/`[name of copyright owner]` placeholders, `NOTICE` says the
  same for the Rust code and keeps the upstream-weight list below it, and
  `docs/LICENSES.md` §1 repeats it. The name is the one the project's own site
  uses for itself (`DeepForgeHub`, from the domain); it is deliberately **not**
  the bare `DeepForge`, which an unrelated 755-star deep-learning IDE already
  occupies on GitHub. If a registered legal entity owns this code, that name
  should replace it here and nowhere else needs to change.
- `rust-version` corrected from 1.75 to **1.88**, and the MSRV job in CI now checks
  the configuration that number is actually about. The old value held only for
  `--no-default-features`: `ort` and `ort-sys` rc.13 declare a 1.88 floor, so
  `cargo +1.85 check` is refused outright and `cargo +1.75 check` fails on
  `ort-sys`'s `edition2024` manifest — while the CI job asked only about the
  engine-free build, which meant the badge was green and false at the same time.
  The crate was never published, so no one is being moved. Worth being exact about
  the side effect: raising the number surfaced two
  `clippy::manual_is_multiple_of` sites the old value had legitimately silenced
  (`u*::is_multiple_of` is 1.87+), and applying that method in `graph.rs` is what
  killed the last remnant of the old claim — the engine-free build no longer
  compiles on 1.75 either, so the first version of this very bullet, which said CI
  would check "1.75 without an engine", was corrected before it could sit in a
  workflow that fails on its own premise.
- `Cargo.toml` gained `repository` and `homepage`; the README's install block
  names the real git URL instead of an `OWNER` placeholder, so a reader can install
  this today without waiting for a release. The crate docs and `rust-roformer --help`
  each state the home URL once, and a test asserts the help text keeps it.
- The README carries a CI badge. The placeholder comment above the badges had been
  holding three of them back until "the remote and the first release exist"; the
  remote exists now and the workflow passes on it (fmt-clippy, msrv, and the test
  matrix on Linux/macOS/Windows, all green on `5c57d71`), so the CI badge is filled
  in and the crates.io and docs.rs badges stay withheld — the crate is still
  unpublished, and their URLs would 404.
- Two `chunks_exact(<constant>)` call sites became `as_chunks::<N>().0`: clippy
  1.98.0 added `chunks_exact_to_as_chunks`, and the `fmt-clippy` job runs `-D
  warnings` against rolling stable, so the badge went red on the first push after
  that release landed without a single line of this crate's code changing. Both
  sites are test helpers, not the shipped read path — one whole-buffer WAV oracle
  in `audio.rs`, one gguf-float decoder in the MLX module — and the rewrite is
  iteration-identical, so the suites report the same counts as before it
  (`as_chunks` is 1.88-stable, which is also this crate's floor, and the MSRV job
  checks `--all-targets` so it would notice if that stopped being true). Worth
  naming the shape of the failure rather than burying it: a gate pinned to a
  moving target is a gate that will go red on someone else's release day, and that
  is a maintenance decision, not an accident of this diff.
- The README, the crate docs and `docs/benchmarks.md` now name the application this
  code came out of — DeepVideo — rather than describing it as "a desktop
  video-translation application". It is the same sentence that says where the
  45.8-minute and Windows figures were instrumented, which is the honest way to
  label a provenance: the numbers a public test suite cannot carry are exactly the
  ones a reader most wants to know the origin of. Nothing else about the
  relationship changed: no product code, no weights, no audio and no dependency on
  the application live here, and the crate still runs with the network off.
