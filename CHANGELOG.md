# Changelog

All notable changes to this crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (unreleased)

First extraction from a shipping desktop application. The public repository is
`https://github.com/Crazyjay1982/rust-roformer`; `cargo install --git` against it
works today, and `repository` is set in `Cargo.toml` accordingly. Crates.io is still
unpublished, so `cargo install rust-roformer` and the crates.io/docs.rs badges wait
for that first release rather than pointing at something that 404s.

### Added

- `graph`: locates and rewrites the input window baked into a Mel-Band RoFormer
  export, in memory: four `int64` `Constant` payloads and the input/output last
  dimensions. Refuses to grow past the iSTFT normalisation table's capacity, to
  touch a graph that carries the window inside a weight, and to rewrite a
  dimension varint that would change width (that case belongs to the offline
  tool, which re-serialises).
- `audio`: chunk-at-a-time WAV decode and a chunked resampler, so peak memory
  does not scale with track length. The exact-vs-last-bit behaviour across sample
  rates is asserted in tests (48 kHz→16 kHz is bit-identical; 44.1 kHz→16 kHz is
  not, by arithmetic, not by neglect).
- `stream`: `WavSource` (windowed reads) and `StemWriter` (both stems written in
  one pass, 44-byte header rewritten at each window boundary so the files on disk
  are always a playable prefix), plus a resume record checked field by field:
  mismatch means start over, never "continue from a stale prefix" and never
  "fail with a broken file behind you". Every failure, cancellation included, keeps
  that checkpoint; [`stream::discard_staging`] is the caller's own, explicit way to
  say the partial result is worthless, and no engine calls it for you. One arm of
  this crate used to throw the pair away on `Cancelled`, which contradicted the
  trait and the other arm, and a user who pressed stop is the last person who
  wants an hour of inference deleted.
- `mem`: per-platform memory figures (macOS `phys_footprint`, Windows available
  commit, Linux `/proc`), the refusal text classifier, and the pre-flight gate
  that prices a window before the first forward.
- `engine::onnx` is the ONNX Runtime engine: windowed forward, linear crossfade,
  streaming output, checkpoint/resume, cancellation between windows, the gate.
  The arena is off by default; that choice is measured (8.83 GB peak with it on,
  3,440 MB with it off, identical output).
- `engine::mlx` (`--features mlx`, Apple Silicon): the architecture
  re-implemented in Rust: own STFT, mel band split, alternating time/frequency
  transformer blocks, mask estimator. Accepted against a PyTorch FFT reference
  rather than the ONNX graph's output, for the reason recorded in its module
  docs. `--features mlx` needs a provisioned MLX; see README.
- `tools/reduce_window.py`: the same window edit, offline and byte-exact;
  `tools/extract_onnx_weights.py`: the 672-tensor `.safetensors` the MLX engine
  loads, derived from the stock export. Neither redistributes weights, and this
  repository contains no model file and no audio.
- `rust-roformer`, the command line: `--model`, `--out`, `--window` (a sample
  count, or seconds with an `s` suffix), `--engine`, `--threads`, `--fresh`,
  `--quiet`. Seven options, each one a thing the library can already be asked to
  do and none a promise the crate does not make, hence no `--gpu` (the ONNX arm
  is CPU, the MLX arm is the Apple Silicon path) and no decoder beyond WAV. It
  builds with either engine or neither, and `--version` says which it got; exit
  codes are 0 done, 1 failed, 2 mistyped; a failure that means "this machine is
  too small for that window" says so and names the knob to turn. Stopping it is
  safe by construction rather than by a signal handler, because the pair on disk
  *is* the checkpoint: measured on a 12-second track interrupted eight seconds in
  (`SIGINT`, two of five windows flushed), the identical command resumed from
  frame 220,500 and wrote stems byte-identical to the uninterrupted run.
- `examples/bench.rs`: `synth` (deterministic test track, no model needed),
  `run`, and `resume-check` (cancelled-then-continued output must be
  byte-identical to uninterrupted, or it fails with the offset of the first
  difference). `--window` reshapes the stock export at load time, and every row
  ends with the window read back off the live session plus the count of windows
  that came off disk, so a pass that quietly re-ran everything cannot be mistaken
  for a resume. `scripts/bench_pair.sh` + `scripts/pair_summary.awk` do
  same-session, order-swapped A/B and print `inconclusive` when the difference is
  inside the position term or the scatter; `--window-a/--window-b` let one file be
  measured at two windows.
- `scripts/audit_public.sh`: refuses to let machine paths, account-looking
  strings, credentials or model/audio binaries into the tree. Its own banned list
  is split in two, because a gate that is published cannot carry the identifiers
  it guards: generic shapes (home directories, AWS key patterns, private-key
  headers, email domains, URL credentials) live in the script, the naming-specific
  patterns live in a gitignored `scripts/audit-local.txt` that a public clone does
  not have. The script fails if that file is ever tracked, and each class of leak
  (path, private name, weight file, tracked local list) was planted and proved to
  turn it red. The scan has a second subject as well: **commit metadata**, because
  that is the one thing `git push` publishes which a file walk cannot see. The walk
  excludes `.git`, which is right for weights and wrong for identities. The same two
  pattern lists are run over `git log`'s author and committer lines; the check was
  motivated by eleven commits in this repository's own pre-rewrite history carrying a
  personal address that every file in the tree had been held clean of, and the
  history that is pushed now carries none. Being exact about the ordering: an earlier
  entry here described this sweep, and the commit that carried it touched only
  `CHANGELOG.md`, so the description sat in a published file with no code behind it.
  That is the failure mode this gate exists to catch in someone else's work, so it is
  named rather than smoothed over: the sweep exists as of this entry, and its probe
  is an empty commit authored by a banned address, which turns the run red on one
  generic domain pattern and one local name pattern. Commits reachable only from
  other refs (a rewrite's backup branch, filter-branch's own `refs/original`) produce
  a note carrying a *count*, not the identities: a plain push does not send them,
  failing on them would block the push that lands the fix, and printing them would
  write into the transcript the strings this gate keeps out of the repository.
- `scripts/demo.sh` + `docs/demo.md`: a listening example that costs the
  repository nothing it does not already refuse to carry: two recordings from
  Wikimedia Commons (an aria over an orchestra, and an a cappella control), named
  by URL, pinned by SHA-256 at both the source file and the 12-second excerpt,
  fetched and decoded on the user's machine under their attribution-only licences
  (CC BY 2.5 / CC BY 3.0). The page reports what those exact bytes measure:
  63.3 dB of sub-140 Hz given up by the vocal stem while the voice band holds to
  0.1 dB, the background stem 39.8 dB under the input on material that has nothing
  to remove. It also says that none of it is an SI-SDR figure, because
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
  without default features, and an MSRV job so `rust-version` is checked and not
  just declared.

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
  `ort-sys`'s `edition2024` manifest, while the CI job asked only about the
  engine-free build, which meant the badge was green and false at the same time.
  The crate was never published, so no one is being moved. Worth being exact about
  the side effect: raising the number surfaced two
  `clippy::manual_is_multiple_of` sites the old value had legitimately silenced
  (`u*::is_multiple_of` is 1.87+), and applying that method in `graph.rs` is what
  killed the last remnant of the old claim. The engine-free build no longer
  compiles on 1.75 either, so the first version of this bullet, which said CI
  would check "1.75 without an engine", was itself wrong, and is corrected here.
- `Cargo.toml` gained `repository` and `homepage`; the README's install block
  names the real git URL instead of an `OWNER` placeholder, so a reader can install
  this today without waiting for a release. The crate docs and `rust-roformer --help`
  each state the home URL once, and a test asserts the help text keeps it.
- The README carries a CI badge. The placeholder comment above the badges had been
  holding three of them back until "the remote and the first release exist"; the
  remote exists now and the workflow passes on it (fmt-clippy, msrv, and the test
  matrix on Linux/macOS/Windows, all green on `5c57d71`), so the CI badge is filled
  in and the crates.io and docs.rs badges stay withheld: the crate is still
  unpublished, and their URLs would 404.
- `docs/benchmarks.md` now says plainly that the two engines' pre-flight numbers are
  different kinds of objects. They read alike in a table and are not alike: the ONNX
  arm consults `estimate_forward_mb`, a `c + q·T²` curve through the two measured
  Windows commit anchors that moves with `--window` and labels its own answer
  measured / interpolated / extrapolated; the MLX arm consults `per_forward_mb`, an
  analytic sum of the graph's shapes pinned to 2,800–3,200 MB by a test, which
  cannot move because `resolve_window` accepts only the native 352,800; there a
  shorter window is a different attention length, a quality change rather than a
  buffer resize. So the 4 s and 2 s figures are ONNX figures, and there is no
  shorter-window MLX number to add.
- Two ONNX windows measured with this crate's own binary, one fresh process each,
  order-swapped: **2 s peaks 4,567 / 4,573 MB, 4 s peaks 6,919 / 6,640 MB** (16 GB
  Apple Silicon, release build, 4 threads, stock export reshaped at load, the 12 s
  Commons excerpt `docs/demo.md` pins by SHA-256; `/usr/bin/time -l` independently
  reported 4,789,358,864 B = that same 4,567, with a lower maximum resident set, as
  it should be). The finding is in the comparison rather than the numbers: measured,
  the 4 s/2 s ratio is **1.45–1.52** while the fitted curve implies **3.34**. That is
  a floor, not a contradiction: about 3 GB of the peak is weights and runtime, which
  the per-forward estimate never claimed to price and which `preflight` cannot see,
  because `OnnxEngine::load` only inspects the graph header and the session is built
  afterwards. The documented consequence is a band, not a slogan: a machine whose
  free memory sits *between* the estimate and the real peak (1.5 GB to 4.6 GB for a
  2 s window) is accepted by the gate and then overflows it. **Whether the gate
  should also price the graph itself is left open, not decided here**: that is a
  behaviour change to a shipped check, so the doc records what was measured and does
  not pre-empt the call.
- `mem::tests::peak_mb_while_tracks_a_sustained_climb` stopped asserting something
  only true in a quiet process. It was the one red job on `9ce24da`, a docs-only
  commit on code that had passed the same job twice, and the failure was
  `panicked at src/mem.rs:1133: peak below the reading taken before the bracket`
  from `cargo test --no-default-features` on the Windows runner, exit 1, a real
  assertion rather than a compile error. Root cause, read out of the code rather
  than guessed: `peak_mb_while` returns the max of **samples** of `held_mb`, and
  `held_mb` is a *current* quantity on Windows (`PagefileUsage`) and Linux
  (`VmSize`): both fall the moment memory is freed. `cargo test` runs the arms as
  threads in one process, so a sibling arm releasing its buffers inside the bracket
  can push every sample below the reading taken before it. macOS cannot fail this
  way, which is why twelve local runs were green and the Windows runner was not: its
  footprint ledger stays charged.
- The cross-bracket assertion is gone and an in-bracket one takes its place: no
  sample may exceed the OS's own lifetime high-water (`PeakPagefileUsage` / `VmPeak`
  / `peak_footprint`), which holds under any amount of sibling activity. Both
  directions were proved rather than asserted: making the sampler over-report by
  4,000,000 MB fails the new check (`sampled peak 4000097 MB exceeds the OS-reported
  lifetime peak Some(97) MB`), and killing the sampler entirely fails the surviving
  not-trailing check, which is also why a third assertion that was sketched along
  the way got deleted: the mutation showed the existing one already fires, so it
  would have been dead weight. One caveat on the proof, stated because it nearly
  made the proof worthless: the first mutation run "passed" only because
  `--exact mem::tests::…` was filtered by bare name and matched **zero** tests; the
  rerun that printed `running 1 test` is the one that counts.
- The first fix was not enough. `d4c1ebc` removed the
  cross-bracket assertion and Windows still failed the same job, because the
  assertion left standing (`peak + 1 >= during`) compares the same two kinds of
  values: a max over samples and one live reading, taken at different instants, of
  a quantity that falls. Dropping one invalid comparison and keeping another was
  editing the symptom. The arm is now `tests/peak_sampler.rs`, an integration test,
  which cargo builds as its own binary and therefore its own process: inside it the
  only thing that moves the process's memory is its own 96 MiB, so both comparisons
  are sound and neither depends on thread luck. Same class as #189 in the
  application tree: an arm reading a process-wide counter while a parallel suite
  runs. It was the second time that shape bit.
- Recorded because it bounds what the new file is worth: **on macOS this arm cannot
  fail.** Gutting the sampler's loop entirely leaves it green here, because
  `phys_footprint` never falls and the single read taken on the way out of the
  bracket still reports the high-water. Its teeth are on Windows and Linux, the two
  platforms whose counters do fall, which is also the only place it has ever
  reported a failure. The file says so in its own header rather than claiming a
  portable guard, and the CI matrix (not a local green run) is the experiment that
  validates it.
- `docs/benchmarks.md` gains the Linux row its instrument table never had, and says
  out loud what the above implies for every number this file prints as
  `peak N MB`: on Windows and Linux it is a **lower bound** on the OS-recorded
  high-water, because a spike shorter than the sampling interval is missed. macOS
  is the exception, and the exception is verified: the printed peak matched
  `/usr/bin/time -l` byte for byte.
- Left open, not silently changed: **whether the reported peak should become the OS's
  own lifetime counter** rather than a sampled max. It would make the number exact on
  all three platforms and the test trivial; it would also change what every peak
  figure in this repository's output and docs means on Windows and Linux (they would
  rise, and would include maxima this sampler never saw). That is a user-visible
  semantics change to a shipped figure, so it waits for a decision instead of riding
  in with a test fix.
- `actions/checkout` went from v4 to v7 in `ci.yml`. Not cosmetic: v4 still
  declares the Node 20 runtime the runners dropped, so every job printed a
  deprecation warning next to an otherwise clean run — and a warning that is always
  there stops being a signal, which is the same failure mode as a badge that is
  always green. Checked before bumping rather than after: v5/v6/v7's `action.yml`
  all say `using: node24`, and the one breaking change shared across those three
  (the safer `pull_request_target` defaults, backported as far down as v4) does not
  reach this workflow, which triggers on `pull_request` where the token is already
  read-only and passes no inputs to checkout at all.
- The README's window examples are now the round set `8 s / 4 s / 2 s`, and a `2.5s`
  one is gone: what a reader copies out of a README should be what the memory figures
  cover. The parser still takes any hop multiple including fractional seconds (a
  test exercises `2.5s` as a *format*, which is what it was always testing), and the
  hop rule refuses what it always refused.
- Two `chunks_exact(<constant>)` call sites became `as_chunks::<N>().0`: clippy
  1.98.0 added `chunks_exact_to_as_chunks`, and the `fmt-clippy` job runs `-D
  warnings` against rolling stable, so the badge went red on the first push after
  that release landed without a single line of this crate's code changing. Both
  sites are test helpers, not the shipped read path (one whole-buffer WAV oracle
  in `audio.rs`, one gguf-float decoder in the MLX module), and the rewrite is
  iteration-identical, so the suites report the same counts as before it
  (`as_chunks` is 1.88-stable, which is also this crate's floor, and the MSRV job
  checks `--all-targets` so it would notice if that stopped being true). Worth
  naming the shape of the failure rather than burying it: a gate pinned to a
  moving target is a gate that will go red on someone else's release day, and that
  is a maintenance decision, not an accident of this diff.
- The README, the crate docs and `docs/benchmarks.md` now name the application this
  code came out of (DeepVideo) rather than describing it as "a desktop
  video-translation application". It is the same sentence that says where the
  45.8-minute and Windows figures were instrumented, which is what makes a
  provenance usable: the numbers a public test suite cannot carry are exactly the
  ones a reader most needs an origin for. Nothing else about the
  relationship changed: no product code, no weights, no audio and no dependency on
  the application live here, and the crate still runs with the network off.
