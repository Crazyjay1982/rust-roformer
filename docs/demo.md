# Listening

This crate ships no audio and no weights, so nothing here lets you *hear* it
yet — and every quality claim in [benchmarks.md](benchmarks.md) is about memory,
windows and resumption rather than about sound. This page is the missing part:
two recordings you are allowed to run a remover on, one command, and the numbers
those exact bytes produce.

```sh
./scripts/demo.sh --model melband_roformer_vocals.onnx            # aria + orchestra
./scripts/demo.sh --model melband_roformer_vocals.onnx --source bright   # a cappella
```

The script needs `curl` and `ffmpeg` (the crate reads WAV only, and both sources
are Ogg). It fetches the file, checks its SHA-256, cuts a 12-second excerpt, runs
`examples/bench.rs` on it, and prints where the three files are. Everything lands
in `demo-audio/` and `demo-out/window-<n>/` under the directory you ran it from;
neither is in the repository, and `.gitignore` makes sure a stray `git add -A`
cannot put them there. `--window`, `--seconds` and `--source` are passed straight
to the harness, so this is also the shortest way to watch the memory gate refuse:

```text
$ ./scripts/demo.sh --model melband_roformer_vocals.onnx --window 352800
bench: memory gate: this window needs ~19400 MB per forward, 14281 MB available
```

That is exit 1 on a 16 GB machine before anything is allocated. The window the
script asks for by default (176400 = 4 s) is not a workaround baked into a
different file: it is the same 953 MB export, reshaped in memory at load time.

Once the excerpt exists, the tool you would actually install can run on it —
the harness is for measuring, the command line is for working:

```sh
rust-roformer --model melband_roformer_vocals.onnx --window 4s -o stems demo-audio/aria_12s.wav
```

Both paths produce the same bytes for the same window (checked on these excerpts,
`vocals.wav` and `background.wav` each identical), which is the least this crate
could do while claiming the harness measures what it ships. The README's
[Quick start](../README.md#quick-start) shows that command's resume behaviour
verbatim: a run stopped eight seconds into a twelve-second track, the same line
typed again, and `continued from frame 220,500` in the summary.

## The two sources

Both are on Wikimedia Commons, and the licence line below is what the file's own
page declares — read off the API and the page templates on 2026-09-28, not
inferred from the filename. Neither recording is ours, neither is distributed by
this repository, and the credit lines belong with any audio you make from them.

| | `aria` | `bright` |
| --- | --- | --- |
| What it is | Mozart, "Der Hölle Rache" (*Die Zauberflöte*), live 2006 Bangkok Opera performance: Sandra Partridge, Siam Philharmonic Orchestra cond. Trisdee na Patalung | "Bright College Years", vv. 1 and 3, sung by the 2006 Yale Whiffenpoofs (lyrics Henry Durand, 1881) |
| Why it is here | the hard case: a soprano over a full orchestra, with coloratura that occupies the same bands as much of the accompaniment | the control: a cappella, so there is nothing to remove and any output in the background stem is a false positive |
| Licence as declared | uploader's own work, offered under GFDL / CC BY-SA 3.0 / **CC BY 2.5**; used here under CC BY 2.5 (the attribution-only option) | **CC BY 3.0** for the recording (file page states lyrics are public domain) |
| File page | [File:Der Hoelle Rache.ogg](https://commons.wikimedia.org/wiki/File:Der_Hoelle_Rache.ogg) | [File:Bright College Years.ogg](https://commons.wikimedia.org/wiki/File:Bright_College_Years.ogg) |
| Bytes | 3,757,004 | 1,407,002 |
| SHA-256 (the `.ogg` as served) | `bbdf0a8d4c151aee5a21fb71ed86894b1aae5c7dba9ea767f7af6c0f752915c2` | `066e4f0dea963de65966c0531e4bdead2f296ec8ef6076c2eaffcbe054f3c6b7` |
| Decoded | 44.1 kHz stereo Vorbis, 190.6 s | 44.1 kHz stereo Vorbis, 103.7 s |
| Excerpt the script cuts | 12 s from t = 30 s | first 12 s |
| SHA-256 of the excerpt WAV | `8c223add4a20c14bda1250af87a41b543db7092963d4bb81a702e9ff6937314a` | `536051d1b179833ff56b6492f92b5615aefef125520e708771d79a07531cc8ea` |

The excerpt hash matters more than it looks: `ffmpeg` gives a different file for
`-ss` before `-i` than for `-ss` after it (fast container seek versus decode and
discard), and both differ from what you get by re-encoding at another sample
rate. The script pins the command *and* the result, so the numbers below are
attached to bytes rather than to a description of a procedure. The aria excerpt
starts at 30 s because the track opens with an orchestral tuti, and a demo of a
vocal remover should not spend its first twelve seconds on instruments.

## What those bytes measure

Separated on the stock export at `--window 176400`, on a 16 GB Apple Silicon
laptop, 2026-09-28. Numbers are RMS or band-energy ratios relative to the input,
computed from the excerpt and the two stems; `band` sums `|X(f)|²` over one
window-length FFT of the whole 12 s.

| | `aria` (soprano + orchestra) | `bright` (a cappella) |
| --- | --- | --- |
| RMS: input / vocals / background | −21.7 / −22.1 / −32.7 dBFS | −15.3 / −15.3 / −55.1 dBFS |
| Background stem, relative to input | −11.0 dB | **−39.8 dB** |
| Under 140 Hz, in the vocals stem | **−63.3 dB** | −0.0 dB |
| Under 140 Hz, in the background stem | −0.0 dB | −32.0 dB |
| Voice band 200 Hz – 4 kHz, vocals | −0.1 dB | +0.0 dB |
| Voice band 200 Hz – 4 kHz, background | −16.1 dB | **−53.0 dB** |
| Above 5 kHz, vocals / background | −0.3 / −14.7 dB | +0.0 / −57.8 dB |
| SI-SDR(vocals + background, input) | 75.4 dB | 80.6 dB |
| Peak of the loudest stem | −5.9 dBFS | −1.7 dBFS |

What each row is for:

* **The aria's low end is gone from the vocal stem** — 63 dB of it — and is still
  in the background stem (−0.0 dB versus the input). That is a real separation,
  not a gain change: the two stems differ in *content*, not in level.
* **The voice band survives** at −0.1 dB in the vocal stem while the background
  loses 16 dB of it. Colouratura at the top of the range and cymbals share a
  band, which is why 16 dB is a modest-looking number for an audible result.
* **`SI-SDR(vocals + background, input)` ≈ 75–80 dB** says the model partitions
  the mixture: adding the stems gives the input back to within numerical noise.
  A mask that only attenuated would show up here as a large residual instead.
* **The a cappella is the negative control.** With nothing to remove, the
  background stem sits 39.8 dB under the input and 53 dB down in the voice band,
  and the vocal stem is the input at 39.7 dB SI-SDR. Nothing was invented to
  occupy the empty stem.

Neither row is an SI-SDR *improvement* figure, and it would be wrong to quote it
as one: there is no ground-truth a cappella track for the aria, so the honest
measurements are the ratios above. Papers report 6–10 dB SI-SDR on MUSDB-style
test sets with exact stems; those numbers are not comparable to anything here.

## Please do not judge quality with synthesized audio

`examples/bench.rs synth` — the fixture the [README transcript](../README.md#quick-start)
runs on — writes two alternating tones (196 Hz and 980 Hz, with one harmonic
each) in 1.7-second phrases, plus deterministic hash-noise to keep every window
non-silent. It is the right input for everything that is about *shape and
behaviour*: which windows the gate refuses, which windows the hop rule refuses,
what a resume checkpoint records, whether a resumed run is byte-identical. It is
the wrong input for asking "does this work".

Run it through the same 12 s / 176400 path and the answer is blunt:

```text
vocals stem:   all 2,116,800 samples are exactly 0
background:    1.000 of the input's energy
```

Not "poor quality" — *digital silence*. The checkpoint decided a sine wave is not
a voice and put none of it in the vocal stem. That is the model behaving as
trained, and it is also why the byte-identity claims in this repository are still
worth something: a pipeline that returns exactly zero on input it rejects is the
same pipeline that returns exactly the same bytes on restart.

It gets worse before it gets better. A hand-built "voice-like" signal — two
harmonic singers with formant weighting, vibrato and syllable envelopes over a
guitar-ish bed, with the exact vocal component retained as ground truth — went
through the same path, and this time the vocal stem was not empty:

| Comparison | SI-SDR against the true vocal |
| --- | --- |
| Doing nothing (the mixture itself) | **3.89 dB** |
| The vocals stem this crate produced | **−10.35 dB** |
| RMS of that stem, relative to the ground-truth vocal | 0.16× |

So the separator did less good than not running it at all, because most of the
"voice" stayed in the *background* stem. Nothing in the port is broken here — the
model has never seen a sawtooth with vibrato — but a bug report filed with
synthesized input will read as one. If you need a reproducible input that is not
someone's recording, use the `bright` control: real voice, nothing to remove, and
a number (−39.8 dB in the background stem) that says whether the pipeline is
intact.

