#!/usr/bin/env bash
# One-command listening example on real, freely licensed audio.
#
# WHY THIS SCRIPT EXISTS: every quality claim in this crate's docs is about
# memory, windows and resumption, and a reader still cannot *hear* the thing
# without finding a recording they are allowed to run a remover on. The two
# sources below are on Wikimedia Commons, one under CC BY 3.0 and one under
# CC BY 2.5, i.e. redistribution-with-attribution licences that survive being
# named in a demo. The bytes are pinned by SHA-256, so the excerpt you hear is
# the excerpt the numbers in docs/demo.md were measured on.
#
# This crate ships no audio and no weights (see NOTICE); the download goes to
# your working directory, not into the repository.
#
#   ./scripts/demo.sh --model melband_roformer_vocals.onnx
#   ./scripts/demo.sh --model melband_roformer_vocals.onnx --source bright
#   ./scripts/demo.sh --model melband_roformer_vocals.onnx --window 352800
#
# The last one is the interesting failure: on a 16 GB machine the memory gate
# refuses the stock 8 s window before allocating anything, which is the point
# the README makes in prose. Run it once, read the message, then drop --window.
set -uo pipefail
# The downloaded audio is resolved against where the user stood, not against the
# repository: this script has to `cd` to the crate root to find cargo and the
# built example, and a 12-second WAV landing next to `src/` is a `git add -A`
# accident waiting to happen.
invoked_at="$PWD"
cd "$(dirname "$0")/.." || exit 2

model=""; source_name="aria"; seconds=12; window=176400
work=""; out=""

usage() {
  cat <<'EOF'
usage: ./scripts/demo.sh --model PATH [--source aria|bright] [--seconds N]
                         [--window N] [--work DIR] [--out DIR]

  --model   the ONNX export (see the README; nothing is downloaded for you)
  --source  aria  = soprano with orchestra, CC BY 2.5   (default)
            bright = a cappella, CC BY 3.0
  --window  samples per forward; 176400 = 4 s, the default. 352800 is the stock
            8 s graph and is what the memory gate refuses on a 16 GB machine —
            worth seeing once. Empty string means "whatever the graph declares".
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)   model="$2";   shift 2 ;;
    --source)  source_name="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    --window)  window="$2";   shift 2 ;;
    --work)    work="$2";     shift 2 ;;
    --out)     out="$2";      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "demo: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$model" ]; then
  echo "demo: --model is required (the ONNX export; see the README on where to get it)" >&2
  exit 2
fi
if [ ! -f "$model" ]; then
  echo "demo: no model at '$model'" >&2
  exit 2
fi
for tool in curl ffmpeg; do
  command -v "$tool" >/dev/null 2>&1 || { echo "demo: needs '$tool' on PATH to fetch and decode the source" >&2; exit 2; }
done
sha() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# The two sources, pinned. `offset` is where the excerpt starts: the aria opens
# with an orchestral tuti, and a demo of a *vocal* remover should not spend its
# first twelve seconds on instruments.
case "$source_name" in
  aria)
    url="https://upload.wikimedia.org/wikipedia/commons/7/7d/Der_Hoelle_Rache.ogg"
    want="bbdf0a8d4c151aee5a21fb71ed86894b1aae5c7dba9ea767f7af6c0f752915c2"
    offset=30; rate_note="44.1 kHz stereo, 190.6 s in full"
    credit='Wolfgang Amadeus Mozart, "Der Hölle Rache" (Die Zauberflöte): live recording of the 2006 Bangkok Opera production by Sandra Partridge (soprano), the Siam Philharmonic Orchestra conducted by Trisdee na Patalung. File: Wikimedia Commons, https://commons.wikimedia.org/wiki/File:Der_Hoelle_Rache.ogg, offered by the uploader under GFDL / CC BY-SA 3.0 / CC BY 2.5 and used here under CC BY 2.5 (attribution only). No affiliation with Commons or the performers; this is not a recommendation of their work.'
    ;;
  bright)
    url="https://upload.wikimedia.org/wikipedia/commons/b/ba/Bright_College_Years.ogg"
    want="066e4f0dea963de65966c0531e4bdead2f296ec8ef6076c2eaffcbe054f3c6b7"
    offset=0; rate_note="44.1 kHz stereo, 103.7 s in full"
    credit='"Bright College Years", verses 1 and 3, performed by the 2006 Yale Whiffenpoofs (lyrics Henry Durand, 1881). File: Wikimedia Commons, https://commons.wikimedia.org/wiki/File:Bright_College_Years.ogg, CC BY 3.0 for the recording (the lyrics are public domain).'
    ;;
  *) echo "demo: --source is aria or bright, not '$source_name'" >&2; exit 2 ;;
esac

[ -n "$work" ] || work="$invoked_at/demo-audio"
# The window is part of the default output directory because the harness writes
# one directory per label and clears it before starting: run the interesting
# failure (`--window 352800`, which exits after deleting) and a following
# successful run would otherwise find (and remove) the pair from the run that
# just succeeded. Two invocations that asked for different graphs should not
# share a directory.
if [ -z "$out" ]; then
  out="$invoked_at/demo-out/window-${window:-native}"
fi
mkdir -p "$work" "$out" || exit 2
raw="$work/${source_name}.ogg"
wav="$work/${source_name}_${seconds}s.wav"

if [ -f "$raw" ]; then
  echo "demo: reusing ${raw} ($(wc -c <"$raw" | tr -d ' ') bytes)"
else
  echo "demo: downloading $url"
  curl -sSL --fail --retry 3 --retry-delay 2 -m 300 \
    -A "rust-roformer demo/0.1 (fetching a freely licensed test file)" \
    -o "$raw" "$url" || { echo "demo: download failed" >&2; exit 1; }
fi
got=$(sha "$raw")
if [ "$got" != "$want" ]; then
  echo "demo: SHA-256 mismatch, refusing to run on bytes we did not measure:" >&2
  echo "      expected $want" >&2
  echo "      got      $got" >&2
  echo "      Commons may have re-encoded or moved the file; if so, docs/demo.md's" >&2
  echo "      numbers are stale for this source and should be re-measured." >&2
  exit 1
fi
echo "demo: checksum ok ($got), $rate_note"

# `-ss` goes AFTER `-i` on purpose. Before `-i`, ffmpeg seeks by container and
# starts at whatever the decoder lands on, so the excerpt depends on how the
# encoder laid out its pages; after `-i`, it decodes from the start and discards
# N seconds, which is exact. The excerpt's bytes are what docs/demo.md measured,
# so exactness is the whole point.
ffmpeg -v error -y -i "$raw" -ss "$offset" -t "$seconds" \
  -acodec pcm_s16le -ar 44100 -ac 2 "$wav" || { echo "demo: ffmpeg decode failed" >&2; exit 1; }

bin="target/release/examples/bench"
if [ ! -x "$bin" ]; then
  echo "demo: building the bench example (release, default features)"
  cargo build --release --example bench || { echo "demo: build failed" >&2; exit 1; }
fi

echo "demo: separating $wav"
# No array here: macOS still ships bash 3.2, where an empty "${arr[@]}" is an
# unbound-variable error under `set -u`.
if [ -n "$window" ]; then
  "$bin" run --model "$model" --input "$wav" --out "$out" --label "$source_name" --window "$window"
else
  "$bin" run --model "$model" --input "$wav" --out "$out" --label "$source_name"
fi
rc=$?
if [ $rc -ne 0 ]; then
  echo "demo: bench exited $rc. If the message is the memory gate, that is the" >&2
  echo "      documented behaviour on a machine that cannot fund the window you" >&2
  echo "      asked for — try --window 176400, or a smaller multiple of 441." >&2
  exit $rc
fi

echo
echo "demo: listen to the three files"
echo "      input      $wav"
echo "      vocals     $out/$source_name#0/vocals.wav"
echo "      background $out/$source_name#0/background.wav"
echo
echo "demo: what to check, and what the measured numbers say: docs/demo.md"
echo "      Attribution for the source (kept with the audio, not in the crate):"
echo "      $credit"
