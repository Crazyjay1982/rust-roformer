#!/usr/bin/env bash
# Paired A/B timing for rust-roformer.
#
# WHY THIS SCRIPT EXISTS INSTEAD OF `time cargo run …` twice:
# on the machine this crate was developed on, the SAME build, SAME model and SAME
# input measured 88 s in one session and 158 s in another. Anything you conclude
# from two separate sessions is a statement about the time of day. So: every
# configuration runs both first and second, in one session, and a difference only
# counts if it survives the swap.
#
#   ./scripts/bench_pair.sh --model-a melband_roformer_vocals.onnx \
#                          --model-b melband_roformer_vocals_4s.onnx \
#                          --input track.wav
#
# Or, to measure the in-memory reshape instead of a second file: the SAME model
# on both arms, one of them given a window:
#
#   ./scripts/bench_pair.sh --model-a melband_roformer_vocals.onnx \
#                          --model-b melband_roformer_vocals.onnx --window-b 176400 \
#                          --input track.wav
#
# Use --engine-a/--engine-b instead of --model-* to compare onnx vs mlx on one
# file you can load both ways (the MLX engine wants the .safetensors produced by
# tools/extract_onnx_weights.py, so usually you pass both --model-* separately).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

model_a=""; model_b=""; input=""; engine_a="onnx"; engine_b="onnx"
window_a=""; window_b=""
repeat=1; threads=4; out="bench-out"; features="onnx"

while [ $# -gt 0 ]; do
  case "$1" in
    --model-a)  model_a="$2";  shift 2 ;;
    --model-b)  model_b="$2";  shift 2 ;;
    --engine-a) engine_a="$2"; shift 2 ;;
    --engine-b) engine_b="$2"; shift 2 ;;
    --window-a) window_a="$2"; shift 2 ;;
    --window-b) window_b="$2"; shift 2 ;;
    --input)    input="$2";    shift 2 ;;
    --repeat)   repeat="$2";   shift 2 ;;
    --threads)  threads="$2";  shift 2 ;;
    --features) features="$2"; shift 2 ;;
    --out)      out="$2";      shift 2 ;;
    -h|--help)  sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "bench_pair: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -n "$input" ] && [ -f "$input" ] || { echo "bench_pair: --input FILE is required" >&2; exit 2; }
if [ -z "$model_a" ] || [ -z "$model_b" ]; then
  if [ "$engine_a" = "$engine_b" ]; then
    echo "bench_pair: give --model-a and --model-b (or make --engine-a/--engine-b differ)" >&2
    exit 2
  fi
fi
[ "$model_a" = "$model_b" ] && [ "$engine_a" = "$engine_b" ] &&
  [ "$window_a" = "$window_b" ] && {
  echo "bench_pair: A and B are the same configuration; nothing to pair" >&2; exit 2; }

BENCH=target/release/examples/bench
echo "bench_pair: building (features: $features)" >&2
cargo build --release --features "$features" --example bench || exit 1
[ -x "$BENCH" ] || { echo "bench_pair: $BENCH not produced" >&2; exit 1; }

mkdir -p "$out"
ts=$(date +%Y%m%d-%H%M%S)
log="$out/pair-$ts.log"

label_a="A($engine_a:$(basename "${model_a:-none}")${window_a:+@$window_a})"
label_b="B($engine_b:$(basename "${model_b:-none}")${window_b:+@$window_b})"

run() { # $1 label  $2 engine  $3 model  $4 window
  echo "  · $1" >&2
  "$BENCH" run --engine "$2" ${3:+--model "$3"} --input "$input" \
      ${4:+--window "$4"} --threads "$threads" --repeat "$repeat" \
      --out "$out/$ts" --label "$1"
}

{
  echo "# session $(date -u +%FT%TZ)  input=$input  threads=$threads  repeat=$repeat"
  echo "## order A,B"
  run "$label_a" "$engine_a" "$model_a" "$window_a"
  run "$label_b" "$engine_b" "$model_b" "$window_b"
  echo "## order B,A"
  run "$label_b" "$engine_b" "$model_b" "$window_b"
  run "$label_a" "$engine_a" "$model_a" "$window_a"
} 2>&1 | tee "$log"

echo
awk -f scripts/pair_summary.awk "$log"

echo
echo "Read this before concluding anything:"
echo "  * the two numbers above are the SAME configuration in the two orders; a"
echo "    gap between them is the position effect, not a result"
echo "  * peak_MB is a process-lifetime high-water mark on macOS, so compare it"
echo "    across separate processes only — which is what each run above is"
echo "  * absolute seconds from this session do not transfer to another session"
echo "  * log: $log"
