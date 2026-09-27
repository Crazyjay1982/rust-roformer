#!/usr/bin/env bash
# Pre-publication gate: nothing from the environment this code was written in may
# survive into the repository. Run before every push to a public remote:
#
#     ./scripts/audit_public.sh
#
# Exits non-zero and prints the offending lines. Patterns are literal-ish; add to
# BANNED rather than deleting a finding.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

BANNED=(
  'deepvideo'
  'DeepVideo'
  'echo-frame'
  'sunjie'
  '/Users/'
  'C:\\Users\\'
  'angi_'
  '@163\.com'
  '@qq\.com'
  'AKIA[0-9A-Z]{16}'
  'aws_secret'
  'AWS_SECRET'
  'sentry'
  'SENTRY'
  'r2\.dev'
  'cloudflareinsights'
  'sgp3'
  'gitee'
  'models/roformer'
  'ROFORMER_MLX_WEIGHTS'
  '~/tmp/'
)

status=0
for pat in "${BANNED[@]}"; do
  hits=$(grep -rIn --binary-files=without-match \
      --exclude-dir=target --exclude-dir=.git --exclude-dir=bench-out \
      --exclude=audit_public.sh \
      -E "$pat" . 2>/dev/null)
  if [ -n "$hits" ]; then
    printf '\n\033[31mHIT\033[0m %s\n%s\n' "$pat" "$hits"
    status=1
  fi
done

# Any real weight or audio file committed? None may be.
weights=$(find . -path ./target -prune -o \
  \( -name '*.onnx' -o -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' \
     -o -name '*.wav' -o -name '*.flac' -o -name '*.mp3' \) -print 2>/dev/null)
if [ -n "$weights" ]; then
  printf '\n\033[31mBINARY ASSET\033[0m model weights or audio in the tree:\n%s\n' "$weights"
  status=1
fi

if [ "$status" = 0 ]; then
  echo "audit OK: no forbidden strings, no weight/audio files"
fi
exit $status
