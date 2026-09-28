#!/usr/bin/env bash
# Pre-publication gate: nothing from the environment this code was written in may
# survive into the repository. Run before every push to a public remote:
#
#     ./scripts/audit_public.sh
#
# Exits non-zero and prints the offending lines. Patterns are literal-ish; add to
# BANNED rather than deleting a finding.
#
# The gate is itself published, so it cannot carry the identifiers it guards: a
# banned-string list that names an account prefix discloses that account prefix.
# The patterns therefore split in two. What follows is the generic part — shapes
# and paths that are public knowledge about Rust projects. The personal and
# project-specific part lives in `scripts/audit-local.txt`, which is gitignored,
# one regex per line, `#` for comments. A clone without that file still runs the
# generic gate; it just cannot check for somebody else's account names, and that
# is the correct behaviour for a public repository.
#
# Two subjects, not one: the file walk below cannot see `git log`, and a push
# publishes author and committer identities exactly as loudly as it publishes
# source lines. Hence the identity sweep further down, over the same patterns.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

BANNED=(
  '/Users/'
  'C:\\Users\\'
  '/home/[a-z0-9_.-]+/'
  '~/'
  'AKIA[0-9A-Z]{16}'
  'aws_secret'
  'AWS_SECRET'
  'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY'
  '@163\.com'
  '@qq\.com'
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(cn|ru|pro|top)\b'
  'https?://[^ ]*:[^@ ]*@'
  '\.env\b'
  '\.pem\b|\.p12\b|\.key\b'
  'TODO.*(remove|before publishing|secret)'
)

LOCAL_LIST=scripts/audit-local.txt
if [ -f "$LOCAL_LIST" ]; then
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    BANNED+=("$line")
  done <"$LOCAL_LIST"
  echo "audit: generic list + $(grep -cvE '^\s*(#|$)' "$LOCAL_LIST") local pattern(s) from $LOCAL_LIST"
else
  echo "audit: generic list only — $LOCAL_LIST not present (expected in a public clone)"
fi

status=0
for pat in "${BANNED[@]}"; do
  hits=$(grep -rIn --binary-files=without-match \
      --exclude-dir=target --exclude-dir=.git --exclude-dir=bench-out --exclude-dir=rehearsal \
      --exclude=audit_public.sh --exclude=audit-local.txt --exclude=Cargo.lock \
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

# Second subject for the same pattern lists: `git push` publishes author and
# committer identities as loudly as it publishes source lines, GitHub search
# indexes them, and a rewrite after the fact is visible in every fork that was
# cloned in between. The file walk above cannot see any of that because it
# excludes `.git` — which is the right call for weights and wrong for metadata.
identities=$(git log --format='author %an <%ae>%ncommitter %cn <%ce>' HEAD 2>/dev/null)
if [ -z "$identities" ]; then
  echo "audit: no commits reachable from HEAD — identity sweep skipped" >&2
else
  for pat in "${BANNED[@]}"; do
    hits=$(printf '%s\n' "$identities" | grep -E "$pat" 2>/dev/null)
    if [ -n "$hits" ]; then
      printf '\n\033[31mHIT (commit metadata)\033[0m %s\n%s\n' "$pat" "$hits"
      status=1
    fi
  done
  # Commits that only other refs reach — a rewrite's backup branch, filter-branch's
  # own refs/original — are not sent by a plain `git push`, so this is a note and not
  # a failure: failing here would block the push that deletes them. The count is
  # printed and the identities are not, because naming them here would put the exact
  # strings this gate exists to keep out of the transcript.
  other=$(git rev-list --count --all --not HEAD 2>/dev/null || echo 0)
  if [ "$other" != 0 ]; then
    printf '\naudit: note — %s commit(s) reachable only from refs other than HEAD; not pushed by `git push`, deleted after it lands\n' "$other"
  fi
fi

# A tracked file that only exists on this machine is the same leak in a different
# shape: `git ls-files` is what a public remote would publish. And crates.io
# publishes what `include` selects, which outranks .gitignore entirely — so the
# packaging manifest gets checked too, not just the index.
if [ -f "$LOCAL_LIST" ]; then
  if git ls-files --error-unmatch "$LOCAL_LIST" >/dev/null 2>&1; then
    printf '\n\033[31mTRACKED\033[0m %s is tracked — the private pattern list must stay local\n' "$LOCAL_LIST"
    status=1
  fi
  packaged=$(cargo package --list --allow-dirty 2>/dev/null)
  if printf '%s\n' "$packaged" | grep -qxF "$LOCAL_LIST"; then
    # One format string on one line: a backslash-newline inside single quotes is a
    # literal backslash, and this message would print it.
    printf '\n\033[31mPACKAGED\033[0m %s would go into the crates.io tarball — name the published files in Cargo.toml include instead of globbing the directory\n' "$LOCAL_LIST"
    status=1
  fi
fi

if [ "$status" = 0 ]; then
  echo "audit OK: no forbidden strings in files or commit metadata, no weight/audio files, private list untracked"
fi
exit $status
