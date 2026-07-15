#!/bin/bash
# Fail if a production .rs file (>=100 lines) is dense with scratch notes.
#
# CLAUDE.md, "実装" section forbids メモ書き・作業ログ的コメント — NOT rustdoc.
# So this gate measures the NOTE line ratio only: `//` and `////`+ (which Rust
# does not treat as doc). rustdoc (`///`, `//!`) is deliberately NOT policed
# here — its per-block length is already bounded by check_comment_block_size.sh
# (MAX_DOC=24), and a rustdoc-density cap can only suppress the algorithm/API
# docs CLAUDE.md wants. Policing total comment density is exactly what falsely
# flagged conic/mod.rs (28.7% doc, 4.4% note) and qcqp_guard.rs (30.5% doc,
# 0.8% note): heavily-documented solver internals, not memo dumps.
#
# THRESHOLD_MEMO=20: measured over otspot-{core,io,model,dev}/src, the note-line
# ratio tops out at 16.6% (presolve/transforms/tests.rs — section banners plus
# test-intent notes), next 14.5% / 13.1%; all legitimate correctness/intent
# notes, none work logs. 20% clears that ceiling with headroom (so ordinary
# well-commented code is not churned by false positives) while flagging a file
# where >1 line in 5 is a scratch note — a density only a work log or a
# commented-out-code dump reaches. No allowlist is needed: every production
# file sits below 20% note density once rustdoc is excluded.
set -euo pipefail

# Max fraction (%) of note lines (`//`, `////`+) allowed in a production file.
readonly THRESHOLD_MEMO=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

HITS=""
while IFS= read -r f; do
  total=$(wc -l < "$f" | tr -d ' ')
  [ "$total" -lt 100 ] && continue
  # Count note lines only: a `//`-prefixed line that is NOT rustdoc (exactly
  # `///` or `//!`). `////`+ is a note, matching the block gate's split.
  memo=$(awk '
    /^[[:space:]]*\/\//{
      if ($0 !~ /^[[:space:]]*(\/\/\/([^\/]|$)|\/\/!)/) c++
    }
    END { print c + 0 }
  ' "$f")
  ratio=$(awk -v c="$memo" -v t="$total" 'BEGIN { printf "%.1f", c*100/t }')
  if awk -v r="$ratio" -v th="$THRESHOLD_MEMO" 'BEGIN { exit !(r >= th) }'; then
    HITS+="$ratio% note lines  $f"$'\n'
  fi
done < <(find otspot-core/src otspot-io/src otspot-model/src otspot-dev/src \
  -name '*.rs' -type f)

if [ -n "${HITS//[[:space:]]/}" ]; then
  echo "::error::Note-comment ratio >= ${THRESHOLD_MEMO}% (CLAUDE.md \"実装\" section):"
  echo "$HITS"
  exit 1
fi
echo "comment ratio check: OK"
