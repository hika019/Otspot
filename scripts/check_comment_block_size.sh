#!/bin/bash
# Fail if any production .rs file has a consecutive comment block >= MAX_BLOCK lines.
set -euo pipefail

MAX_BLOCK=40
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

declare -a ALLOWLIST=(
  "otspot-core/src/simplex/pricing.rs"
  "otspot-core/src/tolerances.rs"
  "otspot-core/src/simplex/dual_advanced/bounded_core.rs"
  "otspot-core/src/simplex/dual_advanced/phase1.rs"
)

HITS=$(find otspot-core/src otspot-io/src otspot-model/src otspot-dev/src \
  -name '*.rs' -type f -print0 | \
  xargs -0 awk -v max="$MAX_BLOCK" '
    /^[[:space:]]*\/\//{
      if (block_start == 0) block_start = FNR
      block_count++
      next
    }
    {
      if (block_count >= max) {
        printf "%s:%d-%d: comment block %d lines (max %d)\n",
          FILENAME, block_start, FNR-1, block_count, max
      }
      block_start = 0; block_count = 0
    }
    END {
      if (block_count >= max) {
        printf "%s:%d-%d: comment block %d lines (max %d)\n",
          FILENAME, block_start, FNR, block_count, max
      }
    }
  ')

FILTERED=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  skip=0
  for allow in "${ALLOWLIST[@]}"; do
    [[ "$line" == *"$allow"* ]] && skip=1 && break
  done
  [ "$skip" -eq 0 ] && FILTERED+="$line"$'\n'
done <<< "$HITS"

if [ -n "${FILTERED//[[:space:]]/}" ]; then
  echo "::error::Comment block size violation (max $MAX_BLOCK lines, CLAUDE.md L45):"
  echo "$FILTERED"
  exit 1
fi
echo "comment block size check: OK"
