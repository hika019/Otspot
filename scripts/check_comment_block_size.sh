#!/bin/bash
# Fail if any production .rs file has a consecutive comment block >= MAX_BLOCK lines.
# CLAUDE.md L45 「コメントが過多. OSSとしてふさわしい程度にしろ」.
#
# MAX_BLOCK=18: reviewer-203 P1-1 で user 指摘 core.rs:28-45 (18 行 docstring) を
# catch するため、`block_count >= max` 経路で MAX=18 に設定 (元 40、20 では catch
# 不能で「上っ面修正」抵触)。allowlist は file-leading の長大 module/item docstring
# のみ (mid-file の 18 行 block は trim 対象として fail させる)。
set -euo pipefail

MAX_BLOCK=18
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

declare -a ALLOWLIST=(
  "otspot-core/src/simplex/pricing.rs"
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
        printf "%s:%d-%d: comment block %d lines (>= MAX_BLOCK=%d)\n",
          FILENAME, block_start, FNR-1, block_count, max
      }
      block_start = 0; block_count = 0
    }
    END {
      if (block_count >= max) {
        printf "%s:%d-%d: comment block %d lines (>= MAX_BLOCK=%d)\n",
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
  echo "::error::Comment block size violation (>= ${MAX_BLOCK} lines, CLAUDE.md L45):"
  echo "$FILTERED"
  exit 1
fi
echo "comment block size check: OK"
