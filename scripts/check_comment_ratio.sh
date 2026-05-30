#!/bin/bash
# Fail if any production .rs file (>=100 lines) has comment ratio >= THRESHOLD%.
# CLAUDE.md L45 「コメントが過多. OSSとしてふさわしい程度にしろ」.
#
# THRESHOLD=27: reviewer-203 P1-1 で user 指摘 core.rs (27.9% ratio) を catch
# するため、`r >= threshold` 経路で 27 に設定 (元 35、30 では 27.9% を catch 不能で
# 「上っ面修正」抵触)。warning → hard-fail (exit 1) 昇格は user L45 強調 directive。
set -euo pipefail

THRESHOLD=27
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

declare -a ALLOWLIST=(
  "otspot-core/src/tolerances.rs"
  "otspot-core/src/simplex/pricing.rs"
)

HITS=""
while IFS= read -r f; do
  total=$(wc -l < "$f" | tr -d ' ')
  [ "$total" -lt 100 ] && continue
  comment=$(grep -cE '^[[:space:]]*//' "$f" || echo 0)
  ratio=$(awk -v c="$comment" -v t="$total" 'BEGIN { printf "%.1f", c*100/t }')
  if awk -v r="$ratio" -v th="$THRESHOLD" 'BEGIN { exit !(r >= th) }'; then
    skip=0
    for allow in "${ALLOWLIST[@]}"; do
      [[ "$f" == *"$allow"* ]] && skip=1 && break
    done
    [ "$skip" -eq 0 ] && HITS+="$ratio% $f"$'\n'
  fi
done < <(find otspot-core/src otspot-io/src otspot-model/src otspot-dev/src \
  -name '*.rs' -type f)

if [ -n "${HITS//[[:space:]]/}" ]; then
  echo "::error::Comment ratio >= ${THRESHOLD}% (CLAUDE.md L45):"
  echo "$HITS"
  exit 1
fi
echo "comment ratio check: OK"
