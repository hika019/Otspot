#!/bin/bash
# Fail if any production .rs file (>=100 lines) has comment ratio >= THRESHOLD%.
# CLAUDE.md L45 「コメントが過多. OSSとしてふさわしい程度にしろ」.
#
# THRESHOLD=27: hard-fail (exit 1) で密集コメント file を block。
# user 指摘 core.rs (元 27.9% → trim 後 25.8%) は block_size gate (MAX=18)
# が L25-42 の 18 行 LEX_PERTURB_REL block を primary catch。本 ratio gate は
# 密集小コメント file の別 vector 検出が主目的 (lib.rs 30.5%、bound_flip.rs 30.4% 等)。
# warning → hard-fail 昇格は CLAUDE.md L45 強調 directive。
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
