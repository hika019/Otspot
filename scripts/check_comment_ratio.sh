#!/bin/bash
# Warn if any production .rs file (>=100 lines) has comment ratio >= THRESHOLD%.
set -euo pipefail

THRESHOLD=35
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
  comment=$(grep -cE '^[[:space:]]*(//|///|//!)' "$f" || echo 0)
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
  echo "::warning::Comment ratio >= ${THRESHOLD}% (review attention, CLAUDE.md L45):"
  echo "$HITS"
fi
echo "comment ratio check: OK"
