#!/bin/bash
# Production memo-comment grep gate (CLAUDE.md L46 「コードにメモ書きコメント禁止」).
# Run from repo root. Exits 1 on hit, 0 on clean.
# Single source of truth for `.github/workflows/audit.yml` と
# `scripts/pre-merge-audit.sh` (二重実装防止).
set -eo pipefail

HITS=$(grep -rnE '(TODO|FIXME|XXX|HACK|todo!\()' \
  otspot-core/src otspot-io/src otspot-model/src otspot-dev/src \
  --include='*.rs' 2>/dev/null \
  | grep -vE 'tests/.*\.rs:' \
  || true)

if [ -n "$HITS" ]; then
  echo "::error::Production code must not contain TODO/FIXME/XXX/HACK/todo!() (CLAUDE.md L46)"
  echo "$HITS"
  exit 1
fi
echo "memo grep: OK"
