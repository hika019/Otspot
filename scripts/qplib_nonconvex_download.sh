#!/bin/bash
# QPLIB official non-convex (pure continuous QP) downloader.
#
# Source: https://qplib.zib.de/  (CC-BY 4.0, free redistribution with attribution)
# Selection: pure continuous variables + linear/box constraints + CONVEX=false +
#            indefinite objective. QCQP and binary-QP instances are excluded
#            because the current QP solver targets continuous QP with linear/box
#            constraints only.
#
# IDs are extracted from per-instance metadata pages (probtype field). Reference
# optima are recorded separately in data/baseline_objectives/qplib_nonconvex_official.csv.
#
# Usage:
#   bash scripts/qplib_nonconvex_download.sh [OUT_DIR]
# OUT_DIR omitted -> data/qplib_nonconvex_official

set -euo pipefail

OUT_DIR="${1:-data/qplib_nonconvex_official}"
mkdir -p "$OUT_DIR"

# QCL (Quadratic objective, Continuous vars, Linear constraints), CONVEX=false.
# Verified via https://qplib.zib.de/QPLIB_<id>.html (2026-05-18).
QPLIB_IDS=(
  "0018"
  "0343"
  "2712"
  "2761"
)

total=${#QPLIB_IDS[@]}
ok=0
skip=0
fail=0
fail_ids=()

for id in "${QPLIB_IDS[@]}"; do
  out="$OUT_DIR/QPLIB_${id}.qplib"
  if [[ -s "$out" ]]; then
    skip=$((skip + 1))
    continue
  fi
  tmp=$(mktemp)
  if curl -fsSL "https://qplib.zib.de/qplib/QPLIB_${id}.qplib" -o "$tmp"; then
    mv "$tmp" "$out"
    ok=$((ok + 1))
    echo "[ok]   QPLIB_${id}"
  else
    rm -f "$tmp"
    fail=$((fail + 1))
    fail_ids+=("$id")
    echo "[fail] QPLIB_${id}"
  fi
done

echo ""
echo "[summary] qplib_nonconvex_official: ok=$ok skip=$skip fail=$fail total=$total -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-ids] ${fail_ids[*]}"
  exit 1
fi
