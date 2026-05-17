#!/bin/bash
# QPLIB unsupported subset (11 instances) downloader.
#
# 現 solver が未対応の structure を持つ QPLIB instance 群。
# Source: https://qplib.zib.de/qplib/QPLIB_XXXX.qplib
# ID list は data/qplib_unsupported/ の既存 file 名から抽出 (fact-based)。
#
# Usage:
#   bash scripts/qplib_unsupported_download.sh [OUT_DIR]
# OUT_DIR 省略時は data/qplib_unsupported

set -euo pipefail

OUT_DIR="${1:-data/qplib_unsupported}"
mkdir -p "$OUT_DIR"

QPLIB_IDS=(
  "1055" "1143" "1157" "1353" "1423" "1493"
  "3913" "3980"
  "10050" "10056" "10069"
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
echo "[summary] qplib_unsupported: ok=$ok skip=$skip fail=$fail total=$total -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-ids] ${fail_ids[*]}"
  exit 1
fi
