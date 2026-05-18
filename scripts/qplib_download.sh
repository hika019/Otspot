#!/bin/bash
# QPLIB instance subset (41 instances、convex + non-convex 混在) downloader.
#
# 注: 旧 doc は「convex subset」と表記していたが、QPLIB 公式メタ確認で
# 0018/0343/2546/2712/2761/2981/3080/3297 等が CONVEX=false (binary QP / QCQP
# 含む)。pure-continuous nonconvex は data/qplib_nonconvex_official/ (4 件、
# CC-BY 4.0) に分離、本 script は既存 data/qplib/ 41 件 (= IDs 抽出元) を維持。
#
# Source: https://qplib.zib.de/qplib/QPLIB_XXXX.qplib
# ID list は data/qplib/ の既存 file 名から抽出 (fact-based, 推測なし)。
#
# Usage:
#   bash scripts/qplib_download.sh [OUT_DIR]
# OUT_DIR 省略時は data/qplib

set -euo pipefail

OUT_DIR="${1:-data/qplib}"
mkdir -p "$OUT_DIR"

QPLIB_IDS=(
  "0018" "0343"
  "2546" "2712" "2761" "2981"
  "3080" "3297" "3913" "3980"
  "8495" "8500" "8505" "8515" "8547" "8559" "8567" "8585" "8595"
  "8602" "8605" "8616" "8683" "8685"
  "8777" "8785" "8790" "8792"
  "8803" "8810" "8845" "8906" "8938" "8991"
  "9002" "9008"
  "10034" "10038" "10050" "10056" "10069"
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
echo "[summary] qplib: ok=$ok skip=$skip fail=$fail total=$total -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-ids] ${fail_ids[*]}"
  exit 1
fi
