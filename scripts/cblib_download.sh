#!/bin/bash
# CBLIB SOCP/MISOCP small-to-medium instances downloader.
#
# 小～中規模の SOCP (連続) と MISOCP (混合整数) インスタンスを取得。
# Source: https://cblib.zib.de/download/all/
#
# Usage:
#   bash scripts/cblib_download.sh [OUT_DIR]
# OUT_DIR 省略時は data/cblib_socp

set -euo pipefail

OUT_DIR="${1:-data/cblib_socp}"
mkdir -p "$OUT_DIR"

# SOCP instances (continuous, 11 problems)
# Source: chained singular (academic), limit analysis (nql/qssp), portfolio optimization
# サイズ実測 (cblib index) で gz < 1MB を確認済。展開後 50MB 以下。
SOCP_INSTANCES=(
  "chainsing-1000-1.cbf.gz"
  "nql30.cbf.gz"
  "nql60.cbf.gz"
  "qssp30.cbf.gz"
  "20_0_1_w.cbf.gz"
  "20_0_2_w.cbf.gz"
  "50_0_1_w.cbf.gz"
  "50_0_2_w.cbf.gz"
  "100_0_1_w.cbf.gz"
  "150_0_1_w.cbf.gz"
  "200_0_1_w.cbf.gz"
)

# MISOCP instances (mixed-integer, 11 problems)
# Source: topology optimization (truss design), stochastic service system, process network synthesis
MISOCP_INSTANCES=(
  "2x3_3bars.cbf.gz"
  "2x4_3bars.cbf.gz"
  "2x5_3bars.cbf.gz"
  "2D-TopOpt-Cantilever_60x40_50.cbf.gz"
  "sssd-strong-15-4.cbf.gz"
  "sssd-strong-20-4.cbf.gz"
  "syn10m.cbf.gz"
  "syn15m.cbf.gz"
  "syn20m.cbf.gz"
  "classical_20_0.cbf.gz"
  "classical_30_0.cbf.gz"
)

ALL_INSTANCES=("${SOCP_INSTANCES[@]}" "${MISOCP_INSTANCES[@]}")

total=${#ALL_INSTANCES[@]}
ok=0
skip=0
fail=0
fail_ids=()

BASE_URL="http://cblib.zib.de/download/all"

for instance in "${ALL_INSTANCES[@]}"; do
  # Extract base name without .gz extension
  base_name="${instance%.gz}"
  out="$OUT_DIR/$base_name"

  if [[ -f "$out" ]]; then
    skip=$((skip + 1))
    continue
  fi

  tmp_gz=$(mktemp --suffix=.gz)
  if curl -fsSL "$BASE_URL/$instance" -o "$tmp_gz"; then
    # Extract .gz to target location
    if gunzip -c "$tmp_gz" > "$out" 2>/dev/null; then
      ok=$((ok + 1))
      echo "[ok]   $instance"
      rm -f "$tmp_gz"
    else
      fail=$((fail + 1))
      fail_ids+=("$instance")
      echo "[fail] $instance (extraction failed)"
      rm -f "$tmp_gz" "$out"
    fi
  else
    rm -f "$tmp_gz"
    fail=$((fail + 1))
    fail_ids+=("$instance")
    echo "[fail] $instance (download failed)"
  fi
done

echo ""
echo "[summary] cblib_socp: ok=$ok skip=$skip fail=$fail total=$total -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-ids] ${fail_ids[*]}"
  exit 1
fi
