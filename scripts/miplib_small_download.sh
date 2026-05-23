#!/bin/bash
# MIPLIB 2017 small/easy MILP subset downloader.
#
# Source: https://miplib.zib.de/WebData/instances/<name>.mps.gz
# 小規模〜中規模の MILP を取得 -> gunzip -> .mps として配置。
# otspot の B&B (cut/presolve なしの素朴 best-bound) で 100s 級 timeout 内に
# 探索が観測できる規模を中心に選定 (巨大インスタンスは除外)。
#
# Usage:
#   bash scripts/miplib_small_download.sh [OUT_DIR]
# OUT_DIR 省略時は data/miplib_small
#
# 各ファイルは gzip マジック (1f8b) を検証してから展開する。MIPLIB サーバは
# 未知名に対し HTML を 200 で返すため、HTTP コードだけでは判定できない。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${1:-data/miplib_small}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

BASE_URL="https://miplib.zib.de/WebData/instances"

# 選定インスタンス (size は HiGHS load line で実測; rows x cols, int 数)。
# tiny: 素朴 B&B で短時間に解ける。hard-tiny: 小さいが探索が爆発 (market share 等)。
# moderate: 100s 内で timeout/部分探索になりうる (otspot の限界を観測する用)。
NAMES=(
  # tiny (≤ 50 cols)
  flugpl          # 18x18,   11 int  general integer
  gen-ip002       # 24x41,   41 int  general integer
  gen-ip016       # 24x28,   28 int
  gen-ip021       # 28x35,   35 int
  gen-ip054       # 27x30,   30 int
  gr4x6           # 34x48,   24 bin  assignment
  # hard-tiny (small but combinatorially hard)
  markshare_4_0   # 4x34,    30 bin  market share
  markshare_5_0   # 5x45,    40 bin
  markshare1      # 6x62,    50 bin
  # small/moderate
  pk1             # 45x86,   55 int
  neos5           # 63x63,   53 int
  gt2             # 29x188,  188 int
  mas74           # 13x151,  150 bin  knapsack-like
  mas76           # 12x151,  150 bin
  enlight_hard    # 100x200, 200 int
  noswot          # 182x128, 100 int
  p0201           # 133x201, 201 bin  classic 0/1
  timtab1         # 171x397, 171 int
  dcmulti         # 290x548, 75 int
  khb05250        # 101x1350, 24 int
)

ok=0; miss=0
for name in "${NAMES[@]}"; do
  out_mps="$OUT_DIR/$name.mps"
  if [[ -f "$out_mps" ]]; then
    echo "[miplib] cached: $name"
    ok=$((ok + 1))
    continue
  fi
  tmp="$OUT_DIR/$name.dl"
  curl -sL --max-time 60 -o "$tmp" "$BASE_URL/$name.mps.gz" 2>/dev/null || true
  magic=$(xxd -p -l2 "$tmp" 2>/dev/null || echo "")
  if [[ "$magic" == "1f8b" ]]; then
    mv "$tmp" "$out_mps.gz"
    gunzip -f "$out_mps.gz"
    echo "[miplib] downloaded: $name"
    ok=$((ok + 1))
  else
    rm -f "$tmp"
    echo "[miplib] MISSING (not a gzip): $name" >&2
    miss=$((miss + 1))
  fi
done

echo "[miplib] done: $ok ok, $miss missing -> $OUT_DIR"
