#!/bin/bash
# bench_mini.sh — 開発用 mini bench (代表 14 問、デフォルト timeout=60s)
#
# TDD サイクルで頻繁に走らせる用。bench_parallel.sh を内部で呼び出し、
# 一時ディレクトリに代表問題の symlink 集合を作って渡す。
#
# 使い方:
#   bash scripts/bench_mini.sh
#   TIMEOUT=30 EPS=1e-6 bash scripts/bench_mini.sh
#
# 環境変数:
#   TIMEOUT  問題あたり timeout 秒 (default: 60)
#   EPS      精度 (default: 1e-4)
#   JOBS     並列数 (default: 8)
#   OUTPUT   出力ファイル (default: logs/bench_mini/mini.txt)
#
# 選定基準:
# - 様々な presolve transform (Doubleton/SingletonRow/RedundantConstraint/...) を踏む
# - 小〜中規模 (各問題 < 30s)
# - 過去 DFEAS_FAIL 経験あり (regression 検出)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROBLEMS=(
  afiro
  blend
  sc50a
  sc50b
  sc105
  adlittle
  kb2
  agg
  brandy
  share1b
  scorpion
  scfxm1
  sctap1
  recipe
)

TIMEOUT="${TIMEOUT:-60}"
EPS="${EPS:-1e-4}"
JOBS="${JOBS:-8}"
OUTPUT="${OUTPUT:-$ROOT/logs/bench_mini/mini_eps${EPS}_t${TIMEOUT}.txt}"

TMPDIR="$(mktemp -d /tmp/bench_mini_XXXXXX)"
trap "rm -rf $TMPDIR" EXIT

missing=0
for p in "${PROBLEMS[@]}"; do
  src="$ROOT/data/lp_problems/$p.QPS"
  if [ -f "$src" ]; then
    ln -s "$src" "$TMPDIR/$p.QPS"
  else
    echo "[WARN] $src not found, skipped" >&2
    missing=$((missing + 1))
  fi
done

if [ "$missing" -gt 0 ]; then
  echo "[WARN] $missing problems missing" >&2
fi

mkdir -p "$(dirname "$OUTPUT")"
echo "[bench_mini] problems=${#PROBLEMS[@]} (missing=$missing) timeout=${TIMEOUT}s eps=$EPS jobs=$JOBS"
echo "[bench_mini] output: $OUTPUT"

SOLVER_DIR="$ROOT" bash "$SCRIPT_DIR/bench_parallel.sh" \
  --data-dir "$TMPDIR" \
  --solver concurrent \
  --timeout "$TIMEOUT" \
  --eps "$EPS" \
  --jobs "$JOBS" \
  --output "$OUTPUT"
