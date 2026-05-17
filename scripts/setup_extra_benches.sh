#!/bin/bash
# OSQP / MPC QP 追加ベンチセットを一括セットアップする。
#
# 1. tmp/external/ に upstream リポジトリを clone (既にあれば pull)
# 2. cvxpy 必須 (OSQP 生成器が import する)
# 3. data/osqp_bench/ と data/mpc_qp/ に .qps を書き出す
#
# baseline_objectives/{osqp_bench,mpc_qp}.csv の populate は別途
# scripts/bench_parallel.sh を回して scripts/baseline_from_bench_log.py で抽出。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

EXT="$REPO_ROOT/tmp/external"
mkdir -p "$EXT"

# --- OSQP benchmarks repo ---
if [[ -d "$EXT/osqp_benchmarks/.git" ]]; then
  echo "[setup_extra_benches] osqp_benchmarks: 既にあるので pull"
  git -C "$EXT/osqp_benchmarks" pull --ff-only --quiet || echo "  (pull 失敗。手動更新推奨)"
else
  echo "[setup_extra_benches] osqp_benchmarks を clone"
  git clone --depth 1 https://github.com/osqp/osqp_benchmarks.git "$EXT/osqp_benchmarks"
fi

# --- MPC QP benchmark repo ---
if [[ -d "$EXT/mpc_qpbenchmark/.git" ]]; then
  echo "[setup_extra_benches] mpc_qpbenchmark: 既にあるので pull"
  git -C "$EXT/mpc_qpbenchmark" pull --ff-only --quiet || echo "  (pull 失敗。手動更新推奨)"
else
  echo "[setup_extra_benches] mpc_qpbenchmark を clone"
  git clone --depth 1 https://github.com/qpsolvers/mpc_qpbenchmark.git "$EXT/mpc_qpbenchmark"
fi

# --- Python deps チェック ---
if ! python3 -c "import cvxpy" >/dev/null 2>&1; then
  echo "[setup_extra_benches] ERROR: cvxpy が無い。 pip install cvxpy で導入してから再実行"
  exit 1
fi

# --- Python deps チェック (SuiteSparse 用) ---
if ! python3 -c "import tables, ssgetpy" >/dev/null 2>&1; then
  echo "[setup_extra_benches] WARN: tables / ssgetpy が無い (SuiteSparse 系をスキップする場合 OK)"
  echo "  追加: pip install tables ssgetpy"
fi

# --- 生成 ---
echo "[setup_extra_benches] OSQP synthetic .qps 生成 -> data/osqp_bench/"
python3 scripts/gen_osqp_bench.py

echo "[setup_extra_benches] OSQP SuiteSparse .qps 生成 -> data/osqp_bench/"
if python3 -c "import tables, ssgetpy" >/dev/null 2>&1; then
  python3 scripts/gen_osqp_suitesparse.py
else
  echo "  (skip: tables / ssgetpy 未導入)"
fi

echo "[setup_extra_benches] MPC .qps 生成 -> data/mpc_qp/"
python3 scripts/gen_mpc_qp.py

OSQP_N=$(ls data/osqp_bench/*.qps 2>/dev/null | wc -l | tr -d ' ')
MPC_N=$(ls data/mpc_qp/*.qps 2>/dev/null | wc -l | tr -d ' ')
echo "[setup_extra_benches] 完了: osqp_bench=$OSQP_N qps, mpc_qp=$MPC_N qps"
echo ""
echo "次の手順 (ベースライン収集):"
echo "  SOLVER_DIR=\$PWD bash scripts/bench_parallel.sh \\"
echo "    --data-dir data/osqp_bench --timeout 300 \\"
echo "    --eps 1e-6 --jobs 4 --output bench_results/baseline_osqp.txt"
echo "  python3 scripts/baseline_from_bench_log.py \\"
echo "    bench_results/baseline_osqp.txt data/baseline_objectives/osqp_bench.csv \\"
echo "    --source bench_\$(date +%Y-%m-%d)_osqp --merge"
echo "(同様に mpc_qp も実行)"
