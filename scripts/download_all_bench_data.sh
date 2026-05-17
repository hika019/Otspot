#!/bin/bash
# Bench data 一括取得スクリプト
#
# 用途: data/ 配下の LP / QP bench data dir を全て取得・生成する。
# 既に存在する dir は skip。
#
# 前提:
#   - curl, python3 (3.8+), git, cargo (Rust toolchain)
#   - python pkg: scipy, numpy, cvxpy (osqp/mpc 生成器が import)
#   - emps コンパイル済み (Netlib 用、未存在なら自動 build)
#
# 使い方:
#   bash scripts/download_all_bench_data.sh           # 全部
#   bash scripts/download_all_bench_data.sh --lp      # LP のみ
#   bash scripts/download_all_bench_data.sh --qp      # QP のみ
#   bash scripts/download_all_bench_data.sh --check   # 取得済み/未取得確認のみ

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="all"
case "${1:-}" in
  --lp) MODE="lp" ;;
  --qp) MODE="qp" ;;
  --check) MODE="check" ;;
  "") ;;
  *) echo "usage: $0 [--lp | --qp | --check]"; exit 1 ;;
esac

check_dir() {
  local dir=$1
  local expect=$2
  if [[ -d "$dir" ]]; then
    local n=$(ls "$dir" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$n" -ge "$expect" ]]; then
      echo "  [ok]      $dir ($n files, expected >= $expect)"
      return 0
    else
      echo "  [partial] $dir ($n files, expected >= $expect)"
      return 1
    fi
  else
    echo "  [missing] $dir"
    return 1
  fi
}

run_or_skip() {
  local dir=$1
  local cmd=$2
  if [[ -d "$dir" && $(ls "$dir" 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[skip] $dir already populated"
  else
    echo "[run] $cmd"
    eval "$cmd"
  fi
}

##############################################################################
# Check mode
##############################################################################
if [[ "$MODE" == "check" ]]; then
  echo "=== LP ==="
  check_dir data/lp_problems 109
  check_dir data/lp_problems_infeas 29
  check_dir data/lp_problems_extra 4
  check_dir data/lp_problems_hard 53
  check_dir data/lp_problems_canary 27
  check_dir data/lp_problems_unbounded 12

  echo "=== QP ==="
  check_dir data/maros_meszaros 139
  check_dir data/mpc_qp 64
  check_dir data/osqp_bench 62
  check_dir data/osqp_bench_extra 238
  check_dir data/osqp_bench_illscaled 126
  check_dir data/osqp_bench_xl 2
  check_dir data/qp_dense_a 8
  check_dir data/qp_infeasible 12
  check_dir data/qp_unbounded 9
  check_dir data/qplib 41
  check_dir data/qplib_nonconvex 45
  check_dir data/qplib_unsupported 11
  exit 0
fi

##############################################################################
# LP suites
##############################################################################
if [[ "$MODE" == "all" || "$MODE" == "lp" ]]; then
  echo ""
  echo "########## LP data ##########"

  run_or_skip data/lp_problems         "bash scripts/netlib_lp_download.sh"
  run_or_skip data/lp_problems_infeas  "bash scripts/netlib_lp_infeas_download.sh"
  run_or_skip data/lp_problems_extra   "bash scripts/lp_extra_download.sh"
  run_or_skip data/lp_problems_hard    "bash scripts/lp_hard_download.sh"
  run_or_skip data/lp_problems_canary  "bash scripts/lp_canary_setup.sh data/lp_problems_canary"
  run_or_skip data/lp_problems_unbounded "python3 scripts/gen_unbounded_lp.py"
fi

##############################################################################
# QP suites
##############################################################################
if [[ "$MODE" == "all" || "$MODE" == "qp" ]]; then
  echo ""
  echo "########## QP data ##########"

  # external repo 経由 (osqp_bench, mpc_qp)
  run_or_skip data/osqp_bench   "bash scripts/setup_extra_benches.sh && python3 scripts/gen_osqp_bench.py"
  run_or_skip data/mpc_qp       "python3 scripts/gen_mpc_qp.py"

  # gen 系
  run_or_skip data/osqp_bench_extra     "python3 scripts/gen_osqp_bench_extra.py"
  run_or_skip data/osqp_bench_illscaled "python3 scripts/gen_osqp_bench_illscaled.py"
  run_or_skip data/osqp_bench_xl        "python3 scripts/gen_osqp_bench_xl.py"
  run_or_skip data/qp_dense_a           "python3 scripts/gen_dense_a_qp.py"
  run_or_skip data/qp_infeasible        "python3 scripts/gen_infeasible_qp.py"
  run_or_skip data/qp_unbounded         "python3 scripts/gen_unbounded_qp.py"
  run_or_skip data/qplib_nonconvex      "python3 scripts/gen_nonconvex_qp.py"

  # Manual setup required (no download script yet)
  if [[ ! -d data/maros_meszaros || $(ls data/maros_meszaros 2>/dev/null | wc -l) -eq 0 ]]; then
    cat <<'EOF'

[manual] data/maros_meszaros/ (139 .QPS files) — download script 未整備
  source: https://github.com/YimingYAN/QP-Test-Problems (MAT_Files/)
  対応:
    1. .mat ファイルを cache: scripts/run_maros_all.py 内の download_mat() を参考
    2. .mat -> .QPS 変換が別途必要 (scipy.io + qp_to_qps.py の組合せ)
  TODO: 自動 download スクリプト化
EOF
  fi

  if [[ ! -d data/qplib || $(ls data/qplib 2>/dev/null | wc -l) -eq 0 ]]; then
    cat <<'EOF'

[manual] data/qplib/ (41 .qplib files) — download script 未整備
  source: https://qplib.zib.de/ (個別問題 instance)
  対応:
    各 QPLIB_XXXX.qplib を https://qplib.zib.de/data/QPLIB_XXXX.qplib から取得
  TODO: convex subset の自動 download スクリプト化
EOF
  fi

  if [[ ! -d data/qplib_unsupported || $(ls data/qplib_unsupported 2>/dev/null | wc -l) -eq 0 ]]; then
    cat <<'EOF'

[manual] data/qplib_unsupported/ (11 .qplib files) — download script 未整備
  source: 同 QPLIB.zib.de (現 solver で未対応の構造を持つ instance 群)
  TODO: 自動 download スクリプト化
EOF
  fi
fi

echo ""
echo "[done] check status:"
bash "$0" --check
