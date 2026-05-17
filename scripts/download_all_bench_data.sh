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

check_python_qp_deps() {
  if ! python3 -c "import numpy, scipy" 2>/dev/null; then
    echo "[error] QP bench data 生成に必要な Python pkg (numpy / scipy) が host にない。" >&2
    echo "         Docker で実行 (推奨):" >&2
    echo "           docker run --rm -v \"\$PWD\":/workspace -w /workspace solver-dev \\" >&2
    echo "             bash scripts/download_all_bench_data.sh${MODE:+ --$MODE}" >&2
    echo "         または host へ install:" >&2
    echo "           pip install numpy scipy cvxpy clarabel" >&2
    echo "         (cvxpy / clarabel は osqp_bench 系 generator のみで必要)" >&2
    exit 1
  fi
}

run_or_skip() {
  local dir=$1
  local expect=$2
  local cmd=$3
  local n=0
  if [[ -d "$dir" ]]; then
    n=$(ls "$dir" 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [[ "$n" -ge "$expect" ]]; then
    echo "[skip] $dir already populated ($n/$expect)"
    return 0
  fi
  if [[ "$n" -gt 0 ]]; then
    echo "[partial] $dir ($n/$expect) — re-running to fill missing files"
  fi
  echo "[run] $cmd"
  eval "$cmd"
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
  check_dir data/maros_meszaros 138
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
  check_dir data/qplib_unsupported 6
  exit 0
fi

##############################################################################
# LP suites
##############################################################################
if [[ "$MODE" == "all" || "$MODE" == "lp" ]]; then
  echo ""
  echo "########## LP data ##########"

  run_or_skip data/lp_problems           109 "bash scripts/netlib_lp_download.sh"
  run_or_skip data/lp_problems_infeas    29  "bash scripts/netlib_lp_infeas_download.sh"
  run_or_skip data/lp_problems_extra     4   "bash scripts/lp_extra_download.sh"
  run_or_skip data/lp_problems_hard      53  "bash scripts/lp_hard_download.sh"
  run_or_skip data/lp_problems_canary    27  "bash scripts/lp_canary_setup.sh data/lp_problems_canary"
  run_or_skip data/lp_problems_unbounded 12  "python3 scripts/gen_unbounded_lp.py"
fi

##############################################################################
# QP suites
##############################################################################
if [[ "$MODE" == "all" || "$MODE" == "qp" ]]; then
  echo ""
  echo "########## QP data ##########"

  check_python_qp_deps

  # external repo 経由 (osqp_bench, mpc_qp)
  run_or_skip data/osqp_bench           62  "bash scripts/setup_extra_benches.sh && python3 scripts/gen_osqp_bench.py"
  run_or_skip data/mpc_qp               64  "python3 scripts/gen_mpc_qp.py"

  # gen 系
  run_or_skip data/osqp_bench_extra     238 "python3 scripts/gen_osqp_bench_extra.py"
  run_or_skip data/osqp_bench_illscaled 126 "python3 scripts/gen_osqp_bench_illscaled.py"
  run_or_skip data/osqp_bench_xl        2   "python3 scripts/gen_osqp_bench_xl.py"
  run_or_skip data/qp_dense_a           8   "python3 scripts/gen_dense_a_qp.py"
  run_or_skip data/qp_infeasible        12  "python3 scripts/gen_infeasible_qp.py"
  run_or_skip data/qp_unbounded         9   "python3 scripts/gen_unbounded_qp.py"
  run_or_skip data/qplib_nonconvex      45  "python3 scripts/gen_nonconvex_qp.py"

  # Maros-Meszaros / QPLIB: 専用 download script
  run_or_skip data/maros_meszaros     138 "bash scripts/maros_meszaros_download.sh"
  run_or_skip data/qplib              41  "bash scripts/qplib_download.sh"
  run_or_skip data/qplib_unsupported  6   "bash scripts/qplib_unsupported_download.sh"
fi

echo ""
echo "[done] check status:"
bash "$0" --check
