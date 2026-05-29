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
#   bash scripts/download_all_bench_data.sh                     # 全部
#   bash scripts/download_all_bench_data.sh --lp                # LP のみ
#   bash scripts/download_all_bench_data.sh --qp                # QP のみ
#   bash scripts/download_all_bench_data.sh --miplib-ext        # MIPLIB 2017 benchmark のみ
#   bash scripts/download_all_bench_data.sh --check             # 取得済み/未取得確認のみ
#   bash scripts/download_all_bench_data.sh --ci-subset         # CI subset (11 dataset, ~1.2 GiB) のみ取得
#   bash scripts/download_all_bench_data.sh --ci-subset --check # CI subset の確認のみ

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="all"
case "${1:-}" in
  --lp) MODE="lp" ;;
  --qp) MODE="qp" ;;
  --miplib-ext) MODE="miplib-ext" ;;
  --check) MODE="check" ;;
  --ci-subset)
    case "${2:-}" in
      --check) MODE="ci-subset-check" ;;
      "") MODE="ci-subset" ;;
      *) echo "usage: $0 --ci-subset [--check]"; exit 1 ;;
    esac
    ;;
  "") ;;
  *) echo "usage: $0 [--lp | --qp | --miplib-ext | --check | --ci-subset [--check]]"; exit 1 ;;
esac

# Build usage flags string for error messages (MODE "all" → no flags; compound modes → split flags)
case "$MODE" in
  all)              USAGE_FLAGS="" ;;
  ci-subset-check)  USAGE_FLAGS="--ci-subset --check" ;;
  *)                USAGE_FLAGS="--$MODE" ;;
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
    echo "             bash scripts/download_all_bench_data.sh${USAGE_FLAGS:+ $USAGE_FLAGS}" >&2
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

# Verify a sample file can be parsed (header sanity, not full parse).
# qplib: line 1 = problem name, line 2 = 3-char type code.
# qps/QPS: line 1 must start with "NAME" (Maros-Meszaros format).
smoke_qplib() {
  local dir=$1
  local sample
  sample=$(ls "$dir"/*.qplib 2>/dev/null | head -1)
  if [[ -z "$sample" ]]; then return 0; fi
  local name type
  name=$(sed -n '1p' "$sample" | tr -d '[:space:]')
  type=$(sed -n '2p' "$sample" | tr -d '[:space:]')
  if [[ -z "$name" || ${#type} -ne 3 ]]; then
    echo "  [smoke-fail] $sample: unexpected header (name='$name' type='$type')"
    return 1
  fi
  echo "  [smoke-ok]  $sample (name=$name type=$type)"
}

smoke_qps() {
  local dir=$1
  local sample
  sample=$(ls "$dir"/*.QPS "$dir"/*.qps 2>/dev/null | head -1)
  if [[ -z "$sample" ]]; then return 0; fi
  local first
  first=$(head -1 "$sample" | tr -d '[:space:]')
  if [[ "$first" != NAME* && "$first" != "NAME"* ]]; then
    echo "  [smoke-fail] $sample: first line does not start with NAME (got: $first)"
    return 1
  fi
  echo "  [smoke-ok]  $sample (header ok)"
}

smoke_mps() {
  local dir=$1
  local sample
  sample=$(ls "$dir"/*.mps 2>/dev/null | head -1)
  if [[ -z "$sample" ]]; then return 0; fi
  local first
  first=$(head -1 "$sample" | tr -d '[:space:]')
  if [[ "$first" != NAME* && "$first" != "NAME"* ]]; then
    echo "  [smoke-fail] $sample: first line does not start with NAME (got: $first)"
    return 1
  fi
  echo "  [smoke-ok]  $sample (header ok)"
}

if [[ "$MODE" == "check" ]]; then
  fail=0

  echo "=== LP ==="
  check_dir data/lp_problems 109        || fail=1
  check_dir data/lp_problems_infeas 29  || fail=1
  check_dir data/lp_problems_extra 4    || fail=1
  check_dir data/lp_problems_hard 53    || fail=1
  check_dir data/lp_problems_canary 27  || fail=1
  check_dir data/lp_problems_unbounded 12 || fail=1

  echo "=== QP ==="
  check_dir data/maros_meszaros 138     || fail=1
  check_dir data/mpc_qp 64             || fail=1
  check_dir data/osqp_bench 62         || fail=1
  check_dir data/osqp_bench_extra 238  || fail=1
  check_dir data/osqp_bench_illscaled 126 || fail=1
  check_dir data/osqp_bench_xl 2       || fail=1
  check_dir data/qp_dense_a 8          || fail=1
  check_dir data/qp_infeasible 12      || fail=1
  check_dir data/qp_unbounded 9        || fail=1
  check_dir data/qplib 41              || fail=1
  check_dir data/qplib_nonconvex 45    || fail=1
  check_dir data/qplib_nonconvex_official 4 || fail=1
  check_dir data/qplib_unsupported 6   || fail=1

  echo ""
  echo "=== parse smoke ==="
  smoke_qplib data/qplib              || fail=1
  smoke_qplib data/qplib_nonconvex_official || fail=1
  smoke_qps   data/maros_meszaros     || fail=1
  smoke_mps   data/miplib_small       || fail=1
  smoke_mps   data/miplib_2017        || fail=1

  echo ""
  echo "=== baseline CSV vs data dir ==="
  if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/check_data_coverage.py" --repo-root "$REPO_ROOT" || fail=1
  else
    echo "  [skip] python3 not found; skipping CSV coverage check"
  fi

  if [[ "$fail" -eq 0 ]]; then
    echo ""
    echo "[check] all ok"
  else
    echo ""
    echo "[check] FAILED — see above" >&2
    exit 1
  fi
  exit 0
fi

##############################################################################
# CI subset check mode
##############################################################################

if [[ "$MODE" == "ci-subset-check" ]]; then
  fail=0

  echo "=== CI subset LP ==="
  check_dir data/lp_problems         109 || fail=1
  check_dir data/lp_problems_infeas   29 || fail=1
  check_dir data/lp_problems_extra     4 || fail=1

  echo "=== CI subset QP ==="
  check_dir data/maros_meszaros      138 || fail=1
  check_dir data/osqp_bench           30 || fail=1  # ci-subset: synthetic only (SuiteSparse 抜き、--all で 62)
  check_dir data/qplib                41 || fail=1
  check_dir data/mpc_qp               64 || fail=1
  check_dir data/miplib_small         20 || fail=1
  check_dir data/qplib_nonconvex      45 || fail=1
  check_dir data/qplib_nonconvex_official 4 || fail=1
  check_dir data/qp_dense_a            8 || fail=1

  echo ""
  echo "=== parse smoke ==="
  smoke_qplib data/qplib                    || fail=1
  smoke_qplib data/qplib_nonconvex_official || fail=1
  smoke_qps   data/maros_meszaros           || fail=1

  if [[ "$fail" -eq 0 ]]; then
    echo ""
    echo "[ci-subset check] all ok"
  else
    echo ""
    echo "[ci-subset check] FAILED — see above" >&2
    exit 1
  fi
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
  run_or_skip data/maros_meszaros            138 "bash scripts/maros_meszaros_download.sh"
  run_or_skip data/qplib                     41  "bash scripts/qplib_download.sh"
  run_or_skip data/qplib_nonconvex_official  4   "bash scripts/qplib_nonconvex_download.sh"
  run_or_skip data/qplib_unsupported         6   "bash scripts/qplib_unsupported_download.sh"
fi

##############################################################################
# MIP suites (--miplib-ext)
##############################################################################
if [[ "$MODE" == "miplib-ext" ]]; then
  echo ""
  echo "########## MIP data ##########"

  # MIPLIB 2017 benchmark: 240+ instances, ~317 MB (explicit opt-in, not in --all)
  # License: ZIB (academic/research), instances in MPS format
  # Source: https://miplib.zib.de/downloads/benchmark.zip
  run_or_skip data/miplib_2017 1 "bash scripts/miplib_2017_download.sh"
fi

##############################################################################
# CI subset suites
##############################################################################
if [[ "$MODE" == "ci-subset" ]]; then
  echo ""
  echo "########## CI subset LP data ##########"

  run_or_skip data/lp_problems           109 "bash scripts/netlib_lp_download.sh"
  run_or_skip data/lp_problems_infeas    29  "bash scripts/netlib_lp_infeas_download.sh"
  run_or_skip data/lp_problems_extra     4   "bash scripts/lp_extra_download.sh"

  echo ""
  echo "########## CI subset QP data ##########"

  check_python_qp_deps

  run_or_skip data/osqp_bench            30  "bash scripts/setup_extra_benches.sh --no-suitesparse && python3 scripts/gen_osqp_bench.py"
  run_or_skip data/mpc_qp               64   "python3 scripts/gen_mpc_qp.py"
  run_or_skip data/qp_dense_a            8   "python3 scripts/gen_dense_a_qp.py"
  run_or_skip data/qplib_nonconvex      45   "python3 scripts/gen_nonconvex_qp.py"
  run_or_skip data/maros_meszaros       138  "bash scripts/maros_meszaros_download.sh"
  run_or_skip data/qplib                41   "bash scripts/qplib_download.sh"
  run_or_skip data/qplib_nonconvex_official 4 "bash scripts/qplib_nonconvex_download.sh"
  run_or_skip data/miplib_small         20   "bash scripts/miplib_small_download.sh"
fi

echo ""
echo "[done] check status:"
if [[ "$MODE" == "ci-subset" ]]; then
  bash "$0" --ci-subset --check
else
  bash "$0" --check
fi
