#!/bin/bash
# solver_bench.sh — ベンチマークバイナリのラッパー
# 他の bench suite と並行実行禁止 (CLAUDE.md L72 PC リソース contention 回避、各 suite 順次実行)
# --release ビルドを強制する。`--eps` は呼出側 (bench_parallel.sh) から指定する。
# qps_benchmark, bench_qplib, milp_solve の三種に対応。
#
# 使い方:
#   ./scripts/solver_bench.sh <bin> <data_dir> [追加オプション]
#
#   <bin>      : qps_benchmark, bench_qplib, または milp_solve
#   <data_dir> : ベンチ対象ディレクトリ
#   追加オプション: --timeout N, --features parallel 等
#
# SOLVER_DIR 環境変数でソルバーリポジトリを指定可能。
# デフォルト: カレントディレクトリ。

set -euo pipefail

# bench_parallel.sh 経由でのみ実行可能（直接実行禁止）
if [[ "${_BENCH_PARALLEL_CALLER:-}" != "1" ]]; then
  echo "[solver_bench.sh] エラー: 直接実行禁止。bench_parallel.sh 経由で実行せよ。" >&2
  echo "[solver_bench.sh] 使い方: bash scripts/bench_parallel.sh --data-dir DIR --timeout SEC --output FILE --jobs N" >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "使い方: $0 <bin> <data_dir> [追加オプション]" >&2
  echo "  <bin>: qps_benchmark または bench_qplib" >&2
  exit 1
fi

bin="$1"; shift
data_dir="$1"; shift

solver_dir="${SOLVER_DIR:-$(pwd)}"
cd "$solver_dir"

echo "[solver_bench.sh] cwd: $(pwd)"
echo "[solver_bench.sh] bin: $bin  data_dir: $data_dir"
echo "[solver_bench.sh] solver_commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "[solver_bench.sh] solver_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "[solver_bench.sh] timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# --features を $@ から分離（cargo build 用）
EXTRA_FEATURES=""
BINARY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --features)
      EXTRA_FEATURES="$2"
      shift 2
      ;;
    *)
      BINARY_ARGS+=("$1")
      shift
      ;;
  esac
done

BUILD_FEATURES="parallel"
if [[ -n "$EXTRA_FEATURES" ]]; then
  BUILD_FEATURES="parallel,$EXTRA_FEATURES"
fi
# 必ず cargo build を実行する。cargo の依存追跡 (mtime + Cargo.lock + features) に
# 任せれば不要な再コンパイルは走らない。
#
# 旧実装は `if [[ ! -f target/release/$bin ]]` で skip していたが、`git checkout`
# 後でも binary が残存していれば旧ソースのままで実行され、bench log の
# `solver_commit: <HEAD>` 表記と実バイナリのソースが乖離する重大バグだった
# (2026-05-17 bisecter 報告)。stale-binary により historical bench log は
# 該当日付までの結果が「現 worktree HEAD の挙動」を保証しない。
echo "[solver_bench.sh] ビルドを開始... (--features $BUILD_FEATURES)"
cargo build --release --features "$BUILD_FEATURES" 2>&1

# milp_solve: single-file binary — iterate over .mps files in data_dir and aggregate.
if [[ "$bin" == "milp_solve" ]]; then
  KNOWN_OPT_FILE=""
  MIP_TIMEOUT="100"
  MIP_EPS="1e-6"
  _idx=0
  while [[ $_idx -lt ${#BINARY_ARGS[@]} ]]; do
    case "${BINARY_ARGS[$_idx]}" in
      --known-optimal) _idx=$(( _idx + 1 )); KNOWN_OPT_FILE="${BINARY_ARGS[$_idx]}" ;;
      --timeout)       _idx=$(( _idx + 1 )); MIP_TIMEOUT="${BINARY_ARGS[$_idx]}" ;;
      --eps)           _idx=$(( _idx + 1 )); MIP_EPS="${BINARY_ARGS[$_idx]}" ;;
    esac
    _idx=$(( _idx + 1 ))
  done

  echo "[solver_bench.sh] milp_solve mode: timeout=${MIP_TIMEOUT}s eps=${MIP_EPS} known-optimal=${KNOWN_OPT_FILE:-none}"

  MIP_TIMEOUT_CMD=$(command -v gtimeout 2>/dev/null || command -v timeout 2>/dev/null || echo "")
  if [[ -n "$MIP_TIMEOUT_CMD" ]]; then
    echo "[solver_bench.sh] 外部timeout: ${MIP_TIMEOUT_CMD} $((MIP_TIMEOUT + 10))s (内部 ${MIP_TIMEOUT}s + 10s 猶予)"
  else
    echo "[solver_bench.sh] 警告: 外部timeout 利用不可（gtimeout/timeout コマンドが見つからない）"
  fi

  n_pass=0; n_checked_noref=0; n_pass_infeasible=0; n_pass_unbounded=0
  n_timeout=0; n_external_timeout=0; n_fail=0; n_maxiter=0; n_error=0; n_skip=0
  n_dfeas_fail=0; n_pfeas_fail=0; n_obj_mismatch=0; n_kkt_fail=0
  n_nonconvex=0; n_suboptimal=0; n_total=0

  printf "\n%-20s %8s %8s %20s %12s %s\n" "NAME" "N_CONS" "N_VARS" "STATUS" "TIME_S" "NOTE"
  printf '%0.s-' {1..80}; echo

  while IFS= read -r mps_file; do
    prob_name=$(basename "$mps_file" .mps)
    n_total=$(( n_total + 1 ))

    mps_out=""
    mps_exit=0
    if [[ -n "$MIP_TIMEOUT_CMD" ]]; then
      mps_out=$("$MIP_TIMEOUT_CMD" "$((MIP_TIMEOUT + 10))" ./target/release/milp_solve "$mps_file" \
        --timeout "$MIP_TIMEOUT" --eps "$MIP_EPS" 2>/dev/null) || mps_exit=$?
    else
      mps_out=$(./target/release/milp_solve "$mps_file" \
        --timeout "$MIP_TIMEOUT" --eps "$MIP_EPS" 2>/dev/null) || mps_exit=$?
    fi

    status_raw=$(printf '%s\n' "$mps_out" | awk '/^status:/ { print $2; exit }')
    obj_str=$(printf '%s\n' "$mps_out" | awk '/^objective:/ { print $2; exit }')
    wall_ms=$(printf '%s\n' "$mps_out" | awk '/^wall_ms:/ { print $2; exit }')
    n_vars=$(printf '%s\n' "$mps_out" | awk '/^n_vars:/ { print $2; exit }')
    n_cons=$(printf '%s\n' "$mps_out" | awk '/^n_cons:/ { print $2; exit }')
    time_s=$(awk "BEGIN{printf \"%.3f\", ${wall_ms:-0}/1000.0}")

    bench_status="FAIL"
    note=""

    # Look up reference objective from known-optimal baseline (empty if not found).
    ref_obj=""
    if [[ -n "$KNOWN_OPT_FILE" && -f "$KNOWN_OPT_FILE" ]]; then
      ref_obj=$(awk -v name="$prob_name" '
        /^#/ { next }
        /^problem_name/ { next }
        { split($0, parts, ","); if (parts[1] == name) { print parts[2]; exit } }
      ' "$KNOWN_OPT_FILE")
    fi

    case "${status_raw:-}" in
      PARSE_ERROR)
        bench_status="ERROR"; note="parse_error"
        n_error=$(( n_error + 1 ))
        ;;
      Optimal)
        obj_check="NOREF"
        if [[ -n "$KNOWN_OPT_FILE" && -f "$KNOWN_OPT_FILE" ]]; then
          obj_check=$(awk -v name="$prob_name" \
            -v solver_obj="${obj_str:-nan}" \
            -v eps_val="$MIP_EPS" '
            /^#/ { next }
            /^problem_name/ { next }
            {
              split($0, parts, ",")
              if (parts[1] == name) {
                known = parts[2] + 0
                denom = known < 0 ? -known : known
                if (denom < 1.0) denom = 1.0
                if (solver_obj == "inf" || solver_obj == "-inf" || solver_obj == "nan") {
                  print "MISMATCH:inf"
                } else {
                  diff = solver_obj - known
                  if (diff < 0) diff = -diff
                  rel_err = diff / denom
                  if (rel_err > eps_val) print "MISMATCH:" rel_err
                  else print "OK:" rel_err
                }
                found = 1; exit
              }
            }
            END { if (!found) print "NOREF" }
          ' "$KNOWN_OPT_FILE")
        fi
        case "${obj_check:-NOREF}" in
          NOREF)
            bench_status="CHECKED[no_ref]"
            n_checked_noref=$(( n_checked_noref + 1 ))
            ;;
          OK:*)
            bench_status="PASS"; note="obj=${obj_str}"
            n_pass=$(( n_pass + 1 ))
            ;;
          MISMATCH:*)
            bench_status="OBJ_MISMATCH"
            note="obj=${obj_str},rel_err=${obj_check#MISMATCH:}"
            n_obj_mismatch=$(( n_obj_mismatch + 1 ))
            ;;
          *)
            bench_status="CHECKED[no_ref]"
            n_checked_noref=$(( n_checked_noref + 1 ))
            ;;
        esac
        ;;
      Infeasible)
        if [ -n "$ref_obj" ]; then
          bench_status="FAIL:false_infeasible"
          n_fail=$(( n_fail + 1 ))
        else
          bench_status="PASS:Infeasible"
          n_pass_infeasible=$(( n_pass_infeasible + 1 ))
        fi
        ;;
      Unbounded)
        if [ -n "$ref_obj" ]; then
          bench_status="FAIL:false_unbounded"
          n_fail=$(( n_fail + 1 ))
        else
          bench_status="PASS:Unbounded"
          n_pass_unbounded=$(( n_pass_unbounded + 1 ))
        fi
        ;;
      Timeout)
        bench_status="TIMEOUT"
        n_timeout=$(( n_timeout + 1 ))
        ;;
      MaxIterations)
        bench_status="MAXITER"
        n_maxiter=$(( n_maxiter + 1 ))
        ;;
      SuboptimalSolution)
        bench_status="SUBOPTIMAL"; note="obj=${obj_str}"
        n_suboptimal=$(( n_suboptimal + 1 ))
        ;;
      NumericalError)
        bench_status="ERROR"; note="numerical_error"
        n_error=$(( n_error + 1 ))
        ;;
      "")
        if [[ "$mps_exit" == "124" ]]; then
          bench_status="EXTERNAL_TIMEOUT"; note="external_timeout_exit=${mps_exit}"
          n_external_timeout=$(( n_external_timeout + 1 ))
        else
          bench_status="ERROR"; note="no_output_exit=${mps_exit}"
          n_error=$(( n_error + 1 ))
        fi
        ;;
      *)
        bench_status="ERROR"; note="unknown_status=${status_raw}"
        n_error=$(( n_error + 1 ))
        ;;
    esac

    printf "%-20s %8s %8s %20s %12.3f %s\n" \
      "$prob_name" "${n_cons:-0}" "${n_vars:-0}" "$bench_status" "$time_s" "$note"
  done < <(find "$data_dir" -maxdepth 1 -iname "*.mps" | sort)

  printf '%0.s-' {1..80}; echo
  echo ""
  echo "=== Summary ==="
  printf "  PASS:              %d\n" "$n_pass"
  printf "  CHECKED[no_ref]:   %d\n" "$n_checked_noref"
  printf "  PASS:Infeasible:   %d\n" "$n_pass_infeasible"
  printf "  PASS:Unbounded:    %d\n" "$n_pass_unbounded"
  printf "  PFEAS_FAIL:        %d\n" "$n_pfeas_fail"
  printf "  DFEAS_FAIL:        %d\n" "$n_dfeas_fail"
  printf "  SUBOPTIMAL:        %d\n" "$n_suboptimal"
  printf "  OBJ_MISMATCH:      %d\n" "$n_obj_mismatch"
  printf "  MAXITER:           %d\n" "$n_maxiter"
  printf "  TIMEOUT:           %d\n" "$n_timeout"
  printf "  EXTERNAL_TIMEOUT:  %d\n" "$n_external_timeout"
  printf "  NONCONVEX:         %d\n" "$n_nonconvex"
  printf "  FAIL:              %d\n" "$n_fail"
  printf "  ERROR:             %d\n" "$n_error"
  printf "  KKT_FAIL:          %d\n" "$n_kkt_fail"
  printf "  SKIP:              %d\n" "$n_skip"
  printf "  TOTAL:             %d\n" "$n_total"
  exit 0
fi

echo "[solver_bench.sh] ./target/release/$bin $data_dir ${BINARY_ARGS[*]}"

# T5修正: 外部タイムアウト安全網（ソルバー内部timeout未設定時の暴走防止）
# macOS: gtimeout（brew install coreutils）、Linux: timeout
TIMEOUT_CMD=$(command -v gtimeout 2>/dev/null || command -v timeout 2>/dev/null || echo "")
if [[ -n "$TIMEOUT_CMD" ]]; then
  echo "[solver_bench.sh] 外部timeout: ${TIMEOUT_CMD} ${EXTERNAL_TIMEOUT:-120}s"
  exec "$TIMEOUT_CMD" "${EXTERNAL_TIMEOUT:-120}" "./target/release/$bin" "$data_dir" "${BINARY_ARGS[@]}"
else
  echo "[solver_bench.sh] 外部timeout: 利用不可（gtimeout/timeout コマンドが見つからない）"
  exec "./target/release/$bin" "$data_dir" "${BINARY_ARGS[@]}"
fi
