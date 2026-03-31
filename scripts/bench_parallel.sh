#!/bin/bash
# bench_parallel.sh — 汎用ベンチ並列実行スクリプト
#
# solver_bench.sh経由で --jobs 数のグループを並列実行し、結果を集計する。
# .qps / .qplib の両形式に対応。
#
# 使い方:
#   SOLVER_DIR=/path/to/solver \
#   bash scripts/bench_parallel.sh \
#     --data-dir <dir> \
#     --solver <solver> \
#     --timeout <sec> \
#     --output <file> \
#     [--eps <eps>]      (default: 1e-6)
#     --jobs <N>         (必須。暗黙のデフォルト禁止)
#     [--features <feat>]
#
# 注意:
# - solver_bench.sh 経由（§43準拠）。直接バイナリ呼び出し禁止
# - .qps と .qplib の混在ディレクトリは非対応（エラーで終了）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# デフォルト値
EPS="1e-6"
JOBS=""
FEATURES=""
DATA_DIR=""
SOLVER=""
TIMEOUT=""
OUTPUT=""

# 引数パース
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)  DATA_DIR="$2";  shift 2 ;;
    --solver)    SOLVER="$2";    shift 2 ;;
    --timeout)   TIMEOUT="$2";   shift 2 ;;
    --eps)       EPS="$2";       shift 2 ;;
    --jobs)      JOBS="$2";      shift 2 ;;
    --output)    OUTPUT="$2";    shift 2 ;;
    --features)  FEATURES="$2";  shift 2 ;;
    *) echo "エラー: 不明な引数 '$1'" >&2
       echo "使い方: $0 --data-dir DIR --solver SOLVER --timeout SEC --output FILE [--eps EPS] [--jobs N] [--features FEAT]" >&2
       exit 1 ;;
  esac
done

# 必須引数チェック（--solver含む全て必須。暗黙のデフォルトモード禁止）
if [[ -z "$DATA_DIR" || -z "$SOLVER" || -z "$TIMEOUT" || -z "$OUTPUT" || -z "$JOBS" ]]; then
  echo "エラー: --data-dir, --solver, --timeout, --output, --jobs は全て必須" >&2
  echo "使い方: $0 --data-dir DIR --solver SOLVER --timeout SEC --output FILE --jobs N [--eps EPS] [--features FEAT]" >&2
  echo "  --solver: concurrent|ipm|ippmm_new（暗黙のデフォルト禁止）" >&2
  echo "  --jobs: 並列数を明示せよ（暗黙のデフォルト禁止）" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "エラー: --data-dir '$DATA_DIR' が存在しない" >&2
  exit 1
fi

DATA_DIR=$(realpath "$DATA_DIR")

# 元の DATA_DIR 名から正解値CSVパスを決定
DATA_DIR_LOWER=$(echo "$DATA_DIR" | tr '[:upper:]' '[:lower:]')
SOLVER_ROOT="${SOLVER_DIR:-$(pwd)}"
if echo "$DATA_DIR_LOWER" | grep -q "maros"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/maros_meszaros.csv"
elif echo "$DATA_DIR_LOWER" | grep -q "qplib"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/qplib.csv"
else
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp.csv"
fi

if [[ ! -f "$KNOWN_OPTIMAL" ]]; then
  echo "警告: 正解値CSV '$KNOWN_OPTIMAL' が見つからない。PASS[no_ref]になる可能性あり" >&2
fi

# ファイル拡張子の自動判別
QPS_COUNT=$(find "$DATA_DIR" -maxdepth 1 \( -iname "*.qps" \) | wc -l | tr -d ' ')
QPLIB_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name "*.qplib" | wc -l | tr -d ' ')

if [[ "$QPS_COUNT" -gt 0 && "$QPLIB_COUNT" -gt 0 ]]; then
  echo "エラー: .qps と .qplib が混在している。非対応。" >&2
  exit 1
fi

if [[ "$QPS_COUNT" -eq 0 && "$QPLIB_COUNT" -eq 0 ]]; then
  echo "エラー: '$DATA_DIR' に .qps/.qplib ファイルが存在しない" >&2
  exit 1
fi

FILES=()
if [[ "$QPS_COUNT" -gt 0 ]]; then
  BIN="qps_benchmark"
  while IFS= read -r f; do
    FILES+=("$f")
  done < <(find "$DATA_DIR" -maxdepth 1 \( -iname "*.qps" \) | sort)
else
  BIN="bench_qplib"
  while IFS= read -r f; do
    FILES+=("$f")
  done < <(find "$DATA_DIR" -maxdepth 1 -name "*.qplib" | sort)
fi

TOTAL_FILES=${#FILES[@]}

# トレーサビリティ情報の記録
SCRIPT_VERSION=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
SOLVER_COMMIT=$(git -C "${SOLVER_DIR:-$(pwd)}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
SOLVER_BRANCH=$(git -C "${SOLVER_DIR:-$(pwd)}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "[bench_parallel.sh] === トレーサビリティ ==="
echo "[bench_parallel.sh] script_commit: $SCRIPT_VERSION (solver)"
echo "[bench_parallel.sh] solver_commit: $SOLVER_COMMIT (branch: $SOLVER_BRANCH)"
echo "[bench_parallel.sh] solver_dir: ${SOLVER_DIR:-$(pwd)}"
echo "[bench_parallel.sh] timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[bench_parallel.sh] 対象: $TOTAL_FILES 件 (bin=$BIN, solver=${SOLVER:-default}, timeout=${TIMEOUT}s, eps=$EPS, jobs=$JOBS)"

# jobs をファイル数に合わせて調整
if [[ $JOBS -gt $TOTAL_FILES ]]; then
  JOBS=$TOTAL_FILES
  echo "[bench_parallel.sh] jobs を $JOBS に調整（ファイル数未満）"
fi

# 一時ディレクトリ作成
TMPDIR_BASE="/tmp/bench_parallel_$$"
mkdir -p "$TMPDIR_BASE"

# 終了時クリーンアップ
cleanup() {
  rm -rf "$TMPDIR_BASE"
}
trap cleanup EXIT

# グループディレクトリ作成
for i in $(seq 1 "$JOBS"); do
  mkdir -p "$TMPDIR_BASE/group_$i"
done

# ファイルをラウンドロビンで分配
for idx in "${!FILES[@]}"; do
  f="${FILES[$idx]}"
  group_num=$(( idx % JOBS + 1 ))
  ln -sf "$f" "$TMPDIR_BASE/group_$group_num/$(basename "$f")"
done

# 分割状況を表示
echo "[bench_parallel.sh] グループ分割:"
for i in $(seq 1 "$JOBS"); do
  count=$(ls "$TMPDIR_BASE/group_$i" 2>/dev/null | wc -l | tr -d ' ')
  echo "  グループ $i: $count 件"
done


# features 引数の構築
FEATURES_EXTRA=""
if [[ -n "$FEATURES" ]]; then
  FEATURES_EXTRA="--features $FEATURES"
fi

# 外部タイムアウトをグループ規模に合わせて設定（デフォルト120sでは不足）
MAX_PER_GROUP=$(( (TOTAL_FILES + JOBS - 1) / JOBS ))
EXTERNAL_TIMEOUT=$(( TIMEOUT * MAX_PER_GROUP + 300 ))
export EXTERNAL_TIMEOUT
echo "[bench_parallel.sh] EXTERNAL_TIMEOUT: ${EXTERNAL_TIMEOUT}s (${MAX_PER_GROUP}問 × ${TIMEOUT}s + 300s余裕)"

# 各グループを並列起動
declare -a PIDS
declare -a LOGS
set +e  # 子プロセスの終了コードを個別に確認するため
SOLVER_ARGS=()
if [[ -n "$SOLVER" ]]; then
  SOLVER_ARGS=(--solver "$SOLVER")
fi

for i in $(seq 1 "$JOBS"); do
  LOG="$TMPDIR_BASE/group_$i.log"
  LOGS+=("$LOG")
  # bench_qplib は --known-optimal 未サポート（渡すとdata_dir上書きで異常終了）
  # qps_benchmark のみに渡す
  KNOWN_OPTIMAL_ARG=()
  if [[ "$BIN" == "qps_benchmark" && -n "$KNOWN_OPTIMAL" ]]; then
    KNOWN_OPTIMAL_ARG=(--known-optimal "$KNOWN_OPTIMAL")
  fi

  # ★ --eps は solver_bench.sh が自動注入する（1e-6固定）。ここでは渡さない（二重防止）
  _BENCH_PARALLEL_CALLER=1 \
  SOLVER_DIR="${SOLVER_DIR:-$(pwd)}" \
  bash "$SCRIPT_DIR/solver_bench.sh" "$BIN" "$TMPDIR_BASE/group_$i" \
    "${SOLVER_ARGS[@]}" \
    --timeout "$TIMEOUT" \
    "${KNOWN_OPTIMAL_ARG[@]}" \
    ${FEATURES_EXTRA} > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "[bench_parallel.sh] グループ $i 開始 (PID=$!)"
done

# 全グループの完了待ち
FAILED_GROUPS=()
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  group_num=$(( i + 1 ))
  if wait "$pid"; then
    echo "[bench_parallel.sh] グループ $group_num 完了"
  else
    echo "[bench_parallel.sh] グループ $group_num 異常終了 (exit=$?)" >&2
    FAILED_GROUPS+=("$group_num")
  fi
done

# 集計
TOTAL_PASS=0
TOTAL_TIMEOUT=0
TOTAL_FAIL=0
TOTAL_MAXITER=0
TOTAL_ERROR=0
TOTAL_SKIP=0
TOTAL_PROBLEMS=0
TOTAL_DFEAS_FAIL=0
TOTAL_PFEAS_FAIL=0
TOTAL_PASS_NO_REF=0
TOTAL_OBJ_MISMATCH=0
TOTAL_NONCONVEX=0
TOTAL_SUBOPTIMAL=0

# 問題別詳細行の収集（PARSE/SOLVE/=>行を除く、問題名+STATUS行のみ）
PROBLEM_DETAIL_FILE="$TMPDIR_BASE/problem_details.txt"
: > "$PROBLEM_DETAIL_FILE"

for i in $(seq 1 "$JOBS"); do
  LOG="$TMPDIR_BASE/group_$i.log"
  if [[ ! -f "$LOG" ]]; then
    echo "[bench_parallel.sh] 警告: グループ $i のログが存在しない" >&2
    continue
  fi

  # Summaryから数値を抽出
  pass=$(grep -E "^\s+PASS:" "$LOG" | awk '{print $2}' | head -1)
  timeout=$(grep -E "^\s+TIMEOUT:" "$LOG" | awk '{print $2}' | head -1)
  fail=$(grep -E "^\s+FAIL:" "$LOG" | awk '{print $2}' | head -1)
  maxiter=$(grep -E "^\s+MAXITER:" "$LOG" | awk '{print $2}' | head -1)
  error=$(grep -E "^\s+ERROR:" "$LOG" | awk '{print $2}' | head -1)
  skip=$(grep -E "^\s+SKIP:" "$LOG" | awk '{print $2}' | head -1)
  total=$(grep -E "^\s+TOTAL:" "$LOG" | awk '{print $2}' | head -1)
  dfeas_fail=$(grep -E "^\s+DFEAS_FAIL:" "$LOG" | awk '{print $2}' | head -1)
  pfeas_fail=$(grep -E "^\s+PFEAS_FAIL:" "$LOG" | awk '{print $2}' | head -1)
  pass_no_ref=$(grep -E "^\s+PASS\[no_ref\]:" "$LOG" | awk '{print $2}' | head -1)
  obj_mismatch=$(grep -E "^\s+OBJ_MISMATCH:" "$LOG" | awk '{print $2}' | head -1)
  nonconvex=$(grep -E "^\s+NONCONVEX:" "$LOG" | awk '{print $2}' | head -1)
  suboptimal=$(grep -E "^\s+SUBOPTIMAL:" "$LOG" | awk '{print $2}' | head -1)

  TOTAL_PASS=$(( TOTAL_PASS + ${pass:-0} ))
  TOTAL_TIMEOUT=$(( TOTAL_TIMEOUT + ${timeout:-0} ))
  TOTAL_FAIL=$(( TOTAL_FAIL + ${fail:-0} ))
  TOTAL_MAXITER=$(( TOTAL_MAXITER + ${maxiter:-0} ))
  TOTAL_ERROR=$(( TOTAL_ERROR + ${error:-0} ))
  TOTAL_SKIP=$(( TOTAL_SKIP + ${skip:-0} ))
  TOTAL_PROBLEMS=$(( TOTAL_PROBLEMS + ${total:-0} ))
  TOTAL_DFEAS_FAIL=$(( TOTAL_DFEAS_FAIL + ${dfeas_fail:-0} ))
  TOTAL_PFEAS_FAIL=$(( TOTAL_PFEAS_FAIL + ${pfeas_fail:-0} ))
  TOTAL_PASS_NO_REF=$(( TOTAL_PASS_NO_REF + ${pass_no_ref:-0} ))
  TOTAL_OBJ_MISMATCH=$(( TOTAL_OBJ_MISMATCH + ${obj_mismatch:-0} ))
  TOTAL_NONCONVEX=$(( TOTAL_NONCONVEX + ${nonconvex:-0} ))
  TOTAL_SUBOPTIMAL=$(( TOTAL_SUBOPTIMAL + ${suboptimal:-0} ))

  # 問題別詳細行（PARSE_/SOLVE_/=>行を除く、STATUS含む行）
  grep -E "\s+(PASS(\[no_ref\])?|TIMEOUT|(DFEAS_FAIL|PFEAS_FAIL|FAIL)(:[A-Za-z]+)?|OBJ_MISMATCH|NONCONVEX|SUBOPTIMAL|MAXITER|ERROR)" "$LOG" \
    | grep -v -E "^(PARSE_|SOLVE_)" >> "$PROBLEM_DETAIL_FILE" 2>/dev/null || true
done

# 結果を出力ファイルとstdoutに書き込み
{
  echo "=== bench_parallel.sh 集計結果 ==="
  echo "data-dir : $DATA_DIR"
  echo "solver   : $SOLVER"
  echo "timeout  : ${TIMEOUT}s"
  echo "eps      : $EPS"
  echo "jobs     : $JOBS"
  echo ""
  if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "★ 異常終了グループ: ${FAILED_GROUPS[*]}"
    echo ""
  fi
  echo "=== Summary ==="
  printf "  PASS:           %d\n" "$TOTAL_PASS"
  printf "  PASS[no_ref]:   %d\n" "$TOTAL_PASS_NO_REF"
  printf "  TIMEOUT:        %d\n" "$TOTAL_TIMEOUT"
  printf "  FAIL:           %d\n" "$TOTAL_FAIL"
  printf "  DFEAS_FAIL:     %d\n" "$TOTAL_DFEAS_FAIL"
  printf "  PFEAS_FAIL:     %d\n" "$TOTAL_PFEAS_FAIL"
  printf "  OBJ_MISMATCH:   %d\n" "$TOTAL_OBJ_MISMATCH"
  printf "  NONCONVEX:      %d\n" "$TOTAL_NONCONVEX"
  printf "  SUBOPTIMAL:     %d\n" "$TOTAL_SUBOPTIMAL"
  printf "  MAXITER:        %d\n" "$TOTAL_MAXITER"
  printf "  ERROR:          %d\n" "$TOTAL_ERROR"
  printf "  SKIP:           %d\n" "$TOTAL_SKIP"
  printf "  TOTAL:          %d\n" "$TOTAL_PROBLEMS"
  echo ""
  echo "=== 問題別詳細 ==="
  if [[ -s "$PROBLEM_DETAIL_FILE" ]]; then
    sort "$PROBLEM_DETAIL_FILE"
  else
    echo "  (詳細なし)"
  fi
} | tee "$OUTPUT"

# TOTAL整合性チェック
CATEGORY_SUM=$(( TOTAL_PASS + TOTAL_PASS_NO_REF + TOTAL_TIMEOUT + TOTAL_FAIL + \
  TOTAL_DFEAS_FAIL + TOTAL_PFEAS_FAIL + TOTAL_OBJ_MISMATCH + TOTAL_NONCONVEX + \
  TOTAL_SUBOPTIMAL + TOTAL_MAXITER + TOTAL_ERROR + TOTAL_SKIP ))
if [[ "$CATEGORY_SUM" != "$TOTAL_PROBLEMS" ]]; then
  echo "警告: カテゴリ合算($CATEGORY_SUM) ≠ TOTAL($TOTAL_PROBLEMS)" >&2
fi

echo ""
echo "[bench_parallel.sh] 結果を $OUTPUT に出力した"

# 異常終了グループがあれば exit 1
if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
