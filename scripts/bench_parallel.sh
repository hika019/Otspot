#!/bin/bash
# bench_parallel.sh — 汎用ベンチ並列実行スクリプト (ワークプール方式)
#
# solver_bench.sh経由で --jobs 数のワーカーが問題キューを処理し、結果を集計する。
# .qps / .qplib の両形式に対応。
#
# 使い方:
#   SOLVER_DIR=/path/to/solver \
#   bash scripts/bench_parallel.sh \
#     --data-dir <dir> \
#     --timeout <sec> \
#     --output <file> \
#     [--eps <eps>]      (default: 1e-6)
#     --jobs <N>         (必須。暗黙のデフォルト禁止)
#     [--features <feat>]
#
# 注意:
# - solver_bench.sh 経由（§43準拠）。直接バイナリ呼び出し禁止
# - .qps と .qplib の混在ディレクトリは非対応（エラーで終了）
# - ワークプール方式: 問題を3問/グループに分割し、Nワーカーが動的に取得

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# デフォルト値
EPS="1e-6"
JOBS=""
FEATURES=""
DATA_DIR=""
TIMEOUT=""
OUTPUT=""

# 引数パース
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)  DATA_DIR="$2";  shift 2 ;;
    --timeout)   TIMEOUT="$2";   shift 2 ;;
    --eps)       EPS="$2";       shift 2 ;;
    --jobs)      JOBS="$2";      shift 2 ;;
    --output)    OUTPUT="$2";    shift 2 ;;
    --features)  FEATURES="$2";  shift 2 ;;
    *) echo "エラー: 不明な引数 '$1'" >&2
       echo "使い方: $0 --data-dir DIR --timeout SEC --output FILE [--eps EPS] [--jobs N] [--features FEAT]" >&2
       exit 1 ;;
  esac
done

# 必須引数チェック（暗黙のデフォルトモード禁止）
if [[ -z "$DATA_DIR" || -z "$TIMEOUT" || -z "$OUTPUT" || -z "$JOBS" ]]; then
  echo "エラー: --data-dir, --timeout, --output, --jobs は全て必須" >&2
  echo "使い方: $0 --data-dir DIR --timeout SEC --output FILE --jobs N [--eps EPS] [--features FEAT]" >&2
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
elif echo "$DATA_DIR_LOWER" | grep -qE "qp[_-]?unbounded"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/qp_unbounded.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "qp[_-]?infeasible"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/qp_infeasible.csv"
elif echo "$DATA_DIR_LOWER" | grep -q "qplib"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/qplib.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "osqp[_-]?bench"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/osqp_bench.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "mpc[_-]?qp"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/mpc_qp.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "lp[_-]?problems[_-]?infeas"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp_infeas.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "lp[_-]?problems[_-]?hard"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp_hard.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "lp[_-]?problems[_-]?extra"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp_extra.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "lp[_-]?problems[_-]?canary"; then
  # canary は standard / infeas 両方の問題を含むので、専用 baseline を使う
  # (生成: lp_canary_setup.sh と一緒に手動メンテ。docs/canary_suite.md 参照)
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp_canary.csv"
elif echo "$DATA_DIR_LOWER" | grep -qE "lp[_-]?problems"; then
  KNOWN_OPTIMAL="$SOLVER_ROOT/data/baseline_objectives/netlib_lp.csv"
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
echo "[bench_parallel.sh] 対象: $TOTAL_FILES 件 (bin=$BIN, timeout=${TIMEOUT}s, eps=$EPS, jobs=$JOBS)"

# ワークプールのグループサイズ (1問/グループ)。
# 重い問題を含むグループが timeout に到達すると、そのワーカーが長時間塞がれ
# 他ワーカーが手伝えない non-work-stealing の弊害が大きいため、最小粒度で
# 動的に分配する。問題数の log ファイルが増える代わりに最後まで JOBS が活用される。
GROUP_SIZE=1
TOTAL_GROUPS=$(( (TOTAL_FILES + GROUP_SIZE - 1) / GROUP_SIZE ))

# jobs をグループ数に合わせて調整
if [[ $JOBS -gt $TOTAL_GROUPS ]]; then
  JOBS=$TOTAL_GROUPS
  echo "[bench_parallel.sh] jobs を $JOBS に調整（グループ数未満）"
fi

# 一時ディレクトリ作成
TMPDIR_BASE="/tmp/bench_parallel_$$"
mkdir -p "$TMPDIR_BASE"

# 終了時クリーンアップ
cleanup() {
  if [[ "${BENCH_KEEP_LOGS:-0}" != "1" ]]; then rm -rf "$TMPDIR_BASE"; else echo "[DEBUG] keeping $TMPDIR_BASE" >&2; fi
}
trap cleanup EXIT

# グループディレクトリ作成（3問ずつ）
for g in $(seq 1 "$TOTAL_GROUPS"); do
  mkdir -p "$TMPDIR_BASE/group_$(printf '%03d' "$g")"
done

# ファイルを3問ずつグループに分配
for idx in "${!FILES[@]}"; do
  f="${FILES[$idx]}"
  group_num=$(( idx / GROUP_SIZE + 1 ))
  ln -sf "$f" "$TMPDIR_BASE/group_$(printf '%03d' "$group_num")/$(basename "$f")"
done

echo "[bench_parallel.sh] グループ分割: ${TOTAL_GROUPS}グループ (最大${GROUP_SIZE}問/グループ, ${JOBS}ワーカー)"


# features 引数の構築
FEATURES_EXTRA=""
if [[ -n "$FEATURES" ]]; then
  FEATURES_EXTRA="--features $FEATURES"
fi

# 外部タイムアウト (1ワーカーが最大GROUP_SIZE問担当)
EXTERNAL_TIMEOUT=$(( TIMEOUT * GROUP_SIZE + 300 ))
export EXTERNAL_TIMEOUT
echo "[bench_parallel.sh] EXTERNAL_TIMEOUT: ${EXTERNAL_TIMEOUT}s (${GROUP_SIZE}問 × ${TIMEOUT}s + 300s余裕)"

# ワーカープール設定
COUNTER_FILE="$TMPDIR_BASE/counter"
COUNTER_LOCK="$TMPDIR_BASE/counter.lock"
echo "1" > "$COUNTER_FILE"
: > "$COUNTER_LOCK"
FAILED_GROUPS_FILE="$TMPDIR_BASE/failed_groups.txt"
: > "$FAILED_GROUPS_FILE"

set +e  # 子プロセスの終了コードを個別に確認するため
KNOWN_OPTIMAL_ARG=()
if [[ -n "$KNOWN_OPTIMAL" ]]; then
  KNOWN_OPTIMAL_ARG=(--known-optimal "$KNOWN_OPTIMAL")
fi

# ワーカー関数：キューからグループを取得して処理
worker_func() {
  local worker_id="$1"
  while true; do
    # アトミックに次のグループ番号を取得
    local group_num
    group_num=$(
      (
        flock -x 9
        n=$(cat "$COUNTER_FILE")
        echo $(( n + 1 )) > "$COUNTER_FILE"
        echo "$n"
      ) 9>"$COUNTER_LOCK"
    )

    if [[ $group_num -gt $TOTAL_GROUPS ]]; then
      break
    fi

    local group_name
    group_name="group_$(printf '%03d' "$group_num")"
    local group_dir="$TMPDIR_BASE/$group_name"
    local log="$TMPDIR_BASE/${group_name}.log"

    echo "[bench_parallel.sh] ワーカー $worker_id: $group_name 開始"

    local exit_code=0
    _BENCH_PARALLEL_CALLER=1 \
    SOLVER_DIR="${SOLVER_DIR:-$(pwd)}" \
    bash "$SCRIPT_DIR/solver_bench.sh" "$BIN" "$group_dir" \
      --eps "$EPS" \
      --timeout "$TIMEOUT" \
      "${KNOWN_OPTIMAL_ARG[@]}" \
      ${FEATURES_EXTRA} > "$log" 2>&1 || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
      echo "$group_name" >> "$FAILED_GROUPS_FILE"
      echo "[bench_parallel.sh] ワーカー $worker_id: $group_name 異常終了 (exit=$exit_code)" >&2
    else
      echo "[bench_parallel.sh] ワーカー $worker_id: $group_name 完了"
    fi
  done
}

# N個のワーカーを起動
declare -a WORKER_PIDS
for w in $(seq 1 "$JOBS"); do
  worker_func "$w" &
  WORKER_PIDS+=($!)
  echo "[bench_parallel.sh] ワーカー $w 起動 (PID=$!)"
done

# 全ワーカーの完了待ち
for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
done

# 失敗グループ収集
FAILED_GROUPS=()
if [[ -s "$FAILED_GROUPS_FILE" ]]; then
  while IFS= read -r g; do
    [[ -n "$g" ]] && FAILED_GROUPS+=("$g")
  done < "$FAILED_GROUPS_FILE"
fi

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
TOTAL_PASS_INFEASIBLE=0
TOTAL_PASS_UNBOUNDED=0
TOTAL_OBJ_MISMATCH=0
TOTAL_NONCONVEX=0
TOTAL_SUBOPTIMAL=0

# 問題別詳細行の収集（PARSE/SOLVE/=>行を除く、問題名+STATUS行のみ）
PROBLEM_DETAIL_FILE="$TMPDIR_BASE/problem_details.txt"
: > "$PROBLEM_DETAIL_FILE"

for g in $(seq 1 "$TOTAL_GROUPS"); do
  group_name="group_$(printf '%03d' "$g")"
  LOG="$TMPDIR_BASE/${group_name}.log"
  if [[ ! -f "$LOG" ]]; then
    echo "[bench_parallel.sh] 警告: $group_name のログが存在しない" >&2
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
  pass_infeasible=$(grep -E "^\s+PASS:Infeasible:" "$LOG" | awk '{print $2}' | head -1)
  pass_unbounded=$(grep -E "^\s+PASS:Unbounded:" "$LOG" | awk '{print $2}' | head -1)
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
  TOTAL_PASS_INFEASIBLE=$(( TOTAL_PASS_INFEASIBLE + ${pass_infeasible:-0} ))
  TOTAL_PASS_UNBOUNDED=$(( TOTAL_PASS_UNBOUNDED + ${pass_unbounded:-0} ))
  TOTAL_OBJ_MISMATCH=$(( TOTAL_OBJ_MISMATCH + ${obj_mismatch:-0} ))
  TOTAL_NONCONVEX=$(( TOTAL_NONCONVEX + ${nonconvex:-0} ))
  TOTAL_SUBOPTIMAL=$(( TOTAL_SUBOPTIMAL + ${suboptimal:-0} ))

  # 問題別詳細行（PARSE_/SOLVE_/=>行を除く、STATUS含む行）
  grep -E "\s+(PASS((\[no_ref\])|(:Infeasible)|(:Unbounded))?|TIMEOUT|(DFEAS_FAIL|PFEAS_FAIL|FAIL)(:[A-Za-z]+)?|OBJ_MISMATCH|NONCONVEX|SUBOPTIMAL|MAXITER|ERROR)" "$LOG" \
    | grep -v -E "^(PARSE_|SOLVE_)" >> "$PROBLEM_DETAIL_FILE" 2>/dev/null || true
done

# 結果を出力ファイルとstdoutに書き込み
{
  echo "=== bench_parallel.sh 集計結果 ==="
  echo "data-dir : $DATA_DIR"
  echo "timeout  : ${TIMEOUT}s"
  echo "eps      : $EPS"
  echo "jobs     : $JOBS"
  echo ""
  if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "★ 異常終了グループ: ${FAILED_GROUPS[*]}"
    echo ""
  fi
  echo "=== Summary ==="
  printf "  PASS:              %d\n" "$TOTAL_PASS"
  printf "  PASS[no_ref]:      %d\n" "$TOTAL_PASS_NO_REF"
  printf "  PASS:Infeasible:   %d\n" "$TOTAL_PASS_INFEASIBLE"
  printf "  PASS:Unbounded:    %d\n" "$TOTAL_PASS_UNBOUNDED"
  printf "  TIMEOUT:           %d\n" "$TOTAL_TIMEOUT"
  printf "  FAIL:              %d\n" "$TOTAL_FAIL"
  printf "  DFEAS_FAIL:        %d\n" "$TOTAL_DFEAS_FAIL"
  printf "  PFEAS_FAIL:        %d\n" "$TOTAL_PFEAS_FAIL"
  printf "  OBJ_MISMATCH:      %d\n" "$TOTAL_OBJ_MISMATCH"
  printf "  NONCONVEX:         %d\n" "$TOTAL_NONCONVEX"
  printf "  SUBOPTIMAL:        %d\n" "$TOTAL_SUBOPTIMAL"
  printf "  MAXITER:           %d\n" "$TOTAL_MAXITER"
  printf "  ERROR:             %d\n" "$TOTAL_ERROR"
  printf "  SKIP:              %d\n" "$TOTAL_SKIP"
  printf "  TOTAL:             %d\n" "$TOTAL_PROBLEMS"
  echo ""
  echo "=== 問題別詳細 ==="
  if [[ -s "$PROBLEM_DETAIL_FILE" ]]; then
    sort "$PROBLEM_DETAIL_FILE"
  else
    echo "  (詳細なし)"
  fi
} | tee "$OUTPUT"

# TOTAL整合性チェック
CATEGORY_SUM=$(( TOTAL_PASS + TOTAL_PASS_NO_REF + TOTAL_PASS_INFEASIBLE + TOTAL_PASS_UNBOUNDED + \
  TOTAL_TIMEOUT + TOTAL_FAIL + \
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
