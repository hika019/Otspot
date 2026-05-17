#!/bin/bash
# run_lp_bench.sh — LP ベンチマーク統合実行スクリプト
#
# 使い方:
#   bash scripts/run_lp_bench.sh [オプション]
#
# オプション:
#   --suite <suite>    実行する問題セット (デフォルト: all)
#                      all | standard | infeas | hard | extra | canary
#   --eps <eps>        許容誤差 (デフォルト: 1e-6)
#                      "all" を指定すると 1e-4, 1e-6, 1e-8 の 3 パターンを順次実行
#   --jobs <N>         並列数 (デフォルト: 8、canary は single-worker を推奨)
#   --timeout <sec>    タイムアウト秒数 (デフォルト: 1000、canary は 60 を推奨)
#
# 機能:
#   1. データが存在しなければ対応するダウンロードスクリプトを自動実行
#   2. 指定 suite のベンチを bench_parallel.sh 経由で実行
#   3. eps=1e-4, 1e-6, 1e-8 の 3 パターン実行 (--eps all 指定時)
#   4. 結果を bench_results/lp_<suite>_<timestamp>/ に保存
#   5. 完了後にサマリーを表示 (PASS/FAIL/TIMEOUT 件数)
#
# Suite 定義:
#   standard : data/lp_problems/         (Netlib 標準 109 問)
#   infeas   : data/lp_problems_infeas/  (Netlib 実行不可能 29 問)
#   extra    : data/lp_problems_extra/   (Mittelmann Large LP 拡張 4 問)
#   hard     : data/lp_problems_hard/    (数値困難 + 大規模 LP)
#   canary   : data/lp_problems_canary/  (subset 27 問、PR loop 用 5 分以内)
#                                        詳細: docs/canary_suite.md
#                                        ※ canary は all に含まれない
#   all      : standard / infeas / extra / hard を順次実行
#              (CLAUDE.md 規約: 各 suite は逐次)
#
# ベンチ実行は bench_parallel.sh 経由のみ (直接バイナリ呼び出し禁止)。
#
# 例:
#   bash scripts/run_lp_bench.sh
#   bash scripts/run_lp_bench.sh --suite standard --eps all
#   bash scripts/run_lp_bench.sh --suite hard --eps 1e-6 --jobs 4
#   bash scripts/run_lp_bench.sh --suite all --timeout 600
#   bash scripts/run_lp_bench.sh --suite canary --jobs 1 --timeout 60

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# デフォルト値
SUITE="all"
EPS="1e-6"
JOBS="8"
TIMEOUT="1000"

# 引数パース
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)   SUITE="$2";   shift 2 ;;
        --eps)     EPS="$2";     shift 2 ;;
        --jobs)    JOBS="$2";    shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --help|-h)
            sed -n '/^# 使い方/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "エラー: 不明な引数 '$1'" >&2
            echo "使い方: $0 [--suite all|standard|infeas|hard|extra] [--eps EPS|all] [--jobs N] [--timeout SEC]" >&2
            exit 1 ;;
    esac
done

# Suite の妥当性チェック
case "$SUITE" in
    all|standard|infeas|hard|extra|canary) ;;
    *)
        echo "エラー: --suite は all|standard|infeas|hard|extra|canary のいずれか" >&2
        exit 1 ;;
esac

# eps パターンの展開
if [[ "$EPS" == "all" ]]; then
    EPS_LIST=("1e-4" "1e-6" "1e-8")
else
    EPS_LIST=("$EPS")
fi

# Suite ごとのデータディレクトリ・ダウンロードスクリプト定義
suite_data_dir() {
    case "$1" in
        standard) echo "$SOLVER_ROOT/data/lp_problems" ;;
        infeas)   echo "$SOLVER_ROOT/data/lp_problems_infeas" ;;
        extra)    echo "$SOLVER_ROOT/data/lp_problems_extra" ;;
        hard)     echo "$SOLVER_ROOT/data/lp_problems_hard" ;;
        canary)   echo "$SOLVER_ROOT/data/lp_problems_canary" ;;
    esac
}

suite_download_script() {
    case "$1" in
        standard) echo "$SCRIPT_DIR/netlib_lp_download.sh" ;;
        infeas)   echo "$SCRIPT_DIR/netlib_lp_infeas_download.sh" ;;
        extra)    echo "$SCRIPT_DIR/lp_extra_download.sh" ;;
        hard)     echo "$SCRIPT_DIR/lp_hard_download.sh" ;;
        # canary は既存 standard/infeas データへの symlink を貼るだけなので
        # emps バイナリ不要。lp_canary_setup.sh が内部で依存元の有無を確認し、
        # 必要なら netlib_lp_download.sh / netlib_lp_infeas_download.sh を呼ぶ。
        canary)   echo "$SCRIPT_DIR/lp_canary_setup.sh" ;;
    esac
}

# タイムスタンプ
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 結果格納ディレクトリ
RESULTS_BASE="$SOLVER_ROOT/bench_results"
mkdir -p "$RESULTS_BASE"

# ----------------------------------------------------------------
# emps バイナリの確認 (データ取得が必要になる場合に備えて)
# ----------------------------------------------------------------
EMPS="/tmp/emps"
ensure_emps() {
    if [ ! -x "$EMPS" ]; then
        echo "[run_lp_bench] emps バイナリが見つからないため、コンパイルします..." >&2
        curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
        cc -o "$EMPS" /tmp/emps.c
        echo "[run_lp_bench] emps コンパイル完了" >&2
    fi
}

# ----------------------------------------------------------------
# 単一 suite + 単一 eps のベンチ実行
# ----------------------------------------------------------------
run_one() {
    local suite="$1"
    local eps="$2"
    local data_dir
    data_dir="$(suite_data_dir "$suite")"
    local dl_script
    dl_script="$(suite_download_script "$suite")"
    local result_dir="$RESULTS_BASE/lp_${suite}_${TIMESTAMP}"
    local output_file="$result_dir/result_eps${eps}.txt"

    echo "" >&2
    echo "================================================================" >&2
    echo "[run_lp_bench] Suite: $suite  eps: $eps  jobs: $JOBS  timeout: ${TIMEOUT}s" >&2
    echo "================================================================" >&2

    # データが存在しなければダウンロード
    local qps_count
    qps_count=$(find "$data_dir" -maxdepth 1 -iname "*.qps" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$qps_count" -eq 0 ]]; then
        echo "[run_lp_bench] $data_dir に問題ファイルがないため、ダウンロードします..." >&2
        ensure_emps
        EMPS_BIN="$EMPS" bash "$dl_script" "$data_dir"
        qps_count=$(find "$data_dir" -maxdepth 1 -iname "*.qps" 2>/dev/null | wc -l | tr -d ' ')
        echo "[run_lp_bench] ダウンロード完了: ${qps_count} 問" >&2
    else
        echo "[run_lp_bench] データ確認済み: ${qps_count} 問 ($data_dir)" >&2
    fi

    if [[ "$qps_count" -eq 0 ]]; then
        echo "[run_lp_bench] 警告: $data_dir に問題ファイルがない。suite=$suite をスキップ" >&2
        return 0
    fi

    mkdir -p "$result_dir"

    # bench_parallel.sh 経由でベンチ実行
    SOLVER_DIR="$SOLVER_ROOT" \
    bash "$SCRIPT_DIR/bench_parallel.sh" \
        --data-dir "$data_dir" \
        --timeout "$TIMEOUT" \
        --eps "$eps" \
        --jobs "$JOBS" \
        --output "$output_file"

    echo "" >&2
    echo "[run_lp_bench] 結果保存: $output_file" >&2
}

# ----------------------------------------------------------------
# メイン: suite と eps の全組み合わせを実行
# ----------------------------------------------------------------
echo "[run_lp_bench] 開始: suite=$SUITE, eps=${EPS_LIST[*]}, jobs=$JOBS, timeout=${TIMEOUT}s" >&2
echo "[run_lp_bench] SOLVER_ROOT: $SOLVER_ROOT" >&2

if [[ "$SUITE" == "all" ]]; then
    # CLAUDE.md 規約: 各 suite を順次実行
    for suite in standard infeas extra hard; do
        for eps in "${EPS_LIST[@]}"; do
            run_one "$suite" "$eps"
        done
    done
else
    for eps in "${EPS_LIST[@]}"; do
        run_one "$SUITE" "$eps"
    done
fi

echo "" >&2
echo "[run_lp_bench] 全ベンチ完了。結果: $RESULTS_BASE/" >&2
echo "[run_lp_bench] 結果一覧:" >&2
find "$RESULTS_BASE" -name "result_eps*.txt" -newer "$SCRIPT_DIR/run_lp_bench.sh" 2>/dev/null | sort | while read -r f; do
    local_pass=$(grep "  PASS:" "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)
    local_fail=$(grep "  FAIL:" "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)
    local_timeout=$(grep "  TIMEOUT:" "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)
    local_total=$(grep "  TOTAL:" "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)
    printf "  %s: PASS=%s FAIL=%s TIMEOUT=%s TOTAL=%s\n" \
        "$(basename "$f")" "${local_pass:-?}" "${local_fail:-?}" "${local_timeout:-?}" "${local_total:-?}" >&2
done
