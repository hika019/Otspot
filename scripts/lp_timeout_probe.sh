#!/bin/bash
# lp_timeout_probe.sh — LP hard timeout probe for cont1/cont11/cont4
#
# 目的:
#   cont1 / cont11 / cont4 を最小セットで実行し、
#   TIMEOUT が internal か external かを自動判別して整形出力する。
#
# 使い方:
#   SOLVER_DIR=/path/to/solver \
#   bash scripts/lp_timeout_probe.sh \
#     [--hard-dir DIR] \
#     [--timeout SEC] \
#     [--eps EPS] \
#     [--jobs N] \
#     [--ext-timeout-buffer SEC] \
#     [--from-bench-output FILE] \
#     [--bench-output FILE] \
#     [--report FILE] \
#     [--class-tsv FILE]
#
# 出力:
#   1) bench_parallel.sh の生ログ: --bench-output
#   2) 判別レポート: --report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_ROOT="${SOLVER_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

HARD_DIR="$SOLVER_ROOT/data/lp_problems_hard"
TIMEOUT="120"
EPS="1e-6"
JOBS="1"
BENCH_OUTPUT=""
REPORT_OUTPUT=""
FROM_BENCH_OUTPUT=""
EXT_TIMEOUT_BUFFER=""
CLASS_TSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hard-dir) HARD_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --eps) EPS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --ext-timeout-buffer) EXT_TIMEOUT_BUFFER="$2"; shift 2 ;;
        --from-bench-output) FROM_BENCH_OUTPUT="$2"; shift 2 ;;
        --bench-output) BENCH_OUTPUT="$2"; shift 2 ;;
        --report) REPORT_OUTPUT="$2"; shift 2 ;;
        --class-tsv) CLASS_TSV="$2"; shift 2 ;;
        --help|-h)
            sed -n '/^# 目的:/,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "エラー: 不明な引数 '$1'" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$HARD_DIR" ]]; then
    echo "エラー: hard ディレクトリが存在しない: $HARD_DIR" >&2
    exit 1
fi

TS="$(date '+%Y%m%d_%H%M%S')"
BENCH_OUTPUT="${BENCH_OUTPUT:-/private/tmp/lp_timeout_probe_bench_${TS}.txt}"
REPORT_OUTPUT="${REPORT_OUTPUT:-/private/tmp/lp_timeout_probe_report_${TS}.txt}"
CLASS_TSV="${CLASS_TSV:-/private/tmp/lp_timeout_probe_class_${TS}.tsv}"

WORK_DIR="$(mktemp -d "/private/tmp/lp_timeout_probe.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ -z "$FROM_BENCH_OUTPUT" ]]; then
    for base in cont1 cont11 cont4; do
        src="$HARD_DIR/${base}.QPS"
        if [[ ! -f "$src" ]]; then
            src="$HARD_DIR/${base}.qps"
        fi
        if [[ ! -f "$src" ]]; then
            echo "エラー: 問題ファイルが見つからない: ${base}.QPS" >&2
            exit 1
        fi
        ln -s "$src" "$WORK_DIR/$(basename "$src")"
    done

    BENCH_ENV=()
    if [[ -n "$EXT_TIMEOUT_BUFFER" ]]; then
        BENCH_ENV+=("BENCH_EXT_TIMEOUT_BUFFER=$EXT_TIMEOUT_BUFFER")
    fi

    env "${BENCH_ENV[@]}" SOLVER_DIR="$SOLVER_ROOT" \
    bash "$SCRIPT_DIR/bench_parallel.sh" \
        --data-dir "$WORK_DIR" \
        --timeout "$TIMEOUT" \
        --eps "$EPS" \
        --jobs "$JOBS" \
        --output "$BENCH_OUTPUT"
else
    BENCH_OUTPUT="$FROM_BENCH_OUTPUT"
    if [[ ! -f "$BENCH_OUTPUT" ]]; then
        echo "エラー: --from-bench-output が存在しない: $BENCH_OUTPUT" >&2
        exit 1
    fi
fi

DETAIL_FILE="$WORK_DIR/details.txt"
awk '
    /^=== 問題別詳細 ===$/ { in_detail = 1; next }
    /^=== カテゴリ別 問題名一覧 ===$/ { in_detail = 0 }
    in_detail { print }
' "$BENCH_OUTPUT" > "$DETAIL_FILE"

classify_line() {
    local line="$1"
    if [[ "$line" == *"TIMEOUT"* ]]; then
        if [[ "$line" == *"external_timeout="* ]]; then
            echo "external_timeout"
        else
            echo "internal_timeout"
        fi
        return
    fi
    if [[ "$line" == *"SUBOPTIMAL"* ]]; then
        echo "suboptimal"
        return
    fi
    if [[ "$line" == *"PASS"* ]]; then
        echo "optimal_or_pass"
        return
    fi
    if [[ "$line" == *"FAIL"* ]] || [[ "$line" == *"ERROR"* ]]; then
        echo "failure"
        return
    fi
    echo "unknown"
}

report_one() {
    local base="$1"
    local line
    line="$(grep -Ei "^[[:space:]]*${base}(\\.qps)?[[:space:]]" "$DETAIL_FILE" | head -n 1 || true)"
    if [[ -z "$line" ]]; then
        printf "%s\t%s\t%s\n" "$base" "not_found" "(detail line not found)" >> "$CLASS_TSV"
        printf "%-8s | %-17s | %s\n" "$base" "not_found" "(detail line not found)"
        return
    fi
    local cls
    cls="$(classify_line "$line")"
    printf "%s\t%s\t%s\n" "$base" "$cls" "$line" >> "$CLASS_TSV"
    printf "%-8s | %-17s | %s\n" "$base" "$cls" "$line"
}

>"$CLASS_TSV"
echo -e "problem\tclass\tdetail" >> "$CLASS_TSV"

{
    echo "=== lp_timeout_probe ==="
    echo "solver_root: $SOLVER_ROOT"
    echo "hard_dir:    $HARD_DIR"
    echo "timeout:     ${TIMEOUT}s"
    echo "eps:         $EPS"
    echo "jobs:        $JOBS"
    echo "bench_log:   $BENCH_OUTPUT"
    echo
    report_one "cont1"
    report_one "cont11"
    report_one "cont4"
    echo
    echo "summary:"
    awk -F'\t' 'NR>1 { cnt[$2]++ } END { for (k in cnt) printf "  %s: %d\n", k, cnt[k] }' "$CLASS_TSV" | sort
} | tee "$REPORT_OUTPUT"

echo
echo "[lp_timeout_probe] report: $REPORT_OUTPUT"
echo "[lp_timeout_probe] class_tsv: $CLASS_TSV"
