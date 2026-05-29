#!/bin/bash
# Netlib LP infeasible問題セットのダウンロード・変換スクリプト
#
# 用途: Netlibの実行不可能LP問題 (https://www.netlib.org/lp/infeas/) を
#       ダウンロードし、QPS形式で data/lp_problems_infeas/ に保存する。
#
# 使用方法:
#   bash scripts/netlib_lp_infeas_download.sh [出力ディレクトリ]
#   デフォルト出力先: data/lp_problems_infeas/
#
# 依存:
#   - curl (ダウンロード)
#   - emps (Netlib emps形式デコーダー, /tmp/emps にコンパイル済みが必要)
#
# empsのコンパイル手順:
#   curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
#   cc -o /tmp/emps /tmp/emps.c
#
# 参考: https://www.netlib.org/lp/infeas/readme
#   Gay, D.M. (1993). "Infeasible linear programming test problems."
#   これらの問題は全て実行不可能 (INFEASIBLE) であることが既知。
#   ベンチマーク用途: ソルバーが正しく Infeasible を検出できるか検証する。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$SOLVER_DIR/data/lp_problems_infeas}"
NETLIB_INFEAS_BASE="https://www.netlib.org/lp/infeas"
EMPS="${EMPS_BIN:-/tmp/emps}"

# empsバイナリ確認
if [ ! -x "$EMPS" ]; then
    echo "ERROR: emps binary not found at $EMPS"
    echo "Compile with: curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c && cc -o /tmp/emps /tmp/emps.c"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Netlib infeasible問題一覧 (29問)
INFEAS_PROBLEMS=(
    bgdbg1 bgetam bgindy bgprtr
    box1 ceria3d chemcom cplex1 cplex2
    ex72a ex73a forest6 galenet
    gosh gran greenbea
    itest2 itest6
    klein1 klein2 klein3
    mondou2 pang pilot4i
    qual reactor refinery vol1 woodinfe
)

SUCCESS=0
FAIL=0
SKIP=0

echo "=== Netlib LP Infeasible Set Download ===" >&2
echo "Output: $OUTPUT_DIR" >&2
echo "" >&2

download_infeas() {
    local prob="$1"
    local OUT="$OUTPUT_DIR/${prob}.QPS"

    if [ -f "$OUT" ]; then
        echo "EXISTS: $prob" >&2
        SKIP=$((SKIP + 1))
        return 0
    fi

    local TMP
    TMP=$(mktemp)

    # Netlibはemps圧縮形式 (curl -L でリダイレクト追跡)
    if ! curl -L -s "${NETLIB_INFEAS_BASE}/${prob}" -o "$TMP" 2>/dev/null; then
        echo "FAIL (download): $prob" >&2
        rm -f "$TMP"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if "$EMPS" "$TMP" > "$OUT" 2>/dev/null && [ -s "$OUT" ]; then
        echo "OK: $prob -> $(basename "$OUT")" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (emps): $prob" >&2
        rm -f "$OUT"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$TMP"
}

for prob in "${INFEAS_PROBLEMS[@]}"; do
    download_infeas "$prob"
done

echo "" >&2
echo "=== 完了 ===" >&2
echo "SUCCESS: $SUCCESS, FAIL: $FAIL, SKIP: $SKIP" >&2
echo "Output dir: $OUTPUT_DIR" >&2
echo "" >&2
echo "注意: 正解値CSV は data/baseline_objectives/netlib_lp_infeas.csv" >&2
echo "  全問題の expected status = INFEASIBLE" >&2
echo "  ベンチ実行: SOLVER_DIR=. bash scripts/bench_parallel.sh \\" >&2
echo "    --data-dir data/lp_problems_infeas \\" >&2
echo "    --timeout 300 --eps 1e-6 --jobs 8 \\" >&2
echo "    --output /tmp/infeas_bench.txt" >&2
