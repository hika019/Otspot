#!/bin/bash
# Mittelmann Large LP / 数値困難問題セットのダウンロードスクリプト
#
# 用途: plato.asu.edu から大規模LP問題をダウンロードし、
#       QPS形式で data/lp_problems_extra/ に保存する。
#
# 追加問題カテゴリ:
#   1. 数値的に悪質な問題 (ill-conditioned):
#      - fome11, fome12: FOME (Forest Management) stochastic LP
#        ill-conditioned due to long planning horizons and column generation structure
#   2. Mittelmann Large LP (pds series 拡張, QAP LP 緩和):
#      - pds-30: Patient Distribution System (50k rows) - Kennington pds-20 より大規模
#      - qap15:  QAP-15 の LP 緩和 (6331 rows, 22275 vars)
#
# 使用方法:
#   bash scripts/lp_extra_download.sh [出力ディレクトリ]
#   デフォルト出力先: data/lp_problems_extra/
#
# 依存:
#   - curl (ダウンロード)
#   - bunzip2 (bz2 展開)
#   - emps (Netlib emps形式デコーダー, /tmp/emps にコンパイル済みが必要)
#
# empsのコンパイル手順:
#   curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
#   cc -o /tmp/emps /tmp/emps.c

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$SOLVER_DIR/data/lp_problems_extra}"
PLATO_BASE="https://plato.asu.edu/ftp/lptestset"
EMPS="${EMPS_BIN:-/tmp/emps}"

# empsバイナリ確認
if [ ! -x "$EMPS" ]; then
    echo "ERROR: emps binary not found at $EMPS"
    echo "Compile with: curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c && cc -o /tmp/emps /tmp/emps.c"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SUCCESS=0
FAIL=0
SKIP=0

echo "=== LP Extra Problem Download ===" >&2
echo "Output: $OUTPUT_DIR" >&2
echo "" >&2

# bz2 + emps形式のダウンロード・変換関数 (fome系, pds系)
download_bz2_emps() {
    local name="$1"     # 出力ファイル名 (拡張子なし)
    local url="$2"      # ダウンロードURL
    local OUT="$OUTPUT_DIR/${name}.QPS"

    if [ -f "$OUT" ]; then
        echo "EXISTS: $name" >&2
        SKIP=$((SKIP + 1))
        return 0
    fi

    local TMP_BZ2 TMP_RAW
    TMP_BZ2=$(mktemp)
    TMP_RAW=$(mktemp)

    if ! curl -L -s "$url" -o "$TMP_BZ2" 2>/dev/null; then
        echo "FAIL (download): $name" >&2
        rm -f "$TMP_BZ2" "$TMP_RAW"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if ! bunzip2 -c "$TMP_BZ2" > "$TMP_RAW" 2>/dev/null; then
        echo "FAIL (bunzip2): $name" >&2
        rm -f "$TMP_BZ2" "$TMP_RAW"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if "$EMPS" "$TMP_RAW" > "$OUT" 2>/dev/null && [ -s "$OUT" ]; then
        echo "OK: $name -> $(basename "$OUT") ($(wc -l < "$OUT") lines)" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (emps): $name" >&2
        rm -f "$OUT"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$TMP_BZ2" "$TMP_RAW"
}

# bz2 + plain MPS形式のダウンロード (qap15はそのままMPS)
download_bz2_mps() {
    local name="$1"    # 出力ファイル名 (拡張子なし)
    local url="$2"     # ダウンロードURL
    local OUT="$OUTPUT_DIR/${name}.QPS"

    if [ -f "$OUT" ]; then
        echo "EXISTS: $name" >&2
        SKIP=$((SKIP + 1))
        return 0
    fi

    local TMP_BZ2
    TMP_BZ2=$(mktemp)

    if ! curl -L -s "$url" -o "$TMP_BZ2" 2>/dev/null; then
        echo "FAIL (download): $name" >&2
        rm -f "$TMP_BZ2"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if bunzip2 -c "$TMP_BZ2" > "$OUT" 2>/dev/null && [ -s "$OUT" ]; then
        echo "OK: $name -> $(basename "$OUT") ($(wc -l < "$OUT") lines)" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (bunzip2): $name" >&2
        rm -f "$OUT"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$TMP_BZ2"
}

echo "=== 数値困難問題 (FOME stochastic LP) ===" >&2
# FOME: Forest Management LP - ill-conditioned stochastic LP
# fome11: 12143 rows, fome12: 24285 rows
download_bz2_emps "fome11" "${PLATO_BASE}/fome/fome11.bz2"
download_bz2_emps "fome12" "${PLATO_BASE}/fome/fome12.bz2"

echo "" >&2
echo "=== Mittelmann Large LP ===" >&2
# pds-30: Patient Distribution System (50k rows, Kennington pds-20 の拡張)
download_bz2_emps "pds-30" "${PLATO_BASE}/pds/pds-30.bz2"

# qap15: LP relaxation of QAP-15 (6331 rows, 22275 vars, plain MPS format)
download_bz2_mps "qap15" "${PLATO_BASE}/qap15.mps.bz2"

# pds-30 の NAME行が空の場合は修正
if [ -f "$OUTPUT_DIR/pds-30.QPS" ]; then
    FIRST_LINE=$(head -1 "$OUTPUT_DIR/pds-30.QPS")
    if [ "$FIRST_LINE" = "NAME" ]; then
        TMP=$(mktemp)
        { echo "NAME          PDS-30"; tail -n +2 "$OUTPUT_DIR/pds-30.QPS"; } > "$TMP"
        mv "$TMP" "$OUTPUT_DIR/pds-30.QPS"
        echo "  (pds-30: NAME行を修正)" >&2
    fi
fi

echo "" >&2
echo "=== 完了 ===" >&2
echo "SUCCESS: $SUCCESS, FAIL: $FAIL, SKIP: $SKIP" >&2
echo "Output dir: $OUTPUT_DIR" >&2
echo "" >&2
echo "注意: 正解値CSV は data/baseline_objectives/netlib_lp_extra.csv" >&2
echo "  初回ベンチ後に optimal_obj を更新すること (現在は no_ref)" >&2
echo "  ベンチ実行: SOLVER_DIR=. bash scripts/bench_parallel.sh \\" >&2
echo "    --data-dir data/lp_problems_extra \\" >&2
echo "    --timeout 1000 --eps 1e-6 --jobs 4 \\" >&2
echo "    --output /tmp/extra_bench.txt" >&2
