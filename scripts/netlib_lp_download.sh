#!/bin/bash
# Netlib LP問題セットのダウンロード・変換スクリプト
#
# 用途: Netlibの圧縮emps形式LPファイルをダウンロードし、
#       QPSパーサー互換のMPS形式（拡張子.QPS）に変換する。
#
# 使用方法:
#   bash scripts/netlib_lp_download.sh [出力ディレクトリ]
#   デフォルト出力先: data/lp_problems/
#
# 依存:
#   - curl (ダウンロード)
#   - emps (Netlib emps形式デコーダー, /tmp/emps にコンパイル済みが必要)
#
# empsのコンパイル手順:
#   curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
#   cc -o /tmp/emps /tmp/emps.c

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$SOLVER_DIR/data/lp_problems}"
NETLIB_BASE="https://www.netlib.org/lp/data"
EMPS="${EMPS_BIN:-/tmp/emps}"

# empsバイナリ確認
if [ ! -x "$EMPS" ]; then
    echo "ERROR: emps binary not found at $EMPS"
    echo "Compile with: curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c && cc -o /tmp/emps /tmp/emps.c"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 非LPファイル（ドキュメント・ツール）をスキップ
SKIP_FILES="readme changes ascii minos"

SUCCESS=0
FAIL=0
SKIP=0

echo "=== Netlib LP Download ===" >&2
echo "Output: $OUTPUT_DIR" >&2
echo "" >&2

# ファイル一覧取得
FILES=$(curl -s "$NETLIB_BASE/" | grep -oE 'href="[a-z0-9_-]+"' | sed 's/href="//;s/"//' | grep -v '\.')

for f in $FILES; do
    # ドキュメントファイルをスキップ
    if echo "$SKIP_FILES" | grep -qw "$f"; then
        echo "SKIP (non-LP): $f" >&2
        SKIP=$((SKIP + 1))
        continue
    fi

    OUT_FILE="$OUTPUT_DIR/${f}.QPS"

    # 既存ファイルはスキップ
    if [ -f "$OUT_FILE" ]; then
        echo "EXISTS: $f" >&2
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    # ダウンロード
    TMP_FILE=$(mktemp)
    if ! curl -s -f "$NETLIB_BASE/$f" -o "$TMP_FILE" 2>/dev/null; then
        echo "FAIL (download): $f" >&2
        rm -f "$TMP_FILE"
        FAIL=$((FAIL + 1))
        continue
    fi

    # emps展開
    if "$EMPS" "$TMP_FILE" > "$OUT_FILE" 2>/dev/null; then
        echo "OK: $f -> $(basename "$OUT_FILE")" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (emps): $f" >&2
        rm -f "$OUT_FILE"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$TMP_FILE"
done

echo "" >&2
echo "=== 完了 ===" >&2
echo "SUCCESS: $SUCCESS, FAIL: $FAIL, SKIP: $SKIP" >&2
echo "Output dir: $OUTPUT_DIR" >&2
