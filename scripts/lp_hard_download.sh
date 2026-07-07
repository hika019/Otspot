#!/bin/bash
# 数値困難 LP 問題セット + Mittelmann Large LP のダウンロード・変換スクリプト
#
# 用途: plato.asu.edu および Netlib から数値困難・大規模 LP 問題をダウンロードし、
#       QPS パーサー互換の MPS 形式 (.QPS) で data/lp_problems_hard/ に保存する。
#
# 取得カテゴリ:
#   1. FOME stochastic LP (数値的に困難):
#      fome13, fome21 — Forest Management (ill-conditioned, large stochastic LP)
#      (fome11, fome12 は lp_extra_download.sh でカバー済み)
#   2. PDS Patient Distribution System (Kennington 拡張):
#      pds-40, pds-50, pds-60, pds-70, pds-80, pds-90, pds-100
#      (pds-02/06/10/20 は netlib_lp_download.sh, pds-30 は lp_extra_download.sh でカバー済み)
#   3. NUG QAP LP 緩和 (極悪条件数):
#      nug08-3rd, nug20, nug30
#   4. Misc large LP (stormG2, watson, neos, ns, sgpf, cont):
#      stormG2_1000, watson_1, watson_2
#      neos, neos1, neos2, neos3
#      ns1687037, ns1688926
#      sgpf5y6
#      cont1, cont4, cont11
#   5. Rail set cover problems:
#      rail507, rail516, rail582, rail2586, rail4284
#   6. FCTP fixed-charge transportation:
#      n370a, n370b, n370c, n370d, n370e
#      n3700..n3709
#      ran4x64, ran6x43, ran8x32, ran10x26, ran12x21, ran14x18, ran16x16, ran17x17
#
# 使用方法:
#   bash scripts/lp_hard_download.sh [出力ディレクトリ]
#   デフォルト出力先: data/lp_problems_hard/
#
#   LP_HARD_ONLY="name1 name2" bash scripts/lp_hard_download.sh
#     指定した instance 名 (拡張子なし、例: "neos") のみ取得する。未設定/空なら全件
#     (デフォルトの local full run 挙動は変えない)。CI で全53件は不要な特定 test
#     のみが要求するデータを取りに行く用途 (例: test-heavy.yml の neos.QPS)。
#
# 依存:
#   - curl
#   - bunzip2
#   - emps (Netlib emps 形式デコーダー, /tmp/emps にコンパイル済みが必要)
#
# emps のコンパイル手順:
#   curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
#   cc -o /tmp/emps /tmp/emps.c

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$SOLVER_DIR/data/lp_problems_hard}"
PLATO_BASE="https://plato.asu.edu/ftp/lptestset"
EMPS="${EMPS_BIN:-/tmp/emps}"

# emps バイナリ確認
if [ ! -x "$EMPS" ]; then
    echo "ERROR: emps binary not found at $EMPS" >&2
    echo "Compile with: curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c && cc -o /tmp/emps /tmp/emps.c" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SUCCESS=0
FAIL=0
SKIP=0
FAIL_NAMES=()

# LP_HARD_ONLY (space区切り) が設定されていれば、その instance 名だけを対象にする。
declare -A ONLY_SET=()
if [ -n "${LP_HARD_ONLY:-}" ]; then
    for _n in $LP_HARD_ONLY; do
        ONLY_SET["$_n"]=1
    done
fi

# instance が対象かどうか (LP_HARD_ONLY 未設定なら常に true)
wanted() {
    local name="$1"
    [ ${#ONLY_SET[@]} -eq 0 ] && return 0
    [ -n "${ONLY_SET[$name]:-}" ]
}

echo "=== LP Hard Problem Download ===" >&2
echo "Output: $OUTPUT_DIR" >&2
if [ ${#ONLY_SET[@]} -gt 0 ]; then
    echo "Filter (LP_HARD_ONLY): ${!ONLY_SET[*]}" >&2
fi
echo "" >&2

# ----------------------------------------------------------------
# ヘルパー関数
# ----------------------------------------------------------------

# bz2 + emps 形式のダウンロード・変換 (MPC 圧縮されたもの)
# 引数: name url
download_bz2_emps() {
    local name="$1"
    local url="$2"
    local OUT="$OUTPUT_DIR/${name}.QPS"

    if ! wanted "$name"; then
        return 0
    fi

    if [ -f "$OUT" ]; then
        echo "EXISTS: $name" >&2
        SKIP=$((SKIP + 1))
        return 0
    fi

    local TMP_BZ2 TMP_RAW
    TMP_BZ2=$(mktemp)
    TMP_RAW=$(mktemp)

    if ! curl -L -s -f "$url" -o "$TMP_BZ2" 2>/dev/null; then
        echo "FAIL (download): $name  ($url)" >&2
        rm -f "$TMP_BZ2" "$TMP_RAW"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
        return 1
    fi

    if ! bunzip2 -c "$TMP_BZ2" > "$TMP_RAW" 2>/dev/null; then
        echo "FAIL (bunzip2): $name" >&2
        rm -f "$TMP_BZ2" "$TMP_RAW"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
        return 1
    fi

    if "$EMPS" "$TMP_RAW" > "$OUT" 2>/dev/null && [ -s "$OUT" ]; then
        echo "OK: $name -> $(basename "$OUT") ($(wc -l < "$OUT") lines)" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (emps): $name" >&2
        rm -f "$OUT"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
    fi

    rm -f "$TMP_BZ2" "$TMP_RAW"
}

# bz2 + plain MPS 形式のダウンロード (emps 不要)
# 引数: name url
download_bz2_mps() {
    local name="$1"
    local url="$2"
    local OUT="$OUTPUT_DIR/${name}.QPS"

    if ! wanted "$name"; then
        return 0
    fi

    if [ -f "$OUT" ]; then
        echo "EXISTS: $name" >&2
        SKIP=$((SKIP + 1))
        return 0
    fi

    local TMP_BZ2
    TMP_BZ2=$(mktemp)

    if ! curl -L -s -f "$url" -o "$TMP_BZ2" 2>/dev/null; then
        echo "FAIL (download): $name  ($url)" >&2
        rm -f "$TMP_BZ2"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
        return 1
    fi

    if bunzip2 -c "$TMP_BZ2" > "$OUT" 2>/dev/null && [ -s "$OUT" ]; then
        echo "OK: $name -> $(basename "$OUT") ($(wc -l < "$OUT") lines)" >&2
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL (bunzip2/empty): $name" >&2
        rm -f "$OUT"
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("$name")
    fi

    rm -f "$TMP_BZ2"
}

# NAME 行が空の場合にヘッダーを補完する
fix_empty_name_row() {
    local file="$1"
    local label="$2"
    if [ -f "$file" ]; then
        local first
        first=$(head -1 "$file")
        if [ "$first" = "NAME" ]; then
            local TMP
            TMP=$(mktemp)
            { printf "NAME          %s\n" "$label"; tail -n +2 "$file"; } > "$TMP"
            mv "$TMP" "$file"
            echo "  (NAME行補完: $label)" >&2
        fi
    fi
}

# ----------------------------------------------------------------
# 1. FOME stochastic LP (数値的に困難)
# ----------------------------------------------------------------
echo "=== 1. FOME stochastic LP (fome13, fome21) ===" >&2
# fome13: 24286 行。fome12 の倍スケール
download_bz2_emps "fome13" "${PLATO_BASE}/fome/fome13.bz2"
# fome21: 解列が異なる大規模変形
download_bz2_emps "fome21" "${PLATO_BASE}/fome/fome21.bz2"

# ----------------------------------------------------------------
# 2. PDS Patient Distribution System (大規模 Kennington 拡張)
# ----------------------------------------------------------------
echo "" >&2
echo "=== 2. PDS (pds-40..pds-100) ===" >&2
for size in 40 50 60 70 80 90 100; do
    download_bz2_emps "pds-${size}" "${PLATO_BASE}/pds/pds-${size}.bz2"
    fix_empty_name_row "$OUTPUT_DIR/pds-${size}.QPS" "PDS-${size}"
done

# ----------------------------------------------------------------
# 3. NUG QAP LP 緩和 (極悪条件数)
# ----------------------------------------------------------------
echo "" >&2
echo "=== 3. NUG QAP LP 緩和 (nug08-3rd, nug20, nug30) ===" >&2
# nug08-3rd: Nugent 8x8 QAP LP 緩和 (3rd formulation)
download_bz2_emps "nug08-3rd" "${PLATO_BASE}/nug/nug08-3rd.bz2"
# nug20: 14240 rows
download_bz2_emps "nug20"     "${PLATO_BASE}/nug/nug20.bz2"
# nug30: 52260 rows (最大規模)
download_bz2_emps "nug30"     "${PLATO_BASE}/nug/nug30.bz2"

# ----------------------------------------------------------------
# 4. Misc large LP (stormG2, watson, neos, ns, sgpf, cont)
# ----------------------------------------------------------------
echo "" >&2
echo "=== 4. Misc large LP ===" >&2

# stormG2_1000: 大規模 stochastic LP (1000 シナリオ)
download_bz2_emps "stormG2_1000" "${PLATO_BASE}/misc/stormG2_1000.bz2"

# watson_1, watson_2: Watson problems (transportation)
download_bz2_emps "watson_1" "${PLATO_BASE}/misc/watson_1.bz2"
download_bz2_emps "watson_2" "${PLATO_BASE}/misc/watson_2.bz2"

# neos, neos1, neos2, neos3: NEOS server problems (plain MPS 形式)
for n in neos neos1 neos2 neos3; do
    download_bz2_emps "$n" "${PLATO_BASE}/misc/${n}.bz2"
done

# ns1687037, ns1688926: NS problems
download_bz2_emps "ns1687037" "${PLATO_BASE}/misc/ns1687037.bz2"
download_bz2_emps "ns1688926" "${PLATO_BASE}/misc/ns1688926.bz2"

# sgpf5y6: set packing LP
download_bz2_emps "sgpf5y6" "${PLATO_BASE}/misc/sgpf5y6.bz2"

# cont1, cont4, cont11: continuous LP (数値的に困難)
for c in cont1 cont4 cont11; do
    download_bz2_emps "$c" "${PLATO_BASE}/misc/${c}.bz2"
done

# ----------------------------------------------------------------
# 5. Rail set cover problems (plain MPS 形式)
# ----------------------------------------------------------------
echo "" >&2
echo "=== 5. Rail set cover (rail507, rail516, rail582, rail2586, rail4284) ===" >&2
for r in rail507 rail516 rail582 rail2586 rail4284; do
    download_bz2_emps "$r" "${PLATO_BASE}/rail/${r}.bz2"
done

# ----------------------------------------------------------------
# 6. FCTP Fixed-charge transportation (plain MPS 形式)
# ----------------------------------------------------------------
echo "" >&2
echo "=== 6. FCTP fixed-charge transportation ===" >&2
# 中規模問題 (n370 系)
for p in n370a n370b n370c n370d n370e n3700 n3701 n3702 n3703 n3704 n3705 n3706 n3707 n3708 n3709; do
    download_bz2_mps "$p" "${PLATO_BASE}/fctp/${p}.mps.bz2"
done
# ran 系 (大規模)
for p in ran4x64 ran6x43 ran8x32 ran10x26 ran12x21 ran14x18 ran16x16 ran17x17; do
    download_bz2_mps "$p" "${PLATO_BASE}/fctp/${p}.mps.bz2"
done

# ----------------------------------------------------------------
# 完了サマリー
# ----------------------------------------------------------------
echo "" >&2
echo "=== 完了 ===" >&2
echo "SUCCESS: $SUCCESS, FAIL: $FAIL, SKIP: $SKIP" >&2
echo "Output dir: $OUTPUT_DIR" >&2

if [ "${#FAIL_NAMES[@]}" -gt 0 ]; then
    echo "" >&2
    echo "--- 失敗した問題 ---" >&2
    for n in "${FAIL_NAMES[@]}"; do
        echo "  FAIL: $n" >&2
    done
fi

echo "" >&2
echo "注意: 正解値 CSV は data/baseline_objectives/netlib_lp_hard.csv" >&2
echo "  ベンチ実行: SOLVER_DIR=. bash scripts/bench_parallel.sh \\" >&2
echo "    --data-dir data/lp_problems_hard \\" >&2
echo "    --timeout 1000 --eps 1e-6 --jobs 6 \\" >&2
echo "    --output /tmp/hard_bench.txt" >&2
