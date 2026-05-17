#!/bin/bash
# lp_canary_setup.sh — canary subset (28 問) のセットアップ
#
# 役割: scripts/run_lp_bench.sh --suite canary が要求する
#   data/lp_problems_canary/ ディレクトリを構築する。
#
#   構築方式: 既存の data/lp_problems/ および data/lp_problems_infeas/ の
#   .QPS ファイルへの相対 symlink を貼る (重複ダウンロード回避)。
#
#   依存先のデータが未取得なら、対応する netlib_lp_download.sh /
#   netlib_lp_infeas_download.sh を先に呼ぶ。
#
# 引数:
#   $1 — canary ディレクトリへの絶対 / 相対パス
#        (例: $SOLVER_ROOT/data/lp_problems_canary)
#
# 選定方針 (詳細: docs/canary_suite.md):
#   - 軽 sanity (16 問): afiro, sc50a/b, sc105/205, adlittle, blend, kb2,
#     share1b/2b, scagr7, recipe, stocfor1, boeing2, brandy, agg
#   - 中規模 (4 問): scfxm1, etamacro, capri, wood1p
#   - 構造的 bug class (5 問): perold (postsolve y), d6cube/cycle (退化),
#     greenbea (#14 dual-only), pds-10 (1000s class 代表)
#   - 等式・不等式 cold-start (2 問): klein1, klein2 (infeasible)
#
# 計 27 問 (28 行目はヘッダ)。timeout=60s, eps=1e-6 で合計 5 分以内に収まる
# 設計 (CLAUDE.md: PR loop 用 fallback)。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "使い方: $0 <canary_dir>" >&2
  echo "  例: $0 \"$SOLVER_ROOT/data/lp_problems_canary\"" >&2
  exit 1
fi

CANARY_DIR="$1"

# --- 依存元ディレクトリ ---
STD_DIR="$SOLVER_ROOT/data/lp_problems"
INFEAS_DIR="$SOLVER_ROOT/data/lp_problems_infeas"

# --- 依存元データが揃っているか確認、無ければダウンロード ---
ensure_emps() {
  if [ ! -x "/tmp/emps" ]; then
    curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
    cc -o /tmp/emps /tmp/emps.c
  fi
}

ensure_std() {
  if [[ ! -d "$STD_DIR" ]] || [[ -z "$(find "$STD_DIR" -maxdepth 1 -iname '*.qps' -print -quit 2>/dev/null)" ]]; then
    echo "[lp_canary_setup] $STD_DIR が空のため netlib_lp_download.sh を呼ぶ" >&2
    ensure_emps
    EMPS_BIN="/tmp/emps" bash "$SCRIPT_DIR/netlib_lp_download.sh" "$STD_DIR"
  fi
}

ensure_infeas() {
  if [[ ! -d "$INFEAS_DIR" ]] || [[ -z "$(find "$INFEAS_DIR" -maxdepth 1 -iname '*.qps' -print -quit 2>/dev/null)" ]]; then
    echo "[lp_canary_setup] $INFEAS_DIR が空のため netlib_lp_infeas_download.sh を呼ぶ" >&2
    ensure_emps
    EMPS_BIN="/tmp/emps" bash "$SCRIPT_DIR/netlib_lp_infeas_download.sh" "$INFEAS_DIR"
  fi
}

ensure_std
ensure_infeas

# --- canary ディレクトリ初期化 ---
mkdir -p "$CANARY_DIR"

# 既存 symlink を初期化 (依存元の構成が変わった場合の追従)
find "$CANARY_DIR" -maxdepth 1 -type l -delete

# 選定問題 (CLAUDE.md: マジックを散らさず 1 箇所に集約)
STD_PROBLEMS=(
  # 軽 sanity (16)
  afiro sc50a sc50b sc105 sc205 adlittle blend kb2
  share1b share2b scagr7 recipe stocfor1 boeing2 brandy agg
  # 中規模 (4)
  scfxm1 etamacro capri wood1p
  # 構造的 bug class (5)
  perold d6cube cycle greenbea pds-10
)
INFEAS_PROBLEMS=(
  klein1 klein2
)

link_one() {
  local src_rel="$1"  # e.g. ../lp_problems/afiro.QPS
  local name="$2"     # e.g. afiro.QPS
  if [[ ! -e "$CANARY_DIR/$src_rel" ]]; then
    echo "[lp_canary_setup] WARN: $CANARY_DIR/$src_rel が解決できない (依存元未取得?)" >&2
    return 1
  fi
  ln -sf "$src_rel" "$CANARY_DIR/$name"
}

for p in "${STD_PROBLEMS[@]}"; do
  link_one "../lp_problems/${p}.QPS" "${p}.QPS"
done
for p in "${INFEAS_PROBLEMS[@]}"; do
  link_one "../lp_problems_infeas/${p}.QPS" "${p}.QPS"
done

# --- 結果検証 ---
COUNT=$(find "$CANARY_DIR" -maxdepth 1 -iname '*.qps' 2>/dev/null | wc -l | tr -d ' ')
EXPECTED=$(( ${#STD_PROBLEMS[@]} + ${#INFEAS_PROBLEMS[@]} ))
echo "[lp_canary_setup] canary 構築完了: ${COUNT}/${EXPECTED} 問 (${CANARY_DIR})" >&2

if [[ "$COUNT" -ne "$EXPECTED" ]]; then
  echo "[lp_canary_setup] WARN: 期待 ${EXPECTED} 問だが ${COUNT} 問しかリンクされなかった" >&2
  exit 1
fi
