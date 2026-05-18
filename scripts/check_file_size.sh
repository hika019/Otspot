#!/bin/bash
# CLAUDE.md L27 trigger: 800 行以上 file は マイクロアーキテクチャ確認 + 統合/分割 + test 追加
#
# 使い方:
#   bash scripts/check_file_size.sh              # default threshold 800 行
#   bash scripts/check_file_size.sh 1000         # custom threshold
#
# agent dispatch 前 / agent 作業冒頭 / reviewer 観点 / lead 定期 audit で使用。
# 範囲: src/ + tests/ の .rs file。

set -eu

THRESHOLD="${1:-800}"

result=$(find src/ tests/ -name "*.rs" -type f -exec wc -l {} \; 2>/dev/null \
  | awk -v t="$THRESHOLD" '$1 >= t' \
  | sort -rn)

if [[ -z "$result" ]]; then
  echo "[check_file_size] OK: 全 .rs file が ${THRESHOLD} 行未満"
  exit 0
fi

count=$(echo "$result" | wc -l | tr -d ' ')
echo "[check_file_size] ${THRESHOLD}+ 行 file: ${count} 件 (CLAUDE.md L27 trigger 該当)"
echo "$result"
echo ""
echo "[check_file_size] 対処:"
echo "  - 新 fn 追加なら別 file 切り出し検討"
echo "  - 既存 file の責任を確認、統合/分割を検討"
echo "  - 分割後は内部 helper に直接 unit test 追加 (memory feedback_micro_architecture_for_testability)"
exit 1
