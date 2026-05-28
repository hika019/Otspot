#!/bin/bash
# CLAUDE.md L38: 2500 行以上 (非test) file はマイクロアーキテクチャ確認 + 責任分割/機能共通化/test 追加
#
# 使い方:
#   bash scripts/check_file_size.sh              # default threshold 2500 行
#   bash scripts/check_file_size.sh 800          # custom threshold (より厳しい監視)
#
# agent dispatch 前 / agent 作業冒頭 / reviewer 観点 / lead 定期 audit で使用。
# 範囲: workspace 全 member crate の src/ (.rs)。tests/ は除外。
# カウント: 非 test 行 (file 先頭〜最初の `#[cfg(test)]` or `mod tests` の直前まで)。

set -eu

THRESHOLD="${1:-2500}"
violations=""

for crate_src in src otspot-core/src otspot-io/src otspot-model/src otspot-dev/src; do
  [ -d "$crate_src" ] || continue
  while IFS= read -r f; do
    test_start=$(grep -n '^#\[cfg(test)\]\|^mod tests' "$f" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [ -n "$test_start" ]; then
      nontest=$((test_start - 1))
    else
      nontest=$(wc -l < "$f" | tr -d ' ')
    fi
    if [ "$nontest" -ge "$THRESHOLD" ]; then
      violations="${violations}${nontest} ${f}"$'\n'
    fi
  done < <(find "$crate_src" -name '*.rs' -type f)
done

if [ -z "$violations" ]; then
  echo "[check_file_size] OK: 全 production .rs file が ${THRESHOLD} 行 (非test) 未満"
  exit 0
fi

echo "[check_file_size] ${THRESHOLD}+ 行 (非test) file: CLAUDE.md L38 trigger 該当"
printf '%s' "$violations" | sort -rn
echo
echo "[check_file_size] 対処: micro-architecture 検討 (責任分割/機能共通化/test 追加)"
exit 1
