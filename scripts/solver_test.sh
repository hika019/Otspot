#!/bin/zsh
# solver_test.sh — cargo nextest run のラッパー
# --release --features parallel を強制注入（debugビルド禁止）
# 使い方: ./scripts/solver_test.sh [nextest追加オプション]
#
# SOLVER_DIR 環境変数または第1引数がディレクトリなら作業ディレクトリとして使用。
# デフォルト: カレントディレクトリ。

set -euo pipefail

if [[ -n "${SOLVER_DIR:-}" ]]; then
  target_dir="$SOLVER_DIR"
elif [[ $# -gt 0 && -d "$1" ]]; then
  target_dir="$1"
  shift
else
  target_dir="$(pwd)"
fi

cd "$target_dir"

echo "[solver_test.sh] cwd: $(pwd)"
echo "[solver_test.sh] cargo nextest run --release --features parallel $*"
exec cargo nextest run --release --features parallel "$@"
