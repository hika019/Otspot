#!/bin/zsh
# solver_bench.sh — ベンチマークバイナリのラッパー
# --release ビルドを強制する。`--eps` は呼出側 (bench_parallel.sh) から指定する。
# qps_benchmark と bench_qplib の両方に対応。
#
# 使い方:
#   ./scripts/solver_bench.sh <bin> <data_dir> [追加オプション]
#
#   <bin>      : qps_benchmark または bench_qplib
#   <data_dir> : ベンチ対象ディレクトリ
#   追加オプション: --timeout N, --features parallel 等
#
# SOLVER_DIR 環境変数でソルバーリポジトリを指定可能。
# デフォルト: カレントディレクトリ。

set -euo pipefail

# bench_parallel.sh 経由でのみ実行可能（直接実行禁止）
if [[ "${_BENCH_PARALLEL_CALLER:-}" != "1" ]]; then
  echo "[solver_bench.sh] エラー: 直接実行禁止。bench_parallel.sh 経由で実行せよ。" >&2
  echo "[solver_bench.sh] 使い方: bash scripts/bench_parallel.sh --data-dir DIR --solver SOLVER --timeout SEC --output FILE --jobs N" >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "使い方: $0 <bin> <data_dir> [追加オプション]" >&2
  echo "  <bin>: qps_benchmark または bench_qplib" >&2
  exit 1
fi

bin="$1"; shift
data_dir="$1"; shift

solver_dir="${SOLVER_DIR:-$(pwd)}"
cd "$solver_dir"

echo "[solver_bench.sh] cwd: $(pwd)"
echo "[solver_bench.sh] bin: $bin  data_dir: $data_dir"
echo "[solver_bench.sh] solver_commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "[solver_bench.sh] solver_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "[solver_bench.sh] timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# --solver フラグ必須チェック（暗黙のデフォルトモード禁止）
has_solver=false
for arg in "$@"; do
  if [[ "$arg" == "--solver" ]]; then
    has_solver=true
    break
  fi
done
if [[ "$has_solver" == false ]]; then
  echo "[solver_bench.sh] エラー: --solver フラグが指定されていない。" >&2
  echo "[solver_bench.sh] 暗黙のデフォルトモード禁止。--solver concurrent|ipm|ippmm_new を明示せよ。" >&2
  exit 1
fi

# --features と --solver を $@ から分離（cargo build 用）
EXTRA_FEATURES=""
SOLVER_ARG=""
BINARY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --features)
      EXTRA_FEATURES="$2"
      shift 2
      ;;
    --solver)
      SOLVER_ARG="$2"
      BINARY_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      BINARY_ARGS+=("$1")
      shift
      ;;
  esac
done

BUILD_FEATURES="parallel"
if [[ -n "$EXTRA_FEATURES" ]]; then
  BUILD_FEATURES="parallel,$EXTRA_FEATURES"
fi
if [[ ! -f "target/release/$bin" ]] || [[ -n "$EXTRA_FEATURES" ]]; then
  echo "[solver_bench.sh] ビルドを開始... (--features $BUILD_FEATURES)"
  cargo build --release --features "$BUILD_FEATURES" 2>&1
fi

echo "[solver_bench.sh] ./target/release/$bin $data_dir ${BINARY_ARGS[*]}"

# T5修正: 外部タイムアウト安全網（ソルバー内部timeout未設定時の暴走防止）
# macOS: gtimeout（brew install coreutils）、Linux: timeout
TIMEOUT_CMD=$(command -v gtimeout 2>/dev/null || command -v timeout 2>/dev/null || echo "")
if [[ -n "$TIMEOUT_CMD" ]]; then
  echo "[solver_bench.sh] 外部timeout: ${TIMEOUT_CMD} ${EXTERNAL_TIMEOUT:-120}s"
  exec "$TIMEOUT_CMD" "${EXTERNAL_TIMEOUT:-120}" "./target/release/$bin" "$data_dir" "${BINARY_ARGS[@]}"
else
  echo "[solver_bench.sh] 外部timeout: 利用不可（gtimeout/timeout コマンドが見つからない）"
  exec "./target/release/$bin" "$data_dir" "${BINARY_ARGS[@]}"
fi
