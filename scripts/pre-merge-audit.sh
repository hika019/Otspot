#!/bin/bash
# lead が merge 判定前に走らせる pre-merge gate.
# CLAUDE.md 「実施者とレビュアーは別エージェント」+ memory feedback_agent_self_report_unreliable_verify_independently 対応.
set -eo pipefail

echo "=== pre-merge audit ==="
echo

# 1. build + test + clippy + file size
cargo build --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --release --test-threads 3
bash scripts/check_file_size.sh
python3 tests/test_check_data_coverage.py

# 2. commit 情報
echo
echo "=== branch diff vs main ==="
git log main..HEAD --pretty='%h %s'
git diff --stat main..HEAD | tail -3

# 3. 公開 API diff (cargo-public-api installed 前提)
echo
echo "=== public API diff ==="
if command -v cargo-public-api >/dev/null 2>&1; then
  cargo public-api diff main 2>&1 | head -30 || true
else
  echo "(cargo-public-api 未 install、CI で確認)"
fi

# 4. コメント品質 (CLAUDE.md L45-46)
# diff scope ではなく full-scan を使用: gate を後付けする以前の commit に
# 違反が残存しうるため (#203 設置時点で複数 file が threshold 超過、PR 段階
# で trim 議論)。main に違反が確定混入した場合は ALLOWLIST 追加 or
# threshold 調整で復帰、本 section の scope (full-scan) は保つ。
echo
echo "=== comment quality ==="
bash scripts/lib/check_memo_grep.sh
bash scripts/check_comment_block_size.sh
bash scripts/check_comment_ratio.sh

# 5. magic 検出 (diff scope のみ、memory feedback_review_magic_detection)
echo
echo "=== magic number scan (diff のみ) ==="
echo "--- 新規 const without /// docstring ---"
git diff main..HEAD --unified=2 -- '*.rs' | awk '
  /^\+(pub(\([^)]+\))?[[:space:]]+)?(const|static) [A-Z_]+:/ {
    if (prev !~ /^[ +]\/\/\//) print "(new const w/o doc) " $0
  }
  { prev = $0 }
' | head -10
echo "--- 新規 inline numeric literal (production source 内、2桁以上) ---"
git diff main..HEAD --unified=0 -- 'otspot-core/src/*.rs' 'otspot-io/src/*.rs' 'otspot-model/src/*.rs' | \
  grep -E '^\+[^+]*\b[0-9]{2,}(_[0-9]+)*(\.[0-9]+)?([eE][+-]?[0-9]+)?\b' | \
  grep -vE 'cfg\(test\)|//|test_' | head -20 || true

echo
echo "=== audit complete ==="
