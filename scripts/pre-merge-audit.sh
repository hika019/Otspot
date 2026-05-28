#!/bin/bash
# lead が merge 判定前に走らせる pre-merge gate.
# CLAUDE.md 「実施者とレビュアーは別エージェント」+ memory feedback_agent_self_report_unreliable_verify_independently 対応.
set -e

echo "=== pre-merge audit ==="
echo

# 1. build + test + clippy + file size
cargo build --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --release --lib --test-threads 3
bash scripts/check_file_size.sh

# 2. commit 情報
echo
echo "=== branch diff vs main ==="
git log main..HEAD --pretty='%h %s'
git diff --stat main..HEAD | tail -3

# 3. 公開 API diff (cargo-public-api installed 前提)
echo
echo "=== public API diff ==="
if command -v cargo-public-api >/dev/null 2>&1; then
  cargo public-api diff main..HEAD 2>&1 | head -30 || true
else
  echo "(cargo-public-api 未 install、CI で確認)"
fi

# 4. TODO/FIXME 増分
echo
echo "=== TODO/FIXME 増分 ==="
git diff main..HEAD | grep -E '^\+.*TODO|^\+.*FIXME|^\+.*XXX|^\+.*HACK' | head -10 || echo "(増分なし)"

# 5. magic 検出 (diff scope のみ、memory feedback_review_magic_detection)
echo
echo "=== magic number scan (diff のみ) ==="
echo "--- 新規 const without /// docstring ---"
git diff main..HEAD --unified=2 -- '*.rs' | awk '
  /^\+(pub )?(const|static) [A-Z_]+:/ {
    if (prev !~ /^\+\/\/\//) print "(new const w/o doc) " $0
  }
  { prev = $0 }
' | head -10
echo "--- 新規 inline numeric literal (production source 内) ---"
git diff main..HEAD --unified=0 -- 'otspot-core/src/*.rs' 'otspot-io/src/*.rs' 'otspot-model/src/*.rs' | \
  grep -E '^\+[^+]*\b[0-9]+(_[0-9]+)*(\.[0-9]+)?([eE][+-]?[0-9]+)?\b' | \
  grep -vE 'cfg\(test\)|//|test_' | head -20

echo
echo "=== audit complete ==="
