#!/bin/bash
# lead が merge 判定前に走らせる pre-merge gate.
# CLAUDE.md 「実施者とレビュアーは別エージェント」+ memory feedback_agent_self_report_unreliable_verify_independently 対応.
set -eo pipefail

echo "=== pre-merge audit ==="
echo

# 1. CI workflow/data bootstrap checks
bash -n scripts/ensure_emps.sh
cargo fmt --all -- --check

if grep -R "curl .*emps\\.c" .github/workflows scripts 2>/dev/null \
  | grep -v "scripts/ensure_emps.sh" \
  | grep -v "Compile with:" \
  | grep -vE '^[^:]+:[[:space:]]*#'; then
  echo "ERROR: emps.c download must go through scripts/ensure_emps.sh" >&2
  exit 1
fi

# 2. build + test + clippy + file size
cargo build --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --release --test-threads 6
# `cargo test --doc` runs doctests; it does not lint intra-doc links. The
# rustdoc gate below is a separate check and CI runs both (ci.yml `docs` job).
cargo test --doc
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
bash scripts/check_file_size.sh
# grep exits 1 on zero matches, which is the passing state here; `|| true`
# keeps `set -e -o pipefail` from aborting the audit exactly when it succeeds.
TODO_COUNT=$(grep -rEc 'TODO|FIXME|XXX|HACK' \
  otspot-core/src otspot-io/src otspot-model/src otspot-dev/src src 2>/dev/null \
  | awk -F: '{n += $NF} END {print n + 0}' || true)
if [ "$TODO_COUNT" -gt 0 ]; then
  echo "ERROR: TODO/FIXME/XXX/HACK count $TODO_COUNT > 0" >&2
  exit 1
fi
python3 tests/test_check_data_coverage.py
python3 tests/test_check_test_data_requirements.py
python3 tests/test_iso_25010_quality_matrix.py

# 1b. merge gate regression scan
echo
echo "=== merge gate regression scan ==="
if git diff main..HEAD --unified=0 -- '*.rs' | grep -E '^\+.*#\[ignore' >/tmp/pre_merge_added_ignore.txt; then
  cat /tmp/pre_merge_added_ignore.txt >&2
  echo "::error::新規 #[ignore] は merge gate で禁止。heavy 隔離が必要なら test-heavy.yml 側の明示 gate と一緒に追加すること" >&2
  exit 1
fi
if git diff main..HEAD --unified=0 -- .github scripts 'otspot-dev/src/bin/*.rs' | grep -E '^\+.*PASS\[no_ref\]' >/tmp/pre_merge_pass_noref.txt; then
  cat /tmp/pre_merge_pass_noref.txt >&2
  echo "::error::PASS[no_ref] は偽PASS経路。CHECKED[no_ref] 等の非PASS分類を使うこと" >&2
  exit 1
fi
if git diff main..HEAD --unified=0 -- scripts | grep -E '^\+[[:space:]]*echo "  .* TIMEOUT \(external_timeout=|^\+[[:space:]]*echo "    TIMEOUT: 1"' >/tmp/pre_merge_external_timeout.txt; then
  cat /tmp/pre_merge_external_timeout.txt >&2
  echo "::error::external_timeout を通常 TIMEOUT に混入させてはいけない。EXTERNAL_TIMEOUT として集計し、1件以上で失敗させること" >&2
  exit 1
fi

# 2. commit 情報
echo
echo "=== branch diff vs main ==="
git log main..HEAD --pretty='%h %s'
git diff --stat main..HEAD | tail -3

# 4. 公開 API diff (cargo-public-api installed 前提)
echo
echo "=== public API diff ==="
if command -v cargo-public-api >/dev/null 2>&1; then
  cargo public-api diff main 2>&1 | head -30 || true
else
  echo "(cargo-public-api 未 install、CI で確認)"
fi

# 5. コメント品質 (CLAUDE.md L45-46)
# diff scope ではなく full-scan を使用: gate を後付けする以前の commit に
# 違反が残存しうるため (#203 設置時点で複数 file が threshold 超過、PR 段階
# で trim 議論)。main に違反が確定混入した場合は ALLOWLIST 追加 or
# threshold 調整で復帰、本 section の scope (full-scan) は保つ。
echo
echo "=== comment quality ==="
bash scripts/lib/check_memo_grep.sh
bash scripts/check_comment_block_size.sh
bash scripts/check_comment_ratio.sh

# 6. magic 検出 (diff scope のみ、memory feedback_review_magic_detection)
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
