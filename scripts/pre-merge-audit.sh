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
# --features parallel: 目録ゲート (下の test inventory gate) と release ビルドを
# 共有し、concurrent.rs の並列テストを目録に含めるため。この feature は非-parallel
# の上位集合 (テストは減らない) なので full-suite を弱めない。
cargo nextest run --release --features parallel --test-threads 6
# heavy 専用 data (lp_problems_hard/neos.QPS、osqp_bench_illscaled 126 件) の充足
# チェック。CI test-heavy.yml では実行前の明示 step だがローカル worktree では
# 未配置になりやすく、無いと heavy 中盤のテスト panic ('data missing' 設計上
# silent skip しない) で失敗理由が分かりにくい。CI と同じ実体を先に走らせ
# 欠落パスと補完手順 (test-heavy.yml の data-prep) を読みやすく示す。あえて
# full-suite (default profile) の後: default は #[ignore] を走らせず heavy 専用
# data を要しないため、ここが唯一の必須地点。実体の単体テストは下のゲート列の
# tests/test_check_test_data_requirements.py。
python3 scripts/check_test_data_requirements.py
# heavy: AGENTS.md の `--profile heavy --run-ignored all` に --features parallel を
# 足す。full-suite と同じ feature 集合で release バイナリを共有し (上位集合なので
# テストは減らない)、concurrent.rs の並列テストも heavy 側で実走させるため。
# 長時間ステージ: #[ignore] 済み tier-2 sentinel を全実走する。
cargo nextest run --release --features parallel --profile heavy --run-ignored all --test-threads 6
# Doctest bodies get no -C debug-assertions/overflow-checks/opt-level from
# rustdoc, so they run with debug_assertions=true in both modes; --release only
# changes the linked lib build (release lib, debug-assertions off; AGENTS.md).
# Runs after the suite for partial reuse (workspace crates still rebuild on the
# feature difference; cold without the suite is 82 units). Intra-doc links are
# covered by the separate rustdoc gate below.
cargo test --doc --release
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
bash scripts/check_file_size.sh
# grep exits 1 on zero matches, which is the passing state here; `|| true`
# keeps `set -e -o pipefail` from aborting the audit exactly when it succeeds.
TODO_COUNT=$(grep -rEc 'TODO|FIXME|XXX|HACK' \
  otspot-core/src otspot-num/src otspot-io/src otspot-model/src otspot-dev/src src 2>/dev/null \
  | awk -F: '{n += $NF} END {print n + 0}' || true)
if [ "$TODO_COUNT" -gt 0 ]; then
  echo "ERROR: TODO/FIXME/XXX/HACK count $TODO_COUNT > 0" >&2
  exit 1
fi
python3 tests/test_check_data_coverage.py
python3 tests/test_check_test_data_requirements.py
python3 tests/test_iso_25010_quality_matrix.py
python3 tests/test_check_new_ignore_attrs.py
python3 tests/test_liswet_exact_certificate.py
python3 tests/test_certificate_contract.py
python3 scripts/check_architecture.py
python3 tests/test_check_architecture.py
python3 scripts/check_docs_consistency.py
python3 tests/test_check_docs_consistency.py
python3 scripts/check_function_size.py
python3 tests/test_check_function_size.py

# 1b. test inventory gate (silent test disable / delete)
# cargo nextest list を ground truth に、コミット済み目録 tests/test_inventory.txt
# との乖離を検出する。上の full-suite で release バイナリは概ね built (この gate は
# parallel feature 込みで list するので otspot-core の部分再ビルドが入りうる)。
echo
echo "=== test inventory gate ==="
set +e
python3 scripts/lib/check_new_ignore_attrs.py
inv_rc=$?
set -e
if [ "$inv_rc" -eq 1 ]; then
  echo "::error::テスト目録が cargo nextest list と乖離 (テストの ignore 化 / 追加 / 削除)。意図的なら python3 scripts/lib/check_new_ignore_attrs.py --update で再生成し diff をコミットせよ" >&2
  exit 1
elif [ "$inv_rc" -ne 0 ]; then
  echo "::error::check_new_ignore_attrs.py が異常終了 (exit $inv_rc): nextest 不在 or ビルド失敗。上記メッセージ参照" >&2
  exit "$inv_rc"
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

# 5. コメント品質 (CLAUDE.md, "実装" section)
# diff scope ではなく full-scan を使用: gate を後付けする以前の commit に
# 違反が残存しうるため (このゲート設置時点で複数 file が threshold 超過、PR 段階
# で trim 議論)。main に違反が確定混入した場合は ALLOWLIST 追加 or
# threshold 調整で復帰、本 section の scope (full-scan) は保つ。
echo
echo "=== comment quality ==="
bash scripts/lib/check_memo_grep.sh
bash scripts/check_comment_block_size.sh
bash scripts/test_comment_block_size.sh
bash scripts/check_comment_ratio.sh
bash scripts/test_comment_ratio.sh
python3 scripts/lib/check_dead_issue_refs.py
python3 tests/test_check_dead_issue_refs.py

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
git diff main..HEAD --unified=0 -- 'otspot-core/src/*.rs' 'otspot-num/src/*.rs' 'otspot-io/src/*.rs' 'otspot-model/src/*.rs' | \
  grep -E '^\+[^+]*\b[0-9]{2,}(_[0-9]+)*(\.[0-9]+)?([eE][+-]?[0-9]+)?\b' | \
  grep -vE 'cfg\(test\)|//|test_' | head -20 || true

echo
echo "=== audit complete ==="
