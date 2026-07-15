#!/bin/bash
# test_comment_ratio.sh — sentinel for check_comment_ratio.sh.
#
# Pins the note-only ratio gate: a rustdoc-heavy file must NOT be flagged (the
# false positive that the old total-comment gate produced on conic/mod.rs /
# qcqp_guard.rs), while a note-dense file must be. Runs the REAL gate (a copy
# of the script) over synthetic fixtures in a throwaway repo tree.
#
# 使い方: bash scripts/test_comment_ratio.sh
# 実行時間: ~1s。CI / 手動 sanity 用。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$SCRIPT_DIR/check_comment_ratio.sh"

PASS=0
FAIL=0
TMP_ROOT=$(mktemp -d /tmp/test_comment_ratio.XXXXXX)
trap 'rm -rf "$TMP_ROOT"' EXIT

# Build a fixture .rs with `d` rustdoc lines, `m` note lines, `q` `////` note
# lines, padded to `total` lines with code. Echoes the file content.
fixture() {
  local d="$1" m="$2" q="$3" total="$4" i code
  code=$((total - d - m - q))
  for ((i = 1; i <= d; i++)); do printf '/// doc line %d\n' "$i"; done
  for ((i = 1; i <= m; i++)); do printf '// note line %d\n' "$i"; done
  for ((i = 1; i <= q; i++)); do printf '//// note line %d\n' "$i"; done
  for ((i = 1; i <= code; i++)); do printf 'let _x%d = %d;\n' "$i" "$i"; done
}

# Run the real gate over a single fixture. Echoes "<exit_code>\n<stdout>".
run_gate() {
  local content="$1" work out rc
  work=$(mktemp -d "$TMP_ROOT/repo.XXXXXX")
  mkdir -p "$work/scripts" "$work/otspot-core/src" \
    "$work/otspot-io/src" "$work/otspot-model/src" "$work/otspot-dev/src"
  cp "$GATE" "$work/scripts/check_comment_ratio.sh"
  # `\n`: `$(...)` strips the fixture's trailing newline; restore it so the
  # line count (and thus the >=100 scope gate) matches the intended total.
  printf '%s\n' "$content" > "$work/otspot-core/src/fixture.rs"
  out=$(bash "$work/scripts/check_comment_ratio.sh" 2>&1)
  rc=$?
  echo "$rc"$'\n'"$out"
}

assert_violation() {
  local label="$1" content="$2" res rc out
  res=$(run_gate "$content"); rc=$(head -1 <<<"$res"); out=$(tail -n +2 <<<"$res")
  if [[ "$rc" == "1" ]] && grep -qF -- "Note-comment ratio" <<<"$out"; then
    echo "  PASS: $label"; PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (rc=$rc, expected exit 1 + violation)" >&2
    echo "$out" >&2; FAIL=$((FAIL + 1))
  fi
}

assert_clean() {
  local label="$1" content="$2" res rc out
  res=$(run_gate "$content"); rc=$(head -1 <<<"$res"); out=$(tail -n +2 <<<"$res")
  if [[ "$rc" == "0" ]]; then
    echo "  PASS: $label"; PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (rc=$rc, expected exit 0)" >&2
    echo "$out" >&2; FAIL=$((FAIL + 1))
  fi
}

echo "== rustdoc must not be policed (the false positive we fixed) =="
# 50% rustdoc, 0% note: the old total-comment gate flagged this; note-only must not.
assert_clean "rustdoc 50% / note 0% -> clean" "$(fixture 60 0 0 120)"
# Doc-heavy + moderate note: total 40% (old gate flags), note 15% (< 20 -> clean).
assert_clean "doc 25% + note 15% (total 40%) -> clean" "$(fixture 25 15 0 100)"

echo "== note-dense files must be flagged =="
assert_violation "note 33% -> violation"        "$(fixture 0 40 0 120)"
assert_violation "//// counted as note: 30% -> violation" "$(fixture 0 0 30 100)"

echo "== threshold boundary (>= 20%) =="
assert_violation "note 20% -> violation (== THRESHOLD)" "$(fixture 0 20 0 100)"
assert_clean     "note 19% -> clean"                    "$(fixture 0 19 0 100)"

echo "== small files (<100 lines) are out of scope =="
assert_clean "note 60% but only 50 lines -> skipped/clean" "$(fixture 0 30 0 50)"

echo "== live repo =="
if (cd "$SCRIPT_DIR/.." && bash scripts/check_comment_ratio.sh >/dev/null 2>&1); then
  echo "  PASS: live repo passes the gate (exit 0)"; PASS=$((PASS + 1))
else
  echo "  FAIL: live repo does NOT pass the gate" >&2; FAIL=$((FAIL + 1))
fi

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
