#!/bin/bash
# test_comment_block_size.sh — sentinel for check_comment_block_size.sh.
#
# Pins the run-based gate against the gate-gaming holes fixed alongside it:
# a long comment run must not be splittable below its threshold by injecting
# one line of the other comment kind, and `////`+ must count as a note (not
# rustdoc). Runs the REAL gate (a copy of the script) over synthetic fixtures
# in a throwaway repo tree, so a regression in the awk makes a case flip.
#
# 使い方: bash scripts/test_comment_block_size.sh
# 実行時間: ~1s。CI / 手動 sanity 用。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$SCRIPT_DIR/check_comment_block_size.sh"

PASS=0
FAIL=0
TMP_ROOT=$(mktemp -d /tmp/test_comment_block_size.XXXXXX)
trap 'rm -rf "$TMP_ROOT"' EXIT

# Run the real gate over a single fixture file placed in a throwaway repo tree.
# Echoes "<exit_code> <gate_stdout>".
run_gate() {
  local fixture_content="$1"
  local work
  work=$(mktemp -d "$TMP_ROOT/repo.XXXXXX")
  mkdir -p "$work/scripts" "$work/otspot-core/src" \
    "$work/otspot-io/src" "$work/otspot-model/src" "$work/otspot-dev/src"
  cp "$GATE" "$work/scripts/check_comment_block_size.sh"
  printf '%s\n' "$fixture_content" > "$work/otspot-core/src/fixture.rs"
  local out rc
  out=$(bash "$work/scripts/check_comment_block_size.sh" 2>&1)
  rc=$?
  echo "$rc"$'\n'"$out"
}

assert_violation() {
  local label="$1" content="$2" needle="$3"
  local res rc out
  res=$(run_gate "$content")
  rc=$(head -1 <<<"$res")
  out=$(tail -n +2 <<<"$res")
  if [[ "$rc" == "1" ]] && grep -qF -- "$needle" <<<"$out"; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (rc=$rc, expected exit 1 + '$needle')" >&2
    echo "  --- gate output ---" >&2
    echo "$out" >&2
    FAIL=$((FAIL + 1))
  fi
}

assert_clean() {
  local label="$1" content="$2"
  local res rc out
  res=$(run_gate "$content")
  rc=$(head -1 <<<"$res")
  out=$(tail -n +2 <<<"$res")
  if [[ "$rc" == "0" ]]; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (rc=$rc, expected exit 0)" >&2
    echo "  --- gate output ---" >&2
    echo "$out" >&2
    FAIL=$((FAIL + 1))
  fi
}

# Repeat a comment line N times.
rep() { local marker="$1" n="$2" i; for ((i = 1; i <= n; i++)); do printf '%s note %d\n' "$marker" "$i"; done; }

echo "== gate-gaming: a decoy of the other kind must not split a run =="

# (a) 28 `//` notes with one `///` decoy in the middle: MEMO tally over the
# whole run is 28 (> 14), even though no single same-kind segment reaches 14.
MEMO28_SPLIT="$(rep '//' 14)"$'\n'"/// decoy"$'\n'"$(rep '//' 14)"
assert_violation "memo28 split by /// -> MEMO violation" "$MEMO28_SPLIT" "MAX_MEMO=14"

# (b) 47 `///` doc lines with one `//` decoy in the middle: DOC tally is 47.
DOC47_SPLIT="$(rep '///' 24)"$'\n'"// decoy"$'\n'"$(rep '///' 23)"
assert_violation "rustdoc47 split by // -> DOC violation" "$DOC47_SPLIT" "MAX_DOC=24"

echo "== P3: //// (4+ slashes) is a note, not rustdoc =="

# (c) 25 `////` lines: Rust does not treat these as doc, so they must count as
# notes (> MAX_MEMO=14), NOT as rustdoc (which would pass under MAX_DOC=24).
FOURSLASH25="$(rep '////' 25)"
assert_violation "//// x25 -> MEMO violation" "$FOURSLASH25" "MAX_MEMO=14"

# 17 `////` lines: this is the actual escape route — as a note it is over
# MAX_MEMO=14, but classifying `////` as rustdoc (the pre-fix bug) would let it
# slip under MAX_DOC=24. Must be flagged as a note.
FOURSLASH17="$(rep '////' 17)"
assert_violation "//// x17 -> MEMO violation (escape route)" "$FOURSLASH17" "MAX_MEMO=14"

echo "== boundaries: thresholds hold on same-kind runs =="

assert_clean     "doc x24 -> clean (== MAX_DOC)"   "$(rep '///' 24)"
assert_violation "doc x25 -> DOC violation"        "$(rep '///' 25)" "MAX_DOC=24"
assert_clean     "memo x14 -> clean (== MAX_MEMO)" "$(rep '//' 14)"
assert_violation "memo x15 -> MEMO violation"      "$(rep '//' 15)" "MAX_MEMO=14"

echo "== no false positives on legitimate comments =="

# A long rustdoc block, then real code, then a short note: separated by a
# non-comment line, so two independent runs, both under threshold.
LEGIT_DOC_THEN_NOTE="$(rep '///' 20)"$'\n'"fn foo() {}"$'\n'"    // a short 3-line"$'\n'"    // inline note"$'\n'"    // about foo"
assert_clean "rustdoc(20) + code + note(3) -> clean" "$LEGIT_DOC_THEN_NOTE"

# Paragraph breaks (bare marker lines) inside a doc block are not counted, so a
# 22-line doc block split into paragraphs stays clean.
DOC_WITH_BLANKS="$(rep '///' 11)"$'\n'"///"$'\n'"$(rep '///' 11)"
assert_clean "doc(22) with blank marker paragraph break -> clean" "$DOC_WITH_BLANKS"

# The real repo must stay green under the gate.
echo "== live repo =="
if (cd "$SCRIPT_DIR/.." && bash scripts/check_comment_block_size.sh >/dev/null 2>&1); then
  echo "  PASS: live repo passes the gate (exit 0)"
  PASS=$((PASS + 1))
else
  echo "  FAIL: live repo does NOT pass the gate" >&2
  FAIL=$((FAIL + 1))
fi

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
