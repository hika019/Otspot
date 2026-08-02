#!/bin/bash
# Guards otspot-py/api_manifest.json against silent drift from the real
# public API it maps to Python. Two-stage check (P1#4c review finding: no
# guard existed against Rust-side ADDITIONS, only the manually-written
# manifest):
#
#   1. This script: regenerate each `cargo public-api` snapshot below and
#      diff it against its checked-in file. Any difference (addition,
#      removal, or signature change to a pub item) fails the build -- the
#      snapshot must be regenerated and reviewed.
#   2. otspot-py/tests/public_api_snapshot_coverage.rs: reads the
#      (already-generated, checked-in) snapshots and asserts every symbol
#      name in them is represented in api_manifest.json's methods/types/
#      variants or explicitly listed in out_of_scope. Pure text processing,
#      no nightly toolchain needed at that point.
#
# Two snapshots, not one (P2 review finding: otspot-model's own public-api
# dump only shows *references* to otspot_core::problem::SolveStatus /
# otspot_core::options::Tolerance as field/parameter types -- it does not
# enumerate their variants, since they are defined in a different crate):
#   - otspot_model_api_snapshot.txt: full `-p otspot-model` dump (the crate
#     otspot-py binds directly).
#   - otspot_core_status_tolerance_snapshot.txt: `-p otspot-core` dump
#     filtered to lines mentioning SolveStatus/Tolerance -- the only two
#     otspot_core types this manifest tracks variants for. Unfiltered would
#     be otspot-core's entire public API (thousands of lines), almost all
#     irrelevant to otspot-py.
#
# Usage: bash scripts/check_otspot_model_public_api.sh [--update]
#   --update: overwrite the checked-in snapshots instead of diffing (run
#             this, inspect the diff, and update api_manifest.json
#             accordingly, whenever the tracked public API intentionally
#             changes).
#
# Requires: rustup nightly-2026-05-01 (matches .github/workflows/ci.yml's
# `public-api` job) and `cargo-public-api` (cargo install cargo-public-api).

set -eu

cd "$(dirname "$0")/.."

TOOLCHAIN="nightly-2026-05-01"
UPDATE="${1:-}"
FAILED=0

check_snapshot() {
  local snapshot="$1"
  local current="$2"
  if [ "$UPDATE" = "--update" ]; then
    cp "$current" "$snapshot"
    echo "[check_otspot_model_public_api] updated $snapshot"
    return 0
  fi
  if ! diff -u "$snapshot" "$current"; then
    echo
    echo "[check_otspot_model_public_api] $snapshot drifted from the live public API"
    FAILED=1
    return 1
  fi
  echo "[check_otspot_model_public_api] OK: $snapshot matches"
}

MODEL_CURRENT=$(mktemp)
CORE_RAW=$(mktemp)
CORE_CURRENT=$(mktemp)
trap 'rm -f "$MODEL_CURRENT" "$CORE_RAW" "$CORE_CURRENT"' EXIT

# `cargo public-api` prints build progress (Compiling/Documenting) to stderr
# on success, so it is discarded there -- but a real failure (compile error,
# missing toolchain) also writes to stderr and leaves stdout short or empty.
# Piping stderr to /dev/null unconditionally hid that: a failed run produced
# an empty/truncated "current" snapshot, which then diffed as "every line
# was deleted" -- a misleading "snapshot needs regenerating" verdict instead
# of "the tool itself failed" (found by reviewer injecting a build failure
# into otspot-core). Check the exit status *before* trusting the output, and
# only suppress stderr after confirming success.
run_public_api() {
  local pkg="$1" out="$2" errlog status
  errlog=$(mktemp)
  status=0
  RUSTUP_TOOLCHAIN="$TOOLCHAIN" cargo public-api -p "$pkg" -sss >"$out" 2>"$errlog" || status=$?
  if [ "$status" -ne 0 ]; then
    echo "[check_otspot_model_public_api] cargo public-api -p $pkg failed (exit $status):" >&2
    cat "$errlog" >&2
    rm -f "$errlog"
    exit 1
  fi
  rm -f "$errlog"
}

run_public_api otspot-model "$MODEL_CURRENT"
run_public_api otspot-core "$CORE_RAW"
grep -E 'SolveStatus|Tolerance' "$CORE_RAW" > "$CORE_CURRENT" || true

check_snapshot "otspot-py/otspot_model_api_snapshot.txt" "$MODEL_CURRENT" || true
check_snapshot "otspot-py/otspot_core_status_tolerance_snapshot.txt" "$CORE_CURRENT" || true

if [ "$UPDATE" = "--update" ]; then
  echo "[check_otspot_model_public_api] now update otspot-py/api_manifest.json (methods/types/variants/out_of_scope) for any new/changed items"
  exit 0
fi

if [ "$FAILED" -ne 0 ]; then
  echo "[check_otspot_model_public_api] run: bash scripts/check_otspot_model_public_api.sh --update"
  echo "[check_otspot_model_public_api] then update otspot-py/api_manifest.json for the diff(s) above"
  exit 1
fi
