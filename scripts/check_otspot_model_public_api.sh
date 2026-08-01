#!/bin/bash
# Guards otspot-py/api_manifest.json against silent drift from otspot-model's
# real public API. Two-stage check (P1#4c review finding: no guard existed
# against Rust-side ADDITIONS, only the manually-written manifest):
#
#   1. This script: regenerate the `cargo public-api` snapshot and diff it
#      against the checked-in otspot-py/otspot_model_api_snapshot.txt. Any
#      difference (addition, removal, or signature change to a pub item)
#      fails the build -- the snapshot must be regenerated and reviewed.
#   2. otspot-py/tests/api_manifest_rust.rs's
#      `public_api_snapshot_is_covered_by_manifest_or_out_of_scope` test:
#      reads the (already-generated, checked-in) snapshot and asserts every
#      symbol name in it is represented in api_manifest.json's methods/types
#      or explicitly listed in out_of_scope. Pure text processing, no
#      nightly toolchain needed at that point.
#
# Usage: bash scripts/check_otspot_model_public_api.sh [--update]
#   --update: overwrite the checked-in snapshot instead of diffing (run this,
#             inspect the diff, and update api_manifest.json accordingly,
#             whenever otspot-model's public API intentionally changes).
#
# Requires: rustup nightly-2026-05-01 (matches .github/workflows/ci.yml's
# `public-api` job) and `cargo-public-api` (cargo install cargo-public-api).

set -eu

cd "$(dirname "$0")/.."

SNAPSHOT="otspot-py/otspot_model_api_snapshot.txt"
TOOLCHAIN="nightly-2026-05-01"

CURRENT=$(mktemp)
trap 'rm -f "$CURRENT"' EXIT

RUSTUP_TOOLCHAIN="$TOOLCHAIN" cargo public-api -p otspot-model -sss 2>/dev/null > "$CURRENT"

if [ "${1:-}" = "--update" ]; then
  cp "$CURRENT" "$SNAPSHOT"
  echo "[check_otspot_model_public_api] updated $SNAPSHOT"
  echo "[check_otspot_model_public_api] now update api_manifest.json's methods/types/out_of_scope for any new/changed items"
  exit 0
fi

if ! diff -u "$SNAPSHOT" "$CURRENT"; then
  echo
  echo "[check_otspot_model_public_api] otspot-model's public API drifted from $SNAPSHOT"
  echo "[check_otspot_model_public_api] run: bash scripts/check_otspot_model_public_api.sh --update"
  echo "[check_otspot_model_public_api] then update otspot-py/api_manifest.json (methods/types/out_of_scope) for the diff above"
  exit 1
fi

echo "[check_otspot_model_public_api] OK: otspot-model public API matches $SNAPSHOT"
