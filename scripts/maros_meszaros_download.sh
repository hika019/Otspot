#!/bin/bash
# Maros-Meszaros QP test set (138 problems) downloader.
#
# Source: the original QPS files (Maros & Meszaros QPDATA) mirrored in
#   https://github.com/YimingYAN/QP-Test-Problems (QPS_Files/), pinned to a commit.
# Files are stored verbatim so the objective constant (objective-row RHS) is kept.
# Every file is checked against scripts/maros_meszaros_qps.sha256; an existing
# file whose hash does not match (e.g. one converted from .mat by an older
# version of this script) is downloaded again.
#
# Usage:
#   bash scripts/maros_meszaros_download.sh [OUT_DIR]
# OUT_DIR defaults to data/maros_meszaros.
#
# Requires: curl, sha256sum

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${1:-data/maros_meszaros}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

MANIFEST="$SCRIPT_DIR/maros_meszaros_qps.sha256"
COMMIT="871cc300154cfb438acc20f52b2c585db3f11e31"
BASE_URL="https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/$COMMIT/QPS_Files"
EXPECTED_COUNT=138

file_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

mapfile -t ENTRIES < <(grep -v '^#' "$MANIFEST")
if (( ${#ENTRIES[@]} != EXPECTED_COUNT )); then
  echo "[error] $MANIFEST lists ${#ENTRIES[@]} files, expected $EXPECTED_COUNT" >&2
  exit 1
fi

ok=0
skip=0
fail=0
fail_names=()

for entry in "${ENTRIES[@]}"; do
  read -r want file <<<"$entry"
  out="$OUT_DIR/$file"
  if [[ -f "$out" && "$(file_sha256 "$out")" == "$want" ]]; then
    skip=$((skip + 1))
    continue
  fi

  tmp=$(mktemp "$OUT_DIR/.${file}.XXXXXX")
  if ! curl -fsSL "$BASE_URL/$file" -o "$tmp"; then
    rm -f "$tmp"
    fail=$((fail + 1))
    fail_names+=("$file(download)")
    echo "[fail] $file (download)"
    continue
  fi
  got=$(file_sha256 "$tmp")
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp"
    fail=$((fail + 1))
    fail_names+=("$file(sha256)")
    echo "[fail] $file (sha256 $got != $want)"
    continue
  fi
  chmod 0644 "$tmp"
  mv "$tmp" "$out"
  ok=$((ok + 1))
  echo "[ok]   $file"
done

echo ""
echo "[summary] maros_meszaros: ok=$ok skip=$skip fail=$fail total=$EXPECTED_COUNT -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-names] ${fail_names[*]}"
  exit 1
fi
