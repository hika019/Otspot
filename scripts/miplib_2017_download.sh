#!/bin/bash
# MIPLIB 2017 Benchmark download and setup
#
# Desc: Download MIPLIB 2017 benchmark set (240+ instances, ~317 MB)
#
# Files are in .mps format (standard Mixed Integer Program format)
# License: ZIB (academic/research, copyright © Zuse Institute Berlin)
# Source: https://miplib.zib.de/downloads/benchmark.zip
#
# Output: data/miplib_2017/ (uncompressed .mps files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="data/miplib_2017"
TEMP_ZIP="$(mktemp /tmp/miplib_2017.XXXXXX.zip)"

# Ensure data dir exists
mkdir -p "$DATA_DIR"

echo "[miplib_2017] Downloading MIPLIB 2017 benchmark (317 MB)..."
# Download benchmark.zip (v2, updated June 2019)
if ! curl -L -f -o "$TEMP_ZIP" "https://miplib.zib.de/downloads/benchmark.zip"; then
  rm -f "$TEMP_ZIP"
  echo "[error] Failed to download MIPLIB 2017 benchmark" >&2
  exit 1
fi

echo "[miplib_2017] Extracting to $DATA_DIR..."
if ! unzip -q "$TEMP_ZIP" -d "$DATA_DIR"; then
  rm -f "$TEMP_ZIP"
  echo "[error] Failed to extract MIPLIB 2017 benchmark" >&2
  exit 1
fi

# The zip contains .mps.gz files at the root
# Decompress each .mps.gz file to plain .mps (for parser compatibility)
echo "[miplib_2017] Decompressing .mps.gz files..."
for gz_file in "$DATA_DIR"/*.mps.gz; do
  if [[ -f "$gz_file" ]]; then
    mps_file="${gz_file%.gz}"
    gunzip -f "$gz_file" || {
      echo "[warning] Failed to decompress $gz_file" >&2
    }
  fi
done

rm -f "$TEMP_ZIP"

# Verify extraction
file_count=$(find "$DATA_DIR" -maxdepth 1 -name "*.mps" 2>/dev/null | wc -l)
if [[ "$file_count" -eq 0 ]]; then
  echo "[error] No .mps files found after decompression" >&2
  exit 1
fi

disk_usage=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
echo "[miplib_2017] Done: $file_count instances, $disk_usage"
