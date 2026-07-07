#!/bin/bash
# run_highs.sh — solves every problem in a directory with HiGHS (highspy),
# sequentially, and writes one CSV row per problem.
#
# Supports: *.mps, *.qps/*.QPS (direct), *.qplib (via dump_problem converter).
# *.cbf is out of scope for HiGHS (no SOCP solver) — reported as
# Unsupported(no-SOCP-in-HiGHS) without attempting a solve.
#
# Usage: run_highs.sh <data_dir> [timeout_sec] [output_csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?usage: run_highs.sh <data_dir> [timeout_sec] [output_csv]}"
TIMEOUT="${2:-1000}"
OUT_CSV="${3:-$(basename "${DATA_DIR%/}")_highs.csv}"
LOG_FILE="${OUT_CSV%.csv}.log"
HARD_TIMEOUT=$((${TIMEOUT%.*} + 30))

: > "$LOG_FILE"
echo "problem,status,objective,time_sec" > "$OUT_CSV"

shopt -s nullglob nocaseglob
FILES=("$DATA_DIR"/*.qplib "$DATA_DIR"/*.cbf "$DATA_DIR"/*.mps "$DATA_DIR"/*.qps)
shopt -u nocaseglob
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "[run_highs] no .mps/.qps/.qplib/.cbf files in $DATA_DIR" >&2
    exit 1
fi
mapfile -t SORTED < <(printf '%s\n' "${FILES[@]}" | sort)

echo "[run_highs] data=$DATA_DIR problems=${#SORTED[@]} timeout=${TIMEOUT}s -> $OUT_CSV"
for f in "${SORTED[@]}"; do
    name="$(basename "${f%.*}")"
    row="$(timeout "${HARD_TIMEOUT}" python3 "$SCRIPT_DIR/solve_one_highs.py" "$f" --timeout "$TIMEOUT" 2>>"$LOG_FILE")" \
        || row="${name},TIMEOUT_HARD,,${TIMEOUT}"
    echo "$row" >> "$OUT_CSV"
    echo "[run_highs] $row"
done
echo "[run_highs] done -> $OUT_CSV"
