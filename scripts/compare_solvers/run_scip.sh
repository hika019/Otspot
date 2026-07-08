#!/bin/bash
# run_scip.sh — solves every problem in a directory with SCIP (pyscipopt),
# sequentially, and writes one CSV row per problem.
#
# Supports: *.mps, *.qps/*.QPS (direct), *.qplib and *.cbf (via dump_problem
# converter; SOC cones as general quadratic constraints).
#
# Usage: run_scip.sh <data_dir> [timeout_sec] [output_csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?usage: run_scip.sh <data_dir> [timeout_sec] [output_csv]}"
TIMEOUT="${2:-1000}"
OUT_CSV="${3:-$(basename "${DATA_DIR%/}")_scip.csv}"
LOG_FILE="${OUT_CSV%.csv}.log"
HARD_TIMEOUT=$((${TIMEOUT%.*} + 30))

: > "$LOG_FILE"
echo "problem,status,objective,time_sec" > "$OUT_CSV"

shopt -s nullglob nocaseglob
FILES=("$DATA_DIR"/*.qplib "$DATA_DIR"/*.cbf "$DATA_DIR"/*.mps "$DATA_DIR"/*.qps)
shopt -u nocaseglob
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "[run_scip] no .mps/.qps/.qplib/.cbf files in $DATA_DIR" >&2
    exit 1
fi
mapfile -t SORTED < <(printf '%s\n' "${FILES[@]}" | sort)

# GNU timeout exit code when the limit fires (`man timeout`); any other
# non-zero exit is a solver-process crash (e.g. 132=SIGILL, 139=SIGSEGV).
GNU_TIMEOUT_EXIT=124

echo "[run_scip] data=$DATA_DIR problems=${#SORTED[@]} timeout=${TIMEOUT}s -> $OUT_CSV"
for f in "${SORTED[@]}"; do
    name="$(basename "${f%.*}")"
    rc=0
    row="$(timeout "${HARD_TIMEOUT}" python3 "$SCRIPT_DIR/solve_one_scip.py" "$f" --timeout "$TIMEOUT" 2>>"$LOG_FILE")" || rc=$?
    if [[ $rc -eq $GNU_TIMEOUT_EXIT ]]; then
        row="${name},TIMEOUT_HARD,,${TIMEOUT}"
    elif [[ $rc -ne 0 ]]; then
        row="${name},CRASH(exit=${rc}),,"
    fi
    echo "$row" >> "$OUT_CSV"
    echo "[run_scip] $row"
done
echo "[run_scip] done -> $OUT_CSV"
