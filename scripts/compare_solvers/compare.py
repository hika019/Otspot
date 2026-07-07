#!/usr/bin/env python3
"""Cross-references Otspot's own per-suite bench results against the new
HiGHS/SCIP result CSVs (from run_highs.sh / run_scip.sh) to find problems
where an external solver reaches a conclusive result but Otspot does not.

Otspot result formats (produced by existing scripts, not by this tool):
  - `bench_parallel.sh` text output (LP/QP/QPLIB suites): a
    "=== 問題別詳細 ===" section with lines
    `name  n  m  status  time  note...`.
  - `examples/solve_cbf.rs` CSV output (cblib_socp):
    `problem,status,objective,iterations,time_sec`.

This tool auto-detects which of the two by file extension/content and
normalizes both to {name: {status, time}}.

Usage:
  compare.py --otspot PATH --highs CSV [--scip CSV] [--out CSV]
"""
import argparse
import csv
import re
import sys

OTSPOT_PASS = {"PASS", "PASS:Infeasible", "PASS:Unbounded", "CHECKED[no_ref]"}
SOLVER_PASS = {"Optimal", "optimal", "Infeasible", "infeasible", "Unbounded", "unbounded"}

# Mirrors lp_vs_highs.sh's KNOWN_STATUSES so bench_parallel.sh's free-text
# note field (which may itself contain spaces) doesn't get misparsed as
# additional status tokens.
KNOWN_STATUSES = {
    "PASS", "CHECKED[no_ref]", "PASS:Infeasible", "PASS:Unbounded",
    "TIMEOUT", "EXTERNAL_TIMEOUT", "MAXITER", "ERROR", "SKIP", "PARSE_ERR",
    "NONCONVEX", "SUBOPTIMAL", "KKT_FAIL", "OBJ_MISMATCH",
    "PFEAS_FAIL", "DFEAS_FAIL", "FAIL", "FAIL:Unknown", "FAIL:NumericalError",
}


def parse_bench_parallel_txt(path: str) -> dict:
    results = {}
    with open(path) as f:
        for line in f:
            if not line or not line[0].strip() or line[0] in "=[ -":
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name, status = parts[0], parts[3]
            if not (status in KNOWN_STATUSES or status.startswith("PASS") or status.startswith("FAIL")):
                continue
            try:
                time_val = float(parts[4])
            except ValueError:
                continue
            note = " ".join(parts[5:]) if len(parts) > 5 else ""
            m = re.search(r"obj=([+-]?[0-9.eE+-]+)", note)
            results[name] = {
                "status": status,
                "time": time_val,
                "objective": m.group(1) if m else "",
            }
    return results


def parse_solve_cbf_csv(path: str) -> dict:
    results = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("problem")
            if not name:
                continue
            results[name] = {
                "status": row.get("status", ""),
                "time": _to_float(row.get("time_sec")),
                "objective": row.get("objective", ""),
            }
    return results


def parse_solver_csv(path: str) -> dict:
    """run_highs.sh / run_scip.sh output: problem,status,objective,time_sec."""
    results = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("problem")
            if not name:
                continue
            results[name] = {
                "status": row.get("status", ""),
                "time": _to_float(row.get("time_sec")),
                "objective": row.get("objective", ""),
            }
    return results


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_otspot(path: str) -> dict:
    if path.endswith(".csv"):
        return parse_solve_cbf_csv(path)
    return parse_bench_parallel_txt(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--otspot", required=True, help="Otspot's own result file (.txt from bench_parallel.sh or .csv from solve_cbf)")
    ap.add_argument("--highs", help="run_highs.sh output CSV")
    ap.add_argument("--scip", help="run_scip.sh output CSV")
    ap.add_argument("--out", help="merged comparison CSV path")
    args = ap.parse_args()

    otspot = parse_otspot(args.otspot)
    highs = parse_solver_csv(args.highs) if args.highs else {}
    scip = parse_solver_csv(args.scip) if args.scip else {}

    names = sorted(set(otspot) | set(highs) | set(scip))
    if not names:
        print("[compare] no problems found in any input", file=sys.stderr)
        sys.exit(1)

    rows = []
    frontier = []  # other-solver-can, otspot-cannot
    for name in names:
        o = otspot.get(name, {})
        h = highs.get(name, {})
        s = scip.get(name, {})
        o_status = o.get("status", "MISSING")
        h_status = h.get("status", "MISSING")
        s_status = s.get("status", "MISSING")
        o_pass = o_status in OTSPOT_PASS
        h_pass = h_status in SOLVER_PASS
        s_pass = s_status in SOLVER_PASS
        other_wins = (not o_pass) and (h_pass or s_pass) and (name in otspot)
        rows.append({
            "problem": name,
            "otspot_status": o_status,
            "otspot_time": o.get("time", ""),
            "highs_status": h_status,
            "highs_time": h.get("time", ""),
            "highs_obj": h.get("objective", ""),
            "scip_status": s_status,
            "scip_time": s.get("time", ""),
            "scip_obj": s.get("objective", ""),
            "other_solver_wins": other_wins,
        })
        if other_wins:
            frontier.append(name)

    out_path = args.out or "compare.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[compare] {len(rows)} problems -> {out_path}")
    print(f"[compare] otspot non-PASS: {sum(1 for r in rows if not (otspot.get(r['problem'], {}).get('status') in OTSPOT_PASS))}")
    print(f"[compare] frontier (other-solver-can, otspot-cannot): {len(frontier)}")
    for name in frontier:
        print(f"[compare]   {name}")


if __name__ == "__main__":
    main()
