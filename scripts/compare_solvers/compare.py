#!/usr/bin/env python3
"""Cross-references Otspot's own per-suite bench results against the new
HiGHS/SCIP result CSVs (from run_highs.sh / run_scip.sh) to find problems
where an external solver reaches a conclusive result but Otspot does not.

Otspot result formats (produced by existing scripts, not by this tool):
  - `bench_parallel.sh` text output (LP/QP/QPLIB suites): a
    "=== 問題別詳細 ===" section with lines
    `name  n  m  status  time  note...`, plus 2-space-indented fallback
    entries `  name.ext  EXTERNAL_TIMEOUT (...)` / `  name.ext  ERROR ...`
    appended when a worker group was killed externally or crashed before
    producing per-problem lines (bench_parallel.sh:290).
  - `examples/solve_cbf.rs` CSV output (cblib_socp):
    `problem,status,objective,iterations,time_sec` with `SolveStatus`
    Display vocabulary (Optimal / Infeasible / Unbounded / Timeout /
    NumericalError / MaxIterations / Unsupported / ...).

Status semantics differ between the two formats:
  - bench_parallel statuses are *verified* (obj checked against baseline):
    PASS / PASS:Infeasible / PASS:Unbounded / CHECKED[no_ref] count as PASS.
  - solve_cbf statuses are raw solver claims. `Optimal` counts as PASS
    (objective printed, comparable). `Infeasible`/`Unbounded` are conclusive
    *claims with no baseline cross-check* (data/baseline_objectives/
    cblib_socp.csv is self-referential, source=otspot_self, so it cannot
    independently verify them); they are flagged in the
    `otspot_unverified_claim` column instead of being counted as PASS, and
    they suppress `other_solver_wins` when the external solver reached the
    *same* conclusion (agreement between the two is not a win).

Usage:
  compare.py --otspot PATH --highs CSV [--scip CSV] [--out CSV]
"""
import argparse
import csv
import re
import sys

# bench_parallel.sh detail-line statuses that mean "Otspot solved and the
# result was verified against the suite baseline".
OTSPOT_PASS = {"PASS", "PASS:Infeasible", "PASS:Unbounded", "CHECKED[no_ref]"}

# solve_cbf.rs (SolveStatus Display) vocabulary.
CBF_PASS = {"Optimal"}
CBF_UNVERIFIED_CONCLUSIVE = {"Infeasible", "Unbounded"}

SOLVER_PASS = {"Optimal", "optimal", "Infeasible", "infeasible", "Unbounded", "unbounded"}

# Normalized conclusion labels, used to detect agreement between an
# unverified Otspot claim and the external solver's conclusion.
_CONCLUSION = {
    "Optimal": "optimal", "optimal": "optimal",
    "Infeasible": "infeasible", "infeasible": "infeasible",
    "Unbounded": "unbounded", "unbounded": "unbounded",
}

# Mirrors lp_vs_highs.sh's KNOWN_STATUSES so bench_parallel.sh's free-text
# note field (which may itself contain spaces) doesn't get misparsed as
# additional status tokens.
KNOWN_STATUSES = {
    "PASS", "CHECKED[no_ref]", "PASS:Infeasible", "PASS:Unbounded",
    "TIMEOUT", "EXTERNAL_TIMEOUT", "MAXITER", "ERROR", "SKIP", "PARSE_ERR",
    "NONCONVEX", "NONCONVEX_LOCAL", "NONCONVEX_GLOBAL", "NOT_SUPPORTED",
    "SUBOPTIMAL", "KKT_FAIL", "OBJ_MISMATCH",
    "PFEAS_FAIL", "DFEAS_FAIL", "FAIL", "FAIL:Unknown", "FAIL:NumericalError",
}

# Statuses of bench_parallel.sh fallback entries (`  name.ext STATUS note...`,
# 2-space indent, no n/m/time columns) appended when a worker group was
# killed by the external timeout or crashed without per-problem output.
INDENTED_FALLBACK_STATUSES = {"EXTERNAL_TIMEOUT", "ERROR"}


def _strip_ext(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def parse_bench_parallel_txt(path: str) -> dict:
    results = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            # Indented fallback entries: `  name.ext  EXTERNAL_TIMEOUT (...)`
            # or `  name.ext  ERROR worker_exit=N`. The summary counter line
            # `    EXTERNAL_TIMEOUT: 1` is excluded because its first token
            # ends with a colon.
            if (
                line[0] == " "
                and len(parts) >= 2
                and parts[1] in INDENTED_FALLBACK_STATUSES
                and not parts[0].endswith(":")
            ):
                results[_strip_ext(parts[0])] = {
                    "status": parts[1],
                    "time": None,
                    "objective": "",
                    "format": "bench_parallel",
                }
                continue
            if line[0] in "=[ -" or not line[0].strip():
                continue
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
                "format": "bench_parallel",
            }
    return results


def parse_solve_cbf_csv(path: str) -> dict:
    results = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("problem")
            # Concatenated runs repeat the header line mid-file
            # (observed in bench_results/cblib_20260707_postfix).
            if not name or name == "problem":
                continue
            results[name] = {
                "status": row.get("status", ""),
                "time": _to_float(row.get("time_sec")),
                "objective": row.get("objective", ""),
                "format": "solve_cbf",
            }
    return results


def parse_solver_csv(path: str) -> dict:
    """run_highs.sh / run_scip.sh output: problem,status,objective,time_sec."""
    results = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("problem")
            if not name or name == "problem":
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


def otspot_pass(entry: dict) -> bool:
    status = entry.get("status", "")
    if entry.get("format") == "solve_cbf":
        return status in CBF_PASS
    return status in OTSPOT_PASS


def otspot_unverified_claim(entry: dict) -> bool:
    return (
        entry.get("format") == "solve_cbf"
        and entry.get("status", "") in CBF_UNVERIFIED_CONCLUSIVE
    )


def build_rows(otspot: dict, highs: dict, scip: dict) -> list:
    """Merges the three result maps into comparison rows.

    `notes` documents the SCIP fairness caveat (heuristics disabled on
    nonlinear models — AVX-less CPU workaround, see README.md) on every row
    that carries a SCIP result, so the caveat travels with the CSV.
    """
    names = sorted(set(otspot) | set(highs) | set(scip))
    rows = []
    for name in names:
        o = otspot.get(name, {})
        h = highs.get(name, {})
        s = scip.get(name, {})
        o_status = o.get("status", "MISSING")
        h_status = h.get("status", "MISSING")
        s_status = s.get("status", "MISSING")
        o_pass = otspot_pass(o) if name in otspot else False
        o_unverified = otspot_unverified_claim(o)
        h_pass = h_status in SOLVER_PASS
        s_pass = s_status in SOLVER_PASS

        # An external solver "wins" when it reaches a conclusive result and
        # Otspot does not. An unverified Otspot claim (cbf Infeasible/
        # Unbounded) does not count as PASS, but when the external solver
        # reached the *same* conclusion the two agree — not a win.
        o_claim = _CONCLUSION.get(o_status) if o_unverified else None
        h_wins = h_pass and _CONCLUSION.get(h_status) != o_claim
        s_wins = s_pass and _CONCLUSION.get(s_status) != o_claim
        other_wins = (not o_pass) and (h_wins or s_wins) and (name in otspot)

        rows.append({
            "problem": name,
            "otspot_status": o_status,
            "otspot_time": o.get("time", ""),
            "otspot_unverified_claim": o_unverified,
            "highs_status": h_status,
            "highs_time": h.get("time", ""),
            "highs_obj": h.get("objective", ""),
            "scip_status": s_status,
            "scip_time": s.get("time", ""),
            "scip_obj": s.get("objective", ""),
            "other_solver_wins": other_wins,
            "notes": "scip_heuristics_off_on_nonlinear" if name in scip else "",
        })
    return rows


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

    rows = build_rows(otspot, highs, scip)
    if not rows:
        print("[compare] no problems found in any input", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or "compare.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    frontier = [r["problem"] for r in rows if r["other_solver_wins"]]
    n_nonpass = sum(
        1 for r in rows
        if r["problem"] in otspot and not otspot_pass(otspot[r["problem"]])
    )
    print(f"[compare] {len(rows)} problems -> {out_path}")
    print(f"[compare] otspot non-PASS: {n_nonpass}")
    print(f"[compare] frontier (other-solver-can, otspot-cannot): {len(frontier)}")
    for name in frontier:
        print(f"[compare]   {name}")


if __name__ == "__main__":
    main()
