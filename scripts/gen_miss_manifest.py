"""gen_miss_manifest.py — bench log → miss-case manifest (JSON + CSV)

Parses one or more bench_parallel.sh output logs and generates a structured
manifest of non-PASS benchmark cases.  Each miss is classified by root cause:

  timeout             — solver did not converge within the time limit
  external_timeout    — external watchdog killed the worker before the
                        solver honored its internal timeout
  status_honesty      — solver returned wrong status (e.g. Infeasible when
                        Optimal was expected)
  objective_mismatch  — Optimal claimed but objective differs from baseline
  residual_drift      — Optimal claimed but primal/dual residuals exceed eps
  suboptimal          — MaxIterations or SuboptimalSolution
  numerical           — NumericalError, parse failure, or unknown error
  skip_nonconvex      — problem outside scope (SKIP / NONCONVEX / NONCONVEX_LOCAL
                        / NONCONVEX_GLOBAL / NOT_SUPPORTED)
  unchecked_reference — feasible result without an external/baseline reference

Usage:
  python scripts/gen_miss_manifest.py LOGFILE [LOGFILE ...]
      [--out DIR]      output directory (default: reports/)
      [--format json|csv|both]
      [--include-skip] include SKIP/NONCONVEX/NOT_SUPPORTED entries

Output (gitignored under reports/):
  reports/miss_manifest_YYYYMMDD_HHMMSS.json
  reports/miss_manifest_YYYYMMDD_HHMMSS.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

PASS_STATUSES = frozenset({"PASS", "PASS:Infeasible", "PASS:Unbounded"})
SKIP_STATUSES = frozenset(
    {"SKIP", "NONCONVEX", "NONCONVEX_LOCAL", "NONCONVEX_GLOBAL", "NOT_SUPPORTED"}
)


def classify(status: str) -> str:
    if status in PASS_STATUSES:
        return "pass"
    if status == "CHECKED[no_ref]":
        return "unchecked_reference"
    if status in SKIP_STATUSES:
        return "skip_nonconvex"
    if status == "TIMEOUT":
        return "timeout"
    if status == "EXTERNAL_TIMEOUT":
        return "external_timeout"
    if status == "MAXITER":
        return "suboptimal"
    if status == "SUBOPTIMAL":
        return "suboptimal"
    if status == "OBJ_MISMATCH":
        return "objective_mismatch"
    if status in ("DFEAS_FAIL", "PFEAS_FAIL", "KKT_FAIL"):
        return "residual_drift"
    if status in ("FAIL:Infeasible", "FAIL:Unbounded"):
        return "status_honesty"
    if status.startswith("FAIL") or status in ("ERROR", "PARSE_ERR"):
        # FAIL:NumericalError, FAIL:Unknown, bare FAIL
        return "numerical"
    return "unknown"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Main problem detail line (from qps_benchmark / bench_qplib):
#   NAME   rows  cols         STATUS   time.ddd  [note...]
# The format is: {:<20} {:>6} {:>6} {:>15} {:>10.3} {}
_STATUS_PAT = (
    r"PASS(?::Infeasible|:Unbounded)?|CHECKED\[no_ref\]"
    r"|TIMEOUT|EXTERNAL_TIMEOUT|MAXITER|ERROR|SKIP|PARSE_ERR"
    r"|NONCONVEX_LOCAL|NONCONVEX_GLOBAL|NONCONVEX|SUBOPTIMAL"
    r"|NOT_SUPPORTED"
    r"|KKT_FAIL|OBJ_MISMATCH|PFEAS_FAIL|DFEAS_FAIL"
    r"|FAIL(?::[A-Za-z]+)?"
)
DETAIL_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.\-]+)\s+"
    r"(?P<rows>\d+)\s+"
    r"(?P<cols>\d+)\s+"
    r"(?P<status>" + _STATUS_PAT + r")\s+"
    r"(?P<time_s>[\d.]+)\s*"
    r"(?P<note>.*)?$"
)

# External-timeout fallback line written by bench_parallel.sh worker_func:
#   "  PROB_NAME  EXTERNAL_TIMEOUT (external_timeout=Xs, ...)"
EXT_TIMEOUT_RE = re.compile(
    r"^\s+(?P<name>\S+)\s+EXTERNAL_TIMEOUT\s+\(external_timeout=(?P<ext>\S+)"
)

# Info line written after each solve by qps_benchmark/bench_qplib:
#   "  => solver=METHOD iters=N ... | n=N m=M nnz=Z"
# Note: bench_parallel.sh's AWK aggregation filters these out, so INFO_RE only
# matches when parsing raw per-group worker logs (not the --manifest-out path).
INFO_RE = re.compile(
    r"^\s+=>\s+solver=(?P<solver>\S+)\s+iters=(?P<iters>\d+)"
    r"(?:\s+pf=(?P<pf>\S+)\s+df=(?P<df>\S+)\s+gap=(?P<gap>\S+))?"
)

# Header fields written inside the tee block by bench_parallel.sh
_META_PATTERNS = {
    "data_dir":        re.compile(r"^data-dir\s*:\s*(.+)$"),
    "bench_timeout_s": re.compile(r"^timeout\s*:\s*([\d.]+)s"),
    "bench_eps":       re.compile(r"^eps\s*:\s*(.+)$"),
    "solver_commit":   re.compile(r"^solver_commit\s*:\s*(\S+)"),
    "solver_branch":   re.compile(r"^solver_branch\s*:\s*(.+)$"),
    "bench_timestamp": re.compile(r"^bench_timestamp\s*:\s*(.+)$"),
}


# ---------------------------------------------------------------------------
# Note-field field extraction
# ---------------------------------------------------------------------------

def _float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_note(note: str) -> dict:
    """Parse structured fields out of the note column of a detail line."""
    fields: dict = {}

    # algorithm route: [ipm], [lp-ipm], [lp-simplex]
    m = re.search(r"\[([a-z][a-z\-]*)\]", note)
    if m:
        fields["route"] = m.group(1)

    for key, pat in [
        ("obj_solver",      r"\bobj=([-+]?[\d.e+\-]+)"),
        ("obj_ref",         r"\bknown=([-+]?[\d.e+\-]+)"),
        ("obj_rel_err_pct", r"\berr=([\d.e+\-]+)%"),
        ("pfeas",           r"\bpf=([\d.e+\-]+)"),
        ("pfeas_norm",      r"\bpfn=([\d.e+\-]+)"),
        ("dfeas",           r"\bdf=([\d.e+\-]+)"),
        ("dfeas_rel",       r"\bdfr=([\d.e+\-]+)"),
        ("iterations",      r"\biters=(\d+)"),
    ]:
        m = re.search(pat, note)
        if m and m.group(1) not in ("NA", "nan"):
            v = _float(m.group(1))
            if v is not None:
                fields[key] = int(v) if key == "iterations" else v

    return fields


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_log(path: Path) -> tuple[dict, list[dict]]:
    """Parse one bench log file.

    Returns (meta, records) where records contains one dict per problem.
    """
    meta: dict = {"log_file": str(path)}
    records: list[dict] = []
    current: dict | None = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # --- Metadata header fields ---------------------------------
            matched_meta = False
            for key, pat in _META_PATTERNS.items():
                m = pat.match(line.strip())
                if m:
                    val = m.group(1).strip()
                    if key == "bench_timeout_s":
                        meta[key] = _float(val)
                    else:
                        meta[key] = val
                    matched_meta = True
                    break
            if matched_meta:
                continue

            # --- External-timeout fallback line -------------------------
            m = EXT_TIMEOUT_RE.match(line)
            if m:
                if current:
                    records.append(current)
                name = m.group("name")
                current = _new_record(
                    name=name,
                    status="EXTERNAL_TIMEOUT",
                    time_s=meta.get("bench_timeout_s"),
                    note=f"external_timeout={m.group('ext')}",
                )
                continue

            # --- Main problem detail line --------------------------------
            m = DETAIL_RE.match(line)
            if m:
                if current:
                    records.append(current)
                note = (m.group("note") or "").strip()
                nf = extract_note(note)
                current = _new_record(
                    name=m.group("name"),
                    status=m.group("status"),
                    time_s=_float(m.group("time_s")),
                    note=note,
                    rows=int(m.group("rows")),
                    cols=int(m.group("cols")),
                    **nf,
                )
                continue

            # --- Info line (=> solver=... iters=...) --------------------
            m = INFO_RE.match(line)
            if m and current:
                if current["route"] is None:
                    current["route"] = m.group("solver")
                if current["iterations"] is None:
                    current["iterations"] = int(m.group("iters"))
                if current["pfeas"] is None and m.group("pf"):
                    current["pfeas"] = _float(m.group("pf"))
                if current["dfeas"] is None and m.group("df"):
                    current["dfeas"] = _float(m.group("df"))
                continue

    if current:
        records.append(current)

    return meta, records


def _new_record(*, name: str, status: str, time_s, note: str = "",
                rows=None, cols=None, **extra) -> dict:
    return {
        "problem":        name,
        "status":         status,
        "miss_class":     classify(status),
        "solve_time_s":   time_s,
        "iterations":     extra.pop("iterations", None),
        "route":          extra.pop("route", None),
        "obj_solver":     extra.pop("obj_solver", None),
        "obj_ref":        extra.pop("obj_ref", None),
        "obj_rel_err_pct": extra.pop("obj_rel_err_pct", None),
        "pfeas":          extra.pop("pfeas", None),
        "pfeas_norm":     extra.pop("pfeas_norm", None),
        "dfeas":          extra.pop("dfeas", None),
        "dfeas_rel":      extra.pop("dfeas_rel", None),
        "rows":           rows,
        "cols":           cols,
        "note":           note,
    }


def merge_logs(paths: list[Path]) -> tuple[dict, list[dict]]:
    combined_meta: dict = {"log_files": [str(p) for p in paths]}
    all_records: list[dict] = []
    for p in paths:
        meta, records = parse_log(p)
        combined_meta.update({k: v for k, v in meta.items()
                               if v is not None and k != "log_file"})
        all_records.extend(records)
    return combined_meta, all_records


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(meta: dict, records: list[dict]) -> dict:
    # Deduplicate: last occurrence wins (handles re-run logs)
    seen: dict[str, dict] = {}
    for r in records:
        seen[r["problem"]] = r

    miss_cases = [r for r in seen.values() if r["miss_class"] not in ("pass",)]
    miss_cases.sort(key=lambda r: (r["miss_class"], r["problem"]))

    n_total = len(seen)
    n_pass = sum(1 for r in seen.values() if r["miss_class"] == "pass")
    by_class: dict[str, int] = {}
    for r in miss_cases:
        by_class[r["miss_class"]] = by_class.get(r["miss_class"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bench_params": {
            "data_dir":        meta.get("data_dir"),
            "timeout_s":       meta.get("bench_timeout_s"),
            "eps":             meta.get("bench_eps"),
            "solver_commit":   meta.get("solver_commit"),
            "solver_branch":   meta.get("solver_branch"),
            "bench_timestamp": meta.get("bench_timestamp"),
            "log_files":       meta.get("log_files") or ([meta["log_file"]] if "log_file" in meta else []),
        },
        "summary": {
            "total_problems": n_total,
            "pass":           n_pass,
            "miss":           len(miss_cases),
            "by_class":       by_class,
        },
        "miss_cases": miss_cases,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "problem", "miss_class", "status", "solve_time_s", "iterations",
    "route", "obj_solver", "obj_ref", "obj_rel_err_pct",
    "pfeas", "pfeas_norm", "dfeas", "dfeas_rel", "rows", "cols", "note",
]


def write_csv(manifest: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for case in manifest["miss_cases"]:
            w.writerow({k: case.get(k, "") for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate miss-case manifest from bench log(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("logs", nargs="*", metavar="LOGFILE",
                    help="bench_parallel.sh output log file(s)")
    ap.add_argument("--out", metavar="DIR", default="reports",
                    help="Output directory (default: reports/)")
    ap.add_argument("--format", choices=["json", "csv", "both"], default="both",
                    help="Output format (default: both)")
    ap.add_argument("--include-skip", action="store_true",
                    help="Include SKIP/NONCONVEX entries in the manifest")
    args = ap.parse_args()

    if not args.logs:
        ap.print_help()
        return 1

    log_paths = [Path(p) for p in args.logs]
    missing = [str(p) for p in log_paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"error: log file not found: {m}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, records = merge_logs(log_paths)
    manifest = build_manifest(meta, records)

    if not args.include_skip:
        manifest["miss_cases"] = [
            r for r in manifest["miss_cases"]
            if r["miss_class"] != "skip_nonconvex"
        ]
        skipped = manifest["summary"]["by_class"].pop("skip_nonconvex", 0)
        manifest["summary"]["miss"] -= skipped

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"miss_manifest_{ts}"

    if args.format in ("json", "both"):
        json_path = base.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"JSON: {json_path}")

    if args.format in ("csv", "both"):
        csv_path = base.with_suffix(".csv")
        write_csv(manifest, csv_path)
        print(f"CSV:  {csv_path}")

    s = manifest["summary"]
    print(f"\n{s['total_problems']} problems — {s['pass']} PASS, {s['miss']} miss")
    for cls, cnt in sorted(s["by_class"].items()):
        names = [r["problem"] for r in manifest["miss_cases"] if r["miss_class"] == cls]
        print(f"  {cls:<25} {cnt:3d}  [{', '.join(names[:6])}{'...' if len(names) > 6 else ''}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
