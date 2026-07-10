#!/usr/bin/env python3
"""Verify baseline CSV vs actual data directory consistency.

Checks:
  1. Every required CSV row has a corresponding data file (CSV_GAP → exit 1).
  2. Optional CSV rows (name starts with ``SS_``) with no data file emit
     [opt-gap] but do not fail.
  3. Files present in the directory but missing from the CSV are reported as
     no_ref.  no_ref alone does not fail unless --strict is passed.

Exit code: 0 = all checks pass, 1 = CSV_GAP (required row missing) or --strict + no_ref.

Usage:
    python3 scripts/check_data_coverage.py [--repo-root DIR] [--strict]

Options:
    --repo-root DIR   Path to repo root (default: parent of this script's dir).
    --strict          Exit 1 if any no_ref files exist.

Optional-row detection: rows whose ``problem_name`` starts with ``SS_`` are
treated as optional.  This prefix-based rule is regen-stable — tools such as
``baseline_from_bench_log.py --merge`` that emit only 3 CSV columns cannot
silently strip the optional marker.

Policies (fact-based, updated 2026-05-30):
  MIPLIB gate (miplib_small):
    No CSV baseline. MIP bench is informational only; not a CI gate until
    B&B cut/presolve lands (Plan#9 #22). When promoted: create
    data/baseline_objectives/miplib_small.csv and add routing to
    bench_parallel.sh.

  Minimum tracked fixtures (#17):
    Small QP/LP fixture files (< 50KB) can be tracked in git under
    tests/fixtures/ (or tests/data/). These bypass the data/ download
    requirement for unit tests. Implementation: task 17.
    Candidates: QPLIB_8495, AUG2D (Maros), afiro (Netlib LP).

  osqp_bench optional rows (SS_*):
    SuiteSparse matrix problems added by setup_extra_benches.sh (skipped
    with --no-suitesparse). Detected by ``SS_`` name prefix (regen-stable;
    4th-column marker was dropped to eliminate fragility).
    Missing SS_* files → [opt-gap] warning only, exit 0.

    Design note: an earlier split design (case a) used a separate
    osqp_bench_optional.csv loaded only by check_data_coverage.py, not by
    bench_utils::detect_csv_path. This caused SS_* baselines to be silently
    excluded from bench runner regression checks (PASS[no_ref] regression).
    Single-file design (case b, current) avoids that coverage gap.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple


class Dataset(NamedTuple):
    name: str
    csv_path: str | None
    exts: list[str] | None
    origin: str


# origin: "official" | "official_gen" | "synthetic"
DATASETS: list[Dataset] = [
    # QP official
    Dataset("maros_meszaros", "data/baseline_objectives/maros_meszaros.csv",
            [".QPS"], "official"),
    Dataset("qplib", "data/baseline_objectives/qplib.csv",
            [".qplib"], "official"),
    Dataset("qplib_nonconvex_official", "data/baseline_objectives/qplib_nonconvex_official.csv",
            [".qplib"], "official"),
    # QP official-derived (generated from official problem definitions)
    # osqp_bench.csv contains both required OSQP_* rows and optional SS_* rows
    # (detected by SS_* prefix); bench_utils::detect_csv_path loads this single file.
    Dataset("osqp_bench", "data/baseline_objectives/osqp_bench.csv",
            [".qps"], "official_gen"),
    Dataset("mpc_qp", "data/baseline_objectives/mpc_qp.csv",
            [".qps"], "official_gen"),
    # QP synthetic (no external reference by design)
    Dataset("osqp_bench_extra", None, None, "synthetic"),
    Dataset("osqp_bench_illscaled", None, None, "synthetic"),
    Dataset("qplib_nonconvex", "data/baseline_objectives/qplib_nonconvex_synthetic.csv",
            [".qplib", ".npz", ".mat"], "synthetic"),
    # LP official
    Dataset("lp_problems", "data/baseline_objectives/netlib_lp.csv",
            [".mps", ".lp", ".SIF"], "official"),
    Dataset("lp_problems_infeas", "data/baseline_objectives/netlib_lp_infeas.csv",
            [".mps", ".lp", ".SIF"], "official"),
    Dataset("lp_problems_canary", "data/baseline_objectives/netlib_lp_canary.csv",
            [".mps", ".lp", ".SIF", ".gz"], "official"),
    # LP without baseline (extension sets or synthetic)
    Dataset("lp_problems_extra", None, None, "official"),
    Dataset("lp_problems_hard", None, None, "official"),
    Dataset("lp_problems_unbounded", "data/baseline_objectives/lp_problems_unbounded.csv",
            [".QPS", ".mps", ".lp"], "synthetic"),
    # QP generated infeasible / unbounded
    Dataset("qp_infeasible", "data/baseline_objectives/qp_infeasible.csv",
            [".npz", ".mat"], "synthetic"),
    Dataset("qp_unbounded", "data/baseline_objectives/qp_unbounded.csv",
            [".npz", ".mat"], "synthetic"),
    # MIP
    Dataset("miplib_small", None, None, "official"),
]


def read_csv_rows(csv_path: Path) -> list[tuple[str, bool]]:
    """Return (problem_name, is_optional) for each data row in the CSV.

    Rows whose name starts with ``SS_`` are optional (SuiteSparse problems
    added by setup_extra_benches.sh without --no-suitesparse).  All other
    rows are required.  The 4th column is not used; prefix-based detection
    is regen-stable across tools that emit only 3 columns.
    """
    rows: list[tuple[str, bool]] = []
    if not csv_path.exists():
        return rows
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if parts[0] == "problem_name":
            continue
        name = parts[0]
        optional = name.startswith("SS_")
        rows.append((name, optional))
    return rows


def data_files(data_dir: Path, extensions: list[str] | None) -> set[str]:
    """Return stems of data files (excluding Highs.log and hidden files).

    If *extensions* is None, all file suffixes are accepted.
    """
    result: set[str] = set()
    if not data_dir.is_dir():
        return result
    for entry in data_dir.iterdir():
        if entry.name.startswith(".") or entry.name == "Highs.log":
            continue
        if not entry.is_file():
            continue
        if extensions is None or entry.suffix in extensions:
            result.add(entry.stem)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=None,
                        help="Repo root directory (default: parent of scripts/)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any no_ref data files exist")
    args = parser.parse_args()

    if args.repo_root:
        root = Path(args.repo_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    ok = True
    rows = []

    for ds in DATASETS:
        data_dir = root / "data" / ds.name
        csv_path = root / ds.csv_path if ds.csv_path else None

        if not data_dir.is_dir():
            rows.append({
                "dataset": ds.name,
                "origin": ds.origin,
                "files": 0,
                "csv_rows": 0,
                "no_ref": 0,
                "csv_gap": 0,
                "opt_gap": 0,
                "status": "MISSING_DIR",
            })
            continue

        all_files = data_files(data_dir, ds.exts)
        csv_row_list = read_csv_rows(csv_path) if csv_path else []
        csv_all_names = {name for name, _ in csv_row_list}
        csv_required = {name for name, opt in csv_row_list if not opt}
        csv_optional = {name for name, opt in csv_row_list if opt}

        no_ref = all_files - csv_all_names
        csv_gap = csv_required - all_files      # required rows with no file → fail
        opt_gap = csv_optional - all_files      # optional rows with no file → warn only

        status_parts = []
        if csv_gap:
            status_parts.append(f"CSV_GAP:{len(csv_gap)}")
            ok = False
        if opt_gap:
            status_parts.append(f"opt-gap:{len(opt_gap)}")
        if no_ref and args.strict:
            status_parts.append(f"NO_REF:{len(no_ref)}")
            ok = False
        status = " ".join(status_parts) if status_parts else "ok"

        rows.append({
            "dataset": ds.name,
            "origin": ds.origin,
            "files": len(all_files),
            "csv_rows": len(csv_row_list),
            "no_ref": len(no_ref),
            "csv_gap": len(csv_gap),
            "opt_gap": len(opt_gap),
            "status": status,
            "no_ref_names": sorted(no_ref),
            "csv_gap_names": sorted(csv_gap),
            "opt_gap_names": sorted(opt_gap),
        })

    # Print summary table
    col_w = [max(len(r["dataset"]) for r in rows) + 2, 14, 5, 7, 9, 7, 7, 22]
    header = ["dataset", "origin", "files", "csv_rows", "no_ref", "csv_gap", "opt_gap", "status"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(
            r["dataset"], r["origin"], r["files"], r["csv_rows"],
            r["no_ref"], r["csv_gap"], r["opt_gap"], r["status"],
        ))

    # Detail for gaps
    for r in rows:
        if r.get("csv_gap_names"):
            print(f"\n[CSV_GAP] {r['dataset']}: in CSV but not in dir ({len(r['csv_gap_names'])} files):")
            for n in r["csv_gap_names"]:
                print(f"  {n}")
        if r.get("opt_gap_names"):
            print(f"\n[opt-gap] {r['dataset']}: optional CSV rows with no data file ({len(r['opt_gap_names'])} files):")
            for n in r["opt_gap_names"][:20]:
                print(f"  {n}")
            if len(r["opt_gap_names"]) > 20:
                print(f"  ... ({len(r['opt_gap_names']) - 20} more)")
        if r.get("no_ref_names") and (args.strict or r.get("csv_gap_names")):
            print(f"\n[no_ref] {r['dataset']}: in dir but not in CSV ({len(r['no_ref_names'])} files):")
            for n in sorted(r["no_ref_names"])[:20]:
                print(f"  {n}")
            if len(r["no_ref_names"]) > 20:
                print(f"  ... ({len(r['no_ref_names']) - 20} more)")

    print()
    if ok:
        print("[check_data_coverage] all checks passed")
    else:
        print("[check_data_coverage] FAILED — see CSV_GAP entries above", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
