#!/usr/bin/env python3
"""Verify baseline CSV vs actual data directory consistency.

Checks:
  1. File count in data dir matches or exceeds CSV row count.
  2. Every CSV name has a corresponding data file (CSV_GAP → exit 1).
  3. Reports files present in the directory but missing from the CSV (no_ref gap).
     no_ref alone does not fail unless --strict is passed.

Exit code: 0 = all checks pass, 1 = CSV_GAP found (or --strict + no_ref).

Usage:
    python3 scripts/check_data_coverage.py [--repo-root DIR] [--strict]

Options:
    --repo-root DIR   Path to repo root (default: parent of this script's dir).
    --strict          Exit 1 if any no_ref files exist.

Policies (fact-based, updated 2026-05-26):
  MIPLIB gate (miplib_small):
    No CSV baseline. MIP bench is informational only; not a CI gate until
    B&B cut/presolve lands (Plan#9 #22). When promoted: create
    data/baseline_objectives/miplib_small.csv and add routing to
    bench_parallel.sh.

  Minimum tracked fixtures (#17):
    Small QP/LP fixture files (< 50KB) can be tracked in git under
    tests/fixtures/ (or tests/data/). These bypass the data/ download
    requirement for unit tests. Implementation: task #17.
    Candidates: QPLIB_8495, AUG2D (Maros), afiro (Netlib LP).
"""

import argparse
import os
import sys
from pathlib import Path

# (dir_glob, csv_path, file_extensions, origin)
# origin: "official" | "official_gen" | "synthetic"
DATASETS = [
    # QP official
    ("maros_meszaros", "data/baseline_objectives/maros_meszaros.csv",
     [".QPS"], "official"),
    ("qplib", "data/baseline_objectives/qplib.csv",
     [".qplib"], "official"),
    ("qplib_nonconvex_official", "data/baseline_objectives/qplib_nonconvex_official.csv",
     [".qplib"], "official"),
    # QP official-derived (generated from official problem definitions)
    ("osqp_bench", "data/baseline_objectives/osqp_bench.csv",
     [".npz", ".mat"], "official_gen"),
    ("mpc_qp", "data/baseline_objectives/mpc_qp.csv",
     [".npz", ".mat"], "official_gen"),
    # QP synthetic (no external reference by design)
    ("osqp_bench_extra", None, None, "synthetic"),
    ("osqp_bench_illscaled", None, None, "synthetic"),
    ("osqp_bench_xl", None, None, "synthetic"),
    ("qplib_nonconvex", "data/baseline_objectives/qplib_nonconvex_synthetic.csv",
     [".qplib", ".npz", ".mat"], "synthetic"),
    # LP official
    ("lp_problems", "data/baseline_objectives/netlib_lp.csv",
     [".mps", ".lp", ".SIF"], "official"),
    ("lp_problems_infeas", "data/baseline_objectives/netlib_lp_infeas.csv",
     [".mps", ".lp", ".SIF"], "official"),
    ("lp_problems_canary", "data/baseline_objectives/netlib_lp_canary.csv",
     [".mps", ".lp", ".SIF", ".gz"], "official"),
    # LP without baseline (extension sets or synthetic)
    ("lp_problems_extra", None, None, "official"),
    ("lp_problems_hard", None, None, "official"),
    ("lp_problems_unbounded", "data/baseline_objectives/lp_problems_unbounded.csv",
     [".QPS", ".mps", ".lp"], "synthetic"),
    # QP generated infeasible / unbounded
    ("qp_infeasible", "data/baseline_objectives/qp_infeasible.csv",
     [".npz", ".mat"], "synthetic"),
    ("qp_unbounded", "data/baseline_objectives/qp_unbounded.csv",
     [".npz", ".mat"], "synthetic"),
    # MIP
    ("miplib_small", None, None, "official"),
]


def read_csv_names(csv_path: Path) -> set[str]:
    names: set[str] = set()
    if not csv_path.exists():
        return names
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if parts[0] == "problem_name":
            continue
        names.add(parts[0])
    return names


def data_files(data_dir: Path, extensions: list[str] | None) -> set[str]:
    """Return stems of data files (excluding Highs.log and hidden files)."""
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
        elif extensions is None:
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

    for (dir_name, csv_rel, exts, origin) in DATASETS:
        data_dir = root / "data" / dir_name
        csv_path = root / csv_rel if csv_rel else None

        if not data_dir.is_dir():
            rows.append({
                "dataset": dir_name,
                "origin": origin,
                "files": 0,
                "csv_rows": 0,
                "no_ref": 0,
                "csv_gap": 0,
                "status": "MISSING_DIR",
            })
            continue

        # Count actual data files (exclude Highs.log, symlinks-as-dirs, hidden)
        all_files = set()
        for entry in data_dir.iterdir():
            if entry.name.startswith(".") or entry.name == "Highs.log":
                continue
            if entry.is_file():
                all_files.add(entry.stem)

        csv_names = read_csv_names(csv_path) if csv_path else set()
        no_ref = all_files - csv_names  # in dir, not in CSV
        csv_gap = csv_names - all_files  # in CSV, not in dir

        status_parts = []
        if csv_gap:
            status_parts.append(f"CSV_GAP:{len(csv_gap)}")
            ok = False
        if no_ref and args.strict:
            status_parts.append(f"NO_REF:{len(no_ref)}")
            ok = False
        status = " ".join(status_parts) if status_parts else "ok"

        rows.append({
            "dataset": dir_name,
            "origin": origin,
            "files": len(all_files),
            "csv_rows": len(csv_names),
            "no_ref": len(no_ref),
            "csv_gap": len(csv_gap),
            "status": status,
            "no_ref_names": sorted(no_ref),
            "csv_gap_names": sorted(csv_gap),
        })

    # Print summary table
    col_w = [max(len(r["dataset"]) for r in rows) + 2, 14, 7, 9, 7, 9, 20]
    header = ["dataset", "origin", "files", "csv_rows", "no_ref", "csv_gap", "status"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(
            r["dataset"], r["origin"], r["files"], r["csv_rows"],
            r["no_ref"], r["csv_gap"], r["status"],
        ))

    # Detail for gaps
    for r in rows:
        if r.get("csv_gap_names"):
            print(f"\n[CSV_GAP] {r['dataset']}: in CSV but not in dir ({len(r['csv_gap_names'])} files):")
            for n in r["csv_gap_names"]:
                print(f"  {n}")
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
