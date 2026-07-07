#!/usr/bin/env python3
"""Verify that every data/ path referenced by test code exists on disk.

Purpose (PR #25 issue #4): test-heavy.yml runs `--run-ignored all` as a
must-pass gate, and data-requiring tests panic on missing data by design
(no silent skip).  The set of data CI provides and the set of data tests
require are maintained in different places (workflow steps vs test sources)
and drift apart silently.  This script closes that gap structurally: run it
after CI data preparation and before the test run, and it fails fast with a
readable list instead of an opaque panic deep inside a long nextest run.

What it checks:
  1. Scans Rust sources for ``data/...`` string references:
     tests/**/*.rs, */tests/**/*.rs, and crate src/**/*.rs unit tests
     (excluding src/bin/ and examples/, which are not executed by nextest).
  2. A reference ending in a known data-file extension is a *file*
     requirement (the exact file must exist).  Any other reference is a
     *directory* requirement (the directory must exist and be non-empty).
  3. Missing requirements in GRACEFUL_SKIP_DATASETS are warnings only
     (the referencing tests skip cleanly when data is absent — a known
     coverage gap, surfaced but not fatal).  All other missing
     requirements fail with exit 1.

Exit code: 0 = all hard requirements satisfied, 1 = at least one missing.

Usage:
    python3 scripts/check_test_data_requirements.py [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Datasets whose referencing tests skip gracefully when the data is absent
# (verified by reading the test sources — they early-return, they do not
# panic).  Missing entries here are surfaced as warnings (coverage gap),
# not failures.  If a new test *panics* on data from one of these datasets,
# remove the dataset from this list.
GRACEFUL_SKIP_DATASETS: dict[str, str] = {
    # tests/cbf_feasibility.rs: `if !path.exists() { eprintln!(skip); return; }`
    "cblib_socp": "cbf_feasibility.rs skips when absent",
    # otspot-io/src/qplib/mod.rs unit tests: `if !path.exists() { return; }`
    "qplib_unsupported": "otspot-io qplib unit tests skip when absent",
    # diag_greenbea_dfeas.rs / diag_etamacro_dfeas.rs list the canary dir as
    # the first candidate and fall back to data/lp_problems (which is a hard
    # requirement checked separately via its own references).
    "lp_problems_canary": "candidate path with fallback to data/lp_problems",
}

# Extensions that identify a reference as a concrete data-file requirement.
DATA_FILE_EXTS = {
    ".qps", ".qplib", ".mps", ".cbf", ".csv", ".npz", ".mat", ".lp", ".sif",
}

# data/<something>; stops at quotes, braces ({name} in format! strings),
# whitespace and other non-path characters.  The negative lookbehind skips
# "/data/..." (absolute paths, e.g. CSV-routing name-matching fixtures in
# otspot-dev/src/bench_utils.rs) — only repo-relative refs are requirements.
DATA_REF_RE = re.compile(r"(?<![/\w-])data/[A-Za-z0-9_][A-Za-z0-9_.\-/]*")


def rust_sources(root: Path) -> list[Path]:
    """Rust files whose tests nextest executes: integration tests and
    in-crate unit tests.  Excludes bins/examples (not run as tests) and
    build artifacts / nested worktrees."""
    results: list[Path] = []
    for path in root.rglob("*.rs"):
        rel_parts = path.relative_to(root).parts
        if any(p in ("target", ".claude", ".codex-worktrees", "examples") for p in rel_parts):
            continue
        if "bin" in rel_parts and "src" in rel_parts:
            continue
        if "tests" in rel_parts or "src" in rel_parts:
            results.append(path)
    return results


def collect_requirements(root: Path) -> dict[str, set[str]]:
    """Map of data-path reference → set of referencing source files."""
    reqs: dict[str, set[str]] = {}
    for src in rust_sources(root):
        try:
            text = src.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in DATA_REF_RE.finditer(text):
            ref = m.group(0).rstrip("/.")
            # "data" alone or "data/<x>" with no dataset dir is not a
            # requirement (prose in comments like "data/ 配下").
            if ref.count("/") < 1 or ref == "data":
                continue
            reqs.setdefault(ref, set()).add(str(src.relative_to(root)))
    return reqs


def classify(ref: str) -> str:
    """'file' if the last segment carries a known data extension, else 'dir'."""
    suffix = Path(ref).suffix.lower()
    return "file" if suffix in DATA_FILE_EXTS else "dir"


def check(root: Path) -> int:
    reqs = collect_requirements(root)
    missing_hard: list[tuple[str, str, list[str]]] = []
    missing_soft: list[tuple[str, str, list[str]]] = []
    ok_count = 0

    for ref in sorted(reqs):
        kind = classify(ref)
        path = root / ref
        if kind == "file":
            present = path.is_file()
        else:
            present = path.is_dir() and any(path.iterdir())
        if present:
            ok_count += 1
            continue
        dataset = ref.split("/")[1]
        entry = (ref, kind, sorted(reqs[ref]))
        if dataset in GRACEFUL_SKIP_DATASETS:
            missing_soft.append(entry)
        else:
            missing_hard.append(entry)

    print(f"[check_test_data_requirements] {len(reqs)} referenced data paths: "
          f"{ok_count} ok, {len(missing_soft)} skip-tolerant missing, "
          f"{len(missing_hard)} REQUIRED missing")

    for ref, kind, sources in missing_soft:
        dataset = ref.split("/")[1]
        reason = GRACEFUL_SKIP_DATASETS[dataset]
        print(f"\n[warn] missing {kind}: {ref}  (graceful skip: {reason})")
        for s in sources:
            print(f"       referenced by {s}")

    sys.stdout.flush()
    for ref, kind, sources in missing_hard:
        print(f"\n[FAIL] missing {kind}: {ref}", file=sys.stderr)
        for s in sources:
            print(f"       referenced by {s}", file=sys.stderr)

    if missing_hard:
        print(
            "\n[check_test_data_requirements] FAILED — tests referencing the "
            "paths above will panic ('data missing'). Provide the data in the "
            "CI data-preparation steps (see .github/workflows/test-heavy.yml) "
            "or, if the referencing test skips gracefully, add its dataset to "
            "GRACEFUL_SKIP_DATASETS with a reason.",
            file=sys.stderr,
        )
        return 1
    print("[check_test_data_requirements] all required data present")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", default=None,
                        help="Repo root directory (default: parent of scripts/)")
    args = parser.parse_args()
    root = (Path(args.repo_root).resolve() if args.repo_root
            else Path(__file__).resolve().parent.parent)
    return check(root)


if __name__ == "__main__":
    sys.exit(main())
