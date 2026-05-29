"""Sentinel tests for scripts/check_data_coverage.py.

Covers the four core branches of optional-row handling:
  1. required row + file present  → ok (exit 0)
  2. required row + file absent   → CSV_GAP (exit 1)
  3. optional row + file present  → ok (exit 0)
  4. optional row + file absent   → opt-gap (exit 0, warning only)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_data_coverage.py"
SENTINEL_CSV_HEADER = "problem_name,optimal_obj,source,optional?\n"


def _build_repo(tmp: Path, csv_content: str, data_stems: list[str]) -> Path:
    """Create a minimal fake repo under *tmp*.

    Layout:
      <tmp>/scripts/               (empty, satisfies parent-of-script heuristic)
      <tmp>/data/osqp_bench/       (populated with .qps stubs)
      <tmp>/data/baseline_objectives/osqp_bench.csv
    """
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir()

    data_dir = tmp / "data" / "osqp_bench"
    data_dir.mkdir(parents=True)
    for stem in data_stems:
        (data_dir / f"{stem}.qps").write_text("")

    baseline_dir = tmp / "data" / "baseline_objectives"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "osqp_bench.csv").write_text(csv_content)
    return tmp


def _run(repo_root: Path, strict: bool = False) -> tuple[int, str]:
    cmd = [sys.executable, str(SCRIPT), "--repo-root", str(repo_root)]
    if strict:
        cmd.append("--strict")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def test_required_present_ok():
    """required row + file present → ok, exit 0."""
    csv = SENTINEL_CSV_HEADER + "OSQP_TEST_A,1.0,source\n"
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), csv, ["OSQP_TEST_A"])
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "CSV_GAP" not in out


def test_required_absent_csv_gap():
    """required row + file absent → CSV_GAP, exit 1."""
    csv = SENTINEL_CSV_HEADER + "OSQP_MISSING,1.0,source\n"
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), csv, [])
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    assert "CSV_GAP" in out


def test_optional_present_ok():
    """optional row (optional?=*) + file present → ok, exit 0."""
    csv = SENTINEL_CSV_HEADER + "SS_TEST_A,1.0,source,*\n"
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), csv, ["SS_TEST_A"])
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "CSV_GAP" not in out
    assert "opt-gap" not in out


def test_optional_absent_warn_only():
    """optional row + file absent → opt-gap warning, exit 0."""
    csv = SENTINEL_CSV_HEADER + "SS_MISSING,1.0,source,*\n"
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), csv, [])
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "opt-gap" in out
    assert "CSV_GAP" not in out


if __name__ == "__main__":
    tests = [
        test_required_present_ok,
        test_required_absent_csv_gap,
        test_optional_present_ok,
        test_optional_absent_warn_only,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    sys.exit(failed)
