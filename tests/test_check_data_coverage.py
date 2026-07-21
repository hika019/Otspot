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


def _build_qplib_repo(tmp: Path, qplib_csv: str, qcqp_csv: str, data_stems: list[str]) -> Path:
    """Fake repo with the "qplib" dataset's extra_csv_path wiring (PR #25 review):

      <tmp>/data/qplib/                       (.qplib stubs)
      <tmp>/data/baseline_objectives/qplib.csv
      <tmp>/data/baseline_objectives/qplib_qcqp.csv
    """
    (tmp / "scripts").mkdir()
    data_dir = tmp / "data" / "qplib"
    data_dir.mkdir(parents=True)
    for stem in data_stems:
        (data_dir / f"{stem}.qplib").write_text("")

    baseline_dir = tmp / "data" / "baseline_objectives"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "qplib.csv").write_text(qplib_csv)
    (baseline_dir / "qplib_qcqp.csv").write_text(qcqp_csv)
    return tmp


_QPLIB_CSV = "problem_name,optimal_obj,source\nQPLIB_10034,-1.0,src\n"
_QCQP_CSV = (
    "problem_name,optimal_obj,problem_type,objsense,source\n"
    "QPLIB_2546,-8668213.409,CCQ,minimize,qplib_official\n"
)


def test_qplib_qcqp_extra_csv_merged_into_coverage():
    """`qplib_qcqp.csv` rows must count toward `qplib` coverage, not `no_ref`.

    `QPLIB_2546` is listed only in `qplib_qcqp.csv`, never in `qplib.csv`; the
    `Dataset("qplib", ...).extra_csv_path` wiring must merge it in so its
    `.qplib` file isn't flagged `no_ref` under `--strict` (the QCQP-route
    obj-regression blind spot this closes -- PR #25 review).

    Sentinel: dropping `extra_csv_path` from the `qplib` `Dataset` entry (or
    the merge in `main()`) makes the `--strict` assertion below FAIL:
    `QPLIB_2546` would show up as `no_ref` (present in `data/qplib`, absent
    from the csv `read_csv_rows` actually reads).
    """
    with tempfile.TemporaryDirectory() as td:
        repo = _build_qplib_repo(Path(td), _QPLIB_CSV, _QCQP_CSV, ["QPLIB_10034", "QPLIB_2546"])
        code, out = _run(repo)
        assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
        assert "CSV_GAP" not in out

        code_strict, out_strict = _run(repo, strict=True)
    assert code_strict == 0, f"Expected exit 0 under --strict, got {code_strict}.\nOutput:\n{out_strict}"
    assert "NO_REF" not in out_strict, out_strict


def test_qplib_qcqp_row_without_file_is_still_csv_gap():
    """A `qplib_qcqp.csv` row with no matching `.qplib` file must still CSV_GAP.

    Confirms the merge adds rows for coverage checking rather than
    unconditionally suppressing gaps.
    """
    with tempfile.TemporaryDirectory() as td:
        # QPLIB_2546 listed in qplib_qcqp.csv but no .qplib file created for it.
        repo = _build_qplib_repo(Path(td), _QPLIB_CSV, _QCQP_CSV, ["QPLIB_10034"])
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1 (CSV_GAP), got {code}.\nOutput:\n{out}"
    assert "CSV_GAP" in out
    assert "QPLIB_2546" in out


if __name__ == "__main__":
    tests = [
        test_required_present_ok,
        test_required_absent_csv_gap,
        test_optional_present_ok,
        test_optional_absent_warn_only,
        test_qplib_qcqp_extra_csv_merged_into_coverage,
        test_qplib_qcqp_row_without_file_is_still_csv_gap,
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
