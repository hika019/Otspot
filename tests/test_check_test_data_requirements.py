"""Sentinel tests for scripts/check_test_data_requirements.py.

Covers the core branches of the heavy-data alignment guard (PR #25 issue #4):
  1. file requirement + file present            → ok (exit 0)
  2. file requirement + file absent             → FAIL (exit 1)
  3. dir requirement + dir present & non-empty  → ok (exit 0)
  4. dir requirement + dir absent               → FAIL (exit 1)
  5. graceful-skip dataset + data absent        → warn only (exit 0)
  6. absolute "/data/..." string                → not a requirement (exit 0)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent
          / "scripts" / "check_test_data_requirements.py")


def _build_repo(tmp: Path, test_source: str) -> Path:
    """Minimal fake repo: scripts/ + tests/fake_test.rs with *test_source*."""
    (tmp / "scripts").mkdir()
    tests_dir = tmp / "tests"
    tests_dir.mkdir()
    (tests_dir / "fake_test.rs").write_text(test_source, encoding="utf-8")
    return tmp


def _run(repo_root: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root)],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_file_requirement_present_ok():
    src = 'const P: &str = "data/lp_problems_hard/neos.QPS";\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        d = repo / "data" / "lp_problems_hard"
        d.mkdir(parents=True)
        (d / "neos.QPS").write_text("NAME NEOS\n")
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "FAIL" not in out


def test_file_requirement_absent_fails():
    src = 'const P: &str = "data/lp_problems_hard/neos.QPS";\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        (repo / "data" / "lp_problems_hard").mkdir(parents=True)
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    assert "data/lp_problems_hard/neos.QPS" in out
    assert "fake_test.rs" in out, f"missing referencing-source listing:\n{out}"


def test_dir_requirement_present_ok():
    src = 'let p = format!("data/lp_problems/{}.QPS", name);\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        d = repo / "data" / "lp_problems"
        d.mkdir(parents=True)
        (d / "afiro.QPS").write_text("NAME AFIRO\n")
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"


def test_dir_requirement_absent_fails():
    src = 'let p = format!("data/lp_problems/{}.QPS", name);\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    assert "data/lp_problems" in out


def test_graceful_skip_dataset_warns_only():
    src = 'const P: &str = "data/cblib_socp/classical_20_0.cbf";\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0 (graceful skip), got {code}.\nOutput:\n{out}"
    assert "warn" in out, f"expected coverage-gap warning:\n{out}"
    assert "FAIL" not in out


def test_absolute_path_string_is_not_a_requirement():
    src = 'let p = detect_csv_path("/data/qplib-nonconvex", None, root);\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "qplib-nonconvex" not in out


if __name__ == "__main__":
    tests = [
        test_file_requirement_present_ok,
        test_file_requirement_absent_fails,
        test_dir_requirement_present_ok,
        test_dir_requirement_absent_fails,
        test_graceful_skip_dataset_warns_only,
        test_absolute_path_string_is_not_a_requirement,
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
