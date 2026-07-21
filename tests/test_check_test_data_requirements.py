"""Sentinel tests for scripts/check_test_data_requirements.py.

Covers the core branches of the heavy-data alignment guard (PR #25 review item 4):
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


# ---------------------------------------------------------------------
# PR #25 review ("Check formatted data files instead of only their
# directory"): a `format!("data/<dir>/{name}.QPS", name)` file-path helper
# called with literal names (`netlib_lp_extra.rs`, `diag_liswet_basin.rs`,
# `qp_clarabel_extra_assertions.rs`, `diag_lp_simplex_stall_sentinel.rs`,
# `clarabel_cross_check.rs`, `israel_objective_mismatch.rs`,
# `diag_bounded_grow_feasibility.rs` style) must resolve to the *exact* file
# each call site needs, not just its containing directory.
# ---------------------------------------------------------------------

_FORMAT_HELPER_SRC = (
    'fn solve_and_check(name: &str, expected_obj: f64, max_secs: u64) {\n'
    '    let path_str = format!("data/lp_problems/{}.QPS", name);\n'
    '    let path = std::path::Path::new(&path_str);\n'
    '    assert!(path.exists());\n'
    '}\n'
    '\n'
    '#[test]\n'
    'fn t() {\n'
    '    solve_and_check("afiro", 464.75, 30);\n'
    '}\n'
)


def test_format_helper_named_file_present_ok():
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), _FORMAT_HELPER_SRC)
        d = repo / "data" / "lp_problems"
        d.mkdir(parents=True)
        (d / "afiro.QPS").write_text("NAME AFIRO\n")
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "FAIL" not in out


def test_format_helper_named_file_absent_fails_even_if_dir_nonempty():
    """Confirmed repro of the reviewed bug: the directory is non-empty (a
    *different* file is present), which the pre-fix directory-only check
    would have accepted; the specific file the call site needs is missing
    and must be a hard failure."""
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), _FORMAT_HELPER_SRC)
        d = repo / "data" / "lp_problems"
        d.mkdir(parents=True)
        (d / "some_other_problem.QPS").write_text("NAME OTHER\n")
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    assert "data/lp_problems/afiro.QPS" in out, (
        f"must name the specific missing file, not just the directory:\n{out}"
    )


def test_format_helper_positional_and_inline_placeholder_both_detected():
    """Both `format!("...{}...", name)` (positional) and
    `format!("...{name}...")` (inline capture) must resolve."""
    src = (
        'fn parse(name: &str) -> Option<()> {\n'
        '    let path = format!("data/lp_problems/{name}.QPS");\n'
        '    None\n'
        '}\n'
        '\n'
        '#[test]\n'
        'fn t() {\n'
        '    parse("neos");\n'
        '}\n'
    )
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    assert "data/lp_problems/neos.QPS" in out


def test_format_helper_for_loop_array_is_detected():
    """`for name in ["a", "b"] { helper(name) }` (conic_qps_integration.rs
    style: literal names passed via a loop variable, not a direct literal
    call-site argument) must resolve each array element to a file."""
    src = (
        'fn try_load(name: &str) -> Option<()> {\n'
        '    let p = format!("data/maros_meszaros/{name}");\n'
        '    None\n'
        '}\n'
        '\n'
        '#[test]\n'
        'fn t() {\n'
        '    for name in ["HS21.QPS", "TAME.QPS"] {\n'
        '        let _ = try_load(name);\n'
        '    }\n'
        '}\n'
    )
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        d = repo / "data" / "maros_meszaros"
        d.mkdir(parents=True)
        (d / "HS21.QPS").write_text("NAME HS21\n")
        # TAME.QPS deliberately absent.
        code, out = _run(repo)
    assert code == 1, f"Expected exit 1, got {code}.\nOutput:\n{out}"
    fail_lines = [line for line in out.splitlines() if "[FAIL]" in line]
    assert not any("HS21.QPS" in line for line in fail_lines), (
        f"HS21.QPS is present and must not be reported missing:\n{out}"
    )
    assert any("TAME.QPS" in line for line in fail_lines), (
        f"TAME.QPS is absent and must be reported missing:\n{out}"
    )


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


def test_crate_relative_literal_is_detected():
    """`.join("../data/cblib_socp")` (crate-relative, cbf_cblib_integration.rs
    style) must be resolved — the old regex missed it because of the `../`."""
    src = (
        'fn cblib_dir() -> PathBuf {\n'
        '    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/cblib_socp")\n'
        '}\n'
    )
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    # cblib_socp is graceful-skip → warn, but it MUST be seen (not silently dropped).
    assert "data/cblib_socp" in out, f"crate-relative ref not detected:\n{out}"
    assert code == 0, f"cblib_socp is skip-tolerant, expected exit 0:\n{out}"


def test_const_join_helper_is_detected():
    """`const SUB = "lp_problems"; fn data_dir(s){..join("../data").join(s)}
    data_dir(SUB)` (presolve_correctness_sweep.rs style) must resolve to
    data/lp_problems.  This is the P1 regression: the pure-regex extractor
    could not see it, so drift in this dataset went undetected."""
    src = (
        'const LP_SUBDIR: &str = "lp_problems";\n'
        'fn data_root() -> PathBuf {\n'
        '    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")\n'
        '}\n'
        'fn data_dir(sub: &str) -> PathBuf { data_root().join(sub) }\n'
        'fn run() { let d = std::fs::read_dir(data_dir(LP_SUBDIR)).unwrap(); }\n'
    )
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        # data/lp_problems absent → must be a HARD failure (not skip-tolerant).
        code, out = _run(repo)
    assert "data/lp_problems" in out, f"const-join helper not resolved:\n{out}"
    assert code == 1, f"Expected exit 1 (hard missing lp_problems):\n{out}"


def test_metadata_substring_is_not_a_requirement():
    """A literal like "metadata/foo" must not be misread as a data/ ref."""
    src = 'const M: &str = "metadata/foo";\n'
    with tempfile.TemporaryDirectory() as td:
        repo = _build_repo(Path(td), src)
        code, out = _run(repo)
    assert code == 0, f"Expected exit 0, got {code}.\nOutput:\n{out}"
    assert "metadata" not in out


if __name__ == "__main__":
    tests = [
        test_file_requirement_present_ok,
        test_file_requirement_absent_fails,
        test_dir_requirement_present_ok,
        test_dir_requirement_absent_fails,
        test_format_helper_named_file_present_ok,
        test_format_helper_named_file_absent_fails_even_if_dir_nonempty,
        test_format_helper_positional_and_inline_placeholder_both_detected,
        test_format_helper_for_loop_array_is_detected,
        test_graceful_skip_dataset_warns_only,
        test_absolute_path_string_is_not_a_requirement,
        test_crate_relative_literal_is_detected,
        test_const_join_helper_is_detected,
        test_metadata_substring_is_not_a_requirement,
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
