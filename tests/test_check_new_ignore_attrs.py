"""Unit tests for scripts/lib/check_new_ignore_attrs.py.

The gate compares a committed inventory against `cargo nextest list`. These
tests exercise the JSON parsing/sorting, inventory rendering, and drift
comparison without invoking cargo: `list_testcases()` is fed a canned
nextest JSON string through the `_nextest_list_output` seam, and the
higher-level drift tests stub `list_testcases()` directly. rustc's
resolution of cfg_attr predicates and macro expansion is the compiler's
job and is verified separately end-to-end.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lib" / "check_new_ignore_attrs.py"

_spec = importlib.util.spec_from_file_location("check_new_ignore_attrs", SCRIPT)
gate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["check_new_ignore_attrs"] = gate
_spec.loader.exec_module(gate)

TARGET = "host-default"
VERSIONS = "nextest=0.9.137 rustc=1.95.0"
# (binary-id, test-name, ignored)
BASE_ROWS = [
    ("otspot::diag_x", "heavy_case", True),
    ("otspot::diag_x", "quick_case", False),
    ("otspot-core", "mod::unit_a", False),
    ("otspot-core", "mod::unit_b", False),
]

# Real implementations captured before any test stubs a seam, so each test
# starts from a pristine module regardless of run order.
_REAL = {
    "tool_versions": gate.tool_versions,
    "list_testcases": gate.list_testcases,
    "_nextest_list_output": gate._nextest_list_output,
    "subprocess_run": gate.subprocess.run,
}


def _reset():
    gate.tool_versions = _REAL["tool_versions"]
    gate.list_testcases = _REAL["list_testcases"]
    gate._nextest_list_output = _REAL["_nextest_list_output"]
    gate.subprocess.run = _REAL["subprocess_run"]


def _nextest_json(rows) -> str:
    """Build a nextest-list-shaped JSON string from (bid, name, ignored)."""
    suites: dict = {}
    for bid, name, ig in rows:
        suite = suites.setdefault(bid, {"binary-id": bid, "testcases": {}})
        suite["testcases"][name] = {"ignored": ig}
    return json.dumps({"rust-suites": suites})


def _install(rows):
    # Drive the REAL list_testcases via the subprocess seam (canned JSON), so
    # the drift tests also exercise JSON parsing and sorting.
    _reset()
    gate.tool_versions = lambda: VERSIONS
    gate._nextest_list_output = lambda: _nextest_json(rows)


def _run(argv_tail, inventory: Path) -> tuple[int, str]:
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        code = gate.main(["prog", "--inventory", str(inventory), *argv_tail])
    return code, err.getvalue() + out.getvalue()


def _committed(rows, path: Path) -> None:
    path.write_text(gate.render_inventory(sorted(rows), TARGET, VERSIONS))


# ---------------------------------------------------------------------------
# list_testcases(): subprocess seam, JSON key access, sorting  (P2-1)
# ---------------------------------------------------------------------------

def test_list_testcases_parses_and_sorts_json():
    # Feed UNSORTED suites/testcases; expect sorted rows with correct flags.
    _reset()
    gate._nextest_list_output = lambda: _nextest_json(list(reversed(BASE_ROWS)))
    try:
        rows = gate.list_testcases()
    except gate.NextestError as e:
        raise AssertionError(f"list_testcases raised instead of parsing: {e}")
    assert rows == sorted(BASE_ROWS)


def test_list_testcases_missing_rust_suites_key_is_env_error():
    _reset()
    gate._nextest_list_output = lambda: json.dumps({"unexpected": {}})
    try:
        gate.list_testcases()
    except gate.NextestError:
        return
    raise AssertionError("expected NextestError on missing rust-suites key")


def test_nextest_nonzero_returncode_raises_env_error():
    class _Proc:
        returncode = 101
        stdout = ""
        stderr = "error[E0432]: build failed"

    _reset()
    gate.subprocess.run = lambda *a, **k: _Proc()
    try:
        gate._nextest_list_output()
        raised = False
    except gate.NextestError:
        raised = True
    except Exception:
        raised = False
    assert raised, "nonzero nextest exit must raise NextestError"


def test_nextest_missing_binary_raises_env_error():
    def _boom(*a, **k):
        raise FileNotFoundError("cargo")

    _reset()
    gate.subprocess.run = _boom
    try:
        gate._nextest_list_output()
        raised = False
    except gate.NextestError:
        raised = True
    except Exception:
        raised = False  # a raw FileNotFoundError is not an acceptable failure
    assert raised


# ---------------------------------------------------------------------------
# Drift comparison
# ---------------------------------------------------------------------------

def test_matching_inventory_passes():
    _install(BASE_ROWS)
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        _committed(BASE_ROWS, inv)
        code, out = _run([], inv)
    assert code == gate.EXIT_OK, out


def test_ignore_flip_is_detected():
    _install([(b, n, True if n == "quick_case" else ig) for b, n, ig in BASE_ROWS])
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        _committed(BASE_ROWS, inv)
        code, out = _run([], inv)
    assert code == gate.EXIT_DRIFT, out
    assert "quick_case" in out


def test_added_test_is_detected():
    _install(BASE_ROWS + [("otspot-core", "mod::unit_c", False)])
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        _committed(BASE_ROWS, inv)
        code, out = _run([], inv)
    assert code == gate.EXIT_DRIFT, out
    assert "unit_c" in out


def test_removed_active_test_is_detected():
    _install([r for r in BASE_ROWS if r[1] != "quick_case"])
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        _committed(BASE_ROWS, inv)
        code, out = _run([], inv)
    assert code == gate.EXIT_DRIFT, out
    assert "quick_case" in out


def test_toolchain_version_change_alone_does_not_drift():
    # Committed with old versions, live with new versions, same tests: OK.
    _install(BASE_ROWS)
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        inv.write_text(gate.render_inventory(sorted(BASE_ROWS), TARGET, "nextest=0.0.1 rustc=0.0.1"))
        code, out = _run([], inv)
    assert code == gate.EXIT_OK, out


def test_missing_inventory_is_drift_with_hint():
    _install(BASE_ROWS)
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "absent.txt"
        code, out = _run([], inv)
    assert code == gate.EXIT_DRIFT, out
    assert "--update" in out


def test_update_writes_inventory_then_passes():
    _install(BASE_ROWS)
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        code, _ = _run(["--update"], inv)
        assert code == gate.EXIT_OK
        assert inv.exists()
        code2, out2 = _run([], inv)
    assert code2 == gate.EXIT_OK, out2


def test_nextest_env_error_is_exit_2():
    def boom():
        raise gate.NextestError("cargo/nextest not found")

    _reset()
    gate.host_target = lambda: TARGET
    gate.tool_versions = lambda: VERSIONS
    gate._nextest_list_output = boom  # propagates through real list_testcases
    with tempfile.TemporaryDirectory() as td:
        inv = Path(td) / "inv.txt"
        _committed(BASE_ROWS, inv)
        code, out = _run([], inv)
    assert code == gate.EXIT_ENV, out
    assert "not found" in out


# ---------------------------------------------------------------------------
# Inventory rendering (white-box)
# ---------------------------------------------------------------------------

def test_render_marks_ignored_and_keeps_active():
    text = gate.render_inventory(sorted(BASE_ROWS), TARGET, VERSIONS)
    assert "otspot::diag_x\theavy_case\tignored" in text
    assert "otspot::diag_x\tquick_case\tactive" in text  # active tests kept
    assert f"target={TARGET}" in text and "profile=release" in text
    assert f"{gate.DIAGNOSTIC_PREFIX} {VERSIONS}" in text  # versions recorded
    assert "1 ignored" in text


def test_render_is_sorted_and_deterministic():
    a = gate.render_inventory(sorted(reversed(BASE_ROWS)), TARGET, VERSIONS)
    b = gate.render_inventory(sorted(BASE_ROWS), TARGET, VERSIONS)
    assert a == b
    body = [ln for ln in a.splitlines() if not ln.startswith("#")]
    assert body == sorted(body)


def test_comparable_strips_only_diagnostic_lines():
    text = gate.render_inventory(sorted(BASE_ROWS), TARGET, VERSIONS)
    comp = gate._comparable(text)
    assert gate.DIAGNOSTIC_PREFIX not in comp
    assert "profile=release" in comp  # config line still compared
    assert "heavy_case\tignored" in comp
    # testcase count is derived from the text, not a hardcoded header length
    assert gate._testcase_count(text) == len(BASE_ROWS)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
