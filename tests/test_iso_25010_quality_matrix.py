import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "iso-25010-quality-matrix.md"

CHARACTERISTICS = [
    "Functional suitability",
    "Performance efficiency",
    "Compatibility",
    "Usability",
    "Reliability",
    "Security",
    "Maintainability",
    "Portability",
]

REQUIRED_REFERENCES = [
    ".github/workflows/ci.yml",
    ".github/workflows/audit.yml",
    ".github/workflows/test-heavy.yml",
    ".config/nextest.toml",
    "scripts/pre-merge-audit.sh",
    "scripts/check_file_size.sh",
    "scripts/check_comment_block_size.sh",
    "scripts/check_comment_ratio.sh",
    "scripts/lib/check_memo_grep.sh",
    "tests/test_check_data_coverage.py",
]

COMMAND_RE = re.compile(r"`(cargo nextest run [^`]+)`")
SELECTOR_EXPR_RE = re.compile(r"-E\s+'([^']+)'")
NEXT_RUN_ONLY_ARGS = {"--no-fail-fast", "--test-threads"}
HEAVY_SURVEILLANCE_COUNT = 11
EXPECTED_GATE_SELECTOR_EXPRESSIONS = {
    "Functional suitability": [],
    "Performance efficiency": [
        "binary(memory_regression) | binary(diag_bench_timeout_honored) | binary(diag_dfl001_postsolve_speedup)",
        "binary(diag_lp_simplex_stall_sentinel)",
    ],
    "Compatibility": [],
    "Usability": [
        "test(model_api) | binary(api_correctness) | binary(solver_wide_api_contract)",
    ],
    "Reliability": [],
    "Security": [],
    "Maintainability": [],
    "Portability": [],
}
EXPECTED_NONEMPTY_BINARY_GATE_SELECTORS = {
    "Performance efficiency": [
        {
            "selector": "binary(memory_regression)",
            "command": "cargo nextest run --release --features parallel -E 'binary(memory_regression)' --test-threads 3",
        },
        {
            "selector": "binary(diag_bench_timeout_honored)",
            "command": "cargo nextest run --release --features parallel -E 'binary(diag_bench_timeout_honored)' --test-threads 3",
        },
        {
            "selector": "binary(diag_dfl001_postsolve_speedup)",
            "command": "cargo nextest run --release --features parallel -E 'binary(diag_dfl001_postsolve_speedup)' --test-threads 3",
        },
        {
            "selector": "binary(diag_lp_simplex_stall_sentinel)",
            "command": "cargo nextest run --release --features parallel --profile heavy --run-ignored all -E 'binary(diag_lp_simplex_stall_sentinel)' --test-threads 3",
        },
    ],
    "Usability": [
        {
            "selector": "binary(api_correctness)",
            "command": "cargo nextest run --release -E 'binary(api_correctness)'",
        },
        {
            "selector": "binary(solver_wide_api_contract)",
            "command": "cargo nextest run --release -E 'binary(solver_wide_api_contract)'",
        },
    ],
}


def _nextest_list_command(command):
    args = shlex.split(command)
    assert args[:3] == ["cargo", "nextest", "run"], command
    list_args = ["cargo", "nextest", "list"]
    index = 3
    while index < len(args):
        arg = args[index]
        if arg in NEXT_RUN_ONLY_ARGS:
            index += 2 if arg == "--test-threads" else 1
            continue
        list_args.append(arg)
        index += 1
    return list_args


def _listed_test_count(command):
    result = subprocess.run(
        _nextest_list_command(command),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return len(_parse_nextest_list_tests(result.stdout))


def _listed_tests(command):
    result = subprocess.run(
        _nextest_list_command(command),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_nextest_list_tests(result.stdout)


def _parse_nextest_list_tests(output):
    tests = []
    for line in output.splitlines():
        name = line.strip()
        if not name or name.startswith("package:") or name.endswith(":"):
            continue
        if line[:1].isspace():
            tests.append(name)
        elif " " in name and "::" in name.split()[0]:
            tests.append(name)
    return tests


def _documented_nextest_selector_commands(text):
    for command in COMMAND_RE.findall(text):
        if " -E " in command:
            yield command


def _selector_expr(command):
    match = SELECTOR_EXPR_RE.search(command)
    assert match, command
    return match.group(1)


def _characteristic_row(text, name):
    prefix = f"| {name} |"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(name)


def _characteristic_selector_exprs(text, name):
    row = _characteristic_row(text, name)
    return [_selector_expr(command) for command in _documented_nextest_selector_commands(row)]


def test_iso_25010_matrix_covers_product_quality_characteristics():
    text = MATRIX.read_text(encoding="utf-8")
    for name in CHARACTERISTICS:
        assert f"| {name} |" in text, name


def test_iso_25010_matrix_references_existing_repo_gates():
    text = MATRIX.read_text(encoding="utf-8")
    for rel in REQUIRED_REFERENCES:
        assert rel in text, rel
        assert (ROOT / rel).exists(), rel


def test_iso_25010_matrix_keeps_gates_actionable():
    text = MATRIX.read_text(encoding="utf-8")
    assert "certification claim" in text
    assert "Do not add ISO process artifacts" in text
    assert "cargo nextest run" in text
    assert "cargo clippy" in text


def test_nextest_list_parser_counts_root_level_tests_without_counting_headers():
    sample = """
package: otspot-dev
diag_lp_simplex_stall_sentinel:
    detects_stall_at_root
    honors_timeout_budget
    simplex::phase_one_recovers
otspot::diag_lp_simplex_stall_sentinel root_level_from_full_output
"""
    expected = [
        "detects_stall_at_root",
        "honors_timeout_budget",
        "simplex::phase_one_recovers",
        "otspot::diag_lp_simplex_stall_sentinel root_level_from_full_output",
    ]
    assert _parse_nextest_list_tests(sample) == expected
    assert len(_parse_nextest_list_tests(sample)) == 4

def test_iso_25010_nextest_release_gate_selectors_are_nonempty():
    text = MATRIX.read_text(encoding="utf-8")
    commands = list(_documented_nextest_selector_commands(text))
    assert commands, "expected documented nextest selectors"
    for command in commands:
        assert _listed_test_count(command) > 0, command


def test_iso_25010_expected_gate_selector_expressions_match_documented_rows_exactly():
    text = MATRIX.read_text(encoding="utf-8")
    assert set(EXPECTED_GATE_SELECTOR_EXPRESSIONS) == set(CHARACTERISTICS)
    for characteristic, expected_exprs in EXPECTED_GATE_SELECTOR_EXPRESSIONS.items():
        actual_exprs = _characteristic_selector_exprs(text, characteristic)
        assert actual_exprs == expected_exprs, (
            f"{characteristic}: expected {expected_exprs}, got {actual_exprs}"
        )


def test_iso_25010_expected_binary_gate_selectors_are_documented_and_nonempty():
    text = MATRIX.read_text(encoding="utf-8")
    for characteristic, selectors in EXPECTED_NONEMPTY_BINARY_GATE_SELECTORS.items():
        selector_exprs = _characteristic_selector_exprs(text, characteristic)
        for entry in selectors:
            selector = entry["selector"]
            assert any(selector in expr for expr in selector_exprs), (
                f"{characteristic}: {selector}"
            )
            tests = _listed_tests(entry["command"])
            assert tests, f"{characteristic}: {selector}"


def test_iso_25010_heavy_surveillance_selector_count_is_pinned():
    text = MATRIX.read_text(encoding="utf-8")
    commands = [
        command
        for command in _documented_nextest_selector_commands(text)
        if "--run-ignored" in command
    ]
    assert commands, "expected documented ignored/heavy nextest selectors"
    for command in commands:
        assert _listed_test_count(command) == HEAVY_SURVEILLANCE_COUNT, command


if __name__ == "__main__":
    test_iso_25010_matrix_covers_product_quality_characteristics()
    test_iso_25010_matrix_references_existing_repo_gates()
    test_iso_25010_matrix_keeps_gates_actionable()
    test_nextest_list_parser_counts_root_level_tests_without_counting_headers()
    test_iso_25010_nextest_release_gate_selectors_are_nonempty()
    test_iso_25010_expected_gate_selector_expressions_match_documented_rows_exactly()
    test_iso_25010_expected_binary_gate_selectors_are_documented_and_nonempty()
    test_iso_25010_heavy_surveillance_selector_count_is_pinned()
    print("iso 25010 quality matrix check: OK")
