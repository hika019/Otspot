import json
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


def _parse_nextest_list_json(stdout):
    stdout = stdout.strip()
    if not stdout:
        return []
    try:
        payloads = [json.loads(stdout)]
    except json.JSONDecodeError:
        payloads = [json.loads(line) for line in stdout.splitlines() if line.strip()]

    tests = []
    for payload in payloads:
        for suite in payload.get("rust-suites", {}).values():
            tests.extend(suite.get("testcases", {}).keys())
    return tests


def _parse_nextest_list_human(stdout):
    tests = []
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped == "(no tests)":
            continue
        if not raw_line[:1].isspace():
            continue
        if stripped.startswith(("bin: ", "cwd: ", "build platform: ")):
            continue
        if stripped.endswith(" (skipped)"):
            stripped = stripped[: -len(" (skipped)")]
        tests.append(stripped)
    return tests


def _list_nextest_tests(command):
    base_args = _nextest_list_command(command)
    try:
        result = subprocess.run(
            [*base_args, "--message-format", "json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return _parse_nextest_list_json(result.stdout)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if "--message-format" not in stderr:
            raise

    result = subprocess.run(
        [*base_args, "--message-format", "human"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_nextest_list_human(result.stdout)


def _listed_test_count(command):
    return len(_list_nextest_tests(command))


def _listed_tests(command):
    return _list_nextest_tests(command)


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


def test_nextest_json_listing_parser_counts_root_level_names():
    stdout = """
{
  "test-count": 3,
  "rust-suites": {
    "otspot::api_correctness": {
      "testcases": {
        "smoke_test": {"ignored": false},
        "module::nested_case": {"ignored": false}
      }
    },
    "otspot::solver_wide_api_contract": {
      "testcases": {"contract_case": {"ignored": false}}
    }
  }
}
"""
    assert _parse_nextest_list_json(stdout) == [
        "smoke_test",
        "module::nested_case",
        "contract_case",
    ]


def test_nextest_human_listing_parser_ignores_headers_and_keeps_root_level_names():
    stdout = """
nextest-runner:
    smoke_test
    module::nested_case
nextest-runner::integration:
    basic::test_list_tests
nextest-runner::bin/passthrough:
    (no tests)
"""
    assert _parse_nextest_list_human(stdout) == [
        "smoke_test",
        "module::nested_case",
        "basic::test_list_tests",
    ]


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
    test_nextest_json_listing_parser_counts_root_level_names()
    test_nextest_human_listing_parser_ignores_headers_and_keeps_root_level_names()
    test_iso_25010_nextest_release_gate_selectors_are_nonempty()
    test_iso_25010_expected_gate_selector_expressions_match_documented_rows_exactly()
    test_iso_25010_expected_binary_gate_selectors_are_documented_and_nonempty()
    test_iso_25010_heavy_surveillance_selector_count_is_pinned()
    print("iso 25010 quality matrix check: OK")
