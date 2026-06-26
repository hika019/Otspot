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
NEXT_RUN_ONLY_ARGS = {"--no-fail-fast", "--test-threads"}
HEAVY_SURVEILLANCE_COUNT = 11


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
    return sum(1 for line in result.stdout.splitlines() if "::" in line)


def _documented_nextest_selector_commands(text):
    for command in COMMAND_RE.findall(text):
        if " -E " in command:
            yield command


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


def test_iso_25010_nextest_release_gate_selectors_are_nonempty():
    text = MATRIX.read_text(encoding="utf-8")
    commands = list(_documented_nextest_selector_commands(text))
    assert commands, "expected documented nextest selectors"
    for command in commands:
        assert _listed_test_count(command) > 0, command


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
    test_iso_25010_nextest_release_gate_selectors_are_nonempty()
    test_iso_25010_heavy_surveillance_selector_count_is_pinned()
    print("iso 25010 quality matrix check: OK")
