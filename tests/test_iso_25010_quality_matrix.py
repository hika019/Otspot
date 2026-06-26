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


if __name__ == "__main__":
    test_iso_25010_matrix_covers_product_quality_characteristics()
    test_iso_25010_matrix_references_existing_repo_gates()
    test_iso_25010_matrix_keeps_gates_actionable()
    print("iso 25010 quality matrix check: OK")
