#!/usr/bin/env python3
"""Ratchet direct matrix-storage field access while legacy code is migrated.

Tracks two distinct bypasses of the `CscMatrix`/`SparseVec` constructor API:
dot-access on the storage fields (`.col_ptr`, `.values`, ...) and direct
struct-literal construction (`CscMatrix { col_ptr: ..., ... }`), which dot-access
regexes alone cannot see. Both count against the same per-file ratchet.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/csc_field_access_baseline.txt"
ROOTS = ("otspot-core/src", "otspot-io/src", "otspot-model/src", "otspot-dev/src")
DIRECT = re.compile(r"\.(col_ptr|row_ind|values|nrows|ncols)\b(?!\s*\()")

# The types' own definition module is where `Self { .. }`/`TypeName { .. }`
# construction is legitimate; struct-literal detection never applies there.
TYPE_DEFINITION_ROOT = "otspot-num/src/sparse"

STRUCT_LITERAL = re.compile(r"\b(CscMatrix|SparseVec)\s*\{")
# rustfmt always keeps a struct/impl/fn-signature's opening `{` attached to the
# preceding token, so a bare type name at end-of-line whose very next physical
# line opens with `{` is likewise a construction site under canonical (fmt
# --check-clean) formatting. Caught here as defense-in-depth for code that
# bypasses `cargo fmt` locally; `cargo fmt --all -- --check` is the CI gate
# that keeps this the only brace-placement this scanner needs to handle.
TYPE_AT_EOL = re.compile(r"\b(CscMatrix|SparseVec)\s*$")
# A `TypeName {` match is a signature/definition, not a construction
# expression, when it's a function/closure return type (preceded by `->`) or
# the line opens an `impl`/`struct` item.
SIGNATURE_OR_DEFINITION = re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(impl|struct)\b")


def _is_construction_site(code: str, match_start: int) -> bool:
    if code[:match_start].rstrip().endswith("->"):
        return False
    return not SIGNATURE_OR_DEFINITION.match(code)


def _construction_matches(code_lines: list[str], i: int) -> int:
    code = code_lines[i]
    count = sum(
        1 for m in STRUCT_LITERAL.finditer(code) if _is_construction_site(code, m.start())
    )
    if i + 1 < len(code_lines):
        m = TYPE_AT_EOL.search(code)
        if m and code_lines[i + 1].lstrip().startswith("{"):
            count += 1 if _is_construction_site(code, m.start()) else 0
    return count


def counts(root: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    for base in ROOTS:
        for path in (root / base).rglob("*.rs"):
            rel = str(path.relative_to(root))
            if (
                path.name == "tests.rs"
                or path.name.startswith("tests_")
                or "/tests/" in f"/{rel}/"
            ):
                continue
            skip_construction = rel.replace("\\", "/").startswith(f"{TYPE_DEFINITION_ROOT}/")
            lines = path.read_text().splitlines()
            test_start = next(
                (i for i, line in enumerate(lines) if re.match(r"^\s*mod\s+tests\b", line)),
                len(lines),
            )
            code_lines = [line.split("//", 1)[0] for line in lines[:test_start]]
            count = 0
            for i, code in enumerate(code_lines):
                count += len(DIRECT.findall(code))
                if not skip_construction:
                    count += _construction_matches(code_lines, i)
            if count:
                found[rel] = count
    return found


def read_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    result = {}
    for line in BASELINE.read_text().splitlines():
        if line and not line.startswith("#"):
            path, count = line.split("\t")
            result[path] = int(count)
    return result


def violations(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    return [
        f"{path}: {count} > baseline {baseline.get(path, 0)}"
        for path, count in current.items()
        if count > baseline.get(path, 0)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    current = counts(ROOT)
    if args.update:
        rows = ["# direct storage-field accesses + struct-literal construction; counts may decrease but not grow"]
        rows += [f"{path}\t{count}" for path, count in sorted(current.items())]
        BASELINE.write_text("\n".join(rows) + "\n")
        print(f"[encapsulation] wrote {BASELINE} ({sum(current.values())} accesses)")
        return 0
    baseline = read_baseline()
    failures = violations(current, baseline)
    if failures:
        print("[encapsulation] direct field access grew:", file=sys.stderr)
        print("\n".join(f"  - {item}" for item in failures), file=sys.stderr)
        return 1
    print(f"[encapsulation] OK: {sum(current.values())} accesses (no growth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
