#!/usr/bin/env python3
"""Ratchet direct matrix-storage field access while legacy code is migrated."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/csc_field_access_baseline.txt"
ROOTS = ("otspot-core/src", "otspot-io/src", "otspot-model/src")
DIRECT = re.compile(r"\.(col_ptr|row_ind|values|nrows|ncols)\b(?!\s*\()")


def counts(root: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    for base in ROOTS:
        for path in (root / base).rglob("*.rs"):
            if (
                path.name == "tests.rs"
                or path.name.startswith("tests_")
                or "/tests/" in f"/{path.relative_to(root)}/"
            ):
                continue
            lines = path.read_text().splitlines()
            test_start = next(
                (i for i, line in enumerate(lines) if re.match(r"^\s*mod\s+tests\b", line)),
                len(lines),
            )
            count = 0
            for line in lines[:test_start]:
                code = line.split("//", 1)[0]
                count += len(DIRECT.findall(code))
            if count:
                found[str(path.relative_to(root))] = count
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
        rows = ["# direct storage-field accesses; counts may decrease but not grow"]
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
