#!/usr/bin/env python3
"""Ratchet long Rust functions: no new giants and no growth of legacy giants."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/function_size_baseline.txt"
LIMIT = 220
ROOTS = ("otspot-core/src", "otspot-num/src", "otspot-ir/src", "otspot-io/src", "otspot-model/src")
FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)")


def functions(root: Path) -> dict[tuple[str, str, int], int]:
    result = {}
    for base in ROOTS:
        for path in (root / base).rglob("*.rs"):
            if path.name == "tests.rs" or "tests" in path.parts[path.parts.index("src") + 1:-1]:
                continue
            lines = path.read_text().splitlines()
            test_start = next((i for i, line in enumerate(lines) if re.match(r"^\s*mod\s+tests\b", line)), len(lines))
            i = 0
            while i < test_start:
                match = FN.match(lines[i])
                if not match:
                    i += 1
                    continue
                start, depth, opened, j = i, 0, False, i
                in_block = False
                while j < test_start:
                    line = lines[j]
                    code = ""
                    k = 0
                    while k < len(line):
                        if in_block:
                            end = line.find("*/", k)
                            if end < 0:
                                k = len(line)
                                continue
                            in_block, k = False, end + 2
                        elif line.startswith("/*", k):
                            in_block, k = True, k + 2
                        elif line.startswith("//", k):
                            break
                        elif line[k] == '"':
                            k += 1
                            while k < len(line):
                                if line[k] == "\\":
                                    k += 2
                                elif line[k] == '"':
                                    k += 1
                                    break
                                else:
                                    k += 1
                        else:
                            code += line[k]
                            k += 1
                    depth += code.count("{") - code.count("}")
                    opened |= "{" in code
                    if opened and depth <= 0:
                        break
                    j += 1
                key = (str(path.relative_to(root)), match.group(1), start + 1)
                result[key] = j - start + 1
                i = j + 1
    return result


def read_baseline() -> dict[tuple[str, str, int], int]:
    rows = {}
    if BASELINE.exists():
        for line in BASELINE.read_text().splitlines():
            if line and not line.startswith("#"):
                path, name, start, size = line.split("\t")
                rows[(path, name, int(start))] = int(size)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    current = {key: size for key, size in functions(ROOT).items() if size > LIMIT}
    if args.update:
        lines = [f"# functions over {LIMIT} lines; growth is forbidden, shrink/removal is allowed"]
        lines += [f"{p}\t{n}\t{s}\t{size}" for (p, n, s), size in sorted(current.items())]
        BASELINE.write_text("\n".join(lines) + "\n")
        print(f"[function-size] wrote {BASELINE} ({len(current)} legacy exceptions)")
        return 0
    baseline = read_baseline()
    failures = []
    for key, size in current.items():
        allowed = baseline.get(key)
        if allowed is None:
            failures.append(f"new oversized function {key}: {size} > {LIMIT}")
        elif size > allowed:
            failures.append(f"function grew {key}: {size} > baseline {allowed}")
    if failures:
        print("[function-size] violations:", file=sys.stderr)
        print("\n".join(f"  - {x}" for x in failures), file=sys.stderr)
        return 1
    print(f"[function-size] OK: no new/grown functions over {LIMIT} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
