#!/usr/bin/env python3
"""Architecture-shape ratchet for the layered solver design."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_INTERNAL = {"otspot-num": set()}
FOUNDATION_MAX_LINES = 1600
MODULE_ROOT_MAX_LINES = 200
LEGACY_FACADE_MAX_LINES = 100
FACADE_FORBIDDEN_PATHS = (
    "crate::sparse::",
    "crate::linalg::",
    "crate::error::SolverError",
    "crate::SolverError",
)
OWNERS = {
    "pub struct CscMatrix": "otspot-num/src/sparse/csc.rs",
    "pub struct SparseVec": "otspot-num/src/sparse/vec.rs",
    "pub trait CscMatrixView": "otspot-num/src/sparse/view.rs",
    "pub trait KktBackend": "otspot-num/src/kkt.rs",
}


def _internal_dependencies(root: Path, crate: str) -> set[str]:
    data = tomllib.loads((root / crate / "Cargo.toml").read_text())
    return {name for name in data.get("dependencies", {}) if name.startswith("otspot-")}


def check(root: Path = ROOT) -> list[str]:
    failures: list[str] = []
    for crate, allowed in ALLOWED_INTERNAL.items():
        forbidden = _internal_dependencies(root, crate) - allowed
        if forbidden:
            failures.append(f"{crate} forbidden dependencies: {', '.join(sorted(forbidden))}")

    for crate in ALLOWED_INTERNAL:
        for source in (root / crate / "src").rglob("*.rs"):
            for line in source.read_text().splitlines():
                code = line.split("//", 1)[0]
                for forbidden in ("otspot_core", "otspot_io", "otspot_model", "otspot_dev"):
                    if forbidden in code:
                        failures.append(f"{source.relative_to(root)} references {forbidden}")

    # Implementation ownership: canonical primitives must have one physical home.
    sources = list((root / "otspot-num/src").rglob("*.rs"))
    sources += list((root / "otspot-core/src").rglob("*.rs"))
    for declaration, owner in OWNERS.items():
        found = [
            str(path.relative_to(root))
            for path in sources
            if declaration in path.read_text()
        ]
        if found != [owner]:
            failures.append(f"{declaration} owner must be {owner}; found {found}")

    # Legacy namespaces must remain thin compatibility facades.
    forbidden_dirs = [root / "otspot-core/src/linalg", root / "otspot-core/src/sparse/csc.rs"]
    for path in forbidden_dirs:
        if path.exists():
            failures.append(f"legacy implementation path reintroduced: {path.relative_to(root)}")
    for relative in ("otspot-core/src/linalg.rs", "otspot-core/src/sparse/mod.rs"):
        path = root / relative
        if len(path.read_text().splitlines()) > LEGACY_FACADE_MAX_LINES:
            failures.append(f"legacy facade exceeds {LEGACY_FACADE_MAX_LINES} lines: {relative}")

    # otspot-core internals must depend on otspot-num directly; the legacy
    # crate::sparse / crate::linalg / crate::error::SolverError paths are a
    # public facade for downstream crates only, not for internal use.
    for source in (root / "otspot-core/src").rglob("*.rs"):
        for lineno, line in enumerate(source.read_text().splitlines(), start=1):
            code = line.split("//", 1)[0]
            for forbidden in FACADE_FORBIDDEN_PATHS:
                if forbidden in code:
                    failures.append(
                        f"{source.relative_to(root)}:{lineno} uses legacy facade path "
                        f"{forbidden!r}; depend on otspot_num directly"
                    )

    # Foundation files and module roots should stay cohesive.
    for crate in ("otspot-num",):
        for path in (root / crate / "src").rglob("*.rs"):
            lines = len(path.read_text().splitlines())
            limit = MODULE_ROOT_MAX_LINES if path.name in {"lib.rs", "mod.rs"} else FOUNDATION_MAX_LINES
            if lines > limit:
                failures.append(f"{path.relative_to(root)} has {lines} lines (limit {limit})")
    return failures


def main() -> int:
    failures = check()
    if failures:
        print("[architecture] violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("[architecture] OK: dependency, ownership, facade, and cohesion rules pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
