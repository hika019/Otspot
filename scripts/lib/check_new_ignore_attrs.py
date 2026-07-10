#!/usr/bin/env python3
"""Guard against silently disabling or deleting tests, using rustc as the
ground truth for what "a test" and "ignored" mean.

`cargo nextest list` reports the real, post-expansion test set with an
`ignored` flag that reflects macro expansion, `#[cfg_attr(pred, ignore)]`
predicate evaluation, conditional compilation, and every test macro
(`#[test]`, `#[tokio::test]`, `proptest!`, ...). We normalise that into a
committed inventory (`tests/test_inventory.txt`) and fail if the live list
drifts from it. An author who adds, removes, or ignores a test regenerates
the inventory (`--update`); the change then appears as an inventory diff in
the PR, so both a silent disable and a silent delete become explicit,
reviewable edits.

Why not parse the Rust source: a regex/AST reader cannot evaluate a cfg
predicate -- `#[cfg_attr(windows, ignore)]` runs on Linux but a text
matcher reads it as ignored -- and mis-expands macros (repetition
matchers, second-arm `#[test]`, cross-file `#[macro_export]`,
dispatchers). Measured on this tree, the previous parser reported 22
ignored where rustc lists 28. Only the compiler is authoritative, so the
parser is gone.

Inventory scope: the full `(binary-id, test-name, status)` set, not just
the ignored ones -- the ignored subset alone cannot detect a silently
deleted active test. The list is pinned to one profile/feature/target
(recorded in the header) because `ignored` and test existence are
cfg-dependent; regenerating under a different configuration diffs on
purpose. `binary-id` embeds the integration-test file stem, so a file
rename diffs too -- also intentional, a rename is not a silent change. The
nextest/rustc versions are recorded in the header for diagnosis but
excluded from the comparison, so a toolchain bump alone does not force a
regenerate.
"""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

# Pinned to the workspace's authoritative must-pass test configuration
# (ci.yml's test-integration job): release profile + the `parallel` feature,
# which gates otspot-core/src/qp/tests/concurrent.rs. Where a build with
# these exact flags already exists (that CI job; pre-merge-audit.sh, which
# now passes --features parallel to its full-suite), `list` is a sub-second
# no-rebuild step. A target that has never built this feature set pays one
# recompile of otspot-core and its dependents (~50s) the first time.
NEXTEST_LIST_ARGS = [
    "cargo", "nextest", "list",
    "--release", "--features", "parallel",
    "--message-format", "json",
]
PROFILE_LABEL = "release"
FEATURES_LABEL = "parallel"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "tests" / "test_inventory.txt"

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_ENV = 2  # nextest missing, build failure, unreadable output

_UPDATE_HINT = "python3 scripts/lib/check_new_ignore_attrs.py --update"
# Header lines beginning with this prefix are diagnostic only: recorded on
# --update but stripped before the drift comparison, so a nextest/rustc
# version bump does not by itself fail the gate.
DIAGNOSTIC_PREFIX = "# tools:"


class NextestError(RuntimeError):
    """cargo/nextest could not produce a usable test list."""


def _tool_version(args: list[str]) -> str:
    """Second whitespace field of `<tool> --version` (the version number)."""
    try:
        out = subprocess.run(args, capture_output=True, text=True).stdout.split()
    except OSError:
        return "unknown"
    return out[1] if len(out) >= 2 else "unknown"


def tool_versions() -> str:
    nextest = _tool_version(["cargo", "nextest", "--version"])
    rustc = _tool_version(["rustc", "--version"])
    return f"nextest={nextest} rustc={rustc}"


def host_target() -> str:
    try:
        out = subprocess.run(
            ["rustc", "-vV"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise NextestError(f"could not run rustc to determine host target: {e}")
    for line in out.splitlines():
        if line.startswith("host: "):
            return line[len("host: ") :].strip()
    raise NextestError("rustc -vV did not report a host target")


def _nextest_list_output() -> str:
    """Raw stdout of `cargo nextest list` (subprocess seam, mockable)."""
    try:
        proc = subprocess.run(NEXTEST_LIST_ARGS, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise NextestError(f"cargo/nextest not found: {e}")
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-15:]
        raise NextestError(
            "`cargo nextest list` failed (build error or nextest missing):\n"
            + "\n".join(tail)
        )
    return proc.stdout


def list_testcases() -> list[tuple[str, str, bool]]:
    """Return sorted (binary-id, test-name, ignored) for the live test set."""
    try:
        data = json.loads(_nextest_list_output())
        suites = data["rust-suites"]
    except (json.JSONDecodeError, KeyError) as e:
        raise NextestError(f"could not parse `cargo nextest list` JSON: {e}")

    rows: list[tuple[str, str, bool]] = []
    for suite in suites.values():
        binary_id = suite["binary-id"]
        for name, case in (suite.get("testcases") or {}).items():
            rows.append((binary_id, name, bool(case.get("ignored"))))
    rows.sort()
    return rows


def render_inventory(
    rows: list[tuple[str, str, bool]], target: str, versions: str
) -> str:
    """Deterministic text: pinned-config header + one line per testcase."""
    ignored = sum(1 for _, _, ig in rows if ig)
    header = [
        "# cargo nextest test inventory -- do NOT edit by hand.",
        f"# Regenerate after adding/removing/ignoring any test: {_UPDATE_HINT}",
        f"# config: profile={PROFILE_LABEL} features={FEATURES_LABEL} target={target}",
        f"{DIAGNOSTIC_PREFIX} {versions}  (diagnostic only; not compared)",
        f"# totals: {len(rows)} testcases, {ignored} ignored",
        "# columns: <binary-id> \\t <test-name> \\t <ignored|active>",
        "# 'ignored' is the cfg-evaluated result for the config above;",
        "# regenerating under a different profile/feature/target will diff.",
    ]
    body = [
        f"{binary_id}\t{name}\t{'ignored' if ig else 'active'}"
        for binary_id, name, ig in rows
    ]
    return "\n".join(header + body) + "\n"


def build_current() -> str:
    return render_inventory(list_testcases(), host_target(), tool_versions())


def _comparable(text: str) -> str:
    """Drop diagnostic (toolchain-version) lines before comparing."""
    return "".join(
        ln for ln in text.splitlines(keepends=True)
        if not ln.startswith(DIAGNOSTIC_PREFIX)
    )


def _testcase_count(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln and not ln.startswith("#"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the committed test inventory against `cargo nextest list`."
    )
    parser.add_argument(
        "--update", action="store_true",
        help="regenerate the inventory file instead of checking it",
    )
    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_INVENTORY,
        help=f"inventory path (default: {DEFAULT_INVENTORY})",
    )
    args = parser.parse_args(argv[1:])

    try:
        current = build_current()
    except NextestError as e:
        print(f"check_new_ignore_attrs: {e}", file=sys.stderr)
        return EXIT_ENV

    if args.update:
        args.inventory.write_text(current)
        print(
            f"check_new_ignore_attrs: wrote {args.inventory} "
            f"({_testcase_count(current)} testcases)"
        )
        return EXIT_OK

    if not args.inventory.exists():
        print(
            f"check_new_ignore_attrs: inventory {args.inventory} is missing; "
            f"create it with `{_UPDATE_HINT}`",
            file=sys.stderr,
        )
        return EXIT_DRIFT

    committed = args.inventory.read_text()
    if _comparable(committed) == _comparable(current):
        print(
            f"check_new_ignore_attrs: OK ({_testcase_count(current)} testcases "
            f"match {args.inventory.name})"
        )
        return EXIT_OK

    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"{args.inventory.name} (committed)",
        tofile="cargo nextest list (live)",
    )
    sys.stderr.writelines(diff)
    print(
        "\ncheck_new_ignore_attrs: live test set differs from the committed "
        f"inventory. If this change is intentional, run `{_UPDATE_HINT}` and "
        "commit the diff so the added/removed/ignored tests are reviewable.",
        file=sys.stderr,
    )
    return EXIT_DRIFT


if __name__ == "__main__":
    sys.exit(main(sys.argv))
