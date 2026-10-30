#!/usr/bin/env python3
"""Machine-check the operational claims in CLAUDE.md / AGENTS.md against the repo.

Every claim is a `Claim` entry with an explicit anchor in the prose and a check
that reads a structured artifact (TOML, YAML run blocks, shell command lines,
`git ls-files`). There is no prose inference: if an anchor disappears or an
artifact shape changes, the claim fails closed instead of silently passing.
Rewording a checked sentence therefore requires updating its anchor here -- that
is deliberate, so a claim cannot be dropped by an innocent-looking edit.

Known drift is a ratchet, not an allowlist. A failure whose exact
`(claim, key)` is listed in `KNOWN_DRIFT` is printed as KNOWN and does not fail
the gate; any other failure fails. When the underlying repair lands, the old key
stops matching and the entry becomes STALE, which fails the gate until the entry
(and the matching prose pointer) is removed. A new regression always has a new
key, so it is never absorbed by an existing entry.

Deliberately not checked: CLAUDE.md's "1 test finishes within 3 minutes" versus
nextest.toml's slow-timeout. The default profile's slow-timeout is a kill
ceiling that deliberately exceeds 3 minutes for promoted tier-2 sentinels, and
tier overrides are per-test; the operational reading of the 3-minute sentence is
not settled. Encoding either reading would manufacture a false failure. The
claim registry makes this a one-function addition once the policy is fixed.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
DOC_NAMES = ("AGENTS.md", "CLAUDE.md")
CI_THREAD_LIMIT_RE = re.compile(r"CI\s+(\d+)")
LOCAL_THREAD_RE = re.compile(r"ローカル\s*(\d+)\s*スレッド\s*/\s*CI\s+(\d+)")
CLAUDE_THREAD_RE = re.compile(r"ローカル\s*(\d+)\s*/\s*GitHub Actions\s+(\d+)")
PROFILE_RE = re.compile(r"--profile\s+([A-Za-z0-9_-]+)")
PATH_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])((?:scripts|tests|\.config|\.github|data)/[A-Za-z0-9_./*-]*)")
SCRIPT_COMMAND_RE = re.compile(r"\b(?:bash|sh|python3?)\s+(scripts/[^\s`]+)([^`\n]*)")
YAML_RUN_RE = re.compile(r"^(\s*)(?:-\s+)?run:\s*(.*)$")
NEXTEST_RUN_RE = re.compile(r"\bcargo\s+nextest\s+run\b")
NEXTEST_RUN_START_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:time\s+)?cargo\s+nextest\s+run\b"
)
NEXTEST_FILTER_RE = re.compile(r"^(test|binary)\((.*)\)$")


class StructureError(RuntimeError):
    """A required doc, config, catalog, or directory is missing or malformed."""


@dataclass(frozen=True)
class Failure:
    """One doc-versus-reality mismatch. `key` is stable across line shifts."""

    claim: str
    key: str
    doc_ref: str
    actual: str
    expected: str


@dataclass(frozen=True)
class KnownDrift:
    """A reviewed drift entry that the gate tolerates until its repair lands."""

    claim: str
    key: str
    reason: str


@dataclass(frozen=True)
class Doc:
    rel: str
    lines: tuple[str, ...]

    @classmethod
    def load(cls, root: Path, rel: str) -> "Doc":
        path = root / rel
        if not path.is_file():
            return cls(rel, ())
        return cls(rel, tuple(path.read_text().splitlines()))

    def find(self, needle: str) -> int | None:
        for lineno, line in enumerate(self.lines, start=1):
            if needle in line:
                return lineno
        return None

    def find_re(self, pattern: str | re.Pattern[str]) -> tuple[int, re.Match[str]] | None:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        for lineno, line in enumerate(self.lines, start=1):
            match = compiled.search(line)
            if match:
                return lineno, match
        return None

    def ref(self, lineno: int | None) -> str:
        if lineno is None:
            return f"{self.rel}: (anchor missing)"
        text = self.lines[lineno - 1].strip()
        return f"{self.rel}:{lineno}: {text[:180]}"


@dataclass(frozen=True)
class Catalog:
    cases: tuple[tuple[str, str], ...]
    source: str


@dataclass(frozen=True)
class RunCommand:
    path: str
    lineno: int
    text: str


@dataclass
class Context:
    root: Path
    catalog: Catalog
    _docs: dict[str, Doc] = field(default_factory=dict)
    _nextest: dict | None = None

    def doc(self, rel: str) -> Doc:
        if rel not in self._docs:
            self._docs[rel] = Doc.load(self.root, rel)
        return self._docs[rel]

    @property
    def nextest(self) -> dict:
        if self._nextest is None:
            self._nextest = load_nextest(self.root)
        return self._nextest


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


def _strip_shell_comment(text: str) -> str:
    """Cut an unquoted `#` comment from a command line."""
    quote: str | None = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or text[index - 1].isspace()):
            return text[:index].rstrip()
    return text.rstrip()


def _filter_matches(expr: str, cases: Iterable[tuple[str, str]]) -> set[tuple[str, str]] | None:
    """Evaluate the nextest filter subset used by this repo.

    Supported: `test(...)`, `binary(...)`, `test(=exact)`, `+` unions, and
    `&` intersections -- the operators this repo's overrides actually use.
    Anything else returns None; the claim fails closed instead of guessing.
    """
    cases = tuple(cases)
    branches = _split_top_level(expr, "+")
    if branches is None:
        return None
    matched: set[tuple[str, str]] = set()
    for branch in branches:
        conjuncts = _split_top_level(branch, "&")
        if conjuncts is None:
            return None
        intersected: set[tuple[str, str]] | None = None
        for conjunct in conjuncts:
            part = _match_single_filter(conjunct, cases)
            if part is None:
                return None
            intersected = part if intersected is None else intersected & part
        matched |= intersected if intersected is not None else set()
    return matched


def _match_single_filter(
    expr: str, cases: tuple[tuple[str, str], ...]
) -> set[tuple[str, str]] | None:
    """Match one `test(...)`/`binary(...)` atom, with optional leading `=`."""
    match = NEXTEST_FILTER_RE.match(expr.strip())
    if not match:
        return None
    kind, arg = match.groups()
    exact = arg.startswith("=")
    if exact:
        arg = arg[1:]
    if not arg or "(" in arg or ")" in arg:
        return None
    matched: set[tuple[str, str]] = set()
    for binary, name in cases:
        value = name if kind == "test" else binary
        if (value == arg) if exact else (arg in value):
            matched.add((binary, name))
    return matched


def _split_top_level(expr: str, operator: str) -> list[str] | None:
    """Split on top-level `operator` (`+` union or `&` intersection).

    The remaining set operators (`-` subtraction, `|` union sugar, `!`
    negation) stay rejected, so unsupported syntax keeps failing closed.
    """
    if operator not in ("+", "&"):
        raise AssertionError(f"unsupported split operator: {operator!r}")
    rejected = frozenset("-|!")
    pieces: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and char == operator:
            pieces.append(expr[start:index])
            start = index + 1
        elif depth == 0 and char in rejected:
            return None
    if depth != 0:
        return None
    pieces.append(expr[start:])
    return [piece for piece in pieces if piece.strip()]


def load_catalog(root: Path, path: Path | None = None) -> Catalog:
    if path is not None and not path.is_absolute():
        path = root / path
    catalog_path = path or (root / "tests" / "test_inventory.txt")
    if not catalog_path.is_file():
        raise StructureError(f"test catalog not found: {catalog_path}")
    text = catalog_path.read_text()
    cases: list[tuple[str, str]] = []
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
            suites = data["rust-suites"]
            for suite in suites.values():
                binary = suite["binary-id"]
                cases.extend((binary, name) for name in (suite.get("testcases") or {}))
        except (ValueError, KeyError, AttributeError) as exc:
            raise StructureError(f"could not parse nextest JSON catalog {catalog_path}: {exc}")
    else:
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise StructureError(
                    f"{catalog_path}:{lineno}: expected <binary>\\t<test>\\t<status>"
                )
            cases.append((parts[0], parts[1]))
    if not cases:
        raise StructureError(f"test catalog is empty: {catalog_path}")
    try:
        label = str(catalog_path.resolve().relative_to(root.resolve()))
    except ValueError:
        label = str(catalog_path)
    return Catalog(tuple(sorted(cases)), label)


def load_nextest(root: Path) -> dict:
    path = root / ".config" / "nextest.toml"
    if not path.is_file():
        raise StructureError(f"missing {path.relative_to(root)}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise StructureError(f"{path.relative_to(root)}: {exc}")
    profiles = data.get("profile")
    if not isinstance(profiles, dict) or "default" not in profiles:
        raise StructureError(f"{path.relative_to(root)}: [profile.default] is missing")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise StructureError(f"{path.relative_to(root)}: profile {name!r} is not a table")
    return profiles


def profile_threads(profiles: dict, name: str) -> int | None:
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        return None
    value = profile.get("test-threads")
    if value is None:
        value = profiles["default"].get("test-threads")
    return value if isinstance(value, int) else None


def _logical_lines(text: str, start: int) -> Iterator[tuple[int, str]]:
    current = ""
    current_line = start
    for offset, raw in enumerate(text.splitlines()):
        if not current:
            current_line = start + offset
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        current += stripped
        command = _strip_shell_comment(current).strip()
        if command:
            yield current_line, command
        current = ""
    if current.strip():
        command = _strip_shell_comment(current).strip()
        if command:
            yield current_line, command


def iter_run_commands(path: Path, root: Path) -> list[RunCommand]:
    """Extract `run:` scalars (inline and block) from a workflow file."""
    lines = path.read_text().splitlines()
    commands: list[RunCommand] = []
    index = 0
    while index < len(lines):
        match = YAML_RUN_RE.match(lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        value = match.group(2).strip()
        start = index + 1
        if re.fullmatch(r"[|>][+-]?\s*(#.*)?", value):
            body: list[str] = []
            cursor = index + 1
            while cursor < len(lines):
                line = lines[cursor]
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    break
                body.append(line)
                cursor += 1
            for lineno, command in _logical_lines("\n".join(body), start):
                commands.append(RunCommand(path.name, lineno, command))
            index = cursor
        else:
            for lineno, command in _logical_lines(value, start):
                commands.append(RunCommand(path.name, lineno, command))
            index += 1
    return commands


def iter_script_commands(path: Path) -> list[RunCommand]:
    if not path.is_file():
        return []
    return [
        RunCommand(path.name, lineno, command)
        for lineno, command in _logical_lines(path.read_text(), 1)
    ]


def _workflow_name(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        match = re.match(r"^name:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip("'\"")
    return None


def _workflow_jobs(path: Path) -> set[str]:
    jobs: set[str] = set()
    in_jobs = False
    for line in path.read_text().splitlines():
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if in_jobs:
            if line and not line.startswith(" "):
                break
            match = re.match(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$", line)
            if match:
                jobs.add(match.group(1))
    return jobs


def _preceding_comment(lines: list[str], lineno: int) -> tuple[int, str] | None:
    """Nearest comment block above a filter, walking through the section header."""
    cursor = lineno - 2
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if not stripped:
            cursor -= 1
            continue
        if stripped.startswith("["):
            cursor -= 1
            continue
        break
    if cursor < 0 or not lines[cursor].lstrip().startswith("#"):
        return None
    end = cursor
    while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
        cursor -= 1
    return cursor + 2, lines[cursor + 1].strip()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def _accepted_flags(script: Path) -> set[str]:
    text = script.read_text()
    if script.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return set()
        flags: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--"):
                        flags.add(arg.value.split("=", 1)[0])
        if flags:
            return flags
    flags = set()
    for line in text.splitlines():
        code = _strip_shell_comment(line)
        if not code.lstrip().startswith("#"):
            flags.update(
                re.findall(r"(?<![A-Za-z0-9_-])(--[A-Za-z0-9][A-Za-z0-9-]*)", code)
            )
    return flags


def _anchor_failure(
    claim: str, key: str, doc: Doc, needle: str, expected: str
) -> Failure:
    return Failure(
        claim=claim,
        key=key,
        doc_ref=f"{doc.rel}: (anchor missing: {needle!r})",
        actual="the sentence this claim is anchored to was not found",
        expected=expected,
    )


def check_doc_files(ctx: Context) -> list[Failure]:
    failures: list[Failure] = []
    for rel in DOC_NAMES:
        doc = ctx.doc(rel)
        if not doc.lines:
            failures.append(
                Failure(
                    claim="doc_files_present",
                    key=rel,
                    doc_ref=f"{rel}: (missing or empty)",
                    actual="file missing or empty",
                    expected="the operational doc exists and is readable",
                )
            )
    return failures


def check_toolchain_pin(ctx: Context) -> list[Failure]:
    claim = "toolchain_pin"
    doc = ctx.doc("AGENTS.md")
    found = doc.find_re(r"`rust-toolchain\.toml` で (\d+\.\d+\.\d+) に固定")
    if found is None:
        return [
            _anchor_failure(
                claim,
                "agents-toolchain",
                doc,
                "`rust-toolchain.toml` で X.Y.Z に固定",
                "AGENTS.md pins the toolchain version and names rust-toolchain.toml",
            )
        ]
    lineno, match = found
    declared = match.group(1)
    path = ctx.root / "rust-toolchain.toml"
    if not path.is_file():
        return [
            Failure(
                claim=claim,
                key="rust-toolchain-missing",
                doc_ref=doc.ref(lineno),
                actual="rust-toolchain.toml is missing",
                expected=f"channel = \"{declared}\"",
            )
        ]
    raw = path.read_text().splitlines()
    try:
        channel = tomllib.loads(path.read_text())["toolchain"]["channel"]
    except (tomllib.TOMLDecodeError, KeyError):
        return [
            Failure(
                claim=claim,
                key="rust-toolchain-parse",
                doc_ref=doc.ref(lineno),
                actual="rust-toolchain.toml has no parsable [toolchain].channel",
                expected=f"channel = \"{declared}\"",
            )
        ]
    if channel != declared:
        channel_line = next(
            (i for i, line in enumerate(raw, start=1) if line.strip().startswith("channel")),
            1,
        )
        return [
            Failure(
                claim=claim,
                key="channel",
                doc_ref=doc.ref(lineno),
                actual=f"rust-toolchain.toml:{channel_line}: channel = \"{channel}\"",
                expected=f"channel matches the documented {declared}",
            )
        ]
    return []


def _section(doc: Doc, heading: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    inside = False
    for lineno, line in enumerate(doc.lines, start=1):
        if line.startswith(heading):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside:
            rows.append((lineno, line))
    return rows


def check_workspace_crates(ctx: Context) -> list[Failure]:
    claim = "workspace_crates"
    doc = ctx.doc("AGENTS.md")
    rows = _section(doc, "## 構成")
    documented: list[tuple[int, str]] = []
    for lineno, line in rows:
        if not line.lstrip().startswith("|"):
            continue
        match = re.match(r"^\s*\|\s*`([^`]+)`", line)
        if match:
            documented.append((lineno, match.group(1).strip()))
    if not documented:
        return [
            _anchor_failure(
                claim,
                "crate-table",
                doc,
                "| `crate` | role |",
                "the crate table in ## 構成 lists every workspace member",
            )
        ]

    cargo_path = ctx.root / "Cargo.toml"
    try:
        cargo = tomllib.loads(cargo_path.read_text())
        members = list(cargo["workspace"]["members"])
        root_package = cargo["package"]["name"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        return [
            Failure(
                claim=claim,
                key="cargo-toml",
                doc_ref=doc.ref(documented[0][0]),
                actual=f"could not read workspace.members/package.name: {exc}",
                expected="Cargo.toml exposes the documented workspace members",
            )
        ]

    def canonical(name: str) -> str:
        return root_package if name == "." else name

    actual = {canonical(name) for name in members}
    expected = {canonical(name) for _, name in documented}
    failures: list[Failure] = []
    if actual != expected:
        failures.append(
            Failure(
                claim=claim,
                key="members",
                doc_ref=doc.ref(documented[0][0]),
                actual=f"workspace.members: {', '.join(sorted(actual))}",
                expected=f"documented crates: {', '.join(sorted(expected))}",
            )
        )

    pending = doc.find_re(r"`otspot-py`.*default-members 外")
    if pending is None:
        failures.append(
            _anchor_failure(
                claim,
                "otspot-py-default-members",
                doc,
                "`otspot-py` ... default-members 外",
                "the doc states otspot-py is excluded from default-members",
            )
        )
    else:
        lineno, _ = pending
        default_members = {
            canonical(name) for name in cargo["workspace"].get("default-members", [])
        }
        if "otspot-py" not in actual:
            failures.append(
                Failure(
                    claim=claim,
                    key="otspot-py-member",
                    doc_ref=doc.ref(lineno),
                    actual="otspot-py is not in workspace.members",
                    expected="otspot-py is a workspace member but not a default member",
                )
            )
        if "otspot-py" in default_members:
            failures.append(
                Failure(
                    claim=claim,
                    key="otspot-py-default-member",
                    doc_ref=doc.ref(lineno),
                    actual=f"workspace.default-members: {', '.join(sorted(default_members))}",
                    expected="otspot-py is absent from default-members",
                )
            )
    return failures


def check_nextest_local_threads(ctx: Context) -> list[Failure]:
    claim = "nextest_local_threads"
    agents = ctx.doc("AGENTS.md")
    claude = ctx.doc("CLAUDE.md")
    failures: list[Failure] = []
    local_values: list[tuple[str, int, str]] = []
    ci_values: list[tuple[str, int, str]] = []

    agents_match = agents.find_re(LOCAL_THREAD_RE)
    if agents_match is None:
        failures.append(
            _anchor_failure(
                claim,
                "agents-declaration",
                agents,
                "ローカル 6 スレッド / CI 3",
                "AGENTS.md declares the local and CI thread counts",
            )
        )
    else:
        lineno, match = agents_match
        local_values.append(("agents-local", int(match.group(1)), agents.ref(lineno)))
        ci_values.append(("agents-ci", int(match.group(2)), agents.ref(lineno)))

    claude_match = claude.find_re(CLAUDE_THREAD_RE)
    if claude_match is None:
        failures.append(
            _anchor_failure(
                claim,
                "claude-declaration",
                claude,
                "ローカル 6 / GitHub Actions 3",
                "CLAUDE.md declares the local and CI thread counts",
            )
        )
    else:
        lineno, match = claude_match
        local_values.append(("claude-local", int(match.group(1)), claude.ref(lineno)))
        ci_values.append(("claude-ci", int(match.group(2)), claude.ref(lineno)))

    profiles = ctx.nextest
    default_threads = profile_threads(profiles, "default")
    if default_threads is None:
        failures.append(
            Failure(
                claim=claim,
                key="toml-default",
                doc_ref=agents.ref(agents_match[0] if agents_match else None),
                actual=".config/nextest.toml has no integer [profile.default].test-threads",
                expected="[profile.default].test-threads is an integer",
            )
        )
    else:
        source_line = next(
            (
                i
                for i, line in enumerate(
                    (ctx.root / ".config" / "nextest.toml").read_text().splitlines(), start=1
                )
                if re.match(r"\s*test-threads\s*=", line)
            ),
            1,
        )
        for key, value, ref in local_values:
            if value != default_threads:
                failures.append(
                    Failure(
                        claim=claim,
                        key=key,
                        doc_ref=ref,
                        actual=f".config/nextest.toml:{source_line}: test-threads = {default_threads}",
                        expected=f"the doc and the default profile agree on {value}",
                    )
                )

    for pattern, doc, key in (
        (r"cargo nextest run --release --features parallel --test-threads (\d+)", agents, "agents-command"),
        (r"full-suite `cargo nextest run --release --test-threads (\d+)`", claude, "claude-command"),
    ):
        found = doc.find_re(pattern)
        if found is None:
            failures.append(
                _anchor_failure(
                    claim,
                    key,
                    doc,
                    "full-suite command with --test-threads",
                    "the documented local full-suite command is present",
                )
            )
            continue
        lineno, match = found
        value = int(match.group(1))
        if default_threads is not None and value != default_threads:
            failures.append(
                Failure(
                    claim=claim,
                    key=key,
                    doc_ref=doc.ref(lineno),
                    actual=f"documented --test-threads {value}",
                    expected=f"the documented local command uses {default_threads}",
                )
            )

    if len({value for _, value, _ in ci_values}) > 1:
        failures.append(
            Failure(
                claim=claim,
                key="ci-agreement",
                doc_ref=ci_values[0][2],
                actual=" / ".join(f"{value}" for _, value, _ in ci_values),
                expected="AGENTS.md and CLAUDE.md declare the same CI thread count",
            )
        )
    return failures


def check_nextest_ci_threads(ctx: Context) -> list[Failure]:
    claim = "nextest_ci_threads"
    agents = ctx.doc("AGENTS.md")
    limit_match = agents.find_re(CI_THREAD_LIMIT_RE)
    if limit_match is None:
        return [
            _anchor_failure(
                claim,
                "ci-limit",
                agents,
                "CI 3",
                "AGENTS.md declares the CI nextest thread count",
            )
        ]
    limit_line, limit_re_match = limit_match
    limit = int(limit_re_match.group(1))
    workflows = ctx.root / ".github" / "workflows"
    if not workflows.is_dir():
        return [
            Failure(
                claim=claim,
                key="workflows-missing",
                doc_ref=agents.ref(limit_line),
                actual=".github/workflows is missing",
                expected="workflow nextest invocations resolve to at most "
                f"{limit} threads",
            )
        ]
    profiles = ctx.nextest
    failures: list[Failure] = []
    invocations = 0
    for path in sorted(workflows.glob("*.yml")):
        for command in iter_run_commands(path, ctx.root):
            if not NEXTEST_RUN_START_RE.search(command.text):
                continue
            invocations += 1
            text = command.text
            explicit = re.search(r"--test-threads(?:=|\s+)(\d+)", text)
            profile_match = PROFILE_RE.search(text)
            profile = profile_match.group(1) if profile_match else None
            if explicit:
                effective = int(explicit.group(1))
                source = "CLI --test-threads"
            elif profile:
                effective = profile_threads(profiles, profile)
                source = f"--profile {profile} (nextest.toml)"
            else:
                effective = profile_threads(profiles, "default")
                source = "default profile (no CLI override)"
            if effective is None:
                failures.append(
                    Failure(
                        claim=claim,
                        key=f"{command.path}::{_norm_ws(text)}",
                        doc_ref=agents.ref(limit_line),
                        actual=f"{command.path}:{command.lineno}: {text}",
                        expected=f"effective CI threads <= {limit}; could not resolve "
                        f"profile {profile!r}",
                    )
                )
                continue
            if effective > limit:
                failures.append(
                    Failure(
                        claim=claim,
                        key=f"{command.path}::{_norm_ws(text)}",
                        doc_ref=agents.ref(limit_line),
                        actual=f"{command.path}:{command.lineno}: {text} -> {effective} "
                        f"threads ({source})",
                        expected=f"effective CI threads <= {limit}",
                    )
                )
    if invocations == 0:
        failures.append(
            Failure(
                claim=claim,
                key="no-invocations",
                doc_ref=agents.ref(limit_line),
                actual="no `cargo nextest run` command found in .github/workflows/*.yml",
                expected="every CI nextest invocation is inspectable",
            )
        )
    return failures


def _filter_lines(root: Path) -> list[tuple[int, str, str]]:
    """(lineno, profile, filter) for every `filter =` line in nextest.toml."""
    path = root / ".config" / "nextest.toml"
    if not path.is_file():
        return []
    profile = ""
    rows: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        header = re.match(r"^\s*\[\[profile\.([A-Za-z0-9_-]+)\.overrides\]\]", line)
        if header:
            profile = header.group(1)
            continue
        table = re.match(r"^\s*\[profile\.([A-Za-z0-9_-]+)\]", line)
        if table:
            profile = table.group(1)
            continue
        match = re.match(r'^\s*filter\s*=\s*"(.*)"\s*$', line)
        if match:
            rows.append((lineno, profile, match.group(1)))
    return rows


def check_override_filters(ctx: Context) -> list[Failure]:
    claim = "nextest_override_filters_live"
    path = ctx.root / ".config" / "nextest.toml"
    rows = _filter_lines(ctx.root)
    if not path.is_file() or not rows:
        return [
            Failure(
                claim=claim,
                key="filters-missing",
                doc_ref=".config/nextest.toml: (no filter = lines)",
                actual="no override filters found",
                expected="every override filter matches at least one catalog testcase",
            )
        ]
    raw_lines = path.read_text().splitlines()
    failures: list[Failure] = []
    parsed_overrides = 0
    for profile_name, profile in ctx.nextest.items():
        overrides = profile.get("overrides", [])
        if not isinstance(overrides, list):
            failures.append(
                Failure(
                    claim=claim,
                    key=f"structure::{profile_name}",
                    doc_ref=f".config/nextest.toml: [profile.{profile_name}]",
                    actual="overrides is not an array of tables",
                    expected="every override is a table with a filter",
                )
            )
            continue
        for override in overrides:
            parsed_overrides += 1
            if not isinstance(override, dict) or "filter" not in override:
                failures.append(
                    Failure(
                        claim=claim,
                        key=f"structure::{profile_name}::override",
                        doc_ref=f".config/nextest.toml: [profile.{profile_name}.overrides]",
                        actual="override has no filter",
                        expected="every override declares a filter",
                    )
                )
    if len(rows) != parsed_overrides:
        failures.append(
            Failure(
                claim=claim,
                key="structure::count",
                doc_ref=".config/nextest.toml: filter lines",
                actual=f"{len(rows)} textual filter line(s), {parsed_overrides} parsed override(s)",
                expected="the textual filter scan covers every parsed override",
            )
        )
    for lineno, profile, expr in rows:
        matched = _filter_matches(expr, ctx.catalog.cases)
        key = f"{profile}::{expr}"
        comment = _preceding_comment(raw_lines, lineno)
        doc_ref = (
            f".config/nextest.toml:{comment[0]}-{lineno}: {comment[1][:160]}"
            if comment
            else f".config/nextest.toml:{lineno}: {raw_lines[lineno - 1].strip()}"
        )
        if matched is None:
            failures.append(
                Failure(
                    claim=claim,
                    key=key,
                    doc_ref=doc_ref,
                    actual=f"unsupported filter syntax: {expr}",
                    expected="filter uses the supported nextest subset "
                    "(test/binary, optional =, + unions, & intersections)",
                )
            )
        elif not matched:
            failures.append(
                Failure(
                    claim=claim,
                    key=key,
                    doc_ref=doc_ref,
                    actual=f"0 of {len(ctx.catalog.cases)} catalog testcases match {expr}",
                    expected=f"at least one match in {ctx.catalog.source}",
                )
            )
    return failures


def _git_ignored(root: Path, relative: str) -> bool:
    result = _git(root, "check-ignore", "-q", "--", relative)
    return result.returncode == 0


def _produced_by_scripts(root: Path, relative: str) -> bool:
    name = Path(relative).name
    if not name:
        return False
    for script in (root / "scripts").rglob("*"):
        if not script.is_file() or "__pycache__" in script.parts:
            continue
        try:
            if name in script.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def check_doc_paths(ctx: Context) -> list[Failure]:
    claim = "doc_path_references"
    failures: list[Failure] = []
    seen: set[tuple[str, str]] = set()
    for rel in DOC_NAMES:
        doc = ctx.doc(rel)
        for lineno, line in enumerate(doc.lines, start=1):
            for match in PATH_TOKEN_RE.finditer(line):
                token = match.group(1).rstrip(".,;:)。、")
                if not token or (rel, token) in seen:
                    continue
                seen.add((rel, token))
                target = ctx.root / token
                if "*" in token:
                    exists = bool(list(ctx.root.glob(token)))
                else:
                    exists = target.exists()
                if exists:
                    continue
                if (
                    token.startswith("data/")
                    and _git_ignored(ctx.root, token)
                    and _produced_by_scripts(ctx.root, token)
                ):
                    continue
                failures.append(
                    Failure(
                        claim=claim,
                        key=f"{rel}::{token}",
                        doc_ref=doc.ref(lineno),
                        actual=f"{token} does not exist",
                        expected="the referenced path exists (data/ paths must be "
                        "gitignored and produced by scripts/)",
                    )
                )
    return failures


def check_doc_script_flags(ctx: Context) -> list[Failure]:
    claim = "doc_script_flags"
    failures: list[Failure] = []
    seen: set[tuple[str, str]] = set()
    for rel in DOC_NAMES:
        doc = ctx.doc(rel)
        for lineno, line in enumerate(doc.lines, start=1):
            for match in SCRIPT_COMMAND_RE.finditer(line):
                script_rel = match.group(1).rstrip(".,;:)。、")
                args = _strip_shell_comment(match.group(2))
                script = ctx.root / script_rel
                if not script.is_file():
                    key = f"{rel}::{script_rel}::missing"
                    if key not in seen:
                        seen.add(key)
                        failures.append(
                            Failure(
                                claim=claim,
                                key=key,
                                doc_ref=doc.ref(lineno),
                                actual=f"{script_rel} does not exist",
                                expected="documented script exists and accepts its flags",
                            )
                        )
                    continue
                accepted = _accepted_flags(script)
                for flag in re.findall(r"(?<!\S)(--[A-Za-z0-9][A-Za-z0-9-]*)", args):
                    key = f"{rel}::{script_rel}::{flag}"
                    if key in seen or flag in accepted:
                        continue
                    seen.add(key)
                    failures.append(
                        Failure(
                            claim=claim,
                            key=key,
                            doc_ref=doc.ref(lineno),
                            actual=f"{script_rel} does not accept {flag}",
                            expected="the flag is present in the script's argument parsing",
                        )
                    )
    return failures


def check_doc_name_references(ctx: Context) -> list[Failure]:
    claim = "doc_name_references"
    profiles = ctx.nextest
    workflows = ctx.root / ".github" / "workflows"
    workflow_names: dict[str, str] = {}
    job_sets: dict[str, set[str]] = {}
    if workflows.is_dir():
        for path in workflows.glob("*.yml"):
            name = _workflow_name(path)
            if name:
                workflow_names.setdefault(name, path.name)
            job_sets[path.name] = _workflow_jobs(path)
    failures: list[Failure] = []
    seen: set[tuple[str, str]] = set()
    job_refs: list[tuple[int, str]] = []

    for rel in DOC_NAMES:
        doc = ctx.doc(rel)
        for lineno, line in enumerate(doc.lines, start=1):
            for match in PROFILE_RE.finditer(line):
                name = match.group(1)
                key = f"{rel}::profile::{name}"
                if key in seen:
                    continue
                seen.add(key)
                if name not in profiles:
                    failures.append(
                        Failure(
                            claim=claim,
                            key=key,
                            doc_ref=doc.ref(lineno),
                            actual=f"--profile {name} is not a [profile.*] table",
                            expected="the named profile exists in .config/nextest.toml",
                        )
                    )
            for match in re.finditer(r"`([A-Za-z][A-Za-z0-9 _-]*)`\s*workflow", line):
                name = match.group(1)
                key = f"{rel}::workflow::{name}"
                if key in seen:
                    continue
                seen.add(key)
                if name not in workflow_names:
                    failures.append(
                        Failure(
                            claim=claim,
                            key=key,
                            doc_ref=doc.ref(lineno),
                            actual=f"no workflow has name: {name}",
                            expected="a .github/workflows/*.yml declares that name",
                        )
                    )
            for match in re.finditer(r"\b([A-Za-z0-9_-]+\.yml)\b", line):
                name = match.group(1)
                key = f"{rel}::workflow-file::{name}"
                if key in seen:
                    continue
                seen.add(key)
                if not (workflows / name).is_file():
                    failures.append(
                        Failure(
                            claim=claim,
                            key=key,
                            doc_ref=doc.ref(lineno),
                            actual=f".github/workflows/{name} is missing",
                            expected="the referenced workflow file exists",
                        )
                    )
            job_match = re.search(r"ci\.yml の非テストゲート \(([^)]+)\)", line)
            if job_match:
                for name in (part.strip() for part in job_match.group(1).split(",")):
                    if name:
                        job_refs.append((lineno, name))

    ci_name = "ci.yml"
    if job_refs and ci_name not in job_sets:
        failures.append(
            Failure(
                claim=claim,
                key="jobs-file-missing",
                doc_ref=f"CLAUDE.md: (ci.yml job reference)",
                actual=".github/workflows/ci.yml is missing",
                expected="the referenced CI jobs exist",
            )
        )
    else:
        for lineno, name in job_refs:
            key = f"ci.yml::job::{name}"
            if key in seen:
                continue
            seen.add(key)
            if name not in job_sets.get(ci_name, set()):
                failures.append(
                    Failure(
                        claim=claim,
                        key=key,
                        doc_ref=ctx.doc("CLAUDE.md").ref(lineno),
                        actual=f"ci.yml jobs: {', '.join(sorted(job_sets.get(ci_name, set())))}",
                        expected=f"job {name!r} exists",
                    )
                )
    return failures


def check_data_tracking(ctx: Context) -> list[Failure]:
    claim = "data_tracking"
    doc = ctx.doc("AGENTS.md")
    found = doc.find_re(r"`data/baseline_objectives/\*\.csv` のみ git 追跡")
    if found is None:
        return [
            _anchor_failure(
                claim,
                "anchor",
                doc,
                "`data/baseline_objectives/*.csv` のみ git 追跡",
                "AGENTS.md states the only tracked data files are the baseline CSVs",
            )
        ]
    lineno, _ = found
    result = _git(ctx.root, "ls-files", "-z", "--", "data")
    if result.returncode != 0:
        return [
            Failure(
                claim=claim,
                key="git-ls-files",
                doc_ref=doc.ref(lineno),
                actual=f"git ls-files failed: {result.stderr.strip()}",
                expected="tracked data files can be enumerated",
            )
        ]
    tracked = [item for item in result.stdout.split("\0") if item]
    failures: list[Failure] = []
    if not tracked:
        failures.append(
            Failure(
                claim=claim,
                key="no-tracked",
                doc_ref=doc.ref(lineno),
                actual="git ls-files data returned no files",
                expected="data/baseline_objectives/*.csv is tracked",
            )
        )
    for item in tracked:
        if not re.fullmatch(r"data/baseline_objectives/[^/]+\.csv", item):
            failures.append(
                Failure(
                    claim=claim,
                    key=f"tracked::{item}",
                    doc_ref=doc.ref(lineno),
                    actual=f"tracked data file: {item}",
                    expected="only data/baseline_objectives/*.csv is tracked",
                )
            )
    if not any(item.startswith("data/baseline_objectives/") for item in tracked):
        failures.append(
            Failure(
                claim=claim,
                key="no-baseline",
                doc_ref=doc.ref(lineno),
                actual="no tracked file under data/baseline_objectives/",
                expected="at least one baseline CSV is tracked",
            )
        )
    if not _git_ignored(ctx.root, "data/__docs_consistency_untracked_probe__"):
        failures.append(
            Failure(
                claim=claim,
                key="data-not-ignored",
                doc_ref=doc.ref(lineno),
                actual="data/ runtime files are not gitignored",
                expected="data/ is gitignored except data/baseline_objectives/*.csv",
            )
        )
    return failures


def _heavy_in_premerge(root: Path) -> tuple[bool, str]:
    script = root / "scripts" / "pre-merge-audit.sh"
    for command in iter_script_commands(script):
        if (
            NEXTEST_RUN_START_RE.search(command.text)
            and "--profile heavy" in command.text
            and "--run-ignored all" in command.text
        ):
            return True, f"scripts/pre-merge-audit.sh:{command.lineno}"
    return False, "scripts/pre-merge-audit.sh has no heavy nextest command"


PENDING_RESOLUTION: dict[str, Callable[[Path], tuple[bool, str]]] = {
    "fix/ci-doc-consistency": _heavy_in_premerge,
}


def check_pending_pointers(ctx: Context) -> list[Failure]:
    claim = "pending_pointer_freshness"
    failures: list[Failure] = []
    seen: set[tuple[str, str]] = set()
    for rel in DOC_NAMES:
        doc = ctx.doc(rel)
        for lineno, line in enumerate(doc.lines, start=1):
            for match in re.finditer(r"`([A-Za-z0-9._/-]+)`\s*で対応中", line):
                branch = match.group(1)
                key = f"{rel}::{branch}"
                if key in seen:
                    continue
                seen.add(key)
                resolver = PENDING_RESOLUTION.get(branch)
                if resolver is None:
                    failures.append(
                        Failure(
                            claim=claim,
                            key=f"{rel}::unknown::{branch}",
                            doc_ref=doc.ref(lineno),
                            actual=f"no resolution predicate is registered for {branch}",
                            expected="every '対応中' pointer has a machine-checkable "
                            "resolution predicate",
                        )
                    )
                    continue
                resolved, evidence = resolver(ctx.root)
                if resolved:
                    failures.append(
                        Failure(
                            claim=claim,
                            key=f"{rel}::stale::{branch}",
                            doc_ref=doc.ref(lineno),
                            actual=f"the repair is already present: {evidence}",
                            expected="remove the '対応中' sentence now that it is stale",
                        )
                    )
    return failures


def check_premerge_gate(ctx: Context) -> list[Failure]:
    claim = "premerge_gate_commands"
    agents = ctx.doc("AGENTS.md")
    claude = ctx.doc("CLAUDE.md")
    anchors: list[tuple[str, Doc, str, int | None]] = []
    full = agents.find("マージ前ゲートを一発で")
    heavy = agents.find_re(r"heavy.*も必須")
    doctest = agents.find("cargo test --doc --release")
    claude_gate = claude.find("integrate へのマージ前")
    anchors.extend(
        [
            ("full-suite", agents, "マージ前ゲートを一発で", full),
            ("heavy", agents, "heavy ... も必須", heavy[0] if heavy else None),
            ("doctest-release", agents, "`cargo test --doc --release`", doctest),
            ("claude-gate", claude, "integrate へのマージ前", claude_gate),
        ]
    )
    failures: list[Failure] = []
    missing_anchor = [item for item in anchors if item[3] is None]
    for requirement, doc, needle, _ in missing_anchor:
        failures.append(
            _anchor_failure(
                claim,
                f"anchor::{requirement}",
                doc,
                needle,
                "the merge-gate prose still makes the checked claim",
            )
        )
    script = ctx.root / "scripts" / "pre-merge-audit.sh"
    commands = iter_script_commands(script)
    if not script.is_file():
        failures.append(
            Failure(
                claim=claim,
                key="script-missing",
                doc_ref=agents.ref(full),
                actual="scripts/pre-merge-audit.sh is missing",
                expected="the one-shot merge gate script exists",
            )
        )
        return failures
    if not commands:
        failures.append(
            Failure(
                claim=claim,
                key="no-commands",
                doc_ref=agents.ref(full),
                actual="scripts/pre-merge-audit.sh contains no runnable commands",
                expected="the gate script is inspectable",
            )
        )
        return failures

    local_threads = profile_threads(ctx.nextest, "default")
    nextest_commands = [
        cmd for cmd in commands if NEXTEST_RUN_START_RE.search(cmd.text)
    ]
    doc_test_commands = [
        cmd for cmd in commands if re.search(r"\bcargo\s+test\b", cmd.text)
    ]
    has_full = any(
        "--release" in cmd.text
        and re.search(r"--test-threads(?:=|\s+)(\d+)", cmd.text)
        and int(re.search(r"--test-threads(?:=|\s+)(\d+)", cmd.text).group(1))
        == local_threads
        for cmd in nextest_commands
    )
    has_heavy = any(
        "--profile heavy" in cmd.text and "--run-ignored all" in cmd.text
        for cmd in nextest_commands
    )
    has_doctest = any(
        re.search(r"--doc\b", cmd.text) and "--release" in cmd.text
        for cmd in doc_test_commands
    )
    rendered = "; ".join(f"{cmd.path}:{cmd.lineno}: {cmd.text}" for cmd in commands)
    checks = [
        (
            "full-suite",
            has_full,
            agents.ref(full),
            f"full-suite nextest run with --release and --test-threads {local_threads}",
        ),
        (
            "heavy",
            has_heavy,
            agents.ref(heavy[0] if heavy else None),
            "heavy nextest run with --profile heavy --run-ignored all",
        ),
        (
            "doctest-release",
            has_doctest,
            agents.ref(doctest),
            "cargo test --doc --release",
        ),
    ]
    for requirement, present, ref, expected in checks:
        if not present:
            failures.append(
                Failure(
                    claim=claim,
                    key=f"missing::{requirement}",
                    doc_ref=ref,
                    actual=f"scripts/pre-merge-audit.sh commands: {rendered[:600]}",
                    expected=f"the one-shot gate includes {expected}",
                )
            )
    return failures


@dataclass(frozen=True)
class Claim:
    id: str
    summary: str
    check: Callable[[Context], list[Failure]]


CLAIMS: tuple[Claim, ...] = (
    Claim("doc_files_present", "CLAUDE.md and AGENTS.md are readable", check_doc_files),
    Claim("toolchain_pin", "documented toolchain equals rust-toolchain.toml", check_toolchain_pin),
    Claim(
        "workspace_crates",
        "documented crate table equals workspace members; otspot-py is excluded "
        "from default-members",
        check_workspace_crates,
    ),
    Claim(
        "nextest_local_threads",
        "local thread declarations and the default full-suite command agree with "
        "nextest.toml",
        check_nextest_local_threads,
    ),
    Claim(
        "nextest_ci_threads",
        "every workflow nextest invocation resolves to at most the documented CI "
        "thread count",
        check_nextest_ci_threads,
    ),
    Claim(
        "nextest_override_filters_live",
        "every nextest override filter matches at least one real testcase",
        check_override_filters,
    ),
    Claim(
        "doc_path_references",
        "paths named in the docs exist (or are gitignored runtime data produced "
        "by scripts/)",
        check_doc_paths,
    ),
    Claim(
        "doc_script_flags",
        "flags used in documented script commands are accepted by those scripts",
        check_doc_script_flags,
    ),
    Claim(
        "doc_name_references",
        "referenced profiles, workflows, workflow files, and CI jobs exist",
        check_doc_name_references,
    ),
    Claim(
        "data_tracking",
        "only data/baseline_objectives/*.csv is git-tracked under data/",
        check_data_tracking,
    ),
    Claim(
        "pending_pointer_freshness",
        "'repair in progress on branch X' pointers are removed once the repair "
        "lands",
        check_pending_pointers,
    ),
    Claim(
        "premerge_gate_commands",
        "the one-shot merge gate script contains the commands the docs require",
        check_premerge_gate,
    ),
)


KNOWN_DRIFT: tuple[KnownDrift, ...] = ()


def evaluate(root: Path = ROOT, catalog_path: Path | None = None) -> tuple[Context, list[Failure]]:
    ctx = Context(root, load_catalog(root, catalog_path))
    failures: list[Failure] = []
    for claim in CLAIMS:
        try:
            failures.extend(claim.check(ctx))
        except StructureError as exc:
            failures.append(
                Failure(
                    claim=claim.id,
                    key="structure",
                    doc_ref=f"{claim.id}: (structure)",
                    actual=str(exc),
                    expected="the claim's source artifacts are readable and well formed",
                )
            )
    deduped: dict[tuple[str, str], Failure] = {}
    for failure in failures:
        deduped.setdefault((failure.claim, failure.key), failure)
    return ctx, [deduped[key] for key in sorted(deduped)]


def classify(
    failures: list[Failure], debts: tuple[KnownDrift, ...], strict: bool
) -> tuple[list[Failure], list[tuple[Failure, KnownDrift]], list[KnownDrift]]:
    debt_map = {(debt.claim, debt.key): debt for debt in debts}
    new: list[Failure] = []
    known: list[tuple[Failure, KnownDrift]] = []
    failure_keys = {(failure.claim, failure.key) for failure in failures}
    for failure in failures:
        debt = debt_map.get((failure.claim, failure.key))
        if debt is not None and not strict:
            known.append((failure, debt))
        else:
            new.append(failure)
    stale = [debt for debt in debts if (debt.claim, debt.key) not in failure_keys]
    return new, known, stale


def _print_failure(label: str, failure: Failure, stream) -> None:
    print(f"{label} [{failure.claim}] {failure.key}", file=stream)
    print(f"  doc: {failure.doc_ref}", file=stream)
    print(f"  actual: {failure.actual}", file=stream)
    print(f"  expected: {failure.expected}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify CLAUDE.md / AGENTS.md operational claims against the repo."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="test catalog: tests/test_inventory.txt or `cargo nextest list "
        "--message-format json` output (default: tests/test_inventory.txt)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="ignore KNOWN_DRIFT: every failure must be repaired before this passes",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered claims and known-drift entries, then exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for claim in CLAIMS:
            print(f"{claim.id}: {claim.summary}")
        print()
        for debt in KNOWN_DRIFT:
            print(f"known-drift {debt.claim} [{debt.key}]: {debt.reason}")
        return 0

    try:
        ctx, failures = evaluate(ROOT, args.catalog)
    except StructureError as exc:
        print(f"[docs-consistency] environment error: {exc}", file=sys.stderr)
        return 2

    new, known, stale = classify(failures, KNOWN_DRIFT, args.strict)
    for failure in new:
        _print_failure("FAIL", failure, sys.stderr)
    for failure, debt in known:
        _print_failure("KNOWN", failure, sys.stdout)
        print(f"  known-drift reason: {debt.reason}", file=sys.stdout)
    for debt in stale:
        print(
            f"STALE [{debt.claim}] {debt.key}: the known drift no longer reproduces; "
            "remove this KNOWN_DRIFT entry (and its prose pointer)",
            file=sys.stderr,
        )
        print(f"  recorded reason: {debt.reason}", file=sys.stderr)

    mode = "strict" if args.strict else "ratchet"
    print(
        f"[docs-consistency] {mode}: {len(CLAIMS)} claims, "
        f"{len(new)} new failure(s), {len(known)} known drift, {len(stale)} stale "
        f"entr{'y' if len(stale) == 1 else 'ies'}; catalog={ctx.catalog.source} "
        f"({len(ctx.catalog.cases)} testcases)"
    )
    return 1 if new or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
