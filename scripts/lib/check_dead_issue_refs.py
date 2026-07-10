#!/usr/bin/env python3
"""Fail if the tracked tree cites an issue-tracker-shaped reference that
nothing in this repo can resolve.

Shapes flagged: ``Task #NN``, ``memo#NN``, ``audit#NN``, ``issue #NN``,
``open #NN``, ``tracked #NN`` / ``tracked in #NN``, ``review #NN`` (and the
same triggers separated from the number by a colon/comma, an opening paren,
or one-to-two connector words -- see PATTERN). Matching is case-insensitive.

Single source of truth for ``.github/workflows/audit.yml`` (comment-hygiene
job) and ``scripts/pre-merge-audit.sh``; behaviour pinned by
tests/test_check_dead_issue_refs.py.

Why this exists
---------------
This repo has ZERO GitHub issues (``gh issue list --state all`` is empty),
so any ``issue #NN`` / ``Task #NN`` / ``memo#NN`` label is a private,
unresolvable in-code note. ``gh issue view <n>`` for a small ``n`` does not
404 -- it falls back to the same-numbered PR (PRs 1-26 exist) -- so a dead
``issue #14`` looks legitimate when checked naively. Only ``PR #NN`` is
resolvable and allowed; ``PR`` is not a trigger word, so a bare ``PR #25``
never matches. A PR-internal finding is written as prose ("PR #25 review
40"), carrying no ``#NN`` label.

Why we read git BLOBS, not the working tree
-------------------------------------------
The scan target is the CONTENT OF THE TRACKED TREE, not whatever the working
directory currently holds. Reading the worktree (``open(path)``) is wrong on
three counts, all of which let real dead refs through while reporting "OK": a
tracked symlink is *followed*, so the gate reads a file OUTSIDE the repo and
reports its content as ours; the symlink's own tracked blob (mode 120000,
whose content is the link-target string) is never scanned even when that
string is ``issue #14``; and an ``rm``-ed / ``chmod 000`` / sparse-checkout
entry cannot be read at all. So we ask git for the blob of every tracked
entry via a single ``git cat-file --batch``: a symlink blob is scanned as
text (its link-target string), never followed, so nothing outside the repo
is read; a gitlink (mode 160000 submodule) is not a blob and is skipped and
listed; a blob is always retrievable regardless of worktree state.

The scan root is the true top level (``git rev-parse --show-toplevel``), not
a path guessed from this file's location, so enumeration cannot be silently
scoped to a subtree (which would satisfy the coverage invariant on the wrong
set of files). ``GIT_DIR`` / object-directory overrides are stripped so an
inherited value (git hooks export these) cannot redirect enumeration or
object lookup. Coverage is proven, not assumed: every ls-files entry is
accounted for as scanned, skipped-binary, or skipped-gitlink; a mismatch is
a hard error.

This gate scans WIDER than `git grep -I`, matching only its NUL window, not
its verdict. `git grep -I` first short-circuits on the `binary`/`-diff`
.gitattributes (grep.c grep_source_is_binary -> the driver's binary flag),
and only falls back to content (buffer_is_binary, 8000-byte NUL window) when
no attribute is set. We deliberately do NOT honor that attribute short-cut --
otherwise setting `-diff` on a file would hide it from this gate (a real
escape hatch seen in an earlier revision) -- so a `-diff` file that git would
call binary is still scanned here. We DO reuse git's content window: the NUL
check looks only at the first 8000 bytes (see _GIT_BINARY_SNIFF_BYTES).

The invariant we guarantee is "nothing is skipped silently" -- every skip is
a hard error or a listed entry -- NOT "all text is scanned". One disclosed
hole where a real dead ref can go unreported: a BOM-less UTF-16 blob whose
first 8000 bytes are NUL-free (e.g. a long Greek UTF-16BE prefix) is not
flagged binary, is decoded as mojibake, and counts as scanned; a later ASCII
ref (NUL-interleaved) neither matches the pattern nor lands in the skip list.
Shrinking the NUL check from the whole blob to git's 8000-byte window opened
this sub-case; we keep the window at git's value deliberately and do not
claim to scan what we cannot. (A BOM-less UTF-16 blob with a NUL in the first
8000 bytes is instead flagged binary and listed -- visible, though its own
ref still will not match through the interleaved NULs.)

Trigger vocabulary and the ordinal-heading trade-off
----------------------------------------------------
The connection rule is uniform across triggers: between a trigger and
``#NN`` we allow an optional ``:``/``,``, whitespace, up to two connector
words from a closed list, and an optional ``(``. The list is closed and the
chain must end at ``#NN`` within two words, so an unrelated downstream number
cannot bridge a long gap. A regex cannot separate a dead label ``Task #29``
from an ordinal heading ``## Task #1: Setup`` -- same shape -- and ``Task`` /
``review`` / ``audit`` are the bulk of the real dead refs, so we flag the
ordinal form on purpose: the convention is to write an internal number
WITHOUT ``#`` (``## Task 1: Setup``), and that one-character fix clears the
gate. Genuine non-tracker ordinals use non-trigger nouns ("priority #1", MPS
``x#1#1``) and do not match.
"""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys
from pathlib import Path

# Error handler for the UTF-8 fallback: map every undecodable byte to '_'
# (U+005F, a \w word char). The only property this gate needs from a byte it
# cannot decode is that it not forge a word boundary; making it a word char
# guarantees that regardless of the byte's true (unknown) encoding. It does
# NOT touch bytes that decode cleanly, so a genuine U+FFFD written in valid
# UTF-8 keeps its real (non-word) meaning.
codecs.register_error(
    "deadref_wordsafe",
    lambda exc: ("_" * (exc.end - exc.start), exc.end),
)

# Repo-root-relative POSIX paths excluded: this gate and its test both carry
# trigger-shaped example strings.
SELF_EXCLUDE = {
    "scripts/lib/check_dead_issue_refs.py",
    "tests/test_check_dead_issue_refs.py",
}

TRIGGERS = r"(?:Task|memo|audit|issue|open|tracked|review)"
CONNECTOR = r"(?:in|as|at|to|under|via|see|ref|commit)"
GAP = r"[\s:,]*(?:" + CONNECTOR + r"\s+){0,2}\(?\s*"
PATTERN = re.compile(r"\b" + TRIGGERS + GAP + r"#[0-9]+", re.IGNORECASE)

# git's buffer_is_binary() (xdiff-interface.c) inspects only the first 8000
# bytes for a NUL: `if (FIRST_FEW_BYTES < size) size = FIRST_FEW_BYTES;
# return !!memchr(ptr, 0, size);`. We match that window exactly so this
# gate's text/binary verdict equals `git grep -I`'s.
_GIT_BINARY_SNIFF_BYTES = 8000

_STRIP_GIT_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM",
)
_GITLINK_MODE = "160000"


class ScanError(RuntimeError):
    pass


def _clean_git_env() -> dict:
    env = dict(os.environ)
    for key in _STRIP_GIT_ENV:
        env.pop(key, None)
    return env


def _git(cwd: str, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, env=_clean_git_env(),
                          stdout=subprocess.PIPE, check=True).stdout


def toplevel(start: Path) -> Path:
    root = Path(os.fsdecode(_git(str(start), "rev-parse", "--show-toplevel")).strip())
    if not (root / ".git").exists():
        raise ScanError(f"{root} is not a git repository top level")
    return root


def list_entries(root: Path):
    """(mode, blob_hash, rel_path) for each ls-files entry."""
    out = _git(str(root), "ls-files", "-s", "-z")
    entries = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, blob, _stage = meta.split(b" ")
        entries.append((mode.decode(), blob.decode(), os.fsdecode(path)))
    return entries


def read_blobs(root: Path, hashes: list[str]) -> dict:
    """Batch-read blob contents by hash in ONE git process. Raise on any
    unresolved object (e.g. a bogus object directory)."""
    if not hashes:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=str(root), env=_clean_git_env(),
        input=("\n".join(hashes) + "\n").encode(),
        stdout=subprocess.PIPE, check=True,
    )
    buf = proc.stdout
    out: dict = {}
    pos, n = 0, len(buf)
    while pos < n:
        nl = buf.index(b"\n", pos)
        header = buf[pos:nl].split(b" ")
        pos = nl + 1
        oid = header[0].decode()
        if len(header) < 3 or header[1] in (b"missing", b"ambiguous"):
            raise ScanError(f"git cat-file could not resolve object {oid}")
        size = int(header[2])
        out[oid] = buf[pos:pos + size]
        pos += size + 1  # trailing newline
    return out


def classify(data: bytes):
    """('text', str) or ('binary', None).

    1. UTF-16 BOM -> decode UTF-16 and scan.
    2. else NUL in the first _GIT_BINARY_SNIFF_BYTES -> binary.
    3. else -> decode UTF-8 strict; on failure decode UTF-8 again mapping each
       undecodable byte to '_' (a word char).

    We do NOT guess the source encoding of an invalid byte. The one property
    the scan needs is that an undecodable byte never forge a word boundary
    (which would make e.g. "<byte>audit #12" match "audit #12" and
    "iss<byte>ue #12" match "issue"). Mapping the byte to a word char '_'
    guarantees that whatever its true encoding: "_audit" / "iss_ue" are single
    words, not "audit" / "issue". Earlier attempts to guess (latin-1, then
    cp1252) forged boundaries on some byte range every time -- latin-1 on
    0x80-0x9F (C1 controls), cp1252 on its five undefined bytes and on
    truncated multibyte UTF-8. Bytes that decode cleanly are untouched, so a
    genuine U+FFFD written in valid UTF-8 keeps its real non-word meaning.
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return "text", data.decode("utf-16")
        except UnicodeDecodeError:
            return "binary", None
    if b"\x00" in data[:_GIT_BINARY_SNIFF_BYTES]:
        return "binary", None
    try:
        return "text", data.decode("utf-8")
    except UnicodeDecodeError:
        return "text", data.decode("utf-8", errors="deadref_wordsafe")


def scan(root: Path, classify_fn=classify):
    """Return (hits, skipped_binary, skipped_gitlink). Raise ScanError if any
    entry is unaccounted for."""
    entries = list_entries(root)
    want = sorted({b for (m, b, _p) in entries if m != _GITLINK_MODE})
    blobs = read_blobs(root, want)

    hits: list[str] = []
    skipped_binary: list[str] = []
    skipped_gitlink: list[str] = []
    scanned = 0
    for mode, blob, rel in entries:
        if mode == _GITLINK_MODE:
            skipped_gitlink.append(rel)
            continue
        if rel in SELF_EXCLUDE:
            scanned += 1
            continue
        kind, text = classify_fn(blobs[blob])
        if kind == "text":
            scanned += 1
            for lineno, line in enumerate(text.splitlines(), 1):
                if PATTERN.search(line):
                    hits.append(f"{rel}:{lineno}:{line.strip()}")
        elif kind == "binary":
            skipped_binary.append(rel)
    total = scanned + len(skipped_binary) + len(skipped_gitlink)
    if total != len(entries):
        raise ScanError(
            f"coverage invariant broken: entries={len(entries)} scanned={scanned} "
            f"binary={len(skipped_binary)} gitlink={len(skipped_gitlink)}"
        )
    return hits, skipped_binary, skipped_gitlink


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    start = Path(__file__).resolve().parent
    if "--repo-root" in argv:
        start = Path(argv[argv.index("--repo-root") + 1]).resolve()
    try:
        root = toplevel(start)
        hits, skipped_binary, skipped_gitlink = scan(root)
    except (ScanError, subprocess.CalledProcessError) as exc:
        print(f"::error::dead issue-ref scan failed; tree not fully scanned: {exc}",
              file=sys.stderr)
        return 2

    for rel in skipped_gitlink:
        print(f"skipped submodule (gitlink, not scanned): {rel}")
    if skipped_binary:
        print(f"skipped {len(skipped_binary)} binary file(s) (not scanned):")
        for rel in skipped_binary:
            print(f"  {rel}")

    if hits:
        print("::error::Unresolvable issue-tracker-shaped reference (this repo "
              "has zero GitHub issues; see scripts/lib/check_dead_issue_refs.py):")
        for h in hits:
            print(h)
        print()
        print("Describe what happened / what is still open in prose. For an item "
              "with no tracker, write the number without a '#' (e.g. \"issue 31\", "
              "not \"#31\"). Only 'PR #NN' is resolvable; a PR finding is prose "
              "(e.g. \"PR #25 review 40\").")
        return 1

    print("dead issue ref check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
