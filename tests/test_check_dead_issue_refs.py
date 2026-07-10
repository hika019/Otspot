"""Self-test for scripts/lib/check_dead_issue_refs.py.

Pins the SCAN-COVERAGE defenses (the gate must never report a green "OK"
while tracked content went unscanned, and must never read outside the repo)
and the PATTERN decision. Run directly:
``python3 tests/test_check_dead_issue_refs.py`` (exit == failing cases).
Wired into pre-merge-audit.sh and .github/workflows/audit.yml.

Each coverage test pins one defense; reverting that defense turns the paired
test red:
  - read git blobs, not the worktree  -> test_symlink_outside_repo_not_read,
                                         test_symlink_blob_is_scanned
  - true toplevel (rev-parse)         -> test_subdir_repo_root_scans_toplevel
  - gitlink skipped + listed          -> test_gitlink_is_skipped_listed
  - each _STRIP_GIT_ENV var            -> test_env_*_stripped / *_in_strip_set
  - cat-file size-based slicing        -> test_cat_file_size_based_slicing
  - coverage invariant assert          -> test_coverage_invariant_break_errors
  - NUL(<8000 window)=binary           -> test_nul_within_8000_window_is_binary,
                                          test_nul_past_8000_window_is_text,
                                          test_true_binary_is_listed
  - undecodable byte -> word char '_'  -> test_undecodable_byte_never_forges_word_boundary,
                                          test_undecodable_byte_line_still_reports_real_ref,
                                          test_genuine_ufffd_in_valid_utf8_stays_non_word
  - UTF-16 BOM decoded / listed         -> test_utf16_bom_scanned,
                                          test_utf16_no_bom_skipped_listed
  - content classify (not attrs)       -> test_diff_attr_file_is_scanned
  - blob enumeration (name-safe)       -> test_leading_hyphen_name_detected
  - uniform connector rule             -> test_connector_forms_detected
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = (Path(__file__).resolve().parent.parent
        / "scripts" / "lib" / "check_dead_issue_refs.py")

_spec = importlib.util.spec_from_file_location("dead_issue_gate", GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _clean_env() -> dict:
    env = dict(os.environ)
    for k in gate._STRIP_GIT_ENV:
        env.pop(k, None)
    return env


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(root), env=_clean_env(),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _init(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    _git(tmp, "init")
    _git(tmp, "config", "user.email", "t@t")
    _git(tmp, "config", "user.name", "t")
    return tmp


def _make_repo(tmp: Path, files: dict) -> Path:
    """files maps relative path (str) -> bytes; committed into a fresh repo."""
    _init(tmp)
    for name, data in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-m", "init")
    return tmp


def _run(root: Path, extra_env: dict | None = None):
    env = _clean_env()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, str(GATE), "--repo-root", str(root)],
                       capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


# --------------------------------------------------------------------------
# Scan-coverage defenses
# --------------------------------------------------------------------------

def test_clean_repo_ok():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"a.txt": b"nothing to see\n"})
        code, out = _run(repo)
    assert code == 0, out


def test_direct_dead_ref_detected():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"a.rs": b"// see issue #14 for context\n"})
        code, out = _run(repo)
    assert code == 1, out
    assert "issue #14" in out


def test_symlink_outside_repo_not_read():
    """A tracked symlink pointing OUTSIDE the repo must not be followed: the
    blob content is the link-target path string, not the target's bytes."""
    with tempfile.TemporaryDirectory() as td:
        outside = Path(td) / "outside_secret.txt"
        outside.write_text("secret contents with issue #77777\n")
        repo = _init(Path(td) / "repo")
        (repo / "escape.txt").symlink_to(outside)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "link")
        code, out = _run(repo)
    assert code == 0, out          # target's dead ref must NOT surface
    assert "77777" not in out


def test_symlink_blob_is_scanned():
    """A symlink whose blob content (link target string) is itself a dead ref
    must be caught."""
    with tempfile.TemporaryDirectory() as td:
        repo = _init(Path(td))
        (repo / "hides.txt").symlink_to("issue #14")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "hide")
        code, out = _run(repo)
    assert code == 1, out
    assert "hides.txt" in out and "issue #14" in out


def test_subdir_repo_root_scans_toplevel():
    """--repo-root pointing at a SUBDIR must resolve to the true toplevel, so
    a dead ref at repo root is not silently out of scope."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {
            "root_level.txt": b"issue #55555\n",
            "sub/x.txt": b"clean\n",
        })
        code, out = _run(repo / "sub")
    assert code == 1, out
    assert "root_level.txt" in out


def test_gitlink_is_skipped_listed():
    with tempfile.TemporaryDirectory() as td:
        sub = _make_repo(Path(td) / "subrepo", {"s.txt": b"hi\n"})
        repo = _make_repo(Path(td) / "main", {"top.txt": b"clean\n"})
        _git(repo, "-c", "protocol.file.allow=always",
             "submodule", "add", str(sub), "mysub")
        _git(repo, "commit", "-m", "withsub")
        code, out = _run(repo)
    assert code == 0, out
    assert "mysub" in out and "gitlink" in out.lower()


def test_deleted_worktree_file_still_scanned():
    """Blob-based scan: a tracked file removed from the worktree is still read
    from its blob (no exit-2 error, and its dead ref is caught)."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"gone.txt": b"issue #1\n"})
        (repo / "gone.txt").unlink()
        code, out = _run(repo)
    assert code == 1, out
    assert "gone.txt" in out


def test_unreadable_worktree_file_still_scanned():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"secret.txt": b"issue #2\n"})
        (repo / "secret.txt").chmod(0o000)
        try:
            code, out = _run(repo)
        finally:
            (repo / "secret.txt").chmod(0o644)
    assert code == 1, out
    assert "secret.txt" in out


def test_diff_attr_file_is_scanned():
    """A file marked ``-diff`` (git treats as binary) is still read, because
    classification is by content, not .gitattributes (the Cargo.lock hole)."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {
            ".gitattributes": b"locked.txt -diff\n",
            "locked.txt": b"# tracked in issue #9999\n",
        })
        code, out = _run(repo)
    assert code == 1, out
    assert "locked.txt" in out


def test_leading_hyphen_name_detected():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"-dashfile.txt": b"issue #55\n"})
        code, out = _run(repo)
    assert code == 1, out
    assert "-dashfile.txt" in out


def test_space_in_name_detected():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"a file.txt": b"Task #7\n"})
        code, out = _run(repo)
    assert code == 1, out


def test_newline_in_name_detected():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"weird\nname.txt": b"Task #8\n"})
        code, out = _run(repo)
    assert code == 1, out


def test_utf16_bom_scanned():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"u16.txt": "issue #77\n".encode("utf-16")})
        code, out = _run(repo)
    assert code == 1, out
    assert "issue #77" in out


def test_utf16_no_bom_skipped_listed():
    """BOM-less UTF-16 is NUL-laden and indistinguishable from binary, so it
    is skipped -- but LISTED, never silently dropped (invariant: no silent
    skip, not "all text scanned")."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"n.txt": "issue #4242".encode("utf-16-le")})
        code, out = _run(repo)
    assert code == 0, out
    assert "n.txt" in out and "skipped" in out.lower()


def test_true_binary_is_listed():
    """A genuine binary (NUL byte, git's own binary rule) is skipped+listed."""
    with tempfile.TemporaryDirectory() as td:
        blob = b"\x89PNG\r\n\x1a\n\x00\x00 issue #5 \xff\xd8\xff"
        repo = _make_repo(Path(td), {"blob.png": blob})
        code, out = _run(repo)
    assert code == 0, out
    assert "blob.png" in out and "skipped" in out.lower()


def test_invalid_byte_file_still_scanned():
    """A file with an invalid-UTF-8 byte and NO NUL is text per git; a
    strict-decode design skipped the whole file. The gate must scan it so a
    dead ref elsewhere on the line is still found."""
    with tempfile.TemporaryDirectory() as td:
        blob = b"note iss\x92 tracked in issue #22222 ok\n"
        repo = _make_repo(Path(td), {"doc.md": blob})
        code, out = _run(repo)
    assert code == 1, out
    assert "22222" in out


def test_nul_past_8000_window_is_text():
    """git's binary sniff only looks at the first 8000 bytes; a NUL past that
    window does not make the file binary, so a dead ref before it is scanned.
    (Reverting the window to whole-blob skips this file and turns this red.)"""
    with tempfile.TemporaryDirectory() as td:
        blob = b"tracked in issue #22222 here\n" + b"x" * 8600 + b"\x00tail\n"
        assert blob.index(b"\x00") > 8000
        repo = _make_repo(Path(td), {"late.txt": blob})
        code, out = _run(repo)
    assert code == 1, out
    assert "22222" in out


def test_nul_within_8000_window_is_binary():
    """A NUL inside the 8000-byte window marks binary (matches git)."""
    with tempfile.TemporaryDirectory() as td:
        blob = b"x" * 100 + b"\x00 tracked in issue #333 here\n"
        repo = _make_repo(Path(td), {"early.txt": blob})
        code, out = _run(repo)
    assert code == 0, out
    assert "early.txt" in out and "skipped" in out.lower()


def test_undecodable_byte_never_forges_word_boundary():
    """The invariant: an undecodable byte is mapped to '_' (a word char) so it
    can never forge a word boundary, whatever its true encoding. A trigger
    directly after such a byte is mid-word (no match); a byte inside a word
    keeps it one word (not the trigger word). This covers every byte class an
    encoding guess got wrong: cp1252's 8 word letters, cp1252's 5 UNDEFINED
    bytes, and a truncated multibyte UTF-8 sequence. Reverting the '_' mapping
    to U+FFFD (non-word) turns this red."""
    files = {}
    # cp1252 word letters (0x83 8A 8C 8E 9A 9C 9E 9F) AND undefined (0x81 8D
    # 8F 90 9D): all -> '_' -> "_audit"/"_issue"/... single word, no match.
    for b in (0x83, 0x8A, 0x8C, 0x8E, 0x9A, 0x9C, 0x9E, 0x9F,
              0x81, 0x8D, 0x8F, 0x90, 0x9D):
        files[f"s{b:02x}.txt"] = bytes([b]) + b"audit #12\n"
    files["inside.txt"] = b"iss\x81ue #12\n"          # iss_ue != issue
    files["trunc.txt"] = b"\xe3\x81audit #12\n"        # truncated UTF-8, adjacent
    files["cafe.txt"] = b"caf\xe9issue #12\n"          # caf_issue mid-word
    files["iss.txt"] = b"iss\x92ue #12\n"              # iss_ue
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), files)
        code, out = _run(repo)
    assert code == 0, out


def test_undecodable_byte_line_still_reports_real_ref():
    """The '_' mapping must not swallow a genuine dead ref elsewhere on a line
    that also carries an undecodable byte."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"u.txt": b"weird\x81 tracked in issue #22222 ok\n"})
        code, out = _run(repo)
    assert code == 1, out
    assert "22222" in out


def test_genuine_ufffd_in_valid_utf8_stays_non_word():
    """A real U+FFFD written in VALID UTF-8 decodes cleanly (strict path), so
    it is NOT remapped to '_' and keeps its true non-word meaning: it forms a
    boundary, so 'x<FFFD>audit #12' exposes a standalone 'audit #12' (match).
    This pins that the '_' remap fires ONLY on the decode-failure path."""
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"f.txt": "x�audit #12\n".encode("utf-8")})
        code, out = _run(repo)
    assert code == 1, out


# --- _STRIP_GIT_ENV: one sentinel per var. Each sets the var to a value that,
# if honored, redirects/breaks the scan; the gate must strip it and stay
# clean. Removing that var from _STRIP_GIT_ENV turns the paired test red.
# Six are behaviorally load-bearing in a sandbox; the two object-search
# extras cannot break a clean scan behaviorally (an alternate only ADDS
# sources; discovery-across-fs needs a
# mount boundary) so they are pinned by membership.

def _env_poison_repos(td: Path):
    """(target-clean, poison-repo). Poison holds a unique dead ref #13131."""
    poison = _make_repo(td / "poison", {"p.txt": b"poison issue #13131\n"})
    target = _make_repo(td / "target", {"ok.txt": b"clean\n", "sub/s.txt": b"clean\n"})
    return target, poison


def _assert_env_stripped(var: str, value_of, start_subdir: bool = False):
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        target, poison = _env_poison_repos(td)
        start = target / "sub" if start_subdir else target
        code, out = _run(start, extra_env={var: value_of(target, poison)})
    assert code == 0, f"{var}: expected clean exit 0, got {code}\n{out}"
    assert "13131" not in out, f"{var}: poison ref leaked\n{out}"


def test_env_git_dir_stripped():
    _assert_env_stripped("GIT_DIR", lambda t, p: str(p / ".git"))


def test_env_git_index_file_stripped():
    _assert_env_stripped("GIT_INDEX_FILE", lambda t, p: str(p / ".git" / "index"))


def test_env_git_work_tree_stripped():
    _assert_env_stripped("GIT_WORK_TREE", lambda t, p: str(p))


def test_env_git_common_dir_stripped():
    _assert_env_stripped("GIT_COMMON_DIR", lambda t, p: str(p / ".git"))


def test_env_git_object_directory_stripped():
    _assert_env_stripped("GIT_OBJECT_DIRECTORY", lambda t, p: "/nonexistent/bogus")


def test_env_git_ceiling_directories_stripped():
    # From a subdir, a ceiling at the repo root blocks toplevel discovery.
    _assert_env_stripped("GIT_CEILING_DIRECTORIES",
                         lambda t, p: str(t), start_subdir=True)


def test_env_alternate_object_dirs_in_strip_set():
    # An alternate only ADDS object sources, so a bogus one cannot break a
    # clean scan (no behavioral sentinel possible); pin it by membership.
    assert "GIT_ALTERNATE_OBJECT_DIRECTORIES" in gate._STRIP_GIT_ENV


def test_env_discovery_across_fs_in_strip_set():
    # Needs a filesystem mount boundary to exercise; pin by membership.
    assert "GIT_DISCOVERY_ACROSS_FILESYSTEM" in gate._STRIP_GIT_ENV


def test_cat_file_size_based_slicing():
    """cat-file --batch output is sliced by the declared byte SIZE, not by
    lines. A blob whose CONTENT contains a line that looks like a batch header
    (``<sha> blob <n>``) followed by a dead ref must be parsed correctly: a
    naive line-based reader desyncs on the fake header and drops the ref."""
    fake_header = b"0" * 40 + b" blob 100000\n"
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {
            "a.txt": fake_header + b"poison issue #31337 here\n",
            "b.txt": b"clean tail\n",
        })
        code, out = _run(repo)
    assert code == 1, out
    assert "31337" in out


def test_coverage_invariant_break_errors():
    with tempfile.TemporaryDirectory() as td:
        repo = _make_repo(Path(td), {"a.txt": b"hi\n"})
        raised = False
        try:
            gate.scan(gate.toplevel(repo), classify_fn=lambda data: ("mystery", None))
        except gate.ScanError:
            raised = True
    assert raised, "coverage invariant did not raise on unaccounted file"


# --------------------------------------------------------------------------
# Pattern decision
# --------------------------------------------------------------------------

def _match(s: str) -> bool:
    return gate.PATTERN.search(s) is not None


def test_real_dead_ref_shapes_match():
    for s in ["Task #29", "memo#22", "audit#141", "issue #4", "review #40",
              "tracked in #88", "task #88"]:
        assert _match(s), s


def test_connector_forms_detected():
    for s in ["Issue: #12", "Task: #5", "Tracked: #12", "review (#12)",
              "tracked in commit #451", "opened as issue, see #14",
              "marked open, ref #22"]:
        assert _match(s), s


def test_connector_chain_broken_not_matched():
    assert not _match("audit, see appendix #3")


def test_pr_bare_is_allowed():
    assert not _match("PR #25")
    assert not _match("PR #25 review 40")


def test_review_bare_hash_banned():
    assert _match("review #40")
    assert not _match("review 40")


def test_structural_index_not_matched():
    for s in ["cold #0", "Eq#1", "Run #1:", "x#1#1", "priority #1",
              "player #2", "see line #40"]:
        assert not _match(s), s


def test_ordinal_heading_is_intentionally_flagged():
    for s in ["## Task #1: Setup", "### Review #2 findings",
              "memo #4: draft outline", "audit #2 of the quarterly report"]:
        assert _match(s), s


TESTS = [
    test_clean_repo_ok,
    test_direct_dead_ref_detected,
    test_symlink_outside_repo_not_read,
    test_symlink_blob_is_scanned,
    test_subdir_repo_root_scans_toplevel,
    test_gitlink_is_skipped_listed,
    test_deleted_worktree_file_still_scanned,
    test_unreadable_worktree_file_still_scanned,
    test_diff_attr_file_is_scanned,
    test_leading_hyphen_name_detected,
    test_space_in_name_detected,
    test_newline_in_name_detected,
    test_utf16_bom_scanned,
    test_utf16_no_bom_skipped_listed,
    test_true_binary_is_listed,
    test_invalid_byte_file_still_scanned,
    test_nul_past_8000_window_is_text,
    test_nul_within_8000_window_is_binary,
    test_undecodable_byte_never_forges_word_boundary,
    test_undecodable_byte_line_still_reports_real_ref,
    test_genuine_ufffd_in_valid_utf8_stays_non_word,
    test_env_git_dir_stripped,
    test_env_git_index_file_stripped,
    test_env_git_work_tree_stripped,
    test_env_git_common_dir_stripped,
    test_env_git_object_directory_stripped,
    test_env_git_ceiling_directories_stripped,
    test_env_alternate_object_dirs_in_strip_set,
    test_env_discovery_across_fs_in_strip_set,
    test_cat_file_size_based_slicing,
    test_coverage_invariant_break_errors,
    test_real_dead_ref_shapes_match,
    test_connector_forms_detected,
    test_connector_chain_broken_not_matched,
    test_pr_bare_is_allowed,
    test_review_bare_hash_banned,
    test_structural_index_not_matched,
    test_ordinal_heading_is_intentionally_flagged,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(failed)
