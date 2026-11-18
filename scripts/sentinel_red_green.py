#!/usr/bin/env python3
"""Sentinel test 真正性ゲート (red/green メタ検証).

目的
====
CLAUDE.md「実装」の「修正には sentinel test を付ける (revert すると fail する
ことを確認)」を CI で機械的に強制する。手でやっていた「revert して fail したか
ら green に変えた」という運用を、"base commit で red・HEAD で green" という
検証可能な形に置き換える (meta originally failed and turned into pass)。

規約 (convention)
==================
回帰張り (regression guard) となるテスト関数は、doc ブロック内に 1 行の
マーカーを持つ:

    /// sentinel: <何がどの回帰を張っているか 1 行>

- マーカー行はテスト関数の直前の doc ブロックの任意の位置に置いてよい
  (doc 行と `#[test]` 属性の間に挟んでもよい)。
- 記述 (コロン以降) が空のマーカーは malformed として lint fail。
- マーカーを付けた関数名は tests/test_inventory.txt に実在しなければならない
  (inventory は `python3 scripts/lib/check_new_ignore_attrs.py --update` で
  再生成する成果物であり、本スクリプトは membership 検査のみ行う。再生成は
  しない)。テストではない helper 関数にマーカーを付けても lint fail。
- 既存テストはマーカーなしで従来どおり動作する。マーカーのない新規テストは
  red 検証を要求されない (リファクタリング安全)。マーカーは「これは revert
  すれば fail するはずの張りである」という著者の主張であり、red モードが
  その主張を base commit に対して検証する。

モード
======
lint  : 静的検査のみ (cargo 不要)。マーカー書式・テスト関数への束縛・
        inventory membership を検査する。
red   : PR の BASE_SHA に対して新規マーカー付きテストを抽出し、base worktree
        へテスト定義ファイルを overlay して `cargo nextest run` を実行する。
        red 判定基準は「exit code != 0」(テスト fail でもビルド error でも
        red)。exit 0 = base では pass する = 回帰を張れていない = VIOLATION。
        実行は環境変数 SENTINEL_CI_RUN=1 のときだけ (GitHub runner 限定)。
        それ以外は --dry-run 扱いで計画 (新テスト集合・overlay 対象ファイル・
        nextest コマンド) を表示するだけ。ローカル開発では cargo を走らせない。
--self-test : 合成 git repository を一時ディレクトリに作り、lint/red の抽出・
        判定ロジックを実検査する (cargo 不要。SENTINEL_CI_FAKE_RESULT で
        nextest の exit code をスタブして判定経路も検証する)。

既知の限界 (正直な記録)
========================
- overlay は「PR が触った、テスト fn を含むファイル」のみ。本番コードは base
  のままなので、テストが PR 新規の内部 API に依存すると base でビルド error
  になる。これは exit != 0 として RED OK に数えられるが、真の「テスト fail」
  の証拠ではない (判定は exit code のみが見える)。出力上は区別できない。
- nextest のセレクタが 0 件だと cargo nextest は非ゼロ終了するため、同じく
  「0 件による red」と「fail による red」は本ゲートでは区別しない。
- inventory の status が active から ignored へ変わったテストは新規扱いに
  ならない (binary-id / test-name の集合で差分を取るため)。

使い方
======
    python3 scripts/sentinel_red_green.py --mode lint
    BASE_SHA=<pr-base-sha> python3 scripts/sentinel_red_green.py --mode red --dry-run
    BASE_SHA=<pr-base-sha> SENTINEL_CI_RUN=1 python3 scripts/sentinel_red_green.py --mode red
    python3 scripts/sentinel_red_green.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXIT_OK = 0
EXIT_FAIL = 1  # lint error / red violation
EXIT_ENV = 2   # 環境・引数の不足 (BASE_SHA 未指定、git 失敗など)

INVENTORY_REL = "tests/test_inventory.txt"
# マーカー行: `/// sentinel: <記述>`。`///` 直後 (空白可) に sentinel: が
# 来ること。本文途中の "sentinel:" は誤認しない (行頭锚定)。
MARKER_RE = re.compile(r"^\s*///\s*sentinel\s*:\s*(.*)$")
# マーカー行と fn 定義の間に許す行: doc 継続行、属性行、空行なし (doc
# ブロックは空行で切れるので空行を跨ぐ束縛は拒否する)。
DOC_CONT_RE = re.compile(r"^\s*///")
ATTR_LINE_RE = re.compile(r"^\s*#\[")
FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?extern\s+.*?fn\s+(\w+)|^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")

# GitHub Actions が計画有無を機械判定するための行 (workflow との契約)。
PLAN_COUNT_PREFIX = "SENTINEL_PLAN_COUNT="


class GateError(RuntimeError):
    """ゲートを実行できない環境・入力問題 (violation とは区別する)。"""


def run_git(args: list[str], cwd: Path, check: bool = True) -> str:
    """`git` を実行して stdout を返す。失敗時は GateError。"""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise GateError(f"git が見つからない: {e}")
    if check and proc.returncode != 0:
        raise GateError(
            f"git {' '.join(args)} が失敗 (exit {proc.returncode}): "
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no stderr'}"
        )
    return proc.stdout


def repo_root(cwd: Path | None = None) -> Path:
    out = run_git(["rev-parse", "--show-toplevel"], cwd or Path.cwd())
    return Path(out.strip())


def tracked_rs_files(root: Path) -> list[str]:
    """working tree の追跡 .rs ファイル (target/ 等は追跡されていないので除外)。"""
    out = run_git(["ls-files", "-z", "--", "*.rs"], root)
    return sorted(p for p in out.split("\0") if p)


def scan_markers(root: Path) -> tuple[list[dict], list[str]]:
    """`/// sentinel:` マーカーを走査し (entries, errors) を返す。

    entry: {file, line, fn, desc}。エラーは書式不正・fn 未束縛。
    """
    entries: list[dict] = []
    errors: list[str] = []
    for rel in tracked_rs_files(root):
        try:
            lines = (root / rel).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            errors.append(f"{rel}: 読み取り失敗 ({e})")
            continue
        for i, line in enumerate(lines):
            m = MARKER_RE.match(line)
            if not m:
                continue
            desc = m.group(1).strip()
            if not desc:
                errors.append(
                    f"{rel}:{i + 1}: マーカーの記述が空 — "
                    f"`/// sentinel: <何を張るか 1 行>` の形式にすること"
                )
                continue
            fn_name, why = _bind_fn(lines, i)
            if fn_name is None:
                errors.append(f"{rel}:{i + 1}: {why}")
                continue
            entries.append(
                {"file": rel, "line": i + 1, "fn": fn_name, "desc": desc}
            )
    return entries, errors


def _bind_fn(lines: list[str], marker_idx: int) -> tuple[str | None, str]:
    """マーカー行の直後 (doc 継続・属性を挟んで) 最初の fn を見つける。"""
    j = marker_idx + 1
    in_attr = False
    while j < len(lines):
        line = lines[j]
        if in_attr:
            # 複数行属性 (`#[a(\n ...)]`) の継続行は属性終端まで読み飛ばす。
            if line.rstrip().endswith("]"):
                in_attr = False
            j += 1
            continue
        if DOC_CONT_RE.match(line):
            j += 1
            continue
        if ATTR_LINE_RE.match(line):
            if not line.rstrip().endswith("]"):
                in_attr = True
            j += 1
            continue
        m = FN_RE.match(line)
        if m:
            return (m.group(1) or m.group(2), "")
        return (
            None,
            "マーカーがテスト関数に束縛できない — doc の直後のコード行が "
            f"fn ではない: {line.strip()!r} (マーカーは fn 直前の doc に置く)",
        )
    return (None, "マーカー以降に fn が見つからない")


def parse_inventory(text: str) -> set[tuple[str, str]]:
    """inventory テキストから {(binary-id, test-name)} 集合を作る。"""
    out: set[tuple[str, str]] = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        out.add((cols[0], cols[1]))
    return out


def inventory_at(root: Path, rev: str) -> set[tuple[str, str]]:
    out = run_git(["show", f"{rev}:{INVENTORY_REL}"], root)
    return parse_inventory(out)


def fn_segment(test_name: str) -> str:
    """inventory の test-name (モジュールパス付き) から最終 fn 名を取り出す。"""
    return test_name.split("::")[-1]


def mode_lint(root: Path) -> int:
    entries, errors = scan_markers(root)
    try:
        inv = inventory_at(root, "HEAD")
    except GateError as e:
        print(f"lint: inventory が読めない: {e}", file=sys.stderr)
        return EXIT_ENV
    inv_fns = {fn_segment(name) for _, name in inv}

    ok_rows, bad_rows = [], []
    for ent in entries:
        if ent["fn"] in inv_fns:
            ok_rows.append(ent)
        else:
            ent = dict(ent)
            ent["why"] = (
                f"関数 {ent['fn']} が {INVENTORY_REL} に存在しない — "
                "inventory 再生成 (check_new_ignore_attrs.py --update) が必要 "
                "か、マーカーがテスト以外の fn に付いている"
            )
            bad_rows.append(ent)

    print("=== sentinel lint ===")
    for ent in ok_rows + bad_rows:
        mark = "OK " if "why" not in ent else "NG "
        print(f"{mark}| {ent['file']}:{ent['line']} | {ent['fn']} | {ent['desc']}")
        if "why" in ent:
            print(f"   | {ent['why']}")
    for err in errors:
        print(f"NG | {err}")
    print(
        f"=== 結果: マーカー {len(entries)} 件 / 書式・束縛エラー {len(errors)} 件"
        f" / inventory 欠落 {len(bad_rows)} 件 ==="
    )
    if errors or bad_rows:
        print("sentinel lint: FAIL", file=sys.stderr)
        return EXIT_FAIL
    print("sentinel lint: OK")
    return EXIT_OK


def new_marked_tests(
    root: Path, base: str
) -> tuple[list[tuple[str, str]], list[str], dict[tuple[str, str], list[str]]]:
    """base..HEAD で追加されたマーカー付きテストを返す。

    戻り値: (新規マーカー付き (binary, name) 列, PR が触った .rs, overlay 対象 map)
    """
    base_inv = inventory_at(root, base)
    head_inv = inventory_at(root, "HEAD")
    new_ids = sorted(head_inv - base_inv)
    entries, _errs = scan_markers(root)
    marked_fns = {e["fn"] for e in entries}
    targets = [(b, n) for (b, n) in new_ids if fn_segment(n) in marked_fns]

    if not targets:
        return targets, [], {}

    changed_raw = run_git(
        ["diff", "--name-only", base, "HEAD", "--"], root
    ).splitlines()
    changed = [p for p in changed_raw if p.endswith(".rs")]

    overlays: dict[tuple[str, str], list[str]] = {}
    for bid, name in targets:
        fn = fn_segment(name)
        # HEAD のツリーから fn を定義する .rs を探す (working tree dirty 対策で
        # 明示的に rev 指定)。
        out = run_git(
            ["grep", "-l", "-E", rf"\bfn\s+{re.escape(fn)}\b", "HEAD", "--", "*.rs"],
            root,
            check=False,
        )
        containers = [ln.split(":", 1)[1] for ln in out.splitlines() if ":" in ln]
        overlay = sorted(set(containers) & set(changed))
        if not overlay:
            print(
                f"警告: {name} の定義ファイル ({containers}) が PR で変更されて"
                "いない — overlay 対象なし (マクロ生成テスト等の境界ケース。"
                "base 側では 0 セレクション => red として扱われる)",
                file=sys.stderr,
            )
        overlays[(bid, name)] = overlay
    return targets, changed, overlays


def nextest_filter(bid: str, name: str) -> str:
    """`binary_id(<bid 部分一致>) & test(/^<name>$/)` を作る。

    test 名は正規表現メタ文字 (evolved test の名前には空白・括弧が現れる) を
    エスケープして完全一致に锚定する。binary-id は inventory の列値の部分一致で
    十分 (完全一致锚定は nextest の binary-id 表記揺れに強くないため)。
    """
    esc = re.sub(r"([.*+?^${}()|\[\]\\])", r"\\\1", name)
    return f"binary_id({bid}) & test(/^" + esc + "$/)"


def build_base_worktree(root: Path, base: str, tmpdir: Path) -> Path:
    wt = tmpdir / "base-worktree"
    if wt.exists():
        raise GateError(f"worktree 先が既にある: {wt}")
    run_git(["worktree", "add", "--detach", str(wt), base], root)
    # data/ は gitignored で base worktree に現れない。HEAD 側に実体があれば
    # symlink で共有する (ci.yml の test-integration と同じ data を使わせる)。
    src_data = root / "data"
    dst_data = wt / "data"
    if src_data.is_dir() and not dst_data.exists():
        os.symlink(str(src_data.resolve()), str(dst_data))
    return wt


def overlay_files(root: Path, wt: Path, files: list[str]) -> None:
    """HEAD 版のファイルを base worktree へ書き出す (git show 経由。dirty 対策)。"""
    for rel in files:
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{rel}"],
            capture_output=True,
            check=True,
        ).stdout
        dst = wt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)


def red_exit_code(cmd: list[str], wt: Path) -> int:
    """SENTINEL_CI_FAKE_RESULT があればそれを、なければ実際の cargo exit code。"""
    fake = os.environ.get("SENTINEL_CI_FAKE_RESULT")
    if fake is not None:
        print(f"  (stub) SENTINEL_CI_FAKE_RESULT={fake} を exit code として使用")
        return int(fake)
    proc = subprocess.run(cmd, cwd=str(wt))
    return proc.returncode


def mode_red(root: Path, base: str, dry: bool) -> int:
    try:
        targets, _changed, overlays = new_marked_tests(root, base)
    except GateError as e:
        print(f"red: {e}", file=sys.stderr)
        return EXIT_ENV

    print(f"=== sentinel red (base={base[:12]}) ===")
    print(f"{PLAN_COUNT_PREFIX}{len(targets)}")
    if not targets:
        print("新規マーカー付きテストなし — base 検証はスキップ (VIOLATION なし)")
        print("sentinel red: OK (対象 0 件)")
        return EXIT_OK

    run = (not dry) and os.environ.get("SENTINEL_CI_RUN") == "1"
    if not run:
        print("(dry-run: 計画表示のみ。実行には SENTINEL_CI_RUN=1)")

    tmpdir: Path | None = None
    wt: Path | None = None
    violations: list[tuple[str, str, int]] = []
    reds = 0
    try:
        for bid, name in targets:
            fn = fn_segment(name)
            cmd = [
                "cargo", "nextest", "run", "--release", "--features", "parallel",
                "-E", nextest_filter(bid, name),
            ]
            print(f"\n-- テスト: {bid} :: {name}")
            print(f"   overlay 対象: {', '.join(overlays[(bid, name)]) or '(なし)'}")
            print(f"   実行コマンド: (base worktree 内で) {' '.join(cmd)}")
            if not run:
                continue
            if wt is None:
                tmpdir = Path(tempfile.mkdtemp(prefix="sentci-wt-"))
                wt = build_base_worktree(root, base, tmpdir)
                print(f"   (base worktree を用意した: {wt})")
            assert wt is not None  # 直前に必ず build 済み
            overlay_files(root, wt, overlays[(bid, name)])
            print("   base checkout での実行結果:")
            rc = red_exit_code(cmd, wt)
            verdict = "RED OK (base で fail — 張りは本物)" if rc != 0 else \
                "VIOLATION (base で pass — 回帰を張れていない)"
            print(f"   exit={rc} => {verdict}")
            if rc != 0:
                reds += 1
            else:
                violations.append((bid, name, rc))
    finally:
        if wt is not None:
            subprocess.run(
                ["git", "-C", str(root), "worktree", "remove", "--force", str(wt)],
                capture_output=True, text=True,
            )
            shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n=== 判定表 ===")
    print(f"{'テスト':<70} | 判定")
    print("-" * 100)
    for bid, name in targets:
        v = next((x for x in violations if x[0] == bid and x[1] == name), None)
        verdict = f"VIOLATION (exit {v[2]})" if v else ("RED OK" if run else "計画のみ")
        print(f"{bid} :: {name:<.50} | {verdict}")
    print(
        f"=== 結果: 対象 {len(targets)} 件 / RED OK {reds} 件 / "
        f"VIOLATION {len(violations)} 件 ==="
    )
    if violations:
        print(
            "sentinel red: FAIL — 上のテストは base commit で pass する。"
            "revert しても fail しない張りは sentinel ではない (CLAUDE.md「実装」)。",
            file=sys.stderr,
        )
        return EXIT_FAIL
    print("sentinel red: OK")
    return EXIT_OK


# ---------------------------------------------------------------------------
# self-test: 合成 git repository で抽出・判定ロジックを実検査する (cargo 不要)
# ---------------------------------------------------------------------------

BASE_LIB = """\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        assert_eq!(add(1, 2), 3);
    }
}
"""

HEAD_LIB = """\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        assert_eq!(add(1, 2), 3);
    }

    /// sentinel: add が減算へ逆戻りすると fail — 回帰張りそのものの合成例
    #[test]
    fn add_guard_sentinel() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn new_unmarked_test() {
        assert_eq!(add(0, 0), 0);
    }
}
"""

INV_HEADER = (
    "# cargo nextest test inventory -- do NOT edit by hand.\n"
    "# columns: <binary-id> \\t <test-name> \\t <ignored|active>\n"
)


def _git(repo: Path, *args: str) -> str:
    return run_git(list(args), repo)


def _make_synthetic_repo(root: Path, extra_bad: str = "") -> str:
    """base commit と HEAD commit を持つ合成 repo を作り base sha を返す。"""
    root.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests").mkdir()
    _git(root, "init", "-b", "main", "-q")
    _git(root, "config", "user.email", "sentci@invalid")
    _git(root, "config", "user.name", "sentci-selftest")
    (root / "src" / "lib.rs").write_text(BASE_LIB)
    (root / "tests" / "test_inventory.txt").write_text(
        INV_HEADER + "synthetic\ttests::add_works\tactive\n"
    )
    _git(root, "add", "src/lib.rs", "tests/test_inventory.txt")
    _git(root, "commit", "-q", "-m", "base")
    base_sha = _git(root, "rev-parse", "HEAD").strip()
    (root / "src" / "lib.rs").write_text(HEAD_LIB + extra_bad)
    (root / "tests" / "test_inventory.txt").write_text(
        INV_HEADER
        + "synthetic\ttests::add_works\tactive\n"
        + "synthetic\ttests::add_guard_sentinel\tactive\n"
        + "synthetic\ttests::new_unmarked_test\tactive\n"
    )
    _git(root, "add", "src/lib.rs", "tests/test_inventory.txt")
    _git(root, "commit", "-q", "-m", "head: 張りを追加")
    return base_sha


class _Captured:
    """stdout/stderr を Capture して returncode を覚えるヘルパ。"""

    def __init__(self) -> None:
        import io
        from contextlib import redirect_stderr, redirect_stdout, ExitStack

        self.out, self.err = io.StringIO(), io.StringIO()
        self._stack = ExitStack()
        self._stack.enter_context(redirect_stdout(self.out))
        self._stack.enter_context(redirect_stderr(self.err))

    def close(self) -> str:
        self._stack.close()
        return self.out.getvalue() + self.err.getvalue()


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="sentci-selftest-"))
    print(f"self-test 作業先: {tmp}")
    try:
        repo = tmp / "synthetic"
        base_sha = _make_synthetic_repo(repo)

        # 1) lint: 正しく張いれば OK
        cap = _Captured()
        rc = mode_lint(repo)
        out = cap.close()
        check("lint: 正規マーカーで exit 0", rc == EXIT_OK, out)
        check("lint: マーカーテーブルに関数名", "add_guard_sentinel" in out)

        # 2) red --dry-run: 新規マーカー付きだけを計画し、張りのない新規テストは無視
        cap = _Captured()
        rc = mode_red(repo, base_sha, dry=True)
        out = cap.close()
        check("red(dry): exit 0", rc == EXIT_OK, out)
        check("red(dry): 新規マーカー付きを抽出", "add_guard_sentinel" in out)
        check("red(dry): 無印の新規テストは対象外", "new_unmarked_test" not in out)
        check("red(dry): overlay 対象ファイル検出", "src/lib.rs" in out)
        check("red(dry): nextest コマンド計画", "cargo nextest run --release --features parallel -E binary_id(synthetic) & test(/^tests::add_guard_sentinel$/)" in out)
        check("red(dry): 機械判定行", "SENTINEL_PLAN_COUNT=1" in out)

        # 3) SENTINEL_CI_FAKE_RESULT=1 (base で fail) => RED OK, worktree 後始末
        os.environ["SENTINEL_CI_RUN"] = "1"
        os.environ["SENTINEL_CI_FAKE_RESULT"] = "1"
        try:
            cap = _Captured()
            rc = mode_red(repo, base_sha, dry=False)
            out = cap.close()
            check("red(fake=1): RED OK として exit 0", rc == EXIT_OK, out)
            check("red(fake=1): worktree 片付け", not list(tmp.glob("sentci-wt-*")))
            leftovers = _git(repo, "worktree", "list").splitlines()
            check("red(fake=1): git worktree 台帳もクリーン", len(leftovers) == 1, str(leftovers))

            # 4) FAKE_RESULT=0 (base で pass) => VIOLATION
            os.environ["SENTINEL_CI_FAKE_RESULT"] = "0"
            cap = _Captured()
            rc = mode_red(repo, base_sha, dry=False)
            out = cap.close()
            check("red(fake=0): VIOLATION で exit 1", rc == EXIT_FAIL, out)
            check("red(fake=0): 判定表に VIOLATION", "VIOLATION" in out)
        finally:
            del os.environ["SENTINEL_CI_FAKE_RESULT"]
            del os.environ["SENTINEL_CI_RUN"]

        # 5) malformed marker (記述空) => lint fail
        repo2 = tmp / "malformed"
        _make_synthetic_repo(
            repo2,
            extra_bad="\n/// sentinel:\n#[test]\nfn malformed_marker_case() {}\n",
        )
        cap = _Captured()
        rc = mode_lint(repo2)
        out = cap.close()
        check("lint: 記述空マーカーで exit 1", rc == EXIT_FAIL, out)

        # 6) マーカーが fn 以外に束縛 => lint fail
        repo3 = tmp / "unbound"
        _make_synthetic_repo(repo3, extra_bad="\n/// sentinel: 構造体に付いた誤配置\nstruct NotATest;\n")
        cap = _Captured()
        rc = mode_lint(repo3)
        out = cap.close()
        check("lint: fn 未束縛マーカーで exit 1", rc == EXIT_FAIL, out)

        # 7) マーカー fn が inventory に無い => lint fail
        repo4 = tmp / "noinv"
        base4 = _make_synthetic_repo(
            repo4,
            extra_bad="\n/// sentinel: inventory 未登録の張り\n#[test]\nfn missing_from_inventory() {}\n",
        )
        cap = _Captured()
        rc = mode_lint(repo4)
        out = cap.close()
        check("lint: inventory 欠落マーカーで exit 1", rc == EXIT_FAIL, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"=== self-test 結果: {len(failures)} 失敗 ===")
    for f in failures:
        print(f"  FAIL: {f}")
    return EXIT_OK if not failures else EXIT_FAIL


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Sentinel test red/green メタ検証ゲート"
    )
    parser.add_argument("--mode", choices=["lint", "red"], help="実行モード")
    parser.add_argument(
        "--base", default=None,
        help="red モードの base commit SHA (省略時は環境変数 BASE_SHA)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="red モードで計画表示のみ (cargo を実行しない)。デフォルト動作",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="合成 git repository でゲートロジックを検査する (cargo 不要)",
    )
    args = parser.parse_args(argv[1:])

    if args.self_test:
        return self_test()
    if not args.mode:
        parser.error("--mode lint|red のいずれか、または --self-test が必要")
    try:
        root = repo_root()
    except GateError as e:
        print(f"sentinel_red_green: {e}", file=sys.stderr)
        return EXIT_ENV
    if args.mode == "lint":
        return mode_lint(root)
    base = args.base or os.environ.get("BASE_SHA", "")
    if not base:
        print(
            "sentinel_red_green: red モードには base commit が必要 "
            "(環境変数 BASE_SHA か --base)",
            file=sys.stderr,
        )
        return EXIT_ENV
    return mode_red(root, base, dry=not (os.environ.get("SENTINEL_CI_RUN") == "1"))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
