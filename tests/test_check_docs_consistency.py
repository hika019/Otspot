"""Fixture tests for scripts/check_docs_consistency.py.

The fixture is a miniature repo whose docs and artifacts agree. Each mutation
changes exactly one side of a claim. `test_unrelated_mutation...` and
`test_line_shift...` pin the ratchet property: keys survive line shifts, and a
single-artifact mutation only moves its own claim.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "docs_consistency", ROOT / "scripts" / "check_docs_consistency.py"
)
gate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["docs_consistency"] = gate
_spec.loader.exec_module(gate)

AGENTS = """# AGENTS.md

toolchain は `rust-toolchain.toml` で 1.95.0 に固定。

## 構成

| クレート | 役割 |
|---|---|
| `.` (otspot) | facade |
| `otspot-num` | numeric primitives |
| `otspot-core` | solver engine |
| `otspot-io` | parsers |
| `otspot-model` | modeling API |
| `otspot-dev` | dev binaries |
| `otspot-py` | bindings。**default-members 外** |

- 層構造は CI (`Architecture` workflow / `python3 scripts/check_architecture.py`) が強制。

## テスト・検証コマンド

nextest 必須。設定は `.config/nextest.toml` (ローカル 6 スレッド / CI 3):

```bash
cargo nextest run --release --features parallel --test-threads 6
cargo nextest run --release --profile lib-only
cargo nextest run --release --profile heavy --run-ignored all
cargo test --doc --release
```

マージ前ゲートを一発で: `bash scripts/pre-merge-audit.sh`。integrate へのマージには full-suite に加え heavy (`--profile heavy --run-ignored all`) も必須 (一次ソースは CLAUDE.md「git」)。

## data/

gitignored。`data/baseline_objectives/*.csv` のみ git 追跡 (期待値参照)。

```bash
bash scripts/download_all_bench_data.sh --ci-subset
python3 scripts/lib/check_new_ignore_attrs.py --update
```
"""

CLAUDE = """# エージェント運用

- integrate へのマージ前: full-suite `cargo nextest run --release --test-threads 6` (+ heavy 相当 `--profile heavy --run-ignored all`)。マージ後にも full-suite。
- スレッド数はローカル 6 / GitHub Actions 3 (nextest.toml default が 6、CI は workflow で 3 を明示)。
- push 前ゲート: ci.yml の非テストゲート (docs, todo-budget, file-size)。
"""

CI_YML = """name: CI
on:
  push:
    branches: ["**"]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - run: cargo nextest run --release --test-threads 3
  todo-budget:
    runs-on: ubuntu-latest
    steps:
      - run: echo todo
  file-size:
    runs-on: ubuntu-latest
    steps:
      - run: echo size
"""

ARCH_YML = """name: Architecture
on: [push]
jobs:
  layers:
    runs-on: ubuntu-latest
    steps:
      - run: python3 scripts/check_architecture.py
"""

HEAVY_YML = """name: CI (heavy)
on: [push]
jobs:
  test-heavy:
    runs-on: ubuntu-latest
    steps:
      - run: |
          cargo nextest run --release --features parallel --profile heavy \\
            --run-ignored all --test-threads 3
"""

NEXTEST = """[profile.default]
test-threads = 6

# fixture override comment
[[profile.default.overrides]]
filter = "test(fixture_case)"
threads-required = "num-cpus"

[profile.lib-only]
test-threads = 3

[profile.heavy]
test-threads = 6

[[profile.heavy.overrides]]
filter = "test(fixture_heavy)"
threads-required = "num-cpus"
"""

INVENTORY = """# fixture catalog
fixture::suite\tfixture_case\tactive
fixture::suite\tfixture_heavy\tignored
"""

PRE_MERGE = """#!/bin/bash
set -eo pipefail
cargo nextest run --release --features parallel --test-threads 6
cargo nextest run --release --profile heavy --run-ignored all
cargo test --doc --release
"""

DOWNLOAD = """#!/bin/bash
case "${1:-}" in
  --ci-subset) shift ;;
  --check) shift ;;
  *) exit 1 ;;
esac
"""

IGNORE_GATE = """import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--update", action="store_true")
"""

GITIGNORE = """/data/*
!/data/baseline_objectives/
/data/baseline_objectives/*
!/data/baseline_objectives/*.csv
"""


class DocsConsistencyTests(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        files = {
            "AGENTS.md": AGENTS,
            "CLAUDE.md": CLAUDE,
            "Cargo.toml": (
                "[workspace]\n"
                'members = [".", "otspot-core", "otspot-num", "otspot-io", '
                '"otspot-model", "otspot-dev", "otspot-py"]\n'
                'default-members = [".", "otspot-core", "otspot-num", "otspot-io", '
                '"otspot-model", "otspot-dev"]\n\n'
                '[package]\nname = "otspot"\nversion = "0.1.0"\n'
            ),
            "rust-toolchain.toml": '[toolchain]\nchannel = "1.95.0"\n',
            ".config/nextest.toml": NEXTEST,
            "tests/test_inventory.txt": INVENTORY,
            ".gitignore": GITIGNORE,
            ".github/workflows/ci.yml": CI_YML,
            ".github/workflows/architecture.yml": ARCH_YML,
            ".github/workflows/test-heavy.yml": HEAVY_YML,
            "scripts/pre-merge-audit.sh": PRE_MERGE,
            "scripts/download_all_bench_data.sh": DOWNLOAD,
            "scripts/lib/check_new_ignore_attrs.py": IGNORE_GATE,
            "scripts/check_architecture.py": "",
            "data/baseline_objectives/expected.csv": "name,value\nx,1\n",
        }
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "add", "data/baseline_objectives/expected.csv"],
            check=True,
        )
        return root

    def evaluate(self, root):
        return gate.evaluate(root)

    def claims(self, failures):
        return {(failure.claim, failure.key) for failure in failures}

    def assert_failure(self, root, claim, key_part, actual_part=None):
        _, failures = self.evaluate(root)
        matches = [
            failure
            for failure in failures
            if failure.claim == claim and key_part in failure.key
        ]
        self.assertTrue(
            matches,
            f"expected {claim} failure containing {key_part!r}; got "
            f"{[(f.claim, f.key) for f in failures]}",
        )
        if actual_part is not None:
            self.assertIn(actual_part, matches[0].actual)
        return matches[0]

    def test_consistent_fixture_passes(self):
        root = self.fixture()
        _, failures = self.evaluate(root)
        self.assertEqual(
            [(f.claim, f.key, f.actual) for f in failures],
            [],
        )

    def test_missing_doc_file_fails(self):
        root = self.fixture()
        (root / "CLAUDE.md").unlink()
        self.assert_failure(root, "doc_files_present", "CLAUDE.md", "missing or empty")

    def test_toolchain_mismatch_fails(self):
        root = self.fixture()
        (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.94.0"\n')
        self.assert_failure(root, "toolchain_pin", "channel", "1.94.0")

    def test_missing_toolchain_anchor_fails_closed(self):
        root = self.fixture()
        text = (root / "AGENTS.md").read_text().replace(
            "toolchain は `rust-toolchain.toml` で 1.95.0 に固定。\n", ""
        )
        (root / "AGENTS.md").write_text(text)
        self.assert_failure(root, "toolchain_pin", "agents-toolchain", "anchor")

    def test_workspace_member_mismatch_fails(self):
        root = self.fixture()
        text = (root / "Cargo.toml").read_text().replace(', "otspot-io"', "")
        (root / "Cargo.toml").write_text(text)
        self.assert_failure(root, "workspace_crates", "members", "workspace.members")

    def test_otspot_py_default_member_fails(self):
        root = self.fixture()
        text = (root / "Cargo.toml").read_text().replace(
            '"otspot-model", "otspot-dev"]', '"otspot-model", "otspot-dev", "otspot-py"]'
        )
        (root / "Cargo.toml").write_text(text)
        self.assert_failure(root, "workspace_crates", "otspot-py-default-member")

    def test_local_thread_declaration_mismatch_fails(self):
        root = self.fixture()
        text = (root / "AGENTS.md").read_text().replace(
            "ローカル 6 スレッド / CI 3", "ローカル 7 スレッド / CI 3"
        )
        (root / "AGENTS.md").write_text(text)
        self.assert_failure(root, "nextest_local_threads", "agents-local", "test-threads = 6")

    def test_ci_inherited_default_threads_fails(self):
        root = self.fixture()
        path = root / ".github/workflows/ci.yml"
        path.write_text(path.read_text() + "      - run: cargo nextest run --release\n")
        self.assert_failure(
            root,
            "nextest_ci_threads",
            "ci.yml::cargo nextest run --release",
            "-> 6 threads (default profile",
        )

    def test_dead_override_filter_fails(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_test)"\n'
        )
        failure = self.assert_failure(
            root, "nextest_override_filters_live", "test(no_such_test)", "0 of"
        )
        self.assertIn(".config/nextest.toml:", failure.doc_ref)

    def test_unsupported_filter_syntax_fails_closed(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(fixture_case) - test(x)"\n'
        )
        self.assert_failure(root, "nextest_override_filters_live", "test(fixture_case) - test(x)", "unsupported")

    def test_intersection_filter_matching_catalog_passes(self):
        # The `.config/nextest.toml` shape this repo actually uses: binary()
        # narrows an exact test() name down to one testcase of one suite.
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\n'
            'filter = "binary(fixture::suite) & test(=fixture_case)"\n'
        )
        _, failures = self.evaluate(root)
        self.assertEqual([(f.claim, f.key) for f in failures], [])

    def test_intersection_with_unknown_binary_fails_as_zero_match(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\n'
            'filter = "binary(no_such_binary) & test(=fixture_case)"\n'
        )
        self.assert_failure(
            root,
            "nextest_override_filters_live",
            "binary(no_such_binary)",
            "0 of",
        )

    def test_negation_filter_still_fails_closed(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "not(test(fixture_case))"\n'
        )
        self.assert_failure(
            root, "nextest_override_filters_live", "not(test(fixture_case))", "unsupported"
        )

    def test_missing_doc_path_fails(self):
        root = self.fixture()
        (root / "scripts/check_architecture.py").unlink()
        self.assert_failure(
            root,
            "doc_path_references",
            "AGENTS.md::scripts/check_architecture.py",
            "does not exist",
        )

    def test_missing_script_flag_fails(self):
        root = self.fixture()
        path = root / "scripts/download_all_bench_data.sh"
        path.write_text("#!/bin/bash\nexit 0\n")
        self.assert_failure(
            root,
            "doc_script_flags",
            "scripts/download_all_bench_data.sh::--ci-subset",
            "does not accept",
        )

    def test_flag_only_in_comment_is_not_accepted(self):
        root = self.fixture()
        path = root / "scripts/download_all_bench_data.sh"
        path.write_text("#!/bin/bash\n# --ci-subset is documented here\nexit 0\n")
        self.assert_failure(
            root,
            "doc_script_flags",
            "scripts/download_all_bench_data.sh::--ci-subset",
        )

    def test_missing_profile_reference_fails(self):
        root = self.fixture()
        path = root / "AGENTS.md"
        path.write_text(path.read_text().replace("--profile heavy", "--profile nonexistent"))
        self.assert_failure(root, "doc_name_references", "AGENTS.md::profile::nonexistent")

    def test_missing_workflow_name_fails(self):
        root = self.fixture()
        path = root / ".github/workflows/architecture.yml"
        path.write_text(path.read_text().replace("name: Architecture", "name: Layers"))
        self.assert_failure(root, "doc_name_references", "AGENTS.md::workflow::Architecture")

    def test_tracked_non_csv_fails(self):
        root = self.fixture()
        (root / "data/raw.txt").write_text("not baseline\n")
        subprocess.run(
            ["git", "-C", str(root), "add", "-f", "data/raw.txt"], check=True
        )
        self.assert_failure(root, "data_tracking", "tracked::data/raw.txt")

    def test_stale_pending_pointer_fails(self):
        root = self.fixture()
        path = root / "AGENTS.md"
        path.write_text(
            path.read_text()
            + "\nheavy ステージのスクリプトへの組み込みは `fix/ci-doc-consistency` で対応中。\n"
        )
        failure = self.assert_failure(
            root, "pending_pointer_freshness", "stale::fix/ci-doc-consistency", "already present"
        )
        self.assertIn("対応中", failure.doc_ref)

    def test_unresolved_pending_pointer_is_fresh(self):
        root = self.fixture()
        path = root / "AGENTS.md"
        path.write_text(
            path.read_text()
            + "\nheavy ステージのスクリプトへの組み込みは `fix/ci-doc-consistency` で対応中。\n"
        )
        script = root / "scripts/pre-merge-audit.sh"
        script.write_text(
            script.read_text().replace(
                "cargo nextest run --release --profile heavy --run-ignored all\n", ""
            )
        )
        _, failures = self.evaluate(root)
        self.assertFalse(
            any(f.claim == "pending_pointer_freshness" for f in failures),
            [f for f in failures if f.claim == "pending_pointer_freshness"],
        )
        self.assertTrue(
            any(
                f.claim == "premerge_gate_commands" and f.key == "missing::heavy"
                for f in failures
            )
        )

    def test_premerge_missing_heavy_fails(self):
        root = self.fixture()
        script = root / "scripts/pre-merge-audit.sh"
        script.write_text(
            script.read_text().replace(
                "cargo nextest run --release --profile heavy --run-ignored all\n", ""
            )
        )
        self.assert_failure(root, "premerge_gate_commands", "missing::heavy")

    def test_premerge_doctest_without_release_fails(self):
        root = self.fixture()
        script = root / "scripts/pre-merge-audit.sh"
        script.write_text(script.read_text().replace("cargo test --doc --release", "cargo test --doc"))
        self.assert_failure(
            root, "premerge_gate_commands", "missing::doctest-release", "cargo test --doc"
        )

    def test_missing_catalog_raises_structure_error(self):
        root = self.fixture()
        (root / "tests/test_inventory.txt").unlink()
        with self.assertRaises(gate.StructureError):
            gate.load_catalog(root)

    def test_catalog_accepts_nextest_json(self):
        # A `cargo nextest list --message-format json` dump is an accepted
        # catalog source, so the filter liveness check can run against a live
        # list without rebuilding (the default committed inventory stays used
        # when --catalog is not given).
        root = self.fixture()
        catalog = root / "nextest-list.json"
        catalog.write_text(
            '{"rust-suites": {'
            '"fixture::suite": {"binary-id": "fixture::suite", "testcases": '
            '{"fixture_case": {"kind": "test"}, "fixture_heavy": {"kind": "test"}},'
            '"other::suite": {"binary-id": "other::suite", "testcases": {}}}}}'
        )
        loaded = gate.load_catalog(root, catalog)
        self.assertEqual(loaded.cases, (("fixture::suite", "fixture_case"), ("fixture::suite", "fixture_heavy")))
        _, failures = gate.evaluate(root, catalog_path=catalog)
        self.assertEqual(failures, [])
        catalog.write_text('{"rust-suites": {"x": {"testcases": null}}}')
        with self.assertRaises(gate.StructureError):
            gate.load_catalog(root, catalog)

    def test_echoed_nextest_command_is_not_an_invocation(self):
        root = self.fixture()
        path = root / ".github/workflows/ci.yml"
        path.write_text(
            path.read_text()
            + '      - run: echo "cargo nextest run --release --test-threads 9"\n'
        )
        _, failures = self.evaluate(root)
        self.assertFalse(any(f.claim == "nextest_ci_threads" for f in failures))

    def test_workflow_block_continuation_is_parsed(self):
        root = self.fixture()
        commands = gate.iter_run_commands(
            root / ".github/workflows/test-heavy.yml", root
        )
        heavy = [cmd for cmd in commands if "cargo nextest run" in cmd.text]
        self.assertEqual(len(heavy), 1)
        self.assertIn("--profile heavy", heavy[0].text)
        self.assertIn("--test-threads 3", heavy[0].text)

    def test_filter_union_and_exact_semantics(self):
        cases = [
            ("suite", "mod::alpha"),
            ("suite", "beta"),
            ("other", "alpha2"),
        ]
        self.assertEqual(
            gate._filter_matches("test(alpha) + binary(other)", cases),
            {("suite", "mod::alpha"), ("other", "alpha2")},
        )
        self.assertEqual(
            gate._filter_matches("test(=beta)", cases), {("suite", "beta")}
        )
        self.assertIsNone(gate._filter_matches("test(alpha) - test(beta)", cases))

    def test_filter_intersection_semantics(self):
        cases = [
            ("suite", "mod::alpha"),
            ("suite", "beta"),
            ("other", "alpha2"),
        ]
        self.assertEqual(
            gate._filter_matches("binary(suite) & test(mod::alpha)", cases),
            {("suite", "mod::alpha")},
        )
        self.assertEqual(
            gate._filter_matches("binary(suite) & test(=beta)", cases),
            {("suite", "beta")},
        )
        # an unknown conjunct narrows the set to empty (0-match, not unknown)
        self.assertEqual(
            gate._filter_matches("binary(no_such) & test(=beta)", cases), set()
        )
        # `+` binds looser than `&`: each union branch intersects on its own
        self.assertEqual(
            gate._filter_matches("test(=beta) + binary(other) & test(=alpha2)", cases),
            {("suite", "beta"), ("other", "alpha2")},
        )
        # fail-closed is kept for unknown syntax inside or beside a `&`
        self.assertIsNone(gate._filter_matches("not(test(x)) & test(beta)", cases))
        self.assertIsNone(
            gate._filter_matches("binary(suite) & test(beta) - test(=beta)", cases)
        )

    def test_known_drift_suppresses_exact_key_and_becomes_stale(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_test)"\n'
        )
        _, failures = self.evaluate(root)
        self.assertEqual(len(failures), 1)
        debt = gate.KnownDrift(failures[0].claim, failures[0].key, "fixture")
        new, known, stale = gate.classify(failures, (debt,), strict=False)
        self.assertEqual(new, [])
        self.assertEqual(len(known), 1)
        self.assertEqual(stale, [])

        path.write_text(NEXTEST)
        _, fixed = self.evaluate(root)
        self.assertEqual(fixed, [])
        new, known, stale = gate.classify(fixed, (debt,), strict=False)
        self.assertEqual(new, [])
        self.assertEqual(known, [])
        self.assertEqual(stale, [debt])

    def test_new_failure_is_not_absorbed_by_known_drift(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_a)"\n'
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_b)"\n'
        )
        _, failures = self.evaluate(root)
        self.assertEqual(len(failures), 2)
        debt = gate.KnownDrift(failures[0].claim, failures[0].key, "fixture")
        new, known, stale = gate.classify(failures, (debt,), strict=False)
        self.assertEqual(len(new), 1)
        self.assertEqual(len(known), 1)
        self.assertEqual(stale, [])

    def test_strict_does_not_suppress_known_drift(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_test)"\n'
        )
        _, failures = self.evaluate(root)
        debt = gate.KnownDrift(failures[0].claim, failures[0].key, "fixture")
        new, known, stale = gate.classify(failures, (debt,), strict=True)
        self.assertEqual(len(new), 1)
        self.assertEqual(known, [])
        self.assertEqual(stale, [])

    def test_line_shift_does_not_change_failure_keys(self):
        root = self.fixture()
        path = root / ".config/nextest.toml"
        path.write_text(
            path.read_text()
            + '\n[[profile.default.overrides]]\nfilter = "test(no_such_test)"\n'
        )
        _, before = self.evaluate(root)
        agents = root / "AGENTS.md"
        agents.write_text("\n\n" + agents.read_text())
        _, after = self.evaluate(root)
        self.assertEqual(self.claims(before), self.claims(after))

    def test_unrelated_doc_mutation_only_changes_its_claim(self):
        root = self.fixture()
        _, before = self.evaluate(root)
        self.assertEqual(before, [])
        text = (root / "AGENTS.md").read_text().replace(
            "ローカル 6 スレッド / CI 3", "ローカル 7 スレッド / CI 3"
        )
        (root / "AGENTS.md").write_text(text)
        _, after = self.evaluate(root)
        self.assertEqual({f.claim for f in after}, {"nextest_local_threads"})


if __name__ == "__main__":
    unittest.main()
