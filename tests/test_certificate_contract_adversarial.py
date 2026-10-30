"""契約ツール破棄経路と契約定数の sentinel (レビュー B P2)。

既存の tests/test_certificate_contract.py は受理経路と code freeze を中心に検証するが、
(1) 契約定数そのもの、(2) enclosure 欠落候補の拒否、(3) previous 値乖離検出、
(4) checker_digest の対象集合、(5) 偽造 proof・保存時ガード、(6) LISWET 行の
厳密経路ルーティング は pin していなかった。各テストは対応する実装を revert
または緩和すると fail する (実測済みの sentinel)。
"""
import contextlib
import gzip
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
# highs_baseline_truth を先に spec 読み込みすると scripts/ が sys.path に入る
# (test_certificate_contract.py と同一の import 手順)。
SPEC = importlib.util.spec_from_file_location("truth", ROOT / "scripts/highs_baseline_truth.py")
truth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(truth)
sys.modules.setdefault("highs_baseline_truth", truth)
# 共有インスタンス再バインド (test_certificate_contract.py と同理由): 後続
# spec 読込ファイルの patch が sys.modules 越しに無効化されるのを防ぐ。
truth = sys.modules["highs_baseline_truth"]
import baseline_certificate as checker
import liswet_exact_certificate as liswet
import revalidate_baselines as revalidation


def scalar_problem():
    # min 1/2 x^2 - 3x over x >= 0: 手計算 optimum -9/2 at x = 3。
    return dict(c0=0.0, c=np.array([-3.0]), A=sp.csr_matrix((0, 1)),
                L=np.zeros(0), U=np.zeros(0), lo=np.zeros(1),
                up=np.array([math.inf]), Q=sp.eye(1, format="csr"), n=1, m=0)


class ContractConstantPins(unittest.TestCase):
    """契約文書 (docs/baseline-certificate-contract.md) の数値リテラル pin。

    定数を緩めると (例 1e-7 -> 1e-6) このテストが fail し、契約の緩和を
    暗黙の diff にできない。
    """

    def test_reference_tolerance_is_one_in_ten_million(self):
        self.assertEqual(checker.REFERENCE_TOLERANCE, Fraction(1, 10_000_000))
        self.assertEqual(float(checker.REFERENCE_TOLERANCE), 1e-7)
        self.assertEqual(checker.CHECKER_VERSION, "binary64-enclosure-v1")

    def test_benchmark_tolerance_derivations_are_pinned(self):
        # 契約は「ベンチ既定許容 1e-6 の 1/10」を共有定数で表現する。
        self.assertEqual(truth.BENCH_EPS, 1e-6)
        self.assertEqual(truth.REL_GAP_TOL, truth.BENCH_EPS / 10)
        self.assertEqual(truth.REL_GAP_TOL, 1e-7)
        self.assertEqual(truth.UNCERTAINTY_TOL, truth.REL_GAP_TOL)
        self.assertEqual(truth.OBJ_CONSISTENCY_TOL, truth.BENCH_EPS * 1e-3)

    def test_csv_mantissa_decimals_are_pinned_to_twelve(self):
        # 13 有効桁 (仮数部 12 桁) は共有契約。二モジュールが同一値であることも pin。
        self.assertEqual(truth.CSV_MANTISSA_DECIMALS, 12)
        self.assertEqual(liswet.CSV_MANTISSA_DECIMALS, 12)
        self.assertEqual(liswet.CSV_MANTISSA_DECIMALS, truth.CSV_MANTISSA_DECIMALS)


class CheckerDigestStructureTests(unittest.TestCase):
    """checker_digest の対象集合・フレーム構造・書式分離の sentinel。
    	対象リストは実装側 CHECKER_FILES から derive する (片側二重管理の解消)。
    	・v1 (生 bytes 連結) で生存した分割シフト (shebang を他ファイル末尾へ
    	  移す PoC) で digest が変わることを digest_of で直接 pin。
    	・v1 値は v2 の値域に到達せず、worker の pinned digest として拒否される
    	  (旧書式 digest の混在受理禁止)。
    """

    @staticmethod
    def target_files():
        return checker.CHECKER_FILES

    def test_target_file_list_is_nonempty_and_present(self):
        files = self.target_files()
        self.assertTrue(len(files) >= 1)
        for name in files:
            path = ROOT / "scripts" / name
            self.assertTrue(path.is_file(), msg=name)
            self.assertGreater(path.stat().st_size, 0, msg=name)

    @staticmethod
    def digest_from_copy(tamper):
        # コピー先ディレクトリで baseline_certificate を再 import すると、その
        # 時点の全対象ファイル bytes から LOADED_CHECKER_DIGEST が計算される。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in CheckerDigestStructureTests.target_files():
                source = ROOT / "scripts" / name
                data = source.read_bytes()
                if name == tamper:
                    data = data + b"\n# tampered\n"
                (root / name).write_bytes(data)
            spec = importlib.util.spec_from_file_location("digest_probe", root / "baseline_certificate.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.LOADED_CHECKER_DIGEST

    def test_untampered_copy_hashes_to_the_running_digest(self):
        self.assertEqual(self.digest_from_copy(None), checker.checker_digest())

    def test_every_targeted_file_is_included_in_the_digest(self):
        for name in self.target_files():
            with self.subTest(file=name):
                self.assertNotEqual(self.digest_from_copy(name), checker.checker_digest())

    def test_loaded_digest_matches_fresh_digest(self):
        # worker の code freeze は LOADED と実行時 digest の一致に依存する。
        self.assertEqual(revalidation.LOADED_CHECKER_DIGEST, checker.checker_digest())

    def test_partition_shift_across_files_changes_digest(self):
        # レビュー PoC: 同じバイト列をファイル間で切る位置をずらす (境界シフト)
        # と v1 連結は完全一致だった。v2 はフレーム (name/length/content-hash)
        # 単位のため digest が変わる。
        stream = b"#!/bin/shift\nrest-of-a" + b"tail-of-b"
        cut_a = {"a.py": stream[:18], "b.py": stream[18:]}
        cut_b = {"a.py": stream[:15], "b.py": stream[15:]}  # 3 バイト後方へ移動
        self.assertEqual(cut_a["a.py"] + cut_a["b.py"], cut_b["a.py"] + cut_b["b.py"])
        self.assertNotEqual(checker.digest_of(cut_a), checker.digest_of(cut_b))

    def test_length_resegmentation_changes_digest(self):
        # 連結結果は同一 (b"ab") のまま境界を 1 バイトずらす移動は、
        # name+length フレームにより v2 では検出される。
        self.assertNotEqual(checker.digest_of({"a.py": b"ab", "b.py": b""}),
                            checker.digest_of({"a.py": b"a", "b.py": b"b"}))

    def test_content_swap_between_files_changes_digest(self):
        self.assertNotEqual(
            checker.digest_of({"a.py": b"AAA", "b.py": b"BBB"}),
            checker.digest_of({"a.py": b"BBB", "b.py": b"AAA"}))

    @staticmethod
    def v1_digest_of_repo_files():
        root = ROOT / "scripts"
        return hashlib.sha256(b"".join((root / name).read_bytes()
                                       for name in CheckerDigestStructureTests.target_files())).hexdigest()

    def test_v1_digest_value_is_unreachable_and_rejected_as_pin(self):
        v1 = self.v1_digest_of_repo_files()
        self.assertNotEqual(v1, checker.checker_digest())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = RevalidationPreviousValueTests.scalar_case(root, "-4.500000000000e0")
            config["checker_sha256"] = v1  # 旧書式 digest を pinned 値に混ぜる試み
            result = RevalidationPreviousValueTests.run_worker(config, root)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "checker_code_changed")


class ReplayEvidenceRejectionTests(unittest.TestCase):
    """保存された bounds/model/reference/input hash の改竄を各経路で拒否する。"""

    INPUT_HASH = hashlib.sha256(b"synthetic replay input").hexdigest()

    def honest_payload(self):
        model = checker.ExactModel(scalar_problem())
        enclosure = model.check([Fraction(3)], [Fraction(3)], [])
        payload = checker.evidence(enclosure, "-4.500000000000e0", self.INPUT_HASH)
        return enclosure, payload

    def test_honest_payload_replays(self):
        # 対照群: 真正な proof は必ず通る。拒否系を「全部弾く」実装への退行検出。
        _, payload = self.honest_payload()
        enclosure = checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)
        self.assertEqual(enclosure.lower, Fraction(-9, 2))

    def test_forged_stored_bounds_are_rejected(self):
        # 保存 bounds を再計算と不一致に改竄 (enclosure を 0.1 だけ広げる)。
        _, payload = self.honest_payload()
        payload["lower"] = str(Fraction(payload["lower"]) - Fraction(1, 10))
        with self.assertRaisesRegex(ValueError, "stored bounds"):
            checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)

    def test_forged_model_hash_is_rejected(self):
        _, payload = self.honest_payload()
        payload["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)

    def test_forged_reference_budget_breach_is_rejected(self):
        # tolerance 改竄: bounds は正当だが reference を enclosure 外へ書き換える。
        _, payload = self.honest_payload()
        payload["reference"] = "-4.400000000000e0"
        with self.assertRaisesRegex(ValueError, "error budget"):
            checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)

    def test_forged_input_hash_is_rejected(self):
        _, payload = self.honest_payload()
        payload["input_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input hash mismatch"):
            checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)


class SaveEvidenceGuardTests(unittest.TestCase):
    """save_evidence の保存前ガード (予算・digest) の発動と無副作用を検証する。"""

    def setUp(self):
        self._temp_dirs = []

    def tearDown(self):
        for path in self._temp_dirs:
            shutil.rmtree(path, ignore_errors=True)

    def scratch_dir(self):
        path = Path(tempfile.mkdtemp())
        self._temp_dirs.append(path)
        return path

    @staticmethod
    def enclosure():
        return checker.ExactModel(scalar_problem()).check([Fraction(3)], [Fraction(3)], [])

    def test_reference_outside_budget_raises_and_writes_nothing(self):
        path = self.scratch_dir() / "SCALAR.proof.json.gz"
        with self.assertRaisesRegex(ValueError, "outside its certified error budget"):
            checker.save_evidence(path, self.enclosure(), "-4.400000000000e0",
                                  hashlib.sha256(b"x").hexdigest())
        self.assertFalse(path.exists())

    def test_stale_expected_digest_raises_and_writes_nothing(self):
        path = self.scratch_dir() / "SCALAR.proof.json.gz"
        with self.assertRaisesRegex(ValueError, "checker changed during verification"):
            checker.save_evidence(path, self.enclosure(), "-4.500000000000e0",
                                  hashlib.sha256(b"x").hexdigest(),
                                  expected_digest="0" * 64)
        self.assertFalse(path.exists())


class CertifyAcceptanceTests(unittest.TestCase):
    """certify の受理条件が verified enclosure の存在であることを固定する。

    solver status=Optimal・報告値と再計算値の一致 (objdiff=0)・残差小を揃えても、
    enclosure が計算できなければ拒否。旧受理ロジック (status/残差ベース) への
    revert では下のテストが fail する。
    """

    def test_infeasible_consistent_optimal_claim_is_rejected(self):
        prob = scalar_problem()
        prob["up"] = np.array([2.0])
        # 上界を 2^-20 だけ超える点。報告値は再計算と一致させ、違反は bench 許容
        # より小さい (残差・status ベースなら受理してしまう候補)。
        x = np.array([2.0 + 2.0 ** -20])
        reported = float(prob["c0"] + prob["c"] @ x + 0.5 * x @ (prob["Q"] @ x))
        with mock.patch.object(truth, "polish", return_value=None), \
             mock.patch.object(truth, "_exact_certificate", return_value=None), \
             mock.patch.object(truth, "_sparse_exact_certificate", return_value=None):
            ok, _, msg = truth.certify(prob, ("Optimal", x, np.zeros(0), reported))
        self.assertFalse(ok)
        # 拒否理由は objdiff ではなく enclosure 欠落であること: msg 内の objdiff は 0。
        self.assertIn("reported_vs_p=0.000e+00", msg)


class RevalidationPreviousValueTests(unittest.TestCase):
    """「Revalidation compares every row with its previous value」の発動テスト。

    previous 値が enclosure から reference-budget 外へ乖離したら再認証値を再発行し、
    budget 内なら公開済みの文字列をそのまま保持する。previous 値を無条件に再発行
    する実装 (比較を省いた revert) では drift テストが fail する。
    """

    @staticmethod
    def scalar_case(root, previous):
        data, candidates, out = root / "data", root / "candidates", root / "out"
        for directory in (data, candidates, out):
            directory.mkdir()
        qps = data / "SCALAR.QPS"
        qps.write_text("NAME SCALAR\nROWS\n N OBJ\nCOLUMNS\n    X1  OBJ -3\nRHS\n"
                       "BOUNDS\n FR B X1\nQUADOBJ\n    X1  X1 1\nENDATA\n")
        np.savez(candidates / "SCALAR.npz", x=np.array([3.0]), y=np.zeros(0))
        (candidates / "SCALAR.json").write_text(json.dumps(
            {"input_sha256": hashlib.sha256(qps.read_bytes()).hexdigest()}))
        return dict(rows=[dict(problem_name="SCALAR", optimal_obj=previous,
                               source="previous_cert")],
                    data_dir=str(data), candidate_dir=str(candidates), output_dir=str(out),
                    gaps={}, timeout=60, checker_sha256=checker.checker_digest())

    @staticmethod
    def run_worker(config, root):
        revalidation.write_json(root / "run.json", config)
        revalidation.worker(str(root / "run.json"), "SCALAR")
        return json.loads((root / "out" / "SCALAR.result.json").read_text())

    def test_drifted_previous_value_is_replaced_by_certified_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.scalar_case(root, "-4.400000000000e0")
            result = self.run_worker(config, root)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["old_value"], "-4.400000000000e0")
            self.assertEqual(result["reference"], "-4.500000000000e0")
            code = revalidation.finalize([result], root / "out", config["checker_sha256"])
            self.assertEqual(code, 0)
            csv_text = (root / "out" / "baseline.csv").read_text()
            self.assertIn("SCALAR,-4.500000000000e0,exact_enclosure_v1", csv_text)
            self.assertNotIn("-4.4", csv_text)

    def test_previous_value_within_budget_is_retained_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 前回公開値は真値から 5e-8 (relative 9.1e-9 <= 1e-7) ずれている。
            config = self.scalar_case(root, "-4.500000050000e0")
            result = self.run_worker(config, root)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["reference"], "-4.500000050000e0")


# LISWET 構造 (行 i = x_i - 2 x_{i+1} + x_{i+2}, Q = I, 全変数 FR, 行境界 >= 0)。
# test_liswet_exact_certificate.py の hand-computed one_gap ケースを再利用:
# x = (1,1,1,1,2,3) に行 2 のみ無効 (二次差分 1)、大域最適値 -8 (c0 = 1/2 込み)。
LISWET_X = [Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(2), Fraction(3)]
LISWET_ATY = [Fraction(1), Fraction(0), Fraction(-3), Fraction(3), Fraction(-2), Fraction(1)]


class RevalidateLiswetRoutingTests(unittest.TestCase):
    """revalidate の LISWET 行は候補 (.npz) 経路でなく厳密帯 KKT 経路で証明される。

    candidate 用のメタデータを hash 不一致で poisoned し、npz 自体を置かない。
    LISWET 分岐を外して候補経路に流す revert では missing_candidate /
    candidate_input_hash_mismatch で拒否され、このテストは fail する。
    """

    @staticmethod
    def write_liswet(path):
        c = [LISWET_ATY[j] - LISWET_X[j] for j in range(6)]
        rows = [{i: 1.0, i + 1: -2.0, i + 2: 1.0} for i in range(4)]
        lines = ["NAME LISWETS", "ROWS", " N OBJ"]
        lines += [f" G R{i + 1}" for i in range(4)]
        lines.append("COLUMNS")
        for j in range(6):
            pairs = ["OBJ", str(float(c[j]))]
            for i, row in enumerate(rows):
                if j in row:
                    pairs += [f"R{i + 1}", str(row[j])]
            lines.append(f"    X{j + 1}  " + "  ".join(pairs))
        lines += ["RHS", "    RHS OBJ -0.5", "BOUNDS"]
        lines += [f" FR BOUNDS X{j + 1}" for j in range(6)]
        lines += ["QUADOBJ"] + [f"    X{j + 1} X{j + 1} 1" for j in range(6)]
        lines.append("ENDATA")
        path.write_text("\n".join(lines) + "\n")

    def test_liswet_row_uses_exact_path_not_candidate_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, candidates, out = root / "data", root / "candidates", root / "out"
            for directory in (data, candidates, out):
                directory.mkdir()
            self.write_liswet(data / "LISWETS.QPS")
            # poison: npz は無く、json の input_sha256 も意図的に不正な値。
            (candidates / "LISWETS.json").write_text(json.dumps({"input_sha256": "0" * 64}))
            config = dict(rows=[dict(problem_name="LISWETS", optimal_obj="-8.000000000000e0",
                                      source="previous_cert")],
                          data_dir=str(data), candidate_dir=str(candidates),
                          output_dir=str(out), gaps={"LISWETS": [2]}, timeout=60,
                          checker_sha256=checker.checker_digest())
            revalidation.write_json(root / "run.json", config)
            revalidation.worker(str(root / "run.json"), "LISWETS")
            result = json.loads((out / "LISWETS.result.json").read_text())
            self.assertTrue(result["accepted"], msg=result.get("reason"))
            self.assertEqual(result["reason"], "certified")
            self.assertEqual(result["reference"], "-8.000000000000e0")
            # 候補経路の痕跡 (method) が無く、proposal は liswet_banded であること。
            self.assertNotIn("method", result)
            self.assertEqual(result["proposal"]["method"], "liswet_banded")
            self.assertEqual(result["proposal"]["gaps"], [2])
            # 保存 proof は共有 enclosure で replay できる (厳密経路の証跡)。
            # result["proof"] の CSV 基準相対参照で解決できることも同時に pin。
            payload = json.loads(gzip.decompress(
                (out / result["proof"]).read_bytes()).decode())
            checker.replay_evidence(truth.load_qps(str(data / "LISWETS.QPS")), payload,
                                    result["input_sha256"])


class OutputSourceProvenanceTests(unittest.TestCase):
    """受理行の source 列は契約が要求する traceability (CSV 基準の proof 参照・
    proof 内容 digest・入力 hash・checker version) を載せる。必須キー欠落時は
    旧書式 (exact_enclosure_v1 素) へ silent に劣化させず raise する fail-closed の
    sentinel。reviewer 指摘「proof= が basename のみ・sha256 なし・キー欠落時
    silent 劣化」の解消。"""

    @staticmethod
    def full_result():
        return dict(accepted=True, problem_name="A", reference="1.0",
                    proof="proofs/A.proof.json.gz", proof_sha256="c" * 64,
                    input_sha256="a" * 64, checker_sha256="b" * 64)

    def test_accepted_source_carries_relative_proof_and_all_hashes(self):
        self.assertEqual(revalidation.output_source(self.full_result()),
                         "exact_enclosure_v1;proof=proofs/A.proof.json.gz;"
                         "proof_sha256=" + "c" * 64 + ";"
                         "input_sha256=" + "a" * 64 + ";checker_sha256=" + "b" * 64)

    def test_accepted_source_missing_any_provenance_key_raises(self):
        for key in revalidation.CLAIM_KEYS:
            broken = self.full_result()
            del broken[key]
            with self.assertRaisesRegex(ValueError, key):
                revalidation.output_source(broken)

    def test_accepted_source_empty_provenance_value_raises(self):
        # キーがあっても空値は provenance でない (無言の項目欠落への sentinel)。
        for key in revalidation.CLAIM_KEYS:
            broken = self.full_result()
            broken[key] = ""
            with self.assertRaisesRegex(ValueError, key):
                revalidation.output_source(broken)

    def test_failed_source_keeps_uncertified_suffix(self):
        result = dict(accepted=False, reason="missing_candidate")
        self.assertEqual(revalidation.output_source(result),
                         "revalidation_missing_candidate_UNCERTIFIED")


class SourceColumnResolveTests(unittest.TestCase):
    """finalize が書き出す source 列の proof= 参照は CSV の位置から解決でき、
    proof_sha256 は trust でなく独立再計算で実ファイルと一致する。

    proof= を basename (または CSV 基準で解けない絶対配置) へ、あるいは
    proof_sha256 を run 時の記憶値 (ファイル実体と無関係) へ戻す revert では
    このテストが fail する。
    """

    def test_written_csv_resolves_its_own_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = RevalidationPreviousValueTests.scalar_case(root, "-4.500000000000e0")
            result = RevalidationPreviousValueTests.run_worker(config, root)
            self.assertTrue(result["accepted"], msg=result.get("reason"))
            code = revalidation.finalize([result], root / "out", config["checker_sha256"])
            self.assertEqual(code, 0)
            csv_path = root / "out" / "baseline.csv"
            rows = list(csv.DictReader(line for line in csv_path.read_text().splitlines()
                                       if not line.startswith("#")))
            self.assertEqual([row["problem_name"] for row in rows], ["SCALAR"])
            fields = dict(item.split("=", 1) for item in rows[0]["source"].split(";")[1:])
            self.assertEqual(set(fields), set(revalidation.CLAIM_KEYS))
            # 配置も pin: 証跡は CSV 基準の proofs/ 副次目录下 (committed layout と同形)。
            self.assertEqual(fields["proof"], revalidation.proof_rel("SCALAR"))
            proof = (csv_path.parent / fields["proof"]).resolve()  # CSV 基準の解決
            self.assertTrue(proof.is_file())
            self.assertEqual(fields["proof_sha256"],
                             hashlib.sha256(proof.read_bytes()).hexdigest())
            self.assertEqual(fields["input_sha256"], hashlib.sha256(
                (root / "data" / "SCALAR.QPS").read_bytes()).hexdigest())
            self.assertEqual(fields["checker_sha256"], checker.checker_digest())


class CertifyObjdiffGuardTests(unittest.TestCase):
    """certify 3 接眼のうち objdiff 接眼の哨兵 (敵対レビュー M02 生存対応)。

    enclosure も reference 予算も無傷で objdiff だけが壊れる受理 (報告値だけを
    真の再計算値から ~1.8e-5 ずらした solved) を拒否し、拒否理由を msg の
    reported_vs_p で固定する。接眼を 1 つ外した変異では必ず fail する。
    """

    def test_report_shift_is_rejected_with_objdiff_reason(self):
        prob = scalar_problem()
        x = np.array([3.0])
        honest = float(prob["c0"] + prob["c"] @ x + 0.5 * x @ (prob["Q"] @ x))
        ok, _, msg = truth.certify(prob, ("Optimal", x, np.zeros(0), honest))
        self.assertTrue(ok, msg=msg)  # 対照: 報告値が正直なら受理 (enclosure 無傷)
        # |r-(-4.5)|/(1+4.5) = 9.9e-5/5.5 = 1.8e-5 (OBJ_CONSISTENCY_TOL=1e-9 の 4 桁上)
        ok, _, msg = truth.certify(prob, ("Optimal", x, np.zeros(0), -4.500099))
        self.assertFalse(ok)
        self.assertIn("reported_vs_p=1.800e-05", msg)


class CertifyReferenceBudgetProbeTests(unittest.TestCase):
    """certify の reference_error 接眼の効果窓を絶対値境界 probe で固定 (M03 対応)。

    enclosure が計算できて objdiff=0 でも、CSV 値 (format_csv_value(p)) が
    enclosure から相対 1.1e-7 以上離れていれば拒否。site に倍率 (×1000 等) が
    混入しても 1.1e-7 probe が受理されて fail する。期待境界は
    REFERENCE_TOLERANCE からderive せず絶対リテラルで書く (定数との関係は
    下記の独立 assert で pin)。
    """

    @staticmethod
    def cert_with_lower(text):
        # r = format_csv_value(0) = 0 なので reference_error = |lower|。
        enclosure = checker.Enclosure(-Fraction(text), Fraction(0),
                                      (Fraction(3),), (Fraction(3),), (), "synthetic")
        return truth.Certificate(Fraction(0), Fraction(0), 0.0, 0.0, 0.0, 0,
                                 "synthetic", (Fraction(3),), enclosure)

    def probe(self, lower):
        prob = scalar_problem()
        solved = ("Optimal", np.array([3.0]), np.zeros(0), -4.5)
        with mock.patch.object(truth, "best_certificate",
                               return_value=self.cert_with_lower(lower)):
            return truth.certify(prob, solved)

    def test_constants_under_test_are_absolute(self):
        # probe は絶対リテラル。定数との対応关系はここで独立に pin する。
        self.assertEqual(checker.REFERENCE_TOLERANCE, Fraction(1, 10_000_000))
        self.assertLess(float(Fraction("9.9e-8")), float(checker.REFERENCE_TOLERANCE))
        self.assertGreater(float(Fraction("1.1e-7")), float(checker.REFERENCE_TOLERANCE))

    def test_budget_window_accepts_relative_9_9e_minus_8(self):
        ok, _, msg = self.probe("9.9e-8")
        self.assertTrue(ok, msg=msg)
        self.assertIn("uncertainty=9.900e-08", msg)

    def test_budget_window_rejects_relative_1_1e_minus_7(self):
        ok, _, msg = self.probe("1.1e-7")
        self.assertFalse(ok)
        self.assertIn("uncertainty=1.100e-07", msg)

    def test_budget_window_rejects_the_1_4e_minus_6_survivor_scale(self):
        # M03 生存変異が実際に受理した規模 (相対 ~1.4e-6) を明示的に拒否。
        ok, _, msg = self.probe("1.4e-6")
        self.assertFalse(ok)
        self.assertIn("uncertainty=1.400e-06", msg)


class BudgetSiteBoundaryProbeTests(unittest.TestCase):
    """REFERENCE_TOLERANCE の各使用 site の効果窓を 9.9e-8/1.1e-7 で 1 本ずつ pin。

    「定数の値」でなく「使用 site の効果窓」が目的。site に倍率が混入すれば
    (×1000 変異は rel 9.1e-5 を認証した) 1.1e-7 probe が通過して fail する。
    境界: |r-(-4.5)| = 5.445e-7 → /(1+4.5+) = 9.9e-8-, 6.05e-7 → 1.1e-7+。
    """

    INSIDE = "-4.500000544500e0"    # 相対 9.9e-8 (以内: 受理側)
    OUTSIDE = "-4.500000605000e0"   # 相対 1.1e-7 (外れ: 拒否側)
    HONEST = "-4.500000000000e0"

    def setUp(self):
        self._temp_dirs = []

    def tearDown(self):
        for path in self._temp_dirs:
            shutil.rmtree(path, ignore_errors=True)

    def scratch_dir(self):
        path = Path(tempfile.mkdtemp())
        self._temp_dirs.append(path)
        return path

    @staticmethod
    def tight_enclosure():
        return checker.ExactModel(scalar_problem()).check([Fraction(3)], [Fraction(3)], [])

    def test_save_evidence_accepts_inside_budget(self):
        path = self.scratch_dir() / "SCALAR.proof.json.gz"
        compact = checker.save_evidence(path, self.tight_enclosure(), self.INSIDE,
                                        hashlib.sha256(b"x").hexdigest())
        self.assertTrue(path.is_file())
        self.assertEqual(compact["reference"], self.INSIDE)

    def test_save_evidence_rejects_outside_budget(self):
        path = self.scratch_dir() / "SCALAR.proof.json.gz"
        with self.assertRaisesRegex(ValueError, "outside its certified error budget"):
            checker.save_evidence(path, self.tight_enclosure(), self.OUTSIDE,
                                  hashlib.sha256(b"x").hexdigest())
        self.assertFalse(path.exists())

    INPUT_HASH = hashlib.sha256(b"synthetic replay input").hexdigest()

    def test_replay_accepts_inside_budget(self):
        payload = checker.evidence(self.tight_enclosure(), self.HONEST, self.INPUT_HASH)
        payload["reference"] = self.INSIDE
        enclosure = checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)
        self.assertEqual(enclosure.lower, Fraction(-9, 2))

    def test_replay_rejects_outside_budget(self):
        payload = checker.evidence(self.tight_enclosure(), self.HONEST, self.INPUT_HASH)
        payload["reference"] = self.OUTSIDE
        with self.assertRaisesRegex(ValueError, "error budget"):
            checker.replay_evidence(scalar_problem(), payload, self.INPUT_HASH)

    def run_with_previous(self, previous):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = RevalidationPreviousValueTests.scalar_case(root, previous)
            result = RevalidationPreviousValueTests.run_worker(config, root)
            self.assertTrue(result["accepted"], msg=result.get("reason"))
            return result

    def test_retain_site_keeps_previous_at_boundary_inside(self):
        self.assertEqual(self.run_with_previous(self.INSIDE)["reference"], self.INSIDE)

    def test_retain_site_replaces_previous_at_boundary_outside(self):
        self.assertEqual(self.run_with_previous(self.OUTSIDE)["reference"], self.HONEST)


class InputToctouTests(unittest.TestCase):
    """検証途中の入力改変 (TOCTOU) を evaluate_row と liswet main が拒否する哨兵。

    検証開始後に QPS 本体を書き換えても受理される変異 (再 hash 比較の削除) では
    両テストが fail する。
    """

    def test_worker_rejects_qps_mutated_during_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = RevalidationPreviousValueTests.scalar_case(root, "-4.500000000000e0")
            qps = root / "data" / "SCALAR.QPS"
            real = truth.best_certificate

            def mutate_during_check(prob, x, y):
                cert = real(prob, x, y)
                qps.write_bytes(qps.read_bytes() + b"\n# mutated after hashing\n")
                return cert

            truth.best_certificate = mutate_during_check
            try:
                result = RevalidationPreviousValueTests.run_worker(config, root)
            finally:
                truth.best_certificate = real
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "error")
            self.assertIn("input changed during verification", result["error"])
            self.assertFalse((root / "out" / revalidation.proof_rel("SCALAR")).exists())

    # 独立オラクル済みの SYNTH3 (x=0 で 7、DirectCsvProvenanceTests と同一モデル)。
    SYNTH3 = ("NAME SYNTH3\nROWS\n N OBJ\n G R1\nCOLUMNS\n"
              "    X1  OBJ 0  R1 1\n    X2  OBJ 0  R1 -2\n    X3  OBJ 0  R1 1\n"
              "RHS\n    RHS OBJ -7\nBOUNDS\n FR B X1\n FR B X2\n FR B X3\n"
              "QUADOBJ\n    X1 X1 1\n    X2 X2 1\n    X3 X3 1\nENDATA\n")

    def test_liswet_main_rejects_qps_mutated_during_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qps = root / "SYNTH3.QPS"
            qps.write_text(self.SYNTH3)
            evidence = root / "ev"
            real = liswet.certify

            def mutate_after_certify(path, *args, **kwargs):
                cert = real(path, *args, **kwargs)
                qps.write_bytes(qps.read_bytes() + b"\n# mutated\n")
                return cert

            liswet.certify = mutate_after_certify
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    code = liswet.main(["--csv", "--evidence-dir", str(evidence), str(qps)])
            finally:
                liswet.certify = real
            self.assertEqual(code, 1)
            self.assertIn("input changed during verification", err.getvalue())
            self.assertFalse((evidence / "SYNTH3.proof.json.gz").exists())


class GapsJsonPropagationTests(unittest.TestCase):
    """LISWET 行の gaps は gaps.json (config) が勝つ。フォールバック提案を
    参照した変異 (M19) では solve_highs が呼ばれて AssertionError となり、
    accepted=False でこのテストが fail する。
    """

    def test_gaps_from_config_win_over_fallback_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, candidates, out = root / "data", root / "candidates", root / "out"
            for directory in (data, candidates, out):
                directory.mkdir()
            RevalidateLiswetRoutingTests.write_liswet(data / "LISWETS.QPS")
            (candidates / "LISWETS.json").write_text(json.dumps(
                {"input_sha256": "0" * 64, "gaps": [0, 1, 3]}))  # 候補側は別物
            config = dict(rows=[dict(problem_name="LISWETS",
                                      optimal_obj="-8.000000000000e0",
                                      source="previous_cert")],
                          data_dir=str(data), candidate_dir=str(candidates),
                          output_dir=str(out), gaps={"LISWETS": [2]}, timeout=60,
                          checker_sha256=checker.checker_digest())
            def forbidden(*args, **kwargs):
                raise AssertionError("fallback proposal consulted instead of gaps.json")
            with mock.patch.object(truth, "solve_highs", side_effect=forbidden):
                revalidation.write_json(root / "run.json", config)
                revalidation.worker(str(root / "run.json"), "LISWETS")
            result = json.loads((out / "LISWETS.result.json").read_text())
            self.assertTrue(result["accepted"], msg=result.get("reason"))
            self.assertEqual(result["proposal"]["gaps"], [2])  # gaps.json が伝播


class ModelConsistencyTriggerTests(unittest.TestCase):
    """HiGHS 提案経路 (reported 有り) で MODEL_CONSISTENCY_TOL が発動する。

    exact 目的値と HiGHS 報告値が相対 1e-6 より離れていれば CertificateError、
    一致していれば受理 (対照)。照合経路ごと削除した変異では発動側が fail。
    """

    X_HONEST = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0])  # hand-computed optimum

    def run_liswet(self, reported, data):
        def proposed(prob, time_limit=None):
            return ("Optimal", self.X_HONEST, np.zeros(prob["m"]), reported)
        with mock.patch.object(truth, "solve_highs", side_effect=proposed):
            return liswet.certify(data / "LISWETS.QPS")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        RevalidateLiswetRoutingTests.write_liswet(self.root / "LISWETS.QPS")

    def tearDown(self):
        self._tmp.cleanup()

    def test_honest_report_passes_consistency(self):
        cert = self.run_liswet(-8.0, self.root)
        self.assertEqual(cert.gaps, (2,))
        self.assertAlmostEqual(float(cert.objective), -8.0)

    def test_shifted_report_triggers_model_consistency(self):
        # |-8 - (-8.00002)|/(1+8.00002) = 2.22e-6 > MODEL_CONSISTENCY_TOL (1e-6)。
        self.assertEqual(liswet.MODEL_CONSISTENCY_TOL, 1e-6)
        with self.assertRaisesRegex(liswet.CertificateError, "differs from HiGHS report"):
            self.run_liswet(-8.00002, self.root)


class SourceNameCharsetTests(unittest.TestCase):
    """source 列の proof パスに埋め込まれる問題名は separator 文字を含まない。

    現状実装の実読: Path(name).name != name は区切り (/, ..) だけを見ており
    ';', '=' は素通りだったので、混入しうる文字は入口で reject に変更した。
    """

    def run_main_with_name(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "in.csv"
            csv_path.write_text("# comment\nproblem_name,optimal_obj,source\n"
                                f"{name},1.0,old_cert\n")
            argv = ["--csv", str(csv_path), "--data-dir", str(root / "d"),
                    "--candidate-dir", str(root / "c"), "--output-dir", str(root / "o"),
                    "--jobs", "1"]
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit):
                    revalidation.main(argv)
            return err.getvalue()

    def test_separator_names_are_rejected(self):
        # ';' '=' は source 列の key separator、空白は path 起因の崩れ。CSV
        # 第 1 列自体の ',' は csv フレーミングで守られるので対象外。
        for name in ("A;B", "A=B", "A B"):
            with self.subTest(name=name):
                self.assertIn("invalid problem name", self.run_main_with_name(name))

    def test_committed_baseline_names_are_all_in_the_safe_charset(self):
        # 正方向 pin: 実在 committed 名はすべて許可される (過剰拒否の検出)。
        found = 0
        for path in sorted((ROOT / "data" / "baseline_objectives").glob("*.csv")):
            rows = list(csv.DictReader(line for line in path.read_text().splitlines()
                                       if not line.startswith("#")))
            for row in rows:
                if row.get("problem_name"):
                    found += 1
                    self.assertTrue(revalidation.SAFE_NAME.fullmatch(row["problem_name"]),
                                    msg=f"{path.name}:{row['problem_name']}")
        self.assertGreater(found, 0)


if __name__ == "__main__":
    unittest.main()
