"""契約ツール破棄経路と契約定数の sentinel (レビュー B P2)。

既存の tests/test_certificate_contract.py は受理経路と code freeze を中心に検証するが、
(1) 契約定数そのもの、(2) enclosure 欠落候補の拒否、(3) previous 値乖離検出、
(4) checker_digest の対象集合、(5) 偽造 proof・保存時ガード、(6) LISWET 行の
厳密経路ルーティング は pin していなかった。各テストは対応する実装を revert
または緩和すると fail する (実測済みの sentinel)。
"""
import gzip
import hashlib
import importlib.util
import json
import math
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


class CheckerDigestScopeTests(unittest.TestCase):
    """checker_digest は指定 4 ファイル全体の hash であり、1 つでも外れると変わる。

    対象ファイルが集合から除外される (または追加ファイルが増える) 退行を、
    「そのファイルの 1 バイト改竄で digest が変わる」ことで検出する。
    """

    FILES = ("baseline_certificate.py", "highs_baseline_truth.py",
             "liswet_exact_certificate.py", "revalidate_baselines.py")

    @staticmethod
    def digest_from_copy(tamper):
        # コピー先ディレクトリで baseline_certificate を再 import すると、その
        # 時点の 4 ファイル bytes から LOADED_CHECKER_DIGEST が計算される。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in CheckerDigestScopeTests.FILES:
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
        for name in self.FILES:
            with self.subTest(file=name):
                self.assertNotEqual(self.digest_from_copy(name), checker.checker_digest())

    def test_loaded_digest_matches_fresh_digest(self):
        # worker の code freeze は LOADED と実行時 digest の一致に依存する。
        self.assertEqual(revalidation.LOADED_CHECKER_DIGEST, checker.checker_digest())


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

    @staticmethod
    def enclosure():
        return checker.ExactModel(scalar_problem()).check([Fraction(3)], [Fraction(3)], [])

    def test_reference_outside_budget_raises_and_writes_nothing(self):
        path = Path(tempfile.mkdtemp()) / "SCALAR.proof.json.gz"
        with self.assertRaisesRegex(ValueError, "outside its certified error budget"):
            checker.save_evidence(path, self.enclosure(), "-4.400000000000e0",
                                  hashlib.sha256(b"x").hexdigest())
        self.assertFalse(path.exists())

    def test_stale_expected_digest_raises_and_writes_nothing(self):
        path = Path(tempfile.mkdtemp()) / "SCALAR.proof.json.gz"
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
            payload = json.loads(gzip.decompress(
                (out / "LISWETS.proof.json.gz").read_bytes()).decode())
            # 保存 proof は共有 enclosure で replay できる (厳密経路の証跡)。
            checker.replay_evidence(truth.load_qps(str(data / "LISWETS.QPS")), payload,
                                    result["input_sha256"])


class OutputSourceProvenanceTests(unittest.TestCase):
    """受理行の source 列は契約が要求する traceability (proof 参照・入力 hash・
    checker version) を載せる。reviewer 指摘「committed CSV は契約ツールの成果物で
    ない」の解消として、revalidate の出力そのものを committed データにする。"""

    def test_accepted_source_carries_proof_and_hashes(self):
        result = dict(accepted=True, problem_name="A", reference="1.0",
                      proof="A.proof.json.gz",
                      input_sha256="a" * 64, checker_sha256="b" * 64)
        self.assertEqual(revalidation.output_source(result),
                         "exact_enclosure_v1;proof=A.proof.json.gz;"
                         "input_sha256=" + "a" * 64 + ";checker_sha256=" + "b" * 64)

    def test_failed_source_keeps_uncertified_suffix(self):
        result = dict(accepted=False, reason="missing_candidate")
        self.assertEqual(revalidation.output_source(result),
                         "revalidation_missing_candidate_UNCERTIFIED")


if __name__ == "__main__":
    unittest.main()
