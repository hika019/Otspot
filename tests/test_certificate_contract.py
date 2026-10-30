"""Independent counterexamples for the baseline certificate contract."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("truth", ROOT / "scripts/highs_baseline_truth.py")
truth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(truth)
sys.modules.setdefault("highs_baseline_truth", truth)
# revalidate は evaluate_row 内で `import highs_baseline_truth` するため、mock や
# 属性差替がそこから見えるのは sys.modules のインスタンスだけ。spec 読込を複数
# テストファイルで combined pytest すると setdefault 先行ファイルのインスタンスが
# 残り、後続ファイルの patch が无声で無効化する (test_worker_fails_when_checker_
# source_changes_during_verification が combined で fail する)。共有インスタンスへ
# 再バインドし、ファイル間で同一 object を指すようにする。
truth = sys.modules["highs_baseline_truth"]
import baseline_certificate as checker
import revalidate_baselines as revalidation
import liswet_exact_certificate as liswet


def scalar_problem():
    return dict(c0=0.0, c=np.array([-3.0]), A=sp.csr_matrix((0, 1)),
                L=np.zeros(0), U=np.zeros(0), lo=np.zeros(1),
                up=np.array([math.inf]), Q=sp.eye(1, format="csr"), n=1, m=0)


class CertificateContractTests(unittest.TestCase):
    def test_small_negative_curvature_cannot_certify_stationary_origin(self):
        a = 1.0 + 1e-12
        prob = dict(c0=0.0, c=np.zeros(2), A=sp.csr_matrix((0, 2)),
                    L=np.zeros(0), U=np.zeros(0), lo=np.full(2, -1e8),
                    up=np.full(2, 1e8), Q=sp.csr_matrix([[1.0, a], [a, 1.0]]),
                    n=2, m=0, psd=True)
        self.assertLess((Fraction(1) - Fraction(a)) * 10**16, -10000)
        ok, _, _ = truth.certify(prob, ("Optimal", np.zeros(2), np.zeros(0), 0.0))
        self.assertFalse(ok)

    def test_feasible_nonoptimal_candidate_has_certified_interval(self):
        prob = scalar_problem()
        x = np.array([3.0 + 2.0**-12])
        optimum = Fraction(-9, 2)
        with mock.patch.object(truth, "polish", return_value=None), \
             mock.patch.object(truth, "_exact_certificate", return_value=None), \
             mock.patch.object(truth, "_sparse_exact_certificate", return_value=None):
            reported = float(x[0]**2 / 2 - 3 * x[0])
            ok, value, _ = truth.certify(prob, ("Optimal", x, np.zeros(0), reported))
        self.assertTrue(ok)
        self.assertGreater(Fraction(value), optimum)
        self.assertLess((Fraction(value) - optimum) / (1 + abs(Fraction(value))),
                        Fraction(1, 10**7))

    def test_dual_checker_recomputes_reduced_cost(self):
        data = truth._ExactData(scalar_problem())
        self.assertIsNone(truth._exact_d(data, [Fraction(0)], [], [Fraction(0)]))

    def test_invalid_model_bounds_and_dimensions_are_rejected(self):
        changes = [dict(lo=np.array([math.inf])), dict(lo=np.array([math.nan])),
                   dict(up=np.array([-math.inf])), dict(lo=np.zeros(0)),
                   dict(c=np.zeros(0)), dict(lo=np.array([2.0]), up=np.array([1.0])),
                   dict(A=sp.csr_matrix((1, 1))), dict(c=np.array([math.nan]))]
        for change in changes:
            with self.subTest(change=change):
                prob = scalar_problem()
                prob.update(change)
                with self.assertRaises(ValueError):
                    checker.ExactModel(prob)

    def test_model_snapshot_is_immutable_and_not_changed_by_source(self):
        prob = scalar_problem()
        model = checker.ExactModel(prob)
        fingerprint = model.fingerprint()
        prob["c"][0] = 100
        self.assertEqual(model.c, (Fraction(-3),))
        self.assertEqual(model.fingerprint(), fingerprint)
        with self.assertRaises(AttributeError):
            model.c = (Fraction(100),)

    def test_exact_primal_and_separate_tangent_enclose_hand_computed_optimum(self):
        model = checker.ExactModel(scalar_problem())
        point = Fraction(3) + Fraction(1, 4096)
        enclosure = model.check([point], [3], [])
        self.assertEqual(enclosure.lower, Fraction(-9, 2))
        self.assertEqual(enclosure.upper, Fraction(-9, 2) + Fraction(1, 2 * 4096**2))
        self.assertIsNone(model.check([-1], [3], []))

    def test_asymmetric_hessian_is_rejected(self):
        q = sp.csr_matrix([[1.0, 2.0], [0.0, 1.0]])
        self.assertFalse(truth._psd_ok(q))

    def test_exact_psd_budget_exhaustion_is_unknown_not_proof(self):
        q = checker.columns(sp.csr_matrix([[1.0, 2.0], [2.0, 5.0]]))
        checker.prove_psd.cache_clear()
        with mock.patch.object(checker, "PSD_OPERATIONS", 0):
            self.assertIsNone(checker.prove_psd(q))
        checker.prove_psd.cache_clear()
        self.assertTrue(checker.prove_psd(q))

    def test_serialized_binary64_value_can_exceed_decimal_error_budget(self):
        r, tol = Fraction("0.1"), checker.REFERENCE_TOLERANCE
        interval = checker.Enclosure(r - tol * (1 + r), r, (), (), (), "synthetic")
        self.assertEqual((r - interval.lower) / (1 + r), tol)
        self.assertGreater(checker.reference_error(interval, "0.1"), tol)
        exact = checker.Enclosure(Fraction(0), tol, (), (), (), "synthetic")
        self.assertEqual(checker.reference_error(exact, "0"), tol)

    def test_verified_candidate_survives_better_unverified_score(self):
        enclosure = checker.Enclosure(Fraction(-1), Fraction(0), (0,), (0,), (), "test")
        verified = truth.Certificate(0., -1., 0., 0., 0., 0, "verified", True, enclosure)
        guessed = truth.Certificate(0., 0., 0., 0., 0., 0, "heuristic", False)
        self.assertLess(truth.certificate_rank(verified), truth.certificate_rank(guessed))
        with mock.patch.object(truth, "polish", return_value=None), \
             mock.patch.object(truth, "verify", return_value=verified), \
             mock.patch.object(truth, "_exact_certificate", return_value=guessed):
            self.assertIs(truth.best_certificate(scalar_problem(), np.zeros(1), np.zeros(0)), verified)

    def test_failed_revalidation_drops_old_certification_label(self):
        for reason in ("psd_unknown", "nonconvex", "missing_candidate", "verification_timeout"):
            result = dict(accepted=False, reason=reason, old_source="exact_rational_kkt")
            self.assertTrue(revalidation.output_source(result).endswith("_UNCERTIFIED"))
            self.assertNotIn("exact_rational_kkt", revalidation.output_source(result))


class RevalidationCodeFreezeTests(unittest.TestCase):
    """Each revalidation row must be bound to the checker code pinned at run start."""

    @staticmethod
    def scalar_case(root):
        # min 1/2 x^2 - 3x over free x: hand optimum -9/2 at rational x = 3.
        data, candidates, out = root / "data", root / "candidates", root / "out"
        data.mkdir()
        candidates.mkdir()
        out.mkdir()
        qps = data / "SCALAR.QPS"
        qps.write_text("NAME SCALAR\nROWS\n N OBJ\nCOLUMNS\n    X1  OBJ -3\nRHS\n"
                       "BOUNDS\n FR B X1\nQUADOBJ\n    X1  X1 1\nENDATA\n")
        np.savez(candidates / "SCALAR.npz", x=np.array([3.0]), y=np.zeros(0))
        (candidates / "SCALAR.json").write_text(json.dumps(
            {"input_sha256": hashlib.sha256(qps.read_bytes()).hexdigest()}))
        return dict(rows=[dict(problem_name="SCALAR", optimal_obj="-4.5",
                               source="previous_cert")],
                    data_dir=str(data), candidate_dir=str(candidates), output_dir=str(out),
                    gaps={}, timeout=60, checker_sha256=checker.checker_digest())

    @staticmethod
    def run_worker(config, root):
        config_path = root / "run.json"
        revalidation.write_json(config_path, config)
        revalidation.worker(str(config_path), "SCALAR")
        return json.loads((root / "out" / "SCALAR.result.json").read_text())

    def test_worker_certifies_under_pinned_digest_with_replayable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.scalar_case(root)
            result = self.run_worker(config, root)
            self.assertEqual(result["reason"], "certified")
            self.assertTrue(result["accepted"])
            self.assertEqual(result["reference"], "-4.5")
            self.assertEqual(result["checker_sha256"], config["checker_sha256"])
            payload = json.loads(gzip.decompress(
                (root / "out" / result["proof"]).read_bytes()).decode())
            self.assertEqual(payload["checker_sha256"], config["checker_sha256"])
            self.assertEqual(payload["lower"], payload["upper"])
            prob = truth.load_qps(str(root / "data" / "SCALAR.QPS"))
            checker.replay_evidence(prob, payload, result["input_sha256"])

    def test_worker_rejects_digest_other_than_the_pinned_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.scalar_case(root)
            config["checker_sha256"] = "0" * 64
            stale = root / "out" / revalidation.proof_rel("SCALAR")
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"stale evidence from an earlier run")
            result = self.run_worker(config, root)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "checker_code_changed")
            self.assertFalse(stale.exists())

    def test_failed_worker_run_clears_stale_proof_of_earlier_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.scalar_case(root)
            config = dict(rows=[], data_dir=str(root / "data"),
                          candidate_dir=str(root / "candidates"), output_dir=str(root / "out"),
                          gaps={}, timeout=60, checker_sha256=checker.checker_digest())
            config_path = root / "run.json"
            revalidation.write_json(config_path, config)
            row = dict(problem_name="MISSING", optimal_obj="1.0", source="previous_cert")
            stale = root / "out" / revalidation.proof_rel("MISSING")
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"stale evidence")
            result = revalidation.run_row(config_path, row, root / "out", 60)
            self.assertEqual(result["reason"], "worker_failed")
            self.assertFalse(stale.exists())

    def test_worker_fails_when_checker_source_changes_during_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.scalar_case(root)
            target = ROOT / "scripts" / "highs_baseline_truth.py"
            original = target.read_bytes()
            real = truth.best_certificate

            def tampered(*args, **kwargs):
                cert = real(*args, **kwargs)
                target.write_bytes(original + b"\n# rewritten while the worker ran\n")
                return cert

            truth.best_certificate = tampered
            try:
                result = self.run_worker(config, root)
            finally:
                target.write_bytes(original)
                truth.best_certificate = real
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "checker_code_changed")
            self.assertFalse((root / "out" / revalidation.proof_rel("SCALAR")).exists())
            self.assertTrue(revalidation.output_source(result).endswith("_UNCERTIFIED"))


class CertificationRunCodeFreezeTests(unittest.TestCase):
    """A checker digest drift across the run voids every certification in it."""

    @staticmethod
    def rows(output, proof_bytes=b"gzip-placeholder"):
        # 受理行は CLAIM_KEYS すべてを備える (欠落は output_source が fail-closed)。
        proof = revalidation.proof_rel("A")
        (output / proof).parent.mkdir(parents=True, exist_ok=True)
        (output / proof).write_bytes(proof_bytes)
        accepted = dict(problem_name="A", old_value="1.0", old_source="exact_enclosure_v1",
                        reference="1.0", accepted=True, reason="certified",
                        proof=proof, proof_sha256="c" * 64, input_sha256="a" * 64,
                        checker_sha256="b" * 64)
        failed = dict(problem_name="B", old_value="2.0", old_source="old_cert",
                      reference="2.0", accepted=False, reason="missing_candidate")
        return [accepted, failed]

    def test_digest_drift_at_finalize_voids_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = revalidation.finalize(self.rows(output), output, "0" * 64)
            self.assertEqual(code, 1)
            csv_text = (output / "baseline.csv").read_text()
            self.assertNotIn("exact_enclosure_v1", csv_text)
            self.assertEqual(csv_text.count("checker_code_changed_UNCERTIFIED"), 2)
            self.assertFalse((output / revalidation.proof_rel("A")).exists())
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["code_frozen"])
            self.assertEqual(summary["certified"], 0)
            self.assertEqual(summary["checker_sha256"], "0" * 64)

    def test_frozen_run_retains_certified_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = revalidation.finalize(self.rows(output), output, checker.checker_digest())
            self.assertEqual(code, 0)
            csv_text = (output / "baseline.csv").read_text()
            self.assertIn("A,1.0,exact_enclosure_v1", csv_text)
            self.assertIn("missing_candidate_UNCERTIFIED", csv_text)
            self.assertTrue((output / revalidation.proof_rel("A")).exists())
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["code_frozen"])
            self.assertEqual(summary["certified"], 1)


class DirectCsvProvenanceTests(unittest.TestCase):
    """A direct --csv claim persists contract provenance via the shared checker."""

    # Independent oracle: f(x) = 7 + ||x||^2 / 2 under A x >= 0 for the second
    # difference row; x = 0 attains 7 and no feasible point can be lower.
    QPS = ("NAME SYNTH3\nROWS\n N OBJ\n G R1\nCOLUMNS\n"
           "    X1  OBJ 0  R1 1\n    X2  OBJ 0  R1 -2\n    X3  OBJ 0  R1 1\n"
           "RHS\n    RHS OBJ -7\nBOUNDS\n FR B X1\n FR B X2\n FR B X3\n"
           "QUADOBJ\n    X1 X1 1\n    X2 X2 1\n    X3 X3 1\nENDATA\n")
    OPTIMUM = "7.000000000000e0"

    def assert_contract_evidence(self, payload, qps):
        self.assertEqual(payload["reference"], self.OPTIMUM)
        self.assertEqual(payload["lower"], "7")
        self.assertEqual(payload["upper"], "7")
        self.assertEqual(payload["checker"], checker.CHECKER_VERSION)
        self.assertEqual(payload["input_sha256"], hashlib.sha256(qps.read_bytes()).hexdigest())
        self.assertLessEqual(Fraction(payload["relative_error"]), checker.REFERENCE_TOLERANCE)
        checker.replay_evidence(truth.load_qps(str(qps)), payload, payload["input_sha256"])

    def test_liswet_csv_path_saves_replayable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qps = root / "SYNTH3.QPS"
            qps.write_text(self.QPS)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = liswet.main(["--csv", "--evidence-dir", str(root / "ev"), str(qps)])
            self.assertEqual(code, 0)
            self.assertIn(f"SYNTH3,{self.OPTIMUM},exact_rational_kkt", out.getvalue())
            payload = json.loads(gzip.decompress(
                (root / "ev" / "SYNTH3.proof.json.gz").read_bytes()).decode())
            self.assertEqual(payload["proposal"]["method"], "liswet_banded")
            self.assert_contract_evidence(payload, qps)

    def test_truth_csv_path_saves_replayable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qps = root / "SYNTH3.QPS"
            qps.write_text(self.QPS)
            run = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "highs_baseline_truth.py"),
                 "--csv", "--evidence-dir", str(root / "ev"), str(qps)],
                capture_output=True, text=True, check=True, timeout=600)
            self.assertIn(f"SYNTH3,{self.OPTIMUM},highs", run.stdout)
            payload = json.loads(gzip.decompress(
                (root / "ev" / "SYNTH3.proof.json.gz").read_bytes()).decode())
            self.assertEqual(payload["proposal"]["solver"], "highs")
            self.assert_contract_evidence(payload, qps)


class TruthModuleIdentityTests(unittest.TestCase):
    """spec 読み込みした truth は revalidate が import する sys.modules インスタンスと
    同一 object であること。combined pytest でファイルごとに別 instance が残ると
    属性差替が无声で無効化される (import 再バインド revert + 2 ファイル combined
    実行で本テストが fail する sentinel)。
    """

    def test_truth_module_is_shared_instance(self):
        import highs_baseline_truth as seen
        self.assertIs(seen, truth)
        self.assertIs(sys.modules["highs_baseline_truth"], truth)


if __name__ == "__main__":
    unittest.main()
