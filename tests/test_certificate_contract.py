"""Independent counterexamples for the baseline certificate contract."""
import importlib.util
import math
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
import baseline_certificate as checker
import revalidate_baselines as revalidation


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


if __name__ == "__main__":
    unittest.main()
