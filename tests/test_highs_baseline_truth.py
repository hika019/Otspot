"""scripts/highs_baseline_truth.py の検算 (弱双対下界) とパーサの sentinel。"""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("truth", ROOT / "scripts/highs_baseline_truth.py")
truth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(truth)

# 手計算の最適値: min 1/2 x^2 - 3x (x >= 0) は x* = 3 で -9/2。
HAND_OPTIMUM = -4.5
ABS_TOL = 1e-12


def scalar_problem():
    """min 1/2 x^2 - 3x, x >= 0 (制約行なし)。"""
    return dict(c0=0.0, c=np.array([-3.0]), A=sp.csr_matrix((0, 1)), L=np.zeros(0),
                U=np.zeros(0), lo=np.array([0.0]), up=np.array([math.inf]),
                Q=sp.csr_matrix(np.array([[1.0]])), n=1, m=0)


class VerifyLowerBoundTests(unittest.TestCase):
    def test_dropped_dual_residual_term_keeps_lower_bound_valid(self):
        # x = 1 は最適でない: z = x - 3 = -2 は上限無限の変数に負の乗数を要求するので捨てる
        # (r = -2)。Q delta = -r で delta = 2。下界 = -1/2 - 2 (r x) - 2 (1/2 r delta)
        # = -9/2 で、x = 3 の厳密な最適値に一致する。r'x を落とすと -5/2 (= p(x)) になり
        # 最適値 -9/2 を超える (下界ではない)。
        p, d, _, resid, _ = truth.verify(scalar_problem(), np.array([1.0]), np.zeros(0))
        self.assertAlmostEqual(p, -2.5, delta=ABS_TOL)
        self.assertLessEqual(d, p + ABS_TOL)
        self.assertAlmostEqual(d, HAND_OPTIMUM, delta=ABS_TOL)
        self.assertEqual(resid, 0.0)

    def test_optimal_point_has_zero_gap(self):
        p, d, *_ = truth.verify(scalar_problem(), np.array([3.0]), np.zeros(0))
        self.assertAlmostEqual(p, HAND_OPTIMUM, delta=ABS_TOL)
        self.assertAlmostEqual(d, HAND_OPTIMUM, delta=ABS_TOL)


FIXED_QPS = (
    "NAME          T\n"
    "ROWS\n"
    " N  OBJ\n"
    "COLUMNS\n"
    "    X         OBJ       1.0\n"
    "RHS\n"
    "BOUNDS\n"
    " {kind} BND       X         1.0\n"
    "ENDATA\n"
)


class FixedBoundTypeTests(unittest.TestCase):
    def parse(self, kind):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.qps"
            path.write_text(FIXED_QPS.format(kind=kind))
            return truth.parse_qps(str(path), fixed=True)

    def test_unknown_bound_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("XX")

    def test_known_bound_type_is_accepted(self):
        self.assertEqual(self.parse("UP")["up"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
