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


SEED = 20260821
N_PROBLEMS = 100
POINTS_PER_PROBLEM = 5
# 各分岐を踏んだ試行数の下限。生成器が分岐を踏まなくなって検査が空洞化するのを防ぐ。
MIN_TRIALS_PER_BRANCH = 20
# oracle の HiGHS 時間上限 [s]。小規模 QP は 数 ms で解ける。超える問題 (非有界など) は oracle 不成立として捨てる。
ORACLE_TIME_LIMIT_SECS = 1.0
# 最適解近傍の摂動幅 10^U(lo, hi)。上限 1e-3 は HiGHS 解の主双対誤差より十分大きく、
# 下限 1e-9 は認証を通りうる最小の摂動 (UNCERTAINTY_TOL = 1e-7 の 1 桁以上下)。
PERTURB_LOG10_RANGE = (-9.0, -3.0)
# 最適解近傍の摂動点のうち、認証を通る試行数の下限 (step=None 経路を含む)。
MIN_ACCEPTED_STEP_NONE = 3
# HiGHS の解を「最適」とみなす条件: 検算付き gap の相対上限と主実行不能度の上限。
ORACLE_GAP_TOL = 1e-7
ORACLE_PINF_TOL = 1e-9
# 認証を通った候補の主目的値と真の最適値の相対差の上限。認証許容と oracle 自身の
# gap 許容の和 (どちらも片側の不確かさ上界)。
ACCEPTED_ERR_TOL = truth.UNCERTAINTY_TOL + ORACLE_GAP_TOL
# d <= 真の最適値 を許す数値余裕 (相対)。HiGHS の最適値自身の収束誤差 (gap 上限) に揃える。
BOUND_SLACK = 10 * ORACLE_GAP_TOL
BOUND_DOMAIN = 9.0
# 両側有限変数の Q 列を 0 にする確率。特異 Q かつ r in range(Q) の経路を十分な頻度で踏む値。
ZERO_COLUMN_PROB = 0.7


def random_problem(rng):
    """凸 QP (Q = B'B, B は k 行 <= n なので特異もある)。x0 が実行可能になるよう行範囲を作る。"""
    n, m = int(rng.integers(2, 6)), int(rng.integers(0, 4))
    B = rng.normal(size=(int(rng.integers(0, n + 1)), n)).round(1)
    A = rng.normal(size=(m, n)).round(1)
    lo = np.where(rng.random(n) < .5, -rng.random(n).round(1), -math.inf)
    up = np.where(rng.random(n) < .5, 1 + rng.random(n).round(1), math.inf)
    fixed = (rng.random(n) < .15) & np.isfinite(lo)
    up = np.where(fixed, lo, up)
    # 両側有限境界の変数の Q 列を確率的に 0 にする: Q は特異になり、捨てられる残差 r
    # (片側無限の変数にだけ立つ) は range(Q) に収まる。
    two_sided = np.isfinite(lo) & np.isfinite(up)
    B[:, two_sided & (rng.random(n) < ZERO_COLUMN_PROB)] = 0.0
    x0 = np.clip(rng.normal(size=n), np.where(np.isfinite(lo), lo, -BOUND_DOMAIN),
                 np.where(np.isfinite(up), up, BOUND_DOMAIN))
    ax0 = A @ x0
    L = np.where(rng.random(m) < .5, ax0 - rng.random(m), -math.inf)
    U = np.where(rng.random(m) < .5, ax0 + rng.random(m), math.inf)
    eq = rng.random(m) < .2
    L, U = np.where(eq, ax0, L), np.where(eq, ax0, U)
    prob = dict(c0=float(rng.normal()), c=rng.normal(size=n).round(1), A=sp.csr_matrix(A),
                L=L, U=U, lo=lo, up=up, Q=sp.csr_matrix(B.T @ B), n=n, m=m)
    return prob, x0


def oracle_optimum(prob):
    """HiGHS の QP 解。検算 (gap, 主実行不能度) を通らなければ None。"""
    status, x, y, obj = truth.solve_highs(prob, ORACLE_TIME_LIMIT_SECS)
    if status != "Optimal":
        return None
    p, d, pinf, resid, _ = truth.verify(prob, x, y)
    if resid != 0.0 or pinf > ORACLE_PINF_TOL or abs(p - d) > ORACLE_GAP_TOL * (1 + abs(p)):
        return None
    return obj, x, y


class RandomLowerBoundTests(unittest.TestCase):
    def test_dual_bound_never_exceeds_true_optimum(self):
        rng = np.random.default_rng(SEED)
        counts = dict(step_none=0, step_singular=0, step_nonsingular=0, ranges=0, fixed_var=0,
                      row_multiplier=0, mixed_bounds=0)
        for _ in range(N_PROBLEMS):
            prob, x0 = random_problem(rng)
            oracle = oracle_optimum(prob)
            if oracle is None:
                continue
            opt, x_opt, _ = oracle
            n = prob["n"]
            singular = np.linalg.matrix_rank(prob["Q"].toarray()) < n
            ranges = bool(np.any(np.isfinite(prob["L"]) & np.isfinite(prob["U"])
                                 & (prob["L"] < prob["U"])))
            fixed_var = bool(np.any(prob["lo"] == prob["up"]))
            finite = np.isfinite(prob["lo"]) | np.isfinite(prob["up"])
            mixed = bool(finite.any() and not finite.all())
            for _ in range(POINTS_PER_PROBLEM):
                xb = x0 + rng.normal(size=n) * rng.choice([0, .3, 2])
                xb = np.where(rng.random(n) < .3, x_opt, xb)
                yb = rng.normal(size=prob["m"]) * rng.choice([0, .5, 3])
                _, d, _, resid, _ = truth.verify(prob, xb, yb)
                if resid > 0:
                    # step=None: d は下界でなく見積りで、resid が不確かさに入る (下の認証テスト)。
                    counts["step_none"] += 1
                    continue
                self.assertLessEqual(d, opt + BOUND_SLACK * (1 + abs(opt)))
                if truth_dropped(prob, xb, yb):
                    counts["step_singular" if singular else "step_nonsingular"] += 1
                counts["ranges"] += ranges
                counts["fixed_var"] += fixed_var
                counts["row_multiplier"] += bool(np.any(yb != 0))
                counts["mixed_bounds"] += mixed
        for branch, count in counts.items():
            self.assertGreaterEqual(count, MIN_TRIALS_PER_BRANCH, f"{branch}: {counts}")

    def test_accepted_certificate_matches_true_optimum(self):
        """最適解近傍の (x, y) で認証 (pinf, uncertainty) を通った p は真の最適値に一致する。
        step=None 経路 (resid > 0) もここで検査する。"""
        rng = np.random.default_rng(SEED + 1)
        accepted = accepted_step_none = 0
        for _ in range(N_PROBLEMS):
            prob, _ = random_problem(rng)
            oracle = oracle_optimum(prob)
            if oracle is None:
                continue
            opt, x_opt, y_opt = oracle
            for _ in range(POINTS_PER_PROBLEM):
                scale = 10.0 ** rng.uniform(*PERTURB_LOG10_RANGE)
                xb = x_opt + scale * rng.normal(size=prob["n"]) * (rng.random(prob["n"]) < .5)
                yb = y_opt + scale * rng.normal(size=prob["m"]) * (rng.random() < .5)
                result = truth.verify(prob, xb, yb)
                if result[2] > truth.PRIMAL_INF_TOL or truth.uncertainty(result) > truth.UNCERTAINTY_TOL:
                    continue
                accepted += 1
                accepted_step_none += result[3] > 0
                self.assertLessEqual(abs(result[0] - opt), ACCEPTED_ERR_TOL * (1 + abs(opt)))
        self.assertGreaterEqual(accepted_step_none, MIN_ACCEPTED_STEP_NONE, f"{accepted}")


def truth_dropped(prob, x, y):
    """verify が z の無限側要求成分を捨てたか (r != 0 で step 経路を通ったか)。"""
    yp = y.copy()
    yp[(yp > 0) & ~np.isfinite(prob["L"])] = 0.0
    yp[(yp < 0) & ~np.isfinite(prob["U"])] = 0.0
    z = prob["c"] + prob["Q"] @ x - prob["A"].T @ yp
    return bool(truth._bound_term(z, prob["lo"], prob["up"])[1].any())


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
