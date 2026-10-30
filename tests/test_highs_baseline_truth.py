"""scripts/highs_baseline_truth.py の検算 (弱双対下界) とパーサの sentinel。"""
import csv
import importlib.util
import json
import itertools
import math
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("truth", ROOT / "scripts/highs_baseline_truth.py")
truth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(truth)

# 手計算の最適値: min 1/2 x^2 - 3x (x >= 0) は x* = 3 で -9/2。
HAND_OPTIMUM = Fraction(-9, 2)
# 下界がどれだけ緩いかの上限 (絶対)。dual_bound は丸め誤差上界ぶん -inf 側へ倒すので、
# 最適な線形化点でも d は最適値より O(u) だけ小さい。
HAND_SLACK = 1e-12


def scalar_problem():
    """min 1/2 x^2 - 3x, x >= 0 (制約行なし)。"""
    return dict(c0=0.0, c=np.array([-3.0]), A=sp.csr_matrix((0, 1)), L=np.zeros(0),
                U=np.zeros(0), lo=np.array([0.0]), up=np.array([math.inf]),
                Q=sp.csr_matrix(np.array([[1.0]])), n=1, m=0)


class VerifyLowerBoundTests(unittest.TestCase):
    def test_unbounded_side_residual_is_repaired_to_exact_bound(self):
        # x = 1 は最適でない: g = x - 3 = -2 は上限無限の変数に負の係数を要求するので、
        # 線形化点のままでは下界 -inf。_repair が xh を Q xh + c = 0 (xh = 3) へ動かすと
        # 下界は -1/2 * 9 = -9/2 (= 最適値) になる。
        cert = truth.verify(scalar_problem(), np.array([1.0]), np.zeros(0))
        self.assertEqual(cert.p, -2.5)
        self.assertGreater(cert.repair_iters, 0)
        self.assertLessEqual(Fraction(cert.d), HAND_OPTIMUM)
        self.assertLessEqual(HAND_OPTIMUM - Fraction(cert.d), HAND_SLACK)

    def test_unrepaired_unbounded_side_gives_minus_infinity(self):
        d = truth.dual_bound(scalar_problem(), np.array([1.0]), np.zeros(0))
        self.assertEqual(d, -math.inf)

    def test_optimal_point_has_zero_gap(self):
        cert = truth.verify(scalar_problem(), np.array([3.0]), np.zeros(0))
        self.assertEqual(cert.p, -4.5)
        self.assertLessEqual(Fraction(cert.d), HAND_OPTIMUM)
        self.assertLessEqual(HAND_OPTIMUM - Fraction(cert.d), HAND_SLACK)

    def test_free_lp_variable_cost_is_repaired_through_row_multiplier(self):
        # min x1 s.t. x1 - x2 = 0, x1 >= 0, x2 free (最適値 0)。y = 1 - 2^-40 では自由変数 x2 の
        # 縮約費用 g_2 = y != 0 で下界 -inf。x2 は Q 列 0 なので xh では直せず、_repair は
        # 等式行の乗数を y = 0 へ動かす。そこで g = (1, 0) となり d = 0。
        prob = dict(c0=0.0, c=np.array([1.0, 0.0]), A=sp.csr_matrix(np.array([[1.0, -1.0]])),
                    L=np.zeros(1), U=np.zeros(1), lo=np.array([0.0, -math.inf]),
                    up=np.array([math.inf, math.inf]), Q=sp.csr_matrix((2, 2)), n=2, m=1)
        y = np.array([1.0 - 2.0 ** -40])
        self.assertEqual(truth.dual_bound(prob, np.zeros(2), y), -math.inf)
        cert = truth.verify(prob, np.zeros(2), y)
        self.assertGreater(cert.repair_iters, 0)
        self.assertLessEqual(cert.d, 0.0)
        self.assertGreaterEqual(cert.d, -HAND_SLACK)


    def test_bound_is_rounded_toward_minus_infinity(self):
        # min 0.1 x, x >= 3: 最適値は fl(0.1) * 3 (有理数として厳密)。浮動小数の積 fl(0.1 * 3) =
        # 0.30000000000000004 は真値 0.3000000000000000166... より大きいので、丸め誤差を
        # -inf 側へ倒さない評価は下界にならない。
        prob = dict(c0=0.0, c=np.array([0.1]), A=sp.csr_matrix((0, 1)), L=np.zeros(0),
                    U=np.zeros(0), lo=np.array([3.0]), up=np.array([math.inf]),
                    Q=sp.csr_matrix((1, 1)), n=1, m=0)
        opt = Fraction(0.1) * 3
        self.assertGreater(Fraction(0.1 * 3.0), opt)
        cert = truth.verify(prob, np.array([3.0]), np.zeros(0))
        self.assertLessEqual(Fraction(cert.d), opt)
        self.assertLessEqual(opt - Fraction(cert.d), HAND_SLACK)


    def test_free_variable_with_exactly_zero_cost_contributes_zero(self):
        # min 0 * x (x 自由) の最適値は 0。y も xh も 0 なら g = 0 は厳密 (丸めなし) で、
        # 有限な境界を持たない成分の項は 0。
        prob = dict(c0=0.0, c=np.zeros(1), A=sp.csr_matrix((0, 1)), L=np.zeros(0),
                    U=np.zeros(0), lo=np.array([-math.inf]), up=np.array([math.inf]),
                    Q=sp.csr_matrix((1, 1)), n=1, m=0)
        d = truth.dual_bound(prob, np.zeros(1), np.zeros(0))
        self.assertLessEqual(d, 0.0)
        self.assertGreaterEqual(d, -HAND_SLACK)


class LinearizedDualFallbackTests(unittest.TestCase):
    def test_linearized_dual_runs_when_repair_makes_no_progress(self):
        # min x1 + x2 s.t. x1 + x2 >= 1, x >= 0。y = 0 では g = (1, 1) >= 0 なので
        # _repair は 0 回で終わり d = 0。線形化 LP の双対 y = 1 では d = 1 になり p = 1 と一致する。
        prob = dict(c0=0.0, c=np.array([1.0, 1.0]),
                    A=sp.csr_matrix(np.array([[1.0, 1.0]])),
                    L=np.array([1.0]), U=np.array([math.inf]),
                    lo=np.zeros(2), up=np.array([math.inf, math.inf]),
                    Q=sp.csr_matrix((2, 2)), n=2, m=1)
        cert = truth.verify(prob, np.array([1.0, 0.0]), np.array([0.0]))
        self.assertEqual(cert.repair_iters, 0)
        self.assertEqual(cert.method, "linearized")
        self.assertLessEqual(cert.d, 1.0)
        self.assertGreater(cert.d, 1.0 - 1e-8)
        self.assertLessEqual(truth.uncertainty(cert), truth.UNCERTAINTY_TOL)


class ExactRationalRecoveryTests(unittest.TestCase):
    def test_negative_reduced_cost_is_repaired_exactly(self):
        # min 1/2 x^2 - 3x (x >= 0), x* = 3, f* = -9/2。xh = 2.9 では g = -0.1 < 0。
        # g = 0 を Q_PP s = -g で厳密に解くと xh = 3、d = -9/2。
        prob = scalar_problem()
        data = truth._ExactData(prob)
        s = truth._exact_sets(prob, np.array([3.0]))
        out = truth._recover_exact(prob, data, s, np.array([2.9]), np.zeros(0),
                                   {0}, np.zeros(0, dtype=int),
                                   time.monotonic() + 10.0)
        self.assertIsNotNone(out)
        xh, y, gex, iters = out
        self.assertEqual(iters, 1)
        d = truth._exact_d(data, xh, y, gex)
        self.assertEqual(d, HAND_OPTIMUM)
        self.assertLessEqual(d, HAND_OPTIMUM)
        self.assertEqual(truth._exact_float(d), -4.5)

    def test_exact_d_rejects_negative_cost_on_lower_only_column(self):
        data = truth._ExactData(scalar_problem())
        self.assertIsNone(truth._exact_d(data, [Fraction(0)], [], [Fraction(-1)]))

    def test_exact_float_rounds_down_and_up(self):
        # 1/10 は最近接 float が真値より上、3/10 は下。どちらも下界は真値以下、
        # 上界は真値以上になること。
        for exact in (Fraction(1, 10), Fraction(3, 10)):
            down = truth._exact_float(exact)
            up = truth._fraction_float_up(exact)
            self.assertLessEqual(Fraction(down), exact)
            self.assertLess(exact, Fraction(up))
        self.assertEqual(truth._exact_float(HAND_OPTIMUM), -4.5)
class ExactFreeVariableCertificateTests(unittest.TestCase):
    """自由変数の縮約費用を厳密に 0 にする証明と、非対角 Q の厳密 KKT の sentinel。"""

    def test_diagonal_fr_uses_exact_zero_reduced_cost(self):
        # min 1/2 * 3 x^2 + x (x 自由)。最適値 -1/6。x = fl(-1/3) では
        # g = 1 + 3x が浮動小数の区間で 0 を跨ぐため dual_bound は -inf。
        # 有理数で xh = -1/3 を構成すると g = 0 が厳密になり下界が閉じる。
        prob = dict(c0=0.0, c=np.array([1.0]), A=sp.csr_matrix((0, 1)), L=np.zeros(0),
                    U=np.zeros(0), lo=np.array([-math.inf]), up=np.array([math.inf]),
                    Q=sp.csr_matrix(np.array([[3.0]])), n=1, m=0)
        x = np.array([-1.0 / 3.0])
        self.assertEqual(truth.dual_bound(prob, x, np.zeros(0)), -math.inf)
        cert = truth.verify(prob, x, np.zeros(0))
        opt = Fraction(-1, 6)
        self.assertLessEqual(Fraction(cert.d), opt)
        self.assertLessEqual(opt - Fraction(cert.d), HAND_SLACK)
        # 証明書は厳密 KKT (xh 自由列の g = 0 が厳密なので補完性が成立) となり、
        # 参照値 p も厳密な最適値 -1/6 に置き換わる (primal ゲートを通る)。
        # pimpact は d の下向き丸め分 (有理数で境界付け、0 でなくても厳密) になる。
        self.assertTrue(cert.primal)
        self.assertEqual(cert.p, -1.0 / 6.0)
        self.assertLessEqual(Fraction(cert.pimpact), Fraction(1, 2) ** -40)

    def test_q0_free_column_is_corrected_exactly(self):
        # min x1 s.t. x1 - x2 = 0, x1 >= 0, x2 自由 (最適値 0)。y = 1 - 2^-40 の
        # ままでは q=0 の自由列 x2 の縮約費用 g2 = -y != 0。_sparse_correction が
        # A_F0' y = c_F0 を厳密に解いて y = 0 にし、下界 0 を与える。
        prob = dict(c0=0.0, c=np.array([1.0, 0.0]),
                    A=sp.csr_matrix(np.array([[1.0, -1.0]])), L=np.zeros(1), U=np.zeros(1),
                    lo=np.array([0.0, -math.inf]), up=np.array([math.inf, math.inf]),
                    Q=sp.csr_matrix((2, 2)), n=2, m=1)
        y = np.array([1.0 - 2.0 ** -40])
        self.assertEqual(truth.dual_bound(prob, np.zeros(2), y), -math.inf)
        fr = truth._fr_certificate(prob, np.zeros(2), y)
        self.assertIsNotNone(fr)
        self.assertLessEqual(fr[0], 0.0)
        self.assertLessEqual(-Fraction(fr[0]), HAND_SLACK)

    def test_non_diagonal_q_falls_back_to_exact_kkt(self):
        # min 1/2 x'Qx + c'x (x 自由), Q = [[2,1],[1,2]], c = [-1,-2]。最適値 -1。
        # Q が非対角なので _fr_certificate は使えず、厳密 KKT が下界 -1 を与える。
        prob = dict(c0=0.0, c=np.array([-1.0, -2.0]), A=sp.csr_matrix((0, 2)),
                    L=np.zeros(0), U=np.zeros(0),
                    lo=np.array([-math.inf, -math.inf]), up=np.array([math.inf, math.inf]),
                    Q=sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]])), n=2, m=0)
        x = np.array([0.0, 1.0])
        self.assertIsNone(truth._fr_certificate(prob, x, np.zeros(0)))
        cert = truth.verify(prob, x, np.zeros(0))
        opt = Fraction(-1)
        self.assertLessEqual(Fraction(cert.d), opt)
        self.assertLessEqual(opt - Fraction(cert.d), HAND_SLACK)
        self.assertEqual(cert.pimpact, 0.0)

    def test_exact_psd_rejects_indefinite_q_and_skips_kkt(self):
        # Q = [[1,2],[2,1]] は固有値 -1, 3 で非凸。浮動小数の固有値判定では
        # ない厳密 LDL が負 pivot を見つけ、KKT 証明を出さない。
        Q = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
        self.assertFalse(truth._psd_exact(Q))
        prob = dict(c0=0.0, c=np.zeros(2), A=sp.csr_matrix((0, 2)), L=np.zeros(0),
                    U=np.zeros(0), lo=np.array([-math.inf, -math.inf]),
                    up=np.array([math.inf, math.inf]), Q=Q, n=2, m=0, psd=False)
        self.assertIsNone(truth._kkt_certificate(prob, np.zeros(2)))
        self.assertEqual(truth.verify(prob, np.zeros(2), np.zeros(0)).d, -math.inf)

    def test_exact_psd_accepts_semidefinite_and_zero_matrices(self):
        self.assertTrue(truth._psd_exact(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))))
        self.assertTrue(truth._psd_exact(sp.csr_matrix((2, 2))))
        self.assertFalse(truth._psd_exact(sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))))


class SparseExactCertificateTests(unittest.TestCase):
    """dense 上限を超える疎問題を厳密有理数 dual で認証する経路の sentinel。"""

    def test_sparse_path_repairs_small_problem(self):
        # min 1/2 x^2 - 3x (x >= 0), x* = 3。x = 2.9 の縮約費用 g = -0.1 を
        # 疎な厳密解で xh = 3 へ修復し、d = -9/2 を返す。
        prob = scalar_problem()
        x = np.array([2.9])
        base = truth.verify(prob, x, np.zeros(0))
        cert = truth._sparse_exact_certificate(prob, x, base, np.zeros(0))
        self.assertIsNotNone(cert)
        self.assertEqual(cert.method, "exact-sparse")
        self.assertEqual(cert.d, -4.5)
        self.assertLessEqual(Fraction(cert.d), HAND_OPTIMUM)

    def test_free_column_strategy_uses_all_rows_when_active_rows_fail(self):
        # 自由列を含む問題で、active 行だけを候補に残すと、active 行に触れない
        # 内部列の方程式 0 = -g_j (空係数、右辺非零) が不整合になる。全行を候補に
        # 残し、自由列の g_j = 0 だけを初期方程式にして修復する経路がこれを直す。
        #
        # 列: f (自由, Q 列 0, 費用 1, 行 0 係数 1), z (両側有限 [-1,1], Q 列 0,
        # 費用 1, 行 1 係数 1), b (両側有限 [0,2], Q = 2, 費用 -2, 行 0 係数 1)。
        # 行: 0 は f + b = 2 (等式), 1 は z ∈ [0,1] (ranged)。候補 x = (1, 0.5, 1),
        # y = (0.5, 0): g_f = 1 - 0.5 = 0.5 != 0 で下界 -inf、z は内部で
        # g_z = 1 - y_1 = 1 != 0 だが z が触れる行 1 は非 active なので
        # active 行だけの系は 0 = -1 で不整合。
        prob = dict(c0=0.0, c=np.array([1.0, 1.0, -2.0]),
                    Q=sp.csr_matrix(np.array([[0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0],
                                              [0.0, 0.0, 2.0]])),
                    A=sp.csr_matrix(np.array([[1.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0]])),
                    L=np.array([2.0, 0.0]), U=np.array([2.0, 1.0]),
                    lo=np.array([-math.inf, -1.0, 0.0]),
                    up=np.array([math.inf, 1.0, 2.0]), n=3, m=2)
        x = np.array([1.0, 0.5, 1.0])
        y = np.array([0.5, 0.0])
        base = truth.verify(prob, x, y)
        # repair が等式行の乗数で g_f を直すので base.d は有限だが、緩すぎ
        # (uncertainty > tol) のまま厳密経路が走る。
        self.assertGreater(truth.uncertainty(base), truth.UNCERTAINTY_TOL)
        # active 行だけを候補に残す旧経路は不整合 (空方程式 0 = -1) で None。
        s = truth._exact_sets(prob, x)
        data = truth._ExactData(prob)
        active = np.flatnonzero(s["atL"] | s["atU"])
        self.assertEqual(list(active), [0])
        xh0, y0 = truth._exact_base(prob, s, x, y, active)
        self.assertIsNone(truth._recover_exact(
            prob, data, s, xh0, y0,
            {int(j) for j in np.flatnonzero(s["interior"])}, active,
            time.monotonic() + 60))
        cert = truth._sparse_exact_certificate(prob, x, base, y)
        self.assertIsNotNone(cert)
        self.assertEqual(cert.method, "exact-sparse-free")
        # y_0 = 1 で g_f = 0 厳密、d = h - 1/2 x'Qx + 境界項 = 2 - 1 - 1 - 2 = -2。
        self.assertEqual(cert.d, -2.0)
        self.assertLessEqual(Fraction(cert.d), Fraction(cert.p))

    def test_banded_free_certificate_solves_all_free_all_ranged(self):
        # 全列自由・全行 ranged (LASER 形) で Q が帯行列の問題: 弱双対は
        # g = Qxh - A'y = 0 (厳密) だけを要求し、y は符号自由。帯消去で
        # Q xh = A'y - c を厳密に解いて d = h(y) - 1/2 xh'Qxh を返す。
        # Q = [[2,1,0],[1,2,1],[0,1,2]], A = [[1,1,0],[0,1,1]], 候補
        # x = (0,1,0), y = (1,1): g = (1,2,1) - (1,2,1) = 0 厳密で、
        # xh = x 厳密、h = 1*0 + 1*0 = 0、d = -1/2 * (x'Qx) = -1。
        prob = dict(c0=0.0, c=np.zeros(3),
                    Q=sp.csr_matrix(np.array([[2.0, 1.0, 0.0],
                                              [1.0, 2.0, 1.0],
                                              [0.0, 1.0, 2.0]])),
                    A=sp.csr_matrix(np.array([[1.0, 1.0, 0.0],
                                              [0.0, 1.0, 1.0]])),
                    L=np.zeros(2), U=np.full(2, 2.0),
                    lo=np.full(3, -math.inf), up=np.full(3, math.inf),
                    n=3, m=2)
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([1.0, 1.0])
        out = truth._banded_free_certificate(prob, x, y, 1.0)
        self.assertIsNotNone(out)
        self.assertEqual(out[0], -1.0)
        self.assertLessEqual(out[1], truth.UNCERTAINTY_TOL)
        # stationarity 残差が BANDED_G0_TOL を超える候補は前に断す (有効な下界は
        # 近特異方向の増幅で作れない)。
        self.assertIsNone(truth._banded_free_certificate(
            prob, x, np.array([2.0, 1.0]), 1.0))

    def test_zero_cost_dummy_columns_are_certified_by_sparse_path(self):
        # 浮動小数では認証できない既知の小問題 (SEED=20260821, i=5, 6 番目の摂動) に、
        # 目的・制約・曲率を持たない零コスト列を EXACT_MAX_COLS 本足す。n が dense 経路の
        # 上限を超えるので _exact_certificate は候補を作れず、疎経路だけが認証できる。
        prob, x, y = _padded_sentinel_problem(truth.EXACT_MAX_COLS)
        self.assertGreater(prob["n"], truth.EXACT_MAX_COLS)
        base = truth.verify(prob, x, y)
        self.assertGreater(truth.uncertainty(base), truth.UNCERTAINTY_TOL)
        self.assertIsNone(truth._exact_certificate(prob, x, base, y))
        cert = truth.verify(prob, x, y, exact=True)
        self.assertEqual(cert.method, "exact-sparse")
        self.assertLessEqual(truth.uncertainty(cert), truth.UNCERTAINTY_TOL)
        self.assertEqual(cert.d, PADDED_SENTINEL_D)
        self.assertLessEqual(Fraction(cert.d), Fraction(cert.p))
        # polish() が候補を作れない場合でも、best_certificate() が疎経路へ落ちること。
        with mock.patch.object(truth, "polish", return_value=None):
            best = truth.best_certificate(prob, x, y)
        self.assertEqual(best.method, "exact-sparse")
        self.assertLessEqual(truth.uncertainty(best), truth.UNCERTAINTY_TOL)


def _padded_sentinel_problem(pad):
    """小問題 (SEED=20260821, i=5) の x/y を保ったまま零コスト列を pad 本足す。"""
    prob = dict(c0=0.4847014283021232, c=np.array([-0.4, 0.2, -0.6]),
                Q=sp.csr_matrix(np.array([[6.61, -3.85, -1.33],
                                          [-3.85, 2.7800000000000002, 0.69],
                                          [-1.33, 0.69, 3.54]])),
                A=sp.csr_matrix(np.array([[1.8, 0.1, 0.3]])),
                L=np.array([-math.inf]), U=np.array([math.inf]),
                lo=np.array([-0.3, -math.inf, -math.inf]),
                up=np.array([math.inf, 1.9, 1.2]), n=3, m=1)
    x = np.array([0.15792544888563192, 0.09444327429383367, 0.21033222907624805])
    y = np.array([0.0012936184205120916])
    padded = dict(prob, n=prob["n"] + pad,
                  c=np.concatenate([prob["c"], np.zeros(pad)]),
                  Q=sp.block_diag([prob["Q"], sp.csr_matrix((pad, pad))], format="csr"),
                  A=sp.hstack([prob["A"], sp.csr_matrix((prob["m"], pad))], format="csr"),
                  lo=np.concatenate([prob["lo"], np.zeros(pad)]),
                  up=np.concatenate([prob["up"], np.full(pad, math.inf)]))
    return padded, np.concatenate([x, np.zeros(pad)]), y


# 疎経路が返す厳密下界 (この問題の active set は一意なので再現可能)。
PADDED_SENTINEL_D = 0.3994564252974797


SEED = 20260821
N_PROBLEMS = 200
POINTS_PER_PROBLEM = 5
# 各分岐を踏んだ試行数の下限。生成器が分岐を踏まなくなって検査が空洞化するのを防ぐ。
MIN_TRIALS_PER_BRANCH = 20
# oracle の HiGHS 時間上限 [s]。小規模 QP は数 ms で解ける。超える問題 (非有界など) は捨てる。
ORACLE_TIME_LIMIT_SECS = 1.0
# 最適解近傍の摂動幅 10^U(lo, hi)。上限 1e-3 は HiGHS 解の主双対誤差より十分大きく、
# 下限 1e-9 は認証を通りうる最小の摂動 (UNCERTAINTY_TOL = 1e-7 の 1 桁以上下)。
PERTURB_LOG10_RANGE = (-9.0, -3.0)
# 最適解近傍の摂動点のうち、_repair を経て認証を通る試行数の下限。
MIN_ACCEPTED_REPAIRED = 20
# 認証を通った候補の主目的値と真の最適値の相対差の上限 (認証許容そのもの)。
ACCEPTED_ERR_TOL = truth.UNCERTAINTY_TOL
# x0 を無限側の境界方向へ切り詰める範囲。N(0,1) の標本を実質切らない幅。
BOUND_DOMAIN = 9.0
# 両側有限変数の Q 列を 0 にする確率。特異 Q かつ片側無限変数に曲率が残る経路を踏む値。
ZERO_COLUMN_PROB = 0.7
# Q = 0 (LP) にする確率。曲率のない片側無限変数で縮約費用の符号を y の修正だけで直す
# 経路 (旧実装の step=None に当たる) を十分な頻度で踏む値。
LP_PROB = 0.35
# 活性判定 (oracle の候補 active set) の許容。HiGHS の主実行可能性許容 1e-7 と同じ。
ORACLE_ACTIVE_TOL = 1e-7
# active set 列挙で浮動小数の KKT 解を厳密検算へ回すかの許容 (緩く取り、判定は厳密に行う)。
ORACLE_SCREEN_TOL = 1e-6


def random_problem(rng):
    """凸 QP (Q = B'B, B は k 行 <= n なので特異もある; LP_PROB で Q = 0)。x0 が実行可能に
    なるよう行範囲を作る。"""
    n, m = int(rng.integers(2, 6)), int(rng.integers(0, 4))
    k = 0 if rng.random() < LP_PROB else int(rng.integers(0, n + 1))
    B = rng.normal(size=(k, n)).round(1)
    A = rng.normal(size=(m, n)).round(1)
    lo = np.where(rng.random(n) < .5, -rng.random(n).round(1), -math.inf)
    up = np.where(rng.random(n) < .5, 1 + rng.random(n).round(1), math.inf)
    fixed = (rng.random(n) < .15) & np.isfinite(lo)
    up = np.where(fixed, lo, up)
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


def _solve_exact(M, r):
    """有理数行列 M u = r の解 (自由変数は 0) を Gauss-Jordan で返す。不整合なら None。"""
    rows, cols = len(M), len(M[0]) if M else 0
    aug = [list(M[i]) + [r[i]] for i in range(rows)]
    pivots, row = [], 0
    for col in range(cols):
        piv = next((i for i in range(row, rows) if aug[i][col] != 0), None)
        if piv is None:
            continue
        aug[row], aug[piv] = aug[piv], aug[row]
        inv = 1 / aug[row][col]
        aug[row] = [v * inv for v in aug[row]]
        for i in range(rows):
            if i != row and aug[i][col] != 0:
                f = aug[i][col]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[row])]
        pivots.append(col)
        row += 1
    if any(aug[i][cols] != 0 for i in range(row, rows)):
        return None
    sol = [Fraction(0)] * cols
    for i, col in enumerate(pivots):
        sol[col] = aug[i][cols]
    return sol


class PrimalExactCertificateTests(unittest.TestCase):
    """参照値の主上界 (P1 指摘) の sentinel: 厳密 KKT が立てない候補は未認証。

    min 1/2 x^2 (x 自由), 1e-8 x = 1e-8。違反 1e-8 は PRIMAL_INF_TOL (1e-7) 未満で、
    乗数 y=0 だと一次感度 (pimpact) は 0 になる。浮動小数の候補 x=0 の p=0 は
    「最適値の上界」として何も支えず、真の最適値は 0.5。
    """

    def setUp(self):
        self.prob = dict(c0=0.0, c=np.array([0.0]),
                         A=sp.csr_matrix(np.array([[1e-8]])), L=np.array([1e-8]),
                         U=np.array([1e-8]), lo=np.array([-math.inf]),
                         up=np.array([math.inf]),
                         Q=sp.csr_matrix(np.array([[1.0]])), n=1, m=1)

    def solved(self, x, y, reported=None):
        if reported is None:
            reported = float(self.prob["c0"] + self.prob["c"] @ x
                             + 0.5 * x @ (self.prob["Q"] @ x))
        return ("Optimal", np.array(x), np.array(y), reported)

    def test_exact_primal_upgrade_certifies_true_optimum(self):
        # 厳密 KKT (xh = 1, y = 1e8 を有理数で) が立つので、参照値は浮動小数の
        # 候補の p=0 でなく厳密な最適値 0.5 に置き換わる。
        ok, p, msg = truth.certify(self.prob, self.solved([0.0], [0.0], 0.0))
        self.assertTrue(ok, msg)
        self.assertEqual(p, 0.5)
        cert = truth.best_certificate(self.prob, np.array([0.0]), np.array([0.0]))
        self.assertTrue(cert.primal)
        self.assertEqual(cert.d, 0.5)
        self.assertEqual(cert.pinf, 0.0)

    def test_primal_gate_rejects_candidate_without_exact_upgrade(self):
        # 厳密有理数の候補が時間・サイズ上限で得られない問題では、浮動小数の
        # 候補だけで参照値を「認証」しない (以前は p=d=pimpact=0 で合格していた)。
        with mock.patch.object(truth, "_exact_certificate", return_value=None), \
                mock.patch.object(truth, "_sparse_exact_certificate",
                                  return_value=None):
            ok, p, msg = truth.certify(self.prob, self.solved([0.0], [0.0], 0.0))
            self.assertFalse(ok, msg)
            cert = truth.best_certificate(self.prob, np.array([0.0]),
                                          np.array([0.0]))
            self.assertFalse(cert.primal)

    def test_degenerate_zero_multiplier_equality_row_upgrades(self):
        # TAME 型: Q が特異 (ベクトル (1,-1) の 2 乗) で等式行の最適乗数が厳密 0
        # になる (min (x1-x2)^2 s.t. x1+x2=1, x >= 0 の最適 (1/2, 1/2) は内部点)。
        # ソルバ乗数に 5e-8 のノイズが残っていても、interior 列の g = 0 と行 pin
        # (pin 方程式は行をまたぐ interior 列すべてを覆う) で厳密 KKT が立ち、
        # 参照値は厳密な最適値 0 に置き換わる。pin を 1 列で立てる・g の方程式を
        # interior 列に展開しない旧実装では f(xh) == d が立たない (sentinel)。
        prob = dict(c0=0.0, c=np.array([0.0, 0.0]),
                    A=sp.csr_matrix(np.array([[1.0, 1.0]])),
                    L=np.array([1.0]), U=np.array([1.0]),
                    lo=np.array([0.0, 0.0]),
                    up=np.array([math.inf, math.inf]),
                    Q=sp.csr_matrix(np.array([[2.0, -2.0], [-2.0, 2.0]])),
                    n=2, m=1)
        solved = ("Optimal", np.array([0.5, 0.5]), np.array([5e-8]), 0.0)
        ok, p, msg = truth.certify(prob, solved)
        self.assertTrue(ok, msg)
        self.assertEqual(p, 0.0)

    def test_pin_equation_spans_all_row_columns(self):
        # pin 方程式 (A xh = 境界) は行をまたぐ全 interior 列を覆う必要がある。
        # 1 列分しか式を立てないと残りの列が動いた分だけ Axh が境界から外れ、
        # f(xh) == d の厳密一致が立たない。境界を float 和の 1 ulp 先に置いて
        # pin の右辺を厳密に非零にし、interior の 2 列をそろって動かして初めて
        # 境界に一致する状況を作る。
        s = 0.1 + 0.4
        bound = float(np.nextafter(s, np.inf))
        prob = dict(c0=0.0, c=np.array([0.0, 0.0]),
                    A=sp.csr_matrix(np.array([[1.0, 1.0]])),
                    L=np.array([bound]), U=np.array([bound]),
                    lo=np.array([0.0, 0.0]),
                    up=np.array([math.inf, math.inf]),
                    Q=sp.csr_matrix(np.diag([2.0, 0.5])), n=2, m=1)
        # 最適は 2x1 = 0.5x2 = y, x1+x2 = bound より x1 = bound/5, x2 = 4*bound/5。
        # 浮動小数の候補 x は KKT から 1e-17 程度しか外れていない。
        x = np.array([0.1, 0.4])
        y = np.array([0.2])
        reported = float(0.5 * x @ (prob["Q"] @ x))
        ok, p, msg = truth.certify(prob, ("Optimal", x, y, reported))
        self.assertTrue(ok, msg)
        # 厳密な最適値 f* = x1*^2 + x2*^2 / 4 (f = 1/2 x'Qx, Q = diag(2, 1/2))。
        expected = float(Fraction(bound) ** 2 / 5)
        self.assertEqual(p, expected)

    def test_newly_pinned_row_counts_as_recovery_progress(self):
        # 進捨の基準を新 pin の挿入前に取る (Codex P2 4370997713)。候補 x=1, y=0
        # は g = -1e-8 のノイズを残し、初反復で E 増して xh が境界外へ動き、次の
        # 反復だけ行を pin する (E/pin_rows 増えない)。pin 挿入後に基準を取る旧
        # 実装は no-progress と誤判定して None を返す。下限 -10 (有限) は自由列の
        # fr 証明を閉け、_dual_qp_candidate を退させる (A2/B の救済経路を閉け、
        # clarabel の成否で sentinel が揺れるのを防ぐ)。厳密修復が
        # _recover_exact() の pin 進捨だけになるので、旧実装の早退 None は
        # certify の合格に直に効く。min 0.5x^2 - 1.00000001x s.t. x <= 1,
        # x >= -10 の厳密最適は x = 1 で f* = 1/2 - 1.00000001 (float 1.00000001
        # の有理数値)。
        prob = dict(c0=0.0, c=np.array([-1.00000001]),
                    A=sp.csr_matrix(np.array([[1.0]])),
                    L=np.array([-math.inf]), U=np.array([1.0]),
                    lo=np.array([-10.0]), up=np.array([math.inf]),
                    Q=sp.csr_matrix(np.array([[1.0]])), n=1, m=1)
        solved = ("Optimal", np.array([1.0]), np.array([0.0]),
                  float(0.5 - 1.00000001))
        with mock.patch.object(truth, "_dual_qp_candidate", return_value=None):
            ok, p, msg = truth.certify(prob, solved)
        self.assertTrue(ok, msg)
        expected = float(Fraction(1, 2) - Fraction(float(1.00000001)))
        self.assertEqual(p, expected)


class ExactProblem:
    """浮動小数データを Fraction に移した問題 (QPS 由来の浮動小数は有理数として厳密)。"""

    def __init__(self, prob):
        self.n, self.m = prob["n"], prob["m"]
        frac = lambda v: [Fraction(float(e)) if math.isfinite(e) else None for e in v]
        self.c0 = Fraction(prob["c0"])
        self.c = frac(prob["c"])
        self.A = [frac(r) for r in prob["A"].toarray()]
        self.Q = [frac(r) for r in prob["Q"].toarray()]
        self.L, self.U = frac(prob["L"]), frac(prob["U"])
        self.lo, self.up = frac(prob["lo"]), frac(prob["up"])

    def objective(self, x):
        qx = [sum(self.Q[i][j] * x[j] for j in range(self.n)) for i in range(self.n)]
        return self.c0 + sum(ci * xi for ci, xi in zip(self.c, x)) + sum(
            xi * qi for xi, qi in zip(x, qx)) / 2

    def kkt_point(self, var_state, row_state):
        """active set (変数: 'l'/'u'/'f', 行: 'l'/'u'/'f') の KKT を厳密に解き、
        主実行可能・乗数の符号が正しければ最適解 x (Fraction) を返す (凸 QP の十分条件)。"""
        n = self.n
        free = [j for j in range(n) if var_state[j] == "f"]
        act = [i for i in range(self.m) if row_state[i] != "f"]
        x_fix = {j: (self.lo[j] if var_state[j] == "l" else self.up[j])
                 for j in range(n) if var_state[j] != "f"}
        k = len(free) + len(act)
        M, r = [], []
        for j in free:
            M.append([self.Q[j][f] for f in free] + [-self.A[i][j] for i in act])
            r.append(-self.c[j] - sum(self.Q[j][b] * v for b, v in x_fix.items()))
        for i in act:
            M.append([self.A[i][f] for f in free] + [Fraction(0)] * len(act))
            b = self.L[i] if row_state[i] == "l" else self.U[i]
            r.append(b - sum(self.A[i][b2] * v for b2, v in x_fix.items()))
        sol = _solve_exact(M, r) if k else []
        if sol is None:
            return None
        x = [x_fix.get(j, Fraction(0)) for j in range(n)]
        for t, j in enumerate(free):
            x[j] = sol[t]
        y = [Fraction(0)] * self.m
        for t, i in enumerate(act):
            y[i] = sol[len(free) + t]
        for j in range(n):
            if (self.lo[j] is not None and x[j] < self.lo[j]) or (
                    self.up[j] is not None and x[j] > self.up[j]):
                return None
        for i in range(self.m):
            ax = sum(self.A[i][j] * x[j] for j in range(n))
            if (self.L[i] is not None and ax < self.L[i]) or (
                    self.U[i] is not None and ax > self.U[i]):
                return None
            if self.L[i] != self.U[i] and (
                    (row_state[i] == "l" and y[i] < 0) or (row_state[i] == "u" and y[i] > 0)):
                return None
        for j in range(n):
            z = self.c[j] + sum(self.Q[j][k2] * x[k2] for k2 in range(n)) - sum(
                self.A[i][j] * y[i] for i in range(self.m))
            if self.lo[j] != self.up[j] and (
                    (var_state[j] == "l" and z < 0) or (var_state[j] == "u" and z > 0)):
                return None
        return x


def _states(lower, upper):
    """変数/行の取りうる active 状態 (有限側の境界と 'f')。等式 (lower == upper) は 'l' のみ。"""
    if lower is not None and lower == upper:
        return ("l",)
    return tuple(s for s, ok in (("l", lower is not None), ("u", upper is not None), ("f", True))
                 if ok)


def _guided_state(val, lower, upper):
    near = lambda b: b is not None and abs(val - float(b)) <= ORACLE_ACTIVE_TOL * (1 + abs(float(b)))
    if lower is not None and lower == upper:
        return "l"
    return "l" if near(lower) else "u" if near(upper) else "f"


def exact_optimum(prob):
    """(真の最適値 Fraction, 最適解 x* (Fraction), HiGHS の x, HiGHS の y)。
    HiGHS が最適を返さない問題と、厳密 KKT 点が見つからない問題は None。

    HiGHS の解の active set を先に厳密 KKT で試し、通らなければ全 active set を列挙する
    (浮動小数の KKT 解で篩ってから厳密に検算)。返す値は KKT の厳密検算を通った点の目的値で、
    凸 QP では大域最適値そのもの (HiGHS の精度に依存しない)。
    """
    status, x, y, _ = truth.solve_highs(prob, ORACLE_TIME_LIMIT_SECS)
    if status != "Optimal":
        return None
    ex = ExactProblem(prob)
    ax = prob["A"] @ x
    guided = ([_guided_state(x[j], ex.lo[j], ex.up[j]) for j in range(ex.n)],
              [_guided_state(ax[i], ex.L[i], ex.U[i]) for i in range(ex.m)])
    candidates = itertools.chain([guided], itertools.product(
        itertools.product(*[_states(ex.lo[j], ex.up[j]) for j in range(ex.n)]),
        itertools.product(*[_states(ex.L[i], ex.U[i]) for i in range(ex.m)])))
    for var_state, row_state in candidates:
        if (list(var_state), list(row_state)) != guided and not _float_screen(
                prob, var_state, row_state):
            continue
        xs = ex.kkt_point(var_state, row_state)
        if xs is not None:
            return ex.objective(xs), xs, x, y
    return None


def _float_screen(prob, var_state, row_state):
    """active set の KKT を浮動小数の最小二乗で解き、主実行可能性が緩い許容で成り立つか。"""
    n, A, Q, c = prob["n"], prob["A"].toarray(), prob["Q"].toarray(), prob["c"]
    free = [j for j in range(n) if var_state[j] == "f"]
    act = [i for i in range(prob["m"]) if row_state[i] != "f"]
    x = np.array([0.0 if s == "f" else (prob["lo"][j] if s == "l" else prob["up"][j])
                  for j, s in enumerate(var_state)])
    if free or act:
        M = np.block([[Q[np.ix_(free, free)], -A[np.ix_(act, free)].T],
                      [A[np.ix_(act, free)], np.zeros((len(act), len(act)))]])
        b = np.array([prob["L"][i] if row_state[i] == "l" else prob["U"][i] for i in act])
        rhs = np.concatenate([-(c[free] + Q[np.ix_(free, range(n))] @ x),
                              b - A[np.ix_(act, range(n))] @ x])
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        if np.abs(M @ sol - rhs).max(initial=0.0) > ORACLE_SCREEN_TOL:
            return False
        x[free] = sol[: len(free)]
    tol = ORACLE_SCREEN_TOL
    with np.errstate(invalid="ignore"):
        ok_x = np.all(~(x < prob["lo"] - tol) & ~(x > prob["up"] + tol))
        ax = A @ x
        ok_r = np.all(~(ax < prob["L"] - tol) & ~(ax > prob["U"] + tol))
    return bool(ok_x and ok_r)


def lower_bound_holds(cert, opt):
    """d <= 真の最適値 (有理数で厳密比較)。d = -inf は常に下界。"""
    return cert.d == -math.inf or Fraction(cert.d) <= opt


class RandomLowerBoundTests(unittest.TestCase):
    def test_dual_bound_never_exceeds_true_optimum(self):
        counts = dict(repaired=0, repaired_lp=0, linearized=0, unrepaired=0,
                      singular=0, ranges=0,
                      fixed_var=0, row_multiplier=0, mixed_bounds=0)
        minus_inf = 0
        for i in range(N_PROBLEMS):
            rng = np.random.default_rng([SEED, i])
            prob, x0 = random_problem(rng)
            oracle = exact_optimum(prob)
            if oracle is None:
                continue
            opt, x_opt, _, _ = oracle
            x_opt_f = np.array([float(v) for v in x_opt])
            n = prob["n"]
            lp = prob["Q"].nnz == 0
            singular = np.linalg.matrix_rank(prob["Q"].toarray()) < n
            ranges = bool(np.any(np.isfinite(prob["L"]) & np.isfinite(prob["U"])
                                 & (prob["L"] < prob["U"])))
            fixed_var = bool(np.any(prob["lo"] == prob["up"]))
            finite = np.isfinite(prob["lo"]) | np.isfinite(prob["up"])
            mixed = bool(finite.any() and not finite.all())
            for _ in range(POINTS_PER_PROBLEM):
                xb = x0 + rng.normal(size=n) * rng.choice([0, .3, 2])
                xb = np.where(rng.random(n) < .3, x_opt_f, xb)
                yb = rng.normal(size=prob["m"]) * rng.choice([0, .5, 3])
                cert = truth.verify(prob, xb, yb)
                self.assertTrue(lower_bound_holds(cert, opt),
                                f"problem {i}: d={cert.d!r} > opt={float(opt)!r} {cert}")
                if cert.d == -math.inf:
                    minus_inf += 1
                if cert.repair_iters:
                    counts["repaired"] += 1
                    counts["repaired_lp"] += lp
                    counts["linearized"] += cert.repair_iters >= truth.REPAIR_ITERS
                else:
                    counts["unrepaired"] += 1
                counts["singular"] += singular
                counts["ranges"] += ranges
                counts["fixed_var"] += fixed_var
                counts["row_multiplier"] += bool(np.any(yb != 0))
                counts["mixed_bounds"] += mixed
        print(f"\nlower-bound branch counts: {counts} minus_inf={minus_inf}")
        for branch, count in counts.items():
            self.assertGreaterEqual(count, MIN_TRIALS_PER_BRANCH, f"{branch}: {counts}")

    def test_accepted_certificate_matches_true_optimum(self):
        """最適解近傍の (x, y) で認証 (pinf, uncertainty) を通った p は真の最適値に一致し、
        d は真の最適値を超えない。_repair を経た経路 (旧 step=None 相当を含む) もここで検査する。"""
        accepted = accepted_repaired = accepted_repaired_lp = 0
        for i in range(N_PROBLEMS):
            rng = np.random.default_rng([SEED, i])
            prob, _ = random_problem(rng)
            oracle = exact_optimum(prob)
            if oracle is None:
                continue
            opt, x_opt, x_hi, y_hi = oracle
            for _ in range(POINTS_PER_PROBLEM):
                scale = 10.0 ** rng.uniform(*PERTURB_LOG10_RANGE)
                xb = x_hi + scale * rng.normal(size=prob["n"]) * (rng.random(prob["n"]) < .5)
                yb = y_hi + scale * rng.normal(size=prob["m"]) * (rng.random() < .5)
                cert = truth.verify(prob, xb, yb)
                self.assertTrue(lower_bound_holds(cert, opt),
                                f"problem {i}: d={cert.d!r} > opt={float(opt)!r} {cert}")
                if cert.pinf > truth.PRIMAL_INF_TOL or truth.uncertainty(cert) > truth.UNCERTAINTY_TOL:
                    continue
                accepted += 1
                if cert.repair_iters:
                    accepted_repaired += 1
                    accepted_repaired_lp += prob["Q"].nnz == 0
                self.assertLessEqual(abs(Fraction(cert.p) - opt),
                                     Fraction(ACCEPTED_ERR_TOL) * (1 + abs(opt)))
        print(f"\naccepted={accepted} repaired={accepted_repaired} "
              f"repaired_lp={accepted_repaired_lp}")
        self.assertGreaterEqual(accepted_repaired, MIN_ACCEPTED_REPAIRED,
                                f"accepted={accepted} repaired={accepted_repaired}")
        self.assertGreaterEqual(accepted_repaired_lp, MIN_ACCEPTED_REPAIRED,
                                f"accepted={accepted} repaired_lp={accepted_repaired_lp}")


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


class PsdAndOverflowTests(unittest.TestCase):
    def test_psd_accepts_semidefinite_blocks(self):
        self.assertTrue(truth._psd_ok(sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))))
        # 特異 (固有値 0, 2) も PSD。
        self.assertTrue(truth._psd_ok(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))))

    def test_psd_rejects_indefinite_blocks(self):
        self.assertFalse(truth._psd_ok(sp.csr_matrix(np.array([[-1.0]]))))
        self.assertFalse(truth._psd_ok(sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))))

    def test_psd_rejects_negative_diagonal_below_tolerance(self):
        # P1 指摘: 許容差 (PSD_REL_TOL) で受け入れると min -0.5e-11 x^2 over [0, 1e8]
        # のような非 PSD Q に凸接線下界が使われ、x=0 の p=d=0 が証明書として誤って
        # 合格する。対角成分はパース値そのもの (丸め誤差なし) なので、どんなに小さく
        # ても負なら非 PSD。
        self.assertFalse(truth._psd_ok(sp.csr_matrix(np.array([[-1e-11]]))))
        self.assertFalse(truth._psd_ok(
            sp.csr_matrix(np.array([[1.0, 1.0], [1.0, -1e-11]]))))
        # 受け入れ側の境界も固定する (半正定値の微小値と -0.0)。
        self.assertTrue(truth._psd_ok(sp.csr_matrix(np.array([[1e-11]]))))
        self.assertTrue(truth._psd_ok(sp.csr_matrix(np.array([[-0.0]]))))

    def test_minimal_negative_curvature_is_not_certified(self):
        # P1 指摘の end-to-end sentinel: min -0.5e-11 x^2 over [0, 1e8] の真の
        # 最適値は -50000。旧コードは許容差で Q を PSD と判定し、x=0 の p=d=0 を
        # 重み 0 で証明書として合格していた。
        prob = scalar_problem()
        prob["c"] = np.zeros(1)
        prob["Q"] = sp.csr_matrix(np.array([[-1e-11]]))
        prob["up"] = np.array([1e8])
        # load_qps と同じ流れで psd フラグを詰める。
        prob["psd"] = truth._psd_ok(prob["Q"])
        cert = truth.verify(prob, np.array([0.0]), np.zeros(0))
        self.assertEqual(cert.d, -math.inf)
        ok, _p, _msg = truth.certify(prob, ("Optimal", np.array([0.0]), np.zeros(0), 0.0))
        self.assertFalse(ok)

    def test_nonconvex_problem_is_not_certified(self):
        prob = scalar_problem()
        prob["Q"] = sp.csr_matrix(np.array([[-1.0]]))
        prob["lo"] = np.array([-1.0])
        prob["up"] = np.array([1.0])
        prob["psd"] = False
        cert = truth.verify(prob, np.array([0.0]), np.zeros(0))
        self.assertEqual(cert.d, -math.inf)
        self.assertFalse(truth.certify(prob, ("Optimal", np.array([0.0]), np.zeros(0), 0.0))[0])

    def test_overflowing_quad_form_returns_minus_inf(self):
        Q = sp.csr_matrix(np.array([[1.0, -2.0], [-2.0, 5.0]]))
        val, err = truth._quad_form(Q, np.array([1e200, 1e200]))
        self.assertEqual(val, -math.inf)
        self.assertEqual(err, math.inf)


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


FIXED_QPS_FIRST_SECOND = (
    "NAME t\n"
    "ROWS\n"
    " N  OBJ\n"
    " E  r\n"
    "COLUMNS\n"
    "    X         OBJ       0.0\n"
    "    X         r         1.0\n"
    "QUADOBJ\n"
    "    X         X         1.0\n"
    "RHS\n"
    "    FIRST     r         1.0\n"
    "    SECOND    r         2.0\n"
    "ENDATA\n"
)

FREE_QPS_FIRST_SECOND = (
    "NAME t\n"
    "ROWS\n"
    " N OBJ\n"
    " E r\n"
    "COLUMNS\n"
    " X OBJ 0.0\n"
    " X r 1.0\n"
    "QUADOBJ\n"
    " X X 1.0\n"
    "RHS\n"
    " FIRST r 1.0\n"
    " SECOND r 2.0\n"
    "ENDATA\n"
)


class VectorSectionTests(unittest.TestCase):
    """RHS/RANGES の複数名前付きベクタ: 最初のベクタが勝つ (otspot-io と同じ)。"""

    def parse(self, text, fixed=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.qps"
            path.write_text(text)
            return truth.parse_qps(str(path), fixed=fixed)

    def test_rhs_multiple_named_vectors_first_wins_fixed(self):
        # P1 指摘: 第二ベクタで上書きすると、独立オラクルが第一ベクタを採用する
        # production パーサ (otspot-io/src/common/mod.rs:948) と異なるモデルを
        # 証明してしまう (test_qps_rhs_multiple_named_vectors_first_wins と同じ形状)。
        prob = self.parse(FIXED_QPS_FIRST_SECOND, fixed=True)
        self.assertEqual(prob["L"][0], 1.0)
        self.assertEqual(prob["U"][0], 1.0)

    def test_rhs_multiple_named_vectors_first_wins_free(self):
        prob = self.parse(FREE_QPS_FIRST_SECOND)
        self.assertEqual(prob["L"][0], 1.0)
        self.assertEqual(prob["U"][0], 1.0)

    def test_ranges_multiple_named_vectors_first_wins(self):
        # RANGES も同じ上書き問題を持つ。G 行 r (rhs は FIRST ベクタの 1.0) に
        # FIRST = 1.0 → U = 2.0。SECOND = 4.0 を受けても 2.0 のままであること。
        text = (
            "NAME t\n"
            "ROWS\n"
            " N OBJ\n"
            " G r\n"
            "COLUMNS\n"
            " X OBJ 0.0\n"
            " X r 1.0\n"
            "QUADOBJ\n"
            " X X 1.0\n"
            "RHS\n"
            " FIRST r 1.0\n"
            "RANGES\n"
            " FIRST r 1.0\n"
            " SECOND r 4.0\n"
            "ENDATA\n"
        )
        prob = self.parse(text)
        self.assertEqual(prob["L"][0], 1.0)
        self.assertEqual(prob["U"][0], 2.0)

    def test_named_vector_after_unnamed_entries_is_rejected(self):
        text = (
            "NAME t\n"
            "ROWS\n"
            " N OBJ\n"
            " E r\n"
            "COLUMNS\n"
            " X OBJ 0.0\n"
            " X r 1.0\n"
            "QUADOBJ\n"
            " X X 1.0\n"
            "RHS\n"
            " r 1.0\n"
            " FIRST r 2.0\n"
            "ENDATA\n"
        )
        with self.assertRaises(ValueError):
            self.parse(text)

    def test_duplicate_entry_in_first_vector_is_rejected(self):
        text = (
            "NAME t\n"
            "ROWS\n"
            " N OBJ\n"
            " E r\n"
            "COLUMNS\n"
            " X OBJ 0.0\n"
            " X r 1.0\n"
            "QUADOBJ\n"
            " X X 1.0\n"
            "RHS\n"
            " FIRST r 1.0\n"
            " FIRST r 2.0\n"
            "ENDATA\n"
        )
        with self.assertRaises(ValueError):
            self.parse(text)

    def test_duplicate_entry_in_discarded_later_vector_is_rejected(self):
        # 後続ベクタは棄てられるが、不正入力の黙認はしない (production と同じ)。
        text = (
            "NAME t\n"
            "ROWS\n"
            " N OBJ\n"
            " E r\n"
            "COLUMNS\n"
            " X OBJ 0.0\n"
            " X r 1.0\n"
            "QUADOBJ\n"
            " X X 1.0\n"
            "RHS\n"
            " FIRST r 1.0\n"
            " SECOND r 2.0\n"
            " SECOND r 3.0\n"
            "ENDATA\n"
        )
        with self.assertRaises(ValueError):
            self.parse(text)

    def test_unnamed_line_after_named_vector_is_duplicate_in_same_vector(self):
        text = (
            "NAME t\n"
            "ROWS\n"
            " N OBJ\n"
            " E r\n"
            "COLUMNS\n"
            " X OBJ 0.0\n"
            " X r 1.0\n"
            "QUADOBJ\n"
            " X X 1.0\n"
            "RHS\n"
            " FIRST r 1.0\n"
            " r 0.0\n"
            "ENDATA\n"
        )
        # 名前を省いた行は FIRST へ属性されるので、(FIRST, r) の重複として退けられる。
        with self.assertRaises(ValueError):
            self.parse(text)


class BaselineCsvSeparationTests(unittest.TestCase):
    """certified external CSV と selfbest regression CSV の役割分離を固定する。"""

    @staticmethod
    def load(name):
        path = ROOT / "data" / "baseline_objectives" / name
        with path.open() as f:
            lines = [line for line in f if not line.startswith("#")]
        return {row["problem_name"]: row for row in csv.DictReader(lines)}

    def test_reissued_csv_has_88_contract_claims_and_50_uncertified_rows(self):
        rows = self.load("maros_meszaros.csv")
        self.assertEqual(len(rows), 138)
        claims = {n: r for n, r in rows.items() if r["source"].startswith("exact_enclosure_v1;")}
        uncer = {n: r for n, r in rows.items() if r["source"].endswith("_UNCERTIFIED")}
        self.assertEqual(len(claims), 88)
        self.assertEqual(len(uncer), 50)
        reasons = {}
        for r in uncer.values():
            reason = r["source"].split("revalidation_")[1].split("_UNCERTIFIED")[0]
            reasons[reason] = reasons.get(reason, 0) + 1
        # 2026-10-31 frozen run の実測内訳。件数主張は CSV から自己記述で読める。
        self.assertEqual(reasons, {"no_verified_enclosure": 33, "psd_unknown": 16,
                                   "nonconvex": 1})
        for r in uncer.values():
            self.assertTrue(r["source"].startswith(
                "claim=pre_contract_numeric_reference_not_certified;"))
        self.assertIn("VALUES", uncer)

    def test_claim_rows_carry_the_full_contract_provenance(self):
        rows = self.load("maros_meszaros.csv")
        claims = {n: r for n, r in rows.items() if r["source"].startswith("exact_enclosure_v1;")}
        for name, r in claims.items():
            parts = dict(item.split("=", 1) for item in r["source"].split(";")[1:])
            self.assertEqual(set(parts), {"proof", "proof_sha256", "input_sha256",
                                          "checker_sha256"})
            expected = ("proofs/liswet/manifest.json" if name.startswith("LISWET")
                        else f"proofs/{name}.proof.json.lzma")
            self.assertEqual(parts["proof"], expected)
            for key in ("proof_sha256", "input_sha256", "checker_sha256"):
                self.assertRegex(parts[key], r"^[0-9a-f]{64}$")
        # LISWET の有理数証跡は MB 級でリポジトリに載せない。manifest が 12 行を
        # 網羅し、期待ハッシュが CSV の stamp と一致すること (検証器と同じ結線)。
        liswet = sorted(n for n in claims if n.startswith("LISWET"))
        self.assertEqual(len(liswet), 12)
        manifest = json.loads((ROOT / "data/baseline_objectives/proofs/liswet/manifest.json")
                              .read_text())["rows"]
        self.assertEqual(sorted(manifest), liswet)
        for name in liswet:
            self.assertEqual(manifest[name]["proof_sha256"],
                             dict(item.split("=", 1) for item in
                                  claims[name]["source"].split(";")[1:])["proof_sha256"])
            self.assertEqual(manifest[name]["reference"], claims[name]["optimal_obj"])
        for name in claims:
            if name not in liswet:
                self.assertTrue((ROOT / "data/baseline_objectives" /
                                 f"proofs/{name}.proof.json.lzma").is_file(), name)

    def test_reissue_flip_side_names_match_the_frozen_run(self):
        rows = self.load("maros_meszaros.csv")
        still = ("QSTAIR", "QBRANDY", "QPCSTAIR", "PRIMALC1", "BOYD2", "QSCSD8", "LISWET7")
        flipped_uncertified = {"CVXQP1_L", "CVXQP2_L", "CVXQP3_L", "LASER",
                               "HUES-MOD", "HUESTIS", "QCAPRI", "QSCFXM3"}
        for name in still:
            self.assertTrue(rows[name]["source"].startswith("exact_enclosure_v1;"), name)
        for name in flipped_uncertified:
            self.assertTrue(rows[name]["source"].endswith("_UNCERTIFIED"), name)

    def test_published_values_are_stable_across_the_reissue(self):
        # 再発行は source 列だけを書き換える。値の 1 文字の漂移も検出する。
        rows = self.load("maros_meszaros.csv")
        pinned = {"AUG2D": "1.687411752897e6", "BOYD2": "2.125676700395e1",
                  "HS51": "0.000000000000e0", "LISWET11": "4.952395713866e1",
                  "LISWET7": "4.988408907861e2", "QPCSTAIR": "6.204387477317e6",
                  "QSCSD8": "9.407635742185e2", "QSTAIR": "7.985452756104e6",
                  "VALUES": "-1.396621144714e0"}
        for name, value in pinned.items():
            self.assertEqual(rows[name]["optimal_obj"], value, name)

    def test_selfbest_csv_is_a_regression_subset_with_full_objective_liswet(self):
        certified = self.load("maros_meszaros.csv")
        selfbest = self.load("maros_meszaros_selfbest.csv")
        self.assertEqual(len(selfbest), 131)
        self.assertTrue(set(selfbest) <= set(certified))
        for name in ("LISWET1", "LISWET7", "LISWET10", "LISWET11", "LISWET12"):
            # main の LISWET 値は c0 抜きだった。selfbest は c0 込みへ変換済み。
            self.assertGreater(float(selfbest[name]["optimal_obj"]), 0.0)
            self.assertIn("full objective", selfbest[name]["source"])
            self.assertTrue(
                math.isclose(
                    float(selfbest[name]["optimal_obj"]),
                    float(certified[name]["optimal_obj"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )


if __name__ == "__main__":
    unittest.main()
