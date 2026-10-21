#!/usr/bin/env python3
"""Maros-Meszaros QPS 問題の baseline CSV 参照値を、検算付きで出力する。

QPS を独立にパースした問題
    min c0 + c'x + 1/2 x'Qx   s.t.  L <= Ax <= U,  l <= x <= u
を HiGHS に直接渡して解き (HiGHS の MPS reader は使わない; to_highs_model 参照)、
その解 (x, y) から主実行可能性と弱双対下界を独立に再計算して検算する。HiGHS の解が
検算を通らない (QP solver の Solve error・時間切れ・dual 精度不足) 問題は Clarabel
(行列スケーリングあり、なしの順) で解き直し、同じ検算にかける。どのソルバの解も検算を通らない問題は非ゼロ終了する。
参照値の正しさはソルバの status でなく検算 (certify) が担保する。

目的定数 c0 は目的行 RHS の符号反転 (c0 = -RHS; Gurobi MPS 仕様・HiGHS・CPLEX と同じ)。
対応範囲: NAME/ROWS/COLUMNS/RHS/RANGES/BOUNDS(UP/LO/FX/FR/MI/PL)/QUADOBJ/ENDATA の
free format (RHS/RANGES/BOUNDS の set 名省略も可) と、free format で読めない場合の
fixed format。OBJSENSE、MARKER (整数変数)、2 本目以降の N 行への非ゼロは ValueError
で拒否する (黙って誤値を出さない)。

検算: active set で KKT を解き直した候補 (polish) も含め、
  - 弱双対: y を行種別に合う符号へ射影し、z = c + Qx - A'y を x の固定点で取った
    Lagrange 双対 d = c0 - 1/2 x'Qx + sum_i h_i(y_i) + sum_j g_j(z_j)
    (h_i(y) = y L_i (y>0) / y U_i (y<0)、g_j 同様) は凸 QP の最適値の下界。z_j が無限側の
    境界を要求する成分 r は 0 へ射影する。残った Lagrangian の x 方向最小値は
    -1/2 x'Qx + r'x + 1/2 r'delta (Q delta = -r) なので、d にその分 (r'x + 1/2 r'delta)
    を足して厳密な下界に戻す (r が range(Q) に無ければ見積りへ退く)。
  - 相対不確かさ |p - d|/(1+|p|) + 双対残差影響 + 主違反影響 <= UNCERTAINTY_TOL、
    相対主実行不能度 <= PRIMAL_INF_TOL、ソルバ報告値と自前再計算の一致。

Usage:
  python3 scripts/highs_baseline_truth.py data/maros_meszaros/LISWET7.QPS ...
  python3 scripts/highs_baseline_truth.py --csv --time-limit 1200 data/maros_meszaros/*.QPS

診断出力 (1 ソルバ試行 1 行): problem,solver,PASS|FAIL,status= reported= p= d= uncertainty= ...
--csv: 検算を通った問題だけ、参照値 p (検算した候補の主目的値) の baseline CSV 行
  `NAME,<mantissa>e<exp>,<solver>_<ver>_<status>_<date>` を stdout に出し、診断行は
  stderr へ回す。
"""
import argparse
import datetime
import math
import os
import sys
import warnings

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, spsolve

# baseline 比較の許容 eps (bench 基準, CLAUDE.md: eps=1e-6)。参照値の誤差は比較側の
# 許容より十分小さくなければならないので、検算許容は eps の 1/10 以下に取る。
BENCH_EPS = 1e-6
# 相対 gap |p-d|/(1+|p|) の上限。参照値の不確かさ上界 = BENCH_EPS/10。
REL_GAP_TOL = BENCH_EPS / 10
# 参照値の相対不確かさ (gap + 双対残差影響 + 主違反影響, uncertainty()) の上限。
UNCERTAINTY_TOL = REL_GAP_TOL
# 相対主実行不能度 (違反量 / (1 + |境界|)) の上限。HiGHS の既定主実行可能性許容 1e-7 と
# 同じ。目的値への影響は別途 pimpact (乗数 × 違反量) として REL_GAP_TOL 予算で検査する。
PRIMAL_INF_TOL = 1e-7
# HiGHS 報告目的値と独立再計算 p(x) の相対差の上限。両者は同じ x の目的値なので浮動小数の
# 丸め (~1e-12) しか差が出ないはずで、超えたら QPS 解釈 (Q, c, c0, 符号) の食い違いを示す。
OBJ_CONSISTENCY_TOL = BENCH_EPS * 1e-3
# baseline CSV の仮数部の小数桁数 (小数点以下 12 桁 = 13 有効桁)。
CSV_MANTISSA_DECIMALS = 12
# active 判定の相対許容 (|値 - 境界| <= ACTIVE_TOL (1 + |境界|))。HiGHS の主実行可能性
# 許容 1e-7 に揃える。誤判定しても polish() の候補が劣るだけで、検算の妥当性は崩れない。
ACTIVE_TOL = 1e-7
# 特異 KKT を LSQR で解くときの停止許容と反復上限 (倍精度の丸めに近い水準まで詰める)。
LSQR_TOL = 1e-15
LSQR_ITER_LIMIT = 200_000
# _range_step の Q delta = -r の解の採否 (残差 / |r|_inf の上限)。r が range(Q) にあるか
# の判定で、超えたら Lagrangian 下界を厳密に取れないとみなす。
RANGE_RESID_TOL = 1e-10
# Clarabel (HiGHS が検算に通る解を出せない問題の代替ソルバ) の停止許容と時間上限。
CLARABEL_TOL = 1e-10
CLARABEL_TIME_LIMIT_SECS = 1800.0
# MPS の無限大表記 (|v| >= 1e20 は無限大; HiGHS kHighsInf 判定と同じ閾値)。
MPS_INFINITY = 1e20

SECTIONS = {"NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "ENDATA"}
ROW_TYPES = {"N", "G", "L", "E"}
BOUND_TYPES_WITH_VALUE = {"UP", "LO", "FX"}
BOUND_TYPES_NO_VALUE = {"FR", "MI", "PL"}
# fixed format のフィールド位置 (0 始まり, [start, end)): 1:2-3, 2:5-12, 3:15-22, 4:25-36,
# 5:40-47, 6:50-61 (MPS 標準のカラム位置)。
FIXED_FIELDS = ((1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61))


def _fixed_tokens(line):
    return [line[a:b].strip() for a, b in FIXED_FIELDS]


def _pairs(tokens, has_set_name):
    rest = tokens[1:] if has_set_name else tokens
    if len(rest) % 2:
        raise ValueError(f"odd name/value list {tokens}")
    return [(rest[k], float(rest[k + 1])) for k in range(0, len(rest), 2)]


def parse_qps(path, fixed=False):
    """独立 QPS パーサ。辞書 (c0, c, A(csr), L, U, l, u, Q(csr, 対称), n, m) を返す。"""
    section = None
    obj = None
    free_rows = set()
    row_index, row_type = {}, []
    cols = {}
    c, a_i, a_j, a_v = {}, [], [], []
    rhs, rng, bnd, qent = {}, {}, [], []
    with open(path, errors="strict") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                tok = line.split()[0]
                if tok not in SECTIONS:
                    raise ValueError(f"unsupported section {tok}")
                section = tok
                continue
            if fixed:
                ft = _fixed_tokens(line)
            t = line.split()
            if section == "ROWS":
                kind, name = (ft[0], ft[1]) if fixed else (t[0], t[1])
                if kind not in ROW_TYPES:
                    raise ValueError(f"row type {kind}")
                if kind == "N":
                    if obj is None:
                        obj = name
                    else:
                        free_rows.add(name)
                else:
                    row_index[name] = len(row_type)
                    row_type.append(kind)
            elif section == "COLUMNS":
                if "'MARKER'" in line:
                    raise ValueError("integer MARKER unsupported")
                if fixed:
                    col = ft[1]
                    pairs = [(ft[k], float(ft[k + 1])) for k in (2, 4) if ft[k]]
                else:
                    col = t[0]
                    pairs = _pairs(t[1:], False)
                j = cols.setdefault(col, len(cols))
                for r, v in pairs:
                    if r == obj:
                        c[j] = c.get(j, 0.0) + v
                    elif r in row_index:
                        a_i.append(row_index[r])
                        a_j.append(j)
                        a_v.append(v)
                    elif r in free_rows:
                        continue
                    else:
                        raise ValueError(f"unknown row {r}")
            elif section in ("RHS", "RANGES"):
                if fixed:
                    pairs = [(ft[k], float(ft[k + 1])) for k in (2, 4) if ft[k]]
                else:
                    pairs = _pairs(t, len(t) % 2 == 1)
                dest = rhs if section == "RHS" else rng
                for r, v in pairs:
                    if r in free_rows:
                        if v != 0.0:
                            raise ValueError(f"nonzero {section} on free row {r}")
                        continue
                    dest[r] = v
            elif section == "BOUNDS":
                if fixed:
                    kind, col = ft[0], ft[2]
                    val = float(ft[3]) if ft[3] else None
                    if kind in BOUND_TYPES_WITH_VALUE:
                        if val is None:
                            raise ValueError(f"bound {kind} without value: {line!r}")
                    elif kind not in BOUND_TYPES_NO_VALUE:
                        raise ValueError(f"bound type {kind}")
                else:
                    kind = t[0]
                    if kind in BOUND_TYPES_WITH_VALUE:
                        col, val = (t[2], float(t[3])) if len(t) == 4 else (t[1], float(t[2]))
                    elif kind in BOUND_TYPES_NO_VALUE:
                        col, val = (t[2] if len(t) == 3 else t[1]), None
                    else:
                        raise ValueError(f"bound type {kind}")
                bnd.append((kind, cols[col], val))
            elif section == "QUADOBJ":
                if fixed:
                    a, b, v = ft[1], ft[2], float(ft[3])
                else:
                    a, b, v = t[0], t[1], float(t[2])
                qent.append((cols[a], cols[b], v))
            else:
                raise ValueError(f"data line in section {section}")
    n, m = len(cols), len(row_type)
    cv = np.zeros(n)
    for j, v in c.items():
        cv[j] = v
    A = sp.csr_matrix((a_v, (a_i, a_j)), shape=(m, n))
    L, U = np.full(m, -math.inf), np.full(m, math.inf)
    for name, i in row_index.items():
        b = rhs.get(name, 0.0)
        kind = row_type[i]
        if kind == "G":
            L[i] = b
        elif kind == "L":
            U[i] = b
        else:
            L[i] = U[i] = b
        if name in rng:
            r = rng[name]
            if kind == "G":
                U[i] = b + abs(r)
            elif kind == "L":
                L[i] = b - abs(r)
            elif r >= 0:
                U[i] = b + r
            else:
                L[i] = b + r
    lo, up = np.zeros(n), np.full(n, math.inf)
    lo_set = np.zeros(n, dtype=bool)
    for kind, j, v in bnd:
        if kind == "UP":
            up[j] = v
            # MPS 慣行: 下限未指定で負の上限なら下限は -inf (HiGHS/CPLEX と同じ)。
            if v < 0 and not lo_set[j]:
                lo[j] = -math.inf
        elif kind == "LO":
            lo[j] = v
            lo_set[j] = True
        elif kind == "FX":
            lo[j] = up[j] = v
            lo_set[j] = True
        elif kind == "FR":
            lo[j], up[j] = -math.inf, math.inf
        elif kind == "MI":
            lo[j] = -math.inf
        elif kind == "PL":
            up[j] = math.inf
    for arr in (L, U, lo, up):
        arr[arr >= MPS_INFINITY] = math.inf
        arr[arr <= -MPS_INFINITY] = -math.inf
    qi = [e[0] for e in qent] + [e[1] for e in qent if e[0] != e[1]]
    qj = [e[1] for e in qent] + [e[0] for e in qent if e[0] != e[1]]
    qv = [e[2] for e in qent] + [e[2] for e in qent if e[0] != e[1]]
    Q = sp.csr_matrix((qv, (qi, qj)), shape=(n, n))
    c0 = (-rhs.get(obj, 0.0) if obj is not None else 0.0) + 0.0
    return dict(c0=c0, c=cv, A=A, L=L, U=U, lo=lo, up=up, Q=Q, n=n, m=m)


def load_qps(path):
    try:
        return parse_qps(path)
    except (ValueError, IndexError, KeyError):
        return parse_qps(path, fixed=True)


def _bound_term(mult, lower, upper):
    """sum h(mult) と、無限側を要求して捨てた成分のマスク。"""
    term = np.zeros_like(mult)
    pos, neg = mult > 0, mult < 0
    lo_ok = np.isfinite(lower)
    up_ok = np.isfinite(upper)
    term[pos & lo_ok] = mult[pos & lo_ok] * lower[pos & lo_ok]
    term[neg & up_ok] = mult[neg & up_ok] * upper[neg & up_ok]
    dropped = (pos & ~lo_ok) | (neg & ~up_ok)
    return float(term.sum()), dropped


def _rel_violation(val, lower, upper):
    with np.errstate(invalid="ignore"):
        below = np.where(np.isfinite(lower), (lower - val) / (1.0 + np.abs(lower)), 0.0)
        above = np.where(np.isfinite(upper), (val - upper) / (1.0 + np.abs(upper)), 0.0)
    v = np.maximum(np.maximum(below, above), 0.0)
    return float(v.max()) if len(v) else 0.0


def _at_bound(val, bound):
    with np.errstate(invalid="ignore"):
        return np.isfinite(bound) & (np.abs(val - bound) <= ACTIVE_TOL * (1.0 + np.abs(bound)))


def polish(prob, x, y):
    """x の active set で KKT 方程式を直接解き直した (x, y) を返す (解けなければ None)。

    HiGHS の QP 解は主側が正確でも row dual に ~1e-6 の stationarity 残差が残り、弱双対
    gap が bench 許容の 1/10 を割れない。active な境界の変数を境界値に固定し、active 行
    (等式行を含む) を等式として
        [Q_FF  -A_RF'] [x_F]   [-(c_F + Q_FB x_B)]
        [A_RF    0   ] [y_R] = [ b_R - A_RB x_B  ]
    を解く。返す (x, y) は候補にすぎず、主実行可能性と下界の妥当性は verify() が検査する。
    """
    A, Q, c = prob["A"].tocsc(), prob["Q"].tocsc(), prob["c"]
    ax = A @ x
    at_lo_row, at_up_row = _at_bound(ax, prob["L"]), _at_bound(ax, prob["U"])
    rows = np.flatnonzero(at_lo_row | at_up_row)
    b_rows = np.where(at_lo_row, prob["L"], prob["U"])[rows]
    at_lo, at_up = _at_bound(x, prob["lo"]), _at_bound(x, prob["up"])
    fixed = at_lo | at_up
    xb = np.where(at_lo, prob["lo"], prob["up"])
    F, B = np.flatnonzero(~fixed), np.flatnonzero(fixed)
    x_new = x.copy()
    x_new[B] = xb[B]
    A_R = A[rows]
    kkt = sp.bmat([[Q[F][:, F], -A_R[:, F].T], [A_R[:, F], None]], format="csc")
    rhs = np.concatenate([-(c[F] + Q[F][:, B] @ x_new[B]), b_rows - A_R[:, B] @ x_new[B]])
    if kkt.shape[0] == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = None
        # active 行が自由変数より多い KKT は必ず特異。SuperLU (spsolve) は特異行列で
        # segfault することがある (STADAT1 で実測) ので、その場合は直接 LSQR へ回す。
        if len(rows) <= len(F):
            try:
                sol = spsolve(kkt, rhs)
            except RuntimeError:
                sol = None
        if sol is None or not np.all(np.isfinite(sol)):
            # 縮退 (active 制約が自由次元を超える) で KKT が特異: 最小ノルム解を取る。
            x0 = np.concatenate([x[F], y[rows]])
            sol = lsqr(kkt, rhs, atol=LSQR_TOL, btol=LSQR_TOL, iter_lim=LSQR_ITER_LIMIT,
                       x0=x0)[0]
    if not np.all(np.isfinite(sol)):
        return None
    x_new[F] = sol[: len(F)]
    y_new = np.zeros_like(y)
    y_new[rows] = sol[len(F):]
    return x_new, y_new


def _abs_violation(val, lower, upper):
    with np.errstate(invalid="ignore"):
        below = np.where(np.isfinite(lower), lower - val, 0.0)
        above = np.where(np.isfinite(upper), val - upper, 0.0)
    return np.maximum(np.maximum(below, above), 0.0)


def verify(prob, x, y):
    """独立再計算 (docstring 冒頭の定義)。

    返り値 (p, d, pinf, resid, pimpact):
      p       主目的値 c0 + c'x + 1/2 x'Qx
      d       Lagrange 双対下界
      pinf    相対主実行不能度 max 違反 / (1 + |境界|)
      resid   射影で捨てた双対残差の目的値影響 sum |z_j x_j| / (1 + |p|)
      pimpact 主違反の目的値影響 (sum |y_i| 行違反_i + sum |z_j| 列違反_j) / (1 + |p|)。
              x が境界を僅かに破るとき p(x) は真の最適値の上界でなくなるが、その差は
              一次で乗数 × 違反量 (感度) で抑えられる。
    """
    A, Q, c, c0 = prob["A"], prob["Q"], prob["c"], prob["c0"]
    ax = A @ x
    qx = Q @ x
    pinf = max(_rel_violation(ax, prob["L"], prob["U"]),
               _rel_violation(x, prob["lo"], prob["up"]))
    p = float(c0 + c @ x + 0.5 * x @ qx)
    yp = y.copy()
    yp[(yp > 0) & ~np.isfinite(prob["L"])] = 0.0
    yp[(yp < 0) & ~np.isfinite(prob["U"])] = 0.0
    z = c + qx - A.T @ yp
    row_term, _ = _bound_term(yp, prob["L"], prob["U"])
    col_term, dropped = _bound_term(z, prob["lo"], prob["up"])
    d = float(c0 - 0.5 * x @ qx + row_term + col_term)
    scale = 1.0 + abs(p)
    resid = 0.0
    if dropped.any():
        r = np.where(dropped, z, 0.0)
        step = _range_step(Q, r)
        if step is None:
            resid = float(np.abs(z[dropped] * x[dropped]).sum()) / scale
        else:
            d += float(r @ x) + 0.5 * float(r @ step)
    zk = np.where(dropped, 0.0, z)
    pimpact = float(np.abs(yp) @ _abs_violation(ax, prob["L"], prob["U"])
                    + np.abs(zk) @ _abs_violation(x, prob["lo"], prob["up"])) / scale
    return p, d, pinf, resid, pimpact


def _range_step(Q, r):
    """Q delta = -r を解いた delta (r が range(Q) に無ければ None)。

    捨てた双対残差 r を残したままの Lagrangian の x 方向の部分は、x = xbar + delta として
    1/2 x'Qx + (r - Q xbar)'x = -1/2 xbar'Q xbar + r'xbar + r'delta + 1/2 delta'Q delta。
    delta の最小値は 1/2 r'delta (Q delta = -r)。呼び出し側は d に r'xbar + 1/2 r'delta
    を足す (-1/2 xbar'Q xbar は d の本体に含まれる)。r が range(Q) に無ければ下界は -inf になるので None を返し、呼び出し側は
    sum |z_j x_j| の見積りへ退く。
    """
    # Q は特異でありうる (SuperLU は特異行列で segfault しうる) ので LSQR で解く。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = lsqr(Q, -r, atol=LSQR_TOL, btol=LSQR_TOL, iter_lim=LSQR_ITER_LIMIT)[0]
    rnorm = float(np.abs(r).max())
    if not np.all(np.isfinite(step)) or np.abs(Q @ step + r).max() > RANGE_RESID_TOL * rnorm:
        return None
    return step


def uncertainty(result):
    """参照値 p の相対不確かさ上界の見積り: gap + 双対残差影響 + 主違反影響。"""
    p, d, _, resid, pimpact = result
    return abs(p - d) / (1.0 + abs(p)) + resid + pimpact


def best_certificate(prob, x, y):
    """HiGHS の (x, y) と polish() の (x, y) のうち、相対主実行不能度が PRIMAL_INF_TOL
    以下で uncertainty() 最小の候補の verify 結果。該当候補が無ければ HiGHS 側を返す。"""
    cands = [(x, y)]
    polished = polish(prob, x, y)
    if polished is not None:
        cands.append(polished)
    results = [verify(prob, cx, cy) for cx, cy in cands]
    feasible = [r for r in results if r[2] <= PRIMAL_INF_TOL]
    if not feasible:
        return results[0]
    return min(feasible, key=uncertainty)


def to_highs_model(prob):
    """独立パース結果を HighsModel にする。

    HiGHS の MPS reader には通さない: set 名が行名と同じ字面の free format 行
    (DPKLO1 の RHS `1  2  0.0` = set "1", 行 "2") を HiGHS は set 名省略と読み、
    別の問題を解いてしまう。
    """
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = prob["n"], prob["m"]
    lp.offset_ = prob["c0"]
    lp.col_cost_ = prob["c"]
    lp.col_lower_, lp.col_upper_ = _highs_inf(prob["lo"]), _highs_inf(prob["up"])
    lp.row_lower_, lp.row_upper_ = _highs_inf(prob["L"]), _highs_inf(prob["U"])
    a = prob["A"].tocsc()
    a.sort_indices()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = prob["n"], prob["m"]
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
    model = highspy.HighsModel()
    model.lp_ = lp
    q = sp.tril(prob["Q"]).tocsc()
    q.sort_indices()
    if q.nnz:
        hess = highspy.HighsHessian()
        hess.dim_ = prob["n"]
        hess.format_ = highspy.HessianFormat.kTriangular
        hess.start_, hess.index_, hess.value_ = q.indptr, q.indices, q.data
        model.hessian_ = hess
    return model


def _highs_inf(v):
    return np.clip(v, -highspy.kHighsInf, highspy.kHighsInf)


def format_csv_value(v):
    """baseline CSV の値表記 (例 -2.489287597915e3)。"""
    mant, exp = f"{v:.{CSV_MANTISSA_DECIMALS}e}".split("e")
    return f"{mant}e{int(exp)}"


def solve_highs(prob, time_limit):
    """(status, x, y, 報告目的値)。"""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit is not None:
        h.setOptionValue("time_limit", time_limit)
    h.passModel(to_highs_model(prob))
    h.run()
    sol = h.getSolution()
    return (h.modelStatusToString(h.getModelStatus()), np.array(sol.col_value),
            np.array(sol.row_dual), h.getInfo().objective_function_value)


def solve_clarabel(prob, time_limit, equilibrate=True):
    """(status, x, y, 報告目的値)。y は verify() と同じ符号規約 (c + Qx - A'y - z = 0)。

    Clarabel 形式 Mx + s = b, s in K へ、等式行 (L=U)、上限側、下限側 (符号反転) の順に
    積む。Clarabel の双対 w (Qx + c + M'w = 0) から y_i = w_lo_i - w_up_i - w_eq_i。
    equilibrate=False は Clarabel の行列スケーリングを切る (BOYD1 は既定だと AlmostSolved で
    相対 gap 5e-6 に止まり、切ると Solved で 2.5e-11 まで詰まる)。
    """
    import clarabel

    A = prob["A"].tocsr()
    L, U, lo, up = prob["L"], prob["U"], prob["lo"], prob["up"]
    n = prob["n"]
    eq = np.flatnonzero(L == U)
    ub = np.flatnonzero((L != U) & np.isfinite(U))
    lb = np.flatnonzero((L != U) & np.isfinite(L))
    I = sp.identity(n, format="csr")
    xeq = np.flatnonzero(lo == up)
    xub = np.flatnonzero((lo != up) & np.isfinite(up))
    xlb = np.flatnonzero((lo != up) & np.isfinite(lo))
    M = sp.vstack([A[eq], I[xeq], A[ub], -A[lb], I[xub], -I[xlb]]).tocsc()
    b = np.concatenate([U[eq], up[xeq], U[ub], -L[lb], up[xub], -lo[xlb]])
    n_zero = len(eq) + len(xeq)
    cones = [clarabel.ZeroConeT(n_zero), clarabel.NonnegativeConeT(len(b) - n_zero)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.equilibrate_enable = equilibrate
    settings.time_limit = time_limit if time_limit is not None else CLARABEL_TIME_LIMIT_SECS
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = CLARABEL_TOL
    sol = clarabel.DefaultSolver(sp.triu(prob["Q"]).tocsc(), prob["c"], M, b, cones,
                                 settings).solve()
    w = np.array(sol.z)
    y = np.zeros(prob["m"])
    k = 0
    for rows, sign in ((eq, -1.0), (xeq, 0.0), (ub, -1.0), (lb, 1.0)):
        if sign != 0.0:
            np.add.at(y, rows, sign * w[k:k + len(rows)])
        k += len(rows)
    return str(sol.status), np.array(sol.x), y, float(sol.obj_val) + prob["c0"]


SOLVERS = (
    ("highs", lambda: highspy.Highs().version(), solve_highs),
    ("clarabel", lambda: __import__("clarabel").__version__, solve_clarabel),
    ("clarabel-noequil", lambda: __import__("clarabel").__version__,
     lambda prob, time_limit: solve_clarabel(prob, time_limit, equilibrate=False)),
)


def certify(prob, solved):
    """solve_* の結果を検算し (ok, 参照値 p, 診断文字列) を返す。"""
    status, x, y, reported = solved
    if len(x) != prob["n"] or len(y) != prob["m"] or not np.all(np.isfinite(x)):
        return False, math.nan, f"status={status} no solution"
    # 報告値と自前再計算の食い違いは QPS 解釈 (Q, c, c0, 符号) の食い違いを示す。
    p_raw = float(prob["c0"] + prob["c"] @ x + 0.5 * x @ (prob["Q"] @ x))
    objdiff = abs(p_raw - reported) / (1.0 + abs(p_raw))
    result = best_certificate(prob, x, y)
    p, d, pinf, resid, pimpact = result
    unc = uncertainty(result)
    gap = abs(p - d) / (1.0 + abs(p))
    # NaN は比較が常に False になるので、合格条件を肯定形で書き否定を取る。
    ok = unc <= UNCERTAINTY_TOL and pinf <= PRIMAL_INF_TOL and objdiff <= OBJ_CONSISTENCY_TOL
    msg = (f"status={status} reported={reported!r} p={p!r} d={d!r} uncertainty={unc:.3e} "
           f"relgap={gap:.3e} dual_resid={resid:.3e} pimpact={pimpact:.3e} "
           f"verify_pinf={pinf:.3e} reported_vs_p={objdiff:.3e} c0={prob['c0']!r}")
    return ok, p, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--time-limit", type=float, default=None, help="各ソルバの時間上限 [s]")
    ap.add_argument("--csv", action="store_true", help="baseline CSV 行を stdout へ出す")
    args = ap.parse_args()
    date = datetime.date.today().strftime("%Y%m%d")
    diag = sys.stderr if args.csv else sys.stdout
    bad = 0
    for path in args.paths:
        name = os.path.splitext(os.path.basename(path))[0]
        prob = load_qps(path)
        accepted = None
        for solver, version, solve in SOLVERS:
            solved = solve(prob, args.time_limit)
            ok, p, msg = certify(prob, solved)
            print(f"{name},{solver},{'PASS' if ok else 'FAIL'},{msg}", file=diag, flush=True)
            if ok:
                status = solved[0].replace(" ", "")
                accepted = (p, f"{solver}_{version()}_{status}_{date}")
                break
        if accepted is None:
            bad += 1
            print(f"FAIL {name}: no solver passed the certificate", file=sys.stderr, flush=True)
        elif args.csv:
            print(f"{name},{format_csv_value(accepted[0])},{accepted[1]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
