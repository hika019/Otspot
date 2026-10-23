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
  - 弱双対 (dual_bound): 符号を射影した y と任意の線形化点 xh で
        f* >= c0 + sum_i h_i(y_i) - 1/2 xh'Q xh + sum_j min_{lo_j<=x_j<=up_j} g_j x_j,
        g = c + Q xh - A'y
    (h_i(y) = y L_i (y>0) / y U_i (y<0); Q の凸性 1/2 x'Qx >= 1/2 xh'Q xh + (Q xh)'(x-xh))。
    g を丸め誤差上界つきの区間で、各項と和を -inf 方向へ丸めて評価するので、得る値は
    浮動小数の丸めを含めて最適値の下界になる。g_j が無限側の境界を要求する成分があれば
    下界は -inf で、_repair が xh と y を動かして消す (収束しなければ線形化 LP の双対から
    やり直す; _linearized_dual)。消せなければ検算失敗 (その問題は UNCERTIFIED とする)。
    自由変数 (FR) を含む問題で _repair が -inf のままなら、対角 Q は有理数で縮約費用を
    厳密に 0 にする FR 証明 (_fr_certificate)、非対角 Q は厳密 KKT 証明
    (_kkt_certificate) を追加で試す。どちらも下界の妥当性は同じ弱双対式で、厳密な
    有理数評価により浮動小数の区間では作れない [0, 0] の縮約費用を与える。
  - 相対不確かさ |p - d|/(1+|p|) + 丸め影響 + 主違反影響 <= UNCERTAINTY_TOL、
    相対主実行不能度 <= PRIMAL_INF_TOL、ソルバ報告値と自前再計算の一致。
  - Q の半正定値性: load_qps が非ゼロ非対角グラフのブロックごとに確認する。負固有値が
    丸め許容を超える場合と、ブロックが大きすぎて判定できない場合は検算不能
    (d = -inf) とし、その問題は UNCERTIFIED とする (非凸 QP は対象外)。

Usage:
  python3 scripts/highs_baseline_truth.py data/maros_meszaros/LISWET7.QPS ...
  python3 scripts/highs_baseline_truth.py --csv --time-limit 1200 data/maros_meszaros/*.QPS

診断出力 (1 ソルバ試行 1 行): problem,solver,PASS|FAIL,status= reported= p= d= uncertainty= ...
--csv: 検算を通った問題だけ、参照値 p (検算した候補の主目的値) の baseline CSV 行
  `NAME,<mantissa>e<exp>,<solver>_<ver>_<status>_<date>` を stdout に出し、診断行は
  stderr へ回す。
"""
import argparse
import collections
import datetime
import heapq
import math
import os
import sys
import time
import warnings
from fractions import Fraction

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import linprog
from scipy.sparse import csgraph
from scipy.sparse.linalg import lsqr, spsolve

# baseline 比較の許容 eps (bench 基準, CLAUDE.md: eps=1e-6)。参照値の誤差は比較側の
# 許容より十分小さくなければならないので、検算許容は eps の 1/10 以下に取る。
BENCH_EPS = 1e-6
# 相対 gap |p-d|/(1+|p|) の上限。参照値の不確かさ上界 = BENCH_EPS/10。
REL_GAP_TOL = BENCH_EPS / 10
# 参照値の相対不確かさ (gap + 丸め影響 + 主違反影響, uncertainty()) の上限。
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
# _repair() の LSQR は「候補を 1 つ作る」ためだけに使う。200k 反復は大規模問題で
# 数時間かかることがあるため、反復数と壁時計時間の両方に上限を置く。上限に達した
# 候補は検算が通らない可能性があるが、下界の妥当性は変わらない (失敗側に倒れる)。
REPAIR_LSQR_ITER_LIMIT = 20_000
REPAIR_TIME_BUDGET = 60.0
# Q の PSD 判定。丸めで生じる微小な負固有値を許す相対許容 (行列の 1-ノルム基準)。
# これを超える負固有値がある Q は非凸として検算しない。ブロックが大きすぎて判定
# できない場合も検算しない (fail-safe)。
PSD_REL_TOL = 1e-10
PSD_DENSE_MAX = 8_000
# IEEE 754 binary64 の単位丸め u = 2^-53 (丸め誤差上界 gamma_k の基準)。
UNIT_ROUNDOFF = np.finfo(float).eps / 2
# 積が subnormal へ下位桁あふれしたときの 1 演算あたりの絶対誤差上界 (subnormal の間隔)。
UNDERFLOW_ABS = np.finfo(float).smallest_subnormal
# 丸め誤差上界の安全係数。上界そのものを浮動小数で計算する相対誤差 (<= gamma_k << 1) を
# 吸収するのに 1 + gamma_k で足りるところを、桁の余裕を見て 2 とする。
ERROR_BOUND_SAFETY = 2.0
# _repair() の反復上限。各反復で無限側を要求する成分が修正対象に加わり、対象が基底成分を
# 覆うまで反復が続く。収束する問題は数回で収束し、32 回で収束しないものは循環しているので
# _linearized_dual() からやり直す。
REPAIR_ITERS = 32
# 片側無限変数の修正目標 tau = REPAIR_MARGIN * (修正対象の縮約費用区間の最大半幅)。修正後の
# 区間幅の変動と LSQR の残差 (成分ごとでなく系全体の大きさで決まる) を超えて、区間全体を
# 有限側へ置く余裕。有限側の境界が 0 の成分では下界を損なわない。
REPAIR_MARGIN = 4.0
# _repair() で基底成分 (境界に無い変数) とみなす縮約費用の相対上限 |g_j| / (1 + mag_j)。
# ソルバの双対許容 (HiGHS 1e-7、Clarabel 1e-10) を上回り、非基底成分の縮約費用
# (問題データの桁) を下回る値。誤判定しても下界の妥当性は崩れず、緩さが変わるだけ。
REPAIR_PIN_TOL = 1e-6
# _linearized_dual() の HiGHS の主・双対実行可能性許容 (HiGHS が受け付ける最小値)。残る
# 双対残差は後続の _repair() が丸め誤差の水準まで消す。
LINEARIZED_LP_TOL = 1e-10
# _linearized_dual() の LP の時間上限。候補を作るためだけの LP なので、大規模問題で
# 際限なく待たず、時間切れなら候補なし (UNCERTIFIED) に倒す。
LINEARIZED_LP_TIME_LIMIT = 60.0
# 厳密有理数修復の初期候補を作る双対 QP (Clarabel) の停止許容。候補の妥当性は
# 浮動小数の双対 QP の精度ではなく、最後の有理数評価 (_exact_d) が担保する。
DUAL_QP_TOL = 1e-11
# margin LP の境界近傍幅と行乗数の探索幅。零接触集合 (縮約費用が厳密に 0 の列) を
# 同定するためだけに使い、得られた候補は厳密有理数で再検査する。
MARGIN_LP_DELTA = 1e-3
MARGIN_LP_DY = 1e-3
# 厳密有理数修復を試す問題の上限。有理数 Gauss-Jordan の作業量は非ゼロ数とピボット数に
# 比例して増えるため、これを超える問題は候補を作らず UNCERTIFIED のままにする
# (証明できないものを時間切れで曖昧にしない fail-safe)。
EXACT_MAX_COLS = 10_000
EXACT_MAX_ROWS = 20_000
# 厳密消去の作業量上限 (メモリ・時間の暴走を防ぐ fail-safe)。超えた候補は捨てる。
EXACT_MAX_NNZ = 4_000_000
EXACT_MAX_OPS = 200_000_000
EXACT_MAX_PIVOTS = 300_000
EXACT_TIME_BUDGET = 120.0
# Clarabel (HiGHS が検算に通る解を出せない問題の代替ソルバ) の停止許容と時間上限。
CLARABEL_TOL = 1e-10
CLARABEL_TIME_LIMIT_SECS = 1800.0
# MPS の無限大表記 (|v| >= 1e20 は無限大; HiGHS kHighsInf 判定と同じ閾値)。
MPS_INFINITY = 1e20
# 自由変数の縮約費用を厳密に 0 にする補正 (Q 対角 0 の自由列) の疎消去上限。
# これを超える系は有理数の桁数が爆発するため証明を諦める (下界 -inf のまま)。
FR_MAX_NNZ = 3_000_000
FR_MAX_OPS = 50_000_000
# 非対角 Q の厳密 KKT 証明は密な有理数消去なので、未知数と行列要素数に上限を置く。
# 上限内でも係数成長で遅い場合は呼び側の時間予算で打ち切る (証明できない側に倒れる)。
KKT_MAX_UNKNOWNS = 300
KKT_MAX_DENSE = 200_000
# 非対角 Q の厳密 PSD 判定 (有理数 LDL) を行う最大次元。これを超える非対角 Q は
# 厳密な凸性を証明できないので KKT 証明を諦める (対角 Q は対角成分だけで判定する)。
KKT_EXACT_PSD_MAX = 64

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
        prob = parse_qps(path)
    except (ValueError, IndexError, KeyError):
        prob = parse_qps(path, fixed=True)
    prob["psd"] = _psd_ok(prob["Q"])
    return prob


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


def _gamma(k):
    """k 回の浮動小数演算の相対誤差上界 gamma_k = k u / (1 - k u) (Higham 2002, §3.1)。"""
    ku = k * UNIT_ROUNDOFF
    return ku / (1.0 - ku)


def _pattern(M):
    """M の非ゼロ配置を 1 にした行列。"""
    P = M.tocsr(copy=True)
    P.data = (P.data != 0).astype(float)
    return P


def _max_col_nnz(M):
    return int(np.diff(M.tocsc().indptr).max(initial=0))


def _safe_fsum(values):
    """math.fsum の ±inf 混在・overflow を nan に落とす (呼び側で -inf 扱い)。"""
    try:
        return math.fsum(values)
    except (ValueError, OverflowError):
        return math.nan


def _psd_ok(Q):
    """Q が半正定値か。True / False / None (判定不能) を返す。

    Q の非ゼロ非対角グラフの連結成分 (ブロック対角構造) ごとに固有値を調べる。
    ブロックが PSD_DENSE_MAX 以下のときだけ密固有値計算し、それより大きいブロックは
    「判定不能」として None を返す (証明できない Q は検算しない)。対角行列は対角成分
    だけで判定する。
    """
    n = Q.shape[0]
    if n == 0:
        return True
    diag = np.asarray(Q.diagonal()).ravel()
    if not np.all(np.isfinite(diag)):
        return False
    A = Q.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()
    if A.nnz == 0:
        scale = max(1.0, float(np.abs(diag).max()))
        return bool(diag.min() >= -PSD_REL_TOL * scale)
    _, labels = csgraph.connected_components(A, directed=False)
    unknown = False
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 1:
            d = float(diag[idx[0]])
            scale = max(1.0, abs(d))
            if d < -PSD_REL_TOL * scale:
                return False
            continue
        if len(idx) > PSD_DENSE_MAX:
            unknown = True
            continue
        sub = Q[idx][:, idx]
        scale = max(1.0, float(np.abs(sub).sum(axis=1).max()))
        try:
            # 最小固有値だけを求める。8000 列の CVXQP_L ブロックでも
            # 512 MiB 程度の作業領域で済み、全固有値を求めるより速い。
            w = eigh(sub.toarray(), subset_by_index=[0, 0],
                     eigvals_only=True, driver="evr")
        except (np.linalg.LinAlgError, ValueError):
            unknown = True
            continue
        if not np.all(np.isfinite(w)):
            unknown = True
            continue
        if float(w[0]) < -PSD_REL_TOL * scale:
            return False
    return None if unknown else True


def _down(v):
    return np.nextafter(v, -math.inf)


def _up(v):
    return np.nextafter(v, math.inf)


def _project_dual(prob, y):
    """行乗数を行種別に合う符号へ射影する (y_i > 0 は L_i 有限、y_i < 0 は U_i 有限を要求)。"""
    yp = y.copy()
    yp[(yp > 0) & ~np.isfinite(prob["L"])] = 0.0
    yp[(yp < 0) & ~np.isfinite(prob["U"])] = 0.0
    return yp


def _quad_form(Q, x):
    """(fl(x'Qx), 誤差上界)。|fl(x'Qx) - x'Qx| <= 誤差上界。

    非有限・overflow では下界を諦めて (-inf, inf) を返す (呼び側で d=-inf)。
    """
    with np.errstate(over="ignore", invalid="ignore"):
        q = Q @ x
        k = _max_col_nnz(Q)
        err_q = _gamma(k) * (abs(Q) @ np.abs(x)) + k * UNDERFLOW_ABS
        prod = x * q
        abs_prod = np.abs(prod)
        try:
            val = math.fsum(prod)
            err = (math.fsum(np.abs(x) * err_q) + UNIT_ROUNDOFF * math.fsum(abs_prod)
                   + len(x) * UNDERFLOW_ABS + UNIT_ROUNDOFF * abs(val))
        except (ValueError, OverflowError):
            return -math.inf, math.inf
    if not math.isfinite(val):
        return val, math.inf
    return val, ERROR_BOUND_SAFETY * err


def _reduced_cost_interval(prob, xh, y):
    """g = c + Q xh - A'y を真値で含む区間 [gl, gu] (浮動小数計算の誤差上界で外側へ広げる)。

    返り値 (gl, gu, mag): mag = |c| + |Q||xh| + |A'||y| (g の各成分の大きさ)。
    """
    A, Q, c = prob["A"], prob["Q"], prob["c"]
    g = c + Q @ xh - A.T @ y
    mag = np.abs(c) + abs(Q) @ np.abs(xh) + abs(A).T @ np.abs(y)
    k = _max_col_nnz(Q) + _max_col_nnz(A) + 2
    # 下位桁あふれしうるのは両因子が非ゼロの積だけ。その数が 0 で mag も 0 なら g は厳密に 0。
    nonzero_products = ((c != 0).astype(float) + _pattern(Q) @ (xh != 0).astype(float)
                        + _pattern(A).T @ (y != 0).astype(float))
    err = ERROR_BOUND_SAFETY * (_gamma(k) * mag + nonzero_products * UNDERFLOW_ABS)
    exact = err == 0
    return np.where(exact, g, _down(g - err)), np.where(exact, g, _up(g + err)), mag


def _classify(prob, xh, y):
    """縮約費用区間と、成分ごとの box 最小値の下界。

    返り値 (gl, gu, col, unbounded):
      col       sum_j min { g x : g in [gl_j, gu_j], lo_j <= x <= up_j } の下界 (unbounded 成分を除く)
      unbounded 区間が無限側の境界を要求する成分 (その成分の最小値は -inf)

    区間が無限側の境界を跨ぐ成分は、g の符号を丸め誤差の内で確定できないので下界を
    作れない (conceded のような「0 扱い」はしない)。g が厳密に 0 と分かる成分だけは
    自由変数・片側無限変数でも項が 0 になり、そのまま扱える。
    """
    gl, gu, _ = _reduced_cost_interval(prob, xh, y)
    lo, up = prob["lo"], prob["up"]
    lo_ok, up_ok = np.isfinite(lo), np.isfinite(up)
    corner = np.full_like(gl, math.inf)
    with np.errstate(invalid="ignore"):
        for ok, bound in ((lo_ok, lo), (up_ok, up)):
            for gv in (gl, gu):
                corner = np.where(ok, np.minimum(corner, _down(gv * bound)), corner)
    # 上限なしなら g < 0 で -inf、下限なしなら g > 0 で -inf。区間が非有限 (オーバーフロー)
    # のときは符号を確定できないので、同じく検算不能として扱う。
    unbounded = (~lo_ok & (gu > 0)) | (~up_ok & (gl < 0))
    unbounded |= ~np.isfinite(gl) | ~np.isfinite(gu)
    # 両側無限で unbounded でない成分だけがここに残る (区間が厳密に [0, 0])。
    corner = np.where(~lo_ok & ~up_ok, 0.0, corner)
    col = _safe_fsum(corner[~unbounded])
    if not math.isfinite(col):
        col = -math.inf
    else:
        col = _down(col)
    return gl, gu, col, unbounded


def _repair(prob, xh, y):
    """無限側を要求する縮約費用成分を、xh と y の修正で有限側 (自由変数は 0) へ寄せる。

    下界 dual_bound(xh, y) は任意の (xh, y) で成り立つので、修正は下界の妥当性に影響せず、
    下界が有限になるかと緩さだけを左右する。修正対象 P は、相補性から g_j = 0 であるべき
    成分 (自由変数と、xh_j が境界に無く |g_j| <= REPAIR_PIN_TOL (1 + mag_j) の基底成分) と、
    無限側を要求した成分の累積。P の外の成分は g を保ったまま境界で抑えられる。基底成分を
    固定しないと、y の修正で g_j が膨らみ下界が g_j (境界 - xh_j) だけ緩む。P の g を目標
    (片側無限は有限側へ tau、それ以外は 0) へ合わせる最小ノルム解を、P 成分の xh (Q_PP s) と
    行乗数 y で取る (LSQR)。片側有限行の乗数が逆符号へ動いたら 0 へ射影し、以後その行は
    動かさない。一度入れた成分は P から外さない (外すと次の修正で相補的な成分の符号が倒れて
    循環する)。
    """
    A, Q, lo, up = prob["A"], prob["Q"], prob["lo"], prob["up"]
    both = np.isfinite(prob["L"]) & np.isfinite(prob["U"])
    frozen = np.zeros(prob["m"], dtype=bool)
    lo_inf, up_inf = ~np.isfinite(lo), ~np.isfinite(up)
    gl, gu, mag = _reduced_cost_interval(prob, xh, y)
    small = np.abs(0.5 * (gl + gu)) <= REPAIR_PIN_TOL * (1.0 + mag)
    pinned = (lo_inf & up_inf) | (small & ~_at_bound(xh, lo) & ~_at_bound(xh, up))
    started = time.monotonic()
    for it in range(REPAIR_ITERS):
        if time.monotonic() - started > REPAIR_TIME_BUDGET:
            break
        gl, gu, _, unbounded = _classify(prob, xh, y)
        if not unbounded.any():
            return xh, y, it
        pinned |= unbounded
        P = np.flatnonzero(pinned)
        g = 0.5 * (gl[P] + gu[P])
        tau = REPAIR_MARGIN * float(0.5 * (gu[P] - gl[P]).max())
        one_sided = lo_inf[P] != up_inf[P]
        target = np.where(one_sided, np.where(lo_inf[P], -tau, tau), 0.0)
        # 両側無限の行の乗数も自由に動かせる (符号制約なし)。片側有限行だけが射影で凍る。
        R = np.flatnonzero(~frozen)
        M = sp.hstack([Q[P][:, P], -A[R][:, P].T]).tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # LSQR の収束は行数程度で頭打ちになる。小さい系で 20k 回も回さない。
            iter_cap = min(REPAIR_LSQR_ITER_LIMIT, max(1_000, 4 * M.shape[0]))
            sol = lsqr(M, target - g, atol=LSQR_TOL, btol=LSQR_TOL,
                       iter_lim=iter_cap)[0]
        if not np.all(np.isfinite(sol)):
            break
        xh = xh.copy()
        xh[P] += sol[: len(P)]
        y = y.copy()
        y[R] += sol[len(P):]
        projected = _project_dual(prob, y)
        frozen |= ~both & (projected != y)
        y = projected
    return xh, y, REPAIR_ITERS


def _linearized_dual(prob, xh, y_ref):
    """xh で目的を線形化した LP の行双対 (解けなければ None)。

    Q の凸性から f(x) >= f(xh) + (c + Q xh)'(x - xh) なので、線形化 LP  min (c + Q xh)'x
    (元の制約) の双対は dual_bound() にそのまま渡せる下界の y で、xh = x* なら下界は f* に届く。
    基底の片側無限成分は最適双対で g_j = 0 となり、丸め誤差の区間が 0 を跨いで -inf になる
    ので、その成分の費用を有限側へ tau_j (区間半幅と LP の双対許容の REPAIR_MARGIN 倍) だけ
    ずらした LP を解く (Neumaier-Shcherbina の摂動): 双対は g_j >= tau_j を満たし、
    dual_bound() は元の c で評価するので下界の妥当性は変わらない。半幅は y_ref で見積もる。
    費用 0 の後退方向 (A d = 0, d >= 0 の組) があるとずらした LP は非有界になる (その方向の
    成分は g'd = 0 から g_j >= tau_j を同時に満たせない) ので、ずらさない LP へ退く。
    _repair() が縮退した双対で収束しないときに使う。
    """
    lo_inf, up_inf = ~np.isfinite(prob["lo"]), ~np.isfinite(prob["up"])
    gl, gu, _ = _reduced_cost_interval(prob, xh, y_ref)
    tau = REPAIR_MARGIN * (0.5 * (gu - gl) + LINEARIZED_LP_TOL)
    shift = np.where(lo_inf == up_inf, 0.0, np.where(lo_inf, tau, -tau))
    lin_c = prob["c"] + prob["Q"] @ xh
    for c_lin in (lin_c + shift, lin_c):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", LINEARIZED_LP_TIME_LIMIT)
        h.setOptionValue("primal_feasibility_tolerance", LINEARIZED_LP_TOL)
        h.setOptionValue("dual_feasibility_tolerance", LINEARIZED_LP_TOL)
        h.passModel(to_highs_model(dict(prob, c=c_lin, Q=sp.csr_matrix(prob["Q"].shape))))
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            y = _project_dual(prob, np.array(h.getSolution().row_dual))
            # ずらした LP の双対が有限な下界を与えないなら、ずらさない LP も試す。
            if math.isfinite(dual_bound(prob, xh, y)):
                return y
    return None


def _fiv(fr):
    """Fraction を、その値を含む float 区間 [lo, hi] へ丸める。"""
    try:
        f = float(fr)
    except OverflowError:
        return -math.inf, math.inf
    if math.isinf(f) or math.isnan(f):
        return f, f
    lo = f if Fraction(f) <= fr else math.nextafter(f, -math.inf)
    hi = f if Fraction(f) >= fr else math.nextafter(f, math.inf)
    return lo, hi


def _imul(a, b):
    """区間積。NaN は呼び側で検算不能として扱う。"""
    if any(math.isnan(v) for v in (*a, *b)):
        return math.nan, math.nan
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    if any(math.isnan(v) for v in p):
        return math.nan, math.nan
    return min(_down(v) for v in p), max(_up(v) for v in p)


def _isum(ivs):
    """区間和。fsum の結果を外側へ丸める。"""
    try:
        lo = math.fsum(v[0] for v in ivs)
        hi = math.fsum(v[1] for v in ivs)
    except (ValueError, OverflowError):
        return -math.inf, math.inf
    if math.isnan(lo) or math.isnan(hi):
        return math.nan, math.nan
    return _down(lo), _up(hi)


def _square(iv):
    """区間の二乗。"""
    lo, hi = iv
    if math.isnan(lo) or math.isnan(hi):
        return math.nan, math.nan
    if lo >= 0.0:
        return _down(lo * lo), _up(hi * hi)
    if hi <= 0.0:
        return _down(hi * hi), _up(lo * lo)
    return 0.0, _up(max(lo * lo, hi * hi))


def _sparse_correction(prob, y0):
    """Q 対角 0 の自由列について A_F0' y = c_F0 を厳密に解き、y を補正する。

    y0 からの差分を疎な有理数 Gauss-Jordan で求める。返す y は厳密に
    A_F0' y = c_F0 を満たす。係数成長が上限を超えたら (None, ops) を返す。
    """
    A = prob["A"].tocsc(); c = prob["c"]
    q = np.asarray(prob["Q"].diagonal()).ravel()
    lo, up = prob["lo"], prob["up"]
    free0 = np.flatnonzero(~np.isfinite(lo) & ~np.isfinite(up) & (q == 0))
    if len(free0) == 0:
        return [Fraction(float(v)) for v in y0], 0
    frac = lambda v: Fraction(float(v))
    y0f = [frac(v) for v in y0]
    eqs = []
    for j in free0:
        a = A[:, j].tocoo()
        d = {int(i): frac(v) for i, v in zip(a.row, a.data)}
        rhs = frac(c[j]) - sum(v * y0f[i] for i, v in d.items())
        eqs.append([d, rhs])
    col_all, col_alive = {}, {}
    for k, (d, _) in enumerate(eqs):
        for i in d:
            col_all.setdefault(i, set()).add(k)
            col_alive.setdefault(i, set()).add(k)
    alive = set(range(len(eqs)))
    heap = [(len(d), k) for k, (d, _) in enumerate(eqs)]
    heapq.heapify(heap)
    pivots_list = []
    nnz = sum(len(d) for d, _ in eqs)
    ops = 0
    while heap:
        ln, k = heapq.heappop(heap)
        if k not in alive:
            continue
        d, rhs = eqs[k]
        if ln != len(d):
            heapq.heappush(heap, (len(d), k))
            continue
        if not d:
            if rhs != 0:
                return None, ops
            alive.discard(k)
            continue
        col = min(d, key=lambda i: (len(col_alive.get(i, ())), abs(d[i])))
        inv = Fraction(1) / d[col]
        d2 = {i: v * inv for i, v in d.items()}
        rhs2 = rhs * inv
        eqs[k][0] = d2
        eqs[k][1] = rhs2
        alive.discard(k)
        for i in d2:
            col_alive.get(i, set()).discard(k)
        pivots_list.append((col, k))
        for kk in list(col_all.get(col, ())):
            if kk == k:
                continue
            dk, rhsk = eqs[kk]
            fac = dk.pop(col, None)
            if fac is None:
                continue
            col_all.get(col, set()).discard(kk)
            col_alive.get(col, set()).discard(kk)
            nnz -= 1
            for i, v in d2.items():
                if i == col:
                    continue
                old = dk.get(i)
                nv = (old if old is not None else Fraction(0)) - fac * v
                if nv == 0:
                    if old is not None:
                        dk.pop(i, None)
                        col_all.get(i, set()).discard(kk)
                        col_alive.get(i, set()).discard(kk)
                        nnz -= 1
                else:
                    if old is None:
                        col_all.setdefault(i, set()).add(kk)
                        if kk in alive:
                            col_alive.setdefault(i, set()).add(kk)
                        nnz += 1
                    dk[i] = nv
            rhsk -= fac * rhs2
            eqs[kk][1] = rhsk
            heapq.heappush(heap, (len(dk), kk))
            ops += len(d2)
            if nnz > FR_MAX_NNZ or ops > FR_MAX_OPS:
                return None, ops
        if len(pivots_list) > len(eqs) + 10:
            return None, ops
    for k in alive:
        if eqs[k][0] or eqs[k][1] != 0:
            return None, ops
    pivots = {col: eqs[k][1] for col, k in pivots_list}
    y = [frac(v) for v in y0]
    for i, v in pivots.items():
        y[i] += v
    return y, ops


def _fr_certificate(prob, x, y, debug=False):
    """対角 Q の FR 問題で、縮約費用を厳密に 0 にした下界 (d, pimpact)。

    Q 対角 q_j > 0 の列は xh_j = clamp((A_j'y - c_j)/q_j, lo_j, up_j) を有理数で
    構成し、g_j を厳密に評価する。q_j = 0 の自由列は _sparse_correction() で
    A_j'y = c_j を厳密に満たす。これにより自由列の縮約費用区間は厳密に [0, 0] と
    なり、浮動小数の区間演算では作れない弱双対下界を与える。二次項だけは
    xh の float 区間で上界を作る。
    """
    Q = prob["Q"]; A = prob["A"].tocsc(); c = prob["c"]
    lo, up = prob["lo"], prob["up"]
    n, m = prob["n"], prob["m"]
    if prob.get("psd", True) is not True:
        return None
    free = (~np.isfinite(lo)) & (~np.isfinite(up))
    if not free.any():
        return None
    q = np.asarray(Q.diagonal()).ravel()
    qc = Q.tocoo()
    if np.any((qc.data != 0) & (qc.row != qc.col)):
        return None
    if np.any(q < 0):
        return None
    y0 = _project_dual(prob, y)
    y1, ops = _sparse_correction(prob, y0)
    if y1 is None:
        if debug:
            print("  correction failed", flush=True)
        return None
    frac = lambda v: Fraction(float(v))
    h_frac = Fraction(0)
    for i in range(m):
        yi = y1[i]
        if yi > 0:
            if not math.isfinite(prob["L"][i]):
                if debug:
                    print("  row sign", i, flush=True)
                return None
            h_frac += yi * frac(prob["L"][i])
        elif yi < 0:
            if not math.isfinite(prob["U"][i]):
                if debug:
                    print("  row sign", i, flush=True)
                return None
            h_frac += yi * frac(prob["U"][i])
    xh_iv = []
    g_frac = []
    for j in range(n):
        a = A[:, j].tocoo()
        aty = sum(frac(v) * y1[int(i)] for i, v in zip(a.row, a.data))
        if q[j] != 0:
            xr = (aty - frac(c[j])) / frac(q[j])
            if math.isfinite(lo[j]):
                xr = max(xr, frac(lo[j]))
            if math.isfinite(up[j]):
                xr = min(xr, frac(up[j]))
            xh_iv.append(_fiv(xr))
            g_frac.append(frac(c[j]) + frac(q[j]) * xr - aty)
            continue
        if free[j]:
            if frac(c[j]) - aty != 0:
                if debug:
                    print("  free q0 not zero", j, flush=True)
                return None
            xh_iv.append((0.0, 0.0))
            g_frac.append(Fraction(0))
            continue
        xj = float(x[j])
        xh_iv.append((xj, xj))
        g_frac.append(frac(c[j]) - aty)
    col_frac = Fraction(0)
    for j in range(n):
        g = g_frac[j]
        if not math.isfinite(lo[j]) and not math.isfinite(up[j]):
            if g != 0:
                if debug:
                    print("  free col", j, flush=True)
                return None
        elif math.isfinite(lo[j]) and not math.isfinite(up[j]):
            if g < 0:
                if debug:
                    print("  lower col", j, flush=True)
                return None
            col_frac += g * frac(lo[j])
        elif not math.isfinite(lo[j]) and math.isfinite(up[j]):
            if g > 0:
                if debug:
                    print("  upper col", j, flush=True)
                return None
            col_frac += g * frac(up[j])
        else:
            col_frac += min(g * frac(lo[j]), g * frac(up[j]))
    q_terms = []
    for j in range(n):
        if q[j] == 0:
            continue
        if not math.isfinite(q[j]) or q[j] < 0:
            return None
        sq = _square(xh_iv[j])
        if math.isnan(sq[0]):
            return None
        q_terms.append(_imul((float(q[j]), float(q[j])), sq))
    q_iv = _isum(q_terms)
    if not math.isfinite(q_iv[1]) or math.isnan(q_iv[1]):
        return None
    total_frac = frac(prob["c0"]) + h_frac + col_frac - Fraction(q_iv[1]) / 2
    d = _fiv(total_frac)[0]
    if math.isnan(d) or d == -math.inf:
        return None
    yf = np.array([float(v) for v in y1])
    ax = prob["A"] @ x
    row_viol = _abs_violation(ax, prob["L"], prob["U"])
    var_viol = _abs_violation(x, lo, up)
    gabs = np.zeros(n)
    for j in range(n):
        giv = _fiv(g_frac[j])
        gabs[j] = max(abs(giv[0]), abs(giv[1]))
    p = float(prob["c0"] + prob["c"] @ x + 0.5 * x @ (prob["Q"] @ x))
    scale = 1.0 + abs(p)
    pimpact = float(np.abs(yf) @ row_viol + gabs @ var_viol) / scale
    if debug:
        print(f"  fr d={d!r} qhi={q_iv[1]!r} h={float(h_frac)!r} "
              f"col={float(col_frac)!r} ops={ops}", flush=True)
    return d, pimpact


def _solve_exact(M, r):
    """有理数 Gauss-Jordan。解けなければ None。"""
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


def _psd_exact(Q):
    """Q が有理数として厳密に半正定値かを LDL 分解で判定する。

    PSD 行列は対角成分が非負で、正の対角 pivot で Schur 補元を取っても PSD。
    対角が全て 0 になったら、PSD であるためには残りの非対角も全て 0 でなければ
    ならない (0 対角で非対角が非零だと 2x2 小行列式が負)。浮動小数の固有値判定と
    違い、丸めで生じた微小な負曲率を見逃さない。
    """
    n = Q.shape[0]
    A = [[Fraction(float(Q[i, j])) for j in range(n)] for i in range(n)]
    for k in range(n):
        piv = None
        for t in range(k, n):
            if A[t][t] < 0:
                return False
            if A[t][t] > 0 and piv is None:
                piv = t
        if piv is None:
            for i in range(k, n):
                for j in range(k, n):
                    if A[i][j] != 0:
                        return False
            return True
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            for i in range(n):
                A[i][k], A[i][piv] = A[i][piv], A[i][k]
        p = A[k][k]
        for i in range(k + 1, n):
            f = A[i][k] / p
            if f == 0:
                continue
            for j in range(k + 1, n):
                A[i][j] -= f * A[k][j]
    return True


def _kkt_certificate(prob, x, debug=False):
    """非対角 Q 用の厳密 KKT 下界 (d, pimpact)。

    active set を固定して KKT 方程式を有理数で解き、得た (xs, y) で双対下界
         c0 + h(y) - 1/2 xs'Qxs + sum_j min_{box} (c + Qxs - A'y)_j x_j
    を厳密に評価する。xs の主実行可能性は要求しない (下界の妥当性に不要)。
    自由列の g が厳密に 0 でない、または係数が無限大になる場合は None。
    """
    n, m = prob["n"], prob["m"]
    if prob.get("psd", True) is not True:
        return None
    Qsp = prob["Q"].tocsc()
    if (Qsp != Qsp.T).nnz != 0:
        return None
    qd = np.asarray(prob["Q"].diagonal()).ravel()
    qc = Qsp.tocoo()
    if np.any((qc.data != 0) & (qc.row != qc.col)):
        if n > KKT_EXACT_PSD_MAX or not _psd_exact(prob["Q"]):
            return None
    elif np.any(qd < 0):
        return None
    lo, up, L, U = prob["lo"], prob["up"], prob["L"], prob["U"]
    ax = prob["A"] @ x
    at_lo, at_up = _at_bound(x, lo), _at_bound(x, up)
    fixed = at_lo | at_up
    xb = np.where(at_lo, lo, up)
    F = np.flatnonzero(~fixed)
    B = np.flatnonzero(fixed)
    rows = np.flatnonzero(_at_bound(ax, L) | _at_bound(ax, U))
    if len(F) + len(rows) > KKT_MAX_UNKNOWNS or n * m > KKT_MAX_DENSE:
        return None
    A = prob["A"].toarray(); Q = prob["Q"].toarray(); c = prob["c"]
    Asp = prob["A"].tocsr()
    frac = lambda v: Fraction(float(v))
    xf = [frac(xb[j]) for j in B]
    b_rows = np.where(_at_bound(ax, L), L, U)[rows]
    k = len(F) + len(rows)
    M, r = [], []
    for j in F:
        M.append([frac(Q[j, kk]) for kk in F] + [-frac(A[i, j]) for i in rows])
        r.append(-frac(c[j]) - sum(frac(Q[j, b]) * xf[t] for t, b in enumerate(B)))
    for ii, i in enumerate(rows):
        M.append([frac(A[i, j]) for j in F] + [Fraction(0)] * len(rows))
        r.append(frac(b_rows[ii]) - sum(frac(A[i, b]) * xf[t] for t, b in enumerate(B)))
    sol = _solve_exact(M, r) if k else []
    if sol is None:
        return None
    xs = [Fraction(0)] * n
    for t, j in enumerate(F):
        xs[j] = sol[t]
    for t, j in enumerate(B):
        xs[j] = xf[t]
    yv = [Fraction(0)] * m
    for ii, i in enumerate(rows):
        yv[i] = sol[len(F) + ii]
    h = Fraction(0)
    for i in range(m):
        yi = yv[i]
        if yi > 0:
            if not math.isfinite(L[i]):
                return None
            h += yi * frac(L[i])
        elif yi < 0:
            if not math.isfinite(U[i]):
                return None
            h += yi * frac(U[i])
    Qxs = [Fraction(0)] * n
    for j in range(n):
        for kk, v in zip(Qsp.indices[Qsp.indptr[j]:Qsp.indptr[j + 1]],
                         Qsp.data[Qsp.indptr[j]:Qsp.indptr[j + 1]]):
            Qxs[j] += frac(v) * xs[int(kk)]
    g = [frac(c[j]) + Qxs[j] for j in range(n)]
    for i in range(m):
        if yv[i] == 0:
            continue
        yi = yv[i]
        for j, v in zip(Asp.indices[Asp.indptr[i]:Asp.indptr[i + 1]],
                        Asp.data[Asp.indptr[i]:Asp.indptr[i + 1]]):
            g[int(j)] -= yi * frac(v)
    qterm = Fraction(1, 2) * sum(xs[j] * Qxs[j] for j in range(n))
    box = Fraction(0)
    for j in range(n):
        if not math.isfinite(lo[j]) and not math.isfinite(up[j]):
            if g[j] != 0:
                return None
        elif math.isfinite(lo[j]) and not math.isfinite(up[j]):
            if g[j] < 0:
                return None
            box += g[j] * frac(lo[j])
        elif not math.isfinite(lo[j]) and math.isfinite(up[j]):
            if g[j] > 0:
                return None
            box += g[j] * frac(up[j])
        else:
            box += min(g[j] * frac(lo[j]), g[j] * frac(up[j]))
    total = frac(prob["c0"]) + h - qterm + box
    d = _fiv(total)[0]
    if math.isnan(d) or d == -math.inf:
        return None
    yf = np.array([float(v) for v in yv])
    row_viol = _abs_violation(ax, L, U)
    var_viol = _abs_violation(x, lo, up)
    gabs = np.zeros(n)
    for j in range(n):
        giv = _fiv(g[j])
        gabs[j] = max(abs(giv[0]), abs(giv[1]))
    p = float(prob["c0"] + prob["c"] @ x + 0.5 * x @ (prob["Q"] @ x))
    scale = 1.0 + abs(p)
    pimpact = float(np.abs(yf) @ row_viol + gabs @ var_viol) / scale
    if debug:
        print(f"  kkt d={d!r} k={k} pimpact={pimpact!r}", flush=True)
    return d, pimpact


def dual_bound(prob, xh, y):
    """凸 QP の最適値の厳密な下界 (取れなければ -inf)。

    実行可能な x と符号を射影した y で y_i (Ax)_i >= h_i(y_i) (h_i(y) = y L_i (y > 0) /
    y U_i (y < 0))、Q の凸性で 1/2 x'Qx >= -1/2 xh'Q xh + (Q xh)'x なので
        f(x) >= c0 + sum_i h_i(y_i) - 1/2 xh'Q xh + sum_j min_{lo_j <= x_j <= up_j} g_j x_j,
        g = c + Q xh - A'y
    が任意の xh で成り立つ。g は丸め誤差上界で広げた区間で評価し、各項・和を -inf 方向へ
    丸めるので、返す値は浮動小数計算の結果でなく真の右辺の下界である。無限側の境界を
    要求する成分の符号が区間で確定できなければ -inf を返す (0 扱いの見積りはしない)。
    """
    if prob.get("psd", True) is not True:
        return -math.inf
    yp = _project_dual(prob, y)
    _, _, col, unbounded = _classify(prob, xh, yp)
    if unbounded.any():
        return -math.inf
    L, U = prob["L"], prob["U"]
    with np.errstate(invalid="ignore"):
        h = np.where(yp > 0, _down(yp * L), np.where(yp < 0, _down(yp * U), 0.0))
    row = _safe_fsum(h)
    if not math.isfinite(row):
        return -math.inf
    row = _down(row)
    xqx, xqx_err = _quad_form(prob["Q"], xh)
    half_quad_up = _up(0.5 * _up(xqx + xqx_err))
    total = _safe_fsum([prob["c0"], row, -half_quad_up, col])
    if not math.isfinite(total):
        return -math.inf
    return float(_down(total))


def _dual_qp_candidate(prob):
    """弱双対下界を最大化する双対 QP を Clarabel で解き、(xh, y) を返す (失敗なら None)。

    双対 QP は
        max  sum_i h_i(y_i) - 1/2 xh'Qxh + sum_j min_{lo_j<=x_j<=up_j} g_j x_j,
        g = c + Qxh - A'y
    を y の符号制約のもとで解く問題で、box 最小値は補助変数 t_j と
        t_j <= g_j x_j  (x_j は lo/up の端点)
    で表す。返す候補は浮動小数の解にすぎず、下界の妥当性は _exact_d() が有理数で検査する。
    """
    try:
        import clarabel
    except ImportError:
        return None

    A, Q, c = prob["A"].tocsr(), prob["Q"].tocsc(), prob["c"]
    lo, up, L, U = prob["lo"], prob["up"], prob["L"], prob["U"]
    n, m = prob["n"], prob["m"]
    pidx = np.flatnonzero(np.isfinite(L))
    nidx = np.flatnonzero(np.isfinite(U))
    P, N = len(pidx), len(nidx)
    off_y = n
    off_yn = n + P
    off_t = n + P + N
    nv = n + P + N + n
    q = np.zeros(nv)
    q[off_y:off_y + P] = -L[pidx]
    q[off_yn:off_yn + N] = U[nidx]
    q[off_t:] = -1.0
    rows, cols, vals, b = [], [], [], []
    r = 0
    Acsc = A.tocsc()
    for j in range(n):
        aj = Acsc[:, j].tocoo()
        # 有限な下限・上限それぞれで t_j <= g_j * bound を課す。
        for bound in (lo[j], up[j]):
            if not np.isfinite(bound):
                continue
            rows.append(r)
            cols.append(off_t + j)
            vals.append(1.0)
            for i, v in zip(aj.row, aj.data):
                ip = np.searchsorted(pidx, i)
                if ip < P and pidx[ip] == i:
                    rows.append(r)
                    cols.append(off_y + ip)
                    vals.append(bound * v)
                iN = np.searchsorted(nidx, i)
                if iN < N and nidx[iN] == i:
                    rows.append(r)
                    cols.append(off_yn + iN)
                    vals.append(-bound * v)
            qj = Q[:, j].tocoo()
            for i, v in zip(qj.row, qj.data):
                rows.append(r)
                cols.append(i)
                vals.append(-bound * v)
            b.append(bound * c[j])
            r += 1
        # 片側無限変数では有限側の端点だけを考えればよい (反対側は g の符号で消える)。
        if np.isfinite(lo[j]) and lo[j] != up[j] and not np.isfinite(up[j]):
            for i, v in zip(aj.row, aj.data):
                ip = np.searchsorted(pidx, i)
                if ip < P and pidx[ip] == i:
                    rows.append(r)
                    cols.append(off_y + ip)
                    vals.append(v)
                iN = np.searchsorted(nidx, i)
                if iN < N and nidx[iN] == i:
                    rows.append(r)
                    cols.append(off_yn + iN)
                    vals.append(-v)
            qj = Q[:, j].tocoo()
            for i, v in zip(qj.row, qj.data):
                rows.append(r)
                cols.append(i)
                vals.append(-v)
            b.append(c[j])
            r += 1
        if np.isfinite(up[j]) and up[j] != lo[j] and not np.isfinite(lo[j]):
            for i, v in zip(aj.row, aj.data):
                ip = np.searchsorted(pidx, i)
                if ip < P and pidx[ip] == i:
                    rows.append(r)
                    cols.append(off_y + ip)
                    vals.append(-v)
                iN = np.searchsorted(nidx, i)
                if iN < N and nidx[iN] == i:
                    rows.append(r)
                    cols.append(off_yn + iN)
                    vals.append(v)
            qj = Q[:, j].tocoo()
            for i, v in zip(qj.row, qj.data):
                rows.append(r)
                cols.append(i)
                vals.append(v)
            b.append(-c[j])
            r += 1
    # y = yp - yn, yp, yn >= 0 (非負錐で課す)。
    for k in range(P):
        rows.append(r)
        cols.append(off_y + k)
        vals.append(-1.0)
        b.append(0.0)
        r += 1
    for k in range(N):
        rows.append(r)
        cols.append(off_yn + k)
        vals.append(-1.0)
        b.append(0.0)
        r += 1
    M = sp.coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsc()
    Pmat = sp.lil_matrix((nv, nv))
    Pmat[:n, :n] = Q
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = DUAL_QP_TOL
    settings.max_iter = 2000
    try:
        sol = clarabel.DefaultSolver(sp.triu(Pmat).tocsc(), q, M, np.array(b),
                                     [clarabel.NonnegativeConeT(r)], settings).solve()
    except Exception:
        return None
    if str(sol.status) != "Solved":
        return None
    xh = np.array(sol.x[:n])
    yp = np.array(sol.x[off_y:off_y + P])
    yn = np.array(sol.x[off_yn:off_yn + N])
    y = np.zeros(m)
    y[pidx] += yp
    y[nidx] -= yn
    if not (np.all(np.isfinite(xh)) and np.all(np.isfinite(y))):
        return None
    return xh, y


def _margin_lp(prob, x, y_ref):
    """零接触集合を同定する LP を解き、(xh, y, 集合) を返す (失敗なら None)。

    x の active set を固定し、interior 列では g_j = 0、active な片側無限列では
    g_j に符号付きの margin tau を要求する。tau を最大化して、tau > 0 の列を
    「零接触でない」、tau = 0 の列を「零接触」と判定する。得られた候補は
    厳密有理数の修復初期値に使うだけで、下界の妥当性は _exact_d() が検査する。
    """
    A, Q, c = prob["A"].tocsc(), prob["Q"].tocsc(), prob["c"]
    lo, up, L, U = prob["lo"], prob["up"], prob["L"], prob["U"]
    n, m = prob["n"], prob["m"]
    s = _exact_sets(prob, x)
    atL, atU, atlo, atup, fixed = s["atL"], s["atU"], s["atlo"], s["atup"], s["fixed"]
    interior = s["interior"]
    I = np.flatnonzero(interior)
    R = np.flatnonzero(atL | atU)
    nv = len(I) + len(R) + 1
    rows, cols, vals, b = [], [], [], []
    r = 0

    def add(j, sign):
        q = Q[:, j].tocoo()
        for i, v in zip(q.row, q.data):
            k = np.searchsorted(I, i)
            if k < len(I) and I[k] == i:
                rows.append(r)
                cols.append(int(k))
                vals.append(sign * float(v))
        a = A[:, j].tocoo()
        for i, v in zip(a.row, a.data):
            k = np.searchsorted(R, i)
            if k < len(R) and R[k] == i:
                rows.append(r)
                cols.append(len(I) + int(k))
                vals.append(-sign * float(v))

    # interior 列: Q xh - A'y = -c。
    for j in np.flatnonzero(interior):
        add(j, 1.0)
        b.append(-c[j])
        r += 1
    neq = r
    # active な片側無限列: g >= tau (下限側) / g <= -tau (上限側)。
    for j in range(n):
        if fixed[j]:
            continue
        if atlo[j] and not atup[j]:
            add(j, -1.0)
            rows.append(r)
            cols.append(nv - 1)
            vals.append(1.0)
            b.append(c[j])
            r += 1
        elif atup[j] and not atlo[j]:
            add(j, 1.0)
            rows.append(r)
            cols.append(nv - 1)
            vals.append(1.0)
            b.append(-c[j])
            r += 1
    lb = np.full(nv, -np.inf)
    ub = np.full(nv, np.inf)
    for k, i in enumerate(R):
        hlb = -math.inf
        hub = math.inf
        if not np.isfinite(L[i]):
            hub = 0.0
        if not np.isfinite(U[i]):
            hlb = 0.0
        ctr = min(max(y_ref[i], hlb), hub)
        lb[len(I) + k] = max(hlb, ctr - MARGIN_LP_DY)
        ub[len(I) + k] = min(hub, ctr + MARGIN_LP_DY)
    lb[-1] = 0.0
    for k, j in enumerate(I):
        lb[k] = x[j] - MARGIN_LP_DELTA
        ub[k] = x[j] + MARGIN_LP_DELTA
    M = sp.coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    Aeq, beq = M[:neq], np.array(b[:neq])
    Aub, bub = M[neq:], np.array(b[neq:])
    cv = np.zeros(nv)
    cv[-1] = -1.0
    res = linprog(cv, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=list(zip(lb, ub)), method="highs",
                  options=dict(dual_feasibility_tolerance=1e-10,
                               primal_feasibility_tolerance=1e-10))
    if res.status != 0 or not np.all(np.isfinite(res.x)):
        return None
    xh = np.zeros(n)
    xh[I] = res.x[:len(I)]
    xh[atlo & ~fixed] = lo[atlo & ~fixed]
    xh[atup & ~fixed] = up[atup & ~fixed]
    xh[fixed] = lo[fixed]
    y = np.zeros(m)
    y[R] = res.x[len(I):len(I) + len(R)]
    return xh, y, s


class _ExactData:
    """浮動小数の問題データを有理数へ移したもの。

    QPS の浮動小数は有理数として厳密に扱える。列ごとの非ゼロ (行番号, Fraction) を
    保持し、厳密な縮約費用と弱双対値を評価する。
    """

    def __init__(self, prob):
        self.n, self.m = prob["n"], prob["m"]
        self.c0 = Fraction(float(prob["c0"]))
        self.c = [Fraction(float(v)) for v in prob["c"]]
        self.A = self._columns(prob["A"].tocsc())
        self.Q = self._columns(prob["Q"].tocsc())
        finite = lambda v: Fraction(float(v)) if math.isfinite(v) else None
        self.L = [finite(v) for v in prob["L"]]
        self.U = [finite(v) for v in prob["U"]]
        self.lo = [finite(v) for v in prob["lo"]]
        self.up = [finite(v) for v in prob["up"]]

    @staticmethod
    def _columns(M):
        return [[(int(M.indices[k]), Fraction(float(M.data[k])))
                 for k in range(M.indptr[j], M.indptr[j + 1]) if M.data[k] != 0.0]
                for j in range(M.shape[1])]


def _exact_g(data, xh, y):
    """縮約費用 g = c + Q xh - A'y を有理数で評価する。"""
    out = []
    for j in range(data.n):
        g = data.c[j]
        for i, v in data.Q[j]:
            g += v * xh[i]
        for i, v in data.A[j]:
            g -= v * y[i]
        out.append(g)
    return out


def _exact_d(data, xh, y, gex=None):
    """弱双対の右辺 c0 + h(y) - 1/2 xh'Qxh + sum_j min g_j x_j を有理数で評価する。

    符号制約 (y_i > 0 なら L_i 有限、y_i < 0 なら U_i 有限、片側無限列では g_j が
    有限側を要求しないこと、自由列では g_j = 0) を満たさなければ None を返す。
    返す Fraction は浮動小数の丸めを含まない厳密な下界である。
    """
    h = Fraction(0)
    for i, yi in enumerate(y):
        if yi > 0:
            if data.L[i] is None:
                return None
            h += yi * data.L[i]
        elif yi < 0:
            if data.U[i] is None:
                return None
            h += yi * data.U[i]
    if gex is None:
        gex = _exact_g(data, xh, y)
    quad = Fraction(0)
    for j in range(data.n):
        if xh[j]:
            s = Fraction(0)
            for i, v in data.Q[j]:
                s += v * xh[i]
            quad += xh[j] * s
    col = Fraction(0)
    for j, g in enumerate(gex):
        lo, up = data.lo[j], data.up[j]
        if lo is not None and up is not None:
            col += min(g * lo, g * up)
        elif lo is not None:
            if g < 0:
                return None
            col += g * lo
        elif up is not None:
            if g > 0:
                return None
            col += g * up
        else:
            if g != 0:
                return None
    return data.c0 + h - quad / 2 + col


def _exact_float(d):
    """有理数の下界 d を、d 以下の float へ丸める (丸めで上へ出さない)。"""
    try:
        df = float(d)
    except OverflowError:
        return -math.inf
    if not math.isfinite(df):
        return -math.inf
    if Fraction(df) > d:
        df = math.nextafter(df, -math.inf)
    return df


def _fraction_float_up(v):
    """非負の有理数 v を v 以上の float へ丸める。"""
    try:
        f = float(v)
    except OverflowError:
        return math.inf
    if not math.isfinite(f):
        return math.inf
    if Fraction(f) < v:
        f = math.nextafter(f, math.inf)
    return f


def _exact_sets(prob, x):
    """x の active set と interior 列。"""
    ax = prob["A"] @ x
    atL, atU = _at_bound(ax, prob["L"]), _at_bound(ax, prob["U"])
    atlo, atup = _at_bound(x, prob["lo"]), _at_bound(x, prob["up"])
    fixed = prob["lo"] == prob["up"]
    interior = ~(fixed | atlo | atup)
    return dict(ax=ax, atL=atL, atU=atU, atlo=atlo, atup=atup, fixed=fixed,
                interior=interior)


def _solve_sparse_exact(rows, rhs, deadline):
    """疎な有理数連立一次方程式を Markowitz 風の Gauss-Jordan で解く。

    rows は {未知数番号: 係数} の辞書、rhs は右辺。返り値は (ピボット解, 演算数)。
    非ピボット変数は 0 とし、不整合・作業量上限・時間切れなら None。
    """
    eqs = [[dict(d), r] for d, r in zip(rows, rhs)]
    col_all, col_alive = {}, {}
    for k, (d, _) in enumerate(eqs):
        for i in d:
            col_all.setdefault(i, set()).add(k)
            col_alive.setdefault(i, set()).add(k)
    alive = set(range(len(eqs)))
    heap = [(len(d), k) for k, (d, _) in enumerate(eqs)]
    heapq.heapify(heap)
    pivot_pairs = []
    nnz = sum(len(d) for d, _ in eqs)
    ops = 0
    while heap:
        if time.monotonic() > deadline:
            return None
        ln, k = heapq.heappop(heap)
        if k not in alive:
            continue
        d, r = eqs[k]
        if ln != len(d):
            heapq.heappush(heap, (len(d), k))
            continue
        if not d:
            if r != 0:
                return None
            alive.discard(k)
            continue
        col = min(d, key=lambda i: (len(col_alive.get(i, ())), abs(d[i])))
        inv = Fraction(1) / d[col]
        d2 = {i: v * inv for i, v in d.items()}
        r2 = r * inv
        eqs[k] = [d2, r2]
        alive.discard(k)
        for i in d2:
            col_alive.get(i, set()).discard(k)
        pivot_pairs.append((col, k))
        for kk in list(col_all.get(col, ())):
            if kk == k:
                continue
            dk, rk = eqs[kk]
            fac = dk.pop(col, None)
            if fac is None:
                continue
            col_all.get(col, set()).discard(kk)
            col_alive.get(col, set()).discard(kk)
            nnz -= 1
            for i, v in d2.items():
                if i == col:
                    continue
                old = dk.get(i)
                nv = (old if old is not None else Fraction(0)) - fac * v
                if nv == 0:
                    if old is not None:
                        dk.pop(i)
                        col_all.get(i, set()).discard(kk)
                        col_alive.get(i, set()).discard(kk)
                        nnz -= 1
                else:
                    if old is None:
                        col_all.setdefault(i, set()).add(kk)
                        if kk in alive:
                            col_alive.setdefault(i, set()).add(kk)
                        nnz += 1
                    dk[i] = nv
            eqs[kk][1] = rk - fac * r2
            if kk in alive:
                heapq.heappush(heap, (len(dk), kk))
            ops += len(d2)
            if nnz > EXACT_MAX_NNZ or ops > EXACT_MAX_OPS:
                return None
        if len(pivot_pairs) > EXACT_MAX_PIVOTS:
            return None
    for k in alive:
        d, r = eqs[k]
        if d or r != 0:
            return None
    return {col: eqs[k][1] for col, k in pivot_pairs}, ops


def _recover_exact(prob, data, s, xh0, y0, E0, R, deadline):
    """有理数の active-set 双対候補を、縮約費用の符号が厳密に正しくなるまで修正する。

    E0 の列は g_j = 0 を方程式として解き、xh と行乗数を動かす。片側無限列の g の
    符号がまだ悪い列、自由列の g != 0、符号制約に反する行乗数を検出したら、その列/
    行を方程式に加えて再解する。返す (xh, y, gex, 反復数) はすべて Fraction。
    """
    n, m = data.n, data.m
    I = np.flatnonzero(s["interior"])
    ridx = {int(r): k for k, r in enumerate(R)}
    xh = [Fraction(float(v)) for v in xh0]
    y = [Fraction(float(v)) for v in y0]
    gex = _exact_g(data, xh, y)
    lo_side = np.isfinite(prob["lo"]) & ~np.isfinite(prob["up"]) & ~s["fixed"]
    up_side = np.isfinite(prob["up"]) & ~np.isfinite(prob["lo"]) & ~s["fixed"]
    free = ~np.isfinite(prob["lo"]) & ~np.isfinite(prob["up"]) & ~s["fixed"]
    E = {int(j) for j in E0}
    E |= {int(j) for j in np.flatnonzero(lo_side) if gex[j] < 0}
    E |= {int(j) for j in np.flatnonzero(up_side) if gex[j] > 0}
    E |= {int(j) for j in np.flatnonzero(free) if gex[j] != 0}
    pin_rows = set()
    max_iter = 15
    for it in range(max_iter):
        if time.monotonic() > deadline:
            return None
        cols = np.array(sorted(E), dtype=int)
        rows, rhs = [], []
        for j in cols:
            d = {}
            for i, v in data.Q[j]:
                k = np.searchsorted(I, i)
                if k < len(I) and I[k] == i:
                    d[int(k)] = d.get(int(k), Fraction(0)) + v
            for i, v in data.A[j]:
                k = ridx.get(int(i))
                if k is not None:
                    d[len(I) + k] = d.get(len(I) + k, Fraction(0)) - v
            rows.append({k: v for k, v in d.items() if v})
            rhs.append(-gex[j])
        for i in sorted(pin_rows):
            k = ridx.get(int(i))
            if k is not None:
                rows.append({len(I) + k: Fraction(1)})
                rhs.append(-y[i])
        out = _solve_sparse_exact(rows, rhs, deadline)
        if out is None:
            return None
        piv, _ = out
        new_xh = list(xh)
        new_y = list(y)
        for k, v in piv.items():
            if k < len(I):
                new_xh[I[k]] += v
            else:
                new_y[R[k - len(I)]] += v
        xh, y = new_xh, new_y
        gex = _exact_g(data, xh, y)
        badlo = [int(j) for j in np.flatnonzero(lo_side) if gex[j] < 0]
        badup = [int(j) for j in np.flatnonzero(up_side) if gex[j] > 0]
        badfree = [int(j) for j in np.flatnonzero(free) if gex[j] != 0]
        rowbad = [int(i) for i in R
                  if (y[i] > 0 and not math.isfinite(prob["L"][i]))
                  or (y[i] < 0 and not math.isfinite(prob["U"][i]))]
        if not badlo and not badup and not badfree and not rowbad:
            return xh, y, gex, it + 1
        old = (len(E), len(pin_rows))
        E |= set(badlo) | set(badup) | set(badfree)
        pin_rows |= set(rowbad)
        if (len(E), len(pin_rows)) == old:
            return None
    return None


def _exact_base(prob, s, xh_base, y_base, R):
    """厳密修復の初期値を、x の active set に合わせて境界値へ寄せた float で返す。"""
    xh = np.array(xh_base, dtype=float, copy=True)
    y = np.array(y_base, dtype=float, copy=True)
    fixed = s["fixed"]
    lo_at = s["atlo"] & ~fixed
    up_at = s["atup"] & ~fixed
    xh[fixed] = prob["lo"][fixed]
    xh[lo_at] = prob["lo"][lo_at]
    xh[up_at] = prob["up"][up_at]
    y = _project_dual(prob, y)
    mask = np.ones(prob["m"], dtype=bool)
    mask[R] = False
    y[mask] = 0.0
    return xh, y


def _exact_violations(data, x):
    """x の主実行不能度を有理数で評価し、(相対違反の上界, 行違反, 列違反) を返す。"""
    xf = [Fraction(float(v)) for v in x]
    ax = [Fraction(0)] * data.m
    for j, xj in enumerate(xf):
        if xj:
            for i, v in data.A[j]:
                ax[i] += v * xj
    pinf = Fraction(0)
    row_v = []
    for i, val in enumerate(ax):
        v = Fraction(0)
        lo, up = data.L[i], data.U[i]
        if lo is not None and val < lo:
            v = lo - val
            pinf = max(pinf, v / (1 + abs(lo)))
        if up is not None and val > up:
            d = val - up
            v = max(v, d)
            pinf = max(pinf, d / (1 + abs(up)))
        row_v.append(v)
    col_v = []
    for j, val in enumerate(xf):
        v = Fraction(0)
        lo, up = data.lo[j], data.up[j]
        if lo is not None and val < lo:
            v = lo - val
            pinf = max(pinf, v / (1 + abs(lo)))
        if up is not None and val > up:
            d = val - up
            v = max(v, d)
            pinf = max(pinf, d / (1 + abs(up)))
        col_v.append(v)
    return _fraction_float_up(pinf), row_v, col_v


def _exact_pimpact(p, y, gex, row_v, col_v):
    """厳密候補の主違反影響を、有理数の乗数・縮約費用から上界側で評価する。"""
    total = Fraction(0)
    for i, yi in enumerate(y):
        if yi and row_v[i]:
            total += abs(yi) * row_v[i]
    for j, gj in enumerate(gex):
        if gj and col_v[j]:
            total += abs(gj) * col_v[j]
    return _fraction_float_up(total / Fraction(1.0 + abs(p)))


def _exact_certificate(prob, x, base, y_hint):
    """厳密有理数の双対候補を作り、base と同じ p/pinf/roundoff を持つ Certificate を返す。

    双対 QP の候補を出発点に、負の縮約費用だけを修復する戦略 A と、margin LP で
    零接触集合を同定してから修復する戦略 B を試す。返す d は _exact_d() の Fraction を
    下向きに丸めた float なので、浮動小数の丸めを含めても最適値の下界である。
    """
    if prob.get("psd", True) is not True:
        return None
    if prob["n"] > EXACT_MAX_COLS or prob["m"] > EXACT_MAX_ROWS:
        return None
    deadline = time.monotonic() + EXACT_TIME_BUDGET
    data = _ExactData(prob)
    s = _exact_sets(prob, x)
    pinf_exact, row_v, col_v = _exact_violations(data, x)
    all_rows = np.arange(prob["m"])
    active_rows = np.flatnonzero(s["atL"] | s["atU"])
    candidates = []

    def evaluate(method, xh, y, gex):
        d = _exact_d(data, xh, y, gex)
        if d is None:
            return None
        cert = Certificate(base.p, _exact_float(d), max(base.pinf, pinf_exact),
                           base.roundoff,
                           _exact_pimpact(base.p, y, gex, row_v, col_v),
                           base.repair_iters, method)
        candidates.append(cert)
        return cert

    def margin_set(margin):
        xh_lp, y_lp, _ = margin
        glp = _exact_g(data, [Fraction(float(v)) for v in xh_lp],
                       [Fraction(float(v)) for v in y_lp])
        lo_only = s["atlo"] & ~s["atup"] & ~s["fixed"]
        up_only = s["atup"] & ~s["atlo"] & ~s["fixed"]
        E = {int(j) for j in np.flatnonzero(s["interior"])}
        E |= {int(j) for j in np.flatnonzero(lo_only) if glp[j] <= 0}
        E |= {int(j) for j in np.flatnonzero(up_only) if glp[j] >= 0}
        return E

    qp = _dual_qp_candidate(prob)
    y_ref = _project_dual(prob, y_hint)
    if qp is not None:
        y_ref = _project_dual(prob, qp[1])
    margin = _margin_lp(prob, x, y_ref)
    E_margin = margin_set(margin) if margin is not None else None

    def try_recover(method, xh_base, y_base, E0, R):
        xh0, y0 = _exact_base(prob, s, xh_base, y_base, R)
        out = _recover_exact(prob, data, s, xh0, y0, E0, R, deadline)
        if out is None:
            return None
        xh, y, gex, _ = out
        return evaluate(method, xh, y, gex)

    if qp is not None:
        # 戦略 A: 全行を候補に残し、厳密に負の縮約費用だけを修復する。
        xh_qp, y_qp = qp
        xh0, y0 = _exact_base(prob, s, xh_qp, y_qp, all_rows)
        gex0 = _exact_g(data, [Fraction(float(v)) for v in xh0],
                        [Fraction(float(v)) for v in y0])
        lo_side = np.isfinite(prob["lo"]) & ~np.isfinite(prob["up"]) & ~s["fixed"]
        up_side = np.isfinite(prob["up"]) & ~np.isfinite(prob["lo"]) & ~s["fixed"]
        E_neg = {int(j) for j in np.flatnonzero(lo_side) if gex0[j] < 0}
        E_neg |= {int(j) for j in np.flatnonzero(up_side) if gex0[j] > 0}
        cert = try_recover("exact-neg", xh_qp, y_qp, E_neg, all_rows)
        if cert is not None and uncertainty(cert) <= UNCERTAINTY_TOL:
            return cert
        # 戦略 B: margin LP の零接触集合を使い、active 行だけを候補に残す。
        if margin is not None:
            cert = try_recover("exact-margin", xh_qp, y_qp, E_margin, active_rows)
            if cert is not None and uncertainty(cert) <= UNCERTAINTY_TOL:
                return cert
    # 双対 QP が使えない/失敗した場合の保険: margin LP 自体を初期値にする。
    if margin is not None:
        xh_lp, y_lp, _ = margin
        cert = try_recover("exact-margin-lp", xh_lp, y_lp, E_margin, active_rows)
        if cert is not None and uncertainty(cert) <= UNCERTAINTY_TOL:
            return cert
    if not candidates:
        return None
    return min(candidates, key=uncertainty)


Certificate = collections.namedtuple(
    "Certificate", "p d pinf roundoff pimpact repair_iters method", defaults=("float",))


def _float_certificate(prob, x, ax, p, pinf, roundoff, xh, y, iters, method):
    """浮動小数の候補 (xh, y) から Certificate を作る。"""
    d = dual_bound(prob, xh, y)
    gl, gu, _ = _reduced_cost_interval(prob, xh, y)
    g = 0.5 * (gl + gu)
    pimpact = float(np.abs(y) @ _abs_violation(ax, prob["L"], prob["U"])
                    + np.abs(g) @ _abs_violation(x, prob["lo"], prob["up"])) / (1.0 + abs(p))
    return Certificate(p, d, pinf, roundoff, pimpact, iters, method)


def verify(prob, x, y, exact=False):
    """独立再計算 (docstring 冒頭の定義)。Certificate を返す。

      p             主目的値 c0 + c'x + 1/2 x'Qx (浮動小数値)
      d             _repair() した (xh, y) での dual_bound()。最適値の厳密な下界 (なければ -inf)。
                    自由変数を含み dual_bound() が -inf の場合は _fr_certificate() /
                    _kkt_certificate() の厳密証明も試し、より強い下界を採用する。
      pinf          相対主実行不能度 max 違反 / (1 + |境界|)
      roundoff      p の丸め誤差上界 / (1 + |p|)
      pimpact       主違反の目的値影響 (sum |y_i| 行違反_i + sum |g_j| 列違反_j) / (1 + |p|)。
                    x が境界を僅かに破るとき p(x) は真の最適値の上界でなくなるが、その差は
                    一次で乗数 × 違反量 (感度) で抑えられる。
      repair_iters  _repair() の反復数 (0 は修正不要)
      method        d を作った経路 ("solver" / "repair" / "linearized" / "exact-*")

    exact=True のとき、浮動小数の候補が UNCERTAINTY_TOL を超える場合に厳密有理数の
    双対候補 (_exact_certificate) も試す。best_certificate() は候補ごとの二重実行を
    避けるため exact=False で比較してから、最良の主候補に一度だけ厳密候補を適用する。
    """
    A, Q, c, c0 = prob["A"], prob["Q"], prob["c"], prob["c0"]
    if not np.all(np.isfinite(x)):
        return Certificate(math.nan, -math.inf, math.inf, math.inf, math.inf, 0, "invalid")
    ax = A @ x
    pinf = max(_rel_violation(ax, prob["L"], prob["U"]),
               _rel_violation(x, prob["lo"], prob["up"]))
    xqx, xqx_err = _quad_form(Q, x)
    cx = c * x
    try:
        p = math.fsum([c0, math.fsum(cx), 0.5 * xqx])
        cx_abs = math.fsum(np.abs(cx))
    except (ValueError, OverflowError):
        p = math.nan
        cx_abs = math.inf
    p_err = ERROR_BOUND_SAFETY * (UNIT_ROUNDOFF * (cx_abs + abs(p))
                                  + len(x) * UNDERFLOW_ABS) + 0.5 * xqx_err
    scale = 1.0 + abs(p)
    roundoff = float(p_err) / scale
    if prob.get("psd", True) is not True:
        return Certificate(p, -math.inf, pinf, roundoff, 0.0, 0, "nonconvex")
    if not math.isfinite(p) or not math.isfinite(xqx):
        return Certificate(p, -math.inf, pinf, roundoff, math.inf, 0, "overflow")
    y_proj = _project_dual(prob, y)
    xh, yr, iters = _repair(prob, x.copy(), y_proj)
    method = "repair" if iters else "solver"
    cert = _float_certificate(prob, x, ax, p, pinf, roundoff, xh, yr, iters, method)
    tried_linearized = iters == REPAIR_ITERS
    if tried_linearized:
        y_lin = _linearized_dual(prob, x, y_proj)
        if y_lin is not None:
            xh2, yr2, more = _repair(prob, x.copy(), y_lin)
            cert2 = _float_certificate(prob, x, ax, p, pinf, roundoff, xh2, yr2,
                                       iters + more, "linearized")
            if uncertainty(cert2) < uncertainty(cert):
                cert = cert2
    elif uncertainty(cert) > UNCERTAINTY_TOL:
        # _repair() が 0 回で終わっても、双対候補が悪ければ線形化 LP の双対を試す。
        # QRECIPE はこの経路で ~1e-6 の gap を ~1e-11 まで詰める。
        y_lin = _linearized_dual(prob, x, y_proj)
        if y_lin is not None:
            xh2, yr2, more = _repair(prob, x.copy(), y_lin)
            cert2 = _float_certificate(prob, x, ax, p, pinf, roundoff, xh2, yr2,
                                       iters + more, "linearized")
            if uncertainty(cert2) < uncertainty(cert):
                cert = cert2
    if exact and uncertainty(cert) > UNCERTAINTY_TOL:
        ex = _exact_certificate(prob, x, cert, y)
        if (ex is not None and ex.pinf <= PRIMAL_INF_TOL
                and uncertainty(ex) < uncertainty(cert)):
            cert = ex
    # 自由変数を含み、通常の dual_bound() が -inf のときの厳密証明。
    # _fr_certificate は対角 Q 用、_kkt_certificate は小規模な非対角 Q 用。
    free = (~np.isfinite(prob["lo"])) & (~np.isfinite(prob["up"]))
    if free.any() and cert.d == -math.inf:
        fr = _fr_certificate(prob, x, y)
        if fr is not None:
            cand = Certificate(cert.p, fr[0], cert.pinf, cert.roundoff,
                               fr[1], cert.repair_iters, "fr")
            if uncertainty(cand) < uncertainty(cert):
                cert = cand
        if cert.d == -math.inf or uncertainty(cert) > UNCERTAINTY_TOL:
            kk = _kkt_certificate(prob, x)
            if kk is not None:
                cand = Certificate(cert.p, kk[0], cert.pinf, cert.roundoff,
                                   kk[1], cert.repair_iters, "kkt")
                if uncertainty(cand) < uncertainty(cert):
                    cert = cand
    return cert


def uncertainty(cert):
    """参照値 p の相対不確かさ: |p - d| / (1 + |p|) + roundoff + pimpact (d = -inf なら inf)。"""
    if not math.isfinite(cert.d):
        return math.inf
    return abs(cert.p - cert.d) / (1.0 + abs(cert.p)) + cert.roundoff + cert.pimpact


def best_certificate(prob, x, y):
    """ソルバの (x, y) と polish() の (x, y) のうち、相対主実行不能度が PRIMAL_INF_TOL
    以下で uncertainty() 最小の候補を返す。浮動小数の候補が許容を超える場合は、最良の
    主候補に対して厳密有理数の双対候補も試す。該当候補が無ければソルバ側を返す。"""
    if prob.get("psd", True) is not True:
        return verify(prob, x, y)
    cands = [(x, y)]
    polished = polish(prob, x, y)
    if polished is not None:
        cands.append(polished)
    results = [verify(prob, cx, cy) for cx, cy in cands]
    feasible = [(r, cx, cy) for r, (cx, cy) in zip(results, cands)
                if r.pinf <= PRIMAL_INF_TOL]
    if not feasible:
        return results[0]
    best = min(feasible, key=lambda t: uncertainty(t[0]))
    if uncertainty(best[0]) <= UNCERTAINTY_TOL:
        return best[0]
    for r, cx, cy in sorted(feasible, key=lambda t: uncertainty(t[0])):
        ex = _exact_certificate(prob, cx, r, cy)
        if (ex is not None and ex.pinf <= PRIMAL_INF_TOL
                and uncertainty(ex) < uncertainty(best[0])):
            best = (ex, cx, cy)
        if uncertainty(best[0]) <= UNCERTAINTY_TOL:
            break
    return best[0]


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
    cert = best_certificate(prob, x, y)
    unc = uncertainty(cert)
    gap = abs(cert.p - cert.d) / (1.0 + abs(cert.p))
    # NaN は比較が常に False になるので、合格条件を肯定形で書き否定を取る。
    ok = (unc <= UNCERTAINTY_TOL and cert.pinf <= PRIMAL_INF_TOL
          and objdiff <= OBJ_CONSISTENCY_TOL)
    msg = (f"status={status} reported={reported!r} p={cert.p!r} d={cert.d!r} "
           f"uncertainty={unc:.3e} relgap={gap:.3e} roundoff={cert.roundoff:.3e} "
           f"pimpact={cert.pimpact:.3e} "
           f"repair_iters={cert.repair_iters} method={cert.method} "
           f"verify_pinf={cert.pinf:.3e} "
           f"reported_vs_p={objdiff:.3e} c0={prob['c0']!r}")
    return ok, cert.p, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--time-limit", type=float, default=None, help="各ソルバの時間上限 [s]")
    ap.add_argument("--csv", action="store_true", help="baseline CSV 行を stdout へ出す")
    args = ap.parse_args()
    date = datetime.date.today().strftime("%Y%m%d")
    # --csv の stdout は CSV 行だけにする。ソルバの C ライブラリ (HiGHS/Clarabel) が
    # stdout へ出す LAPACK 警告は、プロセス全体で fd 1 を stderr へ向けて逃がし、
    # CSV は元の stdout の複製 fd へ書く。C のバッファは終了時に fd 1 (= stderr) へ
    # flush されるので、CSV に混入しない。
    if args.csv:
        csv_fd = os.dup(1)
        os.dup2(2, 1)
        csv_out = os.fdopen(csv_fd, "w", buffering=1)
    else:
        csv_out = sys.stdout
    diag = sys.stderr if args.csv else sys.stdout
    try:
        bad = 0
        for path in args.paths:
            name = os.path.splitext(os.path.basename(path))[0]
            prob = load_qps(path)
            accepted = None
            for solver, version, solve in SOLVERS:
                solved = solve(prob, args.time_limit)
                ok, p, msg = certify(prob, solved)
                print(f"{name},{solver},{'PASS' if ok else 'FAIL'},{msg}",
                      file=diag, flush=True)
                if ok:
                    status = solved[0].replace(" ", "")
                    accepted = (p, f"{solver}_{version()}_{status}_{date}")
                    break
            if accepted is None:
                bad += 1
                print(f"FAIL {name}: no solver passed the certificate",
                      file=sys.stderr, flush=True)
            elif args.csv:
                print(f"{name},{format_csv_value(accepted[0])},{accepted[1]}",
                      file=csv_out, flush=True)
        return 1 if bad else 0
    finally:
        if args.csv:
            # fd 1 は stderr のまま終了する。C/Fortran の stdout バッファが終了時に
            # flush されても CSV ファイルへ混入しない (CSV は複製 fd に書いている)。
            csv_out.close()


if __name__ == "__main__":
    sys.exit(main())
