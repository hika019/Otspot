#!/usr/bin/env python3
"""Maros-Meszaros QPS 問題を HiGHS (highspy) で解き、baseline CSV 用の参照値を出力する。

HiGHS の解 (x, y) を、QPS を独立にパースした (A, c, Q, bounds, 行種別) から再計算し、
主実行可能性 (Ax と行種別) と弱双対 gap を検算する。HiGHS の status が Optimal でない、
または検算が許容値 (REL_GAP_TOL / PRIMAL_INF_TOL) を満たさない (NaN を含む) 問題は
非ゼロ終了する。

対応範囲: NAME/ROWS/COLUMNS/BOUNDS(FR のみ)/QUADOBJ/ENDATA のみの free format で、
RHS なし (b=0)・全変数 FR・N 行は 1 本 (LISWET 家族)。これ以外のセクション
(RHS, RANGES, QMATRIX, QSECTION, OBJSENSE 等)、FR 以外の境界、MARKER (整数変数)、
G/L/E 以外の行種別は ValueError で拒否する (黙って誤値を出さない)。

Usage:
  python3 scripts/highs_baseline_truth.py data/maros_meszaros/LISWET7.QPS ...
  python3 scripts/highs_baseline_truth.py --csv data/maros_meszaros/LISWET7.QPS ...

通常出力 (stdout, 1 問 1 行):
  problem,objective,status,highs_pinf,highs_dinf,verify_pinf,p=,d=,relgap=,ysign_fix=
  highs_pinf/highs_dinf は HiGHS 自己申告、verify_* は独立再計算、p/d は再計算した
  主/双対目的値、ysign_fix は row dual の符号射影量。
--csv: 検算 (HiGHS 報告値と p(x) の相対差 OBJ_CONSISTENCY_TOL 以下を含む) を通った問題だけ、
  独立再計算した p(x) を値とする baseline CSV 行 `NAME,<mantissa>e<exp>,highs_<ver>_<status>_<date>`
  を stdout に出し、診断行は stderr へ回す。
"""
import argparse
import datetime
import os
import sys
import tempfile

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# baseline 比較の許容 eps (bench 基準, CLAUDE.md: eps=1e-6)。参照値の誤差は比較側の
# 許容より十分小さくなければならないので、検算許容は eps の 1/10 以下に取る。
BENCH_EPS = 1e-6
# 相対 gap |p-d|/(1+|p|) の上限。参照値の不確かさ上界 = BENCH_EPS/10。
# LISWET 家族 7 問の実測最大は 9.4e-9 (LISWET10) で十分余裕がある。
REL_GAP_TOL = BENCH_EPS / 10
# 行違反 (b=0 なので絶対値) の上限。目的値へ効く誤差が BENCH_EPS の 1e-3 倍以下に
# なる水準。実測最大は 8.2e-15。
PRIMAL_INF_TOL = BENCH_EPS * 1e-3
# HiGHS 報告目的値と独立再計算 p(x) の相対差の上限。両者は同じ x の目的値なので浮動小数の
# 丸め (~1e-12) しか差が出ないはずで、超えたら QPS 解釈 (Q, c, 符号) の食い違いを示す。
OBJ_CONSISTENCY_TOL = BENCH_EPS * 1e-3
# baseline CSV の仮数部の小数桁数 (小数点以下 12 桁 = 既存行と同じ 13 有効桁)。
CSV_MANTISSA_DECIMALS = 12

SUPPORTED_SECTIONS = {"NAME", "ROWS", "COLUMNS", "BOUNDS", "QUADOBJ", "ENDATA"}
ROW_TYPES = {"G", "L", "E"}


def parse_qps(path):
    """(c, A(csr), row_types, Q(csr, 対称), n) を返す。docstring の対応範囲外は ValueError。"""
    section = None
    rows, cols, qent = {}, {}, []
    row_type = []
    obj = None
    c, a_i, a_j, a_v = {}, [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("*"):
                continue
            if not line.startswith(" "):
                tok = line.split()[0]
                if tok not in SUPPORTED_SECTIONS:
                    raise ValueError(f"unsupported section {tok}")
                section = tok
                continue
            t = line.split()
            if section == "ROWS":
                if t[0] == "N":
                    if obj is not None:
                        raise ValueError("multiple N rows")
                    obj = t[1]
                elif t[0] in ROW_TYPES:
                    rows[t[1]] = len(row_type)
                    row_type.append(t[0])
                else:
                    raise ValueError(f"row type {t[0]}")
            elif section == "COLUMNS":
                if "'MARKER'" in t:
                    raise ValueError("integer MARKER unsupported")
                j = cols.setdefault(t[0], len(cols))
                for r, v in zip(t[1::2], t[2::2]):
                    if r == obj:
                        c[j] = c.get(j, 0.0) + float(v)
                    else:
                        a_i.append(rows[r]); a_j.append(j); a_v.append(float(v))
            elif section == "BOUNDS":
                if t[0] != "FR":
                    raise ValueError(f"bound type {t[0]}")
            elif section == "QUADOBJ":
                qent.append((cols[t[0]], cols[t[1]], float(t[2])))
            else:
                raise ValueError(f"data line in section {section}")
    n, m = len(cols), len(row_type)
    cv = np.zeros(n)
    for j, v in c.items():
        cv[j] = v
    A = sp.csr_matrix((a_v, (a_i, a_j)), shape=(m, n))
    qi = [e[0] for e in qent] + [e[1] for e in qent if e[0] != e[1]]
    qj = [e[1] for e in qent] + [e[0] for e in qent if e[0] != e[1]]
    qv = [e[2] for e in qent] + [e[2] for e in qent if e[0] != e[1]]
    Q = sp.csr_matrix((qv, (qi, qj)), shape=(n, n))
    return cv, A, row_type, Q, n


def verify(path, x, y):
    """独立再計算: 主実行不能度 pinf と弱双対 gap (p(x) - d(y))。

    min 1/2 x'Qx + c'x  s.t. G: Ax>=0, L: Ax<=0, E: Ax=0 (b=0), x free。
    y は HiGHS の row dual を、行種別に合う符号 (G:y>=0, L:y<=0) へ射影した値で使う。
    d(y) = min_x 1/2 x'Qx + (c - A'y)'x = -1/2 z'Q^+ z (z = c - A'y が range(Q) 内のとき)。
    Q が正則 (LISWET は Q=I) なら Q^{-1} を解いて評価する。
    """
    c, A, rt, Q, n = parse_qps(path)
    ax = A @ x
    sign = np.array([{"G": -1.0, "L": 1.0, "E": 0.0}[t] for t in rt])
    viol = np.where(np.array(rt) == "E", np.abs(ax), np.maximum(sign * ax, 0.0))
    pinf = float(viol.max()) if len(viol) else 0.0
    p = float(0.5 * x @ (Q @ x) + c @ x)
    # y を符号制約へ射影 (min 問題: G の multiplier>=0, L の multiplier<=0)
    yp = y.copy()
    yp[np.array(rt) == "G"] = np.maximum(yp[np.array(rt) == "G"], 0.0)
    yp[np.array(rt) == "L"] = np.minimum(yp[np.array(rt) == "L"], 0.0)
    sign_flip = float(np.abs(yp - y).max()) if len(y) else 0.0
    z = c - A.T @ yp
    w = spsolve(Q.tocsc(), z)
    d = float(-0.5 * z @ w)
    return p, d, pinf, sign_flip


def format_csv_value(v):
    """baseline CSV の値表記 (例 -2.489287597915e3)。"""
    mant, exp = f"{v:.{CSV_MANTISSA_DECIMALS}e}".split("e")
    return f"{mant}e{int(exp)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--gap-tol", type=float, default=REL_GAP_TOL, help="相対 gap 上限")
    ap.add_argument("--csv", action="store_true", help="baseline CSV 行を stdout へ出す")
    args = ap.parse_args()
    version = highspy.Highs().version()
    date = datetime.date.today().strftime("%Y%m%d")
    diag = sys.stderr if args.csv else sys.stdout
    bad = 0
    for path in args.paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with tempfile.TemporaryDirectory() as td:
            link = os.path.join(td, name + ".mps")
            os.symlink(os.path.abspath(path), link)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(link)
            h.run()
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        sol = h.getSolution()
        x = np.array(sol.col_value)
        y = np.array(sol.row_dual)
        p, d, pinf, sflip = verify(path, x, y)
        gap = abs(p - d) / (1.0 + abs(p))
        print(f"{name},{info.objective_function_value!r},{status},"
              f"{info.max_primal_infeasibility:.3e},{info.max_dual_infeasibility:.3e},"
              f"{pinf:.3e},p={p!r},d={d!r},relgap={gap:.3e},ysign_fix={sflip:.3e}",
              file=diag)
        # NaN は比較が常に False になるので、合格条件を肯定形で書き否定を取る。
        hobj = info.objective_function_value
        objdiff = abs(p - hobj) / (1.0 + abs(p))
        ok = (status == "Optimal" and gap <= args.gap_tol and pinf <= PRIMAL_INF_TOL
              and objdiff <= OBJ_CONSISTENCY_TOL)
        if not ok:
            bad += 1
            print(f"FAIL {name}: status={status} relgap={gap:.3e} verify_pinf={pinf:.3e} "
                  f"highs_vs_p={objdiff:.3e}",
                  file=sys.stderr)
        elif args.csv:
            print(f"{name},{format_csv_value(p)},"
                  f"highs_{version}_{status}_{date}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
