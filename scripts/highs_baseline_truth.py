#!/usr/bin/env python3
"""Maros-Meszaros QPS 問題を HiGHS (highspy) で解き、baseline CSV 用の参照値を出力する。

HiGHS の解 (x, y) を、QPS を独立にパースした (A, c, Q, bounds, 行種別) から再計算し、
主実行可能性 (Ax と行/変数境界) と弱双対 gap を検算する。HiGHS の status が Optimal で
ない、または検算 gap が --gap-tol を超える問題は非ゼロ終了する。

対応範囲: RHS/RANGES なし・全変数 FR の QUADOBJ 形式 (LISWET 家族)。それ以外の
セクション/境界種別は未対応としてエラーにする (黙って誤値を出さない)。

Usage: python3 scripts/highs_baseline_truth.py data/maros_meszaros/LISWET7.QPS ...
出力: problem,objective(repr),status,max_primal_infeas,max_dual_infeas,verify_pinf,verify_gap
"""
import argparse
import os
import sys
import tempfile

import highspy
import numpy as np
import scipy.sparse as sp

ROW_TYPES = {"G", "L", "E"}


def parse_qps(path):
    """(c, A(csr), row_types, Q(csr, 対称), n) を返す。free format、RHS/RANGES 非対応。"""
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
                if tok in ("RHS", "RANGES"):
                    raise ValueError(f"unsupported section {tok}")
                section = tok
                continue
            t = line.split()
            if section == "ROWS":
                if t[0] == "N":
                    obj = t[1] if obj is None else obj
                elif t[0] in ROW_TYPES:
                    rows[t[1]] = len(row_type)
                    row_type.append(t[0])
                else:
                    raise ValueError(f"row type {t[0]}")
            elif section == "COLUMNS":
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
    from scipy.sparse.linalg import spsolve
    w = spsolve(Q.tocsc(), z)
    d = float(-0.5 * z @ w)
    return p, d, pinf, sign_flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--gap-tol", type=float, default=1e-5, help="相対 gap 上限 (弱双対側の求解精度で決まる)")
    args = ap.parse_args()
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
              f"{pinf:.3e},p={p!r},d={d!r},relgap={gap:.3e},ysign_fix={sflip:.3e}")
        if status != "Optimal" or gap > args.gap_tol or pinf > 1e-7:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
