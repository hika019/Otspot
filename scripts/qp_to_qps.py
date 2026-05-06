"""OSQP-form (P, q, A, l, u) を QPS (free-form) に書き出す。

OSQP: min 1/2 x' P x + q' x  s.t.  l <= A x <= u
QPS  (qps.rs パーサ仕様):
  - N 行: 目的関数。COLUMNS の係数 = q
  - L/G/E/RANGES: 制約。l/u の組み合わせから決定
  - BOUNDS: 全変数 FR (OSQP は明示の変数 bound を持たない)
  - QUADOBJ: upper-triangular of P (1/2 規約)

注意:
  - 数値は Python の repr (full precision) で書く。`%.17g` で round-trip 保証。
  - 行名/列名は固定形式幅 8 文字を超えないよう zero-pad。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as spa


# 数値書式: round-trip 保証 (IEEE754 double)
NUM_FMT = "{:.17g}"
INF = float("inf")


def _fmt(v: float) -> str:
    return NUM_FMT.format(float(v))


def _col_name(i: int) -> str:
    # X00000001 形式 (8 文字以内)
    return f"X{i + 1:07d}"


def _row_name(i: int) -> str:
    return f"R{i + 1:07d}"


def write_qps(
    name: str,
    P: spa.spmatrix,
    q: np.ndarray,
    A: spa.spmatrix,
    l: np.ndarray,
    u: np.ndarray,
    out_path: Path,
    var_lb: np.ndarray | None = None,
    var_ub: np.ndarray | None = None,
) -> None:
    """OSQP-form 問題を QPS に書き出す。

    Parameters
    ----------
    name : QPS NAME (8 文字を超えると一部パーサで切れる可能性。警告のみ)
    P    : (n,n) 対称半正定値、Hessian。1/2 x'Px + q'x の規約
    q    : (n,)
    A    : (m,n) 制約行列 (l <= Ax <= u)
    l, u : (m,) 各成分は finite か ±inf
    out_path : 出力 .qps パス
    var_lb, var_ub : (n,) 変数 bound。None の場合は全変数 FR。
                     成分ごとに ±inf 可。
    """
    n = P.shape[0]
    m = A.shape[0]
    assert P.shape == (n, n)
    assert q.shape == (n,)
    assert A.shape == (m, n)
    assert l.shape == (m,) and u.shape == (m,)

    P = spa.csc_matrix(P)
    A = spa.csc_matrix(A)

    if len(name) > 8:
        # NAME 行は parser 側で問題名を読まないので致命的ではないが、識別性のため警告
        print(f"[qp_to_qps] warn: NAME='{name}' は 8 文字超 (parser 影響なし)")

    obj_row = "OBJ"

    # --- 各制約の type/rhs/range_val を決定 ---
    rtype: list[str] = []         # 'L' / 'G' / 'E' / None (skip)
    rhs: list[float] = []
    range_val: list[float | None] = []
    for i in range(m):
        li, ui = float(l[i]), float(u[i])
        l_inf = math.isinf(li) and li < 0
        u_inf = math.isinf(ui) and ui > 0
        if l_inf and u_inf:
            rtype.append("")
            rhs.append(0.0)
            range_val.append(None)
        elif l_inf:
            rtype.append("L")
            rhs.append(ui)
            range_val.append(None)
        elif u_inf:
            rtype.append("G")
            rhs.append(li)
            range_val.append(None)
        elif li == ui:
            rtype.append("E")
            rhs.append(li)
            range_val.append(None)
        else:
            # 両端 finite で l < u: L 型 + RANGES (parser: L で b=u, |r|=u-l → lower=l)
            if li > ui:
                raise ValueError(f"row {i}: l={li} > u={ui} (infeasible bound)")
            rtype.append("L")
            rhs.append(ui)
            range_val.append(ui - li)

    active_rows = [i for i in range(m) if rtype[i]]

    # --- COLUMNS 用: 各列に対し (row, val) リストを構築 ---
    # 目的関数 (q) と制約 (A 列) をマージ
    A_csc = A
    cols_data: list[list[tuple[str, float]]] = [[] for _ in range(n)]
    # q
    for j in range(n):
        if q[j] != 0.0:
            cols_data[j].append((obj_row, float(q[j])))
    # A
    indptr = A_csc.indptr
    indices = A_csc.indices
    data = A_csc.data
    for j in range(n):
        for k in range(indptr[j], indptr[j + 1]):
            i = indices[k]
            if not rtype[i]:
                continue  # 両端 ∞ の制約 → スキップ
            v = float(data[k])
            if v == 0.0:
                continue
            cols_data[j].append((_row_name(i), v))

    # --- QUADOBJ: P の upper triangular (i <= j; col1=col_i, col2=col_j) ---
    # qps.rs の規約: parts[0]=col_idx_a, parts[1]=col_idx_b, parts[2]=value で
    # symmetry を補完する。i==j は対角。i!=j は片側のみ。
    P_upper = spa.triu(P, k=0).tocoo()
    quad_entries: list[tuple[int, int, float]] = []
    for i, j, v in zip(P_upper.row, P_upper.col, P_upper.data):
        if v == 0.0:
            continue
        # COLUMNS 順 (列 i, 列 j) で書き出すため (i,j) で出力。i<=j 保証。
        quad_entries.append((int(i), int(j), float(v)))

    # --- 出力 ---
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"NAME          {name}\n")
        f.write("ROWS\n")
        f.write(f" N  {obj_row}\n")
        for i in active_rows:
            f.write(f" {rtype[i]}  {_row_name(i)}\n")

        f.write("COLUMNS\n")
        for j in range(n):
            entries = cols_data[j]
            if not entries:
                # 列が完全に 0 でも、変数として残す必要があるなら obj 行に 0 を書く。
                # ただし parser は COLUMNS で初出の列を変数登録するため、欠落列は変数に含まれない。
                # OSQP 問題で q[j]=0 かつ A[:,j]=0 はめったに無いが、念のため obj に 0 を出す。
                f.write(f"    {_col_name(j)}  {obj_row}  0\n")
                continue
            for (rname, val) in entries:
                f.write(f"    {_col_name(j)}  {rname}  {_fmt(val)}\n")

        # RHS
        rhs_lines = [(i, rhs[i]) for i in active_rows if rhs[i] != 0.0]
        if rhs_lines:
            f.write("RHS\n")
            for (i, v) in rhs_lines:
                f.write(f"    RHS  {_row_name(i)}  {_fmt(v)}\n")

        # RANGES
        rng_lines = [(i, range_val[i]) for i in active_rows if range_val[i] is not None]
        if rng_lines:
            f.write("RANGES\n")
            for (i, v) in rng_lines:
                f.write(f"    RNG  {_row_name(i)}  {_fmt(v)}\n")

        # BOUNDS
        # var_lb/var_ub 未指定 → 全変数 FR
        # 指定時: 成分ごとに以下を出力 (MPS 標準):
        #   lb=-inf, ub=+inf : FR
        #   lb=-inf, ub finite : MI + UP <ub>
        #   lb finite, ub=+inf : LO <lb> (デフォルト ub=+inf)
        #   lb finite, ub finite, lb==ub : FX <lb>
        #   lb finite, ub finite, lb<ub  : LO <lb> + UP <ub>
        # qps.rs パーサのデフォルト: 値未指定の bound type の既定を確認すること。
        #   MPS 標準では LO 未指定 → 0、UP 未指定 → +∞。
        #   ここでは下限が 0 でない問題でも明示的に LO を書くため OK。
        f.write("BOUNDS\n")
        if var_lb is None and var_ub is None:
            for j in range(n):
                f.write(f" FR BND  {_col_name(j)}\n")
        else:
            lb_arr = (np.full(n, -INF) if var_lb is None else np.asarray(var_lb, dtype=float))
            ub_arr = (np.full(n,  INF) if var_ub is None else np.asarray(var_ub, dtype=float))
            assert lb_arr.shape == (n,) and ub_arr.shape == (n,)
            for j in range(n):
                lj, uj = float(lb_arr[j]), float(ub_arr[j])
                l_inf = math.isinf(lj) and lj < 0
                u_inf = math.isinf(uj) and uj > 0
                if l_inf and u_inf:
                    f.write(f" FR BND  {_col_name(j)}\n")
                elif l_inf:
                    f.write(f" MI BND  {_col_name(j)}\n")
                    f.write(f" UP BND  {_col_name(j)}  {_fmt(uj)}\n")
                elif u_inf:
                    # ub=+inf を明示するため PL は省略可。MPS 既定 ub=+inf。
                    # ただし lb!=0 の場合 LO で下限を上書きする必要あり。
                    # MPS 既定 lb=0 のため lb=0 でも明示的に LO 0 を書いて確定させる。
                    f.write(f" LO BND  {_col_name(j)}  {_fmt(lj)}\n")
                elif lj == uj:
                    f.write(f" FX BND  {_col_name(j)}  {_fmt(lj)}\n")
                else:
                    if lj > uj:
                        raise ValueError(f"var {j}: lb={lj} > ub={uj}")
                    f.write(f" LO BND  {_col_name(j)}  {_fmt(lj)}\n")
                    f.write(f" UP BND  {_col_name(j)}  {_fmt(uj)}\n")

        # QUADOBJ
        if quad_entries:
            f.write("QUADOBJ\n")
            for (i, j, v) in quad_entries:
                f.write(f"    {_col_name(i)}  {_col_name(j)}  {_fmt(v)}\n")

        f.write("ENDATA\n")


def write_osqp_problem(name: str, qp_problem: dict, out_path: Path) -> None:
    """OSQP benchmarks の `qp_problem` dict (P,q,A,l,u,n,m) から書き出す。"""
    write_qps(
        name=name,
        P=qp_problem["P"],
        q=qp_problem["q"],
        A=qp_problem["A"],
        l=qp_problem["l"],
        u=qp_problem["u"],
        out_path=out_path,
    )
