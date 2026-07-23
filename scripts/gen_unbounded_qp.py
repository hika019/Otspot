"""QP unbounded problem generator (QPLIB format).

数学的正当性:
  minimize (1/2) x^T Q x + c^T x  s.t.  lb <= Ax <= ub,  lo <= x <= hi
  unbounded の条件: feasible ray d が存在して
    (1) Q d = 0  (d が Q の null space にある)
    (2) c^T d < 0  (d 方向に目的が減少)
    (3) A d = 0  (等号制約を満たす) または  lb <= A(x+td) <= ub for all t > 0

  各パターンで以下を設計:
    - null(Q) を明示的に設定
    - c を null(Q) 方向に負になるよう設定
    - 制約を ray d が feasible のまま保てるよう設計
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple


INF = 1e30


class QpProblem(NamedTuple):
    name: str
    n: int
    q_lower_tri: list[tuple[int, int, float]]  # 1-indexed (i>=j)
    c: list[float]
    con_type: str   # 'L' or 'B'
    a_triplets: list[tuple[int, int, float]]   # (row1idx, col1idx, val)
    m: int
    con_bounds: list[tuple[float, float]]      # (lb, ub) per constraint
    var_bounds: list[tuple[float, float]]      # (lb, ub) per variable


def write_qplib(prob: QpProblem, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(prob.name)
    lines.append(f"QC{prob.con_type}")
    lines.append("minimize")
    lines.append(f"{prob.n}")
    if prob.con_type == 'L':
        lines.append(f"{prob.m}")

    lines.append(f"{len(prob.q_lower_tri)}")
    for i, j, v in prob.q_lower_tri:
        lines.append(f"{i} {j} {v:.15g}")

    nondefault_c = [(i + 1, v) for i, v in enumerate(prob.c) if v != 0.0]
    lines.append("0.0")
    lines.append(f"{len(nondefault_c)}")
    for idx, v in nondefault_c:
        lines.append(f"{idx} {v:.15g}")

    lines.append("0.0")

    if prob.con_type == 'L':
        lines.append(f"{len(prob.a_triplets)}")
        for k, i, v in prob.a_triplets:
            lines.append(f"{k} {i} {v:.15g}")

    lines.append(f"{INF:.6g}")

    if prob.con_type == 'L':
        lb_con = [lb for lb, _ in prob.con_bounds]
        ub_con = [ub for _, ub in prob.con_bounds]

        lines.append(f"{-INF:.6g}")
        nondefault_lb = [(k + 1, lb) for k, lb in enumerate(lb_con) if lb > -INF * 0.99]
        lines.append(f"{len(nondefault_lb)}")
        for k, lb in nondefault_lb:
            lines.append(f"{k} {lb:.15g}")

        lines.append(f"{INF:.6g}")
        nondefault_ub = [(k + 1, ub) for k, ub in enumerate(ub_con) if ub < INF * 0.99]
        lines.append(f"{len(nondefault_ub)}")
        for k, ub in nondefault_ub:
            lines.append(f"{k} {ub:.15g}")

    lb_var = [lb for lb, _ in prob.var_bounds]
    ub_var = [ub for _, ub in prob.var_bounds]

    lines.append("0.0")
    nondefault_lb_var = [(i + 1, lb) for i, lb in enumerate(lb_var) if lb != 0.0]
    lines.append(f"{len(nondefault_lb_var)}")
    for i, lb in nondefault_lb_var:
        lines.append(f"{i} {lb:.15g}")

    lines.append(f"{INF:.6g}")
    nondefault_ub_var = [(i + 1, ub) for i, ub in enumerate(ub_var) if ub < INF * 0.99]
    lines.append(f"{len(nondefault_ub_var)}")
    for i, ub in nondefault_ub_var:
        lines.append(f"{i} {ub:.15g}")

    # QPLIB optional tail: starting-point (primal/constraint-dual/bound-dual)
    # + names sections. All default/empty here. constraint-dual is present
    # only when the problem has general constraints (con_type in 'L','Q');
    # the constraint-names count is unconditional (present even for con_type='B').
    lines.append("0.0")  # default variable primal value in starting point
    lines.append("0")    # number of non-default variable primal values
    if prob.con_type in ('L', 'Q'):
        lines.append("0.0")  # default constraint dual value in starting point
        lines.append("0")    # number of non-default constraint dual values
    lines.append("0.0")  # default variable bound dual value in starting point
    lines.append("0")    # number of non-default variable bound dual values
    lines.append("0")    # number of non-default variable names
    lines.append("0")    # number of non-default constraint names

    out_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# QP unbounded 問題生成関数
# =============================================================================

def gen_qp_unbd_q0_free_n2() -> QpProblem:
    """Q=0 (LP退化) + 自由変数 → 非有界.

    minimize c^T x = -x1 - x2  (Q=0)
    制約なし, x1 x2 free
    ray d=(1,1): c^T d = -2 < 0, Q d = 0 なので (1/2) d^T Q d = 0.
    目的 f(x+td) = -x1-x2 + t*(-2) → -∞.
    """
    name = "UNBD_QP_Q0_FREE_N2"
    n = 2
    q_lower_tri: list = []   # Q=0
    c = [-1.0, -1.0]
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


def gen_qp_unbd_q0_ineq_n3() -> QpProblem:
    """Q=0 + 不等号制約が緩い → 非有界.

    minimize -x1 - x2 - x3  (Q=0)
    制約: x1 + x2 >= 1 (lb=1, G 型)
    x1, x2, x3 >= 0

    ray d=(0,0,1): c^T d = -1 < 0, Q d = 0,
      A*d: C1: 0+0 = 0 >= 0 OK (元の制約 >= 1 は feasible point で満たし、
      ray 方向 d=(0,0,1) は C1 に影響しない)。
    x3 >= 0 で d3=1 > 0 OK。
    """
    name = "UNBD_QP_Q0_INEQ_N3"
    n = 3
    q_lower_tri: list = []
    c = [-1.0, -1.0, -1.0]
    a_triplets = [(1, 1, 1.0), (1, 2, 1.0)]
    m = 1
    con_bounds = [(1.0, INF)]
    var_bounds = [(0.0, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_qp_unbd_rank_deficient_n3() -> QpProblem:
    """Q が rank-deficient (rank=2, null space=1 次元) + null space 方向に c < 0.

    Q = diag(1, 1, 0) → null space = span{e3}
    c = (0, 0, -1) → c^T e3 = -1 < 0

    制約なし, x3 free。
    f(x + t*e3) = (1/2)(x1^2 + x2^2) + c1 x1 + c2 x2 + (-1)(x3 + t) → -∞ as t→∞.
    """
    name = "UNBD_QP_RANKDEF_N3"
    n = 3
    q_lower_tri = [(1, 1, 1.0), (2, 2, 1.0)]  # Q33 = 0 (省略)
    c = [0.0, 0.0, -1.0]
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


def gen_qp_unbd_null_space_n4() -> QpProblem:
    """Q = diag(2,2,0,0) → null space = span{e3, e4}.

    c = (0, 0, -1, -2) → null space 方向 d=(0,0,1,1) で c^T d = -3 < 0.
    制約: x1 + x2 = 1 (等号), x3, x4 free.
    ray d=(0,0,1,1): A*d = 0 (C1 に x3, x4 係数なし) OK.
    Q*d = 0 (Q33=Q44=0) OK.
    """
    name = "UNBD_QP_NULLSP_N4"
    n = 4
    q_lower_tri = [(1, 1, 2.0), (2, 2, 2.0)]  # Q=diag(2,2,0,0)
    c = [0.0, 0.0, -1.0, -2.0]
    a_triplets = [(1, 1, 1.0), (1, 2, 1.0)]
    m = 1
    con_bounds = [(1.0, 1.0)]  # equality
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_qp_unbd_q0_eq_only_n5() -> QpProblem:
    """Q=0 + 等号制約のみ → null space に c < 0 方向が存在.

    n=5, m=2 等号制約
    A = [[1,1,0,0,0], [0,0,1,1,0]] → null space 次元 = 3
    b = [1, 1]
    c = (0, 0, 0, 0, -1) → null space ベクトル e5 で c^T e5 = -1 < 0.
    x5 free → ray d=e5 が feasible。
    """
    name = "UNBD_QP_Q0_EQ_N5"
    n = 5
    q_lower_tri: list = []   # Q=0
    c = [0.0, 0.0, 0.0, 0.0, -1.0]
    a_triplets = [
        (1, 1, 1.0), (1, 2, 1.0),
        (2, 3, 1.0), (2, 4, 1.0),
    ]
    m = 2
    con_bounds = [(1.0, 1.0), (1.0, 1.0)]
    var_bounds = [(0.0, INF), (0.0, INF), (0.0, INF), (0.0, INF), (-INF, INF)]
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_qp_unbd_box_no_ub_n6() -> QpProblem:
    """Q PSD (半正定値) + 上界なし変数 → 非有界.

    Q = diag(1, 1, 1, 1, 0, 0) → rank 4, null space = span{e5, e6}
    c = (0, 0, 0, 0, -1, -2)
    no constraints (box のみ), x5, x6 >= 0 (下界 0, 上界 +inf).
    ray d = (0,0,0,0,1,1): Q d = 0, c^T d = -3 < 0,
      x5 >= 0 で d5=1 > 0 OK, x6 >= 0 で d6=1 > 0 OK.
    """
    name = "UNBD_QP_BOX_NOUB_N6"
    n = 6
    q_lower_tri = [(1,1,1.0),(2,2,1.0),(3,3,1.0),(4,4,1.0)]
    c = [0.0, 0.0, 0.0, 0.0, -1.0, -2.0]
    var_bounds = [(-INF, INF)] * 4 + [(0.0, INF)] * 2
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


def gen_qp_unbd_sparse_a_n8() -> QpProblem:
    """n=8, スパース A + rank-deficient Q → 非有界.

    Q = diag(1,1,1,1,1,1,0,0) → null = {e7, e8}
    c = (0,...,0,-3,-2)
    制約: x1+x2 <= 10, x3+x4 >= -5, x5+x6 = 0
    ray d=(0,...,0,1,1) (e7+e8 方向):
      Q*d = 0 (Q77=Q88=0)
      c^T d = -3 - 2 = -5 < 0
      C1: 0 <= 10 OK
      C2: 0 >= -5 OK
      C3: 0 = 0 OK (等号制約 x5+x6=0 には影響しない)
    x7, x8 free (-INF, INF) なので d7=d8=1 可能。
    """
    name = "UNBD_QP_SPARSE_A_N8"
    n = 8
    q_lower_tri = [(i+1,i+1,1.0) for i in range(6)]   # Q=diag(1,1,1,1,1,1,0,0)
    c = [0.0]*6 + [-3.0, -2.0]
    a_triplets = [
        (1,1,1.0),(1,2,1.0),           # x1+x2 <= 10
        (2,3,1.0),(2,4,1.0),           # x3+x4 >= -5
        (3,5,1.0),(3,6,1.0),           # x5+x6 = 0
    ]
    m = 3
    con_bounds = [(-INF, 10.0), (-5.0, INF), (0.0, 0.0)]
    var_bounds = [(-INF, INF)]*8
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_qp_unbd_large_null_n10() -> QpProblem:
    """n=10, null space が広い (rank 5, null dim 5).

    Q = diag(2,2,2,2,2,0,0,0,0,0)
    c = (0,...,0,-1,-2,-1,-2,-1)
    制約: x1+x2 = 1 (等号), x3+x4+x5 >= 0
    ray d = (0,0,0,0,0,1,1,1,1,1): null(Q) 方向
      c^T d = -1-2-1-2-1 = -7 < 0
      C1: 0 = 0 (d1=d2=0) OK
      C2: 0 >= 0 OK
    x6..x10 free → d6..d10 = 1 可能。
    """
    name = "UNBD_QP_LARGE_NULL_N10"
    n = 10
    q_lower_tri = [(i+1,i+1,2.0) for i in range(5)]
    c = [0.0]*5 + [-1.0,-2.0,-1.0,-2.0,-1.0]
    a_triplets = [
        (1,1,1.0),(1,2,1.0),
        (2,3,1.0),(2,4,1.0),(2,5,1.0),
    ]
    m = 2
    con_bounds = [(1.0,1.0), (0.0,INF)]
    var_bounds = [(-INF,INF)]*n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_qp_unbd_q0_no_con_n15() -> QpProblem:
    """Q=0, 制約なし, x >= 0, 全係数が負 → 非有界.

    ray d=(1,...,1): c^T d = sum(c) < 0.
    """
    name = "UNBD_QP_Q0_NOCON_N15"
    n = 15
    q_lower_tri: list = []
    rng = random.Random(7)
    c = [-rng.uniform(0.5, 3.0) for _ in range(n)]
    var_bounds = [(0.0, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


# =============================================================================
# PROBLEMS リスト
# =============================================================================

PROBLEMS = [
    gen_qp_unbd_q0_free_n2,
    gen_qp_unbd_q0_ineq_n3,
    gen_qp_unbd_rank_deficient_n3,
    gen_qp_unbd_null_space_n4,
    gen_qp_unbd_q0_eq_only_n5,
    gen_qp_unbd_box_no_ub_n6,
    gen_qp_unbd_sparse_a_n8,
    gen_qp_unbd_large_null_n10,
    gen_qp_unbd_q0_no_con_n15,
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="QP unbounded problem generator (QPLIB format)")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "qp_unbounded"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen_fn in PROBLEMS:
        prob = gen_fn()
        out_path = out_dir / f"{prob.name}.qplib"
        write_qplib(prob, out_path)
        print(f"  {prob.name}: n={prob.n} m={prob.m} nnz_Q={len(prob.q_lower_tri)} -> {out_path.name}")

    print(f"\n生成完了: {len(PROBLEMS)} 問題 -> {out_dir}")


if __name__ == "__main__":
    main()
