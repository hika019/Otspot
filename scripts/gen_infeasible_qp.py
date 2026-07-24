"""QP infeasible problem generator (QPLIB format).

数学的正当性:
  infeasible QP = 実行可能領域が空集合。
  各パターンで矛盾を明示的に設計する。

パターン別根拠:
  1. 等号制約の矛盾: Ax = b で b が range(A) 外
  2. lb > ub 変数: 変数の下界が上界を超える
  3. 不等式の矛盾: Ax <= b1 かつ Ax >= b2 (b2 > b1 で矛盾)
  4. 等号 + 不等号の混在矛盾
  5. 変数境界と制約の矛盾
  6. 大規模 infeasible
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
    q_lower_tri: list[tuple[int, int, float]]
    c: list[float]
    con_type: str
    a_triplets: list[tuple[int, int, float]]
    m: int
    con_bounds: list[tuple[float, float]]
    var_bounds: list[tuple[float, float]]


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
# QP infeasible 問題生成関数
# =============================================================================

def gen_infeas_bound_contradiction_n1() -> QpProblem:
    """lb > ub: x1 in [5, 3] → 矛盾 (直接矛盾).

    lb=5 > ub=3 → 実行不可能。
    """
    name = "INFEAS_QP_BOUND_N1"
    n = 1
    q_lower_tri = [(1, 1, 1.0)]
    c = [1.0]
    var_bounds = [(5.0, 3.0)]  # lb > ub !
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


def gen_infeas_bound_contradiction_n3() -> QpProblem:
    """複数変数の lb > ub 矛盾.

    x1 in [10, 5], x2 in [0, -1], x3 in [3, 1].
    """
    name = "INFEAS_QP_BOUND_N3"
    n = 3
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [1.0]*n
    var_bounds = [(10.0, 5.0), (0.0, -1.0), (3.0, 1.0)]
    return QpProblem(name, n, q_lower_tri, c, 'B', [], 0, [], var_bounds)


def gen_infeas_eq_contradiction_n2() -> QpProblem:
    """等号制約の矛盾: x1 + x2 = 5 かつ x1 + x2 = 10.

    同一行ベクトル [1,1] に対して異なる右辺値 → infeasible。
    """
    name = "INFEAS_QP_EQ_N2"
    n = 2
    q_lower_tri = [(1,1,1.0),(2,2,1.0)]
    c = [0.0, 0.0]
    a_triplets = [
        (1,1,1.0),(1,2,1.0),  # row1: x1+x2
        (2,1,1.0),(2,2,1.0),  # row2: x1+x2 (同じ)
    ]
    m = 2
    # row1: x1+x2 = 5, row2: x1+x2 = 10 → 矛盾
    con_bounds = [(5.0, 5.0), (10.0, 10.0)]
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_ineq_contradiction_n3() -> QpProblem:
    """不等式の矛盾: x1+x2+x3 <= 5 かつ x1+x2+x3 >= 10.

    同一線形式に矛盾する上下界。
    """
    name = "INFEAS_QP_INEQ_N3"
    n = 3
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [1.0]*n
    a_triplets = [
        (1,1,1.0),(1,2,1.0),(1,3,1.0),  # row1: x1+x2+x3
        (2,1,1.0),(2,2,1.0),(2,3,1.0),  # row2: x1+x2+x3
    ]
    m = 2
    # row1 <= 5, row2 >= 10 → 矛盾
    con_bounds = [(-INF, 5.0), (10.0, INF)]
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_eq_range_contradiction_n4() -> QpProblem:
    """等号制約が range(A) 外の b を要求.

    A = [[1,1,0,0],[0,0,1,1],[1,1,1,1]]
    row3 = row1 + row2 なので b3 = b1 + b2 でないと infeasible。
    b = [3, 4, 100] → b3=100 != b1+b2=7 → 矛盾。
    """
    name = "INFEAS_QP_EQ_RANGE_N4"
    n = 4
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n
    a_triplets = [
        (1,1,1.0),(1,2,1.0),              # row1: x1+x2 = 3
        (2,3,1.0),(2,4,1.0),              # row2: x3+x4 = 4
        (3,1,1.0),(3,2,1.0),(3,3,1.0),(3,4,1.0),  # row3: x1+x2+x3+x4 = 100 (矛盾: 3+4=7≠100)
    ]
    m = 3
    con_bounds = [(3.0,3.0),(4.0,4.0),(100.0,100.0)]
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_var_bound_vs_constraint_n4() -> QpProblem:
    """変数境界と制約の矛盾.

    x1 in [5, 10], x2 in [5, 10]
    制約: x1 + x2 <= 8 → max(x1+x2) = 10+10 = 20 > 8,
    しかし min(x1+x2) = 5+5 = 10 > 8 → infeasible。
    """
    name = "INFEAS_QP_VARBOUND_N4"
    n = 4
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [1.0]*n
    a_triplets = [(1,1,1.0),(1,2,1.0)]
    m = 1
    con_bounds = [(-INF, 8.0)]  # x1+x2 <= 8
    var_bounds = [(5.0, 10.0), (5.0, 10.0), (0.0, INF), (0.0, INF)]
    # x1 >= 5, x2 >= 5 → x1+x2 >= 10 > 8
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_mixed_eq_ineq_n5() -> QpProblem:
    """等号 + 不等号の混在矛盾.

    等号: x1 + x2 = 10
    不等式: x1 >= 8, x2 >= 8
    → x1+x2 >= 16 > 10 と等号 x1+x2=10 が矛盾。
    """
    name = "INFEAS_QP_MIXED_N5"
    n = 5
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [1.0]*n
    a_triplets = [
        (1,1,1.0),(1,2,1.0),   # row1: x1+x2 = 10
        (2,1,1.0),             # row2: x1 >= 8
        (3,2,1.0),             # row3: x2 >= 8
    ]
    m = 3
    con_bounds = [(10.0,10.0),(8.0,INF),(8.0,INF)]
    var_bounds = [(0.0, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_cycle_n6() -> QpProblem:
    """循環する等号制約の矛盾.

    x1 = x2 + 1 (x1 - x2 = 1)
    x2 = x3 + 1 (x2 - x3 = 1)
    x3 = x1 + 1 (x3 - x1 = 1)
    → 足し合わせると 0 = 3 → 矛盾。
    """
    name = "INFEAS_QP_CYCLE_N6"
    n = 6  # x1..x3 が問題、x4..x6 はダミー（有界）
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n
    a_triplets = [
        (1,1,1.0),(1,2,-1.0),   # row1: x1-x2 = 1
        (2,2,1.0),(2,3,-1.0),   # row2: x2-x3 = 1
        (3,3,1.0),(3,1,-1.0),   # row3: x3-x1 = 1
    ]
    m = 3
    con_bounds = [(1.0,1.0),(1.0,1.0),(1.0,1.0)]
    var_bounds = [(-INF, INF)]*3 + [(0.0, 10.0)]*3
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_overdetermined_n8() -> QpProblem:
    """過決定 (m > n) で矛盾する等号制約.

    n=8, m=10 等号制約
    A を rank n=8 で設計し、最後の 2 行は他の行の線形結合だが rhs が矛盾。
    """
    name = "INFEAS_QP_OVER_N8"
    n = 8
    rng = random.Random(100)

    # まず feasible point x0 を作る
    x0 = [rng.uniform(1.0, 5.0) for _ in range(n)]

    # 最初の n=8 制約は線形独立 (単位行列的に構成)
    a_rows = []
    b_rows = []
    for i in range(n):
        row = [0.0]*n
        row[i] = 1.0
        a_rows.append(row)
        b_rows.append(x0[i])

    # 最後の 2 制約は row0+row1 と row2+row3 の和だが、rhs を矛盾させる
    row_extra1 = [a_rows[0][j] + a_rows[1][j] for j in range(n)]
    b_extra1 = b_rows[0] + b_rows[1] + 100.0  # 矛盾: 正しくは b0+b1

    row_extra2 = [a_rows[2][j] + a_rows[3][j] for j in range(n)]
    b_extra2 = b_rows[2] + b_rows[3] + 50.0   # 矛盾

    a_rows.append(row_extra1)
    b_rows.append(b_extra1)
    a_rows.append(row_extra2)
    b_rows.append(b_extra2)

    m = len(a_rows)
    a_triplets = []
    for k, row in enumerate(a_rows):
        for j, v in enumerate(row):
            if abs(v) > 1e-12:
                a_triplets.append((k+1, j+1, v))

    con_bounds = [(b, b) for b in b_rows]

    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_large_n100() -> QpProblem:
    """大規模 infeasible QP: n=100.

    設計: まず feasible な n=100 の QP を作り、
    1 つの制約を意図的に矛盾させる。
    制約行列 A ∈ R^{50×100}: ランダムスパース
    feasible point x0 = e (all ones)
    b_i = A_i * x0 for i < 49
    b_49 = A_49 * x0 + 1000 (矛盾させる)
    加えて x1 in [1,10], ..., x100 in [1,10] → A_{49}x >= b_{49}=A_{49}x0+1000 > A_{49}*x0
    さらに x1+...+x100 <= 100 (sum <= 100, 各 x >= 1 なので sum >= 100, ちょうど等号だが
    矛盾追加制約で infeasible)。
    """
    name = "INFEAS_QP_LARGE_N100"
    n = 100
    rng = random.Random(200)

    x0 = [1.0] * n  # feasible point

    # 制約行列: 50 行 (うち 1 行が矛盾)
    m = 50
    a_triplets = []
    b_rows = []
    for k in range(m):
        row = [0.0]*n
        # 各行に 10 個の非零要素
        cols = rng.sample(range(n), 10)
        for j in cols:
            v = rng.uniform(-1.0, 1.0)
            row[j] = v
            a_triplets.append((k+1, j+1, v))
        b_k = sum(row[j]*x0[j] for j in range(n))
        if k == m-1:
            # 最後の行を矛盾させる: ub を x0 での値より大幅に小さく設定
            # row_{m-1} * x0 = b_k。各変数が [1,10] なので、
            # max(row_{m-1}*x) = sum(|row[j]|*10) が上界。
            # lb を b_k + 1000 に設定 → x が [1,10] 内では達成不可能。
            b_rows.append((b_k + 1000.0, INF))  # lb > achievable max → infeasible
        else:
            # feasible: lb = b_k - 5, ub = b_k + 5
            b_rows.append((b_k - 5.0, b_k + 5.0))

    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n
    var_bounds = [(1.0, 10.0)] * n  # x in [1,10]
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, b_rows, var_bounds)


def gen_infeas_large_n200() -> QpProblem:
    """大規模 infeasible QP: n=200.

    設計: n=200, m=80。等号制約に矛盾を注入。
    A ∈ R^{80×200}, b_k = A_k * x0 for k < 79
    row_79 = row_0 + row_1 (線形従属), b_79 = b_0 + b_1 + 500 (矛盾)
    """
    name = "INFEAS_QP_LARGE_N200"
    n = 200
    rng = random.Random(201)

    x0 = [rng.uniform(0.5, 2.0) for _ in range(n)]

    m = 80
    a_rows = []
    b_rows_val = []
    a_triplets = []

    for k in range(m-1):
        row = [0.0]*n
        cols = rng.sample(range(n), 15)
        for j in cols:
            v = rng.gauss(0, 1)
            row[j] = v
        a_rows.append(row)
        b_val = sum(row[j]*x0[j] for j in range(n))
        b_rows_val.append(b_val)

    # 最後の行: row0 + row1 (線形従属), b に矛盾を追加
    row_extra = [a_rows[0][j] + a_rows[1][j] for j in range(n)]
    b_extra = b_rows_val[0] + b_rows_val[1] + 500.0  # 矛盾
    a_rows.append(row_extra)
    b_rows_val.append(b_extra)

    for k, row in enumerate(a_rows):
        for j, v in enumerate(row):
            if abs(v) > 1e-10:
                a_triplets.append((k+1, j+1, v))

    con_bounds = [(b, b) for b in b_rows_val]

    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n
    var_bounds = [(-INF, INF)] * n
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_infeas_large_n500() -> QpProblem:
    """大規模 infeasible QP: n=500.

    設計: n=500, m=100。直接矛盾: 変数境界 + 制約の組み合わせ。
    x_i in [2.0, 5.0] for all i
    制約: sum(x) <= 100 (sum >= 2*500=1000 > 100 → infeasible)
    """
    name = "INFEAS_QP_LARGE_N500"
    n = 500
    q_lower_tri = [(i+1,i+1,1.0) for i in range(n)]
    c = [0.0]*n

    # sum(x) <= 100 だが x_i >= 2 なので sum >= 1000 → 矛盾
    a_triplets = [(1, j+1, 1.0) for j in range(n)]
    m = 1
    con_bounds = [(-INF, 100.0)]
    var_bounds = [(2.0, 5.0)] * n  # lb=2, sum_min = 2*500=1000 > 100
    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


# =============================================================================
# PROBLEMS リスト
# =============================================================================

PROBLEMS = [
    gen_infeas_bound_contradiction_n1,
    gen_infeas_bound_contradiction_n3,
    gen_infeas_eq_contradiction_n2,
    gen_infeas_ineq_contradiction_n3,
    gen_infeas_eq_range_contradiction_n4,
    gen_infeas_var_bound_vs_constraint_n4,
    gen_infeas_mixed_eq_ineq_n5,
    gen_infeas_cycle_n6,
    gen_infeas_overdetermined_n8,
    gen_infeas_large_n100,
    gen_infeas_large_n200,
    gen_infeas_large_n500,
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="QP infeasible problem generator (QPLIB format)")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "qp_infeasible"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen_fn in PROBLEMS:
        prob = gen_fn()
        out_path = out_dir / f"{prob.name}.qplib"
        write_qplib(prob, out_path)
        print(f"  {prob.name}: n={prob.n} m={prob.m} -> {out_path.name}")

    print(f"\n生成完了: {len(PROBLEMS)} 問題 -> {out_dir}")


if __name__ == "__main__":
    main()
