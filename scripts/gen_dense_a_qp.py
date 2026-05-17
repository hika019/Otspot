"""Dense-A convex QP generator (QPLIB format).

条件:
  - Q は PSD (半正定値, 凸)
  - A 行列が密 (nnz/mn > 50%)
  - n=500〜2000 の中規模
  - feasible な問題 (optimal solution が存在)

Q の構成:
  - 対角 PSD: Q = diag(d), d_i > 0
  - sparse PSD: Q = D + V^T V (D 対角 + ランクr 更新)
  - dense PSD: Q = L L^T (Cholesky 分解で構成) ← 大規模では避ける

A の構成:
  - 全要素を乱数で埋めて密にする
  - feasible point x0 を先に決めて b = A x0 ± slack とする
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

    out_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# ヘルパー
# =============================================================================

def make_diag_psd_q(n: int, rng: random.Random, scale: float = 1.0) -> list[tuple[int,int,float]]:
    """対角 PSD: Q = diag(d), d_i ~ Uniform(0.1, 2.0) * scale."""
    return [(i+1, i+1, rng.uniform(0.1, 2.0) * scale) for i in range(n)]


def make_sparse_psd_q(n: int, rng: random.Random, rank: int = 10,
                       reg: float = 0.1) -> list[tuple[int,int,float]]:
    """Sparse PSD: Q = diag(reg) + V^T V, V in R^{rank x n} (各要素ランダム).

    Q[i][j] = reg * delta_{ij} + sum_k V[k][i] * V[k][j]
    正定値: reg > 0 かつ V の選び方による最小固有値 > 0 を保証。
    Q の下三角のみ格納。nnz ≒ n (対角) + rank*n^2/2 は大規模では大きすぎるため、
    V を疎に保つ (各 V[k] は n/rank 個の非零要素のみ)。
    """
    # まず対角正則化
    diag_vals = [reg] * n

    # V の外積を累積: 各 k に対して V[k] を生成
    # Q_lower_tri: {(i,j): val} の辞書で管理 (i >= j, 1-indexed)
    q_dict: dict[tuple[int,int], float] = {}
    for i in range(n):
        q_dict[(i+1, i+1)] = diag_vals[i]

    nnz_per_row = max(1, n // rank)
    for _k in range(rank):
        # V[k]: n 個の要素から nnz_per_row 個を非零にする
        cols = rng.sample(range(n), min(nnz_per_row, n))
        v = [0.0] * n
        for j in cols:
            v[j] = rng.gauss(0, 1.0 / math.sqrt(nnz_per_row))

        # V[k] * V[k]^T の下三角
        nonzero_cols = [j for j in range(n) if v[j] != 0.0]
        for a_idx, a in enumerate(nonzero_cols):
            for b in nonzero_cols[:a_idx+1]:
                key = (a+1, b+1)
                q_dict[key] = q_dict.get(key, 0.0) + v[a] * v[b]

    return [(i, j, val) for (i,j), val in sorted(q_dict.items()) if abs(val) > 1e-15]


def make_dense_a(n: int, m: int, rng: random.Random,
                 density: float = 1.0) -> tuple[list[list[float]], list[tuple[int,int,float]]]:
    """密な制約行列 A ∈ R^{m×n}.

    density=1.0 → 全要素が非零 (100% 密)
    density=0.7 → 70% の要素が非零

    返り値: (A_matrix, a_triplets)
    """
    A = [[0.0]*n for _ in range(m)]
    a_triplets = []
    for k in range(m):
        for j in range(n):
            if rng.random() < density:
                v = rng.gauss(0, 1.0 / math.sqrt(n))
                A[k][j] = v
                a_triplets.append((k+1, j+1, v))
    return A, a_triplets


def compute_ax(A: list[list[float]], x: list[float]) -> list[float]:
    """A * x を計算。"""
    m = len(A)
    n = len(x)
    result = [0.0] * m
    for k in range(m):
        result[k] = sum(A[k][j] * x[j] for j in range(n))
    return result


# =============================================================================
# Dense-A QP 生成関数
# =============================================================================

def gen_dense_a_n500_m100_diag_q() -> QpProblem:
    """n=500, m=100, A dense, Q 対角 PSD."""
    name = "DENSE_A_N500_M100_DIAG"
    n, m = 500, 100
    rng = random.Random(1001)

    q_lower_tri = make_diag_psd_q(n, rng)
    c = [rng.gauss(0, 1.0) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    # feasible point: x0 = 0 (全ゼロ、変数は x >= 0)
    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)

    slack = 1.0
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(0.0, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n500_m200_dense_q() -> QpProblem:
    """n=500, m=200, A dense (100%), Q sparse PSD (rank-20 update + diag).

    Q が sparse PSD で A が dense という組み合わせ。
    """
    name = "DENSE_A_N500_M200_SPARSEQ"
    n, m = 500, 200
    rng = random.Random(1002)

    q_lower_tri = make_sparse_psd_q(n, rng, rank=20, reg=0.5)
    c = [rng.gauss(0, 1.0) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)
    slack = 2.0
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n1000_m100_diag_q() -> QpProblem:
    """n=1000, m=100, A dense, Q 対角 PSD."""
    name = "DENSE_A_N1000_M100_DIAG"
    n, m = 1000, 100
    rng = random.Random(1003)

    q_lower_tri = make_diag_psd_q(n, rng)
    c = [rng.gauss(0, 1.0) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)
    slack = 1.5
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(0.0, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n1000_m300_sparse_q() -> QpProblem:
    """n=1000, m=300, A dense, Q sparse PSD."""
    name = "DENSE_A_N1000_M300_SPARSEQ"
    n, m = 1000, 300
    rng = random.Random(1004)

    q_lower_tri = make_sparse_psd_q(n, rng, rank=30, reg=0.3)
    c = [rng.gauss(0, 0.5) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)
    slack = 1.0
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n2000_m200_diag_q() -> QpProblem:
    """n=2000, m=200, A dense, Q 対角 PSD."""
    name = "DENSE_A_N2000_M200_DIAG"
    n, m = 2000, 200
    rng = random.Random(1005)

    q_lower_tri = make_diag_psd_q(n, rng)
    c = [rng.gauss(0, 1.0) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)
    slack = 2.0
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(0.0, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n2000_m500_sparse_q() -> QpProblem:
    """n=2000, m=500, A dense, Q sparse PSD."""
    name = "DENSE_A_N2000_M500_SPARSEQ"
    n, m = 2000, 500
    rng = random.Random(1006)

    q_lower_tri = make_sparse_psd_q(n, rng, rank=50, reg=0.5)
    c = [rng.gauss(0, 0.5) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [0.0] * n
    Ax0 = compute_ax(A, x0)
    slack = 1.0
    con_bounds = [(Ax0[k] - slack, Ax0[k] + slack) for k in range(m)]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n500_m500_eq_diag_q() -> QpProblem:
    """n=500, m=500 (m=n), A dense, Q 対角 PSD (等号制約).

    等号制約: lb = ub = b。A が正方 (m=n) かつ full rank の場合は一意解。
    A の feasibility: x0=0 での Ax0=0 → con_bounds = (0,0) でも OK だが、
    full rank A に対して b=Ax0 (x0=0) → b=0 が解。それだと trivial なので
    x0 を適度に設定。
    """
    name = "DENSE_A_N500_M500_EQ_DIAG"
    n, m = 500, 500
    rng = random.Random(1007)

    q_lower_tri = make_diag_psd_q(n, rng)
    c = [rng.gauss(0, 1.0) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    # x0 = (1/n, 1/n, ...) で feasible な b を設定
    x0 = [1.0 / n] * n
    Ax0 = compute_ax(A, x0)

    # 等号制約: lb = ub = Ax0
    con_bounds = [(Ax0[k], Ax0[k]) for k in range(m)]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


def gen_dense_a_n1000_m1000_eq_sparse_q() -> QpProblem:
    """n=1000, m=1000 (m=n), A dense, Q sparse PSD (等号制約)."""
    name = "DENSE_A_N1000_M1000_EQ_SPARSEQ"
    n, m = 1000, 1000
    rng = random.Random(1008)

    q_lower_tri = make_sparse_psd_q(n, rng, rank=20, reg=0.5)
    c = [rng.gauss(0, 0.5) for _ in range(n)]

    A, a_triplets = make_dense_a(n, m, rng, density=1.0)

    x0 = [1.0 / n] * n
    Ax0 = compute_ax(A, x0)

    con_bounds = [(Ax0[k], Ax0[k]) for k in range(m)]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(name, n, q_lower_tri, c, 'L', a_triplets, m, con_bounds, var_bounds)


# =============================================================================
# PROBLEMS リスト
# =============================================================================

PROBLEMS = [
    gen_dense_a_n500_m100_diag_q,
    gen_dense_a_n500_m200_dense_q,
    gen_dense_a_n1000_m100_diag_q,
    gen_dense_a_n1000_m300_sparse_q,
    gen_dense_a_n2000_m200_diag_q,
    gen_dense_a_n2000_m500_sparse_q,
    gen_dense_a_n500_m500_eq_diag_q,
    gen_dense_a_n1000_m1000_eq_sparse_q,
]


def nnz_density(a_triplets: list, n: int, m: int) -> float:
    """A 行列の密度 (nnz / (m*n)) を計算。"""
    if m == 0 or n == 0:
        return 0.0
    return len(a_triplets) / (m * n)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dense-A convex QP generator (QPLIB format)")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "qp_dense_a"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen_fn in PROBLEMS:
        prob = gen_fn()
        out_path = out_dir / f"{prob.name}.qplib"
        write_qplib(prob, out_path)
        density = nnz_density(prob.a_triplets, prob.n, prob.m)
        print(f"  {prob.name}: n={prob.n} m={prob.m} nnz_Q={len(prob.q_lower_tri)}"
              f" nnz_A={len(prob.a_triplets)} density={density:.3f} -> {out_path.name}")

    print(f"\n生成完了: {len(PROBLEMS)} 問題 -> {out_dir}")


if __name__ == "__main__":
    main()
