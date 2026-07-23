"""Synthetic nonconvex (indefinite) QP problem generator.

目的:
  ソルバーの NonConvex 検出機能を検証するため、明示的に非凸（不定Q行列）な
  QP 問題を QPLIB 形式で生成する。

生成する問題の種別:
  - 対角Q行列に負値を含むもの（最も単純な非凸）
  - 密Q行列で混合固有値を持つもの
  - スパースQ行列（制約付き）
  - n>1000 の大規模問題（対角チェックは通過する形式含む）
  - n>1000 の大規模問題（対角に明示的な負値を持つもの）

QPLIB フォーマット (QCL / QCB タイプ):
  - O = Q: 不定目的関数（indefinite）
  - V = C: 連続変数
  - C = L: 線形制約 / C = B: 境界制約のみ
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import NamedTuple


class QpProblem(NamedTuple):
    """生成するQP問題の記述。minimize (1/2) x^T Q x + c^T x s.t. constraints."""
    name: str
    n: int
    # Q の下三角トリプレット: (i, j, v) で i >= j、1-indexed
    q_lower_tri: list[tuple[int, int, float]]
    # 線形目的係数
    c: list[float]
    # 制約タイプ: 'L' or 'B'
    con_type: str
    # con_type='L' のとき: [(a_row_1indexed, b_col_1indexed, val), ...]
    a_triplets: list[tuple[int, int, float]]
    m: int  # 制約数 (con_type='L') または 0 (con_type='B')
    # con_type='L' のとき: (lb, ub) の list (inf=1e30 で表現)
    con_bounds: list[tuple[float, float]]
    # 変数境界 (lb, ub), len=n
    var_bounds: list[tuple[float, float]]


INF = 1e30


def write_qplib(prob: QpProblem, out_path: Path) -> None:
    """QpProblem を QPLIB 形式ファイルに書き出す。"""
    lines: list[str] = []
    lines.append(prob.name)
    # タイプコード: Q (indefinite obj) + C (continuous) + L/B (constraints)
    lines.append(f"QC{prob.con_type}")
    lines.append("minimize")
    lines.append(f"{prob.n}")
    if prob.con_type == 'L':
        lines.append(f"{prob.m}")

    # 目的関数二次項（下三角）
    lines.append(f"{len(prob.q_lower_tri)}")
    for i, j, v in prob.q_lower_tri:
        lines.append(f"{i} {j} {v:.15g}")

    # 線形目的係数
    # デフォルト=0、非デフォルトのみ列挙
    nondefault_c = [(i + 1, v) for i, v in enumerate(prob.c) if v != 0.0]
    lines.append("0.0")  # default_b0 = 0
    lines.append(f"{len(nondefault_c)}")
    for idx, v in nondefault_c:
        lines.append(f"{idx} {v:.15g}")

    # 目的定数
    lines.append("0.0")

    # 制約線形項
    if prob.con_type == 'L':
        lines.append(f"{len(prob.a_triplets)}")
        for k, i, v in prob.a_triplets:
            lines.append(f"{k} {i} {v:.15g}")

    # 無限大の定義値
    lines.append(f"{INF:.6g}")

    # 制約下界・上界
    if prob.con_type == 'L':
        lb_con = [lb for lb, _ in prob.con_bounds]
        ub_con = [ub for _, ub in prob.con_bounds]

        # 下界
        lines.append(f"{-INF:.6g}")  # default = -inf
        nondefault_lb = [(k + 1, lb) for k, lb in enumerate(lb_con) if lb > -INF * 0.99]
        lines.append(f"{len(nondefault_lb)}")
        for k, lb in nondefault_lb:
            lines.append(f"{k} {lb:.15g}")

        # 上界
        lines.append(f"{INF:.6g}")  # default = +inf
        nondefault_ub = [(k + 1, ub) for k, ub in enumerate(ub_con) if ub < INF * 0.99]
        lines.append(f"{len(nondefault_ub)}")
        for k, ub in nondefault_ub:
            lines.append(f"{k} {ub:.15g}")

    # 変数下界・上界
    lb_var = [lb for lb, _ in prob.var_bounds]
    ub_var = [ub for _, ub in prob.var_bounds]

    lines.append("0.0")  # default lb = 0
    nondefault_lb_var = [(i + 1, lb) for i, lb in enumerate(lb_var) if lb != 0.0]
    lines.append(f"{len(nondefault_lb_var)}")
    for i, lb in nondefault_lb_var:
        lines.append(f"{i} {lb:.15g}")

    lines.append(f"{INF:.6g}")  # default ub = +inf
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
# 問題生成関数群
# =============================================================================

def gen_indef_diag(name: str, n: int, seed: int, neg_frac: float = 0.3,
                   scale: float = 1.0, with_bounds: bool = False) -> QpProblem:
    """対角Q行列に負値を含む非凸QP。

    Q = diag(d) where some d[i] < 0.
    これは最も単純な非凸QP。neg_frac の割合の変数が負の曲率を持つ。
    制約: 変数の和 = 1 (1-simplex) or box bounds [0, 1]
    """
    rng = random.Random(seed)
    d = []
    for i in range(n):
        if rng.random() < neg_frac:
            d.append(-rng.uniform(0.5, 3.0) * scale)
        else:
            d.append(rng.uniform(0.1, 2.0) * scale)

    q_lower_tri = [(i + 1, i + 1, d[i]) for i in range(n)]
    c = [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]

    if with_bounds:
        # box constraints only
        var_bounds = [(0.0, 1.0)] * n
        return QpProblem(
            name=name, n=n,
            q_lower_tri=q_lower_tri, c=c,
            con_type='B', a_triplets=[], m=0, con_bounds=[],
            var_bounds=var_bounds,
        )
    else:
        # sum(x) = 1 constraint
        a_triplets = [(1, i + 1, 1.0) for i in range(n)]
        con_bounds = [(1.0, 1.0)]
        var_bounds = [(-INF, INF)] * n
        return QpProblem(
            name=name, n=n,
            q_lower_tri=q_lower_tri, c=c,
            con_type='L', a_triplets=a_triplets, m=1, con_bounds=con_bounds,
            var_bounds=var_bounds,
        )


def gen_indef_dense(name: str, n: int, seed: int, neg_ev_frac: float = 0.3) -> QpProblem:
    """密Q行列で混合固有値を持つ非凸QP。

    構成: Q = V * diag(lambda) * V^T で lambda の一部が負。
    V は直交行列（QR分解で生成）。
    問題: minimize (1/2) x^T Q x + c^T x  s.t. -10 <= x_i <= 10
    """
    rng = random.Random(seed)

    # ランダム直交行列 V (n×n) をハウスホルダー反射で生成
    def random_orthogonal(n: int, rng: random.Random) -> list[list[float]]:
        # Gram-Schmidt
        basis = []
        for _ in range(n):
            v = [rng.gauss(0, 1) for _ in range(n)]
            for u in basis:
                dot = sum(v[k] * u[k] for k in range(n))
                v = [v[k] - dot * u[k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-12:
                # fallback: basis vector
                v = [1.0 if k == len(basis) else 0.0 for k in range(n)]
                norm = 1.0
            v = [x / norm for x in v]
            basis.append(v)
        return basis  # basis[i] = i-th row of V

    # 固有値: neg_ev_frac の割合が負
    n_neg = max(1, int(n * neg_ev_frac))
    lambdas = []
    for i in range(n):
        if i < n_neg:
            lambdas.append(-rng.uniform(0.1, 2.0))
        else:
            lambdas.append(rng.uniform(0.1, 2.0))

    # Q = V^T * diag(lambda) * V (V の各行が固有ベクトル)
    V = random_orthogonal(n, rng)  # V[i] = i-th eigenvector

    # Q[a][b] = sum_k lambda[k] * V[k][a] * V[k][b]
    # 下三角のみ格納
    q_lower_tri = []
    for a in range(n):
        for b in range(a + 1):
            q_ab = sum(lambdas[k] * V[k][a] * V[k][b] for k in range(n))
            if abs(q_ab) > 1e-14:
                q_lower_tri.append((a + 1, b + 1, q_ab))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # 変数境界: [-10, 10]
    var_bounds = [(-10.0, 10.0)] * n

    # 制約なし（box のみ）
    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='B', a_triplets=[], m=0, con_bounds=[],
        var_bounds=var_bounds,
    )


def gen_indef_sparse_constrained(name: str, n: int, m: int, seed: int,
                                 nnz_per_col: int = 3) -> QpProblem:
    """スパースQ行列 + 線形制約付き非凸QP。

    Q: スパース不定行列 (対角に負値を含む)
    A: ランダムスパース制約行列
    制約: lb <= Ax <= ub
    """
    rng = random.Random(seed)

    # Q: 対角は交互に正負、一部オフ対角あり
    q_lower_tri = []
    for i in range(n):
        # 対角: 30%が負
        diag_val = (-1.0 if i % 3 == 0 else 1.0) * rng.uniform(0.5, 2.0)
        q_lower_tri.append((i + 1, i + 1, diag_val))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # A: ランダムスパース
    a_triplets = []
    for k in range(m):
        # 各制約にランダムな変数を選ぶ
        cols = rng.sample(range(n), min(nnz_per_col, n))
        for j in cols:
            v = rng.uniform(-1.0, 1.0)
            if abs(v) > 0.05:
                a_triplets.append((k + 1, j + 1, v))

    # 制約: lb <= Ax <= ub (feasible region: sum-of-ones type)
    # feasible x を x=0 として Ax = 0 を含む区間にする
    con_bounds = [(-5.0, 5.0)] * m

    var_bounds = [(-5.0, 5.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='L', a_triplets=a_triplets, m=m, con_bounds=con_bounds,
        var_bounds=var_bounds,
    )


def gen_indef_large_diag_neg(name: str, n: int, m: int, seed: int) -> QpProblem:
    """n>1000 で対角に明示的な負値を持つ非凸QP。

    目的: n>1000 でも対角チェックで NonConvex が検出されることを確認。
    対角の一部 (30%) を負にする。
    """
    rng = random.Random(seed)

    q_lower_tri = []
    for i in range(n):
        if i % 3 == 0:  # 1/3 が負
            q_lower_tri.append((i + 1, i + 1, -rng.uniform(0.1, 1.0)))
        else:
            q_lower_tri.append((i + 1, i + 1, rng.uniform(0.1, 1.0)))

    c = [rng.uniform(-0.5, 0.5) for _ in range(n)]

    # 制約: box
    if m == 0:
        var_bounds = [(-1.0, 1.0)] * n
        return QpProblem(
            name=name, n=n,
            q_lower_tri=q_lower_tri, c=c,
            con_type='B', a_triplets=[], m=0, con_bounds=[],
            var_bounds=var_bounds,
        )
    else:
        # sparse linear constraints
        a_triplets = []
        for k in range(m):
            cols = rng.sample(range(n), min(5, n))
            for j in cols:
                a_triplets.append((k + 1, j + 1, rng.choice([-1.0, 1.0])))
        con_bounds = [(-3.0, 3.0)] * m
        var_bounds = [(-1.0, 1.0)] * n
        return QpProblem(
            name=name, n=n,
            q_lower_tri=q_lower_tri, c=c,
            con_type='L', a_triplets=a_triplets, m=m, con_bounds=con_bounds,
            var_bounds=var_bounds,
        )


def gen_indef_offdiag_only(name: str, n: int, seed: int) -> QpProblem:
    """対角が全て正だが密なオフ対角で不定になるQ行列。

    目的: 対角チェックをパスするが Cholesky で検出される非凸問題。
    n <= 1000 のみで意味がある (n>1000 は Cholesky スキップ)。

    構成: Q = D + E where D = diag(large positive) and E = small dense indefinite.
    実は D の固有値が E の最小固有値を超えると PSD になる。
    代わりに: Q[i][j] = delta_{ij} * d_i + A[i][j] で
    A が dense でその最小固有値が -d_min より小さい場合。

    確実に不定にする方法: Cauchy-Schwarz 的に Q = [[1, 2],[2, 1]] は固有値 3,-1 で不定。
    n 個の 2x2 ブロックを並べる: diag block [[1, r],[r, 1]] with r>1.
    """
    rng = random.Random(seed)
    assert n >= 4 and n % 2 == 0, f"n must be even and >= 4, got {n}"

    q_lower_tri = []
    for i in range(0, n, 2):
        d = 1.0  # 対角
        r = rng.uniform(1.2, 2.0)  # off-diag > 1 → 固有値 d±r, 最小 = d-r < 0
        q_lower_tri.append((i + 1, i + 1, d))
        q_lower_tri.append((i + 2, i + 1, r))  # 下三角: (row, col) = (i+2, i+1)
        q_lower_tri.append((i + 2, i + 2, d))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # 制約: sum of each pair = 0 (overdetermined, but feasible)
    a_triplets = []
    m = n // 2
    for k in range(m):
        a_triplets.append((k + 1, 2 * k + 1, 1.0))
        a_triplets.append((k + 1, 2 * k + 2, 1.0))

    con_bounds = [(-2.0, 2.0)] * m
    var_bounds = [(-5.0, 5.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='L', a_triplets=a_triplets, m=m, con_bounds=con_bounds,
        var_bounds=var_bounds,
    )


def gen_eq_only_constrained(name: str, n: int, m: int, seed: int) -> QpProblem:
    """等号制約のみを持つ非凸QP。

    制約: A x = b  (等号制約のみ、不等号なし)
    Q: 対角不定行列
    目的: 等号制約付きのKKT条件で saddle-point システムが生じる状況を検証。
    """
    rng = random.Random(seed)
    assert m < n, f"m={m} must be < n={n} for feasibility"

    d = []
    for i in range(n):
        if rng.random() < 0.3:
            d.append(-rng.uniform(0.5, 2.0))
        else:
            d.append(rng.uniform(0.1, 2.0))

    q_lower_tri = [(i + 1, i + 1, d[i]) for i in range(n)]
    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # A: dense equality constraints, x0 = (1/n, ..., 1/n) is feasible
    x0 = 1.0 / n
    a_triplets = []
    b_vals = []
    for k in range(m):
        row = [rng.gauss(0, 1) for _ in range(n)]
        b = sum(row[j] * x0 for j in range(n))
        for j, v in enumerate(row):
            if abs(v) > 1e-10:
                a_triplets.append((k + 1, j + 1, v))
        b_vals.append(b)

    # equality: lb = ub = b
    con_bounds = [(b, b) for b in b_vals]
    var_bounds = [(-INF, INF)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='L', a_triplets=a_triplets, m=m, con_bounds=con_bounds,
        var_bounds=var_bounds,
    )


def gen_overdetermined_constrained(name: str, n: int, m: int, seed: int) -> QpProblem:
    """高密度制約 (m >> n) の非凸QP。

    制約: lb <= A x <= ub  (m >> n: many constraints per variable)
    Q: 対角不定行列
    目的: 制約が多い場合の検証。ランクが n の制約行列で実現可能性を保証。
    """
    rng = random.Random(seed)
    assert m > n, f"m={m} must be > n={n} for overdetermined"

    d = []
    for i in range(n):
        if rng.random() < 0.4:
            d.append(-rng.uniform(0.3, 2.0))
        else:
            d.append(rng.uniform(0.1, 3.0))

    q_lower_tri = [(i + 1, i + 1, d[i]) for i in range(n)]
    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # feasible point: x* = 0
    a_triplets = []
    for k in range(m):
        for j in range(n):
            v = rng.gauss(0, 1)
            if abs(v) > 0.3:  # moderate sparsity
                a_triplets.append((k + 1, j + 1, v))

    # Ax* = 0 at x=0, so [-3, 3] bounds include 0
    con_bounds = [(-3.0, 3.0)] * m
    var_bounds = [(-2.0, 2.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='L', a_triplets=a_triplets, m=m, con_bounds=con_bounds,
        var_bounds=var_bounds,
    )


def gen_one_neg_eigenvalue(name: str, n: int, seed: int) -> QpProblem:
    """固有値が1つだけ負の境界非凸QP。

    最小固有値が1つだけ負 (他はすべて正) → 最も境界に近い非凸ケース。
    構成: Q = V diag(lambda) V^T, lambda = [neg_val, pos_1, ..., pos_{n-1}]
    """
    rng = random.Random(seed)

    def random_orthogonal(n: int, rng: random.Random) -> list[list[float]]:
        basis = []
        for _ in range(n):
            v = [rng.gauss(0, 1) for _ in range(n)]
            for u in basis:
                dot = sum(v[k] * u[k] for k in range(n))
                v = [v[k] - dot * u[k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-12:
                v = [1.0 if k == len(basis) else 0.0 for k in range(n)]
                norm = 1.0
            v = [x / norm for x in v]
            basis.append(v)
        return basis

    # 固有値: 1つだけ負、他はすべて正
    neg_val = -rng.uniform(0.1, 1.0)
    lambdas = [neg_val] + [rng.uniform(0.5, 3.0) for _ in range(n - 1)]

    V = random_orthogonal(n, rng)

    q_lower_tri = []
    for a in range(n):
        for b in range(a + 1):
            q_ab = sum(lambdas[k] * V[k][a] * V[k][b] for k in range(n))
            if abs(q_ab) > 1e-14:
                q_lower_tri.append((a + 1, b + 1, q_ab))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    var_bounds = [(-5.0, 5.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='B', a_triplets=[], m=0, con_bounds=[],
        var_bounds=var_bounds,
    )


def gen_ill_conditioned_indef(name: str, n: int, seed: int,
                               cond_ratio: float = 1e4) -> QpProblem:
    """固有値分布の偏りが大きい（高 conditioning）非凸QP。

    構成: Q = V diag(lambda) V^T
    lambda: 負の固有値は小さく (-1 程度)、正の固有値は大きく (cond_ratio 程度)
    → condition number が非常に大きく、数値的に困難な問題。
    """
    rng = random.Random(seed)

    def random_orthogonal(n: int, rng: random.Random) -> list[list[float]]:
        basis = []
        for _ in range(n):
            v = [rng.gauss(0, 1) for _ in range(n)]
            for u in basis:
                dot = sum(v[k] * u[k] for k in range(n))
                v = [v[k] - dot * u[k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-12:
                v = [1.0 if k == len(basis) else 0.0 for k in range(n)]
                norm = 1.0
            v = [x / norm for x in v]
            basis.append(v)
        return basis

    n_neg = max(1, n // 5)  # 20% 負固有値
    lambdas = []
    for i in range(n):
        if i < n_neg:
            lambdas.append(-rng.uniform(0.5, 2.0))
        else:
            # 正の固有値: log-uniform で大きなレンジ
            log_scale = rng.uniform(0.0, math.log10(cond_ratio))
            lambdas.append(10 ** log_scale)

    V = random_orthogonal(n, rng)

    q_lower_tri = []
    for a in range(n):
        for b in range(a + 1):
            q_ab = sum(lambdas[k] * V[k][a] * V[k][b] for k in range(n))
            if abs(q_ab) > 1e-14:
                q_lower_tri.append((a + 1, b + 1, q_ab))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    var_bounds = [(-10.0, 10.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='B', a_triplets=[], m=0, con_bounds=[],
        var_bounds=var_bounds,
    )


def gen_saddle_point(name: str, n: int, seed: int) -> QpProblem:
    """正負の固有値が半々の saddle point 型非凸QP。

    構成: Q = V diag(lambda) V^T
    lambda: n/2 個が負、n/2 個が正 (絶対値は同程度)
    → 完全な saddle point 型の目的関数
    """
    rng = random.Random(seed)
    assert n % 2 == 0, f"n must be even, got {n}"

    def random_orthogonal(n: int, rng: random.Random) -> list[list[float]]:
        basis = []
        for _ in range(n):
            v = [rng.gauss(0, 1) for _ in range(n)]
            for u in basis:
                dot = sum(v[k] * u[k] for k in range(n))
                v = [v[k] - dot * u[k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-12:
                v = [1.0 if k == len(basis) else 0.0 for k in range(n)]
                norm = 1.0
            v = [x / norm for x in v]
            basis.append(v)
        return basis

    n_neg = n // 2
    lambdas = []
    for i in range(n):
        scale = rng.uniform(0.5, 2.0)
        if i < n_neg:
            lambdas.append(-scale)
        else:
            lambdas.append(scale)

    V = random_orthogonal(n, rng)

    q_lower_tri = []
    for a in range(n):
        for b in range(a + 1):
            q_ab = sum(lambdas[k] * V[k][a] * V[k][b] for k in range(n))
            if abs(q_ab) > 1e-14:
                q_lower_tri.append((a + 1, b + 1, q_ab))

    c = [rng.uniform(-0.5, 0.5) for _ in range(n)]

    # 制約: sum(x) = 0 (equal split)
    a_triplets = [(1, i + 1, 1.0) for i in range(n)]
    con_bounds = [(0.0, 0.0)]
    var_bounds = [(-3.0, 3.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='L', a_triplets=a_triplets, m=1, con_bounds=con_bounds,
        var_bounds=var_bounds,
    )


def gen_near_psd(name: str, n: int, seed: int, min_neg_ev: float = -1e-4) -> QpProblem:
    """near-PSD: 最小固有値がわずかに負の非凸QP。

    構成: Q = V diag(lambda) V^T
    lambda: 1つだけ min_neg_ev (小さな負値)、他は適度な正値
    → PSD との境界付近、数値的に検出が困難なケース
    """
    rng = random.Random(seed)

    def random_orthogonal(n: int, rng: random.Random) -> list[list[float]]:
        basis = []
        for _ in range(n):
            v = [rng.gauss(0, 1) for _ in range(n)]
            for u in basis:
                dot = sum(v[k] * u[k] for k in range(n))
                v = [v[k] - dot * u[k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-12:
                v = [1.0 if k == len(basis) else 0.0 for k in range(n)]
                norm = 1.0
            v = [x / norm for x in v]
            basis.append(v)
        return basis

    # 1つだけ小さな負の固有値、他は正
    lambdas = [min_neg_ev] + [rng.uniform(0.5, 2.0) for _ in range(n - 1)]
    V = random_orthogonal(n, rng)

    q_lower_tri = []
    for a in range(n):
        for b in range(a + 1):
            q_ab = sum(lambdas[k] * V[k][a] * V[k][b] for k in range(n))
            if abs(q_ab) > 1e-15:
                q_lower_tri.append((a + 1, b + 1, q_ab))

    c = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    var_bounds = [(-5.0, 5.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='B', a_triplets=[], m=0, con_bounds=[],
        var_bounds=var_bounds,
    )


def gen_maxcut_relaxation(name: str, n: int, seed: int) -> QpProblem:
    """Max-Cut SDP 緩和の QP 形式。

    Max-Cut: maximize (1/4) * x^T (D-W) x s.t. x_i in {-1,+1}
    連続緩和: minimize -(1/4) x^T L x s.t. x_i^2 <= 1 (box: -1<=x_i<=1)
    L = D - W はラプラシアン行列 (PSD), -L は不定 → 非凸 minimize 問題。
    """
    rng = random.Random(seed)

    # ランダムグラフ (Erdos-Renyi, p=0.3)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.3:
                w = rng.uniform(0.5, 1.5)
                W[i][j] = w
                W[j][i] = w

    # Laplacian L = D - W, D = diag(sum of W rows)
    # We minimize -(1/4) x^T L x = (1/4) x^T (W - D) x
    # Objective matrix: Q_obj = (1/4) (W - D) = -(1/4) L (indefinite!)

    # 下三角トリプレット
    q_lower_tri = []
    for i in range(n):
        deg_i = sum(W[i])
        # 対角: -(1/4) * D[i][i] = -(1/4) * deg_i
        q_diag = -(1.0 / 4.0) * deg_i
        if abs(q_diag) > 1e-12:
            q_lower_tri.append((i + 1, i + 1, q_diag))
        # 下三角オフ対角: +(1/4) * W[i][j]
        for j in range(i):
            q_ij = (1.0 / 4.0) * W[i][j]
            if abs(q_ij) > 1e-12:
                q_lower_tri.append((i + 1, j + 1, q_ij))

    c = [0.0] * n  # no linear term

    # Box constraints: -1 <= x_i <= 1
    var_bounds = [(-1.0, 1.0)] * n

    return QpProblem(
        name=name, n=n,
        q_lower_tri=q_lower_tri, c=c,
        con_type='B', a_triplets=[], m=0, con_bounds=[],
        var_bounds=var_bounds,
    )


# =============================================================================
# メイン
# =============================================================================

PROBLEMS = [
    # (generator_func, kwargs)
    # 小規模: 対角Q、NonConvex 検出確認用 (n <= 1000 → Cholesky 検出)
    lambda: gen_indef_diag("NONCONVEX_DIAG_N10",    n=10,  seed=1, neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_N50",    n=50,  seed=2, neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_N200",   n=200, seed=3, neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_N500",   n=500, seed=4, neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_BOX_N50", n=50, seed=5, neg_frac=0.3, with_bounds=True),

    # 密Q: オフ対角も含む不定行列 (n <= 1000 → Cholesky 検出)
    lambda: gen_indef_dense("NONCONVEX_DENSE_N20",  n=20,  seed=10, neg_ev_frac=0.3),
    lambda: gen_indef_dense("NONCONVEX_DENSE_N50",  n=50,  seed=11, neg_ev_frac=0.25),
    lambda: gen_indef_dense("NONCONVEX_DENSE_N100", n=100, seed=12, neg_ev_frac=0.2),

    # オフ対角のみ不定: 対角チェックをパスするが Cholesky で検出
    lambda: gen_indef_offdiag_only("NONCONVEX_OFFDIAG_N20", n=20, seed=20),
    lambda: gen_indef_offdiag_only("NONCONVEX_OFFDIAG_N100", n=100, seed=21),

    # スパースQ + 線形制約
    lambda: gen_indef_sparse_constrained("NONCONVEX_SPARSE_N100_M30", n=100, m=30, seed=30),
    lambda: gen_indef_sparse_constrained("NONCONVEX_SPARSE_N300_M100", n=300, m=100, seed=31),

    # Max-Cut 緩和
    lambda: gen_maxcut_relaxation("NONCONVEX_MAXCUT_N20", n=20, seed=40),
    lambda: gen_maxcut_relaxation("NONCONVEX_MAXCUT_N50", n=50, seed=41),

    # 大規模: n > 1000, 対角に明示的な負値 → 対角チェックで検出
    lambda: gen_indef_large_diag_neg("NONCONVEX_LARGE_N1200_BOX", n=1200, m=0, seed=50),
    lambda: gen_indef_large_diag_neg("NONCONVEX_LARGE_N2000_M200", n=2000, m=200, seed=51),
    lambda: gen_indef_large_diag_neg("NONCONVEX_LARGE_N5000_BOX", n=5000, m=0, seed=52),

    # 既存カテゴリの中間サイズ追加
    lambda: gen_indef_diag("NONCONVEX_DIAG_N100",   n=100,  seed=6,  neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_N300",   n=300,  seed=7,  neg_frac=0.3),
    lambda: gen_indef_diag("NONCONVEX_DIAG_BOX_N200", n=200, seed=8, neg_frac=0.4, with_bounds=True),
    lambda: gen_indef_dense("NONCONVEX_DENSE_N30",  n=30,  seed=13, neg_ev_frac=0.3),
    lambda: gen_indef_dense("NONCONVEX_DENSE_N75",  n=75,  seed=14, neg_ev_frac=0.4),
    lambda: gen_indef_sparse_constrained("NONCONVEX_SPARSE_N50_M20",  n=50,  m=20,  seed=32),
    lambda: gen_indef_sparse_constrained("NONCONVEX_SPARSE_N200_M60", n=200, m=60,  seed=33),
    lambda: gen_maxcut_relaxation("NONCONVEX_MAXCUT_N30", n=30, seed=42),
    lambda: gen_maxcut_relaxation("NONCONVEX_MAXCUT_N100", n=100, seed=43),

    # 等号制約のみ (EQ_ONLY)
    lambda: gen_eq_only_constrained("NONCONVEX_EQ_N30_M10",   n=30,  m=10,  seed=60),
    lambda: gen_eq_only_constrained("NONCONVEX_EQ_N100_M30",  n=100, m=30,  seed=61),
    lambda: gen_eq_only_constrained("NONCONVEX_EQ_N200_M50",  n=200, m=50,  seed=62),

    # 高密度制約 (m >> n)
    lambda: gen_overdetermined_constrained("NONCONVEX_OVER_N20_M100",   n=20,  m=100,  seed=70),
    lambda: gen_overdetermined_constrained("NONCONVEX_OVER_N50_M300",   n=50,  m=300,  seed=71),
    lambda: gen_overdetermined_constrained("NONCONVEX_OVER_N100_M500",  n=100, m=500,  seed=72),

    # 固有値が1つだけ負 (ONE_NEG_EV): 境界非凸ケース
    lambda: gen_one_neg_eigenvalue("NONCONVEX_ONENEG_N20",  n=20,  seed=80),
    lambda: gen_one_neg_eigenvalue("NONCONVEX_ONENEG_N50",  n=50,  seed=81),
    lambda: gen_one_neg_eigenvalue("NONCONVEX_ONENEG_N100", n=100, seed=82),

    # 高 conditioning + 不定 (ILL_CONDITIONED)
    lambda: gen_ill_conditioned_indef("NONCONVEX_ILL_N30",  n=30,  seed=90),
    lambda: gen_ill_conditioned_indef("NONCONVEX_ILL_N50",  n=50,  seed=91),
    lambda: gen_ill_conditioned_indef("NONCONVEX_ILL_N100", n=100, seed=92, cond_ratio=1e6),

    # saddle point 型 (SADDLE): 正負半々
    lambda: gen_saddle_point("NONCONVEX_SADDLE_N20",  n=20,  seed=100),
    lambda: gen_saddle_point("NONCONVEX_SADDLE_N50",  n=50,  seed=101),
    lambda: gen_saddle_point("NONCONVEX_SADDLE_N100", n=100, seed=102),

    # near-PSD: 最小固有値がわずかに負
    lambda: gen_near_psd("NONCONVEX_NEARPSD_N20_EPS1E4",  n=20,  seed=110, min_neg_ev=-1e-4),
    lambda: gen_near_psd("NONCONVEX_NEARPSD_N50_EPS1E4",  n=50,  seed=111, min_neg_ev=-1e-4),
    lambda: gen_near_psd("NONCONVEX_NEARPSD_N20_EPS1E6",  n=20,  seed=112, min_neg_ev=-1e-6),
    lambda: gen_near_psd("NONCONVEX_NEARPSD_N50_EPS1E6",  n=50,  seed=113, min_neg_ev=-1e-6),
]


def main():
    ap = argparse.ArgumentParser(description="Synthetic nonconvex QP generator")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "qplib_nonconvex"),
        help="Output directory (default: data/qplib_nonconvex/)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for gen_fn in PROBLEMS:
        prob = gen_fn()
        out_path = out_dir / f"{prob.name}.qplib"
        write_qplib(prob, out_path)
        n_q = len(prob.q_lower_tri)
        print(f"  {prob.name}: n={prob.n} m={prob.m} nnz_Q={n_q} -> {out_path.name}")
        generated.append(prob.name)

    print(f"\n生成完了: {len(generated)} 問題 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
