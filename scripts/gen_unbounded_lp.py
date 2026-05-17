"""LP unbounded problem generator (MPS format).

数学的正当性:
  minimize c^T x  s.t.  Ax >= b,  lb <= x <= ub
  unbounded の条件: feasible ray d が存在して
    c^T d < 0  かつ  Ad >= 0  かつ  d[i] >= 0 for i s.t. lb[i] = 0
  → 任意の feasible x に対して x + t*d も feasible で、t→∞ で目的値 → -∞

各パターンで d, A, b を先に設計してから c を設定する。
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple


# =============================================================================
# MPS writer
# =============================================================================

class LpProblem(NamedTuple):
    """LP 問題記述。minimize c^T x s.t. constraints, lb <= x <= ub."""
    name: str
    n: int              # 変数数
    var_names: list[str]
    obj_name: str
    # 行: (row_name, row_type, {col_name: coeff}, rhs)
    # row_type: 'L' (<=), 'G' (>=), 'E' (=)
    rows: list[tuple[str, str, dict[str, float], float]]
    # 境界: {col_name: (lb, ub)}  None = 0 <= x < +inf (デフォルト)
    bounds: dict[str, tuple[float | None, float | None]]


def write_mps(prob: LpProblem, out_path: Path) -> None:
    """LpProblem を free MPS 形式で書き出す。"""
    lines: list[str] = []

    lines.append(f"NAME          {prob.name}")
    lines.append("ROWS")
    lines.append(f" N  {prob.obj_name}")
    for row_name, row_type, _, _ in prob.rows:
        lines.append(f" {row_type}  {row_name}")

    lines.append("COLUMNS")
    # 列ごとに係数を集める
    col_entries: dict[str, list[tuple[str, float]]] = {col: [] for col in prob.var_names}
    for row_name, _, coeffs, _ in prob.rows:
        for col, val in coeffs.items():
            col_entries[col].append((row_name, val))

    for col in prob.var_names:
        # 目的係数
        # (目的は別途 obj_coeffs で管理してないが rows には入れない → obj はここ)
        pass  # obj coeffs は rows に N 行として含まれていない。別途 obj_row を使う
    # 実装を整理: obj 係数は rows に含む方式にする (row_type='N')
    # ここでは LpProblem.rows が N 行を含まないので、obj 係数は別途 obj_coeffs フィールドが必要
    # → 再設計: write_mps2 を使う

    raise NotImplementedError("use write_mps2")


class LpProblem2(NamedTuple):
    """LP 問題記述 v2。"""
    name: str
    obj_name: str
    # 目的係数: {col_name: coeff}
    obj_coeffs: dict[str, float]
    # 制約行: (row_name, row_type, {col_name: coeff}, rhs)
    rows: list[tuple[str, str, dict[str, float], float]]
    # 変数名リスト (順序保持)
    var_names: list[str]
    # 境界 {col_name: (lb, ub)}  lb=None → -inf、ub=None → +inf
    bounds: dict[str, tuple[float | None, float | None]]


def write_mps2(prob: LpProblem2, out_path: Path) -> None:
    """LpProblem2 を MPS 形式で書き出す。"""
    lines: list[str] = []

    lines.append(f"NAME          {prob.name}")

    # ROWS
    lines.append("ROWS")
    lines.append(f" N  {prob.obj_name}")
    for row_name, row_type, _, _ in prob.rows:
        lines.append(f" {row_type}  {row_name}")

    # COLUMNS section
    lines.append("COLUMNS")

    # 変数ごとに (row, val) を集める
    col_data: dict[str, list[tuple[str, float]]] = {v: [] for v in prob.var_names}

    # 目的係数
    for col, val in prob.obj_coeffs.items():
        if col in col_data:
            col_data[col].append((prob.obj_name, val))

    # 制約係数
    for row_name, _, coeffs, _ in prob.rows:
        for col, val in coeffs.items():
            if col in col_data:
                col_data[col].append((row_name, val))

    for col in prob.var_names:
        entries = col_data[col]
        if not entries:
            # ダミーエントリ (obj = 0)
            entries = [(prob.obj_name, 0.0)]
        # 2 エントリずつ 1 行に出力
        for i in range(0, len(entries), 2):
            pair = entries[i:i+2]
            row1, val1 = pair[0]
            line = f"    {col:<10}  {row1:<12}  {val1:>12g}"
            if len(pair) == 2:
                row2, val2 = pair[1]
                line += f"   {row2:<12}  {val2:>12g}"
            lines.append(line)

    # RHS
    rhs_entries: list[tuple[str, float]] = []
    for row_name, _, _, rhs in prob.rows:
        if rhs != 0.0:
            rhs_entries.append((row_name, rhs))

    if rhs_entries:
        lines.append("RHS")
        for row_name, rhs in rhs_entries:
            lines.append(f"    RHS           {row_name:<12}  {rhs:>12g}")

    # BOUNDS
    # デフォルト: 0 <= x < +inf
    # FR: -inf < x < +inf (free)
    # MI: -inf <= x (lb = -inf, ub >= 0 または +inf)
    # PL: 0 <= x < +inf (デフォルトと同じ)
    # LO: 下界設定
    # UP: 上界設定
    # FX: 固定
    bound_lines: list[str] = []
    for col, (lb, ub) in prob.bounds.items():
        if lb is None and ub is None:
            # FR: 完全自由変数
            bound_lines.append(f" FR BND           {col}")
        elif lb is None and ub is not None:
            # MI + UP
            bound_lines.append(f" MI BND           {col}")
            if ub != float('inf') and ub is not None:
                bound_lines.append(f" UP BND           {col:<12}  {ub:>12g}")
        elif lb is not None and lb != 0.0:
            bound_lines.append(f" LO BND           {col:<12}  {lb:>12g}")
            if ub is not None and ub != float('inf'):
                bound_lines.append(f" UP BND           {col:<12}  {ub:>12g}")
        elif lb == 0.0:
            # デフォルト下界
            if ub is not None and ub != float('inf'):
                bound_lines.append(f" UP BND           {col:<12}  {ub:>12g}")
            # else: デフォルトのまま

    if bound_lines:
        lines.append("BOUNDS")
        lines.extend(bound_lines)

    lines.append("ENDATA")
    out_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# LP unbounded 問題生成関数
# =============================================================================

def make_col_names(n: int, prefix: str = "x") -> list[str]:
    return [f"{prefix}{i+1}" for i in range(n)]


def gen_lp_unbd_free_var_1d() -> LpProblem2:
    """n=1, x free, minimize -x → unbounded (ray d=1).

    構造: minimize -x (c=-1), no constraints.
    自由変数 x → -inf 方向に無制限。
    ray: d=1, c^T d = -1 < 0, Ad = 0 >= 0.
    """
    name = "UNBD_LP_FREE1D"
    var_names = ["x1"]
    obj_coeffs = {"x1": -1.0}
    rows: list = []
    bounds = {"x1": (None, None)}  # free
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_free_var_2d() -> LpProblem2:
    """n=2, x1 free, minimize -(x1 + x2), x2 >= 0 制約あり.

    ray: d=(1,0) → c^T d = -1 < 0, x1 free なので境界なし、Ad = 0 (制約なし).
    """
    name = "UNBD_LP_FREE2D"
    var_names = ["x1", "x2"]
    obj_coeffs = {"x1": -1.0, "x2": -1.0}
    # x2 >= 1 制約 (これは x1 に無関係で ray d=(1,0) は可能)
    rows = [("C1", "G", {"x2": 1.0}, 1.0)]
    bounds = {
        "x1": (None, None),  # free
        "x2": (0.0, None),   # x2 >= 0
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_ineq_loose_n3() -> LpProblem2:
    """n=3, 不等式が緩い → ray d=(1,1,0) が存在.

    制約: x1 + x2 <= 100 (緩い上界), x3 = 1 (固定)
    目的: minimize -(x1 + x2)
    ray d=(1,1,0): Ad = 1+1=2 > 0 となるが L 制約なので Ad <= 0 が必要 → 別の設計へ。

    正しい設計:
      制約なし (x1,x2 は x >= 0 のみ), 目的 minimize -(x1 + x2 + x3)
      → x = (t, t, t) → -∞
    しかし x >= 0 で下界なし → unbounded。
    """
    name = "UNBD_LP_INEQ_N3"
    var_names = ["x1", "x2", "x3"]
    obj_coeffs = {"x1": -1.0, "x2": -2.0, "x3": -1.0}
    # no constraints (x >= 0 はデフォルト)
    rows: list = []
    bounds = {
        "x1": (0.0, None),
        "x2": (0.0, None),
        "x3": (0.0, None),
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_ray_mixed_n4() -> LpProblem2:
    """n=4, 複数変数が絡む unbounded ray.

    設計:
      minimize -(x1 + x3)
      制約: x1 - x2 <= 5, x3 - x4 <= 5, x2 >= 0, x4 >= 0
      変数: x1, x3 free; x2, x4 >= 0

    ray d=(1,1,1,1): x1 free なので d1=1 可能, x3 free なので d3=1 可能。
      A*d = d1 - d2 = 0 (<= 5: 0 <= 5 OK), d3 - d4 = 0 (<= 5: OK)
      c^T d = -1 - 1 = -2 < 0 → unbounded。
    """
    name = "UNBD_LP_RAY_N4"
    var_names = ["x1", "x2", "x3", "x4"]
    obj_coeffs = {"x1": -1.0, "x3": -1.0}
    rows = [
        ("C1", "L", {"x1": 1.0, "x2": -1.0}, 5.0),
        ("C2", "L", {"x3": 1.0, "x4": -1.0}, 5.0),
    ]
    bounds = {
        "x1": (None, None),  # free
        "x2": (0.0, None),
        "x3": (None, None),  # free
        "x4": (0.0, None),
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_eq_only_free_n5() -> LpProblem2:
    """n=5, 等号制約のみ + 自由変数 → 等号が縮退して ray が存在.

    設計:
      等号制約 m=2: A x = b, rank(A) = 2 < n=5
      null space に c^T d < 0 となる方向 d が存在。

    A = [[1,1,0,0,0], [0,0,1,1,0]]
    b = [1, 1]
    null space には e.g. d=(1,-1,0,0,0), d=(0,0,1,-1,0), d=(0,0,0,0,1) が含まれる。
    c = (0,0,0,0,-1) → d=(0,0,0,0,1) で c^T d = -1 < 0, Ad = 0 (制約は等号のまま).
    x5 free → d 方向に進める。
    """
    name = "UNBD_LP_EQ_FREE_N5"
    var_names = ["x1", "x2", "x3", "x4", "x5"]
    obj_coeffs = {"x5": -1.0}
    rows = [
        ("C1", "E", {"x1": 1.0, "x2": 1.0}, 1.0),
        ("C2", "E", {"x3": 1.0, "x4": 1.0}, 1.0),
    ]
    bounds = {
        "x1": (0.0, None),
        "x2": (0.0, None),
        "x3": (0.0, None),
        "x4": (0.0, None),
        "x5": (None, None),  # free
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_loose_ineq_n2() -> LpProblem2:
    """n=2, 不等式制約が緩い (x >= 0 のみ) → 無制限.

    設計:
      minimize -(3x1 + 2x2)
      制約: -x1 + x2 <= 10 (ray に平行またはネガティブ方向でOK)
      x1, x2 >= 0

    ray d=(1,1): c^T d = -3 - 2 = -5 < 0
      A*d = -1+1 = 0 <= 10 OK → unbounded。
    """
    name = "UNBD_LP_LOOSE_N2"
    var_names = ["x1", "x2"]
    obj_coeffs = {"x1": -3.0, "x2": -2.0}
    rows = [
        ("C1", "L", {"x1": -1.0, "x2": 1.0}, 10.0),
    ]
    bounds = {
        "x1": (0.0, None),
        "x2": (0.0, None),
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_medium_n10() -> LpProblem2:
    """n=10, 中規模 unbounded LP.

    設計:
      変数 x1..x5: free
      変数 x6..x10: >= 0
      制約: x1 + x2 + x3 = 0 (等号), x4 + x5 >= -5
      目的: minimize -(x1 + x6) → x1 free で無制限
      ray d = e1 (x1 方向): c^T d = -1 < 0, A*d = 1 (等号 C1 = 1+0+0 ≠ 0)
      → 等号制約を満たさない。修正: x1 を等号から除く。

    修正設計:
      等号: x2 + x3 = 0 (x2, x3 は free で相殺できる)
      目的: minimize -(x1 + x6), x1 free
      ray d = (1,0,...,0): c^T d = -1 < 0
        A*d = 0 (等号: d2+d3 = 0 OK), x6 は自由でなく d6=0 でOK。
    """
    name = "UNBD_LP_MEDIUM_N10"
    n = 10
    var_names = make_col_names(n)
    obj_coeffs = {"x1": -1.0, "x6": -1.0}
    rows = [
        ("C1", "E", {"x2": 1.0, "x3": 1.0}, 0.0),
        ("C2", "G", {"x4": 1.0, "x5": 1.0}, -5.0),
        ("C3", "L", {"x7": 1.0, "x8": 1.0}, 20.0),
    ]
    bounds: dict[str, tuple] = {}
    for i in range(5):
        bounds[f"x{i+1}"] = (None, None)  # x1..x5 free
    for i in range(5, 10):
        bounds[f"x{i+1}"] = (0.0, None)   # x6..x10 >= 0
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_medium_n15() -> LpProblem2:
    """n=15, 中規模 unbounded LP with multiple free variables.

    設計:
      x1..x8 free, x9..x15 >= 0
      制約: x1+x2=1 (等号), x3+x4+x5 >= 0, x6-x7 <= 10,
            x9+x10 <= 100, x11+x12 >= 0
      目的: minimize -(2x1 + x8 + x13)
      ray d = (1,−1, 0,...,0, 0,...): c^T d = -2 < 0,
        A*d: C1: d1+d2 = 0 OK, C2: d3+d4+d5 = 0 OK,
        C3: d6-d7 = 0 OK, 他は 0。
      x1 free なので d1=1, x2 free で d2=-1, others 0。
    """
    name = "UNBD_LP_MEDIUM_N15"
    n = 15
    var_names = make_col_names(n)
    obj_coeffs = {"x1": -2.0, "x8": -1.0, "x13": -1.0}
    rows = [
        ("C1", "E", {"x1": 1.0, "x2": 1.0}, 1.0),
        ("C2", "G", {"x3": 1.0, "x4": 1.0, "x5": 1.0}, 0.0),
        ("C3", "L", {"x6": 1.0, "x7": -1.0}, 10.0),
        ("C4", "L", {"x9": 1.0, "x10": 1.0}, 100.0),
        ("C5", "G", {"x11": 1.0, "x12": 1.0}, 0.0),
    ]
    bounds: dict[str, tuple] = {}
    for i in range(8):
        bounds[f"x{i+1}"] = (None, None)
    for i in range(8, n):
        bounds[f"x{i+1}"] = (0.0, None)
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_no_constraints_n20() -> LpProblem2:
    """n=20, 制約なし (変数は全て x >= 0 のみ).

    minimize 負の係数をもつ目的関数 → 全変数が無制限に増大。
    ray d = (1,1,...,1): c^T d = sum(c) < 0 (全 ci < 0).
    """
    name = "UNBD_LP_NOCON_N20"
    n = 20
    var_names = make_col_names(n)
    rng = random.Random(42)
    obj_coeffs = {f"x{i+1}": -rng.uniform(0.5, 3.0) for i in range(n)}
    rows: list = []
    bounds = {f"x{i+1}": (0.0, None) for i in range(n)}
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_sparse_constraints_n12() -> LpProblem2:
    """n=12, スパース制約 + 一方向に非有界.

    設計:
      x1..x4 free; x5..x12 >= 0
      制約:
        x1 + x2 <= 10 (L)
        x3 - x4 >= -5 (G, both free)
        x5 + x6 <= 20 (L)
        x7 + x8 + x9 >= 0 (G)
      目的: minimize -(x1 + x3 + x5 + x7)
      ray d = (1, -1, 1, 1, 0, ..., 0):
        x1 free d1=1, x2 free d2=-1, x3 free d3=1, x4 free d4=1
        c^T d = -1 - 1 = -2 (x5..x12 係数なし) < 0
        C1: d1+d2 = 0 <= 10 OK
        C2: d3-d4 = 0 >= -5 OK
        C3: 0 <= 20 OK
        C4: 0 >= 0 OK
    """
    name = "UNBD_LP_SPARSE_N12"
    n = 12
    var_names = make_col_names(n)
    obj_coeffs = {"x1": -1.0, "x3": -1.0, "x5": -1.0, "x7": -1.0}
    rows = [
        ("C1", "L", {"x1": 1.0, "x2": 1.0}, 10.0),
        ("C2", "G", {"x3": 1.0, "x4": -1.0}, -5.0),
        ("C3", "L", {"x5": 1.0, "x6": 1.0}, 20.0),
        ("C4", "G", {"x7": 1.0, "x8": 1.0, "x9": 1.0}, 0.0),
    ]
    bounds: dict[str, tuple] = {}
    for i in range(4):
        bounds[f"x{i+1}"] = (None, None)
    for i in range(4, n):
        bounds[f"x{i+1}"] = (0.0, None)
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_neg_lower_bound_n5() -> LpProblem2:
    """n=5, 負の下界を持つ変数 → ray 方向に無制限.

    設計:
      x1 >= -100 (lb=-100, ub=+inf)
      制約: x1 + x2 + x3 >= 0
      目的: minimize -(x1 + x2 + x3) → x=(t,t,t) → -∞
      ray d=(1,1,1,0,0): c^T d = -3 < 0
        C1: d1+d2+d3 = 3 >= 0 OK
    """
    name = "UNBD_LP_NEGLB_N5"
    n = 5
    var_names = make_col_names(n)
    obj_coeffs = {"x1": -1.0, "x2": -1.0, "x3": -1.0}
    rows = [
        ("C1", "G", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, 0.0),
    ]
    bounds = {
        "x1": (-100.0, None),
        "x2": (0.0, None),
        "x3": (0.0, None),
        "x4": (0.0, None),
        "x5": (0.0, None),
    }
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


def gen_lp_unbd_multi_free_n8() -> LpProblem2:
    """n=8, 複数の自由変数 + 制約が ray を許容.

    設計:
      x1..x4 free; x5..x8 >= 0
      等号制約: x1+x2=0, x3+x4=0 (null space に (1,-1,0,0,0,0,0,0) が存在)
      目的: minimize -(x1 + x3) → ray d=(1,-1,1,-1,0,0,0,0)
        c^T d = -1 - 1 = -2 < 0
        C1: d1+d2 = 0 OK (等号)
        C2: d3+d4 = 0 OK (等号)
    """
    name = "UNBD_LP_MULTIFREE_N8"
    n = 8
    var_names = make_col_names(n)
    obj_coeffs = {"x1": -1.0, "x3": -1.0}
    rows = [
        ("C1", "E", {"x1": 1.0, "x2": 1.0}, 0.0),
        ("C2", "E", {"x3": 1.0, "x4": 1.0}, 0.0),
    ]
    bounds: dict[str, tuple] = {}
    for i in range(4):
        bounds[f"x{i+1}"] = (None, None)
    for i in range(4, n):
        bounds[f"x{i+1}"] = (0.0, None)
    return LpProblem2(name, "OBJ", obj_coeffs, rows, var_names, bounds)


# =============================================================================
# PROBLEMS リスト
# =============================================================================

PROBLEMS = [
    gen_lp_unbd_free_var_1d,
    gen_lp_unbd_free_var_2d,
    gen_lp_unbd_ineq_loose_n3,
    gen_lp_unbd_ray_mixed_n4,
    gen_lp_unbd_eq_only_free_n5,
    gen_lp_unbd_loose_ineq_n2,
    gen_lp_unbd_medium_n10,
    gen_lp_unbd_medium_n15,
    gen_lp_unbd_no_constraints_n20,
    gen_lp_unbd_sparse_constraints_n12,
    gen_lp_unbd_neg_lower_bound_n5,
    gen_lp_unbd_multi_free_n8,
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="LP unbounded problem generator (MPS format)")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "lp_problems_unbounded"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen_fn in PROBLEMS:
        prob = gen_fn()
        out_path = out_dir / f"{prob.name}.QPS"
        write_mps2(prob, out_path)
        print(f"  {prob.name}: n={len(prob.var_names)} m={len(prob.rows)} -> {out_path.name}")

    print(f"\n生成完了: {len(PROBLEMS)} 問題 -> {out_dir}")


if __name__ == "__main__":
    main()
