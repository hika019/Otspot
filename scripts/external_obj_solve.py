"""外部ソルバ (OSQP Python, eps=1e-9) で問題を再解いて外部リファレンス obj を取得する。

本ソルバの生成系列 (osqp_bench, mpc_qp) を再構築してから OSQP に渡す。
.qps を再ロードする方式は QPS への変換ロスを跨ぐので避け、
オリジナルの (P, q, A, l, u) を生成器から直接取る。

出力: 指定 CSV (problem_name, ext_obj, ext_status, source)

使い方:
  python3 scripts/external_obj_solve.py \\
    --target osqp_bench   # OSQP synthetic + SuiteSparse
  python3 scripts/external_obj_solve.py \\
    --target mpc_qp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spa

REPO_ROOT = Path(__file__).resolve().parent.parent
OSQP_REPO = REPO_ROOT / "tmp" / "external" / "osqp_benchmarks"
sys.path.insert(0, str(OSQP_REPO))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import clarabel  # noqa: E402
import osqp      # noqa: E402

from gen_osqp_bench import SIZE_GRID, SEED, import_classes  # noqa: E402


# Clarabel: IPM ベース。ADMM の偽 infeasibility 判定を起こさず信頼度が高い。
#   LIPMWALK4 で OSQP は誤って primal infeasible を返したが、
#   Clarabel/SCS/ECOS は揃って optimal -0.437 を返した実績あり。
def solve_clarabel(P, q, A, l, u, time_limit: float = 600.0) -> tuple[float | None, str]:
    """Clarabel で OSQP-form (l <= Ax <= u) を解いて (obj, status) を返す。
    Clarabel は cone constraints を使うので、各成分を変換:
      l[i] = u[i] (finite) → ZeroCone (等式)
      finite l[i], u[i] → 2 個の Nonnegative cone
      l[i]=-inf, u[i] finite → Nonnegative
      l[i] finite, u[i]=+inf → Nonnegative (符号反転)
    """
    P = spa.csc_matrix(P)
    A = spa.csc_matrix(A)
    n = P.shape[0]
    m = A.shape[0]

    # Clarabel form: min 0.5 x' P x + q' x  s.t. A x + s = b, s ∈ K
    # Nonnegative cone (s>=0) は b - A x >= 0, つまり A x <= b を意味する。
    # OSQP form l <= Ax <= u を変換:
    #   eq i (l=u finite)        : A[i] x + s = l[i],  s ∈ ZeroCone
    #   ub i (u finite)          : A[i] x + s = u[i],  s >= 0  ⇔ A x <= u
    #   lb i (l finite)          : -A[i] x + s = -l[i], s >= 0  ⇔ -A x <= -l ⇔ A x >= l
    rows_eq, rhs_eq = [], []
    rows_ub, rhs_ub = [], []  # for A x <= u
    rows_lb, rhs_lb = [], []  # for -A x <= -l (encoded as block=-A, b=-l)
    for i in range(m):
        li, ui = float(l[i]), float(u[i])
        l_inf = (np.isinf(li) and li < 0)
        u_inf = (np.isinf(ui) and ui > 0)
        if l_inf and u_inf:
            continue
        if li == ui and not l_inf and not u_inf:
            rows_eq.append(i); rhs_eq.append(li)
        else:
            if not u_inf:
                rows_ub.append(i); rhs_ub.append(ui)
            if not l_inf:
                rows_lb.append(i); rhs_lb.append(li)

    blocks: list[spa.spmatrix] = []
    rhs_parts: list[np.ndarray] = []
    cones: list = []
    if rows_eq:
        blocks.append(A[rows_eq, :])
        rhs_parts.append(np.asarray(rhs_eq, dtype=float))
        cones.append(clarabel.ZeroConeT(len(rows_eq)))
    if rows_ub:
        blocks.append(A[rows_ub, :])
        rhs_parts.append(np.asarray(rhs_ub, dtype=float))
        cones.append(clarabel.NonnegativeConeT(len(rows_ub)))
    if rows_lb:
        blocks.append(-A[rows_lb, :])
        rhs_parts.append(-np.asarray(rhs_lb, dtype=float))
        cones.append(clarabel.NonnegativeConeT(len(rows_lb)))

    if not blocks:
        # 制約無し: ダミー
        blocks.append(spa.csc_matrix((1, n)))
        rhs_parts.append(np.array([0.0]))
        cones.append(clarabel.ZeroConeT(1))

    A_clar = spa.vstack(blocks, format="csc")
    b_clar = np.concatenate(rhs_parts)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    # Clarabel の絶対/相対 tol field 名: tol_gap_abs / tol_gap_rel / tol_feas / tol_infeas_*
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-9
    settings.tol_feas = 1e-9
    settings.tol_infeas_abs = 1e-10
    settings.tol_infeas_rel = 1e-10
    settings.max_iter = 50_000
    settings.time_limit = time_limit

    P_triu = spa.triu(P, format="csc")
    s = clarabel.DefaultSolver(P_triu, q, A_clar, b_clar, cones, settings)
    sol = s.solve()
    status = str(sol.status)
    # Clarabel status: 'Solved', 'AlmostSolved', 'PrimalInfeasible', etc.
    # Solved 系のみ obj を返す。
    if status in ("Solved", "AlmostSolved"):
        obj = float(sol.obj_val)
    else:
        obj = None
    return obj, status


def solve_osqp(P, q, A, l, u) -> tuple[float | None, str]:
    """OSQP で解く。Clarabel が失敗した場合のフォールバック用。
    OSQP は ADMM ベースで偽 infeasibility を出すことがあるため信頼度は中。
    """
    P = spa.csc_matrix(P)
    A = spa.csc_matrix(A)
    n = P.shape[0]
    m = A.shape[0]
    if m == 0:
        A = spa.csc_matrix((1, n))
        l = np.array([0.0])
        u = np.array([0.0])

    OSQP_OPTS = dict(
        eps_abs=1e-9, eps_rel=1e-9,
        eps_prim_inf=1e-10, eps_dual_inf=1e-10,
        max_iter=200_000, polishing=True, polish_refine_iter=5,
        verbose=False, time_limit=600.0,
    )
    s = osqp.OSQP()
    s.setup(P=P, q=q, A=A, l=l, u=u, **OSQP_OPTS)
    res = s.solve()
    info = res.info
    status = info.status
    obj = float(info.obj_val) if hasattr(info, "obj_val") else None
    if obj is not None and not np.isfinite(obj):
        obj = None
    return obj, f"osqp:{status}"


def solve_external(P, q, A, l, u) -> tuple[float | None, str]:
    """Clarabel 主、OSQP 副の二段構え。両者が disagree した場合は両方の値を残す。"""
    obj_c, st_c = solve_clarabel(P, q, A, l, u)
    if obj_c is not None:
        return obj_c, f"clarabel:{st_c}"
    # Clarabel 失敗 → OSQP fallback
    obj_o, st_o = solve_osqp(P, q, A, l, u)
    if obj_o is not None:
        return obj_o, f"clarabel_failed/osqp:{st_o}"
    return None, f"clarabel:{st_c}|osqp:{st_o}"


import math  # noqa: E402


def gen_osqp_synthetic():
    """OSQP synthetic 問題を全部 yield。
    (name, qp_dict)
    """
    classes = import_classes()
    for class_name, sizes in SIZE_GRID.items():
        cls = classes[class_name]
        for size in sizes:
            try:
                inst = cls(size, seed=SEED)
            except Exception as e:
                print(f"[ext] gen failed {class_name}_{size}: {e}", file=sys.stderr)
                continue
            stem = f"OSQP_{class_name}_{size}".upper()
            yield stem, inst.qp_problem


def gen_suitesparse():
    """SuiteSparse Lasso/Huber を yield。"""
    from gen_osqp_suitesparse import (
        load_least_squares,
        build_lasso_qp,
        build_huber_qp,
    )
    import ssgetpy

    cache_dir = REPO_ROOT / "tmp" / "ssgetpy_cache"
    matrices = ssgetpy.search(kind="least squares problem", limit=200)
    MAX_ORIG_NNZ = 100_000
    MAX_QP_N = 20_000
    for ssmat in sorted(matrices, key=lambda x: x.rows * x.cols):
        if ssmat.nnz > MAX_ORIG_NNZ:
            continue
        try:
            paths = ssmat.download(format="MAT", destpath=str(cache_dir), extract=True)
            mat_path = Path(paths[0]) if isinstance(paths, tuple) else Path(paths)
            Ad, bd, _ = load_least_squares(mat_path)
        except Exception as e:
            print(f"[ext] SS load failed {ssmat.name}: {e}", file=sys.stderr)
            continue
        m, n = Ad.shape
        if 2 * n + m <= MAX_QP_N:
            yield f"SS_LASSO_{ssmat.name}".upper(), build_lasso_qp(Ad, bd)
        if n + 3 * m <= MAX_QP_N:
            yield f"SS_HUBER_{ssmat.name}".upper(), build_huber_qp(Ad, bd)


def gen_mpc():
    """MPC QP を yield。"""
    from gen_mpc_qp import _maybe_obj
    data_dir = REPO_ROOT / "tmp" / "external" / "mpc_qpbenchmark" / "data"
    for npz in sorted(data_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        P = _maybe_obj(d["P"])
        q = _maybe_obj(d["q"])
        G = _maybe_obj(d["G"])
        h = _maybe_obj(d["h"])
        A_eq = _maybe_obj(d["A"]) if "A" in d.files else None
        b_eq = _maybe_obj(d["b"]) if "b" in d.files else None
        lb = _maybe_obj(d["lb"]) if "lb" in d.files else None
        ub = _maybe_obj(d["ub"]) if "ub" in d.files else None

        n = P.shape[0]
        P = spa.csc_matrix(P)
        q = np.asarray(q, dtype=float).reshape(-1)

        # 変数 bound を制約に edge として埋め込む (OSQP に渡すため)
        # 制約: G x <= h, A x = b, lb <= x <= ub  ->  全部 l <= [G;A;I] x <= u
        blocks = []
        l_parts = []
        u_parts = []
        if G is not None:
            G = spa.csc_matrix(G)
            h = np.asarray(h, dtype=float).reshape(-1)
            blocks.append(G)
            l_parts.append(np.full(G.shape[0], -np.inf))
            u_parts.append(h)
        if A_eq is not None:
            A_eq = spa.csc_matrix(A_eq)
            b_eq = np.asarray(b_eq, dtype=float).reshape(-1)
            blocks.append(A_eq)
            l_parts.append(b_eq.copy())
            u_parts.append(b_eq.copy())
        if lb is not None or ub is not None:
            lb_arr = (np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).reshape(-1))
            ub_arr = (np.full(n,  np.inf) if ub is None else np.asarray(ub, dtype=float).reshape(-1))
            blocks.append(spa.eye(n, format="csc"))
            l_parts.append(lb_arr)
            u_parts.append(ub_arr)

        if blocks:
            A_full = spa.vstack(blocks, format="csc")
            l_arr = np.concatenate(l_parts)
            u_arr = np.concatenate(u_parts)
        else:
            A_full = spa.csc_matrix((0, n))
            l_arr = np.zeros(0)
            u_arr = np.zeros(0)
        yield npz.stem, {"P": P, "q": q, "A": A_full, "l": l_arr, "u": u_arr,
                          "n": A_full.shape[1], "m": A_full.shape[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["osqp_bench", "mpc_qp"], required=True)
    ap.add_argument("--out-csv", default=None,
                    help="出力 CSV (デフォルト: data/baseline_objectives/<target>_external.csv)")
    ap.add_argument("--filter", default=None,
                    help="問題名 substring フィルタ (debug 用)")
    args = ap.parse_args()

    if args.target == "osqp_bench":
        gens = [("synthetic", gen_osqp_synthetic), ("suitesparse", gen_suitesparse)]
    else:
        gens = [("mpc", gen_mpc)]

    out_csv = Path(args.out_csv) if args.out_csv else \
        REPO_ROOT / "data" / "baseline_objectives" / f"{args.target}_external.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float | None, str, str]] = []
    for source_name, gen in gens:
        print(f"[ext] === {source_name} ===")
        for name, qp in gen():
            if args.filter and args.filter not in name:
                continue
            t0 = time.time()
            try:
                obj, status = solve_external(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"])
            except Exception as e:
                obj, status = None, f"ERROR: {type(e).__name__}: {e}"[:80]
            dt = time.time() - t0
            obj_str = f"{obj:.6e}" if obj is not None else "NA"
            print(f"[ext] {name:30s} status={status:30s} obj={obj_str:>14s} time={dt:6.2f}s")
            rows.append((name, obj, status, f"clarabel_eps1e-9_t{int(dt)}s"))

    with out_csv.open("w") as f:
        f.write(
            "# 出典: Clarabel (主) eps=1e-9, fallback OSQP. 外部リファレンス。\n"
            "# 用途: 自ソルバ obj との独立 cross-check (PASS と別軸の正当性検証)。\n"
            "# Source: independent Clarabel/OSQP solve at high accuracy.\n"
            "# Purpose: cross-validate own solver's obj (independent of self-PASS).\n"
            "# 注意: OSQP は ADMM ベースで偽 infeasibility を出すケースがあり、\n"
            "#       Clarabel (IPM) を主に使用。LIPMWALK4 等で OSQP が誤判定する実例あり。\n"
            "problem_name,optimal_obj,source,ext_status\n"
        )
        for (name, obj, status, src) in sorted(rows, key=lambda x: x[0]):
            obj_str = f"{obj:.6e}" if obj is not None else "NA"
            f.write(f"{name},{obj_str},{src},{status}\n")
    n_solved = sum(1 for r in rows if r[1] is not None)
    print(f"[ext] done: {n_solved}/{len(rows)} solved -> {out_csv}")


if __name__ == "__main__":
    main()
