"""SuiteSparse Matrix Collection の "least squares problem" から
OSQP の Lasso / Huber QP を構築して .qps として書き出す。

参考実装: osqp_benchmarks/problem_classes/suitesparse_{lasso,huber}.py
ダウンロード: ssgetpy (Julia の MatrixDepot.jl 代替)

ファイル名: SS_LASSO_<name>.qps / SS_HUBER_<name>.qps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as spio
import scipy.sparse as spa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from qp_to_qps import write_qps  # noqa: E402


SEED = 0  # download.jl の Random.seed!(0) と同じ


def load_least_squares(mat_path: Path):
    """SuiteSparse .mat (Problem struct) を読み、(A, b) を返す。
    b が無い場合は固定 seed の synthetic b = A x0 + s0 を作る。
    download.jl と同じロジック。
    """
    d = spio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    P = d["Problem"]
    A = spa.csc_matrix(P.A)
    m, n = A.shape
    if hasattr(P, "b"):
        b = np.asarray(P.b, dtype=float).reshape(-1)
        assert b.shape == (m,), f"b shape {b.shape} != ({m},)"
        b_source = "real"
    else:
        rng = np.random.RandomState(SEED)
        x0 = rng.randn(n)
        s0 = rng.randn(m)
        b = A @ x0 + s0
        b_source = "synthetic"
    return A, b, b_source


def build_lasso_qp(Ad: spa.csc_matrix, bd: np.ndarray) -> dict:
    """suitesparse_lasso.py と同一の QP を構築。
    minimize  y'y + lambda 1'.t
    s.t.      y = Ax - b
              -t <= x <= t
    変数順序: x (n), y (m), t (n). 全 n+m+n 次元。
    """
    m, n = Ad.shape
    lambda_max = float(np.linalg.norm(Ad.T @ bd, np.inf))
    lambda_param = (1.0 / 5.0) * lambda_max

    P = spa.block_diag(
        (spa.csc_matrix((n, n)), 2 * spa.eye(m), spa.csc_matrix((n, n))),
        format="csc",
    )
    q = np.concatenate([np.zeros(m + n), lambda_param * np.ones(n)])
    In = spa.eye(n)
    Onm = spa.csc_matrix((n, m))
    A = spa.vstack([
        spa.hstack([Ad, -spa.eye(m), spa.csc_matrix((m, n))]),  # y = Ax - b
        spa.hstack([In, Onm, -In]),                              # x - t <= 0
        spa.hstack([In, Onm, In]),                               # x + t >= 0
    ]).tocsc()
    l = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
    u = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])
    return {"P": P, "q": q, "A": A, "l": l, "u": u, "n": A.shape[1], "m": A.shape[0]}


def build_huber_qp(Ad: spa.csc_matrix, bd: np.ndarray) -> dict:
    """suitesparse_huber.py と同一の QP。
    minimize  1/2 z'z + 1'(r + s)
    s.t.      Ax - b - z = r - s
              r >= 0, s >= 0
    変数順序: x (n), z (m), r (m), s (m). 全 n+3m 次元。
    """
    m, n = Ad.shape
    Im = spa.eye(m)
    P = spa.block_diag(
        (spa.csc_matrix((n, n)), Im, spa.csc_matrix((2 * m, 2 * m))),
        format="csc",
    )
    q = np.hstack([np.zeros(n + m), np.ones(2 * m)])
    A = spa.bmat([
        [Ad,    -Im,   -Im,   Im],
        [None,  None,   Im,   None],
        [None,  None,   None, Im],
    ], format="csc")
    l = np.hstack([bd, np.zeros(2 * m)])
    u = np.hstack([bd, np.inf * np.ones(2 * m)])
    return {"P": P, "q": q, "A": A, "l": l, "u": u, "n": A.shape[1], "m": A.shape[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "osqp_bench"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "tmp" / "ssgetpy_cache"))
    ap.add_argument("--max-orig-nnz", type=int, default=100_000,
                    help="元行列 nnz 上限 (拡張後 QP nnz はこれより数倍大きい)")
    ap.add_argument("--max-qp-n", type=int, default=20_000,
                    help="生成 QP の変数次元上限")
    ap.add_argument("--limit", type=int, default=200,
                    help="ssgetpy.search の取得上限")
    args = ap.parse_args()

    import ssgetpy

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = ssgetpy.search(kind="least squares problem", limit=args.limit)
    print(f"[gen_ss] SuiteSparse 'least squares problem': {len(matrices)} matrices")

    written = 0
    skipped = 0
    for ssmat in sorted(matrices, key=lambda x: x.rows * x.cols):
        if ssmat.nnz > args.max_orig_nnz:
            print(f"[gen_ss] skip {ssmat.name}: nnz={ssmat.nnz} > {args.max_orig_nnz}")
            skipped += 2
            continue

        # ダウンロード (キャッシュあれば再利用)
        try:
            paths = ssmat.download(format="MAT", destpath=str(cache_dir), extract=True)
            mat_path = Path(paths[0]) if isinstance(paths, tuple) else Path(paths)
        except Exception as e:
            print(f"[gen_ss] FAIL download {ssmat.name}: {e}", file=sys.stderr)
            skipped += 2
            continue

        try:
            Ad, bd, b_source = load_least_squares(mat_path)
        except Exception as e:
            print(f"[gen_ss] FAIL load {ssmat.name}: {e}", file=sys.stderr)
            skipped += 2
            continue

        m, n = Ad.shape

        # Lasso
        n_qp_lasso = 2 * n + m
        if n_qp_lasso > args.max_qp_n:
            print(f"[gen_ss] skip LASSO {ssmat.name}: qp_n={n_qp_lasso} > {args.max_qp_n}")
        else:
            qp = build_lasso_qp(Ad, bd)
            stem = f"SS_LASSO_{ssmat.name}".upper()
            out = out_dir / f"{stem}.qps"
            try:
                write_qps(stem[:8], qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], out)
                written += 1
                print(f"[gen_ss] wrote {out.name}  m={m} n={n} qp_n={qp['n']} qp_m={qp['m']} b={b_source}")
            except Exception as e:
                print(f"[gen_ss] FAIL write LASSO {ssmat.name}: {e}", file=sys.stderr)
                skipped += 1

        # Huber
        n_qp_huber = n + 3 * m
        if n_qp_huber > args.max_qp_n:
            print(f"[gen_ss] skip HUBER {ssmat.name}: qp_n={n_qp_huber} > {args.max_qp_n}")
        else:
            qp = build_huber_qp(Ad, bd)
            stem = f"SS_HUBER_{ssmat.name}".upper()
            out = out_dir / f"{stem}.qps"
            try:
                write_qps(stem[:8], qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], out)
                written += 1
                print(f"[gen_ss] wrote {out.name}  m={m} n={n} qp_n={qp['n']} qp_m={qp['m']} b={b_source}")
            except Exception as e:
                print(f"[gen_ss] FAIL write HUBER {ssmat.name}: {e}", file=sys.stderr)
                skipped += 1

    print(f"[gen_ss] done: wrote={written} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
