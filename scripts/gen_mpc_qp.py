"""qpsolvers/mpc_qpbenchmark の .npz から QPS を生成する。

入力 schema (qpsolvers convention):
  P, q       : 目的 (1/2 x'Px + q'x)
  G, h       : 不等式 G x <= h            (None なら無し)
  A, b       : 等式   A x = b              (None なら無し)
  lb, ub     : 変数 bound  lb <= x <= ub   (None なら ±inf)

QPS への変換: G/A を縦に積んで constraint 行列とし、
  G 行: l = -inf, u = h
  A 行: l = u = b
を統合した A_full, l, u を qp_to_qps に渡す。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as spa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from qp_to_qps import write_qps  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "tmp" / "external" / "mpc_qpbenchmark" / "data"


def _maybe_obj(x):
    """object dtype の 0-dim array → 中身。それ以外はそのまま。"""
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    return x


def convert_npz(npz_path: Path, out_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
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

    blocks: list[spa.spmatrix] = []
    l_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []

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

    if blocks:
        A_full = spa.vstack(blocks, format="csc")
        l_arr = np.concatenate(l_parts)
        u_arr = np.concatenate(u_parts)
    else:
        # 制約無し (Hessian だけ)。空の constraint 行列。
        A_full = spa.csc_matrix((0, n))
        l_arr = np.zeros(0)
        u_arr = np.zeros(0)

    var_lb = None if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    var_ub = None if ub is None else np.asarray(ub, dtype=float).reshape(-1)

    name_stem = npz_path.stem
    write_qps(
        name=name_stem[:8],
        P=P, q=q,
        A=A_full, l=l_arr, u=u_arr,
        var_lb=var_lb, var_ub=var_ub,
        out_path=out_path,
    )
    return {
        "n": n,
        "m_ineq": (G.shape[0] if G is not None else 0),
        "m_eq": (A_eq.shape[0] if A_eq is not None else 0),
        "nnz_P": P.nnz,
        "nnz_A": A_full.nnz,
        "has_var_bounds": (var_lb is not None or var_ub is not None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="入力 .npz ディレクトリ")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "mpc_qp"),
                    help="出力 .qps ディレクトリ")
    ap.add_argument("--max-n", type=int, default=None,
                    help="変数数 n がこれを超える問題はスキップ")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.npz"))
    if not files:
        print(f"[gen_mpc_qp] no .npz in {data_dir}", file=sys.stderr)
        return 1

    written = 0
    skipped = 0
    for npz in files:
        try:
            info_pre = np.load(npz, allow_pickle=True)
            n_pre = int(info_pre["P"].shape[0])
            info_pre.close()
            if args.max_n is not None and n_pre > args.max_n:
                print(f"[gen_mpc_qp] skip {npz.name}: n={n_pre} > max_n={args.max_n}")
                skipped += 1
                continue
            out = out_dir / f"{npz.stem}.qps"
            info = convert_npz(npz, out)
            print(
                f"[gen_mpc_qp] wrote {out.name}  n={info['n']} "
                f"m_ineq={info['m_ineq']} m_eq={info['m_eq']} "
                f"nnz(P)={info['nnz_P']} nnz(A)={info['nnz_A']} "
                f"vbnd={'Y' if info['has_var_bounds'] else 'N'}"
            )
            written += 1
        except Exception as e:
            print(f"[gen_mpc_qp] FAIL {npz.name}: {e}", file=sys.stderr)
            skipped += 1
    print(f"[gen_mpc_qp] done: wrote={written} skipped={skipped} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
