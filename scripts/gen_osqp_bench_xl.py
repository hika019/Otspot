"""Large-scale OSQP-style QP generator.

Produces medium-to-large QP instances in QPLIB/QPS format under
data/osqp_bench_xl/.  Problem classes mirror the OSQP benchmark suite
(portfolio, lasso, huber, svm, control); dense-P classes (random_qp,
eq_qp) are excluded because they become intractable above n≈1 000.

SIZE_GRID is tuned to stay well within a 7 GB RAM budget (GH Actions
runner limit).  A pre-generation byte estimate guards against edge cases
where the grid values would still exceed available memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OSQP_REPO = REPO_ROOT / "tmp" / "external" / "osqp_benchmarks"
sys.path.insert(0, str(OSQP_REPO))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qp_to_qps import write_osqp_problem  # noqa: E402

# Size grid tuned for ≤7 GB RAM (GH Actions runner).
# portfolio/lasso: dense factor F[n,k] where n=100·k → 100·k² elements × 8 B
#   k=500  → 200 MB,  k=1000 → 800 MB  (both safe)
# huber/svm: dense feature matrix [m×n], m=100·n → 100·n² elements × 8 B
#   n=1000 → 800 MB,  n=2000 → 3.2 GB  (both safe with headroom)
# control: sparse but state size drives n/m; kept conservative
SIZE_GRID: dict[str, list[int]] = {
    "portfolio": [500, 1000],
    "lasso":     [500, 1000],
    "huber":     [1000, 2000],
    "svm":       [1000, 2000],
    "control":   [200, 400],
}

SEEDS = [2]

# Hard cap: skip instance if estimated dense bytes exceed this threshold.
MAX_RAM_BYTES: int = 2 * 1024 ** 3  # 2 GB per instance


def estimate_factor_bytes(name: str, size: int) -> int:
    """Return an upper-bound byte estimate for a single problem instance.

    Uses the dominant matrix allocation for each class:
    - portfolio: dense factor F[n, k], n=100*size, k=size
    - lasso/huber/svm: sparse Ad with density=0.15, m=100*size, n=size;
      CSC storage ≈ density * nnz * (8 float64 + 4 row_ind + 4 col_ptr) bytes
    - control: sparse; conservative 10 MB floor
    """
    if name == "portfolio":
        return 100 * size * size * 8
    if name in ("lasso", "huber", "svm"):
        # density=0.15, m=100*size, n=size → nnz = 0.15 * 100 * size²
        # CSC: 8 (float64) + 4 (row_ind) + 4 (col_ptr estimate) = 16 bytes/nnz
        return int(0.15 * 100 * size * size * 16)
    # control and unknowns: sparse; safe floor
    return 10 * 1024 * 1024  # 10 MB


def import_classes() -> dict:
    from problem_classes.portfolio import PortfolioExample
    from problem_classes.lasso import LassoExample
    from problem_classes.huber import HuberExample
    from problem_classes.svm import SVMExample
    from problem_classes.control import ControlExample
    return {
        "portfolio": PortfolioExample,
        "lasso":     LassoExample,
        "huber":     HuberExample,
        "svm":       SVMExample,
        "control":   ControlExample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "osqp_bench_xl"))
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--max-nnz", type=int, default=200_000_000,
                    help="P+A combined nnz upper bound")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = import_classes()
    target_names = args.classes if args.classes else list(SIZE_GRID.keys())

    written = 0
    skipped = 0
    for name in target_names:
        if name not in classes:
            continue
        cls = classes[name]
        for size in SIZE_GRID[name]:
            for seed in args.seeds:
                est = estimate_factor_bytes(name, size)
                if est > MAX_RAM_BYTES:
                    print(
                        f"[xl] skip {name}_{size}_s{seed}: "
                        f"estimated_bytes={est} > MAX_RAM_BYTES={MAX_RAM_BYTES}"
                    )
                    skipped += 1
                    continue

                try:
                    inst = cls(size, seed=seed)
                except Exception as e:
                    print(f"[xl] skip {name}_{size}_s{seed}: gen failed ({e})", file=sys.stderr)
                    skipped += 1
                    continue

                qp = inst.qp_problem
                nnz = qp["P"].nnz + qp["A"].nnz
                if nnz > args.max_nnz:
                    print(f"[xl] skip {name}_{size}_s{seed}: nnz={nnz} > max_nnz")
                    skipped += 1
                    continue

                stem = f"XL_{name}_{size}_s{seed}".upper()
                out = out_dir / f"{stem}.qps"
                try:
                    name_in_qps = f"XL{name[:3]}{size:05d}".upper()[:8]
                    write_osqp_problem(name_in_qps, qp, out)
                except Exception as e:
                    print(f"[xl] skip {name}_{size}_s{seed}: write failed ({e})", file=sys.stderr)
                    skipped += 1
                    continue

                print(f"[xl] wrote {out.name}  n={qp['n']} m={qp['m']} nnz(P+A)={nnz}")
                written += 1

    print(f"[xl] done: wrote={written} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
