"""超大規模 (1M 級) OSQP-style QP を生成する。

QPLIB suite に既に 1M 級が複数あるが (QPLIB_8500/8547/9008)、
OSQP-style の問題構造で 1M 級が無いため、IPPMM チューニング時の
汎用性担保 (大規模で挙動が変わらないか) のために追加する。

class 別の現実的な 1M 級設定:
- portfolio: k=5000 (factor model、n=k×100=500k 程度、A はブロック sparse)
- lasso: n_features=10000 (m=100×n=1M)
- control: state_size=2000 (n≈30k、m≈50k、sparse)
- random_qp / eq_qp は dense P で生成自体が NG (n=1000 で nnz=1M)

データは data/osqp_bench_xl/ に出力。
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

# 1M 級 size grid。dense P 系 (random_qp / eq_qp) は除外。
SIZE_GRID: dict[str, list[int]] = {
    "portfolio": [2500, 5000, 10000],   # factor 数 k、assets ≈ 100k
    "lasso":     [2500, 5000, 10000],   # features n、samples = 100×n
    "huber":     [2500, 5000],
    "svm":       [2500, 5000],
    "control":   [800, 1500, 2500],     # state size
}

# 1M 級は seed 1 つだけで十分 (生成・bench とも数十分級なので)
SEEDS = [2]


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
    ap.add_argument("--max-nnz", type=int, default=200_000_000,  # 200M 上限 (生成可能内)
                    help="P+A 合計 nnz 上限")
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
