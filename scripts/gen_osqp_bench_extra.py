"""OSQP benchmark suite から **追加** QPS を生成する (gen_osqp_bench.py の拡張)。

既存 data/osqp_bench/ (62 問題、SEED=1 単一) に対し、
- 複数 seed (2..6) で多様化
- 小〜大規模 (size grid 拡張)
を増やして data/osqp_bench_extra/ に出力する。

目的: IPPMM チューニングが既存 4 suite に overfit しないよう、別 seed・別 size の
新規問題で汎用性を担保する。
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

# 拡張 size grid: 小 (10 級) 〜 超大規模 (sparse 系で 1k-2k 級) まで。
# 既存 gen_osqp_bench.py が中規模中心 (max 500-1000) なので、両端を補強する。
#
# 注: random_qp / eq_qp は **dense P** (n×n) を生成する。size=3000 で 9M nnz、
# size=7000 で 49M nnz になり生成自体が分単位かかる。これらは size を 2000 で打ち止め。
# sparse 系 (lasso / huber / svm / portfolio / control) は m が n の 100x 程度の
# 細長い A を生成 (sparse) なので大規模化しても生成・解とも実用的。
SIZE_GRID: dict[str, list[int]] = {
    "random_qp": [10, 30, 70, 150, 300, 700, 1500],          # dense P, max 1500
    "eq_qp":     [10, 30, 70, 150, 300, 700, 1500],          # dense P, max 1500
    "portfolio": [3, 8, 15, 30, 70, 150, 300, 600, 1200],   # sparse, factor model
    "lasso":     [5, 15, 30, 70, 150, 300, 600, 1200],
    "huber":     [5, 15, 30, 70, 150, 300, 600, 1200],
    "svm":       [5, 15, 30, 70, 150, 300, 600, 1200],
    "control":   [5, 15, 30, 70, 100, 200, 400],            # MPC-like, sparse
}

# 別 seed (固定 1 だけだった既存と区別、複数 seed で類似分布の独立サンプル)。
SEEDS = [2, 3, 4, 5, 6]
LARGE_SEEDS = [2, 3]  # size > LARGE_SIZE_THRESHOLD では seed 数を絞る
LARGE_SIZE_THRESHOLD = 600


def import_classes() -> dict:
    from problem_classes.random_qp import RandomQPExample
    from problem_classes.eq_qp import EqQPExample
    from problem_classes.portfolio import PortfolioExample
    from problem_classes.lasso import LassoExample
    from problem_classes.huber import HuberExample
    from problem_classes.svm import SVMExample
    from problem_classes.control import ControlExample
    return {
        "random_qp": RandomQPExample,
        "eq_qp":     EqQPExample,
        "portfolio": PortfolioExample,
        "lasso":     LassoExample,
        "huber":     HuberExample,
        "svm":       SVMExample,
        "control":   ControlExample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "osqp_bench_extra"),
                    help="出力先")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="対象 class (省略時は全部)")
    ap.add_argument("--max-nnz", type=int, default=20_000_000,
                    help="P+A 合計 nnz 上限 (超大規模含む)")
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
            print(f"[extra] unknown class: {name}", file=sys.stderr)
            continue
        cls = classes[name]
        for size in SIZE_GRID[name]:
            # 大規模では seed 数を絞ってトータル時間を抑える。
            seeds = LARGE_SEEDS if size > LARGE_SIZE_THRESHOLD else args.seeds
            for seed in seeds:
                try:
                    inst = cls(size, seed=seed)
                except Exception as e:
                    print(f"[extra] skip {name}_{size}_s{seed}: gen failed ({e})", file=sys.stderr)
                    skipped += 1
                    continue
                qp = inst.qp_problem
                nnz = qp["P"].nnz + qp["A"].nnz
                if nnz > args.max_nnz:
                    print(f"[extra] skip {name}_{size}_s{seed}: nnz={nnz} > max_nnz")
                    skipped += 1
                    continue
                stem = f"X_{name}_{size}_s{seed}".upper()
                out = out_dir / f"{stem}.qps"
                try:
                    # write_osqp_problem の name 引数は QPS NAME 行用なので 8 文字制限あり
                    name_in_qps = f"X{name[:3]}{size:04d}{seed}".upper()[:8]
                    write_osqp_problem(name_in_qps, qp, out)
                except Exception as e:
                    print(f"[extra] skip {name}_{size}_s{seed}: write failed ({e})", file=sys.stderr)
                    skipped += 1
                    continue
                print(f"[extra] wrote {out.name}  n={qp['n']} m={qp['m']} nnz(P+A)={nnz}")
                written += 1

    print(f"[extra] done: wrote={written} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
