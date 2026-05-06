"""OSQP benchmark suite (osqp/osqp_benchmarks の problem_classes/) から QPS を生成する。

各 problem class × 複数サイズで問題を生成し、data/osqp_bench/ に書き出す。
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

# 各 class の sizes は「中程度: 数十〜数百次元」に統一。
# サイズが大きすぎる (>1000 次元 or >100k nnz) と 1000s で解けない可能性があり、
# benchmark grid として観測ノイズが増えるため避ける。
# class 名 -> (importer, sizes)
#   importer: lambda size, seed -> instance with .qp_problem
#   sizes:    list of size parameters
SIZE_GRID: dict[str, list[int]] = {
    "random_qp": [20, 50, 100, 200, 500],
    "eq_qp":     [20, 50, 100, 200, 500],
    "portfolio": [5, 10, 20, 50, 100],          # k (assets ≈ 100k)
    "lasso":     [10, 20, 50, 100],             # n features (m=100n)
    "huber":     [10, 20, 50, 100],
    "svm":       [10, 20, 50, 100],
    "control":   [10, 20, 50],                  # state size
}

# seed は固定 (再現性)
SEED = 1


def import_classes() -> dict:
    """OSQP benchmark の generator class を import する。"""
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
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "osqp_bench"),
                    help="出力先 (.qps を平置き)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="対象クラス (省略時は全部)")
    ap.add_argument("--max-nnz", type=int, default=2_000_000,
                    help="P+A の合計 nnz 上限 (超える size はスキップ)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = import_classes()
    target_names = args.classes if args.classes else list(SIZE_GRID.keys())

    written = 0
    skipped = 0
    for name in target_names:
        if name not in classes:
            print(f"[gen_osqp_bench] unknown class: {name}", file=sys.stderr)
            continue
        cls = classes[name]
        for size in SIZE_GRID[name]:
            try:
                inst = cls(size, seed=SEED)
            except Exception as e:
                print(f"[gen_osqp_bench] skip {name}_{size}: gen failed ({e})", file=sys.stderr)
                skipped += 1
                continue
            qp = inst.qp_problem
            nnz = qp["P"].nnz + qp["A"].nnz
            if nnz > args.max_nnz:
                print(f"[gen_osqp_bench] skip {name}_{size}: nnz={nnz} > max_nnz")
                skipped += 1
                continue
            # ファイル名: OSQP_<class>_<size>.qps  (大文字でベンチ系統と揃える)
            stem = f"OSQP_{name}_{size}".upper()
            out = out_dir / f"{stem}.qps"
            try:
                write_osqp_problem(stem[:8] if len(stem) > 8 else stem, qp, out)
            except Exception as e:
                print(f"[gen_osqp_bench] skip {name}_{size}: write failed ({e})", file=sys.stderr)
                skipped += 1
                continue
            print(f"[gen_osqp_bench] wrote {out.name}  n={qp['n']} m={qp['m']} nnz(P+A)={nnz}")
            written += 1

    print(f"[gen_osqp_bench] done: wrote={written} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
