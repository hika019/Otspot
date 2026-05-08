"""ill-scaled OSQP-style QP を生成する (検証空白を埋めるための合成 ill-scaling)。

動機: data/osqp_bench_extra/ (238 問) は値が O(1) 正規化された合成 QP で、
Ruiz scaling が σ_total ≈ 0.1〜0.7 で済む well-scaled サンプルしかカバーしない。
一方 Maros 138 の fail 問題は LP→QP 変換で σ_total = 1e-5 級の極端 ill-scaling。
合成データではその病理を再現しないため、solver の数値ロバスト性を Maros 1 種に
依存して評価せざるを得ない (検証空白)。

CLAUDE.md「fail を範疇外で分離するな。検証空白を埋めるテスト追加 or 真因対処」
への直接対応として、合成 ill-scaled QP を生成する。

スケール強度の制御:
  各列 j に対角スケール D[j] = 10^u[j] を適用 (u[j] ~ Uniform(0, log10_max))。
    Q' = D Q D, A' = A D, c' = D c, bounds' = bounds / D
  解空間は x = D^{-1} x' で等価 (substitution)。
  Ruiz が完全吸収するなら問題なしだが、現実には D の非一様性が σ_total に残る。

  log10_max = 3 → σ_total ≈ 1e-3 級
  log10_max = 6 → σ_total ≈ 1e-6 級

出力: data/osqp_bench_illscaled/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as spa

REPO_ROOT = Path(__file__).resolve().parent.parent
OSQP_REPO = REPO_ROOT / "tmp" / "external" / "osqp_benchmarks"
sys.path.insert(0, str(OSQP_REPO))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qp_to_qps import write_osqp_problem  # noqa: E402

# ill-scaled bench は coverage gap を埋める目的なので size 中規模で十分
# (各 size × scale × seed で組み合わせ爆発するため、size は控えめに)。
SIZE_GRID: dict[str, list[int]] = {
    "portfolio": [30, 100, 300],
    "lasso":     [30, 100, 300],
    "huber":     [30, 100, 300],
    "svm":       [30, 100, 300],
    "control":   [30, 70, 150],
    "random_qp": [30, 70, 150],
    "eq_qp":     [30, 70, 150],
}

# log10(D) の上限。Maros 5 fails の σ_total を概ね覆うレンジ。
SCALE_LOG10_GRID = [3, 5]   # σ_total ≈ 1e-3 / 1e-5 級

SEEDS = [11, 22, 33]  # gen_osqp_bench_extra の SEEDS と被らない seed


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


def apply_column_scaling(qp: dict, log10_max: float, seed: int) -> dict:
    """qp に対角列スケール D = diag(10^u) を適用して ill-scaled な等価問題に変換する。

    元 OSQP-form: min 1/2 x' P x + q' x  s.t. l <= A x <= u
    変換後:       min 1/2 (Dx')' P (Dx') + q' (Dx')
                = min 1/2 x'' (DPD) x'' + (Dq)' x''
                  s.t. l <= AD x'' <= u
    つまり P' = DPD, A' = AD, q' = Dq, l/u 不変 (列スケールは行に作用しない)。

    解 x* に対し x*'' = D^{-1} x*。Ruiz が D を完全吸収すれば solver の動作は不変だが、
    現実には Ruiz の収束限界・double 精度限界で σ_total に D の非一様性が残る。
    """
    rng = np.random.default_rng(seed)
    n = qp["n"]
    # 一様分布 (0, log10_max) — half は控えめ、half は強い
    u = rng.uniform(0.0, log10_max, size=n)
    d = 10.0 ** u  # D[j] = 10^u[j]
    D = spa.diags(d)

    P = qp["P"]
    A = qp["A"]
    q = qp["q"]
    # P' = D P D, A' = A D, q' = D q
    P_new = (D @ P @ D).tocsc()
    A_new = (A @ D).tocsc()
    q_new = d * q
    # l/u は行 bound なので列スケールでは変化しない
    return {
        "n": n,
        "m": qp["m"],
        "P": P_new,
        "A": A_new,
        "q": q_new,
        "l": qp["l"].copy(),
        "u": qp["u"].copy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "osqp_bench_illscaled"))
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--max-nnz", type=int, default=20_000_000)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scales", nargs="+", type=int, default=SCALE_LOG10_GRID)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = import_classes()
    target_names = args.classes if args.classes else list(SIZE_GRID.keys())

    written = 0
    skipped = 0
    for name in target_names:
        if name not in classes:
            print(f"[illscaled] unknown class: {name}", file=sys.stderr)
            continue
        cls = classes[name]
        for size in SIZE_GRID[name]:
            for seed in args.seeds:
                try:
                    inst = cls(size, seed=seed)
                    qp_orig = inst.qp_problem
                except Exception as e:
                    print(f"[illscaled] skip {name}_{size}_s{seed}: gen failed ({e})", file=sys.stderr)
                    skipped += 1
                    continue
                for log10_max in args.scales:
                    qp_scaled = apply_column_scaling(qp_orig, float(log10_max), seed * 100 + log10_max)
                    nnz = qp_scaled["P"].nnz + qp_scaled["A"].nnz
                    if nnz > args.max_nnz:
                        skipped += 1
                        continue
                    stem = f"IS_{name}_{size}_s{seed}_e{log10_max}".upper()
                    out = out_dir / f"{stem}.qps"
                    name_in_qps = f"I{name[:2]}{size:03d}{seed:02d}{log10_max}".upper()[:8]
                    try:
                        write_osqp_problem(name_in_qps, qp_scaled, out)
                    except Exception as e:
                        print(f"[illscaled] skip {stem}: write failed ({e})", file=sys.stderr)
                        skipped += 1
                        continue
                    print(f"[illscaled] wrote {out.name}  n={qp_scaled['n']} m={qp_scaled['m']} nnz={nnz} log10D={log10_max}")
                    written += 1

    print(f"[illscaled] done: wrote={written} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
