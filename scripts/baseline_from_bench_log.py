"""bench_parallel.sh の出力 (集計結果ファイル) から baseline CSV を生成する。

集計結果の「問題別詳細」セクションから PASS / CHECKED[no_ref] / SUBOPTIMAL 行を
抽出し、 obj=<value> を読んで baseline_objectives 形式で出力する。

形式 (1行/問題):
  problem_name,optimal_obj,source

PASS_NO_REF / SUBOPTIMAL も含めるかは --include で選択可能。
TIMEOUT / FAIL / ERROR は除外。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# 問題行の例:
#   OSQP_RANDOM_QP_20        20    200    CHECKED[no_ref]   0.013 [-] obj=1.14e1 pf=0.0e0 ...
LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s+\d+\s+\d+\s+([A-Z_\[\]a-z]+)\s+[\d.]+.*?\bobj=([-+0-9.eE]+|NA)"
)

PASS_STATUSES = {"PASS", "CHECKED[no_ref]"}


def parse_log(log_path: Path, include_suboptimal: bool):
    """ログから (name, status, obj) を抽出。"""
    results = []
    seen = set()
    with log_path.open() as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            name, status, obj_str = m.group(1), m.group(2), m.group(3)
            if status not in PASS_STATUSES and not (
                include_suboptimal and status == "SUBOPTIMAL"
            ):
                continue
            if obj_str == "NA":
                continue
            try:
                obj = float(obj_str)
            except ValueError:
                continue
            # 同じ問題が複数回出る場合は最初のみ採用
            if name in seen:
                continue
            seen.add(name)
            results.append((name, obj, status))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="bench_parallel.sh の出力ファイル")
    ap.add_argument("out_csv", help="出力 baseline CSV (上書き)")
    ap.add_argument("--source", required=True,
                    help="baseline の source カラムに書く識別子 (例: bench_2026-05-06_300s)")
    ap.add_argument("--include-suboptimal", action="store_true",
                    help="SUBOPTIMAL も含める")
    ap.add_argument("--merge", action="store_true",
                    help="既存 CSV と統合 (新規分は上書き、未掲載分は保持)")
    args = ap.parse_args()

    results = parse_log(Path(args.log), args.include_suboptimal)
    print(f"[baseline] extracted {len(results)} entries from {args.log}", file=sys.stderr)

    # 既存 entries (merge モード)
    existing: dict[str, tuple[float, str]] = {}
    out_path = Path(args.out_csv)
    if args.merge and out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#") or line.startswith("problem_name"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        existing[parts[0].strip()] = (
                            float(parts[1].strip()),
                            parts[2].strip() if len(parts) >= 3 else "",
                        )
                    except ValueError:
                        pass
        print(f"[baseline] merged with {len(existing)} existing entries", file=sys.stderr)

    # 新規で上書き
    for name, obj, _ in results:
        existing[name] = (obj, args.source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(
            "# 出典: 自ソルバーの計測値（ベンチマーク実行結果）。外部参照の最適値ではない。\n"
            "# 用途: 退行検知（過去の自己ベストとの比較）。\n"
            "# Source: measured by this solver's benchmark runs. NOT externally verified optimal values.\n"
            "# Purpose: regression detection (comparison against own historical best).\n"
            "problem_name,optimal_obj,source\n"
        )
        for name in sorted(existing.keys()):
            obj, src = existing[name]
            f.write(f"{name},{obj:.6e},{src}\n")
    print(f"[baseline] wrote {len(existing)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
