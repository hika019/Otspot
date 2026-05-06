"""自ソルバ baseline と外部リファレンス obj を比較。
不一致を report する。

使い方:
  python3 scripts/cross_check_obj.py \\
    --self  data/baseline_objectives/osqp_bench.csv \\
    --ext   data/baseline_objectives/osqp_bench_external.csv \\
    --rel-tol 1e-2

出力 (stdout):
  AGREE: <name>  self=<x>  ext=<y>  rel=<r>
  DISAGREE: <name>  self=<x>  ext=<y>  rel=<r>
  SELF_ONLY: <name>  self=<x>
  EXT_ONLY: <name>  ext=<y>  ext_status=<s>
  サマリ末尾。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def load_csv(path: Path) -> dict[str, tuple[float | None, str]]:
    """problem_name -> (obj or None, full_line_for_debug)."""
    out: dict[str, tuple[float | None, str]] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("problem_name"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            try:
                obj = float(parts[1].strip()) if parts[1].strip() != "NA" else None
            except ValueError:
                obj = None
            extra = ",".join(parts[2:]) if len(parts) >= 3 else ""
            out[name] = (obj, extra)
    return out


def rel_err(a: float, b: float) -> float:
    denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", dest="self_csv", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--rel-tol", type=float, default=1e-2)
    args = ap.parse_args()

    self_map = load_csv(Path(args.self_csv))
    ext_map = load_csv(Path(args.ext))

    all_names = sorted(set(self_map) | set(ext_map))

    n_agree = n_disagree = n_self_only = n_ext_only = n_ext_failed = 0
    disagree_lines: list[str] = []

    for name in all_names:
        s = self_map.get(name)
        e = ext_map.get(name)
        if s is not None and e is not None:
            s_obj, _ = s
            e_obj, e_extra = e
            if e_obj is None:
                n_ext_failed += 1
                print(f"EXT_FAILED:   {name}  self={s_obj:.4e}  (ext_status={e_extra})")
                continue
            if s_obj is None:
                n_self_only += 1
                print(f"SELF_NONE:    {name}  ext={e_obj:.4e}")
                continue
            r = rel_err(s_obj, e_obj)
            if r <= args.rel_tol:
                n_agree += 1
                # AGREE 行は出力しない (静かに)
            else:
                n_disagree += 1
                line = f"DISAGREE:     {name}  self={s_obj:.6e}  ext={e_obj:.6e}  rel={r:.2e}"
                print(line)
                disagree_lines.append(line)
        elif s is not None:
            n_self_only += 1
            print(f"SELF_ONLY:    {name}  self={s[0]}")
        else:
            assert e is not None
            n_ext_only += 1
            obj, extra = e
            obj_s = f"{obj:.4e}" if obj is not None else "NA"
            print(f"EXT_ONLY:     {name}  ext={obj_s}  ({extra})")

    print()
    print(f"=== summary (rel_tol={args.rel_tol:.0e}) ===")
    print(f"  AGREE:       {n_agree}")
    print(f"  DISAGREE:    {n_disagree}")
    print(f"  EXT_FAILED:  {n_ext_failed}")
    print(f"  SELF_ONLY:   {n_self_only}")
    print(f"  SELF_NONE:   (counted under EXT_ONLY-style)")
    print(f"  EXT_ONLY:    {n_ext_only}")
    print(f"  TOTAL:       {len(all_names)}")
    if disagree_lines:
        print()
        print("DISAGREE 詳細:")
        for line in disagree_lines:
            print(f"  {line}")


if __name__ == "__main__":
    main()
