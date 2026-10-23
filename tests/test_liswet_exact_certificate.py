"""scripts/liswet_exact_certificate.py の sentinel (合成 QPS、外部データ不要)。"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "liswet_exact_certificate", ROOT / "scripts" / "liswet_exact_certificate.py"
)
exact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exact
spec.loader.exec_module(exact)

# LISWET と同じ n=6, m=4 の二次差分。行 i は列 i, i+1, i+2 に 1, -2, 1。
ROWS = tuple(
    {i: Fraction(1), i + 1: Fraction(-2), i + 2: Fraction(1)} for i in range(4)
)


def _decimal(value: Fraction) -> str:
    """QPS の 10 進リテラルとして書く (テスト値は全て有限小数)。"""
    with localcontext() as ctx:
        ctx.prec = 50
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def write_qps(path: Path, c: list[Fraction], c0: Fraction, rows) -> None:
    lines = ["NAME SYNTH", "ROWS", " N OBJ"]
    lines += [f" G R{i + 1}" for i in range(len(rows))]
    lines.append("COLUMNS")
    for j in range(len(c)):
        pairs = ["OBJ", _decimal(c[j])]
        for i, row in enumerate(rows):
            if j in row:
                pairs += [f"R{i + 1}", _decimal(row[j])]
        lines.append(f"    X{j + 1}  " + "  ".join(pairs))
    if c0:
        lines += ["RHS", f"    RHS OBJ {_decimal(-c0)}"]
    else:
        lines.append("RHS")
    lines += ["BOUNDS"] + [f" FR BOUNDS X{j + 1}" for j in range(len(c))]
    lines += ["QUADOBJ"] + [f"    X{j + 1} X{j + 1} 1" for j in range(len(c))]
    lines.append("ENDATA")
    path.write_text("\n".join(lines) + "\n")


def objective_at(
    c: list[Fraction], c0: Fraction, rows, x: list[Fraction]
) -> tuple[Fraction, list[Fraction]]:
    value = c0 + sum(c[j] * x[j] for j in range(len(x)))
    value += Fraction(1, 2) * sum(v * v for v in x)
    ax = [x[i] - 2 * x[i + 1] + x[i + 2] for i in range(len(rows))]
    return value, ax


class ExactCertificateTests(unittest.TestCase):
    def test_all_active_rows_match_hand_computed_kkt_point(self):
        # x はアフィン (全行有効)、y > 0。c = A'y - x を解析的に作り、目的値を
        # 独立に手計算した値と比較する。
        x = [Fraction(1), Fraction(11, 10), Fraction(6, 5),
             Fraction(13, 10), Fraction(7, 5), Fraction(3, 2)]
        y = [Fraction(1), Fraction(2), Fraction(3), Fraction(4)]
        aty = [
            y[0],
            -2 * y[0] + y[1],
            y[0] - 2 * y[1] + y[2],
            y[1] - 2 * y[2] + y[3],
            y[2] - 2 * y[3],
            y[3],
        ]
        c = [aty[j] - x[j] for j in range(6)]
        c0 = Fraction(7, 8)
        expected, ax = objective_at(c, c0, ROWS, x)
        self.assertEqual(ax, [0] * 4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all_active.qps"
            write_qps(path, c, c0, ROWS)
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.gaps, ())
        self.assertEqual(cert.objective, expected)
        self.assertEqual(cert.objective, Fraction(-39, 10))
        self.assertGreater(cert.y_min, 0)

    def test_inactive_row_matches_hand_computed_kkt_point(self):
        # 行 2 だけ無効。x = (1,1,1,1,2,3) は行 0,1,3 で二次差分ゼロ、
        # 行 2 で 1 > 0。y_2 = 0。
        x = [Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(2), Fraction(3)]
        aty = [Fraction(1), Fraction(0), Fraction(-3), Fraction(3), Fraction(-2), Fraction(1)]
        c = [aty[j] - x[j] for j in range(6)]
        c0 = Fraction(1, 2)
        expected, ax = objective_at(c, c0, ROWS, x)
        self.assertEqual(ax, [0, 0, 1, 0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one_gap.qps"
            write_qps(path, c, c0, ROWS)
            cert = exact.certify(path, gaps=(2,))
        self.assertEqual(cert.gaps, (2,))
        self.assertEqual(cert.objective, expected)
        self.assertEqual(cert.objective, Fraction(-8))
        self.assertEqual(cert.inactive_min, 1)

    def test_wrong_active_set_is_rejected_without_refinement(self):
        x = [Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(2), Fraction(3)]
        aty = [Fraction(1), Fraction(0), Fraction(-3), Fraction(3), Fraction(-2), Fraction(1)]
        c = [aty[j] - x[j] for j in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong_gap.qps"
            write_qps(path, c, Fraction(1, 2), ROWS)
            with self.assertRaises(exact.CertificateError):
                exact.certify(path, gaps=(), refine=False)

    def test_refinement_recovers_missing_gap(self):
        x = [Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(2), Fraction(3)]
        aty = [Fraction(1), Fraction(0), Fraction(-3), Fraction(3), Fraction(-2), Fraction(1)]
        c = [aty[j] - x[j] for j in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refine.qps"
            write_qps(path, c, Fraction(1, 2), ROWS)
            cert = exact.certify(path, gaps=(), refine=True)
        self.assertEqual(cert.gaps, (2,))
        self.assertEqual(cert.objective, Fraction(-8))

    def test_non_liswet_row_is_rejected(self):
        c = [Fraction(0)] * 6
        rows = list(ROWS)
        rows[1] = {1: Fraction(1), 2: Fraction(-3), 3: Fraction(1)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.qps"
            write_qps(path, c, Fraction(0), rows)
            with self.assertRaises(exact.UnsupportedProblem):
                exact.certify(path, gaps=())

    def test_csv_format_keeps_thirteen_significant_digits(self):
        self.assertEqual(exact.format_rational_csv(Fraction(1, 3)), "3.333333333333e-1")
        self.assertEqual(
            exact.format_rational_csv(Fraction(4988408907861147, 10**13)),
            "4.988408907861e2",
        )


if __name__ == "__main__":
    unittest.main()
