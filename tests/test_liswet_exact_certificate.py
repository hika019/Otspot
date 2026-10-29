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


def write_vector_section_qps(path: Path, rhs_lines: list[str]) -> None:
    """n=3, m=1 の LISWET 形 QPS を書く。RHS セクションの中身を rhs_lines に任す。

    行 R1 は列 1, 2, 3 に 1, -2, 1 (二次差分)。c = (1, -2, 1)、Q = I、
    全変数 FR で、x = 0・y = 1 は厳密 KKT 点 (手計算: x + c = A'y で
    (1, -2, 1) = 1·(1, -2, 1)、A x = 0 で R1 有効、y >= 0) なので
    c0 = 0 のとき大域最適値は 0。
    """
    lines = [
        "NAME SYNTH",
        "ROWS",
        " N OBJ",
        " G R1",
        "COLUMNS",
        "    X1  OBJ 1  R1 1",
        "    X2  OBJ -2  R1 -2",
        "    X3  OBJ 1  R1 1",
        "RHS",
        *rhs_lines,
        "BOUNDS",
        " FR BOUNDS X1",
        " FR BOUNDS X2",
        " FR BOUNDS X3",
        "QUADOBJ",
        "    X1 X1 1",
        "    X2 X2 1",
        "    X3 X3 1",
        "ENDATA",
    ]
    path.write_text("\n".join(lines) + "\n")


class ExactCertificateTests(unittest.TestCase):
    def test_binary64_all_active_optimum_matches_affine_projection(self):
        # Orthogonal projection onto span(1,j), computed independently of KKT recovery.
        c = [Fraction(0), Fraction(-11, 10), Fraction(-6, 5),
             Fraction(-13, 10), Fraction(-32, 5), Fraction(5, 2)]
        binary_c = [Fraction(float(v)) for v in c]
        s0 = -sum(binary_c)
        s1 = -sum(j * v for j, v in enumerate(binary_c))
        intercept = (55 * s0 - 15 * s1) / 105
        slope = (6 * s1 - 15 * s0) / 105
        optimum = Fraction(7, 8) - sum((intercept + j * slope)**2 for j in range(6)) / 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary64_active.qps"
            write_qps(path, c, Fraction(7, 8), ROWS)
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.objective, optimum)
        self.assertEqual(cert.enclosure.lower, optimum)
        self.assertEqual(cert.enclosure.upper, optimum)
        self.assertNotEqual(optimum, Fraction(-39, 10))
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

    def test_banded_kkt_handles_more_gaps_than_legacy_gram_limit(self):
        # 旧 Gram 方式の上限 256 を超える 292 個の無効行を持つ KKT 点。
        # 有効行では二次差分 0、無効行では二次差分 1 になる x を作り、
        # 有効行だけ 1 の y から c = A'y - x を定義する。
        n = 300
        rows = tuple(
            {i: Fraction(1), i + 1: Fraction(-2), i + 2: Fraction(1)}
            for i in range(n - 2)
        )
        m = len(rows)
        active = tuple(range(0, m, 50))
        active_set = set(active)
        gaps = tuple(i for i in range(m) if i not in active_set)
        d = [0 if i in active_set else 1 for i in range(m)]
        x = [Fraction(0)] * n
        for i in range(m):
            x[i + 2] = d[i] + 2 * x[i + 1] - x[i]
        y = [Fraction(1) if i in active_set else Fraction(0) for i in range(m)]
        aty = [Fraction(0)] * n
        for i, row in enumerate(rows):
            for j, value in row.items():
                aty[j] += value * y[i]
        c = [aty[j] - x[j] for j in range(n)]
        c0 = Fraction(5, 2)
        expected, ax = objective_at(c, c0, rows, x)
        self.assertEqual([ax[i] for i in active], [0] * len(active))
        self.assertEqual([ax[i] for i in gaps], [1] * len(gaps))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "many_gaps.qps"
            write_qps(path, c, c0, rows)
            cert = exact.certify(path, gaps=gaps)
        self.assertGreater(len(cert.gaps), 256)
        self.assertEqual(cert.gaps, gaps)
        self.assertEqual(cert.objective, expected)
        self.assertEqual(cert.y_min, 1)
        self.assertEqual(cert.inactive_min, 1)

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


class VectorSectionTests(unittest.TestCase):
    """複数 named RHS ベクタの first-wins (production parser 同規則; Codex P1)."""

    def test_rhs_multiple_named_vectors_first_wins(self):
        # Codex P1 (4366021328) の例。FIRST R1 0 が開き、SECOND OBJ -100 を
        # 棄てる。旧実装は disjoint 行を merge して c0=100 を誤認証する;
        # production parser (otspot-io) は first-wins で c0=0、x=0 が大域最適。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first_wins.qps"
            write_vector_section_qps(path, ["    FIRST R1 0", "    SECOND OBJ -100"])
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.objective, Fraction(0))

    def test_named_vector_after_unnamed_entries_is_rejected(self):
        # 名前の無い行の後で名前付きベクタが現れると 2 通りの解釈で値が異なる
        # ので曖昧として退ける (production 同規則)。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambiguous.qps"
            write_vector_section_qps(path, ["    R1 0", "    SECOND OBJ -100"])
            with self.assertRaises(exact.UnsupportedProblem):
                exact.parse_exact_qps(path)

    def test_duplicate_entry_in_first_vector_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup_first.qps"
            write_vector_section_qps(path, ["    FIRST R1 0", "    FIRST R1 5"])
            with self.assertRaises(exact.UnsupportedProblem):
                exact.parse_exact_qps(path)

    def test_duplicate_entry_in_discarded_later_vector_is_rejected(self):
        # 棄てるベクタ内の重複も検査して不正入力の黙認を防ぐ (production 同規則)。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup_discarded.qps"
            write_vector_section_qps(
                path, ["    FIRST R1 0", "    SECOND R1 5", "    SECOND R1 6"])
            with self.assertRaises(exact.UnsupportedProblem):
                exact.parse_exact_qps(path)

    def test_same_row_in_second_vector_is_discarded(self):
        # 同一行が 2 ベクタに現れるときは後続ベクタを棄てる (production 同規則;
        # 旧実装は "duplicate RHS" を退けるので first-wins で受理できる QPS を
        # 証明器側で拒絶する食い違いも直す)。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "discard.qps"
            write_vector_section_qps(path, ["    FIRST R1 0", "    SECOND R1 5"])
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.objective, Fraction(0))

    def test_rhs_vector_name_does_not_shadow_problem_name(self):
        # RHS 行のベクタ名は専変数で受ける; name は NAME セクション以来の問題名
        # (ExactQp.name → Certificate.problem → CSV 行の problem 列)。旧実装は
        # 匿名行で None、named 行で最後のベクタ名に上書きし、基準行の問題列が
        # 破れる (Codex P2 4370997709)。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anon.qps"
            write_vector_section_qps(path, ["    R1 0"])
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.problem, "SYNTH")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "named.qps"
            write_vector_section_qps(path, ["    FIRST R1 0", "    SECOND OBJ -100"])
            cert = exact.certify(path, gaps=())
        self.assertEqual(cert.problem, "SYNTH")


if __name__ == "__main__":
    unittest.main()
