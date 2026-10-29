#!/usr/bin/env python3
"""LISWET QP の厳密有理 KKT 証明。

共有 QPS パーサの binary64 モデルを有理数として厳密に読み、LISWET の構造

    min  c0 + c'x + 1/2 x'x
    s.t. A x >= 0,  x free

(A の行 i は x_i - 2 x_{i+1} + x_{i+2}) に対する KKT 点を有理数で構成する。
Q = I なので KKT は

    x + c = A_S' y,  A_S x = 0,  y >= 0,  A_{S^c} x >= 0

であり (S は有効制約)、これらを厳密に満たす点は凸 QP の大域最適解である。
浮動小数の KKT 残差ではなく、有理数の等式・不等式だけで判定する。

有効行 S を固定すると、x を消去した y の方程式は 5 点差分

    y_{i-2} - 4 y_{i-1} + 6 y_i - 4 y_{i+1} + y_{i+2}
        = c_i - 2 c_{i+1} + c_{i+2}       (i in S)

になる。無効行の y は 0 なので、有効行だけを未知数にすれば係数行列は
帯幅 2 の帯行列である。部分ピボット付き Gauss 消去で厳密に解き、x を
stationarity から復元する。密な Gram 行列を使う旧方式は無効行数 g に対して
O(g^3) の有理数演算を要したが、この方式は O(|S|) 回の帯内消去で済む。

使い方:
  python3 scripts/liswet_exact_certificate.py data/maros_meszaros/LISWET7.QPS
  python3 scripts/liswet_exact_certificate.py --gaps 3179,3180,9786,9792 LISWET9.QPS
  python3 scripts/liswet_exact_certificate.py --csv data/maros_meszaros/LISWET7.QPS

--gaps を省略すると HiGHS の浮動小数解から無効行を提案し、厳密検証に通るまで
active set を 1 行ずつ修正する。証明の成否は最後の厳密検証だけで決まり、
HiGHS の解は候補生成にしか使わない。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import highs_baseline_truth as truth
from baseline_certificate import ExactModel, reference_error, REFERENCE_TOLERANCE, save_evidence

# 有効行の候補判定 (境界 0 からの絶対偏差)。LISWET の数値解では有効行の
# A x は 1e-15 程度、無効行は 1e-7 以上離れる。誤判定は厳密検証と
# active set 修正で回復する。
ACTIVE_TOL = 1e-7
# 有効行の KKT 方程式に現れる 5 点差分の係数。オフセットは行 i からの差。
_KKT_STENCIL = ((-2, 1), (-1, -4), (0, 6), (1, -4), (2, 1))
# 5 点差分の片側帯幅。部分ピボットはこの範囲内だけで十分。
_KKT_BANDWIDTH = 2
# active set 修正の反復上限。候補が循環したら打ち切る。
MAX_REFINE_ITERS = 64
# baseline CSV と同じ 13 有効桁 (小数点以下 12 桁)。
CSV_MANTISSA_DECIMALS = 12
# HiGHS の報告値と厳密値の相対差の上限 (bench eps=1e-6)。これを超える差は
# QPS 解釈 (c0 の符号・Q の 1/2 規約など) の食い違いを示す。
MODEL_CONSISTENCY_TOL = 1e-6


class UnsupportedProblem(ValueError):
    """この証明器が対象としない QPS。"""


class CertificateError(RuntimeError):
    """厳密 KKT 検証に失敗した。"""


@dataclass(frozen=True)
class ExactQp:
    name: str
    n: int
    m: int
    c0: Fraction
    c: tuple[Fraction, ...]
    rows: tuple[tuple[tuple[int, Fraction], ...], ...]
    model: ExactModel


@dataclass(frozen=True)
class Certificate:
    problem: str
    objective: Fraction
    c0: Fraction
    gaps: tuple[int, ...]
    y_min: Fraction
    inactive_min: Fraction | None
    reported: float | None
    n: int
    m: int
    enclosure: object

    @property
    def objective_float(self) -> float:
        return float(self.objective)

    @property
    def relative_reported_gap(self) -> float | None:
        if self.reported is None:
            return None
        return abs(self.objective_float - self.reported) / (1.0 + abs(self.reported))


def parse_exact_qps(path: str | Path) -> ExactQp:
    """Read the shared binary64 model and require the LISWET structure."""
    path = Path(path)
    try:
        prob = truth.load_qps(path)
        model = ExactModel(prob)
    except (ValueError, IndexError, KeyError) as exc:
        raise UnsupportedProblem(str(exc)) from exc
    n, m = model.n, model.m
    if not n or not m or m != n - 2:
        raise UnsupportedProblem(f"expected m = n - 2, got n={n}, m={m}")
    if any(v is not None for v in model.lo + model.up):
        raise UnsupportedProblem("all variables must be FR")
    if model.Q != tuple(((j, Fraction(1)),) for j in range(n)):
        raise UnsupportedProblem("Q must be the identity")
    if any(v != 0 for v in model.L) or any(v is not None for v in model.U):
        raise UnsupportedProblem("rows must have lower bound zero and no upper bound")
    for i, row in enumerate(model.Arow):
        expected = ((i, Fraction(1)), (i + 1, Fraction(-2)), (i + 2, Fraction(1)))
        if row != expected:
            raise UnsupportedProblem(f"row {i} is not the LISWET second difference")
    name = path.stem
    with path.open() as handle:
        for line in handle:
            if line.startswith("NAME"):
                tokens = line.split()
                if len(tokens) > 1:
                    name = tokens[1]
                break
    return ExactQp(name, n, m, model.c0, model.c, model.Arow, model)


def _solve_banded_system(
    rows: list[dict[int, Fraction]], rhs: list[Fraction]
) -> list[Fraction]:
    """帯幅 2 の有理数連立一次方程式を部分ピボット付き Gauss 消去で解く。"""
    size = len(rhs)
    a = [dict(row) for row in rows]
    b = list(rhs)
    for col in range(size):
        # 下側帯幅は消去と帯内ピボット交換で 2 のままなので、候補は col..col+2。
        pivot = next(
            (
                r
                for r in range(col, min(size, col + _KKT_BANDWIDTH + 1))
                if a[r].get(col, Fraction(0)) != 0
            ),
            None,
        )
        if pivot is None:
            raise CertificateError(f"singular exact banded KKT system at column {col}")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        pivot_value = a[col][col]
        for r in range(col + 1, min(size, col + _KKT_BANDWIDTH + 1)):
            factor = a[r].get(col, Fraction(0)) / pivot_value
            if factor == 0:
                continue
            for k, value in a[col].items():
                updated = a[r].get(k, Fraction(0)) - factor * value
                if updated == 0:
                    a[r].pop(k, None)
                else:
                    a[r][k] = updated
            b[r] -= factor * b[col]
    result = [Fraction(0)] * size
    for r in range(size - 1, -1, -1):
        value = b[r]
        for k, coefficient in a[r].items():
            if k != r:
                value -= coefficient * result[k]
        result[r] = value / a[r][r]
    return result


def _solve_kkt_banded(
    qp: ExactQp, gaps: tuple[int, ...]
) -> tuple[list[Fraction], list[Fraction], set[int]]:
    """指定した無効行集合に対する KKT 点を帯行列の厳密解から構成する。"""
    gaps = tuple(sorted(set(gaps)))
    if any(g < 0 or g >= qp.m for g in gaps):
        raise CertificateError(f"gap index out of range: {gaps!r}")
    active = sorted(set(range(qp.m)) - set(gaps))
    position = {row: index for index, row in enumerate(active)}

    rows: list[dict[int, Fraction]] = []
    rhs: list[Fraction] = []
    for i in active:
        coefficients: dict[int, Fraction] = {}
        for offset, value in _KKT_STENCIL:
            row = i + offset
            if 0 <= row < qp.m and row in position:
                coefficients[position[row]] = Fraction(value)
        rows.append(coefficients)
        rhs.append(qp.c[i] - 2 * qp.c[i + 1] + qp.c[i + 2])

    y_active = _solve_banded_system(rows, rhs)
    y = [Fraction(0)] * qp.m
    for index, row in enumerate(active):
        y[row] = y_active[index]

    # x + c = A'y を列ごとに評価する。
    x = [Fraction(0)] * qp.n
    for j in range(qp.n):
        value = -qp.c[j]
        if j < qp.m:
            value += y[j]
        if 0 <= j - 1 < qp.m:
            value -= 2 * y[j - 1]
        if 0 <= j - 2 < qp.m:
            value += y[j - 2]
        x[j] = value

    for i in active:
        if x[i] - 2 * x[i + 1] + x[i + 2] != 0:
            raise CertificateError(f"active row {i} is not exactly zero")
    return x, y, set(active)


def _certificate_from_point(
    qp: ExactQp,
    x: list[Fraction],
    y: list[Fraction],
    active: set[int],
    gaps: tuple[int, ...],
    reported: float | None,
) -> Certificate:
    # 主実行可能性 (A x >= 0) と相補性 (有効行は A x = 0)。
    ax = [x[i] - 2 * x[i + 1] + x[i + 2] for i in range(qp.m)]
    for i in active:
        if ax[i] != 0:
            raise CertificateError(f"active row {i} has A x = {ax[i]}")
    inactive = [i for i in range(qp.m) if i not in active]
    for i in inactive:
        if ax[i] < 0:
            raise CertificateError(f"inactive row {i} violates A x >= 0: {ax[i]}")
    # 双対実行可能性。
    for i in active:
        if y[i] < 0:
            raise CertificateError(f"active row {i} has negative multiplier {y[i]}")
    for i in inactive:
        if y[i] != 0:
            raise CertificateError(f"inactive row {i} has nonzero multiplier {y[i]}")

    # 定常性 (x + c - A'y = 0) を全列で厳密に確認。
    aty = [Fraction(0)] * qp.n
    for i, row in enumerate(qp.rows):
        if y[i] == 0:
            continue
        for j, value in row:
            aty[j] += value * y[i]
    for j in range(qp.n):
        if x[j] + qp.c[j] - aty[j] != 0:
            raise CertificateError(f"stationarity residual at column {j}")

    objective = qp.c0 + sum(qp.c[j] * x[j] for j in range(qp.n))
    objective += Fraction(1, 2) * sum(x[j] * x[j] for j in range(qp.n))
    dual = qp.c0 - Fraction(1, 2) * sum(x[j] * x[j] for j in range(qp.n))
    if objective != dual:
        raise CertificateError("primal and dual objective values differ")
    complementarity = sum(y[i] * ax[i] for i in range(qp.m))
    if complementarity != 0:
        raise CertificateError("complementarity residual is nonzero")

    enclosure = qp.model.check(x, x, y)
    if enclosure is None or enclosure.lower != enclosure.upper:
        raise CertificateError("shared model checker rejected the KKT witness")
    if reference_error(enclosure, format_rational_csv(objective)) > REFERENCE_TOLERANCE:
        raise CertificateError("serialized reference exceeds its error budget")
    inactive_min = min((ax[i] for i in inactive), default=None)
    return Certificate(
        problem=qp.name,
        objective=objective,
        c0=qp.c0,
        gaps=gaps,
        y_min=min((y[i] for i in active), default=Fraction(0)),
        inactive_min=inactive_min,
        reported=reported,
        n=qp.n,
        m=qp.m,
        enclosure=enclosure,
    )


def propose_gaps(path: str | Path, time_limit: float) -> tuple[tuple[int, ...], float]:
    """HiGHS の浮動小数解から無効行の候補を作る (証明には使わない)。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import highs_baseline_truth as truth
    except ImportError as exc:
        raise CertificateError(
            "HiGHS proposal unavailable (install numpy/scipy/highspy or pass --gaps)"
        ) from exc

    prob = truth.load_qps(str(path))
    status, x, _y, reported = truth.solve_highs(prob, time_limit)
    if status != "Optimal":
        raise CertificateError(f"HiGHS did not return Optimal: {status}")
    ax = prob["A"] @ x
    gaps = tuple(i for i in range(prob["m"]) if float(ax[i]) > ACTIVE_TOL)
    return gaps, float(reported)


def certify(
    path: str | Path,
    gaps: tuple[int, ...] | None = None,
    time_limit: float = 300.0,
    refine: bool = True,
) -> Certificate:
    """QPS を厳密に証明する。gaps 省略時は HiGHS から候補を作る。"""
    qp = parse_exact_qps(path)
    reported = None
    if gaps is None:
        gaps, reported = propose_gaps(path, time_limit)
    current = tuple(sorted(set(gaps)))
    seen: set[tuple[int, ...]] = set()
    for _ in range(MAX_REFINE_ITERS):
        if current in seen:
            raise CertificateError(f"active set refinement cycled at {len(current)} gaps")
        seen.add(current)
        x, y, active = _solve_kkt_banded(qp, current)
        ax = [x[i] - 2 * x[i + 1] + x[i + 2] for i in range(qp.m)]
        negative = [i for i in active if y[i] < 0]
        violated = [i for i in current if ax[i] < 0]
        if not negative and not violated:
            result = _certificate_from_point(qp, x, y, active, current, reported)
            relative = result.relative_reported_gap
            if relative is not None and relative > MODEL_CONSISTENCY_TOL:
                raise CertificateError(
                    f"exact objective differs from HiGHS report by {relative:.3e} "
                    f"(> {MODEL_CONSISTENCY_TOL:g}); check Q/c0 interpretation"
                )
            return result
        if not refine:
            detail = (f"negative multipliers at {negative[:4]}"
                      if negative else f"inactive violations at {violated[:4]}")
            raise CertificateError(f"not a KKT point: {detail}")
        if negative:
            worst = min(negative, key=lambda i: y[i])
            current = tuple(sorted(current + (worst,)))
        else:
            worst = min(violated, key=lambda i: ax[i])
            current = tuple(i for i in current if i != worst)
    raise CertificateError(f"active set refinement did not converge in {MAX_REFINE_ITERS} iterations")


def format_rational_csv(value: Fraction) -> str:
    """13 有効桁の baseline CSV 表記 (例 4.988408907861e2)。"""
    with localcontext() as ctx:
        ctx.prec = 50
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        mantissa, exponent = f"{decimal:.{CSV_MANTISSA_DECIMALS}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _json_payload(cert: Certificate) -> dict:
    inactive = None if cert.inactive_min is None else str(cert.inactive_min)
    return {
        "problem": cert.problem,
        "n": cert.n,
        "m": cert.m,
        "objective": _rational_text(cert.objective),
        "objective_decimal": repr(cert.objective_float),
        "csv": format_rational_csv(cert.objective),
        "c0": _rational_text(cert.c0),
        "gaps": list(cert.gaps),
        "active_rows": cert.m - len(cert.gaps),
        "y_min": str(cert.y_min),
        "inactive_min": inactive,
        "reported": cert.reported,
        "relative_reported_gap": cert.relative_reported_gap,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="LISWET QPS files")
    parser.add_argument("--gaps", help="comma separated inactive row indices")
    parser.add_argument("--time-limit", type=float, default=300.0, help="HiGHS time limit [s]")
    parser.add_argument("--csv", action="store_true", help="print baseline CSV rows")
    parser.add_argument("--json", action="store_true", help="print one JSON object per problem")
    parser.add_argument("--evidence-dir", type=Path, default=Path("baseline-evidence"))
    args = parser.parse_args(argv)

    gaps = None
    if args.gaps is not None:
        if len(args.paths) != 1:
            parser.error("--gaps accepts exactly one problem")
        gaps = tuple(int(token) for token in args.gaps.split(",") if token.strip())

    status = 0
    for path in args.paths:
        try:
            input_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            cert = certify(path, gaps=gaps, time_limit=args.time_limit)
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != input_hash:
                raise CertificateError("input changed during verification")
            save_evidence(args.evidence_dir / (Path(path).stem + ".proof.json.gz"),
                          cert.enclosure, format_rational_csv(cert.objective), input_hash,
                          dict(method="liswet_banded", gaps=list(cert.gaps)))
        except (UnsupportedProblem, CertificateError, OSError) as exc:
            print(f"{Path(path).stem}: NOT CERTIFIED: {exc}", file=sys.stderr)
            status = 1
            continue
        if args.json:
            print(json.dumps(_json_payload(cert), sort_keys=True))
            continue
        if args.csv:
            date = datetime.date.today().strftime("%Y%m%d")
            print(f"{cert.problem},{format_rational_csv(cert.objective)},exact_rational_kkt_{date}")
            continue
        inactive = "none" if cert.inactive_min is None else f"{float(cert.inactive_min):.3e}"
        reported = "n/a" if cert.reported is None else f"{cert.relative_reported_gap:.2e}"
        print(
            f"{cert.problem}: CERTIFIED objective={float(cert.objective):.16e} "
            f"csv={format_rational_csv(cert.objective)} gaps={len(cert.gaps)} "
            f"y_min={float(cert.y_min):.3e} inactive_min={inactive} "
            f"reported_rel={reported}"
        )
        print(f"  rational={_rational_text(cert.objective)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
