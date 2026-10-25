#!/usr/bin/env python3
"""LISWET QP の厳密有理 KKT 証明。

QPS の数値リテラルを 10 進有理数として厳密に読み、LISWET の構造

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
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

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

    @property
    def objective_float(self) -> float:
        return float(self.objective)

    @property
    def relative_reported_gap(self) -> float | None:
        if self.reported is None:
            return None
        return abs(self.objective_float - self.reported) / (1.0 + abs(self.reported))


def _pairs(tokens: list[str], has_set_name: bool) -> list[tuple[str, str]]:
    start = 1 if has_set_name else 0
    if (len(tokens) - start) % 2:
        raise UnsupportedProblem(f"odd token count in data line: {tokens!r}")
    return [(tokens[k], tokens[k + 1]) for k in range(start, len(tokens), 2)]


class _VectorSection:
    """RHS の名前付きベクタ状態。最初に開いたベクタだけを採用する。

    MPS/QPS 標準は同一セクションに複数の名前付きベクタを許し、各行は慣行
    (GLPK/CPLEX、otspot-io/src/common/mod.rs の VectorSectionState と同じ) により
    最初に現れたベクタの値を採用して以降のベクタを捨てる。名前を省いた行は
    セクションで開いているベクタへ属性する。名前の無い行の後で名前付きベクタが
    現れると 2 通りの解釈で値が異なるので曖昧として退ける。同一 (ベクタ, 行) の
    重複は、後続ベクタを棄てるときでも検査して不正入力の黙認を防ぐ。production
    parser (first-wins) と独立証明器が同ファイルで異なる規則を取ると、証明器は
    production が解さない別モデルを認証してしまう (Codex P1 4366021328)。
    """

    ANONYMOUS = ""

    def __init__(self):
        self.seen = set()
        self.first = None

    def record(self, target, section, vector_name, row, value):
        if self.first == self.ANONYMOUS and vector_name is not None:
            raise UnsupportedProblem(
                f"{section}: vector {vector_name!r} is named after unnamed entries "
                f"were already read; the section is ambiguous (one vector with the "
                f"name omitted earlier, or a second vector whose entries must be "
                f"discarded)")
        if vector_name is not None:
            vector = vector_name
        elif self.first is not None:
            vector = self.first
        else:
            vector = self.ANONYMOUS
        if (vector, row) in self.seen:
            raise UnsupportedProblem(
                f"{section}: duplicate entry for row {row!r} in vector {vector!r}")
        self.seen.add((vector, row))
        if self.first is None:
            self.first = vector
        if self.first == vector:
            target[row] = value


def parse_exact_qps(path: str | Path) -> ExactQp:
    """QPS を 10 進有理数として読む。LISWET の構造だけを受け付ける。"""
    path = Path(path)
    section = None
    name = path.stem
    objective_row = None
    row_names: list[str] = []
    row_index: dict[str, int] = {}
    col_index: dict[str, int] = {}
    c: list[Fraction] = []
    rows: list[dict[int, Fraction]] = []
    rhs: dict[str, Fraction] = {}
    rhs_vectors = _VectorSection()
    free: set[str] = set()
    q_diag: set[str] = set()

    def intern_column(token: str) -> int:
        if token not in col_index:
            col_index[token] = len(c)
            c.append(Fraction(0))
        return col_index[token]

    with path.open(errors="strict") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                tokens = line.split()
                if tokens[0] not in {
                    "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "ENDATA",
                }:
                    raise UnsupportedProblem(f"unsupported section {tokens[0]!r}")
                section = tokens[0]
                if section == "NAME" and len(tokens) > 1:
                    name = tokens[1]
                continue
            tokens = line.split()
            if section == "ROWS":
                if len(tokens) != 2:
                    raise UnsupportedProblem(f"bad ROWS line: {line!r}")
                kind, row_name = tokens
                if kind == "N":
                    if objective_row is not None:
                        raise UnsupportedProblem("multiple N rows")
                    objective_row = row_name
                elif kind == "G":
                    if row_name in row_index:
                        raise UnsupportedProblem(f"duplicate row {row_name!r}")
                    row_index[row_name] = len(row_names)
                    row_names.append(row_name)
                    rows.append({})
                else:
                    raise UnsupportedProblem(f"row type {kind!r} (only G rows supported)")
            elif section == "COLUMNS":
                column = intern_column(tokens[0])
                for row_name, value in _pairs(tokens[1:], False):
                    if row_name == objective_row:
                        c[column] += Fraction(value)
                        continue
                    if row_name not in row_index:
                        raise UnsupportedProblem(f"unknown row {row_name!r} in COLUMNS")
                    i = row_index[row_name]
                    rows[i][column] = rows[i].get(column, Fraction(0)) + Fraction(value)
            elif section == "RHS":
                # ベクタ名は専変数で受ける。name は NAME セクション以来の問題名
                # (ExactQp.name → Certificate.problem → CSV 行の problem 列) で、
                # ここで上書きすると基準行の問題列が破れる (Codex P2 4370997709)。
                vector_name = tokens[0] if len(tokens) % 2 == 1 else None
                for row_name, value in _pairs(tokens, len(tokens) % 2 == 1):
                    rhs_vectors.record(rhs, "RHS", vector_name, row_name, Fraction(value))
            elif section == "RANGES":
                raise UnsupportedProblem("RANGES is not supported")
            elif section == "BOUNDS":
                if len(tokens) not in (2, 3):
                    raise UnsupportedProblem(f"bad BOUNDS line: {line!r}")
                if tokens[0] != "FR":
                    raise UnsupportedProblem(f"bound type {tokens[0]!r} (only FR supported)")
                column = tokens[-1]
                if column not in col_index:
                    raise UnsupportedProblem(f"bound for unknown column {column!r}")
                free.add(column)
            elif section == "QUADOBJ":
                if len(tokens) != 3:
                    raise UnsupportedProblem(f"bad QUADOBJ line: {line!r}")
                left, right, value = tokens
                if left != right or Fraction(value) != 1:
                    raise UnsupportedProblem(f"Q is not the identity: {line!r}")
                if left in q_diag:
                    raise UnsupportedProblem(f"duplicate QUADOBJ entry {left!r}")
                q_diag.add(left)
            elif section == "ENDATA":
                break
            else:
                raise UnsupportedProblem(f"data line in section {section!r}")

    if objective_row is None:
        raise UnsupportedProblem("missing objective N row")
    unknown_rhs = set(rhs) - set(row_names) - {objective_row}
    if unknown_rhs:
        raise UnsupportedProblem(f"RHS for unknown row {sorted(unknown_rhs)[:4]}")
    n = len(c)
    m = len(rows)
    if n == 0 or m == 0:
        raise UnsupportedProblem("empty problem")
    if len(free) != n:
        missing = sorted(set(col_index) - free)
        raise UnsupportedProblem(f"all variables must be FR (missing {missing[:4]})")
    if q_diag != set(col_index):
        raise UnsupportedProblem("QUADOBJ must list every diagonal entry of Q = I")
    for row_name in row_names:
        if rhs.get(row_name, Fraction(0)) != 0:
            raise UnsupportedProblem(f"G row {row_name!r} has nonzero RHS")
    if objective_row in rhs:
        c0 = -rhs[objective_row]
    else:
        c0 = Fraction(0)

    # LISWET の構造: m = n - 2、行 i は列 i, i+1, i+2 に 1, -2, 1。
    if m != n - 2:
        raise UnsupportedProblem(f"expected m = n - 2, got n={n}, m={m}")
    normalized: list[tuple[tuple[int, Fraction], ...]] = []
    for i, row in enumerate(rows):
        entries = tuple(sorted((j, v) for j, v in row.items() if v != 0))
        expected = ((i, Fraction(1)), (i + 1, Fraction(-2)), (i + 2, Fraction(1)))
        if entries != expected:
            raise UnsupportedProblem(f"row {i} is not the LISWET second difference: {entries!r}")
        normalized.append(entries)
    return ExactQp(name=name, n=n, m=m, c0=c0, c=tuple(c), rows=tuple(normalized))


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
    if not active:
        raise CertificateError("no active rows")
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

    inactive_min = min((ax[i] for i in inactive), default=None)
    return Certificate(
        problem=qp.name,
        objective=objective,
        c0=qp.c0,
        gaps=gaps,
        y_min=min(y[i] for i in active),
        inactive_min=inactive_min,
        reported=reported,
        n=qp.n,
        m=qp.m,
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
    args = parser.parse_args(argv)

    gaps = None
    if args.gaps is not None:
        if len(args.paths) != 1:
            parser.error("--gaps accepts exactly one problem")
        gaps = tuple(int(token) for token in args.gaps.split(",") if token.strip())

    status = 0
    for path in args.paths:
        try:
            cert = certify(path, gaps=gaps, time_limit=args.time_limit)
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
