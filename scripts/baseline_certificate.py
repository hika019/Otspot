"""Exact checks for the binary64 QP model, independent of candidate recovery."""
from collections import namedtuple
from fractions import Fraction
from functools import lru_cache
import hashlib
import gzip
import json
import math
from pathlib import Path
import time

# Resource exhaustion means unknown; these limits never relax a proof.
PSD_SECONDS = 30.0
PSD_OPERATIONS = 2_000_000
REFERENCE_TOLERANCE = Fraction(1, 10_000_000)
CHECKER_VERSION = "binary64-enclosure-v1"


# v1 (生 bytes 連結の単一 hash) は分割シフト攻撃に無防備だった: バイト列
# (例: shebang 行) をファイル境界を跨いで移動すると連結結果が同一で digest が
# 不変だった (レビュー PoC 実測)。v2 はファイルごとに自己区切りフレーム
# (name \0 length \0 content-hash \0) を個別 hash し、その連結を外层 hash する。
# 名前と長さ自体が preimage に含まれるため再分割もファイル間の内容入れ替えも
# digest を保存できず、scheme tag により v1 値は v2 の値域に到達不能
# (旧書式 digest の混在受理が構造的に起きない)。
CHECKER_FILES = ("baseline_certificate.py", "highs_baseline_truth.py",
                 "liswet_exact_certificate.py", "revalidate_baselines.py")


def digest_of(files):
    """files: ordered mapping name -> bytes. Pure; checker_digest feeds the repo."""
    parts = [b"checker-files-v2\0"]
    for name, content in files.items():
        frame = (name.encode("utf-8") + b"\0" + str(len(content)).encode() + b"\0"
                 + hashlib.sha256(content).hexdigest().encode() + b"\0")
        parts.append(hashlib.sha256(frame).digest())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def checker_digest():
    root = Path(__file__).parent
    return digest_of({name: (root / name).read_bytes() for name in CHECKER_FILES})


LOADED_CHECKER_DIGEST = checker_digest()


def rational(value):
    return value if isinstance(value, Fraction) else Fraction(float(value))


def columns(matrix):
    matrix = matrix.tocsc(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return tuple(tuple((int(matrix.indices[k]), rational(matrix.data[k]))
                       for k in range(matrix.indptr[j], matrix.indptr[j + 1])
                       if matrix.data[k]) for j in range(matrix.shape[1]))


@lru_cache(maxsize=8)
def prove_psd(q):
    """Exact symmetric diagonal dominance or sparse rational LDL; None is unknown."""
    rows = [dict(col) for col in q]
    for i, row in enumerate(rows):
        if row.get(i, 0) < 0:
            return False
        if any(rows[j].get(i, 0) != value for j, value in row.items()):
            return False
    if all(row.get(i, 0) >= sum(abs(v) for j, v in row.items() if j != i)
           for i, row in enumerate(rows)):
        return True
    deadline = time.monotonic() + PSD_SECONDS
    operations = 0
    for k, row in enumerate(rows):
        if time.monotonic() >= deadline:
            return None
        pivot = row.get(k, 0)
        neighbours = [(j, v) for j, v in row.items() if j > k and v]
        if pivot < 0 or (pivot == 0 and neighbours):
            return False
        if pivot == 0:
            continue
        for offset, (i, vi) in enumerate(neighbours):
            for j, vj in neighbours[offset:]:
                operations += 1
                if operations > PSD_OPERATIONS or time.monotonic() >= deadline:
                    return None
                value = rows[i].get(j, 0) - vi * vj / pivot
                if value:
                    rows[i][j] = value
                    rows[j][i] = value
                else:
                    rows[i].pop(j, None)
                    rows[j].pop(i, None)
            rows[i].pop(k, None)
        rows[k] = {}
    return True


Enclosure = namedtuple("Enclosure", "lower upper primal linearization dual model_hash")


_ModelData = namedtuple("_ModelData", "n m c0 c A Q L U lo up Arow")


class ExactModel(_ModelData):
    """An immutable, validated numeric snapshot of one binary64 model."""
    __slots__ = ()

    def __new__(cls, prob):
        n, m = prob["n"], prob["m"]
        if not isinstance(n, int) or not isinstance(m, int) or n < 0 or m < 0:
            raise ValueError("invalid model dimensions")
        if prob["A"].shape != (m, n) or prob["Q"].shape != (n, n):
            raise ValueError("matrix shape does not match model dimensions")
        for key, length in (("c", n), ("L", m), ("U", m), ("lo", n), ("up", n)):
            if getattr(prob[key], "shape", (len(prob[key]),)) != (length,):
                raise ValueError(f"{key} shape does not match model dimensions")
        c0, c = rational(prob["c0"]), tuple(map(rational, prob["c"]))
        A, Q = columns(prob["A"]), columns(prob["Q"])

        def bounds(values, lower):
            result = []
            for value in values:
                value = float(value)
                if math.isnan(value) or value == (math.inf if lower else -math.inf):
                    raise ValueError("invalid bound")
                result.append(None if not math.isfinite(value) else rational(value))
            return tuple(result)

        L, U = bounds(prob["L"], True), bounds(prob["U"], False)
        lo, up = bounds(prob["lo"], True), bounds(prob["up"], False)
        for lower, upper in ((L, U), (lo, up)):
            if any(a is not None and b is not None and a > b for a, b in zip(lower, upper)):
                raise ValueError("lower bound exceeds upper bound")
        rows = [[] for _ in range(m)]
        for j, col in enumerate(A):
            for i, value in col:
                rows[i].append((j, value))
        return super().__new__(cls, n, m, c0, c, A, Q, L, U, lo, up, tuple(map(tuple, rows)))

    def fingerprint(self):
        numeric = (self.n, self.m, self.c0, self.c, self.A, self.Q,
                   self.L, self.U, self.lo, self.up)
        return hashlib.sha256(repr(numeric).encode()).hexdigest()

    def reduced_cost(self, xh, y):
        return tuple(self.c[j] + sum(v * xh[i] for i, v in self.Q[j])
                     - sum(v * y[i] for i, v in self.A[j]) for j in range(self.n))

    def objective(self, x):
        return self.c0 + sum(self.c[j] * x[j] + x[j] *
                             sum(v * x[i] for i, v in self.Q[j]) / 2
                             for j in range(self.n))

    def primal_bound(self, x):
        if len(x) != self.n:
            return None
        ax = [sum(v * x[j] for j, v in row) for row in self.Arow]
        for values, lower, upper in ((x, self.lo, self.up), (ax, self.L, self.U)):
            if any((lo is not None and v < lo) or (up is not None and v > up)
                   for v, lo, up in zip(values, lower, upper)):
                return None
        return self.objective(x)

    def dual_bound(self, xh, y):
        if len(xh) != self.n or len(y) != self.m or prove_psd(self.Q) is not True:
            return None
        value = self.c0 - sum(xh[j] * sum(v * xh[i] for i, v in self.Q[j])
                              for j in range(self.n)) / 2
        for yi, lo, up in zip(y, self.L, self.U):
            bound = lo if yi > 0 else up
            if yi and bound is None:
                return None
            if yi:
                value += yi * bound
        for gj, lo, up in zip(self.reduced_cost(xh, y), self.lo, self.up):
            bound = lo if gj > 0 else up
            if gj and bound is None:
                return None
            if gj:
                value += gj * bound
        return value

    def check(self, x, xh, y):
        x, xh, y = tuple(map(rational, x)), tuple(map(rational, xh)), tuple(map(rational, y))
        upper, lower = self.primal_bound(x), self.dual_bound(xh, y)
        if upper is None or lower is None or lower > upper:
            return None
        return Enclosure(lower, upper, x, xh, y, self.fingerprint())

    def diagonal_linearization(self, xh, y):
        """Minimize the separable quadratic Lagrangian to propose a tangent point."""
        if any(i != j for j, col in enumerate(self.Q) for i, _ in col):
            return None
        point = list(map(rational, xh))
        y = tuple(map(rational, y))
        for j, col in enumerate(self.Q):
            qj = dict(col).get(j, 0)
            if qj > 0:
                point[j] = (sum(v * y[i] for i, v in self.A[j]) - self.c[j]) / qj
                if self.lo[j] is not None:
                    point[j] = max(point[j], self.lo[j])
                if self.up[j] is not None:
                    point[j] = min(point[j], self.up[j])
        return point


def reference_error(enclosure, text):
    """Bound both the written decimal and the binary64 value read by benchmarks."""
    values = (Fraction(text), Fraction(float(text)))
    return max(max(abs(r - enclosure.lower), abs(r - enclosure.upper)) / (1 + abs(r))
               for r in values)


def evidence(enclosure, text, input_sha256):
    return dict(checker=CHECKER_VERSION, input_sha256=input_sha256,
                model_sha256=enclosure.model_hash, reference=text,
                lower=str(enclosure.lower), upper=str(enclosure.upper),
                relative_error=str(reference_error(enclosure, text)),
                primal=list(map(str, enclosure.primal)),
                linearization=list(map(str, enclosure.linearization)),
                dual=list(map(str, enclosure.dual)))


def save_evidence(path, enclosure, reference, input_sha256, proposal=None,
                  expected_digest=LOADED_CHECKER_DIGEST):
    """Persist the witness and return compact provenance for a retained CSV claim."""
    if checker_digest() != expected_digest:
        raise ValueError("checker changed during verification; rerun with a fixed source tree")
    if reference_error(enclosure, reference) > REFERENCE_TOLERANCE:
        raise ValueError("reference is outside its certified error budget")
    payload = evidence(enclosure, reference, input_sha256)
    payload.update(checker_sha256=expected_digest, proposal=proposal)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle, allow_nan=False)
    compact = {k: v for k, v in payload.items() if k not in ("primal", "linearization", "dual")}
    compact.update(proof=path.name, proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return compact


def replay_evidence(prob, payload, input_sha256):
    """Recompute a persisted witness; serialized bounds are checked, never trusted."""
    if payload["input_sha256"] != input_sha256:
        raise ValueError("input hash mismatch")
    model = ExactModel(prob)
    enclosure = model.check(*([Fraction(v) for v in payload[key]]
                             for key in ("primal", "linearization", "dual")))
    if enclosure is None or enclosure.model_hash != payload["model_sha256"]:
        raise ValueError("invalid witness or model hash mismatch")
    if enclosure.lower != Fraction(payload["lower"]) or enclosure.upper != Fraction(payload["upper"]):
        raise ValueError("stored bounds do not match independently recomputed bounds")
    if reference_error(enclosure, payload["reference"]) > REFERENCE_TOLERANCE:
        raise ValueError("stored reference exceeds its error budget")
    return enclosure
