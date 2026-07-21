"""Shared helpers for the HiGHS/SCIP comparison runners.

Neither HiGHS (highspy) nor the SCIP build bundled in the `pyscipopt` PyPI
wheel (SCIP 10.0.2 — no reader_cbf.c/reader_qplib.c compiled in; verified via
`strings` on libscip-*.so) can read `.qplib` or `.cbf` files natively. Both
*can* read the QPS-family files directly once presented with a `.mps`
extension (extension-based dispatch; case-sensitive to the literal `mps`
suffix for HiGHS, but SCIP's `readProblem(path, extension=...)` accepts an
explicit override).

For `.qplib`/`.cbf`, this module shells out to the `dump_problem` Rust
example (built from otspot's own, already-tested QPLIB/CBF parsers) which
serializes the parsed problem as a plain whitespace-token stream. Building a
second, independent QPLIB/CBF parser in Python would duplicate nontrivial,
already-correct logic (range-constraint splitting, MAX-sense sign flips,
MILP/MIQP dispatch, cone reordering, MISOCP integrality) and risk subtly
disagreeing with the parser Otspot itself uses to produce baseline results.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DUMP_PROBLEM_BIN = os.path.join(REPO_ROOT, "target", "release", "examples", "dump_problem")

INF = float("inf")

# QPS-family suites store LP/QP problems as extended MPS (QUADOBJ/QMATRIX
# sections) but use a `.QPS`/`.qps` suffix; both HiGHS and SCIP only
# recognize MPS-family readers via `.mps`.
MPS_LIKE_EXTENSIONS = {"mps", "qps"}


def classify(path: str) -> str:
    """Returns 'mps', 'qplib', or 'cbf' based on file extension."""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext in MPS_LIKE_EXTENSIONS:
        return "mps"
    if ext == "qplib":
        return "qplib"
    if ext == "cbf":
        return "cbf"
    raise ValueError(f"unrecognized extension for {path!r}: {ext!r}")


class TokenReader:
    """Pull-based reader over a whitespace-delimited token stream."""

    def __init__(self, path: str):
        with open(path) as f:
            self._toks = f.read().split()
        self._i = 0

    def tok(self) -> str:
        t = self._toks[self._i]
        self._i += 1
        return t

    def i(self) -> int:
        return int(self.tok())

    def f(self) -> float:
        return float(self.tok())


@dataclass
class QpDump:
    n: int
    m: int
    obj_offset: float
    bounds: list  # [(lb, ub), ...] length n
    c: list  # length n
    q: list  # [(row, col, val), ...] full symmetric storage
    a: list  # [(row, col, val), ...] column-sorted
    b: list  # length m
    ctypes: list  # 0=Le, 1=Ge, 2=Eq; length m
    qc: list = field(default_factory=list)  # [] or length m of triplet lists
    integer_vars: list = field(default_factory=list)


@dataclass
class ConicDump:
    n: int
    p: int
    m: int
    l: int
    soc_dims: list
    maximize: bool
    obj_offset: float
    c: list
    a: list  # equality rows, [(row, col, val), ...]
    b: list
    g: list  # conic rows, [(row, col, val), ...]
    h: list
    integers: list  # [(idx, lb, ub), ...]


def _read_triplets(r: TokenReader, nnz: int):
    return [(r.i(), r.i(), r.f()) for _ in range(nnz)]


def read_qp_dump(path: str) -> QpDump:
    r = TokenReader(path)
    tag = r.tok()
    assert tag == "QP", f"expected QP tag, got {tag!r} in {path}"
    n = r.i()
    m = r.i()
    obj_offset = r.f()
    bounds = [(r.f(), r.f()) for _ in range(n)]
    c = [r.f() for _ in range(n)]
    q_nnz = r.i()
    q = _read_triplets(r, q_nnz)
    a_nnz = r.i()
    a = _read_triplets(r, a_nnz)
    b = [r.f() for _ in range(m)]
    ctypes = [r.i() for _ in range(m)]
    has_qc = r.i()
    qc = []
    if has_qc:
        for _ in range(m):
            nnz_k = r.i()
            qc.append(_read_triplets(r, nnz_k))
    num_int = r.i()
    integer_vars = [r.i() for _ in range(num_int)]
    return QpDump(n, m, obj_offset, bounds, c, q, a, b, ctypes, qc, integer_vars)


def read_conic_dump(path: str) -> ConicDump:
    r = TokenReader(path)
    tag = r.tok()
    assert tag == "CONIC", f"expected CONIC tag, got {tag!r} in {path}"
    n = r.i()
    p = r.i()
    m = r.i()
    l = r.i()
    num_soc = r.i()
    soc_dims = [r.i() for _ in range(num_soc)]
    maximize = bool(r.i())
    obj_offset = r.f()
    c = [r.f() for _ in range(n)]
    a_nnz = r.i()
    a = _read_triplets(r, a_nnz)
    b = [r.f() for _ in range(p)]
    g_nnz = r.i()
    g = _read_triplets(r, g_nnz)
    h = [r.f() for _ in range(m)]
    num_int = r.i()
    integers = [(r.i(), r.f(), r.f()) for _ in range(num_int)]
    return ConicDump(n, p, m, l, soc_dims, maximize, obj_offset, c, a, b, g, h, integers)


def dump_problem(path: str) -> str:
    """Runs the `dump_problem` Rust binary on `path`, returns the dump file path.

    Raises RuntimeError with the tool's stderr on parse failure (e.g. a
    deliberately-infeasible-by-construction bounds file, or a format the
    otspot-io parser rejects — `Unsupported`/`ParseError`).
    """
    if not os.path.isfile(DUMP_PROBLEM_BIN):
        raise RuntimeError(
            f"dump_problem binary not found at {DUMP_PROBLEM_BIN}; "
            "build it first: cargo build --release --example dump_problem"
        )
    fd, out_path = tempfile.mkstemp(prefix="dump_", suffix=".txt")
    os.close(fd)
    result = subprocess.run(
        [DUMP_PROBLEM_BIN, path, out_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        os.unlink(out_path)
        raise RuntimeError(result.stderr.strip() or f"dump_problem failed on {path}")
    return out_path


def group_by_row(triplets):
    """Groups (row, col, val) triplets into {row: [(col, val), ...]}."""
    rows: dict = {}
    for r, c, v in triplets:
        rows.setdefault(r, []).append((c, v))
    return rows


def coo_column_sorted_to_csc(triplets, ncols):
    """Builds CSC (start, index, value) arrays from triplets already in
    non-decreasing column order (as emitted by `dump_problem`)."""
    start = [0] * (ncols + 1)
    index = []
    value = []
    for r, c, v in triplets:
        start[c + 1] += 1
        index.append(r)
        value.append(v)
    for j in range(ncols):
        start[j + 1] += start[j]
    return start, index, value


def csv_escape(s: str) -> str:
    return str(s).replace(",", ";").replace("\n", " ")
