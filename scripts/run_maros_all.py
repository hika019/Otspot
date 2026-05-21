#!/usr/bin/env python3
"""
Maros-Meszaros QP benchmark runner for the Rust solver.

Downloads all 138 .mat files from YimingYAN/QP-Test-Problems, converts
each to our text format, and runs the qp_runner binary.

Usage:
    python3 scripts/run_maros_all.py [options]

Options:
    --timeout SECS   Per-problem time limit in seconds (default: 10)
    --build          Build qp_runner before running
    --problems N [N] Run only these problem names (default: all 138)
    --output FILE    CSV output path (default: reports/maros_meszaros_raw.csv)
    --subset N       Run only the first N problems (useful for testing)
    -h, --help       Show this message and exit
"""

import argparse
import gc
import subprocess
import sys
import io
import time
import os
import csv
import urllib.request
from collections import Counter
from pathlib import Path

import scipy.io
import scipy.sparse
import numpy as np

BASE_URL = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/MAT_Files/"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_EPS = 1e-6  # qp_runner の既定許容と一致
RUNNER_BIN = str(Path(__file__).parent.parent / "target" / "release" / "qp_runner")
CACHE_DIR = Path("/tmp/maros_meszaros_mat")
DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "reports" / "maros_meszaros_raw.csv"

# Values beyond this magnitude are encoded as ±INF_ENCODE in the text protocol.
INF_ENCODE = 1e300
# Threshold below which a matrix entry is treated as structural zero.
SPARSE_ZERO_TOL = 1e-15
# Threshold for considering a bound finite.
INF_BOUND_THRESHOLD = 1e200
# |rl - ru| below this treats a constraint as an equality (rl == ru).
EQ_TOL = 1e-10

PROBLEM_NAMES = [
    "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP",
    "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
    "BOYD1", "BOYD2",
    "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300",
    "CVXQP1_L", "CVXQP1_M", "CVXQP1_S",
    "CVXQP2_L", "CVXQP2_M", "CVXQP2_S",
    "CVXQP3_L", "CVXQP3_M", "CVXQP3_S",
    "DPKLO1", "DTOC3",
    "DUAL1", "DUAL2", "DUAL3", "DUAL4",
    "DUALC1", "DUALC2", "DUALC5", "DUALC8",
    "EXDATA",
    "GENHS28",
    "GOULDQP2", "GOULDQP3",
    "HS118", "HS21", "HS268", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76",
    "HUES-MOD", "HUESTIS",
    "KSIP",
    "LASER",
    "LISWET1", "LISWET10", "LISWET11", "LISWET12",
    "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7", "LISWET8", "LISWET9",
    "LOTSCHD",
    "MOSARQP1", "MOSARQP2",
    "POWELL20",
    "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4",
    "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
    "Q25FV47", "QADLITTL", "QAFIRO", "QBANDM", "QBEACONF", "QBORE3D",
    "QBRANDY", "QCAPRI", "QE226", "QETAMACR", "QFFFFF80", "QFORPLAN",
    "QGFRDXPN", "QGROW15", "QGROW22", "QGROW7", "QISRAEL",
    "QPCBLEND", "QPCBOEI1", "QPCBOEI2", "QPCSTAIR",
    "QPILOTNO", "QPTEST", "QRECIPE", "QSC205",
    "QSCAGR25", "QSCAGR7", "QSCFXM1", "QSCFXM2", "QSCFXM3",
    "QSCORPIO", "QSCRS8", "QSCSD1", "QSCSD6", "QSCSD8",
    "QSCTAP1", "QSCTAP2", "QSCTAP3", "QSEBA",
    "QSHARE1B", "QSHARE2B", "QSHELL",
    "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S", "QSHIP12L", "QSHIP12S",
    "QSIERRA", "QSTAIR", "QSTANDAT",
    "S268",
    "STADAT1", "STADAT2", "STADAT3",
    "STCQP1", "STCQP2",
    "TAME", "UBH1", "VALUES", "YAO", "ZECEVIC2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maros-Meszaros QP benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, metavar="SECS",
        help=f"per-problem time limit in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="run 'cargo build --release --bin qp_runner' before benchmarking",
    )
    parser.add_argument(
        "--problems", nargs="+", metavar="NAME",
        help="run only these problem names (default: all)",
    )
    parser.add_argument(
        "--subset", type=int, metavar="N",
        help="run only the first N problems (for quick testing)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_CSV, metavar="FILE",
        help=f"CSV output path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--eps", type=float, default=DEFAULT_EPS, metavar="EPS",
        help=f"solver tolerance forwarded to qp_runner --eps (default: {DEFAULT_EPS})",
    )
    return parser.parse_args()


def download_mat(name):
    """Download .mat file and return as bytes, using cache if available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.mat"
    if cache_path.exists():
        return cache_path.read_bytes()
    url = BASE_URL + f"{name}.mat"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        cache_path.write_bytes(data)
        return data
    except Exception:
        return None


def load_mat_problem(data_bytes):
    """Load .mat bytes and return (Q, c, A, rl, ru, lb, ub) with sparse Q and A."""
    mat = scipy.io.loadmat(io.BytesIO(data_bytes))
    Q = mat['Q']
    c = mat['c'].flatten().astype(float)
    A = mat['A']
    rl = mat['rl'].flatten().astype(float)
    ru = mat['ru'].flatten().astype(float)
    lb = mat['lb'].flatten().astype(float)
    ub = mat['ub'].flatten().astype(float)
    return Q, c, A, rl, ru, lb, ub


def convert_to_ineq(A, rl, ru):
    """Convert rl <= Ax <= ru to Ax_ub <= b_ub without densifying A.

    Returns (coo_rows, coo_cols, coo_vals, b_ub, m_ub) in COO format.
    """
    A_csr = scipy.sparse.csr_matrix(A) if not scipy.sparse.issparse(A) else A.tocsr()
    m_orig, _ = A_csr.shape

    coo_rows = []
    coo_cols = []
    coo_vals = []
    b_ub_list = []
    out_row = 0

    for i in range(m_orig):
        row = A_csr.getrow(i)
        indices = row.indices
        data = row.data
        rli = rl[i]
        rui = ru[i]

        if np.isfinite(rui):
            # Ax[i] <= ru[i]
            for j, v in zip(indices, data):
                if abs(v) > SPARSE_ZERO_TOL:
                    coo_rows.append(out_row)
                    coo_cols.append(j)
                    coo_vals.append(v)
            b_ub_list.append(rui)
            out_row += 1

        if np.isfinite(rli):
            # Ax[i] >= rl[i]  =>  -Ax[i] <= -rl[i]
            for j, v in zip(indices, data):
                if abs(v) > SPARSE_ZERO_TOL:
                    coo_rows.append(out_row)
                    coo_cols.append(j)
                    coo_vals.append(-v)
            b_ub_list.append(-rli)
            out_row += 1

    m_ub = out_row
    b_ub = np.array(b_ub_list, dtype=float)
    return coo_rows, coo_cols, coo_vals, b_ub, m_ub


def q_to_upper_coo(Q, n):
    """Extract upper triangular entries of Q without full densification."""
    if scipy.sparse.issparse(Q):
        Q_up = scipy.sparse.triu(Q, format="coo")
        mask = np.abs(Q_up.data) > SPARSE_ZERO_TOL
        return Q_up.row[mask].tolist(), Q_up.col[mask].tolist(), Q_up.data[mask].tolist()
    else:
        Q_arr = np.asarray(Q)
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(i, n):
                v = Q_arr[i, j]
                if abs(v) > SPARSE_ZERO_TOL:
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)
        return rows, cols, vals


def make_input_text(n, m_ub, c, lb, ub, q_rows, q_cols, q_vals, a_rows, a_cols, a_vals, b_ub):
    """Build the text protocol for qp_runner."""
    def fmt(v):
        if v > INF_BOUND_THRESHOLD:
            return f"{INF_ENCODE}"
        if v < -INF_BOUND_THRESHOLD:
            return f"{-INF_ENCODE}"
        return f"{v:.15g}"

    lines = [
        f"{n} {m_ub}",
        " ".join(fmt(v) for v in c),
        " ".join(fmt(v) for v in lb),
        " ".join(fmt(v) for v in ub),
        str(len(q_rows)),
    ]
    for r, c_idx, v in zip(q_rows, q_cols, q_vals):
        lines.append(f"{r} {c_idx} {v:.15g}")
    lines.append(str(len(a_rows)))
    for r, c_idx, v in zip(a_rows, a_cols, a_vals):
        lines.append(f"{r} {c_idx} {v:.15g}")
    if m_ub > 0:
        lines.append(" ".join(fmt(v) for v in b_ub))
    return "\n".join(lines) + "\n"


def run_problem(name, input_text, timeout_sec, eps):
    """Run qp_runner with input_text; return (status, objective, iterations, elapsed, error)."""
    try:
        t0 = time.time()
        proc = subprocess.run(
            [RUNNER_BIN, "--eps", str(eps)],
            input=input_text.encode(),
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed = time.time() - t0
        out = proc.stdout.decode().strip()
        if not out:
            return "ERROR", None, None, elapsed, "no output"
        parts = out.split()
        if len(parts) >= 3:
            return parts[0], float(parts[1]), int(parts[2]), elapsed, None
        return "ERROR", None, None, elapsed, f"bad output: {out}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, None, timeout_sec, None
    except Exception as e:
        return "ERROR", None, None, 0.0, str(e)


def classify_problem(rl, ru, lb, ub):
    """Return a dict of problem properties for the CSV."""
    has_bounds = bool(np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)))
    has_equality = bool(
        np.any(np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) < EQ_TOL))
    )
    return {"has_bounds": has_bounds, "has_equality": has_equality}


def build_runner():
    repo_root = Path(__file__).parent.parent
    print("Building qp_runner...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "qp_runner"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print("Build failed.")
        sys.exit(1)
    print("Build succeeded.")


def main():
    args = parse_args()

    if args.build:
        build_runner()

    if not os.path.exists(RUNNER_BIN):
        print(f"ERROR: qp_runner not found at {RUNNER_BIN}")
        print("Build with: cargo build --release --bin qp_runner")
        sys.exit(1)

    problems = args.problems if args.problems else PROBLEM_NAMES
    if args.subset is not None:
        problems = problems[: args.subset]

    output_csv = args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    timeout_sec = args.timeout
    total = len(problems)

    print(f"Running {total} Maros-Meszaros problems (timeout={timeout_sec}s)")
    print(f"Binary:  {RUNNER_BIN}")
    print(f"Output:  {output_csv}")
    print("-" * 80)

    results = []

    for i, name in enumerate(problems, 1):
        data = download_mat(name)
        if data is None:
            print(f"[{i}/{total}] {name}: DOWNLOAD_ERROR")
            results.append(_error_row(name, "DOWNLOAD_ERROR", "download failed"))
            continue

        try:
            Q, c, A, rl, ru, lb, ub = load_mat_problem(data)
        except Exception as e:
            print(f"[{i}/{total}] {name}: PARSE_ERROR {e}")
            results.append(_error_row(name, "PARSE_ERROR", str(e)))
            continue
        finally:
            del data

        n = len(c)
        m_orig = len(rl)
        props = classify_problem(rl, ru, lb, ub)

        try:
            a_rows, a_cols, a_vals, b_ub, m_ub = convert_to_ineq(A, rl, ru)
            q_rows, q_cols, q_vals = q_to_upper_coo(Q, n)
        except Exception as e:
            print(f"[{i}/{total}] {name}: CONVERT_ERROR {e}")
            results.append(_error_row(name, "CONVERT_ERROR", str(e), n=n, m_orig=m_orig, **props))
            continue
        finally:
            del Q, A

        lb_clipped = np.where(np.isfinite(lb), lb, -INF_ENCODE)
        ub_clipped = np.where(np.isfinite(ub), ub, INF_ENCODE)
        input_text = make_input_text(
            n, m_ub, c, lb_clipped, ub_clipped,
            q_rows, q_cols, q_vals, a_rows, a_cols, a_vals, b_ub,
        )
        del c, lb, ub, lb_clipped, ub_clipped, rl, ru
        del a_rows, a_cols, a_vals, b_ub, q_rows, q_cols, q_vals

        status, obj, iters, elapsed, err = run_problem(name, input_text, timeout_sec, args.eps)
        del input_text
        gc.collect()

        flag = {"Optimal": "✓", "TIMEOUT": "⏱", "Infeasible": "✗"}.get(status, "!")
        print(
            f"[{i:3d}/{total}] {name:<15} n={n:6d} m_ub={m_ub:6d}  "
            f"{flag} {status:<14} t={elapsed:.2f}s  iters={iters}"
        )

        results.append({
            "name": name, "n": n, "m_orig": m_orig, "m_ub": m_ub,
            "status": status,
            "objective": obj if obj is not None else "",
            "iterations": iters if iters is not None else "",
            "elapsed": f"{elapsed:.3f}",
            "error": err or "",
            **props,
        })

    fieldnames = [
        "name", "n", "m_orig", "m_ub", "status", "objective",
        "iterations", "elapsed", "error", "has_bounds", "has_equality",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    counts = Counter(r["status"] for r in results)
    print(f"TOTAL: {len(results)} problems")
    for st, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {st}: {cnt}")
    print(f"\nResults saved to: {output_csv}")


def _error_row(name, status, error, *, n="?", m_orig="?", m_ub="?",
               has_bounds="", has_equality=""):
    return {
        "name": name, "n": n, "m_orig": m_orig, "m_ub": m_ub,
        "status": status, "objective": "", "iterations": "",
        "elapsed": "", "error": error,
        "has_bounds": has_bounds, "has_equality": has_equality,
    }


if __name__ == "__main__":
    main()
