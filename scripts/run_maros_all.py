#!/usr/bin/env python3
"""
Maros-Meszaros QP benchmark runner for the Rust solver.

Downloads all 138 .mat files from YimingYAN/QP-Test-Problems, converts
each to our text format, and runs the qp_runner binary.

Usage:
    python3 scripts/run_maros_all.py [--timeout=10] [--build]

Results are written to reports/cmd126_maros_meszaros_raw.csv
"""

import subprocess
import sys
import io
import time
import os
import csv
import urllib.request
from pathlib import Path

import scipy.io
import numpy as np
from scipy.sparse import issparse

BASE_URL = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/MAT_Files/"
TIMEOUT_SEC = 10.0
RUNNER_BIN = str(Path(__file__).parent.parent / "target" / "release" / "qp_runner")
CACHE_DIR = Path("/tmp/maros_meszaros_mat")
OUTPUT_CSV = Path("/Users/hika019/Develop/multi-agent-shogun/reports/cmd126_maros_meszaros_raw.csv")

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

INF_ENCODE = 1e300

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
    except Exception as e:
        return None

def load_mat_problem(data_bytes):
    """Load .mat bytes and return (n, m_orig, Q, c, A, rl, ru, lb, ub)."""
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
    """Convert rl <= Ax <= ru to list of (A_row, b_val) inequalities.
    Returns (A_ub_coo_rows, A_ub_coo_cols, A_ub_coo_vals, b_ub) with Ax_ub <= b_ub."""
    if issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.array(A)
    m_orig, n = A_dense.shape
    rows_list = []
    for i in range(m_orig):
        row = A_dense[i, :]
        rli = rl[i]
        rui = ru[i]
        # upper bound: Ax[i] <= ru[i]
        if np.isfinite(rui):
            rows_list.append((row, rui))
        # lower bound: Ax[i] >= rl[i] => -Ax[i] <= -rl[i]
        if np.isfinite(rli):
            rows_list.append((-row, -rli))
    m_ub = len(rows_list)
    if m_ub == 0:
        return [], [], [], m_ub, 0
    A_ub = np.array([r for r, _ in rows_list])
    b_ub = np.array([b for _, b in rows_list])
    # Convert to COO
    coo_rows = []
    coo_cols = []
    coo_vals = []
    for r in range(m_ub):
        for c_idx in range(n):
            v = A_ub[r, c_idx]
            if abs(v) > 1e-15:
                coo_rows.append(r)
                coo_cols.append(c_idx)
                coo_vals.append(v)
    return coo_rows, coo_cols, coo_vals, b_ub, m_ub

def q_to_upper_coo(Q, n):
    """Convert Q matrix to upper triangular COO format."""
    if issparse(Q):
        Q = Q.toarray()
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j in range(i, n):
            v = Q[i, j]
            if abs(v) > 1e-15:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    return rows, cols, vals

def make_input_text(n, m_ub, c, lb, ub, q_rows, q_cols, q_vals, a_rows, a_cols, a_vals, b_ub):
    """Build the text format for qp_runner."""
    def fmt_float(v):
        if v == float('inf') or v > 1e200:
            return f"{INF_ENCODE}"
        if v == float('-inf') or v < -1e200:
            return f"{-INF_ENCODE}"
        return f"{v:.15g}"

    lines = []
    lines.append(f"{n} {m_ub}")
    lines.append(" ".join(fmt_float(v) for v in c))
    lines.append(" ".join(fmt_float(v) for v in lb))
    lines.append(" ".join(fmt_float(v) for v in ub))
    lines.append(str(len(q_rows)))
    for r, c_idx, v in zip(q_rows, q_cols, q_vals):
        lines.append(f"{r} {c_idx} {v:.15g}")
    lines.append(str(len(a_rows)))
    for r, c_idx, v in zip(a_rows, a_cols, a_vals):
        lines.append(f"{r} {c_idx} {v:.15g}")
    if m_ub > 0:
        lines.append(" ".join(fmt_float(v) for v in b_ub))
    return "\n".join(lines) + "\n"

def run_problem(name, input_text, timeout_sec):
    """Run qp_runner with input_text; return (status, objective, iterations, elapsed, error)."""
    try:
        t0 = time.time()
        proc = subprocess.run(
            [RUNNER_BIN],
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
            status = parts[0]
            obj = float(parts[1])
            iters = int(parts[2])
            return status, obj, iters, elapsed, None
        return "ERROR", None, None, elapsed, f"bad output: {out}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, None, timeout_sec, None
    except Exception as e:
        return "ERROR", None, None, 0.0, str(e)

def classify_problem(Q_sp, A, rl, ru, lb, ub):
    """Classify problem properties for failure analysis."""
    n = lb.shape[0]
    has_bounds = any(np.isfinite(lb)) or any(np.isfinite(ub))
    has_equality = any(np.isfinite(rl) & np.isfinite(ru) & (np.abs(rl - ru) < 1e-10))
    has_ineq = any(np.isfinite(rl) | np.isfinite(ru))
    m_orig = len(rl)
    return {
        'n': n,
        'm_orig': m_orig,
        'has_bounds': has_bounds,
        'has_equality': has_equality,
    }

def main():
    # Check binary exists
    if not os.path.exists(RUNNER_BIN):
        print(f"ERROR: qp_runner binary not found at {RUNNER_BIN}")
        print("Build first with: cargo build --release --bin qp_runner")
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(PROBLEM_NAMES)

    print(f"Running {total} Maros-Meszaros problems (timeout={TIMEOUT_SEC}s)")
    print(f"Binary: {RUNNER_BIN}")
    print("-" * 80)

    for i, name in enumerate(PROBLEM_NAMES, 1):
        # Download
        data = download_mat(name)
        if data is None:
            print(f"[{i}/{total}] {name}: DOWNLOAD_ERROR")
            results.append({
                'name': name, 'n': '?', 'm_orig': '?', 'm_ub': '?',
                'status': 'DOWNLOAD_ERROR', 'objective': '', 'iterations': '',
                'elapsed': '', 'error': 'download failed',
                'has_bounds': '', 'has_equality': '',
            })
            continue

        # Parse
        try:
            Q, c, A, rl, ru, lb, ub = load_mat_problem(data)
        except Exception as e:
            print(f"[{i}/{total}] {name}: PARSE_ERROR {e}")
            results.append({
                'name': name, 'n': '?', 'm_orig': '?', 'm_ub': '?',
                'status': 'PARSE_ERROR', 'objective': '', 'iterations': '',
                'elapsed': '', 'error': str(e),
                'has_bounds': '', 'has_equality': '',
            })
            continue

        n = len(c)
        m_orig = len(rl)
        props = classify_problem(Q, A, rl, ru, lb, ub)

        # Convert to our format
        try:
            a_rows, a_cols, a_vals, b_ub, m_ub = convert_to_ineq(A, rl, ru)
            q_rows, q_cols, q_vals = q_to_upper_coo(Q, n)
        except Exception as e:
            print(f"[{i}/{total}] {name}: CONVERT_ERROR {e}")
            results.append({
                'name': name, 'n': n, 'm_orig': m_orig, 'm_ub': '?',
                'status': 'CONVERT_ERROR', 'objective': '', 'iterations': '',
                'elapsed': '', 'error': str(e),
                'has_bounds': props['has_bounds'], 'has_equality': props['has_equality'],
            })
            continue

        # Build input
        lb_clip = np.where(np.isfinite(lb), lb, -INF_ENCODE)
        ub_clip = np.where(np.isfinite(ub), ub, INF_ENCODE)
        input_text = make_input_text(n, m_ub, c, lb_clip, ub_clip,
                                      q_rows, q_cols, q_vals,
                                      a_rows, a_cols, a_vals, b_ub)

        # Run
        status, obj, iters, elapsed, err = run_problem(name, input_text, TIMEOUT_SEC)

        flag = ""
        if status == "Optimal": flag = "✓"
        elif status == "TIMEOUT": flag = "⏱"
        elif status == "Infeasible": flag = "✗"
        else: flag = "!"

        print(f"[{i:3d}/{total}] {name:<15} n={n:6d} m_ub={m_ub:6d}  {flag} {status:<14} "
              f"t={elapsed:.2f}s  iters={iters}")

        results.append({
            'name': name, 'n': n, 'm_orig': m_orig, 'm_ub': m_ub,
            'status': status, 'objective': obj if obj is not None else '',
            'iterations': iters if iters is not None else '',
            'elapsed': f"{elapsed:.3f}", 'error': err or '',
            'has_bounds': props['has_bounds'], 'has_equality': props['has_equality'],
        })

    # Write CSV
    fieldnames = ['name', 'n', 'm_orig', 'm_ub', 'status', 'objective',
                  'iterations', 'elapsed', 'error', 'has_bounds', 'has_equality']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 80)
    from collections import Counter
    counts = Counter(r['status'] for r in results)
    total_ran = len(results)
    print(f"TOTAL: {total_ran} problems")
    for st, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {st}: {cnt}")
    print(f"\nResults saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
