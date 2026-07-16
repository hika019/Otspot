#!/usr/bin/env python3
"""Solves a single problem file with HiGHS (highspy) and prints one CSV row:

    problem,status,objective,time_sec

Supports the QPS-family (LP + convex QP objective, via `.mps`-extension
dispatch) and `.qplib` (via the `dump_problem` Rust converter). `.cbf`
(SOCP/MISOCP) is out of scope for HiGHS — it has no conic solver — and is
reported as `Unsupported(no-SOCP)` without attempting a solve.

Usage: solve_one_highs.py <path> [--timeout SEC]
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import highspy

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    classify,
    coo_column_sorted_to_csc,
    csv_escape,
    dump_problem,
    read_qp_dump,
)

# HiGHS QCQP is not supported by this crate's Python bindings (no quadratic
# constraint API); QCQP-class QPLIB files are reported as Unsupported.
UNSUPPORTED_QC = "Unsupported(QCQP-not-supported-by-HiGHS)"
UNSUPPORTED_CBF = "Unsupported(no-SOCP-in-HiGHS)"


def build_highs_from_qp_dump(d):
    if d.qc:
        return None
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_ = d.n
    lp.num_row_ = d.m
    lp.col_cost_ = list(d.c)
    lp.col_lower_ = [b[0] for b in d.bounds]
    lp.col_upper_ = [b[1] for b in d.bounds]
    lp.offset_ = d.obj_offset
    lp.sense_ = highspy.ObjSense.kMinimize

    row_lower, row_upper = [], []
    for bi, ct in zip(d.b, d.ctypes):
        if ct == 0:  # Le
            row_lower.append(-highspy.kHighsInf)
            row_upper.append(bi)
        elif ct == 1:  # Ge
            row_lower.append(bi)
            row_upper.append(highspy.kHighsInf)
        else:  # Eq
            row_lower.append(bi)
            row_upper.append(bi)
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper

    a_start, a_index, a_value = coo_column_sorted_to_csc(d.a, d.n)
    A = highspy.HighsSparseMatrix()
    A.format_ = highspy.MatrixFormat.kColwise
    A.num_col_ = d.n
    A.num_row_ = d.m
    A.start_ = a_start
    A.index_ = a_index
    A.value_ = a_value
    lp.a_matrix_ = A

    if d.integer_vars:
        integrality = [highspy.HighsVarType.kContinuous] * d.n
        for idx in d.integer_vars:
            integrality[idx] = highspy.HighsVarType.kInteger
        lp.integrality_ = integrality

    if d.q:
        upper = [(r, c, v) for (r, c, v) in d.q if r <= c]
        h_start, h_index, h_value = coo_column_sorted_to_csc(upper, d.n)
        hess = highspy.HighsHessian()
        hess.dim_ = d.n
        hess.format_ = highspy.HessianFormat.kTriangular
        hess.start_ = h_start
        hess.index_ = h_index
        hess.value_ = h_value
        model.hessian_ = hess

    st = h.passModel(model)
    if st != highspy.HighsStatus.kOk:
        return None
    return h


def status_name(model_status) -> str:
    return model_status.name.removeprefix("k")


def solve_mps_family(path: str, timeout: float):
    fd, tmp_mps = tempfile.mkstemp(prefix="highs_", suffix=".mps")
    os.close(fd)
    try:
        # HiGHS dispatches purely on filename suffix; copy to a `.mps` name
        # regardless of the source suffix's case (`.QPS`/`.qps`/`.mps`).
        shutil.copyfile(path, tmp_mps)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(timeout))
        rm_status = h.readModel(tmp_mps)
        if rm_status != highspy.HighsStatus.kOk:
            return "ParseError", None, 0.0
        start = time.time()
        h.run()
        elapsed = time.time() - start
        status = status_name(h.getModelStatus())
        obj = None
        if status in ("Optimal", "TimeLimit", "IterationLimit"):
            info = h.getInfo()
            if info.primal_solution_status == highspy.kSolutionStatusFeasible:
                obj = info.objective_function_value
        return status, obj, elapsed
    finally:
        os.unlink(tmp_mps)


def solve_qplib(path: str, timeout: float):
    dump_path = dump_problem(path)
    try:
        d = read_qp_dump(dump_path)
    finally:
        os.unlink(dump_path)
    if d.qc:
        return UNSUPPORTED_QC, None, 0.0
    if d.integer_vars and d.q:
        # HiGHS does not support MIQP (integer + nonzero Hessian); let it
        # attempt anyway and report whatever status it returns as real signal.
        pass
    h = build_highs_from_qp_dump(d)
    if h is None:
        return UNSUPPORTED_QC, None, 0.0
    h.setOptionValue("time_limit", float(timeout))
    start = time.time()
    h.run()
    elapsed = time.time() - start
    status = status_name(h.getModelStatus())
    obj = None
    if status in ("Optimal", "TimeLimit", "IterationLimit"):
        info = h.getInfo()
        if info.primal_solution_status == highspy.kSolutionStatusFeasible:
            obj = info.objective_function_value
    return status, obj, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--timeout", type=float, default=1000.0)
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.path))[0]
    kind = classify(args.path)

    try:
        if kind == "mps":
            status, obj, elapsed = solve_mps_family(args.path, args.timeout)
        elif kind == "qplib":
            status, obj, elapsed = solve_qplib(args.path, args.timeout)
        else:  # cbf
            status, obj, elapsed = UNSUPPORTED_CBF, None, 0.0
    except Exception as e:  # noqa: BLE001 - surface as a CSV row, not a crash
        status, obj, elapsed = f"Error({csv_escape(str(e))})", None, 0.0

    obj_str = f"{obj:.10e}" if obj is not None else ""
    print(f"{name},{status},{obj_str},{elapsed:.3f}")


if __name__ == "__main__":
    main()
