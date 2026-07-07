#!/usr/bin/env python3
"""Solves a single problem file with SCIP (pyscipopt) and prints one CSV row:

    problem,status,objective,time_sec

Supports the QPS-family (LP/QP/MILP/MIQP, via `.mps`-extension dispatch with
an explicit `extension="mps"` override), `.qplib` (LP/QP/MILP/MIQP/QCQP, via
the `dump_problem` Rust converter), and `.cbf` (SOCP/MISOCP, same converter;
SOC cones are represented as general quadratic constraints since this SCIP
build has no dedicated SOC constraint handler exposed through pyscipopt).

Environment note (see scripts/compare_solvers/README.md): this machine's
virtual CPU exposes no AVX/AVX2/FMA (SSE4.2 only). SCIP's presolve
heuristics for nonlinear/quadratic constraints reliably crash with
`Illegal instruction` on this CPU unless heuristics are disabled — this
runner disables them whenever a model contains a nonlinear (quadratic
objective epigraph or QCQP/SOC) constraint. Pure LP/MILP models are
unaffected and keep heuristics on.

Usage: solve_one_scip.py <path> [--timeout SEC]
"""
import argparse
import os
import sys
import time

import pyscipopt as scip

sys.path.insert(0, os.path.dirname(__file__))
from common import classify, csv_escape, dump_problem, group_by_row, read_conic_dump, read_qp_dump

CTYPE_OPS = {0: "<=", 1: ">=", 2: "=="}


def _bound_or_none(v: float):
    return None if v in (float("inf"), float("-inf")) else v


def _linear_expr(xs, terms):
    return scip.quicksum(v * xs[c] for c, v in terms)


def _add_row(m, expr, ctype: int, rhs: float):
    op = CTYPE_OPS[ctype]
    if op == "<=":
        m.addCons(expr <= rhs)
    elif op == ">=":
        m.addCons(expr >= rhs)
    else:
        m.addCons(expr == rhs)


def build_scip_from_qp_dump(d):
    """Returns (model, has_nonlinear) for a QpDump (QP/MILP/MIQP/QCQP)."""
    m = scip.Model()
    m.hideOutput()
    int_set = set(d.integer_vars)
    xs = [
        m.addVar(
            lb=_bound_or_none(lb),
            ub=_bound_or_none(ub),
            vtype="I" if j in int_set else "C",
        )
        for j, (lb, ub) in enumerate(d.bounds)
    ]

    has_nonlinear = bool(d.q) or bool(d.qc)

    obj_terms = [0.5 * v * xs[r] * xs[c] for r, c, v in d.q]
    obj_terms += [v * xs[j] for j, v in enumerate(d.c) if v != 0.0]
    obj_expr = scip.quicksum(obj_terms) if obj_terms else 0.0

    if d.q:
        z = m.addVar(lb=None, ub=None, vtype="C", obj=1.0)
        m.addCons(obj_expr <= z)
        m.setMinimize()
    else:
        m.setObjective(obj_expr, sense="minimize")

    a_rows = group_by_row(d.a)
    for i in range(d.m):
        terms = a_rows.get(i, [])
        expr = _linear_expr(xs, terms)
        if d.qc and d.qc[i]:
            expr = expr + scip.quicksum(0.5 * v * xs[r] * xs[c] for r, c, v in d.qc[i])
            has_nonlinear = True
        _add_row(m, expr, d.ctypes[i], d.b[i])

    return m, has_nonlinear


def build_scip_from_conic_dump(d):
    """Returns (model, xs) for a ConicDump; SOC cones as quadratic constraints."""
    m = scip.Model()
    m.hideOutput()
    int_idx = {idx: (lb, ub) for idx, lb, ub in d.integers}
    xs = []
    for j in range(d.n):
        if j in int_idx:
            lb, ub = int_idx[j]
            xs.append(m.addVar(lb=lb, ub=ub, vtype="I"))
        else:
            xs.append(m.addVar(lb=None, ub=None, vtype="C"))

    obj_expr = scip.quicksum(v * xs[j] for j, v in enumerate(d.c) if v != 0.0)
    m.setObjective(obj_expr, sense="minimize")

    a_rows = group_by_row(d.a)
    for i in range(d.p):
        expr = _linear_expr(xs, a_rows.get(i, []))
        m.addCons(expr == d.b[i])

    g_rows = group_by_row(d.g)

    def s_expr(i):
        return d.h[i] - _linear_expr(xs, g_rows.get(i, []))

    for i in range(d.l):
        m.addCons(s_expr(i) >= 0)

    row = d.l
    for dim in d.soc_dims:
        t_e = s_expr(row)
        m.addCons(t_e >= 0)
        u_exprs = [s_expr(row + k) for k in range(1, dim)]
        if u_exprs:
            m.addCons(scip.quicksum(u * u for u in u_exprs) <= t_e * t_e)
        row += dim

    return m, xs


def status_name(m) -> str:
    return m.getStatus()


def solve_mps_family(path: str, timeout: float):
    m = scip.Model()
    m.hideOutput()
    m.setParam("limits/time", float(timeout))
    try:
        m.readProblem(path, extension="mps")
    except Exception:
        return "ParseError", None, 0.0
    start = time.time()
    m.optimize()
    elapsed = time.time() - start
    status = status_name(m)
    obj = m.getObjVal() if m.getNSols() > 0 else None
    return status, obj, elapsed


def solve_qplib(path: str, timeout: float):
    dump_path = dump_problem(path)
    try:
        d = read_qp_dump(dump_path)
    finally:
        os.unlink(dump_path)
    m, has_nonlinear = build_scip_from_qp_dump(d)
    m.setParam("limits/time", float(timeout))
    if has_nonlinear:
        # Works around an `Illegal instruction` crash in SCIP's presolve
        # heuristics for nonlinear constraints on this machine's AVX-less
        # virtual CPU (see module docstring).
        m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    start = time.time()
    m.optimize()
    elapsed = time.time() - start
    status = status_name(m)
    obj = m.getObjVal() if m.getNSols() > 0 else None
    return status, obj, elapsed


def solve_cbf(path: str, timeout: float):
    dump_path = dump_problem(path)
    try:
        d = read_conic_dump(dump_path)
    finally:
        os.unlink(dump_path)
    m, _xs = build_scip_from_conic_dump(d)
    m.setParam("limits/time", float(timeout))
    m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    start = time.time()
    m.optimize()
    elapsed = time.time() - start
    status = status_name(m)
    obj = None
    if m.getNSols() > 0:
        raw = m.getObjVal()
        signed = -raw if d.maximize else raw
        obj = signed + d.obj_offset
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
            status, obj, elapsed = solve_cbf(args.path, args.timeout)
    except Exception as e:  # noqa: BLE001 - surface as a CSV row, not a crash
        status, obj, elapsed = f"Error({csv_escape(str(e))})", None, 0.0

    obj_str = f"{obj:.10e}" if obj is not None else ""
    print(f"{name},{status},{obj_str},{elapsed:.3f}")


if __name__ == "__main__":
    main()
