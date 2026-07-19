#!/usr/bin/env python3
"""Unit tests for compare.py's parsers and frontier ("other_solver_wins")
logic. Runs standalone (`python3 test_compare.py`) or under pytest.

Fixtures are synthetic but mirror the exact producers:
  - bench_parallel.sh detail lines and its EXTERNAL_TIMEOUT fallback entry
    (2-space indent + basename with extension, bench_parallel.sh:290),
  - solve_cbf.rs CSV with SolveStatus Display vocabulary and the repeated
    header lines observed in bench_results/cblib_20260707_postfix.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from compare import (
    build_rows,
    parse_bench_parallel_txt,
    parse_solve_cbf_csv,
    parse_solver_csv,
)

BENCH_PARALLEL_FIXTURE = """\
=== bench_parallel.sh 集計結果 ===
data-dir         : /x/data/qplib
timeout          : 1000s

=== Summary ===
  PASS:              1
  EXTERNAL_TIMEOUT:  1
  TOTAL:   3

=== 問題別詳細 ===
QPLIB_0018                   50      1         STALLED      0.648 [ipm] iters=139 obj=-1.259197e1 x_inf=3.14e-1
QPLIB_8495                27543   8000            PASS      4.420 [ipm] obj=4.285750e4 obj_err=0.000%
  QPLIB_9999.qplib  EXTERNAL_TIMEOUT (external_timeout=6300s, solver internal timeout 未機能)
    EXTERNAL_TIMEOUT: 1
    TOTAL:   1
  QPLIB_7777.qplib  ERROR worker_exit=134
"""

SOLVE_CBF_FIXTURE = """\
problem,status,objective,iterations,time_sec
nql30,Optimal,-9.4604999300e-1,25,12.500
100_0_1_w,NumericalError,inf,1,12.550
qssp30,Infeasible,,3,1.200
problem,status,objective,iterations,time_sec
syn10m,Unbounded,,2,0.900
"""

# bench_qplib.rs (NONCONVEX_LOCAL/NONCONVEX_GLOBAL, bench_qplib.rs:571/581)
# and qps_benchmark.rs (NOT_SUPPORTED, qps_benchmark.rs:952) detail lines,
# same `{name:<24} {n:>6} {m:>6} {status:>15} {time:>10.3} {note}` layout as
# BENCH_PARALLEL_FIXTURE.
NONCONVEX_AND_UNSUPPORTED_FIXTURE = """\
=== 問題別詳細 ===
QPLIB_1111                   10      2 NONCONVEX_LOCAL      0.100 [ipm] obj=1.000000e0 kkt=1.0e-7 gap=1.0e-4
QPLIB_2222                   10      2 NONCONVEX_GLOBAL      0.200 [bb] obj=2.000000e0 kkt=1.0e-7 gap=0.0e0
QPLIB_3333                   10      2   NOT_SUPPORTED      0.050 [ipm] unsupported cone type
"""


def _write(content: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _solver_csv(rows) -> str:
    body = "problem,status,objective,time_sec\n" + "".join(
        f"{n},{s},{o},{t}\n" for n, s, o, t in rows
    )
    return _write(body, ".csv")


class TestBenchParallelParser(unittest.TestCase):
    def setUp(self):
        self.path = _write(BENCH_PARALLEL_FIXTURE, ".txt")
        self.parsed = parse_bench_parallel_txt(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_detail_lines_parsed(self):
        self.assertEqual(self.parsed["QPLIB_0018"]["status"], "STALLED")
        self.assertEqual(self.parsed["QPLIB_8495"]["status"], "PASS")
        self.assertAlmostEqual(self.parsed["QPLIB_8495"]["time"], 4.420)

    def test_external_timeout_fallback_line_parsed(self):
        """bench_parallel.sh:290 fallback entry: 2-space indent, basename
        with extension, no n/m/time columns."""
        self.assertIn("QPLIB_9999", self.parsed)
        self.assertEqual(self.parsed["QPLIB_9999"]["status"], "EXTERNAL_TIMEOUT")

    def test_error_fallback_line_parsed(self):
        self.assertIn("QPLIB_7777", self.parsed)
        self.assertEqual(self.parsed["QPLIB_7777"]["status"], "ERROR")

    def test_summary_counter_line_not_a_problem(self):
        """`    EXTERNAL_TIMEOUT: 1` is a Summary counter, not a problem."""
        self.assertNotIn("EXTERNAL_TIMEOUT", self.parsed)
        self.assertNotIn("EXTERNAL_TIMEOUT:", self.parsed)


class TestNonconvexAndUnsupportedStatuses(unittest.TestCase):
    """P2 sentinel: NONCONVEX_LOCAL/NONCONVEX_GLOBAL (bench_qplib.rs) and
    NOT_SUPPORTED (qps_benchmark.rs) detail lines must not be dropped by
    KNOWN_STATUSES, or the problem silently disappears from otspot_pass
    triage (an external-solver win goes undetected). Reverting the
    KNOWN_STATUSES additions makes these lines fail the whitelist check and
    the problems vanish from the parsed map."""

    def setUp(self):
        self.path = _write(NONCONVEX_AND_UNSUPPORTED_FIXTURE, ".txt")
        self.parsed = parse_bench_parallel_txt(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_nonconvex_local_not_dropped(self):
        self.assertIn("QPLIB_1111", self.parsed)
        self.assertEqual(self.parsed["QPLIB_1111"]["status"], "NONCONVEX_LOCAL")

    def test_nonconvex_global_not_dropped(self):
        self.assertIn("QPLIB_2222", self.parsed)
        self.assertEqual(self.parsed["QPLIB_2222"]["status"], "NONCONVEX_GLOBAL")

    def test_not_supported_not_dropped(self):
        self.assertIn("QPLIB_3333", self.parsed)
        self.assertEqual(self.parsed["QPLIB_3333"]["status"], "NOT_SUPPORTED")

    def test_external_optimal_over_these_statuses_is_a_win(self):
        """End-to-end sentinel for the actual harm Codex flagged: build_rows
        computes `other_solver_wins = (not o_pass) and (h_wins or s_wins)
        and (name in otspot)` (compare.py). Before the whitelist fix these
        detail lines were dropped from the parsed map, `name in otspot` was
        False, and other_solver_wins was forced False — hiding every
        externally-solvable NONCONVEX_LOCAL / NONCONVEX_GLOBAL /
        NOT_SUPPORTED problem from triage. Removing the three statuses from
        KNOWN_STATUSES flips these asserts back to False."""
        highs_path = _solver_csv([
            ("QPLIB_1111", "optimal", "1.0e0", "3.0"),
            ("QPLIB_2222", "optimal", "2.0e0", "3.0"),
            ("QPLIB_3333", "optimal", "3.0e0", "3.0"),
        ])
        try:
            highs = parse_solver_csv(highs_path)
        finally:
            os.unlink(highs_path)
        rows = {r["problem"]: r for r in build_rows(self.parsed, highs, {})}
        for name in ("QPLIB_1111", "QPLIB_2222", "QPLIB_3333"):
            self.assertTrue(
                rows[name]["other_solver_wins"],
                f"{name}: externally solved, otspot non-PASS -> must be a win",
            )


class TestSolveCbfParser(unittest.TestCase):
    def setUp(self):
        self.path = _write(SOLVE_CBF_FIXTURE, ".csv")
        self.parsed = parse_solve_cbf_csv(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_statuses_and_repeated_header_skipped(self):
        self.assertEqual(len(self.parsed), 4)
        self.assertNotIn("problem", self.parsed)
        self.assertEqual(self.parsed["nql30"]["status"], "Optimal")
        self.assertEqual(self.parsed["100_0_1_w"]["status"], "NumericalError")
        self.assertEqual(self.parsed["nql30"]["format"], "solve_cbf")


class TestFrontierLogic(unittest.TestCase):
    def _rows(self, otspot, scip_rows):
        scip_path = _solver_csv(scip_rows)
        try:
            scip = parse_solver_csv(scip_path)
        finally:
            os.unlink(scip_path)
        return {r["problem"]: r for r in build_rows(otspot, {}, scip)}

    def test_external_timeout_vs_scip_optimal_is_a_win(self):
        """P0-2 sentinel: the most important case — Otspot's worker was
        killed externally (EXTERNAL_TIMEOUT fallback line) while SCIP solved
        the problem. Reverting the indented-fallback parsing drops the
        problem from the otspot map and other_solver_wins becomes False."""
        txt = _write(BENCH_PARALLEL_FIXTURE, ".txt")
        try:
            otspot = parse_bench_parallel_txt(txt)
        finally:
            os.unlink(txt)
        rows = self._rows(otspot, [("QPLIB_9999", "optimal", "-1.0e0", "5.0")])
        self.assertTrue(rows["QPLIB_9999"]["other_solver_wins"])

    def test_bench_parallel_pass_is_not_a_win(self):
        txt = _write(BENCH_PARALLEL_FIXTURE, ".txt")
        try:
            otspot = parse_bench_parallel_txt(txt)
        finally:
            os.unlink(txt)
        rows = self._rows(otspot, [("QPLIB_8495", "optimal", "4.28575e4", "2.0")])
        self.assertFalse(rows["QPLIB_8495"]["other_solver_wins"])

    def _cbf_otspot(self):
        path = _write(SOLVE_CBF_FIXTURE, ".csv")
        try:
            return parse_solve_cbf_csv(path)
        finally:
            os.unlink(path)

    def test_cbf_optimal_is_pass_not_a_win(self):
        """P0-1 sentinel: solve_cbf's `Optimal` must count as Otspot PASS.
        Reverting to the bench_parallel-only PASS vocabulary makes every cbf
        row non-PASS and this row becomes a spurious win."""
        rows = self._rows(self._cbf_otspot(), [("nql30", "optimal", "-0.946", "3.0")])
        self.assertFalse(rows["nql30"]["other_solver_wins"])
        self.assertFalse(rows["nql30"]["otspot_unverified_claim"])

    def test_cbf_numerical_error_vs_scip_optimal_is_a_win(self):
        rows = self._rows(self._cbf_otspot(), [("100_0_1_w", "optimal", "1.5e2", "8.0")])
        self.assertTrue(rows["100_0_1_w"]["other_solver_wins"])

    def test_cbf_infeasible_agreement_is_not_a_win(self):
        """Both claim infeasible: agreement, not a win — but the Otspot side
        is flagged as an unverified claim."""
        rows = self._rows(self._cbf_otspot(), [("qssp30", "infeasible", "", "1.0")])
        self.assertFalse(rows["qssp30"]["other_solver_wins"])
        self.assertTrue(rows["qssp30"]["otspot_unverified_claim"])

    def test_cbf_infeasible_vs_scip_optimal_is_a_win(self):
        """Conflicting conclusions: SCIP found an optimum where Otspot
        claimed (unverified) infeasibility — flag as a win for triage."""
        rows = self._rows(self._cbf_otspot(), [("qssp30", "optimal", "2.0e0", "1.0")])
        self.assertTrue(rows["qssp30"]["other_solver_wins"])
        self.assertTrue(rows["qssp30"]["otspot_unverified_claim"])

    def test_problem_only_in_solver_csv_is_not_a_win(self):
        """A problem missing from the Otspot results entirely (not run) must
        not appear as a frontier entry."""
        rows = self._rows(self._cbf_otspot(), [("brand_new", "optimal", "1.0", "1.0")])
        self.assertFalse(rows["brand_new"]["other_solver_wins"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
