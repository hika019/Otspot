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
QPLIB_0018                   50      1      SUBOPTIMAL      0.648 [ipm] iters=139 obj=-1.259197e1 x_inf=3.14e-1
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
        self.assertEqual(self.parsed["QPLIB_0018"]["status"], "SUBOPTIMAL")
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
