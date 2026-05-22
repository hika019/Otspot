//! Regression guard: `cycle.QPS` (Netlib LP canary) must converge to its known
//! optimum within `REL_TOL`. Known objective is taken from
//! `data/baseline_objectives/netlib_lp_canary.csv`.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::{solve_with, QpProblem};
use std::path::Path;
use std::time::Instant;

fn make_lp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

/// Task #26: cycle.QPS must reach the known optimum, not NumericalError.
#[test]
fn diag_cycle_must_reach_known_objective() {
    let path = Path::new("data/lp_problems/cycle.QPS");
    if !path.exists() {
        panic!(
            "data missing: {:?}. Symlink data/lp_problems/cycle.QPS into the worktree.",
            path
        );
    }
    let qp = parse_qps(path).expect("parse cycle.QPS");
    let lp = make_lp(&qp);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed_s = t0.elapsed().as_secs_f64();

    let tb = r.timing_breakdown.unwrap_or_default();
    eprintln!(
        "[cycle] elapsed={:.2}s status={:?} obj={:.10e} iters={} sol_len={}/n={}",
        elapsed_s,
        r.status,
        r.objective,
        r.iterations,
        r.solution.len(),
        lp.num_vars,
    );
    eprintln!(
        "[cycle] timing_us: presolve={} solve={} postsolve={} (total_ms={:.1})",
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
        (tb.presolve_us + tb.solve_us + tb.postsolve_us) as f64 / 1000.0,
    );

    const KNOWN_OBJ: f64 = -5.2263930248924400e+00;
    const REL_TOL: f64 = 1.0e-4;

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "[cycle] expected Optimal, got {:?} (obj={:.6e})",
        r.status,
        r.objective,
    );

    let rel_err = (r.objective - KNOWN_OBJ).abs() / KNOWN_OBJ.abs();
    assert!(
        rel_err < REL_TOL,
        "[cycle] obj={:.10e} differs from known {:.10e} by rel {:.3e} (>{:.0e})",
        r.objective,
        KNOWN_OBJ,
        rel_err,
        REL_TOL,
    );
}
