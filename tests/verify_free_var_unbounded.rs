//! Verify that truly unbounded LPs with free variables are still detected as
//! Unbounded after the step11_dual_fixing free-variable guard (pilot-ja fix).

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::solve_with;
use std::path::Path;

fn load_lp(filename: &str) -> LpProblem {
    let path = Path::new("data/lp_problems_unbounded").join(filename);
    assert!(path.exists(), "{} not found", path.display());
    let qp = parse_qps(&path).unwrap_or_else(|e| panic!("parse {filename} failed: {e}"));
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        Some(filename.to_string()),
    )
    .unwrap_or_else(|e| panic!("LP construction for {filename} failed: {e}"))
}

fn solve_presolve(lp: &LpProblem) -> SolveStatus {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(30.0);
    solve_with(lp, &opts).status
}

/// min -x1, x1 free, no constraints → x1 → +∞, obj → -∞.
#[test]
fn free_1d_still_unbounded_with_presolve() {
    let lp = load_lp("UNBD_LP_FREE1D.QPS");
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "UNBD_LP_FREE1D must remain Unbounded after is_free guard"
    );
}

/// x1 free, min -x1 - x2, s.t. x2 ≥ 1 → x1 unbounded.
#[test]
fn free_2d_still_unbounded_with_presolve() {
    let lp = load_lp("UNBD_LP_FREE2D.QPS");
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "UNBD_LP_FREE2D must remain Unbounded after is_free guard"
    );
}

/// x5 free, obj = -x5, equality constraints on x1..x4 → x5 unbounded.
#[test]
fn eq_free_n5_still_unbounded_with_presolve() {
    let lp = load_lp("UNBD_LP_EQ_FREE_N5.QPS");
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "UNBD_LP_EQ_FREE_N5 must remain Unbounded after is_free guard"
    );
}

/// Multiple free variables; obj = -x1 - x3, equality constraints → unbounded.
#[test]
fn multifree_n8_still_unbounded_with_presolve() {
    let lp = load_lp("UNBD_LP_MULTIFREE_N8.QPS");
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "UNBD_LP_MULTIFREE_N8 must remain Unbounded after is_free guard"
    );
}
