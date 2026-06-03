//! Verify that truly unbounded LPs with free variables are still detected as
//! Unbounded after the step11_dual_fixing free-variable guard (pilot-ja fix).
//!
//! All LPs are constructed in-code (no data files required) to mirror the
//! problem definitions in scripts/gen_unbounded_lp.py.

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::CscMatrix;

fn solve_presolve(lp: &LpProblem) -> SolveStatus {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(30.0);
    solve_with(lp, &opts).status
}

/// min -x1, x1 free, no constraints → x1 → +∞, obj → -∞.
///
/// Mirrors gen_lp_unbd_free_var_1d in scripts/gen_unbounded_lp.py.
#[test]
fn free_1d_still_unbounded_with_presolve() {
    let c = vec![-1.0_f64];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let ct: Vec<ConstraintType> = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("FREE1D".to_string())).unwrap();
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "FREE1D must remain Unbounded after is_free guard"
    );
}

/// x1 free, min -x1 - x2, s.t. x2 >= 1 → x1 unbounded.
///
/// Mirrors gen_lp_unbd_free_var_2d in scripts/gen_unbounded_lp.py.
#[test]
fn free_2d_still_unbounded_with_presolve() {
    let c = vec![-1.0_f64, -1.0];
    let a = CscMatrix::from_triplets(&[0usize], &[1usize], &[1.0_f64], 1, 2).unwrap();
    let b = vec![1.0_f64];
    let ct = vec![ConstraintType::Ge];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("FREE2D".to_string())).unwrap();
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "FREE2D must remain Unbounded after is_free guard"
    );
}

/// x5 free, obj = -x5, equality constraints x1+x2=1, x3+x4=1 → x5 unbounded.
///
/// Mirrors gen_lp_unbd_eq_only_free_n5 in scripts/gen_unbounded_lp.py.
#[test]
fn eq_free_n5_still_unbounded_with_presolve() {
    let c = vec![0.0_f64, 0.0, 0.0, 0.0, -1.0];
    let rows = [0usize, 0, 1, 1];
    let cols = [0usize, 1, 2, 3];
    let vals = [1.0_f64, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 5).unwrap();
    let b = vec![1.0_f64, 1.0];
    let ct = vec![ConstraintType::Eq, ConstraintType::Eq];
    let bounds = vec![
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
    ];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("EQ_FREE_N5".to_string())).unwrap();
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "EQ_FREE_N5 must remain Unbounded after is_free guard"
    );
}

/// Multiple free variables; obj = -x1 - x3, equality constraints x1+x2=0,
/// x3+x4=0 → unbounded ray d=(1,-1,1,-1,0,0,0,0).
///
/// Mirrors gen_lp_unbd_multi_free_n8 in scripts/gen_unbounded_lp.py.
#[test]
fn multifree_n8_still_unbounded_with_presolve() {
    let c = vec![-1.0_f64, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let rows = [0usize, 0, 1, 1];
    let cols = [0usize, 1, 2, 3];
    let vals = [1.0_f64, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 8).unwrap();
    let b = vec![0.0_f64, 0.0];
    let ct = vec![ConstraintType::Eq, ConstraintType::Eq];
    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
    ];
    let lp =
        LpProblem::new_general(c, a, b, ct, bounds, Some("MULTIFREE_N8".to_string())).unwrap();
    assert_eq!(
        solve_presolve(&lp),
        SolveStatus::Unbounded,
        "MULTIFREE_N8 must remain Unbounded after is_free guard"
    );
}
