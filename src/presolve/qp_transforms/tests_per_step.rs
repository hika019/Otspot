//! Per-step QP presolve sentinels. Each step in the split modules
//! (`steps_bounds`, `steps_free`, `steps_parallel`) is driven directly via the
//! `Workspace` so a no-op rewrite of a step body produces an observable FAIL.

use super::state::{QpPostsolveStep, QpPresolveResult, Workspace};
use super::steps_bounds::{step10_implied_bounds, step11_dual_fixing};
use super::steps_free::step7_free_var;
use super::steps_parallel::step8_parallel_row;
use crate::problem::ConstraintType;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

#[allow(clippy::too_many_arguments)]
fn make_qp(
    q_rows: &[usize],
    q_cols: &[usize],
    q_vals: &[f64],
    n: usize,
    c: Vec<f64>,
    a_rows: &[usize],
    a_cols: &[usize],
    a_vals: &[f64],
    m: usize,
    b: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    cts: Vec<ConstraintType>,
) -> QpProblem {
    let q = if q_vals.is_empty() {
        CscMatrix::new(n, n)
    } else {
        CscMatrix::from_triplets(q_rows, q_cols, q_vals, n, n).unwrap()
    };
    let a = if a_vals.is_empty() {
        CscMatrix::new(m, n)
    } else {
        CscMatrix::from_triplets(a_rows, a_cols, a_vals, m, n).unwrap()
    };
    QpProblem::new(q, c, a, b, bounds, cts).unwrap()
}

/// QpPresolveResult doesn't implement Debug, so .unwrap() can't be used. Wrap.
fn expect_ok(r: Result<(), QpPresolveResult>, msg: &str) {
    assert!(r.is_ok(), "{msg}");
}

// -----------------------------------------------------------
// step10_implied_bounds
// -----------------------------------------------------------

#[test]
fn step10_le_infeasible_when_implied_ub_below_lb() {
    // x + y <= -1, x in [0,5], y in [0,5] → implied_ub for x = -1 < lb 0.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![-1.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Le],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step10_implied_bounds(&prob, &mut ws, None);
    assert!(res.is_err(), "step10 must detect infeasibility");
}

#[test]
fn step10_eq_infeasible_when_required_value_outside_bounds() {
    // x + y = 100, x,y in [0,5] → both directions force tight bounds outside [0,5].
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![100.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Eq],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step10_implied_bounds(&prob, &mut ws, None);
    assert!(res.is_err(), "Eq blowup must be flagged");
}

#[test]
fn step10_feasible_problem_returns_ok() {
    // x + y <= 10, x,y in [0,5] → comfortably feasible.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![10.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Le],
    );
    let mut ws = Workspace::from_problem(&prob);
    assert!(step10_implied_bounds(&prob, &mut ws, None).is_ok());
}

// -----------------------------------------------------------
// step11_dual_fixing
// -----------------------------------------------------------

#[test]
fn step11_positive_cost_fixes_to_lb() {
    // c=+1, lb=2, ub=5, no Q/A nz → minimize pushes x to lb=2.
    let prob = make_qp(
        &[],
        &[],
        &[],
        1,
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![(2.0, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step11_dual_fixing(&prob, &mut ws), "step11 ok");
    assert!(ws.removed_cols[0]);
    assert_eq!(ws.bounds[0], (2.0, 2.0));
    let pushed = matches!(
        ws.postsolve_stack.steps.last(),
        Some(QpPostsolveStep::EmptyCol { idx: 0, val }) if (*val - 2.0).abs() < 1e-12
    );
    assert!(pushed, "EmptyCol with val=2.0 must be pushed");
}

#[test]
fn step11_negative_cost_fixes_to_ub() {
    // c=-1, lb=2, ub=5 → minimize pushes x to ub=5.
    let prob = make_qp(
        &[],
        &[],
        &[],
        1,
        vec![-1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![(2.0, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step11_dual_fixing(&prob, &mut ws), "step11 ok");
    assert!(ws.removed_cols[0]);
    assert_eq!(ws.bounds[0], (5.0, 5.0));
}

#[test]
fn step11_zero_cost_does_not_fix() {
    let prob = make_qp(
        &[],
        &[],
        &[],
        1,
        vec![0.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![(2.0, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step11_dual_fixing(&prob, &mut ws), "step11 ok");
    assert!(!ws.removed_cols[0]);
}

#[test]
fn step11_skips_when_q_nonzero() {
    // c=+1 but Q has a nonzero diagonal → step11 must not fix (column is quadratic-coupled).
    let prob = make_qp(
        &[0],
        &[0],
        &[2.0],
        1,
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![(2.0, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step11_dual_fixing(&prob, &mut ws), "step11 ok");
    assert!(!ws.removed_cols[0]);
}

// -----------------------------------------------------------
// step7_free_var (QP)
// -----------------------------------------------------------

#[test]
fn step7_qp_eliminates_free_var_via_singleton_eq() {
    // Eq singleton row 0: 2·z = 6 → val=3. z free, no Q nz on z.
    // Le row 1: x + z <= 10 (still uses z, but step7 only requires singleton Eq, not unique row).
    // Reusing same z across rows is fine; step7 still picks the singleton-Eq row to fix z=3.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 1, 1],
        &[1, 0, 1],
        &[2.0, 1.0, 1.0],
        2,
        vec![6.0, 10.0],
        vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Eq, ConstraintType::Le],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step7_free_var(&prob, &mut ws, None), "step7 ok");
    assert!(ws.removed_cols[1], "free col z should be eliminated");
    assert!(ws.removed_rows[0], "singleton Eq row consumed");
    let pushed = matches!(
        ws.postsolve_stack.steps.last(),
        Some(QpPostsolveStep::SingletonRow { row: 0, col: 1, val }) if (*val - 3.0).abs() < 1e-10
    );
    assert!(pushed, "SingletonRow(row=0,col=1,val=3) expected");
}

#[test]
fn step7_qp_skips_free_var_with_nonzero_q() {
    // z free, Q[z,z]=2 → step7 declines (quadratic coupling).
    let prob = make_qp(
        &[1],
        &[1],
        &[2.0],
        2,
        vec![0.0, 0.0],
        &[0],
        &[1],
        &[2.0],
        1,
        vec![6.0],
        vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Eq],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step7_free_var(&prob, &mut ws, None), "step7 ok");
    assert!(!ws.removed_cols[1], "Q-coupled col must not be eliminated");
}

#[test]
fn step7_qp_skips_when_no_singleton_eq_row() {
    // z free but every Eq row has 2 active vars → no eligible singleton.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![5.0],
        vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Eq],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step7_free_var(&prob, &mut ws, None), "step7 ok");
    assert!(!ws.removed_cols[1]);
}

// -----------------------------------------------------------
// step8_parallel_row
// -----------------------------------------------------------

#[test]
fn step8_parallel_le_drops_looser_row() {
    // x+y<=5 and 2x+2y<=8 (alpha=2). eff_b2 = 4 < b[0]=5 → row 0 (looser) removed.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 2.0, 2.0],
        2,
        vec![5.0, 8.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Le, ConstraintType::Le],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step8_parallel_row(&prob, &mut ws, None), "step8 ok");
    let removed: Vec<usize> = (0..2).filter(|&i| ws.removed_rows[i]).collect();
    assert_eq!(removed.len(), 1, "exactly one parallel row removed");
    assert_eq!(removed[0], 0, "the looser Le row (#0 with eff_b=5 > 4) is dropped");
}

#[test]
fn step8_parallel_eq_same_b_removes_second() {
    // x+y=3 and 2x+2y=6 (consistent) → second row removed.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 2.0, 2.0],
        2,
        vec![3.0, 6.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Eq, ConstraintType::Eq],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step8_parallel_row(&prob, &mut ws, None), "step8 ok");
    assert!(ws.removed_rows[0] ^ ws.removed_rows[1], "exactly one row removed");
}

#[test]
fn step8_parallel_eq_inconsistent_is_infeasible() {
    // x+y=3 and 2x+2y=7 → eff_b2=3.5 ≠ 3 → Infeasible.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 2.0, 2.0],
        2,
        vec![3.0, 7.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Eq, ConstraintType::Eq],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step8_parallel_row(&prob, &mut ws, None);
    assert!(res.is_err(), "inconsistent parallel Eq pair must be Infeasible");
}

#[test]
fn step8_non_parallel_rows_untouched() {
    // x+y<=5 and x+2y<=5 — same first-col bucket but not parallel.
    let prob = make_qp(
        &[],
        &[],
        &[],
        2,
        vec![0.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 2.0],
        2,
        vec![5.0, 5.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Le, ConstraintType::Le],
    );
    let mut ws = Workspace::from_problem(&prob);
    expect_ok(step8_parallel_row(&prob, &mut ws, None), "step8 ok");
    assert!(!ws.removed_rows[0] && !ws.removed_rows[1]);
}
