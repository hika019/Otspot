//! Per-step QP presolve sentinels. Each step in the split modules
//! (`steps_bounds`, `steps_free`, `steps_parallel`) is driven directly via the
//! `Workspace` so a no-op rewrite of a step body produces an observable FAIL.

use super::state::{QpPostsolveStep, QpPresolveResult, QpPresolveStatus, Workspace};
use super::steps_basic::step4_empty;
use super::steps_bounds::{step10_implied_bounds, step11_dual_fixing};
use super::steps_free::step7_free_var;
use super::steps_parallel::step8_parallel_row;
use super::helpers::early_infeasibility_check;
use crate::options::SolverOptions;
use crate::presolve::qp_transforms::run_qp_presolve_phase1;
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

// -----------------------------------------------------------
// step4_empty: empty-col Unbounded vs Feasible
//
// Covers the matrix:
//   c sign × (lb finite, ub finite) → val=lb / val=ub / val=0
//   c sign × (lb=-inf)              → Unbounded (positive c only)
//   c sign × (ub=+inf)              → Unbounded (negative c only)
// plus the corner where Q is non-zero (empty-col path must be skipped).
// -----------------------------------------------------------

/// 2-var, both empty cols, both bounded → step4_empty fixes each to its lb;
/// must NOT report Unbounded.
#[test]
fn step4_empty_bounded_columns_positive_cost_fix_to_lb() {
    let prob = make_qp(
        &[], &[], &[],
        2,
        vec![1.0, 1.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(0.0, 1.0), (-2.0, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_ok(), "bounded empty cols with c>0 → fix to lb, never Unbounded");
    assert!(ws.removed_cols[0] && ws.removed_cols[1]);
    let pushed: Vec<_> = ws.postsolve_stack.steps.iter().filter(|s| matches!(s, QpPostsolveStep::EmptyCol { .. })).collect();
    assert_eq!(pushed.len(), 2, "two EmptyCol steps pushed");
    // obj_offset = 1*0 + 1*(-2) = -2
    assert!((ws.obj_offset - (-2.0)).abs() < 1e-12, "obj_offset = c·lb = -2, got {}", ws.obj_offset);
}

/// Negative cost, finite ub → fix to ub.
#[test]
fn step4_empty_bounded_columns_negative_cost_fix_to_ub() {
    let prob = make_qp(
        &[], &[], &[],
        1,
        vec![-3.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(-10.0, 2.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_ok());
    assert!(ws.removed_cols[0]);
    let val_ok = matches!(
        ws.postsolve_stack.steps.last(),
        Some(QpPostsolveStep::EmptyCol { idx: 0, val }) if (*val - 2.0).abs() < 1e-12
    );
    assert!(val_ok, "val must be ub=2, got {:?}", ws.postsolve_stack.steps.last());
    assert!((ws.obj_offset - (-6.0)).abs() < 1e-12);
}

/// Zero cost, both bounds finite → val=lb (deterministic fallback).
#[test]
fn step4_empty_zero_cost_finite_bounds_uses_lb() {
    let prob = make_qp(
        &[], &[], &[],
        1,
        vec![0.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(3.0, 7.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_ok());
    assert!(ws.removed_cols[0]);
    let val_ok = matches!(
        ws.postsolve_stack.steps.last(),
        Some(QpPostsolveStep::EmptyCol { idx: 0, val }) if (*val - 3.0).abs() < 1e-12
    );
    assert!(val_ok);
}

/// Positive cost, lb = -∞ → genuine Unbounded.
#[test]
fn step4_empty_positive_cost_no_lower_bound_is_unbounded() {
    let prob = make_qp(
        &[], &[], &[],
        1,
        vec![1.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(f64::NEG_INFINITY, 5.0)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_err(), "positive c with lb=-∞ is genuinely Unbounded");
    if let Err(r) = res {
        assert!(matches!(r.presolve_status, QpPresolveStatus::Unbounded));
    }
}

/// Negative cost, ub = +∞ → genuine Unbounded.
#[test]
fn step4_empty_negative_cost_no_upper_bound_is_unbounded() {
    let prob = make_qp(
        &[], &[], &[],
        1,
        vec![-2.5],
        &[], &[], &[],
        0,
        vec![],
        vec![(0.0, f64::INFINITY)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_err());
    if let Err(r) = res {
        assert!(matches!(r.presolve_status, QpPresolveStatus::Unbounded));
    }
}

/// Q is non-zero on this column → step4 must skip the empty-col branch
/// entirely, even if A is empty and the only finite bound would otherwise
/// have triggered Unbounded reporting in a buggy implementation.
#[test]
fn step4_empty_skips_when_q_nonzero_even_with_infinite_lb() {
    // Q[0,0]=2, c=+1, lb=-∞.  Quadratic term keeps it bounded;
    // step4_empty must NOT enter the empty-col Unbounded check.
    let prob = make_qp(
        &[0], &[0], &[2.0],
        1,
        vec![1.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
        vec![],
    );
    let mut ws = Workspace::from_problem(&prob);
    let res = step4_empty(&prob, &mut ws);
    assert!(res.is_ok(), "Q-coupled col must NOT trip empty-col Unbounded");
    assert!(!ws.removed_cols[0], "Q-coupled col must remain");
}

/// run_qp_presolve_phase1 end-to-end: bounded empty cols with non-zero c must
/// converge to Feasible status, not Unbounded.  Table-driven over signs.
#[test]
fn phase1_bounded_empty_cols_are_feasible_not_unbounded() {
    // (cj, lb, ub, expected val pushed)
    let cases: [(f64, f64, f64, f64); 4] = [
        ( 1.0, 0.0,  1.0, 0.0),  // c>0, both finite → val=lb
        (-1.0, 0.0,  1.0, 1.0),  // c<0, both finite → val=ub
        ( 0.0, 2.0,  3.0, 2.0),  // c=0, both finite → val=lb
        ( 1.0, 4.0, f64::INFINITY, 4.0), // c>0, ub=+∞ but lb finite → val=lb (NOT unbounded)
    ];
    for (cj, lb, ub, expected_val) in cases {
        let prob = make_qp(
            &[], &[], &[],
            1,
            vec![cj],
            &[], &[], &[],
            0,
            vec![],
            vec![(lb, ub)],
            vec![],
        );
        let result = run_qp_presolve_phase1(&prob, &SolverOptions::default());
        assert!(
            !matches!(result.presolve_status, QpPresolveStatus::Unbounded),
            "case cj={cj} lb={lb} ub={ub} must be Feasible, got Unbounded"
        );
        let expected_offset = cj * expected_val;
        assert!(
            (result.obj_offset - expected_offset).abs() < 1e-12,
            "case cj={cj} lb={lb} ub={ub}: obj_offset={} expected={}",
            result.obj_offset, expected_offset
        );
    }
}

/// run_qp_presolve_phase1 end-to-end: genuinely unbounded inputs MUST still be
/// reported as Unbounded (sentinel against over-correction).
#[test]
fn phase1_truly_unbounded_empty_cols_still_detected() {
    // Free var (-∞,+∞) with c=+1 is the canonical unbounded case for an empty col.
    let prob = make_qp(
        &[], &[], &[],
        1,
        vec![1.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
        vec![],
    );
    let result = run_qp_presolve_phase1(&prob, &SolverOptions::default());
    assert!(
        matches!(result.presolve_status, QpPresolveStatus::Unbounded),
        "free col with c=+1 must be Unbounded"
    );
}

/// `early_infeasibility_check` reports Unbounded only when all bounds are
/// infinite AND every Q diagonal is strictly negative.  Bounded vars must
/// never trigger it, even with negative Q diag.
#[test]
fn early_check_does_not_flag_bounded_empty_cols() {
    // Q diag = -1 on a bounded var; m=0, but lb/ub finite.
    let prob = make_qp(
        &[0], &[0], &[-1.0],
        1,
        vec![0.0],
        &[], &[], &[],
        0,
        vec![],
        vec![(0.0, 1.0)],
        vec![],
    );
    let status = early_infeasibility_check(&prob);
    assert!(status.is_none(), "bounded var must not trigger early Unbounded, got {:?}", status);
}
