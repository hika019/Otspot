//! Per-step LP presolve sentinels. Each step submodule (`bounds`, `doubleton`,
//! `free`, `substitution`) is exercised directly with multiple data patterns so
//! a no-op rewrite of a step body produces a visible FAIL.

use super::bounds::step5_bounds_tightening;
use super::doubleton::step6_doubleton_equation;
use super::free::{step7_free_var_substitution, step8_free_singleton_col};
use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use super::substitution::fill_in_exceeds_budget;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;

fn make_state(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
    cts: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
) -> PresolveState {
    let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
    let lp = LpProblem::new_general(c, a, b, cts, bounds, None).unwrap();
    PresolveState::from_problem(&lp)
}

fn count_bounds_tightened(st: &PresolveState) -> usize {
    st.postsolve_stack
        .iter()
        .filter(|s| matches!(s, PostsolveStep::BoundsTightened { .. }))
        .count()
}

fn count_linear_subst(st: &PresolveState) -> usize {
    st.postsolve_stack
        .iter()
        .filter(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }))
        .count()
}

// -----------------------------------------------------------
// step5_bounds_tightening
// -----------------------------------------------------------

#[test]
fn step5_le_positive_coeff_tightens_ub() {
    // x + 2y <= 4, x in [0,1], y in [0,10] → y_ub becomes (4 - 0) / 2 = 2.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 2.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 10.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed).unwrap();
    let (lb_y, ub_y) = st.bounds[1];
    assert_eq!(lb_y, 0.0);
    assert!((ub_y - 2.0).abs() < 1e-10, "y_ub should tighten to 2, got {ub_y}");
    assert!(count_bounds_tightened(&st) >= 1);
}

#[test]
fn step5_ge_positive_coeff_tightens_lb() {
    // 2x + y >= 6, x in [0,2], y in [0,5] → rest_ub_x = 5, implied_lb_x = (6-5)/2 = 0.5.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 1.0],
        1,
        2,
        vec![6.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 2.0), (0.0, 5.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed).unwrap();
    let (lb_x, _) = st.bounds[0];
    assert!((lb_x - 0.5).abs() < 1e-10, "x_lb should tighten to 0.5, got {lb_x}");
    assert!(count_bounds_tightened(&st) >= 1);
}

#[test]
fn step5_le_infeasible_negative_rhs() {
    // x + y <= -1, x in [0,5], y in [0,5] → implied_ub for x = -1 < lb 0.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![-1.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let mut fixed = 0usize;
    let res = step5_bounds_tightening(&mut st, &mut fixed);
    assert_eq!(res, Err(PresolveStatus::Infeasible));
}

#[test]
fn step5_eq_tightens_finite_ub() {
    // x + y = 3, both in [0,10] → both tightened to ub=3 in same pass.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed).unwrap();
    assert!((st.bounds[0].1 - 3.0).abs() < 1e-10);
    assert!((st.bounds[1].1 - 3.0).abs() < 1e-10);
    assert!(count_bounds_tightened(&st) >= 2);
}

// -----------------------------------------------------------
// step6_doubleton_equation
// -----------------------------------------------------------

#[test]
fn step6_basic_eliminates_one_var() {
    // x + y = 5, both in [0,10]. Equal magnitudes → first picked as pivot.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0)],
    );
    let mut subst = 0usize;
    step6_doubleton_equation(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 1);
    assert!(st.removed_cols[0]);
    assert!(st.removed_rows[0]);
    assert_eq!(count_linear_subst(&st), 1);
}

#[test]
fn step6_prefers_free_pivot() {
    // 2x + 3y = 6, x bounded, y free. Code picks y (free) even though |a_y|=3 > |a_x|.
    // Both branches "free side" select the free col; symmetric proof.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 3.0],
        1,
        2,
        vec![6.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step6_doubleton_equation(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 1);
    assert!(st.removed_cols[1], "free col y should be pivot, not bounded x");
    assert!(!st.removed_cols[0]);
}

#[test]
fn step6_infeasible_when_doubleton_impossible() {
    // x + y = 10, both in [0,3]. Implied other-bound = max(0, 10-3)=7 > 3 → Infeasible.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![10.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let mut subst = 0usize;
    let res = step6_doubleton_equation(&mut st, &mut subst);
    assert_eq!(res, Err(PresolveStatus::Infeasible));
}

// -----------------------------------------------------------
// step7_free_var_substitution
// -----------------------------------------------------------

#[test]
fn step7_eliminates_free_var_with_eq_row() {
    // x + y + z = 5 (Eq), x + y <= 10 (Le). z free; step7 picks z, eliminates Eq row.
    let mut st = make_state(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 0, 1, 1],
        &[0, 1, 2, 0, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![5.0, 10.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step7_free_var_substitution(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 1);
    assert!(st.removed_cols[2]);
    assert!(st.removed_rows[0]);
    assert_eq!(count_linear_subst(&st), 1);
}

#[test]
fn step7_skips_free_var_when_no_eq_row() {
    // z free, but the only row is Le → step7 must skip; col_entries Eq scan empty.
    let mut st = make_state(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[1.0, 1.0, 1.0],
        1,
        3,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step7_free_var_substitution(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 0, "no Eq row → free var stays");
    assert!(!st.removed_cols[2]);
}

#[test]
fn step7_picks_largest_magnitude_pivot() {
    // Two Eq rows both containing free z. Coefs 1.0 vs 3.0 → step7 picks the |3|.
    // Verifying postsolve LinearSubstitution.pivot magnitude = 3.0.
    let mut st = make_state(
        vec![0.0, 0.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 2, 1, 2],
        &[1.0, 1.0, 1.0, 3.0],
        2,
        3,
        vec![4.0, 12.0],
        vec![ConstraintType::Eq, ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step7_free_var_substitution(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 1);
    let pivot_mag = st.postsolve_stack.iter().find_map(|s| match s {
        PostsolveStep::LinearSubstitution { pivot, orig_col, .. } if *orig_col == 2 => Some(pivot.abs()),
        _ => None,
    });
    assert!(pivot_mag.is_some(), "free col z should be eliminated");
    assert!((pivot_mag.unwrap() - 3.0).abs() < 1e-10, "should pick |3| over |1|");
}

// -----------------------------------------------------------
// step8_free_singleton_col
// -----------------------------------------------------------

#[test]
fn step8_eliminates_free_singleton_eq() {
    // z appears only in row 1 (Eq), free. step8 must eliminate z + remove row 1.
    let mut st = make_state(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![3.0, 7.0],
        vec![ConstraintType::Ge, ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step8_free_singleton_col(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 1);
    assert!(st.removed_cols[2]);
    assert!(st.removed_rows[1]);
}

#[test]
fn step8_skips_free_singleton_in_le_row() {
    // z free + singleton col, but the row is Le not Eq → step8 refuses.
    let mut st = make_state(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![3.0, 7.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let mut subst = 0usize;
    step8_free_singleton_col(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 0);
    assert!(!st.removed_cols[2]);
}

#[test]
fn step8_skips_non_free_singleton() {
    // z is singleton col + Eq row, but bounded → step8 must not eliminate.
    let mut st = make_state(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![3.0, 7.0],
        vec![ConstraintType::Ge, ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0), (0.0, 5.0)],
    );
    let mut subst = 0usize;
    step8_free_singleton_col(&mut st, &mut subst).unwrap();
    assert_eq!(subst, 0);
    assert!(!st.removed_cols[2]);
}

// -----------------------------------------------------------
// substitution::fill_in_exceeds_budget
// -----------------------------------------------------------

#[test]
fn fill_in_budget_allows_dense_pivot_with_few_targets() {
    // Pivot row 0 = x+y+z=2 (Eq), only one other row touches x.
    // piv_others=2, col_j_others=1 → new_entries ≤ 2 ≤ 3*(1+2+1)=12 → allowed.
    let st = make_state(
        vec![0.0; 3],
        &[0, 0, 0, 1],
        &[0, 1, 2, 0],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![2.0, 5.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, 10.0); 3],
    );
    assert!(!fill_in_exceeds_budget(&st, 0, 0), "low fill-in must not be skipped");
}

#[test]
fn fill_in_budget_blocks_high_fill() {
    // Pivot row 0: x0+x1+x2+x3+x4+x5+x6+x7=8 (Eq, 8 cols including x0).
    // Rows 1..8: x0+x_(k+7)=k (Le, only x0 and one disjoint col each).
    // piv_others=7 (x1..x7), col_j_others=7 (rows 1..7), all disjoint → fill=49.
    // removed_nnz = 1+7+7=15, budget=45. 49 > 45 → blocked.
    let n_extra = 7usize; // x1..x7
    let n_rows_extra = 7usize; // rows 1..7
    let n_cols = 1 + n_extra + n_rows_extra;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..=n_extra {
        rows.push(0);
        cols.push(j);
        vals.push(1.0);
    }
    for r in 1..=n_rows_extra {
        rows.push(r);
        cols.push(0);
        vals.push(1.0);
        rows.push(r);
        cols.push(n_extra + r);
        vals.push(1.0);
    }
    let m = 1 + n_rows_extra;
    let mut cts = vec![ConstraintType::Eq];
    cts.extend(std::iter::repeat(ConstraintType::Le).take(n_rows_extra));
    let mut b = vec![8.0];
    b.extend((1..=n_rows_extra).map(|k| k as f64));
    let bounds = vec![(0.0, 10.0); n_cols];
    let st = make_state(
        vec![0.0; n_cols],
        &rows,
        &cols,
        &vals,
        m,
        n_cols,
        b,
        cts,
        bounds,
    );
    assert!(fill_in_exceeds_budget(&st, 0, 0), "dense disjoint fill should exceed budget");
}
