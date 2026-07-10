//! Per-step LP presolve sentinels. Each step submodule (`bounds`, `doubleton`,
//! `free`, `substitution`) is exercised directly with multiple data patterns so
//! a no-op rewrite of a step body produces a visible FAIL.

use super::bounds::step5_bounds_tightening;
use super::doubleton::step6_doubleton_equation;
use super::empty_redundant::{step3a_empty_row, step3b_empty_column, step4_redundant_constraint};
use super::fixed::step1_fixed_variable;
use super::free::{step7_free_var_substitution, step8_free_singleton_col};
use super::singleton::step2_singleton_row;
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
        .filter(|s| matches!(s, PostsolveStep::BoundsTightened))
        .count()
}

fn count_linear_subst(st: &PresolveState) -> usize {
    st.postsolve_stack
        .iter()
        .filter(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }))
        .count()
}

// -----------------------------------------------------------
// PresolveState sparse-entry update hot path
// -----------------------------------------------------------

#[test]
fn add_to_entry_updates_prunes_and_keeps_row_col_in_sync() {
    let mut st = make_state(
        vec![0.0; 3],
        &[0, 0, 1],
        &[0, 1, 1],
        &[1.0, 2.0, 3.0],
        2,
        3,
        vec![0.0; 2],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, 10.0); 3],
    );

    st.add_to_entry(0, 1, 0.5);
    assert_eq!(st.coeff(0, 1), 2.5);
    assert!(st.row_entries[0].contains(&(1, 2.5)));
    assert!(st.col_entries[1].contains(&(0, 2.5)));

    st.add_to_entry(1, 2, -4.0);
    assert_eq!(st.coeff(1, 2), -4.0);
    assert!(st.row_entries[1].contains(&(2, -4.0)));
    assert!(st.col_entries[2].contains(&(1, -4.0)));

    st.add_to_entry(0, 1, -2.5);
    assert_eq!(st.coeff(0, 1), 0.0);
    assert!(!st.row_entries[0].iter().any(|&(j, _)| j == 1));
    assert!(!st.col_entries[1].iter().any(|&(i, _)| i == 0));

    st.add_to_entry(1, 2, crate::tolerances::ZERO_TOL * 0.5);
    assert_eq!(st.coeff(1, 2), -4.0);
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
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    let (lb_y, ub_y) = st.bounds[1];
    assert_eq!(lb_y, 0.0);
    assert!(
        (ub_y - 2.0).abs() < 1e-10,
        "y_ub should tighten to 2, got {ub_y}"
    );
    assert!(count_bounds_tightened(&st) >= 1);
}

/// `revert_redundant_added_bounds` must drop a presolve-added implied ub that a
/// retained row already forces, while preserving genuine original model bounds.
///
/// x + y <= 5 with x in [0, +inf) and y in [0, 2]. step5 derives an implied
/// ub = 5 for the originally-unbounded x; that bound is redundant (the retained
/// Le row enforces it), so the reversion must restore x's ub to +inf. y's ub is
/// an original model bound and must survive.
///
/// No-op proof: deleting the reversion leaves x's ub = 5, which the simplex
/// standard form would materialize as an explicit UB row — the osa-60 row-blowup
/// this guards against.
#[test]
fn revert_redundant_added_bounds_restores_infinite_ub_but_keeps_original() {
    let mut st = make_state(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, 2.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    assert!(
        st.bounds[0].1.is_finite(),
        "precondition: step5 must add a finite implied ub to the unbounded x, got {}",
        st.bounds[0].1
    );

    super::bounds::revert_redundant_added_bounds(&mut st);

    assert_eq!(
        st.bounds[0].1,
        f64::INFINITY,
        "redundant implied ub on originally-unbounded x must revert to +inf"
    );
    assert_eq!(st.bounds[0].0, 0.0, "x lower bound must be untouched");
    assert_eq!(
        st.bounds[1],
        (0.0, 2.0),
        "original model ub on y must be preserved, not reverted"
    );
}

/// Companion lb-side branch: a presolve-added implied *lower* bound on an
/// originally `-inf` variable that a retained row forces must revert to `-inf`,
/// while an original finite lb survives.
///
/// x + y >= 3 with x in (-inf, +inf) and y in [0, 5]. step5 derives implied
/// lb_x = (3 - y_ub)/1 = -2 (independent oracle). The retained Ge row enforces
/// x >= 3 - y >= 3 - 5 = -2, so that bound is redundant and reverts to -inf.
///
/// No-op proof: deleting the lb branch in `revert_redundant_added_bounds` leaves
/// x's lb at -2, failing the `-inf` assertion.
#[test]
fn revert_redundant_added_bounds_restores_infinite_lb_but_keeps_original() {
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, 5.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    assert!(
        (st.bounds[0].0 - (-2.0)).abs() < 1e-10,
        "precondition: step5 must add implied lb = -2 to x, got {}",
        st.bounds[0].0
    );

    super::bounds::revert_redundant_added_bounds(&mut st);

    assert_eq!(
        st.bounds[0].0,
        f64::NEG_INFINITY,
        "redundant implied lb on originally-unbounded-below x must revert to -inf"
    );
    assert_eq!(
        st.bounds[0].1,
        f64::INFINITY,
        "x upper bound must be untouched"
    );
    assert_eq!(
        st.bounds[1].0, 0.0,
        "original model lb on y must be preserved"
    );
}

/// Safety-valve branch: an added upper bound that *no retained row implies* must
/// be KEPT. This is load-bearing — reverting it would enlarge the feasible region.
///
/// x + y <= 5 (row 0) and x + y >= 0 (row 1), x,y in [0, +inf). step5 derives
/// ub = 5 for x and y from row 0. Row 0 is then removed (as `step4` does when a
/// row is redundant *given* the tightened bounds), making x's ub the only thing
/// enforcing x <= 5. The remaining Ge row implies no finite ub (implied_ub = +inf
/// > 5), so the reversion must keep x's ub at 5; dropping it would let x → +inf.
///
/// No-op proof: making the ub reversion unconditional (removing the
/// `implied_ub[j] <= ub` guard) reverts x's ub to +inf, failing the assertion —
/// this is the guard that preserves the feasible region.
#[test]
fn revert_redundant_added_bounds_keeps_ub_no_retained_row_implies() {
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        2,
        vec![5.0, 0.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    assert!(
        (st.bounds[0].1 - 5.0).abs() < 1e-10,
        "precondition: step5 must add ub = 5 to x from the Le row, got {}",
        st.bounds[0].1
    );
    // Simulate step4 removing the source row; the added ub is now load-bearing.
    st.removed_rows[0] = true;

    super::bounds::revert_redundant_added_bounds(&mut st);

    assert_eq!(
        st.bounds[0].1, 5.0,
        "an added ub no retained row implies must be KEPT (reverting would unbound x)"
    );
}

/// Fixing-skip branch: a variable fixed by tightening (`lb == ub`) must never be
/// reverted, even when a retained row implies the bound.
///
/// x in [0, +inf) with the row x <= 3. x is fixed to [3, 3] (as dual-fixing /
/// step1 would). The row implies x <= 3, but the `ub - lb <= ZERO_TOL` guard must
/// skip the fixed box so it survives intact.
///
/// No-op proof: removing the fixing skip reverts x's ub to +inf, turning the
/// fixed [3,3] into [3, +inf) and failing the assertion.
#[test]
fn revert_redundant_added_bounds_skips_fixed_variable() {
    let mut st = make_state(
        vec![1.0],
        &[0],
        &[0],
        &[1.0],
        1,
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
    );
    st.bounds[0] = (3.0, 3.0);

    super::bounds::revert_redundant_added_bounds(&mut st);

    assert_eq!(
        st.bounds[0],
        (3.0, 3.0),
        "a variable fixed by tightening (lb == ub) must not be reverted"
    );
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
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    let (lb_x, _) = st.bounds[0];
    assert!(
        (lb_x - 0.5).abs() < 1e-10,
        "x_lb should tighten to 0.5, got {lb_x}"
    );
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
    let res = step5_bounds_tightening(&mut st, &mut fixed, None);
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
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
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
    step6_doubleton_equation(&mut st, &mut subst, None).unwrap();
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
    step6_doubleton_equation(&mut st, &mut subst, None).unwrap();
    assert_eq!(subst, 1);
    assert!(
        st.removed_cols[1],
        "free col y should be pivot, not bounded x"
    );
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
    let res = step6_doubleton_equation(&mut st, &mut subst, None);
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
    step7_free_var_substitution(&mut st, &mut subst, None).unwrap();
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
    step7_free_var_substitution(&mut st, &mut subst, None).unwrap();
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
    step7_free_var_substitution(&mut st, &mut subst, None).unwrap();
    assert_eq!(subst, 1);
    let pivot_mag = st.postsolve_stack.iter().find_map(|s| match s {
        PostsolveStep::LinearSubstitution {
            pivot, orig_col, ..
        } if *orig_col == 2 => Some(pivot.abs()),
        _ => None,
    });
    assert!(pivot_mag.is_some(), "free col z should be eliminated");
    assert!(
        (pivot_mag.unwrap() - 3.0).abs() < 1e-10,
        "should pick |3| over |1|"
    );
}

#[test]
fn step7_respects_expired_deadline() {
    // no-op proof: with a live deadline, Step7 performs one substitution.
    let mut st_live = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 1.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Eq],
        vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, 10.0)],
    );
    let mut subst_live = 0usize;
    step7_free_var_substitution(
        &mut st_live,
        &mut subst_live,
        Some(std::time::Instant::now() + std::time::Duration::from_millis(200)),
    )
    .unwrap();
    assert_eq!(subst_live, 1, "live deadline should allow substitution");

    // Sentinel: expired deadline must return before mutating state.
    let mut st_expired = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 1.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Eq],
        vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, 10.0)],
    );
    let mut subst_expired = 0usize;
    step7_free_var_substitution(
        &mut st_expired,
        &mut subst_expired,
        Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
    )
    .unwrap();
    assert_eq!(subst_expired, 0);
    assert!(
        !st_expired.removed_cols[0] && !st_expired.removed_rows[0],
        "expired deadline must short-circuit before elimination",
    );
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
    step8_free_singleton_col(&mut st, &mut subst, None).unwrap();
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
    step8_free_singleton_col(&mut st, &mut subst, None).unwrap();
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
    step8_free_singleton_col(&mut st, &mut subst, None).unwrap();
    assert_eq!(subst, 0);
    assert!(!st.removed_cols[2]);
}

// -----------------------------------------------------------
// substitution::fill_in_exceeds_budget
// -----------------------------------------------------------

#[test]
fn fill_in_budget_allows_dense_pivot_with_few_targets() {
    // Pivot row 0 = x+y+z=2 (Eq), only one other row touches x.
    // piv_others=2, col_j_others=1 → Markowitz fill 1*2=2 ≤ removed 1+2+1=4 → allowed.
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
    assert!(
        !fill_in_exceeds_budget(&st, 0, 0),
        "low fill-in must not be skipped"
    );
}

#[test]
fn fill_in_budget_blocks_high_fill() {
    // Pivot row 0: x0+x1+x2+x3+x4+x5+x6+x7=8 (Eq, 8 cols including x0).
    // Rows 1..8: x0+x_(k+7)=k (Le, only x0 and one disjoint col each).
    // piv_others=7 (x1..x7), col_j_others=7 (rows 1..7) → Markowitz fill 7*7=49.
    // removed_nnz = 1+7+7=15. 49 > 15 → blocked.
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
    cts.extend(std::iter::repeat_n(ConstraintType::Le, n_rows_extra));
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
    assert!(
        fill_in_exceeds_budget(&st, 0, 0),
        "dense disjoint fill should exceed budget"
    );
}

// -----------------------------------------------------------
// step1_fixed_variable — negative fix value, multi-row b update, obj_offset sign
// -----------------------------------------------------------

#[test]
fn step1_fixed_negative_value_updates_b_and_offset() {
    // x0 fixed at -3 (lb==ub), appears in row0 (coef 1, b=10) and row1 (coef 4, b=20).
    // After fixing: b0 = 10 - 1*(-3) = 13, b1 = 20 - 4*(-3) = 32, offset = c0*(-3) = 2*(-3) = -6.
    let mut st = make_state(
        vec![2.0, 0.0],
        &[0, 1, 1],
        &[0, 0, 1],
        &[1.0, 4.0, 1.0],
        2,
        2,
        vec![10.0, 20.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(-3.0, -3.0), (0.0, 10.0)],
    );
    step1_fixed_variable(&mut st, None).unwrap();
    assert!(st.removed_cols[0], "fixed col must be removed");
    assert!(
        (st.b[0] - 13.0).abs() < 1e-12,
        "b0 expected 13, got {}",
        st.b[0]
    );
    assert!(
        (st.b[1] - 32.0).abs() < 1e-12,
        "b1 expected 32, got {}",
        st.b[1]
    );
    assert!(
        (st.obj_offset - (-6.0)).abs() < 1e-12,
        "offset expected -6, got {}",
        st.obj_offset
    );
}

#[test]
fn step1_detects_lb_gt_ub() {
    // Bound inconsistency injected post-construction → step1 must report Infeasible.
    let mut st = make_state(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        1,
        vec![],
        vec![],
        vec![(0.0, 1.0)],
    );
    st.bounds[0] = (3.0, 2.0);
    assert_eq!(
        step1_fixed_variable(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

// -----------------------------------------------------------
// step2_singleton_row — negative coefficient, Le-row skip
// -----------------------------------------------------------

#[test]
fn step2_singleton_negative_coeff_solves_negative_value() {
    // Eq row0: -2*x0 = 6 → x0 = -3 (in [-5,0]). x0 also in Le row1 (coef 1, b=10).
    // After: b1 = 10 - 1*(-3) = 13, offset = c0*(-3) = 1*(-3) = -3.
    let mut st = make_state(
        vec![1.0, 0.0],
        &[0, 1, 1],
        &[0, 0, 1],
        &[-2.0, 1.0, 1.0],
        2,
        2,
        vec![6.0, 10.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(-5.0, 0.0), (0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_cols[0] && st.removed_rows[0]);
    assert!(
        (st.b[1] - 13.0).abs() < 1e-12,
        "b1 expected 13, got {}",
        st.b[1]
    );
    assert!(
        (st.obj_offset - (-3.0)).abs() < 1e-12,
        "offset expected -3, got {}",
        st.obj_offset
    );
}

#[test]
fn step2_singleton_infeasible_out_of_bounds() {
    // 2*x0 = 6 → x0 = 3, but x0 in [0,1] → Infeasible.
    let mut st = make_state(
        vec![1.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![6.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0)],
    );
    assert_eq!(
        step2_singleton_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

// -----------------------------------------------------------
// step3a_empty_row — Eq / Ge feasibility branches (Le covered in tests.rs)
// -----------------------------------------------------------

/// Build a 2-row state where row1 is structurally empty (no column touches it).
fn empty_second_row_state(ct1: ConstraintType, b1: f64) -> PresolveState {
    make_state(
        vec![1.0],
        &[0],
        &[0],
        &[1.0],
        2,
        1,
        vec![5.0, b1],
        vec![ConstraintType::Le, ct1],
        vec![(0.0, f64::INFINITY)],
    )
}

#[test]
fn step3a_empty_eq_row_nonzero_rhs_infeasible() {
    // Empty Eq row with b != 0 → 0 == b infeasible.
    let mut st = empty_second_row_state(ConstraintType::Eq, 5.0);
    assert_eq!(
        step3a_empty_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

#[test]
fn step3a_empty_eq_row_zero_rhs_feasible() {
    // Empty Eq row with b == 0 → 0 == 0 feasible, row removed.
    let mut st = empty_second_row_state(ConstraintType::Eq, 0.0);
    step3a_empty_row(&mut st, None).unwrap();
    assert!(st.removed_rows[1], "empty 0==0 row must be removed");
}

#[test]
fn step3a_empty_ge_row_positive_rhs_infeasible() {
    // Empty Ge row: 0 >= b with b > 0 → infeasible.
    let mut st = empty_second_row_state(ConstraintType::Ge, 5.0);
    assert_eq!(
        step3a_empty_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

#[test]
fn step3a_empty_ge_row_nonpositive_rhs_feasible() {
    // Empty Ge row: 0 >= b with b <= 0 → feasible, removed.
    let mut st = empty_second_row_state(ConstraintType::Ge, -2.0);
    step3a_empty_row(&mut st, None).unwrap();
    assert!(st.removed_rows[1]);
}

// -----------------------------------------------------------
// step3b_empty_column — cost-sign × bound branches
// -----------------------------------------------------------

/// Build a 2-col state where col1 is structurally empty (no row touches it).
fn empty_second_col_state(c1: f64, bounds1: (f64, f64)) -> PresolveState {
    make_state(
        vec![0.0, c1],
        &[0],
        &[0],
        &[1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), bounds1],
    )
}

#[test]
fn step3b_empty_col_positive_cost_no_lower_bound_unbounded() {
    // c1 > 0, lb = -inf → minimizing drives x1 → -inf → Unbounded.
    let mut st = empty_second_col_state(1.0, (f64::NEG_INFINITY, 10.0));
    assert_eq!(
        step3b_empty_column(&mut st, None),
        Err(PresolveStatus::Unbounded)
    );
}

#[test]
fn step3b_empty_col_negative_cost_fixes_at_upper_bound() {
    // c1 < 0, ub finite → x1 = ub = 5, offset += c1*ub = -2*5 = -10.
    let mut st = empty_second_col_state(-2.0, (0.0, 5.0));
    step3b_empty_column(&mut st, None).unwrap();
    assert!(st.removed_cols[1]);
    assert!(
        (st.obj_offset - (-10.0)).abs() < 1e-12,
        "offset expected -10, got {}",
        st.obj_offset
    );
}

#[test]
fn step3b_empty_col_zero_cost_finite_lb_fixes_at_lower_bound() {
    // c1 == 0, lb finite → x1 = lb = 3, offset unchanged.
    let mut st = empty_second_col_state(0.0, (3.0, 10.0));
    step3b_empty_column(&mut st, None).unwrap();
    assert!(st.removed_cols[1]);
    assert!(st.obj_offset.abs() < 1e-12, "zero-cost col adds no offset");
    let value = st.postsolve_stack.iter().find_map(|s| match s {
        PostsolveStep::EmptyColumn { orig_col: 1, value } => Some(*value),
        _ => None,
    });
    assert!(
        value.is_some_and(|v| (v - 3.0).abs() < 1e-12),
        "empty col value must default to finite lb=3"
    );
}

#[test]
fn step3b_empty_col_zero_cost_free_fixes_at_zero() {
    // c1 == 0, both bounds infinite (free) → x1 = 0.
    let mut st = empty_second_col_state(0.0, (f64::NEG_INFINITY, f64::INFINITY));
    step3b_empty_column(&mut st, None).unwrap();
    assert!(st.removed_cols[1]);
    let value = st.postsolve_stack.iter().find_map(|s| match s {
        PostsolveStep::EmptyColumn { orig_col: 1, value } => Some(*value),
        _ => None,
    });
    assert!(
        value.is_some_and(|v| v.abs() < 1e-12),
        "free zero-cost col value must default to 0"
    );
}

// -----------------------------------------------------------
// step4_redundant_constraint — Ge / Eq branches (Le covered in tests.rs)
// -----------------------------------------------------------

#[test]
fn step4_ge_constraint_redundant_when_activity_floor_dominates() {
    // x0 + x1 >= 1, x0,x1 in [1,5] → row activity min = 2 >= 1 → redundant.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![1.0],
        vec![ConstraintType::Ge],
        vec![(1.0, 5.0), (1.0, 5.0)],
    );
    step4_redundant_constraint(&mut st, None).unwrap();
    assert!(
        st.removed_rows[0],
        "Ge with activity floor >= rhs must be redundant"
    );
}

#[test]
fn step4_ge_constraint_not_redundant_when_floor_below_rhs() {
    // x0 + x1 >= 3, x0,x1 in [0,5] → activity min = 0 < 3 → NOT redundant.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    step4_redundant_constraint(&mut st, None).unwrap();
    assert!(!st.removed_rows[0]);
}

#[test]
fn step4_eq_constraint_redundant_when_activity_pins_rhs() {
    // x0 + x1 = 4 with x0,x1 fixed at 2 → activity range [4,4] == rhs → redundant.
    let mut st = make_state(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Eq],
        vec![(2.0, 2.0), (2.0, 2.0)],
    );
    step4_redundant_constraint(&mut st, None).unwrap();
    assert!(
        st.removed_rows[0],
        "Eq pinned to rhs by fixed vars must be redundant"
    );
}

// -----------------------------------------------------------
// step5_bounds_tightening — fixed-variable counting branch
// -----------------------------------------------------------

#[test]
fn step5_tightening_to_point_increments_new_fixed() {
    // x0 <= 0 with x0 in [0,10] → implied ub = 0, collapsing bounds to [0,0].
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[1.0],
        1,
        1,
        vec![0.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    let mut fixed = 0usize;
    step5_bounds_tightening(&mut st, &mut fixed, None).unwrap();
    assert_eq!(
        fixed, 1,
        "tightening that pins lb==ub must count as a new fix"
    );
    let (lb, ub) = st.bounds[0];
    assert!(
        lb.abs() < 1e-12 && ub.abs() < 1e-12,
        "bounds collapsed to [0,0]"
    );
}

// -----------------------------------------------------------
// step6_doubleton_equation — opposite-sign coefficients (ratio < 0)
// -----------------------------------------------------------

#[test]
fn step6_opposite_sign_coeffs_tightens_other_bound() {
    // x0 - x1 = 2, x0,x1 in [0,10]. Equal magnitudes, a1=1 >= a2=-1 → pivot=x0.
    // ratio = 1/(-1) = -1 < 0 branch. bo = 2/(-1) = -2.
    //   other_lb_impl = bo - ratio*lb_p = -2 - (-1)*0 = -2
    //   other_ub_impl = bo - ratio*ub_p = -2 - (-1)*10 = 8
    // x1 tightened: [max(0,-2), min(10,8)] = [0, 8]. Then x0 eliminated.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, -1.0],
        1,
        2,
        vec![2.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0)],
    );
    let mut subst = 0usize;
    step6_doubleton_equation(&mut st, &mut subst, None).unwrap();
    assert_eq!(subst, 1);
    assert!(
        st.removed_cols[0] && st.removed_rows[0],
        "pivot x0 eliminated"
    );
    let (lb1, ub1) = st.bounds[1];
    assert!(lb1.abs() < 1e-12, "x1 lb stays 0, got {lb1}");
    assert!((ub1 - 8.0).abs() < 1e-12, "x1 ub tightened to 8, got {ub1}");
    assert!(count_linear_subst(&st) >= 1);
    assert!(
        count_bounds_tightened(&st) >= 1,
        "ratio<0 must tighten other bound"
    );
}

// -----------------------------------------------------------
// step2_singleton_row — Le / Ge inequality extensions
// -----------------------------------------------------------

#[test]
fn step2_singleton_le_positive_coeff_tightens_ub() {
    // 3x <= 6, x in [0,10] → x <= 2. Row removed.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[3.0],
        1,
        1,
        vec![6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0], "singleton Le row must be removed");
    assert!(!st.removed_cols[0], "column stays active");
    let (lb, ub) = st.bounds[0];
    assert!((lb - 0.0).abs() < 1e-12);
    assert!((ub - 2.0).abs() < 1e-12, "ub tightened to 2, got {ub}");
}

#[test]
fn step2_singleton_le_negative_coeff_tightens_lb() {
    // -2x <= -6, x in [0,10] → x >= 3. Row removed.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[-2.0],
        1,
        1,
        vec![-6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    let (lb, _ub) = st.bounds[0];
    assert!((lb - 3.0).abs() < 1e-12, "lb tightened to 3, got {lb}");
}

#[test]
fn step2_singleton_ge_positive_coeff_tightens_lb() {
    // 4x >= 8, x in [0,10] → x >= 2. Row removed.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[4.0],
        1,
        1,
        vec![8.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    let (lb, ub) = st.bounds[0];
    assert!((lb - 2.0).abs() < 1e-12, "lb tightened to 2, got {lb}");
    assert!((ub - 10.0).abs() < 1e-12);
}

#[test]
fn step2_singleton_ge_negative_coeff_tightens_ub() {
    // -5x >= -15, x in [0,10] → x <= 3. Row removed.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[-5.0],
        1,
        1,
        vec![-15.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    let (lb, ub) = st.bounds[0];
    assert!((lb - 0.0).abs() < 1e-12);
    assert!((ub - 3.0).abs() < 1e-12, "ub tightened to 3, got {ub}");
}

#[test]
fn step2_singleton_le_infeasible() {
    // 2x <= 1, x in [3,10] → implied ub = 0.5 < lb = 3. Infeasible.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(3.0, 10.0)],
    );
    assert_eq!(
        step2_singleton_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

#[test]
fn step2_singleton_ge_infeasible() {
    // -1x >= 5, x in [0,2] → implied ub = -5 < lb = 0. Infeasible.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[-1.0],
        1,
        1,
        vec![5.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 2.0)],
    );
    assert_eq!(
        step2_singleton_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

#[test]
fn step2_singleton_le_redundant_no_tightening() {
    // 2x <= 30, x in [0,10] → implied ub=15 > current ub=10. No tightening, row still removed.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![30.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    let (lb, ub) = st.bounds[0];
    assert!((lb - 0.0).abs() < 1e-12);
    assert!((ub - 10.0).abs() < 1e-12, "ub unchanged at 10, got {ub}");
}

#[test]
fn step2_singleton_le_free_variable() {
    // 1x <= 5, x in (-inf, +inf) → ub tightened to 5.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[1.0],
        1,
        1,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
    );
    step2_singleton_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    let (lb, ub) = st.bounds[0];
    assert_eq!(lb, f64::NEG_INFINITY);
    assert!((ub - 5.0).abs() < 1e-12, "ub tightened to 5, got {ub}");
}

// -----------------------------------------------------------
// step2b_forcing_row
// -----------------------------------------------------------

use super::forcing::step2b_forcing_row;

#[test]
fn step2b_forcing_le_all_positive() {
    // x + y <= 3, x in [0,1], y in [0,2]. min = 0+0 = 0 < 3. max = 1+2 = 3.
    // a_min=0 < rhs=3 but a_max=3 = rhs → redundant, NOT forcing.
    // For forcing: a_min >= rhs. Let's use: x + y <= 0, x in [0,1], y in [0,2].
    // a_min=0 >= 0. Forcing from min: x→0, y→0.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![0.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
    assert!(
        matches!(
            st.postsolve_stack.last(),
            Some(PostsolveStep::ForcingRow { .. })
        ),
        "must push ForcingRow"
    );
}

#[test]
fn step2b_forcing_ge_all_positive() {
    // x + y >= 3, x in [0,1], y in [0,2]. max = 1+2 = 3 <= rhs=3. Forcing from max: x→1, y→2.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
}

#[test]
fn step2b_forcing_le_mixed_signs() {
    // 2x - y <= -1, x in [0,1], y in [0,3].
    // min = 2*0 + (-1)*3 = -3. -3 < -1 → not forcing.
    // Use: 2x - y <= -3. min = -3 >= -3 → forcing from min. x→0 (pos coeff), y→3 (neg coeff).
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, -1.0],
        1,
        2,
        vec![-3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 3.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
}

#[test]
fn step2b_forcing_eq_forced_from_below() {
    // x + y = 0, x in [0,1], y in [0,2]. min = 0+0 = 0 >= 0. Forced from below.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![0.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
}

#[test]
fn step2b_forcing_eq_forced_from_above() {
    // x + y = 3, x in [0,1], y in [0,2]. max = 1+2 = 3 <= 3. Forced from above.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
}

#[test]
fn step2b_forcing_single_var_skip() {
    // Singleton rows are handled by step2, step2b requires len >= 2.
    let mut st = make_state(
        vec![1.0],
        &[0],
        &[0],
        &[1.0],
        1,
        1,
        vec![0.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(
        !st.removed_rows[0],
        "singleton should not be caught by forcing"
    );
}

#[test]
fn step2b_near_forcing_no_trigger() {
    // x + y <= 1, x in [0,1], y in [0,2]. min=0 < 1. NOT forcing.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(!st.removed_rows[0], "near-forcing should NOT trigger");
}

#[test]
fn step2b_forcing_unbounded_var_skip() {
    // x + y <= 0, x in [0,1], y in (-inf,2]. min involves -inf, so lb_fin=false. Skip.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![0.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (f64::NEG_INFINITY, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    // lb_fin is false due to y's lower bound being -inf, so activity_range min is not finite.
    // The forcing condition lb_fin && row_lb >= rhs fails.
    assert!(
        !st.removed_rows[0],
        "unbounded contributing bound must skip"
    );
}

#[test]
fn step2b_forcing_ge_mixed_signs() {
    // -x + 2y >= 4, x in [0,1], y in [0,2].
    // max = (-1)*0 + 2*2 = 4 <= 4. Forcing from max: x→0 (neg coeff→lb), y→2 (pos coeff→ub).
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[-1.0, 2.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    step2b_forcing_row(&mut st, None).unwrap();
    assert!(st.removed_rows[0]);
    assert!(st.removed_cols[0] && st.removed_cols[1]);
}

// -----------------------------------------------------------
// step2_singleton_row — Ge infeasible
// -----------------------------------------------------------

#[test]
fn step2_singleton_ge_infeasible_lb_exceeds_ub() {
    // 2x >= 6, x in [0, 2] -> implied lb = 6/2 = 3 > ub = 2 -> Infeasible.
    let mut st = make_state(
        vec![0.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![6.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 2.0)],
    );
    assert_eq!(
        step2_singleton_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

// -----------------------------------------------------------
// step2b_forcing_row — infeasible (min activity > rhs)
// -----------------------------------------------------------

#[test]
fn step2b_forcing_le_infeasible_min_exceeds_rhs() {
    // x + y <= -1, x,y in [0,1]. min activity = 0+0 = 0 > rhs = -1 -> Infeasible.
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![-1.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    assert_eq!(
        step2b_forcing_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}

#[test]
fn step2b_forcing_le_infeasible_unbounded_ub() {
    // x + y <= -1, x,y in [0, +inf). lb_fin=true, ub_fin=false.
    // min activity = 0 > rhs = -1 -> Infeasible (lb_fin alone suffices for Le).
    let mut st = make_state(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![-1.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    assert_eq!(
        step2b_forcing_row(&mut st, None),
        Err(PresolveStatus::Infeasible)
    );
}
