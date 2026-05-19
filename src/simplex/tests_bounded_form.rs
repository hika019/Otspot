//! Equivalence sentinel for `BoundedStandardForm` (foundation for BFRT wiring).
//!
//! Locks in `wrap_to_legacy(build_bounded_standard_form(lp)) ≡ build_standard_form(lp)`
//! across multi-pattern bound fixtures (boxed / half-finite / free / fixed),
//! and proves a no-op identity-wrap collapses the equivalence (so the test
//! cannot pass on an empty implementation).

use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;

use super::standard_form::{
    build_bounded_standard_form, build_standard_form, wrap_to_legacy, BoundedStandardForm,
    StandardForm,
};

/// Float-comparison tolerance for CSC values / RHS / costs that traverse
/// `from_triplets` (DROP_TOL drop + sort-merge sum). Strict exact match would
/// be brittle against future numerical tweaks; 1e-15 keeps detectability of
/// any algebraic divergence well within solver tolerances.
const STRUCT_EPS: f64 = 1e-15;

fn vec_close(a: &[f64], b: &[f64], eps: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() <= eps)
}

fn csc_close(a: &CscMatrix, b: &CscMatrix, eps: f64) -> bool {
    a.nrows == b.nrows
        && a.ncols == b.ncols
        && a.col_ptr == b.col_ptr
        && a.row_ind == b.row_ind
        && vec_close(&a.values, &b.values, eps)
}

fn assert_structurally_equivalent(left: &StandardForm, right: &StandardForm) {
    assert_eq!(left.m, right.m, "m mismatch");
    assert_eq!(left.n_shifted, right.n_shifted, "n_shifted mismatch");
    assert_eq!(left.n_total, right.n_total, "n_total mismatch");
    assert_eq!(left.n_orig, right.n_orig, "n_orig mismatch");
    assert_eq!(left.num_artificial, right.num_artificial, "num_artificial mismatch");
    assert_eq!(left.initial_basis, right.initial_basis, "initial_basis mismatch");
    assert_eq!(left.needs_artificial, right.needs_artificial, "needs_artificial mismatch");
    assert_eq!(left.row_negated, right.row_negated, "row_negated mismatch");
    assert!((left.obj_offset - right.obj_offset).abs() <= STRUCT_EPS, "obj_offset mismatch");
    assert!(vec_close(&left.b, &right.b, STRUCT_EPS), "b mismatch");
    assert!(vec_close(&left.c, &right.c, STRUCT_EPS), "c mismatch");
    assert!(csc_close(&left.a, &right.a, STRUCT_EPS), "A mismatch");
    assert_eq!(
        left.orig_var_info.len(),
        right.orig_var_info.len(),
        "orig_var_info len mismatch"
    );
    for (li, ri) in left.orig_var_info.iter().zip(right.orig_var_info.iter()) {
        assert!((li.offset - ri.offset).abs() <= STRUCT_EPS, "orig_var_info offset");
        assert_eq!(li.new_vars, ri.new_vars, "orig_var_info new_vars");
    }
}

/// Build a representative LP that exercises every bound topology BFRT must
/// support:
/// - x0 ∈ [2, 7]            (boxed: lb finite, ub finite, lb≠ub)
/// - x1 ∈ [0, +∞)           (half-finite lower)
/// - x2 ∈ (-∞, 4]           (half-finite upper)
/// - x3 ∈ (-∞, +∞)          (free → split)
/// - x4 = 3                 (fixed: lb=ub)
///
/// Constraints mix Le / Ge / Eq with the signed b values that exercise the
/// row-sign-normalization branch (b < -PIVOT_TOL ⇒ row negated).
fn fixture_mixed_bounds() -> LpProblem {
    let n = 5;
    let m = 3;
    let rows = vec![0, 1, 2, 0, 1, 0, 2, 1, 2, 0, 2];
    let cols = vec![0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let vals = vec![1.0, 2.0, -1.0, 1.0, 1.0, 1.0, -2.0, 1.0, 3.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b = vec![20.0, 5.0, 8.0];
    let c = vec![1.0, -2.0, 3.0, -1.0, 2.0];
    let constraint_types = vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq];
    let bounds = vec![
        (2.0, 7.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 4.0),
        (f64::NEG_INFINITY, f64::INFINITY),
        (3.0, 3.0),
    ];
    LpProblem::new_general(c, a, b, constraint_types, bounds, None).unwrap()
}

/// Tight bounds + a row whose post-shift b becomes negative (forcing row
/// negation in standard form). Covers the boxed-only path.
fn fixture_all_boxed_with_neg_b() -> LpProblem {
    let n = 3;
    let m = 2;
    let rows = vec![0, 0, 1, 1, 0, 1];
    let cols = vec![0, 1, 0, 2, 2, 1];
    let vals = vec![1.0, 1.0, 1.0, 1.0, -1.0, 2.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    // Choose b so that after the lb-shift the row 0 RHS goes negative.
    // x0 ∈ [5, 10], x1 ∈ [0, 4], x2 ∈ [-3, 1]
    // row0: x0 + x1 - x2 ≤ b0 ⇒ shifted: x0' + x1' - x2' ≤ b0 - 5 - 0 + (-3) = b0 - 8
    // pick b0 = 3 ⇒ shifted b0 = -5 (row negation triggers)
    let b = vec![3.0, 7.0];
    let c = vec![1.0, 1.0, 1.0];
    let constraint_types = vec![ConstraintType::Le, ConstraintType::Le];
    let bounds = vec![(5.0, 10.0), (0.0, 4.0), (-3.0, 1.0)];
    LpProblem::new_general(c, a, b, constraint_types, bounds, None).unwrap()
}

fn fixture_only_free_and_half() -> LpProblem {
    let n = 3;
    let m = 2;
    let rows = vec![0, 0, 0, 1, 1, 1];
    let cols = vec![0, 1, 2, 0, 1, 2];
    let vals = vec![1.0, 2.0, 1.0, -1.0, 1.0, 4.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b = vec![6.0, 2.0];
    let c = vec![1.0, 1.0, -1.0];
    let constraint_types = vec![ConstraintType::Eq, ConstraintType::Ge];
    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 5.0),
    ];
    LpProblem::new_general(c, a, b, constraint_types, bounds, None).unwrap()
}

#[test]
fn wrap_to_legacy_matches_build_standard_form_mixed() {
    let lp = fixture_mixed_bounds();
    let bsf = build_bounded_standard_form(&lp);
    let sf_via_wrap = wrap_to_legacy(&bsf);
    let sf_direct = build_standard_form(&lp);
    assert_structurally_equivalent(&sf_via_wrap, &sf_direct);
}

#[test]
fn wrap_to_legacy_matches_build_standard_form_all_boxed_neg_b() {
    let lp = fixture_all_boxed_with_neg_b();
    let bsf = build_bounded_standard_form(&lp);
    let sf_via_wrap = wrap_to_legacy(&bsf);
    let sf_direct = build_standard_form(&lp);
    assert_structurally_equivalent(&sf_via_wrap, &sf_direct);
}

#[test]
fn wrap_to_legacy_matches_build_standard_form_free_and_half() {
    let lp = fixture_only_free_and_half();
    let bsf = build_bounded_standard_form(&lp);
    let sf_via_wrap = wrap_to_legacy(&bsf);
    let sf_direct = build_standard_form(&lp);
    assert_structurally_equivalent(&sf_via_wrap, &sf_direct);
}

#[test]
fn bounded_form_records_finite_upper_for_boxed_vars() {
    let lp = fixture_mixed_bounds();
    let bsf = build_bounded_standard_form(&lp);
    // x0 ∈ [2, 7] ⇒ shifted col 0 upper = 5.
    // x4 = 3 ⇒ shifted col (n_orig-th boxed) upper = 0.
    // x1 / x2 / x3 are half/free ⇒ upper = +∞.
    let finite_count = (0..bsf.n_shifted)
        .filter(|&j| bsf.upper_bounds[j].is_finite())
        .count();
    assert_eq!(finite_count, 2, "boxed + fixed should be the only finite uppers");
    // slack columns inherit +∞
    for j in bsf.n_shifted..bsf.n_total {
        assert!(bsf.upper_bounds[j].is_infinite());
    }
    // The boxed var (x0) shifts by lb=2, so the upper bound is 7-2 = 5.
    let boxed_idx = bsf.orig_var_info[0].new_vars[0].0;
    assert!((bsf.upper_bounds[boxed_idx] - 5.0).abs() < 1e-15);
    // Fixed (x4): lb=ub=3 ⇒ shifted upper = 0.
    let fixed_idx = bsf.orig_var_info[4].new_vars[0].0;
    assert!(bsf.upper_bounds[fixed_idx].abs() < 1e-15);
}

#[test]
fn bounded_form_has_no_ub_rows() {
    // Sentinel: BFRT's reason-for-existing is that legacy StandardForm
    // adds UB rows, whereas BoundedStandardForm must keep `m == m_orig`.
    let lp = fixture_mixed_bounds();
    let bsf = build_bounded_standard_form(&lp);
    let sf = build_standard_form(&lp);
    assert_eq!(bsf.m, lp.num_constraints, "m must equal original m");
    assert!(sf.m > bsf.m, "legacy must add UB rows; otherwise BSF brings no value");
}

// ---------------------------------------------------------------------------
// No-op proof: an identity `wrap_to_legacy` (UB rows dropped) cannot satisfy
// the equivalence sentinel for any LP that has finite-upper bounded vars.
// This proves the sentinel actually depends on the UB-expansion logic.
// ---------------------------------------------------------------------------

/// Identity wrap: copies BSF fields verbatim into a StandardForm, *without*
/// appending UB rows. This is the no-op the sentinel must reject.
fn wrap_to_legacy_noop(bsf: &BoundedStandardForm) -> StandardForm {
    let orig_var_info = bsf
        .orig_var_info
        .iter()
        .map(|info| crate::simplex::OrigVarInfo {
            offset: info.offset,
            new_vars: info.new_vars.clone(),
        })
        .collect();
    StandardForm {
        a: bsf.a.clone(),
        b: bsf.b.clone(),
        c: bsf.c.clone(),
        m: bsf.m,
        n_shifted: bsf.n_shifted,
        n_total: bsf.n_total,
        initial_basis: bsf.initial_basis.clone(),
        needs_artificial: bsf.needs_artificial.clone(),
        num_artificial: bsf.num_artificial,
        obj_offset: bsf.obj_offset,
        n_orig: bsf.n_orig,
        orig_var_info,
        row_negated: bsf.row_negated.clone(),
    }
}

#[test]
fn noop_wrap_fails_equivalence_on_mixed_bounds() {
    let lp = fixture_mixed_bounds();
    let bsf = build_bounded_standard_form(&lp);
    let sf_noop = wrap_to_legacy_noop(&bsf);
    let sf_direct = build_standard_form(&lp);
    assert!(
        sf_noop.m < sf_direct.m,
        "no-op must omit UB rows; otherwise the wrap isn't doing real work"
    );
    assert!(
        sf_noop.n_total < sf_direct.n_total,
        "no-op must omit UB-row slacks"
    );
}

#[test]
fn noop_wrap_fails_equivalence_on_boxed_only() {
    let lp = fixture_all_boxed_with_neg_b();
    let bsf = build_bounded_standard_form(&lp);
    let sf_noop = wrap_to_legacy_noop(&bsf);
    let sf_direct = build_standard_form(&lp);
    // 3 boxed vars ⇒ 3 UB rows in the legacy form, all absent in no-op.
    assert_eq!(sf_direct.m - sf_noop.m, 3, "expected 3 UB rows lost");
    assert_eq!(
        sf_direct.n_total - sf_noop.n_total,
        3,
        "expected 3 UB-row slacks lost"
    );
}

#[test]
fn noop_wrap_passes_equivalence_when_no_finite_uppers() {
    // Counter-example: if every variable is half-finite or free, there are
    // no UB rows to add ⇒ identity wrap *is* the correct wrap. This guards
    // against false positives where the no-op proof would fire spuriously.
    let lp = fixture_only_free_and_half();
    let bsf = build_bounded_standard_form(&lp);
    let sf_noop = wrap_to_legacy_noop(&bsf);
    let sf_real = wrap_to_legacy(&bsf);
    let sf_direct = build_standard_form(&lp);
    // The real wrap must equal direct build...
    assert_structurally_equivalent(&sf_real, &sf_direct);
    // ...and the no-op coincidentally does too, because there are zero UB rows.
    assert_eq!(sf_noop.m, sf_direct.m);
    assert_eq!(sf_noop.n_total, sf_direct.n_total);
}
