//! Bounded-path dispatch logic for dual advanced solver.

use super::bounded_core::{
    bump_eq_ub_dispatch_count, iterate as bounded_iterate, solve_bounded_dual, BoundedDualState,
};
use super::pipeline::{finish_bounded, run_phase1_then_phase2};
use super::{
    fallback_profile_enabled, make_leaving_strategy, warm_basis_is_dual_feasible,
    CRASH_INFEASIBLE_FALLBACKS,
};
use crate::basis::{BasisManager, LuBasis};
use crate::linalg::timeout::deadline_reached;
use crate::presolve::{LpEquilibration, LpScalingResult};
use crate::problem::{LpProblem, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::DROP_TOL;
use std::sync::atomic::Ordering;

use super::super::{scale_upper_bounds, BoundedStandardForm};
use crate::options::SolverOptions;

fn scale_bounded_standard_form(
    bsf: &BoundedStandardForm,
    options: &SolverOptions,
) -> Option<LpScalingResult> {
    if options.use_ruiz_scaling {
        LpEquilibration::scale_with_deadline(&bsf.a, &bsf.b, &bsf.c, options.deadline)
    } else {
        Some((
            bsf.a.clone(),
            bsf.b.clone(),
            bsf.c.clone(),
            vec![1.0; bsf.m],
            vec![1.0; bsf.n_total],
        ))
    }
}

/// Try to solve a Le-only bounded LP via the BFRT-aware dual+primal path.
///
/// Returns `Some(result)` on success or definite failure (Infeasible / Timeout /
/// NumericalError). Returns `None` when `BoundedOutcome::UbViolationOutOfScope`
/// is reached, signalling the caller to fall back to the legacy path.
pub(super) fn try_bounded(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> Option<SolverResult> {
    if deadline_reached(options.deadline) {
        return Some(SolverResult::timeout());
    }
    let Some((a, b, c, row_scale, col_scale)) = scale_bounded_standard_form(bsf, options) else {
        return Some(SolverResult::timeout());
    };
    let ubs = scale_upper_bounds(&bsf.upper_bounds, &col_scale);
    // total_iters is always assigned before read (warm branch overwrites before
    // passing &mut to finish_bounded; cold path overwrites before return).
    let mut total_iters: usize;

    // Warm start: reuse a previously-saved bounded-path basis when the index
    // space matches (basis.len() == bsf.m, all indices < bsf.n_total). Warm
    // starts from the legacy path have basis.len() == sf.m > bsf.m when UBs
    // are present, so they fall through to cold start automatically.
    if let Some(warm) = &options.warm_start {
        if warm.basis.len() == bsf.m && warm.basis.iter().all(|&idx| idx < bsf.n_total) {
            if let Ok(mut basis_mgr) =
                LuBasis::new_timed(&a, &warm.basis, options.max_etas, options.deadline)
            {
                let mut x_b_sv = SparseVec::from_dense(&b);
                basis_mgr.ftran(&mut x_b_sv);
                let x_b = x_b_sv.to_dense();
                // Bounded path warm-start: has_lb_violation 時は cold-start fallback。
                // Reason: bounded_core::iterate の BFRT は lower-bound 列のみ選択、
                // sign flip equivalent なし → lb_violation は repair 不可、cycle→timeout
                // (見つかったのは過去のレビューパス)。真因対処 (`WarmStartBasis` に
                // `at_upper` field を追加 + bounded core repair algorithm) は未実装で、
                // 追跡先も存在しない (この repo に GitHub issue はゼロ件)。それまでは
                // この cold-start fallback が正。
                //
                // Also fall through when the warm basis is dual-infeasible under
                // the new cost vector c: the dual simplex would exit immediately as
                // Optimal with a wrong objective value.
                let has_lb_violation = super::super::has_lb_violation(&x_b, options.primal_tol);
                let is_basic_bounded: Vec<bool> = {
                    let mut v = vec![false; bsf.n_total];
                    for &j in &warm.basis {
                        v[j] = true;
                    }
                    v
                };
                if !has_lb_violation
                    && warm_basis_is_dual_feasible(
                        &a,
                        &c,
                        &mut basis_mgr,
                        &warm.basis,
                        &is_basic_bounded,
                        bsf.n_total,
                        bsf.m,
                        options.dual_tol,
                    )
                {
                    let state = BoundedDualState {
                        basis: warm.basis.clone(),
                        at_upper: vec![false; bsf.n_total],
                        x_b,
                        reduced_costs: vec![0.0; bsf.n_total],
                        is_basic: is_basic_bounded,
                        iterations: 0,
                        price_start: 0,
                    };
                    let mut leaving = make_leaving_strategy(options.dual_pricing, bsf.m);
                    let (dual_out, dual_state) =
                        bounded_iterate(state, bsf, &a, &c, options, &ubs, leaving.as_mut());
                    total_iters = dual_state.iterations;
                    let result = finish_bounded(
                        dual_out,
                        dual_state,
                        bsf,
                        &a,
                        &b,
                        &c,
                        &row_scale,
                        &col_scale,
                        &ubs,
                        problem,
                        options,
                        &mut total_iters,
                    );
                    if result.is_some() {
                        return result;
                    }
                    // UbViolationOutOfScope → cold start
                } // dual-infeasibility: fall through to cold start
            }
            // Singular warm basis → cold start
        }
    }

    // Cold start.
    let mut leaving = make_leaving_strategy(options.dual_pricing, bsf.m);
    let (dual_out, dual_state) =
        solve_bounded_dual(bsf, &a, &b, &c, options, &ubs, leaving.as_mut());
    total_iters = dual_state.iterations;
    finish_bounded(
        dual_out,
        dual_state,
        bsf,
        &a,
        &b,
        &c,
        &row_scale,
        &col_scale,
        &ubs,
        problem,
        options,
        &mut total_iters,
    )
}

/// Build the augmented matrix `[bsf.a | I_art]` for the Eq+UB Phase I path.
/// Returns `(a_aug, art_col_of_row, n_art)`. `art_col_of_row[i] = Some(col)` iff
/// `needs_artificial[i]`; `col` indexes into `[bsf.n_total, n_aug)`.
fn build_a_aug_for_eq(
    bsf: &BoundedStandardForm,
    a_scaled: &CscMatrix,
    needs_artificial: &[bool],
) -> (CscMatrix, Vec<Option<usize>>, usize) {
    let m = bsf.m;
    let n_total = bsf.n_total;
    let mut art_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut n_art = 0usize;
    for i in 0..m {
        if needs_artificial[i] {
            art_col_of_row[i] = Some(n_total + n_art);
            n_art += 1;
        }
    }
    let n_aug = n_total + n_art;

    let mut trip_rows: Vec<usize> = Vec::with_capacity(a_scaled.nnz() + n_art);
    let mut trip_cols: Vec<usize> = Vec::with_capacity(a_scaled.nnz() + n_art);
    let mut trip_vals: Vec<f64> = Vec::with_capacity(a_scaled.nnz() + n_art);
    for j in 0..n_total {
        let (rows, vals) = a_scaled.get_column(j).unwrap();
        for (k, &row) in rows.iter().enumerate() {
            let v = vals[k];
            if v.abs() > DROP_TOL {
                trip_rows.push(row);
                trip_cols.push(j);
                trip_vals.push(v);
            }
        }
    }
    for (i, col_opt) in art_col_of_row.iter().enumerate() {
        if let Some(col) = col_opt {
            trip_rows.push(i);
            trip_cols.push(*col);
            trip_vals.push(1.0);
        }
    }

    let a_aug = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_aug)
        .expect("augmented matrix construction must succeed (deduplicated by build)");
    (a_aug, art_col_of_row, n_art)
}

/// Initial basic values `x_B = B^{-1} b` for the slack/artificial starting
/// basis. That basis is structurally diagonal — each slack column and each
/// artificial column has its single nonzero at its own row — but after Ruiz
/// scaling the Le-slack diagonals are non-unit, so `x_B[i] = b[i] / diag_i`,
/// not `b[i]`. Artificial columns have `diag = 1`. Negative results are clamped
/// to 0: the bounded primal starts inside the box and a feasible slack/art
/// basis yields `x_B ≥ 0` (the clamp is a no-op there and only guards roundoff).
pub(super) fn diag_basis_initial_x_b(a_aug: &CscMatrix, basis: &[usize], b: &[f64]) -> Vec<f64> {
    let m = basis.len();
    let mut x_b = vec![0.0f64; m];
    for i in 0..m {
        let (rows, vals) = a_aug.get_column(basis[i]).unwrap();
        let mut diag = 0.0f64;
        for (k, &r) in rows.iter().enumerate() {
            if r == i {
                diag = vals[k];
                break;
            }
        }
        debug_assert!(
            diag != 0.0,
            "slack/artificial starting basis must be diagonal (nonzero pivot at its own row)"
        );
        let v = b[i] / diag;
        x_b[i] = if v < 0.0 { 0.0 } else { v };
    }
    x_b
}

/// Eq+UB cold start: augment with artificials, run primal Phase I (minimise
/// sum of artificials), then primal Phase II on the augmented matrix with
/// pricing restricted to structural columns.
///
/// Returns `None` to signal "fall through to legacy" when Phase I evidence is
/// inconclusive (e.g. Phase I Timeout on a possibly-feasible LP) — Big-M /
/// primal Phase I in the legacy path may still resolve it.
pub(super) fn try_bounded_phase1_eq(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> Option<SolverResult> {
    if deadline_reached(options.deadline) {
        return Some(SolverResult::timeout());
    }
    let Some((a, b, c, row_scale, col_scale)) = scale_bounded_standard_form(bsf, options) else {
        return Some(SolverResult::timeout());
    };
    let ubs = scale_upper_bounds(&bsf.upper_bounds, &col_scale);

    bump_eq_ub_dispatch_count();

    // Optional crash basis: replaces artificial placeholders with structural
    // columns where a primal-feasible pivot exists. Returns identity-equivalent
    // (basis_pre = bsf.initial_basis, needs_artificial = bsf.needs_artificial)
    // when disabled. Crash is applied opportunistically; if x_b ≥ 0 fails after
    // FTRAN, we fall back to identity (no recursion: re-derive on the spot).
    let identity_state = || {
        let (a_aug, art_col_of_row, n_art) = build_a_aug_for_eq(bsf, &a, &bsf.needs_artificial);
        let n_aug = bsf.n_total + n_art;
        let mut ubs_aug = vec![f64::INFINITY; n_aug];
        ubs_aug[..bsf.n_total].copy_from_slice(&ubs);
        let mut basis: Vec<usize> = bsf.initial_basis.clone();
        for (i, col_opt) in art_col_of_row.iter().enumerate() {
            if let Some(col) = col_opt {
                basis[i] = *col;
            }
        }
        let mut is_basic = vec![false; n_aug];
        for &j in &basis {
            is_basic[j] = true;
        }
        let x_b = diag_basis_initial_x_b(&a_aug, &basis, &b);
        (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b)
    };

    let (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b) = if options.use_lp_crash_basis {
        let (basis_pre, needs_artificial, _n_art_post) = super::super::crash::compute_crash_basis(
            &a,
            &b,
            bsf.m,
            bsf.n_shifted,
            &bsf.initial_basis,
            &bsf.needs_artificial,
        );
        let (a_aug, art_col_of_row, n_art) = build_a_aug_for_eq(bsf, &a, &needs_artificial);
        let n_aug = bsf.n_total + n_art;
        let mut ubs_aug = vec![f64::INFINITY; n_aug];
        ubs_aug[..bsf.n_total].copy_from_slice(&ubs);
        let mut basis: Vec<usize> = basis_pre;
        for (i, col_opt) in art_col_of_row.iter().enumerate() {
            if let Some(col) = col_opt {
                basis[i] = *col;
            }
        }
        let mut is_basic = vec![false; n_aug];
        for &j in &basis {
            is_basic[j] = true;
        }
        // Compute x_B = B^{-1} b. Skip the FTRAN when crash made no change
        // (basis is the slack/art identity, so x_B = b directly).
        let needs_ftran = needs_artificial
            .iter()
            .zip(bsf.needs_artificial.iter())
            .any(|(post, orig)| post != orig);
        let x_b = if needs_ftran {
            match LuBasis::new_timed(&a_aug, &basis, options.max_etas, options.deadline) {
                Ok(mut bm) => {
                    let mut sv = SparseVec::from_dense(&b);
                    bm.ftran(&mut sv);
                    let xb = sv.to_dense();
                    // Reject both lb (xb < 0) and ub (xb > ubs_aug[basis[i]])
                    // violations — primal_simplex_aug starts inside the box and
                    // cannot repair an initial UB overshoot.
                    let infeasible = xb.iter().enumerate().any(|(i, &v)| {
                        if v < -options.primal_tol {
                            return true;
                        }
                        let ub_i = ubs_aug[basis[i]];
                        ub_i.is_finite() && v > ub_i + options.primal_tol
                    });
                    if infeasible {
                        // Crash yielded a bounded-infeasible RHS. The user
                        // explicitly enabled crash; the legacy path (crash +
                        // Big-M, UB rows expanded) can absorb violations the
                        // bounded primal start cannot. Hand off rather than
                        // silently dropping crash's iter savings.
                        if fallback_profile_enabled() {
                            CRASH_INFEASIBLE_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        }
                        return None;
                    }
                    xb
                }
                Err(_) => {
                    return run_phase1_then_phase2(
                        bsf,
                        problem,
                        options,
                        identity_state,
                        &b,
                        &c,
                        &row_scale,
                        &col_scale,
                    );
                }
            }
        } else {
            // Crash made no change → basis is still the slack/art identity
            // (diagonal); use the scaled-diagonal solve, not a raw b.clone().
            diag_basis_initial_x_b(&a_aug, &basis, &b)
        };
        (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b)
    } else {
        identity_state()
    };

    run_phase1_then_phase2(
        bsf,
        problem,
        options,
        move || (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b),
        &b,
        &c,
        &row_scale,
        &col_scale,
    )
}
