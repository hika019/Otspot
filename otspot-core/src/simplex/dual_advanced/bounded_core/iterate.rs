//! Core dual simplex iteration loop for bounded standard form.

use crate::linalg::timeout::deadline_reached;
use super::extract::bounded_obj;
use super::pricing::compute_reduced_costs_into_timed;
use super::{BoundedDualState, BoundedOutcome};
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use std::sync::atomic::Ordering;

use super::super::super::dual_common::{
    basic_obj, compute_dual_vars_into, made_progress_with_floor, recompute_gamma_truth,
    NO_PROGRESS_MIN, NO_PROGRESS_TRIGGER_FACTOR,
};
use super::super::super::pricing::DualLeavingStrategy;
use super::super::super::standard_form::BoundedStandardForm;
use super::super::super::trace::IterTrace;
use super::super::bound_flip::{bfrt_select_entering, ColBound};

#[cfg(test)]
thread_local! {
    static FLIP_APPLY_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_flip_apply_disabled(v: bool) {
    FLIP_APPLY_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn flip_apply_disabled() -> bool {
    FLIP_APPLY_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn flip_apply_disabled() -> bool {
    false
}

/// Per-iteration scratch buffers. Allocated once and reused across iters.
struct IterBuffers {
    rho: Vec<f64>,
    trow: Vec<f64>,
    alpha: Vec<f64>,
    alpha_flip: Vec<f64>,
    sigma: Vec<f64>,
    col_bounds: Vec<ColBound>,
    y: Vec<f64>,
}

impl IterBuffers {
    fn new(m: usize, n_total: usize, upper_bounds: &[f64]) -> Self {
        let col_bounds = (0..n_total)
            .map(|j| ColBound {
                upper: upper_bounds[j],
                at_upper: false,
            })
            .collect();
        Self {
            rho: vec![0.0; m],
            trow: vec![0.0; n_total],
            alpha: vec![0.0; m],
            alpha_flip: vec![0.0; m],
            sigma: vec![0.0; m],
            col_bounds,
            y: vec![0.0; m],
        }
    }
}

/// Inner iteration loop. Accepts a pre-populated state — tests use this to
/// inject synthetic primal infeasibilities; production cold/warm-start callers
/// supply the matching basis. Cost perturbation is applied here so callers
/// don't have to pre-perturb `c`.
///
/// `ubs` is the effective per-column upper bound slice used for bound-violation
/// checks and flip weights. Pass `&bsf.upper_bounds` when the matrices are
/// unscaled; pass Ruiz-scaled bounds (`u_j / col_scale[j]`) when scaled.
pub(crate) fn iterate(
    mut state: BoundedDualState,
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    c: &[f64],
    options: &SolverOptions,
    ubs: &[f64],
    leaving: &mut dyn DualLeavingStrategy,
) -> (BoundedOutcome, BoundedDualState) {
    let m = bsf.m;
    let n_total = bsf.n_total;
    debug_assert_eq!(state.basis.len(), m);
    debug_assert_eq!(state.x_b.len(), m);
    debug_assert_eq!(state.at_upper.len(), n_total);
    debug_assert_eq!(state.is_basic.len(), n_total);

    let mut basis_mgr =
        match LuBasis::new_timed(a, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(crate::error::SolverError::SingularBasis { .. }) => {
                return (BoundedOutcome::SingularBasis, state);
            }
            Err(_) => {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
        };

    // Early-exit before O(m²) γ init; prevents budget overrun on large warm-start solves.
    if deadline_reached(options.deadline)
        || options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    {
        let obj = bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        );
        return (BoundedOutcome::Timeout(obj), state);
    }

    let needs_sigma = leaving.needs_sigma();
    if needs_sigma {
        match recompute_gamma_truth(
            &mut basis_mgr,
            m,
            options.deadline,
            options.cancel_flag.as_deref(),
        ) {
            None => {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            Some(gamma_truth) => leaving.set_initial_gamma(&gamma_truth),
        }
    }

    // Cost perturbation: c̃_j = max(c_j, 0). With slack initial basis (y = 0)
    // every reduced cost is ≥ 0 ⇒ dual feasible. The perturbation is local
    // to this loop; the caller restores the original cost in Phase 2 primal.
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut buf = IterBuffers::new(m, n_total, ubs);

    // Initial reduced costs (r_j = c̃_j − y^T a_j with y = B^{-T} c̃_B).
    if !compute_reduced_costs_into_timed(
        a,
        &c_perturbed,
        &mut basis_mgr,
        &state.is_basic,
        n_total,
        &state.basis,
        &mut buf.y,
        &mut state.reduced_costs,
        options.deadline,
    ) {
        let obj = bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        );
        return (BoundedOutcome::Timeout(obj), state);
    }

    // Anti-cycling: track progress; switch to Bland's rule when stalled.
    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    let mut best_infeas = leaving.progress_metric(&state.x_b, &state.basis);
    let mut iters_since_progress: usize = 0;
    let mut bland_mode = false;
    let mut trace = IterTrace::new("bounded-dual");

    loop {
        state.iterations = state.iterations.saturating_add(1);
        let timed_out = deadline_reached(options.deadline);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            return (BoundedOutcome::Timeout(obj), state);
        }

        if let Some(t) = trace.as_mut() {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            t.log(state.iterations, obj, &state.basis, bland_mode);
        }

        // ub-violation scan (separate from lb-violation leaving selection).
        let mut ub_violation_row: Option<usize> = None;
        for i in 0..m {
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            let xi = state.x_b[i];
            let ub_i = ubs[state.basis[i]];
            if ub_i.is_finite() && xi > ub_i + options.primal_tol {
                ub_violation_row.get_or_insert(i);
            }
        }
        let leaving_row = if bland_mode {
            leaving.bland_leaving(&state.x_b, options.primal_tol, &state.basis)
        } else {
            leaving.select_leaving(&state.x_b, options.primal_tol, &state.basis)
        };
        if leaving_row.is_none() {
            if let Some(row) = ub_violation_row {
                return (BoundedOutcome::UbViolationOutOfScope { row }, state);
            }
            let obj = basic_obj(c, &state.basis, &state.x_b);
            let mut y = vec![0.0; m];
            compute_dual_vars_into(&c_perturbed, &mut basis_mgr, &state.basis, &mut y);
            return (BoundedOutcome::Optimal(obj, y), state);
        }
        let r = leaving_row.unwrap();

        // BTRAN ρ = B^{-T} e_r.
        for slot in buf.rho.iter_mut() {
            *slot = 0.0;
        }
        buf.rho[r] = 1.0;
        basis_mgr.btran_dense(&mut buf.rho);

        // σ = B^{-1} ρ (needed by DSE after_pivot weight update).
        if needs_sigma {
            buf.sigma.copy_from_slice(&buf.rho);
            basis_mgr.ftran_dense(&mut buf.sigma);
        }

        // PRICE trow[j] = ρ^T a_j on non-basic columns.
        for j in 0..n_total {
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            if state.is_basic[j] {
                buf.trow[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut dot = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                dot += buf.rho[row] * vals[k];
            }
            buf.trow[j] = dot;
        }

        // Refresh `col_bounds.at_upper` (uppers themselves never change).
        for j in 0..n_total {
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            buf.col_bounds[j].at_upper = state.at_upper[j];
        }

        let leaving_residual = state.x_b[r]; // negative; BFRT uses |·|
        let bfrt = match bfrt_select_entering(
            &buf.trow,
            &state.reduced_costs,
            &state.is_basic,
            &buf.col_bounds,
            n_total,
            PIVOT_TOL,
            leaving_residual,
        ) {
            None => {
                return (BoundedOutcome::Unbounded, state);
            }
            Some(res) => res,
        };

        // Apply flips: each non-entering bypassed breakpoint switches its
        // bound. x_B picks up Δx_N[k] · α_k per flip — flip from lb (0) to ub
        // (u_k) adds +u_k to x_N[k]; ub→lb adds −u_k. x_B := x_B − α_k · Δ.
        let apply_flip = !flip_apply_disabled();
        for &k in &bfrt.flips {
            let u_k = ubs[k];
            debug_assert!(u_k.is_finite(), "BFRT must not return infinite-upper flips");
            if apply_flip {
                ftran_column(a, &mut basis_mgr, k, m, &mut buf.alpha_flip);
                let direction = if state.at_upper[k] { -1.0 } else { 1.0 };
                let weight = direction * u_k;
                for i in 0..m {
                    state.x_b[i] -= buf.alpha_flip[i] * weight;
                }
            }
            state.at_upper[k] = !state.at_upper[k];
        }

        let entering_col = bfrt.entering_col;
        let theta = bfrt.theta;
        let entering_at_upper = state.at_upper[entering_col];

        // FTRAN α_q = B^{-1} a_q.
        ftran_column(a, &mut basis_mgr, entering_col, m, &mut buf.alpha);
        let pivot_element = buf.alpha[r];
        if pivot_element.abs() < PIVOT_TOL {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return (BoundedOutcome::SingularBasis, state);
                }
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if !compute_reduced_costs_into_timed(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
                options.deadline,
            ) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            continue;
        }

        // Standard column-swap pivot update of x_B.
        let step = state.x_b[r] / pivot_element;
        for i in 0..m {
            state.x_b[i] -= buf.alpha[i] * step;
        }
        state.x_b[r] = step;
        if entering_at_upper {
            let u_q = ubs[entering_col];
            debug_assert!(u_q.is_finite(), "at_upper entering must be finite");
            state.x_b[r] += u_q;
        }
        for val in state.x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // Reduced-cost increment: r_j_new = r_j − θ trow[j] for non-basic j.
        let leaving_col = state.basis[r];
        for j in 0..n_total {
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            if !state.is_basic[j] {
                state.reduced_costs[j] -= theta * buf.trow[j];
            }
        }
        if leaving_col < n_total {
            state.reduced_costs[leaving_col] = -theta;
        }

        // Basis bookkeeping.
        state.is_basic[entering_col] = true;
        state.at_upper[entering_col] = false;
        if leaving_col < n_total {
            state.is_basic[leaving_col] = false;
            state.at_upper[leaving_col] = false;
        }

        // Push the column swap through the LU.
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        let mut alpha_sv_for_update = alpha_sv;
        basis_mgr.ftran(&mut alpha_sv_for_update);
        basis_mgr.update(entering_col, r, &alpha_sv_for_update);
        state.basis[r] = entering_col;

        leaving.after_pivot(r, &buf.alpha, &buf.sigma, pivot_element);

        // Anti-cycling progress check.
        if !bland_mode {
            let current = leaving.progress_metric(&state.x_b, &state.basis);
            if made_progress_with_floor(best_infeas, current, 0.0) {
                best_infeas = current;
                iters_since_progress = 0;
            } else {
                iters_since_progress += 1;
                if iters_since_progress >= k_trigger {
                    bland_mode = true;
                }
            }
        }

        // Refactor + reduced-cost refresh on the LU's request (eta cap).
        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return (BoundedOutcome::SingularBasis, state);
                }
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if !compute_reduced_costs_into_timed(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
                options.deadline,
            ) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
        }
    }
}

/// FTRAN a column of `a` and dump into `out` (length `m`).
pub(super) fn ftran_column(a: &CscMatrix, basis_mgr: &mut LuBasis, col: usize, m: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), m);
    out.fill(0.0);
    let (rows, vals) = a.get_column(col).unwrap();
    for (i, &row) in rows.iter().enumerate() {
        out[row] = vals[i];
    }
    basis_mgr.ftran_dense(out);
}
