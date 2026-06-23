//! Primal simplex phases for bounded standard form.

use crate::linalg::timeout::deadline_reached;
use super::extract::bounded_obj;
use super::iterate::ftran_column;
use super::leaving::{
    bland_entering, select_leaving_bland_bounded, select_leaving_bounded, BoundedLeave,
};
use super::pricing::{partial_price_entering, PartialPrice};
use super::BoundedDualState;
use crate::basis::{BasisManager, LuBasis};
use crate::error::SolverError;
use crate::options::SolverOptions;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use std::sync::atomic::Ordering;

use super::super::super::dual_common::{
    compute_dual_vars_into, NO_PROGRESS_MIN, NO_PROGRESS_TRIGGER_FACTOR,
};
use super::super::super::pricing::{CAP_MULT_OF_M, GAMMA_FLOOR};
use super::super::super::standard_form::{BoundedStandardForm, SimplexOutcome};
use super::super::super::trace::IterTrace;
use super::super::bound_flip::bump_bfrt_flip_invocations;

#[cfg(test)]
thread_local! {
    static FORCE_BLAND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_primal_force_bland(v: bool) {
    FORCE_BLAND.with(|c| c.set(v));
}

#[cfg(test)]
fn primal_force_bland() -> bool {
    FORCE_BLAND.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn primal_force_bland() -> bool {
    false
}

#[cfg(test)]
thread_local! {
    static PRIMAL_ALPHA_SV_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_primal_alpha_sv_disabled(v: bool) {
    PRIMAL_ALPHA_SV_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn primal_alpha_sv_disabled() -> bool {
    PRIMAL_ALPHA_SV_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn primal_alpha_sv_disabled() -> bool {
    false
}

/// Drive primal Phase 2 from a primal-feasible `BoundedDualState`.
///
/// Caller supplies the state produced by `solve_bounded_dual` (perturbed-cost
/// dual phase) and the **original** cost vector `c`. The function minimizes
/// the original objective while maintaining primal feasibility, handling
/// variables at their upper bound via bounded-primal ratio test.
///
/// Pricing: non-basic at lb enters if `rc < 0`; non-basic at ub enters if
/// `rc > 0` (reversed, because decreasing from ub reduces the objective).
/// Ratio test: leaving variable hits either its lb or ub; entering variable
/// may flip to its opposite bound without a basis change (step = `u_q`).
///
/// Returns `(SimplexOutcome, BoundedDualState)` so the caller can extract the
/// solution and dual variables from the terminal state.
/// `ubs` must match the Ruiz-scaling space of `a` and `c` (pass
/// `&bsf.upper_bounds` for unscaled, or scaled bounds from `scale_upper_bounds`).
pub(crate) fn phase2_primal_bounded(
    bsf: &BoundedStandardForm,
    mut state: BoundedDualState,
    a: &CscMatrix,
    c: &[f64],
    options: &SolverOptions,
    iters: &mut usize,
    ubs: &[f64],
) -> (SimplexOutcome, BoundedDualState) {
    let m = bsf.m;
    let n_total = bsf.n_total;

    let timeout_obj = |state: &BoundedDualState| {
        SimplexOutcome::Timeout(bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        ))
    };
    if deadline_reached(options.deadline) {
        return (timeout_obj(&state), state);
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(SolverError::DeadlineExceeded) => return (timeout_obj(&state), state),
            Err(_) => return (SimplexOutcome::SingularBasis, state),
        };

    let mut y = vec![0.0f64; m];
    let mut rc = vec![0.0f64; n_total];
    let mut alpha = vec![0.0f64; m];
    let mut trace = IterTrace::new("bounded-primal");

    loop {
        *iters = iters.saturating_add(1);
        if deadline_reached(options.deadline) {
            return (
                SimplexOutcome::Timeout(bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                )),
                state,
            );
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
            t.log(*iters, obj, &state.basis, false);
        }

        if deadline_reached(options.deadline) {
            return (timeout_obj(&state), state);
        }
        compute_dual_vars_into(c, &mut basis_mgr, &state.basis, &mut y);

        let q = {
            let at_upper = &state.at_upper;
            match partial_price_entering(
                a,
                c,
                &state.is_basic,
                &y,
                &mut rc,
                n_total,
                state.price_start,
                options.deadline,
                |j, rc_j| {
                    let score = if at_upper[j] { rc_j } else { -rc_j };
                    (score > PIVOT_TOL).then_some(score)
                },
            ) {
                PartialPrice::Deadline => return (timeout_obj(&state), state),
                PartialPrice::Optimal { next_start } => {
                    state.price_start = next_start;
                    let obj = bounded_obj(
                        c,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs,
                    );
                    return (SimplexOutcome::Optimal(obj, y), state);
                }
                PartialPrice::Entering {
                    entering,
                    next_start,
                } => {
                    state.price_start = next_start;
                    entering
                }
            }
        };

        let from_ub = state.at_upper[q];
        let dir = if from_ub { -1.0f64 } else { 1.0 };

        ftran_column(a, &mut basis_mgr, q, m, &mut alpha);

        if deadline_reached(options.deadline) {
            return (
                SimplexOutcome::Timeout(bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                )),
                state,
            );
        }

        let ub_q = ubs[q];
        let (r, leaving_at_ub, theta) = match select_leaving_bounded(
            &alpha,
            dir,
            &state.x_b,
            &state.basis,
            ubs,
            ub_q,
            m,
            PIVOT_TOL,
            options.primal_tol,
            None,
        ) {
            BoundedLeave::Flip => {
                bump_bfrt_flip_invocations();
                for i in 0..m {
                    state.x_b[i] -= alpha[i] * dir * ub_q;
                }
                state.at_upper[q] = !from_ub;
                basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
                if basis_mgr.refactor_failed {
                    return if basis_mgr.singular_basis {
                        (SimplexOutcome::SingularBasis, state)
                    } else {
                        (
                            SimplexOutcome::Timeout(bounded_obj(
                                c,
                                &state.basis,
                                &state.x_b,
                                &state.at_upper,
                                &state.is_basic,
                                ubs,
                            )),
                            state,
                        )
                    };
                }
                continue;
            }
            BoundedLeave::Unbounded => return (SimplexOutcome::Unbounded, state),
            BoundedLeave::Pivot { row, at_ub, step } => (row, at_ub, step),
        };

        let leaving_col = state.basis[r];

        for i in 0..m {
            state.x_b[i] -= alpha[i] * dir * theta;
        }
        state.x_b[r] = if from_ub { ub_q - theta } else { theta };

        for v in state.x_b.iter_mut() {
            if v.abs() < options.clamp_tol {
                *v = 0.0;
            }
        }

        state.at_upper[leaving_col] = leaving_at_ub;
        state.at_upper[q] = false;
        state.is_basic[leaving_col] = false;
        state.is_basic[q] = true;
        state.basis[r] = q;

        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec { indices: vec![], values: vec![], len: m }
        } else {
            SparseVec::from_dense(&alpha)
        };
        basis_mgr.update(q, r, &alpha_sv);

        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    (SimplexOutcome::SingularBasis, state)
                } else {
                    (
                        SimplexOutcome::Timeout(bounded_obj(
                            c,
                            &state.basis,
                            &state.x_b,
                            &state.at_upper,
                            &state.is_basic,
                            ubs,
                        )),
                        state,
                    )
                };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn primal_simplex_aug(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
    art_threshold: Option<usize>,
) -> SimplexOutcome {
    let m = state.basis.len();
    let n_aug = state.at_upper.len();
    debug_assert_eq!(state.x_b.len(), m);
    debug_assert_eq!(state.is_basic.len(), n_aug);
    debug_assert_eq!(ubs_aug.len(), n_aug);
    debug_assert_eq!(c_aug.len(), n_aug);
    debug_assert!(n_struct <= n_aug);

    let timeout_obj = |st: &BoundedDualState| {
        SimplexOutcome::Timeout(bounded_obj(
            c_aug,
            &st.basis,
            &st.x_b,
            &st.at_upper,
            &st.is_basic,
            ubs_aug,
        ))
    };
    if deadline_reached(options.deadline) {
        return timeout_obj(state);
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a_aug, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(SolverError::DeadlineExceeded) => return timeout_obj(state),
            Err(_) => return SimplexOutcome::SingularBasis,
        };

    let mut y = vec![0.0f64; m];
    let mut rc = vec![0.0f64; n_struct];
    let mut alpha = vec![0.0f64; m];
    let mut devex_weights = vec![1.0f64; n_struct];
    let mut trace = IterTrace::new("bounded-aug-primal");

    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    let force_bland = primal_force_bland();
    let mut iters_since_progress: usize = 0;
    let mut bland_mode = force_bland;

    loop {
        *iters = iters.saturating_add(1);
        if deadline_reached(options.deadline)
            || options
                .cancel_flag
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
        {
            return timeout_obj(state);
        }

        if let Some(t) = trace.as_mut() {
            let obj = bounded_obj(
                c_aug,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs_aug,
            );
            t.log(*iters, obj, &state.basis, bland_mode);
        }

        if deadline_reached(options.deadline) {
            return timeout_obj(state);
        }
        compute_dual_vars_into(c_aug, &mut basis_mgr, &state.basis, &mut y);

        let q = if bland_mode {
            match bland_entering(
                a_aug,
                c_aug,
                &state.is_basic,
                &state.at_upper,
                &y,
                n_struct,
                PIVOT_TOL,
            ) {
                Some(j) => j,
                None => {
                    let obj = bounded_obj(
                        c_aug,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs_aug,
                    );
                    return SimplexOutcome::Optimal(obj, y);
                }
            }
        } else {
            let at_upper = &state.at_upper;
            match partial_price_entering(
                a_aug,
                c_aug,
                &state.is_basic,
                &y,
                &mut rc,
                n_struct,
                state.price_start,
                options.deadline,
                |j, rc_j| {
                    let violation = if at_upper[j] { rc_j } else { -rc_j };
                    if violation <= PIVOT_TOL {
                        return None;
                    }
                    let gamma = devex_weights[j].max(GAMMA_FLOOR);
                    Some(violation / gamma.sqrt())
                },
            ) {
                PartialPrice::Deadline => return timeout_obj(state),
                PartialPrice::Optimal { next_start } => {
                    state.price_start = next_start;
                    let obj = bounded_obj(
                        c_aug,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs_aug,
                    );
                    return SimplexOutcome::Optimal(obj, y);
                }
                PartialPrice::Entering {
                    entering,
                    next_start,
                } => {
                    state.price_start = next_start;
                    entering
                }
            }
        };

        let from_ub = state.at_upper[q];
        let dir = if from_ub { -1.0f64 } else { 1.0 };

        ftran_column(a_aug, &mut basis_mgr, q, m, &mut alpha);

        let ub_q = ubs_aug[q];
        let leave = if bland_mode {
            select_leaving_bland_bounded(
                &alpha,
                dir,
                &state.x_b,
                &state.basis,
                ubs_aug,
                ub_q,
                m,
                PIVOT_TOL,
            )
        } else {
            select_leaving_bounded(
                &alpha,
                dir,
                &state.x_b,
                &state.basis,
                ubs_aug,
                ub_q,
                m,
                PIVOT_TOL,
                options.primal_tol,
                art_threshold,
            )
        };
        let (r, leaving_at_ub, theta) = match leave {
            BoundedLeave::Flip => {
                bump_bfrt_flip_invocations();
                if let Some(t) = trace.as_mut() {
                    t.note_flip();
                }
                for i in 0..m {
                    state.x_b[i] -= alpha[i] * dir * ub_q;
                }
                state.at_upper[q] = !from_ub;
                iters_since_progress = 0;
                if !force_bland {
                    bland_mode = false;
                }
                basis_mgr.refactor_if_needed_timed(a_aug, &state.basis, options.deadline);
                if basis_mgr.refactor_failed {
                    return if basis_mgr.singular_basis {
                        SimplexOutcome::SingularBasis
                    } else {
                        timeout_obj(state)
                    };
                }
                continue;
            }
            BoundedLeave::Unbounded => return SimplexOutcome::Unbounded,
            BoundedLeave::Pivot { row, at_ub, step } => (row, at_ub, step),
        };
        if let Some(t) = trace.as_mut() {
            t.note_pivot(theta, options.primal_tol);
        }

        if theta > step_zero_threshold {
            iters_since_progress = 0;
            if !force_bland {
                bland_mode = false;
            }
        } else {
            iters_since_progress = iters_since_progress.saturating_add(1);
            if iters_since_progress >= k_trigger {
                bland_mode = true;
            }
        }
        let leaving_col = state.basis[r];

        for i in 0..m {
            state.x_b[i] -= alpha[i] * dir * theta;
        }
        state.x_b[r] = if from_ub { ub_q - theta } else { theta };
        for v in state.x_b.iter_mut() {
            if v.abs() < options.clamp_tol {
                *v = 0.0;
            }
        }

        state.at_upper[leaving_col] = leaving_at_ub;
        state.at_upper[q] = false;
        state.is_basic[leaving_col] = false;
        state.is_basic[q] = true;
        state.basis[r] = q;

        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec { indices: vec![], values: vec![], len: m }
        } else {
            SparseVec::from_dense(&alpha)
        };
        let norm_sq: f64 = alpha.iter().map(|&v| v * v).sum();
        let mut gamma_leaving = 1.0;
        if leaving_col < n_struct {
            gamma_leaving = devex_weights[leaving_col];
            let pivot = alpha[r];
            if pivot.abs() > PIVOT_TOL {
                let cap = CAP_MULT_OF_M * (m as f64).max(1.0);
                let new_weight = (norm_sq / (pivot * pivot))
                    .min(cap)
                    .max(GAMMA_FLOOR);
                devex_weights[leaving_col] = devex_weights[leaving_col].max(new_weight);
                gamma_leaving = devex_weights[leaving_col];
            }
        }
        if q < n_struct {
            devex_weights[q] = if norm_sq > GAMMA_FLOOR {
                (gamma_leaving / norm_sq).max(1.0)
            } else {
                1.0
            };
        }
        basis_mgr.update(q, r, &alpha_sv);

        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a_aug, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    SimplexOutcome::SingularBasis
                } else {
                    timeout_obj(state)
                };
            }
        }
    }
}
