//! Primal simplex phases for bounded standard form.

use super::extract::bounded_obj;
use super::iterate::ftran_column;
use super::leaving::{
    bland_entering, select_leaving_bland_bounded, select_leaving_bounded, BoundedLeave,
};
use super::pricing::{partial_price_entering, PartialPrice};
use super::BoundedDualState;
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::tolerances::PIVOT_TOL;
use otspot_num::linalg::timeout::deadline_reached;
use otspot_num::sparse::{CscMatrix, SparseVec};
use otspot_num::SolverError;
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
pub(crate) fn set_primal_force_bland(v: bool) -> bool {
    FORCE_BLAND.with(|c| c.replace(v))
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
pub(crate) fn set_primal_alpha_sv_disabled(v: bool) -> bool {
    PRIMAL_ALPHA_SV_DISABLE.with(|c| c.replace(v))
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

/// Per-iteration stop-condition check shared by `primal_simplex_aug` and
/// `phase2_primal_bounded`: deadline/cancel, trace log, `max_iters` cap, and
/// the (throttled) objective-plateau bail. Returns `Some(outcome)` when the
/// loop must return immediately, `None` to continue. `compute_obj` recomputes
/// `bounded_obj` for whichever cost/bounds vectors the caller is using.
///
/// `cancelled` is the caller's own `options.cancel_flag` check result (or
/// `false` to skip it): `phase2_primal_bounded` does not check it here, an
/// existing behavior difference this extraction preserves, not introduces.
#[allow(clippy::too_many_arguments)]
fn check_plateau_stop_conditions(
    iters: &mut usize,
    options: &SolverOptions,
    cancelled: bool,
    trace: &mut Option<IterTrace>,
    basis: &[usize],
    bland_mode: bool,
    best_obj: &mut f64,
    iters_since_obj_progress: &mut usize,
    giveup_obj_trigger: usize,
    compute_obj: impl Fn() -> f64,
) -> Option<SimplexOutcome> {
    *iters = iters.saturating_add(1);
    if deadline_reached(options.deadline) || cancelled {
        return Some(SimplexOutcome::Timeout(compute_obj()));
    }
    if let Some(t) = trace.as_mut() {
        t.log(*iters, compute_obj(), basis, bland_mode);
    }
    if options
        .max_iters
        .is_some_and(|limit| *iters as u64 >= limit)
    {
        return Some(SimplexOutcome::Stalled(compute_obj()));
    }
    if iters.is_multiple_of(OBJ_PLATEAU_CHECK_INTERVAL) {
        let obj = compute_obj();
        if obj_plateau_should_bail(
            best_obj,
            obj,
            iters_since_obj_progress,
            OBJ_PLATEAU_CHECK_INTERVAL,
            giveup_obj_trigger,
        ) {
            return Some(SimplexOutcome::Stalled(obj));
        }
    }
    if deadline_reached(options.deadline) {
        return Some(SimplexOutcome::Timeout(compute_obj()));
    }
    None
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
    let mut trace = IterTrace::new("bounded-phase2-primal");

    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    // Bland-mode give-up: see the doc comment on `OBJ_PLATEAU_BAIL_FACTOR` above
    // `primal_simplex_aug` — this loop has the identical bland_mode/Flip
    // structure (and calls the same `select_leaving_bland_bounded`), so it
    // shares the same give-up mechanism and constants.
    let giveup_obj_trigger = (OBJ_PLATEAU_BAIL_FACTOR * m).max(OBJ_PLATEAU_BAIL_MIN);
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    let force_bland = primal_force_bland();
    let mut iters_since_progress: usize = 0;
    let mut iters_since_obj_progress: usize = 0;
    let mut best_obj: f64 = bounded_obj(
        c,
        &state.basis,
        &state.x_b,
        &state.at_upper,
        &state.is_basic,
        ubs,
    );
    let mut bland_mode = force_bland;

    loop {
        if let Some(outcome) = check_plateau_stop_conditions(
            iters,
            options,
            false,
            &mut trace,
            &state.basis,
            bland_mode,
            &mut best_obj,
            &mut iters_since_obj_progress,
            giveup_obj_trigger,
            || {
                bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                )
            },
        ) {
            return (outcome, state);
        }

        compute_dual_vars_into(c, &mut basis_mgr, &state.basis, &mut y);

        let q = if bland_mode {
            match bland_entering(
                a,
                c,
                &state.is_basic,
                &state.at_upper,
                &y,
                n_total,
                PIVOT_TOL,
            ) {
                Some(j) => j,
                None => {
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
            }
        } else {
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
        let leave = if bland_mode {
            select_leaving_bland_bounded(
                &alpha,
                dir,
                &state.x_b,
                &state.basis,
                ubs,
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
                ubs,
                ub_q,
                m,
                PIVOT_TOL,
                options.primal_tol,
                None,
            )
        };
        let (r, leaving_at_ub, theta) = match leave {
            BoundedLeave::Flip => {
                bump_bfrt_flip_invocations();
                for i in 0..m {
                    state.x_b[i] -= alpha[i] * dir * ub_q;
                }
                state.at_upper[q] = !from_ub;
                iters_since_progress = 0;
                if !force_bland {
                    bland_mode = false;
                }
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

        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec::from_raw_parts(vec![], vec![], m)
        } else {
            SparseVec::from_dense(&alpha)
        };
        match basis_mgr.update(q, r, &alpha_sv) {
            Ok(()) => {}
            Err(otspot_num::SolverError::SingularBasis { .. }) => {
                return (SimplexOutcome::SingularBasis, state);
            }
            Err(err) => panic!("internal bounded-primal eta invariant violated: {err}"),
        }

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

/// Objective-plateau bail trigger shared by `primal_simplex_aug` and
/// `phase2_primal_bounded`, mirroring `crate::simplex::primal::core`'s
/// `BAIL_TRIGGER_FACTOR`/`BAIL_TRIGGER_MIN` (Primal Phase I's cycling
/// early-bail): these bounded-primal paths (ordinary B&B node relaxations,
/// RINS/RENS/local-branching sub-MIPs, in-tree cut re-solves) otherwise have
/// no independent stop besides the caller's wall-clock deadline.
///
/// Give up once the objective has not meaningfully improved for `K`
/// iterations, sampled every [`OBJ_PLATEAU_CHECK_INTERVAL`] iterations —
/// unconditional on `bland_mode`/step size, since a cycle can route through
/// `BoundedLeave::Flip` (which unconditionally resets both) without
/// resolving, dodging a bland/step-gated check indefinitely. A flip only
/// counts as progress if it clears [`OBJ_PLATEAU_PROGRESS_REL_TOL`].
///
/// **Also arms during Phase I** (`art_threshold = Some(_)`): deliberate, not
/// an oversight — an unresolved Phase I plateau needs the same bail. Risk
/// traded: misreporting a slow-but-converging-toward-Infeasible Phase I as
/// `Stalled` instead of `Infeasible`. Measured no regression on
/// `lp_problems_infeas` (29/29) / `lp_problems_unbounded` (12/12), including
/// `klein3` (the adversarial case Phase I's own bail cites as its origin).
///
/// Returns the honest [`SimplexOutcome::Stalled`] (→
/// `SuboptimalSolution`/`MaxIterations` via `stop_status`), never a silent
/// `Timeout`.
const OBJ_PLATEAU_BAIL_FACTOR: usize = 10;
const OBJ_PLATEAU_BAIL_MIN: usize = 5_000;

/// Relative objective-improvement floor for [`OBJ_PLATEAU_BAIL_FACTOR`]'s progress
/// check — deliberately looser than `dual_common::NO_PROGRESS_REL_EPS`
/// (1e-12, calibrated for detecting genuine-but-tiny per-pivot progress
/// elsewhere). At `1e-12` relative, floating-point noise in `bounded_obj`'s
/// repeated summation across hundreds of thousands of iterations crosses the
/// threshold often enough to reset the give-up counter indefinitely without
/// any real progress (this is what let the 872,044-pivot RENS calls above
/// evade a `NO_PROGRESS_REL_EPS`-scale check); `1e-9` is 1,000x looser, well
/// above plausible summation noise for problems in scope, while still 1,000x
/// tighter than `OBJ_MATCH_REL_TOL` (1e-4, a solution-acceptance tolerance,
/// not an anti-cycling one).
pub(super) const OBJ_PLATEAU_PROGRESS_REL_TOL: f64 = 1e-9;

/// Iteration interval at which the give-up progress check samples the
/// objective (`bounded_obj`, an `O(m)` dense pass), rather than every
/// iteration. Computing it every iteration to feed a backstop that almost
/// never fires measured as a 2-10% wall-clock regression on MIPLIB problems
/// nowhere near giving up (dcmulti/khb05250/markshare_4_0/p0201). Sampling
/// every `1024` amortizes that cost to near-zero on the non-cycling path,
/// while still bounding a genuine stall to `giveup_obj_trigger +
/// OBJ_PLATEAU_CHECK_INTERVAL` iterations — keyed on `*iters`, not
/// wall-clock, so the stop point stays deterministic.
///
/// `*iters` is threaded across an entire pipeline call (Phase I then Phase
/// II share one counter; only `best_obj`/`iters_since_obj_progress` reset
/// per phase), so a phase's first sample can land 1..=1024 iterations in —
/// this only ever shortens the first window, biasing toward earlier
/// detection (safe, not a correctness gap).
///
/// Lower-bound evidence `1024`/`5_000` don't misfire on well-behaved LPs:
/// the full Netlib `data/lp_problems` suite (109/109) passes unchanged with
/// this bail active, including its slowest members (pilot87, dfl001,
/// pds-20).
pub(super) const OBJ_PLATEAU_CHECK_INTERVAL: usize = 1_024;

/// Shared give-up progress update for `primal_simplex_aug` /
/// `phase2_primal_bounded`: records whether `current_obj` improves on
/// `*best_obj` (resetting `*iters_since_obj_progress` when so), returning
/// `true` once it reaches `giveup_trigger` — the caller then returns
/// `SimplexOutcome::Stalled`. `progress_check_interval` is the iteration
/// count this call stands in for (the caller only samples every
/// [`OBJ_PLATEAU_CHECK_INTERVAL`] iterations), so a no-progress call
/// advances the counter by that amount, not by 1.
///
/// Pulled out as a pure function so it is unit-testable against a synthetic
/// no-progress sequence without hand-constructing a genuinely cycling LP.
///
/// Non-finite `current_obj`/`*best_obj`: a non-finite `current_obj` is
/// explicitly treated as no-progress (not relying on NaN comparisons being
/// `false`); a non-finite `*best_obj` with finite `current_obj` is treated
/// as progress, recovering `*best_obj` rather than latching onto a value no
/// future `current_obj` could ever "improve" on.
pub(super) fn obj_plateau_should_bail(
    best_obj: &mut f64,
    current_obj: f64,
    iters_since_obj_progress: &mut usize,
    progress_check_interval: usize,
    giveup_trigger: usize,
) -> bool {
    let improved = current_obj.is_finite()
        && (!best_obj.is_finite()
            || *best_obj - current_obj > best_obj.abs().max(1.0) * OBJ_PLATEAU_PROGRESS_REL_TOL);
    if improved {
        *best_obj = current_obj;
        *iters_since_obj_progress = 0;
        false
    } else {
        *iters_since_obj_progress =
            iters_since_obj_progress.saturating_add(progress_check_interval);
        *iters_since_obj_progress >= giveup_trigger
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
    let giveup_obj_trigger = (OBJ_PLATEAU_BAIL_FACTOR * m).max(OBJ_PLATEAU_BAIL_MIN);
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    let force_bland = primal_force_bland();
    let mut iters_since_progress: usize = 0;
    let mut iters_since_obj_progress: usize = 0;
    let mut best_obj: f64 = bounded_obj(
        c_aug,
        &state.basis,
        &state.x_b,
        &state.at_upper,
        &state.is_basic,
        ubs_aug,
    );
    let mut bland_mode = force_bland;

    loop {
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if let Some(outcome) = check_plateau_stop_conditions(
            iters,
            options,
            cancelled,
            &mut trace,
            &state.basis,
            bland_mode,
            &mut best_obj,
            &mut iters_since_obj_progress,
            giveup_obj_trigger,
            || {
                bounded_obj(
                    c_aug,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs_aug,
                )
            },
        ) {
            return outcome;
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

        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec::from_raw_parts(vec![], vec![], m)
        } else {
            SparseVec::from_dense(&alpha)
        };
        match basis_mgr.update(q, r, &alpha_sv) {
            Ok(()) => {}
            Err(otspot_num::SolverError::SingularBasis { .. }) => {
                return SimplexOutcome::SingularBasis;
            }
            Err(err) => panic!("internal augmented-primal eta invariant violated: {err}"),
        }

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

        let norm_sq: f64 = alpha.iter().map(|&v| v * v).sum();
        let mut gamma_leaving = 1.0;
        if leaving_col < n_struct {
            gamma_leaving = devex_weights[leaving_col];
            let pivot = alpha[r];
            if pivot.abs() > PIVOT_TOL {
                let cap = CAP_MULT_OF_M * (m as f64).max(1.0);
                let new_weight = (norm_sq / (pivot * pivot)).min(cap).max(GAMMA_FLOOR);
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
