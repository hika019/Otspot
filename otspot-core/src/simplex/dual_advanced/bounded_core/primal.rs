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
            t.log(*iters, obj, &state.basis, bland_mode);
        }
        if options
            .max_iters
            .is_some_and(|limit| *iters as u64 >= limit)
        {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            return (SimplexOutcome::Stalled(obj), state);
        }
        if iters.is_multiple_of(OBJ_PLATEAU_CHECK_INTERVAL) {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            if obj_plateau_should_bail(
                &mut best_obj,
                obj,
                &mut iters_since_obj_progress,
                OBJ_PLATEAU_CHECK_INTERVAL,
                giveup_obj_trigger,
            ) {
                return (SimplexOutcome::Stalled(obj), state);
            }
        }

        if deadline_reached(options.deadline) {
            return (timeout_obj(&state), state);
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
/// `phase2_primal_bounded` (identical bland_mode/Flip loop shape, both built
/// on `select_leaving_bland_bounded`), mirroring
/// `crate::simplex::primal::core`'s `BAIL_TRIGGER_FACTOR`/`BAIL_TRIGGER_MIN`
/// (Primal Phase I's cycling early-bail) by deliberate design symmetry:
/// `enable_phase1_cycling_bail` only arms that bail for Phase I, leaving these
/// bounded-primal paths — used for ordinary warm-started B&B node
/// relaxations, RINS/RENS/local-branching sub-MIP node relaxations, and
/// in-tree cut-separation re-solves alike — with no independent stop besides
/// the caller's wall-clock deadline.
///
/// Give up once the objective (`c_aug^T x_B`) has not meaningfully improved
/// for `K` consecutive iterations, sampled every [`OBJ_PLATEAU_CHECK_INTERVAL`]
/// iterations (not gated on `bland_mode` or the per-pivot step size): a
/// bounded-variable cycle can route through `BoundedLeave::Flip` steps, which
/// unconditionally reset both `bland_mode` and the step-plateau counter
/// without resolving the cycle, so a bland/step-gated check can be dodged
/// indefinitely by a cycle that flips periodically. The objective-plateau
/// check has no such escape hatch — a flip only "counts" as progress if it
/// actually decreases the objective by more than
/// [`OBJ_PLATEAU_PROGRESS_REL_TOL`].
///
/// Motivating measurements (pk1, MIPLIB), both predating this bail:
/// - With `select_leaving_bland_bounded`'s former `PIVOT_TOL`-sized ratio-tie
///   band, one B&B node relaxation revisited 592 distinct bases for
///   ~5,000,000 consecutive 100%-degenerate pivots with zero objective
///   progress.
/// - After tightening that tie band, two RENS sub-MIP node relaxations still
///   each ran 872,044 100%-degenerate pivots — interleaved with a `Flip`
///   roughly every 34 pivots, which is what motivated dropping the
///   `bland_mode`/step-based gate in favor of a plain objective-plateau check.
///
/// A give-up here returns the honest [`SimplexOutcome::Stalled`] (mapped by
/// `stop_status` to `SuboptimalSolution`/`MaxIterations`, never silently
/// reported as `Timeout`), not a symptom-hiding cap.
///
/// **Also arms during Phase I** (`primal_simplex_aug` called with
/// `art_threshold = Some(_)`, minimizing the artificial-variable sum): this
/// is a deliberate choice, not an oversight — a Phase I run that plateaus
/// without reducing the artificial sum to zero is exactly as unresolved as a
/// Phase II plateau, and needs the same bail so it cannot spin indefinitely
/// either. The risk this trades against is misreporting a *slow-but-still-
/// converging-toward-Infeasible* Phase I as `Stalled` (→
/// `SuboptimalSolution`/`MaxIterations`) instead of the correct `Infeasible`.
/// Measured: `data/lp_problems_infeas` (29 certified) and
/// `data/lp_problems_unbounded` (12 certified) both still classify 29/29 and
/// 12/12 correctly with this bail active, including `klein3` — the
/// adversarial cycling instance `crate::simplex::primal::core`'s own Phase I
/// bail cites as its origin case — which still resolves to `Infeasible`
/// (taking 14.9s, the bail never fires because the artificial-sum objective
/// keeps decreasing). No regression found; not proof no LP can ever trigger
/// this risk, but the two suites built specifically to exercise
/// Infeasible/Unbounded classification show no counterexample.
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
/// objective (`bounded_obj`), rather than every iteration.
///
/// `bounded_obj` is an `O(m)` dense pass over the current basis; computing it
/// on every iteration purely to feed a backstop that almost never fires (the
/// overwhelming majority of solves converge long before `giveup_obj_trigger`)
/// measured as a 2-10% wall-clock regression across MIPLIB problems that stay
/// nowhere near giving up (dcmulti/khb05250/markshare_4_0/p0201, each PASS in
/// a few hundred to a few thousand iterations). Sampling every `1024`
/// iterations amortizes that `O(m)` cost to effectively zero on the
/// non-cycling path (the ordinary case), while still bounding a genuine
/// stall to `giveup_obj_trigger + OBJ_PLATEAU_CHECK_INTERVAL` iterations — a
/// small, fixed overshoot against the millions of iterations this backstop
/// replaces (pk1, MIPLIB: 5,000,000+ and 872,044-pivot degenerate cycles
/// before this bail existed). The check remains keyed on `*iters`, not
/// wall-clock, so the stop point stays deterministic.
///
/// `*iters` is a single counter threaded across an entire pipeline call
/// (`dual_advanced::pipeline` declares it once and passes `&mut iters`
/// through Phase I *and* Phase II in sequence), not reset to 0 at the start
/// of `primal_simplex_aug` — only `best_obj`/`iters_since_obj_progress`
/// (local to each call) are fresh per phase. So the first sample a given
/// phase sees can land anywhere from 1 to `OBJ_PLATEAU_CHECK_INTERVAL`
/// iterations after that phase starts, depending on where the *previous*
/// phase left `*iters % OBJ_PLATEAU_CHECK_INTERVAL`. This only ever shortens
/// a phase's first sampling window, never lengthens it, so it biases toward
/// *earlier* detection — safe, not a correctness gap.
///
/// Lower-bound evidence that `1024`/`5_000` do not misfire on well-behaved
/// LPs: the full Netlib `data/lp_problems` suite (109/109, `--timeout 1000
/// --eps 1e-6`) passes unchanged with this bail active, including its
/// largest/slowest members (pilot87, dfl001, pds-20 — each hundreds of
/// seconds of real simplex work) — none plateau long enough to trip the
/// bail on a genuinely converging solve.
pub(super) const OBJ_PLATEAU_CHECK_INTERVAL: usize = 1_024;

/// Shared give-up progress update for `primal_simplex_aug` /
/// `phase2_primal_bounded`: records whether `current_obj` is a meaningful
/// improvement over `*best_obj` (updating it and resetting
/// `*iters_since_obj_progress` when so), and returns `true` once
/// `*iters_since_obj_progress` reaches `giveup_trigger` — the caller must
/// then return `SimplexOutcome::Stalled`.
///
/// `progress_check_interval` is the number of iterations this call is
/// standing in for (the caller only calls this every
/// [`OBJ_PLATEAU_CHECK_INTERVAL`] iterations — see its doc comment), so a
/// no-progress call advances the counter by that amount, not by 1; the
/// trigger threshold stays expressed in iteration units regardless of the
/// sampling interval.
///
/// Pulled out as its own pure function (rather than inlined per-loop) so it
/// is unit-testable against a synthetic no-progress sequence without needing
/// to hand-construct an LP that genuinely cycles.
///
/// `current_obj`/`*best_obj` non-finite (P3-4): a non-finite `current_obj`
/// cannot be assessed as an improvement (the `>` comparison against it is
/// `false` for NaN and typically unreachable for infinities of the same
/// sign, but relying on that IEEE-754 detail rather than stating it is
/// fragile), so it is explicitly treated as no-progress — the plateau
/// counter still advances and the call can still bail. A non-finite
/// `*best_obj` with a finite `current_obj` is explicitly treated as
/// progress (finite is strictly better than non-finite), recovering
/// `*best_obj` instead of latching onto a non-finite value that no future
/// `current_obj` could ever "improve" on via the relative-tolerance formula.
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
        if options
            .max_iters
            .is_some_and(|limit| *iters as u64 >= limit)
        {
            let obj = bounded_obj(
                c_aug,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs_aug,
            );
            return SimplexOutcome::Stalled(obj);
        }
        if iters.is_multiple_of(OBJ_PLATEAU_CHECK_INTERVAL) {
            let obj = bounded_obj(
                c_aug,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs_aug,
            );
            if obj_plateau_should_bail(
                &mut best_obj,
                obj,
                &mut iters_since_obj_progress,
                OBJ_PLATEAU_CHECK_INTERVAL,
                giveup_obj_trigger,
            ) {
                return SimplexOutcome::Stalled(obj);
            }
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
