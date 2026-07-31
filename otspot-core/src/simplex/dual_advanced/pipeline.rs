//! Phase pipeline and cold-start driver for dual advanced solver.

use super::super::dual_common::{basic_obj, lp_unbounded_ray_verified};
use super::bounded_core::{
    bounded_primal_phase1, bounded_primal_phase2_aug, extract_dual_info_bounded,
    extract_solution_bounded, phase2_primal_bounded, BoundedDualState, BoundedOutcome,
};
use super::{
    bounded_obj_from_state, fallback_profile_enabled, make_leaving_strategy,
    maybe_perturb_initial_xb, reconcile_bounded_terminal_state, BoundedTerminalReconcile,
    PHASE1_BOUND_VIOLATION_FALLBACKS, UB_VIOLATION_FALLBACKS,
};
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use otspot_num::sparse::CscMatrix;
use std::sync::atomic::Ordering;

use super::BoundedStandardForm;

/// Builds the honest, clock-rechecked `SolverResult` for a bounded-path
/// internal dead-end (deadline timeout or iteration stall): status is
/// [`super::super::stop_status`], never a raw [`SolveStatus::Timeout`]
/// literal. `stop_status` re-derives the truth from `options` at report
/// time, so a genuine deadline/cancel stop still classifies as `Timeout`;
/// any internal-only dead-end (including one that arrived carrying a stale
/// `SimplexOutcome::Timeout`/`BoundedTerminalReconcile::Timeout` variant) is
/// honestly downgraded to [`SolveStatus::SuboptimalSolution`] / [`SolveStatus::
/// MaxIterations`] instead. A `BoundedTerminalReconcile::BoundViolation`
/// (no verified basis, hence no honest solution/objective to report) is
/// *not* routed here — it maps straight to `SolverResult::numerical_error()`,
/// mirroring `SingularBasis`.
fn honest_stall_result(
    objective: f64,
    solution: Vec<f64>,
    iterations: usize,
    options: &SolverOptions,
) -> SolverResult {
    let status = super::super::stop_status(!solution.is_empty(), options);
    SolverResult {
        status,
        objective,
        solution,
        iterations,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_phase1_then_phase2<F>(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    state_factory: F,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> Option<SolverResult>
where
    F: FnOnce() -> (
        CscMatrix,
        Vec<Option<usize>>,
        Vec<f64>,
        Vec<usize>,
        Vec<bool>,
        Vec<f64>,
    ),
{
    fn mark_eq_ub_path(mut r: SolverResult) -> SolverResult {
        r.stats.bounded_eq_ub_path = true;
        r
    }

    // Distinct name from `honest_stall_result` (the function it wraps): this
    // closure additionally applies `mark_eq_ub_path`, so it is not a mere
    // alias — collapsing the names invited confusing them at call sites.
    let mark_eq_ub_honest_stall = |objective: f64, solution: Vec<f64>, iters: usize| {
        mark_eq_ub_path(honest_stall_result(objective, solution, iters, options))
    };

    let (a_aug, art_col_of_row, mut ubs_aug, basis, is_basic, mut x_b) = state_factory();
    let n_aug = a_aug.ncols();
    maybe_perturb_initial_xb(&mut x_b);
    let mut state = BoundedDualState {
        basis,
        at_upper: vec![false; n_aug],
        x_b,
        reduced_costs: vec![0.0; n_aug],
        is_basic,
        iterations: 0,
        price_start: 0,
    };

    // Phase I: minimise sum of artificials. Structural cost = 0.
    let mut c_p1 = vec![0.0f64; n_aug];
    for col in art_col_of_row.iter().flatten() {
        c_p1[*col] = 1.0;
    }
    let mut iters: usize = 0;
    let p1_out = bounded_primal_phase1(
        &a_aug,
        &c_p1,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p1_out {
        SimplexOutcome::SingularBasis => {
            return Some(mark_eq_ub_path(SolverResult::numerical_error()));
        }
        SimplexOutcome::Unbounded => {
            return None;
        }
        SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            return Some(mark_eq_ub_honest_stall(bsf.obj_offset, solution, iters));
        }
        SimplexOutcome::Optimal(_, _) => {
            let art_sum = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p1, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(_) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_honest_stall(bsf.obj_offset, solution, iters));
                }
                BoundedTerminalReconcile::BoundViolation => {
                    if fallback_profile_enabled() {
                        PHASE1_BOUND_VIOLATION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    }
                    return None;
                }
                BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            if art_sum > options.primal_tol {
                let mut r = SolverResult::infeasible();
                r.iterations = iters;
                return Some(mark_eq_ub_path(r));
            }
        }
    }

    // Pin artificials to ub = 0 for Phase II.
    for col in art_col_of_row.iter().flatten() {
        ubs_aug[*col] = 0.0;
    }

    // Phase II: minimise true objective on augmented matrix.
    let mut c_p2 = vec![0.0f64; n_aug];
    c_p2[..bsf.n_total].copy_from_slice(c);
    let p2_out = bounded_primal_phase2_aug(
        &a_aug,
        &c_p2,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p2_out {
        SimplexOutcome::Optimal(_, y) => {
            let obj = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p2, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(obj) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_honest_stall(
                        obj + bsf.obj_offset,
                        solution,
                        iters,
                    ));
                }
                BoundedTerminalReconcile::BoundViolation => {
                    // A bound violation on reconciliation is a numerical dead-end
                    // (eta-drift / ill-conditioning) with no verified basis to
                    // report a solution/objective from — the same honesty as
                    // `SingularBasis` below, not a stall with a diagnostic
                    // iterate. Never a deadline event by itself: it must not
                    // self-report as Timeout.
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
                BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            let ws = if state.basis.iter().all(|&j| j < bsf.n_total) {
                Some(WarmStartBasis {
                    basis: state.basis.clone(),
                    x_b: state.x_b.clone(),
                })
            } else {
                None
            };
            Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: ws,
                iterations: iters,
                ..Default::default()
            }))
        }
        SimplexOutcome::Unbounded => Some(mark_eq_ub_path(SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            iterations: iters,
            ..Default::default()
        })),
        SimplexOutcome::Timeout(obj) | SimplexOutcome::Stalled(obj) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            Some(mark_eq_ub_honest_stall(
                obj + bsf.obj_offset,
                solution,
                iters,
            ))
        }
        SimplexOutcome::SingularBasis => Some(mark_eq_ub_path(SolverResult::numerical_error())),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_bounded(
    dual_out: BoundedOutcome,
    dual_state: BoundedDualState,
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
    ubs: &[f64],
    problem: &LpProblem,
    options: &SolverOptions,
    total_iters: &mut usize,
) -> Option<SolverResult> {
    match dual_out {
        BoundedOutcome::UbViolationOutOfScope { .. } => {
            if fallback_profile_enabled() {
                UB_VIOLATION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
        BoundedOutcome::Unbounded => Some(SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        }),
        BoundedOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &dual_state, col_scale);
            Some(honest_stall_result(
                obj + bsf.obj_offset,
                solution,
                *total_iters,
                options,
            ))
        }
        BoundedOutcome::SingularBasis => Some(SolverResult::numerical_error()),
        BoundedOutcome::Optimal(_, _) => {
            let (p2_out, mut p2_state) =
                phase2_primal_bounded(bsf, dual_state, a, c, options, total_iters, ubs);
            let p2_out = match p2_out {
                SimplexOutcome::Optimal(_, y) => {
                    let pre_reconcile_x_b = p2_state.x_b.clone();
                    match reconcile_bounded_terminal_state(a, b, c, ubs, &mut p2_state, options) {
                        BoundedTerminalReconcile::Optimal(obj) => SimplexOutcome::Optimal(obj, y),
                        BoundedTerminalReconcile::Timeout(obj) => SimplexOutcome::Timeout(obj),
                        BoundedTerminalReconcile::BoundViolation => {
                            p2_state.x_b = pre_reconcile_x_b;
                            let obj = bounded_obj_from_state(c, ubs, &p2_state);
                            SimplexOutcome::Timeout(obj)
                        }
                        BoundedTerminalReconcile::SingularBasis => SimplexOutcome::SingularBasis,
                    }
                }
                other => other,
            };
            Some(finish_bounded_phase2(
                p2_out,
                p2_state,
                bsf,
                col_scale,
                row_scale,
                problem,
                *total_iters,
                options,
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_bounded_phase2(
    out: SimplexOutcome,
    state: BoundedDualState,
    bsf: &BoundedStandardForm,
    col_scale: &[f64],
    row_scale: &[f64],
    problem: &LpProblem,
    total_iters: usize,
    options: &SolverOptions,
) -> SolverResult {
    match out {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: state.basis,
                x_b: state.x_b,
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        },
        SimplexOutcome::Timeout(obj) | SimplexOutcome::Stalled(obj) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            honest_stall_result(obj + bsf.obj_offset, solution, total_iters, options)
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

// Test-only observability: lets a test force the Phase II ray-verification
// below to report "unverified" on a call that is genuinely, reachably
// unbounded (so the underlying `revised_simplex_core` exit and basis are
// real, not synthetic) — proving `cold_start_advanced`'s wiring reacts to a
// failed verification, independent of whether real eta-drift can be
// manufactured deterministically in a small fixture. `#[cfg(test)]`-only,
// zero production footprint.
#[cfg(test)]
thread_local! {
    static FORCE_RAY_UNVERIFIED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_force_ray_unverified() -> bool {
    FORCE_RAY_UNVERIFIED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
#[inline(always)]
fn test_force_ray_unverified() -> bool {
    false
}

/// Le-only cold startでHarris Dual Simplexを使用する
#[allow(clippy::too_many_arguments)]
pub(super) fn cold_start_advanced(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;

    let mut basis = sf.initial_basis.clone();
    let mut x_b = b.to_vec();

    // コスト摂動: c̃_j = max(c_j, 0) → スラック基底（y=0）で r̃_j = c̃_j ≥ 0 → 双対実行可能
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut leaving = make_leaving_strategy(options.dual_pricing, m);

    let mut total_iters: usize = 0;
    let phase1_outcome = super::core::dual_simplex_core_advanced(
        a,
        &mut x_b,
        &c_perturbed,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        false,
        options,
        leaving.as_mut(),
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            return SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            };
        }
        SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
            return super::super::stop_result_with_incumbent(
                sf,
                problem,
                &basis,
                &x_b,
                col_scale,
                total_iters,
                options,
            );
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {}
    }

    // Phase 2: 元のコストで主実行可能点からPrimal Simplexで最適化
    use super::super::pricing::SteepestEdgePricing;
    let mut pricing = SteepestEdgePricing::new(sf.n_total);
    let phase2_outcome = super::super::revised_simplex_core(
        a,
        &mut x_b,
        c,
        b,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        &mut pricing,
        options,
        &mut total_iters,
        false,
        None,
        false,
        None,
    );

    // Gate a Phase II `Unbounded` on a re-derived recession ray (same verified
    // gate as `dual.rs`'s primal Phase II path and the Big-M path). This is a
    // genuine primal-simplex unboundedness exit (`revised_simplex_core`), so
    // the ray oracle applies directly: an eta-drift false-Unbounded becomes
    // an honest Stalled instead of a wrong `SolveStatus::Unbounded` verdict.
    let phase2_outcome = if matches!(phase2_outcome, SimplexOutcome::Unbounded)
        && (test_force_ray_unverified()
            || !lp_unbounded_ray_verified(a, &basis, c, m, sf.n_total, sf.n_total, options))
    {
        SimplexOutcome::Stalled(basic_obj(c, &basis, &x_b))
    } else {
        phase2_outcome
    };

    let mut result = match phase2_outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: basis.to_vec(),
                x_b: x_b.to_vec(),
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        },
        SimplexOutcome::Timeout(obj) | SimplexOutcome::Stalled(obj) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            honest_stall_result(obj + sf.obj_offset, solution, total_iters, options)
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    };
    result.iterations = total_iters;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentinel (honesty fix, mirrors `dual_common::
    /// outcome_to_result_clock_rechecks_timeout_variant`'s "clock-recheck,
    /// not variant trust" pattern): `honest_stall_result` is the single
    /// choke point every bounded-path internal dead-end (deadline timeout,
    /// iteration stall, *and* a `BoundedTerminalReconcile::BoundViolation`
    /// numerical reconciliation failure) now funnels through, replacing the
    /// raw `SolveStatus::Timeout` literals previously minted separately at
    /// each of `run_phase1_then_phase2`'s three call sites and
    /// `finish_bounded`'s. With `SolverOptions::default()` (no deadline, no
    /// cancel flag) — i.e. no external stop condition ever active — an
    /// internal-only dead-end must never self-report as `Timeout`, whether
    /// or not an incumbent solution is present.
    ///
    /// Sentinel: reverting `honest_stall_result`'s body to
    /// `SolverResult { status: SolveStatus::Timeout, ... }` makes both
    /// assertions below fail.
    #[test]
    fn honest_stall_result_clock_rechecks_timeout_variant() {
        let options = SolverOptions::default();
        assert!(
            options.deadline.is_none() && options.cancel_flag.is_none(),
            "test premise: no external stop condition must be active"
        );

        let with_incumbent = honest_stall_result(1.0, vec![0.5, 1.5], 7, &options);
        assert_eq!(
            with_incumbent.status,
            SolveStatus::SuboptimalSolution,
            "a non-empty solution with no deadline/cancel must downgrade to \
             SuboptimalSolution, not self-report as Timeout; got {:?}",
            with_incumbent.status
        );

        let without_incumbent = honest_stall_result(f64::INFINITY, vec![], 3, &options);
        assert_eq!(
            without_incumbent.status,
            SolveStatus::MaxIterations,
            "an empty solution with no deadline/cancel must downgrade to \
             MaxIterations, not self-report as Timeout; got {:?}",
            without_incumbent.status
        );
    }

    /// A genuine, already-expired deadline must still classify as `Timeout`
    /// through the same choke point — `honest_stall_result` clock-rechecks,
    /// it does not blanket-deny `Timeout`.
    #[test]
    fn honest_stall_result_honours_genuine_deadline() {
        let options = SolverOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
            ..SolverOptions::default()
        };
        let result = honest_stall_result(1.0, vec![0.5], 1, &options);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "an already-expired deadline must still report Timeout; got {:?}",
            result.status
        );
    }

    /// Resets `FORCE_RAY_UNVERIFIED` to `false` on drop (including on
    /// panic), so a failing assertion can never leak the flag into whichever
    /// other test happens to reuse this thread next.
    struct ForceRayUnverifiedGuard;

    impl ForceRayUnverifiedGuard {
        fn new() -> Self {
            FORCE_RAY_UNVERIFIED.with(|f| f.set(true));
            Self
        }
    }

    impl Drop for ForceRayUnverifiedGuard {
        fn drop(&mut self) {
            FORCE_RAY_UNVERIFIED.with(|f| f.set(false));
        }
    }

    /// P2-2 (Codex/Opus review): `cold_start_advanced`'s Phase II
    /// `SimplexOutcome::Unbounded` exit is a genuine primal-simplex
    /// unboundedness witness (`revised_simplex_core` maintains primal
    /// feasibility throughout, unlike the dual loop's dual-feasibility
    /// invariant), so `lp_unbounded_ray_verified` — the same oracle
    /// `dual.rs:194-200` already applies to this exact kind of exit — is
    /// mathematically applicable here, unlike the dual "no ratio-test
    /// candidate" exit reverted from this branch.
    ///
    /// Reachable fixture: `x1 - x2 <= 1, x1,x2 >= 0`, minimize `-x1-x2`.
    /// Genuinely unbounded (`x1=x2=t -> -inf`) and Le-only (no artificial
    /// needed), so this reaches `cold_start_advanced`'s real Phase II exit
    /// through actual `revised_simplex_core` execution — no synthetic
    /// basis/matrix injection. `FORCE_RAY_UNVERIFIED` only substitutes the
    /// *verification outcome*, proving the wiring reacts correctly without
    /// requiring real eta-drift to be manufactured deterministically.
    ///
    /// Sentinel: reverting the gate (dropping the `test_force_ray_unverified()
    /// ||` disjunct, or the whole gate) makes the forced-flag assertion fail
    /// (status stays `Unbounded` instead of downgrading).
    #[test]
    fn cold_start_phase2_unbounded_gate_reacts_to_forced_unverified_ray() {
        use crate::problem::ConstraintType;
        use crate::simplex::standard_form::build_standard_form;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, -1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        assert_eq!(
            sf.num_artificial, 0,
            "test premise: Le-only LP must need no artificial (cold_start_advanced dispatch)"
        );
        let a_m = sf.a.clone();
        let b = sf.b.clone();
        let c = sf.c.clone();
        let row_scale = vec![1.0; sf.m];
        let col_scale = vec![1.0; sf.n_total];
        let options = SolverOptions::default();

        let baseline =
            cold_start_advanced(&sf, &lp, &options, &a_m, &b, &c, &row_scale, &col_scale);
        assert_eq!(
            baseline.status,
            SolveStatus::Unbounded,
            "test premise: this LP must genuinely verify as Unbounded via real execution; \
             got {:?}",
            baseline.status
        );

        let _guard = ForceRayUnverifiedGuard::new();
        let forced = cold_start_advanced(&sf, &lp, &options, &a_m, &b, &c, &row_scale, &col_scale);
        assert_ne!(
            forced.status,
            SolveStatus::Unbounded,
            "an unverified ray must not self-report as Unbounded; got {:?}",
            forced.status
        );
    }
}
