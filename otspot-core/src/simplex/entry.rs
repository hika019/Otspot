//! Simplex public entry + presolve/postsolve orchestration + method dispatch.

use crate::options::{SimplexMethod, SolverOptions, WarmStartBasis};
use crate::presolve;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::qp::certificate::guard_lp_optimal;
use crate::tolerances::PIVOT_TOL;

use super::dual;
use super::dual_advanced;
use super::primal::two_phase_simplex;
use super::standard_form::build_standard_form_with_deadline;

// Test-only hook: forces the was_reduced=true branch to treat the reduced solve
// as if it returned Timeout with a reduced-space solution, bypassing wall-clock.
// This lets sentinels verify the early-return contract deterministically.
#[cfg(test)]
thread_local! {
    static INJECT_REDUCED_TIMEOUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Iteration count the test hook stamps onto the injected reduced-space Timeout,
/// so a sentinel can assert the early-return carries `iterations` through. A 0
/// here would re-introduce the `iters=0` reporting artifact.
#[cfg(test)]
const REDUCED_TIMEOUT_INJECT_ITERS: usize = 7919;

/// Solve an LP with default options (raw simplex, without obj_offset).
///
/// Use [`crate::solve`] for the full pipeline including `obj_offset`.
#[cfg(test)]
pub(crate) fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// Solve an LP with the supplied options. When `options.presolve` is set,
/// presolve runs before the simplex.
///
/// Returns [`SolveStatus::NumericalError`] immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub(crate) fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    if options.validate().is_err() {
        return SolverResult::numerical_error();
    }
    // timeout_secs → deadline (mirrors qp_solve_impl).
    let mut opts_with_deadline;
    let options = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
        opts_with_deadline = options.clone();
        opts_with_deadline.deadline =
            Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(secs));
        &opts_with_deadline
    } else {
        options
    };

    let prof_t0 = std::time::Instant::now();
    // Presolve elapsed time when presolve ran but did not reduce the problem.
    // Used to set timing_breakdown on the fallthrough solve_without_presolve path.
    let mut non_reduced_presolve_us: Option<u64> = None;

    if options.presolve {
        match presolve::run_presolve(problem, options.deadline) {
            Err(presolve::PresolveStatus::Infeasible) => {
                let presolve_us = prof_t0.elapsed().as_micros() as u64;
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: f64::INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    timing_breakdown: Some(crate::problem::TimingBreakdown {
                        presolve_us,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
            }
            Err(presolve::PresolveStatus::Unbounded) => {
                let presolve_us = prof_t0.elapsed().as_micros() as u64;
                return SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    timing_breakdown: Some(crate::problem::TimingBreakdown {
                        presolve_us,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
            }
            Ok(presolve_result) if presolve_result.was_reduced => {
                // Presolve renumbers variables, so a supplied warm_start is invalidated.
                let opts_no_ws = if options.warm_start.is_some() {
                    let mut o = options.clone();
                    o.warm_start = None;
                    o.presolve = false;
                    Some(o)
                } else {
                    None
                };
                let eff_opts = opts_no_ws.as_ref().unwrap_or(options);
                let t_presolve_done = std::time::Instant::now();
                let presolve_us = t_presolve_done.duration_since(prof_t0).as_micros() as u64;
                let raw = solve_without_presolve(&presolve_result.reduced_problem, eff_opts);
                let t_solve_done = std::time::Instant::now();
                let solve_us = t_solve_done.duration_since(t_presolve_done).as_micros() as u64;
                // Test hook: override raw with a Timeout carrying a reduced-space solution.
                #[cfg(test)]
                let raw = if INJECT_REDUCED_TIMEOUT.with(|v| v.get()) {
                    SolverResult {
                        status: SolveStatus::Timeout,
                        solution: vec![0.0; presolve_result.reduced_problem.num_vars],
                        iterations: REDUCED_TIMEOUT_INJECT_ITERS,
                        ..Default::default()
                    }
                } else {
                    raw
                };
                let deadline_expired = eff_opts
                    .deadline
                    .is_some_and(|d| std::time::Instant::now() >= d);
                // In tests the hook also bypasses the wall-clock deadline check so
                // the sentinel doesn't depend on timing.
                #[cfg(test)]
                let deadline_expired =
                    deadline_expired || INJECT_REDUCED_TIMEOUT.with(|v| v.get());
                if raw.status == SolveStatus::Timeout && deadline_expired {
                    // The reduced solve timed out and the deadline is exhausted.
                    // `raw.solution` is in the *reduced* variable space — propagating
                    // it would violate the SolverResult contract (solution must be in
                    // the original variable space or empty).  Return an empty Timeout
                    // result, consistent with the Infeasible/Unbounded early-returns.
                    // `iterations` is reduced-space-independent diagnostic metadata, so
                    // it IS carried over: dropping it reports a misleading `iters=0` for
                    // a solve that actually ran many pivots (masks "slow" vs "stuck").
                    return SolverResult {
                        status: SolveStatus::Timeout,
                        objective: f64::INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        iterations: raw.iterations,
                        timing_breakdown: Some(crate::problem::TimingBreakdown {
                            presolve_us,
                            solve_us,
                            postsolve_us: 0,
                            ..Default::default()
                        }),
                        ..Default::default()
                    };
                }
                // The reduced LP can be unsolvable while the original is fine
                // (SingularBasis on the reduced initial basis, Eq drift in Phase II,
                // or guard_lp_optimal catching a KKT failure on the reduced form).
                // SuboptimalSolution from the guard means KKT failed → fall back.
                // Strip warm_start if present: a stale basis passed to the
                // cold-start retry of the original LP can cause cycling.
                if matches!(
                    raw.status,
                    SolveStatus::NumericalError | SolveStatus::SuboptimalSolution
                ) {
                    let fallback_opts;
                    let fb = if options.warm_start.is_some() {
                        fallback_opts = SolverOptions {
                            warm_start: None,
                            ..options.clone()
                        };
                        &fallback_opts
                    } else {
                        options
                    };
                    return solve_without_presolve(problem, fb);
                }
                // Infeasible/Unbounded on the reduced LP propagates directly to the
                // original (presolve is feasibility-preserving). run_postsolve must not
                // be called: it would fill solution/dual vectors from the postsolve stack
                // (e.g. SingletonRow fixed values), producing a spurious non-empty
                // solution with Infeasible/Unbounded status.
                if matches!(
                    raw.status,
                    SolveStatus::Infeasible | SolveStatus::Unbounded
                ) {
                    let objective = if raw.status == SolveStatus::Infeasible {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    return SolverResult {
                        status: raw.status,
                        objective,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        timing_breakdown: Some(crate::problem::TimingBreakdown {
                            presolve_us,
                            solve_us,
                            postsolve_us: 0,
                            ..Default::default()
                        }),
                        ..Default::default()
                    };
                }
                let mut res = presolve::postsolve::run_postsolve(
                    &raw,
                    &presolve_result,
                    problem,
                    eff_opts.deadline,
                    options.recover_warm_start_basis,
                );
                res = guard_lp_optimal(res, problem);
                let postsolve_us = t_solve_done.elapsed().as_micros() as u64;
                res.timing_breakdown = Some(crate::problem::TimingBreakdown {
                    presolve_us,
                    solve_us,
                    postsolve_us,
                    ..Default::default()
                });
                // Postsolve dfeas above PIVOT_TOL (or guard-caught KKT failure) means
                // dual-recovery cannot reconstruct the structure presolve removed.
                // The original LP solves cleanly, so re-attempt on the remaining deadline.
                let postsolve_bad = res.postsolve_dfeas.is_some_and(|d| d > PIVOT_TOL)
                    || res.status == SolveStatus::SuboptimalSolution;
                if matches!(
                    res.status,
                    SolveStatus::Optimal | SolveStatus::SuboptimalSolution
                ) && postsolve_bad
                {
                    let deadline_ok = options
                        .deadline
                        .is_none_or(|d| std::time::Instant::now() < d);
                    if deadline_ok {
                        let mut opts_off = options.clone();
                        opts_off.presolve = false;
                        // Force primal: 初回試行で feasibility 既知のため Primal で直行。
                        opts_off.simplex_method = crate::options::SimplexMethod::Primal;
                        let t_alt_start = std::time::Instant::now();
                        let mut alt = solve_without_presolve(problem, &opts_off);
                        let alt_solve_us = t_alt_start.elapsed().as_micros() as u64;
                        if alt.status == SolveStatus::Optimal
                            && alt.postsolve_dfeas.is_none()
                            && alt.objective.is_finite()
                        {
                            // Preserve the original presolve/postsolve times: both phases
                            // ran (even if postsolve produced bad duals); only solve_us
                            // reflects the alt direct-solve.
                            alt.timing_breakdown = Some(crate::problem::TimingBreakdown {
                                presolve_us,
                                solve_us: alt_solve_us,
                                postsolve_us,
                                ..Default::default()
                            });
                            return alt;
                        }
                    }
                }
                return res;
            }
            Ok(_) => {
                // Presolve did not reduce; record elapsed for timing_breakdown below.
                non_reduced_presolve_us = Some(prof_t0.elapsed().as_micros() as u64);
            }
        }
    }

    // Catch deadline overrun before build_standard_form (presolve may have
    // returned early without reducing).
    if options
        .deadline
        .is_some_and(|d| std::time::Instant::now() >= d)
    {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }

    let t_solve_start = std::time::Instant::now();
    let mut result = solve_without_presolve(problem, options);
    if let Some(presolve_us) = non_reduced_presolve_us {
        let solve_us = t_solve_start.elapsed().as_micros() as u64;
        result.timing_breakdown = Some(crate::problem::TimingBreakdown {
            presolve_us,
            solve_us,
            postsolve_us: 0,
            ..Default::default()
        });
    }
    result
}

/// Solve without presolve.
pub(crate) fn solve_without_presolve(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // Empty variable box (lb > ub beyond the feasibility tolerance) is structurally
    // infeasible. The bounded simplex path detects this, but the `m == 0` and
    // `n == 0` special cases below do not — they pick a variable bound by cost sign
    // without checking lb ≤ ub, returning a false Optimal. B&B branching routinely
    // produces empty boxes (e.g. an integer var pinned between consecutive integers,
    // ⌈lb⌉ > ⌊ub⌋), so the relaxation solver must report Infeasible for them
    // regardless of constraint count. Presolve previously masked this.
    if problem
        .bounds
        .iter()
        .any(|&(lo, hi)| lo - hi > options.primal_tol)
    {
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

    if n == 0 {
        for i in 0..m {
            let feasible = match problem.constraint_types[i] {
                ConstraintType::Le => problem.b[i] >= -options.primal_tol,
                ConstraintType::Ge => problem.b[i] <= options.primal_tol,
                ConstraintType::Eq => problem.b[i].abs() <= options.primal_tol,
            };
            if !feasible {
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
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![0.0; m],
            reduced_costs: vec![],
            slack: problem.b.clone(),
            warm_start_basis: None,
            ..Default::default()
        };
    }

    if m == 0 {
        let mut x = vec![0.0; n];
        let mut obj = 0.0;
        for (j, x_j) in x.iter_mut().enumerate() {
            let (lb, ub) = problem.bounds[j];
            let cj = problem.c[j];
            if cj > options.dual_tol {
                if !lb.is_finite() {
                    return SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        ..Default::default()
                    };
                }
                *x_j = lb;
            } else if cj < -options.dual_tol {
                if !ub.is_finite() {
                    return SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        ..Default::default()
                    };
                }
                *x_j = ub;
            } else if lb.is_finite() {
                // Zero cost: any feasible bound is optimal. Match presolve
                // step3b_empty_column — lb if finite, else ub, else 0.
                *x_j = lb;
            } else if ub.is_finite() {
                *x_j = ub;
            }
            obj += problem.c[j] * *x_j;
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: obj,
            solution: x,
            dual_solution: vec![],
            reduced_costs: problem.c.clone(),
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }
    let Some(sf) = build_standard_form_with_deadline(problem, options.deadline) else {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    };

    // Copy warm_start_lp.basis into warm_start so the LP-specific slot feeds
    // the existing simplex warm path.
    let warm_lp_opts;
    let options = if let Some(ws_lp) = options.warm_start_lp.as_ref() {
        if options.warm_start.is_none() {
            warm_lp_opts = SolverOptions {
                warm_start: Some(WarmStartBasis {
                    basis: ws_lp.basis.clone(),
                    x_b: Vec::new(),
                }),
                ..options.clone()
            };
            &warm_lp_opts
        } else {
            options
        }
    } else {
        options
    };

    let result = match options.simplex_method {
        SimplexMethod::Primal => two_phase_simplex(&sf, problem, options),
        SimplexMethod::Dual => dual::two_phase_dual_simplex(&sf, problem, options),
        SimplexMethod::DualAdvanced | SimplexMethod::Auto => {
            // Auto uses dual_advanced; it falls back to two_phase_dual_simplex
            // internally for problems with Ge/Eq constraints.
            dual_advanced::solve_dual_advanced(&sf, problem, options)
        }
    };
    guard_lp_optimal(result, problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn make_trivial_lp() -> LpProblem {
        // minimize x  s.t.  x <= 5,  x >= 0
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    /// Empty variable box (lb > ub) is Infeasible for every constraint count.
    ///
    /// B&B branching produces empty boxes (an integer var pinned between
    /// consecutive integers → ⌈lb⌉ > ⌊ub⌋) by *mutating* `bounds` after
    /// construction (`MilpProblem::solve` sets `sub.bounds` directly), bypassing
    /// `new_general`'s lb ≤ ub validation. The relaxation solver must then report
    /// Infeasible, not pick a bound by cost sign and claim Optimal.
    ///
    /// No-op proof: removing the empty-box guard makes the `m == 0` / `n == 0`
    /// special cases return Optimal with x at a bound (false-Optimal), failing
    /// the Infeasible assertion. Cases (a) m=0 and (b) m≥1 both must agree; case
    /// (c) proves a genuinely fixed var (lb == ub) is not over-rejected.
    #[test]
    fn empty_variable_box_is_infeasible_for_all_constraint_counts() {
        // (a) m == 0, single var; box emptied post-construction (B&B style).
        let mut m0 = LpProblem::new_general(
            vec![1.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(0.0, 5.0)],
            None,
        )
        .unwrap();
        m0.bounds = vec![(1.8, 1.2)];
        assert_eq!(
            solve_without_presolve(&m0, &SolverOptions::default()).status,
            SolveStatus::Infeasible,
            "m=0 empty box (lb=1.8 > ub=1.2) must be Infeasible, not false-Optimal"
        );

        // (b) m >= 1: one feasible Le row but the box itself is empty.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut m1 = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0)],
            None,
        )
        .unwrap();
        m1.bounds = vec![(2.0, 1.0)];
        assert_eq!(
            solve_without_presolve(&m1, &SolverOptions::default()).status,
            SolveStatus::Infeasible,
            "m=1 empty box (lb=2 > ub=1) must be Infeasible"
        );

        // (c) A genuinely fixed variable (lb == ub) must stay feasible (no
        // over-rejection): proves the guard uses lb - ub > tol, not lb != ub.
        let fixed = LpProblem::new_general(
            vec![1.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(3.0, 3.0)],
            None,
        )
        .unwrap();
        assert_eq!(
            solve_without_presolve(&fixed, &SolverOptions::default()).status,
            SolveStatus::Optimal,
            "fixed var (lb == ub) must remain feasible, not be rejected as empty"
        );
    }

    /// guard_lp_optimal demotes a corrupt Optimal (x = 1e12 >> b = 5) to SuboptimalSolution.
    /// Primal feasibility and stationarity both fail prove_optimal_lp at LP_CERT_TOL.
    #[test]
    fn guard_lp_optimal_catches_corrupt_result() {
        let lp = make_trivial_lp();
        let corrupt = SolverResult {
            status: SolveStatus::Optimal,
            objective: 1e12,
            solution: vec![1e12],
            dual_solution: vec![0.0],
            reduced_costs: vec![0.0],
            slack: vec![0.0],
            ..Default::default()
        };
        let guarded = guard_lp_optimal(corrupt, &lp);
        assert_eq!(
            guarded.status,
            SolveStatus::SuboptimalSolution,
            "guard must demote false-Optimal with |Ax-b| >> tol to SuboptimalSolution"
        );
    }

    /// guard_lp_optimal is a no-op for non-Optimal statuses.
    #[test]
    fn guard_lp_optimal_passthrough_non_optimal() {
        let lp = make_trivial_lp();
        for status in [
            SolveStatus::Infeasible,
            SolveStatus::Timeout,
            SolveStatus::NumericalError,
        ] {
            let r = SolverResult {
                status: status.clone(),
                ..Default::default()
            };
            let out = guard_lp_optimal(r, &lp);
            assert_eq!(out.status, status, "guard must pass through {status:?}");
        }
    }

    // min x + y  s.t.  2x + y >= 3,  x + 2y >= 3,  x,y >= 0
    // Ge constraints: dual-fix cannot push x,y to 0 (would violate Ge).
    // No singleton rows, no Eq rows, no free vars, no parallel rows → presolve
    // cannot remove any row or column ⇒ was_reduced=false.
    fn make_non_reducible_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
            .unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0, 3.0],
            vec![ConstraintType::Ge, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    // min x + y  s.t.  x = 2 (singleton Eq),  x + y <= 5,  x,y >= 0
    // Singleton equality row fixes x — presolve reduces the problem.
    fn make_reducible_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 0, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    /// timing_breakdown is Some on the was_reduced=false path (non-reducing presolve).
    #[test]
    fn timing_breakdown_set_when_presolve_does_not_reduce() {
        let lp = make_non_reducible_lp();

        // Confirm this LP actually exercises the was_reduced=false path.
        let pr = crate::presolve::run_presolve(&lp, None)
            .expect("non-reducible LP must not be Infeasible/Unbounded at presolve");
        assert!(
            !pr.was_reduced,
            "make_non_reducible_lp() must produce an LP presolve cannot reduce (was_reduced must be false)"
        );

        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            result.timing_breakdown.is_some(),
            "timing_breakdown must be Some even when presolve does not reduce the problem"
        );
    }

    /// timing_breakdown is Some on the was_reduced=true path (reducing presolve).
    #[test]
    fn timing_breakdown_set_when_presolve_reduces() {
        let lp = make_reducible_lp();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            result.timing_breakdown.is_some(),
            "timing_breakdown must be Some when presolve reduces the problem"
        );
        // is_some() is the load-bearing assertion; individual μs values can round
        // to zero on fast machines for this trivial LP.
        let _tb = result.timing_breakdown.unwrap();
    }

    /// Invalid options are rejected at the simplex entry with NumericalError.
    ///
    /// Wiring sentinel: removing the `validate()` call from `solve_with` causes
    /// all cases to panic or produce wrong status instead of NumericalError.
    #[test]
    fn invalid_options_rejected_at_simplex_entry() {
        let lp = make_trivial_lp();
        let cases: &[(&str, SolverOptions)] = &[
            (
                "nan primal_tol",
                SolverOptions {
                    primal_tol: f64::NAN,
                    ..Default::default()
                },
            ),
            (
                "zero primal_tol",
                SolverOptions {
                    primal_tol: 0.0,
                    ..Default::default()
                },
            ),
            (
                "neg dual_tol",
                SolverOptions {
                    dual_tol: -1.0,
                    ..Default::default()
                },
            ),
            (
                "inf timeout",
                SolverOptions {
                    timeout_secs: Some(f64::INFINITY),
                    ..Default::default()
                },
            ),
            (
                "neg timeout",
                SolverOptions {
                    timeout_secs: Some(-1.0),
                    ..Default::default()
                },
            ),
            (
                "zero threads",
                SolverOptions {
                    threads: 0,
                    ..Default::default()
                },
            ),
        ];
        for (label, opts) in cases {
            let result = solve_with(&lp, opts);
            assert_eq!(
                result.status,
                SolveStatus::NumericalError,
                "simplex::solve_with with {label} must return NumericalError"
            );
        }
    }

    #[test]
    fn zero_variable_rows_respect_constraint_type() {
        let empty_a = CscMatrix::new(3, 0);
        let lp = LpProblem::new_general(
            vec![],
            empty_a,
            vec![1.0, -1.0, 0.0],
            vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq],
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(
            solve_without_presolve(&lp, &SolverOptions::default()).status,
            SolveStatus::Optimal
        );

        let bad_ge = LpProblem::new_general(
            vec![],
            CscMatrix::new(1, 0),
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(
            solve_without_presolve(&bad_ge, &SolverOptions::default()).status,
            SolveStatus::Infeasible
        );

        let bad_eq = LpProblem::new_general(
            vec![],
            CscMatrix::new(1, 0),
            vec![1.0],
            vec![ConstraintType::Eq],
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(
            solve_without_presolve(&bad_eq, &SolverOptions::default()).status,
            SolveStatus::Infeasible
        );
    }

    #[test]
    fn zero_constraint_bound_only_lp_uses_correct_bound_direction() {
        let lp = LpProblem::new_general(
            vec![2.0, -3.0, 0.0],
            CscMatrix::new(0, 3),
            vec![],
            vec![],
            vec![(1.0, 5.0), (-2.0, 4.0), (7.0, 9.0)],
            None,
        )
        .unwrap();
        let result = solve_without_presolve(&lp, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.solution, vec![1.0, 4.0, 7.0]);
        assert!((result.objective + 10.0).abs() < 1e-12);

        let unbounded_below = LpProblem::new_general(
            vec![1.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            None,
        )
        .unwrap();
        assert_eq!(
            solve_without_presolve(&unbounded_below, &SolverOptions::default()).status,
            SolveStatus::Unbounded
        );
    }

    /// Zero-cost variables in an empty-constraint LP must land on a feasible
    /// bound, not x=0. The ub-only case (lb=-inf, ub finite) regressed: x stayed
    /// at 0.0, violating ub, yet was returned Optimal.
    ///
    /// Sentinel: dropping the `else if ub.is_finite()` arm leaves x[0]=0.0 > ub=-1
    /// in the first case, so the bound-feasibility assert fails.
    #[test]
    fn zero_cost_empty_constraint_lp_lands_on_feasible_bound() {
        // (c, (lb, ub), expected_x): all four bound topologies under zero cost.
        let cases: &[(f64, (f64, f64), f64)] = &[
            // ub-only (the regressed case): lb=-inf, ub=-1 → must pick ub.
            (0.0, (f64::NEG_INFINITY, -1.0), -1.0),
            // lb-only: lb=2, ub=+inf → must pick lb.
            (0.0, (2.0, f64::INFINITY), 2.0),
            // both finite: pick lb (matches presolve step3b policy).
            (0.0, (3.0, 7.0), 3.0),
            // both infinite: x stays 0.0 (only feasible default).
            (0.0, (f64::NEG_INFINITY, f64::INFINITY), 0.0),
        ];
        for &(c, (lb, ub), expected) in cases {
            let lp = LpProblem::new_general(
                vec![c],
                CscMatrix::new(0, 1),
                vec![],
                vec![],
                vec![(lb, ub)],
                None,
            )
            .unwrap();
            let result = solve_without_presolve(&lp, &SolverOptions::default());
            assert_eq!(
                result.status,
                SolveStatus::Optimal,
                "zero-cost empty-constraint LP with bounds ({lb}, {ub}) must be Optimal"
            );
            assert_eq!(
                result.solution,
                vec![expected],
                "x must land on feasible bound for bounds ({lb}, {ub})"
            );
            // Feasibility: the returned x must respect both bounds.
            assert!(
                result.solution[0] >= lb - 1e-12 && result.solution[0] <= ub + 1e-12,
                "x={} must satisfy lb={lb} ≤ x ≤ ub={ub}",
                result.solution[0]
            );
        }
    }

    /// Nonzero-cost ub/lb-only cases stay correct (cost-sign branches unchanged).
    #[test]
    fn nonzero_cost_empty_constraint_lp_picks_optimal_bound() {
        // c<0 with finite ub → maximize x toward ub.
        let neg_cost = LpProblem::new_general(
            vec![-2.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(f64::NEG_INFINITY, 4.0)],
            None,
        )
        .unwrap();
        let r = solve_without_presolve(&neg_cost, &SolverOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal);
        assert_eq!(r.solution, vec![4.0]);

        // c>0 with finite lb → minimize x toward lb.
        let pos_cost = LpProblem::new_general(
            vec![3.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(-5.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let r = solve_without_presolve(&pos_cost, &SolverOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal);
        assert_eq!(r.solution, vec![-5.0]);
    }

    // LP with 3 variables (x, y, z) where presolve fixes x via a singleton Eq row,
    // but the remaining 2-variable problem (y, z) is the same as `make_non_reducible_lp`
    // which presolve cannot reduce further.
    //
    // Result: was_reduced=true, reduced_num_vars=2 (y,z remain), orig_num_vars=3.
    // This gives 0 < reduced_n < orig_n, required so that a reduced-space Timeout
    // solution (len=2) is visibly wrong (not 0, not 3).
    //
    //   row 0: 1.0*x = 5          (Eq singleton — presolve fixes x=5)
    //   row 1: 2.0*y + 1.0*z >= 3 (Ge — part of the non-reducible 2-var sub-LP)
    //   row 2: 1.0*y + 2.0*z >= 3 (Ge — part of the non-reducible 2-var sub-LP)
    fn make_partial_reducible_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 1, 2, 1, 2],
            &[0, 1, 1, 2, 2],
            &[1.0, 2.0, 1.0, 1.0, 2.0],
            3,
            3,
        )
        .unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![5.0, 3.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Ge, ConstraintType::Ge],
            vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ],
            None,
        )
        .unwrap()
    }

    // min x0+x1+x2+x3  s.t.  x0=5 (Eq singleton),  x1-x2<=-1,  x2-x3<=-1,  x3-x1<=-1
    //
    // The cycle x1-x2<=-1, x2-x3<=-1, x3-x1<=-1 sums to 0<=-3 → infeasible.
    // Presolve reduces (removes x0 via singleton row, was_reduced=true) but
    // cannot detect the cycle infeasibility (bounds propagation is incomplete
    // for this 3-constraint Farkas cycle). Simplex on the reduced problem
    // returns Infeasible.
    //
    // Before the fix: run_postsolve was called and replayed SingletonRow
    // {x0=5.0}, producing solution=[5,0,0,0] with status=Infeasible (spurious).
    // After the fix: early return with solution=[] before run_postsolve.
    fn make_presolve_reduced_infeasible_lp() -> LpProblem {
        // Rows: [x0=5, x1-x2<=-1, x2-x3<=-1, x3-x1<=-1]
        let rows = [0usize, 1, 1, 2, 2, 3, 3];
        let cols = [0usize, 1, 2, 2, 3, 3, 1];
        let vals = [1.0f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 4, 4).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0, 1.0, 1.0],
            a,
            vec![5.0, -1.0, -1.0, -1.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Le,
                ConstraintType::Le,
                ConstraintType::Le,
            ],
            vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ],
            None,
        )
        .unwrap()
    }

    /// Sentinel: presolve Timeout early-return must not leak a reduced-space solution.
    /// `INJECT_REDUCED_TIMEOUT` forces the buggy scenario deterministically.
    /// Pre-fix: `solution.len() == reduced_n` → assertion fails.  Post-fix: `len == 0`.
    #[test]
    fn presolve_timeout_solution_never_leaks_reduced_space() {
        // 3-variable LP where presolve eliminates x but leaves y and z.
        // orig_num_vars=3, reduced_num_vars=2 (y,z remain).
        let lp = make_partial_reducible_lp();
        let orig_n = lp.num_vars; // 3

        let pr = crate::presolve::run_presolve(&lp, None)
            .expect("make_partial_reducible_lp must not be Infeasible/Unbounded at presolve");
        assert!(pr.was_reduced, "make_partial_reducible_lp must produce was_reduced=true");
        let reduced_n = pr.reduced_problem.num_vars;
        assert!(
            reduced_n > 0 && reduced_n < orig_n,
            "reduced_n={reduced_n} must be in (0, {orig_n}) — needed to expose the leak"
        );

        // 1. Optimal path (no deadline): solution must be in original space.
        let r = solve_with(&lp, &SolverOptions { presolve: true, ..Default::default() });
        assert_eq!(r.status, SolveStatus::Optimal);
        assert_eq!(
            r.solution.len(),
            orig_n,
            "Optimal: solution.len() must equal orig_num_vars={orig_n}"
        );

        // 2. Deterministic sentinel via injection hook.
        // The hook forces raw = Timeout(solution: vec![0; reduced_n]) and bypasses
        // the wall-clock deadline check, reliably triggering the early-return path.
        // Pre-fix: early-return returned raw → solution.len() == reduced_n (< orig_n) → FAIL.
        // Post-fix: early-return returns vec![] → solution.len() == 0 → PASS.
        INJECT_REDUCED_TIMEOUT.with(|v| v.set(true));
        let r = solve_with(&lp, &SolverOptions { presolve: true, ..Default::default() });
        INJECT_REDUCED_TIMEOUT.with(|v| v.set(false));
        let n = r.solution.len();
        assert_eq!(r.status, SolveStatus::Timeout, "injected path must return Timeout");
        assert!(
            n == 0 || n == orig_n,
            "injected Timeout: solution.len()={n} must be 0 or {orig_n} (orig), \
             never {reduced_n} (reduced — pre-fix reduced-space leak)",
        );
    }

    /// Sentinel: the reduced-space Timeout early-return must carry the reduced
    /// solve's `iterations` through (diagnostic metadata), not drop it to 0.
    /// The injected raw stamps `REDUCED_TIMEOUT_INJECT_ITERS`; pre-fix the
    /// rebuilt result used `..Default::default()` (iterations=0), masking a
    /// solve that ran many pivots as a misleading `iters=0` (the pds-20
    /// reporting artifact that mimicked a stuck/初回-LU hang).
    #[test]
    fn reduced_timeout_preserves_iteration_count() {
        let lp = make_partial_reducible_lp();
        INJECT_REDUCED_TIMEOUT.with(|v| v.set(true));
        let r = solve_with(&lp, &SolverOptions { presolve: true, ..Default::default() });
        INJECT_REDUCED_TIMEOUT.with(|v| v.set(false));
        assert_eq!(r.status, SolveStatus::Timeout, "injected path must return Timeout");
        assert_eq!(
            r.iterations, REDUCED_TIMEOUT_INJECT_ITERS,
            "reduced-space Timeout early-return must carry raw.iterations ({}); \
             got {} — dropping it reports a misleading iters=0 for a long solve",
            REDUCED_TIMEOUT_INJECT_ITERS, r.iterations
        );
    }

    /// Sentinel: Timeout without presolve must return solution in {0, orig_num_vars}.
    /// `cancel_flag=true` fires at the first simplex iteration → deterministic Timeout.
    /// No-op proof: removing the cancel_flag check lets the LP solve Optimal → status
    /// assert fails.
    #[test]
    fn timeout_no_presolve_solution_is_empty_or_orig() {
        use std::sync::{atomic::AtomicBool, Arc};
        let lp = make_reducible_lp();
        let orig_n = lp.num_vars;
        let opts = SolverOptions {
            presolve: false,
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            ..Default::default()
        };
        let r = solve_with(&lp, &opts);
        assert_eq!(
            r.status,
            SolveStatus::Timeout,
            "cancel_flag=true must produce Timeout"
        );
        let n = r.solution.len();
        assert!(
            n == 0 || n == orig_n,
            "Timeout (no presolve): solution.len()={n} must be 0 or {orig_n}"
        );
    }

    /// Reduced-problem Infeasible must propagate with empty solution, not through
    /// run_postsolve which would fill in postsolve-stack values (e.g. x0=5 from
    /// SingletonRow).
    ///
    /// Sentinel: removing the Infeasible guard before run_postsolve causes
    /// result.solution to be non-empty ([5.0, 0.0, 0.0, 0.0]), failing this test.
    #[test]
    fn reduced_infeasible_propagates_without_postsolve() {
        let lp = make_presolve_reduced_infeasible_lp();

        // Confirm presolve reduces (precondition for the test path).
        let pr = crate::presolve::run_presolve(&lp, None)
            .expect("presolve must not detect infeasibility at its own level for this LP");
        assert!(
            pr.was_reduced,
            "LP must be reduced by presolve (x0 singleton row)"
        );

        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "cycle LP must be Infeasible"
        );
        assert!(
            result.solution.is_empty(),
            "Infeasible result must have empty solution, not postsolve-fabricated values (got {:?})",
            result.solution
        );
        assert!(
            result.dual_solution.is_empty(),
            "Infeasible result must have empty dual_solution"
        );
        assert!(result.slack.is_empty(), "Infeasible result must have empty slack");
        assert_eq!(
            result.objective,
            f64::INFINITY,
            "Infeasible objective must be +∞"
        );
        assert!(
            result.timing_breakdown.is_some(),
            "timing_breakdown must be set even for reduced-Infeasible path"
        );
    }

    // min -x1-x2-x3  s.t.  x0=5 (Eq singleton),
    //   x1-x2<=1, x2-x3<=1, x3-x1<=1
    //
    // After presolve removes x0 (was_reduced=true), the reduced problem is
    // min -(x1+x2+x3) with the three Le constraints.  Direction d=(1,1,1)
    // satisfies all constraints (each difference stays constant), c^T d = -3 < 0
    // → UNBOUNDED.  Presolve cannot detect this: x1,x2,x3 appear in active
    // rows (step3b does not fire) and bounds propagation gives no finite upper
    // bounds (ub_finite=false for all three constraints, step4 stays inactive,
    // step5 gives no tightening).
    //
    // Before the fix: run_postsolve was called, producing solution=[5,0,0,0] with
    // status=Unbounded (spurious x0=5 from SingletonRow).
    // After the fix: early return with solution=[] before run_postsolve.
    fn make_presolve_reduced_unbounded_lp() -> LpProblem {
        // Row 0: x0 = 5 (Eq singleton)
        // Rows 1,2,3: x1-x2<=1, x2-x3<=1, x3-x1<=1
        let rows = [0usize, 1, 1, 2, 2, 3, 3];
        let cols = [0usize, 1, 2, 2, 3, 3, 1];
        let vals = [1.0f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 4, 4).unwrap();
        LpProblem::new_general(
            vec![0.0, -1.0, -1.0, -1.0],
            a,
            vec![5.0, 1.0, 1.0, 1.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Le,
                ConstraintType::Le,
                ConstraintType::Le,
            ],
            vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ],
            None,
        )
        .unwrap()
    }

    /// Reduced-problem Unbounded must propagate with empty solution, not through
    /// run_postsolve which would fill in postsolve-stack values (x0=5 from SingletonRow).
    ///
    /// Sentinel: removing the Unbounded guard before run_postsolve causes
    /// result.solution to be non-empty, failing this test.
    #[test]
    fn reduced_unbounded_propagates_without_postsolve() {
        let lp = make_presolve_reduced_unbounded_lp();

        // Confirm presolve reduces without detecting unboundedness.
        let pr = crate::presolve::run_presolve(&lp, None)
            .expect("presolve must not detect Unbounded at its own level for this LP");
        assert!(
            pr.was_reduced,
            "LP must be reduced by presolve (x0 singleton row)"
        );

        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Unbounded,
            "LP must be Unbounded after presolve reduction"
        );
        assert!(
            result.solution.is_empty(),
            "Unbounded result must have empty solution, not postsolve-fabricated values (got {:?})",
            result.solution
        );
        assert!(
            result.dual_solution.is_empty(),
            "Unbounded result must have empty dual_solution"
        );
        assert!(result.slack.is_empty(), "Unbounded result must have empty slack");
        assert_eq!(
            result.objective,
            f64::NEG_INFINITY,
            "Unbounded objective must be -∞"
        );
        assert!(
            result.timing_breakdown.is_some(),
            "timing_breakdown must be set even for reduced-Unbounded path"
        );
    }

    /// timing_breakdown is set when presolve itself detects Infeasible (H observability).
    ///
    /// LP: x = -1 (Eq singleton), x >= 0 → presolve step2 detects value=-1 < lb=0 → Infeasible.
    /// Sentinel: removing the timing_breakdown from the presolve-Infeasible early return
    /// causes result.timing_breakdown to be None, failing this assertion.
    #[test]
    fn timing_breakdown_set_when_presolve_detects_infeasible() {
        // x = -1 with x >= 0 → immediately Infeasible at presolve (value < lb).
        let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0f64], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![-1.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Infeasible);
        assert!(
            result.timing_breakdown.is_some(),
            "timing_breakdown must be Some when presolve itself detects Infeasible (H observability)"
        );
    }
}
