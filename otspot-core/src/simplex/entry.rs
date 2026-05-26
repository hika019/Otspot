//! Simplex public entry + presolve/postsolve orchestration + method dispatch.

use crate::options::{SimplexMethod, SolverOptions, WarmStartBasis};
use crate::presolve;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::qp::certificate::guard_lp_optimal;
use crate::tolerances::PIVOT_TOL;

use super::dual;
use super::dual_advanced;
use super::primal::two_phase_simplex;
use super::standard_form::build_standard_form;

/// Solve an LP with default options.
pub fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// Solve an LP with the supplied options. When `options.presolve` is set,
/// presolve runs before the simplex.
///
/// Returns [`SolveStatus::NumericalError`] immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    if options.validate().is_err() {
        return SolverResult::numerical_error();
    }
    // timeout_secs → deadline (mirrors qp_solve_impl).
    let mut opts_with_deadline;
    let options = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
        opts_with_deadline = options.clone();
        opts_with_deadline.deadline = Some(
            std::time::Instant::now() + std::time::Duration::from_secs_f64(secs),
        );
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
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
            Err(presolve::PresolveStatus::Unbounded) => {
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
                // The reduced LP can be unsolvable while the original is fine
                // (SingularBasis on the reduced initial basis, Eq drift in Phase II,
                // or guard_lp_optimal catching a KKT failure on the reduced form).
                // SuboptimalSolution from the guard means KKT failed → fall back.
                if matches!(raw.status, SolveStatus::NumericalError | SolveStatus::SuboptimalSolution) {
                    return solve_without_presolve(problem, options);
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
                    presolve_us, solve_us, postsolve_us,
                    ..Default::default()
                });
                // Postsolve dfeas above PIVOT_TOL (or guard-caught KKT failure) means
                // dual-recovery cannot reconstruct the structure presolve removed.
                // The original LP solves cleanly, so re-attempt on the remaining deadline.
                let postsolve_bad = res.postsolve_dfeas.is_some_and(|d| d > PIVOT_TOL)
                    || res.status == SolveStatus::SuboptimalSolution;
                if matches!(res.status, SolveStatus::Optimal | SolveStatus::SuboptimalSolution)
                    && postsolve_bad
                {
                    let deadline_ok = options.deadline
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
    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
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

    if n == 0 {
        for i in 0..m {
            if problem.b[i] < -options.primal_tol {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
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
            if problem.c[j] < -options.primal_tol {
                // Finite upper bound caps the maximizer; infinite ⇒ Unbounded.
                let ub = problem.bounds[j].1;
                if ub.is_infinite() {
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

    let sf = build_standard_form(problem);

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
        for status in [SolveStatus::Infeasible, SolveStatus::Timeout, SolveStatus::NumericalError] {
            let r = SolverResult { status: status.clone(), ..Default::default() };
            let out = guard_lp_optimal(r, &lp);
            assert_eq!(out.status, status, "guard must pass through {status:?}");
        }
    }

    // min x + y  s.t.  2x + y >= 3,  x + 2y >= 3,  x,y >= 0
    // Ge constraints: dual-fix cannot push x,y to 0 (would violate Ge).
    // No singleton rows, no Eq rows, no free vars, no parallel rows → presolve
    // cannot remove any row or column ⇒ was_reduced=false.
    fn make_non_reducible_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[2.0, 1.0, 1.0, 2.0],
            2, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0, 3.0],
            vec![ConstraintType::Ge, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap()
    }

    // min x + y  s.t.  x = 2 (singleton Eq),  x + y <= 5,  x,y >= 0
    // Singleton equality row fixes x — presolve reduces the problem.
    fn make_reducible_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 1.0],
            2, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap()
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

        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
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
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
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
            ("nan primal_tol", SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
            ("zero primal_tol", SolverOptions { primal_tol: 0.0, ..Default::default() }),
            ("neg dual_tol", SolverOptions { dual_tol: -1.0, ..Default::default() }),
            ("inf timeout", SolverOptions { timeout_secs: Some(f64::INFINITY), ..Default::default() }),
            ("neg timeout", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
            ("zero threads", SolverOptions { threads: 0, ..Default::default() }),
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
}
