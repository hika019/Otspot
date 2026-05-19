//! Simplex public entry + presolve/postsolve orchestration + method dispatch.

use crate::options::{SimplexMethod, SolverOptions, WarmStartBasis};
use crate::presolve;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::tolerances::PIVOT_TOL;

use super::dual;
use super::dual_advanced;
use super::primal::two_phase_simplex;
use super::standard_form::build_standard_form;

/// Normalized primal violation threshold for the production sentinel.
/// Solutions with normalized violation > this are corrupt (e.g. |Ax-b| = 1e11).
/// Well-solved LPs typically have normalized violation < 1e-8.
const LP_PRIMAL_SENTINEL_TOL: f64 = 1e-3;

/// Normalized primal violation: max constraint violation / (1 + ||b||∞).
fn lp_primal_violation_normalized(problem: &LpProblem, x: &[f64]) -> f64 {
    let m = problem.b.len();
    if m == 0 || x.is_empty() {
        return 0.0;
    }
    let mut ax = vec![0.0_f64; m];
    for j in 0..x.len().min(problem.a.ncols) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m {
                    ax[row] += vals[k] * x[j];
                }
            }
        }
    }
    let b_inf = problem.b.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let viol: f64 = (0..m)
        .map(|i| match problem.constraint_types[i] {
            ConstraintType::Eq => (ax[i] - problem.b[i]).abs(),
            ConstraintType::Le => (ax[i] - problem.b[i]).max(0.0),
            ConstraintType::Ge => (problem.b[i] - ax[i]).max(0.0),
        })
        .fold(0.0_f64, f64::max);
    viol / (1.0 + b_inf)
}

/// Downgrade a false-Optimal to NumericalError when primal violation is excessive.
///
/// Catches cases like klein3 where the simplex solver returns Optimal with
/// |Ax-b| = 2.9e11 due to numerical corruption in Big-M Phase I/II cycling.
pub(crate) fn guard_lp_optimal(result: SolverResult, problem: &LpProblem) -> SolverResult {
    if result.status != SolveStatus::Optimal || result.solution.is_empty() {
        return result;
    }
    let viol = lp_primal_violation_normalized(problem, &result.solution);
    if viol > LP_PRIMAL_SENTINEL_TOL {
        SolverResult::numerical_error()
    } else {
        result
    }
}

/// Solve an LP with default options.
pub fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// Solve an LP with the supplied options. When `options.presolve` is set,
/// presolve runs before the simplex.
pub fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
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
                // (SingularBasis on the reduced initial basis, Eq drift in Phase II).
                // Fall back to solving the original LP without presolve.
                if raw.status == SolveStatus::NumericalError {
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
                });
                // Postsolve dfeas above PIVOT_TOL means dual-recovery cannot
                // reconstruct the structure presolve removed. The original LP
                // solves cleanly, so re-attempt on the remaining deadline.
                if res.status == SolveStatus::Optimal
                    && res.postsolve_dfeas.is_some_and(|d| d > PIVOT_TOL)
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
                            alt.timing_breakdown = Some(crate::problem::TimingBreakdown {
                                presolve_us: 0,
                                solve_us: alt_solve_us,
                                postsolve_us: 0,
                            });
                            return alt;
                        }
                    }
                }
                return res;
            }
            Ok(_) => {}
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

    solve_without_presolve(problem, options)
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

    // #15: warm_start_lp.basis を warm_start にコピー (LP 専用 path で
    // IPM crossover 等の拡張 slot を吸収) → 既存 simplex の warm path に乗る。
    let warm_lp_opts;
    let options = if let Some(ws_lp) = options.warm_start_lp.as_ref() {
        if options.warm_start.is_none() {
            warm_lp_opts = SolverOptions {
                warm_start: Some(WarmStartBasis {
                    basis: ws_lp.basis.clone(),
                    x_b: Vec::new(),
                    at_upper: vec![],
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
