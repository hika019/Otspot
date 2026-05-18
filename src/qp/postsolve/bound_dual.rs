//! bound dual (z) の postsolve 系操作。
//!
//! - reduced → orig 空間への展開 (`remap_bound_duals_to_orig`)
//! - singleton 列の停留性から導出される y 区間への射影 (`project_duals_from_singleton_columns`)
//! - 明確 slack ある不等式行の dual を 0 にする (`zero_inactive_inequality_duals`)

use crate::problem::SolverResult;
use crate::qp::postsolve::dual_recovery::{
    compute_dual_recovery_row_activity, compute_dual_recovery_row_bounds,
    dual_recovery_row_slack_tol,
};
use crate::qp::problem::QpProblem;

/// reduced bound_duals を元問題空間に展開。除去変数の bound_dual は 0.0 で埋める。
pub(crate) fn remap_bound_duals_to_orig(
    presolve_result: &crate::presolve::QpPresolveResult,
    orig_bounds: &[(f64, f64)],
    reduced_bound_duals: &[f64],
) -> Vec<f64> {
    let n_lb_orig = orig_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_orig = orig_bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    if n_lb_orig + n_ub_orig == 0 {
        return Vec::new();
    }
    let reduced_bounds = &presolve_result.reduced.bounds;
    let n_lb_reduced = reduced_bounds
        .iter()
        .filter(|(lb, _)| lb.is_finite())
        .count();
    let n_reduced = reduced_bounds.len();

    let mut lb_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    let mut ub_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    {
        let mut li = 0usize;
        for (jj, &(lb, _)) in reduced_bounds.iter().enumerate() {
            if lb.is_finite() {
                lb_bd_idx[jj] = Some(li);
                li += 1;
            }
        }
        let mut ui = 0usize;
        for (jj, &(_, ub)) in reduced_bounds.iter().enumerate() {
            if ub.is_finite() {
                ub_bd_idx[jj] = Some(n_lb_reduced + ui);
                ui += 1;
            }
        }
    }

    let mut new_bd = vec![0.0_f64; n_lb_orig + n_ub_orig];
    if !reduced_bound_duals.is_empty() {
        let mut orig_li = 0usize;
        for (j, &(lb, _)) in orig_bounds.iter().enumerate() {
            if lb.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = lb_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[orig_li] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_li += 1;
            }
        }
        let mut orig_ui = 0usize;
        for (j, &(_, ub)) in orig_bounds.iter().enumerate() {
            if ub.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = ub_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[n_lb_orig + orig_ui] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_ui += 1;
            }
        }
    }
    new_bd
}

/// singleton column の停留性から row dual の feasible interval を作り、現在 y を射影する。
/// unconstrained LSQ refine では one-sided bound 列で「非負 z で補正不能な y」が出るのを補正。
pub(crate) fn project_duals_from_singleton_columns(
    problem: &QpProblem,
    result: &mut SolverResult,
) {
    let Some((lower, upper)) = compute_dual_recovery_row_bounds(problem, &result.solution) else {
        return;
    };
    if result.dual_solution.len() != problem.num_constraints {
        return;
    }
    for row in 0..problem.num_constraints {
        let lo = lower[row];
        let hi = upper[row];
        if lo > hi {
            continue;
        }
        let y = &mut result.dual_solution[row];
        if *y < lo {
            *y = lo;
        } else if *y > hi {
            *y = hi;
        }
    }
}

/// 明確に slack ある不等式行の dual を相補性から 0 にする。LSQ/IR は stationarity のみ見るため
/// slack 行に dual が残る場合がある。
pub(crate) fn zero_inactive_inequality_duals(problem: &QpProblem, result: &mut SolverResult) {
    if result.solution.len() != problem.num_vars
        || result.dual_solution.len() != problem.num_constraints
    {
        return;
    }
    let Some((ax, row_abs_activity)) =
        compute_dual_recovery_row_activity(problem, &result.solution)
    else {
        return;
    };
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..problem.num_constraints {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            result.dual_solution[i] = 0.0;
        }
    }
}
