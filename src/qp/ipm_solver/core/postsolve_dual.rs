//! postsolve 直後 (元空間) の y/z LSQ refine + Stage 0 (SingletonRow 後退代入) 反復。
//!
//! いずれも KKT-guard 付き: 各 pass で kkt_residual_rel が悪化すれば revert。

use crate::options::SolverOptions;
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::{
    bound_contrib_at_var, recover_y_for_singleton_row_with_bound, QpPresolveResult,
};
use crate::problem::SolverResult;
use crate::qp::ipm_solver::kkt::kkt_residual_rel;
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::problem::QpProblem;

const POST_LSQ_PROGRESS_EPS: f64 = 1e-12;
const STAGE0_PROGRESS_EPS: f64 = 1e-12;

/// 元空間 dual 一括復元: postsolve_qp_with_dual_recovery は col_first 停留性のみで
/// y[row] を復元するため、関与 fixed col の停留性が z で吸収されない。ここで
/// refine_dual_lsq を回し x/z 固定で y を LSQ-optimal に更新する。
pub(super) fn refine_postsolve_dual_lsq(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    eliminated_cols: &[bool],
    opts: &SolverOptions,
) {
    if !(final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints)
    {
        return;
    }
    let view0 = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
        eliminated_cols,
    };
    let mut prev = kkt_residual_rel(
        &view0,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let mut best_sol = final_sol.clone();
    let mut pass = 0usize;
    let post_trace = std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1");
    loop {
        if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            *final_sol = best_sol;
            return;
        }
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        crate::qp::refine_dual_lsq(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::refine_dual_worst_active_block(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        let cur = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_trace {
            eprintln!(
                "POST_STAGE [postsolve dual_lsq pass {}] kkt_rel={:.3e}",
                pass, cur
            );
        }
        if cur + POST_LSQ_PROGRESS_EPS >= prev {
            *final_sol = best_sol;
            return;
        }
        prev = cur;
        best_sol = final_sol.clone();
        pass += 1;
    }
}

/// Stage 0: postsolve y/z 交互反復。一括 LSQ で残った fixed-row dofs を col_first
/// 停留性で締める。
pub(super) fn refine_postsolve_recovery(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    eliminated_cols: &[bool],
    final_sol: &mut SolverResult,
    opts: &SolverOptions,
) {
    if !(final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints)
    {
        return;
    }
    let view0 = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
        eliminated_cols,
    };
    let mut prev_kkt = kkt_residual_rel(
        &view0,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let mut best_sol = final_sol.clone();
    let mut pass = 0usize;
    let post_trace = std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1");
    loop {
        if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            *final_sol = best_sol;
            return;
        }
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        // y[row] を逆順で SingletonRow/RedundantRowFix から復元 (後退代入)
        for step in presolve_result.postsolve_stack.steps.iter().rev() {
            let (row, col) = match step {
                QpPostsolveStep::SingletonRow { row, col, .. } => (*row, *col),
                _ => continue,
            };
            let bc = bound_contrib_at_var(&orig_problem.bounds, &final_sol.bound_duals, col);
            recover_y_for_singleton_row_with_bound(row, col, orig_problem, final_sol, bc);
        }
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::refine_dual_worst_active_block(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        let cur_kkt = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_trace {
            eprintln!(
                "POST_STAGE [postsolve recovery pass {}] kkt_rel={:.3e}",
                pass, cur_kkt
            );
        }
        if cur_kkt + STAGE0_PROGRESS_EPS >= prev_kkt {
            *final_sol = best_sol;
            return;
        }
        prev_kkt = cur_kkt;
        best_sol = final_sol.clone();
        pass += 1;
    }
}
