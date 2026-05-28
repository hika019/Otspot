//! refine 系を組み合わせた KKT 改善ループ。

use crate::qp::postsolve::bound_dual::{
    project_duals_from_singleton_columns, zero_inactive_inequality_duals,
};
use crate::qp::postsolve::refine::kkt_iterative::refit_bound_duals_kkt;
use crate::qp::postsolve::refine::projected_gradient::refine_dual_projected_gradient;
use crate::qp::postsolve::refine::worst_active::refine_dual_worst_active_block;
use crate::qp::problem::QpProblem;

pub(crate) fn run_dual_recovery_postprocess(
    problem: &QpProblem,
    view: &crate::qp::ipm_solver::outcome::ProblemView<'_>,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) -> f64 {
    let eliminated_cols = view.eliminated_cols;
    let pre_cleanup = result.clone();
    let kkt_before_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    zero_inactive_inequality_duals(problem, result);
    project_duals_from_singleton_columns(problem, result);
    refine_dual_projected_gradient(problem, result, eliminated_cols, deadline);
    refine_dual_worst_active_block(problem, result, eliminated_cols, deadline);

    let pre_z = result.bound_duals.clone();
    let pre_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    refit_bound_duals_kkt(problem, result);
    let post_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if post_refit_kkt > pre_refit_kkt {
        result.bound_duals = pre_z;
    }

    let kkt_after_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if kkt_after_cleanup > kkt_before_cleanup {
        *result = pre_cleanup;
        kkt_before_cleanup
    } else {
        kkt_after_cleanup
    }
}
