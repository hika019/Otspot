//! postsolve 直後 (元空間) の y/z LSQ refine + Stage 0 (SingletonRow 後退代入) 反復。
//!
//! いずれも KKT-guard 付き: 各 pass で kkt_residual_rel が悪化すれば revert。

use crate::options::SolverOptions;
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::{recover_y_for_singleton_row_with_bound, QpPresolveResult};
use crate::problem::SolverResult;
use crate::qp::ipm_solver::kkt::kkt_residual_rel;
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::bound_contrib;
use crate::qp::problem::QpProblem;

use super::post_processing::refit_progress_stalled;

/// postsolve dual-LSQ / Stage 0 反復の stall 判定。refit/IRLS と同一の
/// 相対+絶対 gate (`refit_progress_stalled`) を共有する: 絶対 floor (1e-12)
/// 単独では大残差 regime で drop≈1e-11 を「進捗あり」と誤判定し、deadline
/// まで無駄反復する (0204d515 と同型の欠陥。dfl001 post_recovery=81–91s)。
fn recovery_progress_stalled(prev_kkt: f64, current_kkt: f64) -> bool {
    refit_progress_stalled(prev_kkt, current_kkt)
}

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
    loop {
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            *final_sol = best_sol;
            return;
        }
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, opts.ipm_eps());
        crate::qp::refine_dual_lsq(orig_problem, final_sol, eliminated_cols, opts.deadline);
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        crate::qp::refine_dual_worst_active_block(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, opts.ipm_eps());
        let cur = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if recovery_progress_stalled(prev, cur) {
            *final_sol = best_sol;
            return;
        }
        prev = cur;
        best_sol = final_sol.clone();
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
    loop {
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            *final_sol = best_sol;
            return;
        }
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, opts.ipm_eps());
        // y[row] を逆順で SingletonRow から復元 (後退代入)
        let bc_vec = bound_contrib(&orig_problem.bounds, &final_sol.bound_duals);
        for step in presolve_result.postsolve_stack.steps.iter().rev() {
            let (row, col) = match step {
                QpPostsolveStep::SingletonRow { row, col, .. } => (*row, *col),
                _ => continue,
            };
            let bc = bc_vec.get(col).copied().unwrap_or(0.0);
            recover_y_for_singleton_row_with_bound(row, col, orig_problem, final_sol, bc);
        }
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        crate::qp::refine_dual_worst_active_block(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, opts.ipm_eps());
        let cur_kkt = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if recovery_progress_stalled(prev_kkt, cur_kkt) {
            *final_sol = best_sol;
            return;
        }
        prev_kkt = cur_kkt;
        best_sol = final_sol.clone();
    }
}

#[cfg(test)]
mod recovery_stall_gate_tests {
    //! Sentinels for the postsolve dual-LSQ / Stage 0 stall gate.
    //!
    //! The gate must be the shared relative+absolute `refit_progress_stalled`,
    //! not the pre-fix absolute floor (`cur + 1e-12 >= prev`): at a large
    //! residual (~0.1) f64 rounding noise produces per-pass drops ≈1e-11 that
    //! the absolute floor misreads as progress, looping until the deadline
    //! (dfl001: post_recovery 81–91s of a 1000s budget; same defect class as
    //! the refit/IRLS gate fixed in 0204d515).

    use super::recovery_progress_stalled;

    /// no-op proof: reverting `recovery_progress_stalled` to the absolute
    /// floor (`cur + 1e-12 >= prev`) classifies the 1e-11 micro-drop as
    /// progress → this assertion fails.
    #[test]
    fn micro_drop_at_large_residual_is_stall() {
        let prev = 1.0e-1;
        let cur = prev - 1.0e-11;
        assert!(
            recovery_progress_stalled(prev, cur),
            "drop 1e-11 at residual 0.1 is f64 noise, not progress"
        );
    }

    /// Genuine (even slow linear) convergence must NOT be classified as stall:
    /// a relative drop of 1e-4 per pass is orders above the 1e-8 stall bar.
    #[test]
    fn real_progress_is_not_stall() {
        let prev = 1.0e-1;
        assert!(!recovery_progress_stalled(prev, prev * (1.0 - 1.0e-4)));
        assert!(!recovery_progress_stalled(prev, 5.0e-2));
    }

    /// Near-zero regime keeps the absolute floor: sub-1e-12 wiggle terminates.
    #[test]
    fn near_zero_noise_is_stall() {
        assert!(recovery_progress_stalled(1.0e-13, 9.0e-14));
    }
}
