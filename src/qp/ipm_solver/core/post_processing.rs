//! 元空間 post-processing 3 段階: (1) primal projection, (2) y/z 交互 refit (+ IRLS),
//! (3) saddle-point Krylov IR + 2nd primal projection。

use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::ipm_solver::kkt::{kkt_residual_rel, primal_residual_rel};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::problem::QpProblem;

/// factorize 時間予算ガード。LDL 因子化が分単位かかる規模では skip。
const PRIMAL_PROJECTION_SIZE_LIMIT: usize = 50_000;
const REFIT_PROGRESS_EPS: f64 = 1e-12;
const IRLS_INNER_MAX_ITERS: usize = 30;
const KRYLOV_MAX_ITERS: usize = 400;

pub(super) fn allow_primal_projection(orig_problem: &QpProblem) -> bool {
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    problem_size <= PRIMAL_PROJECTION_SIZE_LIMIT
}

/// IPM 出口で既に user eps を満たした Optimal なら post-processing skip (LSQ が秒単位)。
/// Suboptimal/Timeout は component-wise dfr が残るため skip しない。
pub(super) fn kkt_already_passes(
    orig_problem: &QpProblem,
    final_sol: &SolverResult,
    ipm_status_optimal: bool,
    user_eps: f64,
) -> bool {
    if final_sol.solution.is_empty()
        || orig_problem.num_constraints == 0
        || !ipm_status_optimal
    {
        return false;
    }
    let view = build_view(orig_problem);
    let kkt0 = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let pres0 = primal_residual_rel(&view, &final_sol.solution);
    kkt0 < user_eps && pres0 < user_eps
}

/// Post-processing stage 1+2: primal projection + y/z 交互 refit + IRLS。
/// 各 step は KKT-guard 付きで悪化時 revert。
pub(super) fn refine_post_processing(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    opts: &SolverOptions,
    allow_primal: bool,
) -> f64 {
    let view = build_view(orig_problem);

    // (1) primal projection: 違反制約に対して x を最小ノルム射影。
    if allow_primal {
        let pre_x = final_sol.solution.clone();
        let pre_pres = primal_residual_rel(&view, &final_sol.solution);
        crate::qp::refine_primal_lsq(orig_problem, final_sol, opts.deadline);
        let post_pres = primal_residual_rel(&view, &final_sol.solution);
        if post_pres > pre_pres {
            final_sol.solution = pre_x;
        } else {
            // x 改善時は z を新 x に合わせて refit。
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        }
        if std::env::var("PRIMAL_LSQ_TRACE").ok().as_deref() == Some("1") {
            let post_pres2 = primal_residual_rel(&view, &final_sol.solution);
            let post_kkt2 = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            eprintln!("PRIMAL_LSQ: pre_pres={:.3e} post_pres={:.3e} final_pres={:.3e} final_kkt={:.3e} guard={}",
                pre_pres, post_pres, post_pres2, post_kkt2,
                if post_pres > pre_pres { "REVERT" } else { "ACCEPT" });
        }
    }

    // (2) y/z 交互 refit。
    let mut current_kkt = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    loop {
        if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let prev_kkt = current_kkt;

        let pre_dual_step = final_sol.clone();
        crate::qp::refine_dual_lsq(orig_problem, final_sol, opts.deadline);
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(orig_problem, final_sol, opts.deadline);
        crate::qp::refine_dual_worst_active_block(orig_problem, final_sol, opts.deadline);
        let post_kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            *final_sol = pre_dual_step;
        }

        let pre_z = final_sol.bound_duals.clone();
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
        let post_kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            final_sol.bound_duals = pre_z;
        }

        if current_kkt + REFIT_PROGRESS_EPS >= prev_kkt {
            break;
        }
    }

    // 標準 LSQ が componentwise eps を満たさない場合 IRLS で L∞ 風 y を試行。
    let user_eps = opts.ipm_eps();
    loop {
        if current_kkt <= user_eps {
            break;
        }
        if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let prev_kkt = current_kkt;

        let pre_dual_step = final_sol.clone();
        crate::qp::refine_dual_lsq_irls(
            orig_problem,
            final_sol,
            user_eps,
            IRLS_INNER_MAX_ITERS,
            opts.deadline,
        );
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(orig_problem, final_sol, opts.deadline);
        crate::qp::refine_dual_worst_active_block(orig_problem, final_sol, opts.deadline);
        let post_kkt_irls = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt_irls < current_kkt {
            current_kkt = post_kkt_irls;
            let pre_z = final_sol.bound_duals.clone();
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
            let post_kkt_z = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_kkt_z <= current_kkt {
                current_kkt = post_kkt_z;
            } else {
                final_sol.bound_duals = pre_z;
            }
        } else {
            *final_sol = pre_dual_step;
            break;
        }

        if current_kkt + REFIT_PROGRESS_EPS >= prev_kkt {
            break;
        }
    }

    current_kkt
}

/// Post-processing stage 3: saddle-point Krylov IR (K [dx;dy] = -[r_d;r_p]) +
/// pres 残留時の 2nd primal projection (KKT-guard 付き)。
pub(super) fn refine_krylov_and_projection(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    opts: &SolverOptions,
    allow_primal: bool,
) {
    if final_sol.solution.is_empty() || orig_problem.num_constraints == 0 {
        return;
    }
    let view = build_view(orig_problem);
    let user_eps = opts.ipm_eps();
    let target_pf = user_eps;
    let post_trace = std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1");
    if post_trace {
        let pres_pre = primal_residual_rel(&view, &final_sol.solution);
        let kkt_pre = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        eprintln!(
            "POST_STAGE [pre saddle-point IR] pres_rel={:.3e} kkt_rel={:.3e}",
            pres_pre, kkt_pre
        );
    }
    let refined = crate::qp::refine_kkt_iterative(
        orig_problem,
        final_sol,
        KRYLOV_MAX_ITERS,
        target_pf,
        opts.deadline,
    );
    if post_trace {
        let pres_post = primal_residual_rel(&view, &final_sol.solution);
        let kkt_post = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        eprintln!(
            "POST_STAGE [post saddle-point IR] refined_iters={} pres_rel={:.3e} kkt_rel={:.3e}",
            refined, pres_post, kkt_post
        );
    }

    // (3b) KKT IR 後に pres > eps なら primal projection を 1 回追加。
    // 採用条件: pres 改善 AND kkt <= user_eps を厳守 (df 退行防止)。
    if !allow_primal {
        return;
    }
    if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let pres_post_ir = primal_residual_rel(&view, &final_sol.solution);
    let kkt_post_ir = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    if pres_post_ir > user_eps && kkt_post_ir <= user_eps {
        let pre_sol2 = final_sol.clone();
        crate::qp::refine_primal_lsq(orig_problem, final_sol, opts.deadline);
        let post_pres2 = primal_residual_rel(&view, &final_sol.solution);
        if post_pres2 < pres_post_ir {
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol);
            let kkt_after2 = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if kkt_after2 > user_eps {
                *final_sol = pre_sol2;
            } else if post_trace {
                eprintln!("POST_STAGE [2nd primal proj] pre_pres={:.3e} post_pres={:.3e} kkt_after={:.3e} ACCEPT",
                    pres_post_ir, post_pres2, kkt_after2);
            }
        } else {
            *final_sol = pre_sol2;
        }
    }
}

fn build_view(orig_problem: &QpProblem) -> ProblemView<'_> {
    ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
    }
}
