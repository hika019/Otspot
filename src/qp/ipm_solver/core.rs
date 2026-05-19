//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT)。
//! IpmOutcome は元空間の解と残差のみ持ち、satisfies_eps(user_eps) は常に元空間判定。

mod diagnostics;
mod duality_gap;
mod eps_tighten;
mod post_processing;
mod postsolve_dual;
mod warm_start;

use super::kkt::{bound_violation, complementarity_residual_rel, kkt_residual_rel, primal_residual_rel};
use super::outcome::{IpmOutcome, ProblemView};
use crate::options::SolverOptions;
use crate::presolve::{postsolve_qp_with_dual_recovery, QpPresolveResult};
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;

use duality_gap::compute_duality_gap_rel;
use eps_tighten::tighten_ipm_eps_for_presolve_scale;
use post_processing::{
    allow_primal_projection, kkt_already_passes, refine_krylov_and_projection,
    refine_post_processing,
};
use postsolve_dual::{refine_postsolve_dual_lsq, refine_postsolve_recovery};
use warm_start::translate_warm_start_for_presolve;

pub type InnerSolver = fn(&QpProblem, &SolverOptions) -> crate::problem::SolverResult;

/// 1 回の IPPMM 呼出 + 後処理。元空間の解と残差を返す。
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with(
        orig_problem,
        presolve_result,
        opts,
        crate::qp::ipm_core::solve_qp_ippmm,
    )
}

fn run_ipm_with(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    inner_solver: InnerSolver,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;

    let mut opts_for_ipm = tighten_ipm_eps_for_presolve_scale(opts, presolve_result);
    translate_warm_start_for_presolve(&mut opts_for_ipm, presolve_result, reduced);

    let mut result = inner_solver(reduced, &opts_for_ipm);

    // 確定的 Infeasible/Unbounded/NonConvex は outcome に保持して Timeout 隠蔽を避ける。
    if matches!(
        result.status,
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::NonConvex(_)
    ) {
        return IpmOutcome::infeasibility(result.status);
    }

    // 不定 Q + 慣性修正 IPM 収束時は LocallyOptimal フラグを保持。
    // 後処理は Optimal と同パスで行うため一旦 Optimal に昇格。
    let is_locally_optimal = result.status == SolveStatus::LocallyOptimal;
    if is_locally_optimal {
        result.status = SolveStatus::Optimal;
    }

    let invalid = result.solution.is_empty()
        || result.solution.iter().any(|v| !v.is_finite())
        || matches!(result.status, SolveStatus::NumericalError);
    if invalid {
        return IpmOutcome {
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            objective: f64::INFINITY,
            iterations: result.iterations,
            kkt_residual_rel: f64::INFINITY,
            primal_residual_rel: f64::INFINITY,
            bound_violation: f64::INFINITY,
            complementarity_residual_rel: f64::INFINITY,
            duality_gap_rel: f64::INFINITY,
            numerical_failure: true,
            infeasibility_status: None,
            is_locally_optimal: false,
        };
    }

    let post_trace = diagnostics::trace_enabled();
    if post_trace {
        diagnostics::log_ipm_exit_reduced(reduced, &result);
    }

    // dual の LSQ refine は元空間に戻してから行う。scaled 空間で LSQ を回すと L2 ノルム
    // 最小化が scaled 残差分布に過剰適合し、unscale 後に元空間残差が悪化することがある。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        if post_trace {
            let (x_unscaled, y_unscaled) =
                scaler.unscale_solution(&result.solution, &result.dual_solution);
            diagnostics::log_ruiz_scale_ratio(
                scaler, &result.solution, &result.dual_solution, &x_unscaled, &y_unscaled,
            );
        }
        let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
        result.solution = x;
        result.dual_solution = y;
        result.bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &reduced.bounds);
        if scaler.c.abs() > 1e-300 {
            result.objective /= scaler.c;
        }
    }
    if post_trace {
        diagnostics::log_unscaled_reduced(reduced, &result);
        diagnostics::log_presolve_transforms(presolve_result, reduced, orig_problem);
    }

    // postsolve: reduced 空間 → 元問題空間。eliminated 行 / 固定変数の dual 復元込み。
    let mut final_sol = postsolve_qp_with_dual_recovery(presolve_result, &result, orig_problem);

    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }

    if post_trace {
        diagnostics::log_postsolve_remap_bd(orig_problem, &final_sol);
        diagnostics::log_violation_distribution(orig_problem, presolve_result, reduced, &final_sol);
    }

    // bounds clip (Ruiz unscale 増幅由来の微小違反補正)
    let mut total_bound_clip = 0.0_f64;
    let mut clip_count_pre = 0_usize;
    for (xi, &(lb, ub)) in final_sol.solution.iter_mut().zip(orig_problem.bounds.iter()) {
        let pre = *xi;
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
        let amt = (pre - *xi).abs();
        if amt > 0.0 {
            clip_count_pre += 1;
            total_bound_clip = total_bound_clip.max(amt);
        }
    }
    if post_trace {
        diagnostics::log_bounds_clip(orig_problem, &final_sol, clip_count_pre, total_bound_clip);
    }

    // presolve metadata から削除 col mask を導出 (#92 真因 fix)。orig 空間での
    // kkt_residual_rel / refine 呼出は本 mask を経由してのみ EmptyCol を skip する。
    let eliminated_cols: Vec<bool> = presolve_result
        .col_map
        .iter()
        .map(|c| c.is_none())
        .collect();

    // 元空間 dual 一括復元 + Stage 0 (SingletonRow 後退代入)。両方とも IPM が iterate
    // した場合のみ実施 (cancel/timeout=0 で冷状態 x=0 から後処理が独自解を作らない)。
    if presolve_result.was_reduced && !presolve_result.postsolve_stack.steps.is_empty() {
        refine_postsolve_dual_lsq(orig_problem, &mut final_sol, &eliminated_cols, opts);
        if result.iterations > 0 {
            refine_postsolve_recovery(
                orig_problem, presolve_result, &eliminated_cols, &mut final_sol, opts,
            );
        }
    }

    // 元空間 post-processing 3 段階: (1) primal projection, (2) y/z 交互 refit + IRLS,
    // (3) saddle-point Krylov IR。
    // IPM が 1 度も iterate しなかった場合 (cancel/timeout=0) は冷状態 x=0 から
    // 後処理が独自解を作り cancel/Timeout セマンティクスを破壊するため skip。
    let ipm_made_progress = result.iterations > 0;
    let allow_primal = allow_primal_projection(orig_problem);

    if post_trace {
        diagnostics::log_pre_post_processing(orig_problem, &final_sol);
    }

    let user_eps_for_skip = opts.ipm_eps();
    let kkt_already_pass = kkt_already_passes(
        orig_problem, &final_sol, &eliminated_cols, result.status == SolveStatus::Optimal, user_eps_for_skip,
    );
    let kkt = if !final_sol.solution.is_empty()
        && orig_problem.num_constraints > 0
        && ipm_made_progress
        && !kkt_already_pass
    {
        refine_post_processing(orig_problem, &mut final_sol, &eliminated_cols, opts, allow_primal)
    } else {
        let view = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
            eliminated_cols: &eliminated_cols,
        };
        kkt_residual_rel(
            &view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals,
        )
    };

    if ipm_made_progress {
        refine_krylov_and_projection(orig_problem, &mut final_sol, &eliminated_cols, opts, allow_primal);
    }

    let view = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
        eliminated_cols: &eliminated_cols,
    };
    let kkt_final = kkt_residual_rel(
        &view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals,
    );
    let kkt_out = kkt_final;
    let _ = kkt;

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);
    let comp = complementarity_residual_rel(
        &view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals,
    );
    let dual_gap = compute_duality_gap_rel(orig_problem, &final_sol);

    // Invariant: 報告 objective は返却 x で計算。post-processing 後の整合性を保証。
    let objective_recomputed = {
        let qx = orig_problem
            .q
            .mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_vars]);
        let xqx: f64 = qx.iter().zip(final_sol.solution.iter()).map(|(&q, &x)| q * x).sum();
        let cx: f64 = orig_problem.c.iter().zip(final_sol.solution.iter()).map(|(&c, &x)| c * x).sum();
        0.5 * xqx + cx + orig_problem.obj_offset
    };

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: objective_recomputed,
        iterations: result.iterations,
        kkt_residual_rel: kkt_out,
        primal_residual_rel: pres,
        bound_violation: bv,
        complementarity_residual_rel: comp,
        duality_gap_rel: dual_gap,
        numerical_failure: false,
        infeasibility_status: None,
        is_locally_optimal,
    }
}
