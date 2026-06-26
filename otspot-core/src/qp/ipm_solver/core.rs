//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT)。
//! IpmOutcome は元空間の解と残差のみ持ち、satisfies_eps(user_eps) は常に元空間判定。

mod duality_gap;
mod eps_tighten;
mod post_processing;
mod postsolve_dual;
mod warm_start;

use super::kkt::{
    bound_violation, complementarity_componentwise_rel, complementarity_residual_rel,
    kkt_residual_rel, primal_residual_rel,
};
use super::outcome::{IpmOutcome, ProblemView};
use crate::options::SolverOptions;
use crate::presolve::{postsolve_qp_with_dual_recovery, QpPresolveResult};
use crate::problem::{SolveStatus, TimingBreakdown};
use crate::qp::problem::QpProblem;
use crate::tolerances::any_nonfinite;

pub(crate) use duality_gap::compute_duality_gap_rel;
use eps_tighten::tighten_ipm_eps_for_presolve_scale;
use post_processing::{
    allow_primal_projection, kkt_already_passes, refine_krylov_and_projection,
    refine_post_processing,
};
use postsolve_dual::{refine_postsolve_dual_lsq, refine_postsolve_recovery};
use warm_start::translate_warm_start_for_presolve;

pub type InnerSolver = fn(&QpProblem, &SolverOptions) -> crate::problem::SolverResult;

/// 1 回の IPPMM 呼出 + 後処理。元空間の解と残差を返す。
///
/// Returns an [`IpmOutcome`] with `numerical_failure = true` immediately if
/// `opts` fails validation (negative timeout, zero threads, etc.).
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with_user_eps(orig_problem, presolve_result, opts, opts.ipm_eps())
}

pub(super) fn run_ipm_with_user_eps(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    user_eps: f64,
) -> IpmOutcome {
    if opts.validate().is_err() {
        return IpmOutcome {
            numerical_failure: true,
            ..IpmOutcome::empty()
        };
    }
    run_ipm_with(
        orig_problem,
        presolve_result,
        opts,
        user_eps,
        crate::qp::ipm_core::solve_qp_ippmm,
    )
}

fn run_ipm_with(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    user_eps: f64,
    inner_solver: InnerSolver,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;
    let lp_crossover_will_certify = orig_problem.is_zero_q();

    let mut opts_for_ipm = tighten_ipm_eps_for_presolve_scale(opts, presolve_result);
    translate_warm_start_for_presolve(&mut opts_for_ipm, presolve_result, reduced);

    let mut result = inner_solver(reduced, &opts_for_ipm);

    // 確定的 Infeasible/Unbounded/NonConvex は outcome に保持して Timeout 隠蔽を避ける。
    if matches!(
        result.status,
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::NonConvex(_)
    ) {
        let mut out = IpmOutcome::infeasibility(result.status);
        out.timing = result.timing_breakdown;
        return out;
    }

    // 不定 Q + 慣性修正 IPM 収束時は LocallyOptimal フラグを保持。
    // 後処理は Optimal と同パスで行うため一旦 Optimal に昇格。
    let is_locally_optimal = result.status == SolveStatus::LocallyOptimal;
    if is_locally_optimal {
        debug_assert!(
            !result.solution.is_empty(),
            "LocallyOptimal promotion requires a non-empty solution (IPM convergence invariant)"
        );
        result.status = SolveStatus::Optimal;
    }

    // n_reduced==0: presolve が全変数を除去した場合。solve_unconstrained は
    // solution=vec![] を返すが、これは正常 (postsolve が元空間を復元する)。
    // is_empty() / any(!finite()) を n_reduced>0 の場合のみ適用する。
    let n_reduced = reduced.num_vars;
    let invalid = matches!(result.status, SolveStatus::NumericalError)
        || (n_reduced > 0 && (result.solution.is_empty() || any_nonfinite(&result.solution)));
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
            postsolve_krylov_ir_skipped: false,
            timing: result.timing_breakdown,
        };
    }

    // dual の LSQ refine は元空間に戻してから行う。scaled 空間で LSQ を回すと L2 ノルム
    // 最小化が scaled 残差分布に過剰適合し、unscale 後に元空間残差が悪化することがある。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
        result.solution = x;
        result.dual_solution = y;
        result.bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &reduced.bounds);
    }

    // postsolve: reduced 空間 → 元問題空間。eliminated 行 / 固定変数の dual 復元込み。
    let t_postsolve_map = std::time::Instant::now();
    let mut final_sol = postsolve_qp_with_dual_recovery(presolve_result, &result, orig_problem);

    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }
    let postsolve_map_us = t_postsolve_map.elapsed().as_micros() as u64;

    // bounds clip (Ruiz unscale 増幅由来の微小違反補正)
    for (xi, &(lb, ub)) in final_sol
        .solution
        .iter_mut()
        .zip(orig_problem.bounds.iter())
    {
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
    }
    // presolve metadata から削除 col mask を導出。orig 空間での
    // kkt_residual_rel / refine 呼出は本 mask を経由してのみ EmptyCol を skip する。
    let eliminated_cols: Vec<bool> = presolve_result
        .col_map
        .iter()
        .map(|c| c.is_none())
        .collect();

    // 元空間 dual 一括復元 + Stage 0 (SingletonRow 後退代入)。両方とも IPM が iterate
    // した場合のみ実施 (cancel/timeout=0 で冷状態 x=0 から後処理が独自解を作らない)。
    let t_lsq = std::time::Instant::now();
    let mut postsolve_recovery_us = 0u64;
    if !lp_crossover_will_certify
        && presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
    {
        refine_postsolve_dual_lsq(orig_problem, &mut final_sol, &eliminated_cols, opts);
        let t_recovery = std::time::Instant::now();
        if result.iterations > 0 {
            refine_postsolve_recovery(
                orig_problem,
                presolve_result,
                &eliminated_cols,
                &mut final_sol,
                opts,
            );
        }
        postsolve_recovery_us = t_recovery.elapsed().as_micros() as u64;
    }
    let postsolve_lsq_us = t_lsq.elapsed().as_micros() as u64 - postsolve_recovery_us;

    // 元空間 post-processing 3 段階: (1) primal projection, (2) y/z 交互 refit + IRLS,
    // (3) saddle-point Krylov IR。
    // IPM が 1 度も iterate しなかった場合 (cancel/timeout=0) は冷状態 x=0 から
    // 後処理が独自解を作り cancel/Timeout セマンティクスを破壊するため skip。
    let ipm_made_progress = result.iterations > 0;
    let allow_primal = allow_primal_projection(orig_problem);

    let kkt_already_pass = !lp_crossover_will_certify
        && kkt_already_passes(
            orig_problem,
            &final_sol,
            &eliminated_cols,
            result.status == SolveStatus::Optimal,
            user_eps,
        );
    // Stage 1+2 (primal projection + y/z refit/IRLS): run for side effects on
    // `final_sol` only when the solution does not already meet the tolerance.
    // The returned residual is recomputed below as `kkt_final`.
    let t_refine = std::time::Instant::now();
    if !final_sol.solution.is_empty()
        && orig_problem.num_constraints > 0
        && ipm_made_progress
        && !lp_crossover_will_certify
        && !kkt_already_pass
    {
        refine_post_processing(
            orig_problem,
            &mut final_sol,
            &eliminated_cols,
            opts,
            allow_primal,
        );
    }
    let postsolve_refine_us = t_refine.elapsed().as_micros() as u64;

    let kkt_pass_after_refine = !lp_crossover_will_certify
        && (kkt_already_pass
            || kkt_already_passes(
                orig_problem,
                &final_sol,
                &eliminated_cols,
                result.status == SolveStatus::Optimal,
                user_eps,
            ));

    // Skip the saddle-point Krylov IR when the solution already meets the
    // user tolerance (satisfies_eps: kkt + pres + bv + comp + duality_gap all
    // pass). The IR factorizes the full augmented K = [Q+δI, Aᵀ; A, -δI] (n+m),
    // which fills catastrophically for dense constraint rows (fit2d: ~7s) yet
    // performs zero refinement on an already-converged point. This mirrors the
    // `!kkt_already_pass` gate on `refine_post_processing` above.
    let run_krylov_ir = !lp_crossover_will_certify
        && should_run_krylov_ir(ipm_made_progress, kkt_pass_after_refine);
    let t_krylov = std::time::Instant::now();
    if run_krylov_ir {
        refine_krylov_and_projection(
            orig_problem,
            &mut final_sol,
            &eliminated_cols,
            opts,
            allow_primal,
        );
    }
    let postsolve_krylov_ir_us = t_krylov.elapsed().as_micros() as u64;
    // Sentinel: the IR would run whenever the IPM made progress; this is true iff
    // the gate skipped it. Derived from `run_krylov_ir` (not `kkt_already_pass`)
    // so that dropping the `&& !kkt_already_pass` gate flips this to false.
    let krylov_ir_skipped = ipm_made_progress && !run_krylov_ir;

    // Bound-only QP (num_constraints == 0): all refinement above is gated on
    // num_constraints > 0 and is therefore skipped. After inertia-corrected IPM,
    // the converged KKT is (Q+δI)x + z + c = 0, leaving a stationarity residual
    // of δ*x in the original problem. Refit z directly from Qx+c so that z
    // satisfies the original stationarity Qx + z + c = 0 unconditionally.
    if !final_sol.solution.is_empty() && orig_problem.num_constraints == 0 && ipm_made_progress {
        crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol, opts.ipm_eps());
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
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let kkt_out = kkt_final;

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);
    let comp = complementarity_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    )
    .max(complementarity_componentwise_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    ));
    let dual_gap = compute_duality_gap_rel(orig_problem, &final_sol);

    // Invariant: 報告 objective は返却 x で計算。post-processing 後の整合性を保証。
    let objective_recomputed = {
        let qx = orig_problem
            .q
            .mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_vars]);
        let xqx: f64 = qx
            .iter()
            .zip(final_sol.solution.iter())
            .map(|(&q, &x)| q * x)
            .sum();
        let cx: f64 = orig_problem
            .c
            .iter()
            .zip(final_sol.solution.iter())
            .map(|(&c, &x)| c * x)
            .sum();
        0.5 * xqx + cx + orig_problem.obj_offset
    };
    // IPM inner solver が収集した KKT timing に postsolve timing を合算。
    let ipm_base = result.timing_breakdown.unwrap_or_default();
    let postsolve_total_us = postsolve_map_us
        + postsolve_lsq_us
        + postsolve_recovery_us
        + postsolve_refine_us
        + postsolve_krylov_ir_us;
    let combined_timing = TimingBreakdown {
        postsolve_us: postsolve_total_us,
        postsolve_map_us,
        postsolve_lsq_us,
        postsolve_recovery_us,
        postsolve_refine_us,
        postsolve_krylov_ir_us,
        ..ipm_base
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
        postsolve_krylov_ir_skipped: krylov_ir_skipped,
        timing: Some(combined_timing),
    }
}

fn should_run_krylov_ir(ipm_made_progress: bool, kkt_pass_after_refine: bool) -> bool {
    ipm_made_progress && !kkt_pass_after_refine
}

#[cfg(test)]
mod postsolve_krylov_gate_tests {
    use super::should_run_krylov_ir;

    #[test]
    fn krylov_gate_uses_post_refine_pass_state() {
        assert!(
            !should_run_krylov_ir(true, true),
            "once refit/IRLS brings the original-space certificate under eps, \
             Krylov IR must be skipped; using the pre-refine gate here would run it"
        );
        assert!(
            should_run_krylov_ir(true, false),
            "IPM progress plus a still-failing certificate must run Krylov IR"
        );
        assert!(
            !should_run_krylov_ir(false, false),
            "no IPM progress must not let postsolve synthesize a solution"
        );
    }
}

#[cfg(test)]
mod extended_ir_postsolve_path_tests {
    use super::run_ipm_with;
    use crate::options::SolverOptions;
    use crate::presolve::QpPresolveResult;
    use crate::problem::{ConstraintType, SolveStatus, SolverResult};
    use crate::qp::certificate::prove_optimal;
    use crate::qp::ipm_solver::outcome::ProblemView;
    use crate::qp::kkt_resid::dual_sign_violation;
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    fn stationarity_requires_wrong_sign_le_inner(
        _problem: &QpProblem,
        _opts: &SolverOptions,
    ) -> SolverResult {
        SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            iterations: 1,
            ..Default::default()
        }
    }

    #[test]
    fn extended_ir_postsolve_path_does_not_hide_wrong_sign_dual_from_prove_optimal() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let problem = QpProblem::new(
            q,
            vec![1.0_f64],
            a,
            vec![10.0_f64],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let presolve = QpPresolveResult::no_reduction(&problem);
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;
        opts.ipm.extended_ir = true;

        let outcome = run_ipm_with(
            &problem,
            &presolve,
            &opts,
            opts.ipm_eps(),
            stationarity_requires_wrong_sign_le_inner,
        );
        let view = ProblemView::from_problem(&problem);
        let proof = prove_optimal(
            &view,
            &outcome.solution,
            &outcome.dual_solution,
            &outcome.bound_duals,
            outcome.duality_gap_rel,
            opts.ipm_eps(),
        );
        let dsign = dual_sign_violation(
            &problem.constraint_types,
            &outcome.dual_solution,
            &problem.bounds,
            &outcome.bound_duals,
        );

        assert!(
            !outcome.satisfies_eps(opts.ipm_eps()),
            "extended_ir postsolve path must not accept the wrong-sign Le dual candidate as eps-satisfying"
        );
        assert!(
            proof.is_err(),
            "prove_optimal must reject the same postsolve outcome when the certificate is not met"
        );
        assert!(
            dsign <= opts.ipm_eps(),
            "postsolve must preserve dual-sign feasibility instead of trading it for stationarity, dsign={dsign:.3e}"
        );
    }
}
