//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT) の一貫処理。
//!
//! 設計原則:
//! - 入力は元 QpProblem と presolve 結果。reduced(scaled) は内部で扱う。
//! - 出力 IpmOutcome は **元空間** の解と残差のみを持つ。
//! - これにより `satisfies_eps(user_eps)` が常に元空間判定として機能する。
//!
//! 採用アルゴリズムは設計概要 (`docs/solver_overview_design.md`) に従い IPM/IPPMM のみ。
//! Active Set 法等は採用しない。post-processing は `refine_dual_lsq` (qp/mod.rs の
//! 既存関数、A^T y = -(Qx + c + bound_contrib) の最小二乗解) のみ使用する。

use crate::options::SolverOptions;
use crate::presolve::{postsolve_qp, QpPresolveResult};
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;
use super::outcome::{IpmOutcome, ProblemView};
use super::kkt::{kkt_residual_rel, primal_residual_rel, bound_violation};

/// inner_solver の関数型 (Mehrotra / IP-PMM どちらでも受け取れる)
pub type InnerSolver = fn(&QpProblem, &SolverOptions) -> crate::problem::SolverResult;

/// 1 回の IPM 呼出 + 後処理 (IP-PMM 版)。元空間の解と残差を返す。
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with(orig_problem, presolve_result, opts, crate::qp::ipm::solve_qp_ippmm)
}

/// 1 回の IPM 呼出 + 後処理 (Mehrotra 版)。
/// `solve_qp_v1_wrapped` から呼ばれて IPM (Mehrotra predictor-corrector) を v2 と同じ
/// retry 1 層 / status 1 箇所 / 元空間 KKT 直接判定の枠組みで動かす。
pub fn run_ipm_mehrotra(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with(orig_problem, presolve_result, opts, crate::qp::ipm::solve_qp_ipm)
}

/// 内部 solver を引数に取る一般化 wrapper。
fn run_ipm_with(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    inner_solver: InnerSolver,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;
    let mut result = inner_solver(reduced, opts);

    // 確定的 Infeasible / Unbounded / NonConvex は IpmOutcome に保持して finalize_outcome に
    // 伝える。ここで握りつぶすと外部 status は Timeout に丸められて status 隠蔽になる。
    if matches!(
        result.status,
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::NonConvex(_)
    ) {
        return IpmOutcome::infeasibility(result.status);
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
            numerical_failure: true,
            infeasibility_status: None,
        };
    }

    // dual の post-process refinement (LSQ): scaled 空間で動かす方が IPM 出力との整合性が高い。
    if reduced.num_constraints > 0 {
        crate::qp::refine_dual_lsq(reduced, &mut result);
    }

    // Ruiz unscale: presolve が scaling 適用済みの場合のみ。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
        result.solution = x;
        result.dual_solution = y;
        result.bound_duals = scaler.unscale_bound_duals(
            &result.bound_duals,
            &reduced.bounds,
        );
        if scaler.c.abs() > 1e-300 {
            result.objective /= scaler.c;
        }
    }

    // postsolve: reduced 空間 → 元問題空間
    let mut final_sol = postsolve_qp(presolve_result, &result);

    // bound_duals を元問題空間に remap
    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }

    // bounds clip (Ruiz unscale 増幅由来の微小違反補正)
    for (xi, &(lb, ub)) in final_sol.solution.iter_mut().zip(orig_problem.bounds.iter()) {
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
    }

    // 元空間で KKT 残差を計算 (元空間判定ベース)
    let view = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
    };

    // 元空間 dual refinement (元 v8 セッション 8 と同形 + bound_duals refit を追加):
    //   - y を refine_dual_lsq (LSQ + KKT-guard) で再計算
    //   - z (bound_duals) を refit_bound_duals_kkt (analytic + KKT-guard) で再計算
    //
    // presolve が変数 fix / 行除去すると postsolve は dual_solution・bound_duals に 0 を
    // 埋め込み KKT が破壊される。z の refit は QRECIPE 1 件 (T1.2) を Optimal 化する。
    // 残りの Catastrophic 8 件は本 KKT-guard 後処理で完全復元できず、proper dual postsolve
    // (各 presolve 変換ごとに dual を記録) が必要 — 別 PR 範囲。
    let kkt = if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 {
        let mut current_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        let pre_y = final_sol.dual_solution.clone();
        crate::qp::refine_dual_lsq(orig_problem, &mut final_sol);
        let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            final_sol.dual_solution = pre_y;
        }
        // z refit (QRECIPE 効果): bound_duals が postsolve 0 埋めの修復
        let pre_z = final_sol.bound_duals.clone();
        crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
        let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            final_sol.bound_duals = pre_z;
        }
        current_kkt
    } else {
        kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals)
    };

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: final_sol.objective,
        iterations: result.iterations,
        kkt_residual_rel: kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        numerical_failure: false,
        infeasibility_status: None,
    }
}
