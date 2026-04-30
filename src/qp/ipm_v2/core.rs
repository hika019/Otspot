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
use crate::presolve::{
    postsolve_qp_with_dual_recovery, QpPresolveResult,
    recover_y_for_singleton_row_with_bound, bound_contrib_at_var,
};
use crate::presolve::qp_transforms::QpPostsolveStep;
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
            duality_gap_rel: f64::INFINITY,
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

    // postsolve: reduced 空間 → 元問題空間。dual recovery 付き (T1.4 系列の真因対処)。
    let mut final_sol = postsolve_qp_with_dual_recovery(presolve_result, &result, orig_problem);

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


    // Stage 0: postsolve y/z 交互反復 (bound_duals が orig レイアウト確定後)。
    //
    // postsolve_qp_with_dual_recovery 内の forward pass は bound_contrib=0 仮定で
    // y[row] を復元する。boundary に張り付いた fixed/empty 列が存在する場合、
    // KKT 式の bound_contrib が非ゼロのため y[row] が wrong stays。
    //
    // 本反復で bound_duals (orig 空間) を refit_bound_duals_kkt で計算 → bound_contrib
    // を取得 → recover_y_for_singleton_row_with_bound で y を更新、を交互に行うことで
    // 連立を不動点で解く。実問題で 3 pass で収束する経験。
    if result.iterations > 0
        && presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        /// 連鎖依存解消の固定反復回数。各 pass で y / z を交互更新する。
        const POSTSOLVE_RECOVERY_PASSES: usize = 5;
        for _pass in 0..POSTSOLVE_RECOVERY_PASSES {
            // (i) z (bound_duals) を current y に基づいて refit
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            // (ii) y[row] を SingletonRow / RedundantRowFix step で更新
            //      bound_contrib を bound_duals から取得して KKT 完全式で解く
            for step in presolve_result.postsolve_stack.steps.iter() {
                let (row, col) = match step {
                    QpPostsolveStep::SingletonRow { row, col, .. }
                    | QpPostsolveStep::RedundantRowFix { row, col, .. } => (*row, *col),
                    _ => continue,
                };
                let bc = bound_contrib_at_var(&orig_problem.bounds, &final_sol.bound_duals, col);
                recover_y_for_singleton_row_with_bound(
                    row, col, orig_problem, &mut final_sol, bc,
                );
            }
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

    // 元空間 post-processing:
    //   stage A: x の primal projection (LISWET 系 T1.3)
    //   stage B: y を refine_dual_lsq で再計算
    //   stage C: z (bound_duals) を refit_bound_duals_kkt で再計算 (T1.2 QRECIPE)
    //
    // primal を動かすと dual との整合が崩れるため、stage A は stage A〜C の combined
    // (KKT max-rel + primal max-rel) で guard する。stage B/C は kkt_residual_rel で
    // 個別 guard。
    //
    // presolve が変数 fix / 行除去すると postsolve は dual_solution・bound_duals に 0 を
    // 埋め込み KKT が破壊される。Catastrophic 8 件 (QADLITTL/QBORE3D/QCAPRI/QETAMACR/
    // QFFFFF80/QPCBOEI1/QSEBA/QSHELL) は本後処理では完全復元できず、proper dual
    // postsolve (各 presolve 変換ごとに dual を記録) が必要 — 別 PR で対応 (#11)。
    // IPM が一度も iterate しなかった場合 (cancel_flag 即停止 / timeout=0 等) は
    // post-processing をスキップ。post-processing が冷状態 x=[0,..0] から analytic に
    // 最適解を計算してしまい、cancel_flag セマンティクス (Timeout 期待) を破壊するのを防ぐ。
    let ipm_made_progress = result.iterations > 0;

    // 大規模問題で refine_primal_lsq の AAT factorize が時間予算を圧迫するのを防ぐ。
    // BOYD2 (n+m≈280k) では LDL 因子化に分単位かかり、bench の external timeout を
    // 超える。実問題では LISWET (n+m≈20k) が現実的な上限。
    const PRIMAL_PROJECTION_SIZE_LIMIT: usize = 50_000;
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    let allow_primal_projection = problem_size <= PRIMAL_PROJECTION_SIZE_LIMIT;

    let kkt = if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress {
        // (A) x の primal projection — combined guard 付き (サイズ制限あり)。
        // LISWET 系で primal projection 後の dual LSQ refit は ill-conditioned A
        // (near-rank-deficient AAT) で破綻する (|y_new| が IPM y の 5e4 倍に膨張)
        // ため、ここでは primal x のみを射影し dual は IPM 値を保持する。
        // 多くの問題で primal projection は KKT 改善せず combined guard で revert
        // されるが、QRECIPE / 一部 borderline では効く。LISWET 系の precision floor
        // 突破は IPM 内部の深い数値改修 (Mehrotra centering / step size scheduling)
        // が必要 — 別タスク。
        if allow_primal_projection {
            let pre_x = final_sol.solution.clone();
            let pre_y_for_a = final_sol.dual_solution.clone();
            let pre_z_for_a = final_sol.bound_duals.clone();
            let pre_pres_a = primal_residual_rel(&view, &final_sol.solution);
            let pre_kkt_a = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            let pre_combined_a = pre_pres_a.max(pre_kkt_a);
            crate::qp::refine_primal_lsq(orig_problem, &mut final_sol);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let post_pres_a = primal_residual_rel(&view, &final_sol.solution);
            let post_kkt_a = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            let post_combined_a = post_pres_a.max(post_kkt_a);
            if post_combined_a > pre_combined_a {
                final_sol.solution = pre_x;
                final_sol.dual_solution = pre_y_for_a;
                final_sol.bound_duals = pre_z_for_a;
            }
        }

        // (B) 念のためもう 1 度 y / z refit (stage A 内の内部 guard 後の再評価)
        let mut current_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        let pre_y = final_sol.dual_solution.clone();
        crate::qp::refine_dual_lsq(orig_problem, &mut final_sol);
        let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            final_sol.dual_solution = pre_y;
        }
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
    let dual_gap = compute_duality_gap_rel(orig_problem, &final_sol);

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: final_sol.objective,
        iterations: result.iterations,
        kkt_residual_rel: kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        duality_gap_rel: dual_gap,
        numerical_failure: false,
        infeasibility_status: None,
    }
}

/// 元空間 双対ギャップ相対値: |primal_obj - dual_obj| / max(|p|, |d|, 1)
///
/// QP の弱双対性: dual_obj = -1/2 x^T Q x - b^T y + lb^T z_lb - ub^T z_ub
///   (KKT 停留性 Qx + c + A^T y - z_lb + z_ub = 0 を Lagrangian に代入して導出)
/// 真の Optimal では gap → 0。rank-deficient Q で KKT 残差が小さくても gap が
/// 大きい偽 Optimal (UBH1: gap=9.49 で obj 54% 誤差) を弾くゲート。
///
/// FX (lb=ub) 変数は postsolve で bound_duals が 0 埋めされる慣例 + KKT 評価から
/// 除外される設計のため、result.bound_duals[j] には FX 変数の正しい dual が入って
/// いない。ここでは FX 変数の bound 寄与を「lb_j * 停留性」で解析的に置き換え、
/// 偽の gap 検出を防ぐ (BD-T2: FX 変数 z=3 で gap=1.0 → 0 に修正される)。
fn compute_duality_gap_rel(
    problem: &crate::qp::QpProblem,
    result: &crate::problem::SolverResult,
) -> f64 {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return f64::INFINITY;
    }
    let x = &result.solution;
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        match problem.a.transpose().mat_vec_mul(&result.dual_solution) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        vec![0.0_f64; n]
    };
    let xqx: f64 = qx.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let cx: f64 = problem.c.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let primal_obj = 0.5 * xqx + cx + problem.obj_offset;

    let mut by: f64 = 0.0;
    for (&bi, &yi) in problem.b.iter().zip(result.dual_solution.iter()) {
        by += bi * yi;
    }

    // bnd_term = lb^T z_lb - ub^T z_ub
    // FX (lb=ub=val) は z_lb_j, z_ub_j が postsolve で 0 埋め (refit でも更新されない)
    // のため、解析的に val * net_z_at_j (= val * -(qx+c+aty)) で置換する。
    let mut bnd_term: f64 = 0.0;
    let mut lb_idx = 0_usize;
    let mut ub_idx = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < crate::qp::FX_TOL;
        if is_fx {
            // FX: lb_j * z_lb_j - ub_j * z_ub_j = val * (z_lb - z_ub)。
            // bound_contrib[j] = -z_lb + z_ub = -(qx + c + aty) (停留性) なので
            //   val * (z_lb - z_ub) = -val * bound_contrib = val * (qx + c + aty)
            let stat_no_bnd = qx[j] + problem.c[j] + aty[j];
            bnd_term += lb * stat_no_bnd;
            // bound_duals layout 上 idx は進める (FX 用 slot は使わない)
            if lb_finite { lb_idx += 1; }
            if ub_finite { ub_idx += 1; }
        } else {
            if lb_finite && lb_idx < result.bound_duals.len() {
                bnd_term += lb * result.bound_duals[lb_idx];
                lb_idx += 1;
            }
            if ub_finite && ub_idx < result.bound_duals.len() {
                bnd_term -= ub * result.bound_duals[ub_idx];
                ub_idx += 1;
            }
        }
    }
    let dual_obj = -0.5 * xqx - by + bnd_term + problem.obj_offset;
    let gap_abs = (primal_obj - dual_obj).abs();
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if denom > 0.0 && gap_abs.is_finite() { gap_abs / denom } else { f64::INFINITY }
}
