//! solve_qp_v2: 単一 retry 層 + 単一 status 変換 で解く新規 API。
//!
//! 設計書 (`docs/solver_overview_design.md`) の 3 原則:
//! - retry 1 層 (時間内で eps 厳格化を直線的に進める)
//! - status 変換 1 箇所 (API 境界のみ)
//! - 元空間 KKT 直接判定 (scaled OK で偽 Optimal 出さない)
//!
//! 既存 `solve_qp_with` は temporarily 並行運用。v2 が品質・性能で上回ったら旧版を削除する。

use crate::options::SolverOptions;
use crate::presolve::{
    postsolve_qp, run_qp_presolve_phase1, run_qp_presolve_phase2,
    qp_transforms::QpPresolveStatus,
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use super::core::run_ipm;
use super::outcome::IpmOutcome;
use std::time::Instant;

/// 統合 retry の attempt 配列。各 attempt で (use_ruiz, eps_tighten) を変える。
/// 旧 PV_RETRY × POST_VERIFY = 9 attempts を 6 attempts に直線化する。
const ATTEMPTS: &[(bool, f64)] = &[
    (true,  1.0),    // Ruiz on,  eps × 1
    (true,  10.0),   // Ruiz on,  eps × 1/10
    (true,  100.0),  // Ruiz on,  eps × 1/100
    (false, 1.0),    // Ruiz off, eps × 1
    (false, 10.0),   // Ruiz off, eps × 1/10
    (false, 100.0),  // Ruiz off, eps × 1/100
];
/// eps 事前調整の下限 (double 精度限界近傍)
const EPS_FLOOR: f64 = 1e-15;
/// 1 attempt が消費してよい時間の最低割合 (deadline / 残 attempt 数 が これ以下なら break)
const MIN_TIME_PER_ATTEMPT: f64 = 0.5;

/// QP を v2 設計で解く。既存 `solve_qp_with` と同じ API シグネチャ。
///
/// retry 1 層・status 1 箇所変換・元空間 KKT 判定 の 3 原則で動く。
pub fn solve_qp_v2(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let start_time = Instant::now();
    let mut opts = options.clone();
    let n_orig = problem.num_vars;

    // ── presolve (1 回のみ) ─────────────────────────────
    let presolve_result = if opts.presolve {
        let phase1 = run_qp_presolve_phase1(problem, &opts);
        run_qp_presolve_phase2(phase1, &opts)
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    if presolve_result.presolve_status == QpPresolveStatus::Infeasible {
        return SolverResult::infeasible();
    }

    // ── deadline 確定 (presolve 時間も timeout に算入) ─────
    if opts.deadline.is_none() {
        if let Some(secs) = opts.timeout_secs {
            let elapsed = start_time.elapsed().as_secs_f64();
            let remaining = (secs - elapsed).max(0.0);
            opts.deadline = Some(Instant::now() + std::time::Duration::from_secs_f64(remaining));
            opts.timeout_secs = None;
        }
    }
    let total_deadline = opts.deadline;
    let user_eps = opts.ipm_eps();

    // ── retry 1 層: ATTEMPTS 配列を時間内で順に試行 ────────
    let reduced = &presolve_result.reduced;
    let mut best: Option<IpmOutcome> = None;

    for (idx, &(use_ruiz, tighten)) in ATTEMPTS.iter().enumerate() {
        if let Some(d) = total_deadline {
            let now = Instant::now();
            if now >= d {
                break;
            }
            // 残り時間を残 attempt 数で割って per-attempt deadline を算出
            let remaining = d.saturating_duration_since(now);
            let remaining_attempts = (ATTEMPTS.len() - idx) as u32;
            if remaining.as_secs_f64() < MIN_TIME_PER_ATTEMPT {
                break;
            }
            opts.deadline = Some(now + remaining / remaining_attempts.max(1));
            opts.timeout_secs = None;
        }
        opts.ipm.eps = (user_eps / tighten).max(EPS_FLOOR);
        opts.use_ruiz_scaling = use_ruiz;

        let outcome = run_ipm(reduced, &opts);

        // 早期終了: ユーザー指定精度を真に満たす解
        if outcome.satisfies_eps(user_eps) {
            best = Some(outcome);
            break;
        }
        // best-so-far を更新
        match &best {
            None => best = Some(outcome),
            Some(prev) if outcome.quality_score() < prev.quality_score() => {
                best = Some(outcome);
            }
            _ => {}
        }
    }

    // ── postsolve + status 変換 (1 箇所のみ) ───────────────
    let outcome = best.unwrap_or_else(IpmOutcome::empty);
    finalize_outcome(outcome, &presolve_result, problem, user_eps, n_orig)
}

/// `IpmOutcome` から `SolverResult` (外部 status) への変換 — **status mutation 1 箇所**。
fn finalize_outcome(
    outcome: IpmOutcome,
    presolve_result: &crate::presolve::QpPresolveResult,
    _problem: &QpProblem,
    user_eps: f64,
    n_orig: usize,
) -> SolverResult {
    if outcome.solution.is_empty() {
        // 解なし: timeout として返す
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            iterations: outcome.iterations,
            ..Default::default()
        };
    }

    // reduced_sol を SolverResult 形式に詰めて postsolve_qp で元空間に展開
    let reduced_sol = SolverResult {
        status: SolveStatus::Optimal, // 仮 (postsolve は status を見ない)
        objective: outcome.objective,
        solution: outcome.solution,
        dual_solution: outcome.dual_solution,
        bound_duals: outcome.bound_duals,
        iterations: outcome.iterations,
        ..Default::default()
    };
    let mut final_sol = postsolve_qp(presolve_result, &reduced_sol);
    // bound_duals を元問題空間に remap (postsolve_qp は reduced 空間のままコピーするため)
    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &_problem.bounds,
            &final_sol.bound_duals,
        );
    }
    // bounds clip (Ruiz unscale 増幅由来の微小違反を補正)
    for (xi, &(lb, ub)) in final_sol.solution.iter_mut().zip(_problem.bounds.iter()) {
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
    }

    // ── 単一 status 決定 (元空間 KKT で判定) ──
    // 元問題で再検証 (presolve 後の reduced で KKT OK でも postsolve で違反する可能性)
    let final_view = super::outcome::ProblemView {
        q: &_problem.q,
        a: &_problem.a,
        c: &_problem.c,
        b: &_problem.b,
        bounds: &_problem.bounds,
        constraint_types: &_problem.constraint_types,
    };
    let kkt_orig = super::kkt::kkt_residual_rel(
        &final_view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let pres_orig = super::kkt::primal_residual_rel(&final_view, &final_sol.solution);
    let bv_orig = super::kkt::bound_violation(_problem.bounds.as_slice(), &final_sol.solution);

    final_sol.status = if kkt_orig <= user_eps && pres_orig <= user_eps && bv_orig <= user_eps {
        SolveStatus::Optimal
    } else {
        // ユーザー精度未達 → Timeout (有効解なし扱い)
        // 設計書: 「内部で解を捨てない」 = solution は保持して返す
        SolveStatus::Timeout
    };
    debug_assert_eq!(final_sol.solution.len(), n_orig, "postsolve dimension mismatch");
    final_sol
}
