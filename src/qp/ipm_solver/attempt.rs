//! solve_ipm: 単一 retry 層 + API 境界 1 箇所での status 変換 + 元空間 KKT 直接判定。

use std::cell::Cell;
use std::time::Instant;

#[cfg(test)]
use crate::ScopedDisable;

use crate::options::SolverOptions;
use crate::presolve::{
    run_qp_presolve_phase1, run_qp_presolve_phase2,
    qp_transforms::{QpPresolveStatus, QpPostsolveStep},
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use super::core::run_ipm;
use crate::presolve::QpPresolveResult;
use super::kkt::{kkt_residual_rel, primal_residual_rel, bound_violation};
use super::outcome::{IpmOutcome, ProblemView};

/// Residual threshold above which an Optimal/LocallyOptimal QP result is
/// considered catastrophically corrupt and demoted to NumericalError.
///
/// Set three orders of magnitude above typical convergence (1e-6) so that
/// only catastrophic failures (e.g. undetected postsolve corruption) trigger
/// this guard. Normal near-miss suboptimal results are handled by satisfies_eps.
const QP_GUARD_CATASTROPHIC_TOL: f64 = 1e-1;

thread_local! {
    static QP_GUARD_DISABLED: Cell<bool> = Cell::new(false);
}

/// Runs `f` with `guard_qp_optimal` bypassed.
///
/// Test-only: used as a no-op scope guard in `guard_qp_optimal_no_op_proof`.
/// Pass corrupt data through the guard while disabled and assert it is NOT
/// demoted. The load-bearing evidence lives in the paired test that does NOT
/// disable.
///
/// Thread-safe: affects only the current thread.
/// Panic-safe: the guard is re-enabled even if `f` panics.
#[cfg(test)]
pub(crate) fn with_qp_guard_disabled<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = ScopedDisable::new(
        || QP_GUARD_DISABLED.with(|c| c.set(true)),
        || QP_GUARD_DISABLED.with(|c| c.set(false)),
    );
    f()
}

/// Downgrade false-Optimal/LocallyOptimal QP results with catastrophic residuals
/// to NumericalError. Defense-in-depth applied at the solve_ipm API boundary.
///
/// Recomputes KKT stationarity, primal feasibility, and bound violation from
/// the solution independently of the stored IpmOutcome residuals, so it catches
/// corruption that occurs after satisfies_eps has been evaluated.
pub(crate) fn guard_qp_optimal(result: SolverResult, problem: &QpProblem) -> SolverResult {
    if QP_GUARD_DISABLED.with(|c| c.get()) {
        return result;
    }
    if !matches!(result.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal) {
        return result;
    }
    if result.solution.is_empty() {
        return result;
    }
    let view = ProblemView::from_problem(problem);
    let kkt = kkt_residual_rel(&view, &result.solution, &result.dual_solution, &result.bound_duals);
    let pf = primal_residual_rel(&view, &result.solution);
    let bv = bound_violation(&problem.bounds, &result.solution);
    if kkt > QP_GUARD_CATASTROPHIC_TOL
        || pf > QP_GUARD_CATASTROPHIC_TOL
        || bv > QP_GUARD_CATASTROPHIC_TOL
    {
        SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            iterations: result.iterations,
            ..Default::default()
        }
    } else {
        result
    }
}

/// 1 attempt の IPM 反復上限。500 は Maros/QPLIB 全 PASS が収まる empirical sweet spot。
const MAX_ITER_PER_ATTEMPT: usize = 500;

/// No-presolve fallback: only run on problems this size or smaller. The fallback
/// re-solves the original (non-reduced) problem, bypassing the Ruiz amplification
/// that can stop the inner IPM from converging tightly enough. The cap bounds the
/// cost of re-solving without presolve reduction; it sits below PRESOLVE_SIZE_LIMIT
/// (50_000), so problems in between still get presolve+Ruiz but are deemed too
/// large to re-solve from scratch economically.
const NO_PRESOLVE_FALLBACK_LIMIT: usize = 10_000;

type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions) -> IpmOutcome;

/// presolve スケーリング縮小比率の下限 sigma_total。unscale 残差は 1/sigma_total 倍される。
fn compute_presolve_sigma_total(presolve_result: &QpPresolveResult) -> f64 {
    let mut primal_row_scale_min = 1.0_f64;
    for step in presolve_result.postsolve_stack.steps.iter() {
        if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
            let local_min = row_scales.iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .fold(f64::INFINITY, |a, &v| a.min(v));
            if local_min.is_finite() {
                primal_row_scale_min *= local_min;
            }
        }
    }
    let mut dual_col_scale_min = f64::INFINITY;
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let e_min = scaler.e.iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if e_min.is_finite() {
            primal_row_scale_min *= e_min;
        }
        let d_min = scaler.d.iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if d_min.is_finite() && scaler.c.is_finite() && scaler.c > 0.0 {
            dual_col_scale_min = scaler.c * d_min;
        }
    }
    primal_row_scale_min.min(dual_col_scale_min)
}

/// tighten = ceil_pow10(user_eps / 1e-8) ∈ [1, 1000]。上限 1000 は IPM floor 制約。
fn dynamic_base_tighten(sigma_total: f64, user_eps: f64) -> f64 {
    const REF_EPS: f64 = 1e-8;
    let _ = sigma_total;
    let ratio = user_eps / REF_EPS;
    if ratio <= 1.0 {
        return 1.0;
    }
    let pow = ratio.log10().ceil();
    10_f64.powf(pow.min(3.0))
}

/// Q が対角なら s_j=1/√Q_jj の column scaling で Q'_jj=1 に均等化し、解後 x_orig=D·x_scaled で復元。
pub fn solve_ipm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if let Some((scaled_problem, col_scales)) = try_q_diagonal_scaling(problem) {
        let scaled_options = scale_warm_start_for_q_diag(options, &col_scales);
        let mut result = solve_ipm_with_runner(&scaled_problem, &scaled_options, run_ipm);
        unscale_q_diagonal(&mut result, &col_scales, problem);
        return guard_qp_optimal(result, problem);
    }
    let result = solve_ipm_with_runner(problem, options, run_ipm);
    guard_qp_optimal(result, problem)
}

/// warm_start_qp.x を Q-diag column scaling (x_orig = D·x_scaled) の inverse で scaled 空間に翻訳。
/// y / mu は scaling 不変。長さ不一致は B-2 で扱うため drop + 警告。
fn scale_warm_start_for_q_diag(options: &SolverOptions, col_scales: &[f64]) -> SolverOptions {
    let mut scaled = options.clone();
    if let Some(ws) = scaled.warm_start_qp.as_mut() {
        if ws.x.len() == col_scales.len() {
            for j in 0..col_scales.len() {
                ws.x[j] /= col_scales[j];
            }
        } else {
            eprintln!(
                "[warm_start_qp dropped] q_diag_scaling dim mismatch: ws.x.len={} col_scales.len={}",
                ws.x.len(),
                col_scales.len()
            );
            scaled.warm_start_qp = None;
        }
    }
    scaled
}

fn try_q_diagonal_scaling(problem: &QpProblem) -> Option<(QpProblem, Vec<f64>)> {
    let n = problem.num_vars;
    if n == 0 { return None; }

    let mut q_diag = vec![0.0_f64; n];
    let mut q_offdiag_max = 0.0_f64;
    for col in 0..n {
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            let v = problem.q.values[k];
            if row == col {
                q_diag[col] = v;
            } else {
                q_offdiag_max = q_offdiag_max.max(v.abs());
            }
        }
    }

    const Q_OFFDIAG_TOL: f64 = 1e-10;
    if q_offdiag_max > Q_OFFDIAG_TOL {
        return None;
    }

    let mut q_pos_min = f64::INFINITY;
    let mut q_pos_max = 0.0_f64;
    for &v in &q_diag {
        if v > Q_OFFDIAG_TOL {
            q_pos_min = q_pos_min.min(v);
            q_pos_max = q_pos_max.max(v);
        }
    }
    if !q_pos_min.is_finite() || q_pos_max <= 0.0 {
        return None;
    }
    // dynamic range が狭い Q では IPM K-行列 conditioning 悪化リスクが上回るため gate。
    const Q_DIAG_RANGE_TRIGGER: f64 = 1e6;
    if q_pos_max / q_pos_min < Q_DIAG_RANGE_TRIGGER {
        return None;
    }

    // s_j = 1/√Q_jj (Q_jj=0 の LP-like 列は s_j=1)、Q'_jj = 1。
    let mut col_scales = vec![1.0_f64; n];
    for j in 0..n {
        if q_diag[j] > Q_OFFDIAG_TOL {
            col_scales[j] = 1.0 / q_diag[j].sqrt();
        }
    }

    let mut q_s = problem.q.clone();
    for col in 0..n {
        let cs = q_s.col_ptr[col];
        let ce = q_s.col_ptr[col + 1];
        for k in cs..ce {
            let row = q_s.row_ind[k];
            q_s.values[k] *= col_scales[row] * col_scales[col];
        }
    }

    // A' = A D (column-scale)
    let mut a_s = problem.a.clone();
    for col in 0..n {
        let cs = a_s.col_ptr[col];
        let ce = a_s.col_ptr[col + 1];
        let s = col_scales[col];
        for k in cs..ce {
            a_s.values[k] *= s;
        }
    }

    // c' = D c (column-scale)
    let c_s: Vec<f64> = problem.c.iter().enumerate()
        .map(|(j, &v)| v * col_scales[j])
        .collect();

    // bounds' = bounds / D (s_j > 0 なので符号変わらず)
    let bounds_s: Vec<(f64, f64)> = problem.bounds.iter().enumerate()
        .map(|(j, &(lb, ub))| (lb / col_scales[j], ub / col_scales[j]))
        .collect();

    // QpProblem を作る (b は不変、constraint_types も不変)。
    // obj_offset は scaling 不変なため orig から引き継ぐ。
    let mut scaled = match QpProblem::new(
        q_s, c_s, a_s, problem.b.clone(), bounds_s, problem.constraint_types.clone(),
    ) {
        Ok(p) => p,
        Err(_) => return None,
    };
    scaled.obj_offset = problem.obj_offset;

    Some((scaled, col_scales))
}

/// `try_q_diagonal_scaling` で行った column scaling を逆変換する。
/// x_orig = D × x_scaled, y は不変, y_lb/y_ub /= D.
fn unscale_q_diagonal(
    result: &mut SolverResult,
    col_scales: &[f64],
    orig_problem: &QpProblem,
) {
    let n = orig_problem.num_vars;
    if result.solution.len() == n {
        for j in 0..n {
            result.solution[j] *= col_scales[j];
        }
    }
    // y は scaling 不変。bound_duals layout = [y_lb 群; y_ub 群]。
    if !result.bound_duals.is_empty() {
        let mut idx = 0_usize;
        for (j, &(lb, _)) in orig_problem.bounds.iter().enumerate() {
            if lb.is_finite() && idx < result.bound_duals.len() {
                result.bound_duals[idx] /= col_scales[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in orig_problem.bounds.iter().enumerate() {
            if ub.is_finite() && idx < result.bound_duals.len() {
                result.bound_duals[idx] /= col_scales[j];
                idx += 1;
            }
        }
    }
}

fn solve_ipm_with_runner(
    problem: &QpProblem,
    options: &SolverOptions,
    runner: IpmRunner,
) -> SolverResult {
    let start_time = Instant::now();
    let mut opts = options.clone();
    let n_orig = problem.num_vars;

    // 巨大 QP の presolve 内 hot loop でも deadline を見られるよう先に固定。
    if opts.deadline.is_none() {
        if let Some(secs) = opts.timeout_secs {
            opts.deadline = Some(start_time + std::time::Duration::from_secs_f64(secs));
            opts.timeout_secs = None;
        }
    }
    let total_deadline = opts.deadline;
    let user_eps = opts.ipm_eps();

    // presolve hot loop は deadline を見る (qp_transforms/driver.rs) が、巨大問題では
    // presolve だけで deadline 予算を食い切り IPM が走れなくなる。この上限は
    // 「presolve に予算を配分するか IPM に回すか」の予算配分ガード (時間予算 proxy)。
    // n か m のどちらかが上限超なら presolve を skip し IPM に予算を残す。
    const PRESOLVE_SIZE_LIMIT: usize = 50_000;
    let presolve_result = if opts.presolve
        && problem.num_vars <= PRESOLVE_SIZE_LIMIT
        && problem.num_constraints <= PRESOLVE_SIZE_LIMIT
    {
        let phase1 = run_qp_presolve_phase1(problem, &opts);
        if std::env::var("QP_PRESOLVE_PHASE2").ok().as_deref() == Some("0") {
            phase1
        } else {
            run_qp_presolve_phase2(phase1, &opts)
        }
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    if presolve_result.presolve_status == QpPresolveStatus::Infeasible {
        return SolverResult::infeasible();
    }
    if presolve_result.presolve_status == QpPresolveStatus::Unbounded {
        return SolverResult::unbounded();
    }

    if total_deadline.is_some_and(|d| Instant::now() >= d) {
        return finalize_outcome(IpmOutcome::empty(), user_eps, n_orig, total_deadline, false);
    }

    // presolve Ruiz 済なら IPM 側で重ね掛けしない (二重 scale で誤収束する)。
    let presolve_did_ruiz = presolve_result.ruiz_scaler.is_some();
    let mut best: Option<IpmOutcome> = None;

    // (use_ruiz, eps_tighten) 試行配列。tighten は user_eps/1e-8 から導出、
    // base → base×10 → base/10 → 1 の段階で reg_limit 適応を促す。
    let sigma_total = compute_presolve_sigma_total(&presolve_result);
    let base_tighten = dynamic_base_tighten(sigma_total, user_eps);
    let attempts: Vec<(bool, f64)> = if presolve_did_ruiz {
        let mut v = vec![
            (false, base_tighten),
            (false, base_tighten * 10.0),
        ];
        if base_tighten > 10.0 {
            v.push((false, base_tighten / 10.0));
        }
        if base_tighten > 1.0 {
            v.push((false, 1.0));
        }
        v
    } else {
        let mut v = vec![
            (true,  base_tighten),
            (false, base_tighten),
            (true,  base_tighten * 10.0),
            (false, base_tighten * 10.0),
            (true,  base_tighten * 100.0),
            (false, base_tighten * 100.0),
        ];
        if base_tighten > 10.0 {
            v.push((true,  base_tighten / 10.0));
            v.push((false, base_tighten / 10.0));
        }
        if base_tighten > 1.0 {
            v.push((true,  1.0));
            v.push((false, 1.0));
        }
        v
    };

    for &(use_ruiz, tighten) in attempts.iter() {
        if let Some(d) = total_deadline {
            if Instant::now() >= d { break; }
        }
        opts.deadline = total_deadline;
        opts.timeout_secs = None;
        opts.ipm.max_iter = MAX_ITER_PER_ATTEMPT;
        opts.use_ruiz_scaling = use_ruiz;
        // IPM_EPS_NOISE_FLOOR (100×ε) で統一 (attempt level でも machine noise 直近 eps
        // を回避、reviewer 観点で 3 種共存 → 2 種集約)。
        opts.ipm.eps = (user_eps / tighten).max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);

        let outcome = runner(problem, &presolve_result, &opts);

        if outcome.satisfies_eps(user_eps) {
            best = Some(outcome);
            break;
        }
        match &best {
            None => best = Some(outcome),
            Some(prev) if outcome.quality_score() < prev.quality_score() => {
                best = Some(outcome);
            }
            _ => {}
        }
    }

    // No-presolve fallback: when presolve+Ruiz path fails for small problems, run
    // the inner IPM directly on the original problem. Ruiz equilibration in
    // presolve can force a scaled convergence threshold (eps * sigma_total) that
    // the inner IPM cannot reach numerically, even with all tighten attempts.
    // Without presolve, the IPM operates in the original space (no amplification)
    // and typically converges within user_eps. Size-gated to avoid overhead on
    // problems that are too large to re-solve without reduction.
    let best_ok = best.as_ref().map(|b| b.satisfies_eps(user_eps)).unwrap_or(false);
    if !best_ok && presolve_did_ruiz && n_orig <= NO_PRESOLVE_FALLBACK_LIMIT {
        let fallback_pre = QpPresolveResult::no_reduction(problem);
        for use_ruiz_fb in [false, true] {
            if total_deadline.map_or(false, |d| Instant::now() >= d) {
                break;
            }
            opts.deadline = total_deadline;
            opts.timeout_secs = None;
            opts.ipm.max_iter = MAX_ITER_PER_ATTEMPT;
            opts.use_ruiz_scaling = use_ruiz_fb;
            opts.tolerance = None;
            opts.ipm.eps = user_eps.max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);
            let fb = runner(problem, &fallback_pre, &opts);
            // Replace best only when the fallback actually satisfies user_eps. A
            // non-satisfying fallback must NOT displace the presolve result:
            // quality_score is KKT-only and ignores objective value, so a fallback
            // with smaller kkt_rel but a far-worse objective would wrongly win.
            // (After the attempt loop `best` is always Some, so the prior-None
            // branch was unreachable and is dropped.)
            if fb.satisfies_eps(user_eps) {
                best = Some(fb);
                break;
            }
        }
    }

    let mut outcome = best.unwrap_or_else(IpmOutcome::empty);

    // Ruiz 歪みで偽陽性した LocallyOptimal を、元問題 Q の Gershgorin PSD 判定で打ち消す。
    if outcome.is_locally_optimal {
        let ic = crate::qp::ipm_core::kkt::compute_inertia_correction(&problem.q);
        if ic == 0.0 {
            outcome.is_locally_optimal = false;
        }
    }

    let cancelled = options
        .cancel_flag
        .as_ref()
        .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed));
    finalize_outcome(outcome, user_eps, n_orig, total_deadline, cancelled)
}

/// IpmOutcome → SolverResult: eps 達成→Optimal、外部停止→Timeout、内部停止→Suboptimal、解無し→NumericalError。
fn finalize_outcome(
    outcome: IpmOutcome,
    user_eps: f64,
    n_orig: usize,
    total_deadline: Option<Instant>,
    cancelled: bool,
) -> SolverResult {
    let krylov_ir_skipped = outcome.postsolve_krylov_ir_skipped;
    if let Some(infeas) = outcome.infeasibility_status {
        let objective = match infeas {
            SolveStatus::Infeasible => f64::INFINITY,
            SolveStatus::Unbounded => f64::NEG_INFINITY,
            _ => f64::NAN,
        };
        return SolverResult {
            status: infeas,
            objective,
            iterations: outcome.iterations,
            ..Default::default()
        };
    }

    let timed_out = cancelled || total_deadline.is_some_and(|d| Instant::now() >= d);

    if outcome.solution.is_empty() {
        let status = if timed_out {
            SolveStatus::Timeout
        } else {
            SolveStatus::NumericalError
        };
        return SolverResult {
            status,
            objective: f64::INFINITY,
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            iterations: outcome.iterations,
            ..Default::default()
        };
    }

    let status = if outcome.satisfies_eps(user_eps) {
        if outcome.is_locally_optimal {
            SolveStatus::LocallyOptimal
        } else {
            SolveStatus::Optimal
        }
    } else if timed_out {
        SolveStatus::Timeout
    } else {
        SolveStatus::SuboptimalSolution
    };

    debug_assert_eq!(outcome.solution.len(), n_orig, "outcome solution dimension mismatch");

    SolverResult {
        status,
        objective: outcome.objective,
        solution: outcome.solution,
        dual_solution: outcome.dual_solution,
        bound_duals: outcome.bound_duals,
        iterations: outcome.iterations,
        timing_breakdown: outcome.timing,
        stats: crate::problem::SolveStats {
            postsolve_krylov_ir_skipped: krylov_ir_skipped,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    #[test]
    fn test_q_diagonal_scaling_skips_non_diagonal_q() {
        let q = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        assert!(try_q_diagonal_scaling(&prob).is_none());
    }

    #[test]
    fn test_q_diagonal_scaling_skips_uniform_diagonal() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        assert!(try_q_diagonal_scaling(&prob).is_none());
    }

    /// ill-cond diagonal Q で scaling/unscale が roundtrip する。
    #[test]
    fn test_q_diagonal_scaling_roundtrip() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e-7, 2.0], 2, 2).unwrap();
        let c = vec![-3.0, -4.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0, 100.0), (0.0, 100.0)];
        let prob = QpProblem::new(
            q.clone(), c.clone(), a.clone(), b.clone(), bounds.clone(),
            vec![crate::problem::ConstraintType::Eq],
        ).unwrap();

        let (scaled, col_scales) = try_q_diagonal_scaling(&prob)
            .expect("ill-cond diag Q must trigger");
        let q_s = &scaled.q;
        for col in 0..2 {
            for k in q_s.col_ptr[col]..q_s.col_ptr[col + 1] {
                if q_s.row_ind[k] == col {
                    assert!((q_s.values[k] - 1.0).abs() < 1e-12, "got {} at col {}", q_s.values[k], col);
                }
            }
        }
        assert!((col_scales[0] - 1.0 / (1e-7_f64).sqrt()).abs() < 1e-3);
        assert!((col_scales[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((scaled.bounds[0].1 - 100.0 / col_scales[0]).abs() < 1e-9);
        assert!((scaled.bounds[1].1 - 100.0 / col_scales[1]).abs() < 1e-6);
    }

    /// solve_ipm 経由でも unscale 後に元問題 primal feas を満たすこと。
    #[test]
    fn test_q_diagonal_scaling_unscale_roundtrip() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e-12, 2.0], 2, 2).unwrap();
        let c = vec![-3.0, -4.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0, 100.0), (0.0, 100.0)];
        let prob = QpProblem::new(
            q, c, a, b, bounds,
            vec![crate::problem::ConstraintType::Eq],
        ).unwrap();

        let opts = SolverOptions::default();
        let result = solve_ipm(&prob, &opts);
        assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
        // Full KKT + primal + bound invariant check (P1-C: assert_solver_invariants_qp coverage).
        crate::test_kkt::assert_solver_invariants_qp(&result, &prob);
        let ax = prob.a.mat_vec_mul(&result.solution).unwrap();
        assert!((ax[0] - 1.0).abs() < 1e-6);
        for j in 0..2 {
            let (lb, ub) = prob.bounds[j];
            assert!(result.solution[j] >= lb - 1e-9);
            assert!(result.solution[j] <= ub + 1e-9);
        }
    }

    fn make_simple_eq_qp() -> QpProblem {
        // min x  s.t.  x = 1.0,  x >= 0
        // optimal: x=1, obj=1
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        QpProblem::new(
            q,
            vec![1.0],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY)],
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap()
    }

    /// guard_qp_optimal demotes corrupt Optimal (x=1e12 violates x=1) to NumericalError.
    #[test]
    fn guard_qp_optimal_catches_catastrophic_result() {
        let prob = make_simple_eq_qp();
        let corrupt = SolverResult {
            status: crate::problem::SolveStatus::Optimal,
            objective: 1e12,
            solution: vec![1e12],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..Default::default()
        };
        let guarded = guard_qp_optimal(corrupt, &prob);
        assert_eq!(
            guarded.status,
            crate::problem::SolveStatus::NumericalError,
            "guard must demote catastrophic QP result: primal violation 1e12-1 >> 1e-1"
        );
    }

    /// No-op proof: with_qp_guard_disabled bypasses guard; corrupt result passes through.
    ///
    /// Load-bearing evidence: guard_qp_optimal_catches_catastrophic_result proves the
    /// guard demotes the same corrupt data WITHOUT disabling. Together these tests form
    /// the no-op proof: removing the guard body breaks the first test.
    #[test]
    fn guard_qp_optimal_no_op_proof() {
        let prob = make_simple_eq_qp();
        let corrupt = SolverResult {
            status: crate::problem::SolveStatus::Optimal,
            objective: 1e12,
            solution: vec![1e12],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..Default::default()
        };
        let unguarded = with_qp_guard_disabled(|| guard_qp_optimal(corrupt, &prob));
        assert_eq!(
            unguarded.status,
            crate::problem::SolveStatus::Optimal,
            "with_qp_guard_disabled must pass corrupt result through as Optimal"
        );
    }

    /// guard_qp_optimal passes through valid Optimal results unchanged.
    ///
    /// Uses a 2-variable convex QP that IPM solves without reducing to 0 variables.
    #[test]
    fn guard_qp_optimal_passthrough_valid() {
        // min x1^2 + x2^2 s.t. x1 + x2 = 1, x1,x2 in [0,100]
        // Optimal: x1=x2=0.5, obj=0.25. Strictly convex → IPM solves cleanly.
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let prob = QpProblem::new(
            q, vec![0.0, 0.0], a, vec![1.0],
            vec![(0.0, 100.0), (0.0, 100.0)],
            vec![crate::problem::ConstraintType::Eq],
        ).unwrap();
        let opts = SolverOptions::default();
        let result = solve_ipm(&prob, &opts);
        assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
        // Re-run guard on the already-valid result — must remain Optimal.
        let re_guarded = guard_qp_optimal(result.clone(), &prob);
        assert_eq!(
            re_guarded.status,
            crate::problem::SolveStatus::Optimal,
            "guard must not demote a valid Optimal QP result"
        );
    }

    /// guard_qp_optimal passes through non-Optimal statuses unchanged.
    #[test]
    fn guard_qp_optimal_passthrough_non_optimal() {
        let prob = make_simple_eq_qp();
        for status in [
            crate::problem::SolveStatus::Infeasible,
            crate::problem::SolveStatus::Timeout,
            crate::problem::SolveStatus::NumericalError,
            crate::problem::SolveStatus::SuboptimalSolution,
        ] {
            let r = SolverResult { status: status.clone(), ..Default::default() };
            let out = guard_qp_optimal(r, &prob);
            assert_eq!(out.status, status, "guard must pass through {status:?}");
        }
    }

    /// x = D·x_s、z_orig = z_s/D の逆変換を直接検証。
    #[test]
    fn unscale_q_diagonal_reverses_x_and_bound_duals() {
        use crate::sparse::CscMatrix;
        let n = 3;
        let q = CscMatrix::from_triplets(
            &[0, 1, 2], &[0, 1, 2], &[1.0, 4.0, 9.0], n, n,
        ).unwrap();
        let prob = QpProblem::new_all_le(
            q, vec![1.0_f64; n],
            CscMatrix::new(0, n), vec![],
            vec![(0.0, 5.0), (0.0, f64::INFINITY), (f64::NEG_INFINITY, 3.0)],
        ).unwrap();
        let col_scales = vec![2.0_f64, 0.5, 4.0];
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0, 2.0, 3.0],
            dual_solution: vec![],
            bound_duals: vec![10.0, 20.0, 30.0, 40.0],
            ..SolverResult::default()
        };
        unscale_q_diagonal(&mut result, &col_scales, &prob);
        assert!((result.solution[0] - 2.0).abs() < 1e-12);
        assert!((result.solution[1] - 1.0).abs() < 1e-12);
        assert!((result.solution[2] - 12.0).abs() < 1e-12);
        assert!((result.bound_duals[0] - 5.0).abs() < 1e-12);
        assert!((result.bound_duals[1] - 40.0).abs() < 1e-12);
        assert!((result.bound_duals[2] - 15.0).abs() < 1e-12);
        assert!((result.bound_duals[3] - 10.0).abs() < 1e-12);
    }
}
