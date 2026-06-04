//! solve_ipm: 単一 retry 層 + API 境界 1 箇所での status 変換 + 元空間 KKT 直接判定。

use std::cell::Cell;
use std::time::Instant;

#[cfg(test)]
use crate::ScopedDisable;

use super::core::run_ipm;
use super::kkt::{bound_violation, kkt_residual_rel, primal_residual_rel};
use super::outcome::{IpmOutcome, ProblemView};
use crate::options::SolverOptions;
use crate::presolve::QpPresolveResult;
use crate::presolve::{
    qp_transforms::QpPresolveStatus, run_qp_presolve_phase1, run_qp_presolve_phase2,
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::certificate::prove_optimal;
use crate::qp::problem::QpProblem;
use crate::tolerances::{Q_DIAG_RANGE_TRIGGER, Q_OFFDIAG_ABS, Q_OFFDIAG_REL, UNDERFLOW_GUARD};

/// Residual threshold above which an Optimal/LocallyOptimal QP result is
/// considered catastrophically corrupt and demoted to NumericalError.
///
/// Set three orders of magnitude above typical convergence (1e-6) so that
/// only catastrophic failures (e.g. undetected postsolve corruption) trigger
/// this guard. Normal near-miss suboptimal results are handled by satisfies_eps.
const QP_GUARD_CATASTROPHIC_TOL: f64 = 1e-1;

thread_local! {
    static QP_GUARD_DISABLED: Cell<bool> = const { Cell::new(false) };
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
///
/// `eliminated_cols` is the presolve elimination mask (col_map[j].is_none()). It
/// must match the mask used by `finalize_outcome`/`prove_optimal` so the guard
/// applies the same EmptyCol stationarity convention: a LP-style fully-isolated
/// EmptyCol (A 列空 AND Q 列空) carries the `bd=0` convention residual `c_j` that
/// is NOT corruption. Without the mask the guard re-demotes a valid presolved
/// Optimal that finalize just accepted (Optimal → NumericalError). The narrow
/// skip condition in `kkt_residual_rel` never hides a non-empty column's genuine
/// stationarity violation (= a real false-Optimal), so this stays sound.
/// Pass `&[]` to disable skipping (length != n is ignored downstream).
pub(crate) fn guard_qp_optimal(
    result: SolverResult,
    problem: &QpProblem,
    eliminated_cols: &[bool],
) -> SolverResult {
    if QP_GUARD_DISABLED.with(|c| c.get()) {
        return result;
    }
    if !matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::LocallyOptimal
    ) {
        return result;
    }
    if result.solution.is_empty() {
        return result;
    }
    let view = ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    let kkt = kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
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
/// cost of re-solving without presolve reduction; it sits below
/// [`LARGE_PROBLEM_THRESHOLD`](crate::tolerances::LARGE_PROBLEM_THRESHOLD) (50_000),
/// so problems in between still get presolve+Ruiz but are deemed too large to
/// re-solve from scratch economically.
const NO_PRESOLVE_FALLBACK_LIMIT: usize = 10_000;

type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions) -> IpmOutcome;

/// tighten = ceil_pow10(user_eps / 1e-8) ∈ [1, 1000]。上限 1000 は IPM floor 制約。
///
/// `sigma_total` (minimum Ruiz / row-scale factor) was considered as an additional
/// divisor here, but bench showed it causes over-tightening that the no-presolve
/// fallback (below) must undo anyway. The fallback is the correct fix for ill-scaled
/// problems; removing sigma_total from this path is strictly simpler.
fn dynamic_base_tighten(user_eps: f64) -> f64 {
    const REF_EPS: f64 = 1e-8;
    let ratio = user_eps / REF_EPS;
    if ratio <= 1.0 {
        return 1.0;
    }
    let pow = ratio.log10().ceil();
    10_f64.powf(pow.min(3.0))
}

/// Q が対角なら s_j=1/√Q_jj の column scaling で Q'_jj=1 に均等化し、解後 x_orig=D·x_scaled で復元。
///
/// Returns [`SolverResult`] with [`SolveStatus::NumericalError`] immediately if
/// `options` fails validation (negative timeout, zero threads, etc.).
pub fn solve_ipm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.validate().is_err() {
        return SolverResult::numerical_error();
    }
    // `eliminated_cols` is structural (q-diag column scaling preserves which columns
    // are A-empty/Q-empty and which presolve removes), so the mask derived inside
    // `solve_ipm_with_runner` on the scaled problem is valid for the guard on the
    // original problem after unscale.
    if let Some((scaled_problem, col_scales)) = try_q_diagonal_scaling(problem) {
        let scaled_options = scale_warm_start_for_q_diag(options, &col_scales);
        let (mut result, eliminated_cols) =
            solve_ipm_with_runner(&scaled_problem, &scaled_options, run_ipm);
        unscale_q_diagonal(&mut result, &col_scales, problem);
        return guard_qp_optimal(result, problem, &eliminated_cols);
    }
    let (result, eliminated_cols) = solve_ipm_with_runner(problem, options, run_ipm);
    guard_qp_optimal(result, problem, &eliminated_cols)
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
            log::warn!(
                "warm_start_qp ignored: q_diag_scaling dim mismatch (x: {}, scales: {})",
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
    if n == 0 {
        return None;
    }

    let mut q_diag = vec![0.0_f64; n];
    for col in 0..n {
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            if problem.q.row_ind[k] == col {
                q_diag[col] = problem.q.values[k];
            }
        }
    }

    // Gate 1: each off-diagonal entry is compared against the local diagonal scale
    // min(|Q_ii|, |Q_jj|) so that a dominant unrelated diagonal (e.g. Q_kk >> Q_ii)
    // cannot accept an off-diagonal that would be amplified by column scaling
    // (s_j = 1/√Q_jj amplifies Q_ij by 1/√(Q_ii·Q_jj)).
    for col in 0..n {
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            if row != col {
                let local_scale = q_diag[row].abs().min(q_diag[col].abs());
                let offdiag_eps = Q_OFFDIAG_REL * local_scale + UNDERFLOW_GUARD;
                if problem.q.values[k].abs() > offdiag_eps {
                    return None;
                }
            }
        }
    }

    // Gates 2 & 3: use Q_OFFDIAG_ABS as the absolute floor for diagonal-positive
    // check. Scaling columns with Q_jj < Q_OFFDIAG_ABS produces extreme scale
    // factors (1/√Q_jj > 1e5) that destabilise the IPM.
    let mut q_pos_min = f64::INFINITY;
    let mut q_pos_max = 0.0_f64;
    for &v in &q_diag {
        if v > Q_OFFDIAG_ABS {
            q_pos_min = q_pos_min.min(v);
            q_pos_max = q_pos_max.max(v);
        }
    }
    if !q_pos_min.is_finite() || q_pos_max <= 0.0 {
        return None;
    }
    // Gate on Q diagonal range: only scale when range >= Q_DIAG_RANGE_TRIGGER, since
    // narrow-range Q does not benefit from diagonal scaling.
    if q_pos_max / q_pos_min < Q_DIAG_RANGE_TRIGGER {
        return None;
    }

    // s_j = 1/√Q_jj (Q_jj=0 の LP-like 列は s_j=1)、Q'_jj = 1。
    let mut col_scales = vec![1.0_f64; n];
    for j in 0..n {
        if q_diag[j] > Q_OFFDIAG_ABS {
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
    let c_s: Vec<f64> = problem
        .c
        .iter()
        .enumerate()
        .map(|(j, &v)| v * col_scales[j])
        .collect();

    // bounds' = bounds / D (s_j > 0 なので符号変わらず)
    let bounds_s: Vec<(f64, f64)> = problem
        .bounds
        .iter()
        .enumerate()
        .map(|(j, &(lb, ub))| (lb / col_scales[j], ub / col_scales[j]))
        .collect();

    // QpProblem を作る (b は不変、constraint_types も不変)。
    // obj_offset は scaling 不変なため orig から引き継ぐ。
    let mut scaled = match QpProblem::new(
        q_s,
        c_s,
        a_s,
        problem.b.clone(),
        bounds_s,
        problem.constraint_types.clone(),
    ) {
        Ok(p) => p,
        Err(_) => return None,
    };
    scaled.obj_offset = problem.obj_offset;

    Some((scaled, col_scales))
}

/// `try_q_diagonal_scaling` で行った column scaling を逆変換する。
/// x_orig = D × x_scaled, y は不変, y_lb/y_ub /= D.
fn unscale_q_diagonal(result: &mut SolverResult, col_scales: &[f64], orig_problem: &QpProblem) {
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

/// Returns the solved `SolverResult` plus the presolve elimination mask
/// (`col_map[j].is_none()`). The mask is forwarded to `guard_qp_optimal` so the
/// outer guard applies the same EmptyCol stationarity convention as `finalize_outcome`.
fn solve_ipm_with_runner(
    problem: &QpProblem,
    options: &SolverOptions,
    runner: IpmRunner,
) -> (SolverResult, Vec<bool>) {
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
    let presolve_result = if opts.presolve
        && problem.num_vars <= crate::tolerances::LARGE_PROBLEM_THRESHOLD
        && problem.num_constraints <= crate::tolerances::LARGE_PROBLEM_THRESHOLD
    {
        let phase1 = run_qp_presolve_phase1(problem, &opts);
        if opts.presolve_phase2 {
            run_qp_presolve_phase2(phase1, &opts)
        } else {
            phase1
        }
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    // presolve が物理削除した col の mask (core.rs::run_ipm_with と同方式)。
    // finalize の prove_optimal と外側 guard_qp_optimal が orig 空間 stationarity を
    // 評価する際、LP-style 完全孤立 EmptyCol (A 列空 AND Q 列空) を kkt_residual_rel が
    // skip するために必要。これを欠くと IPM 解に含まれない EmptyCol の bd=0 慣例値が
    // spurious 残差を生み、valid presolved Optimal が false-demote される (kkt.rs の
    // narrow 条件は非空列の本物の stationarity 違反は決して skip しないため AFIRO 等は安全)。
    let eliminated_cols: Vec<bool> = presolve_result
        .col_map
        .iter()
        .map(|c| c.is_none())
        .collect();

    if presolve_result.presolve_status == QpPresolveStatus::Infeasible {
        return (SolverResult::infeasible(), eliminated_cols);
    }
    if presolve_result.presolve_status == QpPresolveStatus::Unbounded {
        return (SolverResult::unbounded(), eliminated_cols);
    }

    let view = ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &eliminated_cols,
    };

    if total_deadline.is_some_and(|d| Instant::now() >= d) {
        let r = finalize_outcome(
            IpmOutcome::empty(),
            user_eps,
            n_orig,
            total_deadline,
            false,
            &view,
        );
        return (r, eliminated_cols);
    }

    // presolve Ruiz 済なら IPM 側で重ね掛けしない (二重 scale で誤収束する)。
    let presolve_did_ruiz = presolve_result.ruiz_scaler.is_some();
    let mut best: Option<IpmOutcome> = None;

    let user_max_iter = options.ipm.max_iter;
    let mut iter_used: usize = 0;

    let base_tighten = dynamic_base_tighten(user_eps);
    // no-Ruiz only when: presolve already Ruiz-scaled (double scaling wrong), or caller
    // explicitly disabled Ruiz (options.use_ruiz_scaling=false, e.g. no-Ruiz fallback path).
    let attempts: Vec<(bool, f64)> = if presolve_did_ruiz || !options.use_ruiz_scaling {
        let mut v = vec![(false, base_tighten), (false, base_tighten * 10.0)];
        if base_tighten > 10.0 {
            v.push((false, base_tighten / 10.0));
        }
        if base_tighten > 1.0 {
            v.push((false, 1.0));
        }
        v
    } else {
        let mut v = vec![
            (true, base_tighten),
            (false, base_tighten),
            (true, base_tighten * 10.0),
            (false, base_tighten * 10.0),
            (true, base_tighten * 100.0),
            (false, base_tighten * 100.0),
        ];
        if base_tighten > 10.0 {
            v.push((true, base_tighten / 10.0));
            v.push((false, base_tighten / 10.0));
        }
        if base_tighten > 1.0 {
            v.push((true, 1.0));
            v.push((false, 1.0));
        }
        v
    };

    for &(use_ruiz, tighten) in attempts.iter() {
        if let Some(d) = total_deadline {
            if Instant::now() >= d {
                break;
            }
        }
        if iter_used >= user_max_iter {
            break;
        }
        let remaining = user_max_iter.saturating_sub(iter_used);
        let per_attempt_cap = MAX_ITER_PER_ATTEMPT.min(remaining);
        opts.deadline = total_deadline;
        opts.timeout_secs = None;
        opts.ipm.max_iter = per_attempt_cap;
        opts.use_ruiz_scaling = use_ruiz;
        opts.ipm.eps = (user_eps / tighten).max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);

        let outcome = runner(problem, &presolve_result, &opts);
        // Charge per_attempt_cap for failed attempts: stall paths return best_iter which
        // can be far below the actual iterations consumed, causing the outer guard to
        // undercount and permit more total iterations than user_max_iter.
        let charged = if outcome.satisfies_eps(user_eps) {
            outcome.iterations
        } else {
            per_attempt_cap
        };
        iter_used = iter_used.saturating_add(charged);

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
    let best_ok = best
        .as_ref()
        .map(|b| b.satisfies_eps(user_eps))
        .unwrap_or(false);
    if !best_ok && presolve_did_ruiz && n_orig <= NO_PRESOLVE_FALLBACK_LIMIT {
        let fallback_pre = QpPresolveResult::no_reduction(problem);
        for use_ruiz_fb in [false, true] {
            if total_deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            if iter_used >= user_max_iter {
                break;
            }
            let remaining = user_max_iter.saturating_sub(iter_used);
            let per_attempt_cap = MAX_ITER_PER_ATTEMPT.min(remaining);
            opts.deadline = total_deadline;
            opts.timeout_secs = None;
            opts.ipm.max_iter = per_attempt_cap;
            opts.use_ruiz_scaling = use_ruiz_fb;
            opts.tolerance = None;
            // Tighten the inner target like the main attempt loop: the IPM stops on
            // the scale-aggregated complementarity, but prove_optimal accepts on the
            // stricter component-wise complementarity. Solving only to user_eps leaves
            // the worst component just above tol (false SuboptimalSolution); base_tighten
            // drives it below tol. acceptance is still gated by satisfies_eps(user_eps).
            opts.ipm.eps = (user_eps / base_tighten).max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);
            let fb = runner(problem, &fallback_pre, &opts);
            let charged_fb = if fb.satisfies_eps(user_eps) {
                fb.iterations
            } else {
                per_attempt_cap
            };
            iter_used = iter_used.saturating_add(charged_fb);
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
    let r = finalize_outcome(outcome, user_eps, n_orig, total_deadline, cancelled, &view);
    (r, eliminated_cols)
}

/// IpmOutcome → SolverResult: eps 達成→Optimal、外部停止→Timeout、内部停止→Suboptimal、解無し→NumericalError。
///
/// eps 達成時は `prove_optimal` で KKT + dual_sign を再検証する。`satisfies_eps` は dual_sign
/// チェックを含まないため、prove_optimal が唯一の Optimal mint 関数として機能する。
/// `prove_optimal` が `Err(NotProven)` を返す場合は SuboptimalSolution に降格する。
///
/// ## Gap 基準の意図的な厳格化
///
/// `IpmOutcome::satisfies_eps` は duality gap を `PROMOTION_GAP_TOL = 1e-1` (10 %) と比較する。
/// これは retry ループで *最良の iterate* を選ぶための構造的な緩い閾値であり、
/// 最終的な Optimal 判定には用いない。
/// `prove_optimal` はすべての KKT 条件 (gap 含む) を `user_eps` で検証するため、
/// gap が (user_eps, PROMOTION_GAP_TOL) の範囲にある解は SuboptimalSolution に降格する。
/// これはユーザが要求した精度での honest な Optimal 定義であり、意図的な supersede である。
fn finalize_outcome(
    outcome: IpmOutcome,
    user_eps: f64,
    n_orig: usize,
    total_deadline: Option<Instant>,
    cancelled: bool,
    view: &ProblemView<'_>,
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

    // numerical_failure は run_ipm の validate ガードまたは内部ソルバー失敗が
    // 明示セットする。solution.is_empty() に依存せず直接 NumericalError へ map
    // することで、numerical_failure=true かつ solution 非空の誤分類を防ぐ。
    if outcome.numerical_failure {
        return SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            iterations: outcome.iterations,
            ..Default::default()
        };
    }

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
        // prove_optimal: KKT 全条件 (stationarity / primal_feas / bound_feas /
        // complementarity / dual_sign / duality_gap) を tol=user_eps で再検証。
        // satisfies_eps が欠く dual_sign チェックを追加し、唯一の Optimal mint 経路にする。
        // z layout: [lb-half (z_lb≥0), ub-half (z_ub≥0)] = bound_contrib 規約に準拠。
        let proven = prove_optimal(
            view,
            &outcome.solution,
            &outcome.dual_solution,
            &outcome.bound_duals,
            outcome.duality_gap_rel,
            user_eps,
        );
        if proven.is_ok() {
            if outcome.is_locally_optimal {
                SolveStatus::LocallyOptimal
            } else {
                SolveStatus::Optimal
            }
        } else {
            // KKT または dual_sign が tol 超 → Optimal を主張しない。
            SolveStatus::SuboptimalSolution
        }
    } else if timed_out {
        SolveStatus::Timeout
    } else {
        SolveStatus::SuboptimalSolution
    };

    debug_assert_eq!(
        outcome.solution.len(),
        n_orig,
        "outcome solution dimension mismatch"
    );

    let result = SolverResult {
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
    };
    debug_assert!(
        result.reduced_costs.is_empty(),
        "IPM SolverResult must never contain reduced_costs; got len={}",
        result.reduced_costs.len(),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// B.1 sentinel: IPM finalize_outcome must never set reduced_costs.
    /// The debug_assert fires if any code path were to populate this field.
    /// Verified by `..Default::default()` contract; sentinel catches future regressions.
    #[test]
    fn ipm_finalize_outcome_reduced_costs_empty() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            a,
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let result = solve_ipm(&prob, &SolverOptions::default());
        assert!(
            result.reduced_costs.is_empty(),
            "IPM result must never contain reduced_costs (len={})",
            result.reduced_costs.len(),
        );
    }

    /// Case D fixture (#15 P2 root): 1 strictly-convex var + 1 LP-style isolated
    /// EmptyCol whose bound-dual recovery the masked postsolve guard reverts.
    ///
    /// min 0.5·x0² + x1  s.t. x0∈[−10,10], x1∈[0,5], NO linear constraints.
    /// Optimal: x0=0, x1=0 (lb), obj=0. x1 is A-empty AND Q-empty (EmptyCol),
    /// c1=1>0 so z_lb1=1. The IPM solves x0 (already exact), so the masked refit
    /// guard reverts x1's z to 0 → original-space stationarity for x1 = c1 = 1.
    fn make_convex_plus_empty_col_qp() -> QpProblem {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
        let a = CscMatrix::new(0, 2);
        QpProblem::new_all_le(
            q,
            vec![0.0, 1.0],
            a,
            vec![],
            vec![(-10.0, 10.0), (0.0, 5.0)],
        )
        .unwrap()
    }

    /// End-to-end regression: solve_ipm must report Optimal for a valid presolved
    /// QP with an isolated EmptyCol. Before the eliminated_cols mask reached
    /// finalize_outcome/guard_qp_optimal, this false-demoted to SuboptimalSolution
    /// (cert) and then NumericalError (guard).
    #[test]
    fn empty_col_qp_solves_optimal_not_false_demoted() {
        let prob = make_convex_plus_empty_col_qp();
        let result = solve_ipm(&prob, &SolverOptions::default());
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::Optimal,
            "isolated EmptyCol QP must not be false-demoted (got {:?})",
            result.status,
        );
        assert!(
            (result.objective - 0.0).abs() < 1e-6,
            "obj={}",
            result.objective
        );
        assert!(result.solution[0].abs() < 1e-6, "x0={}", result.solution[0]);
        assert!(result.solution[1].abs() < 1e-6, "x1={}", result.solution[1]);
    }

    /// No-op proof for the eliminated_cols mask in finalize_outcome.
    ///
    /// Builds the exact IPM iterate the pipeline produces for the EmptyCol fixture
    /// (x1's z_lb reverted to 0). With the mask the EmptyCol stationarity (c1=1) is
    /// excluded → prove_optimal passes → Optimal. WITHOUT the mask (`&[]`) the same
    /// iterate exposes stationarity 0.5 → prove_optimal Err → SuboptimalSolution.
    ///
    /// **Sentinel**: dropping the mask at attempt.rs (reverting to `from_problem`)
    /// makes the masked branch return SuboptimalSolution → this test FAILs.
    #[test]
    fn empty_col_mask_noop_proof_in_finalize() {
        let prob = make_convex_plus_empty_col_qp();
        // bound_duals layout: n_lb=2 (both lb finite), n_ub=2 → [z_lb0, z_lb1, z_ub0, z_ub1].
        // z_lb1=0 reproduces the reverted EmptyCol dual (the bug state).
        let outcome = IpmOutcome {
            solution: vec![0.0, 0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0, 0.0, 0.0],
            objective: 0.0,
            iterations: 5,
            // Stored residuals as computed by the masked core.rs path (EmptyCol excluded).
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 0.0,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
        };
        assert!(
            outcome.satisfies_eps(1e-6),
            "stored residuals must pass satisfies_eps"
        );

        // WITH mask: EmptyCol (col 1, A-empty AND Q-empty AND eliminated) skipped → Optimal.
        let mask = vec![false, true];
        let view_masked = ProblemView {
            q: &prob.q,
            a: &prob.a,
            c: &prob.c,
            b: &prob.b,
            bounds: &prob.bounds,
            constraint_types: &prob.constraint_types,
            eliminated_cols: &mask,
        };
        let r_masked = finalize_outcome(outcome.clone(), 1e-6, 2, None, false, &view_masked);
        assert_eq!(
            r_masked.status,
            crate::problem::SolveStatus::Optimal,
            "masked view must accept the valid presolved Optimal",
        );

        // WITHOUT mask: EmptyCol stationarity (c1=1, rel=0.5) exposed → demote.
        let view_unmasked = ProblemView::from_problem(&prob);
        let r_unmasked = finalize_outcome(outcome, 1e-6, 2, None, false, &view_unmasked);
        assert_eq!(
            r_unmasked.status,
            crate::problem::SolveStatus::SuboptimalSolution,
            "no-op proof: empty mask must false-demote (mask is load-bearing)",
        );
    }

    /// Safety sentinel: the narrow mask must NOT hide a genuine false-Optimal on a
    /// NON-empty (A-non-empty) column — mirrors AFIRO's structure (all columns have
    /// A entries; 0 structurally-empty cols). Even with eliminated_cols[j]=true, a
    /// column with A entries is never skipped, so a real stationarity violation
    /// still demotes to SuboptimalSolution.
    ///
    /// Fixture: min x  s.t. x = 5 (Eq, A col non-empty), x∈[0,10]. Provide a wrong
    /// iterate x=0 with y=0, z=0 → stationarity r = c + Aᵀy + bc = 1 ≠ 0. Mark the
    /// column eliminated (mask=true) to prove the mask does not hide it.
    #[test]
    fn mask_does_not_hide_nonempty_col_false_optimal() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![1.0],
            a,
            vec![5.0],
            vec![(0.0, 10.0)],
            vec![ConstraintType::Eq],
        )
        .unwrap();

        // Wrong iterate: x=0 (violates x=5), y=0, z=0 → stationarity = c = 1, and
        // primal violation too. Stored residuals forced to 0 so satisfies_eps passes
        // and prove_optimal is the gate under test.
        let outcome = IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0, 0.0],
            objective: 0.0,
            iterations: 5,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 0.0,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
        };
        // mask=true on the A-non-empty col: narrow condition (a_empty) is false → not skipped.
        let mask = vec![true];
        let view = ProblemView {
            q: &prob.q,
            a: &prob.a,
            c: &prob.c,
            b: &prob.b,
            bounds: &prob.bounds,
            constraint_types: &prob.constraint_types,
            eliminated_cols: &mask,
        };
        let result = finalize_outcome(outcome, 1e-6, 1, None, false, &view);
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::SuboptimalSolution,
            "mask must NOT hide a non-empty column's genuine violation (AFIRO-safety)",
        );
    }

    /// Gate 1 sentinel: `try_q_diagonal_scaling` uses local diagonal scale
    /// `min(|Q_ii|, |Q_jj|)` for each off-diagonal entry, not `Q_OFFDIAG_ABS`.
    ///
    /// Fixture: Q = diag([1e9, 1e3]) with off-diagonal 5e-10.
    /// - local_scale = min(1e9, 1e3) = 1e3
    /// - offdiag_eps = Q_OFFDIAG_REL × 1e3 = 1e-9
    /// - 5e-10 < 1e-9 → Gate 1 passes; range = 1e6 → Gate 3 passes → Some
    ///
    /// **Sentinel**: reverting Gate 1 to `offdiag_eps = Q_OFFDIAG_ABS = 1e-10`
    /// makes 5e-10 > 1e-10 → Gate 1 fails → None → this test FAILS.
    #[test]
    fn try_q_diagonal_scaling_gate1_local_scale_sentinel() {
        let q =
            CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1e9_f64, 5e-10, 1e3], 2, 2).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(0.0, 1.0), (0.0, 1.0)],
        )
        .unwrap();
        assert!(
            try_q_diagonal_scaling(&prob).is_some(),
            "Gate 1 local scale: 5e-10 < Q_OFFDIAG_REL×1e3=1e-9 → scaling must trigger"
        );
    }

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
            q.clone(),
            c.clone(),
            a.clone(),
            b.clone(),
            bounds.clone(),
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap();

        let (scaled, col_scales) =
            try_q_diagonal_scaling(&prob).expect("ill-cond diag Q must trigger");
        let q_s = &scaled.q;
        for col in 0..2 {
            for k in q_s.col_ptr[col]..q_s.col_ptr[col + 1] {
                if q_s.row_ind[k] == col {
                    assert!(
                        (q_s.values[k] - 1.0).abs() < 1e-12,
                        "got {} at col {}",
                        q_s.values[k],
                        col
                    );
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
        let prob =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

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
        let guarded = guard_qp_optimal(corrupt, &prob, &[]);
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
        let unguarded = with_qp_guard_disabled(|| guard_qp_optimal(corrupt, &prob, &[]));
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
            q,
            vec![0.0, 0.0],
            a,
            vec![1.0],
            vec![(0.0, 100.0), (0.0, 100.0)],
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap();
        let opts = SolverOptions::default();
        let result = solve_ipm(&prob, &opts);
        assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
        // Re-run guard on the already-valid result — must remain Optimal.
        let re_guarded = guard_qp_optimal(result.clone(), &prob, &[]);
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
            let r = SolverResult {
                status: status.clone(),
                ..Default::default()
            };
            let out = guard_qp_optimal(r, &prob, &[]);
            assert_eq!(out.status, status, "guard must pass through {status:?}");
        }
    }

    /// prove_optimal が dual_sign 違反で Err を返す場合、finalize_outcome は
    /// SuboptimalSolution を返す (Optimal を主張しない)。
    ///
    /// 構成: A=[[1],[-1]], b=[1,-1] の cancelling-Le QP で x=1 は両制約が active。
    /// y_bad=[-v,-v] では stationarity = [1,-1]·[-v,-v] = -v+v = 0 (cancels)、
    /// comp = y_i·slack_i = (-v)·0 = 0 (active constraint)。
    /// よって kkt/primal/bound/comp はすべて 0 だが dual_sign は Le で y<0 → 違反。
    /// satisfies_eps はパスするが prove_optimal が dual_sign で Err → SuboptimalSolution。
    #[test]
    fn finalize_outcome_dual_sign_notproven_demotes_to_suboptimal() {
        use crate::problem::ConstraintType;
        use crate::sparse::CscMatrix;

        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0usize, 1], &[0, 0], &[1.0_f64, -1.0], 2, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![1.0, -1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le, ConstraintType::Le],
        )
        .unwrap();
        let view = ProblemView::from_problem(&prob);
        let user_eps = 1e-6_f64;

        // y=[-0.1,-0.1] は Le 制約で dual_sign 違反 (Le → y≥0 required)。
        // stationarity: A^T y = [1,-1]·[-0.1,-0.1] = -0.1+0.1 = 0 (キャンセル)。
        // comp: y_i·slack_i = (-0.1)·0 = 0 (x=1 で両制約が active)。
        let outcome = IpmOutcome {
            solution: vec![1.0],
            dual_solution: vec![-0.1, -0.1],
            bound_duals: vec![],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 0.0,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
        };

        assert!(
            outcome.satisfies_eps(user_eps),
            "satisfies_eps must pass: all residuals=0, gap=0 (dual_sign は未検査)"
        );
        let result = finalize_outcome(outcome, user_eps, 1, None, false, &view);
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::SuboptimalSolution,
            "dual_sign 違反 → prove_optimal が Err → SuboptimalSolution に降格すべき"
        );
    }

    /// finalize_outcome が prove_optimal を通過する正常ケースの確認。
    ///
    /// A=[[1],[-1]], b=[1,-1] で x=1、y=[v,v] (v>0, Le 符号正) は
    /// dual_sign を含む全条件を通過し Optimal が返る。
    #[test]
    fn finalize_outcome_dual_sign_valid_returns_optimal() {
        use crate::problem::ConstraintType;
        use crate::sparse::CscMatrix;

        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0usize, 1], &[0, 0], &[1.0_f64, -1.0], 2, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![1.0, -1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le, ConstraintType::Le],
        )
        .unwrap();
        let view = ProblemView::from_problem(&prob);
        let user_eps = 1e-6_f64;

        // y=[v,v] (v>0) は stationarity キャンセル + Le 符号正 → 全条件通過。
        let outcome = IpmOutcome {
            solution: vec![1.0],
            dual_solution: vec![0.1, 0.1],
            bound_duals: vec![],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 0.0,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
        };

        assert!(outcome.satisfies_eps(user_eps));
        let result = finalize_outcome(outcome, user_eps, 1, None, false, &view);
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::Optimal,
            "有効な dual は prove_optimal を通過し Optimal が返るべき"
        );
    }

    /// finalize_outcome は numerical_failure=true を solution 非空でも NumericalError へ map する。
    ///
    /// **Sentinel**: `finalize_outcome` の `numerical_failure` 明示チェックを削除すると、
    /// non-empty solution は `solution.is_empty()` を通過し `satisfies_eps` 判定に進む。
    /// `numerical_failure=true` は `satisfies_eps` を false にするため `SuboptimalSolution` が返り、
    /// このテストは FAIL する。
    #[test]
    fn finalize_outcome_numerical_failure_maps_to_numerical_error() {
        let prob = make_simple_eq_qp();
        let view = ProblemView::from_problem(&prob);
        // numerical_failure=true だが solution は非空 — solution.is_empty() では捕捉されない。
        let outcome = IpmOutcome {
            solution: vec![1.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0, 0.0],
            objective: 1.0,
            iterations: 3,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 0.0,
            numerical_failure: true,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
        };
        let result = finalize_outcome(outcome, 1e-6, 1, None, false, &view);
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::NumericalError,
            "numerical_failure=true は solution 非空でも NumericalError でなければならない",
        );
    }

    /// x = D·x_s、z_orig = z_s/D の逆変換を直接検証。
    #[test]
    fn unscale_q_diagonal_reverses_x_and_bound_duals() {
        use crate::sparse::CscMatrix;
        let n = 3;
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 4.0, 9.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![1.0_f64; n],
            CscMatrix::new(0, n),
            vec![],
            vec![(0.0, 5.0), (0.0, f64::INFINITY), (f64::NEG_INFINITY, 3.0)],
        )
        .unwrap();
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

    /// Sentinel: per_attempt_cap is charged for failed attempts, not outcome.iterations.
    ///
    /// Injects a mock runner that always returns `iterations=0` (simulating stall paths
    /// where `IpmOutcome.iterations = best_iter << actual iterations consumed`). With
    /// `user_max_iter=2`, the first attempt charges `per_attempt_cap=2`; the guard
    /// `iter_used >= user_max_iter` triggers immediately and the loop stops.
    ///
    /// **Sentinel**: reverting to `iter_used += outcome.iterations` leaves iter_used=0
    /// after the first attempt → the guard never triggers → all attempts run (count > 1).
    #[test]
    fn iter_guard_charges_per_attempt_cap_on_failed_attempt() {
        use std::cell::Cell;
        thread_local! {
            static CALL_COUNT: Cell<usize> = const { Cell::new(0) };
        }

        fn mock_runner(_: &QpProblem, _: &QpPresolveResult, _: &SolverOptions) -> IpmOutcome {
            CALL_COUNT.with(|c| c.set(c.get() + 1));
            // iterations=0 simulates stall best_iter undercount; never converges.
            IpmOutcome::empty()
        }

        let prob = make_simple_eq_qp();
        let mut opts = SolverOptions::default();
        opts.ipm.max_iter = 2;
        opts.presolve = false; // skip presolve to isolate the attempt loop
        CALL_COUNT.with(|c| c.set(0));

        let _ = solve_ipm_with_runner(&prob, &opts, mock_runner);

        let count = CALL_COUNT.with(|c| c.get());
        // With the fix: attempt 1 charges per_attempt_cap=2 → iter_used=2 >= 2 → stops.
        // Without fix (charge outcome.iterations=0): iter_used never advances → all
        // attempts run → count >> 1.
        assert_eq!(
            count, 1,
            "iter guard must stop after 1 attempt when per_attempt_cap charges full budget \
             (got {} runner calls)",
            count
        );
    }
}
