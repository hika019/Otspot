//! solve_ipm: 単一 retry 層 + API 境界 1 箇所での status 変換 + 元空間 KKT 直接判定。

use std::cell::Cell;
use std::time::Instant;

#[cfg(test)]
use crate::ScopedDisable;

use super::core::run_ipm_with_user_eps;
use super::kkt::{bound_violation, kkt_residual_rel, primal_residual_rel};
use super::outcome::{IpmOutcome, IpmTermination, ProblemView};
use crate::options::SolverOptions;
use crate::presolve::QpPresolveResult;
use crate::presolve::{
    qp_transforms::QpPresolveStatus, run_qp_presolve_phase1, run_qp_presolve_phase2,
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::certificate::prove_optimal;
use crate::qp::problem::QpProblem;
use crate::tolerances::{Q_DIAG_RANGE_TRIGGER, Q_OFFDIAG_REL, UNDERFLOW_GUARD};

/// Smallest positive diagonal eligible for optional numerical column scaling.
/// This gate never removes or changes an input Q coefficient.
const Q_DIAG_SCALING_MIN: f64 = 1e-10;

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

/// 同一 lane (use_ruiz) 内で bit 同一の outcome がこの回数連続したら決定的 stall
/// と断定し、残り attempt を打ち切る。
///
/// 根拠 (dfl001 netlib LP, 実測 trace): lane=true は tighten=100→1000 で 2 連続
/// bit 同一 (iter=115 のまま) だったが、tighten=10000 の 3 投目で iter=136 に
/// 変化し脱出できた。2 連続で打ち切ると、この 3 投目や反対 lane
/// (false,10000→Stalled iter=211) が best 候補になる機会を失い、crossover が
/// 完遂できず Optimal→Stalled に退化する (旧: PASS 365s → 退化後: STALLED 137s)。
/// QPLIB_0018 のような真の決定論的 stall は全 attempt が同一のため 3 連続でも
/// 検出・打ち切り可能 (実測: 3 attempt で break、旧 2 attempt から微増)。
const CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK: usize = 3;

type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions, f64) -> IpmOutcome;

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

fn outcome_proves_optimal(outcome: &IpmOutcome, view: &ProblemView<'_>, user_eps: f64) -> bool {
    if !outcome.satisfies_eps(user_eps) {
        return false;
    }
    prove_optimal(
        view,
        &outcome.solution,
        &outcome.dual_solution,
        &outcome.bound_duals,
        outcome.duality_gap_rel,
        user_eps,
    )
    .is_ok()
}

fn outcome_certificate_score(outcome: &IpmOutcome, view: &ProblemView<'_>, user_eps: f64) -> f64 {
    if outcome.solution.is_empty() || outcome.numerical_failure {
        return f64::INFINITY;
    }
    match prove_optimal(
        view,
        &outcome.solution,
        &outcome.dual_solution,
        &outcome.bound_duals,
        outcome.duality_gap_rel,
        user_eps,
    ) {
        Ok(_) => 0.0,
        Err(not_proven) => outcome
            .quality_score()
            .max(not_proven.stationarity_rel.abs())
            .max(not_proven.primal_residual_rel.abs())
            .max(not_proven.bound_violation.abs())
            .max(not_proven.complementarity_rel.abs())
            .max(not_proven.duality_gap_rel.abs())
            .max(not_proven.dual_sign_violation.abs()),
    }
}

fn outcome_is_better_candidate(
    candidate: &IpmOutcome,
    incumbent: &IpmOutcome,
    view: &ProblemView<'_>,
    user_eps: f64,
) -> bool {
    match (
        candidate.infeasibility_status.is_some(),
        incumbent.infeasibility_status.is_some(),
    ) {
        (true, false) => return false,
        (false, true) => return true,
        (true, true) => return false,
        (false, false) => {}
    }

    let candidate_score = outcome_certificate_score(candidate, view, user_eps);
    let incumbent_score = outcome_certificate_score(incumbent, view, user_eps);
    match candidate_score.total_cmp(&incumbent_score) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => candidate.objective.total_cmp(&incumbent.objective).is_lt(),
    }
}

/// 2 連続 attempt が bit 同一の iterate で終わったか (決定的 stall の検出)。
///
/// tighten だけが違う attempt は、stall が inner eps に依存しない位置で起きると
/// 完全に同じ軌道をなぞる (QPLIB_0018: 6 attempt 全てが iter 61 で同一 stall)。
/// bit 同一なら以降の attempt も同じ結果にしかならないため打ち切る。
fn attempts_bitwise_identical(prev: &AttemptFingerprint, outcome: &IpmOutcome) -> bool {
    prev.iterations == outcome.iterations
        && prev.objective_bits == outcome.objective.to_bits()
        && slices_bitwise_equal(&prev.solution, &outcome.solution)
        && slices_bitwise_equal(&prev.dual_solution, &outcome.dual_solution)
}

fn slices_bitwise_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// 直前 attempt の同一性判定に必要な最小コピー。
struct AttemptFingerprint {
    iterations: usize,
    objective_bits: u64,
    solution: Vec<f64>,
    dual_solution: Vec<f64>,
}

impl AttemptFingerprint {
    fn of(outcome: &IpmOutcome) -> Self {
        Self {
            iterations: outcome.iterations,
            objective_bits: outcome.objective.to_bits(),
            solution: outcome.solution.clone(),
            dual_solution: outcome.dual_solution.clone(),
        }
    }
}

fn fallback_can_replace_unproven(
    fallback: &IpmOutcome,
    incumbent: &IpmOutcome,
    view: &ProblemView<'_>,
    user_eps: f64,
) -> bool {
    if incumbent.infeasibility_status.is_some() {
        return false;
    }
    outcome_is_better_candidate(fallback, incumbent, view, user_eps)
        && fallback.objective.total_cmp(&incumbent.objective).is_le()
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
            solve_ipm_with_runner(&scaled_problem, &scaled_options, run_ipm_with_user_eps);
        unscale_q_diagonal(&mut result, &col_scales, problem);
        return guard_qp_optimal(result, problem, &eliminated_cols);
    }
    let (result, eliminated_cols) = solve_ipm_with_runner(problem, options, run_ipm_with_user_eps);
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

    // Gates 2 & 3: use Q_DIAG_SCALING_MIN as the floor for diagonal-positive
    // check. Scaling columns with Q_jj < Q_DIAG_SCALING_MIN produces extreme scale
    // factors (1/√Q_jj > 1e5) that destabilise the IPM.
    let mut q_pos_min = f64::INFINITY;
    let mut q_pos_max = 0.0_f64;
    for &v in &q_diag {
        if v > Q_DIAG_SCALING_MIN {
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
        if q_diag[j] > Q_DIAG_SCALING_MIN {
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
        let expected_bound_duals = orig_problem
            .bounds
            .iter()
            .filter(|&&(lb, _)| lb.is_finite())
            .count()
            + orig_problem
                .bounds
                .iter()
                .filter(|&&(_, ub)| ub.is_finite())
                .count();
        assert_eq!(
            result.bound_duals.len(),
            expected_bound_duals,
            "Q diagonal unscale requires one bound dual per finite bound"
        );
        let mut idx = 0_usize;
        for (j, &(lb, _)) in orig_problem.bounds.iter().enumerate() {
            if lb.is_finite() {
                result.bound_duals[idx] /= col_scales[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in orig_problem.bounds.iter().enumerate() {
            if ub.is_finite() {
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
    // The requested accuracy is captured in `user_eps`; from here on `opts` is
    // the *inner* option set. A still-set `tolerance` would make every
    // downstream `ipm_eps()` (IPPMM convergence target, Ruiz eps adjustment)
    // return `user_eps` and silently bypass the attempt loop's
    // `opts.ipm.eps = user_eps / tighten` inner-target tightening — the IPM
    // then stops at user_eps aggregate accuracy, prove_optimal fails on the
    // stricter component-wise check, and postsolve refine burns the whole
    // budget grinding an under-converged iterate (POWELL20: 0.5s → 1000s).
    opts.tolerance = None;

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
    // use_ruiz ごとに直前 fingerprint を保持 (index: false=0, true=1)。attempts 列は
    // [(true,t),(false,t),(true,10t),(false,10t),...] と tighten 固定・use_ruiz 反転の
    // ペアを含むため、lane を跨いだ比較 (Ruiz flip のみのペア) は「tighten を変えても
    // 変化なし」の証拠にならない。同じ lane 内で tighten が異なる直前 attempt とのみ
    // 比較する。
    let mut prev_by_ruiz: [Option<(f64, AttemptFingerprint)>; 2] = [None, None];
    // 同一 lane 内で連続 bit 同一だった run の長さ (直近 attempt を含む)。tighten が
    // 変わって bit が変化したら 1 にリセットする。break は
    // CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK に到達したときのみ。
    let mut run_len_by_ruiz: [usize; 2] = [0, 0];

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

    opts.schur_hint = Some(crate::qp::ipm_core::ippmm::probe_schur_decision(
        &presolve_result.reduced,
        &opts,
    ));

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

        let outcome = runner(problem, &presolve_result, &opts, user_eps);
        let outcome_satisfies = outcome.satisfies_eps(user_eps);
        let outcome_proven = outcome_satisfies && outcome_proves_optimal(&outcome, &view, user_eps);
        // Charge per_attempt_cap for failed attempts: stall paths return best_iter which
        // can be far below the actual iterations consumed, causing the outer guard to
        // undercount and permit more total iterations than user_max_iter.
        let charged = if outcome_satisfies {
            outcome.iterations
        } else {
            per_attempt_cap
        };
        iter_used = iter_used.saturating_add(charged);

        if outcome_proven {
            best = Some(outcome);
            break;
        }
        // 同一 lane (use_ruiz) 内で tighten を変えても bit 同一な run が
        // CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK 回続いたら残り attempt をスキップ
        // (決定的 stall)。2 連続だけでは早計 (dfl001: tighten=100→1000 で 2 連続
        // 同一でも 3 投目の tighten=10000 で iter が変化し脱出できた)。
        // termination==Converged (scaled 空間では収束したが元空間 eps に届かない
        // 精度床) は「inner eps を変えれば結果も変わる」が定義そのものなので run を
        // リセットし打ち切り対象から外す — tighten ladder はその救済機構。
        let lane = usize::from(use_ruiz);
        let is_converged = outcome.termination == IpmTermination::Converged;
        let bit_matches_prev = !is_converged
            && prev_by_ruiz[lane]
                .as_ref()
                .is_some_and(|(prev_tighten, prev)| {
                    *prev_tighten != tighten && attempts_bitwise_identical(prev, &outcome)
                });
        run_len_by_ruiz[lane] = if is_converged {
            0
        } else if bit_matches_prev {
            run_len_by_ruiz[lane] + 1
        } else {
            1
        };
        prev_by_ruiz[lane] = Some((tighten, AttemptFingerprint::of(&outcome)));
        match &best {
            None => best = Some(outcome),
            Some(prev) if outcome_is_better_candidate(&outcome, prev, &view, user_eps) => {
                best = Some(outcome);
            }
            _ => {}
        }
        if run_len_by_ruiz[lane] >= CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK {
            break;
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
        .map(|b| outcome_proves_optimal(b, &view, user_eps))
        .unwrap_or(false);
    if !best_ok && presolve_did_ruiz && n_orig <= NO_PRESOLVE_FALLBACK_LIMIT {
        opts.schur_hint = None;
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
            // Tighten the inner target like the main attempt loop: the IPM stops on
            // the scale-aggregated complementarity, but prove_optimal accepts on the
            // stricter component-wise complementarity. Solving only to user_eps leaves
            // the worst component just above tol (false SuboptimalSolution); base_tighten
            // drives it below tol. acceptance is still gated by prove_optimal(user_eps).
            opts.ipm.eps = (user_eps / base_tighten).max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);
            let fb = runner(problem, &fallback_pre, &opts, user_eps);
            let fb_satisfies = fb.satisfies_eps(user_eps);
            let fb_proven = fb_satisfies && outcome_proves_optimal(&fb, &view, user_eps);
            let charged_fb = if fb_satisfies {
                fb.iterations
            } else {
                per_attempt_cap
            };
            iter_used = iter_used.saturating_add(charged_fb);
            if fb_proven {
                best = Some(fb);
                break;
            }
            if best
                .as_ref()
                .is_some_and(|prev| fallback_can_replace_unproven(&fb, prev, &view, user_eps))
            {
                best = Some(fb);
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

/// IpmOutcome → SolverResult。解品質を主張する status (Optimal / LocallyOptimal /
/// SuboptimalSolution) は品質ゲートを通った iterate のみが名乗る:
///
/// - `satisfies_eps(user_eps)` + `prove_optimal` Ok → Optimal / LocallyOptimal
///   (prove_optimal が唯一の Optimal mint 関数)
/// - `satisfies_eps(user_eps)` + `prove_optimal` Err → SuboptimalSolution
///   (eps 品質は検証済み、証明書 (dual_sign / gap ≤ user_eps) のみ未達)
/// - eps 未達 + 外部停止 → Timeout
/// - eps 未達 + 内部停止 → 終端条件どおり Stalled (停滞 / 精度床) または
///   MaxIterations (予算枯渇)。解品質は主張しない (solution は診断用 iterate)。
/// - 解無し → NumericalError (外部停止なら Timeout)
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
            // satisfies_eps 済みなので「検証済み解 + 証明書なし」= SuboptimalSolution。
            SolveStatus::SuboptimalSolution
        }
    } else if timed_out {
        SolveStatus::Timeout
    } else {
        // eps 未達の非 timeout 終端。解品質を主張する status は使わず、
        // 内部終端条件をそのまま報告する。Converged (scaled 空間では収束したが
        // 元空間 user_eps に届かない精度床) は前進不能の一形態として Stalled。
        match outcome.termination {
            IpmTermination::IterationLimit => SolveStatus::MaxIterations,
            IpmTermination::Deadline => SolveStatus::Timeout,
            IpmTermination::Stalled | IpmTermination::Converged => SolveStatus::Stalled,
        }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    static GAP_ACCEPTANCE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static INFEAS_RETRY_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn runner_gap_fail_then_proven(
        _problem: &QpProblem,
        _presolve: &QpPresolveResult,
        _options: &SolverOptions,
        _user_eps: f64,
    ) -> IpmOutcome {
        let call = GAP_ACCEPTANCE_CALLS.fetch_add(1, Ordering::SeqCst);
        IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: if call == 0 { 1e-3 } else { 0.0 },
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        }
    }

    #[test]
    #[should_panic(expected = "Q diagonal unscale requires one bound dual per finite bound")]
    fn unscale_q_diagonal_rejects_short_bound_duals() {
        let qp = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, 1.0)],
            vec![],
        )
        .unwrap();
        let mut result = SolverResult {
            solution: vec![0.0],
            bound_duals: vec![0.0], // box variable requires lb and ub slots
            ..Default::default()
        };
        unscale_q_diagonal(&mut result, &[2.0], &qp);
    }

    fn runner_infeasible_then_fallback_suboptimal(
        _problem: &QpProblem,
        presolve: &QpPresolveResult,
        _options: &SolverOptions,
        _user_eps: f64,
    ) -> IpmOutcome {
        if presolve.ruiz_scaler.is_some() {
            return IpmOutcome::infeasibility(crate::problem::SolveStatus::Infeasible);
        }
        IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 1e-3,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        }
    }

    fn runner_infeasible_then_finite_retry(
        _problem: &QpProblem,
        _presolve: &QpPresolveResult,
        _options: &SolverOptions,
        _user_eps: f64,
    ) -> IpmOutcome {
        let call = INFEAS_RETRY_CALLS.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            return IpmOutcome::infeasibility(crate::problem::SolveStatus::Infeasible);
        }
        IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 1e-3,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        }
    }

    fn runner_incumbent_then_worse_fallback(
        _problem: &QpProblem,
        presolve: &QpPresolveResult,
        _options: &SolverOptions,
        _user_eps: f64,
    ) -> IpmOutcome {
        let is_fallback = presolve.ruiz_scaler.is_none();
        IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0],
            objective: if is_fallback { 1.0e9 } else { 0.0 },
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: if is_fallback { 1e-3 } else { 1e-2 },
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        }
    }

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

    /// Case D fixture: 1 strictly-convex var + 1 LP-style isolated
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
            termination: IpmTermination::Converged,
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

    #[test]
    fn attempt_acceptance_requires_prove_optimal_gap() {
        let prob = make_convex_plus_empty_col_qp();
        let mask = vec![false, true];
        let view = ProblemView {
            q: &prob.q,
            a: &prob.a,
            c: &prob.c,
            b: &prob.b,
            bounds: &prob.bounds,
            constraint_types: &prob.constraint_types,
            eliminated_cols: &mask,
        };
        let outcome = IpmOutcome {
            solution: vec![0.0, 0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 1.0, 0.0, 0.0],
            objective: 0.0,
            iterations: 5,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 1e-3,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        };
        assert!(
            outcome.satisfies_eps(1e-6),
            "loose satisfies_eps still accepts gap below promotion tolerance"
        );
        assert!(
            !outcome_proves_optimal(&outcome, &view, 1e-6),
            "attempt acceptance must match prove_optimal gap<=user_eps"
        );
        assert!(
            outcome_certificate_score(&outcome, &view, 1e-6) >= 1e-3,
            "certificate score must include user-eps gap failure"
        );
    }

    #[test]
    fn attempt_acceptance_requires_prove_optimal_dual_sign() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let view = ProblemView::from_problem(&prob);
        let outcome = IpmOutcome {
            solution: vec![1.0],
            dual_solution: vec![-1.0],
            bound_duals: vec![],
            objective: 0.5,
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
            termination: IpmTermination::Converged,
        };
        assert!(
            outcome.satisfies_eps(1e-6),
            "satisfies_eps intentionally has no dual-sign check"
        );
        assert!(
            !outcome_proves_optimal(&outcome, &view, 1e-6),
            "attempt acceptance must not stop on a dual-sign-invalid point"
        );
        assert!(
            outcome_certificate_score(&outcome, &view, 1e-6) > 0.0,
            "certificate score must include dual-sign failure"
        );
    }

    fn runner_extended_ir_dual_sign_invalid_but_metric_clean(
        _problem: &QpProblem,
        _presolve: &QpPresolveResult,
        _options: &SolverOptions,
        _user_eps: f64,
    ) -> IpmOutcome {
        IpmOutcome {
            solution: vec![1.0],
            dual_solution: vec![-1e-4],
            bound_duals: vec![0.0, 1.0 + 1e-4],
            objective: -1.5,
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
            termination: IpmTermination::Converged,
        }
    }

    #[test]
    fn solve_runner_extended_ir_still_demotes_dual_sign_invalid_postsolve_outcome() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![-2.0_f64],
            a,
            vec![1.0_f64],
            vec![(0.0_f64, 1.0_f64)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;
        opts.ipm.max_iter = 1;
        opts.ipm.extended_ir = true;

        let (result, _mask) = solve_ipm_with_runner(
            &prob,
            &opts,
            runner_extended_ir_dual_sign_invalid_but_metric_clean,
        );

        assert_eq!(
            result.status,
            crate::problem::SolveStatus::SuboptimalSolution,
            "extended_ir=true solve path must let prove_optimal demote a metric-clean but dual-sign-invalid postsolve outcome"
        );
    }

    #[test]
    fn candidate_order_prefers_certificate_then_objective() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let view = ProblemView::from_problem(&prob);
        let mut incumbent = IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 1e-2,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        };
        let mut better_cert = incumbent.clone();
        better_cert.objective = 1.0e9;
        better_cert.duality_gap_rel = 1e-3;
        assert!(
            outcome_is_better_candidate(&better_cert, &incumbent, &view, 1e-6),
            "certificate residual improvement is the primary ordering key"
        );

        let mut better_obj = incumbent.clone();
        better_obj.objective = -1.0;
        incumbent.objective = 1.0;
        assert!(
            outcome_is_better_candidate(&better_obj, &incumbent, &view, 1e-6),
            "objective is the tie-breaker when certificate residuals are equal"
        );

        let mut worse_obj_better_cert = incumbent.clone();
        worse_obj_better_cert.objective = 1.0e9;
        worse_obj_better_cert.duality_gap_rel = 1e-3;
        assert!(
            !fallback_can_replace_unproven(&worse_obj_better_cert, &incumbent, &view, 1e-6),
            "unproven fallback must not replace an incumbent when objective gets worse"
        );
    }

    #[test]
    fn candidate_order_preserves_structural_status_over_unproven_iterate() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let view = ProblemView::from_problem(&prob);
        let structural = IpmOutcome::infeasibility(crate::problem::SolveStatus::Infeasible);
        let finite_unproven = IpmOutcome {
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0],
            objective: 0.0,
            iterations: 1,
            kkt_residual_rel: 0.0,
            primal_residual_rel: 0.0,
            bound_violation: 0.0,
            complementarity_residual_rel: 0.0,
            duality_gap_rel: 1e-3,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Converged,
        };

        assert!(
            outcome_is_better_candidate(&finite_unproven, &structural, &view, 1e-6),
            "a finite retry candidate must displace a retry-local infeasibility status"
        );
        assert!(
            !outcome_is_better_candidate(&structural, &finite_unproven, &view, 1e-6),
            "a retry-local infeasibility status must not displace a finite incumbent"
        );
        assert!(
            !fallback_can_replace_unproven(&finite_unproven, &structural, &view, 1e-6),
            "an unproven no-presolve fallback must not displace an incumbent infeasibility status"
        );
    }

    #[test]
    fn attempt_loop_does_not_stop_on_satisfies_only_gap_failure() {
        GAP_ACCEPTANCE_CALLS.store(0, Ordering::SeqCst);
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let mut options = SolverOptions {
            presolve: false,
            use_ruiz_scaling: false,
            ..SolverOptions::default()
        };
        options.ipm.eps = 1e-6;
        options.ipm.max_iter = MAX_ITER_PER_ATTEMPT;

        let (result, _) = solve_ipm_with_runner(&prob, &options, runner_gap_fail_then_proven);

        assert_eq!(
            GAP_ACCEPTANCE_CALLS.load(Ordering::SeqCst),
            2,
            "first satisfies_eps-only outcome must not terminate the attempt loop"
        );
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::Optimal,
            "second proven outcome must be the accepted result"
        );
    }

    #[test]
    fn no_presolve_fallback_does_not_replace_structural_infeasible() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let mut options = SolverOptions {
            presolve: true,
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        options.ipm.eps = 1e-6;
        options.ipm.max_iter = MAX_ITER_PER_ATTEMPT * 4 + 2;

        let (result, _) =
            solve_ipm_with_runner(&prob, &options, runner_infeasible_then_fallback_suboptimal);

        assert_eq!(
            result.status,
            crate::problem::SolveStatus::Infeasible,
            "no-presolve fallback must not replace a structural infeasibility certificate with an unproven finite iterate"
        );
    }

    #[test]
    fn attempt_loop_keeps_finite_retry_over_infeasibility_status() {
        INFEAS_RETRY_CALLS.store(0, Ordering::SeqCst);
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let mut options = SolverOptions {
            presolve: false,
            use_ruiz_scaling: false,
            ..SolverOptions::default()
        };
        options.ipm.eps = 1e-6;
        options.ipm.max_iter = MAX_ITER_PER_ATTEMPT * 2;

        let (result, _) =
            solve_ipm_with_runner(&prob, &options, runner_infeasible_then_finite_retry);

        assert!(
            INFEAS_RETRY_CALLS.load(Ordering::SeqCst) >= 2,
            "retry-local infeasibility must not terminate before a finite retry is considered"
        );
        assert_eq!(
            result.status,
            crate::problem::SolveStatus::SuboptimalSolution,
            "finite unproven retry must be returned instead of a retry-local infeasibility status"
        );
        assert_eq!(result.objective, 0.0);
    }

    #[test]
    fn no_presolve_fallback_does_not_replace_better_objective_incumbent() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let mut options = SolverOptions {
            presolve: true,
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        options.ipm.eps = 1e-6;
        options.ipm.max_iter = MAX_ITER_PER_ATTEMPT * 4 + 2;

        let (result, _) =
            solve_ipm_with_runner(&prob, &options, runner_incumbent_then_worse_fallback);

        assert_eq!(
            result.status,
            crate::problem::SolveStatus::SuboptimalSolution
        );
        assert_eq!(
            result.objective, 0.0,
            "unproven fallback with a worse objective must not displace the incumbent"
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
            termination: IpmTermination::Converged,
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
    /// `min(|Q_ii|, |Q_jj|)` for each off-diagonal entry, not the unrelated
    /// diagonal-scaling floor.
    ///
    /// Fixture: Q = diag([1e9, 1e3]) with off-diagonal 5e-10.
    /// - local_scale = min(1e9, 1e3) = 1e3
    /// - offdiag_eps = Q_OFFDIAG_REL × 1e3 = 1e-9
    /// - 5e-10 < 1e-9 → Gate 1 passes; range = 1e6 → Gate 3 passes → Some
    ///
    /// **Sentinel**: replacing Gate 1 with `Q_DIAG_SCALING_MIN = 1e-10`
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
            termination: IpmTermination::Converged,
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
            termination: IpmTermination::Converged,
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
            termination: IpmTermination::Converged,
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

        fn mock_runner(
            _: &QpProblem,
            _: &QpPresolveResult,
            _: &SolverOptions,
            _: f64,
        ) -> IpmOutcome {
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

    /// The attempt loop tightens `opts.ipm.eps` for the inner IPM solve, but
    /// postsolve/original-space gates must still evaluate against the user eps.
    #[test]
    fn runner_receives_user_eps_separate_from_attempt_eps() {
        use std::cell::Cell;
        thread_local! {
            static SEEN_ATTEMPT_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
            static SEEN_USER_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
        }

        fn mock_runner(
            _: &QpProblem,
            _: &QpPresolveResult,
            options: &SolverOptions,
            user_eps: f64,
        ) -> IpmOutcome {
            SEEN_ATTEMPT_EPS.with(|c| c.set(options.ipm.eps));
            SEEN_USER_EPS.with(|c| c.set(user_eps));
            IpmOutcome::empty()
        }

        let prob = make_simple_eq_qp();
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;
        opts.ipm.max_iter = 1;
        opts.presolve = false;
        SEEN_ATTEMPT_EPS.with(|c| c.set(f64::NAN));
        SEEN_USER_EPS.with(|c| c.set(f64::NAN));

        let _ = solve_ipm_with_runner(&prob, &opts, mock_runner);

        let attempt_eps = SEEN_ATTEMPT_EPS.with(|c| c.get());
        let user_eps = SEEN_USER_EPS.with(|c| c.get());
        assert!(
            attempt_eps < user_eps,
            "attempt eps must be tightened below user eps; attempt={attempt_eps:e} user={user_eps:e}"
        );
        assert_eq!(
            user_eps, 1e-6,
            "runner must receive the external user eps, not the tightened attempt eps"
        );
    }

    /// A caller-set `Tolerance::Custom` must not leak into the inner attempt
    /// options: the inner IPPMM convergence check reads `options.ipm_eps()`
    /// (ippmm/iter.rs), which returns the Custom value when `tolerance` is set
    /// and silently bypasses the attempt loop's `opts.ipm.eps = user_eps /
    /// tighten` inner-target tightening. The IPM then stops at user_eps
    /// aggregate accuracy and postsolve refine burns the remaining budget
    /// (POWELL20 0.5s → 1000s post_refine regression, daf7ab54).
    ///
    /// Sentinel: reverting the `opts.tolerance = None` clear in
    /// `solve_ipm_with_runner` makes the runner-observed `ipm_eps()` equal the
    /// user eps (untightened) and `tolerance` non-None → both asserts fail.
    #[test]
    fn attempt_loop_clears_tolerance_so_inner_eps_is_tightened() {
        use std::cell::Cell;
        thread_local! {
            static SEEN_TOLERANCE_SET: Cell<bool> = const { Cell::new(true) };
            static SEEN_EFFECTIVE_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
        }

        fn mock_runner(
            _: &QpProblem,
            _: &QpPresolveResult,
            options: &SolverOptions,
            _: f64,
        ) -> IpmOutcome {
            SEEN_TOLERANCE_SET.with(|c| c.set(options.tolerance.is_some()));
            SEEN_EFFECTIVE_EPS.with(|c| c.set(options.ipm_eps()));
            IpmOutcome::empty()
        }

        let prob = make_simple_eq_qp();
        let mut opts = SolverOptions {
            tolerance: Some(crate::options::Tolerance::Custom(1e-6)),
            presolve: false,
            ..SolverOptions::default()
        };
        opts.ipm.eps = 1e-6;
        opts.ipm.max_iter = 1;
        SEEN_TOLERANCE_SET.with(|c| c.set(true));
        SEEN_EFFECTIVE_EPS.with(|c| c.set(f64::NAN));

        let _ = solve_ipm_with_runner(&prob, &opts, mock_runner);

        assert!(
            !SEEN_TOLERANCE_SET.with(|c| c.get()),
            "inner attempt options must have tolerance cleared"
        );
        let effective = SEEN_EFFECTIVE_EPS.with(|c| c.get());
        assert!(
            effective < 1e-6,
            "runner-observed ipm_eps() must be the tightened inner target, \
             not the Custom user eps; got {effective:e}"
        );
    }

    /// The no-presolve fallback clears `opts.tolerance` while keeping the inner
    /// IPM target tightened; the runner must still receive the external user eps.
    #[test]
    fn fallback_runner_receives_user_eps_after_tolerance_clear() {
        use std::cell::Cell;
        thread_local! {
            static FALLBACK_ATTEMPT_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
            static FALLBACK_USER_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
        }

        fn mock_runner(
            _: &QpProblem,
            presolve: &QpPresolveResult,
            options: &SolverOptions,
            user_eps: f64,
        ) -> IpmOutcome {
            if presolve.ruiz_scaler.is_none() {
                FALLBACK_ATTEMPT_EPS.with(|c| c.set(options.ipm.eps));
                FALLBACK_USER_EPS.with(|c| c.set(user_eps));
            }
            IpmOutcome {
                solution: vec![0.0],
                dual_solution: vec![],
                bound_duals: vec![0.0],
                objective: 0.0,
                iterations: 1,
                kkt_residual_rel: 0.0,
                primal_residual_rel: 0.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 0.0,
                duality_gap_rel: 1e-3,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Converged,
            }
        }

        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let mut opts = SolverOptions {
            presolve: true,
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        opts.ipm.eps = 1e-6;
        opts.ipm.max_iter = MAX_ITER_PER_ATTEMPT * 4 + 2;
        FALLBACK_ATTEMPT_EPS.with(|c| c.set(f64::NAN));
        FALLBACK_USER_EPS.with(|c| c.set(f64::NAN));

        let _ = solve_ipm_with_runner(&prob, &opts, mock_runner);

        let attempt_eps = FALLBACK_ATTEMPT_EPS.with(|c| c.get());
        let user_eps = FALLBACK_USER_EPS.with(|c| c.get());
        assert!(
            attempt_eps.is_finite() && attempt_eps < user_eps,
            "fallback attempt eps must be tightened below user eps; attempt={attempt_eps:e} user={user_eps:e}"
        );
        assert_eq!(
            user_eps, 1e-6,
            "fallback runner must receive the external user eps after tolerance is cleared"
        );
    }

    /// Q-diagonal scaling calls the same attempt runner on the scaled problem;
    /// the scaled path must keep user eps separate from the tightened attempt eps.
    #[test]
    fn q_diagonal_scaled_runner_receives_user_eps() {
        use std::cell::Cell;
        thread_local! {
            static SEEN_ATTEMPT_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
            static SEEN_USER_EPS: Cell<f64> = const { Cell::new(f64::NAN) };
        }

        fn mock_runner(
            _: &QpProblem,
            _: &QpPresolveResult,
            options: &SolverOptions,
            user_eps: f64,
        ) -> IpmOutcome {
            SEEN_ATTEMPT_EPS.with(|c| c.set(options.ipm.eps));
            SEEN_USER_EPS.with(|c| c.set(user_eps));
            IpmOutcome::empty()
        }

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e-7_f64, 2.0], 2, 2).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(0.0, 1.0), (0.0, 1.0)],
        )
        .unwrap();
        let (scaled_prob, col_scales) =
            try_q_diagonal_scaling(&prob).expect("ill-conditioned diagonal Q must scale");
        let mut scaled_opts = scale_warm_start_for_q_diag(&SolverOptions::default(), &col_scales);
        scaled_opts.ipm.eps = 1e-6;
        scaled_opts.ipm.max_iter = 1;
        scaled_opts.presolve = false;
        SEEN_ATTEMPT_EPS.with(|c| c.set(f64::NAN));
        SEEN_USER_EPS.with(|c| c.set(f64::NAN));

        let _ = solve_ipm_with_runner(&scaled_prob, &scaled_opts, mock_runner);

        let attempt_eps = SEEN_ATTEMPT_EPS.with(|c| c.get());
        let user_eps = SEEN_USER_EPS.with(|c| c.get());
        assert!(
            attempt_eps < user_eps,
            "scaled attempt eps must be tightened below user eps; attempt={attempt_eps:e} user={user_eps:e}"
        );
        assert_eq!(
            user_eps, 1e-6,
            "Q-diagonal scaled runner must receive the external user eps"
        );
    }

    /// Sentinel: `opts.schur_hint` is set to `Some(...)` before the main attempt loop,
    /// then reset to `None` at the start of the no-presolve fallback.
    ///
    /// A mock runner records the `schur_hint` seen during the fallback invocation
    /// (identified by `presolve.ruiz_scaler.is_none()`). The reset ensures the
    /// fallback IPM re-probes Schur suitability on the original (non-reduced) problem
    /// rather than reusing the decision from the presolve-reduced problem.
    ///
    /// **Sentinel**: removing `opts.schur_hint = None` at line ~596 of
    /// `solve_ipm_with_runner` causes the fallback runner to see `Some(_)`, making
    /// `fallback_schur_hint.get()` non-None → this test FAILS.
    #[test]
    fn schur_hint_is_reset_before_fallback() {
        use std::cell::Cell;
        thread_local! {
            static FALLBACK_SCHUR_HINT_IS_NONE: Cell<bool> = const { Cell::new(false) };
            static FALLBACK_REACHED: Cell<bool> = const { Cell::new(false) };
        }

        fn mock_runner(
            _: &QpProblem,
            presolve: &QpPresolveResult,
            options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            if presolve.ruiz_scaler.is_none() {
                FALLBACK_REACHED.with(|c| c.set(true));
                FALLBACK_SCHUR_HINT_IS_NONE.with(|c| c.set(options.schur_hint.is_none()));
            }
            // Always return unproven to force the fallback path to run.
            IpmOutcome {
                solution: vec![0.0],
                dual_solution: vec![],
                bound_duals: vec![0.0],
                objective: 0.0,
                iterations: 1,
                kkt_residual_rel: 0.0,
                primal_residual_rel: 0.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 0.0,
                duality_gap_rel: 1e-3,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Converged,
            }
        }

        // Use a small convex QP that triggers the no-presolve fallback path:
        // presolve=true + use_ruiz_scaling=true causes Ruiz scaling, and when the
        // main attempts do not prove optimal (duality_gap_rel=1e-3 > user_eps), the
        // fallback is invoked on the original problem (ruiz_scaler=None).
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
        )
        .unwrap();
        let mut opts = SolverOptions {
            presolve: true,
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        opts.ipm.eps = 1e-6;
        opts.ipm.max_iter = MAX_ITER_PER_ATTEMPT * 4 + 2;
        FALLBACK_REACHED.with(|c| c.set(false));
        FALLBACK_SCHUR_HINT_IS_NONE.with(|c| c.set(false));

        let _ = solve_ipm_with_runner(&prob, &opts, mock_runner);

        assert!(
            FALLBACK_REACHED.with(|c| c.get()),
            "no-presolve fallback must be reached for this test to be meaningful"
        );
        assert!(
            FALLBACK_SCHUR_HINT_IS_NONE.with(|c| c.get()),
            "opts.schur_hint must be None when the no-presolve fallback runner is called"
        );
    }
    /// 非収束 iterate をどう作っても SuboptimalSolution を名乗れないことの sentinel 群。
    ///
    /// 旧 finalize (`!satisfies_eps && !timed_out → SuboptimalSolution`) を revert
    /// すると 3 テストとも SuboptimalSolution が返り FAIL する。
    mod nonconverged_finalize_sentinels {
        use super::*;
        use crate::problem::SolveStatus;
        use crate::qp::ipm_solver::outcome::IpmTermination;

        /// kkt_max = 1.0 の完全非収束 iterate (QPLIB stall の縮小再現)。
        fn nonconverged_outcome(termination: IpmTermination) -> IpmOutcome {
            IpmOutcome {
                solution: vec![0.0],
                dual_solution: vec![0.0],
                bound_duals: vec![0.0],
                objective: 0.0,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination,
            }
        }

        #[test]
        fn stall_termination_reports_stalled_not_suboptimal() {
            let prob = make_simple_eq_qp();
            let view = ProblemView::from_problem(&prob);
            let r = finalize_outcome(
                nonconverged_outcome(IpmTermination::Stalled),
                1e-6,
                1,
                None,
                false,
                &view,
            );
            assert_eq!(r.status, SolveStatus::Stalled, "got {:?}", r.status);
            assert!(!r.solution.is_empty(), "診断 iterate は保持される");
        }

        #[test]
        fn iteration_budget_termination_reports_max_iterations() {
            let prob = make_simple_eq_qp();
            let view = ProblemView::from_problem(&prob);
            let r = finalize_outcome(
                nonconverged_outcome(IpmTermination::IterationLimit),
                1e-6,
                1,
                None,
                false,
                &view,
            );
            assert_eq!(r.status, SolveStatus::MaxIterations, "got {:?}", r.status);
        }

        /// scaled 空間で収束したが元空間 user_eps に届かない精度床 → Stalled。
        #[test]
        fn accuracy_floor_termination_reports_stalled() {
            let prob = make_simple_eq_qp();
            let view = ProblemView::from_problem(&prob);
            let r = finalize_outcome(
                nonconverged_outcome(IpmTermination::Converged),
                1e-6,
                1,
                None,
                false,
                &view,
            );
            assert_eq!(r.status, SolveStatus::Stalled, "got {:?}", r.status);
        }
    }

    /// 公開 API 経由の sentinel: 到達不能 eps では IPM は有限残差まで収束するが
    /// satisfies_eps は常に false → 解品質を主張する status を返してはならない。
    /// 旧 finalize (:778) を revert すると SuboptimalSolution が返り FAIL する。
    #[test]
    fn unreachable_eps_reports_stalled_via_public_api() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            a,
            vec![-1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-200;
        let r = solve_ipm(&prob, &opts);
        assert_eq!(
            r.status,
            crate::problem::SolveStatus::Stalled,
            "unreachable eps must not mint a solution-claiming status, got {:?}",
            r.status
        );
        assert!(!r.solution.is_empty(), "診断 iterate は保持される");
    }

    /// attempt loop の同一 outcome 早期打ち切り sentinel。
    ///
    /// runner が毎回 bit 同一の非収束 outcome を返す構成 (真の決定的 stall) では、
    /// attempt list (この構成で 4 attempt) を CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK
    /// (=3) attempt で打ち切る。dfl001 退化 (2 連続で打ち切ると 3 投目で脱出できる
    /// ケースを取り逃す) を受けて 2→3 に変更した。早期打ち切りを完全に revert すると
    /// 4 回呼ばれてこのテストが FAIL する。
    mod identical_attempt_early_break {
        use super::*;
        use crate::problem::SolveStatus;
        use crate::qp::ipm_solver::outcome::IpmTermination;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static IDENTICAL_CALLS: AtomicUsize = AtomicUsize::new(0);

        fn runner_identical_stall(
            _problem: &QpProblem,
            _presolve: &QpPresolveResult,
            _options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            IDENTICAL_CALLS.fetch_add(1, Ordering::SeqCst);
            IpmOutcome {
                solution: vec![0.25],
                dual_solution: vec![0.5],
                bound_duals: vec![0.0],
                objective: 0.25,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Stalled,
            }
        }

        #[test]
        fn identical_consecutive_attempts_break_early() {
            let prob = make_simple_eq_qp();
            // presolve/Ruiz 無効 → attempts = [(false, 100), (false, 1000),
            // (false, 10), (false, 1)] の 4 attempt 構成、fallback loop なし。
            let mut opts = SolverOptions {
                presolve: false,
                use_ruiz_scaling: false,
                ..Default::default()
            };
            opts.ipm.eps = 1e-6;

            IDENTICAL_CALLS.store(0, Ordering::SeqCst);
            let (result, _) = solve_ipm_with_runner(&prob, &opts, runner_identical_stall);
            let calls = IDENTICAL_CALLS.load(Ordering::SeqCst);
            assert_eq!(
                calls, 3,
                "bit 同一 stall は 3 連続 (CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK) で \
                 打ち切るべき (2 連続打ち切りに戻すと 2、早期打ち切り自体を revert すると 4)"
            );
            assert_eq!(
                result.status,
                SolveStatus::Stalled,
                "got {:?}",
                result.status
            );
        }
    }

    /// P3-2 sentinel (dual-lane 正方向): Ruiz 有効 (2 lane 交互) の attempts 列でも
    /// 「全 attempt が真に bit 同一の決定論的 stall」なら early-break が実際に
    /// 発火することを検証する。`identical_attempt_early_break` は no-Ruiz
    /// (単一 lane) 構成のみをカバーしており、dual-lane 構成での正方向 (打ち切る
    /// べきときに打ち切る) を検証する sentinel が欠けていた。
    ///
    /// 期待値の導出 (attempts 列構造 + CONSECUTIVE_IDENTICAL_ATTEMPTS_TO_BREAK=3
    /// から独立に計算): presolve 無効 + use_ruiz_scaling=true, eps=1e-6 →
    /// base_tighten=100 → attempts = [(true,100),(false,100),(true,1000),
    /// (false,1000),(true,10000),(false,10000),(true,10),(false,10),(true,1),
    /// (false,1)] (10 要素、lane が true/false と交互)。runner が入力に関わらず
    /// 常に同一 bit を返す (真の決定論的 stall) とき、lane=true の出現順は
    /// 全体 call 番号 1,3,5,... (1-indexed) = 1st,2nd,3rd,... occurrence。
    /// run_len[true] は 1st occurrence で 1、2nd occurrence (call#3) で 2、
    /// 3rd occurrence (call#5) で 3 となり閾値に到達 → call#5 (0-indexed idx4,
    /// (true,10000)) で break。break までに実行された call 数は 1,2,3,4,5 の
    /// 5 回 (idx0..idx4)。revert (early-break 完全削除) すると 10 回全走し FAIL。
    mod dual_lane_true_stall_breaks_at_third_occurrence {
        use super::*;
        use crate::qp::ipm_solver::outcome::IpmTermination;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn runner_always_identical(
            _problem: &QpProblem,
            _presolve: &QpPresolveResult,
            _options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            CALLS.fetch_add(1, Ordering::SeqCst);
            IpmOutcome {
                solution: vec![0.42],
                dual_solution: vec![0.7],
                bound_duals: vec![0.0],
                objective: 0.42,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Stalled,
            }
        }

        #[test]
        fn dual_lane_true_stall_breaks_at_expected_call_count() {
            let prob = make_simple_eq_qp();
            let mut opts = SolverOptions {
                presolve: false,
                use_ruiz_scaling: true,
                ..Default::default()
            };
            opts.ipm.eps = 1e-6;

            CALLS.store(0, Ordering::SeqCst);
            let (result, _) = solve_ipm_with_runner(&prob, &opts, runner_always_identical);
            let calls = CALLS.load(Ordering::SeqCst);
            assert_eq!(
                calls, 5,
                "真の決定論的 stall (dual-lane) は lane=true の 3rd occurrence \
                 (全体 5 call 目) で打ち切るべき, got {calls}"
            );
            assert_eq!(
                result.status,
                SolveStatus::Stalled,
                "got {:?}",
                result.status
            );
        }
    }

    /// P1-2 sentinel: Ruiz on/off だけが違う「tighten 同一」ペア (attempts 列先頭
    /// [(true,t),(false,t)]) が bit 同一でも、それは「Ruiz が no-op」なだけで
    /// tighten を上げても改善しないという証拠ではない。tighten ladder を丸ごと
    /// skip してはいけない。
    ///
    /// runner は options.ipm.eps (= tighten の写像) だけに依存する値を返す (Ruiz
    /// on/off を無視) — Ruiz が完全に no-op な問題を模擬する。lane 分離を revert
    /// すると (true,100) vs (false,100) が誤って「同一」判定され 2 attempt で
    /// 打ち切られる。
    mod ruiz_flip_pair_does_not_false_trigger_stall {
        use super::*;
        use crate::qp::ipm_solver::outcome::IpmTermination;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn runner_tighten_keyed(
            _problem: &QpProblem,
            _presolve: &QpPresolveResult,
            options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            CALLS.fetch_add(1, Ordering::SeqCst);
            IpmOutcome {
                solution: vec![options.ipm.eps],
                dual_solution: vec![0.5],
                bound_duals: vec![0.0],
                objective: options.ipm.eps,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Stalled,
            }
        }

        #[test]
        fn ruiz_flip_pair_does_not_false_trigger_stall_break() {
            let prob = make_simple_eq_qp();
            // presolve 無効 + use_ruiz_scaling=true → attempts = [(true,100),
            // (false,100),(true,1000),(false,1000),(true,10000),(false,10000),
            // (true,10),(false,10),(true,1),(false,1)] の 10 attempt 構成
            // (eps=1e-6 → base_tighten=100)。
            let mut opts = SolverOptions {
                presolve: false,
                use_ruiz_scaling: true,
                ..Default::default()
            };
            opts.ipm.eps = 1e-6;

            CALLS.store(0, Ordering::SeqCst);
            let _ = solve_ipm_with_runner(&prob, &opts, runner_tighten_keyed);
            let calls = CALLS.load(Ordering::SeqCst);
            assert_eq!(
                calls, 10,
                "Ruiz on/off だけが違う同一 tighten ペアで誤って打ち切ってはいけない \
                 (tighten ladder 全 10 attempt を尽くすべき), got {calls}"
            );
        }
    }

    /// P1-2 追加 sentinel (dfl001 退化の再現): 同一 lane 内で 2 連続 bit 一致した
    /// 後、3 投目で bit が変わる合成ケースでは打ち切ってはいけない (全 attempt を
    /// 尽くすべき)。
    ///
    /// 実測 trace (dfl001, netlib LP): lane=true は tighten=100→1000 で 2 連続
    /// bit 同一 (iter=115) だったが、tighten=10000 の 3 投目で iter=136 に変化し
    /// 脱出できた。2 連続で打ち切ると、この 3 投目や反対 lane の attempt
    /// (best 候補になり得た) を試す機会を失い、Optimal→Stalled に退化する。
    ///
    /// runner は tighten∈{100,1000} で同一 bit、tighten∈{10000,10,1} でそれぞれ
    /// 異なる bit を返す (Ruiz on/off に依存しない — 両 lane で同じパターン)。
    /// これにより両 lane とも run_len は最大 2 までしか伸びず、3 に到達しない。
    mod two_consecutive_match_then_diverge_does_not_break {
        use super::*;
        use crate::qp::ipm_solver::outcome::IpmTermination;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);
        const USER_EPS: f64 = 1e-6;
        // tighten=100,1000 それぞれの inner eps。両者は同一 bit を返す (2 連続一致)。
        const EPS_TIGHTEN_100: f64 = USER_EPS / 100.0;
        const EPS_TIGHTEN_1000: f64 = USER_EPS / 1000.0;

        fn runner_two_match_then_diverge(
            _problem: &QpProblem,
            _presolve: &QpPresolveResult,
            options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            CALLS.fetch_add(1, Ordering::SeqCst);
            // tighten=100 と tighten=1000 は同一の診断値 (iter=115 相当) を返す。
            // それ以外 (10000, 10, 1) は options.ipm.eps ごとに異なる値を返す
            // (dfl001 の 3 投目で iter が変化した挙動を模擬)。
            let probe = if options.ipm.eps == EPS_TIGHTEN_100 || options.ipm.eps == EPS_TIGHTEN_1000
            {
                115.0
            } else {
                options.ipm.eps
            };
            IpmOutcome {
                solution: vec![probe],
                dual_solution: vec![0.5],
                bound_duals: vec![0.0],
                objective: probe,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Stalled,
            }
        }

        #[test]
        fn two_consecutive_match_then_diverge_runs_full_ladder() {
            let prob = make_simple_eq_qp();
            // presolve 無効 + use_ruiz_scaling=true → attempts = [(true,100),
            // (false,100),(true,1000),(false,1000),(true,10000),(false,10000),
            // (true,10),(false,10),(true,1),(false,1)] の 10 attempt 構成。
            let mut opts = SolverOptions {
                presolve: false,
                use_ruiz_scaling: true,
                ..Default::default()
            };
            opts.ipm.eps = USER_EPS;

            CALLS.store(0, Ordering::SeqCst);
            let _ = solve_ipm_with_runner(&prob, &opts, runner_two_match_then_diverge);
            let calls = CALLS.load(Ordering::SeqCst);
            assert_eq!(
                calls, 10,
                "同一 lane 2 連続一致は決定的 stall の証拠にならない (3 投目で \
                 bit が変わりうる) — tighten ladder 全 10 attempt を尽くすべき, got {calls}"
            );
        }
    }

    /// P1-2 sentinel: termination==Converged (scaled 空間では収束したが元空間 eps
    /// に届かない精度床) は、tighten を変えても bit 同一の outcome を返しうるが、
    /// それは決定的 stall ではない (定義上 inner eps が変われば結果も変わるはずの
    /// ケース) ため打ち切ってはいけない。
    mod converged_termination_does_not_break_early {
        use super::*;
        use crate::qp::ipm_solver::outcome::IpmTermination;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn runner_static_converged(
            _problem: &QpProblem,
            _presolve: &QpPresolveResult,
            _options: &SolverOptions,
            _user_eps: f64,
        ) -> IpmOutcome {
            CALLS.fetch_add(1, Ordering::SeqCst);
            IpmOutcome {
                solution: vec![0.25],
                dual_solution: vec![0.5],
                bound_duals: vec![0.0],
                objective: 0.25,
                iterations: 61,
                kkt_residual_rel: 1.0,
                primal_residual_rel: 1.0,
                bound_violation: 0.0,
                complementarity_residual_rel: 1.0,
                duality_gap_rel: 1.0,
                numerical_failure: false,
                infeasibility_status: None,
                is_locally_optimal: false,
                postsolve_krylov_ir_skipped: false,
                timing: None,
                termination: IpmTermination::Converged,
            }
        }

        #[test]
        fn converged_termination_runs_full_ladder() {
            let prob = make_simple_eq_qp();
            let mut opts = SolverOptions {
                presolve: false,
                use_ruiz_scaling: true,
                ..Default::default()
            };
            opts.ipm.eps = 1e-6;

            CALLS.store(0, Ordering::SeqCst);
            let _ = solve_ipm_with_runner(&prob, &opts, runner_static_converged);
            let calls = CALLS.load(Ordering::SeqCst);
            assert_eq!(
                calls, 10,
                "termination=Converged (精度床) は bit 同一でも打ち切ってはいけない \
                 (tighten ladder 全 10 attempt を尽くすべき), got {calls}"
            );
        }
    }
}
