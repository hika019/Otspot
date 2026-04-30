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
    run_qp_presolve_phase1, run_qp_presolve_phase2,
    qp_transforms::QpPresolveStatus,
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use super::core::{run_ipm, run_ipm_mehrotra};
use crate::presolve::QpPresolveResult;
use super::outcome::IpmOutcome;
use std::time::Instant;

/// 統合 retry の attempt 配列。各 attempt で (use_ruiz, eps_tighten) を変える。
/// 旧 PV_RETRY × POST_VERIFY = 9 attempts を 6 attempts に直線化する。
/// presolve 済みの場合は (true, X) と (false, X) が等価なので 4 attempts に縮約する
/// (実装は `attempts_for` で動的選択)。
const ATTEMPTS_FULL: &[(bool, f64)] = &[
    (true,  1.0),    // Ruiz on,  eps × 1
    (true,  10.0),   // Ruiz on,  eps × 1/10
    (true,  100.0),  // Ruiz on,  eps × 1/100
    (false, 1.0),    // Ruiz off, eps × 1
    (false, 10.0),   // Ruiz off, eps × 1/10
    (false, 100.0),  // Ruiz off, eps × 1/100
];
/// presolve_did_ruiz=true 時の attempt 配列。Ruiz on/off は等価なので tighten のみ変える。
/// (false, 1000.0) を追加すると IPM が double 精度限界近くで full convergence できず
/// むしろ悪化リスクがあるため 3 attempts に留める。
const ATTEMPTS_PRESOLVE_RUIZ: &[(bool, f64)] = &[
    (false, 1.0),
    (false, 10.0),
    (false, 100.0),
];
/// eps 事前調整の下限 (double 精度限界近傍)
const EPS_FLOOR: f64 = 1e-15;
/// 1 attempt が消費してよい時間の最低割合 (deadline / 残 attempt 数 が これ以下なら break)
const MIN_TIME_PER_ATTEMPT: f64 = 0.5;

/// IpmOutcome を返す runner 関数の型 (run_ipm = IP-PMM、run_ipm_mehrotra = Mehrotra)
type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions) -> IpmOutcome;

/// QP を v2 設計で解く (IP-PMM 経路)。既存 `solve_qp_with` と同じ API シグネチャ。
///
/// retry 1 層・status 1 箇所変換・元空間 KKT 判定 の 3 原則で動く。
pub fn solve_qp_v2(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_qp_v2_with_runner(problem, options, run_ipm)
}

/// QP を v2 設計で解く (Mehrotra 経路)。
///
/// `solve_qp_v2` と同じ retry/status/KKT 判定の 3 原則を Mehrotra IPM に適用した wrapper。
/// 旧 `ipm::solve_qp_ipm` 直叩き経路では偽 Optimal が大量発生 (60s で OBJ_MISMATCH 24)
/// していたが、本 wrapper で v2 同等の元空間 KKT 再判定により抑え込む。
pub fn solve_qp_v1_wrapped(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_qp_v2_with_runner(problem, options, run_ipm_mehrotra)
}

/// 一般化 wrapper: `runner` (IP-PMM or Mehrotra) を選択可能。
fn solve_qp_v2_with_runner(
    problem: &QpProblem,
    options: &SolverOptions,
    runner: IpmRunner,
) -> SolverResult {
    // ユーザーが指定したアルゴリズム (IPM / IP-PMM) で必ず解く。
    // 旧実装は `is_zero_q()` で Q≈0 退化を検出して Simplex (LP) に dispatch していたが、
    // これはユーザー mandate 違反 (qp_solver: IpPmmNew を指定しても Simplex に振り替わる)。
    // IPM/IP-PMM は Q=0 を線形目的の特殊ケースとして扱える (barrier 経由の LP として解く)。
    // 性能差があれば呼び出し側が `qp_solver: Concurrent` で並行実行を選べる。

    // Q 不定値チェック (非凸 QP 検出): IPM は Q 半正定値前提。
    if !crate::qp::check_q_positive_semidefinite(&problem.q) {
        return SolverResult {
            status: SolveStatus::NonConvex(
                "Q matrix is indefinite (non-convex QP). IPM requires Q to be positive semidefinite.".to_string()
            ),
            ..Default::default()
        };
    }

    let start_time = Instant::now();
    let mut opts = options.clone();
    let n_orig = problem.num_vars;

    // ── presolve (1 回のみ) ─────────────────────────────
    let presolve_result = if opts.presolve {
        let phase1 = run_qp_presolve_phase1(problem, &opts);
        // DIAG: QP_PRESOLVE_PHASE2=0 で phase2 (equality_constraint_qr / Ruiz / 大係数 scale)
        // を無効化。dual postsolve が phase2 の row 削除に追従できない場合の回避用。
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

    // presolve が既に Ruiz scaling を適用した場合、IPM 側で重ね掛けすると
    // 二重スケールで解が誤った点に収束する (HS21 で観測: x=(4.31,0) vs 真値 (2,0))。
    let presolve_did_ruiz = presolve_result.ruiz_scaler.is_some();
    let mut best: Option<IpmOutcome> = None;

    // 注: LP-dominant 問題 (例: QSCRS8/QBORE3D) を Q=0 で LP として解く dispatch は試行したが、
    // (1) LP の x で QP の KKT 残差が 0.5〜1.0 で eps=1e-6 未達、PASS 増加なし
    // (2) LISWET 系で simplex が deadline を尊重せず長時間 stuck する副作用
    // により削除した。`run_lp_postprocess` 関数自体は残しており将来の warm-start IPM 用。

    // ── retry 1 層: 動的 attempt 配列を時間内で順に試行 ────────
    // presolve_did_ruiz=true: (true, X) と (false, X) が等価なので 4 attempts に縮約 + (false,1000.0)
    // presolve_did_ruiz=false: 6 attempts (Ruiz on/off × 3 tighten)
    let attempts: &[(bool, f64)] = if presolve_did_ruiz {
        ATTEMPTS_PRESOLVE_RUIZ
    } else {
        ATTEMPTS_FULL
    };

    for (idx, &(use_ruiz, tighten)) in attempts.iter().enumerate() {
        if let Some(d) = total_deadline {
            let now = Instant::now();
            if now >= d {
                break;
            }
            let remaining = d.saturating_duration_since(now);
            if remaining.as_secs_f64() < MIN_TIME_PER_ATTEMPT {
                break;
            }
            // attempt 0 は full deadline を使う (旧 v1 PV_RETRY pv_try=0 と同等の時間予算)。
            // attempt 1+ は残り時間 / 残 attempt 数で公平に分配。
            // attempt 0 を timeshare すると IPM が時間内に収束しきれず不完全解で best-so-far に
            // 入ってしまう (HS21 で観測: deadline=total/6 だと x=(4.31, 0) で停止 vs full deadline で x=(2, 0))。
            opts.deadline = if idx == 0 {
                total_deadline
            } else {
                let remaining_attempts = (attempts.len() - idx) as u32;
                Some(now + remaining / remaining_attempts.max(1))
            };
            opts.timeout_secs = None;
        }
        opts.ipm.eps = (user_eps / tighten).max(EPS_FLOOR);
        // presolve が Ruiz scaling 済みなら IPM での再スケールは抑止する。
        opts.use_ruiz_scaling = if presolve_did_ruiz { false } else { use_ruiz };

        let outcome = runner(problem, &presolve_result, &opts);

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

    // ── status 変換 (1 箇所のみ) ───────────────
    // outcome は既に元空間 (run_ipm 内で unscale + postsolve 済み)。
    let outcome = best.unwrap_or_else(IpmOutcome::empty);
    finalize_outcome(outcome, user_eps, n_orig)
}

/// `IpmOutcome` から `SolverResult` (外部 status) への変換 — **status mutation 1 箇所**。
/// outcome.solution は既に元空間で postsolve / unscale / clip 済み。
fn finalize_outcome(
    outcome: IpmOutcome,
    user_eps: f64,
    n_orig: usize,
) -> SolverResult {
    // 確定的 Infeasible / Unbounded / NonConvex は最優先で外部に伝える (status 隠蔽防止)。
    // objective は status に応じて意味のある値を設定:
    //   Infeasible → +∞ (実行可能解なし、最小化では到達不能)
    //   Unbounded  → -∞ (objective がいくらでも小さくなる方向あり)
    //   NonConvex  → NaN (大域最適保証なし、値に意味がない)
    // SolverResult::infeasible() の慣例 (objective: f64::INFINITY) と整合。
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

    if outcome.solution.is_empty() {
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

    let status = if outcome.satisfies_eps(user_eps) {
        SolveStatus::Optimal
    } else {
        // ユーザー精度未達 → Timeout (設計書: 「内部で解を捨てない」 = solution は保持)
        SolveStatus::Timeout
    };

    debug_assert_eq!(outcome.solution.len(), n_orig, "outcome solution dimension mismatch");

    SolverResult {
        status,
        objective: outcome.objective,
        solution: outcome.solution,
        dual_solution: outcome.dual_solution,
        bound_duals: outcome.bound_duals,
        iterations: outcome.iterations,
        ..Default::default()
    }
}
