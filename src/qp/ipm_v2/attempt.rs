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
use super::core::run_ipm;
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

/// IpmOutcome を返す runner 関数の型 (= run_ipm = IP-PMM のみ)
type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions) -> IpmOutcome;

/// QP を v2 設計で解く (IP-PMM 経路)。既存 `solve_qp_with` と同じ API シグネチャ。
///
/// retry 1 層・status 1 箇所変換・元空間 KKT 判定 の 3 原則で動く。
pub fn solve_qp_v2(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_qp_v2_with_runner(problem, options, run_ipm)
}

/// 一般化 wrapper: 旧実装で IPM/IPPMM を切り替えていた wrapper。
/// 現在 runner は IP-PMM のみ。
fn solve_qp_v2_with_runner(
    problem: &QpProblem,
    options: &SolverOptions,
    runner: IpmRunner,
) -> SolverResult {
    // Q 不定値チェック (非凸 QP 検出): IPPMM は Q 半正定値前提。
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

    // ── deadline 確定を presolve よりも先に行う ──────────
    // 動機: 100 万変数級 QPLIB (QPLIB_8547: n=1.0M, m=1.0M) では presolve 単体で
    // timeout を超過する。deadline が presolve 後に設定されると、presolve 中の
    // hot loop が timeout を尊重できず external gtimeout で force-kill される。
    // deadline を冒頭で固定し、presolve 後/IPM 前後に経過判定を入れる。
    if opts.deadline.is_none() {
        if let Some(secs) = opts.timeout_secs {
            opts.deadline = Some(start_time + std::time::Duration::from_secs_f64(secs));
            opts.timeout_secs = None;
        }
    }
    let total_deadline = opts.deadline;
    let user_eps = opts.ipm_eps();

    // ── presolve (1 回のみ) ─────────────────────────────
    // 100万変数級の QPLIB (8547/9008) では presolve 内ループが O(n*m) で deadline
    // 到達前に巨大時間を消費する。bench 計装と同じ 50k 閾値で skip し、IPM 内 deadline
    // チェックに任せる。閾値内では従来どおり presolve 適用 (Maros 138 PASS 数を維持)。
    const PRESOLVE_SIZE_LIMIT: usize = 50_000;
    let presolve_result = if opts.presolve
        && problem.num_vars <= PRESOLVE_SIZE_LIMIT
        && problem.num_constraints <= PRESOLVE_SIZE_LIMIT
    {
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

    // presolve 自体が timeout を超過した場合 (巨大 QP)、IPM を走らせず即 Timeout。
    // 解なしで返すため finalize_outcome 経由で `Timeout` (best-so-far なし) になる。
    if total_deadline.is_some_and(|d| Instant::now() >= d) {
        return finalize_outcome(IpmOutcome::empty(), user_eps, n_orig, total_deadline, false);
    }

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
            // MIN_TIME_PER_ATTEMPT は「best-so-far がある時に新規 attempt を始めるか」のガード。
            // idx=0 では best-so-far が無いため、たとえ残り時間が短くても 1 回は IPPMM を
            // 呼ぶ必要がある (呼ばなければ outcome=empty + timed_out=false で NumericalError
            // 誤判定になる)。短時間 deadline は IPPMM 内部の should_stop が尊重するので、
            // ここで早期 break する必要はない。
            if idx > 0 && remaining.as_secs_f64() < MIN_TIME_PER_ATTEMPT {
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

    // ── presolve fall-back: best が eps を満たさず時間が残っているなら presolve=false で再試行 ────
    //
    // 動機: presolve の特定変換組み合わせ (例: #2 + #5 + #12) が稀に y_orig を破壊し、
    // IPM 自体は解けるはずの問題 (no-presolve では PASS) でも DFEAS_FAIL になる病理がある
    // (QBORE3D / QCAPRI 等で実証)。汎用 solver として「IPM で解ける問題を解けない」
    // 状態を放置しないため、presolve 経路が失敗したら presolve なしで再試行する自己修復。
    let need_retry_no_presolve = match &best {
        Some(o) => !o.satisfies_eps(user_eps),
        None => true,
    } && options.presolve; // 元 options で presolve=true だった場合のみ
    if need_retry_no_presolve {
        let remaining = match total_deadline {
            Some(d) => d.saturating_duration_since(Instant::now()).as_secs_f64(),
            None => f64::INFINITY,
        };
        if remaining >= MIN_TIME_PER_ATTEMPT * 2.0 {
            let presolve_result_np = crate::presolve::QpPresolveResult::no_reduction(problem);
            let mut opts_np = options.clone();
            opts_np.presolve = false;
            opts_np.deadline = total_deadline;
            opts_np.timeout_secs = None;
            opts_np.ipm.eps = user_eps;
            let outcome_np = runner(problem, &presolve_result_np, &opts_np);
            let prefer_np = match &best {
                None => true,
                Some(prev) => outcome_np.quality_score() < prev.quality_score(),
            };
            if prefer_np {
                best = Some(outcome_np);
            }
        }
    }

    // ── status 変換 (1 箇所のみ) ───────────────
    // outcome は既に元空間 (run_ipm 内で unscale + postsolve 済み)。
    let outcome = best.unwrap_or_else(IpmOutcome::empty);
    // cancel_flag が事前に立っていた / sibling thread が立てた場合も「外部から停止」
    // として deadline 経過と同様に扱う。これがないと「cancel_flag 即停止」で IPM が
    // 一切 iterate しないケースが NumericalError 扱いになり、`A3-C02` 系の cancel 契約
    // (preset cancel → Timeout) を破る。
    let cancelled = options
        .cancel_flag
        .as_ref()
        .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed));
    finalize_outcome(outcome, user_eps, n_orig, total_deadline, cancelled)
}

/// `IpmOutcome` から `SolverResult` (外部 status) への変換 — **status mutation 1 箇所**。
/// outcome.solution は既に元空間で postsolve / unscale / clip 済み。
///
/// status 分類 (status 隠蔽防止):
///   - Optimal             : ユーザー精度 eps 達成
///   - Timeout             : 外部 deadline 経過 (時間切れ、もっと時間あれば改善余地)
///   - SuboptimalSolution  : deadline 内で IPM が内部終了 (alpha_stall/mu_floor/NaN_guard)、
///                            best-so-far 解あり (eps 未達、IPM の数値的限界)
///   - NumericalError      : best-so-far も無し (factorize 失敗 / 即時破綻)
///
/// 旧実装は「eps 未達は全て Timeout」で IPM 内部諦めと真の時間切れを混同していた。
/// QPLIB_9002 で「2s iters=49 TIMEOUT」のような誤分類 (実際は IPM 早期諦め) を解消する。
fn finalize_outcome(
    outcome: IpmOutcome,
    user_eps: f64,
    n_orig: usize,
    total_deadline: Option<Instant>,
    cancelled: bool,
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

    // 外部停止 = deadline 経過 OR cancel_flag セット。後者は cancel_flag 事前設定での
    // 即停止 / parallel sibling からの cooperative cancel をカバーする。
    let timed_out = cancelled || total_deadline.is_some_and(|d| Instant::now() >= d);

    if outcome.solution.is_empty() {
        // best-so-far も無い: 真の時間切れ or 数値破綻 (factorize fail 等)。
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
        SolveStatus::Optimal
    } else if timed_out {
        // 外部 deadline 経過 / cancel_flag セットで精度未達 → 真の Timeout
        // (時間あれば改善余地ありの可能性)
        SolveStatus::Timeout
    } else {
        // deadline 内で精度未達 → IPM が内部で諦めた (数値的限界)。
        // best-so-far 解は保持。bench 側で SUBOPTIMAL として表示される。
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
        ..Default::default()
    }
}
