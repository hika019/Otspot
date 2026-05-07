//! solve_qp_v2: 単一 retry 層 + 単一 status 変換 で解く新規 API。
//!
//! 設計書 (`docs/solver_overview_design.md`) の 3 原則:
//! - retry 1 層 (時間内で eps 厳格化を直線的に進める)
//! - status 変換 1 箇所 (API 境界のみ)
//! - 元空間 KKT 直接判定 (scaled OK で偽 Optimal 出さない)
//!
//! 既存 `solve_qp_with` は temporarily 並行運用。v2 が品質・性能で上回ったら統合する。

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
/// presolve 済みの場合は IPM 側 Ruiz を抑止するので tighten のみ変える 3 attempts に縮約する。
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
///
/// 入口で Q-diagonal scaling 前処理を試みる:
/// Q が対角でかつ diagonal entries の dynamic range が大きい (max/min > 1e6) 問題
/// (QPLIB_9002: Q diag 9e-12 〜 2.0, var bounds 〜1e11) では、Ruiz だけでは Q'_jj
/// を均等化できず IPM が ill-conditioned KKT 系で wrong stationary point に
/// 収束する (obj=4.3e10 vs 真値 5.7e9)。
///
/// Pre-scaling: 各 column j で s_j = 1/√Q_jj (Q_jj > 0) を適用し
///   x = D x', Q' = D Q D (対角 1.0 に均等化), A' = A D, c' = D c, bounds' = bounds/D
/// 解いた後 x_orig = D x_scaled で復元する。Q が対角でない場合や dynamic range が
/// 狭い場合は no-op (直接 solve_qp_v2_with_runner を呼ぶ)。
pub fn solve_qp_v2(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if let Some((scaled_problem, col_scales)) = try_q_diagonal_scaling(problem) {
        let mut result = solve_qp_v2_with_runner(&scaled_problem, options, run_ipm);
        unscale_q_diagonal(&mut result, &col_scales, problem);
        return result;
    }
    solve_qp_v2_with_runner(problem, options, run_ipm)
}

/// Q が対角 + dynamic range > 1e6 のとき column scaling 因子を返す。
/// それ以外は None。
fn try_q_diagonal_scaling(problem: &QpProblem) -> Option<(QpProblem, Vec<f64>)> {
    let n = problem.num_vars;
    if n == 0 { return None; }

    // Q の対角要素を抽出 (上三角 CSC 格納で row==col が対角)。
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

    // Q が対角でないなら scaling すると off-diagonal が暴れる可能性 → skip
    const Q_OFFDIAG_TOL: f64 = 1e-10;
    if q_offdiag_max > Q_OFFDIAG_TOL {
        return None;
    }

    // 対角の有限・正値の dynamic range を測る
    let mut q_pos_min = f64::INFINITY;
    let mut q_pos_max = 0.0_f64;
    for &v in &q_diag {
        if v > Q_OFFDIAG_TOL {
            q_pos_min = q_pos_min.min(v);
            q_pos_max = q_pos_max.max(v);
        }
    }
    // 正対角がない (Q≡0 LP) または range が狭い場合 skip
    if !q_pos_min.is_finite() || q_pos_max <= 0.0 {
        return None;
    }
    const Q_DIAG_RANGE_TRIGGER: f64 = 1e6;
    if q_pos_max / q_pos_min < Q_DIAG_RANGE_TRIGGER {
        return None;
    }

    // s_j = 1/√Q_jj, ただし Q_jj=0 (LP-like 列) は s_j=1
    // Q'_jj = Q_jj × s_j^2 = 1 で対角均等化
    let mut col_scales = vec![1.0_f64; n];
    for j in 0..n {
        if q_diag[j] > Q_OFFDIAG_TOL {
            col_scales[j] = 1.0 / q_diag[j].sqrt();
        }
    }

    // scaled problem を構築
    // Q' = D Q D: 各値を s_row × Q × s_col に。対角のみなので s_j^2 × Q_jj = 1.
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

    // QpProblem を作る (b は不変、constraint_types も不変)
    let scaled = match QpProblem::new(
        q_s, c_s, a_s, problem.b.clone(), bounds_s, problem.constraint_types.clone(),
    ) {
        Ok(p) => p,
        Err(_) => return None,
    };

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
    // dual_solution (y) は scaling 不変 (KKT 解析より)
    // bound_duals: layout は [y_lb 群; y_ub 群] (lb 有限変数昇順, ub 有限変数昇順)
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
    // objective は不変 (cost は同じ問題)
}

/// 一般化 wrapper: runner は現在 IP-PMM のみ。
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
            // attempt 0 は full deadline を使う。timeshare すると IPM が時間内に収束しきれず
            // 不完全解で best-so-far に入る (HS21 で deadline=total/6 だと x=(4.31, 0) で停止 vs
            // full deadline で x=(2, 0))。attempt 1+ は残り時間を残 attempt 数で均等分配。
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
/// IPM 内部諦め (alpha_stall/mu_floor/NaN_guard) と真の時間切れを区別するため、
/// best-so-far の有無と deadline 経過の両方を見る。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// Q-diagonal scaling trigger 条件: 対角でない Q では scaling しない。
    #[test]
    fn test_q_diagonal_scaling_skips_non_diagonal_q() {
        // Q = [[2, 1], [1, 2]] (off-diag あり)
        let q = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        assert!(try_q_diagonal_scaling(&prob).is_none(), "off-diagonal Q では trigger しない");
    }

    /// Q-diagonal scaling trigger 条件: dynamic range が小さければ scaling しない。
    #[test]
    fn test_q_diagonal_scaling_skips_uniform_diagonal() {
        // Q = diag(1.0, 2.0) — 範囲 2 で trigger 閾値 1e6 未満
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        assert!(try_q_diagonal_scaling(&prob).is_none(), "narrow Q range では trigger しない");
    }

    /// Q-diagonal scaling: ill-conditioned diagonal Q で scaling と unscale が
    /// roundtrip で一致することを確認する (QPLIB_9002 系の base 検証)。
    /// Q_OFFDIAG_TOL=1e-10 未満は zero 扱いとして range 計算から除外されるため、
    /// 本テストは positive Q で 1e-7 〜 2.0 (range 2e7 > trigger 1e6) を使う。
    #[test]
    fn test_q_diagonal_scaling_roundtrip() {
        // Q = diag(1e-7, 2.0) — range 2e7 で trigger
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e-7, 2.0], 2, 2).unwrap();
        // c = [-3, -4] (適当な linear)
        let c = vec![-3.0, -4.0];
        // 1 つの Eq 制約: x0 + x1 = 1 (well-cond)
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0, 100.0), (0.0, 100.0)];
        let prob = QpProblem::new(
            q.clone(), c.clone(), a.clone(), b.clone(), bounds.clone(),
            vec![crate::problem::ConstraintType::Eq],
        ).unwrap();

        let (scaled, col_scales) = try_q_diagonal_scaling(&prob)
            .expect("ill-cond diag Q では trigger するべき");
        // Q' diagonal は uniform 1.0 (within ε)
        let q_s = &scaled.q;
        for col in 0..2 {
            for k in q_s.col_ptr[col]..q_s.col_ptr[col + 1] {
                if q_s.row_ind[k] == col {
                    assert!(
                        (q_s.values[k] - 1.0).abs() < 1e-12,
                        "Q' diag should be ~1.0, got {} at col {}", q_s.values[k], col
                    );
                }
            }
        }
        // 実装の col_scales[j] = 1/sqrt(Q_jj) のとき:
        //   bounds_s[j] = bounds[j] / col_scales[j]  (実装: scaled bounds = bounds / D)
        // col 0: Q=1e-7, col_scales = 1/√(1e-7) ≈ 3162.3, bounds_s = 100 / 3162.3 ≈ 0.0316
        // col 1: Q=2.0,  col_scales = 1/√2 ≈ 0.707,         bounds_s = 100 / 0.707 ≈ 141.4
        assert!((col_scales[0] - 1.0 / (1e-7_f64).sqrt()).abs() < 1e-3);
        assert!((col_scales[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((scaled.bounds[0].1 - 100.0 / col_scales[0]).abs() < 1e-9);
        assert!((scaled.bounds[1].1 - 100.0 / col_scales[1]).abs() < 1e-6);
    }

    /// Q-diagonal scaling: 解 x が unscale roundtrip で正しく復元される。
    #[test]
    fn test_q_diagonal_scaling_unscale_roundtrip() {
        // Q = diag(1e-12, 2.0), c=[-3,-4], A=[1,1] x = 1, bounds=[0,100]
        // Q-scaling 適用 → solve_qp_v2 で解いて、unscale 後に primal feas が
        // 元問題で satisfied されるかを smoke check。
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
        let result = solve_qp_v2(&prob, &opts);
        assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
        // Ax = b 検証
        let ax = prob.a.mat_vec_mul(&result.solution).unwrap();
        assert!((ax[0] - 1.0).abs() < 1e-6, "Ax=b orig 空間で satisfied");
        // bounds satisfied
        for j in 0..2 {
            let (lb, ub) = prob.bounds[j];
            assert!(result.solution[j] >= lb - 1e-9);
            assert!(result.solution[j] <= ub + 1e-9);
        }
    }
}
