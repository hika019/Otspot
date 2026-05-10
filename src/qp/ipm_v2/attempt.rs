//! solve_qp_v2: 単一 retry 層 + 単一 status 変換で QP を解く API。
//!
//! 3 原則 (`docs/solver_overview_design.md` 参照):
//! - retry 1 層 (時間内で eps 厳格化を直線的に進める)
//! - status 変換 1 箇所 (API 境界のみ)
//! - 元空間 KKT 直接判定 (scaled OK で偽 Optimal 出さない)

use crate::options::SolverOptions;
use crate::presolve::{
    run_qp_presolve_phase1, run_qp_presolve_phase2,
    qp_transforms::{QpPresolveStatus, QpPostsolveStep},
};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use super::core::run_ipm;
use crate::presolve::QpPresolveResult;
use super::outcome::IpmOutcome;
use std::time::Instant;

/// 1 attempt あたりの IPM 反復上限。
///
/// 役割: marginal 問題で attempt 0 の slow progress を打ち切り、後続 attempt
/// (Ruiz off / eps_tighten) に救済機会を譲る。
///
/// 値の根拠: 実測ベース。理論的 `O(sqrt(n) * log(1/eps))` から導出する `4 * sqrt(n+m)
/// * log10(1/eps)` を試したところ、LISWET12 / YAO で attempt 0 が cap 上限近くまで
/// 走り誤った解に到達する non-monotonic な挙動を確認 (IPM 内 stagnation 検出の脆弱性)。
/// 500 は Maros 138 / QPLIB 41 全 PASS 問題が convergent に収まる empirical
/// sweet spot で、これより大きくも小さくもしない。
const MAX_ITER_PER_ATTEMPT: usize = 500;

type IpmRunner = fn(&QpProblem, &QpPresolveResult, &SolverOptions) -> IpmOutcome;

/// presolve_result の Ruiz / LargeCoeffRowScale から sigma_total を計算する。
///
/// core.rs の同名計算と同一ロジック。sigma_total は presolve スケーリングで
/// 問題が縮小された比率の下限: unscale 時に残差が 1/sigma_total 倍に増幅される。
///
/// 戻り値: sigma_total ∈ (0, +∞]。Ruiz も LargeCoeffRowScale も無ければ 1.0。
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

/// user_eps と sigma_total から eps_tighten の基底値を動的に計算する。
///
/// 問題の物理量:
///   core.rs は `eps_scaled = (user_eps / tighten) × sigma_total` を IPM に渡す。
///   IPM の `PF_FAR_FROM_TARGET_RATIO = 1e2` 閾値は `eps_orig × 100` で reg_limit
///   適応を制御する。eps_orig が大きいとこの閾値も大きくなりデュアル停滞が起きる。
///
/// 実測 (QBORE3D, sigma=2.649e-4):
///   - eps=1e-6: tighten=100 → eps_orig=1e-8 → Optimal_main (iter=149) ✓
///   - eps=1e-6: tighten=1   → eps_orig=1e-6 → residual_stall (50 iter で打ち切り) ✗
///   - eps=1e-8: tighten=1   → eps_orig=1e-8 → Optimal_main ✓  (tighten 不要)
///   - eps=1e-8: tighten=100 → eps_orig=1e-10 → IPM floor 以下を要求、失敗 ✗
///
/// 結論: tighten が必要な量は `user_eps / REF_EPS` に比例。
///   REF_EPS = 1e-8 は「tighten=1 で QBORE3D が解ける eps」。
///   tighten = ceil_pow10(user_eps / REF_EPS)。
///
/// 例:
///   user_eps=1e-6 → ratio=100   → tighten=100
///   user_eps=1e-7 → ratio=10    → tighten=10
///   user_eps=1e-8 → ratio=1     → tighten=1
///   user_eps=1e-4 → ratio=10000 → tighten=10000 (但し上限 1000: IPM floor 制約)
///
/// sigma_total は現在の判定に使用しない: IPM stall は eps_orig の絶対値に依存し、
/// sigma_total の大小に直接依存しない (sigma が変わっても eps_orig の適正値は変わらない)。
/// sigma_total が 1.0 以上 (スケーリングなし) でも公式は同じ。
///
/// 戻り値: tighten ∈ [1, 1000] のべき10に切り上げた値。
fn dynamic_base_tighten(sigma_total: f64, user_eps: f64) -> f64 {
    // 参照 eps: この値以下の user_eps では tighten=1 で十分 (実測)
    const REF_EPS: f64 = 1e-8;
    let _ = sigma_total; // 現在の判定に不使用 (将来の拡張点として保持)
    let ratio = user_eps / REF_EPS;
    if ratio <= 1.0 {
        return 1.0;
    }
    // ceil_pow10(ratio): ratio を上回る最小の 10 のべき乗
    //   ratio=100   → log10=2.0 → ceil=2 → 100
    //   ratio=61.4  → log10=1.788 → ceil=2 → 100
    //   ratio=10    → log10=1.0 → ceil=1 → 10
    let pow = ratio.log10().ceil();
    // 上限 1000: tighten > 1000 では eps_orig が IPM 精度 floor 以下になり逆効果
    10_f64.powf(pow.min(3.0))
}

/// QP を v2 経路 (IP-PMM) で解く。
///
/// Q が対角の場合、`s_j = 1/√Q_jj` の column scaling を入口で適用して `Q'_jj` を 1 に
/// 均等化する (`x = D x'`, `Q' = D Q D`)。解いた後 `x_orig = D x_scaled` で復元する。
pub fn solve_qp_v2(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if let Some((scaled_problem, col_scales)) = try_q_diagonal_scaling(problem) {
        let mut result = solve_qp_v2_with_runner(&scaled_problem, options, run_ipm);
        unscale_q_diagonal(&mut result, &col_scales, problem);
        return result;
    }
    solve_qp_v2_with_runner(problem, options, run_ipm)
}

/// Q が対角のとき column scaling 因子を返す。それ以外は None。
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
    // 正対角がない (Q≡0 LP) 場合 skip
    if !q_pos_min.is_finite() || q_pos_max <= 0.0 {
        return None;
    }
    // dynamic range が狭い Q (例: STADAT2/3 の uniform diag) で常時 pre-scaling すると、
    // IPM の K-行列 conditioning を僅かに悪化させ pfeas が user_eps 周辺で stagnate する
    // 実例あり。広 range (Maros QPLIB_9002 級) のときのみ effective なため、empirical
    // 閾値で gate する。値 1e6 は user_eps 不依存。
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

    // deadline は presolve より先に固定する: 巨大 QP の presolve 内 hot loop も
    // deadline を見られるようにするため。
    if opts.deadline.is_none() {
        if let Some(secs) = opts.timeout_secs {
            opts.deadline = Some(start_time + std::time::Duration::from_secs_f64(secs));
            opts.timeout_secs = None;
        }
    }
    let total_deadline = opts.deadline;
    let user_eps = opts.ipm_eps();

    // 50k 超の巨大問題は presolve を skip (内ループの O(n*m) で deadline を
    // 食い切るため)。それ未満は通常 presolve 適用。
    const PRESOLVE_SIZE_LIMIT: usize = 50_000;
    let presolve_result = if opts.presolve
        && problem.num_vars <= PRESOLVE_SIZE_LIMIT
        && problem.num_constraints <= PRESOLVE_SIZE_LIMIT
    {
        let phase1 = run_qp_presolve_phase1(problem, &opts);
        // QP_PRESOLVE_PHASE2=0 で phase2 (Ruiz / 大係数 scale) を無効化する DIAG hook。
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

    // 巨大 QP の presolve が deadline を食い切ったら IPM を走らせず即 Timeout。
    if total_deadline.is_some_and(|d| Instant::now() >= d) {
        return finalize_outcome(IpmOutcome::empty(), user_eps, n_orig, total_deadline, false);
    }

    // presolve が Ruiz 済なら IPM 側で重ね掛けすると二重スケールで誤収束するため
    // `use_ruiz=false` のみを試す。
    let presolve_did_ruiz = presolve_result.ruiz_scaler.is_some();
    let mut best: Option<IpmOutcome> = None;

    // 試行配列 `(use_ruiz, eps_tighten)`:
    //
    // `eps_tighten` の役割: `opts.ipm.eps = user_eps / tighten` → core.rs が
    // `eps_scaled = (user_eps / tighten) × sigma_total` を IPM に渡す。
    // sigma_total ≪ 1 の問題では IPM の PF_FAR_FROM_TARGET_RATIO 閾値 (eps_orig × 1e2)
    // が大きくなり reg_limit 適応が鈍化してデュアル停滞する。tighten を sigma_total の
    // 逆数オーダーに設定すると eps_orig が十分小さくなり reg_limit 適応が促進される。
    //
    // 動的 base_tighten: user_eps / REF_EPS(1e-8) から導出。
    //   user_eps=1e-4 → ratio=10000 → base=10000。
    //   user_eps=1e-6 → ratio=100   → base=100。
    //   user_eps=1e-8 → ratio=1     → base=1 (tighten 不要)。
    //
    // `use_ruiz` on/off: 行列形状による algorithmic alternative。
    //     on  ... 典型 row/col 不均一行列で cond を改善
    //     off ... `|b|` 巨大 (BOYD2 級) で Ruiz が初期点を歪める系の救済
    let sigma_total = compute_presolve_sigma_total(&presolve_result);
    let base_tighten = dynamic_base_tighten(sigma_total, user_eps);
    // 小バッファ: base × 10 を第2候補とする (sigma 推定誤差 + IPM 非線形余裕)。
    // presolve Ruiz 済なら IPM 側 Ruiz は重複適用しない (use_ruiz=false のみ)。
    // presolve Ruiz なしなら IPM 側 Ruiz on/off を試す。
    //
    // 段階的 tighten: base → base×10 → base/10 → 1 の順で試す。
    // base/10 は旧 EPS_TIGHTEN_FACTORS の中間値 (e.g. eps=1e-6 → base=100 → 10 も試す)。
    // QGFRDXPN (sigma=9.577e-6) は tighten=10 (eps_scaled≈9.6e-13) で収束するが
    // tighten=100 (eps_scaled≈9.6e-14) では IPM floor を下回り stall する。
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
        opts.ipm.eps = (user_eps / tighten).max(f64::EPSILON);

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

    let outcome = best.unwrap_or_else(IpmOutcome::empty);
    // cancel_flag を「外部停止」として deadline 経過と同様に扱う
    // (preset cancel → Timeout 契約のため)。
    let cancelled = options
        .cancel_flag
        .as_ref()
        .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed));
    finalize_outcome(outcome, user_eps, n_orig, total_deadline, cancelled)
}

/// `IpmOutcome` から `SolverResult` への単一 status 変換。
/// solution は既に元空間で postsolve / unscale / clip 済み。
///
/// status 分類:
/// - Optimal: ユーザー精度 eps 達成
/// - Timeout: 外部 deadline 経過 (best-so-far の有無は問わない)
/// - SuboptimalSolution: IPM が内部停止 (alpha_stall / mu_floor / NaN_guard) + best あり
/// - NumericalError: best-so-far も無い (factorize 失敗 / 即時破綻)
fn finalize_outcome(
    outcome: IpmOutcome,
    user_eps: f64,
    n_orig: usize,
    total_deadline: Option<Instant>,
    cancelled: bool,
) -> SolverResult {
    // 確定的 Infeasible / Unbounded / NonConvex は最優先で外部に伝える。
    // objective: Infeasible → +∞ (到達不能), Unbounded → -∞, NonConvex → NaN。
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

    // 外部停止 = deadline 経過 OR cancel_flag セット。
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

    /// Q-diagonal scaling trigger 条件: dynamic range が狭ければ scaling しない。
    #[test]
    fn test_q_diagonal_scaling_skips_uniform_diagonal() {
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

    /// unscale_q_diagonal: x = D x_s, y = y_s, z_orig = z_s / D の関係に従って
    /// 解と bound dual を逆変換することを直接確認する。
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
        // 仮想 scaled 結果: x_s = (1, 2, 3)、y 空、bound_duals: [lb_0, lb_1, ub_0, ub_2]
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0, 2.0, 3.0],
            dual_solution: vec![],
            bound_duals: vec![10.0, 20.0, 30.0, 40.0],
            ..SolverResult::default()
        };
        unscale_q_diagonal(&mut result, &col_scales, &prob);
        // 期待: x_orig = (2*1, 0.5*2, 4*3) = (2, 1, 12)
        assert!((result.solution[0] - 2.0).abs() < 1e-12);
        assert!((result.solution[1] - 1.0).abs() < 1e-12);
        assert!((result.solution[2] - 12.0).abs() < 1e-12);
        // 期待: lb_0 が col 0 の lb_dual (col_scales[0]=2.0 で割る)
        //       lb_1 が col 1 の lb_dual (col_scales[1]=0.5 で割る)
        //       ub_0 が col 0 の ub_dual (col_scales[0]=2.0 で割る)
        //       ub_2 が col 2 の ub_dual (col_scales[2]=4.0 で割る)
        assert!((result.bound_duals[0] - 5.0).abs() < 1e-12, "lb_0 / 2.0");
        assert!((result.bound_duals[1] - 40.0).abs() < 1e-12, "lb_1 / 0.5");
        assert!((result.bound_duals[2] - 15.0).abs() < 1e-12, "ub_0 / 2.0");
        assert!((result.bound_duals[3] - 10.0).abs() < 1e-12, "ub_2 / 4.0");
    }
}
