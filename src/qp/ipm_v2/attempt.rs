//! solve_qp_v2: 単一 retry 層 + 単一 status 変換で QP を解く API。
//!
//! 3 原則 (`docs/solver_overview_design.md` 参照):
//! - retry 1 層 (時間内で eps 厳格化を直線的に進める)
//! - status 変換 1 箇所 (API 境界のみ)
//! - 元空間 KKT 直接判定 (scaled OK で偽 Optimal 出さない)

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
    // - `eps_tighten`: IPM 内 eps を `user_eps × {1, 10, 100}` で締めて、unscale 残差
    //   増幅 (ill-scaled 問題) を吸収する余裕を作る。
    // - `use_ruiz` on/off: 行列形状による algorithmic alternative。
    //     on  ... 典型 row/col 不均一行列で cond を改善
    //     off ... `|b|` 巨大 (BOYD2 級) で Ruiz が初期点を歪める系の救済
    const EPS_TIGHTEN_FACTORS: &[f64] = &[1.0, 10.0, 100.0];
    let attempts: Vec<(bool, f64)> = if presolve_did_ruiz {
        EPS_TIGHTEN_FACTORS.iter().map(|&t| (false, t)).collect()
    } else {
        let mut v = Vec::with_capacity(EPS_TIGHTEN_FACTORS.len() * 2);
        for &t in EPS_TIGHTEN_FACTORS { v.push((true, t)); }
        for &t in EPS_TIGHTEN_FACTORS { v.push((false, t)); }
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
