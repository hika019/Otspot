//! 内点法（IP-PMM: Interior Point Proximal Method of Multipliers）QPソルバー
//!
//! Mehrotra predictor-corrector + IP-PMM 正則化による QP 求解。
//!
//! # ファイル構成
//! - `mod.rs`:     公開 API・定数・テスト
//! - `kkt.rs`:     KKT 行列構築・疎行列演算ヘルパー
//! - `step.rs`:    IPM Mehrotra inner solver（solver_loop 使用）
//! - `ippmm.rs`:   IP-PMM inner solver（solver_loop 使用）
//! - `init.rs`:    初期点計算（Mehrotra heuristic）
//! - `solver_loop.rs`: Predictor-Corrector-Gondzio 共通ループ部品
//! - `scaling.rs`: Ruiz スケーリングラッパー・アンスケール・後検証

pub(crate) mod common;
pub(crate) mod init;
pub(crate) mod kkt;
pub(crate) mod step;
pub(crate) mod ippmm;
pub(crate) mod solver_loop;
pub(crate) mod scaling;

use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::problem::QpProblem;

// scaling モジュールの公開関数を ipm 名前空間に再エクスポート
#[cfg(test)]
pub(crate) use scaling::check_dfeas_status;
#[cfg(test)]
pub(crate) use scaling::check_dfeas_status_relative;

// ---------------------------------------------------------------------------
// IPM 固定パラメータ
// ---------------------------------------------------------------------------

/// fraction-to-boundary τ（solver_loop.rs がこの定数を参照する）
pub(crate) const TAU: f64 = 0.995;
/// IP-PMM 正則化最小値
#[allow(dead_code)]
pub(crate) const DELTA_MIN: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Gondzio multiple centrality correctors パラメータ（solver_loop.rs が参照）
// ---------------------------------------------------------------------------

/// Gondzio: target step size factor β
pub(crate) const BETA_GONDZIO: f64 = 1.0;
/// Gondzio: complementarity lower bound factor
pub(crate) const GAMMA_L: f64 = 0.1;
/// Gondzio: complementarity upper bound factor
pub(crate) const GAMMA_U: f64 = 10.0;
/// Gondzio: step size 改善の最小閾値
pub(crate) const ALPHA_IMPROVE_THRESHOLD: f64 = 1e-3;

// ---------------------------------------------------------------------------
// 公開 API
// ---------------------------------------------------------------------------

/// IPM (Mehrotra predictor-corrector + IP-PMM) で QP を解く
///
/// Ruiz equilibration スケーリングを適用してから内部ソルバーを呼ぶ。
/// options.use_ruiz_scaling=false のときはスケーリングをスキップ。
pub fn solve_qp_ipm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    scaling::solve_with_ruiz_scaling(problem, options, step::solve_qp_ipm_inner)
}

/// IP-PMM（Interior Point-Proximal Method of Multipliers）で QP を解く
///
/// 完全独立実装（step.rs / kkt.rs 不使用）。Ruiz スケーリングラッパー付き。
pub(crate) fn solve_qp_ippmm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    scaling::solve_with_ruiz_scaling(problem, options, ippmm::solve_ippmm_inner)
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::scaling::{unscale_ipm_result, compute_amplification, EPS_FLOOR, post_verify_solution};
    use crate::linalg::ruiz::RuizScaler;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-5;

    fn close(a: f64, b: f64, name: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name,
            b,
            a,
            (a - b).abs()
        );
    }

    fn default_opts() -> SolverOptions {
        SolverOptions::default()
    }

    /// IPM-T1: 2変数基本 QP
    /// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_basic_2d() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x + y >= 1 → -(x+y) <= -1
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T1: status");
        close(result.solution[0], 0.5, "IPM-T1: x[0]");
        close(result.solution[1], 0.5, "IPM-T1: x[1]");
        close(result.objective, 0.5, "IPM-T1: objective");
    }

    /// IPM-T2: 制約なし QP (解析解)
    /// min (x-3)^2 + (y-4)^2 = 1/2*2*(x^2+y^2) - 6x - 8y + const
    /// Q=2I, c=[-6,-8], 制約なし
    /// 期待: x*=3, y*=4, obj=-25
    #[test]
    fn test_ipm_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T2: status");
        close(result.solution[0], 3.0, "IPM-T2: x[0]");
        close(result.solution[1], 4.0, "IPM-T2: x[1]");
        close(result.objective, -25.0, "IPM-T2: objective");
    }

    /// IPM-T3: 等式制約付き QP
    /// min x^2 + y^2  s.t. x + y = 1
    /// 等式を 2 不等式で表現: x+y<=1, -(x+y)<=-1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(10.0), ..default_opts() };
        let result = solve_qp_ipm(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T3: status");
        close(result.solution[0], 0.5, "IPM-T3: x[0]");
        close(result.solution[1], 0.5, "IPM-T3: x[1]");
        close(result.objective, 0.5, "IPM-T3: objective");
    }

    /// IPM-T4: Box 制約付き QP
    /// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
    /// Q=2I, c=[-4,-4], bounds=[0,1]^2
    /// 期待: x*=y*=1, obj=-6
    #[test]
    fn test_ipm_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(10.0), ..default_opts() };
        let result = solve_qp_ipm(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T4: status");
        close(result.solution[0], 1.0, "IPM-T4: x[0]");
        close(result.solution[1], 1.0, "IPM-T4: x[1]");
        close(result.objective, -6.0, "IPM-T4: objective");
    }

    /// IPM-T5: ポートフォリオ最適化（3変数等式+非負制約）
    /// min 1/2 w^T Σ w  s.t. sum(w)=1, w >= 0
    /// Σ = diag(2,2,2), 対称解: w* = [1/3, 1/3, 1/3], obj = 1/3
    #[test]
    fn test_ipm_portfolio() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3,
            3,
        )
        .unwrap();
        let c = vec![0.0, 0.0, 0.0];
        // 等式 sum=1 (2不等式) + 非負制約 w>=0 (3不等式)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5,
            3,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T5: status");
        close(result.solution[0], 1.0 / 3.0, "IPM-T5: w[0]");
        close(result.solution[1], 1.0 / 3.0, "IPM-T5: w[1]");
        close(result.solution[2], 1.0 / 3.0, "IPM-T5: w[2]");
        close(result.objective, 1.0 / 3.0, "IPM-T5: objective");
    }

    /// IPM-T6: タイムアウト動作確認（極小 timeout で Timeout が返ること）
    #[test]
    fn test_ipm_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions { timeout_secs: Some(0.0001), ..Default::default() };
        opts.use_ruiz_scaling = false;
        let result = solve_qp_ipm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "IPM-T6: expected Timeout or Optimal, got {:?}",
            result.status
        );
    }


    /// IPM-T8: post-unscaling 偽Optimal検出テスト
    ///
    /// scaled空間での収束を満たしているが元空間で許容誤差を超過する SolverResult を
    /// unscale_ipm_result に渡し、Optimal が返らないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  s.t. x <= 0.5  (n=1, m=1, A=[[1.0]], b=[0.5])
    ///   RuizScaler: d=[2.0], e=[1.0], c=1.0
    ///   mock解: x_scaled=[1.0]  → x_orig = d[0]*x_s[0] = 2.0
    ///   pfeas = A*x_orig - b = 2.0 - 0.5 = 1.5  >> eps*(1+0.5) ≈ 1.5e-6
    #[test]
    fn test_ipm_post_unscaling_false_optimal_detection() {
        // 問題構築: min x^2  s.t. x <= 0.5
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![0.5];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        // RuizScaler を手動設定: d=[2.0] により x_orig = 2.0*x_scaled
        let mut scaler = RuizScaler::new(1, 1);
        scaler.d = vec![2.0];
        scaler.e = vec![1.0];
        scaler.c = 1.0;

        // scaled解 x_scaled=[1.0] → x_orig=2.0  は  x<=0.5  を大幅違反
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0],      // scaled x_s
            dual_solution: vec![0.0], // scaled y_s
            objective: 1.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // pfeas = 2.0 - 0.5 = 1.5 >> eps*(1+0.5) → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T8: 偽Optimal（元空間pfeas超過）でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T8: 降格先はSuboptimalSolutionであること"
        );
        // unscale後の解が正しいことも確認
        close(result.solution[0], 2.0, "IPM-T8: x[0] unscaled");
    }

    /// IPM-T9: post-unscaling 正常Optimal確認テスト
    ///
    /// 元空間でも許容誤差内に収まる場合、Optimal がそのまま維持されることを確認。
    ///
    /// 設定:
    ///   問題: min x^2  s.t. x <= 1.0  (n=1, m=1, A=[[1.0]], b=[1.0])
    ///   RuizScaler: d=[1.0], e=[1.0], c=1.0 (恒等変換)
    ///   mock解: x_scaled=[0.0]  → x_orig=0.0  pfeas=max(0, 0-1)=0 < eps
    #[test]
    fn test_ipm_post_unscaling_genuine_optimal_preserved() {
        // 問題構築: min x^2  s.t. x <= 1.0
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 1);

        // x_orig=0.0 は x<=1.0 を満たす (pfeas=0)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // pfeas = 0 < eps*(1+1) → 真のOptimal → そのままOptimalを返すべき
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T9: 真のOptimal（元空間pfeas許容内）でOptimalが維持されること"
        );
    }

    /// IPM-T10: bfeas違反の検出テスト
    ///
    /// lb/ubを持つ問題に対し、境界を大幅に超過するmock解を渡した場合に
    /// unscale_ipm_result が Optimal を返さないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  bounds: 0.0 <= x <= 0.5
    ///   RuizScaler: 恒等変換 (d=[1.0], e=[], c=1.0)
    ///   mock解: x=[1.0]  → ub違反: 1.0 - 0.5 = 0.5 >> eps
    #[test]
    fn test_ipm_bfeas_violation_detected() {
        // bounds: 0.0 <= x <= 0.5 (constraint as bounds)
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::new(0, 1); // 制約なし（boundsのみ）
        let b = vec![];
        let bounds = vec![(0.0_f64, 0.5_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー（Ruiz変換なし相当）
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=1.0 は ub=0.5 を大幅違反
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0],
            dual_solution: vec![],
            objective: 1.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // bfeas = 0.5 >> eps*(1+0.5) ≈ 1.5e-6 → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T10: bfeas違反でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T10: 降格先はSuboptimalSolutionであること"
        );
    }

    /// IPM-T11: bfeas OK かつ dfeas OK → Optimal 維持テスト
    ///
    /// lb/ubを持つ問題に対し、KKT最適解のmock解を渡した場合に
    /// unscale_ipm_result が Optimal を維持することを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  bounds: -1.0 <= x <= 1.0
    ///   mock解: x=[0.0] (KKT最適解: ∇f(0)=0, bounds非活性 → bound_duals=0)
    ///   dfeas = |Q*x + c + bound_contrib| = |2*0 + 0 + 0| = 0 < threshold
    ///
    /// NOTE: このテストは bound_duals修正後にdfeasチェックが有効になった。
    /// KKT最適解x=0を使用（旧x=0.5はdfeas=1.0でSUBに降格するため修正）。
    #[test]
    fn test_ipm_bfeas_within_bounds_preserved() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(-1.0_f64, 1.0_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let scaler = RuizScaler::new(1, 0);

        // mock解: x=0.0 (KKT最適解。dfeas = |2*0 + 0| = 0 < threshold)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            objective: 0.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T11: KKT最適解はOptimalを維持すること"
        );
    }

    /// IPM-T12: post-unscaling dfeas違反の検出テスト
    ///
    /// dfeas = ||Q*x + A^T*y + c||_inf が閾値を大幅超過する mock 解を渡したとき
    /// unscale_ipm_result が Optimal を返さないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2 + x  (Q=[[2.0]], c=[1.0], 制約なし)
    ///   真の最適解: x* = -0.5 (dfeas=0)
    ///   mock解: x=[10.0], y=[]
    ///   dfeas = |2*10 + 1| = 21 >> eps*(1+1) ≈ 2e-6
    #[test]
    fn test_ipm_dfeas_violation_detected() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=10.0 → dfeas = |2*10 + 1| = 21 >> eps*(1+1)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![10.0],
            dual_solution: vec![],
            objective: 110.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // dfeas 超過 → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T12: dfeas違反でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T12: 降格先はSuboptimalSolutionであること"
        );
    }

    /// IPM-T13: post-unscaling dfeas正常（Optimal維持）テスト
    ///
    /// dfeas が閾値内に収まる真の最適解を mock で渡したとき
    /// unscale_ipm_result が Optimal を維持することを確認する。
    ///
    /// 設定:
    ///   問題: min x^2 + x  (Q=[[2.0]], c=[1.0], 制約なし)
    ///   mock解: x=-0.5 (真の最適解)
    ///   dfeas = |2*(-0.5) + 1| = 0 < eps*(1+1)
    #[test]
    fn test_ipm_dfeas_within_tolerance_preserved() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=-0.5 → dfeas = |2*(-0.5) + 1| = 0.0 < eps*(1+1)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![-0.5],
            dual_solution: vec![],
            objective: -0.25,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // dfeas = 0 < eps*(1+1) → 真のOptimal → Optimalを維持すること
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T13: dfeas許容内の解はOptimalを維持すること"
        );
    }

    /// IPM-T14: compute_amplification 計算検証
    ///
    /// e_min=0.1, d_min=0.01, c=1.0 → amplification=max(1/0.1, 1/(1.0*0.01))=max(10.0, 100.0)=100.0
    #[test]
    fn test_compute_amplification_calculation() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.d = vec![0.01, 1.0]; // d_min=0.01
        scaler.e = vec![0.1, 1.0]; // e_min=0.1
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!(
            (amp - 100.0).abs() < 1e-10,
            "IPM-T14: expected amplification=100.0, got {:.6}",
            amp
        );
    }

    /// IPM-T15: 強スケーリング環境での adjusted_eps 確認
    ///
    /// amplification > 1.0 → adjusted_eps = user_eps / amplification < user_eps であること
    #[test]
    fn test_adjusted_eps_less_than_user_eps() {
        let mut scaler = RuizScaler::new(2, 1);
        scaler.d = vec![0.1, 0.5]; // d_min=0.1
        scaler.e = vec![0.5];      // e_min=0.5
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!(amp > 1.0, "IPM-T15: amplification > 1.0 を期待, got {:.6}", amp);
        let user_eps = 1e-6_f64;
        let adjusted_eps = (user_eps / amp).max(EPS_FLOOR);
        assert!(
            adjusted_eps < user_eps,
            "IPM-T15: adjusted_eps ({:.2e}) < user_eps ({:.2e}) であること",
            adjusted_eps,
            user_eps
        );
    }

    /// IPM-T16: lb-only 活性ケースの bound_duals 値検証（TC-01, TC-02対応）
    ///
    /// min x s.t. x >= 1.0  (lb活性: x*=1, y_lb=1.0 > 0)
    /// Q=[[1]], c=[0] → 実際には Q=[[2]]（「1/2あり」規約: min 1/2*2*x^2 = min x^2, min x には c=[-1] 相当）
    ///
    /// 問題: min x s.t. x >= 1.0
    /// Q=[[0]] (zero, 線形問題), c=[1.0]（min x）, bounds=[(1.0, +∞)]
    /// ただし Q=0 はIPMに不安定。代わりに:
    /// min 1/2*ε*x^2 + x s.t. x >= 1 (ε→0+, lb活性)
    /// KKT: ε*x + 1 - y_lb = 0, x=1 → y_lb = ε + 1 ≈ 1.0
    /// → bound_duals[0] ≈ 1.0 (lb dual)
    ///
    /// NOTE: bound_duals修正後。y[m_orig..m_ext]が正しく計算される。
    #[test]
    fn test_ipm_bound_duals_lb_only_active() {
        // min 1/2*0.001*x^2 + x s.t. x >= 1.0
        // Q = [[0.001]] (ε = 0.001), c = [1.0], bounds = [(1.0, +∞)]
        // KKT at x=1: 0.001*1 + 1 - y_lb = 0 → y_lb = 1.001
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(1.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T16: status should be Optimal");
        assert!(
            (result.solution[0] - 1.0).abs() < 1e-4,
            "IPM-T16: x*≈1.0, got {:.6}", result.solution[0]
        );
        // bound_duals[0] = y_lb ≈ 1.001 (lb dual at x=1)
        assert!(!result.bound_duals.is_empty(), "IPM-T16: bound_duals should be non-empty");
        assert!(
            (result.bound_duals[0] - 1.001).abs() < 0.1,
            "IPM-T16: bound_duals[0] (lb dual) ≈ 1.001, got {:.6}", result.bound_duals[0]
        );
    }

    /// IPM-T17: ub-only 活性ケースの bound_duals 値検証（TC-01, TC-02対応）
    ///
    /// min -x s.t. x <= 0.5  (ub活性: x*=0.5, y_ub=1.0 > 0)
    /// Q=[[ε]] (小さい正則化), c=[-1.0], bounds=[(-∞, 0.5)]
    /// KKT: ε*x - 1 + y_ub = 0, x=0.5 → y_ub = 1 - ε*0.5 ≈ 1.0
    ///
    /// NOTE: bound_duals修正後。lb有限変数なし→ub分のみ格納。
    #[test]
    fn test_ipm_bound_duals_ub_only_active() {
        // min 1/2*0.001*x^2 - x s.t. x <= 0.5
        // KKT at x=0.5: 0.001*0.5 - 1 + y_ub = 0 → y_ub = 1 - 0.0005 = 0.9995
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![-1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 0.5_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T17: status should be Optimal");
        assert!(
            (result.solution[0] - 0.5).abs() < 1e-4,
            "IPM-T17: x*≈0.5, got {:.6}", result.solution[0]
        );
        // bound_duals[0] = y_ub ≈ 0.9995 (ub dual. lb-infiniteのためindex=0がub dual)
        assert!(!result.bound_duals.is_empty(), "IPM-T17: bound_duals should be non-empty");
        assert!(
            (result.bound_duals[0] - 1.0).abs() < 0.1,
            "IPM-T17: bound_duals[0] (ub dual) ≈ 1.0, got {:.6}", result.bound_duals[0]
        );
    }

    /// IPM-T18: 両端有限 lb活性ケース — T12のbound_dualsアサート追加（TC-01対応）
    ///
    /// T4と同じ問題: min (x-2)^2 + (y-2)^2 s.t. 0<=x<=1, 0<=y<=1
    /// x*=y*=1 (ub活性), ub duals > 0, lb duals = 0 (lb非活性)
    ///
    /// NOTE: bound_duals修正後。
    /// bound_duals格納順: lb_x, lb_y (index 0,1), ub_x, ub_y (index 2,3)
    /// x=y=1 (ub活性) → ub_duals > 0, lb_duals ≈ 0
    #[test]
    fn test_ipm_bound_duals_box_constrained() {
        // min (x-2)^2+(y-2)^2 = 1/2*4*(x^2+y^2) - 4x - 4y + const
        // Q=[[2,0],[0,2]] (「1/2あり」: 1/2*2*x^2=x^2, grad=2x), c=[-4,-4]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c_vec = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        // NOTE: bound_duals修正後もstatus=Optimalを維持することを確認。
        // もしSUBに変化する場合はKKT解析で原因を確認すること。
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T18: status");
        assert!((result.solution[0] - 1.0).abs() < 0.01, "IPM-T18: x*≈1.0");
        assert!((result.solution[1] - 1.0).abs() < 0.01, "IPM-T18: y*≈1.0");

        // bound_duals格納順: [lb_x, lb_y, ub_x, ub_y] (全4変数×両端有限)
        // x=y=1はub活性 → ub_duals(index 2,3) > 0
        // x=y=1はlb非活性 → lb_duals(index 0,1) ≈ 0 (interior point法では正の値になるが小さい)
        assert_eq!(result.bound_duals.len(), 4, "IPM-T18: 4 bound duals expected");
        // KKT: 2*x - 4 - y_lb_x + y_ub_x = 0 → y_ub_x = 4 - 2*1 + y_lb_x ≈ 2
        assert!(
            result.bound_duals[2] > 0.5,
            "IPM-T18: ub_dual_x should be positive (active), got {:.6}", result.bound_duals[2]
        );
        assert!(
            result.bound_duals[3] > 0.5,
            "IPM-T18: ub_dual_y should be positive (active), got {:.6}", result.bound_duals[3]
        );
    }

    /// T9 sanityチェック: Ruiz有効時のPOST_VERIFYループがtimeoutを大幅超過しないこと。
    /// 注意: このテストはT9修正のsanityチェック。
    /// 実問題での効果確認はベンチ（Step5 Maros/QPLIB）で行う。
    #[test]
    fn test_ipm_post_verify_timeout_stays_within_budget() {
        // 適度なサイズの問題でRuiz scaling有効 + 短めのtimeout
        // use_ruiz_scaling=true → POST_VERIFYループを通るパス
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0; 3];
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[-1.0, -1.0, -1.0], 1, 3).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let timeout_secs = 0.01;
        let mut opts = SolverOptions { timeout_secs: Some(timeout_secs), ..Default::default() };
        opts.use_ruiz_scaling = true;  // POST_VERIFYループを通るパス（T9修正対象）

        let start = std::time::Instant::now();
        let result = solve_qp_ipm(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "T9 sanity: expected Timeout or Optimal, got {:?}", result.status
        );
        // T9修正後: deadline固定によりPOST_VERIFYループの超過は最大1イテレーション分のみ
        // 2.0×で十分厳格（バグ残存なら3×超過し検出できる）
        assert!(
            elapsed < timeout_secs * 2.0,
            "T9 sanity: elapsed({:.3}s) > timeout×2({:.3}s). T9バグが残存している可能性",
            elapsed, timeout_secs * 2.0
        );
    }

    /// IPM-T7: Ruiz スケーリング有無で同一解が得られることを確認
    /// T1 と同じ問題 (min x^2+y^2, s.t. x+y>=1) で比較
    #[test]
    fn test_ipm_ruiz_scaling_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());

        let opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);

        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "IPM-T7: ruiz status");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "IPM-T7: no-ruiz status");
        close(result_ruiz.solution[0], result_no_ruiz.solution[0], "IPM-T7: x[0]");
        close(result_ruiz.solution[1], result_no_ruiz.solution[1], "IPM-T7: x[1]");
        close(result_ruiz.objective, result_no_ruiz.objective, "IPM-T7: objective");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TDD赤フェーズ: テスト不足 (△) 項目
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// A2-T02: timeout_secs=0 で即停止（IPM版）
    /// Given: timeout_secs=0, When: solve_qp_ipm, Then: 即 Timeout
    #[test]
    fn test_a2t02_ipm_timeout_zero_returns_immediately() {
        // SPEC: A2-T02
        // timeout_secs=0 → deadline = now() → 最初の should_stop() で Timeout
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { timeout_secs: Some(0.0), ..SolverOptions::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_ipm(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        // timeout_secs=0 → Timeout または Optimal（初期化中に偶然解けた場合）
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "A2-T02: timeout_secs=0 は Timeout または Optimal を返すこと。got: {:?}", result.status
        );
        assert!(
            elapsed < 0.5,
            "A2-T02: timeout_secs=0 は即座に返るべき。elapsed={:.3}s", elapsed
        );
    }

    /// A5-S01: Ruiz scaling 前後で解が等価（別問題でも確認）
    /// 既存 test_ipm_ruiz_scaling_consistency とは異なる制約条件の問題で確認
    #[test]
    fn test_a5s01_scaling_solution_equivalence_constrained() {
        // SPEC: A5-S01
        // 制約あり問題（T2: 等式制約 x+y=1）で scaling 有無で解が一致することを確認
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x+y = 1 を Ax <= b 形式で: x+y<=1 かつ -(x+y)<=-1
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2,
        ).unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());
        let opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..SolverOptions::default() };
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);
        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "A5-S01: ruiz=true → Optimal");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "A5-S01: ruiz=false → Optimal");
        assert!(
            (result_ruiz.solution[0] - result_no_ruiz.solution[0]).abs() < 1e-4,
            "A5-S01: scaling 有無で x[0] が一致すること"
        );
        assert!(
            (result_ruiz.solution[1] - result_no_ruiz.solution[1]).abs() < 1e-4,
            "A5-S01: scaling 有無で x[1] が一致すること"
        );
    }

    /// A5-S02: POST_VERIFY で SuboptimalSolution を外部 API から返さない
    /// IPM/IPPMM パスでは SuboptimalSolution → Timeout 変換済み
    #[test]
    fn test_a5s02_post_verify_no_false_optimal() {
        // SPEC: A5-S02
        // Ruiz scaling で POST_VERIFY が SuboptimalSolution を検出した場合、
        // Timeout に変換されることを確認。
        // 小さな問題では通常 Optimal が返るため、「SuboptimalSolution が返らない」を確認。
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { use_ruiz_scaling: true, ..SolverOptions::default() };
        let result = solve_qp_ipm(&problem, &opts);
        // SuboptimalSolution はバグステータス。返ってはならない。
        assert_ne!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "A5-S02: SuboptimalSolution は外部 API から返ってはならない"
        );
        assert_eq!(result.status, SolveStatus::Optimal, "A5-S02: 正常ケースは Optimal");
    }

    /// IPM-T13: 大係数行と小係数行の混在ケースで小行の違反を行ノルム正規化で正しく検出
    ///
    /// 背景:
    ///   旧方式（norm_b: 全行を最大|b_i|で正規化）では、大係数行が norm_b を支配するため
    ///   小係数行の微小違反が eps 未満に見えて偽PASSになっていた。
    ///   新方式（行ノルム正規化: 各行を自身の行ノルムで正規化）ではこの問題が解消される。
    ///
    /// 問題:
    ///   min x0^2 + x1^2  (Q=2I, c=[0,0])
    ///   s.t. 1e6*x0 = 1e6  (row 0: 大係数、rn_0=1e6)
    ///        x1     = 1    (row 1: 小係数、rn_1=1.0)
    ///   bounds: (-INF, INF)
    ///
    /// mock解: x=[1.0, 1.0+1e-5] (row0は満足、row1は delta=1e-5 の違反)
    ///
    /// 行ノルム正規化 (新方式):
    ///   pfeas_1 = 1e-5 / (1 + rn_1 + |b_1|) = 1e-5 / 3 ≈ 3.3e-6 > eps=1e-6 → 違反検出
    ///
    /// 旧方式 (norm_b = max(|b_i|) = 1e6 を全行に適用):
    ///   pfeas_1_old = 1e-5 / (1 + 1e6) ≈ 1e-11 < eps → 偽PASS (新方式で修正済み)
    #[test]
    fn test_post_verify_row_norm_rejects_small_row_violation() {
        use crate::problem::ConstraintType;

        // 問題構築: 大係数行(row0) + 小係数行(row1) の等式制約
        // A = [[1e6, 0], [0, 1]], b = [1e6, 1]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // CSC: col0→row0(1e6), col1→row1(1.0)
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e6_f64, 1.0], 2, 2).unwrap();
        let b = vec![1e6_f64, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Eq, ConstraintType::Eq],
        ).unwrap();

        // mock解: row0 は正確に満足、row1 は delta=1e-5 の違反
        let delta = 1e-5_f64;
        let mock_result = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            solution: vec![1.0, 1.0 + delta],
            dual_solution: vec![0.0, 0.0],
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = post_verify_solution(mock_result, &problem, eps);

        // 行ノルム正規化: pfeas_1 = 1e-5/3 ≈ 3.3e-6 > eps → SuboptimalSolution のまま
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T13: 小係数行(row1)の delta=1e-5 違反を行ノルム正規化で検出し SuboptimalSolution を返すべき。\
             旧 norm_b 方式では pfeas_old≈1e-11<eps で偽PASSとなる。got {:?}",
            result.status
        );
    }

    /// IPM-T14: 大係数行と小係数行の混在ケースで正確な解をOptimalと判定
    ///
    /// IPM-T13 の逆ケース: 全制約を正確に満足する解は Optimal に昇格すること。
    /// 行ノルム正規化が偽陰性（真のOptimalを拒否）を引き起こさないことを確認。
    ///
    /// 問題: IPM-T13 と同一
    /// mock解: x=[1.0, 1.0] (全制約満足)、y=[-2e-6, -2.0] (KKT双対変数)
    #[test]
    fn test_post_verify_row_norm_accepts_exact_solution() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e6_f64, 1.0], 2, 2).unwrap();
        let b = vec![1e6_f64, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Eq, ConstraintType::Eq],
        ).unwrap();

        // KKT双対変数: Q*x + c + A^T*y = 0 → [2 + 1e6*y0, 2 + y1] = 0 → y=[-2e-6, -2]
        let mock_result = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            solution: vec![1.0, 1.0],
            dual_solution: vec![-2e-6_f64, -2.0],
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = post_verify_solution(mock_result, &problem, eps);

        // pfeas_normalized = 0 (完全満足) → bfeas OK (無制限) → dfeas ≈ 0 < threshold → Optimal
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T14: 全制約を正確に満足する解は Optimal に昇格すべき。got {:?}",
            result.status
        );
    }

    /// C-IPM 防御テスト: Ge制約付きQP
    /// min x²+y²  s.t. x+y≥1 (ConstraintType::Ge)
    /// timeout_secs=5.0、wall-clock 6秒ガード
    /// 期待: Optimal, x*=y*=0.5
    #[test]
    fn test_ipm_ge_defensive() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..SolverOptions::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_ipm(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        assert_eq!(result.status, SolveStatus::Optimal, "C-IPM Ge: status");
        close(result.solution[0], 0.5, "C-IPM Ge: x[0]");
        close(result.solution[1], 0.5, "C-IPM Ge: x[1]");
        assert!(
            elapsed < 6.0,
            "C-IPM Ge: wall-clock guard 6秒超過。elapsed={:.3}s",
            elapsed
        );
    }

    /// F-IPM 退化ケース: 空制約×IPM
    /// Q=I(2x2), c=[-1,-1], A=空(0×2), b=[], timeout_secs=5.0
    /// 期待: Optimal
    #[test]
    fn test_ipm_empty_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..SolverOptions::default() };
        let result = solve_qp_ipm(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "F-IPM 空制約: status");
    }

}
