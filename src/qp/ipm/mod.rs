//! 内点法（IP-PMM: Interior Point Proximal Method of Multipliers）QPソルバー
//!
//! Mehrotra predictor-corrector + IP-PMM 正則化による QP 求解。
//!
//! # ファイル構成
//! - `mod.rs`:  公開 API・定数・Ruiz スケーリングラッパー・テスト
//! - `kkt.rs`:  KKT 行列構築・疎行列演算ヘルパー
//! - `step.rs`: メインループ（`solve_qp_ipm_inner`）・fraction-to-boundary・ユーティリティ
//! - `init.rs`: 初期点計算（Mehrotra heuristic）

pub(crate) mod init;
pub(crate) mod kkt;
pub(crate) mod step;

use crate::linalg::ruiz::RuizScaler;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;

// ---------------------------------------------------------------------------
// IPM 固定パラメータ
// ---------------------------------------------------------------------------

/// fraction-to-boundary τ
pub(crate) const TAU: f64 = 0.995;
/// IP-PMM 正則化最小値
#[allow(dead_code)]
pub(crate) const DELTA_MIN: f64 = 1e-8;
/// n > LDL_THRESHOLD のとき CG パスを自動選択
pub(crate) const LDL_THRESHOLD: usize = 20_000;
/// CG 最大反復数
pub(crate) const CG_MAX_ITER: usize = 1_000;
/// CG 収束判定（残差 L∞ノルム）
pub(crate) const CG_TOL: f64 = 1e-6;
/// Jacobi 前処理の対角最小値（ゼロ除算防止）
pub(crate) const JACOBI_MIN_DIAG: f64 = 1e-14; // PARAM: 根拠=数値的安定性(ゼロ除算防止) | 要検証=なし

// ---------------------------------------------------------------------------
// 公開 API
// ---------------------------------------------------------------------------

/// IPM (Mehrotra predictor-corrector + IP-PMM) で QP を解く
///
/// Ruiz equilibration スケーリングを適用してから内部ソルバーを呼ぶ。
/// options.use_ruiz_scaling=false のときはスケーリングをスキップ。
pub fn solve_qp_ipm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(q_s, c_s, a_s, b_s, bounds_s) {
            let scaled_result = step::solve_qp_ipm_inner(&scaled_problem, options);
            return unscale_ipm_result(scaled_result, &scaler);
        }
        // QpProblem::new 失敗 → 非スケールにフォールバック
    }

    step::solve_qp_ipm_inner(problem, options)
}

/// IPM Schur complement パスで QP を解く
///
/// Concurrent Solver の 4 番目のバリアントとして使用。
/// 通常の solve_qp_ipm（augmented system）の代替パス。
/// n <= LDL_THRESHOLD の問題に対して Schur complement LDL パスを使用。
pub(crate) fn solve_qp_ipm_schur(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(q_s, c_s, a_s, b_s, bounds_s) {
            let scaled_result = step::solve_qp_ipm_schur_inner(&scaled_problem, options);
            return unscale_ipm_result(scaled_result, &scaler);
        }
        // QpProblem::new 失敗 → 非スケールにフォールバック
    }

    step::solve_qp_ipm_schur_inner(problem, options)
}

/// スケール済み IPM 結果を元のスケールに逆変換する
fn unscale_ipm_result(result: SolverResult, scaler: &RuizScaler) -> SolverResult {
    match result.status {
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::MaxIterations => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let obj_orig = result.objective / scaler.c;
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                ..result
            }
        }
        _ => result,
    }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::kkt::{mv_ipm_apply, compute_jacobi_precond_ipm};
    use crate::options::SolverOptions;
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions { timeout_secs: Some(0.0001), ..Default::default() };
        opts.use_ruiz_scaling = false;
        let result = solve_qp_ipm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "IPM-T6: expected Timeout or Optimal, got {:?}",
            result.status
        );
    }

    /// IPM-CG-T1: mv_ipm_apply の正確性テスト
    ///
    /// Q=diag(2,2), A=[[-1,-1]] (1制約), d_inv=[0.5], delta_p=1e-7, v=[1,0]
    /// M*v = Q*v + delta_p*v + A^T D^{-1} A*v
    ///      = [2,0] + [1e-7,0] + [-1,-1]*0.5*(-1) = [2.5+1e-7, 0.5]
    #[test]
    fn test_ipm_mv_apply() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let d_inv = vec![0.5f64];
        let delta_p = 1e-7_f64;
        let v = vec![1.0_f64, 0.0];
        let mut out = vec![0.0_f64; 2];

        mv_ipm_apply(&q, &a, &d_inv, delta_p, &v, &mut out);

        let eps = 1e-10_f64;
        let expected0 = 2.5 + delta_p;
        assert!(
            (out[0] - expected0).abs() < eps,
            "mv[0]: expected {}, got {} (diff={:.2e})",
            expected0, out[0], (out[0] - expected0).abs()
        );
        assert!(
            (out[1] - 0.5).abs() < eps,
            "mv[1]: expected 0.5, got {} (diff={:.2e})",
            out[1], (out[1] - 0.5).abs()
        );
    }

    /// IPM-CG-T2: compute_jacobi_precond_ipm の正確性テスト
    ///
    /// Q=diag(2,2), A=[[-1,-1]], d_inv=[0.5], delta_p=1e-7
    /// diag(M)[j] = Q[j,j] + delta_p + d_inv[0] * A[0,j]^2 = 2 + 1e-7 + 0.5*1 = 2.5 + 1e-7
    /// m_inv[j] = 1 / (2.5 + 1e-7)
    #[test]
    fn test_ipm_jacobi_precond() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let d_inv = vec![0.5f64];
        let delta_p = 1e-7_f64;

        let m_inv = compute_jacobi_precond_ipm(&q, &a, &d_inv, delta_p);

        let expected = 1.0 / (2.0 + delta_p + 0.5 * 1.0);
        let eps = 1e-10_f64;
        assert_eq!(m_inv.len(), 2, "m_inv length");
        assert!(
            (m_inv[0] - expected).abs() < eps,
            "m_inv[0]: expected {:.10}, got {:.10}",
            expected, m_inv[0]
        );
        assert!(
            (m_inv[1] - expected).abs() < eps,
            "m_inv[1]: expected {:.10}, got {:.10}",
            expected, m_inv[1]
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());

        let opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);

        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "IPM-T7: ruiz status");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "IPM-T7: no-ruiz status");
        close(result_ruiz.solution[0], result_no_ruiz.solution[0], "IPM-T7: x[0]");
        close(result_ruiz.solution[1], result_no_ruiz.solution[1], "IPM-T7: x[1]");
        close(result_ruiz.objective, result_no_ruiz.objective, "IPM-T7: objective");
    }
}
