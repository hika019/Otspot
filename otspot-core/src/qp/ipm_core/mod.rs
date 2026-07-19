//! IP-PMM (Pougkakiotis-Gondzio 2021) による QP 求解。

pub(crate) mod common;
pub(crate) mod ippmm;
pub(crate) mod kkt;
pub(crate) mod scaling;
pub(crate) mod solver_loop;

use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::problem::QpProblem;

/// fraction-to-boundary τ。
pub(crate) const TAU: f64 = 0.995;

/// Gondzio multiple centrality correctors: target step size factor β。
pub(crate) const BETA_GONDZIO: f64 = 1.0;
pub(crate) const GAMMA_L: f64 = 0.1;
pub(crate) const GAMMA_U: f64 = 10.0;
pub(crate) const ALPHA_IMPROVE_THRESHOLD: f64 = 1e-3;

/// IPM 内部 eps の machine-noise floor。`eps_orig × σ_total` (core.rs) と
/// `user_eps / amp` (scaling.rs) の両 σ-tightening が `nr_d_rel` 達成可能域
/// (≈ √n × machine_eps、典型 n≈10^4 で √n≈100) を割らないよう下限を共通化。
/// 元空間は post-processing が user_eps で再 gate するため誤判定にはならない。
/// 100 = √n の典型値 + scaled-eps が 2.22e-14 → user_eps=1e-6 × σ_total ≥ 2.22e-8
/// で初めて発動 (well-scaled では no-op)。
pub(crate) const IPM_EPS_NOISE_FLOOR: f64 = 100.0 * f64::EPSILON;

/// Ruiz scaling 付き IP-PMM。
pub fn solve_qp_ippmm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    scaling::solve_with_ruiz_scaling(problem, options, ippmm::solve_ippmm_inner)
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::scaling::{compute_amplification, EPS_FLOOR};
    use super::*;
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

    /// min x^2 + y^2  s.t. x + y >= 1
    #[test]
    fn test_ipm_basic_2d() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = crate::qp::solve_qp_with(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 0.5, "x[0]");
        close(result.solution[1], 0.5, "x[1]");
        close(result.objective, 0.5, "objective");
    }

    /// min (x-3)^2 + (y-4)^2 (制約なし)
    #[test]
    fn test_ipm_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = crate::qp::solve_qp_with(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 3.0, "x[0]");
        close(result.solution[1], 4.0, "x[1]");
        close(result.objective, -25.0, "objective");
    }

    /// min x^2 + y^2  s.t. x + y = 1 (2 不等式表現)
    #[test]
    fn test_ipm_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..default_opts()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 0.5, "x[0]");
        close(result.solution[1], 0.5, "x[1]");
        close(result.objective, 0.5, "objective");
    }

    /// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1.
    /// 中央パス上で |x* - 1| ≈ ipm.eps、obj 勾配 ≈ 4 のため close EPS=1e-5 には eps=1e-7 が要る。
    #[test]
    fn test_ipm_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..default_opts()
        };
        opts.ipm.eps = 1e-7;
        let result = crate::qp::solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 1.0, "x[0]");
        close(result.solution[1], 1.0, "x[1]");
        close(result.objective, -6.0, "objective");
    }

    /// min 1/2 w^T Σ w  s.t. sum(w)=1, w >= 0 (対称ポートフォリオ)
    #[test]
    fn test_ipm_portfolio() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
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

        let result = crate::qp::solve_qp_with(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 1.0 / 3.0, "w[0]");
        close(result.solution[1], 1.0 / 3.0, "w[1]");
        close(result.solution[2], 1.0 / 3.0, "w[2]");
        close(result.objective, 1.0 / 3.0, "objective");
    }

    #[test]
    fn test_ipm_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions {
            timeout_secs: Some(0.0001),
            ..Default::default()
        };
        opts.use_ruiz_scaling = false;
        let result = crate::qp::solve_qp_with(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "got {:?}",
            result.status
        );
    }

    #[test]
    fn test_compute_amplification_calculation() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.d = vec![0.01, 1.0];
        scaler.e = vec![0.1, 1.0];
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!((amp - 100.0).abs() < 1e-10, "got {:.6}", amp);
    }

    #[test]
    fn test_adjusted_eps_less_than_user_eps() {
        let mut scaler = RuizScaler::new(2, 1);
        scaler.d = vec![0.1, 0.5];
        scaler.e = vec![0.5];
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!(amp > 1.0, "got {:.6}", amp);
        let user_eps = 1e-6_f64;
        let adjusted_eps = (user_eps / amp).max(EPS_FLOOR);
        assert!(adjusted_eps < user_eps);
    }

    /// lb 活性: min 1/2·ε·x^2 + x s.t. x ≥ 1 → y_lb ≈ 1 + ε。
    #[test]
    fn test_ipm_bound_duals_lb_only_active() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(1.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions {
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.solution[0] - 1.0).abs() < 1e-4,
            "got {:.6}",
            result.solution[0]
        );
        assert!(!result.bound_duals.is_empty());
        assert!(
            (result.bound_duals[0] - 1.001).abs() < 0.1,
            "got {:.6}",
            result.bound_duals[0]
        );
    }

    /// ub 活性: min 1/2·ε·x^2 − x s.t. x ≤ 0.5 → y_ub ≈ 1 − ε·0.5。
    #[test]
    fn test_ipm_bound_duals_ub_only_active() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![-1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 0.5_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions {
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.solution[0] - 0.5).abs() < 1e-4,
            "got {:.6}",
            result.solution[0]
        );
        assert!(!result.bound_duals.is_empty());
        assert!(
            (result.bound_duals[0] - 1.0).abs() < 0.1,
            "got {:.6}",
            result.bound_duals[0]
        );
    }

    /// 両端有限 box: x=y=1 で ub 活性、bound_duals 順は [lb_x, lb_y, ub_x, ub_y]。
    #[test]
    fn test_ipm_bound_duals_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c_vec = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
        let problem = QpProblem::new_all_le(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions {
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 1.0).abs() < 0.01);
        assert!((result.solution[1] - 1.0).abs() < 0.01);

        assert_eq!(result.bound_duals.len(), 4);
        assert!(
            result.bound_duals[2] > 0.5,
            "got {:.6}",
            result.bound_duals[2]
        );
        assert!(
            result.bound_duals[3] > 0.5,
            "got {:.6}",
            result.bound_duals[3]
        );
    }

    /// Ruiz 有効時 POST_VERIFY ループが deadline を 2× 以内に収めること。
    #[test]
    fn test_ipm_post_verify_timeout_stays_within_budget() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0; 3];
        let a =
            CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[-1.0, -1.0, -1.0], 1, 3).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let timeout_secs = 0.01;
        let mut opts = SolverOptions {
            timeout_secs: Some(timeout_secs),
            ..Default::default()
        };
        opts.use_ruiz_scaling = true;

        let start = std::time::Instant::now();
        let result = crate::qp::solve_qp_with(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "got {:?}",
            result.status
        );
        assert!(elapsed < timeout_secs * 2.0, "elapsed={:.3}s", elapsed);
    }

    /// Ruiz scaling 有無で同一解。
    #[test]
    fn test_ipm_ruiz_scaling_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result_ruiz = crate::qp::solve_qp_with(&problem, &SolverOptions::default());

        let opts_no_ruiz = SolverOptions {
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result_no_ruiz = crate::qp::solve_qp_with(&problem, &opts_no_ruiz);

        assert_eq!(result_ruiz.status, SolveStatus::Optimal);
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal);
        close(result_ruiz.solution[0], result_no_ruiz.solution[0], "x[0]");
        close(result_ruiz.solution[1], result_no_ruiz.solution[1], "x[1]");
        close(result_ruiz.objective, result_no_ruiz.objective, "objective");
    }

    #[test]
    fn test_a2t02_ipm_timeout_zero_returns_immediately() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            ..SolverOptions::default()
        };
        let start = std::time::Instant::now();
        let result = crate::qp::solve_qp_with(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "got: {:?}",
            result.status
        );
        assert!(elapsed < 0.5, "elapsed={:.3}s", elapsed);
    }

    #[test]
    fn test_a5s01_scaling_solution_equivalence_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result_ruiz = crate::qp::solve_qp_with(&problem, &SolverOptions::default());
        let opts_no_ruiz = SolverOptions {
            use_ruiz_scaling: false,
            ..SolverOptions::default()
        };
        let result_no_ruiz = crate::qp::solve_qp_with(&problem, &opts_no_ruiz);
        assert_eq!(result_ruiz.status, SolveStatus::Optimal);
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal);
        assert!((result_ruiz.solution[0] - result_no_ruiz.solution[0]).abs() < 1e-4);
        assert!((result_ruiz.solution[1] - result_no_ruiz.solution[1]).abs() < 1e-4);
    }

    /// この良条件凸 QP は証明書付き Optimal まで収束するべきで、eps 検証済みだが
    /// 証明書なしの SuboptimalSolution に留まってはならない (SuboptimalSolution
    /// 自体は現行 taxonomy で正当な外部 API 戻り値であり、一般に禁止ではない)。
    #[test]
    fn test_a5s02_post_verify_no_false_optimal() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);
        assert_ne!(result.status, SolveStatus::SuboptimalSolution);
        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// Ge制約付き QP の wall-clock ガード。
    #[test]
    fn test_ipm_ge_defensive() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..SolverOptions::default()
        };
        let start = std::time::Instant::now();
        let result = crate::qp::solve_qp_with(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        assert_eq!(result.status, SolveStatus::Optimal);
        close(result.solution[0], 0.5, "x[0]");
        close(result.solution[1], 0.5, "x[1]");
        assert!(elapsed < 6.0, "elapsed={:.3}s", elapsed);
    }

    /// 空制約 IPM。
    #[test]
    fn test_ipm_empty_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..SolverOptions::default()
        };
        let result = crate::qp::solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
    }
}
