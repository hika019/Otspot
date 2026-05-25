use super::super::*;
use super::{assert_close, EPS};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// 全変数固定退化ケース (presolve=false で本体検証)。
#[test]
fn test_qp_all_vars_fixed() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let bounds = vec![(1.0_f64, 1.0_f64)];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let mut opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    opts.presolve = false;
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
}

/// SuboptimalSolution mapping: MaxIterations/NumericalError が外部に漏れないこと。
#[test]
fn test_suboptimal_to_optimal_mapping() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(2.0),
        ipm: crate::options::IpmOptions {
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::MaxIterations);
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
    ), "got {:?}", result.status);
}

/// MaxIterations が外部 API に漏れないこと。
#[test]
fn test_max_iterations_to_timeout_mapping() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ipm: crate::options::IpmOptions {
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::MaxIterations);
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
    ), "got {:?}", result.status);
}

/// 正常解で dfeas check が Optimal を維持。
#[test]
fn test_dfeas_optimal_preserved() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
}

/// スケール不変性 (1e6 倍) で Optimal 維持。
#[test]
fn test_dfeas_scale_invariant() {
    let scale = 1e6_f64;
    let q = CscMatrix::from_triplets(
        &[0, 1],
        &[0, 1],
        &[2.0 * scale * scale, 2.0 * scale * scale],
        2,
        2,
    )
    .unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-scale, -scale], 1, 2).unwrap();
    let b = vec![-scale];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_close(result.solution[0], 0.5, 1e-4, "x[0]");
    assert_close(result.solution[1], 0.5, 1e-4, "x[1]");
}

/// dfeas 悪化解の SuboptimalSolution 降格 (check_dfeas_status 直接呼出)。
#[test]
fn test_dfeas_bad_solution_downgraded() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    // 最適 x=y=0, dfeas=0。bad: x=y=1 で Qx+c=[2,2], dfeas=2.0。
    let bad_x = vec![1.0, 1.0];
    let bad_y: Vec<f64> = vec![];
    let bad_bd: Vec<f64> = vec![];

    let status = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 1e-6);
    assert_eq!(status, SolveStatus::SuboptimalSolution);
    let status_ok = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 10.0);
    assert_eq!(status_ok, SolveStatus::Optimal);

    let status_rel =
        ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 0.01);
    assert_eq!(status_rel, SolveStatus::SuboptimalSolution);
    let status_rel_ok =
        ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 1.0);
    assert_eq!(status_rel_ok, SolveStatus::Optimal);
}

/// 大 KKT スケール (2e12) でも相対閾値が正規化。
#[test]
fn test_dfeas_relative_threshold_large_kkt() {
    let n = 1usize;
    let q = CscMatrix::from_triplets(&[0], &[0], &[2e12], n, n).unwrap();
    let c = vec![-1e6];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert!((result.solution[0] - 5e-7).abs() < 1e-9, "x*=5e-7, got {:.2e}", result.solution[0]);
}

/// 巨大項キャンセレーション (Qx ≈ -A^Ty): 成分相対なら正確に判定。
#[test]
fn test_dfeas_cancellation_pattern() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let big_x = vec![5e9, 5e9];
    let empty_y: Vec<f64> = vec![];
    let empty_bd: Vec<f64> = vec![];
    let status =
        ipm_core::check_dfeas_status_relative(&problem, &big_x, &empty_y, &empty_bd, 0.01);
    assert_eq!(status, SolveStatus::SuboptimalSolution);

    let good_x = vec![1e-12, 1e-12];
    let status_good =
        ipm_core::check_dfeas_status_relative(&problem, &good_x, &empty_y, &empty_bd, 1e-8);
    assert_eq!(status_good, SolveStatus::Optimal);
}
