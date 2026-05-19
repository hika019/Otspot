//! IP-PMM 単体テスト。

use super::iter::solve_ippmm_inner;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

const EPS: f64 = 1e-4; // IP-PMM は標準 IPM より tolerance がゆるめでも通ることを確認

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
    SolverOptions {
        timeout_secs: Some(10.0),
        use_ruiz_scaling: false,
        ..Default::default()
    }
}

/// IPPMM-T1: 2変数基本 QP
/// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
/// 期待: x*=y*=0.5, obj=0.5
#[test]
fn test_ippmm_basic_2d() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T1: status");
    close(result.solution[0], 0.5, "IPPMM-T1: x[0]");
    close(result.solution[1], 0.5, "IPPMM-T1: x[1]");
    close(result.objective, 0.5, "IPPMM-T1: objective");
}

/// IPPMM-T2: 制約なし QP
/// min (x-3)^2 + (y-4)^2  → Q=2I, c=[-6,-8], 制約なし
/// 期待: x*=3, y*=4, obj=-25
#[test]
fn test_ippmm_unconstrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-6.0, -8.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T2: status");
    close(result.solution[0], 3.0, "IPPMM-T2: x[0]");
    close(result.solution[1], 4.0, "IPPMM-T2: x[1]");
    close(result.objective, -25.0, "IPPMM-T2: objective");
}

/// IPPMM-T3: 等式制約付き QP
/// min x^2 + y^2  s.t. x + y = 1  (2不等式で表現)
/// 期待: x*=y*=0.5, obj=0.5
#[test]
fn test_ippmm_equality_constraint() {
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

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T3: status");
    close(result.solution[0], 0.5, "IPPMM-T3: x[0]");
    close(result.solution[1], 0.5, "IPPMM-T3: x[1]");
    close(result.objective, 0.5, "IPPMM-T3: objective");
}

/// IPPMM-T4: Box 制約付き QP
/// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
/// 期待: x*=y*=1, obj=-6
#[test]
fn test_ippmm_box_constrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-4.0, -4.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T4: status");
    close(result.solution[0], 1.0, "IPPMM-T4: x[0]");
    close(result.solution[1], 1.0, "IPPMM-T4: x[1]");
    close(result.objective, -6.0, "IPPMM-T4: objective");
}


/// IPPMM-T5: タイムアウト動作確認
#[test]
fn test_ippmm_timeout() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(0.0001),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(
        result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
        "IPPMM-T5: expected Timeout or Optimal, got {:?}",
        result.status
    );
}

/// IPPMM-T-conv1: 等式制約収束確認
/// min x²+y² s.t. x+y=1 (ConstraintType::Eq)
/// QpProblem::new() を使用
/// 期待: 5秒以内にOptimal、x*=y*=0.5
#[test]
fn test_ippmm_eq_convergence_check() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
    assert_eq!(result.status, SolveStatus::Optimal, "conv-eq: status");
    close(result.solution[0], 0.5, "conv-eq: x[0]");
    close(result.solution[1], 0.5, "conv-eq: x[1]");
}

/// IPPMM-T-conv2: 不等式制約収束確認
/// min x²+y² s.t. x+y>=1 (Le形式: -x-y <= -1、ConstraintType::Le)
#[test]
fn test_ippmm_le_convergence_check() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
    assert_eq!(result.status, SolveStatus::Optimal, "conv-le: status");
    close(result.solution[0], 0.5, "conv-le: x[0]");
    close(result.solution[1], 0.5, "conv-le: x[1]");
}

/// IPPMM-T-Ge1: Ge制約防御テスト
/// min x²+y² s.t. x+y≥1 (ConstraintType::Ge)
#[test]
fn test_ippmm_ge_defensive() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
    assert_eq!(result.status, SolveStatus::Optimal, "ge-defensive: status");
    close(result.solution[0], 0.5, "ge-defensive: x[0]");
    close(result.solution[1], 0.5, "ge-defensive: x[1]");
}

/// IPPMM-T-F1: 空制約退化ケース
/// min 0.5*(x²+y²) - x - y (Q=I, c=[-1,-1], 制約なし)
#[test]
fn test_ippmm_empty_constraints() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::new(0, 2);
    let b: Vec<f64> = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "empty-constraints: status");
    close(result.solution[0], 1.0, "empty-constraints: x[0]");
    close(result.solution[1], 1.0, "empty-constraints: x[1]");
}

/// IPPMM-T-F2: 複数等式制約退化ケース
/// min x²+y²+z² s.t. x+y=1 (Eq), y+z=1 (Eq)
#[test]
fn test_ippmm_multiple_equality_constraints() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
    let c = vec![0.0, 0.0, 0.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 1, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2, 3,
    ).unwrap();
    let b = vec![1.0, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq, ConstraintType::Eq]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "multi-eq: status");
    close(result.solution[0], 1.0 / 3.0, "multi-eq: x[0]");
    close(result.solution[1], 2.0 / 3.0, "multi-eq: x[1]");
    close(result.solution[2], 1.0 / 3.0, "multi-eq: x[2]");
}
