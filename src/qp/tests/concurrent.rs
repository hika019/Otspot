#[cfg(feature = "parallel")]
use super::super::*;
#[cfg(feature = "parallel")]
use super::{assert_close, EPS};
#[cfg(feature = "parallel")]
use crate::problem::SolveStatus;
#[cfg(feature = "parallel")]
use crate::sparse::CscMatrix;

/// Concurrent Eq 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_eq_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent Ge 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_ge_constraint() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent Box 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_box_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.0, EPS, "x[0]");
    assert_close(result.solution[1], 0.0, EPS, "x[1]");
}

/// Concurrent Mixed (Le+Eq)。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_mixed_constraint() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let b = vec![1.0, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq, ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent 無制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_unconstrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-2.0, -2.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
    assert_close(result.solution[1], 1.0, EPS, "x[1]");
}
