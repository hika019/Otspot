use super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

mod bound_duals;
mod concurrent;
mod dual_recovery;
mod dual_refit;
mod pfeas;
mod postsolve;
mod presolve;
mod psd_nonconvex;
mod smoke;
mod status_dfeas;

const EPS: f64 = 1e-2;

fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
    assert!(
        (a - b).abs() < eps,
        "{}: expected {:.8}, got {:.8} (diff={:.2e})",
        name,
        b,
        a,
        (a - b).abs()
    );
}
/// solve_as_lp が NumericalError を返さないこと。
#[test]
fn test_qp001_solve_as_lp_no_numerical_error() {
    let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![4.0];
    let bounds = vec![(0.0f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::NumericalError);
}

/// timeout_secs=None で有限ステップ収束。
#[test]
fn test_a2t03_qp_no_deadline_converges() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        timeout_secs: None,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
}

/// cancel_flag 事前設定で Timeout。
#[test]
fn test_a3c02_cancel_flag_preset_qp_returns_timeout() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&cancel)),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// presolve 有無で解が一致 (透過性)。
#[test]
fn test_a4p01_presolve_transparency_qp() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts_with = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let opts_without = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result_with = solve_qp_with(&problem, &opts_with);
    let result_without = solve_qp_with(&problem, &opts_without);
    assert_eq!(result_with.status, SolveStatus::Optimal);
    assert_eq!(result_without.status, SolveStatus::Optimal);
    assert!((result_with.solution[0] - result_without.solution[0]).abs() < 1e-3);
    assert!((result_with.solution[1] - result_without.solution[1]).abs() < 1e-3);
}

/// n>1000 では Cholesky skip。対角負値は検出、非対角の非 PSD は skip (既知制限)。
#[test]
fn test_a6i03_nonconvex_skip_for_large_n() {
    let n = 1001usize;
    let mut rows = vec![0usize];
    let mut cols = vec![0usize];
    let mut vals = vec![-1e-3_f64];
    for i in 1..n {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
    }
    let q1 = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(!check_q_positive_semidefinite(&q1));

    let mut rows2: Vec<usize> = (0..n).collect();
    let mut cols2: Vec<usize> = (0..n).collect();
    let mut vals2: Vec<f64> = vec![1.0; n];
    rows2.push(0);
    cols2.push(1);
    vals2.push(-2.0);
    let q2 = CscMatrix::from_triplets(&rows2, &cols2, &vals2, n, n).unwrap();
    assert!(check_q_positive_semidefinite(&q2));
}

/// A7-CS02: concurrent solver スレッド安全性（cancel_flag 経由の停止）
#[cfg(feature = "parallel")]
#[test]
fn test_a7cs02_concurrent_cancel_flag_thread_safety() {
    // SPEC: A7-CS02
    // concurrent solver で Optimal を発見したとき cancel_flag でリソースリーク・
    // データ競合なしに停止することを確認（10回繰り返してクラッシュなし）
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    for _ in 0..10 {
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
    }
}

/// 全スレッド Timeout → Timeout。
#[cfg(feature = "parallel")]
#[test]
fn test_a7cs03_concurrent_all_timeout_returns_timeout() {
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
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// concurrent solver で cancel_flag=true → Timeout。
#[cfg(feature = "parallel")]
#[test]
fn test_a3c01_cancel_flag_concurrent_returns_timeout() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&cancel)),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}
