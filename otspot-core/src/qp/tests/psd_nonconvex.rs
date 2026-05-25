use super::super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// 不定 Q (対角負値) → 慣性修正 IPM で NonConvex を返さないこと。
#[test]
fn test_qp_nonconvex_indefinite_q() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1.0, 1.0, 1.0], 3, 3).unwrap();
    let c = vec![0.0, 0.0, 0.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert!(
        !matches!(result.status, SolveStatus::NonConvex(_)),
        "got {:?}", result.status
    );
    assert!(
        matches!(
            result.status,
            SolveStatus::LocallyOptimal | SolveStatus::Optimal
            | SolveStatus::Unbounded | SolveStatus::Timeout
            | SolveStatus::SuboptimalSolution | SolveStatus::NumericalError
        ),
        "got {:?}", result.status
    );
}

/// 不定 Q + bounds → LocallyOptimal/Optimal/Suboptimal。
#[test]
fn test_qp_nonconvex_with_bounds() {
    let q = CscMatrix::from_triplets(
        &[0, 1],
        &[0, 1],
        &[-2.0, 2.0],
        2,
        2,
    ).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let b = vec![];
    let bounds = vec![(-1.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds.clone()).unwrap();

    let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
    let result = solve_qp_with(&problem, &opts);

    assert!(
        !matches!(result.status, SolveStatus::NonConvex(_)),
        "got {:?}", result.status
    );
    assert!(
        matches!(result.status, SolveStatus::LocallyOptimal | SolveStatus::Optimal
            | SolveStatus::SuboptimalSolution | SolveStatus::Timeout),
        "got {:?}", result.status
    );
    if !result.solution.is_empty() {
        for (&xi, &(lb, ub)) in result.solution.iter().zip(bounds.iter()) {
            assert!(xi >= lb - 1e-4 && xi <= ub + 1e-4);
        }
    }
}

/// 半正定値 Q (min eig=0) は PSD 判定。
#[test]
fn test_qp_psd_semidefinite_q() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.0, 1.0, 1.0], 3, 3).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// SolveStatus::NonConvex の Display。
#[test]
fn test_solve_status_display_nonconvex() {
    let msg = "Q matrix is indefinite".to_string();
    let status = SolveStatus::NonConvex(msg.clone());
    assert_eq!(format!("{}", status), format!("NonConvex({})", msg));
}

/// n>1000 対角負値 → NonPSD 検出。
#[test]
fn test_qp_nonconvex_large_diagonal_negative() {
    let n = 1001_usize;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = std::iter::once(-1.0_f64)
        .chain(std::iter::repeat(1.0_f64).take(n - 1))
        .collect();
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(!check_q_positive_semidefinite(&q));
}

/// n>1000 対角全正値 → PSD (偽陽性防止)。
#[test]
fn test_qp_psd_large_diagonal_positive() {
    let n = 1001_usize;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0_f64; n];
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// 閾値 ‖Q‖_max × 1e-6 内の僅かな負対角値は PSD 扱い (QPS encoding noise)。
#[test]
fn test_qp_diagonal_boundary_below_threshold() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-11_f64, 1.0, 1.0], 3, 3)
        .unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// noise floor (Q[0,0]=-1e-7, ‖Q‖_max=1) は PSD。
#[test]
fn test_qp_diagonal_boundary_at_noise_floor() {
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-7_f64, 1.0, 1.0], 3, 3).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// 閾値 |‖Q‖_max × 1e-6| 超 (Q[0,0]=-1e-4) → NonConvex。
#[test]
fn test_qp_diagonal_boundary_above_threshold() {
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-4_f64, 1.0, 1.0], 3, 3).unwrap();
    assert!(!check_q_positive_semidefinite(&q));
}

