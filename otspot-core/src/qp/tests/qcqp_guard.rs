use super::super::*;
use crate::problem::SolveStatus;
use crate::qp::problem::QcqpMatrix;
use crate::sparse::CscMatrix;

/// Build a minimal 2-variable QP problem and attach quadratic constraints.
fn make_qp_with_qcqp_constraints(n_qcqp_nnz: usize) -> QpProblem {
    // min x^2 + y^2  s.t.  x + y >= 1
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let mut problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    // Attach one quadratic constraint matrix with the requested number of non-zeros.
    let mut qc = QcqpMatrix::new(2);
    for i in 0..n_qcqp_nnz {
        qc.triplets.push((i % 2, i % 2, 1.0));
    }
    problem.quadratic_constraints = vec![qc];
    problem
}

/// A QCQP (non-empty quadratic constraint triplets) must be rejected with NotSupported.
///
/// Sentinel: removing `has_qcqp_constraints()` guard from `dispatch_solve_qp` causes the
/// solver to silently treat the problem as a plain QP and return Optimal instead of
/// NotSupported — this test FAILS under that no-op revert.
#[test]
fn qcqp_rejected_with_not_supported() {
    let problem = make_qp_with_qcqp_constraints(1);
    let result = solve_qp(&problem);
    assert!(
        matches!(result.status, SolveStatus::NotSupported(_)),
        "expected NotSupported for QCQP, got {:?}",
        result.status
    );
    // Solution must be empty — no meaningful answer was produced.
    assert!(result.solution.is_empty());
    assert_eq!(result.objective, f64::INFINITY);
}

/// Multiple quadratic triplets in the constraint still trigger the guard.
#[test]
fn qcqp_rejected_multiple_triplets() {
    let problem = make_qp_with_qcqp_constraints(3);
    let result = solve_qp(&problem);
    assert!(
        matches!(result.status, SolveStatus::NotSupported(_)),
        "expected NotSupported, got {:?}",
        result.status
    );
}

/// A plain QP (quadratic_constraints empty) must NOT be rejected.
#[test]
fn plain_qp_not_rejected() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    // quadratic_constraints is empty by default — must reach the solver normally.
    let result = solve_qp(&problem);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "plain QP should solve normally"
    );
}

/// A QpProblem with a quadratic_constraints vec where every QcqpMatrix is empty
/// (zero triplets) is still a pure QP — the guard must not fire.
#[test]
fn empty_qcqp_matrices_not_rejected() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let mut problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    // Attach an empty QcqpMatrix (no triplets) — semantically no quadratic constraint.
    problem.quadratic_constraints = vec![QcqpMatrix::new(2)];
    let result = solve_qp(&problem);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "QpProblem with all-empty QcqpMatrix entries should solve normally"
    );
}
