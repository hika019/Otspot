use super::super::*;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveRoute, SolveStatus};
use crate::qp::problem::QcqpMatrix;
use crate::sparse::CscMatrix;

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

/// min -x0-x1  s.t.  x0^2+x1^2 <= 1,  x >= 0.  Convex QCQP (Le, PSD constraint
/// matrix); optimum is (1/sqrt2, 1/sqrt2), objective -sqrt(2).
fn convex_qcqp_problem() -> QpProblem {
    let n = 2usize;
    let q_obj = CscMatrix::new(n, n);
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let mut qc = QcqpMatrix::new(n);
    qc.triplets.push((0, 0, 2.0));
    qc.triplets.push((1, 1, 2.0));
    let mut problem = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();
    problem.set_quadratic_constraints(vec![qc]).unwrap();
    problem
}

/// Convex QCQP must route to the conic SOCP bridge and reach the true optimum.
///
/// Sentinel: reverting `dispatch_solve_qp` to the old blanket
/// `SolverResult::not_supported` guard makes this FAIL (status would be
/// `NotSupported` instead of `Optimal`, objective would be `INFINITY`).
#[test]
fn convex_qcqp_routes_to_conic_bridge() {
    let problem = convex_qcqp_problem();
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    assert!(
        (result.objective - (-2.0_f64.sqrt())).abs() < 1e-4,
        "objective={}",
        result.objective
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpConvex);
}

/// min x0+x1  s.t.  x0*x1 >= 1,  x in [0.1,3]^2.  Nonconvex (quadratic `>=`)
/// QCQP; global optimum is 2 at (1,1).
fn nonconvex_qcqp_problem(x1_ub: f64) -> QpProblem {
    let n = 2usize;
    let q_obj = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.1, 3.0), (0.1, x1_ub)];
    let mut qc = QcqpMatrix::new(n);
    // (1/2)x^T Qc x = x0*x1  =>  Qc = [[0,1],[1,0]].
    qc.triplets.push((0, 1, 1.0));
    qc.triplets.push((1, 0, 1.0));
    let mut problem = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();
    problem.set_quadratic_constraints(vec![qc]).unwrap();
    problem
}

/// A nonconvex (quadratic `>=`) QCQP with finite variable bounds must fall
/// back to the spatial (McCormick) global solver and reach the global optimum.
///
/// Sentinel: removing the nonconvex fallback (returning `NotSupported`
/// whenever the convex conic bridge fails) makes this FAIL.
#[test]
fn nonconvex_qcqp_routes_to_global_branch_and_bound() {
    let problem = nonconvex_qcqp_problem(3.0);
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    assert!(
        (result.objective - 2.0).abs() < 5e-3,
        "objective={}",
        result.objective
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
}

/// The spatial global solver needs a finite box for every variable (McCormick
/// envelopes are undefined at infinity). A nonconvex QCQP with an unbounded
/// variable must stay `NotSupported` rather than hang or silently misbehave.
#[test]
fn nonconvex_qcqp_with_unbounded_variable_stays_not_supported() {
    let problem = nonconvex_qcqp_problem(f64::INFINITY);
    let result = solve_qp(&problem);
    assert!(
        matches!(result.status, SolveStatus::NotSupported(_)),
        "expected NotSupported, got {:?}",
        result.status
    );
    assert!(result.solution.is_empty());
    assert_eq!(result.objective, f64::INFINITY);
}

/// `timeout_secs: Some(0.0)` must trip the conic IPM's deadline check on (at
/// latest) the first iteration, rather than running to completion or hanging.
///
/// Sentinel: if timeout propagation from `SolverOptions` into `ConicOptions`
/// is dropped, this FAILS (status would be `Optimal`, not `Timeout`).
#[test]
fn convex_qcqp_propagates_timeout() {
    let problem = convex_qcqp_problem();
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout, "{:?}", result.status);
    assert!(result.stats.deadline_triggered);
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpConvex);
}
