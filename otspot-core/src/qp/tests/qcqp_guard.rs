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

/// min x  s.t.  (1/2)*(-2e-10)*x^2 <= -0.1,  x in [-2e5, 2e5].
///
/// The constraint matrix -2e-10 lies inside the convex bridge's Cholesky
/// jitter band, so the bridge misclassifies the problem as convex and its
/// SOCP solve fails numerically (the true feasible set |x| >= sqrt(1e9) is
/// disconnected). The route must detect the unclean outcome and fall back to
/// the global solver, which finds x = -2e5, objective -2e5.
///
/// Sentinel: reverting the clean-outcome gate (accepting any non-NotSupported
/// convex-bridge result) makes this FAIL with a MaxIterations/NaN result.
#[test]
fn jitter_band_indefinite_qcqp_falls_back_to_global() {
    let n = 1usize;
    let q_obj = CscMatrix::new(n, n);
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let b = vec![-0.1];
    let bounds = vec![(-2e5, 2e5)];
    let mut qc = QcqpMatrix::new(n);
    qc.triplets.push((0, 0, -2e-10));
    let mut problem = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();
    problem.set_quadratic_constraints(vec![qc]).unwrap();
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    assert!(
        (result.objective - (-2e5)).abs() < 1.0,
        "objective={}",
        result.objective
    );
    assert!(
        (result.solution[0] - (-2e5)).abs() < 1.0,
        "x={:?}",
        result.solution
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
}

/// min x  s.t.  (1/2)*(-2e-10)*x^2 <= 1 (inactive),  x in [-1, 1].
///
/// The jitter-band matrix is clamped to a PSD approximation under which the
/// SOCP solves cleanly to Optimal (same answer here, x = -1) — but the clamp
/// means nothing about the original problem is proven, so the route must go
/// global regardless of the clean status. This closes the false-Infeasible
/// window: a clamped reformulation could equally produce a certified-looking
/// Infeasible/Unbounded for the wrong problem.
///
/// Sentinel: dropping the `convexity_unproven` check from
/// `is_clean_convex_outcome` makes this FAIL with route=ConicQcqpConvex.
#[test]
fn clamped_cholesky_forces_global_route_even_on_clean_status() {
    let n = 1usize;
    let q_obj = CscMatrix::new(n, n);
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let b = vec![1.0];
    let bounds = vec![(-1.0, 1.0)];
    let mut qc = QcqpMatrix::new(n);
    qc.triplets.push((0, 0, -2e-10));
    let mut problem = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();
    problem.set_quadratic_constraints(vec![qc]).unwrap();
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    assert!(
        (result.objective - (-1.0)).abs() < 1e-4,
        "objective={}",
        result.objective
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
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

/// Nonconvex-path timeout propagation: an expired deadline must surface as
/// `Timeout` from the spatial branch-and-bound (checked per node), not as a
/// false `Infeasible` or a completed solve.
///
/// Sentinel: dropping the deadline check in `nonconvex::global_core` (or the
/// deadline mapping in `conic_options`) makes this FAIL with `Optimal`.
#[test]
fn nonconvex_qcqp_propagates_timeout() {
    let problem = nonconvex_qcqp_problem(3.0);
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout, "{:?}", result.status);
    assert!(result.stats.deadline_triggered);
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
}

/// memo 31 (P1): the QCQP conic route must include `QpProblem::obj_offset`
/// in the reported objective, like every LP/QP route (QPLIB QCQP `q0` flows
/// through this field).
///
/// Sentinel: dropping the offset addition in `solve_qcqp_via_conic` makes
/// this FAIL (objective would be -sqrt(2), not 42.5 - sqrt(2)).
#[test]
fn convex_qcqp_objective_includes_obj_offset() {
    let mut problem = convex_qcqp_problem();
    problem.obj_offset = 42.5;
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    let expect = 42.5 - 2.0_f64.sqrt();
    assert!(
        (result.objective - expect).abs() < 1e-4,
        "objective={} expected {expect}",
        result.objective
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpConvex);
}

/// memo 31, nonconvex fallback path: the spatial global route must also
/// include `obj_offset`.
#[test]
fn nonconvex_qcqp_objective_includes_obj_offset() {
    let mut problem = nonconvex_qcqp_problem(3.0);
    problem.obj_offset = -7.25;
    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
    let expect = 2.0 - 7.25;
    assert!(
        (result.objective - expect).abs() < 5e-3,
        "objective={} expected {expect}",
        result.objective
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
}

/// Codex finding (qcqp_route.rs:115): the McCormick fallback must honor
/// `options.global_optimization` (node budget / gap), not silently use
/// `GlobalOptions::default()` (max_nodes = 50_000).
///
/// Sentinel: with `max_nodes = 1` the search is node-limited after the root,
/// so the status must be `MaxIterations`; reverting to the hardcoded default
/// makes this FAIL with `Optimal`.
#[test]
fn nonconvex_qcqp_honors_global_optimization_node_budget() {
    let problem = nonconvex_qcqp_problem(3.0);
    let opts = SolverOptions {
        global_optimization: Some(crate::options::GlobalOptimizationConfig {
            max_nodes: 1,
            ..Default::default()
        }),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(
        result.status,
        SolveStatus::MaxIterations,
        "{:?}",
        result.status
    );
    assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
}
