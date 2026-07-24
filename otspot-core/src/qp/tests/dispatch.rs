use super::super::*;
use crate::problem::{SolveRoute, SolveStatus};
use crate::sparse::CscMatrix;

// ── Bug B: is_zero_q must use structural (nnz == 0) check, not numerical threshold ──

/// is_zero_q must return false for a matrix with a tiny stored value (1e-13).
///
/// Sentinel (no-op): if is_zero_q uses a threshold >= 1e-13, it returns true and
/// this assert fires.
#[test]
fn is_zero_q_tiny_nonzero_returns_false() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[1e-13], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    assert!(
        !problem.is_zero_q(),
        "is_zero_q must return false for a matrix with stored value 1e-13"
    );
}

/// is_zero_q must return true for a structurally empty CscMatrix (CscMatrix::new).
///
/// Sentinel (no-op): if is_zero_q returns false for empty matrices, LP dispatch breaks.
#[test]
fn is_zero_q_structural_empty_returns_true() {
    let n = 3;
    let q = CscMatrix::new(n, n);
    let c = vec![0.0; n];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    assert!(
        problem.is_zero_q(),
        "is_zero_q must return true for a structurally empty Q"
    );
}

/// Tiny but non-zero diagonal Q (below the old 1e-12 threshold) must route to QP,
/// not to the LP solver.
///
/// Sentinel: if is_zero_q uses a numerical threshold >= 1e-13, the problem is
/// misrouted to LP (route = LpForwardedFromQp) instead of QP (route = QpIpm).
///
/// Problem: min (1/2) * 2e-13 * x^2 + 2e-11 * x,  x ∈ (-∞, ∞)
/// x* = -2e-11 / (2e-13) = -100,  obj* ≈ -1e-9
/// LP (Q ignored): min 2e-11 * x → Unbounded
///
/// c is scaled so x* = -100 is numerically tractable for the IPM.
#[test]
fn tiny_nonzero_q_routes_to_qp_not_lp() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2e-13], 1, 1).unwrap();
    let c = vec![2e-11]; // x* = -c/Q = -100
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(
        result.stats.route,
        SolveRoute::QpIpm,
        "tiny non-zero Q must dispatch to QpIpm, not LP; route={:?}, status={:?}",
        result.stats.route,
        result.status
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "tiny non-zero Q (x*=-100) must be Optimal via QP solver; got {:?}",
        result.status
    );
}

/// Structurally zero Q (CscMatrix::new — no stored entries) must route via
/// LpForwardedFromQp, not QpIpm.
///
/// Sentinel: if is_zero_q returns false for an empty CscMatrix, the problem
/// takes the IPM path and route becomes QpIpm.
#[test]
fn structural_zero_q_routes_to_lp() {
    let n = 2;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 2.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![4.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "LP (Q=0) must be Optimal"
    );
    assert_eq!(
        result.stats.route,
        SolveRoute::LpForwardedFromQp,
        "structural-zero Q must use LpForwardedFromQp route, got {:?}",
        result.stats.route
    );
}

// ── Bug G: LpProblem::new_general failure must return NumericalError, not Infeasible ──

/// solve_as_lp with a QpProblem whose bounds have been invalidated post-construction
/// (NaN injected after bypassing QpProblem::new validation) must return NumericalError,
/// NOT Infeasible.
///
/// Bug G: the old Err(_) arm returned infeasible(), conflating a conversion/input error
/// with a mathematical infeasibility certificate. route was also left unset (Unknown).
///
/// Sentinel: reverting the Err(_) arm to return infeasible() causes the NumericalError
/// assert to fail. Reverting the route assignment causes the SolveRoute assert to fail.
#[test]
fn lp_conversion_error_returns_numerical_error_not_infeasible() {
    use crate::options::SolverOptions;
    use crate::qp::solve_as_lp;

    // Structural-zero Q so the dispatch calls solve_as_lp.
    let mut problem = QpProblem::new(
        CscMatrix::new(1, 1),
        vec![1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(0.0, 1.0)],
        vec![],
    )
    .unwrap();

    // Tamper: inject NaN into bounds post-construction.
    // QpProblem::new validated these; direct field mutation bypasses the guard.
    // LpProblem::new_general will fail with InvalidBounds.
    problem.bounds[0] = (f64::NAN, 1.0);

    let result = solve_as_lp(&problem, &SolverOptions::default());
    assert_eq!(
        result.status,
        SolveStatus::NumericalError,
        "LpProblem conversion failure must return NumericalError, not {:?}",
        result.status
    );
    assert_eq!(
        result.stats.route,
        SolveRoute::LpForwardedFromQp,
        "route must be LpForwardedFromQp on conversion error, got {:?}",
        result.stats.route
    );
}

/// Sentinel: a QP with nonzero Q and an empty variable box (lb > ub) must solve
/// to Infeasible via the IPM path, with presolve either ON or OFF.
///
/// This is the crux gap: with presolve OFF the IPM never sees a bound-consistency
/// check, and its initial point (`(lb+ub)/2`, then bound clamps) assumes `lb <= ub`.
/// The `first_infeasible_bound` guard in `dispatch_solve_qp` must intercept before
/// the IPM runs. Reverting the guard does NOT panic — the clamp margins keep the
/// same sign as the (negative) range, so `clamp` is never handed `min > max` — but
/// the IPM iterates on the empty box and terminates with a silently WRONG
/// `SolveStatus::Stalled` (empirically verified). Detecting that wrong status is
/// exactly what makes this sentinel load-bearing.
#[test]
fn qp_ipm_empty_box_lb_gt_ub_is_infeasible() {
    use crate::options::SolverOptions;
    // min 1/2 x^2  s.t.  (no constraints),  x ∈ [5, 3]  (empty box). Q≠0 → IPM.
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(5.0, 3.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds)
        .expect("lb>ub box must be ACCEPTED at construction");
    assert!(
        !problem.is_zero_q(),
        "Q must be nonzero to route to the IPM"
    );
    for presolve in [true, false] {
        let opts = SolverOptions {
            presolve,
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "QP-IPM empty box must be Infeasible (presolve={presolve}), got {:?}",
            result.status
        );
        assert_eq!(result.stats.route, SolveRoute::QpIpm);
    }
}
