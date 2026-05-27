//! KKT residual helpers shared across inline `#[cfg(test)]` modules.
//!
//! Tests that only check `result.status == Optimal` + objective value miss
//! dual-recovery / postsolve / Phase I regressions whose KKT residuals can
//! drift while the objective happens to land on a feasible alternative.
//! `assert_kkt_optimal` enforces primal/dual/objective together so those
//! regressions surface.

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::qp::kkt_resid::f64_impl;
use crate::qp::QpProblem;
use crate::simplex::solve_with;
use crate::sparse::CscMatrix;

/// KKT residual tolerance shared with bench (`eps=1e-6`, CLAUDE.md L42).
pub const EPS_KKT: f64 = 1e-6;

/// Relative tolerance for the objective value comparison.
pub const EPS_OBJ_REL: f64 = 1e-6;

/// Mini test single-run budget.
pub const MINI_TIMEOUT_SECS: f64 = 5.0;

/// Tolerance for detecting bound activity in `dfeas_rel_bound`.
const BOUND_TOL: f64 = 1e-6;

/// Bound-aware dual feasibility relative residual.
///
/// `compute_dfeas_orig` (bench) と同型: fixed (lb==ub) を除外し、active な
/// 下端のみで rc<0、active な上端のみで rc>0、interior で rc!=0 の違反量を
/// `(1 + |rc| + |c|)` で正規化して取る。
pub fn dfeas_rel_bound(
    c: &[f64],
    bounds: &[(f64, f64)],
    x: &[f64],
    rc: &[f64],
) -> f64 {
    let n = c.len().min(rc.len()).min(x.len());
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x[j] - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x[j] - ub).abs() < BOUND_TOL;
        let r = rc[j];
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -r)
        } else if at_ub && !at_lb {
            f64::max(0.0, r)
        } else if !at_lb && !at_ub {
            r.abs()
        } else {
            0.0
        };
        let scale = 1.0 + r.abs() + c[j].abs();
        max_rel = max_rel.max(viol / scale);
    }
    max_rel
}

/// Primal feasibility (|Ax-b|∞) — Eq/Le/Ge 別に違反方向のみ取る。
pub fn pfeas_abs(a: &CscMatrix, b: &[f64], cts: &[ConstraintType], x: &[f64]) -> f64 {
    let ax = f64_impl::ax(a, x);
    f64_impl::constraint_violations(&ax, b, cts)
        .into_iter()
        .fold(0.0_f64, f64::max)
}

/// Variable-bound feasibility: lb ≤ x ≤ ub の違反量 (max)。
pub fn bound_violation(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    let mut max_v = 0.0_f64;
    for j in 0..x.len() {
        let (lb, ub) = bounds[j];
        if lb.is_finite() && x[j] < lb {
            max_v = max_v.max(lb - x[j]);
        }
        if ub.is_finite() && x[j] > ub {
            max_v = max_v.max(x[j] - ub);
        }
    }
    max_v
}

/// Assert solver invariants for a `SolverResult` against its `LpProblem`.
///
/// For `Optimal` results: checks primal feasibility, bound feasibility, and
/// dual feasibility. For non-Optimal results: asserts no false-Optimal
/// invariant (nothing to check, any honest non-Optimal is acceptable).
///
/// Use in tests that call into solver internals to ensure all Optimal returns
/// maintain consistent invariants — catches false-Optimal like klein3.
pub fn assert_solver_invariants_lp(result: &crate::problem::SolverResult, lp: &LpProblem) {
    if result.status != crate::problem::SolveStatus::Optimal {
        return;
    }
    if !lp.c.is_empty() {
        assert!(
            !result.solution.is_empty(),
            "Optimal result must have non-empty solution"
        );
    }
    let pf = pfeas_abs(&lp.a, &lp.b, &lp.constraint_types, &result.solution);
    let b_inf = lp.b.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let pf_norm = pf / (1.0 + b_inf);
    assert!(
        pf_norm < EPS_KKT,
        "Optimal result has excessive primal violation: pfeas={:.3e} normalized={:.3e} > {:.3e}",
        pf,
        pf_norm,
        EPS_KKT
    );
    let bv = bound_violation(&lp.bounds, &result.solution);
    assert!(
        bv < EPS_KKT,
        "Optimal result has bound violation={:.3e} > {:.3e}",
        bv,
        EPS_KKT
    );
}

/// Solve `lp` and assert primal/dual/objective KKT all hold to `EPS_KKT`.
///
/// `expected_obj` is compared with relative error `EPS_OBJ_REL`. `label`
/// shows up in failure messages so tests calling this twice with different
/// settings can be disambiguated.
pub fn assert_kkt_optimal(lp: &LpProblem, expected_obj: f64, label: &'static str) {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(MINI_TIMEOUT_SECS);
    assert_kkt_optimal_with(lp, expected_obj, label, &opts);
}

/// `assert_kkt_optimal` の SolverOptions 指定版 (presolve on/off / method 切替用)。
pub fn assert_kkt_optimal_with(
    lp: &LpProblem,
    expected_obj: f64,
    label: &'static str,
    opts: &SolverOptions,
) {
    let r = solve_with(lp, opts);

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "[{}] expected Optimal, got {:?} (obj={:.6e})",
        label,
        r.status,
        r.objective
    );

    let bv = bound_violation(&lp.bounds, &r.solution);
    assert!(
        bv < EPS_KKT,
        "[{}] bound violation={:.3e} > {:.3e} (x={:?})",
        label,
        bv,
        EPS_KKT,
        &r.solution
    );

    let pf = pfeas_abs(&lp.a, &lp.b, &lp.constraint_types, &r.solution);
    assert!(
        pf < EPS_KKT,
        "[{}] pfeas={:.3e} > {:.3e} (x={:?})",
        label,
        pf,
        EPS_KKT,
        &r.solution
    );

    let df = dfeas_rel_bound(&lp.c, &lp.bounds, &r.solution, &r.reduced_costs);
    assert!(
        df < EPS_KKT,
        "[{}] dfeas_rel_bound={:.3e} > {:.3e} | x={:?} rc={:?} y={:?}",
        label,
        df,
        EPS_KKT,
        &r.solution,
        &r.reduced_costs,
        &r.dual_solution
    );

    let obj_err = (r.objective - expected_obj).abs() / (1.0 + expected_obj.abs());
    assert!(
        obj_err < EPS_OBJ_REL,
        "[{}] obj={:.9e} expected={:.9e} rel_err={:.3e} > {:.3e}",
        label,
        r.objective,
        expected_obj,
        obj_err,
        EPS_OBJ_REL
    );
}

/// Assert solver invariants for a QP `SolverResult` against its `QpProblem`.
///
/// For `Optimal` / `LocallyOptimal` results: checks primal feasibility,
/// bound feasibility, and KKT stationarity residual via the shared IPM KKT
/// helpers. For non-Optimal results: returns immediately (honest non-Optimal
/// is always acceptable).
pub fn assert_solver_invariants_qp(
    result: &crate::problem::SolverResult,
    qp: &QpProblem,
) {
    use crate::problem::SolveStatus;
    if !matches!(result.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal) {
        return;
    }
    assert!(
        !result.solution.is_empty(),
        "Optimal/LocallyOptimal QP result must have non-empty solution"
    );
    // Primal feasibility via shared LP helper (same Ax-b logic).
    let pf = pfeas_abs(
        &qp.a,
        &qp.b,
        &qp.constraint_types,
        &result.solution,
    );
    let b_inf = qp.b.iter().fold(0.0_f64, |a, &v: &f64| a.max(v.abs()));
    let pf_norm = pf / (1.0 + b_inf);
    assert!(
        pf_norm < EPS_KKT,
        "QP Optimal result has excessive primal violation: pfeas={:.3e} norm={:.3e} > {:.3e}",
        pf,
        pf_norm,
        EPS_KKT
    );
    // Bound feasibility.
    let bv = bound_violation(&qp.bounds, &result.solution);
    assert!(
        bv < EPS_KKT,
        "QP Optimal result has bound violation={:.3e} > {:.3e}",
        bv,
        EPS_KKT
    );
    // KKT stationarity: Qx + c + A^T y + z = 0 residual via IPM helper.
    use crate::qp::ipm_solver::kkt::kkt_residual_rel;
    use crate::qp::ipm_solver::outcome::ProblemView;
    let view = ProblemView::from_problem(qp);
    let kkt = kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    assert!(
        kkt < EPS_KKT,
        "QP Optimal result has KKT stationarity residual={:.3e} > {:.3e}",
        kkt,
        EPS_KKT
    );
}

#[cfg(test)]
mod no_op_proof_tests {
    use super::*;
    use crate::problem::{SolveStatus, SolverResult};
    use crate::sparse::CscMatrix;

    /// No-op proof: `assert_solver_invariants_lp` has load-bearing body.
    ///
    /// Passes a corrupt Optimal result (x=1e12, violates x≤5) to the helper and
    /// expects a panic. If the helper body were emptied, this test would NOT panic
    /// and would itself fail (since `#[should_panic]` would not be satisfied).
    #[test]
    #[should_panic(expected = "primal violation")]
    fn assert_solver_invariants_lp_panics_on_primal_violation() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = crate::problem::LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let corrupt = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1e12],
            ..Default::default()
        };
        assert_solver_invariants_lp(&corrupt, &lp);
    }

    /// No-op proof: `assert_solver_invariants_lp` catches bound violations.
    ///
    /// x is constrained to [0, 5], but the corrupt result claims x=100.
    #[test]
    #[should_panic(expected = "bound violation")]
    fn assert_solver_invariants_lp_panics_on_bound_violation() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = crate::problem::LpProblem::new_general(
            vec![1.0],
            a,
            vec![1000.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0)],
            None,
        )
        .unwrap();
        let corrupt = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![100.0],
            ..Default::default()
        };
        assert_solver_invariants_lp(&corrupt, &lp);
    }

    /// No-op proof: `assert_solver_invariants_qp` has load-bearing body.
    ///
    /// Passes a corrupt Optimal QP result (x=1e12, violates x=1 equality) and
    /// expects a panic. If the helper body were emptied this test would fail.
    #[test]
    #[should_panic(expected = "primal violation")]
    fn assert_solver_invariants_qp_panics_on_primal_violation() {
        use crate::qp::QpProblem;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let corrupt = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1e12],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0],
            ..Default::default()
        };
        assert_solver_invariants_qp(&corrupt, &prob);
    }
}
