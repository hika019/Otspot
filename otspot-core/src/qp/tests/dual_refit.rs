use super::super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// REFIT-T1: lb 活性 + c>0 で y_lb = c を復元。
#[test]
fn test_refit_bound_duals_lb_only_active() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.5_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![],
        bound_duals: vec![0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);
    assert!(
        (result.bound_duals[0] - 2.5).abs() < 1e-9,
        "got {}",
        result.bound_duals[0]
    );
}

/// REFIT-T2: ub 活性 + c<0 で y_ub = -c。
#[test]
fn test_refit_bound_duals_ub_only_active() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-3.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, 5.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![5.0],
        dual_solution: vec![],
        bound_duals: vec![0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);
    assert!(
        (result.bound_duals[0] - 3.0).abs() < 1e-9,
        "got {}",
        result.bound_duals[0]
    );
}

/// REFIT-T3: 内点では y_lb=y_ub=0 維持。
#[test]
fn test_refit_bound_duals_interior_keeps_zero() {
    let n = 1usize;
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
    let c = vec![-4.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, 5.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![2.0],
        dual_solution: vec![],
        bound_duals: vec![0.0, 0.0],
        objective: -4.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);
    assert!(result.bound_duals[0].abs() < 1e-9);
    assert!(result.bound_duals[1].abs() < 1e-9);
}

/// REFIT-T4: KKT-guard が改善なし更新を revert (既値維持)。
#[test]
fn test_refit_bound_duals_kkt_guard_no_regression() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![],
        bound_duals: vec![2.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);
    assert!(
        (result.bound_duals[0] - 2.0).abs() < 1e-9,
        "got {}",
        result.bound_duals[0]
    );
}

/// REFIT-T5: 制約あり (A^T y 非ゼロ) で bound_dual 計算。
#[test]
fn test_refit_bound_duals_with_constraint() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![5.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0, 0.0],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0, 0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);
    assert!((result.bound_duals[0] - 1.0).abs() < 1e-9);
    assert!(result.bound_duals[1].abs() < 1e-9);
}

/// 不可能な正 Le dual を singleton column interval {0} に projection。
#[test]
fn test_project_duals_from_singleton_columns_clamps_infeasible_positive_le_dual() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![0.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0_f64, 1.0], 1, n).unwrap();
    let b = vec![0.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0, 0.0],
        dual_solution: vec![5.0],
        bound_duals: vec![0.0, 0.0],
        ..SolverResult::default()
    };

    project_duals_from_singleton_columns(&problem, &mut result);
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);

    assert!(result.dual_solution[0].abs() < 1e-12);
    assert!(result.bound_duals.iter().all(|v| v.abs() < 1e-12));
}

/// lb-only singleton column の lower bound から y を必要値まで引き上げ。
#[test]
fn test_project_duals_from_singleton_columns_respects_lb_only_lower_bound() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-2.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
    let b = vec![0.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    project_duals_from_singleton_columns(&problem, &mut result);
    refit_bound_duals_kkt(&problem, &mut result, 1e-6);

    assert!((result.dual_solution[0] - 2.0).abs() < 1e-12);
    assert!(result.bound_duals[0].abs() < 1e-12);
}

#[test]
fn test_zero_inactive_inequality_duals_clears_slack_le_rows() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![0.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
    let b = vec![10.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![3.0],
        dual_solution: vec![7.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    zero_inactive_inequality_duals(&problem, &mut result);

    assert!(result.dual_solution[0].abs() < 1e-12);
}

#[test]
fn test_refine_dual_projected_gradient_uses_curvature_scaled_step() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-1.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0_f64], 1, n).unwrap();
    let b = vec![1.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0e-3],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    refine_dual_projected_gradient(&problem, &mut result, &[], None);

    assert!((result.dual_solution[0] - 1.0e-3).abs() < 1e-9);
}

#[test]
fn test_refine_dual_worst_active_block_updates_row_and_bound_duals_together() {
    let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
    let c = vec![-1.0_f64, 0.0_f64];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
    let b = vec![1.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &[],
    };
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0_f64, 0.0_f64],
        dual_solution: vec![0.0_f64],
        bound_duals: vec![0.0_f64, 0.0_f64],
        ..SolverResult::default()
    };

    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    refine_dual_worst_active_block(&problem, &mut result, &[], None);
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    assert!(post < pre);
    assert!(post < 1e-12);
    assert!((result.dual_solution[0] - 1.0).abs() < 1e-9);
    assert!(result.bound_duals[0].abs() < 1e-12);
    assert!((result.bound_duals[1] - 1.0).abs() < 1e-9);
}

/// REFIT-T-COMPCONS-NEAR: comp-consistent criterion assigns bound dual for a
/// near-active variable (GOULDQP2-like: x interior but comp_candidate << comp_tol).
///
/// Setup: box-bounded j with x=0.998, lb=0, ub=1, c=-1e-6 (Q=0, no constraints).
/// stationarity target = -(Qx + c) = 1e-6 > 0 → z_ub candidate = 1e-6.
/// comp_candidate = 1e-6 * (1-0.998) / (1 + 1e-6 * (0.998+1)) ≈ 2e-9 < comp_tol=1e-6.
/// → z_ub MUST be assigned.
///
/// Sentinel: reverting to the old `DUAL_RECOVERY_ACTIVE_TOL_REL=1e-8` activity check
/// causes this test to FAIL (rel_gap ≈ 6.7e-4 >> 1e-8 → z_ub stays 0 → assertion fails).
#[test]
fn test_refit_bound_duals_comp_consistent_near_active_assigns_dual() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    // c = -1e-6: target = -(Qx + c) = 1e-6 > 0, z_ub candidate
    let c = vec![-1e-6_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    // box-bounded: lb=0, ub=1, x=0.998 (interior by 0.002, rel≈6.7e-4)
    let bounds = vec![(0.0_f64, 1.0_f64)];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let mut result = crate::problem::SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.998_f64],
        dual_solution: vec![],
        // layout: [z_lb, z_ub] (n_lb=1 finite, n_ub=1 finite)
        bound_duals: vec![0.0_f64, 0.0_f64],
        ..crate::problem::SolverResult::default()
    };

    refit_bound_duals_kkt(&problem, &mut result, 1e-6);

    // z_ub should be ≈ 1e-6 (absorbs stationarity; comp ≈ 2e-9 < 1e-6)
    let z_ub = result.bound_duals[1];
    assert!(
        (z_ub - 1e-6).abs() < 1e-10,
        "near-active box variable must have z_ub ≈ 1e-6, got z_ub={z_ub:.3e}; \
         reverting to DUAL_RECOVERY_ACTIVE_TOL_REL=1e-8 activity check causes z_ub=0 here"
    );
    // z_lb must stay 0 (x not near lb)
    assert!(result.bound_duals[0].abs() < 1e-12);
}

/// REFIT-T-COMPCONS-FAR: comp-consistent criterion leaves bound dual at 0 for a
/// far-interior variable (QFORPLAN-like: comp_candidate >> comp_tol).
///
/// Setup: box-bounded j with x=0.5, lb=0, ub=1, c=-1.0 (Q=0, no constraints).
/// stationarity target = 1.0, comp_candidate = 1.0*0.5/(1+1.0*1.5) = 0.2 >> 1e-6.
/// → z_ub must NOT be assigned (spurious assignment would violate comp check).
///
/// Sentinel: reverting to the old pre-QFORPLAN-fix behavior (always assign for box
/// bounds regardless of comp) causes bound_duals[1] = 1.0 ≠ 0 → assertion fails.
#[test]
fn test_refit_bound_duals_comp_consistent_far_interior_keeps_zero() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    // c = -1.0: target = 1.0 (large), comp_candidate ≈ 0.2 >> comp_tol=1e-6
    let c = vec![-1.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64)];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let mut result = crate::problem::SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.5_f64],
        dual_solution: vec![],
        bound_duals: vec![0.0_f64, 0.0_f64],
        ..crate::problem::SolverResult::default()
    };

    refit_bound_duals_kkt(&problem, &mut result, 1e-6);

    // z_ub must stay 0 (far interior; assigning would give comp ≈ 0.2 >> 1e-6)
    let z_ub = result.bound_duals[1];
    assert!(
        z_ub.abs() < 1e-12,
        "far-interior box variable must not receive z_ub (comp would be ≈0.2>>1e-6), got {z_ub:.3e}; \
         reverting to always-assign (pre-QFORPLAN-fix) behavior causes z_ub=1.0 here"
    );
}
