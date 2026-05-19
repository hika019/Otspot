use super::super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// BD-T1: baseline (presolve OFF, 全変数 box) → bound_duals.len()=4。
#[test]
fn test_bd_t1_baseline_presolve_off() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 5.0_f64); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0]).abs() < sol_tol);
    assert!((result.solution[1]).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 4);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1] > tol);
    assert!(result.bound_duals[2].abs() < tol);
    assert!(result.bound_duals[3].abs() < tol);
}

/// BD-T2: FixedVar + bound_duals リマップ (z 除去 → bound_duals.len()=6, lb_x≠lb_y で順序検証)。
#[test]
fn test_bd_t2_fixed_var_remap_core() {
    let n = 3usize;
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
    let c = vec![2.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 5.0_f64), (3.0_f64, 3.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 5e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0]).abs() < sol_tol);
    assert!((result.solution[1]).abs() < sol_tol);
    assert!((result.solution[2] - 3.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 6);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1] > tol);
    // lb_x ≠ lb_y で変数順序バグを検出。
    assert!((result.bound_duals[0] - result.bound_duals[1]).abs() > tol);
    assert!((result.bound_duals[2]).abs() < tol);
    assert!(result.bound_duals[3].abs() < 5e-3);
    assert!(result.bound_duals[4].abs() < 5e-3);
    assert!((result.bound_duals[5]).abs() < tol);
    let dual = if result.dual_solution.is_empty() {
        0.0
    } else {
        result.dual_solution[0]
    };
    let kkt_x = 2.0 - dual - result.bound_duals[0] + result.bound_duals[3];
    assert!(kkt_x.abs() < 1e-3);
    let kkt_y = 1.0 - dual - result.bound_duals[1] + result.bound_duals[4];
    assert!(kkt_y.abs() < 1e-3);
}

/// BD-T3: FixedVar + lb_only 変数 → bound_duals.len()=3。
#[test]
fn test_bd_t3_fixed_var_lb_only() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, f64::INFINITY), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.bound_duals.len(), 3);
}

/// BD-T4: EmptyCol の bound_duals を KKT で復元 (refit_bound_duals_kkt が 0 埋めを修復)。
#[test]
fn test_bd_t4_empty_col_kkt_recovered() {
    let n = 3usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![-1.0, -1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![4.0];
    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0_f64, 3.0_f64),
    ];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.bound_duals.len(), 2);
    let z_lb = result.bound_duals[0];
    let z_ub = result.bound_duals[1];
    assert!((z_lb - 1.0).abs() < 1e-3, "z_lb={z_lb}");
    assert!(z_ub.abs() < 1e-3, "z_ub={z_ub}");
}

/// 全変数 ±∞ → bound_duals 空。
#[test]
fn test_bd_t5_unbounded_vars_empty() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(result.bound_duals.is_empty());
}

/// BD-T6: FixedVar + ub 活性変数 (ub_dual 非ゼロ × presolve 残存)。
#[test]
fn test_bd_t6_ub_active_with_presolve() {
    let n = 3usize;
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
    let c = vec![-1.0, -1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 3.0_f64), (0.0_f64, 5.0_f64), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0] - 3.0).abs() < sol_tol);
    assert!((result.solution[1] - 5.0).abs() < sol_tol);
    assert!((result.solution[2] - 2.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 6);
    assert!(result.bound_duals[0].abs() < tol);
    assert!(result.bound_duals[1].abs() < tol);
    assert!((result.bound_duals[2]).abs() < tol);
    assert!(result.bound_duals[3] > tol);
    assert!(result.bound_duals[4] > tol);
    assert!((result.bound_duals[5]).abs() < tol);
}

/// BD-T7: constraint active × lb_dual nonzero × KKT 照合 (x*=2, y*=1)。
#[test]
fn test_bd_t7_constraint_active_lb_dual_nonzero_kkt() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
    let b = vec![-3.0];
    let bounds = vec![(2.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0] - 2.0).abs() < sol_tol);
    assert!((result.solution[1] - 1.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 2);
    let dual = if result.dual_solution.is_empty() {
        0.0
    } else {
        result.dual_solution[0]
    };
    assert!(dual > tol);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1].abs() < tol);
    let kkt_x = result.solution[0] - dual - result.bound_duals[0];
    assert!(kkt_x.abs() < 1e-3);
    let kkt_y = result.solution[1] - dual - result.bound_duals[1];
    assert!(kkt_y.abs() < 1e-3);
}
