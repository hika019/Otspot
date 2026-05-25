use super::super::*;
use super::{assert_close, EPS};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;
use crate::test_kkt::assert_solver_invariants_qp;


/// 大行ノルム制約での Ruiz scaling 耐性 (元空間で pfeas 評価)。
#[test]
fn test_presolve_pfeas_large_row_norm() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0], 1, 1).unwrap();
    let b = vec![500.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
    let ax = problem.a.mat_vec_mul(&result.solution).unwrap();
    let pfeas = ax
        .iter()
        .zip(problem.b.iter())
        .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
        .fold(0.0_f64, f64::max);
    let norm_b = problem
        .b
        .iter()
        .fold(0.0_f64, |a, &bi| a.max(bi.abs()))
        .max(1.0);
    let eps = opts.ipm_eps();
    assert!(pfeas < eps * (1.0 + norm_b), "pfeas={pfeas:.2e}");
}

/// bounds 付き問題で post-postsolve bfeas check が誤降格しないこと。
#[test]
fn test_presolve_bfeas_bounded_problem() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
    let x = result.solution[0];
    assert!(x >= -1e-4, "x >= lb=0, got {x}");
    assert!(x <= 1.0 + 1e-4, "x <= ub=1, got {x}");
}

/// 正常解で post-postsolve pfeas+bfeas check が Optimal を維持。
#[test]
fn test_presolve_pfeas_bfeas_ok() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0_f64, 0.5_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
}

/// presolve=true で post-unscaling check が正常問題に影響しないこと。
#[test]
fn test_solve_qp_with_presolve_path_verified() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    assert!(opts.presolve);
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
    let eps = 1e-3_f64;
    assert!((result.solution[0] - 0.5).abs() < eps);
    assert!((result.solution[1] - 0.5).abs() < eps);
}

/// Eq 制約 presolve ON/OFF で解一致。
#[test]
fn test_presolve_qp_eq_on_off_consistency() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

    let opts_on = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let mut opts_off = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts_off.presolve = false;

    let result_on = solve_qp_with(&problem, &opts_on);
    let result_off = solve_qp_with(&problem, &opts_off);

    assert_eq!(result_on.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result_on, &problem);
    assert_eq!(result_off.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result_off, &problem);
    assert!((result_on.solution[0] - result_off.solution[0]).abs() < 1e-4);
    assert!((result_on.solution[1] - result_off.solution[1]).abs() < 1e-4);
}

/// Box 制約 presolve ON/OFF で解一致。
#[test]
fn test_presolve_qp_box_on_off_consistency() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 2.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts_on = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let mut opts_off = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts_off.presolve = false;

    let result_on = solve_qp_with(&problem, &opts_on);
    let result_off = solve_qp_with(&problem, &opts_off);

    assert_eq!(result_on.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result_on, &problem);
    assert_eq!(result_off.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result_off, &problem);
    assert_close(result_on.solution[0], 0.0, EPS, "ON x[0]");
    assert_close(result_on.solution[1], 0.0, EPS, "ON x[1]");
    assert_close(result_off.solution[0], 0.0, EPS, "OFF x[0]");
    assert_close(result_off.solution[1], 0.0, EPS, "OFF x[1]");
}

/// Ge 制約 + presolve ON。
#[test]
fn test_qp_ge_constraint_with_presolve() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Mixed (Ge+Le) presolve=false (mixed presolve バグ既知)。
#[test]
fn test_qp_mixed_ge_with_presolve() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![0.5, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();

    let mut opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts.presolve = false;
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result, &problem);
    assert_close(result.solution[0], 0.25, EPS, "x[0]");
    assert_close(result.solution[1], 0.25, EPS, "x[1]");
}

/// Mixed (Ge+Le) presolve=ON + Ruiz=ON: pfeas Ge 違反検出 regression。
#[test]
fn test_qp_mixed_ge_le_presolve_ruiz_regression() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![0.5, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_solver_invariants_qp(&result, &problem);
    assert_close(result.solution[0], 0.25, EPS, "x[0]");
    assert_close(result.solution[1], 0.25, EPS, "x[1]");
    let pfeas = {
        let x = &result.solution;
        let ge_viol = (0.5_f64 - (x[0] + x[1])).max(0.0);
        let le_viol = (x[0] - x[1] - 1.0_f64).max(0.0);
        ge_viol.max(le_viol)
    };
    assert!(pfeas < 1e-6, "pfeas={:e}", pfeas);

    let opts_no_presolve = SolverOptions {
        timeout_secs: Some(10.0),
        presolve: false,
        ..Default::default()
    };
    let result_no_presolve = solve_qp_with(&problem, &opts_no_presolve);
    assert_eq!(result_no_presolve.status, SolveStatus::Optimal);
    assert_solver_invariants_qp(&result_no_presolve, &problem);
    assert_close(result_no_presolve.solution[0], 0.25, EPS, "no-presolve x[0]");
    assert_close(result_no_presolve.solution[1], 0.25, EPS, "no-presolve x[1]");
}
