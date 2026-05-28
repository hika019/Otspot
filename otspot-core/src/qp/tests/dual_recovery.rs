use super::super::*;
use crate::problem::SolveStatus;
use crate::qp::postsolve::postprocess::{run_dual_recovery_postprocess, try_dual_only_ir};
use crate::sparse::CscMatrix;

#[test]
fn test_dual_recovery_postprocess_can_improve_without_dual_ir() {
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
    let post = run_dual_recovery_postprocess(&problem, &view, &mut result, None);

    assert!(post < pre);
    assert!(post < 1e-12);
}

#[test]
fn test_dual_only_ir_uses_active_rows_and_keeps_inactive_le_zero() {
    let q = CscMatrix::new(1, 1);
    let c = vec![-1.0_f64];
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0_f64, 1.0_f64], 2, 1).unwrap();
    let b = vec![1.0_f64, 10.0_f64];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![
            crate::problem::ConstraintType::Eq,
            crate::problem::ConstraintType::Le,
        ],
    )
    .unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0_f64],
        dual_solution: vec![0.0_f64, 0.0_f64],
        bound_duals: vec![],
        ..SolverResult::default()
    };

    let accepted = try_dual_only_ir(&problem, &mut result, &[], 1e-8, None);

    assert!(accepted > 0);
    assert!((result.dual_solution[0] - 1.0).abs() < 1e-9);
    assert!(result.dual_solution[1].abs() < 1e-12);
}

#[test]
fn test_dual_only_ir_couples_row_and_bound_duals() {
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
    let accepted = try_dual_only_ir(&problem, &mut result, &[], 1e-8, None);
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    assert!(accepted > 0);
    assert!(post < pre);
    assert!((result.dual_solution[0] - 1.0).abs() < 1e-6);
    assert!((result.bound_duals[1] - 1.0).abs() < 1e-6);
}

/// 加重 Gram (1/scale²) が componentwise 最悪 j を優先削減 (無加重では r_rel 悪化)。
#[test]
fn test_dual_only_ir_weighted_gram_prioritizes_worst_component() {
    let q = CscMatrix::from_triplets(&[1], &[1], &[1.0_f64], 2, 2).unwrap();
    let c = vec![0.0_f64, 3.0_f64];
    let a = CscMatrix::from_triplets(
        &[0usize, 1, 0, 1],
        &[0usize, 0, 1, 1],
        &[-1.0_f64, 1.0, -2.0, 1.0],
        2,
        2,
    )
    .unwrap();
    let b = vec![-10.0_f64, 5.0_f64];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![
            crate::problem::ConstraintType::Eq,
            crate::problem::ConstraintType::Eq,
        ],
    )
    .unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0_f64, 5.0_f64],
        dual_solution: vec![8.0_f64, 8.0_f64 + 1e-6_f64],
        bound_duals: vec![],
        ..SolverResult::default()
    };

    let target_pf = 5e-7;
    let accepted = try_dual_only_ir(&problem, &mut result, &[], target_pf, None);

    assert!(accepted > 0);

    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &[],
    };
    let df_rel = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    assert!(df_rel < target_pf, "got {:.3e}", df_rel);
}

/// rank-deficient Q (e e^T) + 多解で duality gap が偽 Optimal を弾く。
#[test]
fn test_duality_gap_rejects_rank_deficient_false_optimal() {
    use crate::sparse::CscMatrix;
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], n, n).unwrap();
    let c = vec![-1.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
    let b = vec![3.0_f64];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    if result.status == SolveStatus::Optimal {
        assert!((result.objective - (-0.5)).abs() < 1e-3, "got {}", result.objective);
    }
}

/// EmptyCol 変数の bound_dual を統合経路で KKT 復元 (presolve ON)。
#[test]
fn test_refit_integration_emptycol_recovery() {
    let n = 3usize;
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
    let c = vec![-1.0, -1.0, 2.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![5.0_f64];
    let bounds = vec![
        (0.0_f64, f64::INFINITY),
        (0.0_f64, f64::INFINITY),
        (0.0_f64, 10.0_f64),
    ];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.bound_duals.len(), 4);
    let z_lb_z = result.bound_duals[2];
    assert!((z_lb_z - 2.0).abs() < 1e-2, "got {}", z_lb_z);
}

/// 1×1 well-conditioned で compute_lsq_dual_y が解析解 y=-3 を再現。
/// CG + 正則化 (ε ≈ 4e-12) により解バイアスは O(ε/λ_min) ≈ 3e-12 程度。
/// no-op 検証: y を常に 0 返却すると |0-(-3)| = 3 >> 1e-9 で FAIL。
#[test]
fn compute_lsq_dual_y_recovers_exact_solution_on_well_conditioned() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
    let q = CscMatrix::new(1, 1);
    let c = vec![6.0_f64];
    let b = vec![0.0_f64];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![0.0],
        bound_duals: vec![],
        ..SolverResult::default()
    };
    let y = compute_lsq_dual_y(&problem, &result, None).expect("LSQ should succeed");
    // CG 正則化 (ε ≈ 4e-12) によるバイアス ≈ 3e-12 を考慮した許容誤差。
    assert!((y[0] - (-3.0)).abs() < 1e-9, "got {}", y[0]);
}

/// ill-conditioned (cond(AAT)≈1e16) で IR が residual を f64 1-shot 限界以下に縮める。
#[test]
fn compute_lsq_dual_y_ir_improves_ill_conditioned_problem() {
    let delta = 1e-8;
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0_f64, 1.0, 1.0, 1.0 + delta],
        2,
        2,
    )
    .unwrap();
    let q = CscMatrix::new(2, 2);
    let c = vec![-1.0_f64, -1.0];
    let b = vec![0.0_f64; 2];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c.clone(), a.clone(), b, bounds).unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0, 0.0],
        dual_solution: vec![0.0, 0.0],
        bound_duals: vec![],
        ..SolverResult::default()
    };
    let y = compute_lsq_dual_y(&problem, &result, None).expect("LSQ should succeed");

    use twofloat::TwoFloat;
    let target = [1.0_f64, 1.0];
    let mut max_abs_res = 0.0_f64;
    for col in 0..2 {
        let mut s = TwoFloat::from(0.0);
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            s += TwoFloat::new_mul(a.values[k], y[a.row_ind[k]]);
        }
        let r = (f64::from(s) - target[col]).abs();
        max_abs_res = max_abs_res.max(r);
    }
    // f64 1-shot solve は cond²·ε ≈ 2 で打ち止め。IR で <1e-7 に到達できる。
    assert!(max_abs_res < 1e-7, "got {:.3e}", max_abs_res);
}

#[test]
fn compute_lsq_dual_y_respects_singleton_row_fixed_value() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-1.0_f64, 1.0], 2, 2).unwrap();
    let q = CscMatrix::new(2, 2);
    let c = vec![0.0_f64, 5.0];
    let b = vec![0.0_f64; 2];
    let bounds = vec![(0.0_f64, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0, 0.0],
        dual_solution: vec![50.0, 0.0],
        bound_duals: vec![],
        ..SolverResult::default()
    };

    let y = compute_lsq_dual_y(&problem, &result, None).expect("LSQ should succeed");

    assert_eq!(y.len(), 2);
    assert!(y[0].abs() < 1e-10, "got {}", y[0]);
    assert!((y[1] - (-5.0)).abs() < 1e-8, "got {}", y[1]);
}

/// refine_dual_lsq の DD-guard が改善なし y_new を rejection (現状維持)。
#[test]
fn refine_dual_lsq_keeps_y_when_lsq_does_not_strictly_improve() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
    let q = CscMatrix::new(1, 1);
    let c = vec![0.0_f64];
    let b = vec![0.0_f64];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![0.0_f64],
        bound_duals: vec![],
        ..SolverResult::default()
    };
    refine_dual_lsq(&problem, &mut result, &[], None);
    assert!(result.dual_solution[0].abs() < 1e-12, "got {}", result.dual_solution[0]);
}
