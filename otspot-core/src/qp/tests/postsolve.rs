use super::super::*;
use crate::problem::{ConstraintType, SolveStatus};
use crate::sparse::CscMatrix;

/// presolve OFF 基準線。
#[test]
fn test_postsolve_t1_presolve_off_baseline() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![4.0, 3.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.objective).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), n);
    assert!((result.reduced_costs[0] - 2.0).abs() < tol);
    assert!((result.reduced_costs[1] - 3.0).abs() < tol);
}

/// FixedVar + col_map リマップ (rc[2]=0 で展開されること)。
#[test]
fn test_postsolve_t2_fixed_var_col_map() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, n)
        .unwrap();
    let b = vec![4.0, 6.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2] - 5.0).abs() < tol);
    assert!((result.objective - 5.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[0] - 2.0).abs() < tol);
    assert!((result.reduced_costs[1] - 3.0).abs() < tol);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 6.0).abs() < tol);
    // 自由変数 (x, y) のみ複ementarity 検査 (固定 z は lb/ub の dual を持ち得る)。
    for j in 0..2 {
        assert!((result.solution[j] * result.reduced_costs[j]).abs() < 1e-7);
    }
}

/// SingletonRow + row_map: x=2 (Eq) + y≤3。
#[test]
fn test_postsolve_t3_singleton_row() {
    use crate::problem::ConstraintType;
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    // x=2 (Eq), y<=3 (Le)
    let rows = &[0usize, 1usize];
    let cols = &[0usize, 1usize];
    let vals = &[1.0, 1.0];
    let a = CscMatrix::from_triplets(rows, cols, vals, 2, n).unwrap();
    let b = vec![2.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq, ConstraintType::Le],
    )
    .unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 2);
    assert!((result.solution[0] - 2.0).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0]).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 2);
}

/// Ruiz + FixedVar 複合。
#[test]
fn test_postsolve_t4_ruiz_fixed_var() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[10.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![10.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2] - 5.0).abs() < tol);
    assert!((result.objective - 5.0).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 10.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
}

/// LCS (1e7 係数) + Ruiz + FixedVar: slack を元空間 b-Ax で再計算する精度確認。
#[test]
fn test_postsolve_t5_lcs_ruiz_fixed_var() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1e7, 1.0, 1.0, 1.0], 2, n)
        .unwrap();
    let b = vec![1e7, 2.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.5, 0.5)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let x = result.solution[0];
    let y = result.solution[1];
    assert_eq!(result.slack.len(), 2);
    let slack0_expected = 1e7 - 1e7 * x - y;
    let slack1_expected = 2.0 - x - y;
    let tol_rel = 1e-5_f64;
    assert!((result.slack[0] - slack0_expected).abs() <= tol_rel * slack0_expected.abs().max(1.0));
    assert!((result.slack[1] - slack1_expected).abs() <= tol_rel * slack1_expected.abs().max(1.0));
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < 1e-6);
}

/// EmptyCol (z 制約行ゼロ) → z=lb=0 に固定。
#[test]
fn test_postsolve_t6_empty_col() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![4.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 3.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2]).abs() < tol);
    assert!((result.objective).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
}

/// QP IPM 経路では slack=[], reduced_costs=[]。
#[test]
fn test_postsolve_t7_qp_ipm_empty_slack_rc() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![2.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(result.slack.is_empty());
    assert!(result.reduced_costs.is_empty());
}

/// 全変数 FixedVar。
#[test]
fn test_postsolve_e1_all_vars_fixed() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(1.0_f64, 1.0_f64), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.solution.len(), 2);
    assert_eq!(result.reduced_costs.len(), 2);
    assert_eq!(result.slack.len(), 0);
}

/// 制約なし問題: slack=0, rc=n。
#[test]
fn test_postsolve_e2_no_constraints() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0];
    let a = CscMatrix::new(0, n);
    let b: Vec<f64> = vec![];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert_eq!(result.slack.len(), 0);
    assert_eq!(result.reduced_costs.len(), n);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
}

/// presolve=true でも reduction 発動なし → col_map identity。
#[test]
fn test_postsolve_e3_presolve_no_reduction() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![2.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.reduced_costs.len(), n);
    assert_eq!(result.slack.len(), 1);
    let tol = 1e-8_f64;
    assert!((result.slack[0] - 2.0).abs() < tol);
}

/// LCS 発動 + presolve 変数除去なし: slack を b-Ax 元空間再計算。
#[test]
fn test_postsolve_e4_lcs_no_presolve_elimination() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1e7, 1.0], 1, n).unwrap();
    let b = vec![1e7];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let x = result.solution[0];
    let y = result.solution[1];
    assert_eq!(result.slack.len(), 1);
    let slack_expected = 1e7 - 1e7 * x - y;
    let tol_rel = 1e-5_f64;
    assert!((result.slack[0] - slack_expected).abs() <= tol_rel * slack_expected.abs().max(1.0));
    assert_eq!(result.reduced_costs.len(), n);
}

/// Sentinel Fix-1+2 (integration): Le singleton row via SingletonRow postsolve step must yield
/// sign-feasible dual AND status Optimal.
///
/// Construction: x₀ ∈ [1,1] (fixed), Le: 2e5·x₀ ≤ 2e5 (large coeff → step1 LARGE_B_THRESHOLD
/// skips; step2 non-Eq path pushes SingletonRow{row=0,col=0}).
/// x₁ is the optimization variable: min 0.5·x₁².
/// c = [1e3, 0] so KKT for x₀ gives y_Le = -(0 + 1e3 + 0) / 2e5 = -5e-3 < 0 without Fix 1.
///
/// Without Fix 1: y_Le = -5e-3 (sign violation) → prove_optimal fails → SuboptimalSolution.
/// With Fix 1:    y_Le projected to 0 (sign-feasible) → Optimal.
/// The dual_solution for the Le row must be ≥ 0 with the fix.
#[test]
fn test_sentinel_le_singleton_row_sign_feasible_and_optimal() {
    let n = 2usize;
    // Q = diag(0, 1): only x₁ has a quadratic term.
    let q = CscMatrix::from_triplets(&[1], &[1], &[1.0_f64], n, n).unwrap();
    // c = [1e3, 0]: cost on x₀ creates a non-trivial Le dual via KKT.
    let c = vec![1e3_f64, 0.0_f64];
    // Le: 2e5·x₀ ≤ 2e5  (singleton row for x₀; large coeff bypasses step1 LARGE_B_THRESHOLD).
    let a = CscMatrix::from_triplets(&[0], &[0], &[2e5_f64], 1, n).unwrap();
    let b = vec![2e5_f64];
    // x₀ fixed at 1 (lb=ub=1); x₁ free.
    let bounds = vec![(1.0_f64, 1.0_f64), (f64::NEG_INFINITY, f64::INFINITY)];
    let cts = vec![ConstraintType::Le];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Le singleton row (SingletonRow path) must yield Optimal; got {:?}. \
         Without Fix 1 KKT-recovered y_Le = -5e-3 → sign violation → SuboptimalSolution.",
        result.status
    );
    // The Le constraint dual (row 0) must be sign-feasible (≥ 0).
    let y_le = result.dual_solution.get(0).copied().unwrap_or(f64::NAN);
    assert!(
        y_le >= -1e-6,
        "Le dual must be ≥ 0 (sign-feasible); got {y_le:.3e}. Fix 1 reverted?"
    );
}

/// Q=0 (LP) で reduced_costs が理論値と一致 (Simplex 経路保持)。
#[test]
fn test_solve_as_lp_preserves_reduced_costs() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 2.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.reduced_costs.len(), n);
    assert_eq!(result.slack.len(), 1);
}
