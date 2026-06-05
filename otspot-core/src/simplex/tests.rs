#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::options::{SimplexMethod, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::sparse::CscMatrix;
use crate::test_kkt::assert_solver_invariants_lp;
use crate::tolerances::PIVOT_TOL;

fn make_lp(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
) -> LpProblem {
    let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
    LpProblem::new(c, a, b).unwrap()
}

#[test]
fn test_timeout_result_with_incumbent_uses_original_objective() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![3.0, 1.0],
        a,
        vec![1.0],
        vec![ConstraintType::Ge],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    let sf = build_standard_form(&lp);
    let basis = sf.initial_basis.clone();
    let x_b = sf.b.clone();
    let col_scale = vec![1.0; sf.n_total];

    let result = timeout_result_with_incumbent(&sf, &lp, &basis, &x_b, &col_scale, 42);

    assert_eq!(result.status, SolveStatus::Timeout);
    assert_eq!(
        result.iterations, 42,
        "iter arg は SolverResult.iterations へ反映"
    );
    assert_eq!(result.solution.len(), 2);
    let expected_obj =
        lp.c.iter()
            .zip(result.solution.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>();
    assert!(
        (result.objective - expected_obj).abs() < 1e-12,
        "obj={}",
        result.objective
    );
}

/// Sentinel: a SHIFTED LP (lb ≠ 0 ⇒ obj_offset ≠ 0) must NOT double-count the
/// shift constant. `extract_solution` already un-shifts, so `c·solution` is the
/// complete original objective; adding `sf.obj_offset` on top double-counts
/// `Σ c_j·lb_j` (the same defect the Big-M Optimal path was fixed for).
///
/// x0 ∈ [2, ∞), x1 ∈ [0, ∞), min 3x0 + x1, x0+x1 ≥ 1. obj_offset = 3·2 = 6.
/// Incumbent = initial basis (structurals at lb) ⇒ x0=2, x1=0 ⇒ c·x = 6.
///
/// no-op proof: re-adding `+ sf.obj_offset` makes the reported objective 12,
/// failing the `c·solution` equality.
#[test]
fn test_timeout_result_with_incumbent_no_double_count_on_shifted_lp() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![3.0, 1.0],
        a,
        vec![1.0],
        vec![ConstraintType::Ge],
        vec![(2.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    let sf = build_standard_form(&lp);
    assert!(
        sf.obj_offset.abs() > 1e-9,
        "test LP must be shifted (obj_offset != 0); got {}",
        sf.obj_offset
    );
    let basis = sf.initial_basis.clone();
    let x_b = sf.b.clone();
    let col_scale = vec![1.0; sf.n_total];

    let result = timeout_result_with_incumbent(&sf, &lp, &basis, &x_b, &col_scale, 7);

    let expected_obj = lp
        .c
        .iter()
        .zip(result.solution.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>();
    assert!(
        (result.objective - expected_obj).abs() < 1e-9,
        "shifted-LP incumbent obj must be c·solution (no obj_offset double-count); \
         reported {} vs c·solution {} (diff = obj_offset {} ⇒ double-count)",
        result.objective,
        expected_obj,
        sf.obj_offset
    );
}

#[test]
fn test_reconcile_final_basis_state_recomputes_xb_and_y() {
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 2, 1, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
        .unwrap();
    let b = vec![3.0, 5.0];
    let c = vec![4.0, 2.0, 1.0];
    let basis = vec![0usize, 2usize];
    let mut x_b = vec![0.0, 0.0];
    let mut y = vec![0.0, 0.0];

    reconcile_final_basis_state(&a, &b, &c, &basis, &mut x_b, &mut y, 50, None).unwrap();

    assert!((x_b[0] + 2.0).abs() < 1e-12, "x_b[0]={}", x_b[0]);
    assert!((x_b[1] - 5.0).abs() < 1e-12, "x_b[1]={}", x_b[1]);
    assert!((y[0] - 4.0).abs() < 1e-12, "y[0]={}", y[0]);
    assert!((y[1] + 3.0).abs() < 1e-12, "y[1]={}", y[1]);
}

#[test]
fn test_extract_solution_uses_dd_for_split_variable_cancellation() {
    let sf = StandardForm {
        a: CscMatrix::new(3, 3),
        b: vec![0.0, 0.0, 0.0],
        c: vec![0.0, 0.0, 0.0],
        m: 3,
        n_shifted: 3,
        n_total: 3,
        initial_basis: vec![0, 1, 2],
        needs_artificial: vec![false, false, false],
        num_artificial: 0,
        obj_offset: 0.0,
        n_orig: 1,
        orig_var_info: vec![OrigVarInfo {
            offset: 0.0,
            new_vars: vec![(0, 1.0), (1, 1.0), (2, -1.0)],
        }],
        row_negated: vec![false, false, false],
    };
    let basis = vec![0usize, 1usize, 2usize];
    let x_b = vec![1.0_f64, 1.0e16_f64, 1.0e16_f64];
    let col_scale = vec![1.0, 1.0, 1.0];

    let solution = extract_solution(&sf, &basis, &x_b, &col_scale);

    assert_eq!(solution.len(), 1);
    assert!(
        (solution[0] - 1.0).abs() < 1e-12,
        "split-variable recomposition should preserve unit residual, got {}",
        solution[0]
    );
}

#[test]
fn test_basic_2var() {
    let lp = make_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.objective - (-4.0)).abs() < PIVOT_TOL,
        "Expected objective -4.0, got {}",
        result.objective
    );
    let x1 = result.solution[0];
    let x2 = result.solution[1];
    assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x1), "x1={}", x1);
    assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x2), "x2={}", x2);
    assert!((x1 + x2 - 4.0).abs() < PIVOT_TOL);
}

#[test]
fn test_basic_3var() {
    let lp = make_lp(
        vec![-2.0, -3.0, -1.0],
        &[0, 0, 0, 1, 1, 2, 2],
        &[0, 1, 2, 0, 1, 1, 2],
        &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        3,
        3,
        vec![10.0, 14.0, 8.0],
    );
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    let x = &result.solution;
    assert!(x[0] >= -PIVOT_TOL);
    assert!(x[1] >= -PIVOT_TOL);
    assert!(x[2] >= -PIVOT_TOL);
    assert!(x[0] + x[1] + x[2] <= 10.0 + PIVOT_TOL);
    assert!(2.0 * x[0] + x[1] <= 14.0 + PIVOT_TOL);
    assert!(x[1] + x[2] <= 8.0 + PIVOT_TOL);
    assert!(
        (result.objective - (-28.0)).abs() < PIVOT_TOL,
        "Expected objective -28.0, got {}",
        result.objective
    );
}

#[test]
fn test_unbounded() {
    let lp = make_lp(
        vec![-1.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, -1.0],
        1,
        2,
        vec![1.0],
    );
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Unbounded);
}

#[test]
fn test_infeasible() {
    let lp = make_lp(vec![1.0], &[0], &[0], &[1.0], 1, 1, vec![-1.0]);
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Infeasible);
}

#[test]
fn test_degenerate_zero_vars() {
    let a = CscMatrix::new(0, 0);
    let lp = LpProblem::new(vec![], a, vec![]).unwrap();
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!((result.objective).abs() < PIVOT_TOL);
}

#[test]
fn test_zero_constraints_unbounded() {
    let a = CscMatrix::new(0, 1);
    let lp = LpProblem::new(vec![-1.0], a, vec![]).unwrap();
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Unbounded);
}

#[test]
fn test_zero_constraints_optimal() {
    let a = CscMatrix::new(0, 1);
    let lp = LpProblem::new(vec![1.0], a, vec![]).unwrap();
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!((result.objective).abs() < PIVOT_TOL);
}

#[test]
fn test_solve_with_default_options() {
    // SolverOptions::default() で solve() と同じ結果が返ること
    let lp = make_lp(
        vec![-1.0, -2.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );
    let result_default = solve(&lp);
    let result_with = solve_with(&lp, &SolverOptions::default());
    assert_eq!(result_default.status, result_with.status);
    assert!(
        (result_default.objective - result_with.objective).abs() < PIVOT_TOL,
        "solve() and solve_with(default) should return same objective"
    );
}

/// min -x - y s.t. x+y ≥ 1, 0 ≤ x,y ≤ 10 ⇒ x=y=10, obj=-20.
#[test]
fn test_simplex_ge_defensive() {
    use crate::problem::ConstraintType;
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -1.0],
        a,
        vec![1.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 10.0), (0.0, 10.0)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(5.0);
    let start = std::time::Instant::now();
    let result = solve_with(&lp, &opts);
    assert!(
        start.elapsed().as_secs_f64() < 6.0,
        "test_simplex_ge_defensive: wall-clock 6秒超過"
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Status should be Optimal"
    );
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.objective - (-20.0)).abs() < PIVOT_TOL,
        "Expected obj=-20.0, got {}",
        result.objective
    );
    assert!(
        result.solution[0] >= -PIVOT_TOL && result.solution[0] <= 10.0 + PIVOT_TOL,
        "x should be in [0, 10], got {}",
        result.solution[0]
    );
    assert!(
        result.solution[1] >= -PIVOT_TOL && result.solution[1] <= 10.0 + PIVOT_TOL,
        "y should be in [0, 10], got {}",
        result.solution[1]
    );
    assert!(
        (result.solution[0] + result.solution[1] - 20.0).abs() < PIVOT_TOL,
        "x + y should be 20.0, got {}",
        result.solution[0] + result.solution[1]
    );
}

/// Le-only LP: verify dual / slack / reduced costs.
/// min -x1-2x2 s.t. x1+x2≤4, x1≤3, x2≤3, x≥0
///  ⇒ x=(1,3), y=(-1,0,-1), slack=(0,2,0), rc=(0,0).
#[test]
fn test_dual_solution_basic_le_constraints() {
    let lp = make_lp(
        vec![-1.0, -2.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.objective - (-7.0)).abs() < PIVOT_TOL,
        "Expected obj=-7.0, got {}",
        result.objective
    );

    // 双対変数の検証
    assert_eq!(
        result.dual_solution.len(),
        3,
        "dual_solution should have 3 elements"
    );
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < PIVOT_TOL,
        "y[0] should be -1.0, got {}",
        result.dual_solution[0]
    );
    assert!(
        result.dual_solution[1].abs() < PIVOT_TOL,
        "y[1] should be 0.0 (non-binding), got {}",
        result.dual_solution[1]
    );
    assert!(
        (result.dual_solution[2] - (-1.0)).abs() < PIVOT_TOL,
        "y[2] should be -1.0, got {}",
        result.dual_solution[2]
    );

    // スラック変数の検証
    assert_eq!(result.slack.len(), 3, "slack should have 3 elements");
    assert!(
        result.slack[0].abs() < PIVOT_TOL,
        "slack[0] should be 0 (binding), got {}",
        result.slack[0]
    );
    assert!(
        (result.slack[1] - 2.0).abs() < PIVOT_TOL,
        "slack[1] should be 2.0 (non-binding), got {}",
        result.slack[1]
    );
    assert!(
        result.slack[2].abs() < PIVOT_TOL,
        "slack[2] should be 0 (binding), got {}",
        result.slack[2]
    );

    // 被縮小費用の検証（基底変数なのでゼロ）
    assert_eq!(
        result.reduced_costs.len(),
        2,
        "reduced_costs should have 2 elements"
    );
    assert!(
        result.reduced_costs[0].abs() < PIVOT_TOL,
        "rc[0] should be 0 (basic), got {}",
        result.reduced_costs[0]
    );
    assert!(
        result.reduced_costs[1].abs() < PIVOT_TOL,
        "rc[1] should be 0 (basic), got {}",
        result.reduced_costs[1]
    );
}

#[test]
fn test_large_coefficient_lp() {
    // 係数に 1e12 と 1e-12 を混合した問題 → Optimal or 適切なステータス（オーバーフローしない）
    // min -1e12 * x1 + 1e-12 * x2, s.t. x1 + x2 <= 1, x1,x2 >= 0
    // 最適解: x1=1, x2=0, obj=-1e12
    let lp = make_lp(
        vec![-1e12, 1e-12],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![1.0],
    );
    let result = solve(&lp);
    assert!(
        result.status == SolveStatus::Optimal || result.status == SolveStatus::Timeout,
        "Expected Optimal or Timeout, got {:?}",
        result.status
    );
    assert!(!result.objective.is_nan(), "Objective should not be NaN");
    assert!(
        result.objective.is_finite(),
        "Objective should be finite for bounded LP"
    );
    if result.status == SolveStatus::Optimal {
        assert_solver_invariants_lp(&result, &lp);
    }

    // 全係数 0.0 の目的関数 → Optimal, objective=0.0
    // min 0*x1 + 0*x2, s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
    let lp_zero = make_lp(
        vec![0.0, 0.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![2.0, 1.0, 1.0],
    );
    let result_zero = solve(&lp_zero);
    assert_eq!(
        result_zero.status,
        SolveStatus::Optimal,
        "Expected Optimal for zero-objective LP"
    );
    assert_solver_invariants_lp(&result_zero, &lp_zero);
    assert!(
        result_zero.objective.abs() < PIVOT_TOL,
        "Expected objective=0.0, got {}",
        result_zero.objective
    );
}

#[test]
fn test_highly_degenerate_lp() {
    // 高度退化 LP: 3制約が (1,1) で交わる → 基底解が退化
    // min -x1 - x2
    // s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
    // 最適解: x1=1, x2=1, obj=-2（サイクリングせずに到達すること）
    let lp = make_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![2.0, 1.0, 1.0],
    );
    let result = solve(&lp);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Expected Optimal for degenerate LP"
    );
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.objective - (-2.0)).abs() < PIVOT_TOL,
        "Expected objective=-2.0, got {}",
        result.objective
    );
    let x1 = result.solution[0];
    let x2 = result.solution[1];
    assert!((x1 - 1.0).abs() < PIVOT_TOL, "Expected x1=1.0, got {}", x1);
    assert!((x2 - 1.0).abs() < PIVOT_TOL, "Expected x2=1.0, got {}", x2);
}

/// Eq + Le mix: verify dual / slack / reduced costs.
/// min x1+2x2 s.t. x1+x2=6, x2≤5, x≥0
///  ⇒ x=(6,0), y=(1,0), slack=(0,5), rc=(0,1).
#[test]
fn test_dual_solution_equality_constraint() {
    use crate::problem::ConstraintType;
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![6.0, 5.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.objective - 6.0).abs() < PIVOT_TOL,
        "Expected obj=6.0, got {}",
        result.objective
    );

    // 双対変数の検証
    assert_eq!(
        result.dual_solution.len(),
        2,
        "dual_solution should have 2 elements"
    );
    assert!(
        (result.dual_solution[0] - 1.0).abs() < PIVOT_TOL,
        "y[0] (Eq constraint shadow price) should be 1.0, got {}",
        result.dual_solution[0]
    );
    assert!(
        result.dual_solution[1].abs() < PIVOT_TOL,
        "y[1] (Le constraint, non-binding) should be 0.0, got {}",
        result.dual_solution[1]
    );

    // スラック変数の検証
    assert_eq!(result.slack.len(), 2, "slack should have 2 elements");
    assert!(
        result.slack[0].abs() < PIVOT_TOL,
        "slack[0] (Eq constraint) should be 0, got {}",
        result.slack[0]
    );
    assert!(
        (result.slack[1] - 5.0).abs() < PIVOT_TOL,
        "slack[1] (x2<=5, non-binding) should be 5.0, got {}",
        result.slack[1]
    );

    // 被縮小費用の検証
    assert_eq!(
        result.reduced_costs.len(),
        2,
        "reduced_costs should have 2 elements"
    );
    assert!(
        result.reduced_costs[0].abs() < PIVOT_TOL,
        "rc[0] (x1, basic) should be 0.0, got {}",
        result.reduced_costs[0]
    );
    assert!(
        (result.reduced_costs[1] - 1.0).abs() < PIVOT_TOL,
        "rc[1] (x2, non-basic) should be 1.0, got {}",
        result.reduced_costs[1]
    );
}

#[test]
fn test_free_variables_phase_i() {
    // 全変数が自由境界（-INF/INF）のLP
    // minimize x1 + x2
    // s.t. x1 + x2 = 2
    // x1, x2 in (-INF, INF)
    // → Optimal（Infeasibleを返してはならない）
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![2.0],
        vec![crate::problem::ConstraintType::Eq],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        None,
    )
    .unwrap();
    let result = solve(&lp);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Expected Optimal for free-variable LP with Eq constraint, got {:?}",
        result.status
    );
    assert_solver_invariants_lp(&result, &lp);
    // 解の制約充足チェック: x1 + x2 = 2
    assert!(
        (result.solution[0] + result.solution[1] - 2.0).abs() < 1e-6,
        "Expected x1+x2=2, got x1={}, x2={}, sum={}",
        result.solution[0],
        result.solution[1],
        result.solution[0] + result.solution[1]
    );
}

#[test]
fn test_hs51_feasibility_lp() {
    // HS51の実行可能性LP: find_initial_feasible_pointが構築するLPを直接テスト
    // 5変数(全自由), 6Le制約(等式制約を2不等式ペアに変換)
    // b[1]=-4.0 (負のRHS) → build_standard_formで符号反転+人工変数追加
    // 解は存在する(x=[1,1,1,1,1])のでOptimalを返すべき
    let a = CscMatrix::from_triplets(
        &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
        &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
        &[
            1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0,
        ],
        6,
        5,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0; 5],
        a,
        vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
        vec![crate::problem::ConstraintType::Le; 6],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
        None,
    )
    .unwrap();
    let result = solve(&lp);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "HS51 feasibility LP: Expected Optimal, got {:?}",
        result.status
    );
    assert_solver_invariants_lp(&result, &lp);
    // 解が制約を満たすか検証 (x1+3x2=4 かつ x3+x4-2x5=0 かつ x2-x5=0)
    let x = &result.solution;
    assert!(
        (x[0] + 3.0 * x[1] - 4.0).abs() < 1e-6,
        "Constraint x1+3x2=4 violated: {}",
        x[0] + 3.0 * x[1]
    );
}

#[test]
fn test_finite_ub_zero_constraints() {
    // m=0 with maximize x, lb=0, ub=3 ⇒ x=3.
    let a = CscMatrix::new(0, 1);
    let lp = LpProblem::new_general(
        vec![-1.0], // minimize -x (= maximize x)
        a,
        vec![],
        vec![],
        vec![(0.0, 3.0)], // lb=0, ub=3
        None,
    )
    .unwrap();
    let result = solve(&lp);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.solution[0] - 3.0).abs() < PIVOT_TOL,
        "Expected x=3, got {}",
        result.solution[0]
    );
    assert!(
        (result.objective - (-3.0)).abs() < PIVOT_TOL,
        "Expected obj=-3, got {}",
        result.objective
    );
}

// --- m==0 early-return bound-selection tests ---
//
// All tests call `solve_without_presolve` directly so they exercise the
// `if m == 0` branch in entry.rs rather than letting presolve intercept.

/// Helper: solve m=0 problem without presolve and return the result.
fn solve_m0(c: Vec<f64>, bounds: Vec<(f64, f64)>) -> (LpProblem, crate::problem::SolverResult) {
    let n = c.len();
    let a = CscMatrix::new(0, n);
    let lp = LpProblem::new_general(c, a, vec![], vec![], bounds, None).unwrap();
    let result = solve_without_presolve(&lp, &SolverOptions::default());
    (lp, result)
}

/// Regression: c=0, lb=-inf, ub=-1 must return x=-1 (not x=0 which violates ub).
/// Before the fix, the m==0 path left x=0 and returned Optimal, silently violating ub<0.
#[test]
fn test_m0_zero_cost_ub_negative_regression() {
    let (lp, result) = solve_m0(vec![0.0], vec![(f64::NEG_INFINITY, -1.0)]);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "Expected Optimal, got {:?}",
        result.status
    );
    assert_solver_invariants_lp(&result, &lp);
    let x = result.solution[0];
    assert!(
        x <= -1.0 + PIVOT_TOL,
        "x={x} violates ub=-1 (pre-fix bug: x=0 was returned)"
    );
}

/// c>0, lb finite: optimizer drives x to lower bound.
#[test]
fn test_m0_positive_cost_lb_finite() {
    let (lp, result) = solve_m0(vec![1.0], vec![(-3.0, f64::INFINITY)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.solution[0] - (-3.0)).abs() < PIVOT_TOL,
        "Expected x=-3, got {}",
        result.solution[0]
    );
    assert!(
        (result.objective - (-3.0)).abs() < PIVOT_TOL,
        "Expected obj=-3, got {}",
        result.objective
    );
}

/// c>0, lb=-inf: unbounded below.
#[test]
fn test_m0_positive_cost_lb_infinite_unbounded() {
    let (_lp, result) = solve_m0(vec![1.0], vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    assert_eq!(
        result.status,
        SolveStatus::Unbounded,
        "c>0 + lb=-inf must be Unbounded, got {:?}",
        result.status
    );
}

/// c<0, ub=+inf: unbounded above.
#[test]
fn test_m0_negative_cost_ub_infinite_unbounded() {
    let (_lp, result) = solve_m0(vec![-1.0], vec![(0.0, f64::INFINITY)]);
    assert_eq!(
        result.status,
        SolveStatus::Unbounded,
        "c<0 + ub=+inf must be Unbounded, got {:?}",
        result.status
    );
}

/// c<0, ub finite: optimizer drives x to upper bound.
#[test]
fn test_m0_negative_cost_ub_finite() {
    let (lp, result) = solve_m0(vec![-1.0], vec![(0.0, 3.0)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        (result.solution[0] - 3.0).abs() < PIVOT_TOL,
        "Expected x=3, got {}",
        result.solution[0]
    );
    assert!(
        (result.objective - (-3.0)).abs() < PIVOT_TOL,
        "Expected obj=-3, got {}",
        result.objective
    );
}

/// c=0, lb finite: must land at lb (feasible and cost-free).
#[test]
fn test_m0_zero_cost_lb_finite() {
    let (lp, result) = solve_m0(vec![0.0], vec![(2.0, f64::INFINITY)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!(
        result.solution[0] >= 2.0 - PIVOT_TOL,
        "x={} must be >= lb=2",
        result.solution[0]
    );
}

/// c=0, lb=-inf, ub finite: must land at or below ub.
#[test]
fn test_m0_zero_cost_lb_inf_ub_finite() {
    let (lp, result) = solve_m0(vec![0.0], vec![(f64::NEG_INFINITY, -1.0)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    let x = result.solution[0];
    assert!(x <= -1.0 + PIVOT_TOL, "x={x} violates ub=-1");
}

/// c=0, both lb and ub finite: must land within [lb, ub].
#[test]
fn test_m0_zero_cost_both_bounds_finite() {
    let (lp, result) = solve_m0(vec![0.0], vec![(1.0, 5.0)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    let x = result.solution[0];
    assert!(
        (1.0 - PIVOT_TOL..=5.0 + PIVOT_TOL).contains(&x),
        "x={x} outside [1,5]"
    );
}

/// c=0, both bounds infinite: x=0 is a valid feasible point.
#[test]
fn test_m0_zero_cost_both_bounds_infinite() {
    let (lp, result) = solve_m0(vec![0.0], vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
}

/// Multi-variable m=0: mixed cost signs all satisfied simultaneously.
#[test]
fn test_m0_multi_var_mixed_costs() {
    // 3 vars: c=[-1, 1, 0]
    //   var0: c=-1 (negative), lb=0, ub=4      → x=4, obj contribution=-4
    //   var1: c=+1 (positive), lb=-2, ub=inf   → x=-2, obj contribution=-2
    //   var2: c=0  (zero),     lb=3, ub=7      → x∈[3,7]
    let (lp, result) = solve_m0(
        vec![-1.0, 1.0, 0.0],
        vec![(0.0, 4.0), (-2.0, f64::INFINITY), (3.0, 7.0)],
    );
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    let x = &result.solution;
    assert!(
        (x[0] - 4.0).abs() < PIVOT_TOL,
        "var0 (c<0): expected x=4, got {}",
        x[0]
    );
    assert!(
        (x[1] - (-2.0)).abs() < PIVOT_TOL,
        "var1 (c>0): expected x=-2, got {}",
        x[1]
    );
    assert!(
        x[2] >= 3.0 - PIVOT_TOL && x[2] <= 7.0 + PIVOT_TOL,
        "var2 (c=0): expected x∈[3,7], got {}",
        x[2]
    );
    assert!(
        (result.objective - (-6.0)).abs() < PIVOT_TOL,
        "Expected obj=-6, got {}",
        result.objective
    );
}

#[test]
fn test_primal_simplex_timeout() {
    let n = 200usize;
    let m = 100usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let opts = SolverOptions {
        deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

#[test]
fn test_lp_timeout() {
    let n = 200usize;
    let m = 100usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

#[test]
fn test_lp_cancel() {
    use std::sync::Arc;
    let n = 200usize;
    let m = 100usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&cancel)),
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// Singular initial basis (duplicate column) must not yield Optimal.
#[test]
fn test_singular_initial_basis_not_optimal() {
    use crate::simplex::pricing::DantzigPricing;
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let mut x_b = vec![1.0, 0.0];
    let mut basis = vec![0usize, 0];
    let mut pricing = DantzigPricing;
    let opts = SolverOptions::default();
    let b = vec![1.0, 0.0];
    let mut iters = 0usize;
    let outcome = revised_simplex_core(
        &a,
        &mut x_b,
        &c,
        &b,
        &mut basis,
        2,
        2,
        2,
        &mut pricing,
        &opts,
        &mut iters,
        false,
    );
    assert!(!matches!(outcome, SimplexOutcome::Optimal(..)));
}

/// `solve_with` must never surface SolveStatus::MaxIterations.
#[test]
fn test_solve_does_not_return_max_iterations() {
    for method in [SimplexMethod::Primal, SimplexMethod::Dual] {
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            simplex_method: method,
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(
            result.status,
            SolveStatus::MaxIterations,
            "method={:?}",
            method
        );
    }
}

/// refactor_failed with no deadline must yield Optimal/Timeout/SingularBasis.
#[test]
fn test_refactor_failed_no_deadline_returns_timeout() {
    use crate::simplex::pricing::DantzigPricing;
    let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
    let c = vec![-1.0, -1.0, 0.0];
    let mut x_b = vec![4.0];
    let mut basis = vec![2usize];
    let mut pricing = DantzigPricing;
    // max_etas=1 forces an early refactor.
    let opts = SolverOptions {
        deadline: None,
        max_etas: 1,
        ..SolverOptions::default()
    };
    let b = vec![4.0];
    let mut iters = 0usize;
    let outcome = revised_simplex_core(
        &a,
        &mut x_b,
        &c,
        &b,
        &mut basis,
        1,
        3,
        3,
        &mut pricing,
        &opts,
        &mut iters,
        false,
    );
    assert!(matches!(
        outcome,
        SimplexOutcome::Optimal(..) | SimplexOutcome::Timeout(_) | SimplexOutcome::SingularBasis
    ));
}

/// timeout_secs=0 must propagate to Timeout (small LP path).
#[test]
fn test_presolve_respects_deadline_small() {
    let n = 200usize;
    let m = 100usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// At n=2000/m=1000, presolve must early-return on past deadline (no budget overrun).
#[test]
fn test_large_scale_presolve_respects_deadline() {
    let n = 2000usize;
    let m = 1000usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let start = std::time::Instant::now();
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Timeout);
    assert!(
        elapsed.as_secs_f64() < 0.5,
        "elapsed={:.3}s",
        elapsed.as_secs_f64()
    );
}

/// Wall-clock must stay within K · timeout_secs.
#[test]
fn test_timeout_elapsed_within_budget() {
    let n = 200usize;
    let m = 100usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            rows.push(i);
            cols.push(j);
            vals.push(1.0);
        }
    }
    let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
    let timeout_secs = 0.01f64;
    let opts = SolverOptions {
        timeout_secs: Some(timeout_secs),
        presolve: false,
        ..SolverOptions::default()
    };
    let start = std::time::Instant::now();
    let result = solve_with(&lp, &opts);
    let elapsed = start.elapsed().as_secs_f64();
    assert!(matches!(
        result.status,
        SolveStatus::Timeout | SolveStatus::Optimal
    ));
    assert!(
        elapsed < timeout_secs * 3.0 + 0.5,
        "elapsed={:.3}s",
        elapsed
    );
}

/// timeout_secs=None must still converge on a tractable LP.
#[test]
fn test_no_deadline_converges_finite() {
    let lp = make_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );
    let opts = SolverOptions {
        timeout_secs: None,
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
}

/// Optimality at upper bound (min): for x=(2,2) of min -2x1-x2 s.t. x1+x2≤4, 0≤x1≤2, 0≤x2≤3,
/// x[1] basic ⇒ lambda=-1, then rc[0]=c[0]-lambda*a[0,0]=-2+1=-1≤0 under rc=c−A^T y.
#[test]
fn test_extract_dual_info_ub_dual() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let problem = LpProblem::new_general(
        vec![-2.0, -1.0],
        a,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 2.0), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let opts = SolverOptions {
        timeout_secs: None,
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&problem, &opts);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "status should be Optimal"
    );
    assert_solver_invariants_lp(&result, &problem);

    let x = &result.solution;
    assert!(
        (x[0] - 2.0).abs() < 1e-6,
        "x[0]={} should be at upper bound 2.0",
        x[0]
    );
    assert!((x[1] - 2.0).abs() < 1e-6, "x[1]={} should be 2.0", x[1]);

    let rc = &result.reduced_costs;

    // x[0] at upper bound ⇒ optimality requires rc[0] ≤ 0 under rc = c − A^T y.
    assert!(
        rc[0] <= 1e-6,
        "rc[0]={} should be <= 0 (x[0] at upper bound)",
        rc[0]
    );

    // x[1] is strictly between bounds (0 < x[1]=2 < 3) → x[1] is basic → rc[1] ≈ 0
    assert!(
        rc[1].abs() < 1e-6,
        "rc[1]={} should be ≈ 0 (x[1] is basic)",
        rc[1]
    );

    // Upper complementarity for x[0]: (ub - x[0]) * max(-rc[0], 0) ≈ 0
    let ub0 = 2.0_f64;
    let upper_comp = (ub0 - x[0]) * (-rc[0]).max(0.0);
    assert!(
        upper_comp.abs() < 1e-8,
        "upper complementarity={} should be ≈ 0",
        upper_comp
    );
}

/// Degenerate Eq(b=0) artificials must not yield NumericalError.
/// min -x4 s.t. x1+x2=0, x1+x3=0, x2+x4=1, x1+x4≤2, x≥0  ⇒ x=(0,0,0,1).
#[test]
fn test_degenerate_eq_zero_rhs_artificials() {
    use crate::problem::ConstraintType;
    let a = CscMatrix::from_triplets(
        &[0, 1, 3, 0, 2, 1, 2, 3],
        &[0, 0, 0, 1, 1, 2, 3, 3],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        4,
        4,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, 0.0, 0.0, -1.0],
        a,
        vec![0.0, 0.0, 1.0, 2.0],
        vec![
            ConstraintType::Eq,
            ConstraintType::Eq,
            ConstraintType::Eq,
            ConstraintType::Le,
        ],
        vec![(0.0, f64::INFINITY); 4],
        None,
    )
    .unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_ne!(result.status, SolveStatus::NumericalError);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!((result.objective - (-1.0)).abs() < 1e-6);
}

/// Many b=0 Eq constraints (wood1p-style) must not yield NumericalError.
/// min -x5 s.t. x1+x2=0, x2+x3=0, x3+x4=0, x1+x5=1, sum≤2, x≥0  ⇒ x5=1.
#[test]
fn test_multiple_zero_rhs_eq_artificials() {
    use crate::problem::ConstraintType;
    let a = CscMatrix::from_triplets(
        &[0, 3, 4, 0, 1, 4, 1, 2, 4, 2, 4, 3, 4],
        &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        5,
        5,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, 0.0, 0.0, 0.0, -1.0],
        a,
        vec![0.0, 0.0, 0.0, 1.0, 2.0],
        vec![
            ConstraintType::Eq,
            ConstraintType::Eq,
            ConstraintType::Eq,
            ConstraintType::Eq,
            ConstraintType::Le,
        ],
        vec![(0.0, f64::INFINITY); 5],
        None,
    )
    .unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_ne!(result.status, SolveStatus::NumericalError);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
    assert!((result.objective - (-1.0)).abs() < 1e-6);
}

/// hs51 (free vars + Le): degenerate-artificial pivot must not singularize
/// the basis (best_j=None fallback keeps it safe).
#[test]
fn test_hs51_free_var_no_singular_basis() {
    let a = CscMatrix::from_triplets(
        &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
        &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
        &[
            1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0,
        ],
        6,
        5,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0; 5],
        a,
        vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
        vec![crate::problem::ConstraintType::Le; 6],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
        None,
    )
    .unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_with(&lp, &opts);
    assert_ne!(result.status, SolveStatus::NumericalError);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result, &lp);
}

/// Sentinel: `pivot_out_degenerate_artificials` early-exit fires when no
/// degenerate artificials remain after Phase I.
///
/// `pivot_out_degenerate_artificials` lives only on the **primal** path
/// (`two_phase_simplex`); `SimplexMethod::Auto`/`Dual` route Eq rows to the
/// dual Big-M simplex and never reach it, so the cleanup counters stay put.
/// The route must be forced to `Primal` to exercise the function.
///
/// Diagonal LP: m Eq rows `x_i = 1`. Phase I pivots each artificial out at
/// value 1.0 (non-degenerate) → no degenerate artificials remain → early-exit
/// must fire. Removing the early-exit makes `PIVOT_CLEAN_EARLY_EXIT_COUNT`
/// stagnate, failing the assertion below (no-op FAIL).
#[test]
fn primal_pivot_clean_early_exit_fires_when_no_degenerate_artificials() {
    use std::sync::atomic::Ordering;

    let before = primal::PIVOT_CLEAN_EARLY_EXIT_COUNT.load(Ordering::SeqCst);

    // 4 Eq rows: x_i = 1 each. Phase I removes all artificials non-degenerately.
    let m = 4;
    let rows: Vec<usize> = (0..m).collect();
    let cols: Vec<usize> = (0..m).collect();
    let a = CscMatrix::from_triplets(&rows, &cols, &vec![1.0f64; m], m, m).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0f64; m],
        a,
        vec![1.0f64; m],
        vec![ConstraintType::Eq; m],
        vec![(0.0f64, f64::INFINITY); m],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = false; // force artificial path
    opts.simplex_method = SimplexMethod::Primal; // primal-only cleanup path
    let result = solve_with(&lp, &opts);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "diagonal Eq LP must be Optimal"
    );
    assert!(
        (result.objective - 4.0).abs() < 1e-6,
        "obj={}",
        result.objective
    );

    let after = primal::PIVOT_CLEAN_EARLY_EXIT_COUNT.load(Ordering::SeqCst);
    assert!(
        after > before,
        "early-exit must fire when no degenerate artificials remain \
         (before={before}, after={after}); removing the early-exit causes no-op FAIL here"
    );
}

/// Inverse sentinel of the early-exit guard: when a degenerate artificial *is*
/// in the basis, the early-exit must NOT fire and the BTRAN cleanup must run
/// (`PIVOT_CLEAN_CLEANUP_RAN_COUNT` increments). Construction uses redundant Eq
/// rows so Phase I strands duplicates' artificials at value 0. Table-driven over
/// two redundancy patterns; widening the early-exit causes no-op FAIL here.
///
/// `SimplexMethod::Primal` is forced: the cleanup lives only on the primal
/// `two_phase_simplex` path (Auto/Dual route Eq to the dual Big-M simplex).
#[test]
fn primal_pivot_cleanup_runs_when_degenerate_artificial_in_basis() {
    use std::sync::atomic::Ordering;

    // (rows, cols, vals, m, n, b, c, expected_obj, label)
    struct Case {
        a: CscMatrix,
        b: Vec<f64>,
        c: Vec<f64>,
        n: usize,
        expected_obj: f64,
        label: &'static str,
    }

    // Pattern A: `x0 = 1` duplicated → 1 redundant row, 1 artificial degenerate.
    let case_a = Case {
        a: CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap(),
        b: vec![1.0, 1.0],
        c: vec![1.0],
        n: 1,
        expected_obj: 1.0,
        label: "dup_x0_eq_1",
    };

    // Pattern B: `x0 + x1 = 2` tripled → 2 redundant rows, 2 artificials degenerate.
    let case_b = Case {
        a: CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            2,
        )
        .unwrap(),
        b: vec![2.0, 2.0, 2.0],
        c: vec![1.0, 1.0],
        n: 2,
        expected_obj: 2.0,
        label: "triple_x0_plus_x1_eq_2",
    };

    for case in [case_a, case_b] {
        let before = primal::PIVOT_CLEAN_CLEANUP_RAN_COUNT.load(Ordering::SeqCst);

        let lp = LpProblem::new_general(
            case.c,
            case.a,
            case.b.clone(),
            vec![ConstraintType::Eq; case.b.len()],
            vec![(0.0, f64::INFINITY); case.n],
            None,
        )
        .unwrap();

        let mut opts = SolverOptions::default();
        opts.presolve = false; // force artificial path
        opts.simplex_method = SimplexMethod::Primal; // primal-only cleanup path
        let result = solve_with(&lp, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "[{}] redundant Eq LP must be Optimal",
            case.label
        );
        assert!(
            (result.objective - case.expected_obj).abs() < 1e-6,
            "[{}] obj={} expected={}",
            case.label,
            result.objective,
            case.expected_obj
        );

        let after = primal::PIVOT_CLEAN_CLEANUP_RAN_COUNT.load(Ordering::SeqCst);
        assert!(
            after > before,
            "[{}] cleanup must run (early-exit must NOT fire) while a degenerate \
             artificial is basic (before={before}, after={after}); a mis-firing \
             early-exit would strand the artificial and stagnate this counter",
            case.label
        );
    }
}

/// B2 sentinel: `best_obj` は `f64::INFINITY` ではなく初期有限値で初期化すべき。
///
/// `OBJ_PROGRESS_RESET_COUNT` は `revised_simplex_core` 内で
/// `current_obj + progress_eps < best_obj` が成立した回数を記録する。
///
/// **旧実装 (best_obj = INFINITY)**: `progress_eps = ∞` なので
/// `current + ∞ < ∞` が常に false → カウンタが 0 のまま → テスト FAIL。
///
/// **新実装 (best_obj = basic_obj(...))**: `progress_eps` は有限、
/// 目的関数が改善するたびに best_obj が更新され → カウンタが増加 → PASS。
///
/// 2 種類のデータパターン (CLAUDE.md「複数パターンのデータを用意せよ」):
///   - Pattern A: Eq 制約 2 本の可解 LP (Phase I アーティフィシャル 2 個)
///   - Pattern B: 大きな目的関数係数差を持つ Le 制約 LP (Phase II で大幅改善)
#[test]
fn b2_obj_progress_reset_fires_on_improving_objective() {
    use std::sync::atomic::Ordering;

    let before = OBJ_PROGRESS_RESET_COUNT.load(Ordering::SeqCst);

    // Pattern A: Eq 制約 → Phase I が走り、アーティフィシャルを駆逐する過程で
    // 目的関数が減少 → best_obj が更新されてカウンタが増加するはず。
    {
        //   min  x0 + x1 + x2
        //   s.t. x0 + x1       = 3
        //             x1 + x2  = 2
        //   x0,x1,x2 >= 0
        // 最適解: x0=1, x1=2, x2=0, obj=3
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 1, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![3.0, 2.0],
            vec![ConstraintType::Eq; 2],
            vec![(0.0, f64::INFINITY); 3],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.presolve = false;
        let result = solve_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "Pattern A must be Optimal"
        );
        assert!(
            (result.objective - 3.0).abs() < 1e-6,
            "Pattern A obj={}",
            result.objective
        );
    }

    // Pattern B: Le 制約 (Phase II で目的関数が着実に改善する LP)
    {
        //   min  -5*x0 - 4*x1 - 3*x2
        //   s.t.  6*x0 + 4*x1 + 2*x2 <= 240
        //         3*x0 + 2*x1 + 5*x2 <= 270
        //         5*x0 + 6*x1 + 5*x2 <= 420
        //   x0,x1,x2 >= 0
        // 既知最適: 有限の負値 (Phase II で複数の基底交換が起きる)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[6.0, 4.0, 2.0, 3.0, 2.0, 5.0, 5.0, 6.0, 5.0],
            3,
            3,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![-5.0, -4.0, -3.0],
            a,
            vec![240.0, 270.0, 420.0],
            vec![ConstraintType::Le; 3],
            vec![(0.0, f64::INFINITY); 3],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.presolve = false;
        let result = solve_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "Pattern B must be Optimal"
        );
        // 目的関数は負 (最小化問題で負コスト → 最適値は負)
        assert!(
            result.objective < 0.0,
            "Pattern B must have negative optimal, got {}",
            result.objective
        );
    }

    let after = OBJ_PROGRESS_RESET_COUNT.load(Ordering::SeqCst);
    assert!(
        after > before,
        "OBJ_PROGRESS_RESET_COUNT must increment when objective improves \
         (before={before}, after={after}); a non-incrementing counter means \
         best_obj was initialized to INFINITY (B2 バグ復帰) making progress_eps=∞ \
         so the improvement condition never fires"
    );
}

/// Sentinel: `pivot_out_degenerate_artificials` uses the batch path (O(1) LU, zero BTRANs)
/// when many degenerate artificials can each be matched to a unique non-basic structural column.
///
/// LP construction: m=N+1 rows, n=N+1 structural vars (x_0..x_N), all Eq constraints.
///   Row 0:        x_0         = 1   (Phase I pivots x_0 in non-degenerately)
///   Row i=1..N:   x_0 - x_i  = 1   (x_b[i] → 0 after x_0 enters row 0)
///
/// The replacement columns x_1..x_N carry coefficient -1 in rows 1..N. Their Phase I
/// reduced costs = +1 (positive), so Phase I never enters them. The N degenerate
/// artificials stay in the basis until pivot_out.
///
/// No-op proof:
///   Reverting to the O(num_art) sequential path (btran per row) makes
///   `PIVOT_OUT_BTRAN_COUNT` increase by N and `PIVOT_OUT_BATCH_LU_COUNT` stay at 0,
///   failing both assertions below.
#[test]
fn batch_pivot_out_uses_single_lu_and_no_btrans() {
    const N: usize = 50;
    let m = N + 1;
    let n = N + 1;

    // A[0,0]=1; A[i,0]=1, A[i,i]=-1 for i=1..N  (x_0 - x_i = 1)
    let mut trip_rows = vec![0usize];
    let mut trip_cols = vec![0usize];
    let mut trip_vals = vec![1.0f64];
    for i in 1..=N {
        trip_rows.extend_from_slice(&[i, i]);
        trip_cols.extend_from_slice(&[0, i]);
        trip_vals.extend_from_slice(&[1.0, -1.0]);
    }
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n).unwrap();

    // b=[1]*m, c=[1]*n.  Optimal: x_0=1, x_i=0 for i=1..N, obj=1.
    let lp = LpProblem::new_general(
        vec![1.0f64; n],
        a,
        vec![1.0f64; m],
        vec![ConstraintType::Eq; m],
        vec![(0.0f64, f64::INFINITY); n],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.simplex_method = SimplexMethod::Primal;
    opts.use_lp_crash_basis = false;

    let btran_before = primal::PIVOT_OUT_BTRAN_COUNT.with(|c| c.get());
    let batch_lu_before = primal::PIVOT_OUT_BATCH_LU_COUNT.with(|c| c.get());

    let result = solve_with(&lp, &opts);

    let btran_after = primal::PIVOT_OUT_BTRAN_COUNT.with(|c| c.get());
    let batch_lu_after = primal::PIVOT_OUT_BATCH_LU_COUNT.with(|c| c.get());

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "LP with {N} degenerate artificials must reach Optimal; got {:?}",
        result.status
    );
    assert!(
        (result.objective - 1.0).abs() < 1e-6,
        "expected obj=1.0, got {}",
        result.objective
    );

    // Batch path issues zero BTRANs (greedy uses raw |A[r,j]|, not B^{-T}e_i).
    // Reverting to sequential adds N BTRANs → this fails.
    assert_eq!(
        btran_after, btran_before,
        "batch path must issue zero BTRANs for {N} degenerate artificials; \
         sequential path would add {N} (before={btran_before}, after={btran_after})"
    );

    // Batch path does exactly one LU for all N matched rows.
    // Reverting to sequential never increments this → this fails.
    assert_eq!(
        batch_lu_after,
        batch_lu_before + 1,
        "batch path must call exactly one LU for {N} matched rows; \
         sequential path never increments this counter \
         (before={batch_lu_before}, after={batch_lu_after})"
    );
}

/// Sentinel: batch pivot_out falls back to sequential when the greedy-selected column
/// is ill-conditioned — large raw |A[r,j]| but its FTRAN entry at row r nearly cancels.
///
/// 3 rows × 4 cols, all Eq, c=0: x0=[1,1,1], x1=[1,−0.5,0], x2=[0.7,0,1],
/// s=[1,1+δ,0], δ=0.001. Phase I enters x0 and exits with all rc≥0, leaving art1,art2
/// degenerate at rows 1,2 → pivot_out runs. Batch greedy picks s for row 1
/// (|A[1,s]|=1+δ > |A[1,x1]|=0.5) and x2 for row 2; trial basis [x0,s,x2] is
/// non-singular so the LU commits. But FTRAN of s gives d=[1,δ,−1], so
/// |d[1]|/max|d| = δ < PIVOT_STABILITY_THRESHOLD=0.01 → batch rejected, fallback fires
/// (counter++), sequential BTRAN picks the well-conditioned x1, solve reaches Optimal
/// obj=0. No-op proof: drop the stability check → batch accepted, fallback never fires,
/// counter stays zero → assertion fails. The LP is near Ruiz-normal (row/col maxes
/// ≤ 1+δ) so the instability survives scaling.
#[test]
fn batch_pivot_out_falls_back_to_sequential_for_ill_conditioned_basis() {
    // δ: FTRAN cancellation ratio at row 1 (δ/1 = 0.001 < threshold 0.01).
    // Also equals the trial-basis determinant contribution ensuring LU succeeds.
    const DELTA: f64 = 0.001;

    // A (3×4):
    //   col0=x0=[1,1,1], col1=x1=[1,-0.5,0], col2=x2=[0.7,0,1], col3=s=[1,1+δ,0]
    let a = CscMatrix::from_triplets(
        &[0, 1, 2,  0, 1,  0, 2,  0, 1],
        &[0, 0, 0,  1, 1,  2, 2,  3, 3],
        &[1.0, 1.0, 1.0,  1.0, -0.5,  0.7, 1.0,  1.0, 1.0 + DELTA],
        3,
        4,
    )
    .unwrap();

    let lp = LpProblem::new_general(
        vec![0.0; 4],
        a,
        vec![1.0; 3],
        vec![ConstraintType::Eq; 3],
        vec![(0.0, f64::INFINITY); 4],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.use_lp_crash_basis = false;
    opts.simplex_method = SimplexMethod::Primal;

    let fallback_before = primal::PIVOT_OUT_SEQUENTIAL_FALLBACK_COUNT.with(|c| c.get());

    let result = solve_with(&lp, &opts);

    let fallback_after = primal::PIVOT_OUT_SEQUENTIAL_FALLBACK_COUNT.with(|c| c.get());

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "LP with ill-conditioned batch must reach Optimal via sequential fallback; got {:?}",
        result.status
    );
    assert!(
        result.objective.abs() < 1e-8,
        "trivial zero-cost objective must be 0.0; got {}",
        result.objective
    );
    assert!(
        fallback_after > fallback_before,
        "FTRAN stability check must detect ill-conditioned batch (ratio δ={DELTA} < \
         PIVOT_STABILITY_THRESHOLD=0.01) and trigger sequential fallback; \
         PIVOT_OUT_SEQUENTIAL_FALLBACK_COUNT must increase \
         (before={fallback_before}, after={fallback_after}). \
         No-op: removing the stability check keeps this at zero — assertion fails."
    );
}
