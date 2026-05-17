//! API正確性テスト: LP/QP の解析解と実際の解を突き合わせる
//!
//! `tests/model_api_crosscheck.rs` の「両経路一致テスト」とは異なり、
//! 解析的に計算した既知の正解と実際のソルバー出力を直接比較する。
//! 両経路に同じバグがある場合でも検出可能。
//!
//! 問題は `tests/common/` の declarative builder で組み立て、math と 1:1 対応させる
//! (raw triplet 列の人間検証不能性を解消)。`QpProblem` は `min ½ xᵀQx + cᵀx` 規約。

mod common;

use common::{eq, ge, le, lp, qp_diag, INF, NEG_INF};
use solver::model::Model;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use solver::sparse::CscMatrix;

/// 目的関数値の許容誤差 (相対誤差ベース)
const TOL: f64 = 1e-6;
/// 解ベクトルの許容誤差
const TOL_X: f64 = 1e-5;

fn check_optimal(result: &solver::problem::SolverResult, expected_obj: f64, label: &str) {
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "{}: expected Optimal, got {:?}",
        label,
        result.status
    );
    assert!(
        (result.objective - expected_obj).abs() < TOL * (1.0 + expected_obj.abs()),
        "{}: obj={:.9e} expected={:.9e} diff={:.3e}",
        label,
        result.objective,
        expected_obj,
        (result.objective - expected_obj).abs()
    );
}

fn check_sol_elem(x: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (x - expected).abs() < tol * (1.0 + expected.abs()),
        "{}: x={:.9e} expected={:.9e} diff={:.3e}",
        label,
        x,
        expected,
        (x - expected).abs()
    );
}

// ===========================================================================
// LP テスト
// ===========================================================================

#[test]
fn lp_trivial_bound() {
    // min x  s.t. x ≥ 1, 0 ≤ x ≤ 10
    let prob = lp(&[1.0], &[ge(&[1.0], 1.0)], &[(0.0, 10.0)]);
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 1.0, "lp_trivial_bound");
    check_sol_elem(result.solution[0], 1.0, TOL_X, "lp_trivial_bound x");
}

/// 退化頂点: (2,0) or (0,2) 両方 obj=2 (assert は x+y=2 のみチェック)。
#[test]
fn lp_ge_constraint() {
    // min x+y  s.t. x+y ≥ 2, x,y ≥ 0
    let prob = lp(
        &[1.0, 1.0],
        &[ge(&[1.0, 1.0], 2.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 2.0, "lp_ge_constraint");
    let sum = result.solution[0] + result.solution[1];
    assert!(
        (sum - 2.0).abs() < 1e-5,
        "lp_ge_constraint: x+y={:.6e} expected=2.0",
        sum
    );
}

/// c_x=1<c_y=2 で x 優先 → opt (3,0), obj=3 (端点解)。
#[test]
fn lp_eq_constraint() {
    // min x+2y  s.t. x+y = 3, x,y ≥ 0
    let prob = lp(
        &[1.0, 2.0],
        &[eq(&[1.0, 1.0], 3.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 3.0, "lp_eq_constraint");
    check_sol_elem(result.solution[0], 3.0, TOL_X, "lp_eq_constraint x");
    check_sol_elem(result.solution[1], 0.0, TOL_X, "lp_eq_constraint y");
}

/// maximize は c 符号反転で min 化 (c=[-1,-1]); 真の max x+y=4, 内部 obj=-4。
#[test]
fn lp_maximize() {
    // min -(x+y)  s.t. x+y ≤ 4, 0 ≤ x,y ≤ 3
    let prob = lp(
        &[-1.0, -1.0],
        &[le(&[1.0, 1.0], 4.0)],
        &[(0.0, 3.0), (0.0, 3.0)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, -4.0, "lp_maximize (minimize -(x+y))");
    let sum = result.solution[0] + result.solution[1];
    assert!(
        (sum - 4.0).abs() < 1e-5,
        "lp_maximize: x+y={:.6e} expected=4.0",
        sum
    );
}

/// 頂点列挙の最小: opt (0,3), obj=3 (他頂点 (2,1) obj=5, (0,4) obj=4)。
#[test]
fn lp_two_constraints() {
    // min 2x+y  s.t. x+y ≥ 3, x+2y ≥ 4, x,y ≥ 0
    let prob = lp(
        &[2.0, 1.0],
        &[ge(&[1.0, 1.0], 3.0), ge(&[1.0, 2.0], 4.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 3.0, "lp_two_constraints");
    check_sol_elem(result.solution[0], 0.0, TOL_X, "lp_two_constraints x");
    check_sol_elem(result.solution[1], 3.0, TOL_X, "lp_two_constraints y");
}

/// intentionally infeasible: x≥3 ∧ x≤1。
#[test]
fn lp_infeasible() {
    // min x  s.t. x ≥ 3, x ≤ 1, 0 ≤ x ≤ 10
    let prob = lp(
        &[1.0],
        &[ge(&[1.0], 3.0), le(&[1.0], 1.0)],
        &[(0.0, 10.0)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "lp_infeasible: expected Infeasible, got {:?}",
        result.status
    );
}

/// intentionally unbounded: c=-1 + x≥0 で x→+∞。
#[test]
fn lp_unbounded() {
    // min -x  s.t. x ≥ 0 (no upper)
    let prob = lp(&[-1.0], &[], &[(0.0, INF)]);
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(
        result.status,
        SolveStatus::Unbounded,
        "lp_unbounded: expected Unbounded, got {:?}",
        result.status
    );
}

/// 等式 2 本連立 (x+y=1, x-y=1) で unique 解 (1,0), obj=1。
#[test]
fn lp_degenerate() {
    // min x  s.t. x+y = 1, x-y = 1, x,y free
    let prob = lp(
        &[1.0, 0.0],
        &[eq(&[1.0, 1.0], 1.0), eq(&[1.0, -1.0], 1.0)],
        &[(NEG_INF, INF), (NEG_INF, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 1.0, "lp_degenerate");
    check_sol_elem(result.solution[0], 1.0, TOL_X, "lp_degenerate x");
    check_sol_elem(result.solution[1], 0.0, TOL_X, "lp_degenerate y");
}

// ===========================================================================
// QP テスト
// ===========================================================================

/// min (x-1)²+(y-2)² の bound 内 unconstrained 最小; opt (1,2), obj=-5 (定数+5 除く)。
#[test]
fn qp_unconstrained_quadratic() {
    // ½ xᵀQx + cᵀx with Q=diag(2,2), c=(-2,-4) → f = x²+y²-2x-4y
    let prob = qp_diag(
        &[2.0, 2.0],
        &[-2.0, -4.0],
        &[],
        &[(-10.0, 10.0), (-10.0, 10.0)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, -5.0, "qp_unconstrained_quadratic");
    check_sol_elem(result.solution[0], 1.0, TOL_X, "qp_unconstrained_quadratic x");
    check_sol_elem(result.solution[1], 2.0, TOL_X, "qp_unconstrained_quadratic y");
}

/// KKT: 対称な 2x+λ=0 から x=y=0.5, λ=-1, obj=0.5。
#[test]
fn qp_eq_constrained() {
    // min x²+y²  s.t. x+y = 1, x,y ≥ 0
    let prob = qp_diag(
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[eq(&[1.0, 1.0], 1.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 0.5, "qp_eq_constrained");
    check_sol_elem(result.solution[0], 0.5, TOL_X, "qp_eq_constrained x");
    check_sol_elem(result.solution[1], 0.5, TOL_X, "qp_eq_constrained y");
}

/// active Ge 制約; opt (1,1), obj=2。
#[test]
fn qp_ineq_constrained() {
    // min x²+y²  s.t. x+y ≥ 2, x,y ≥ 0
    let prob = qp_diag(
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[ge(&[1.0, 1.0], 2.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 2.0, "qp_ineq_constrained");
    check_sol_elem(result.solution[0], 1.0, TOL_X, "qp_ineq_constrained x");
    check_sol_elem(result.solution[1], 1.0, TOL_X, "qp_ineq_constrained y");
}

/// bounds のみ; opt 原点だが IPM barrier は lb=0 に完全到達せず obj≈0 (許容 5e-6)。
#[test]
fn qp_bounds_only() {
    // min x²+y²  s.t. 0 ≤ x,y ≤ 0.5
    let prob = qp_diag(&[2.0, 2.0], &[0.0, 0.0], &[], &[(0.0, 0.5), (0.0, 0.5)]);
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "qp_bounds_only: expected Optimal, got {:?}",
        result.status
    );
    // IPM バリア法では lb=0 に完全収束しない。5e-6 を許容 (obj=x²+y²)。
    assert!(
        result.objective.abs() < 5e-6,
        "qp_bounds_only: obj={:.6e} should be close to 0",
        result.objective
    );
    assert!(
        result.solution[0] >= -1e-5,
        "qp_bounds_only: x={:.6e} should be >= 0 (lb bound)",
        result.solution[0]
    );
    assert!(
        result.solution[1] >= -1e-5,
        "qp_bounds_only: y={:.6e} should be >= 0 (lb bound)",
        result.solution[1]
    );
}

/// linear c=[1,1] と等式制約の組合せ; opt (1,1), obj=4 (λ=-3)。
#[test]
fn qp_with_linear() {
    // min x²+y² + x+y  s.t. x+y = 2, x,y ≥ 0
    let prob = qp_diag(
        &[2.0, 2.0],
        &[1.0, 1.0],
        &[eq(&[1.0, 1.0], 2.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    check_optimal(&result, 4.0, "qp_with_linear");
    check_sol_elem(result.solution[0], 1.0, TOL_X, "qp_with_linear x");
    check_sol_elem(result.solution[1], 1.0, TOL_X, "qp_with_linear y");
}

/// intentionally infeasible: x≥2 ∧ x≤1。
#[test]
fn qp_infeasible() {
    // min x²  s.t. x ≥ 2, x ≤ 1, x free
    let prob = qp_diag(
        &[2.0],
        &[0.0],
        &[ge(&[1.0], 2.0), le(&[1.0], 1.0)],
        &[(NEG_INF, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "qp_infeasible: expected Infeasible, got {:?}",
        result.status
    );
}

// ===========================================================================
// 双対変数の KKT 検証
// ===========================================================================

/// dual KKT 検証: lp_ge_constraint の y≥0 (shadow price) と c−Aᵀy=reduced_cost≥0 を確認。
#[test]
fn dual_lp_ge_constraint() {
    // min x+y  s.t. x+y ≥ 2, x,y ≥ 0
    let c = vec![1.0_f64, 1.0];
    let prob = lp(
        &c,
        &[ge(&[1.0, 1.0], 2.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(result.status, SolveStatus::Optimal, "dual_lp_ge: Optimal");

    if !result.dual_solution.is_empty() {
        let y = &result.dual_solution;
        assert!(
            y[0] >= -TOL,
            "dual_lp_ge: y[0]={:.6e} should be >= 0 (shadow price of Ge constraint)",
            y[0]
        );
        let rc0 = c[0] - y[0];
        let rc1 = c[1] - y[0];
        assert!(
            rc0 >= -1e-5,
            "dual_lp_ge: reduced cost x: c[0]-y[0]={:.6e} should be >= 0",
            rc0
        );
        assert!(
            rc1 >= -1e-5,
            "dual_lp_ge: reduced cost y: c[1]-y[0]={:.6e} should be >= 0",
            rc1
        );
    }
}

/// dual KKT 検証: qp_eq_constrained で Qx+c+Aᵀλ≈0 (期待 x=(0.5,0.5), λ=-1)。
#[test]
fn dual_qp_eq_constrained() {
    let prob = qp_diag(
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[eq(&[1.0, 1.0], 1.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(result.status, SolveStatus::Optimal, "dual_qp_eq: Optimal");

    if !result.dual_solution.is_empty() {
        let x = &result.solution;
        let lam = result.dual_solution[0];

        // KKT: Q*x + c + A^T*λ ≈ 0 → 2x[i] + λ
        let kkt0 = 2.0 * x[0] + lam;
        let kkt1 = 2.0 * x[1] + lam;
        let kkt_tol = 1e-4;
        assert!(
            kkt0.abs() < kkt_tol,
            "dual_qp_eq: KKT residual x: {:.6e} (expected ~0), x[0]={:.6e}, λ={:.6e}",
            kkt0,
            x[0],
            lam
        );
        assert!(
            kkt1.abs() < kkt_tol,
            "dual_qp_eq: KKT residual y: {:.6e} (expected ~0), x[1]={:.6e}, λ={:.6e}",
            kkt1,
            x[1],
            lam
        );
    }
}

/// dual KKT 検証: lp_two_constraints で active row 1 のみ y=1、x=0 で z_lb=1≥0 (主実行可能性も sanity check)。
#[test]
fn dual_lp_two_constraints() {
    // min 2x+y  s.t. x+y ≥ 3, x+2y ≥ 4, x,y ≥ 0
    let prob = lp(
        &[2.0, 1.0],
        &[ge(&[1.0, 1.0], 3.0), ge(&[1.0, 2.0], 4.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let result = solve_qp_with(&prob, &SolverOptions::default());

    assert_eq!(result.status, SolveStatus::Optimal, "dual_lp_two: Optimal");

    check_sol_elem(result.solution[0], 0.0, 1e-4, "dual_lp_two x");
    check_sol_elem(result.solution[1], 3.0, 1e-4, "dual_lp_two y");

    if !result.dual_solution.is_empty() {
        let x = &result.solution;
        assert!(
            x[0] + x[1] >= 3.0 - 1e-4,
            "dual_lp_two: constraint 1 violated: x+y={:.6e} < 3",
            x[0] + x[1]
        );
        assert!(
            x[0] + 2.0 * x[1] >= 4.0 - 1e-4,
            "dual_lp_two: constraint 2 violated: x+2y={:.6e} < 4",
            x[0] + 2.0 * x[1]
        );
    }
}

// ===========================================================================
// Model API 経由テスト (QpProblem 直接構築と同じ結果になることを確認)
// ===========================================================================

/// Model API ↔ builder 一致確認 (lp_trivial_bound)。
#[test]
fn model_api_lp_trivial_bound() {
    let mut model = Model::new("api_trivial");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint((1.0 * x).geq(1.0));
    model.minimize(x);
    let r_api = model.solve().expect("Model API solve");

    let prob = lp(&[1.0], &[ge(&[1.0], 1.0)], &[(0.0, 10.0)]);
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert!(
        (r_api.objective_value - r_direct.objective).abs() < TOL * (1.0 + r_direct.objective.abs()),
        "model_api_trivial: api={:.6e} direct={:.6e}",
        r_api.objective_value,
        r_direct.objective
    );
    assert!(
        (r_api[x] - r_direct.solution[0]).abs() < TOL_X,
        "model_api_trivial: x: api={:.6e} direct={:.6e}",
        r_api[x],
        r_direct.solution[0]
    );
    assert!((r_api.objective_value - 1.0).abs() < 1e-6, "model_api_trivial: obj should be 1.0");
    assert!((r_api[x] - 1.0).abs() < 1e-5, "model_api_trivial: x should be 1.0");
}

/// Model API ↔ builder 一致確認 (lp_two_constraints)。
#[test]
fn model_api_lp_two_constraints() {
    let mut model = Model::new("api_two");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint((x + y).geq(3.0));
    model.add_constraint((x + 2.0 * y).geq(4.0));
    model.minimize(2.0 * x + y);
    let r_api = model.solve().expect("Model API solve");

    let prob = lp(
        &[2.0, 1.0],
        &[ge(&[1.0, 1.0], 3.0), ge(&[1.0, 2.0], 4.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert!(
        (r_api.objective_value - r_direct.objective).abs() < TOL * (1.0 + r_direct.objective.abs()),
        "model_api_two: api={:.6e} direct={:.6e}",
        r_api.objective_value,
        r_direct.objective
    );
    assert!((r_api.objective_value - 3.0).abs() < 1e-5, "model_api_two: obj should be 3.0");
    assert!((r_api[x] - 0.0).abs() < 1e-4, "model_api_two: x should be 0.0");
    assert!((r_api[y] - 3.0).abs() < 1e-4, "model_api_two: y should be 3.0");
}

/// Model API ↔ builder 一致確認 (qp_eq_constrained)。
#[test]
fn model_api_qp_eq_constrained() {
    let mut model = Model::new("api_qp_eq");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    // Model 経由は generic Q が必要 (set_quadratic_objective が CscMatrix を取る)
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint((x + y).eq_constraint(1.0));
    model.minimize(0.0 * x + 0.0 * y);
    let r_api = model.solve().expect("Model API QP solve");

    let prob = qp_diag(
        &[2.0, 2.0],
        &[0.0, 0.0],
        &[eq(&[1.0, 1.0], 1.0)],
        &[(0.0, INF), (0.0, INF)],
    );
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert!(
        (r_api.objective_value - r_direct.objective).abs() < TOL * (1.0 + r_direct.objective.abs()),
        "model_api_qp_eq: api={:.6e} direct={:.6e}",
        r_api.objective_value,
        r_direct.objective
    );
    assert!(
        (r_api.objective_value - 0.5).abs() < 1e-5,
        "model_api_qp_eq: obj should be 0.5, got {:.6e}",
        r_api.objective_value
    );
    assert!((r_api[x] - 0.5).abs() < 1e-4, "model_api_qp_eq: x should be 0.5");
    assert!((r_api[y] - 0.5).abs() < 1e-4, "model_api_qp_eq: y should be 0.5");
}

// ===========================================================================
// Builder 等価性 regression: raw triplet vs declarative API
// ===========================================================================

/// builder の `lp(...)` が raw `from_triplets` と同一 `QpProblem` を生成することを
/// `lp_two_constraints` (2 row × 2 col, 4 entry) で sentinel 確認。
#[test]
fn builder_matches_raw_triplet_for_lp_two_constraints() {
    use solver::problem::ConstraintType;
    use solver::qp::QpProblem;

    let prob_api = lp(
        &[2.0, 1.0],
        &[ge(&[1.0, 1.0], 3.0), ge(&[1.0, 2.0], 4.0)],
        &[(0.0, INF), (0.0, INF)],
    );

    let q = CscMatrix::new(2, 2);
    let c = vec![2.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 2.0],
        2,
        2,
    )
    .unwrap();
    let b = vec![3.0, 4.0];
    let bounds = vec![(0.0, INF); 2];
    let cts = vec![ConstraintType::Ge, ConstraintType::Ge];
    let prob_raw = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    assert_eq!(prob_api.num_vars, prob_raw.num_vars);
    assert_eq!(prob_api.num_constraints, prob_raw.num_constraints);
    assert_eq!(prob_api.c, prob_raw.c);
    assert_eq!(prob_api.b, prob_raw.b);
    assert_eq!(prob_api.bounds, prob_raw.bounds);
    assert_eq!(prob_api.constraint_types, prob_raw.constraint_types);
    // CSC: col_ptr, row_ind, values が完全一致
    assert_eq!(prob_api.a.col_ptr, prob_raw.a.col_ptr, "A.col_ptr");
    assert_eq!(prob_api.a.row_ind, prob_raw.a.row_ind, "A.row_ind");
    assert_eq!(prob_api.a.values, prob_raw.a.values, "A.values");
}
