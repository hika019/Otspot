//! API正確性テスト: LP/QP の解析解と Model API ソルバー出力を直接比較。
//!
//! `tests/model_api_crosscheck.rs` の「両経路一致テスト」とは異なり、解析的に
//! 計算した既知の正解を直接 assert することで、両経路に共通するバグも検出する。
//! QP は `min ½ xᵀQx + cᵀx` 規約 (Model API と一致)。

use otspot::constraint;
use otspot::model::{Model, ModelError, ModelResult, SolveError};
use otspot::sparse::CscMatrix;

const INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;
/// 目的関数値の許容誤差 (相対誤差ベース)
const TOL: f64 = 1e-6;
/// 解ベクトルの許容誤差
const TOL_X: f64 = 1e-5;

fn check_optimal(result: &ModelResult, expected_obj: f64, label: &str) {
    assert!(
        (result.objective_value - expected_obj).abs() < TOL * (1.0 + expected_obj.abs()),
        "{}: obj={:.9e} expected={:.9e} diff={:.3e}",
        label,
        result.objective_value,
        expected_obj,
        (result.objective_value - expected_obj).abs()
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

fn assert_infeasible(err: ModelError, label: &str) {
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "{}: expected Infeasible, got {:?}",
        label,
        err
    );
}

fn assert_unbounded(err: ModelError, label: &str) {
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Unbounded)),
        "{}: expected Unbounded, got {:?}",
        label,
        err
    );
}

// ===========================================================================
// LP テスト
// ===========================================================================

#[test]
fn lp_trivial_bound() {
    // min x  s.t. x ≥ 1, 0 ≤ x ≤ 10
    let mut model = Model::new("lp_trivial_bound");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!(x >= 1.0));
    model.minimize(x);
    let result = model.solve().expect("lp_trivial_bound solve");

    check_optimal(&result, 1.0, "lp_trivial_bound");
    check_sol_elem(result[x], 1.0, TOL_X, "lp_trivial_bound x");
}

/// 退化頂点: (2,0) or (0,2) 両方 obj=2 (assert は x+y=2 のみチェック)。
#[test]
fn lp_ge_constraint() {
    // min x+y  s.t. x+y ≥ 2, x,y ≥ 0
    let mut model = Model::new("lp_ge_constraint");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) >= 2.0));
    model.minimize(x + y);
    let result = model.solve().expect("lp_ge_constraint solve");

    check_optimal(&result, 2.0, "lp_ge_constraint");
    let sum = result[x] + result[y];
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
    let mut model = Model::new("lp_eq_constraint");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) == 3.0));
    model.minimize(x + 2.0 * y);
    let result = model.solve().expect("lp_eq_constraint solve");

    check_optimal(&result, 3.0, "lp_eq_constraint");
    check_sol_elem(result[x], 3.0, TOL_X, "lp_eq_constraint x");
    check_sol_elem(result[y], 0.0, TOL_X, "lp_eq_constraint y");
}

/// maximize を minimize(-(x+y)) で表現; 真の max x+y=4, 内部 obj=-4。
#[test]
fn lp_maximize() {
    // min -(x+y)  s.t. x+y ≤ 4, 0 ≤ x,y ≤ 3
    let mut model = Model::new("lp_maximize");
    let x = model.add_var("x", 0.0, 3.0);
    let y = model.add_var("y", 0.0, 3.0);
    model.add_constraint(constraint!((x + y) <= 4.0));
    model.minimize(-1.0 * x - 1.0 * y);
    let result = model.solve().expect("lp_maximize solve");

    check_optimal(&result, -4.0, "lp_maximize (minimize -(x+y))");
    let sum = result[x] + result[y];
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
    let mut model = Model::new("lp_two_constraints");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.add_constraint(constraint!((x + 2.0 * y) >= 4.0));
    model.minimize(2.0 * x + y);
    let result = model.solve().expect("lp_two_constraints solve");

    check_optimal(&result, 3.0, "lp_two_constraints");
    check_sol_elem(result[x], 0.0, TOL_X, "lp_two_constraints x");
    check_sol_elem(result[y], 3.0, TOL_X, "lp_two_constraints y");
}

/// intentionally infeasible: x≥3 ∧ x≤1。
#[test]
fn lp_infeasible() {
    // min x  s.t. x ≥ 3, x ≤ 1, 0 ≤ x ≤ 10
    let mut model = Model::new("lp_infeasible");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!(x >= 3.0));
    model.add_constraint(constraint!(x <= 1.0));
    model.minimize(x);
    let err = model.solve().unwrap_err();
    assert_infeasible(err, "lp_infeasible");
}

/// intentionally unbounded: c=-1 + x≥0 で x→+∞。
#[test]
fn lp_unbounded() {
    // min -x  s.t. x ≥ 0 (no upper)
    let mut model = Model::new("lp_unbounded");
    let x = model.add_var("x", 0.0, INF);
    model.minimize(-1.0 * x);
    let err = model.solve().unwrap_err();
    assert_unbounded(err, "lp_unbounded");
}

/// 等式 2 本連立 (x+y=1, x-y=1) で unique 解 (1,0), obj=1。
#[test]
fn lp_degenerate() {
    // min x  s.t. x+y = 1, x-y = 1, x,y free
    let mut model = Model::new("lp_degenerate");
    let x = model.add_var("x", NEG_INF, INF);
    let y = model.add_var("y", NEG_INF, INF);
    model.add_constraint(constraint!((x + y) == 1.0));
    model.add_constraint(constraint!((x - y) == 1.0));
    model.minimize(x);
    let result = model.solve().expect("lp_degenerate solve");

    check_optimal(&result, 1.0, "lp_degenerate");
    check_sol_elem(result[x], 1.0, TOL_X, "lp_degenerate x");
    check_sol_elem(result[y], 0.0, TOL_X, "lp_degenerate y");
}

// ===========================================================================
// QP テスト
//
// Model API は `set_quadratic_objective(CscMatrix)` で Q を直接渡す
// (frontend 拡張は scope 外 / follow-up)。線形項は `minimize()` で。
// ===========================================================================

/// min (x-1)²+(y-2)² の bound 内 unconstrained 最小; opt (1,2), obj=-5 (定数+5 除く)。
#[test]
fn qp_unconstrained_quadratic() {
    // ½ xᵀQx + cᵀx with Q=diag(2,2), c=(-2,-4) → f = x²+y²-2x-4y
    let mut model = Model::new("qp_unconstrained_quadratic");
    let x = model.add_var("x", -10.0, 10.0);
    let y = model.add_var("y", -10.0, 10.0);
    model.minimize(-2.0 * x - 4.0 * y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("qp_unconstrained_quadratic solve");

    check_optimal(&result, -5.0, "qp_unconstrained_quadratic");
    check_sol_elem(result[x], 1.0, TOL_X, "qp_unconstrained_quadratic x");
    check_sol_elem(result[y], 2.0, TOL_X, "qp_unconstrained_quadratic y");
}

/// KKT: 対称な 2x+λ=0 から x=y=0.5, λ=-1, obj=0.5。
#[test]
fn qp_eq_constrained() {
    // min x²+y²  s.t. x+y = 1, x,y ≥ 0
    let mut model = Model::new("qp_eq_constrained");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) == 1.0));
    model.minimize(0.0 * x + 0.0 * y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("qp_eq_constrained solve");

    check_optimal(&result, 0.5, "qp_eq_constrained");
    check_sol_elem(result[x], 0.5, TOL_X, "qp_eq_constrained x");
    check_sol_elem(result[y], 0.5, TOL_X, "qp_eq_constrained y");
}

/// active Ge 制約; opt (1,1), obj=2。
#[test]
fn qp_ineq_constrained() {
    // min x²+y²  s.t. x+y ≥ 2, x,y ≥ 0
    let mut model = Model::new("qp_ineq_constrained");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) >= 2.0));
    model.minimize(0.0 * x + 0.0 * y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("qp_ineq_constrained solve");

    check_optimal(&result, 2.0, "qp_ineq_constrained");
    check_sol_elem(result[x], 1.0, TOL_X, "qp_ineq_constrained x");
    check_sol_elem(result[y], 1.0, TOL_X, "qp_ineq_constrained y");
}

/// bounds のみ; opt 原点だが IPM barrier は lb=0 に完全到達せず obj≈0 (許容 5e-6)。
#[test]
fn qp_bounds_only() {
    // min x²+y²  s.t. 0 ≤ x,y ≤ 0.5
    let mut model = Model::new("qp_bounds_only");
    let x = model.add_var("x", 0.0, 0.5);
    let y = model.add_var("y", 0.0, 0.5);
    model.minimize(0.0 * x + 0.0 * y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("qp_bounds_only solve");

    // IPM バリア法では lb=0 に完全収束しない。5e-6 を許容 (obj=x²+y²)。
    assert!(
        result.objective_value.abs() < 5e-6,
        "qp_bounds_only: obj={:.6e} should be close to 0",
        result.objective_value
    );
    assert!(
        result[x] >= -1e-5,
        "qp_bounds_only: x={:.6e} should be >= 0 (lb bound)",
        result[x]
    );
    assert!(
        result[y] >= -1e-5,
        "qp_bounds_only: y={:.6e} should be >= 0 (lb bound)",
        result[y]
    );
}

/// linear c=[1,1] と等式制約の組合せ; opt (1,1), obj=4 (λ=-3)。
#[test]
fn qp_with_linear() {
    // min x²+y² + x+y  s.t. x+y = 2, x,y ≥ 0
    let mut model = Model::new("qp_with_linear");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) == 2.0));
    model.minimize(x + y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("qp_with_linear solve");

    check_optimal(&result, 4.0, "qp_with_linear");
    check_sol_elem(result[x], 1.0, TOL_X, "qp_with_linear x");
    check_sol_elem(result[y], 1.0, TOL_X, "qp_with_linear y");
}

/// intentionally infeasible: x≥2 ∧ x≤1。
#[test]
fn qp_infeasible() {
    // min x²  s.t. x ≥ 2, x ≤ 1, x free
    let mut model = Model::new("qp_infeasible");
    let x = model.add_var("x", NEG_INF, INF);
    model.add_constraint(constraint!(x >= 2.0));
    model.add_constraint(constraint!(x <= 1.0));
    model.minimize(0.0 * x);
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    model.set_quadratic_objective(q);
    let err = model.solve().unwrap_err();
    assert_infeasible(err, "qp_infeasible");
}

// ===========================================================================
// 双対変数の KKT 検証
//
// LP path は dual_solution が None (現状未供給)。QP path は Some(Vec) を返す。
// 双対が無い場合の sanity 確認はスキップ。
// ===========================================================================

/// dual KKT 検証: lp_ge_constraint の y≥0 (shadow price) と c−Aᵀy=reduced_cost≥0 を確認。
#[test]
fn dual_lp_ge_constraint() {
    // min x+y  s.t. x+y ≥ 2, x,y ≥ 0
    let mut model = Model::new("dual_lp_ge");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) >= 2.0));
    model.minimize(x + y);
    let result = model.solve().expect("dual_lp_ge solve");

    if let Some(dual) = &result.dual_solution {
        let c = [1.0_f64, 1.0];
        assert!(
            dual[0] >= -TOL,
            "dual_lp_ge: y[0]={:.6e} should be >= 0 (shadow price of Ge constraint)",
            dual[0]
        );
        let rc0 = c[0] - dual[0];
        let rc1 = c[1] - dual[0];
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
    let mut model = Model::new("dual_qp_eq");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) == 1.0));
    model.minimize(0.0 * x + 0.0 * y);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    let result = model.solve().expect("dual_qp_eq solve");

    if let Some(dual) = &result.dual_solution {
        let lam = dual[0];
        // KKT: Q*x + c + A^T*λ ≈ 0 → 2x[i] + λ
        let kkt0 = 2.0 * result[x] + lam;
        let kkt1 = 2.0 * result[y] + lam;
        let kkt_tol = 1e-4;
        assert!(
            kkt0.abs() < kkt_tol,
            "dual_qp_eq: KKT residual x: {:.6e} (expected ~0), x={:.6e}, λ={:.6e}",
            kkt0,
            result[x],
            lam
        );
        assert!(
            kkt1.abs() < kkt_tol,
            "dual_qp_eq: KKT residual y: {:.6e} (expected ~0), y={:.6e}, λ={:.6e}",
            kkt1,
            result[y],
            lam
        );
    }
}

/// dual KKT 検証: lp_two_constraints で primal 実行可能性 sanity check。
#[test]
fn dual_lp_two_constraints() {
    // min 2x+y  s.t. x+y ≥ 3, x+2y ≥ 4, x,y ≥ 0
    let mut model = Model::new("dual_lp_two");
    let x = model.add_var("x", 0.0, INF);
    let y = model.add_var("y", 0.0, INF);
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.add_constraint(constraint!((x + 2.0 * y) >= 4.0));
    model.minimize(2.0 * x + y);
    let result = model.solve().expect("dual_lp_two solve");

    check_sol_elem(result[x], 0.0, 1e-4, "dual_lp_two x");
    check_sol_elem(result[y], 3.0, 1e-4, "dual_lp_two y");

    assert!(
        result[x] + result[y] >= 3.0 - 1e-4,
        "dual_lp_two: constraint 1 violated: x+y={:.6e} < 3",
        result[x] + result[y]
    );
    assert!(
        result[x] + 2.0 * result[y] >= 4.0 - 1e-4,
        "dual_lp_two: constraint 2 violated: x+2y={:.6e} < 4",
        result[x] + 2.0 * result[y]
    );
}
