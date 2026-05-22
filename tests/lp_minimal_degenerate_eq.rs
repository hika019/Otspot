//! `check_eq_feasibility` 過剰発火 regression guard。
//! 退化 Eq 制約を持つ小規模 LP で Optimal が返ることを assert する
//! (相対閾値 `feas_rel_tol() * (1 + |b| + |Ax|)` の scale 非依存性検証)。

use otspot::constraint;
use otspot::model::{Model, ModelError, SolveError};

const EPS_KKT: f64 = 1e-6;

/// 退化 Eq 制約を持つ小規模 Model の構築。
///
/// 構造:
/// - n_eq_zero 個の Eq 制約「coeff_i * x_i = 0」(退化頂点)
/// - 1 個の active Eq 「x_{n_eq_zero} = value_last」、目的は min x_{n_eq_zero}
/// - scale_mix 時 i=0 は 1e3、i=1 は 1e-3 で数値条件数悪化
fn build_degenerate_eq_model(n_eq_zero: usize, value_last: f64, scale_mix: bool) -> Model {
    let label = format!("degen_eq_n{}_v{}", n_eq_zero, value_last);
    let mut model = Model::new(&label);
    let n_total = n_eq_zero + 1;
    let vars: Vec<_> = (0..n_total)
        .map(|i| model.add_var(&format!("x{}", i), 0.0, f64::INFINITY))
        .collect();

    for i in 0..n_eq_zero {
        let coeff = if scale_mix && i == 0 { 1e3 }
                    else if scale_mix && i == 1 { 1e-3 }
                    else { 1.0 };
        model.add_constraint(constraint!((coeff * vars[i]) == 0.0));
    }
    let last = vars[n_eq_zero];
    model.add_constraint(constraint!((1.0 * last) == value_last));
    model.minimize(last);
    model
}

fn assert_optimal_with_value(model: &mut Model, expected: f64, label: &str) {
    let r = model
        .solve()
        .unwrap_or_else(|e| panic!("[{}] expected Optimal, got {:?} (check_eq_feasibility 過剰発火の疑い)", label, e));
    eprintln!("[{}] obj={:.6e} expected={:.6e}", label, r.objective_value, expected);
    let obj_err = (r.objective_value - expected).abs() / (1.0 + expected.abs());
    assert!(obj_err < EPS_KKT, "[{}] obj err {:.3e}", label, obj_err);
}

fn assert_infeasible(model: &mut Model, label: &str) {
    let err = model.solve().unwrap_err();
    eprintln!("[{}] err={:?}", label, err);
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "[{}] expected Infeasible, got {:?}", label, err
    );
}

/// 3 個の退化 Eq 制約 (x0=x1=x2=0) + 1 個の active Eq (x3=5)。
#[test]
fn bug5a_degenerate_eq_simple() {
    let mut model = build_degenerate_eq_model(3, 5.0, false);
    assert_optimal_with_value(&mut model, 5.0, "bug5a_degen_eq_simple");
}

/// 5 個の退化 Eq + scale mix (1e3 / 1e-3) で数値条件数を悪化。
#[test]
fn bug5b_degenerate_eq_scale_mix() {
    let mut model = build_degenerate_eq_model(5, 1e-3, true);
    assert_optimal_with_value(&mut model, 1e-3, "bug5b_degen_eq_scale_mix");
}

/// 8 個の退化 Eq + 単純値。退化数の上限ストレステスト。
#[test]
fn bug5c_degenerate_eq_many() {
    let mut model = build_degenerate_eq_model(8, 1.0, false);
    assert_optimal_with_value(&mut model, 1.0, "bug5c_degen_eq_many");
}

/// **境界条件**: Eq 制約 1 行で `|Ax - b|` が exactly 0 (理想 case)。
#[test]
fn bug5d_single_eq_constraint_clean() {
    // min x + y, s.t. x + y = 3, x,y >= 0
    let mut model = Model::new("bug5d_eq_clean");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    model.add_constraint(constraint!((x + y) == 3.0));
    model.minimize(x + y);
    assert_optimal_with_value(&mut model, 3.0, "bug5d_single_eq_clean");
}

/// 大きい b スケール (b=1e6) で退化系の数値ノイズが false NumericalError
/// 化されないこと。相対閾値 `feas_rel_tol() * (1 + |b| + |Ax|)` の scale
/// 非依存性 regression guard。
#[test]
fn bug5e_large_b_scale_degenerate() {
    // 5 退化 Eq (x_i = 0) + 1 active Eq (x_5 = 1e6)
    let mut model = Model::new("bug5e_large_b_scale");
    let vars: Vec<_> = (0..6)
        .map(|i| model.add_var(&format!("x{}", i), 0.0, f64::INFINITY))
        .collect();
    for i in 0..5 {
        model.add_constraint(constraint!((1.0 * vars[i]) == 0.0));
    }
    let x5 = vars[5];
    model.add_constraint(constraint!((1.0 * x5) == 1e6));
    model.minimize(x5);
    assert_optimal_with_value(&mut model, 1e6, "bug5e_large_b_scale_degenerate");
}

/// 大スケール解 (|x|≈1e6) で `|Ax|` も大きいときの LU 残差を相対閾値で許容。
#[test]
fn bug5f_large_solution_scale() {
    // min -x s.t. x = 1e6, x >= 0
    let mut model = Model::new("bug5f_large_x");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.add_constraint(constraint!(x == 1e6));
    model.minimize(-1.0 * x);
    assert_optimal_with_value(&mut model, -1e6, "bug5f_large_solution_scale");
}

// ge/eq cold start infeasible 検出 sanity (klein-style row 矛盾)。

/// 単純 infeasible: x >= 3 と x <= 1 の矛盾 (mini smoke)。
#[test]
fn bug6a_simple_infeasible() {
    let mut model = Model::new("bug6a_simple_inf");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!(x >= 3.0));
    model.add_constraint(constraint!(x <= 1.0));
    model.minimize(x);
    assert_infeasible(&mut model, "bug6a");
}

/// ge / eq 混在 infeasible: x + y >= 5, x + y = 2 (klein-style row 矛盾)。
#[test]
fn bug6b_ge_eq_mix_infeasible() {
    let mut model = Model::new("bug6b_ge_eq_inf");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    model.add_constraint(constraint!((x + y) >= 5.0));
    model.add_constraint(constraint!((x + y) == 2.0));
    model.minimize(x + y);
    assert_infeasible(&mut model, "bug6b");
}

/// 3 var / 3 row klein-style infeasible: 等式系で over-determined。
#[test]
fn bug6c_overdetermined_eq_infeasible() {
    // x + y + z = 5; x + y = 3; z = 3  → z=2 と z=3 矛盾
    let mut model = Model::new("bug6c_overdet_eq");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    let z = model.add_var("z", 0.0, f64::INFINITY);
    model.add_constraint(constraint!((x + y + z) == 5.0));
    model.add_constraint(constraint!((x + y) == 3.0));
    model.add_constraint(constraint!(z == 3.0));
    model.minimize(x + y + z);
    assert_infeasible(&mut model, "bug6c");
}
