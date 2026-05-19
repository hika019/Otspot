//! Model API LP path 拡張テスト
//!
//! - LP path で dual_solution / reduced_costs が populate されること
//! - set_presolve(false) で presolve OFF にしても正答に到達すること

use solver::model::{constraint, Model};

const TOL: f64 = 1e-5;

/// min x  s.t. x >= 1, x in [0, 10]
/// 最適: x = 1, obj = 1, Ge 制約 active → dual != 0
#[test]
fn model_api_lp_returns_dual_solution() {
    let mut model = Model::new("lp_dual_test");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!((x) >= 1.0));
    model.minimize(x);
    let result = model.solve().expect("LP solve");

    assert!(
        (result.objective_value - 1.0).abs() < TOL,
        "obj expected 1.0, got {}",
        result.objective_value
    );

    let dual = result
        .dual_solution
        .as_ref()
        .expect("LP dual_solution must be Some");
    assert_eq!(dual.len(), 1, "1 制約 → dual length 1");
    // active Ge 制約に対する shadow price は非ゼロ (sign convention は実観測)。
    assert!(
        dual[0].abs() > TOL,
        "active Ge constraint dual should be non-zero, got {}",
        dual[0]
    );
}

/// reduced_costs が変数数分 populate されることを確認。
#[test]
fn model_api_lp_returns_reduced_costs() {
    let mut model = Model::new("lp_rc_test");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!((x) >= 1.0));
    model.minimize(x);
    let result = model.solve().expect("LP solve");

    let rc = result
        .reduced_costs
        .as_ref()
        .expect("LP reduced_costs must be Some");
    assert_eq!(rc.len(), 1, "rc length must equal num_vars");
}

/// set_presolve(false) でも正答に到達することを確認。
#[test]
fn model_api_set_presolve_off() {
    let mut model = Model::new("lp_presolve_off");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint(constraint!((x) >= 1.0));
    model.minimize(x);
    model.set_presolve(false);
    let result = model.solve().expect("LP solve presolve=off");
    assert!(
        (result.objective_value - 1.0).abs() < TOL,
        "presolve OFF でも opt 1.0 に到達すべき, got {}",
        result.objective_value
    );
}

/// 2 制約 LP: min x + 2y s.t. x + y >= 3, 2x + 3y <= 12, x in [0,inf), y in [0,10]
/// 最適: x=3, y=0, obj=3, Ge active, Le inactive
#[test]
fn model_api_lp_dual_two_constraints() {
    let mut model = Model::new("lp_dual_two");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
    model.add_constraint((x + y).geq(3.0));
    model.minimize(x + 2.0 * y);
    let result = model.solve().expect("LP solve");

    assert!(
        (result.objective_value - 3.0).abs() < 1e-4,
        "obj expected 3.0, got {}",
        result.objective_value
    );

    let dual = result.dual_solution.as_ref().expect("dual must be Some");
    assert_eq!(dual.len(), 2, "2 constraints → dual length 2");

    let slack = result.slack.as_ref().expect("slack must be Some");
    assert_eq!(slack.len(), 2, "2 constraints → slack length 2");

    let rc = result.reduced_costs.as_ref().expect("rc must be Some");
    assert_eq!(rc.len(), 2, "2 vars → rc length 2");
}

/// presolve=true (default) と presolve=false で同じ最適値を返すこと。
#[test]
fn model_api_set_presolve_on_off_agree() {
    let build = || {
        let mut m = Model::new("agree");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        let y = m.add_var("y", 0.0, 10.0);
        m.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
        m.add_constraint((x + y).geq(3.0));
        m.minimize(x + 2.0 * y);
        (m, x, y)
    };

    let (mut m_on, _, _) = build();
    m_on.set_presolve(true);
    let r_on = m_on.solve().expect("presolve=on solve");

    let (mut m_off, _, _) = build();
    m_off.set_presolve(false);
    let r_off = m_off.solve().expect("presolve=off solve");

    assert!(
        (r_on.objective_value - r_off.objective_value).abs() < 1e-4,
        "presolve on/off で obj 不一致: on={}, off={}",
        r_on.objective_value, r_off.objective_value
    );
}
