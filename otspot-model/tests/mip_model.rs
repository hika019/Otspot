//! Model API tests for MILP / MIQP paths.
//!
//! Moved from `otspot-core/src/mip/tests.rs` to eliminate the core→model
//! test-only dependency. Tests exercise the `Model` high-level API over the
//! MILP branch-and-bound solver.

use otspot_model::{expression::Expression, Model, ModelError, SolveError};

const EPS: f64 = 1e-4;

#[test]
fn model_add_int_var_maximize_branches() {
    // max x s.t. 2x <= 3, x integer in [0,5] → x = 1, obj = 1.
    let mut m = Model::new("milp_int");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.add_constraint((2.0 * x).leq(3.0));
    m.maximize(x);
    let r = m.solve().unwrap();
    assert!((r.objective() - 1.0).abs() < EPS, "obj={}", r.objective());
    assert!((r[x] - 1.0).abs() < EPS, "x={}", r[x]);
}

#[test]
fn model_binary_knapsack() {
    let mut m = Model::new("knapsack");
    let a = m.add_binary_var("a");
    let b = m.add_binary_var("b");
    let c = m.add_binary_var("c");
    let d = m.add_binary_var("d");
    m.add_constraint((5.0 * a + 7.0 * b + 4.0 * c + 3.0 * d).leq(14.0));
    m.maximize(8.0 * a + 11.0 * b + 6.0 * c + 4.0 * d);
    let r = m.solve().unwrap();
    assert!((r.objective() - 21.0).abs() < EPS, "obj={}", r.objective());
    assert_eq!(
        (r[a].round(), r[b].round(), r[c].round(), r[d].round()),
        (0.0, 1.0, 1.0, 1.0)
    );
}

#[test]
fn model_integer_infeasible_errors() {
    let mut m = Model::new("infeasible");
    let x = m.add_int_var("x", 0.0, 10.0);
    m.add_constraint((1.0 * x).geq(1.2));
    m.add_constraint((1.0 * x).leq(1.8));
    m.minimize(x);
    let err = m.solve().unwrap_err();
    assert!(matches!(err, ModelError::SolveError(SolveError::Infeasible)), "got {err:?}");
}

#[test]
fn model_integer_unbounded_errors() {
    let mut m = Model::new("unbounded");
    let x = m.add_int_var("x", 0.0, f64::INFINITY);
    m.maximize(x);
    let err = m.solve().unwrap_err();
    assert!(matches!(err, ModelError::SolveError(SolveError::Unbounded)), "got {err:?}");
}

#[test]
fn model_convex_miqp_branches_to_integer_optimum() {
    // min x^2 - 5x = 1/2·2·x^2 + (-5)x, x integer in [0,5].
    // Continuous min at x=2.5 (fractional → branch); integer optima x=2 or x=3, obj = -6.
    let mut m = Model::new("convex_miqp");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.set_diagonal_q(&[2.0]);
    m.minimize(-5.0 * x);
    let r = m.solve().unwrap();
    assert!((r.objective() - (-6.0)).abs() < EPS, "obj={}", r.objective());
    let xr = r[x].round();
    assert!(xr == 2.0 || xr == 3.0, "x must be 2 or 3, got {}", r[x]);
    assert!((r[x] - xr).abs() < EPS, "x must be integral: {}", r[x]);
}

#[test]
fn model_nonconvex_miqp_errors() {
    // indefinite Q (negative curvature) → must return ModelError::NonConvex, not silent wrong.
    let cases: &[(&str, &[f64], &[f64])] = &[
        ("single neg", &[-2.0], &[1.0]),
        ("neg-pos-2var", &[-3.0, 2.0], &[0.0, 1.0]),
    ];
    for &(name, q_diag, c_vec) in cases {
        let n = q_diag.len();
        let mut m = Model::new(name);
        let vars: Vec<_> = (0..n).map(|i| m.add_int_var(&format!("x{i}"), 0.0, 5.0)).collect();
        m.set_diagonal_q(q_diag);
        let obj = vars.iter().zip(c_vec).fold(
            Expression::from_constant(0.0),
            |acc, (&v, &c)| acc + c * v,
        );
        m.minimize(obj);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::NonConvex(_)),
            "[{name}] expected ModelError::NonConvex, got {err:?}"
        );
    }
}

#[test]
fn model_mixed_integer_continuous() {
    // x integer in [0,5], y continuous in [0,5], x + y <= 3.5.
    // max(x + y) → Optimum: x=3 (integer), y=0.5 → obj 3.5.
    let mut m = Model::new("mixed");
    let x = m.add_int_var("x", 0.0, 5.0);
    let y = m.add_var("y", 0.0, 5.0);
    m.add_constraint((x + y).leq(3.5));
    m.maximize(x + y);
    let r = m.solve().unwrap();
    assert!((r.objective() - 3.5).abs() < EPS, "obj={}", r.objective());
    assert!((r[x].round() - r[x]).abs() < EPS, "x must be integral, x={}", r[x]);
}
