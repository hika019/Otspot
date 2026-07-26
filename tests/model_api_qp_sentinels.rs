// Synthetic Q / data builders index into multiple parallel vecs by row/col.
// The numeric loops can't be rewritten as a single .iter().enumerate() without
// shuffling the helpers; allow the range-loop pattern here.
#![allow(clippy::needless_range_loop)]

//! Hard-data sentinels (SciPy oracles) for QP solve via the Model expression API.
//!
//! Every expected value is hand-computed / SciPy-derived (independent oracle).
//! No expected value is derived by running the solver.
//!
//! These tests build problems through `otspot::model::Model` /
//! `QuadExpr`/`Expression` rather than `otspot_core::qp::QpProblem` directly
//! (that low-level coverage lives in `otspot-core/tests/blackbox_qp.rs`).
//! They live at the workspace root because `otspot` already depends on both
//! `otspot-core` and `otspot-model`; putting Model-API tests in
//! `otspot-core/tests` would need otspot-core (a foundation crate) to
//! dev-depend back on otspot-model (a layer built on top of it) — a
//! layering violation, and it would pull otspot-model's own dependency
//! closure into otspot-core's test build for no reason.
//!
//! QP convention: min 1/2 x'Qx + c'x  (Q is the full Hessian).
//! IMPORTANT: the solver returns 1/2 x'Qx + c'x WITHOUT any constant term.
//! All expected objectives are the internal QP form (no constant offset).

use otspot::model::{Expression, Model, ModelError, QuadExpr, SolveError, Variable};

const EPS_DUAL: f64 = 1e-4;
const HARD_QP_EPS_OBJ: f64 = 5e-5;
const HARD_QP_EPS_X: f64 = 5e-4;
const HARD_QP_TIMEOUT_SECS: f64 = 10.0;
const HARD_QP_ILL_EXPECTED_OBJ: f64 = -2.749_999_999_985_000_4;
const HARD_QP_MICRO_Q: f64 = 1e-14;
const HARD_QP_MICRO_EXPECTED_OBJ: f64 = -999_999.995;
const HARD_QP_DEGENERATE_EXPECTED_OBJ: f64 = -8.25;
const HARD_QP_DUAL_SIGN_EXPECTED_OBJ: f64 = -3.0;
const HARD_QP_LARGE_N: usize = 50;
const HARD_QP_LARGE_M: usize = 10;
const HARD_QP_LARGE_EXPECTED_OBJ: f64 = 0.113_252_558_203_024_54;

fn hard_qp_expr(vars: &[Variable], terms: &[(usize, f64)]) -> Expression {
    let mut expr = Expression::from_constant(0.0);
    for &(var_idx, coeff) in terms {
        expr = expr + coeff * vars[var_idx];
    }
    expr
}

fn hard_qp_obj(
    vars: &[Variable],
    linear_terms: &[(usize, f64)],
    diag_q: &[(usize, f64)],
    offdiag_q: &[(usize, usize, f64)],
) -> QuadExpr {
    let mut obj: QuadExpr = hard_qp_expr(vars, linear_terms).into();
    for &(idx, q_val) in diag_q {
        obj = obj + (0.5 * q_val) * vars[idx] * vars[idx];
    }
    for &(row, col, q_val) in offdiag_q {
        obj = obj + q_val * vars[row] * vars[col];
    }
    obj
}

fn hard_qp_assert_model_obj(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < HARD_QP_EPS_OBJ,
        "{label}: obj={actual:.12e} expected={expected:.12e} rel={rel:.3e}"
    );
}

fn hard_qp_assert_model_x(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_QP_EPS_X,
        "{label}: x={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
    );
}

fn hard_qp_assert_model_route_qp_ipm(route: impl std::fmt::Debug, label: &str) {
    assert_eq!(
        format!("{route:?}"),
        "QpIpm",
        "{label}: route must be QpIpm"
    );
}

/// Hard QP: ill-scaled Q with a tiny off-diagonal value.
///
/// SciPy oracle:
/// SLSQP on `Q=[[1e10,1e-6],[1e-6,2]], c=[-1e5,-3], bounds=[(0,10)]*2`
/// returned success, fun = -2.7499999999850004, x = [9.99999985e-6, 1.5].
#[test]
fn hard_qp_ill_scaled_q_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_ill_scaled_q_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 10.0), model.add_var("y", 0.0, 10.0)];

    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -1e5), (1, -3.0)],
        &[(0, 1e10), (1, 2.0)],
        &[(0, 1, 1e-6)],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_ILL_EXPECTED_OBJ,
        "hard_qp_ill_scaled_q",
    );
    hard_qp_assert_model_x(r[vars[0]], 1.0e-5, "hard_qp_ill_scaled_q x");
    hard_qp_assert_model_x(r[vars[1]], 1.5, "hard_qp_ill_scaled_q y");
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_ill_scaled_q");
}

/// Hard QP: micro curvature just above sparse DROP_TOL must stay on the QP path.
///
/// SciPy oracle:
/// SLSQP on `min 0.5*1e-14*x^2 - x, 0<=x<=1e6`
/// returned success, fun = -999999.995, x = [1e6].
#[test]
fn hard_qp_micro_curvature_routes_to_qp_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_micro_curvature_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 1_000_000.0);
    let vars = [x];

    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -1.0)],
        &[(0, HARD_QP_MICRO_Q)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_MICRO_EXPECTED_OBJ,
        "hard_qp_micro_curvature",
    );
    hard_qp_assert_model_x(r[x], 1_000_000.0, "hard_qp_micro_curvature x at ub");
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_micro_curvature");
}

/// Hard QP: KKT degeneracy with several active bounds and an equality.
///
/// SciPy oracle:
/// SLSQP on `Q=diag(2,2,2), c=[-4,-4,-1], x+y=4, bounds=(0,2),(0,2),(0,0.5)`
/// returned success, fun = -8.25, x = [2, 2, 0.5].
#[test]
fn hard_qp_kkt_degenerate_active_bounds_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_kkt_degenerate_active_bounds_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![
        model.add_var("x", 0.0, 2.0),
        model.add_var("y", 0.0, 2.0),
        model.add_var("z", 0.0, 0.5),
    ];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(4.0));
    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -4.0), (1, -4.0), (2, -1.0)],
        &[(0, 2.0), (1, 2.0), (2, 2.0)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_DEGENERATE_EXPECTED_OBJ,
        "hard_qp_kkt_degenerate",
    );
    for (idx, expected) in [2.0, 2.0, 0.5].into_iter().enumerate() {
        hard_qp_assert_model_x(
            r[vars[idx]],
            expected,
            &format!("hard_qp_kkt_degenerate x{idx}"),
        );
    }
    assert!(
        !r.bound_duals.is_empty(),
        "hard_qp_kkt_degenerate must return bound duals"
    );
}

/// Hard QP: active Le constraint whose dual sign is a tight optimality sentinel.
///
/// SciPy oracle:
/// SLSQP on `Q=diag(2,0.2), c=[-4,-0.05], x+y<=1, x,y>=0`
/// returned success, fun = -3.0, x = [1, 0].
#[test]
fn hard_qp_dual_sign_le_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_dual_sign_le_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 10.0), model.add_var("y", 0.0, 10.0)];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).leq(1.0));
    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -4.0), (1, -0.05)],
        &[(0, 2.0), (1, 0.2)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_DUAL_SIGN_EXPECTED_OBJ,
        "hard_qp_dual_sign",
    );
    hard_qp_assert_model_x(r[vars[0]], 1.0, "hard_qp_dual_sign x");
    hard_qp_assert_model_x(r[vars[1]], 0.0, "hard_qp_dual_sign y");
    let duals = r.dual_solution.as_ref().expect("hard_qp_dual_sign dual");
    assert!(
        duals[0] >= -EPS_DUAL,
        "hard_qp_dual_sign: Le dual must be non-negative in the Model API convention, got {}",
        duals[0]
    );
}

/// Hard QP: n=50 sparse Q + Eq + UB synthetic stress instance.
///
/// SciPy oracle:
/// deterministic Q/c/A/x_ref below, then SLSQP with equality constraints and
/// `[0,1]` bounds returned success, fun = 0.11325255820302454.
#[test]
fn hard_qp_large_sparse_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_large_sparse_eq_ub_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    let x_ref: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| 0.1 + 0.8 * ((j * 23) % HARD_QP_LARGE_N) as f64 / 49.0)
        .collect();

    for i in 0..HARD_QP_LARGE_M {
        let mut terms = Vec::new();
        let mut rhs = 0.0;
        for j in 0..HARD_QP_LARGE_N {
            if matches!((i * 13 + j * 7) % 11, 0 | 2 | 5) {
                let coeff = ((i + j) % 5) as f64 * 0.1 - 0.2
                    + if i == j % HARD_QP_LARGE_M { 0.05 } else { 0.0 };
                terms.push((j, coeff));
                rhs += coeff * x_ref[j];
            }
        }
        model.add_constraint(hard_qp_expr(&vars, &terms).eq_constraint(rhs));
    }

    let linear_terms: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| (j, ((j % 9) as f64 - 4.0) * 0.02))
        .collect();
    let diag_terms: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| (j, 0.2 + (j % 7) as f64 * 0.05))
        .collect();
    let offdiag_terms: Vec<_> = (0..HARD_QP_LARGE_N - 1)
        .filter(|j| j % 5 == 0)
        .map(|j| (j, j + 1, 0.005))
        .collect::<Vec<_>>();
    model.minimize(hard_qp_obj(
        &vars,
        &linear_terms,
        &diag_terms,
        &offdiag_terms,
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_LARGE_EXPECTED_OBJ,
        "hard_qp_large_sparse",
    );
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_large_sparse");
}

/// Hard QP: Eq and UB are contradictory.
///
/// SciPy oracle:
/// SLSQP on `Q=I, c=0, x+y=3, bounds=[(0,1),(0,1)]` returned failure
/// with incompatible constraints.
#[test]
fn hard_qp_infeasible_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_infeasible_eq_ub_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 1.0), model.add_var("y", 0.0, 1.0)];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(3.0));
    model.minimize(hard_qp_obj(&vars, &[], &[(0, 1.0), (1, 1.0)], &[]));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "hard_qp_infeasible_eq_ub: expected infeasible, got {err:?}"
    );
}
