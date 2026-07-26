// Synthetic data builders index into multiple parallel vecs by row/col, so the
// numeric loops can't be rewritten as a single .iter().enumerate() without
// shuffling the helpers; allow the range-loop pattern here.
#![allow(clippy::needless_range_loop)]

//! Hard-data sentinels (SciPy oracles) for LP solve via the Model expression API.
//!
//! Every expected value is hand-computed / SciPy-derived (independent oracle).
//! No expected value is derived by running the solver.
//!
//! These tests build problems through `otspot::model::Model` / `Expression`
//! rather than `otspot_core::problem::LpProblem` directly (that low-level
//! coverage, including the equivalent scaled Eq/Ge hard-data sentinel, lives
//! in `otspot-core/tests/blackbox_lp.rs`). They live at the workspace root
//! because `otspot` already depends on both `otspot-core` and `otspot-model`;
//! putting Model-API tests in `otspot-core/tests` would need otspot-core (a
//! foundation crate) to dev-depend back on otspot-model (a layer built on
//! top of it) — a layering violation, and it would pull otspot-model's own
//! dependency closure into otspot-core's test build for no reason.

use otspot::model::{Expression, Model, ModelError, SolveError, Variable};

const HARD_LP_EPS_OBJ: f64 = 5e-6;
const HARD_LP_EPS_X: f64 = 2e-4;
const HARD_LP_EPS_RESID: f64 = 2e-5;
const HARD_LP_TIMEOUT_SECS: f64 = 10.0;
const HARD_LP_INF: f64 = f64::INFINITY;
const HARD_LP_LARGE_M: usize = 50;
const HARD_LP_LARGE_N: usize = 100;
const HARD_LP_LARGE_EXPECTED_OBJ: f64 = -4.406_871_388_953_238;
const HARD_LP_ILL_EXPECTED_OBJ: f64 = -6.099_999_999_814_999;
const HARD_LP_DEGENERATE_EXPECTED_OBJ: f64 = -1.0;
const HARD_LP_NEAR_TIE_EXPECTED_OBJ: f64 = -1.000_000_005_9;

fn hard_lp_expr(vars: &[Variable], terms: &[(usize, f64)]) -> Expression {
    let mut expr = Expression::from_constant(0.0);
    for &(var_idx, coeff) in terms {
        expr = expr + coeff * vars[var_idx];
    }
    expr
}

fn hard_lp_assert_model_obj(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < HARD_LP_EPS_OBJ,
        "{label}: obj={actual:.12e} expected={expected:.12e} rel={rel:.3e}"
    );
}

fn hard_lp_assert_model_x(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_LP_EPS_X,
        "{label}: x={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
    );
}

fn hard_lp_assert_resid(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_LP_EPS_RESID,
        "{label}: residual actual={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
    );
}

/// Hard LP: Eq + UB with coefficients spanning 1e-10..1e10.
///
/// SciPy oracle:
/// `linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")`
/// returned status 0, fun = -6.099999999814999, x = [4,2,4,0,0,3].
#[test]
fn hard_lp_ill_scaled_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_ill_scaled_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars = vec![
        model.add_var("x0", 0.0, 5.0),
        model.add_var("x1", 0.0, 4.0),
        model.add_var("x2", 0.0, 4.5),
        model.add_var("x3", 0.0, 3.0),
        model.add_var("x4", 0.0, 2.0),
        model.add_var("x5", 0.0, 3.0),
    ];

    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1e-10), (1, 1.0)]).eq_constraint(2.000_000_000_3),
    );
    model.add_constraint(
        hard_lp_expr(&vars, &[(2, 1e10), (3, 1.0)]).eq_constraint(40_000_000_001.0),
    );
    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (2, 1.0), (4, 1.0)]).eq_constraint(8.0));
    model.add_constraint(hard_lp_expr(&vars, &[(1, 1.0), (3, 1.0), (5, 1.0)]).eq_constraint(5.0));
    model.minimize(hard_lp_expr(
        &vars,
        &[
            (0, -1.0),
            (1, 0.25),
            (2, -0.5),
            (3, 0.75),
            (4, 0.1),
            (5, -0.2),
        ],
    ));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_ILL_EXPECTED_OBJ,
        "hard_lp_ill_scaled",
    );
    for (idx, expected) in [4.0, 2.0, 4.000_000_000_1, 0.0, 0.0, 3.0]
        .into_iter()
        .enumerate()
    {
        hard_lp_assert_model_x(
            r[vars[idx]],
            expected,
            &format!("hard_lp_ill_scaled x{idx}"),
        );
    }
}

/// Hard LP: degenerate ratio tie with multiple simultaneous leaving candidates.
///
/// SciPy oracle:
/// `linprog([-1,-1,0,0], A_ub=[[1,1,0,0],[1,0,1,0],[0,1,0,1]], ...)`
/// returned status 0 and fun = -1.0. The primal solution is intentionally
/// non-unique; the sentinel checks objective and active row feasibility.
#[test]
fn hard_lp_degenerate_ratio_tie_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_degenerate_ratio_tie_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..4)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, HARD_LP_INF))
        .collect();

    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0)]).leq(1.0));
    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (2, 1.0)]).leq(1.0));
    model.add_constraint(hard_lp_expr(&vars, &[(1, 1.0), (3, 1.0)]).leq(1.0));
    model.minimize(hard_lp_expr(&vars, &[(0, -1.0), (1, -1.0)]));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_DEGENERATE_EXPECTED_OBJ,
        "hard_lp_degenerate",
    );
    hard_lp_assert_resid(
        r[vars[0]] + r[vars[1]],
        1.0,
        "hard_lp_degenerate active row",
    );
}

/// Hard LP: finite UB becomes active under a near pivot tie.
///
/// SciPy oracle:
/// `linprog(c, A_eq=[[1,1,1],[1,1+1e-10,0]], b_eq=[1+1e-8,1+5e-9], bounds=[(0,1)]*3)`
/// returned status 0, fun = -1.0000000059, x = [0,1,0] within HiGHS feasibility tolerance.
#[test]
fn hard_lp_upper_bound_near_tie_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_upper_bound_near_tie_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..3)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0), (2, 1.0)]).eq_constraint(1.000_000_01),
    );
    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1.0), (1, 1.000_000_000_1)]).eq_constraint(1.000_000_005),
    );
    model.minimize(hard_lp_expr(
        &vars,
        &[(0, -1.0), (1, -1.000_000_001), (2, 0.05)],
    ));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_NEAR_TIE_EXPECTED_OBJ,
        "hard_lp_near_tie",
    );
    hard_lp_assert_model_x(r[vars[1]], 1.0, "hard_lp_near_tie y at ub");
}

/// Hard LP: m=50, n=100 Eq+UB synthetic instance for Phase I + Harris stress.
///
/// SciPy oracle:
/// deterministic A/c/x_ref below, then
/// `linprog(c, A_eq=A, b_eq=A@x_ref, bounds=[(0,1)]*100, method="highs")`
/// returned status 0, fun = -4.406871388953238.
#[test]
fn hard_lp_large_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_large_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..HARD_LP_LARGE_N)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    let mut x_ref = vec![0.0; HARD_LP_LARGE_N];
    for (j, xj) in x_ref.iter_mut().enumerate() {
        *xj = ((j * 37) % HARD_LP_LARGE_N) as f64 / 99.0;
        if j % 17 == 0 {
            *xj = 1.0;
        }
        if j % 19 == 0 {
            *xj = 0.0;
        }
    }

    for i in 0..HARD_LP_LARGE_M {
        let mut terms = Vec::new();
        let mut rhs = 0.0;
        for j in 0..HARD_LP_LARGE_N {
            if matches!((i * 31 + j * 17) % 7, 0 | 3 | 5) {
                let scale = 10.0_f64.powi(((i + j) % 9) as i32 - 4);
                let sign = if (i + 2 * j) % 2 == 0 { 1.0 } else { -1.0 };
                let coeff = sign * scale * (1.0 + ((i * j) % 5) as f64 * 0.1);
                terms.push((j, coeff));
                rhs += coeff * x_ref[j];
            }
        }
        model.add_constraint(hard_lp_expr(&vars, &terms).eq_constraint(rhs));
    }

    let obj_terms: Vec<_> = (0..HARD_LP_LARGE_N)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            (j, sign * (0.01 + (j % 11) as f64 * 0.03))
        })
        .collect();
    model.minimize(hard_lp_expr(&vars, &obj_terms));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_LARGE_EXPECTED_OBJ,
        "hard_lp_large_eq_ub",
    );
}

/// Hard LP: Eq and UB are contradictory.
///
/// SciPy oracle:
/// `linprog([0,0], A_eq=[[1,1]], b_eq=[3], bounds=[(0,1),(0,1)], method="highs")`
/// returned status 2 (infeasible).
#[test]
fn hard_lp_infeasible_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_infeasible_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 1.0);
    let y = model.add_var("y", 0.0, 1.0);
    let vars = [x, y];

    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(3.0));
    model.minimize(Expression::from_constant(0.0));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "hard_lp_infeasible_eq_ub: expected infeasible, got {err:?}"
    );
}

/// Hard LP: Eq row plus ill-scaled cost has an unbounded improving ray.
///
/// SciPy oracle:
/// `linprog([-1e10,0], A_eq=[[0,1e-10]], b_eq=[0], bounds=[(None,None),(0,1)])`
/// returned status 3 (unbounded).
#[test]
fn hard_lp_unbounded_eq_ill_scaled_cost_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_unbounded_eq_ill_scaled_cost_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let ray = model.add_var("ray", f64::NEG_INFINITY, HARD_LP_INF);
    let pinned = model.add_var("pinned", 0.0, 1.0);
    let vars = [ray, pinned];

    model.add_constraint(hard_lp_expr(&vars, &[(1, 1e-10)]).eq_constraint(0.0));
    model.minimize(hard_lp_expr(&vars, &[(0, -1e10)]));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Unbounded)),
        "hard_lp_unbounded_eq_ill_scaled_cost: expected unbounded, got {err:?}"
    );
}
