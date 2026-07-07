//! Input-validation and soundness tests for the Model QCQP/SOCP DSL paths
//! (`add_qc_le` / `add_soc_le`), mirroring the LP/QP Model-path conventions:
//! non-finite user data and foreign-model variables are `InvalidInput`;
//! unsupported solver-path requirements are `NotSupported`; conic solves must
//! honor `set_tolerance` and populate `ModelResult::stats`.

use otspot_core::options::Tolerance;
use otspot_core::problem::{SolveRoute, SolveStatus};
use otspot_model::{Expression, Model, ModelError, SolveError};

// ---------------------------------------------------------------------------
// #5: foreign-model variables in quadratic-constraint quad terms
// ---------------------------------------------------------------------------

#[test]
fn qc_foreign_quad_var_same_index_rejected() {
    // m2's y has index 0, in range for m1 (n=1): without the model_id check
    // the quad term silently constrains m1's x instead.
    let mut m1 = Model::new("m1");
    let x = m1.add_var("x", -5.0, 5.0);
    let mut m2 = Model::new("m2");
    let y = m2.add_var("y", -5.0, 5.0);

    m1.add_qc_le(y * y, 1.0);
    m1.minimize(1.0 * x);
    let result = m1.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "foreign quad var (in-range index) must give InvalidInput, got {result:?}"
    );
}

#[test]
fn qc_foreign_quad_var_out_of_range_rejected_as_invalid_input() {
    // Foreign var with an out-of-range index: must be InvalidInput (user
    // error), not Internal (which quad_to_csc's index check would produce).
    let mut m1 = Model::new("m1");
    let x = m1.add_var("x", -5.0, 5.0);
    let mut m2 = Model::new("m2");
    let _pad = m2.add_var("pad", 0.0, 1.0);
    let y = m2.add_var("y", -5.0, 5.0); // index 1 >= m1.n = 1

    m1.add_qc_le(y * y, 1.0);
    m1.minimize(1.0 * x);
    let result = m1.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "foreign quad var (out-of-range index) must give InvalidInput, got {result:?}"
    );
}

#[test]
fn qc_foreign_linear_var_rejected() {
    // Linear part of the quadratic constraint (existing check — regression guard).
    let mut m1 = Model::new("m1");
    let x = m1.add_var("x", -5.0, 5.0);
    let mut m2 = Model::new("m2");
    let y = m2.add_var("y", -5.0, 5.0);

    m1.add_qc_le(x * x + 1.0 * y, 1.0);
    m1.minimize(1.0 * x);
    let result = m1.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "foreign linear var in qc must give InvalidInput, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #6: non-finite constants/coefficients in quadratic constraints
// ---------------------------------------------------------------------------

#[test]
fn qc_non_finite_rhs_rejected() {
    for &(label, bad) in &[
        ("nan", f64::NAN),
        ("inf", f64::INFINITY),
        ("neg_inf", f64::NEG_INFINITY),
    ] {
        let mut m = Model::new(label);
        let x = m.add_var("x", 0.0, 5.0);
        m.add_qc_le(x * x, bad);
        m.minimize(1.0 * x);
        let result = m.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "[{label}] non-finite qc rhs must give InvalidInput, got {result:?}"
        );
    }
}

#[test]
fn qc_nan_linear_coefficient_rejected() {
    let mut m = Model::new("qc_nan_lin");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_qc_le(x * x + f64::NAN * x, 1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN linear coef in qc must give InvalidInput, got {result:?}"
    );
}

#[test]
fn qc_nan_constant_rejected() {
    let mut m = Model::new("qc_nan_const");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_qc_le(x * x + f64::NAN, 1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN constant in qc must give InvalidInput, got {result:?}"
    );
}

#[test]
fn qc_nan_quad_coefficient_rejected_as_invalid_input() {
    // quad_to_csc rejects the NaN, but the error must surface as InvalidInput
    // (user data), not Internal.
    let mut m = Model::new("qc_nan_quad");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_qc_le(f64::NAN * (x * x), 1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN quad coef in qc must give InvalidInput, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #7: non-finite linear-constraint data on the conic path
// ---------------------------------------------------------------------------

#[test]
fn linear_constraint_nan_rhs_rejected_on_conic_path() {
    // The LP/QP constructors validate b; the conic path must too.
    let mut m = Model::new("lin_nan_rhs");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_constraint((1.0 * x).leq(f64::NAN));
    m.add_qc_le(x * x, 1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN linear rhs on conic path must give InvalidInput, got {result:?}"
    );
}

#[test]
fn linear_constraint_nan_coef_rejected_on_conic_path() {
    let mut m = Model::new("lin_nan_coef");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_constraint((f64::NAN * x).leq(1.0));
    m.add_qc_le(x * x, 1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN linear coef on conic path must give InvalidInput, got {result:?}"
    );
}

#[test]
fn linear_eq_constraint_nan_rhs_rejected_on_conic_path() {
    let mut m = Model::new("lin_nan_eq");
    let x = m.add_var("x", 0.0, 5.0);
    let y = m.add_var("y", 0.0, 5.0);
    m.add_constraint((x + y).eq_constraint(f64::NAN));
    m.add_qc_le(x * x + y * y, 1.0);
    m.minimize(x + y);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN eq rhs on conic path must give InvalidInput, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #8/#9: non-finite SOC affine data
// ---------------------------------------------------------------------------

#[test]
fn soc_nan_bound_constant_rejected() {
    let mut m = Model::new("soc_nan_t");
    let x = m.add_var("x", 0.0, 5.0);
    m.add_soc_le(vec![1.0 * x], Expression::from_constant(f64::NAN));
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN SOC bound constant must give InvalidInput, got {result:?}"
    );
}

#[test]
fn soc_nan_term_constant_rejected() {
    let mut m = Model::new("soc_nan_term_const");
    let x = m.add_var("x", 0.0, 5.0);
    let t = m.add_var("t", 0.0, 5.0);
    m.add_soc_le(vec![1.0 * x + f64::NAN], 1.0 * t);
    m.minimize(1.0 * t);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN SOC term constant must give InvalidInput, got {result:?}"
    );
}

#[test]
fn soc_nan_term_coefficient_rejected() {
    let mut m = Model::new("soc_nan_term_coef");
    let x = m.add_var("x", 0.0, 5.0);
    let t = m.add_var("t", 0.0, 5.0);
    m.add_soc_le(vec![f64::NAN * x], 1.0 * t);
    m.minimize(1.0 * t);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "NaN SOC term coef must give InvalidInput, got {result:?}"
    );
}

#[test]
fn soc_inf_bound_coefficient_rejected() {
    let mut m = Model::new("soc_inf_t_coef");
    let x = m.add_var("x", 0.0, 5.0);
    let t = m.add_var("t", 0.0, 5.0);
    m.add_soc_le(vec![1.0 * x], f64::INFINITY * t);
    m.minimize(1.0 * t);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "Inf SOC bound coef must give InvalidInput, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #10: set_tolerance must reach the conic solver
// ---------------------------------------------------------------------------

#[test]
fn conic_path_honors_loose_custom_tolerance() {
    // max x + y s.t. x^2 + y^2 <= 1: optimum sqrt(2) at (1/sqrt2, 1/sqrt2).
    // With the hardcoded conic tol (1e-9) the IPM converges to ~1e-10 error;
    // a very loose Custom tolerance must stop the IPM early, leaving a
    // measurably larger optimality error. Reverting the tolerance plumbing
    // makes the error collapse below the discrimination threshold.
    let solve_with = |tol: Option<Tolerance>| -> f64 {
        let mut m = Model::new("tol_disk");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        let y = m.add_var("y", 0.0, f64::INFINITY);
        m.add_qc_le(x * x + y * y, 1.0);
        m.maximize(x + y);
        if let Some(t) = tol {
            m.set_tolerance(t);
        }
        let r = m.solve().unwrap();
        (r.objective_value - 2.0_f64.sqrt()).abs()
    };

    let err_default = solve_with(None);
    let err_loose = solve_with(Some(Tolerance::Custom(5e-2)));
    assert!(
        err_default < 1e-7,
        "default tolerance should converge tightly, err={err_default:e}"
    );
    assert!(
        err_loose > 1e-7,
        "loose Custom(5e-2) tolerance must stop the conic IPM early \
         (err_loose={err_loose:e}); if this is tiny, set_tolerance is not \
         reaching ConicOptions::tol"
    );
}

// ---------------------------------------------------------------------------
// #11: stats must be populated on conic paths
// ---------------------------------------------------------------------------

#[test]
fn conic_convex_path_populates_stats_route() {
    let mut m = Model::new("stats_convex");
    let x = m.add_var("x", 0.0, f64::INFINITY);
    let y = m.add_var("y", 0.0, f64::INFINITY);
    m.add_qc_le(x * x + y * y, 1.0);
    m.maximize(x + y);
    let r = m.solve().unwrap();
    assert_eq!(
        r.stats.route,
        SolveRoute::ConicQcqpConvex,
        "convex conic solve must report its route in stats"
    );
}

#[test]
fn conic_socp_path_populates_stats_route() {
    let mut m = Model::new("stats_socp");
    let x = m.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = m.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
    let t = m.add_var("t", 0.0, f64::INFINITY);
    m.add_constraint((x + y).geq(2.0));
    m.add_soc_le(vec![1.0 * x, 1.0 * y], 1.0 * t);
    m.minimize(1.0 * t);
    let r = m.solve().unwrap();
    assert_eq!(
        r.stats.route,
        SolveRoute::ConicQcqpConvex,
        "SOCP solve must report its route in stats"
    );
}

// ---------------------------------------------------------------------------
// #12/#26: nonconvex QCQP must fall back to the global solver
// ---------------------------------------------------------------------------

#[test]
fn nonconvex_qc_solved_via_global_fallback() {
    // min x s.t. x^2 >= 1 (written as -x^2 <= -1), x in [0, 3].
    // Feasible set is [1, 3]; global optimum x* = 1, obj = 1.
    // The convex bridge rejects this (P not PSD); the model path must fall
    // back to the spatial B&B like otspot_core::qp's qcqp_route does.
    let mut m = Model::new("nonconvex_qc");
    let x = m.add_var("x", 0.0, 3.0);
    m.add_qc_le(-(x * x), -1.0);
    m.minimize(1.0 * x);
    let r = m.solve().expect("nonconvex QCQP must be solved globally");
    assert!(
        (r.objective_value - 1.0).abs() < 1e-4,
        "global optimum should be 1.0, got {}",
        r.objective_value
    );
    assert!(
        (r[x] - 1.0).abs() < 1e-3,
        "x* should be 1.0, got {}",
        r[x]
    );
    assert_eq!(
        r.stats.route,
        SolveRoute::ConicQcqpNonconvex,
        "fallback solve must report the nonconvex route"
    );
}

#[test]
fn unproven_convexity_routes_to_global_fallback() {
    // P = diag(2, -1e-10): the -1e-10 pivot sits inside the Cholesky jitter
    // band, so the bridge clamps it and flags convexity_unproven. The model
    // path must not present the clamped SOCP solve as a proven Optimal —
    // it must re-solve via the global route (matching qp::qcqp_route).
    let mut m = Model::new("unproven_qc");
    let x = m.add_var("x", 0.0, 2.0);
    let y = m.add_var("y", 0.0, 1.0);
    m.add_qc_le(x * x - 5e-11 * (y * y), 1.0);
    m.maximize(x + y);
    let r = m.solve().expect("jitter-band QCQP must still be solved");
    assert!(
        (r.objective_value - 2.0).abs() < 1e-3,
        "optimum should be ~2.0 (x=1, y=1), got {}",
        r.objective_value
    );
    assert_eq!(
        r.stats.route,
        SolveRoute::ConicQcqpNonconvex,
        "convexity_unproven result must be re-solved on the global route"
    );
}

#[test]
fn nonconvex_qc_without_finite_bounds_is_not_supported() {
    // Global fallback needs a finite box; without one the model path must
    // report NotSupported (structural), not a silent wrong answer.
    let mut m = Model::new("nonconvex_unbounded");
    let x = m.add_var("x", 0.0, f64::INFINITY);
    m.add_qc_le(-(x * x), -1.0);
    m.minimize(1.0 * x);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::NotSupported(_))),
        "nonconvex QCQP without finite bounds must be NotSupported, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #13: SOC-augmented path must not silently accept convexity_unproven
// ---------------------------------------------------------------------------

#[test]
fn soc_path_rejects_unproven_convexity() {
    // Same jitter-band quadratic as above, but with an SOC constraint the
    // spatial fallback is unavailable — the solve must refuse rather than
    // report a proven optimum for the clamped approximation.
    let mut m = Model::new("soc_unproven");
    let x = m.add_var("x", 0.0, 2.0);
    let y = m.add_var("y", 0.0, 1.0);
    let t = m.add_var("t", 0.0, 5.0);
    m.add_qc_le(x * x - 5e-11 * (y * y), 1.0);
    m.add_soc_le(vec![1.0 * x, 1.0 * y], 1.0 * t);
    m.minimize(1.0 * t);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::NotSupported(_))),
        "unproven convexity with SOC constraints must be NotSupported, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// #15: integer bounds must be finite on the conic B&B paths
// ---------------------------------------------------------------------------

#[test]
fn miqcp_infinite_integer_bound_is_not_supported() {
    let mut m = Model::new("miqcp_inf_int");
    let k = m.add_int_var("k", 0.0, f64::INFINITY);
    m.add_qc_le(k * k, 4.0);
    m.maximize(1.0 * k);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::NotSupported(_))),
        "MIQCP with infinite integer bound must be NotSupported, got {result:?}"
    );
}

#[test]
fn misocp_infinite_integer_bound_is_not_supported() {
    let mut m = Model::new("misocp_inf_int");
    let k = m.add_int_var("k", 0.0, f64::INFINITY);
    let t = m.add_var("t", 0.0, 10.0);
    m.add_soc_le(vec![1.0 * k], 1.0 * t);
    m.minimize(1.0 * t - k);
    let result = m.solve();
    assert!(
        matches!(result, Err(ModelError::NotSupported(_))),
        "MISOCP with infinite integer bound must be NotSupported, got {result:?}"
    );
}

#[test]
fn miqcp_finite_integer_bounds_still_solve() {
    // No false positive: the finite-bound MIQCP keeps working.
    let mut m = Model::new("miqcp_ok");
    let x = m.add_int_var("x", 0.0, 2.0);
    let y = m.add_int_var("y", 0.0, 2.0);
    m.add_qc_le(x * x + y * y, 2.5);
    m.maximize(x + y);
    let r = m.solve().unwrap();
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective_value - 2.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// #27 (adjacent): continuous conic MaxIterations must not become Internal
// ---------------------------------------------------------------------------

#[test]
fn socp_iteration_cap_is_max_iterations_not_internal() {
    // An impossibly tight Custom tolerance exhausts the conic iteration cap.
    // The continuous SOCP path must classify that as MaxIterations (or a
    // NumericalError from IPM breakdown) — never Internal.
    let mut m = Model::new("socp_maxiter");
    let x = m.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = m.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
    let t = m.add_var("t", 0.0, f64::INFINITY);
    m.add_constraint((x + y).geq(2.0));
    m.add_soc_le(vec![1.0 * x, 1.0 * y], 1.0 * t);
    m.minimize(1.0 * t);
    m.set_tolerance(Tolerance::Custom(1e-300));
    let result = m.solve();
    match result {
        Err(ModelError::SolveError(SolveError::MaxIterations))
        | Err(ModelError::SolveError(SolveError::NumericalError)) => {}
        other => panic!(
            "iteration-capped continuous SOCP must be MaxIterations/NumericalError, got {other:?}"
        ),
    }
}
