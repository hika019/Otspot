use otspot::constraint;
use otspot::io::mps::parse_mps;
use otspot::mip::{solve_milp, MilpProblem};
use otspot::model::{Model, ModelError, SolutionProof};
use otspot::options::{MipConfig, SolverOptions, Tolerance};
use otspot::problem::{ConstraintType, LpProblem, SolveRoute, SolveStatus};
use otspot::qp::{solve_qp_with, QpProblem};
use otspot::sparse::CscMatrix;

const OBJ_TOL: f64 = 1e-6;
const FEAS_TOL: f64 = 1e-6;

fn le_ge_eq_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![3.0, 1.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 4.0), (0.0, 4.0)],
        Some("le_ge_eq_lp".into()),
    )
    .unwrap()
}

fn zero_q_from_lp(lp: &LpProblem) -> QpProblem {
    QpProblem::new(
        CscMatrix::new(lp.num_vars, lp.num_vars),
        lp.c.clone(),
        (*lp.a).clone(),
        lp.b.clone(),
        lp.bounds.clone(),
        lp.constraint_types.clone(),
    )
    .unwrap()
}

fn convex_qp() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    QpProblem::new(
        q,
        vec![-2.0, -4.0],
        a,
        vec![3.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

fn tiny_milp() -> MilpProblem {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[2.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-3.0, -2.0],
        a,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
        Some("tiny_milp".into()),
    )
    .unwrap();
    MilpProblem::new(lp, vec![0, 1]).unwrap()
}

fn lp_objective(lp: &LpProblem, x: &[f64]) -> f64 {
    lp.c.iter().zip(x).map(|(c, xi)| c * xi).sum::<f64>() + lp.obj_offset
}

fn qp_objective(qp: &QpProblem, x: &[f64]) -> f64 {
    let qx = qp.q.mat_vec_mul(x).unwrap();
    let quad = 0.5 * x.iter().zip(qx).map(|(xi, qxi)| xi * qxi).sum::<f64>();
    let linear = qp.c.iter().zip(x).map(|(c, xi)| c * xi).sum::<f64>();
    quad + linear + qp.obj_offset
}

fn assert_primal_feasible(
    a: &CscMatrix,
    b: &[f64],
    ct: &[ConstraintType],
    bounds: &[(f64, f64)],
    x: &[f64],
) {
    assert_eq!(bounds.len(), x.len(), "bounds length must match solution");

    for (j, ((lower, upper), xj)) in bounds.iter().zip(x).enumerate() {
        assert!(
            *xj + FEAS_TOL >= *lower,
            "var {j}: expected x >= lower, got {xj} < {lower}"
        );
        assert!(
            *xj <= *upper + FEAS_TOL,
            "var {j}: expected x <= upper, got {xj} > {upper}"
        );
    }

    let ax = a.mat_vec_mul(x).unwrap();
    for (i, ((lhs, rhs), sense)) in ax.iter().zip(b).zip(ct).enumerate() {
        match sense {
            ConstraintType::Le => assert!(
                *lhs <= *rhs + FEAS_TOL,
                "row {i}: expected lhs <= rhs, got {lhs} > {rhs}"
            ),
            ConstraintType::Ge => assert!(
                *lhs + FEAS_TOL >= *rhs,
                "row {i}: expected lhs >= rhs, got {lhs} < {rhs}"
            ),
            ConstraintType::Eq => assert!(
                (*lhs - *rhs).abs() <= FEAS_TOL,
                "row {i}: expected lhs == rhs, got lhs={lhs}, rhs={rhs}"
            ),
            _ => panic!("unexpected constraint type in public API test row {i}"),
        }
    }
}

#[test]
#[should_panic(expected = "var 0: expected x <= upper")]
fn primal_feasibility_oracle_rejects_bound_violations_without_row_violations() {
    assert_primal_feasible(&CscMatrix::new(0, 1), &[], &[], &[(0.0, 1.0)], &[2.0]);
}

fn options_with(f: impl FnOnce(&mut SolverOptions)) -> SolverOptions {
    let mut opts = SolverOptions::default();
    f(&mut opts);
    opts
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let tol = OBJ_TOL * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual:.12e}, expected={expected:.12e}"
    );
}

#[test]
fn solver_options_public_boundary_and_combination_contract() {
    let cases = [
        ("default", SolverOptions::default(), true, "ok"),
        (
            "zero_timeout_allowed",
            options_with(|opts| opts.timeout_secs = Some(0.0)),
            true,
            "ok",
        ),
        (
            "zero_clamp_allowed",
            options_with(|opts| opts.clamp_tol = 0.0),
            true,
            "ok",
        ),
        (
            "zero_threads_rejected",
            options_with(|opts| opts.threads = 0),
            false,
            "threads",
        ),
        (
            "negative_timeout_rejected",
            options_with(|opts| opts.timeout_secs = Some(-f64::MIN_POSITIVE)),
            false,
            "timeout_secs",
        ),
        (
            "custom_tolerance_must_be_positive",
            options_with(|opts| opts.tolerance = Some(Tolerance::Custom(0.0))),
            false,
            "tolerance.Custom",
        ),
        (
            "ipm_validation_still_load_bearing_when_tolerance_overrides_eps",
            options_with(|opts| {
                opts.tolerance = Some(Tolerance::Fast);
                opts.ipm.delta_min = f64::NAN;
            }),
            false,
            "ipm.delta_min",
        ),
    ];

    for (label, opts, valid, field) in cases {
        let result = opts.validate();
        assert_eq!(result.is_ok(), valid, "{label}");
        if !valid {
            assert_eq!(result.unwrap_err().field, field, "{label}");
        }
    }

    let opts = options_with(|opts| {
        opts.tolerance = Some(Tolerance::Fast);
        opts.ipm.eps = 1e-12;
    });
    assert_eq!(opts.ipm_eps(), otspot::options::TOLERANCE_FAST_EPS);
    assert_eq!(
        options_with(|opts| opts.tolerance = Some(Tolerance::Custom(2.5e-5))).ipm_eps(),
        2.5e-5
    );
}

#[test]
fn invalid_solver_options_have_consistent_public_status_across_entries() {
    let lp = le_ge_eq_lp();
    let qp = convex_qp();
    let milp = tiny_milp();
    let cfg = MipConfig::default();
    let bad_options = [
        options_with(|opts| opts.threads = 0),
        options_with(|opts| opts.timeout_secs = Some(f64::NEG_INFINITY)),
        options_with(|opts| opts.tolerance = Some(Tolerance::Custom(f64::NAN))),
    ];

    for opts in bad_options {
        let lp_result = otspot::lp::solve_lp_with(&lp, &opts);
        assert_eq!(lp_result.status, SolveStatus::NumericalError);
        assert_eq!(lp_result.stats.route, SolveRoute::LpDirect);
        assert!(lp_result.solution.is_empty());

        let qp_result = solve_qp_with(&qp, &opts);
        assert_eq!(qp_result.status, SolveStatus::NumericalError);
        assert_eq!(qp_result.stats.route, SolveRoute::Unknown);
        assert!(qp_result.solution.is_empty());

        let mip_result = solve_milp(&milp, &opts, &cfg);
        assert_eq!(mip_result.status, SolveStatus::NumericalError);
        assert_eq!(mip_result.stats.route, SolveRoute::Unknown);
        assert!(mip_result.solution.is_empty());
    }
}

#[test]
fn lp_qp_and_zero_q_entries_preserve_status_objective_and_feasibility_contracts() {
    let lp = le_ge_eq_lp();
    let opts = options_with(|opts| opts.presolve = false);
    let direct = otspot::lp::solve_lp_with(&lp, &opts);
    assert_eq!(direct.status, SolveStatus::Optimal);
    assert_eq!(direct.stats.route, SolveRoute::LpDirect);
    assert_primal_feasible(
        &lp.a,
        &lp.b,
        &lp.constraint_types,
        &lp.bounds,
        &direct.solution,
    );
    assert_close(
        direct.objective,
        lp_objective(&lp, &direct.solution),
        "direct lp obj",
    );
    assert_close(direct.objective, 1.0, "direct lp expected optimum");

    let zero_q = zero_q_from_lp(&lp);
    let forwarded = solve_qp_with(&zero_q, &opts);
    assert_eq!(forwarded.status, SolveStatus::Optimal);
    assert_eq!(forwarded.stats.route, SolveRoute::LpForwardedFromQp);
    assert_primal_feasible(
        &zero_q.a,
        &zero_q.b,
        &zero_q.constraint_types,
        &zero_q.bounds,
        &forwarded.solution,
    );
    assert_close(
        forwarded.objective,
        qp_objective(&zero_q, &forwarded.solution),
        "zero-q qp obj",
    );
    assert_close(
        forwarded.objective,
        direct.objective,
        "zero-q forward equals direct lp",
    );

    let qp = convex_qp();
    let qp_result = solve_qp_with(&qp, &SolverOptions::default());
    assert_eq!(qp_result.status, SolveStatus::Optimal);
    assert_eq!(qp_result.stats.route, SolveRoute::QpIpm);
    assert_primal_feasible(
        &qp.a,
        &qp.b,
        &qp.constraint_types,
        &qp.bounds,
        &qp_result.solution,
    );
    assert_close(
        qp_result.objective,
        qp_objective(&qp, &qp_result.solution),
        "qp recomputed obj",
    );
    assert_close(qp_result.objective, -5.0, "qp expected optimum");
}

#[test]
fn immediate_deadline_is_reported_consistently_for_lp_and_qp_entries() {
    let opts = options_with(|opts| opts.timeout_secs = Some(0.0));

    let lp = le_ge_eq_lp();
    let lp_result = otspot::lp::solve_lp_with(&lp, &opts);
    assert_eq!(lp_result.status, SolveStatus::Timeout);
    assert!(lp_result.stats.deadline_triggered);
    assert_eq!(lp_result.stats.route, SolveRoute::LpDirect);

    let qp = convex_qp();
    let qp_result = solve_qp_with(&qp, &opts);
    assert_eq!(qp_result.status, SolveStatus::Timeout);
    assert!(qp_result.stats.deadline_triggered);
    assert_eq!(qp_result.stats.route, SolveRoute::QpIpm);
}

#[test]
fn model_api_crosses_lp_qp_and_mip_layers_with_consistent_results() {
    let mut lp_model = Model::new("model_lp_contract");
    let x = lp_model.add_var("x", 0.0, 4.0);
    let y = lp_model.add_var("y", 0.0, 4.0);
    lp_model.add_constraint(constraint!((x + y) <= 3.0));
    lp_model.add_constraint(constraint!(x >= 1.0));
    lp_model.minimize(x + 2.0 * y + 7.0);
    let lp_result = lp_model.solve().unwrap();
    assert_eq!(lp_result.status, SolveStatus::Optimal);
    assert_eq!(lp_result.proof, SolutionProof::GlobalOptimal);
    assert_eq!(lp_result.stats.route, SolveRoute::LpDirect);
    assert_close(lp_result.objective_value, 8.0, "model lp objective");

    let mut qp_model = Model::new("model_qp_contract");
    let qx = qp_model.add_var("x", 0.0, 10.0);
    let qy = qp_model.add_var("y", 0.0, 10.0);
    qp_model.add_constraint(constraint!((qx + qy) <= 3.0));
    qp_model.set_tolerance(Tolerance::Fast);
    qp_model.minimize(qx * qx + qy * qy + (-2.0) * qx + (-4.0) * qy);
    let qp_result = qp_model.solve().unwrap();
    assert_eq!(qp_result.status, SolveStatus::Optimal);
    assert_eq!(qp_result.proof, SolutionProof::GlobalOptimal);
    assert_eq!(qp_result.stats.route, SolveRoute::QpIpm);
    assert_close(qp_result.objective_value, -5.0, "model qp objective");

    let mut milp_model = Model::new("model_milp_contract");
    let a = milp_model.add_binary_var("a");
    let b = milp_model.add_binary_var("b");
    milp_model.add_constraint(constraint!((2.0 * a + b) <= 3.0));
    milp_model.set_presolve(false);
    milp_model.set_threads(0);
    milp_model.maximize(3.0 * a + 2.0 * b);
    let milp_result = milp_model.solve().unwrap();
    assert_eq!(milp_result.status, SolveStatus::Optimal);
    assert_eq!(milp_result.proof, SolutionProof::GlobalOptimal);
    assert_close(milp_result.objective_value, 5.0, "model milp objective");
    assert!((milp_result[a] - milp_result[a].round()).abs() <= FEAS_TOL);
    assert!((milp_result[b] - milp_result[b].round()).abs() <= FEAS_TOL);
    assert!(2.0 * milp_result[a] + milp_result[b] <= 3.0 + FEAS_TOL);
}

#[test]
fn parser_model_and_solver_layers_agree_on_minimal_mps_lp() {
    let mps = "\
NAME          MINI
ROWS
 N  COST
 G  DEMAND
COLUMNS
    X         COST      1.0       DEMAND    1.0
RHS
    RHS1      DEMAND    2.0
BOUNDS
 LO BND1      X         0.0
 UP BND1      X         5.0
ENDATA
";
    let parsed = parse_mps(mps).unwrap();
    let parsed_result = otspot::solve(&parsed);
    assert_eq!(parsed_result.status, SolveStatus::Optimal);
    assert_eq!(parsed_result.stats.route, SolveRoute::LpDirect);
    assert_close(parsed_result.objective, 2.0, "parsed mps objective");
    assert_primal_feasible(
        &parsed.a,
        &parsed.b,
        &parsed.constraint_types,
        &parsed.bounds,
        &parsed_result.solution,
    );

    let mut model = Model::new("same_as_mps");
    let x = model.add_var("x", 0.0, 5.0);
    model.add_constraint(constraint!(x >= 2.0));
    model.minimize(x);
    let model_result = model.solve().unwrap();
    assert_eq!(model_result.status, SolveStatus::Optimal);
    assert_close(
        model_result.objective_value,
        parsed_result.objective,
        "model vs parsed mps objective",
    );
}

#[test]
fn model_timeout_and_invalid_option_surface_are_errors_not_success_results() {
    let mut timeout_model = Model::new("timeout_model");
    let x = timeout_model.add_var("x", 0.0, 10.0);
    timeout_model.add_constraint(constraint!(x >= 1.0));
    timeout_model.set_timeout(0.0);
    timeout_model.minimize(x);
    assert!(matches!(timeout_model.solve(), Err(ModelError::Timeout)));

    let mut invalid_tol_model = Model::new("invalid_tol_model");
    let y = invalid_tol_model.add_var("y", 0.0, 10.0);
    invalid_tol_model.set_tolerance(Tolerance::Custom(f64::NAN));
    invalid_tol_model.minimize(y);
    assert!(matches!(
        invalid_tol_model.solve(),
        Err(ModelError::SolveError(
            otspot::model::SolveError::NumericalError
        ))
    ));
}
