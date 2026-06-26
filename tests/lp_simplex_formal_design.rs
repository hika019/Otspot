use otspot::lp::solve_lp_with;
use otspot::options::{SimplexMethod, SolverOptions, WarmStartBasis};
use otspot::presolve;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::sparse::CscMatrix;
use otspot::SolveRoute;

const FEAS_TOL: f64 = 1e-7;
const OBJ_TOL: f64 = 1e-7;

fn lp_obj(lp: &LpProblem, x: &[f64]) -> f64 {
    lp.c.iter().zip(x).map(|(&c, &xj)| c * xj).sum::<f64>() + lp.obj_offset
}

fn assert_lp_feasible(lp: &LpProblem, x: &[f64], label: &str) {
    assert_eq!(x.len(), lp.num_vars, "{label}: solution length");
    let ax = lp.a.mat_vec_mul(x).expect("A*x must succeed");
    for (i, ((&lhs, &rhs), &sense)) in ax
        .iter()
        .zip(lp.b.iter())
        .zip(lp.constraint_types.iter())
        .enumerate()
    {
        let ok = match sense {
            ConstraintType::Le => lhs <= rhs + FEAS_TOL,
            ConstraintType::Ge => lhs + FEAS_TOL >= rhs,
            ConstraintType::Eq => (lhs - rhs).abs() <= FEAS_TOL,
            _ => panic!("{label}: unsupported constraint type at row {i}: {sense:?}"),
        };
        assert!(
            ok,
            "{label}: row {i} infeasible, lhs={lhs:.12e} rhs={rhs:.12e} sense={sense:?}"
        );
    }
    for (j, (&xj, &(lb, ub))) in x.iter().zip(lp.bounds.iter()).enumerate() {
        assert!(
            xj + FEAS_TOL >= lb && xj <= ub + FEAS_TOL,
            "{label}: bound {j} infeasible, x={xj:.12e} bounds=({lb:.12e}, {ub:.12e})"
        );
    }
}

fn assert_optimal_lp(lp: &LpProblem, method: SimplexMethod, expected_obj: f64, label: &str) {
    let mut opts = SolverOptions::default();
    opts.simplex_method = method;
    opts.presolve = false;
    let result = solve_lp_with(lp, &opts);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "{label}/{method:?}: status={:?}",
        result.status
    );
    assert_lp_feasible(lp, &result.solution, &format!("{label}/{method:?}"));
    let recomputed = lp_obj(lp, &result.solution);
    assert!(
        (result.objective - recomputed).abs() <= OBJ_TOL,
        "{label}/{method:?}: reported objective {} must equal c*x {}",
        result.objective,
        recomputed
    );
    assert!(
        (result.objective - expected_obj).abs() <= OBJ_TOL,
        "{label}/{method:?}: objective {} expected {}",
        result.objective,
        expected_obj
    );
}

fn le_box_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 0, 1, 2], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 3, 2)
        .unwrap();
    LpProblem::new_general(
        vec![-3.0, -2.0],
        a,
        vec![4.0, 2.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap()
}

fn ge_cover_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    LpProblem::new_general(
        vec![2.0, 1.0],
        a,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap()
}

fn presolve_noop_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![3.0, 3.0],
        vec![ConstraintType::Ge, ConstraintType::Ge],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap()
}

fn eq_finite_ub_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    LpProblem::new_general(
        vec![-1.0, -2.0],
        a,
        vec![1.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0), (0.0, 1.0)],
        None,
    )
    .unwrap()
}

fn reducible_postsolve_lp() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 0, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let mut lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![2.0, 5.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    lp.obj_offset = 11.0;
    lp
}

#[test]
fn simplex_method_selection_solves_representative_lp_partitions() {
    let methods = [
        SimplexMethod::Primal,
        SimplexMethod::Dual,
        SimplexMethod::DualAdvanced,
        SimplexMethod::Auto,
    ];
    let cases = [
        (le_box_lp(), -10.0, "le_box"),
        (ge_cover_lp(), 3.0, "ge_cover"),
        (eq_finite_ub_lp(), -2.0, "eq_finite_ub"),
    ];

    for (lp, expected_obj, label) in cases {
        for method in methods {
            assert_optimal_lp(&lp, method, expected_obj, label);
        }
    }
}

#[test]
fn warm_start_basis_is_reused_when_valid_and_rejected_when_shape_is_invalid() {
    let lp = eq_finite_ub_lp();
    let mut cold_opts = SolverOptions::default();
    cold_opts.simplex_method = SimplexMethod::Auto;
    cold_opts.presolve = false;
    let cold = solve_lp_with(&lp, &cold_opts);
    assert_eq!(cold.status, SolveStatus::Optimal);
    let valid_basis = cold
        .warm_start_basis
        .clone()
        .expect("cold simplex solve must expose a warm-start basis");

    let mut warm_opts = SolverOptions::default();
    warm_opts.simplex_method = SimplexMethod::Auto;
    warm_opts.presolve = false;
    warm_opts.warm_start = Some(valid_basis);
    let warm = solve_lp_with(&lp, &warm_opts);
    assert_eq!(warm.status, SolveStatus::Optimal);
    assert_lp_feasible(&lp, &warm.solution, "valid warm start");
    assert!((warm.objective + 2.0).abs() <= OBJ_TOL);

    let malformed = WarmStartBasis {
        basis: vec![usize::MAX, usize::MAX],
        x_b: vec![f64::NAN],
    };
    let mut malformed_opts = SolverOptions::default();
    malformed_opts.simplex_method = SimplexMethod::Auto;
    malformed_opts.presolve = false;
    malformed_opts.warm_start = Some(malformed);
    let rejected = solve_lp_with(&lp, &malformed_opts);
    assert_eq!(
        rejected.status,
        SolveStatus::Optimal,
        "malformed warm basis must be rejected and solved cold"
    );
    assert_lp_feasible(&lp, &rejected.solution, "rejected malformed warm start");
    assert!((rejected.objective + 2.0).abs() <= OBJ_TOL);
}

#[test]
fn presolve_reduced_and_unreduced_paths_match_unreduced_lp_semantics() {
    let reduced = reducible_postsolve_lp();
    let presolved =
        presolve::run_presolve(&reduced, None).expect("reducible LP presolve must finish");
    assert!(
        presolved.was_reduced,
        "fixture must exercise reduced postsolve"
    );

    let mut on_opts = SolverOptions::default();
    on_opts.presolve = true;
    on_opts.recover_warm_start_basis = true;
    let on = solve_lp_with(&reduced, &on_opts);
    let mut off_opts = SolverOptions::default();
    off_opts.presolve = false;
    let off = solve_lp_with(&reduced, &off_opts);
    for (label, result) in [("presolve_on", &on), ("presolve_off", &off)] {
        assert_eq!(result.status, SolveStatus::Optimal, "{label}");
        assert_lp_feasible(&reduced, &result.solution, label);
        assert!((result.objective - lp_obj(&reduced, &result.solution)).abs() <= OBJ_TOL);
        assert!((result.objective - 13.0).abs() <= OBJ_TOL);
    }
    assert!(
        on.warm_start_basis.is_some(),
        "recover_warm_start_basis=true must reconstruct an original-space basis"
    );

    let unreduced = presolve_noop_lp();
    let pr = presolve::run_presolve(&unreduced, None).expect("unreduced LP presolve must finish");
    assert!(!pr.was_reduced, "fixture must exercise presolve no-op");
    let mut no_op_opts = SolverOptions::default();
    no_op_opts.presolve = true;
    let no_op = solve_lp_with(&unreduced, &no_op_opts);
    assert_eq!(no_op.status, SolveStatus::Optimal);
    assert_lp_feasible(&unreduced, &no_op.solution, "presolve_noop");
    assert!((no_op.objective - 2.0).abs() <= OBJ_TOL);
}

#[test]
fn timeout_deadline_boundary_returns_no_incumbent_and_sets_route_stats() {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(0.0);
    let result = solve_lp_with(&le_box_lp(), &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
    assert_eq!(result.objective, f64::INFINITY);
    assert!(result.solution.is_empty());
    assert!(result.dual_solution.is_empty());
    assert!(result.reduced_costs.is_empty());
    assert!(result.slack.is_empty());
    assert_eq!(result.stats.route, SolveRoute::LpDirect);
    assert!(result.stats.deadline_triggered);
}

#[test]
fn status_boundaries_for_minimal_lps_are_semantic_not_path_dependent() {
    let optimal = LpProblem::new_general(
        vec![1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![],
        vec![(-2.0, 5.0)],
        None,
    )
    .unwrap();
    let infeasible = LpProblem::new_general(
        vec![],
        CscMatrix::new(1, 0),
        vec![1.0],
        vec![ConstraintType::Eq],
        vec![],
        None,
    )
    .unwrap();
    let unbounded = LpProblem::new_general(
        vec![-1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    let opt = solve_lp_with(&optimal, &opts);
    assert_eq!(opt.status, SolveStatus::Optimal);
    assert_eq!(opt.solution, vec![-2.0]);
    assert!((opt.objective + 2.0).abs() <= OBJ_TOL);

    let inf = solve_lp_with(&infeasible, &opts);
    assert_eq!(inf.status, SolveStatus::Infeasible);
    assert_eq!(inf.objective, f64::INFINITY);
    assert!(inf.solution.is_empty());

    let unb = solve_lp_with(&unbounded, &opts);
    assert_eq!(unb.status, SolveStatus::Unbounded);
    assert_eq!(unb.objective, f64::NEG_INFINITY);
    assert!(unb.solution.is_empty());
}
