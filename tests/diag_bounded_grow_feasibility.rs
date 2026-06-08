use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use otspot_dev::bench_utils::compute_pfeas_normalized;
use std::path::Path;

fn assert_grow_primal_feasible(name: &str) {
    let path = format!("data/lp_problems/{name}.QPS");
    let path = Path::new(&path);
    assert!(path.exists(), "data missing: {}", path.display());

    let problem = parse_qps(path).expect("parse_qps");
    let mut options = SolverOptions::default();
    options.timeout_secs = Some(1000.0);
    options.ipm.eps = 1e-6;

    let result = solve_qp_with(&problem, &options);
    let pfn = compute_pfeas_normalized(&problem, &result.solution);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "{name}: expected Optimal, got {:?} pfn={pfn:.3e}",
        result.status
    );
    assert!(
        pfn <= 1e-6,
        "{name}: bounded simplex returned original-row infeasible solution pfn={pfn:.3e}"
    );
}

#[test]
fn bounded_grow_family_primal_feasible() {
    for name in ["grow7", "grow15", "grow22"] {
        assert_grow_primal_feasible(name);
    }
}

#[test]
fn bounded_pilot_we_certifies_optimal() {
    let path = Path::new("data/lp_problems/pilot-we.QPS");
    assert!(path.exists(), "data missing: {}", path.display());

    let problem = parse_qps(path).expect("parse pilot-we");
    let mut options = SolverOptions::default();
    options.timeout_secs = Some(1000.0);
    options.ipm.eps = 1e-6;
    let expected = -2.7201027439e6 + problem.obj_offset;
    options.known_optimal_obj = Some(expected);

    let result = solve_qp_with(&problem, &options);
    let pfn = compute_pfeas_normalized(&problem, &result.solution);
    let obj_rel = (result.objective - expected).abs() / (1.0 + expected.abs());

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "pilot-we: expected certified Optimal, got {:?} obj={:.12e} pfn={pfn:.3e}",
        result.status,
        result.objective
    );
    assert!(
        obj_rel <= 1e-5,
        "pilot-we: objective mismatch got={:.12e} expected={:.12e} rel={obj_rel:.3e}",
        result.objective,
        expected
    );
}
