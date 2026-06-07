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
