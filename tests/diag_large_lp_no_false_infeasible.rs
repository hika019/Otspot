//! Large LP false Infeasible regression guard。
//! pilot/dfl001/ken-13/ken-18 は Optimal 既知だが、Big-M Phase I の
//! `Timeout + artificials residual → Infeasible` heuristic で誤判定しうる。
//! Timeout が正しい honest answer。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

const TIMEOUT_SEC: f64 = 120.0;
const LARGE_TIMEOUT_SEC: f64 = 60.0;

fn run_with(path_str: &str, timeout_sec: f64) -> (SolveStatus, f64, usize) {
    let path = Path::new(path_str);
    if !path.exists() {
        panic!("data missing: {}", path_str);
    }
    let prob = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_sec);

    let t0 = Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "{} (timeout={:.0}s) -> status={:?} obj={:.6e} wall={:.2}s iters={}",
        path_str, timeout_sec, r.status, r.objective, wall, r.iterations
    );
    (r.status, wall, r.iterations)
}

fn run(path_str: &str) -> (SolveStatus, f64, usize) {
    run_with(path_str, TIMEOUT_SEC)
}

/// Primary TDD red: pilot must not return Infeasible at 120s.
#[test]
fn pilot_no_false_infeasible() {
    let (status, _wall, _iters) = run("data/lp_problems/pilot.QPS");
    assert!(
        !matches!(status, SolveStatus::Infeasible),
        "pilot returned Infeasible — Big-M Phase I Timeout heuristic must not \
         declare infeasibility without certificate"
    );
}

/// Larger LPs — short 60s budget; guard is "not false Infeasible" only.
#[test]
#[ignore = "cross-check: 60s observation, not full convergence"]
fn task37_dfl001_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/dfl001.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}

#[test]
#[ignore = "cross-check: 60s observation, not full convergence"]
fn task37_ken13_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/ken-13.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}

#[test]
#[ignore = "cross-check: 60s observation, not full convergence"]
fn task37_ken18_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/ken-18.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}
