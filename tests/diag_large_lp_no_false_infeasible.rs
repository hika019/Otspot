//! task #37: large LP false Infeasible regression guard.
//!
//! Mittelmann large/standard LPs (pilot, dfl001, ken-13, ken-18) are known
//! Optimal but currently return FAIL:Infeasible at 1000s/eps=1e-6. The
//! dual_advanced router falls through `primal Timeout + empty incumbent →
//! Big-M Phase I`, and Big-M's `Phase I Timeout + artificials residual →
//! Infeasible` heuristic flips the verdict on these merely-slow LPs.
//!
//! Guard: solving pilot at 120s must not return Infeasible. Timeout (with
//! or without incumbent) is the honest answer when we can't certify
//! infeasibility within the deadline.

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

const TIMEOUT_SEC: f64 = 120.0;
const LARGE_TIMEOUT_SEC: f64 = 60.0;

fn run_with(path_str: &str, timeout_sec: f64) -> (SolveStatus, f64, usize) {
    let path = Path::new(path_str);
    if !path.exists() {
        panic!("data missing: {} — required for task #37 guard", path_str);
    }
    let prob = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_sec);

    let t0 = Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[#37] {} (timeout={:.0}s) -> status={:?} obj={:.6e} wall={:.2}s iters={}",
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
        "pilot returned Infeasible — task #37 regression: Big-M Phase I \
         Timeout heuristic must not declare infeasibility without certificate"
    );
}

/// Larger LPs — short 60s budget to fit nextest's per-test cap; guard is
/// "not false Infeasible" only (Optimal would need >>60s). Marked ignored
/// by default; run with `cargo nextest run --release -- --ignored task37`.
#[test]
#[ignore = "task #37 cross-check: 60s observation, not full convergence"]
fn task37_dfl001_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/dfl001.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}

#[test]
#[ignore = "task #37 cross-check: 60s observation, not full convergence"]
fn task37_ken13_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/ken-13.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}

#[test]
#[ignore = "task #37 cross-check: 60s observation, not full convergence"]
fn task37_ken18_no_false_infeasible() {
    let (status, _, _) = run_with("data/lp_problems/ken-18.QPS", LARGE_TIMEOUT_SEC);
    assert!(!matches!(status, SolveStatus::Infeasible));
}
