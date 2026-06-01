//! Diagnose false-Unbounded on Netlib pilot-ja.

use otspot::io::qps::parse_qps;
use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{LpProblem, SolveStatus};
use otspot::{solve_with, solve_with as solve_lp_with_opts};
use std::path::Path;
use std::time::Instant;

fn load_pilot_ja_lp() -> LpProblem {
    let path = Path::new("data/lp_problems/pilot-ja.QPS");
    assert!(
        path.exists(),
        "{} not found — run scripts/netlib_lp_download.sh",
        path.display()
    );
    let qp = parse_qps(path).expect("parse pilot-ja.QPS failed");
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        Some("pilot-ja".to_string()),
    )
    .expect("pilot-ja LP construction failed")
}

#[test]
fn diag_pilot_ja_stage_split() {
    let lp = load_pilot_ja_lp();
    let n_free = lp
        .bounds
        .iter()
        .filter(|&&(lb, ub)| lb == f64::NEG_INFINITY && ub == f64::INFINITY)
        .count();
    eprintln!(
        "pilot-ja dims: n={} m={} free_vars={}",
        lp.num_vars, lp.num_constraints, n_free
    );

    for &presolve in &[false, true] {
        for &method in &[
            SimplexMethod::Auto,
            SimplexMethod::Primal,
            SimplexMethod::Dual,
            SimplexMethod::DualAdvanced,
        ] {
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(180.0);
            opts.presolve = presolve;
            opts.simplex_method = method;
            let t0 = Instant::now();
            let r = solve_with(&lp, &opts);
            eprintln!(
                "pilot-ja presolve={} method={:?} -> status={:?} obj={:.10e} iters={} time={:.2}s",
                presolve,
                method,
                r.status,
                r.objective,
                r.iterations,
                t0.elapsed().as_secs_f64()
            );
        }
    }
}

#[test]
fn pilot_ja_must_not_be_unbounded_with_presolve() {
    let lp = load_pilot_ja_lp();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(180.0);
    opts.presolve = true;
    let r = solve_lp_with_opts(&lp, &opts);
    assert_ne!(
        r.status,
        SolveStatus::Unbounded,
        "pilot-ja is known finite-optimum; presolve must not return Unbounded"
    );
}
