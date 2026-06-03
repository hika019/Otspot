//! Diagnose false-Unbounded on Netlib pilot-ja.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::solve_with as solve_lp_with_opts;
use std::path::Path;

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
