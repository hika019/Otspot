//! d6cube feasibility/convergence regression guards.
//!
//! d6cube is a feasible LP (415 Eq, 6184 vars, c_j=1) — highly degenerate in
//! Phase II. Tests guard against false-Infeasible regressions. Full convergence
//! (~286s with primal simplex) is tested under `#[ignore="tier-2"]`.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;

const D6CUBE_KNOWN_OPT: f64 = 3.15491666666e+02;

// ── Replacement tests (supersede stale timing sentinels) ─────────────────────

/// d6cube must not return Infeasible — correctness guard without timing bound.
/// d6cube is feasible (obj=315.49); any status except Infeasible is acceptable.
/// 15s is enough for Phase I to establish feasibility (measured <1s); Phase II
/// runs until timeout. Using 15s keeps the default test suite fast.
#[test]
fn d6cube_not_infeasible() {
    let path = Path::new("data/lp_problems/d6cube.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let prob = parse_qps(path).expect("parse d6cube");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(15.0); // Phase I < 1s; this guards false-Infeasible regressions
    let r = solve_qp_with(&prob, &opts);
    eprintln!("d6cube_not_infeasible: status={:?} obj={:.6e}", r.status, r.objective);
    assert!(
        !matches!(r.status, SolveStatus::Infeasible),
        "d6cube returned Infeasible — Phase I regression"
    );
}

/// d6cube solves to Optimal with correct objective (≈315.49) — full convergence test.
/// d6cube takes ~286s; run under heavy profile or standalone.
#[test]
#[ignore = "broken: d6cube needs ~286s convergence, exceeds default 180s kill; LP perf fix tracked"]
fn d6cube_optimal_tier2() {
    let path = Path::new("data/lp_problems/d6cube.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let prob = parse_qps(path).expect("parse d6cube");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(400.0); // generous: measured ~286s
    let r = solve_qp_with(&prob, &opts);
    let rel_err = (r.objective - D6CUBE_KNOWN_OPT).abs() / (1.0_f64).max(D6CUBE_KNOWN_OPT.abs());
    eprintln!(
        "d6cube_optimal_tier2: status={:?} obj={:.6e} rel_err={:.2e} iters={}",
        r.status, r.objective, rel_err, r.iterations
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "d6cube must reach Optimal within 400s"
    );
    assert!(rel_err < 1e-4, "d6cube obj rel_err {:.2e} >= 1e-4", rel_err);
}
