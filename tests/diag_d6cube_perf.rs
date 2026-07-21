//! d6cube feasibility/convergence regression guards.
//!
//! d6cube is a feasible LP (415 Eq, 6184 vars, c_j=1) — highly degenerate in
//! Phase II. Tests guard against false-Infeasible regressions. Full convergence
//! is tested under `#[ignore="tier-2"]` in `d6cube_optimal_and_postsolve_fast_tier2`
//! below, which also carries the postsolve cleanup-LP regression guard (see
//! `tests/diag_postsolve_cleanup_gate.rs`, which covers the same gate for
//! wood1p/greenbea) — merged from the former `d6cube_optimal_tier2` and
//! `d6cube_postsolve_under_1s` sentinels so the ~150s solve is paid once.

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
    eprintln!(
        "d6cube_not_infeasible: status={:?} obj={:.6e}",
        r.status, r.objective
    );
    assert!(
        !matches!(r.status, SolveStatus::Infeasible),
        "d6cube returned Infeasible — Phase I regression"
    );
}

/// Solver deadline for the tier-2 convergence+postsolve test below.
///
/// Observed max wall for this solve (presolve=true, LP simplex path): 171.628s
/// (CI 3-thread, `lp_simplex_stall_d6cube_converges`) and 156.601s (local,
/// worst of 3). 400s matches `lp_simplex_stall_d6cube_converges` (360s) and the
/// former `d6cube_optimal_tier2` (400s), giving ~2.3x headroom. The merged
/// `d6cube_postsolve_under_1s` used 60s, but that value was never calibrated for
/// d6cube: a87a800e copied it verbatim from a shared 60s canary-profiling
/// harness (wood1p/greenbea/perold/etamacro), where it left d6cube's postsolve
/// assert vacuous (Optimal never reached within 60s).
const D6CUBE_DEADLINE_SECS: f64 = 400.0;

/// Postsolve regression threshold, inherited unchanged from the removed
/// `d6cube_postsolve_under_1s` (tests/diag_postsolve_cleanup_gate.rs): measured
/// postsolve_s = 0.002389 / 0.002410 / 0.002518s across 3 local runs, so 1.0s
/// keeps ~400x margin.
const D6CUBE_POSTSOLVE_THRESHOLD_SECS: f64 = 1.0;

/// d6cube solves to Optimal with correct objective (≈315.49) and its postsolve
/// cleanup-LP stays under the regression threshold — full convergence test,
/// merged with the postsolve gate since both exercise the identical solve
/// (presolve=true, LP simplex postsolve path) and previously paid for the same
/// ~150s solve twice (see module doc).
#[test]
#[ignore = "heavy/tier2: d6cube reaches Optimal in ~150s here (>30s default budget); run explicitly"]
fn d6cube_optimal_and_postsolve_fast_tier2() {
    let path = Path::new("data/lp_problems/d6cube.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let prob = parse_qps(path).expect("parse d6cube");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(D6CUBE_DEADLINE_SECS);
    let t0 = std::time::Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let postsolve_s = r
        .timing_breakdown
        .map(|t| t.postsolve_us as f64 / 1e6)
        .unwrap_or(f64::NAN);
    let rel_err = (r.objective - D6CUBE_KNOWN_OPT).abs() / (1.0_f64).max(D6CUBE_KNOWN_OPT.abs());
    eprintln!(
        "d6cube_optimal_and_postsolve_fast_tier2: status={:?} obj={:.6e} rel_err={:.2e} iters={} wall={:.3}s postsolve={:.6}s",
        r.status, r.objective, rel_err, r.iterations, wall, postsolve_s
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "d6cube must reach Optimal within {}s; got {:?}",
        D6CUBE_DEADLINE_SECS,
        r.status
    );
    assert!(rel_err < 1e-4, "d6cube obj rel_err {:.2e} >= 1e-4", rel_err);
    assert!(
        postsolve_s < D6CUBE_POSTSOLVE_THRESHOLD_SECS,
        "d6cube postsolve {:.6}s exceeded {}s — cleanup-LP gate likely regressed",
        postsolve_s,
        D6CUBE_POSTSOLVE_THRESHOLD_SECS
    );
}
