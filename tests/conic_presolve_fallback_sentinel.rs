//! Status-regression sentinel for the tier-1 conic presolve: when the
//! REDUCED solve ends inconclusive (here: the reduced nql60 system stalls
//! at the eps-tightened tolerance and the reduced-space honesty gate demotes
//! to `NumericalError`), `solve_socp` must FALL BACK to the unreduced solve
//! and report what it achieves — never map the inconclusive reduced outcome
//! through postsolve as the final answer.
//!
//! Ground truth from the pre-presolve binary (origin/main build, run
//! 2026-11-17): `nql60,Optimal,-9.3505285040e-1,20,0.905`. Accepting the
//! reduced `NumericalError` (the pre-fix `status != MaxIterations` gate)
//! makes this test FAIL — verified during development of the fallback rule.
//!
//! Data is gitignored; the test skips (per `cbf_feasibility.rs` convention)
//! when the file is absent, and is `#[ignore]`d as bench-class runtime
//! (~75 s: reduced solve + fallback solve).

use otspot_core::conic::{solve_socp, ConicOptions, ConicProblem};
use otspot_core::problem::SolveStatus;
use otspot_io::cbf::{parse_cbf, CbfProblem};

/// Legacy (pre-presolve) optimum, [measured] above; the slack covers two
/// independent `1e-6`-tol solves plus the postsolve round-trip.
const NQL60_LEGACY_OBJ: f64 = -9.3505285040e-1;
/// Agreement slack for two `1e-6`-tol solves plus postsolve round-trip.
const OBJ_SLACK: f64 = 1e-4;
/// Feasibility slack (same convention as `cbf_feasibility.rs` `FEAS_SLACK`).
const FEAS_SLACK: f64 = 1e-5;

/// Independent original-space feasibility check with `s` recomputed from
/// `A`/`G` (never trusted from the solver), per the contract's independence
/// rule.
fn assert_conic_feasible(problem: &ConicProblem, x: &[f64]) {
    let ax = problem.a.mat_vec_mul(x).expect("A x");
    for (i, (&axi, &bi)) in ax.iter().zip(&problem.b).enumerate() {
        assert!((axi - bi).abs() <= FEAS_SLACK, "equality row {i} violated");
    }
    let gx = problem.g.mat_vec_mul(x).expect("G x");
    let s: Vec<f64> = problem.h.iter().zip(&gx).map(|(h, g)| h - g).collect();
    for (i, &si) in s[..problem.cone.l].iter().enumerate() {
        assert!(si >= -FEAS_SLACK, "orthant row {i}: s={si}");
    }
    let mut cur = problem.cone.l;
    for (bi, &dim) in problem.cone.soc.iter().enumerate() {
        let t = s[cur];
        let nu: f64 = s[cur + 1..cur + dim]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(t >= nu - FEAS_SLACK, "SOC block {bi}: t={t} ||u||={nu}");
        cur += dim;
    }
}

#[test]
#[ignore = "bench-class runtime (~75 s: reduced solve + fallback); run via --profile heavy --run-ignored all"]
fn nql60_inconclusive_reduced_solve_falls_back_to_optimal() {
    let path = std::path::Path::new("data/cblib_socp/nql60.cbf");
    if !path.exists() {
        eprintln!("[presolve-fallback] skip: data missing: {}", path.display());
        return;
    }
    let CbfProblem::Socp { problem, .. } = parse_cbf(path).expect("parse nql60") else {
        panic!("nql60 must be a continuous SOCP");
    };
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "fallback must recover the pre-presolve status, got {:?}",
        res.status
    );
    assert!(
        (res.objective - NQL60_LEGACY_OBJ).abs() <= OBJ_SLACK,
        "obj {} vs legacy {NQL60_LEGACY_OBJ}",
        res.objective
    );
    assert_conic_feasible(&problem, &res.x);
}
