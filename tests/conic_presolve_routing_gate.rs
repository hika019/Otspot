//! End-to-end sentinel for the conic presolve ROUTING GATE: the solve-time
//! route (`solve_socp` → [`route_reduction`] over the reduction stats) must
//! match the measured band table — the reduced-attempt classes
//! (nql30 / qssp30 / chainsing-1000-1, dense-KKT ≤ [`
//! PRESOLVE_ATTEMPT_DENSE_KKT_GIB_MAX`]) keep attempting the reduced solve;
//! the nql60 class (reduced dense-KKT 5.0 GiB, measured stall-then-fallback
//! of 64.6 s vs 0.9 s presolve-free) must NEVER enter the reduced-attempt
//! path. Reverting the gate (or inflating its threshold) makes the nql60
//! rows here fail — by route table AND by wall-clock — so the regression it
//! guards cannot silently return.
//!
//! The gate decides from reduction stats before any IPM work (see the
//! constant's doc for the measurement basis); this file proves the decision
//! is wired into `solve_socp` end to end. Data is gitignored; each case
//! skips when its file is absent (same convention as `cbf_feasibility.rs`).

use otspot_core::conic::{
    route_reduction, solve_socp, solve_socp_route, ConicOptions, ConicProblem, PresolveRoute,
};
use otspot_core::problem::SolveStatus;
use otspot_io::cbf::{parse_cbf, CbfProblem};
use std::time::Instant;

/// Bench-harness tolerance (AGENTS.md 基準値 eps=1e-6), same basis as the
/// oracle runs and the fallback sentinel.
const BENCH_TOL: f64 = 1e-6;

/// Wall-clock cap for the gate-routed nql60 solve. Measured gate-ON route
/// (legacy direct): 0.837–0.905 s (legacy binaries, 12 GiB band, 2026-11-18).
/// Gate-OFF cost on the same fixture: 64.6 s + 0.9 s re-solve (report §4.4).
/// 30 s sits 33× above the routed measurement and 2× below the regression,
/// so it cannot flake on load yet still fails hard on revert.
const NQL60_GATE_ON_SECS_MAX: f64 = 30.0;

/// Legacy optimum for nql60 (pre-presolve oracle, see
/// `conic_presolve_fallback_sentinel.rs`) and agreement slack for two
/// `1e-6`-tol solves.
const NQL60_LEGACY_OBJ: f64 = -9.3505285040e-1;
const OBJ_SLACK: f64 = 1e-4;

/// Feasibility slack (same convention as `cbf_feasibility.rs` /
/// `conic_presolve_fallback_sentinel.rs` `FEAS_SLACK`).
const FEAS_SLACK: f64 = 1e-5;

/// Band routing table: `(file, expected route)`. Expected routes are the
/// measured verdicts of report §4.4 (converged attempts vs stalled attempt).
const ROUTING_TABLE: &[(&str, PresolveRoute)] = &[
    ("nql30.cbf", PresolveRoute::AttemptReduced),
    ("qssp30.cbf", PresolveRoute::AttemptReduced),
    ("chainsing-1000-1.cbf", PresolveRoute::AttemptReduced),
    ("nql60.cbf", PresolveRoute::LegacyDirect),
];

fn load_socp(name: &str) -> Option<ConicProblem> {
    let path = std::path::Path::new("data/cblib_socp").join(name);
    if !path.exists() {
        eprintln!("[routing-gate] skip: data missing: {}", path.display());
        return None;
    }
    let cbf = parse_cbf(&path).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    match cbf {
        CbfProblem::Socp { problem, .. } => Some(problem),
        CbfProblem::Misocp { problem, .. } => Some(problem.base),
    }
}

/// Independent original-space feasibility check with `s` recomputed from
/// `A`/`G` (never trusted from the solver), per the independence rule.
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

/// (a) Decision table on the REAL reductions: reduce each contracted
/// instance (cheap, pre-solve) and assert the gate's route matches the
/// measured band verdict. Reverting the gate constant makes nql60 flip to
/// `AttemptReduced` here.
#[test]
fn cblib_routing_gate_decision_table_on_real_reductions() {
    let mut verified = 0usize;
    for &(file, want) in ROUTING_TABLE {
        let Some(problem) = load_socp(file) else {
            continue;
        };
        let Some(red) = otspot_core::conic::ConicReduction::reduce(&problem) else {
            panic!("{file}: presolve declined — routing table cannot be checked");
        };
        let stats = red.stats();
        let got = route_reduction(&stats);
        eprintln!(
            "[routing-gate] {file}: kkt {} -> {} ({:.2} GiB dense) => {got:?} (want {want:?})",
            stats.kkt0,
            stats.kkt1,
            otspot_core::conic::dense_kkt_gib(stats.kkt1)
        );
        assert_eq!(got, want, "{file}: gate route differs from measured table");
        verified += 1;
    }
    assert!(
        verified > 0,
        "no CBLIB data present: routing table cannot be checked (silent-skip gate)"
    );
}

/// (b) The nql60 class never enters the reduced-attempt path: `solve_socp`
/// on nql60 must return the legacy `Optimal` (objective = pre-presolve
/// oracle) with legacy-class wall time. Gate reverted → attempt + fallback
/// (~65 s) blows the wall-clock cap → fail.
#[test]
fn nql60_class_never_enters_reduced_attempt_path() {
    let Some(problem) = load_socp("nql60.cbf") else {
        return;
    };
    let opts = ConicOptions {
        tol: BENCH_TOL,
        ..ConicOptions::default()
    };
    let t0 = Instant::now();
    let res = solve_socp(&problem, &opts);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[routing-gate] nql60 gate-routed solve: {secs:.3} s {:?}",
        res.status
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert!(
        (res.objective - NQL60_LEGACY_OBJ).abs() <= OBJ_SLACK,
        "obj {} vs legacy {NQL60_LEGACY_OBJ}",
        res.objective
    );
    assert_conic_feasible(&problem, &res.x);
    assert!(
        secs < NQL60_GATE_ON_SECS_MAX,
        "nql60 took {secs:.1} s — the reduced attempt path was entered \
         (gate regression: expected legacy-direct <1 s)"
    );
}

/// (c) Mis-prediction fallback semantics survive the gate: forcing the
/// reduced attempt on the gate-declined nql60 fixture (the measured stall
/// case) must still fall back to the unreduced solve and report the legacy
/// `Optimal` — the gate is routing, not a status contract change. Reverting
/// the terminal-status acceptance fix (e710daad) makes the forced attempt
/// leak the reduced `NumericalError` and fail here.
#[test]
#[ignore = "bench-class runtime (~75 s: stalled reduced solve + fallback); run via --profile heavy --run-ignored all"]
fn forced_attempt_on_declined_class_still_falls_back_to_optimal() {
    let Some(problem) = load_socp("nql60.cbf") else {
        return;
    };
    let opts = ConicOptions {
        tol: BENCH_TOL,
        ..ConicOptions::default()
    };
    let res = solve_socp_route(&problem, &opts, Some(PresolveRoute::AttemptReduced));
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "forced-attempt fallback must recover the legacy status, got {:?}",
        res.status
    );
    assert!(
        (res.objective - NQL60_LEGACY_OBJ).abs() <= OBJ_SLACK,
        "obj {} vs legacy {NQL60_LEGACY_OBJ}",
        res.objective
    );
    assert_conic_feasible(&problem, &res.x);
}

/// (d) Forced `LegacyDirect` bypasses the presolve stage on the ATTEMPT
/// classes too, with identical results: the gate below threshold must not
/// have changed what the presolve-free pipeline reports (status/objective
/// agreement between routes on small instances is the no-regression floor
/// for the attempt classes).
#[test]
fn forced_legacy_route_agrees_with_gated_route_on_attempt_class() {
    let Some(problem) = load_socp("nql30.cbf") else {
        return;
    };
    let opts = ConicOptions {
        tol: BENCH_TOL,
        ..ConicOptions::default()
    };
    let routed = solve_socp(&problem, &opts);
    let legacy = solve_socp_route(&problem, &opts, Some(PresolveRoute::LegacyDirect));
    assert_eq!(routed.status, SolveStatus::Optimal);
    assert_eq!(legacy.status, SolveStatus::Optimal);
    assert!(
        (routed.objective - legacy.objective).abs() <= OBJ_SLACK,
        "gated {} vs legacy {}",
        routed.objective,
        legacy.objective
    );
    assert_conic_feasible(&problem, &routed.x);
    assert_conic_feasible(&problem, &legacy.x);
}
