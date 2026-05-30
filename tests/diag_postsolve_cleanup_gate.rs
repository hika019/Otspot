//! Postsolve cleanup-LP gate regression guard。
//!
//! `run_postsolve` の cleanup LP は `min(df_loop, df_gs)` の sufficiency check で
//! gate され、perturbation variant の deadline は plain variant の 4× に制限される。
//! 本 test は wall-clock threshold を pin する。
//!
//! Gate は **simplex postsolve** 経路の挙動を pin するもの。`#33` の LP→IPM
//! dispatch (`src/qp/lp_dispatch.rs`) 導入後、サイズ閾値を超える LP は IPM 経路で
//! 解かれ simplex postsolve 自体が走らない。その判定は public accessor
//! `otspot::qp::prefer_ipm_for_size(n, m)` を直接呼ぶ (`timing_breakdown`
//! の NaN proxy は presolve OFF の simplex 経路でも NaN になるため brittle、
//! reviewer C2 指摘で書き換え)。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::{prefer_ipm_for_size, solve_qp_with, QpProblem};
use std::path::Path;

fn load(path: &str) -> QpProblem {
    let p = Path::new(path);
    assert!(p.exists(), "data missing: {}", path);
    parse_qps(p).expect("parse")
}

fn solve(prob: &QpProblem, timeout_s: f64) -> (SolveStatus, f64, f64) {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_s);
    let t0 = std::time::Instant::now();
    let r = solve_qp_with(prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let postsolve_s = r
        .timing_breakdown
        .map(|t| t.postsolve_us as f64 / 1e6)
        .unwrap_or(f64::NAN);
    eprintln!(
        "n={} m={}: status={:?} wall={:.2}s postsolve={:.2}s",
        prob.num_vars, prob.num_constraints, r.status, wall, postsolve_s
    );
    (r.status, wall, postsolve_s)
}

/// wood1p was solving in ~9 s but spending another ~25 s in a cleanup LP
/// that always returned Inf. Postsolve must stay under 2 s.
#[test]
fn wood1p_postsolve_under_2s() {
    let prob = load("data/lp_problems/wood1p.QPS");
    // wood1p: n=2594, m=244 → simplex 経路 (n<3000, m<2000)。gate 適用対象。
    assert!(
        !prefer_ipm_for_size(prob.num_vars, prob.num_constraints),
        "wood1p must remain on simplex path for this gate to apply"
    );
    let (status, _wall, postsolve_s) = solve(&prob, 60.0);
    assert!(
        matches!(status, SolveStatus::Optimal),
        "wood1p must reach Optimal"
    );
    assert!(
        postsolve_s < 2.0,
        "wood1p postsolve {:.2}s exceeded 2s — cleanup-LP gate likely regressed",
        postsolve_s
    );
}

/// d6cube's postsolve used to swallow 15 s on a cleanup LP that returned Inf.
/// The cheap recovery already gives machine-zero dfeas.
#[test]
fn d6cube_postsolve_under_1s() {
    let prob = load("data/lp_problems/d6cube.QPS");
    if prefer_ipm_for_size(prob.num_vars, prob.num_constraints) {
        // IPM 経路で解かれる → simplex cleanup-LP は走らない。
        return;
    }
    let (_status, _wall, postsolve_s) = solve(&prob, 60.0);
    assert!(
        postsolve_s < 1.0,
        "d6cube postsolve {:.2}s exceeded 1s — cleanup-LP gate likely regressed",
        postsolve_s
    );
}

/// greenbea's cleanup_pert returned Inf after ~20 s; the 4× cap on its
/// deadline bounds postsolve well under what it used to consume.
#[test]
fn greenbea_postsolve_under_5s() {
    let prob = load("data/lp_problems/greenbea.QPS");
    if prefer_ipm_for_size(prob.num_vars, prob.num_constraints) {
        return;
    }
    let (_status, _wall, postsolve_s) = solve(&prob, 60.0);
    assert!(
        postsolve_s < 5.0,
        "greenbea postsolve {:.2}s exceeded 5s — cleanup_pert deadline cap likely regressed",
        postsolve_s
    );
}
