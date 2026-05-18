//! Postsolve cleanup-LP gate regression guard。
//!
//! `run_postsolve` の cleanup LP は `min(df_loop, df_gs)` の sufficiency check で
//! gate され、perturbation variant の deadline は plain variant の 4× に制限される。
//! 本 test は wall-clock threshold を pin する。
//!
//! Gate は **simplex postsolve** 経路の挙動を pin するもの。`#33` の LP→IPM
//! dispatch 導入後、サイズ閾値を超える LP は IPM 経路で解かれ simplex postsolve
//! 自体が走らない (`timing_breakdown` が `None`)。その場合 cleanup-LP 退化は
//! 原理的に発生しえないので NaN を「regression 検出不能」として PASS 扱いする。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;

fn solve(path: &str, timeout_s: f64) -> (SolveStatus, f64, f64) {
    let p = Path::new(path);
    assert!(p.exists(), "data missing: {}", path);
    let prob = parse_qps(p).expect("parse");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_s);
    let t0 = std::time::Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let postsolve_s = r.timing_breakdown.map(|t| t.postsolve_us as f64 / 1e6).unwrap_or(f64::NAN);
    eprintln!("{}: status={:?} wall={:.2}s postsolve={:.2}s",
        path, r.status, wall, postsolve_s);
    (r.status, wall, postsolve_s)
}

/// Returns true if `postsolve_s` is NaN, indicating the LP took the IPM
/// dispatch path (no simplex postsolve ran). Cleanup-LP gate cannot
/// regress in that case.
fn ipm_dispatched(postsolve_s: f64) -> bool {
    postsolve_s.is_nan()
}

/// wood1p was solving in ~9 s but spending another ~25 s in a cleanup LP
/// that always returned Inf. Postsolve must stay under 2 s.
#[test]
fn wood1p_postsolve_under_2s() {
    let (status, _wall, postsolve_s) = solve("data/lp_problems/wood1p.QPS", 60.0);
    assert!(matches!(status, SolveStatus::Optimal), "wood1p must reach Optimal");
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
    let (_status, _wall, postsolve_s) = solve("data/lp_problems/d6cube.QPS", 60.0);
    if ipm_dispatched(postsolve_s) {
        // IPM 経路で解かれた → simplex cleanup-LP は走らない (#33)。
        return;
    }
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
    let (_status, _wall, postsolve_s) = solve("data/lp_problems/greenbea.QPS", 60.0);
    if ipm_dispatched(postsolve_s) {
        return;
    }
    assert!(
        postsolve_s < 5.0,
        "greenbea postsolve {:.2}s exceeded 5s — cleanup_pert deadline cap likely regressed",
        postsolve_s
    );
}
