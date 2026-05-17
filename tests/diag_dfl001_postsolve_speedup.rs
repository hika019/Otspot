//! SPEED #4-fix: dfl001 で cleanup 改善ゼロ時に LSQ skip → postsolve 高速化
//!
//! 観測 (dfl001-probe #38): postsolve 2.9-4.5s の 98% が `compute_lsq_dual_y`。
//! cleanup_nopert / cleanup_pert が `cheap_min` を改善できていないのに LSQ が
//! budget を全消費していた。cleanup stagnant 時の LSQ skip が効いていれば
//! postsolve は 1s 未満に収まる。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

#[test]
fn dfl001_postsolve_skips_lsq_when_cleanup_stagnant() {
    let path = Path::new("data/lp_problems/dfl001.QPS");
    assert!(path.exists(), "data missing: dfl001.QPS — lp_download script で取得");
    let problem = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);
    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let postsolve_us = result
        .timing_breakdown
        .as_ref()
        .map(|t| t.postsolve_us)
        .unwrap_or(0);
    let postsolve_s = postsolve_us as f64 / 1e6;
    eprintln!(
        "[dfl001-postsolve] wall={:.3}s postsolve={:.3}s status={:?}",
        wall, postsolve_s, result.status
    );
    assert!(
        postsolve_s < 1.0,
        "postsolve {:.3}s >= 1.0s — LSQ skip not effective (cleanup 改善ゼロでも LSQ が走った)",
        postsolve_s
    );
}
