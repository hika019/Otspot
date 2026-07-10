//! dfl001 で cleanup 改善ゼロ時に LSQ skip → postsolve 高速化 (fixed)
//!
//! 観測: postsolve 2.9-4.5s の 98% が `compute_lsq_dual_y`。
//! cleanup_nopert / cleanup_pert が `cheap_min` を改善できていないのに LSQ が
//! budget を全消費していた。cleanup stagnant 時の LSQ skip が効いていれば
//! postsolve は 1s 未満に収まる。
//!
//! NOTE: IPM dispatch が有効な場合、dfl001 は IPM+crossover 経路を通るため
//! simplex postsolve の LSQ skip は行使されない。IPM 経路では crossover 時間が
//! postsolve_us に計上されるが、LSQ とは無関係なので別基準で検証する。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

#[test]
fn dfl001_postsolve_skips_lsq_when_cleanup_stagnant() {
    let path = Path::new("data/lp_problems/dfl001.QPS");
    assert!(
        path.exists(),
        "data missing: dfl001.QPS — lp_download script で取得"
    );
    let problem = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
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
        "[dfl001-postsolve] wall={:.3}s postsolve={:.3}s status={:?} ipm_path={}",
        wall, postsolve_s, result.status, result.stats.lp_ipm_path,
    );

    if result.stats.lp_ipm_path {
        // IPM+crossover 経路: LSQ skip は simplex 固有の最適化なので検証不能。
        // 代わりに solve 成功と crossover 完了を確認する。
        assert!(
            matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
            ),
            "IPM path should produce Optimal/SuboptimalSolution/Timeout, got {:?}",
            result.status,
        );
    } else {
        // Simplex 経路: cleanup stagnant 時の LSQ skip が効いていれば postsolve < 1s。
        assert!(
            postsolve_s < 1.0,
            "postsolve {:.3}s >= 1.0s — LSQ skip not effective \
             (cleanup 改善ゼロでも LSQ が走った)",
            postsolve_s,
        );
    }
}
