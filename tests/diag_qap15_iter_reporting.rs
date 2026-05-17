//! qap15: Timeout 返却時の iterations フィールドが 0 でないことを assert。
//! observability bug regression sentinel (task #17)。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::solve_qp_with;
use std::path::Path;
use std::time::Instant;

#[test]
fn qap15_timeout_reports_nonzero_iter() {
    let path = Path::new("data/lp_problems_extra/qap15.QPS");
    assert!(path.exists(), "data missing: qap15.QPS");
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(15.0); // 短時間 timeout で確実に Timeout に到達
    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    eprintln!(
        "[qap15-iter] status={:?} wall={:.3}s iters={}",
        result.status,
        t0.elapsed().as_secs_f64(),
        result.iterations
    );
    if result.status == SolveStatus::Timeout {
        assert!(
            result.iterations > 0,
            "Timeout 時 iterations は実 iter 数を返すべき (observability bug)"
        );
    }
}
