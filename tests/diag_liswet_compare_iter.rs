//! LISWET1 (converge OK) vs LISWET9 (wrong basin) の iter 単位 trace 比較 diag。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;

fn solve_with_trace(name: &str) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
    let prob = parse_qps(&path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    eprintln!("\n========= [{}] start =========", name);
    let res = solve_qp_with(&prob, &opts);
    eprintln!("[{}] DONE status={:?} obj={:.6e}", name, res.status, res.objective);
}

#[test]
#[ignore = "diag liswet1 trace"]
fn diag_trace_liswet1() {
    solve_with_trace("LISWET1");
}

#[test]
#[ignore = "diag liswet9 trace"]
fn diag_trace_liswet9() {
    solve_with_trace("LISWET9");
}

#[test]
#[ignore = "diag liswet12 trace"]
fn diag_trace_liswet12() {
    solve_with_trace("LISWET12");
}
