//! task #32 Phase 0 観測 2: LISWET1 (converge OK) vs LISWET9 (wrong basin) 比較。
//! IPPMM_ACTIVE_TRACE=1 を有効化し iter 単位の挙動を比較する。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::qp::solve_qp_with;

fn solve_with_trace(name: &str) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    if !path.exists() { eprintln!("[{}] missing", name); return; }
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
