//! BOYD1 / BOYD2: inner の gap 判定が gap の評価誤差より小さい eps を要求しないこと。
//!
//! Ruiz 増幅が大きい (BOYD1: 1.1e6) と inner eps は `IPM_EPS_NOISE_FLOOR` (2.2e-14) まで
//! 締まる。残差と μ は per-component 相対なので 1e-15 まで下がるが、相対 gap は打ち消し合う
//! 内積の差なので評価誤差の床 (BOYD1: 2.7e-12) に張り付く。gap をこの eps そのものと比べると
//! 収束判定が成立せず、BOYD1 は 40 反復で済む収束点から residual_stall まで 162 反復を回る
//! (BOYD2: 116 → 369 反復)。
//!
//! Sentinel: gap 許容から評価誤差の下限を外すと反復数が上限を超えて fail する。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;
use otspot::SolveStatus;
use std::path::Path;

const TIMEOUT_SECS: f64 = 120.0;

fn solve(name: &str) -> otspot::SolverResult {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/maros_meszaros")
        .join(format!("{name}.QPS"));
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(&path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_SECS);
    let result = solve_qp_with(&prob, &opts);
    eprintln!(
        "{name}: status={:?} obj={:.10e} iters={}",
        result.status, result.objective, result.iterations
    );
    result
}

/// 上限は誤差床を越えられない場合の実測反復数 (162 / 369) と、床を考慮した場合の
/// 実測反復数 (40 / 116) の間に取る。
fn assert_converges_without_stalling_on_gap_floor(name: &str, max_iterations: usize) {
    let result = solve(name);
    assert_eq!(result.status, SolveStatus::Optimal, "{name}");
    assert!(
        result.iterations <= max_iterations,
        "{name}: {} iterations > {max_iterations}: inner kept iterating on the gap \
         evaluation-noise floor",
        result.iterations
    );
}

#[test]
fn boyd1_inner_gap_gate_respects_evaluation_noise() {
    assert_converges_without_stalling_on_gap_floor("BOYD1", 100);
}

#[test]
fn boyd2_inner_gap_gate_respects_evaluation_noise() {
    assert_converges_without_stalling_on_gap_floor("BOYD2", 240);
}
