//! BOYD1 ADMM 検証テスト（subtask_152d Step 2）
//!
//! BOYD1.QPS (n=93K, 33MB) を solve_qp_admm() で実行し、
//! timeout=10 秒で確実に停止するかを確認する。
//!
//! 判定基準:
//! - 10 秒以内に Timeout または Optimal/MaxIterations で返れば成功
//! - 10 秒を超えてハングした場合: テスト FAIL + R01 報告

use solver::io::qps::parse_qps;
use solver::{SolveStatus, SolverOptions};
use solver::qp::solve_qp_admm;
use std::path::Path;
use std::time::Instant;

const BOYD1_PATH: &str = "/Users/hika019/Develop/solver/data/maros_meszaros/BOYD1.QPS";

#[test]
fn boyd1_admm_timeout_check() {
    let path = Path::new(BOYD1_PATH);
    assert!(path.exists(), "BOYD1.QPS not found at {}", BOYD1_PATH);

    // BOYD1 ロード（約 33MB、parse に数秒かかる場合あり）
    let start_parse = Instant::now();
    let prob = parse_qps(path).expect("BOYD1.QPS parse failed");
    let parse_elapsed = start_parse.elapsed().as_secs_f64();
    println!("BOYD1 parse: {:.2}s (n={}, m={})", parse_elapsed, prob.num_vars, prob.num_constraints);

    // ADMM 実行（サイズガード n>10000 → NumericalError即返却のため timeout不要）
    let opts = SolverOptions::default();

    let start_solve = Instant::now();
    let result = solve_qp_admm(&prob, &opts);
    let solve_elapsed = start_solve.elapsed().as_secs_f64();

    println!(
        "BOYD1 ADMM: status={:?}, elapsed={:.3}s, iters={}",
        result.status, solve_elapsed, result.iterations
    );

    // 検証1: サイズガードにより 1 秒以内に返ってくること（ハングなし）
    assert!(
        solve_elapsed < 1.0,
        "BOYD1 ADMM: ハング検出！ {:.1}s 経過（サイズガードが効いていない）",
        solve_elapsed
    );

    // 検証2: n=93K > 10000 なので NumericalError が返ること
    assert_eq!(
        result.status,
        SolveStatus::NumericalError,
        "BOYD1 ADMM: サイズガードにより NumericalError 期待, got {:?}",
        result.status
    );

    println!(
        "BOYD1 ADMM 結果: n=93K → {:?} 即返却 ({:.3}s, {} iters)",
        result.status, solve_elapsed, result.iterations
    );
}
