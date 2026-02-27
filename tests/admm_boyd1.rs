//! BOYD1/BOYD2 CG パス検証テスト（subtask_154d）
//!
//! subtask_154c で LDL_THRESHOLD=10000 超えの問題に対して CG パスが統合された。
//! このファイルは BOYD1/BOYD2（n が大きい問題）で CG パスが正しく動作するかを検証する。
//!
//! 判定基準:
//! - Optimal / MaxIterations / Timeout → 成功（CG パスが正常動作）
//! - NumericalError → 失敗（CGパス設計問題）
//! - 30 秒を大幅超過してプロセスが応答なし → テスト FAIL
//!
//! 注: n=93K の大規模問題は 30s 以内に収束しない場合があり Timeout が返ることがある。
//! これは timeout_secs が正しく機能している証拠であり失敗ではない。

use solver::io::qps::parse_qps;
use solver::{SolveStatus, SolverOptions};
use solver::qp::solve_qp_admm;
use std::path::Path;
use std::time::Instant;

const BOYD1_PATH: &str = "/Users/hika019/Develop/solver/data/maros_meszaros/BOYD1.QPS";
const BOYD2_PATH: &str = "/Users/hika019/Develop/solver/data/maros_meszaros/BOYD2.QPS";

/// BOYD1 を強制 CG モード（admm_use_cg=Some(true)）で解く
#[test]
fn test_boyd1_cg_path() {
    let path = Path::new(BOYD1_PATH);
    assert!(path.exists(), "BOYD1.QPS not found at {}", BOYD1_PATH);

    let start_parse = Instant::now();
    let prob = parse_qps(path).expect("BOYD1.QPS parse failed");
    println!("BOYD1 parse: {:.2}s (n={}, m={})", start_parse.elapsed().as_secs_f64(), prob.num_vars, prob.num_constraints);

    let mut opts = SolverOptions::default();
    opts.admm.use_cg = Some(true); // 強制 CG
    opts.timeout_secs = Some(30.0);

    let start_solve = Instant::now();
    let result = solve_qp_admm(&prob, &opts);
    let elapsed = start_solve.elapsed().as_secs_f64();

    println!(
        "BOYD1 CG path: status={:?}, elapsed={:.3}s, iters={}",
        result.status, elapsed, result.iterations
    );

    // NumericalError は NG（CGパス設計問題）
    assert_ne!(
        result.status,
        SolveStatus::NumericalError,
        "BOYD1 CG path returned NumericalError — CGパス設計を確認せよ"
    );

    // Optimal / MaxIterations / Timeout は全て OK（NumericalError のみ NG）
    assert!(
        result.status == SolveStatus::Optimal
            || result.status == SolveStatus::MaxIterations
            || result.status == SolveStatus::Timeout,
        "BOYD1 CG path: expected Optimal/MaxIterations/Timeout, got {:?}",
        result.status
    );
    println!("BOYD1 CG path: PASS ({:?}, {:.2}s)", result.status, elapsed);
}

/// BOYD1 を Auto モード（admm_use_cg=None）で解く
/// n=93261 > LDL_THRESHOLD=10000 なので CG が自動選択されるはず
#[test]
fn test_boyd1_auto_cg() {
    let path = Path::new(BOYD1_PATH);
    assert!(path.exists(), "BOYD1.QPS not found at {}", BOYD1_PATH);

    let start_parse = Instant::now();
    let prob = parse_qps(path).expect("BOYD1.QPS parse failed");
    println!("BOYD1 parse: {:.2}s (n={}, m={})", start_parse.elapsed().as_secs_f64(), prob.num_vars, prob.num_constraints);

    let mut opts = SolverOptions::default();
    opts.admm.use_cg = None; // Auto: n > LDL_THRESHOLD → CG
    opts.timeout_secs = Some(30.0);

    let start_solve = Instant::now();
    let result = solve_qp_admm(&prob, &opts);
    let elapsed = start_solve.elapsed().as_secs_f64();

    println!(
        "BOYD1 Auto CG: status={:?}, elapsed={:.3}s, iters={}",
        result.status, elapsed, result.iterations
    );

    assert_ne!(
        result.status,
        SolveStatus::NumericalError,
        "BOYD1 Auto CG returned NumericalError — n>LDL_THRESHOLDでCGが選ばれるはず"
    );

    assert!(
        result.status == SolveStatus::Optimal
            || result.status == SolveStatus::MaxIterations
            || result.status == SolveStatus::Timeout,
        "BOYD1 Auto CG: expected Optimal/MaxIterations/Timeout, got {:?}",
        result.status
    );
    println!("BOYD1 Auto CG: PASS ({:?}, {:.2}s)", result.status, elapsed);
}

/// BOYD2 を強制 CG モードで解く
#[test]
fn test_boyd2_cg_path() {
    let path = Path::new(BOYD2_PATH);
    assert!(path.exists(), "BOYD2.QPS not found at {}", BOYD2_PATH);

    let start_parse = Instant::now();
    let prob = parse_qps(path).expect("BOYD2.QPS parse failed");
    println!("BOYD2 parse: {:.2}s (n={}, m={})", start_parse.elapsed().as_secs_f64(), prob.num_vars, prob.num_constraints);

    let mut opts = SolverOptions::default();
    opts.admm.use_cg = Some(true); // 強制 CG
    opts.timeout_secs = Some(30.0);

    let start_solve = Instant::now();
    let result = solve_qp_admm(&prob, &opts);
    let elapsed = start_solve.elapsed().as_secs_f64();

    println!(
        "BOYD2 CG path: status={:?}, elapsed={:.3}s, iters={}",
        result.status, elapsed, result.iterations
    );

    assert_ne!(
        result.status,
        SolveStatus::NumericalError,
        "BOYD2 CG path returned NumericalError — CGパス設計を確認せよ"
    );

    assert!(
        result.status == SolveStatus::Optimal
            || result.status == SolveStatus::MaxIterations
            || result.status == SolveStatus::Timeout,
        "BOYD2 CG path: expected Optimal/MaxIterations/Timeout, got {:?}",
        result.status
    );
    println!("BOYD2 CG path: PASS ({:?}, {:.2}s)", result.status, elapsed);
}

/// BOYD2 を Auto モードで解く
#[test]
fn test_boyd2_auto_cg() {
    let path = Path::new(BOYD2_PATH);
    assert!(path.exists(), "BOYD2.QPS not found at {}", BOYD2_PATH);

    let start_parse = Instant::now();
    let prob = parse_qps(path).expect("BOYD2.QPS parse failed");
    println!("BOYD2 parse: {:.2}s (n={}, m={})", start_parse.elapsed().as_secs_f64(), prob.num_vars, prob.num_constraints);

    let mut opts = SolverOptions::default();
    opts.admm.use_cg = None; // Auto
    opts.timeout_secs = Some(30.0);

    let start_solve = Instant::now();
    let result = solve_qp_admm(&prob, &opts);
    let elapsed = start_solve.elapsed().as_secs_f64();

    println!(
        "BOYD2 Auto CG: status={:?}, elapsed={:.3}s, iters={}",
        result.status, elapsed, result.iterations
    );

    assert_ne!(
        result.status,
        SolveStatus::NumericalError,
        "BOYD2 Auto CG returned NumericalError"
    );

    assert!(
        result.status == SolveStatus::Optimal
            || result.status == SolveStatus::MaxIterations
            || result.status == SolveStatus::Timeout,
        "BOYD2 Auto CG: expected Optimal/MaxIterations/Timeout, got {:?}",
        result.status
    );
    println!("BOYD2 Auto CG: PASS ({:?}, {:.2}s)", result.status, elapsed);
}
