//! task #27: ken-13 deadline 不遵守 fix の TDD diag + 大規模 LP regression guard.
//!
//! 既知症状 (docs/known_bugs.md): ken-13.QPS (~30000 行) で solver の `timeout=1000s`
//! が守られず、外側 gtimeout 1300s まで CPU 100% で暴走する。
//!
//! このテストは短縮版 (timeout=30s) で再現し、watchdog 内に solve が必ず戻ること
//! を GREEN 条件とする。watchdog 超過は test failure。data 欠落時は panic
//! (SKIP 禁止 — CLAUDE.md「検証空白は bug 不在を保証しない」)。
//!
//! 関連 fix commit: 5652027 (cleanup_lp deadline 継承) / 55e4cf7 (parent deadline 未設定時)。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::{LpProblem, SolveStatus};
use solver::{solve_with, QpProblem};
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn make_lp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

/// 共通ヘルパ: 指定 LP を `timeout_secs` で解く。watchdog 内に終わらなければ panic、
/// 終わった場合は (status, wall_secs) を返す。
fn solve_with_watchdog(
    qps_path: &Path,
    timeout_secs: f64,
    watchdog: Duration,
    label: &str,
) -> (SolveStatus, f64) {
    assert!(
        qps_path.exists(),
        "task #27 diag 必須: {:?} が見つからない (SKIP 禁止)",
        qps_path
    );
    let qp = parse_qps(qps_path).expect("parse QPS");
    let lp = make_lp(&qp);
    eprintln!(
        "[{label}] n={} m={} nnz(A)={}",
        lp.num_vars,
        lp.num_constraints,
        lp.a.values.len()
    );

    let (tx, rx) = mpsc::channel();
    let lp_clone = lp.clone();
    let label_owned = label.to_string();
    let handle = thread::Builder::new()
        .name(format!("{label}-solver"))
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(timeout_secs);
            let t0 = Instant::now();
            let r = solve_with(&lp_clone, &opts);
            let elapsed = t0.elapsed();
            let _ = tx.send((r.status, r.objective, elapsed, label_owned));
        })
        .expect("spawn solver thread");

    match rx.recv_timeout(watchdog) {
        Ok((status, obj, elapsed, _)) => {
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "[{label}] status={:?} obj={:.6e} wall={:.3}s (timeout_secs={timeout_secs}, watchdog={}s)",
                status,
                obj,
                secs,
                watchdog.as_secs_f64(),
            );
            let _ = handle.join();
            assert!(
                secs <= watchdog.as_secs_f64(),
                "task #27 [{label}]: wall={:.3}s が watchdog={}s を超過 (deadline 不遵守)",
                secs,
                watchdog.as_secs_f64()
            );
            (status, secs)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "task #27 FAIL [{label}]: solve_with が watchdog {}s 内に return しない (timeout_secs={timeout_secs}). deadline check が抜けている経路あり",
                watchdog.as_secs_f64(),
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("task #27 [{label}]: solver thread panicked before sending result");
        }
    }
}

/// task #27 メイン: ken-13 (~30000 行) — bug 報告本体。
///
/// timeout_secs=30s, watchdog=60s。watchdog 超過 / 異常 status は test failure。
#[test]
fn diag_ken13_deadline_must_stop_within_watchdog() {
    let path = Path::new("data/lp_problems/ken-13.QPS");
    let (status, _secs) = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "ken-13");
    assert!(
        matches!(
            status,
            SolveStatus::Timeout
                | SolveStatus::Optimal
                | SolveStatus::NumericalError
                | SolveStatus::Infeasible
        ),
        "task #27 ken-13: 予期せぬ status {:?} (停止はしたが状態不明)",
        status
    );
}

/// cross-verify: ken-11 (~14000 行) — 元から deadline 正常停止していた問題、
/// 回帰がないことを確認。
#[test]
fn diag_ken11_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/ken-11.QPS");
    let (_status, _secs) = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "ken-11");
}

/// cross-verify: dfl001 (~6000 行) — 1e-4 bench で TIMEOUT 1287s (1000s 超過) 報告あり。
#[test]
fn diag_dfl001_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/dfl001.QPS");
    let (_status, _secs) = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "dfl001");
}

/// cross-verify: pds-20 (~33000 行) — 1000s 級 PFEAS_FAIL 問題。
/// heavy (timeout=60s, watchdog=100s) のため `#[ignore]` で nextest --run-ignored only 経由のみ。
#[test]
#[ignore = "diag heavy ~60s; task #27 cross-verify, run with --run-ignored only"]
fn diag_pds20_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/pds-20.QPS");
    let (_status, _secs) = solve_with_watchdog(path, 60.0, Duration::from_secs(100), "pds-20");
}
