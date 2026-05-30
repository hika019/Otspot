//! Large-LP deadline regression guard.
//!
//! Solves ken-13 and related large Netlib LPs under a short `timeout_secs` and a
//! `mpsc::recv_timeout` watchdog. A solver path that ignores `options.deadline`
//! manifests as either watchdog expiry (hang) or wall ≫ watchdog.
//!
//! Data files are required; missing data panics rather than skips so absence is
//! never silently mistaken for absence of the bug.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::qp::SOLVE_STACK_SIZE;
use otspot::{solve_with, QpProblem};
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

/// Solve `qps_path` with `timeout_secs`; fail if the solver thread does not
/// return within `watchdog` or returns with wall > `watchdog`.
///
/// The 8 MiB stack matches `qp::SOLVE_STACK_SIZE` — faer supernodal recursion
/// overflows the default 2 MiB on large bases.
fn solve_with_watchdog(
    qps_path: &Path,
    timeout_secs: f64,
    watchdog: Duration,
    label: &str,
) -> (SolveStatus, f64) {
    assert!(qps_path.exists(), "data required (no SKIP): {:?}", qps_path);
    let qp = parse_qps(qps_path).expect("parse QPS");
    let lp = make_lp(&qp);
    eprintln!(
        "[{label}] n={} m={} nnz(A)={}",
        lp.num_vars,
        lp.num_constraints,
        lp.a.values().len()
    );

    let (tx, rx) = mpsc::channel();
    let lp_clone = lp.clone();
    let handle = thread::Builder::new()
        .name(format!("{label}-solver"))
        .stack_size(SOLVE_STACK_SIZE)
        .spawn(move || {
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(timeout_secs);
            let t0 = Instant::now();
            let r = solve_with(&lp_clone, &opts);
            let _ = tx.send((r.status, r.objective, t0.elapsed()));
        })
        .expect("spawn solver thread");

    match rx.recv_timeout(watchdog) {
        Ok((status, obj, elapsed)) => {
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "[{label}] status={:?} obj={:.6e} wall={:.3}s (timeout={timeout_secs}s, watchdog={}s)",
                status,
                obj,
                secs,
                watchdog.as_secs_f64(),
            );
            let _ = handle.join();
            assert!(
                secs <= watchdog.as_secs_f64(),
                "[{label}] wall={:.3}s exceeded watchdog={}s",
                secs,
                watchdog.as_secs_f64()
            );
            (status, secs)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "[{label}] solve_with did not return within watchdog {}s (timeout={timeout_secs}s) — deadline path missing",
            watchdog.as_secs_f64(),
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("[{label}] solver thread panicked before reply")
        }
    }
}

#[test]
#[ignore = "tier-2 (Mac ~34s / CI 2.5x ~85s); heavy profile で実行 (#97)"]
fn diag_ken13_deadline_must_stop_within_watchdog() {
    let path = Path::new("data/lp_problems/ken-13.QPS");
    let (status, _) = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "ken-13");
    assert!(
        matches!(
            status,
            SolveStatus::Timeout
                | SolveStatus::Optimal
                | SolveStatus::NumericalError
                | SolveStatus::Infeasible
        ),
        "ken-13: unexpected status {:?}",
        status
    );
}

#[test]
#[ignore = "tier-2 (Mac ~30s / CI 2.5x ~75s); heavy profile で実行 (#97)"]
fn diag_ken11_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/ken-11.QPS");
    let _ = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "ken-11");
}

#[test]
#[ignore = "tier-2 (Mac ~30s / CI 2.5x ~75s); heavy profile で実行 (#97)"]
fn diag_dfl001_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/dfl001.QPS");
    let _ = solve_with_watchdog(path, 30.0, Duration::from_secs(60), "dfl001");
}

/// pds-20 needs ~60s solve budget; measured at 61s, within the 180s nextest cap.
#[test]
#[ignore = "tier-2 (Mac ~62s / CI 2.5x ~155s); heavy profile で実行 (#97)"]
fn diag_pds20_deadline_regression_guard() {
    let path = Path::new("data/lp_problems/pds-20.QPS");
    let _ = solve_with_watchdog(path, 60.0, Duration::from_secs(100), "pds-20");
}
