//! greenbea IPM-time-box regression sentinel.
//!
//! greenbea (feasible, n=5405 m=2392) exceeds the LP-IPM size gate, so the QP
//! dispatch (`solve_qp_with`) runs the IPM first. The IPM does not converge on
//! greenbea (slow meaningful progress, never a hard stall) and, without a budget
//! cap, consumes the whole deadline — starving the simplex fallback, which solves
//! greenbea in ~75s. The fix (`lp_dispatch::ipm_box_deadline`) time-boxes the IPM
//! so simplex gets the remaining budget and reaches Optimal.
//!
//! Load-bearing: removing the time-box (IPM keeps the full deadline) lets the IPM
//! consume the whole budget and return a non-Optimal status (SuboptimalSolution /
//! Timeout) — this test then FAILS on `status == Optimal`. Verified by setting
//! IPM_BUDGET_FRACTION=1.0: greenbea reverts to SuboptimalSolution at 300s.
//!
//! Heavy (~220s: boxed IPM + simplex), so `#[ignore]`d to keep the default
//! `cargo nextest run` under budget. greenbea is inherently slow, so this cannot
//! fit the 3min per-test cap; instead a `.config/nextest.toml` override raises the
//! terminate-after for this test so `cargo nextest run --run-ignored` runs it to
//! completion (the default 180s cap would kill it mid-solve). Data file required;
//! absence panics (never a silent SKIP that would hide the bug).

use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use otspot::io::qps::parse_qps;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// HiGHS / LP DASA (1e-12) / this solver agree on the greenbea optimum.
const GREENBEA_OPT: f64 = -7.2555248130e+07;
const OBJ_REL_TOL: f64 = 1e-3;

#[test]
#[ignore = "heavy ~220s (boxed IPM + simplex fallback); run with `cargo nextest run --run-ignored` (nextest.toml override grants the budget)"]
fn diag_greenbea_ipm_timebox_reaches_optimal() {
    let path = Path::new("data/lp_problems/greenbea.QPS");
    assert!(path.exists(), "data required (no SKIP): {:?}", path);
    let qp = parse_qps(path).expect("parse greenbea.QPS");
    assert_eq!(qp.num_vars, 5405, "greenbea n");
    assert_eq!(qp.num_constraints, 2392, "greenbea m");

    let timeout_secs = 300.0;
    let watchdog = Duration::from_secs(400);

    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .name("greenbea-solver".into())
        // 8 MiB stack: faer supernodal recursion overflows the default on large bases.
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(timeout_secs);
            let t0 = Instant::now();
            let r = solve_qp_with(&qp, &opts);
            let _ = tx.send((r.status, r.objective, t0.elapsed()));
        })
        .expect("spawn solver thread");

    match rx.recv_timeout(watchdog) {
        Ok((status, obj, elapsed)) => {
            let _ = handle.join();
            eprintln!(
                "[greenbea-timebox] status={:?} obj={:.6e} wall={:.1}s (timeout={timeout_secs}s)",
                status, obj, elapsed.as_secs_f64()
            );
            assert_eq!(
                status,
                SolveStatus::Optimal,
                "greenbea must reach Optimal via simplex fallback after the IPM box; \
                 got {status:?} — IPM monopolized the budget (time-box missing?)"
            );
            let rel = (obj - GREENBEA_OPT).abs() / GREENBEA_OPT.abs();
            assert!(
                rel < OBJ_REL_TOL,
                "greenbea obj {obj:.6e} must match {GREENBEA_OPT:.6e} (rel {rel:.2e} < {OBJ_REL_TOL:.0e})"
            );
        }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "[greenbea-timebox] solve did not return within watchdog {}s — deadline path missing",
            watchdog.as_secs_f64()
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("[greenbea-timebox] solver thread panicked")
        }
    }
}
