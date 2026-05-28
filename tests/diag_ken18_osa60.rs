//! TDD diagnostic tests for two LP bench regressions:
//!
//! - **osa-60** (Task #4): solver returns Optimal/Timeout with `obj=0` while the
//!   known optimum is `4.0440725e+06`. Verified root cause:
//!   `simplex::primal::pivot_out_degenerate_artificials` consumes the entire
//!   solve budget (O(n_artificial × n_total) FTRAN) before Phase 2 can iterate.
//!
//! - **ken-18** (Task #3): solver wall-time vastly exceeds its internal deadline
//!   (~3× overrun in single-job repro; >external `gtimeout` in concurrent bench
//!   → SIGKILL = "異常終了"). Verified root cause:
//!   `presolve::postsolve::build_and_solve_cleanup_lp` constructs a massive
//!   second LP (m≈96k, n≈322k) whose 5 s timeout is set via `timeout_secs`
//!   but never converted to a `deadline`, so the long Ruiz/standard-form
//!   construction never checks it.
//!
//! Both tests are `#[ignore]` because their failing wall-time is on the order
//! of 60-180s — too long for default `cargo test`.
//! Run with: `cargo nextest run --run-ignored only --test diag_ken18_osa60`.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::qp::solve_qp_with;
use otspot::{solve_with, QpProblem};
use std::path::Path;
use std::time::Instant;

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

/// Task #4 (osa-60): solver must report a meaningful objective, not 0.
///
/// Known optimum: 4.0440725e+06 (netlib_lp.csv).
///
/// Multi-hypothesis observation: the test records *all* of
/// `status`, `iterations`, `timing_breakdown` (presolve/solve/postsolve μs) and
/// `solution length / |x|_inf` so we can distinguish:
///   (H1) Phase 2 never started (solution=slack basis → obj=0).
///   (H2) Phase 2 ran but state was clobbered to 0 on return.
///   (H3) presolve early-exit returned Optimal trivially.
///
/// At HEAD this asserts:
///   - status is Optimal (after qps_benchmark Timeout→Optimal remap rule)
///   - reported objective relative error vs known < 5 %
///
/// Empirically the solver returns `obj=0`/`Timeout` at 60 s; with 90 s it
/// converges to 4.044e6. So the test FAILS at HEAD with 60 s budget and
/// would PASS once `pivot_out_degenerate_artificials` is sped up.
#[test]
#[ignore = "known failing: LP perf regression (#75 未対応); obj 誤差 5.2% > 5%; 要 data/lp_problems/osa-60.QPS"]
fn diag_osa60_must_reach_known_objective() {
    let path = Path::new("data/lp_problems/osa-60.QPS");
    assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行", path);
    let qp = parse_qps(path).expect("parse osa-60");
    let lp = make_lp(&qp);

    let mut opts = SolverOptions::default();
    // 60 s = bench default at eps=1e-5. Same budget the failing bench used.
    opts.timeout_secs = Some(60.0);

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed_s = t0.elapsed().as_secs_f64();

    // Observe every relevant field — diagnostic only, never assert.
    let abs_x_inf = r
        .solution
        .iter()
        .fold(0.0_f64, |acc: f64, &v: &f64| acc.max(v.abs()));
    let tb = r.timing_breakdown.unwrap_or_default();
    eprintln!(
        "[osa-60] elapsed={:.2}s status={:?} obj={:.6e} iters={} sol_len={}/n={} |x|_inf={:.3e}",
        elapsed_s,
        r.status,
        r.objective,
        r.iterations,
        r.solution.len(),
        lp.num_vars,
        abs_x_inf,
    );
    eprintln!(
        "[osa-60] timing_us: presolve={} solve={} postsolve={} (total_ms={:.1})",
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
        (tb.presolve_us + tb.solve_us + tb.postsolve_us) as f64 / 1000.0,
    );

    // Bench reproduces qps_benchmark.rs:857 — Timeout with full-length solution
    // gets remapped to Optimal. Mimic that so the assertion targets the same
    // post-remap status the bench sees (otherwise we'd report Timeout here and
    // pass while bench fails on OBJ_MISMATCH).
    let original_status = format!("{:?}", r.status);
    let post_remap_status = if matches!(r.status, SolveStatus::Timeout)
        && r.solution.len() == lp.num_vars
    {
        SolveStatus::Optimal
    } else {
        r.status.clone()
    };

    const KNOWN_OBJ: f64 = 4.0440725e6;
    let rel_err = (r.objective - KNOWN_OBJ).abs() / KNOWN_OBJ.abs();

    assert_eq!(
        post_remap_status,
        SolveStatus::Optimal,
        "[osa-60] expected Optimal (or Timeout-with-solution remap), got {}",
        original_status,
    );
    assert!(
        rel_err < 0.05,
        "[osa-60] obj={:.6e} differs from known {:.6e} by {:.1}% (>5%); \
         this is the OBJ_MISMATCH bench reports. timing_us: presolve={} solve={} postsolve={}",
        r.objective,
        KNOWN_OBJ,
        rel_err * 100.0,
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
    );
}

/// Task #3 (ken-18): solver wall-time must respect the internal deadline.
///
/// We do not require ken-18 to *solve* (it's a very large LP and is the next
/// task after the abnormal-exit bug is fixed). The defect under test is the
/// **deadline contract**: a 30 s internal timeout must not produce 100s+ of
/// wall time spent in postsolve `build_and_solve_cleanup_lp`, which is what
/// drives the bench's gtimeout SIGKILL ("異常終了") under concurrent jobs.
///
/// Observation budget: deadline + 30 s slack (matches the bench's 300 s slack
/// design at a smaller scale).
///
/// Multi-hypothesis recording (same as osa-60): status / iters / timing_us
/// are all printed so we can tell whether overrun is in presolve, simplex,
/// or postsolve.
///
/// At HEAD this FAILS: empirically wall ≈ 365 s for a 120 s internal
/// budget. Scaled to 30 s internal it lands well past the 60 s slack ceiling.
#[test]
#[ignore = "known failing: postsolve cleanup_lp deadline violation (wall 72s > 60s budget); 要 data/lp_problems/ken-18.QPS"]
fn diag_ken18_must_respect_internal_deadline() {
    let path = Path::new("data/lp_problems/ken-18.QPS");
    assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行", path);
    let qp = parse_qps(path).expect("parse ken-18");
    let lp = make_lp(&qp);

    const INTERNAL_TIMEOUT_SECS: f64 = 30.0;
    const WALL_SLACK_SECS: f64 = 30.0;

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(INTERNAL_TIMEOUT_SECS);

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed_s = t0.elapsed().as_secs_f64();

    let tb = r.timing_breakdown.unwrap_or_default();
    eprintln!(
        "[ken-18] elapsed={:.2}s (budget={:.1}s+slack={:.1}s) status={:?} obj={:.3e} iters={} sol_len={}/n={}",
        elapsed_s,
        INTERNAL_TIMEOUT_SECS,
        WALL_SLACK_SECS,
        r.status,
        r.objective,
        r.iterations,
        r.solution.len(),
        lp.num_vars,
    );
    eprintln!(
        "[ken-18] timing_us: presolve={} solve={} postsolve={} (total_ms={:.1})",
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
        (tb.presolve_us + tb.solve_us + tb.postsolve_us) as f64 / 1000.0,
    );

    // Defect-of-interest: postsolve cleanup_lp runs an uncapped second simplex.
    // If timing_breakdown captured it, postsolve_us should be the lion's share.
    assert!(
        elapsed_s <= INTERNAL_TIMEOUT_SECS + WALL_SLACK_SECS,
        "[ken-18] wall time {:.2}s exceeds internal deadline {:.1}s + {:.1}s slack. \
         This is the gtimeout-SIGKILL ('異常終了') trigger under concurrent bench. \
         timing_us: presolve={} solve={} postsolve={}",
        elapsed_s,
        INTERNAL_TIMEOUT_SECS,
        WALL_SLACK_SECS,
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
    );

    // Status must be a known enum value — i.e. no panic, no abort.
    // (A panic would have aborted the test before this point; we still
    // gate against silently corrupted statuses for completeness.)
    match r.status {
        SolveStatus::Optimal
        | SolveStatus::Timeout
        | SolveStatus::Infeasible
        | SolveStatus::Unbounded
        | SolveStatus::NumericalError
        | SolveStatus::SuboptimalSolution
        | SolveStatus::LocallyOptimal
        | SolveStatus::MaxIterations
        | SolveStatus::NonConvex(_) => {}
        // SolveStatus is `#[non_exhaustive]`; new variants must trip this test.
        #[allow(unreachable_patterns)]
        _ => panic!("[ken-18] unexpected SolveStatus variant: {:?}", r.status),
    }
}

/// Deadline contract: two-case deterministic sentinel using `SolveStats::deadline_triggered`.
///
/// Case A (generous budget): bore3d solves in < 1 s; a 30 s deadline is never hit.
///   deadline_triggered == false, status == Optimal.
///
/// Case B (forced deadline): a 0.001 s (1 ms) budget forces Timeout before bore3d
///   completes; deadline_triggered == true, status == Timeout.
///
/// Deleting the `deadline_triggered` assignment in lp.rs/dispatch_solve_qp causes
/// Case B to assert false (no-op proof).  No wall-clock measurement.
#[test]
fn diag_deadline_small_lp() {
    let path = Path::new("tests/lp_problems/bore3d.QPS");
    assert!(path.exists(), "tests/lp_problems/bore3d.QPS missing from repo");

    let qp = parse_qps(path).expect("parse bore3d");

    // Case A: generous budget — bore3d finishes before the deadline.
    {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(30.0);
        let r = solve_qp_with(&qp, &opts);
        eprintln!("[bore3d/A] status={:?} obj={:.6e} deadline_triggered={}",
            r.status, r.objective, r.stats.deadline_triggered);
        assert_eq!(r.status, SolveStatus::Optimal,
            "[bore3d/A] expected Optimal with generous budget, got {:?}", r.status);
        assert!(!r.stats.deadline_triggered,
            "[bore3d/A] deadline should not have triggered with 30 s budget");
    }

    // Case B: 1 ms budget — forces Timeout, proving deadline enforcement is wired.
    {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(0.001);
        let r = solve_qp_with(&qp, &opts);
        eprintln!("[bore3d/B] status={:?} obj={:.6e} deadline_triggered={}",
            r.status, r.objective, r.stats.deadline_triggered);
        assert_eq!(r.status, SolveStatus::Timeout,
            "[bore3d/B] expected Timeout with 1 ms budget, got {:?}", r.status);
        assert!(r.stats.deadline_triggered,
            "[bore3d/B] deadline_triggered must be true when status == Timeout");
    }
}
