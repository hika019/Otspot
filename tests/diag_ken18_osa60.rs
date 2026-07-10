//! TDD diagnostic tests for two LP bench regressions:
//!
//! - **osa-60**: with presolve (the default) the solver timed out at 60 s while
//!   presolve=false solved in ~23 s. Verified root cause (measured, not the
//!   earlier `pivot_out_degenerate_artificials` guess, which is disproven —
//!   Phase I completes cleanly, degenerate-pivot fraction ≈ 0.003): presolve's
//!   activity bound-tightening gives every one of the 232 965 reduced variables
//!   a finite upper bound, and `build_standard_form` materializes each finite
//!   upper bound as an explicit constraint row, exploding the standard-form row
//!   count 10 280 → 243 174 (~24×). The primal simplex's per-iteration linear
//!   algebra scales with that row count, so each pivot ran ~2.6× slower on the
//!   same monotonically-converging ~5 400-iteration trajectory and the deadline
//!   fired ~0.2 % short of the optimum. Fixed: presolve now drops the redundant
//!   implied bounds it added on originally-unbounded variables before emitting
//!   the reduced problem, so the standard form no longer grows a row per column —
//!   see `presolve::transforms::bounds::revert_redundant_added_bounds`.
//!
//! - **ken-18**: solver wall-time vastly exceeded its internal deadline
//!   (~3× overrun in single-job repro; >external `gtimeout` in concurrent bench
//!   → SIGKILL = "異常終了"). Verified root cause:
//!   `presolve::postsolve::build_and_solve_cleanup_lp` constructed a massive
//!   second LP (m≈96k, n≈322k) whose 5 s timeout was set via `timeout_secs`
//!   but never converted to a `deadline`, so the long Ruiz/standard-form
//!   construction never checked it. Fixed — see
//!   `diag_ken18_must_respect_internal_deadline`'s ignore reason for the
//!   current measured wall time.
//!
//! `diag_osa60_must_reach_known_objective` now runs in the default profile
//! (~34 s solve, its own thread budget) as the osa-60 perf sentinel. The ken-18
//! deadline test stays `#[ignore]` (its 30 s internal budget exceeds the default
//! per-test wall); run it with
//! `cargo nextest run --run-ignored all --test diag_ken18_osa60`.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
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

/// osa-60: solver must report a meaningful objective, not 0.
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
/// This asserts:
///   - status is Optimal (after qps_benchmark Timeout→Optimal remap rule)
///   - reported objective relative error vs known < 5 %
///
/// Measured after the presolve fix: the default (presolve) path certifies
/// Optimal in ~21 s (well inside the 60 s budget), reported obj 4.0440725e6
/// vs known 4.0440725e6 (rel ≈ 8e-10). Before the fix it timed out at 60 s
/// (see the module doc for the root cause). Reverting the redundant-bound
/// reversion (`presolve::transforms::bounds::revert_redundant_added_bounds`)
/// restores the timeout, so this test is the perf sentinel.
#[test]
fn diag_osa60_must_reach_known_objective() {
    let path = Path::new("data/lp_problems/osa-60.QPS");
    assert!(
        path.exists(),
        "{:?} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path
    );
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
    let post_remap_status =
        if matches!(r.status, SolveStatus::Timeout) && r.solution.len() == lp.num_vars {
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

/// Max relative primal infeasibility of `x` against `lp`:
/// `(max_bound_violation, max_constraint_violation)`, each scaled so the magnitude
/// of the row / bound cannot mask a small absolute violation.
fn osa60_max_primal_infeasibility(lp: &LpProblem, x: &[f64]) -> (f64, f64) {
    let mut max_bound = 0.0_f64;
    for (&xj, &(lo, hi)) in x.iter().zip(lp.bounds.iter()) {
        let span = hi.abs().max(lo.abs()).max(1.0);
        let below = if lo.is_finite() {
            (lo - xj).max(0.0) / span
        } else {
            0.0
        };
        let above = if hi.is_finite() {
            (xj - hi).max(0.0) / span
        } else {
            0.0
        };
        max_bound = max_bound.max(below).max(above);
    }
    let ax = lp.a.mat_vec_mul(x).expect("Ax");
    let mut max_con = 0.0_f64;
    for (&row, (&rhs, &ct)) in ax.iter().zip(lp.b.iter().zip(lp.constraint_types.iter())) {
        let scale = row.abs().max(rhs.abs()).max(1.0);
        let viol = match ct {
            ConstraintType::Le => (row - rhs).max(0.0),
            ConstraintType::Ge => (rhs - row).max(0.0),
            ConstraintType::Eq => (row - rhs).abs(),
            // ConstraintType is `#[non_exhaustive]`; a new sense must trip this test.
            #[allow(unreachable_patterns)]
            _ => panic!("[osa-60] unhandled ConstraintType variant: {:?}", ct),
        } / scale;
        max_con = max_con.max(viol);
    }
    (max_bound, max_con)
}

/// Honest-behavior companion for osa-60: fills the verification blank left by
/// excluding `diag_osa60_must_reach_known_objective` from the heavy gate.
///
/// Measured facts (HEAD, 90 s budget): osa-60 reaches the **exact** optimum —
/// reported obj 4.0440725e6 vs known 4.0440725e6 (rel ≈ 8e-10), primal-feasible
/// to ≈1e-11. The "obj err 5.2%" in the excluded test's ignore message is stale
/// and false; the solver does not mis-value osa-60.
///
/// Solve path: presolve is **disabled**. With presolve enabled (the default), the
/// 60 s deadline lands at a knife-edge: it either returns `SuboptimalSolution`
/// with a full-length solution, or hits the `simplex::entry` timeout early-return
/// (presolve reduced the problem, deadline fires mid-reduced-solve) and returns a
/// solution in *reduced* variable space (length `num_vars - eliminated`, e.g.
/// 232965 ≠ 232966) that never went through postsolve. That malformed-length
/// return is a separate defect; gating on it here would make this sentinel flaky.
/// presolve=false always returns a full-length original-space solution, so the
/// honest contract below is well-defined every run.
///
/// Honest contract (the 5% optimality target is NOT required):
///   - `Optimal`  ⇒ reported obj must match the known optimum (rel <
///     `OPTIMAL_OBJ_TOL`) and the point must be primal-feasible. A miss here is a
///     **false-Optimal correctness bug**.
///   - `SuboptimalSolution` (found a feasible point, precision unmet) ⇒ the point
///     must be primal-feasible AND obj must equal recomputed `c^T x + offset`.
///   - `Timeout` / `MaxIterations` (did not finish) ⇒ only obj self-consistency
///     is required (an unfinished incumbent may be infeasible — that is honest);
///     feasibility is not demanded.
///   - any other status (`Infeasible` / `Unbounded` / …) ⇒ **fail**: osa-60 is a
///     feasible, bounded LP, so those are false verdicts.
///
/// Sentinels (no-op proofs): dropping the `Optimal` obj-match lets a false-Optimal
/// pass; dropping feasibility lets an infeasible "solution" pass; dropping
/// self-consistency lets a mis-reported objective pass.
#[test]
fn diag_osa60_is_feasible_and_honest() {
    let path = Path::new("data/lp_problems/osa-60.QPS");
    assert!(
        path.exists(),
        "{:?} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path
    );
    let qp = parse_qps(path).expect("parse osa-60");
    let lp = make_lp(&qp);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(90.0);
    opts.presolve = false;

    let r = solve_with(&lp, &opts);

    const KNOWN_OBJ: f64 = 4.0440725e6;
    const OPTIMAL_OBJ_TOL: f64 = 1e-5;
    const OBJ_SELF_TOL: f64 = 1e-9;
    const FEAS_TOL: f64 = 1e-6;

    let recomputed_obj =
        lp.c.iter()
            .zip(r.solution.iter())
            .map(|(&c, &x)| c * x)
            .sum::<f64>()
            + lp.obj_offset;
    let obj_vs_known = (r.objective - KNOWN_OBJ).abs() / KNOWN_OBJ.abs();
    let denom = r.objective.abs().max(recomputed_obj.abs()).max(1.0);
    let obj_self_rel = (r.objective - recomputed_obj).abs() / denom;
    let full_length = r.solution.len() == lp.num_vars;
    let (max_bound_viol, max_con_viol) = if full_length {
        osa60_max_primal_infeasibility(&lp, &r.solution)
    } else {
        (f64::NAN, f64::NAN)
    };
    eprintln!(
        "[osa-60/honest] status={:?} reported_obj={:.8e} recomputed_obj={:.8e} \
         obj_vs_known_rel={:.3e} obj_self_rel={:.3e} max_bound_viol={:.3e} \
         max_con_viol={:.3e} sol_len={}/n={}",
        r.status,
        r.objective,
        recomputed_obj,
        obj_vs_known,
        obj_self_rel,
        max_bound_viol,
        max_con_viol,
        r.solution.len(),
        lp.num_vars,
    );

    let assert_self_consistent = || {
        assert!(
            obj_self_rel < OBJ_SELF_TOL,
            "[osa-60] DISHONEST OBJECTIVE: reported {:.8e} ≠ recomputed c^T x + offset {:.8e} \
             (rel {:.3e}); the solver misreports the value of the point it returns.",
            r.objective,
            recomputed_obj,
            obj_self_rel,
        );
    };
    let assert_feasible = || {
        assert!(
            full_length,
            "[osa-60] solution length {} != num_vars {} — feasibility cannot be checked",
            r.solution.len(),
            lp.num_vars,
        );
        assert!(
            max_bound_viol < FEAS_TOL,
            "[osa-60] BOUND INFEASIBLE: max relative bound violation {:.3e} exceeds tol {:.0e}",
            max_bound_viol,
            FEAS_TOL,
        );
        assert!(
            max_con_viol < FEAS_TOL,
            "[osa-60] CONSTRAINT INFEASIBLE: max relative constraint violation {:.3e} exceeds tol {:.0e}",
            max_con_viol,
            FEAS_TOL,
        );
    };

    match r.status {
        SolveStatus::Optimal => {
            assert!(
                obj_vs_known < OPTIMAL_OBJ_TOL,
                "[osa-60] FALSE-OPTIMAL: status==Optimal but reported obj {:.8e} differs from \
                 known optimum {:.8e} by rel {:.3e} (>{:.0e}); claiming Optimal with a wrong \
                 objective is a correctness bug.",
                r.objective,
                KNOWN_OBJ,
                obj_vs_known,
                OPTIMAL_OBJ_TOL,
            );
            assert_self_consistent();
            assert_feasible();
        }
        SolveStatus::SuboptimalSolution => {
            // Found a feasible incumbent but precision unmet: must be feasible + honest.
            assert_self_consistent();
            assert_feasible();
        }
        SolveStatus::Timeout | SolveStatus::MaxIterations => {
            // Did not finish: incumbent may be infeasible (honest). Only require
            // that the solver not lie about the value of whatever it returned.
            if full_length {
                assert_self_consistent();
            }
        }
        ref other => panic!(
            "[osa-60] unexpected status {:?}: osa-60 is a feasible, bounded LP with known \
             optimum {:.8e}; Infeasible/Unbounded/NumericalError are false verdicts.",
            other, KNOWN_OBJ,
        ),
    }
}

/// ken-18: solver wall-time must respect the internal deadline.
///
/// We do not require ken-18 to *solve* (it's a very large LP). The defect
/// under test is the **deadline contract**: a 30 s internal timeout must not
/// produce 100s+ of wall time spent in postsolve `build_and_solve_cleanup_lp`,
/// which is what drove the bench's gtimeout SIGKILL ("異常終了") under
/// concurrent jobs.
///
/// Observation budget: deadline + 30 s slack (matches the bench's 300 s slack
/// design at a smaller scale).
///
/// Multi-hypothesis recording (same as osa-60): status / iters / timing_us
/// are all printed so we can tell whether overrun is in presolve, simplex,
/// or postsolve.
///
/// Originally this FAILED: empirically wall ≈ 365 s for a 120 s internal
/// budget. Fixed (the 5 s cleanup-LP timeout is now converted to a
/// `deadline`); re-measured 2026-07-10 at wall=30.626s for the 30s+30s
/// contract below — see the `#[ignore]` reason for the 2026-06-14 figure.
#[test]
#[ignore = "heavy/timing: measured 30.02s Timeout within 30s+30s wall contract (2026-06-14), \
            but >30s default budget; run explicitly"]
fn diag_ken18_must_respect_internal_deadline() {
    let path = Path::new("data/lp_problems/ken-18.QPS");
    assert!(
        path.exists(),
        "{:?} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path
    );
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
    let path = Path::new("data/lp_problems/bore3d.QPS");
    assert!(
        path.exists(),
        "data/lp_problems/bore3d.QPS missing (run scripts/download_all_bench_data.sh)"
    );

    let qp = parse_qps(path).expect("parse bore3d");

    // Case A: generous budget — bore3d finishes before the deadline.
    {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(30.0);
        let r = solve_qp_with(&qp, &opts);
        eprintln!(
            "[bore3d/A] status={:?} obj={:.6e} deadline_triggered={}",
            r.status, r.objective, r.stats.deadline_triggered
        );
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "[bore3d/A] expected Optimal with generous budget, got {:?}",
            r.status
        );
        assert!(
            !r.stats.deadline_triggered,
            "[bore3d/A] deadline should not have triggered with 30 s budget"
        );
    }

    // Case B: 1 ms budget — forces Timeout, proving deadline enforcement is wired.
    {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(0.001);
        let r = solve_qp_with(&qp, &opts);
        eprintln!(
            "[bore3d/B] status={:?} obj={:.6e} deadline_triggered={}",
            r.status, r.objective, r.stats.deadline_triggered
        );
        assert_eq!(
            r.status,
            SolveStatus::Timeout,
            "[bore3d/B] expected Timeout with 1 ms budget, got {:?}",
            r.status
        );
        assert!(
            r.stats.deadline_triggered,
            "[bore3d/B] deadline_triggered must be true when status == Timeout"
        );
    }
}
