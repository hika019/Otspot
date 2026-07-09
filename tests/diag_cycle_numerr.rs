//! Regression guard: `cycle.QPS` (Netlib LP canary) must converge to its known
//! optimum within `REL_TOL`. Known objective is taken from
//! `data/baseline_objectives/netlib_lp_canary.csv`.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
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

/// Max primal violation of `x`: constraint residual (by sense) and bound
/// violation, in original space. A genuinely feasible solution returns ≈ 0.
fn max_primal_violation(lp: &LpProblem, bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    let m = lp.num_constraints;
    let n = lp.num_vars.min(x.len());
    let mut ax = vec![0.0f64; m];
    for (j, &x_j) in x.iter().enumerate().take(n) {
        if let Ok((rows, vals)) = lp.a.get_column(j) {
            for k in 0..rows.len() {
                ax[rows[k]] += vals[k] * x_j;
            }
        }
    }
    let mut v_max = 0.0f64;
    for (i, &ax_i) in ax.iter().enumerate() {
        let v = match lp.constraint_types[i] {
            ConstraintType::Eq => (ax_i - lp.b[i]).abs(),
            ConstraintType::Le => (ax_i - lp.b[i]).max(0.0),
            ConstraintType::Ge => (lp.b[i] - ax_i).max(0.0),
            _ => 0.0,
        };
        v_max = v_max.max(v);
    }
    for (j, &(lo, hi)) in bounds.iter().enumerate().take(n) {
        v_max = v_max.max((lo - x[j]).max(0.0)).max((x[j] - hi).max(0.0));
    }
    v_max
}

/// cycle.QPS must reach the known optimum, not NumericalError.
#[test]
fn diag_cycle_must_reach_known_objective() {
    let path = Path::new("data/lp_problems/cycle.QPS");
    if !path.exists() {
        panic!(
            "data missing: {:?}. Symlink data/lp_problems/cycle.QPS into the worktree.",
            path
        );
    }
    let qp = parse_qps(path).expect("parse cycle.QPS");
    let lp = make_lp(&qp);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed_s = t0.elapsed().as_secs_f64();

    let tb = r.timing_breakdown.unwrap_or_default();
    eprintln!(
        "[cycle] elapsed={:.2}s status={:?} obj={:.10e} iters={} sol_len={}/n={}",
        elapsed_s,
        r.status,
        r.objective,
        r.iterations,
        r.solution.len(),
        lp.num_vars,
    );
    eprintln!(
        "[cycle] timing_us: presolve={} solve={} postsolve={} (total_ms={:.1})",
        tb.presolve_us,
        tb.solve_us,
        tb.postsolve_us,
        (tb.presolve_us + tb.solve_us + tb.postsolve_us) as f64 / 1000.0,
    );

    const KNOWN_OBJ: f64 = -5.226_393_024_892_44;
    const REL_TOL: f64 = 1.0e-4;

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "[cycle] expected Optimal, got {:?} (obj={:.6e})",
        r.status,
        r.objective,
    );

    let pviol = max_primal_violation(&lp, &qp.bounds, &r.solution);
    const FEAS_TOL: f64 = 1.0e-6;
    assert!(
        pviol <= FEAS_TOL,
        "[cycle] returned solution violates feasibility by {:.3e} (> {:.0e})",
        pviol,
        FEAS_TOL,
    );

    let rel_err = (r.objective - KNOWN_OBJ).abs() / KNOWN_OBJ.abs();
    assert!(
        rel_err < REL_TOL,
        "[cycle] obj={:.10e} differs from known {:.10e} by rel {:.3e} (>{:.0e})",
        r.objective,
        KNOWN_OBJ,
        rel_err,
        REL_TOL,
    );
}

/// Honest-behavior companion to the Optimal guard.
///
/// Does NOT require `Optimal`. Verifies the current, honest contract: cycle.QPS
/// returns a feasible-terminal status whose **returned solution is genuinely
/// feasible** (max primal violation ≤ tol, recomputed independently in original
/// space) and whose objective matches the known optimum. This is the coverage
/// that the feasibility-preserving ratio test + honest backstop must satisfy:
/// no false-feasible / false-Optimal claim. Both asserts stay true whether
/// `solve_with` returns `Optimal` or `SuboptimalSolution`, which is why this
/// companion survived the crossover-certification improvement that later made
/// `Optimal` the deterministic outcome (see `diag_cycle_must_reach_known_objective`).
///
/// tier-2 (~100s): cycle's postsolve dual crossover storm dominates wall time.
#[test]
fn diag_cycle_is_feasible_and_near_optimal() {
    let path = Path::new("data/lp_problems/cycle.QPS");
    if !path.exists() {
        panic!(
            "data missing: {:?}. Symlink data/lp_problems/cycle.QPS into the worktree.",
            path
        );
    }
    let qp = parse_qps(path).expect("parse cycle.QPS");
    let lp = make_lp(&qp);

    let mut opts = SolverOptions::default();
    // Generous timeout: cycle's postsolve dual crossover storm (~126s) plus
    // headroom so the solve deterministically completes (SuboptimalSolution)
    // rather than timing out mid-postsolve.
    opts.timeout_secs = Some(250.0);

    let r = solve_with(&lp, &opts);
    let pviol = max_primal_violation(&lp, &qp.bounds, &r.solution);
    eprintln!(
        "[cycle-honest] status={:?} obj={:.10e} max_primal_violation={:.3e}",
        r.status, r.objective, pviol,
    );

    const KNOWN_OBJ: f64 = -5.226_393_024_892_44;
    const REL_TOL: f64 = 1.0e-4;
    // Primal feasibility tolerance for the recomputed original-space residual.
    const FEAS_TOL: f64 = 1.0e-6;

    // Feasible-terminal status (not a false Infeasible/Unbounded/NumericalError).
    // Optimal is allowed (and, as of the crossover-certification fix, is what
    // `solve_with` now deterministically returns) so this stays passing
    // alongside `diag_cycle_must_reach_known_objective`'s strict Optimal assert.
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "[cycle-honest] expected Optimal/SuboptimalSolution, got {:?} (obj={:.6e})",
        r.status,
        r.objective,
    );

    // The returned solution must actually be feasible (no false-feasibility).
    assert!(
        pviol <= FEAS_TOL,
        "[cycle-honest] returned solution violates feasibility by {:.3e} (> {:.0e})",
        pviol,
        FEAS_TOL,
    );

    // ...and near the known optimum.
    let rel_err = (r.objective - KNOWN_OBJ).abs() / KNOWN_OBJ.abs();
    assert!(
        rel_err < REL_TOL,
        "[cycle-honest] obj={:.10e} differs from known {:.10e} by rel {:.3e} (>{:.0e})",
        r.objective,
        KNOWN_OBJ,
        rel_err,
        REL_TOL,
    );
}
