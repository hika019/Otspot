//! Postsolve dual recovery via primal crossover — pilot-ja Cat A regression guard.
//!
//! pilot-ja solves to the primal optimum under presolve, but its presolve
//! structure (a deleted Eq row serving both a LinearSubstitution pivot and
//! multiple forcing roles) defeats every *local* per-transform dual recovery:
//! greedy `recover_removed_row_dual` lands at dual-infeasibility ≈ 20.8, the
//! forcing pass at ≈ 0.33 — both above the LP certificate tolerance, so
//! `guard_lp_optimal` demoted Optimal → SuboptimalSolution.
//!
//! `crossover_dual_from_primal` reconstructs an optimal basis *at* the primal
//! optimum and reads `y = Bᵀ⁻¹ c_B`, a globally dual-feasible dual. This pins
//! the fix: pilot-ja must report Optimal with a near-zero postsolve dual
//! infeasibility. Reverting the crossover (or its degenerate-pivot Phase II)
//! flips this back to SuboptimalSolution / large dfeas.

use otspot::io::qps::parse_qps;
use otspot::lp::solve_lp_with;
use otspot::options::SolverOptions;
use otspot::presolve;
use otspot::problem::{LpProblem, SolveStatus};
use std::path::Path;

fn load_lp(path: &str) -> LpProblem {
    let qp = parse_qps(Path::new(path)).expect("parse_qps");
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .expect("LpProblem")
}

/// pilot-ja under presolve must certify Optimal with a dual-feasible dual.
///
/// Runs in the default profile (~12s on Mac); not gated behind heavy.
#[test]
fn pilot_ja_postsolve_dual_is_feasible() {
    let prob = load_lp("data/lp_problems/pilot-ja.QPS");
    eprintln!(
        "pilot-ja[parse]: vars={} rows={} nnz={}",
        prob.num_vars,
        prob.num_constraints,
        prob.a.nnz()
    );

    let presolved = presolve::run_presolve(&prob, None).expect("pilot-ja presolve");
    assert!(presolved.was_reduced, "pilot-ja must exercise LP postsolve");
    eprintln!(
        "pilot-ja[presolve]: reduced_vars={} reduced_rows={}",
        presolved.reduced_problem.num_vars, presolved.reduced_problem.num_constraints
    );

    let mut raw_opts = SolverOptions::default();
    raw_opts.presolve = false;
    raw_opts.timeout_secs = Some(400.0);
    let raw = solve_lp_with(&presolved.reduced_problem, &raw_opts);
    eprintln!(
        "pilot-ja[reduced-simplex]: status={:?} obj={:.6e}",
        raw.status, raw.objective
    );
    assert_eq!(
        raw.status,
        SolveStatus::Optimal,
        "pilot-ja reduced LP must solve before postsolve can be blamed"
    );

    let lifted = presolve::postsolve::run_postsolve(&raw, &presolved, &prob, None, false);
    eprintln!(
        "pilot-ja[postsolve]: status={:?} obj={:.6e} dfeas={:?}",
        lifted.status, lifted.objective, lifted.postsolve_dfeas
    );
    assert_eq!(lifted.status, SolveStatus::Optimal);
    assert!(
        lifted.postsolve_dfeas.unwrap_or(f64::INFINITY) < 1e-6,
        "postsolve dual recovery itself must certify dfeas before guard"
    );

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(400.0);
    let r = solve_lp_with(&prob, &opts);
    eprintln!(
        "pilot-ja[presolve=on]: status={:?} obj={:.6e} postsolve_dfeas={:?} timing={:?}",
        r.status, r.objective, r.postsolve_dfeas, r.timing_breakdown
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "pilot-ja must certify Optimal (crossover dual recovery); got {:?} \
         with postsolve_dfeas={:?}. Without the crossover the local recovery \
         stalls at dfeas≈0.33 and guard_lp_optimal demotes to SuboptimalSolution.",
        r.status,
        r.postsolve_dfeas
    );
    let dfeas = r
        .postsolve_dfeas
        .expect("postsolve_dfeas must be populated for an Optimal presolved LP");
    assert!(
        dfeas < 1e-6,
        "pilot-ja postsolve dual infeasibility {dfeas:.3e} must be below the LP \
         certificate tolerance (crossover gives ~1e-12; greedy 20.8 / forcing 0.33 fail)"
    );
}
