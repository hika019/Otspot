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
/// tier-2 (Mac ~12s); heavy profile (#97).
#[test]
#[ignore = "tier-2 (Mac ~12s); heavy profile"]
fn pilot_ja_postsolve_dual_is_feasible() {
    let prob = load_lp("data/lp_problems/pilot-ja.QPS");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(400.0);
    let r = solve_lp_with(&prob, &opts);

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
