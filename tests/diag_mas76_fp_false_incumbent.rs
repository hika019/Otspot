//! mas76 regression guard for FP false incumbents on continuous objective columns.

use std::path::Path;

use otspot::{
    io::mps::parse_milp_file,
    options::{MipConfig, SolverOptions, Tolerance},
    problem::{ConstraintType, SolveStatus},
    solve_milp_with_stats, MilpProblem,
};

const MAS76: &str = "data/miplib_small/mas76.mps";
const MAS76_OBJECTIVE_SENTINEL_UB: f64 = 1.0e9;
const TEST_TIMEOUT_SECS: f64 = 10.0;

fn max_constraint_violation(prob: &MilpProblem, x: &[f64]) -> f64 {
    let ax = prob.lp.a.mat_vec_mul(x).expect("mat_vec_mul");
    ax.iter()
        .zip(&prob.lp.constraint_types)
        .zip(&prob.lp.b)
        .map(|((&lhs, ct), &rhs)| match ct {
            ConstraintType::Le => (lhs - rhs).max(0.0),
            ConstraintType::Ge => (rhs - lhs).max(0.0),
            ConstraintType::Eq => (lhs - rhs).abs(),
            _ => unreachable!("unexpected constraint type in mas76"),
        })
        .fold(0.0, f64::max)
}

#[test]
fn mas76_cuts_does_not_accept_fp_upper_bound_objective() {
    let milp = parse_milp_file(Path::new(MAS76)).expect("parse mas76");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TEST_TIMEOUT_SECS);
    opts.tolerance = Some(Tolerance::Custom(1.0e-6));

    let mut cfg = MipConfig::default();
    cfg.cuts = true;
    cfg.max_nodes = 0;
    let (res, stats) = solve_milp_with_stats(&milp, &opts, &cfg);

    if stats.fp_incumbent_found {
        assert!(
            res.objective.is_finite() && res.objective.abs() < MAS76_OBJECTIVE_SENTINEL_UB,
            "FP incumbent must not preserve mas76 x151 upper-bound objective; obj={}",
            res.objective
        );
        assert!(
            max_constraint_violation(&milp, &res.solution) <= 1.0e-6,
            "accepted incumbent must satisfy original constraints"
        );
    } else {
        assert!(
            matches!(res.status, SolveStatus::MaxIterations | SolveStatus::Timeout),
            "without a repaired FP incumbent, max_nodes=0 should report no incumbent honestly: {:?}",
            res.status
        );
        assert!(
            !res.objective.is_finite(),
            "no incumbent path must not report the mas76 upper-bound objective; obj={}",
            res.objective
        );
    }
}
