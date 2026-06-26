//! RINS (Relaxation Induced Neighborhood Search) heuristic for MILP.
//!
//! Reference: Danna, Rothberg & Le Pape (2005).

use crate::mip::{MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use std::time::Instant;

/// Run RINS every this many B&B nodes.
pub(crate) const RINS_INTERVAL: usize = 100;

/// Node limit for the RINS sub-MIP.
const RINS_NODE_LIMIT: usize = 1_000;

/// Fraction of remaining wall-clock budget given to the sub-MIP.
const RINS_TIME_FRACTION: f64 = 0.10;

/// Absolute upper bound on sub-MIP wall time (seconds).
const RINS_MAX_TIME_SECS: f64 = 10.0;

/// Minimum remaining budget below which RINS is skipped.
const RINS_MIN_REMAINING_SECS: f64 = 1.0;

/// Run the RINS heuristic.
///
/// Fixes integer variables where `round(x_lp[j]) == round(x_inc[j])`, then
/// solves the reduced sub-MIP with a short timeout and node limit. Returns an
/// improved `SolverResult` or `None` when no improvement is found.
///
/// `parent_opts` is cloned and its timeout/deadline overridden so that
/// tolerance, cancellation flag, and other settings are inherited by the sub-MIP.
pub(crate) fn run_rins(
    problem: &MilpProblem,
    x_lp: &[f64],
    x_inc: &[f64],
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_opts: &SolverOptions,
) -> Option<SolverResult> {
    let remaining_secs = remaining_budget(deadline);
    if remaining_secs < RINS_MIN_REMAINING_SECS {
        return None;
    }

    let mut sub_bounds = problem.lp.bounds.clone();
    let mut n_fixed = 0usize;
    for &j in &problem.integer_vars {
        if j >= x_lp.len() || j >= x_inc.len() {
            continue;
        }
        let lp_rounded = x_lp[j].round();
        let inc_rounded = x_inc[j].round();
        if (lp_rounded - inc_rounded).abs() < 0.5 {
            sub_bounds[j] = (inc_rounded, inc_rounded);
            n_fixed += 1;
        }
    }

    if n_fixed == 0 {
        return None;
    }

    let sub_timeout = (remaining_secs * RINS_TIME_FRACTION).min(RINS_MAX_TIME_SECS);

    let mut sub_lp = problem.lp.clone();
    sub_lp.bounds = sub_bounds;
    // integer_vars were already validated on the original; num_vars unchanged.
    let sub_problem = MilpProblem::new(sub_lp, problem.integer_vars.clone()).ok()?;

    let sub_cfg = rins_sub_mip_config(cfg);

    let mut sub_opts = parent_opts.clone();
    sub_opts.timeout_secs = Some(sub_timeout);
    sub_opts.deadline = None;
    sub_opts.warm_start = None;
    sub_opts.warm_start_qp = None;
    sub_opts.warm_start_lp = None;
    sub_opts.known_optimal_obj = None;
    sub_opts.presolve = true;
    sub_opts.use_lp_crash_basis = true;
    sub_opts.recover_warm_start_basis = false;
    sub_opts.threads = 1;

    let result = crate::mip::solve_milp(&sub_problem, &sub_opts, &sub_cfg);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution
    ) && !result.solution.is_empty()
    {
        Some(result)
    } else {
        None
    }
}

fn remaining_budget(deadline: &Option<Instant>) -> f64 {
    match deadline {
        None => f64::INFINITY,
        Some(d) => {
            let now = Instant::now();
            if now >= *d { 0.0 } else { (*d - now).as_secs_f64() }
        }
    }
}

fn rins_sub_mip_config(cfg: &MipConfig) -> MipConfig {
    let mut sub_cfg = cfg.clone();
    sub_cfg.max_nodes = RINS_NODE_LIMIT;
    sub_cfg.rins_enabled = false;
    sub_cfg.rens_enabled = false;
    sub_cfg.local_branching_enabled = false;
    sub_cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::sparse::CscMatrix;

    fn two_var_milp(c: [f64; 2], b: f64) -> MilpProblem {
        let n = 2;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let lp = LpProblem::new_general(
            c.to_vec(),
            a,
            vec![b],
            vec![ConstraintType::Le],
            vec![(0.0, 3.0); n],
            None,
        )
        .unwrap();
        MilpProblem::new(lp, vec![0, 1]).unwrap()
    }

    /// RINS improves a sub-optimal incumbent by fixing an agreeing variable.
    ///
    /// Problem: min -x0 - x1 s.t. x0+x1 <= 3, x0,x1 in {0..3}.
    /// x_lp=(1.4,1.6) rounds to (1,2). x_inc=(1,1).
    /// x0: both round to 1 → fixed to 1. x1 free.
    /// Sub-MIP: min -x1 s.t. 1+x1 <= 3 → x1=2, obj=-3.
    ///
    /// Sentinel: removing `if n_fixed == 0 { return None; }` causes RINS to
    /// call solve_milp on the full problem when no variables agree → wasteful.
    #[test]
    fn rins_improves_suboptimal_incumbent() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        let result = run_rins(&problem, &x_lp, &x_inc, &cfg, &None, &SolverOptions::default())
            .expect("RINS must return Some when at least one variable is fixed");
        assert!(
            result.objective < -1.9,
            "RINS should improve below -2; got {}",
            result.objective
        );
    }

    /// RINS returns None when no integer variable agrees.
    ///
    /// Sentinel: removing `if n_fixed == 0 { return None; }` returns Some → FAILS.
    #[test]
    fn rins_skips_when_no_agreement() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        // x_lp rounds to (0,3); x_inc=(1,2): both disagree
        let x_lp = vec![0.4, 2.6];
        let x_inc = vec![1.0, 2.0];
        assert!(
            run_rins(&problem, &x_lp, &x_inc, &cfg, &None, &SolverOptions::default()).is_none(),
            "RINS must return None when no variable is fixed"
        );
    }

    /// RINS skips when the deadline is already past.
    ///
    /// Sentinel: removing the budget check causes a sub-MIP call with 0 s timeout.
    #[test]
    fn rins_skips_on_expired_deadline() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];
        let past = Instant::now() - std::time::Duration::from_secs(1);
        assert!(
            run_rins(&problem, &x_lp, &x_inc, &cfg, &Some(past), &SolverOptions::default())
                .is_none(),
            "RINS must not run when deadline is expired"
        );
    }

    /// rins_enabled=false still produces an optimal solution.
    ///
    /// Sentinel: if rins_enabled=false broke the solver, status != Optimal.
    #[test]
    fn rins_disabled_cfg_does_not_break_solve() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig { rins_enabled: false, ..MipConfig::default() };
        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let result = crate::mip::solve_milp(&problem, &opts, &cfg);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(result.objective < -2.9, "obj={}", result.objective);
    }

    #[test]
    fn rins_sub_mip_disables_recursive_primal_heuristics() {
        let cfg = MipConfig {
            max_nodes: 99_999,
            rins_enabled: true,
            rens_enabled: true,
            local_branching_enabled: true,
            ..MipConfig::default()
        };

        let sub_cfg = rins_sub_mip_config(&cfg);
        assert_eq!(sub_cfg.max_nodes, RINS_NODE_LIMIT);
        assert!(!sub_cfg.rins_enabled);
        assert!(!sub_cfg.rens_enabled);
        assert!(!sub_cfg.local_branching_enabled);
    }

    #[test]
    fn remaining_budget_past_deadline_is_zero() {
        let past = Instant::now() - std::time::Duration::from_millis(100);
        assert_eq!(remaining_budget(&Some(past)), 0.0);
    }

    #[test]
    fn remaining_budget_no_deadline_is_infinity() {
        assert!(remaining_budget(&None).is_infinite());
    }
}
