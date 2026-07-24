//! RENS (Relaxation Enforced Neighborhood Search) heuristic for MILP.
//!
//! Reference: Berthold (2007), "RENS — the optimal rounding".
//!
//! From a node LP relaxation, integer variables that are already integral are
//! fixed to their value and the fractional ones are restricted to their two
//! surrounding integers `{floor, ceil}`. The resulting sub-MIP — a very small
//! neighborhood around the rounded LP point — is solved to extract a feasible
//! incumbent that the fractional LP solution does not directly provide.

use crate::mip::{MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::SolverResult;
use std::time::Instant;

/// Regular RENS cadence once branch-and-bound already has an incumbent.
///
/// After the first incumbent, RENS is an improvement heuristic competing with
/// the main tree search for time, so keep the historical spacing here.
pub(crate) const RENS_INTERVAL_WITH_INCUMBENT: usize = 200;

/// Node limit for the RENS sub-MIP.
const RENS_NODE_LIMIT: usize = 2_000;

/// Fraction of remaining wall-clock budget given to the sub-MIP.
const RENS_TIME_FRACTION: f64 = 0.10;

/// Absolute upper bound on sub-MIP wall time (seconds).
const RENS_MAX_TIME_SECS: f64 = 10.0;

/// Minimum remaining budget below which RENS is skipped.
const RENS_MIN_REMAINING_SECS: f64 = 1.0;

/// Run the RENS heuristic on a node LP relaxation `x_lp`.
///
/// For every integer variable `j`:
/// - if `x_lp[j]` is integral (within `cfg.integer_feas_tol`), fix it to that
///   integer;
/// - otherwise restrict it to the closed box `[floor(x_lp[j]), ceil(x_lp[j])]`.
///
/// The reduced sub-MIP is solved with a short timeout and node limit. Returns a
/// feasible `SolverResult` or `None` when the LP point is already integral
/// (nothing to enforce) or the sub-MIP finds no feasible point.
///
/// `parent_opts` is cloned and its timeout/deadline overridden so tolerance,
/// cancellation flag, and other settings are inherited by the sub-MIP.
pub(crate) fn run_rens(
    problem: &MilpProblem,
    x_lp: &[f64],
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_opts: &SolverOptions,
) -> Option<SolverResult> {
    let remaining_secs = remaining_budget(deadline);
    if remaining_secs < RENS_MIN_REMAINING_SECS {
        return None;
    }

    let mut sub_bounds = problem.lp.bounds.clone();
    let mut n_fractional = 0usize;
    for &j in &problem.integer_vars {
        if j >= x_lp.len() {
            continue;
        }
        let v = x_lp[j];
        let rounded = v.round();
        if (v - rounded).abs() <= cfg.integer_feas_tol {
            // Already integral: fix to the integer, intersecting the original box.
            let (lb, ub) = problem.lp.bounds[j];
            if rounded < lb || rounded > ub {
                return None;
            }
            sub_bounds[j] = (rounded, rounded);
        } else {
            // Fractional: restrict to {floor, ceil} ∩ original box.
            let lo = v.floor().max(problem.lp.bounds[j].0);
            let hi = v.ceil().min(problem.lp.bounds[j].1);
            if lo > hi {
                return None;
            }
            sub_bounds[j] = (lo, hi);
            n_fractional += 1;
        }
    }

    // No fractional integer var ⇒ the LP point is already integer-feasible and
    // is returned directly by the caller; RENS would add nothing.
    if n_fractional == 0 {
        return None;
    }

    let sub_timeout = (remaining_secs * RENS_TIME_FRACTION).min(RENS_MAX_TIME_SECS);

    let mut sub_lp = problem.lp.clone();
    sub_lp.bounds = sub_bounds;
    // MilpProblem::new only rejects an integer-var index >= lp.num_vars;
    // only `bounds` was mutated above, so `problem.integer_vars` (already valid
    // for `problem` by construction) remains valid for `sub_lp`.
    let sub_problem = MilpProblem::new(sub_lp, problem.integer_vars.clone())
        .expect("bounds-only mutation preserves num_vars; integer_vars already validated");

    let mut sub_cfg = cfg.clone();
    sub_cfg.max_nodes = RENS_NODE_LIMIT;
    sub_cfg.rins_enabled = false;
    sub_cfg.rens_enabled = false;
    sub_cfg.local_branching_enabled = false;

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

    let result = super::solve_sub_milp(&sub_problem, &sub_opts, &sub_cfg);
    super::usable_sub_mip_result_for_original(problem, result, cfg.integer_feas_tol)
}

fn remaining_budget(deadline: &Option<Instant>) -> f64 {
    match deadline {
        None => f64::INFINITY,
        Some(d) => {
            let now = Instant::now();
            if now >= *d {
                0.0
            } else {
                (*d - now).as_secs_f64()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mip::branch::is_integer_feasible;
    use crate::mip::integer_mask;
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    /// min c·x  s.t.  x0 + x1 <= b,  x ∈ {0,1}^2.
    fn knap2(c: [f64; 2], b: f64) -> MilpProblem {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            c.to_vec(),
            a,
            vec![b],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0); 2],
            None,
        )
        .unwrap();
        MilpProblem::new(lp, vec![0, 1]).unwrap()
    }

    /// SENTINEL: RENS produces a feasible integer incumbent that the fractional
    /// root LP does **not** directly give.
    ///
    /// Problem: max x0+x1 (min -x0-x1) s.t. x0+x1 <= 1, x ∈ {0,1}^2.
    /// LP root x_lp = (0.5, 0.5) is fractional (NOT integer-feasible), so the
    /// caller has no incumbent from it. RENS restricts both vars to {0,1} and
    /// solves the sub-MIP → optimal (1,0) or (0,1) with obj = -1.
    ///
    /// A no-op RENS (always `None`) makes `.expect(...)` fail. Returning the raw
    /// fractional LP point would fail the integer-feasibility assertion.
    #[test]
    fn rens_yields_incumbent_root_lp_lacks() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        // Precondition: the LP point itself is not integer-feasible.
        let mask = integer_mask(2, &problem.integer_vars);
        assert!(
            !is_integer_feasible(&x_lp, &mask, cfg.integer_feas_tol),
            "test premise: x_lp must be fractional"
        );

        let res = run_rens(&problem, &x_lp, &cfg, &None, &SolverOptions::default())
            .expect("RENS must produce a feasible incumbent from a fractional LP point");
        assert!(
            is_integer_feasible(&res.solution, &mask, cfg.integer_feas_tol),
            "RENS solution must be integer-feasible: {:?}",
            res.solution
        );
        assert!(
            (res.objective - (-1.0)).abs() < 1e-6,
            "RENS optimum over {{0,1}}^2 with x0+x1<=1 is -1; got {}",
            res.objective
        );
    }

    #[test]
    fn rens_run_path_accepts_feasible_timeout_incumbent() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];
        super::super::set_next_sub_mip_result(SolverResult {
            status: crate::problem::SolveStatus::Timeout,
            objective: -1.0e100,
            solution: vec![1.0, 0.0],
            ..SolverResult::default()
        });

        let result = run_rens(&problem, &x_lp, &cfg, &None, &SolverOptions::default())
            .expect("RENS must keep feasible timeout incumbent from sub-MIP");

        assert_eq!(result.solution, vec![1.0, 0.0]);
        assert_eq!(result.objective, -1.0);
    }

    /// RENS returns `None` when the LP point is already integral (nothing to
    /// enforce — the caller adopts it directly as a leaf).
    ///
    /// Sentinel: removing the `n_fractional == 0` guard re-solves the fully fixed
    /// sub-MIP and returns `Some` → FAILS.
    #[test]
    fn rens_skips_integral_lp_point() {
        let problem = knap2([-1.0, -2.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.0, 1.0];
        assert!(
            run_rens(&problem, &x_lp, &cfg, &None, &SolverOptions::default()).is_none(),
            "RENS must skip an already-integral LP point"
        );
    }

    /// RENS restricts a fractional var to {floor, ceil} only — it cannot jump to
    /// a far integer. With x0+x1 <= 3 and x_lp = (0.4, 0.4), RENS searches
    /// {0,1}×{0,1}; the box-respecting optimum is (1,1) = -2, never (3,3).
    #[test]
    fn rens_neighborhood_is_floor_ceil_only() {
        // bounds widened to [0,3] so a no-op on the floor/ceil restriction could
        // reach -6; the {floor,ceil} restriction caps the optimum at -2.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![6.0],
            vec![ConstraintType::Le],
            vec![(0.0, 3.0); 2],
            None,
        )
        .unwrap();
        let problem = MilpProblem::new(lp, vec![0, 1]).unwrap();
        let cfg = MipConfig::default();
        let x_lp = vec![0.4, 0.4];

        let res = run_rens(&problem, &x_lp, &cfg, &None, &SolverOptions::default())
            .expect("fractional LP point → RENS Some");
        assert!(
            (res.objective - (-2.0)).abs() < 1e-6,
            "RENS over {{0,1}}^2 optimum is -2 (not -6); got {}",
            res.objective
        );
    }

    #[test]
    fn rens_run_path_passes_recursive_sub_mip_config() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            rins_enabled: true,
            rens_enabled: true,
            local_branching_enabled: true,
            ..MipConfig::default()
        };
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let result = run_rens(&problem, &x_lp, &cfg, &None, &SolverOptions::default());
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RENS must call the recursive sub-MIP"
        );
        assert_eq!(
            configs.len(),
            1,
            "RENS run path must solve exactly one sub-MIP"
        );
        let sub_cfg = &configs[0];
        assert_eq!(sub_cfg.max_nodes, RENS_NODE_LIMIT);
        assert!(!sub_cfg.rins_enabled, "recursive RINS must be disabled");
        assert!(!sub_cfg.rens_enabled, "recursive RENS must be disabled");
        assert!(
            !sub_cfg.local_branching_enabled,
            "recursive local branching must be disabled"
        );
    }

    /// RENS skips when the deadline is already past (no work after expiry).
    #[test]
    fn rens_skips_on_expired_deadline() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];
        let past = Instant::now() - std::time::Duration::from_secs(1);
        assert!(
            run_rens(
                &problem,
                &x_lp,
                &cfg,
                &Some(past),
                &SolverOptions::default()
            )
            .is_none(),
            "RENS must not run after the deadline"
        );
    }
}
