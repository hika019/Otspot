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

/// Fixed sub-MIP wall-clock timeout (seconds).
///
/// Phase 1c/1d (P1-A): previously scaled as `(remaining_secs * 0.10).min(10.0)`;
/// see `rins::RINS_MAX_TIME_SECS` for why this was changed to a fixed value
/// (a wall-clock-scaled sub-MIP budget makes the returned incumbent, and
/// hence the parent search's trajectory, run-timing dependent). Phase 1d
/// (P1-B): `sub_cfg.max_lp_iters` (see `heuristics::SUB_MIP_MAX_LP_ITERS`) is
/// now the primary, deterministic stop; this wall-clock cap is expected to
/// stay dormant for the wide majority of calls and fire only as a final
/// safety valve against pathologically slow per-iteration LP solves.
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
/// feasible `SolverResult` (or `None` when the LP point is already integral —
/// nothing to enforce — or the sub-MIP finds no feasible point) together with
/// the sub-MIP's `nodes_processed` and its own recursive `total_simplex_iters`
/// (see `effort`), both reported whenever a sub-MIP solve was actually
/// attempted (0 when skipped before that point).
///
/// `iter_budget` is RENS's own remaining share of `effort::
/// total_simplex_iters` (see `Relaxation::run_rens`'s doc); the sub-MIP's
/// `max_lp_iters` is capped at `min(heuristics::SUB_MIP_MAX_LP_ITERS,
/// iter_budget)`, skipping the call outright when `iter_budget` is below
/// `heuristics::SUB_MIP_MIN_LP_ITERS` (see
/// `heuristics::capped_sub_mip_max_lp_iters`).
///
/// `parent_opts` is cloned and its timeout/deadline overridden so tolerance,
/// cancellation flag, and other settings are inherited by the sub-MIP.
pub(crate) fn run_rens(
    problem: &MilpProblem,
    x_lp: &[f64],
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    iter_budget: u64,
    parent_opts: &SolverOptions,
) -> (Option<SolverResult>, u64, u64) {
    let remaining_secs = remaining_budget(deadline);
    if remaining_secs < RENS_MIN_REMAINING_SECS {
        return (None, 0, 0);
    }
    let Some(max_lp_iters) = super::capped_sub_mip_max_lp_iters(iter_budget) else {
        return (None, 0, 0);
    };

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
                return (None, 0, 0);
            }
            sub_bounds[j] = (rounded, rounded);
        } else {
            // Fractional: restrict to {floor, ceil} ∩ original box.
            let lo = v.floor().max(problem.lp.bounds[j].0);
            let hi = v.ceil().min(problem.lp.bounds[j].1);
            if lo > hi {
                return (None, 0, 0);
            }
            sub_bounds[j] = (lo, hi);
            n_fractional += 1;
        }
    }

    // No fractional integer var ⇒ the LP point is already integer-feasible and
    // is returned directly by the caller; RENS would add nothing.
    if n_fractional == 0 {
        return (None, 0, 0);
    }

    let sub_timeout = RENS_MAX_TIME_SECS;

    let mut sub_lp = problem.lp.clone();
    sub_lp.bounds = sub_bounds;
    // MilpProblem::new only rejects an integer-var index >= lp.num_vars;
    // only `bounds` was mutated above, so `problem.integer_vars` (already valid
    // for `problem` by construction) remains valid for `sub_lp`.
    let sub_problem = MilpProblem::new(sub_lp, problem.integer_vars.clone())
        .expect("bounds-only mutation preserves num_vars; integer_vars already validated");

    let mut sub_cfg = cfg.clone();
    sub_cfg.max_nodes = RENS_NODE_LIMIT;
    sub_cfg.max_lp_iters = Some(max_lp_iters);
    sub_cfg.rins_enabled = false;
    sub_cfg.rens_enabled = false;
    sub_cfg.local_branching_enabled = false;
    // The sub-MIP searches a {floor, ceil} box around the LP point, already
    // a tiny neighborhood; recursive tree-cut separation, symmetry-breaking,
    // and root cut generation pay the parent search's per-node/per-root
    // overhead again for no corresponding benefit here (Phase 1a/1c: freed
    // iteration budget is reallocated to the parent tree search via
    // `mip::effort`).
    sub_cfg.tree_cuts = false;
    sub_cfg.symmetry = false;
    sub_cfg.cuts = false;

    let mut sub_opts = parent_opts.clone();
    sub_opts.deadline = Some(super::sub_mip_deadline(deadline, sub_timeout));
    sub_opts.timeout_secs = None;
    sub_opts.warm_start = None;
    sub_opts.warm_start_qp = None;
    sub_opts.warm_start_lp = None;
    sub_opts.known_optimal_obj = None;
    sub_opts.presolve = true;
    sub_opts.use_lp_crash_basis = true;
    sub_opts.recover_warm_start_basis = false;
    sub_opts.threads = 1;

    let (result, sub_stats) = super::solve_sub_milp(&sub_problem, &sub_opts, &sub_cfg);
    let sub_mip_nodes = sub_stats.nodes_processed as u64;
    let sub_mip_iters = crate::mip::effort::total_simplex_iters(&sub_stats);
    (
        super::usable_sub_mip_result_for_original(problem, result, cfg.integer_feas_tol),
        sub_mip_nodes,
        sub_mip_iters,
    )
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
    use otspot_num::sparse::CscMatrix;

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

        let (res, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let res = res.expect("RENS must produce a feasible incumbent from a fractional LP point");
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

    /// NEW (Phase 0): an attempted RENS sub-MIP solve reports its node count.
    ///
    /// Sentinel: a Phase 0 revert (the `run_rens`/`solve_sub_milp` return type
    /// change stripped back to `Option<SolverResult>`) has no way to expose
    /// this count, so `sub_mip_nodes_total` would stay 0 — this test would fail.
    #[test]
    fn rens_reports_sub_mip_nodes_processed() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        let (res, sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        assert!(res.is_some(), "test premise: RENS must attempt a sub-MIP");
        assert!(
            sub_mip_nodes > 0,
            "an attempted sub-MIP solve must report at least one processed node; got {sub_mip_nodes}"
        );
    }

    /// NEW (Phase 1c): an attempted RENS sub-MIP solve also reports its own
    /// recursive `total_simplex_iters` (the unit `effort::may_run_rens` gates
    /// on), not just its node count.
    ///
    /// Sentinel: a Phase 1c revert (the `run_rens` return type stripped back
    /// to `(Option<SolverResult>, u64)`) has no way to expose this count, so
    /// `rens_iters` would stay 0 — this test would fail.
    #[test]
    fn rens_reports_sub_mip_iters_processed() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        let (res, _sub_mip_nodes, sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        assert!(res.is_some(), "test premise: RENS must attempt a sub-MIP");
        assert!(
            sub_mip_iters > 0,
            "an attempted sub-MIP solve must report at least one simplex iteration; got {sub_mip_iters}"
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

        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let result = result.expect("RENS must keep feasible timeout incumbent from sub-MIP");

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
            run_rens(
                &problem,
                &x_lp,
                &cfg,
                &None,
                u64::MAX,
                &SolverOptions::default()
            )
            .0
            .is_none(),
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

        let (res, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let res = res.expect("fractional LP point → RENS Some");
        assert!(
            (res.objective - (-2.0)).abs() < 1e-6,
            "RENS over {{0,1}}^2 optimum is -2 (not -6); got {}",
            res.objective
        );
    }

    /// NEW (Phase 1a): the disabled tree-cuts/symmetry flags reach the
    /// recursive sub-MIP solve.
    ///
    /// Sentinel: removing `sub_cfg.tree_cuts = false` or
    /// `sub_cfg.symmetry = false` from `run_rens` fails this test via the
    /// recorded sub-MIP config.
    #[test]
    fn rens_run_path_disables_tree_cuts_and_symmetry_recursively() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            tree_cuts: true,
            symmetry: true,
            ..MipConfig::default()
        };
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
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
        assert!(!sub_cfg.tree_cuts, "recursive tree cuts must be disabled");
        assert!(
            !sub_cfg.symmetry,
            "recursive symmetry breaking must be disabled"
        );
    }

    /// NEW (Phase 1c/P2-1): the sub-MIP config also disables root cut
    /// generation, reaching the recursive sub-MIP solve.
    ///
    /// Sentinel: removing `sub_cfg.cuts = false` from `run_rens` fails this
    /// test via the recorded sub-MIP config.
    #[test]
    fn rens_run_path_disables_root_cuts_recursively() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            cuts: true,
            ..MipConfig::default()
        };
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
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
        assert!(
            !configs[0].cuts,
            "recursive root cut generation must be disabled"
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
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
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

    /// Phase 1d (P1-B): the RENS sub-MIP config carries the deterministic
    /// `max_lp_iters` cap, not just the node limit.
    ///
    /// Sentinel: removing `sub_cfg.max_lp_iters = ...` from `run_rens`'s
    /// sub-MIP config construction fails this test.
    #[test]
    fn rens_sub_mip_sets_deterministic_lp_iters_cap() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RENS must call the recursive sub-MIP"
        );
        assert_eq!(configs.len(), 1);
        assert_eq!(
            configs[0].max_lp_iters,
            Some(crate::mip::heuristics::SUB_MIP_MAX_LP_ITERS)
        );
    }

    /// SENTINEL (Codex review, P1): `run_rens`'s sub-MIP `max_lp_iters` is
    /// capped by RENS's own remaining iteration-share budget (`iter_budget`),
    /// not just the flat `SUB_MIP_MAX_LP_ITERS` constant — an "approved"
    /// (`effort::may_run_rens == true`) call could otherwise still hand the
    /// sub-MIP up to `SUB_MIP_MAX_LP_ITERS` iterations regardless of how
    /// little share was actually left. `50_000` is above `heuristics::
    /// SUB_MIP_MIN_LP_ITERS` (30_000), so this exercises the
    /// share-caps-below-the-flat-constant path rather than the skip path
    /// (see `rens_skips_when_remaining_share_budget_is_below_min`).
    ///
    /// Sentinel: reverting `run_rens`'s `sub_cfg.max_lp_iters` assignment
    /// back to the flat `super::SUB_MIP_MAX_LP_ITERS` constant (ignoring
    /// `iter_budget`) makes this assert `Some(SUB_MIP_MAX_LP_ITERS)` instead
    /// of `Some(50_000)`, failing.
    #[test]
    fn rens_sub_mip_max_lp_iters_is_capped_by_remaining_share_budget() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            50_000,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RENS must call the recursive sub-MIP"
        );
        assert_eq!(configs.len(), 1);
        assert_eq!(
            configs[0].max_lp_iters,
            Some(50_000),
            "remaining share budget (50_000) must win over the larger flat \
             SUB_MIP_MAX_LP_ITERS cap"
        );
    }

    /// SENTINEL (markshare_4_0 regression fix): a small but *nonzero*
    /// remaining share budget below `heuristics::SUB_MIP_MIN_LP_ITERS` must
    /// skip RENS outright rather than attempt it with a truncated
    /// `max_lp_iters` — see that constant's doc for why a truncated call is
    /// worse than no call.
    ///
    /// Sentinel: removing the `SUB_MIP_MIN_LP_ITERS` floor from
    /// `capped_sub_mip_max_lp_iters` (reverting to skip only at exactly 0)
    /// makes `result.is_some()` and fails the recorded-config assertions.
    #[test]
    fn rens_skips_when_remaining_share_budget_is_below_min() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, sub_mip_nodes, sub_mip_iters) = run_rens(
            &problem,
            &x_lp,
            &cfg,
            &None,
            crate::mip::heuristics::SUB_MIP_MIN_LP_ITERS - 1,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_none(),
            "a remaining share below SUB_MIP_MIN_LP_ITERS must skip RENS"
        );
        assert_eq!(configs.len(), 0, "the sub-MIP must never be attempted");
        assert_eq!(sub_mip_nodes, 0);
        assert_eq!(sub_mip_iters, 0);
    }

    /// SENTINEL (Codex review, P1): a remaining share budget of exactly 0
    /// skips the sub-MIP call outright — it is never attempted, not
    /// attempted with `Some(0)`.
    ///
    /// Sentinel: removing the `capped_sub_mip_max_lp_iters` early-return from
    /// `run_rens` calls `solve_sub_milp` anyway, failing the recorded-config
    /// count assertion.
    #[test]
    fn rens_skips_when_remaining_share_budget_is_zero() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        super::super::clear_recorded_sub_mip_configs();
        let (result, sub_mip_nodes, sub_mip_iters) =
            run_rens(&problem, &x_lp, &cfg, &None, 0, &SolverOptions::default());
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_none(),
            "zero remaining share budget must skip RENS"
        );
        assert_eq!(configs.len(), 0, "the sub-MIP must never be attempted");
        assert_eq!(sub_mip_nodes, 0);
        assert_eq!(sub_mip_iters, 0);
    }

    /// Codex review (P1): the sub-MIP's `SolverOptions::deadline` must be
    /// `min(parent deadline, RENS_MAX_TIME_SECS)`, not always
    /// `now + RENS_MAX_TIME_SECS` regardless of how little of the parent's
    /// own budget remains — the latter let the sub-MIP run up to
    /// `RENS_MAX_TIME_SECS` past the user's requested overall timeout.
    ///
    /// Sentinel: reverting to `sub_opts.deadline = None` (with only
    /// `sub_opts.timeout_secs = Some(sub_timeout)`) fails the near-deadline
    /// case here, since the recorded deadline would then be `None`.
    #[test]
    fn rens_sub_mip_deadline_is_min_of_parent_and_fixed_cap() {
        let problem = knap2([-1.0, -1.0], 1.0);
        let cfg = MipConfig::default();
        let x_lp = vec![0.5, 0.5];

        // Far parent deadline: the fixed RENS_MAX_TIME_SECS cap must win.
        super::super::clear_recorded_sub_mip_configs();
        let before = Instant::now();
        let far_parent_deadline = before + std::time::Duration::from_secs(1000);
        run_rens(
            &problem,
            &x_lp,
            &cfg,
            &Some(far_parent_deadline),
            u64::MAX,
            &SolverOptions::default(),
        );
        let deadlines = super::super::take_recorded_sub_mip_deadlines();
        assert_eq!(deadlines.len(), 1, "test premise: exactly one sub-MIP call");
        let recorded = deadlines[0].expect("sub-MIP deadline must be set");
        assert!(
            recorded < before + std::time::Duration::from_secs(20),
            "far parent deadline must not override the fixed RENS_MAX_TIME_SECS cap"
        );

        // Near parent deadline (< RENS_MAX_TIME_SECS away, > RENS_MIN_REMAINING_SECS):
        // the parent deadline must win over the fixed cap.
        super::super::clear_recorded_sub_mip_configs();
        let before = Instant::now();
        let near_parent_deadline = before + std::time::Duration::from_secs(2);
        run_rens(
            &problem,
            &x_lp,
            &cfg,
            &Some(near_parent_deadline),
            u64::MAX,
            &SolverOptions::default(),
        );
        let deadlines = super::super::take_recorded_sub_mip_deadlines();
        assert_eq!(deadlines.len(), 1, "test premise: exactly one sub-MIP call");
        let recorded = deadlines[0].expect("sub-MIP deadline must be set");
        assert!(
            recorded <= near_parent_deadline + std::time::Duration::from_millis(50),
            "sub-MIP deadline must not exceed the near parent deadline"
        );
        assert!(
            recorded < before + std::time::Duration::from_secs(9),
            "near parent deadline must win over the fixed RENS_MAX_TIME_SECS cap"
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
                u64::MAX,
                &SolverOptions::default()
            )
            .0
            .is_none(),
            "RENS must not run after the deadline"
        );
    }
}
