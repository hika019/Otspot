//! RINS (Relaxation Induced Neighborhood Search) heuristic for MILP.
//!
//! Reference: Danna, Rothberg & Le Pape (2005).

use crate::mip::{MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::SolverResult;
use std::time::Instant;

/// Run RINS every this many B&B nodes.
pub(crate) const RINS_INTERVAL: usize = 100;

/// Node limit for the RINS sub-MIP.
const RINS_NODE_LIMIT: usize = 1_000;

/// Fixed sub-MIP wall-clock timeout (seconds).
///
/// Phase 1c/1d (P1-A): previously scaled as `(remaining_secs * 0.10).min(10.0)`,
/// making the sub-MIP's own budget a continuous function of wall-clock time —
/// the sub-MIP's explored node count, the incumbent it returns, and hence the
/// parent search's iteration counters and gate decisions all became run-timing
/// dependent (confirmed by direct repro: `gt2` diverged onto genuinely
/// different B&B trajectories, not just cut off at different points, across
/// repeated runs at `--timeout 60`). A fixed timeout removes the *scaling*
/// dependency, but Phase 1d (P1-B) found this fixed cap can still itself be
/// the actively binding, timing-jitter-dependent stop for some instances (see
/// `heuristics::SUB_MIP_MAX_LP_ITERS`) — `sub_cfg.max_lp_iters` is now the
/// primary, deterministic stop; this wall-clock cap is expected to stay
/// dormant for the wide majority of calls and fire only as a final safety
/// valve against pathologically slow per-iteration LP solves.
const RINS_MAX_TIME_SECS: f64 = 10.0;

/// Minimum remaining budget below which RINS is skipped.
const RINS_MIN_REMAINING_SECS: f64 = 1.0;

/// Run the RINS heuristic.
///
/// Fixes integer variables where `round(x_lp[j]) == round(x_inc[j])`, then
/// solves the reduced sub-MIP with a short timeout and node limit. Returns an
/// improved `SolverResult` (or `None` when no improvement is found) together
/// with the sub-MIP's `nodes_processed` and its own recursive
/// `total_simplex_iters` (see `effort`), both reported whenever a sub-MIP
/// solve was actually attempted (0 when skipped before that point).
///
/// `iter_budget` is RINS's own remaining share of `effort::
/// total_simplex_iters` (see `Relaxation::run_rins`'s doc); the sub-MIP's
/// `max_lp_iters` is capped at `min(heuristics::SUB_MIP_MAX_LP_ITERS,
/// iter_budget)`, skipping the call outright when `iter_budget` is below
/// `heuristics::SUB_MIP_MIN_LP_ITERS` (see
/// `heuristics::capped_sub_mip_max_lp_iters`).
///
/// `parent_opts` is cloned and its timeout/deadline overridden so that
/// tolerance, cancellation flag, and other settings are inherited by the sub-MIP.
pub(crate) fn run_rins(
    problem: &MilpProblem,
    x_lp: &[f64],
    x_inc: &[f64],
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    iter_budget: u64,
    parent_opts: &SolverOptions,
) -> (Option<SolverResult>, u64, u64) {
    let remaining_secs = remaining_budget(deadline);
    if remaining_secs < RINS_MIN_REMAINING_SECS {
        return (None, 0, 0);
    }
    let Some(max_lp_iters) = super::capped_sub_mip_max_lp_iters(iter_budget) else {
        return (None, 0, 0);
    };

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
        return (None, 0, 0);
    }

    let sub_timeout = RINS_MAX_TIME_SECS;

    let mut sub_lp = problem.lp.clone();
    sub_lp.bounds = sub_bounds;
    // MilpProblem::new only rejects an integer-var index >= lp.num_vars;
    // only `bounds` was mutated above, so `problem.integer_vars` (already valid
    // for `problem` by construction) remains valid for `sub_lp`.
    let sub_problem = MilpProblem::new(sub_lp, problem.integer_vars.clone())
        .expect("bounds-only mutation preserves num_vars; integer_vars already validated");

    let sub_cfg = rins_sub_mip_config(cfg, max_lp_iters);

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

fn rins_sub_mip_config(cfg: &MipConfig, max_lp_iters: u64) -> MipConfig {
    let mut sub_cfg = cfg.clone();
    sub_cfg.max_nodes = RINS_NODE_LIMIT;
    sub_cfg.max_lp_iters = Some(max_lp_iters);
    sub_cfg.rins_enabled = false;
    sub_cfg.rens_enabled = false;
    sub_cfg.local_branching_enabled = false;
    // The sub-MIP searches a neighborhood RINS already restricted sharply
    // (variables fixed by LP/incumbent agreement); recursive tree-cut
    // separation, symmetry-breaking, and root cut generation pay the parent
    // search's per-node/per-root overhead again for a search space that is
    // already small, so all three are pure overhead here (Phase 1a/1c:
    // freed iteration budget is reallocated to the parent tree search via
    // `mip::effort`).
    sub_cfg.tree_cuts = false;
    sub_cfg.symmetry = false;
    sub_cfg.cuts = false;
    sub_cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use otspot_num::sparse::CscMatrix;

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

        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let result = result.expect("RINS must return Some when at least one variable is fixed");
        assert!(
            result.objective < -1.9,
            "RINS should improve below -2; got {}",
            result.objective
        );
    }

    /// NEW (Phase 0): an attempted RINS sub-MIP solve reports its node count.
    ///
    /// Sentinel: a Phase 0 revert (the `run_rins`/`solve_sub_milp` return type
    /// change stripped back to `Option<SolverResult>`) has no way to expose
    /// this count, so `sub_mip_nodes_total` would stay 0 — this test would fail.
    #[test]
    fn rins_reports_sub_mip_nodes_processed() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        let (result, sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        assert!(
            result.is_some(),
            "test premise: RINS must attempt a sub-MIP"
        );
        assert!(
            sub_mip_nodes > 0,
            "an attempted sub-MIP solve must report at least one processed node; got {sub_mip_nodes}"
        );
    }

    /// NEW (Phase 1c): an attempted RINS sub-MIP solve also reports its own
    /// recursive `total_simplex_iters` (the unit `effort::may_run_rins` gates
    /// on), not just its node count.
    ///
    /// Sentinel: a Phase 1c revert (the `run_rins` return type stripped back
    /// to `(Option<SolverResult>, u64)`) has no way to expose this count, so
    /// `rins_iters` would stay 0 — this test would fail.
    #[test]
    fn rins_reports_sub_mip_iters_processed() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        let (result, _sub_mip_nodes, sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        assert!(
            result.is_some(),
            "test premise: RINS must attempt a sub-MIP"
        );
        assert!(
            sub_mip_iters > 0,
            "an attempted sub-MIP solve must report at least one simplex iteration; got {sub_mip_iters}"
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
            run_rins(
                &problem,
                &x_lp,
                &x_inc,
                &cfg,
                &None,
                u64::MAX,
                &SolverOptions::default()
            )
            .0
            .is_none(),
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
            run_rins(
                &problem,
                &x_lp,
                &x_inc,
                &cfg,
                &Some(past),
                u64::MAX,
                &SolverOptions::default()
            )
            .0
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
        let cfg = MipConfig {
            rins_enabled: false,
            ..MipConfig::default()
        };
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

        let sub_cfg = rins_sub_mip_config(&cfg, crate::mip::heuristics::SUB_MIP_MAX_LP_ITERS);
        assert_eq!(sub_cfg.max_nodes, RINS_NODE_LIMIT);
        assert!(!sub_cfg.rins_enabled);
        assert!(!sub_cfg.rens_enabled);
        assert!(!sub_cfg.local_branching_enabled);
    }

    /// `rins_sub_mip_config` sets `max_lp_iters` to exactly the value it is
    /// given (Codex review, P1: `run_rins` now computes this from RINS's own
    /// remaining iteration-share budget — see the sibling `run_rins`-level
    /// sentinels below — rather than the config builder hardcoding the flat
    /// `SUB_MIP_MAX_LP_ITERS` constant).
    #[test]
    fn rins_sub_mip_config_sets_the_provided_lp_iters_cap() {
        let sub_cfg = rins_sub_mip_config(&MipConfig::default(), 12_345);
        assert_eq!(sub_cfg.max_lp_iters, Some(12_345));
    }

    /// SENTINEL (Codex review, P1): `run_rins`'s sub-MIP `max_lp_iters` is
    /// capped by RINS's own remaining iteration-share budget (`iter_budget`),
    /// not just the flat `SUB_MIP_MAX_LP_ITERS` constant — an "approved"
    /// (`effort::may_run_rins == true`) call could otherwise still hand the
    /// sub-MIP up to `SUB_MIP_MAX_LP_ITERS` iterations regardless of how
    /// little share was actually left, overshooting RINS's iteration share
    /// by up to that entire constant in one call. `50_000` is above
    /// `heuristics::SUB_MIP_MIN_LP_ITERS` (30_000), so this exercises the
    /// share-caps-below-the-flat-constant path rather than the skip path
    /// (see `rins_skips_when_remaining_share_budget_is_below_min`).
    ///
    /// Sentinel: reverting `rins_sub_mip_config`'s `max_lp_iters` assignment
    /// back to the flat `super::SUB_MIP_MAX_LP_ITERS` constant (ignoring
    /// `iter_budget`) makes this assert `Some(SUB_MIP_MAX_LP_ITERS)` instead
    /// of `Some(50_000)`, failing.
    #[test]
    fn rins_sub_mip_max_lp_iters_is_capped_by_remaining_share_budget() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            50_000,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RINS must call the recursive sub-MIP"
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
    /// skip RINS outright rather than attempt it with a truncated
    /// `max_lp_iters` — see that constant's doc for why a truncated call is
    /// worse than no call (fixed per-call overhead with no realistic chance
    /// of finding an improving point before hitting the cap).
    ///
    /// Sentinel: removing the `SUB_MIP_MIN_LP_ITERS` floor from
    /// `capped_sub_mip_max_lp_iters` (reverting to skip only at exactly 0)
    /// makes `result.is_some()` and fails the recorded-config assertions.
    #[test]
    fn rins_skips_when_remaining_share_budget_is_below_min() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, sub_mip_nodes, sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            crate::mip::heuristics::SUB_MIP_MIN_LP_ITERS - 1,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_none(),
            "a remaining share below SUB_MIP_MIN_LP_ITERS must skip RINS"
        );
        assert_eq!(configs.len(), 0, "the sub-MIP must never be attempted");
        assert_eq!(sub_mip_nodes, 0);
        assert_eq!(sub_mip_iters, 0);
    }

    /// An ample remaining share budget (larger than the flat constant) must
    /// not exceed the flat `SUB_MIP_MAX_LP_ITERS` cap.
    #[test]
    fn rins_sub_mip_max_lp_iters_is_flat_cap_when_share_budget_is_ample() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(result.is_some());
        assert_eq!(configs.len(), 1);
        assert_eq!(
            configs[0].max_lp_iters,
            Some(crate::mip::heuristics::SUB_MIP_MAX_LP_ITERS)
        );
    }

    /// SENTINEL (Codex review, P1): a remaining share budget of exactly 0
    /// skips the sub-MIP call outright — it is never attempted, not
    /// attempted with `Some(0)`.
    ///
    /// Sentinel: removing the `capped_sub_mip_max_lp_iters` early-return from
    /// `run_rins` calls `solve_sub_milp` anyway, failing the recorded-config
    /// count assertion.
    #[test]
    fn rins_skips_when_remaining_share_budget_is_zero() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, sub_mip_nodes, sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            0,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_none(),
            "zero remaining share budget must skip RINS"
        );
        assert_eq!(configs.len(), 0, "the sub-MIP must never be attempted");
        assert_eq!(sub_mip_nodes, 0);
        assert_eq!(sub_mip_iters, 0);
    }

    /// NEW (Phase 1a): the sub-MIP config disables recursive tree-cut
    /// separation and symmetry-breaking, not just the three recursive
    /// heuristic flags.
    ///
    /// Sentinel: removing either `sub_cfg.tree_cuts = false` or
    /// `sub_cfg.symmetry = false` from `rins_sub_mip_config` fails this test.
    #[test]
    fn rins_sub_mip_disables_tree_cuts_and_symmetry() {
        let cfg = MipConfig {
            max_nodes: 99_999,
            tree_cuts: true,
            symmetry: true,
            ..MipConfig::default()
        };

        let sub_cfg = rins_sub_mip_config(&cfg, crate::mip::heuristics::SUB_MIP_MAX_LP_ITERS);
        assert!(
            !sub_cfg.tree_cuts,
            "RINS sub-MIP must disable in-tree cut separation"
        );
        assert!(
            !sub_cfg.symmetry,
            "RINS sub-MIP must disable symmetry breaking"
        );
    }

    /// NEW (Phase 1a): the disabled tree-cuts/symmetry flags actually reach
    /// the recursive sub-MIP solve, not just the config-builder unit above.
    ///
    /// Sentinel: removing either flag from `rins_sub_mip_config` fails this
    /// test via the recorded sub-MIP config (same recording hook as the
    /// pre-existing `rins_run_path_passes_recursive_sub_mip_config`).
    #[test]
    fn rins_run_path_disables_tree_cuts_and_symmetry_recursively() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            tree_cuts: true,
            symmetry: true,
            ..MipConfig::default()
        };
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RINS must call the recursive sub-MIP"
        );
        assert_eq!(
            configs.len(),
            1,
            "RINS run path must solve exactly one sub-MIP"
        );
        let sub_cfg = &configs[0];
        assert!(!sub_cfg.tree_cuts, "recursive tree cuts must be disabled");
        assert!(
            !sub_cfg.symmetry,
            "recursive symmetry breaking must be disabled"
        );
    }

    /// Codex review (P1): the sub-MIP's `SolverOptions::deadline` must be
    /// `min(parent deadline, RINS_MAX_TIME_SECS)`, not always
    /// `now + RINS_MAX_TIME_SECS` regardless of how little of the parent's
    /// own budget remains — the latter let the sub-MIP run up to
    /// `RINS_MAX_TIME_SECS` past the user's requested overall timeout.
    ///
    /// Sentinel: reverting to `sub_opts.deadline = None` (with only
    /// `sub_opts.timeout_secs = Some(sub_timeout)`) fails the near-deadline
    /// case here, since the recorded deadline would then be `None`.
    #[test]
    fn rins_sub_mip_deadline_is_min_of_parent_and_fixed_cap() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        // Far parent deadline: the fixed RINS_MAX_TIME_SECS cap must win.
        super::super::clear_recorded_sub_mip_configs();
        let before = Instant::now();
        let far_parent_deadline = before + std::time::Duration::from_secs(1000);
        run_rins(
            &problem,
            &x_lp,
            &x_inc,
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
            "far parent deadline must not override the fixed RINS_MAX_TIME_SECS cap"
        );

        // Near parent deadline (< RINS_MAX_TIME_SECS away, > RINS_MIN_REMAINING_SECS):
        // the parent deadline must win over the fixed cap.
        super::super::clear_recorded_sub_mip_configs();
        let before = Instant::now();
        let near_parent_deadline = before + std::time::Duration::from_secs(2);
        run_rins(
            &problem,
            &x_lp,
            &x_inc,
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
            "near parent deadline must win over the fixed RINS_MAX_TIME_SECS cap"
        );
    }

    /// NEW (Phase 1c/P2-1): the sub-MIP config also disables root cut
    /// generation (`cuts`) — `add_root_cuts` runs a full `CUT_TIME_FRACTION`
    /// pass before the sub-MIP's own B&B even starts, which is pure overhead
    /// against a neighborhood already restricted by RINS's variable fixing.
    ///
    /// Sentinel: removing `sub_cfg.cuts = false` from `rins_sub_mip_config`
    /// fails this test.
    #[test]
    fn rins_sub_mip_disables_root_cuts() {
        let cfg = MipConfig {
            max_nodes: 99_999,
            cuts: true,
            ..MipConfig::default()
        };
        let sub_cfg = rins_sub_mip_config(&cfg, crate::mip::heuristics::SUB_MIP_MAX_LP_ITERS);
        assert!(
            !sub_cfg.cuts,
            "RINS sub-MIP must disable root cut generation"
        );
    }

    /// NEW (Phase 1c/P2-1): the disabled `cuts` flag actually reaches the
    /// recursive sub-MIP solve.
    ///
    /// Sentinel: removing `sub_cfg.cuts = false` from `rins_sub_mip_config`
    /// fails this test via the recorded sub-MIP config.
    #[test]
    fn rins_run_path_disables_root_cuts_recursively() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            cuts: true,
            ..MipConfig::default()
        };
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RINS must call the recursive sub-MIP"
        );
        assert_eq!(
            configs.len(),
            1,
            "RINS run path must solve exactly one sub-MIP"
        );
        assert!(
            !configs[0].cuts,
            "recursive root cut generation must be disabled"
        );
    }

    #[test]
    fn rins_run_path_passes_recursive_sub_mip_config() {
        let problem = two_var_milp([-1.0, -1.0], 3.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            rins_enabled: true,
            rens_enabled: true,
            local_branching_enabled: true,
            ..MipConfig::default()
        };
        let x_lp = vec![1.4, 1.6];
        let x_inc = vec![1.0, 1.0];

        super::super::clear_recorded_sub_mip_configs();
        let (result, _sub_mip_nodes, _sub_mip_iters) = run_rins(
            &problem,
            &x_lp,
            &x_inc,
            &cfg,
            &None,
            u64::MAX,
            &SolverOptions::default(),
        );
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: RINS must call the recursive sub-MIP"
        );
        assert_eq!(
            configs.len(),
            1,
            "RINS run path must solve exactly one sub-MIP"
        );
        let sub_cfg = &configs[0];
        assert_eq!(sub_cfg.max_nodes, RINS_NODE_LIMIT);
        assert!(!sub_cfg.rins_enabled, "recursive RINS must be disabled");
        assert!(!sub_cfg.rens_enabled, "recursive RENS must be disabled");
        assert!(
            !sub_cfg.local_branching_enabled,
            "recursive local branching must be disabled"
        );
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
