//! Phase 0 sentinels for the B&B time-attribution counters (`MipStats`).
//!
//! Phase 0 adds `tree_cut_us` / `rins_us` / `rens_us` / `local_branching_us` /
//! `branch_select_us` / `conflict_us` / `node_loop_other_us` /
//! `sub_mip_nodes_total` purely as measurement — no solver behaviour changes.
//! These tests pin that the new counters actually account for (most of) the
//! search wall clock and that hidden sub-MIP work becomes visible, using the
//! same deterministic knapsack generator the MIP speed bench solves (see
//! `tests/mip_bench_gen_correctness.rs` for the `#[path]` rationale: a single
//! source of truth for the generated problem).

use otspot::{
    options::{MipConfig, SolverOptions},
    problem::{SolveStatus, SolverResult},
    solve_milp_with_stats, MipStats,
};
use std::time::Instant;

#[path = "../otspot-dev/src/bin/mip_speed_bench/kernels.rs"]
mod kernels;
use kernels::gen_knapsack_milp;

/// Size tuned (empirically, seed fixed) so the tight-capacity 0/1 knapsack
/// needs enough B&B nodes to exercise RINS (every 100 nodes), RENS/local
/// branching (every 200 nodes) and in-tree cut separation at least once,
/// while still finishing in well under the 3-minute per-test budget.
const KNAPSACK_N: usize = 70;
const KNAPSACK_SEED: u64 = 11;

fn solve_attribution_instance() -> (SolverResult, MipStats, u64) {
    let problem = gen_knapsack_milp(KNAPSACK_N, 1.0, KNAPSACK_SEED);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);
    let cfg = MipConfig::default();

    let t0 = Instant::now();
    let (res, stats) = solve_milp_with_stats(&problem, &opts, &cfg);
    let wall_us = t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    (res, stats, wall_us)
}

/// SENTINEL: the Phase 0 attribution counters, together with the pre-existing
/// `lp_solve_us_total` / `node_propagation_us`, cover >= 95% of the search
/// wall clock.
///
/// A Phase 0 revert (the new fields never written, staying at their `0`
/// default) collapses the covered sum to `lp_solve_us_total +
/// node_propagation_us` alone; the first assertion below checks that this
/// pre-Phase-0 baseline does NOT already reach 95% of wall clock for this
/// instance, so the second assertion is a real sentinel rather than one that
/// would pass either way.
#[test]
fn attribution_covers_wall_clock() {
    let (res, stats, wall_us) = solve_attribution_instance();
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "test premise: instance must solve to optimality within the budget"
    );
    assert!(
        stats.nodes_processed >= 200,
        "test premise: the instance must explore enough nodes to exercise \
         RINS/RENS/local-branching/tree-cut/branch-select overhead; got {} nodes",
        stats.nodes_processed
    );

    let pre_phase0_covered = stats.lp_solve_us_total + stats.node_propagation_us;
    assert!(
        (pre_phase0_covered as f64) < 0.95 * wall_us as f64,
        "test premise: lp_solve_us_total + node_propagation_us alone must NOT already \
         cover 95% of wall clock (pre_phase0_covered={pre_phase0_covered}us wall={wall_us}us) \
         -- otherwise this instance cannot distinguish a Phase 0 revert"
    );

    let new_counters_us = stats.tree_cut_us
        + stats.rins_us
        + stats.rens_us
        + stats.local_branching_us
        + stats.branch_select_us
        + stats.conflict_us
        + stats.node_loop_other_us;
    let covered_us = pre_phase0_covered + new_counters_us;

    assert!(
        (covered_us as f64) >= 0.95 * wall_us as f64,
        "Phase 0 attribution must cover >= 95% of wall clock: covered={covered_us}us \
         wall={wall_us}us (lp_solve={} propagation={} tree_cut={} rins={} rens={} \
         local_branching={} branch_select={} conflict={} other={})",
        stats.lp_solve_us_total,
        stats.node_propagation_us,
        stats.tree_cut_us,
        stats.rins_us,
        stats.rens_us,
        stats.local_branching_us,
        stats.branch_select_us,
        stats.conflict_us,
        stats.node_loop_other_us,
    );
}

/// SENTINEL: `sub_mip_nodes_total` reports the hidden B&B work RINS/RENS/local
/// branching sub-MIPs do, which is otherwise invisible in `nodes_processed`.
///
/// Fails if `sub_mip_nodes_total` is never populated (Phase 0 reverted) while
/// at least one heuristic sub-MIP call is confirmed attempted.
#[test]
fn sub_mip_nodes_total_reflects_heuristic_work() {
    let (res, stats, _wall_us) = solve_attribution_instance();
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "test premise: instance must solve to optimality within the budget"
    );
    let heuristic_calls = stats.rins_calls + stats.rens_calls + stats.local_branching_calls;
    assert!(
        heuristic_calls > 0,
        "test premise: at least one RINS/RENS/local-branching call must be attempted; \
         got rins_calls={} rens_calls={} local_branching_calls={}",
        stats.rins_calls,
        stats.rens_calls,
        stats.local_branching_calls
    );
    assert!(
        stats.sub_mip_nodes_total > 0,
        "sub_mip_nodes_total must be > 0 when a heuristic sub-MIP was attempted \
         (rins_calls={} rens_calls={} local_branching_calls={}); got 0",
        stats.rins_calls,
        stats.rens_calls,
        stats.local_branching_calls
    );
}
