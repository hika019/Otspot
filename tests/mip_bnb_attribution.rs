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
//!
//! `mip::effort`'s heuristic/separation/strong-branch iteration-share gate
//! sentinels live in `otspot-core/src/mip/tests.rs`
//! (`heuristic_iter_share_is_enforced`, `separation_iter_share_is_enforced`,
//! `strong_branch_iter_share_is_enforced`) as deterministic white-box wiring
//! tests, not here — see the trailing note in this file for why an organic,
//! kernel-driven version of those tests was tried and rejected as unreliable.

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
///
/// Phase 1b's wall-clock `MipEffortBudget` briefly made the B&B trajectory
/// itself timing-dependent (RINS/RENS's *exact* node-count checkpoints could
/// all miss a fractional leaf on an unlucky run at the original N=70), which
/// is why this was widened to N=200. Phase 1c's deterministic
/// simplex-iteration gate (`mip::effort`) removed that source of
/// non-determinism — see `attribution_is_deterministic_across_repeated_runs`
/// below — but N=200 is kept since it is not otherwise harmful and a smaller
/// N is not required now.
const KNAPSACK_N: usize = 200;
const KNAPSACK_SEED: u64 = 11;

fn solve_attribution_instance() -> (SolverResult, MipStats, u64) {
    let problem = gen_knapsack_milp(KNAPSACK_N, 1.0, KNAPSACK_SEED);
    // NEW (Phase 1d/P2-A): no wall-clock timeout at all (was `Some(120.0)`).
    // With a deadline in play, `attribution_is_deterministic_across_repeated_runs`
    // could only be an empirical observation (a run could always, in
    // principle, race against real time and get cut off at a different
    // point) rather than a structural guarantee. Removing the deadline
    // entirely makes determinism a property of the algorithm alone: nothing
    // in the solve path other than `effort::may_run_*` (now iteration-based)
    // can make this instance's trajectory diverge between runs.
    let opts = SolverOptions::default();
    let cfg = MipConfig::default();

    let t0 = Instant::now();
    let (res, stats) = solve_milp_with_stats(&problem, &opts, &cfg);
    let wall_us = t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    (res, stats, wall_us)
}

/// SENTINEL: the Phase 0 attribution counters, together with the pre-existing
/// `lp_solve_us_total` / `node_propagation_us`, cover >= 95% of the *B&B
/// loop's own* wall clock (root-level presolve/feasibility-pump/root-cut
/// time is deliberately excluded here — see
/// `attribution_covers_wall_clock_including_root_overhead` below for the
/// root-inclusive variant).
///
/// NEW (P2-4/P3-1): restores the original Phase 0 test (replaced outright in
/// Phase 1b when root-level `fp_us`/`root_cut_us` became large enough,
/// relative to a much-faster loop, to push the sole coverage definition then
/// in use below the 95% floor) as a SEPARATE test alongside the
/// root-inclusive one — loop-only and root-inclusive attribution are both
/// useful views and coexist rather than replace each other.
///
/// A Phase 0 revert (the new fields never written, staying at their `0`
/// default) collapses the covered sum to `lp_solve_us_total +
/// node_propagation_us` alone; the first assertion below checks that this
/// pre-Phase-0 baseline does NOT already reach 95% of wall clock for this
/// instance, so the second assertion is a real sentinel rather than one that
/// would pass either way.
#[test]
fn attribution_covers_loop_wall_clock() {
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
        "Phase 0 attribution must cover >= 95% of the B&B loop's own wall clock: \
         covered={covered_us}us wall={wall_us}us (lp_solve={} propagation={} tree_cut={} \
         rins={} rens={} local_branching={} branch_select={} conflict={} other={}); if this \
         fails only because root-level fp_us/root_cut_us grew large relative to a fast loop, \
         see attribution_covers_wall_clock_including_root_overhead for the root-inclusive view",
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

/// SENTINEL (P1-1 acceptance criterion): the deterministic simplex-iteration
/// `mip::effort` gate makes `nodes_processed` reproducible across repeated
/// runs of the same deterministic instance, unlike Phase 1b's wall-clock
/// `MipEffortBudget` (which spread this same instance's node count across a
/// wide range run-to-run — see `KNAPSACK_N`'s doc comment).
///
/// Sentinel: reintroducing any wall-clock (`Instant`) input into an
/// `effort::may_run_*` decision would make repeated runs disagree here.
#[test]
fn attribution_is_deterministic_across_repeated_runs() {
    let (res1, stats1, _) = solve_attribution_instance();
    let (res2, stats2, _) = solve_attribution_instance();
    let (res3, stats3, _) = solve_attribution_instance();
    assert_eq!(
        (res1.status, res2.status, res3.status),
        (
            SolveStatus::Optimal,
            SolveStatus::Optimal,
            SolveStatus::Optimal
        ),
        "test premise: all three runs must solve to optimality"
    );
    assert_eq!(
        stats2.nodes_processed, stats1.nodes_processed,
        "run 2's nodes_processed must match run 1's"
    );
    assert_eq!(
        stats3.nodes_processed, stats1.nodes_processed,
        "run 3's nodes_processed must match run 1's"
    );
    assert_eq!(res2.objective, res1.objective);
    assert_eq!(res3.objective, res1.objective);
}

/// SENTINEL: the Phase 0 attribution counters, together with the pre-existing
/// `lp_solve_us_total` / `node_propagation_us` and the root-level
/// `fp_us` / `root_cut_us`, cover >= 95% of the search wall clock.
///
/// A Phase 0 revert (the new fields never written, staying at their `0`
/// default) collapses the covered sum to `lp_solve_us_total +
/// node_propagation_us + fp_us + root_cut_us` alone; the first assertion
/// below checks that this pre-Phase-0 baseline does NOT already reach 95% of
/// wall clock for this instance, so the second assertion is a real sentinel
/// rather than one that would pass either way.
///
/// NEW (Phase 1b): replaces the original `attribution_covers_wall_clock`
/// (CLAUDE.md: behaviour changes are expressed as delete-old + add-new, never
/// an in-place edit of an existing test). The original coverage sum omitted
/// `fp_us` (feasibility pump) and `root_cut_us` (root GMI/MIR generation) —
/// both pre-existing Phase 0 fields — because they were a negligible sliver
/// of this instance's ~700ms wall clock at Phase 0. Phase 1b's
/// `MipEffortBudget` throttles in-tree heuristics/separation, which shrinks
/// this same instance's B&B-loop wall time roughly 5x (to ~110-150ms); the
/// root-level `fp_us + root_cut_us` cost does not shrink with it (it runs
/// once, before the loop, unaffected by the loop's own time-share gates), so
/// it becomes a non-negligible fraction of the now-much-smaller total. This
/// version folds both fields into the sum so the sentinel again reflects the
/// true (root + loop) wall-clock coverage instead of loop-only coverage.
#[test]
fn attribution_covers_wall_clock_including_root_overhead() {
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

    let pre_phase0_covered =
        stats.lp_solve_us_total + stats.node_propagation_us + stats.fp_us + stats.root_cut_us;
    assert!(
        (pre_phase0_covered as f64) < 0.95 * wall_us as f64,
        "test premise: lp_solve_us_total + node_propagation_us + fp_us + root_cut_us alone \
         must NOT already cover 95% of wall clock (pre_phase0_covered={pre_phase0_covered}us \
         wall={wall_us}us) -- otherwise this instance cannot distinguish a Phase 0 revert"
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
         wall={wall_us}us (lp_solve={} propagation={} fp={} root_cut={} tree_cut={} rins={} \
         rens={} local_branching={} branch_select={} conflict={} other={})",
        stats.lp_solve_us_total,
        stats.node_propagation_us,
        stats.fp_us,
        stats.root_cut_us,
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

// NOTE (Phase 1b): `heuristic_iter_share_is_enforced` /
// `separation_iter_share_is_enforced` / `strong_branch_iter_share_is_enforced`
// live in `otspot-core/src/mip/tests.rs`, not here.
//
// An organic, kernel-driven version of `heuristic_iter_share_is_enforced` was
// tried and measured unreliable: RINS/RENS/local-branching sub-MIP timeouts
// are computed from the *overall remaining deadline* (capped at an absolute
// 10s per call, see `RINS_MAX_TIME_SECS` etc.), not from the effort budget's
// remaining share, so a single call approved while elapsed time is still
// small can itself push the post-call ratio far past the target share
// (reproduced directly: one run measured a 79% combined heuristic share for
// this exact instance — *higher* than this same instance's fully-*ungated*
// baseline of ~50%, which makes wall-clock ratio unusable as a pass/fail
// threshold: no single ceiling both tolerates the gated tail and rejects a
// reverted gate). The gate itself is not a bug — it is checked correctly
// before every call — this is a real, separate design gap in how individual
// sub-MIP timeouts are sized, worth a follow-up but out of this task's scope
// (reported to the lead). The white-box tests in `mip/tests.rs` verify the
// wiring deterministically instead, independent of this timing gap.
