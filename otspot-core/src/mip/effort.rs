//! Deterministic per-solve simplex-iteration-share budget for optional B&B
//! work (Phase 1c; replaces Phase 1b's wall-clock `MipEffortBudget`).
//!
//! Phase 1b gated optional work (primal heuristics / in-tree separation /
//! strong branching) on elapsed wall-clock time. That made the B&B
//! trajectory itself timing-dependent: identical deterministic input could
//! explore a different number of nodes on different runs (observed directly:
//! main solved `gt2` to the same fixed 280-node trajectory 3/3 consecutive
//! runs, but the Phase 1b branch spread from 100 to 3000+ nodes across
//! repeats on the same machine). Simplex iteration counts are reproducible
//! for a fixed input and fixed algorithm path, so using them as the budget's
//! "clock" instead of `Instant` makes the whole B&B search deterministic
//! again while keeping the same throttling intent.
//!
//! Each gate compares a component's cumulative simplex iterations against a
//! share of [`total_simplex_iters`] — the iteration count spent anywhere in
//! this B&B search so far (ordinary node relaxations, strong branching,
//! heuristics including most of their sub-MIPs' own recursive cost — see the
//! feasibility-pump gap below — and in-tree separation). The check happens
//! once per would-be invocation, before that unit of work starts; an
//! in-progress sub-MIP solve or separation round always finishes once
//! started (Phase 1b's wall-clock `*_us` counters stay on `MipStats` as pure
//! measurement).
//!
//! **Known measurement gap (P2-B):** every MILP solve — including a RINS/
//! RENS/local-branching sub-MIP's recursive call — unconditionally runs its
//! own feasibility pump before branch-and-bound starts (`solve_milp_with_stats`).
//! That pump's own LP solves are timed into `fp_us` but their simplex
//! iterations are not threaded into `MipStats` at all, so a sub-MIP's
//! recursive `total_simplex_iters` (what `run_rins`/`run_rens`/
//! `run_local_branching` report back as `rins_iters`/etc.) *undercounts* by
//! whatever its own feasibility pump spent. Plumbing this through was judged
//! more invasive than it looks (`run_feasibility_pump` returns early from two
//! call sites, each via a second LP solve in `repair_with_fixed_integers`,
//! and 8 existing call sites — mostly tests — would need updating for a
//! `(SolverResult, iterations)` return), so it is documented here rather than
//! implemented. Root-level (non-recursive) `fp_us` is similarly not part of
//! the top-level `total_simplex_iters` computation used for gating a solve's
//! *own* node loop, only for sub-MIPs feeding their parent.
//!
//! **Known approximation (P2-C):** the shares below were ported directly from
//! Phase 0's wall-clock-measured percentages, not independently re-derived
//! from iteration-rate data. A simplex iteration does not cost the same
//! wall-clock time in every component — a RINS/RENS/local-branching sub-MIP's
//! LP solves run over a deliberately small, already-restricted neighborhood
//! (cheap per iteration), while an in-tree separation cut LP re-solves the
//! *full* node relaxation from a cold basis each round (expensive per
//! iteration) — so porting a wall-clock share directly to an iteration share
//! carries a bias in an unknown direction per component. This has not been
//! corrected: the Phase 1c/1d re-bench (8/20 MIPLIB-small PASS, zero
//! regressions against the Phase 0 baseline) empirically validates that the
//! current thresholds are functional, so independently re-deriving them from
//! iteration-rate data is deferred rather than spurring an untested change.
//!
//! Shares (Phase 0 MIPLIB-small attribution + iteration-rate comparison
//! against HiGHS on the same instances):
//! - RENS/RINS/local-branching: 0.05 each. Phase 1b pooled these at 0.15
//!   combined behind a single gate; because they are tried in a fixed order
//!   (RENS, then RINS, then local branching) within that shared pool, the
//!   first one due could spend the whole 0.15 before the other two ever got
//!   a turn. Splitting into three independent 0.05 shares against the same
//!   [`total_simplex_iters`] denominator gives each heuristic its own
//!   guaranteed slice regardless of try-order, while 0.05×3=0.15 keeps the
//!   same combined ceiling Phase 1b used. Phase 0's TIMEOUT-problem median
//!   combined heuristic share was 66% of wall time with zero of them
//!   converging in that state; HiGHS solves the same instances at
//!   1,574-6,329 nodes/s, which only leaves room for a low heuristic share.
//! - In-tree GMI/MIR separation: 0.15. Phase 0: `enlight_hard` spent 90% and
//!   `timtab1` 87% of wall time in `separate_tree_cuts`, both TIMEOUT with
//!   no bound progress; the Phase 1c re-bench separately found `dcmulti`
//!   spending 45% (one round's cut LP alone ran 27s) — the per-attempt
//!   iteration cap in `cuts::separate_tree_cuts` (see [`separation_iter_budget`])
//!   addresses that single-call case; this share addresses the aggregate.
//! - Strong branching: 0.10. Phase 0: `p0201` was a lone outlier at 34% of
//!   wall time (suite median 0.06%), so this share only clips that outlier
//!   and is a no-op for essentially every other problem.
//!
//! Combined: 0.05×3 + 0.15 + 0.10 = 0.40, so the node loop itself (LP
//! solves, propagation, bookkeeping) structurally keeps at least 60% of the
//! iteration budget.

use super::MipStats;

/// Share of [`total_simplex_iters`] allowed for RENS.
pub(crate) const RENS_ITER_SHARE: f64 = 0.05;
/// Share of [`total_simplex_iters`] allowed for RINS.
pub(crate) const RINS_ITER_SHARE: f64 = 0.05;
/// Share of [`total_simplex_iters`] allowed for local branching.
pub(crate) const LOCAL_BRANCHING_ITER_SHARE: f64 = 0.05;
/// Share of [`total_simplex_iters`] allowed for in-tree GMI/MIR separation.
pub(crate) const SEPARATION_ITER_SHARE: f64 = 0.15;
/// Share of [`total_simplex_iters`] allowed for strong-branching child solves.
pub(crate) const STRONG_BRANCH_ITER_SHARE: f64 = 0.10;

/// Consecutive in-tree separation attempts (calls that passed the node-
/// selection interval and actually ran at least one round, see
/// `cuts::tree_cut_node_selected`) yielding zero accepted rounds, before
/// separation is disabled for the rest of this solve.
///
/// GMI/MIR separation re-solves the node LP through the same tableau-cut
/// machinery each attempt; once several independent attempts (different
/// nodes, different depths) in a row found nothing worth keeping, the
/// surrounding region is unlikely to have easy further gains and continuing
/// is pure overhead. 5 gives multiple independent attempts (at least 5
/// distinct nodes, since each attempt is itself up to `TREE_CUT_MAX_ROUNDS`
/// rounds against a fresh cut pool) before concluding this, while still
/// bounding the wasted work to a small constant multiple of one attempt's
/// cost.
pub(crate) const SEPARATION_DRY_STREAK_LIMIT: usize = 5;

/// Node-count interval at which `tree_cut_dry_streak` is unconditionally
/// reset to 0, independent of the incumbent-improvement reset (P3-A).
///
/// A dry streak hit early in a long search would otherwise disable
/// separation *permanently* for the rest of that solve, even though the
/// region of the tree being explored (and hence the LP relaxations
/// separation would see) changes substantially over hundreds of nodes.
/// 1000 is an order of magnitude above `TREE_CUT_NODE_INTERVAL` (32, the
/// interval separation itself fires on) and `SEPARATION_DRY_STREAK_LIMIT *
/// TREE_CUT_MAX_ROUNDS` (5*4=20, the rounds a full dry streak burns through),
/// so it gives separation multiple fresh chances over a long search without
/// re-trying a genuinely barren region every few dozen nodes.
pub(crate) const SEPARATION_DRY_STREAK_RESET_NODE_INTERVAL: usize = 1000;

/// Cumulative simplex iterations spent anywhere in this B&B search so far:
/// ordinary node relaxations, strong branching, heuristics (including their
/// sub-MIPs' own recursive total), and in-tree separation. The denominator
/// every `may_run_*` gate below shares.
///
/// Floored at `nodes_processed` (P2-D): a relaxation can legitimately need
/// zero simplex iterations (e.g. an already-optimal starting basis, or a
/// convex-MIQP fixed-point leaf that never calls the LP/QP solver), so
/// `lp_iters_total` alone is not guaranteed to grow every node. Without this
/// floor, a run where iteration counts stay near zero while `nodes_processed`
/// climbs would let the *first* nonzero contribution from any one component
/// (e.g. one strong-branch candidate) make `total_simplex_iters` exactly
/// equal to that component's own count — instantly saturating its share and
/// permanently latching it to blocked, since the shared denominator can
/// never again outgrow a component that IS the entire denominator.
/// `nodes_processed` increments once per processed node unconditionally, so
/// it is a strictly-increasing, deterministic lower bound that keeps the
/// denominator growing with genuine search progress regardless of how many
/// relaxations happen to need zero iterations.
pub(crate) fn total_simplex_iters(stats: &MipStats) -> u64 {
    let raw = stats
        .lp_iters_total
        .saturating_add(stats.strong_branch_iters)
        .saturating_add(stats.rins_iters)
        .saturating_add(stats.rens_iters)
        .saturating_add(stats.local_branching_iters)
        .saturating_add(stats.tree_cut_iters);
    raw.max(stats.nodes_processed as u64)
}

/// `component_iters < share * total_iters`, treating a zero total (nothing
/// solved yet) as always-allow so the very first opportunity is never
/// spuriously blocked by an empty denominator.
fn may_run(component_iters: u64, total_iters: u64, share: f64) -> bool {
    if total_iters == 0 {
        return true;
    }
    (component_iters as f64) < share * (total_iters as f64)
}

/// Whether RENS may run another call, given its cumulative iterations so far.
pub(crate) fn may_run_rens(stats: &MipStats) -> bool {
    may_run(
        stats.rens_iters,
        total_simplex_iters(stats),
        RENS_ITER_SHARE,
    )
}

/// Whether RINS may run another call, given its cumulative iterations so far.
pub(crate) fn may_run_rins(stats: &MipStats) -> bool {
    may_run(
        stats.rins_iters,
        total_simplex_iters(stats),
        RINS_ITER_SHARE,
    )
}

/// Whether local branching may run another call, given its cumulative
/// iterations so far.
pub(crate) fn may_run_local_branching(stats: &MipStats) -> bool {
    may_run(
        stats.local_branching_iters,
        total_simplex_iters(stats),
        LOCAL_BRANCHING_ITER_SHARE,
    )
}

/// Whether in-tree separation may run another round, given its cumulative
/// iterations so far and the dry-streak backoff.
///
/// Also requires [`separation_iter_budget`] `> 0` explicitly (P3-C): that
/// budget is computed by truncating `share * total` to a `u64` before
/// subtracting, while the `may_run` check below compares the untruncated
/// float directly. Floor rounding means the two can disagree right at the
/// boundary (e.g. `component=151`, `share*total=151.05`: the float check
/// passes since `151 < 151.05`, but `floor(151.05) - 151 == 0` — zero
/// integer budget). Without this explicit check, `may_run_separation` could
/// say yes while `cuts::separate_tree_cuts` immediately hits its own
/// `max_iters == 0` cap and does no work — consistent but wasteful.
pub(crate) fn may_run_separation(stats: &MipStats) -> bool {
    stats.tree_cut_dry_streak < SEPARATION_DRY_STREAK_LIMIT
        && may_run(
            stats.tree_cut_iters,
            total_simplex_iters(stats),
            SEPARATION_ITER_SHARE,
        )
        && separation_iter_budget(stats) > 0
}

/// Whether strong branching may evaluate another candidate batch, given its
/// cumulative iterations so far.
pub(crate) fn may_run_strong_branch(stats: &MipStats) -> bool {
    may_run(
        stats.strong_branch_iters,
        total_simplex_iters(stats),
        STRONG_BRANCH_ITER_SHARE,
    )
}

/// Per-call iteration budget remaining for one in-tree separation attempt:
/// how many more iterations separation may spend before its cumulative usage
/// would meet or exceed [`SEPARATION_ITER_SHARE`] of [`total_simplex_iters`].
///
/// `cuts::separate_tree_cuts` has no way to cap a *single* LP solve's
/// iteration count — the underlying `revised_simplex_core` has no iteration
/// limit option, relying solely on the wall-clock deadline (confirmed by
/// reading `otspot-core/src/simplex/primal/core.rs`: `let max_iter =
/// usize::MAX; // timeout is the real guard`) — so this cap is checked only
/// at round boundaries (after a round's LP solves complete, before starting
/// the next). It cannot prevent one abnormally expensive single LP solve
/// within a round (Phase 1c re-bench: `dcmulti` had one such 27s solve) but
/// does bound how many *additional* rounds a single attempt may run once the
/// running total is already large.
///
/// Mirrors [`may_run`]'s zero-total early-allow: `share * 0 == 0` would
/// otherwise floor the very first attempt's budget to 0 (in practice
/// unreachable once any node has been processed, since `total_simplex_iters`
/// is floored at `nodes_processed`, but kept as an explicit contract rather
/// than relying on that floor from a different function).
pub(crate) fn separation_iter_budget(stats: &MipStats) -> u64 {
    let total = total_simplex_iters(stats);
    if total == 0 {
        return u64::MAX;
    }
    let allowed = (SEPARATION_ITER_SHARE * total as f64) as u64;
    allowed.saturating_sub(stats.tree_cut_iters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_with(f: impl FnOnce(&mut MipStats)) -> MipStats {
        let mut s = MipStats::default();
        f(&mut s);
        s
    }

    #[test]
    fn zero_total_iters_always_allows() {
        let stats = MipStats::default();
        assert!(may_run_rens(&stats));
        assert!(may_run_rins(&stats));
        assert!(may_run_local_branching(&stats));
        assert!(may_run_separation(&stats));
        assert!(may_run_strong_branch(&stats));
    }

    #[test]
    fn component_at_or_over_its_share_is_blocked() {
        // `total_simplex_iters` includes the tested component's own count, so
        // `lp_iters_total` is set to `1_000_000 - component` for each case,
        // making `total_simplex_iters == 1_000_000` exactly and `component`
        // land exactly at `share * 1_000_000`.
        let base = |lp_iters_total: u64, extra: u64, setter: fn(&mut MipStats, u64)| {
            stats_with(|s| {
                s.lp_iters_total = lp_iters_total;
                setter(s, extra);
            })
        };
        // 0.05 * 1_000_000 = 50_000 threshold.
        let s = base(950_000, 50_000, |s, v| s.rens_iters = v);
        assert!(!may_run_rens(&s), "rens at exactly its share must block");
        let s = base(950_000, 50_000, |s, v| s.rins_iters = v);
        assert!(!may_run_rins(&s));
        let s = base(950_000, 50_000, |s, v| s.local_branching_iters = v);
        assert!(!may_run_local_branching(&s));
        // 0.15 * 1_000_000 = 150_000 threshold.
        let s = base(850_000, 150_000, |s, v| s.tree_cut_iters = v);
        assert!(!may_run_separation(&s));
        // 0.10 * 1_000_000 = 100_000 threshold.
        let s = base(900_000, 100_000, |s, v| s.strong_branch_iters = v);
        assert!(!may_run_strong_branch(&s));
    }

    #[test]
    fn component_comfortably_under_share_is_allowed() {
        let s = stats_with(|s| {
            s.lp_iters_total = 1_000_000;
            s.rens_iters = 1_000;
            s.rins_iters = 1_000;
            s.local_branching_iters = 1_000;
            s.tree_cut_iters = 1_000;
            s.strong_branch_iters = 1_000;
        });
        assert!(may_run_rens(&s));
        assert!(may_run_rins(&s));
        assert!(may_run_local_branching(&s));
        assert!(may_run_separation(&s));
        assert!(may_run_strong_branch(&s));
    }

    #[test]
    fn separation_dry_streak_blocks_regardless_of_iteration_share() {
        let s = stats_with(|s| {
            s.lp_iters_total = 1_000_000;
            s.tree_cut_iters = 0; // nowhere near its iteration share
            s.tree_cut_dry_streak = SEPARATION_DRY_STREAK_LIMIT;
        });
        assert!(
            !may_run_separation(&s),
            "dry-streak limit must block separation even with ample iteration budget"
        );
    }

    #[test]
    fn separation_iter_budget_is_remaining_share_minus_spent() {
        let s = stats_with(|s| {
            // total_simplex_iters includes tree_cut_iters, so lp_iters_total
            // is set to 1_000_000 - 40_000 to make the total exactly 1_000_000.
            s.lp_iters_total = 960_000;
            s.tree_cut_iters = 40_000;
        });
        // 0.15 * 1_000_000 - 40_000 = 110_000.
        assert_eq!(separation_iter_budget(&s), 110_000);
    }

    /// SENTINEL (P2-D): `nodes_processed` floors `total_simplex_iters` so a
    /// component whose own cumulative iterations are self-referential (i.e.
    /// all other relaxations in the search needed zero simplex iterations,
    /// so the shared denominator would otherwise equal exactly that
    /// component's own count) does not get permanently latched to blocked.
    ///
    /// 100 processed nodes with `lp_iters_total == 0` simulates a run where
    /// every ordinary node relaxation happened to need zero iterations (e.g.
    /// already-optimal starting bases); `strong_branch_iters = 5` is the
    /// *only* nonzero iteration count in the whole `MipStats`. Without the
    /// `nodes_processed` floor, `total_simplex_iters` would equal exactly 5
    /// (self-referential), and `5 < 0.10 * 5 == 0.5` is false — permanently
    /// blocked from the very first bit of work it ever does. With the floor,
    /// `total_simplex_iters == max(5, 100) == 100`, and `5 < 0.10 * 100 ==
    /// 10` is true.
    ///
    /// Sentinel: removing `.max(stats.nodes_processed as u64)` from
    /// `total_simplex_iters` makes this FAIL.
    #[test]
    fn strong_branch_share_not_latched_by_zero_iteration_relaxations() {
        let stats = stats_with(|s| {
            s.lp_iters_total = 0;
            s.nodes_processed = 100;
            s.strong_branch_iters = 5;
        });
        assert!(
            may_run_strong_branch(&stats),
            "nodes_processed floor must keep the gate open when relaxation \
             iteration counts stay at zero across many processed nodes"
        );
    }

    /// SENTINEL (P3-C): `may_run_separation` also requires
    /// `separation_iter_budget(stats) > 0`, not just the raw float
    /// `component < share * total` comparison, so the two never disagree at
    /// the integer-truncation boundary.
    ///
    /// `total = 1007` (`lp_iters_total = 856` + `tree_cut_iters = 151`):
    /// `share * total = 0.15 * 1007 = 151.05`. The float comparison
    /// `151 < 151.05` passes, but `separation_iter_budget` truncates
    /// `151.05` to `151` before subtracting, giving exactly `0` remaining
    /// budget.
    ///
    /// Sentinel: removing the `separation_iter_budget(stats) > 0` conjunct
    /// from `may_run_separation` makes this FAIL.
    #[test]
    fn may_run_separation_blocks_when_integer_budget_is_zero_despite_float_check_passing() {
        let stats = stats_with(|s| {
            s.lp_iters_total = 856;
            s.tree_cut_iters = 151;
        });
        assert_eq!(total_simplex_iters(&stats), 1007, "test premise");
        assert_eq!(
            separation_iter_budget(&stats),
            0,
            "test premise: integer budget must be exactly 0"
        );
        assert!(
            !may_run_separation(&stats),
            "may_run_separation must not allow separation when its integer \
             iteration budget is 0, even though the float share check alone \
             would pass"
        );
    }

    #[test]
    fn shares_leave_majority_for_tree_search() {
        let total = RENS_ITER_SHARE
            + RINS_ITER_SHARE
            + LOCAL_BRANCHING_ITER_SHARE
            + SEPARATION_ITER_SHARE
            + STRONG_BRANCH_ITER_SHARE;
        assert!(
            total <= 0.40,
            "combined optional-work share must leave >= 60% for the tree search; got {total}"
        );
    }
}
