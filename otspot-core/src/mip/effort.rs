//! Deterministic per-solve simplex-iteration-share budget for optional B&B
//! work (Phase 1c; replaces Phase 1b's wall-clock `MipEffortBudget`, which
//! made B&B trajectories timing-dependent — `gt2` spread from 100 to 3000+
//! nodes across repeats instead of a fixed 280-node trajectory). Simplex
//! iterations are reproducible for a fixed input, so gating optional work
//! (heuristics / in-tree separation / strong branching) on a share of
//! [`total_simplex_iters`] keeps the search deterministic; a gate is checked
//! only before a unit of work starts, never mid-solve.
//!
//! **Known gaps:** (P2-B) a sub-MIP's own feasibility-pump iterations are not
//! threaded into `MipStats`, so its recursive `total_simplex_iters`
//! undercounts by that amount (deferred: touches 8+ call sites for a
//! return-type change). (P2-C) the shares below were ported from Phase 0's
//! wall-clock percentages, not independently re-derived from iteration-rate
//! data; per-iteration cost differs by component (a separation cut LP
//! re-solves the full node relaxation from a cold basis, while RINS/RENS/
//! local-branching sub-MIPs search a small restricted neighborhood) — a real
//! bias, confirmed by `mas76` regressing to TIMEOUT under Phase 1d despite
//! its iteration share being respected (follow-up task, not fixed here).
//!
//! Shares: RENS/RINS/local-branching 0.05 each (split from Phase 1b's pooled
//! 0.15 so a fixed try-order can't let one heuristic spend the whole pool);
//! separation 0.15 (Phase 0: `enlight_hard`/`timtab1` spent 87-90% of wall
//! time there with no bound progress); strong branching 0.10 (Phase 0:
//! `p0201` was a 34%-of-wall-time outlier). Combined 0.40 — the node loop
//! keeps >= 60% of the iteration budget structurally.

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
/// Deliberately excludes `stats.tree_cut_overhead_iters`: iteration-
/// *equivalent* bookkeeping (`cuts::tree_cut_construction_surcharge`), not
/// real simplex work, seen only by [`separation_component_iters`] —
/// otherwise every other component's `may_run_*` gate loosens too (`gt2`
/// measured: 100-node `Optimal` → 3,744-node `Timeout`).
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
            separation_component_iters(stats),
            total_simplex_iters(stats),
            SEPARATION_ITER_SHARE,
        )
        && separation_iter_budget(stats) > 0
}

/// Separation's own numerator for [`may_run_separation`] and
/// [`separation_iter_budget`]: real simplex iterations
/// (`stats.tree_cut_iters`) plus `cuts::tree_cut_construction_surcharge`'s
/// fixed-cost overhead (`stats.tree_cut_overhead_iters`) — see
/// [`MipStats::tree_cut_overhead_iters`](super::MipStats::
/// tree_cut_overhead_iters) for why the overhead is added *only* here and
/// never folded into [`total_simplex_iters`], the shared denominator every
/// other `may_run_*` gate also reads.
fn separation_component_iters(stats: &MipStats) -> u64 {
    stats
        .tree_cut_iters
        .saturating_add(stats.tree_cut_overhead_iters)
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

/// `floor(share * total_iters) - component_iters`, i.e. the remaining
/// iterations `component_iters` may still spend before reaching `share` of
/// `total_iters`. `u64::MAX` when `total_iters == 0` (nothing solved yet, so
/// the very first opportunity is never budget-starved by an empty
/// denominator) — mirrors [`may_run`]'s zero-total early-allow.
fn iter_budget_remaining(component_iters: u64, total_iters: u64, share: f64) -> u64 {
    if total_iters == 0 {
        return u64::MAX;
    }
    let allowed = (share * total_iters as f64) as u64;
    allowed.saturating_sub(component_iters)
}

/// Per-call iteration budget remaining for one in-tree separation attempt:
/// how many more iterations separation may spend before its cumulative usage
/// would meet or exceed [`SEPARATION_ITER_SHARE`] of [`total_simplex_iters`].
///
/// Starting a new round in `cuts::separate_tree_cuts` requires this to be at
/// least the per-dimension useful minimum (`cuts::tree_cut_min_useful_iters`)
/// — below that, the round is skipped rather than attempted with a
/// truncated `max_iters` (markshare_4_0 regression fix). Historically
/// (Phase 1c) this budget was checked only at round boundaries, and the
/// underlying simplex core had no iteration-limit option of its own, relying
/// solely on the wall-clock deadline, so one abnormally expensive single LP
/// solve within an already-started round could not be capped (re-bench:
/// `dcmulti` had one such 27s solve). `SolverOptions::max_iters` closed that
/// gap (Codex review, P1) — `cuts::separate_tree_cuts` now passes each
/// individual cold solve its own remaining allowance as its `max_iters`, so
/// a single solve's cost is bounded by exactly this budget rather than only
/// by the wall clock.
pub(crate) fn separation_iter_budget(stats: &MipStats) -> u64 {
    iter_budget_remaining(
        separation_component_iters(stats),
        total_simplex_iters(stats),
        SEPARATION_ITER_SHARE,
    )
}

/// Like [`iter_budget_remaining`], but the share ceiling (`share *
/// total_iters`, before subtracting `component_iters`) is floored at
/// [`heuristics::SUB_MIP_MAX_LP_ITERS`](super::heuristics::SUB_MIP_MAX_LP_ITERS).
///
/// `SUB_MIP_MAX_LP_ITERS` was calibrated from a *single* call's own
/// recursive iteration need (see its doc: `khb05250`/`gt2` needed
/// 595,689/386,937) — a different scale from `share * total_simplex_iters`,
/// which never approaches `SUB_MIP_MAX_LP_ITERS / RINS_ITER_SHARE` (≈9.6M)
/// for a search this size. The raw (unfloored) share as a hard per-call cap
/// therefore starves every call — confirmed by regression: `gt2 --timeout
/// 60` went from a deterministic 200-node `Optimal` to a non-deterministic
/// ~2500-node `Timeout`.
///
/// Flooring the ceiling at `SUB_MIP_MAX_LP_ITERS` preserves the old flat
/// budget for a heuristic still below that constant; it is a no-op once
/// `share * total_simplex_iters` exceeds it, where `heuristics::capped_
/// sub_mip_max_lp_iters` still caps every call at `min(SUB_MIP_MAX_LP_
/// ITERS, this)` and skips it below `heuristics::SUB_MIP_MIN_LP_ITERS`.
///
/// Early on, `total_simplex_iters` is small enough that `share * total`
/// alone would starve every call — the floor's early overshoot is what made
/// a call effective at all (Phase 1b-1d, `0eb370f5`: MIPLIB small 5→7 PASS,
/// `markshare_4_0` among the two newly passing). That overshoot is bounded
/// per call at `SUB_MIP_MAX_LP_ITERS` and self-corrects once `share *
/// total_simplex_iters` exceeds it.
fn sub_mip_iter_budget_remaining(component_iters: u64, total_iters: u64, share: f64) -> u64 {
    if total_iters == 0 {
        return u64::MAX;
    }
    let allowed =
        ((share * total_iters as f64) as u64).max(super::heuristics::SUB_MIP_MAX_LP_ITERS);
    allowed.saturating_sub(component_iters)
}

/// Per-call iteration budget remaining for one RINS sub-MIP call: how many
/// more iterations RINS may spend before its cumulative usage would meet or
/// exceed [`RINS_ITER_SHARE`] of [`total_simplex_iters`] (floored at
/// `heuristics::SUB_MIP_MAX_LP_ITERS` — see [`sub_mip_iter_budget_remaining`]).
///
/// Codex review (P1): `may_run_rins` alone only *approves* a call — it does
/// not cap the size of the approved work. Without this, an approved call's
/// sub-MIP `max_lp_iters` was the flat `heuristics::SUB_MIP_MAX_LP_ITERS`
/// constant regardless of how little of RINS's own share was actually left,
/// letting a single "approved" sub-MIP call overshoot its share by up to
/// that entire constant in one call. See
/// `heuristics::capped_sub_mip_max_lp_iters`.
pub(crate) fn rins_iter_budget(stats: &MipStats) -> u64 {
    sub_mip_iter_budget_remaining(
        stats.rins_iters,
        total_simplex_iters(stats),
        RINS_ITER_SHARE,
    )
}

/// Per-call iteration budget remaining for one RENS sub-MIP call. See
/// [`rins_iter_budget`] for the rationale.
pub(crate) fn rens_iter_budget(stats: &MipStats) -> u64 {
    sub_mip_iter_budget_remaining(
        stats.rens_iters,
        total_simplex_iters(stats),
        RENS_ITER_SHARE,
    )
}

/// Per-call iteration budget remaining for one local-branching sub-MIP call.
/// See [`rins_iter_budget`] for the rationale.
pub(crate) fn local_branching_iter_budget(stats: &MipStats) -> u64 {
    sub_mip_iter_budget_remaining(
        stats.local_branching_iters,
        total_simplex_iters(stats),
        LOCAL_BRANCHING_ITER_SHARE,
    )
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

    /// **SENTINEL** (Phase 3b): [`tree_cut_overhead_iters`](super::MipStats::
    /// tree_cut_overhead_iters) — `cuts::tree_cut_construction_surcharge`'s
    /// fixed-cost accounting — must count against separation's *own* gate,
    /// both [`may_run_separation`] and [`separation_iter_budget`], exactly
    /// like real `tree_cut_iters`.
    ///
    /// Sentinel: computing `separation_component_iters` as `stats.
    /// tree_cut_iters` alone (dropping the `+ tree_cut_overhead_iters` term)
    /// makes this FAIL — a search that spent nothing on real separation
    /// iterations but has already accrued a large surcharge would otherwise
    /// still read as comfortably under its share.
    #[test]
    fn separation_overhead_iters_count_against_the_separation_gate() {
        let s = stats_with(|s| {
            s.lp_iters_total = 1_000_000;
            s.tree_cut_iters = 0;
            s.tree_cut_overhead_iters = 150_000; // >= 0.15 * 1_000_000
        });
        assert!(
            !may_run_separation(&s),
            "a surcharge alone reaching the share ceiling must block separation"
        );
        assert_eq!(
            separation_iter_budget(&s),
            0,
            "a surcharge alone reaching the share ceiling must zero the remaining budget"
        );
    }

    /// **SENTINEL** (Phase 3b): [`total_simplex_iters`] — the shared
    /// denominator every `may_run_*` gate (RENS/RINS/local-branching/
    /// strong-branching, not just separation) reads — must be independent of
    /// [`tree_cut_overhead_iters`](super::MipStats::tree_cut_overhead_iters).
    /// A construction-cost surcharge is iteration-*equivalent* bookkeeping
    /// for separation's own gate, not real simplex work; leaking it into the
    /// shared total loosens every other component's gate too.
    ///
    /// Sentinel: folding `stats.tree_cut_overhead_iters` into
    /// `total_simplex_iters`'s sum makes this FAIL — this reproduces the
    /// measured regression directly (an earlier version of this surcharge
    /// took `gt2 --timeout 60` from a deterministic 100-node `Optimal` to a
    /// 3,744-node `Timeout` this way, by loosening RINS/RENS/local-branching/
    /// strong-branching gates that have nothing to do with separation).
    #[test]
    fn total_simplex_iters_is_independent_of_separation_overhead() {
        let without_overhead = stats_with(|s| {
            s.lp_iters_total = 1_000;
        });
        let with_overhead = stats_with(|s| {
            s.lp_iters_total = 1_000;
            s.tree_cut_overhead_iters = 1_000_000_000;
        });
        assert_eq!(
            total_simplex_iters(&without_overhead),
            total_simplex_iters(&with_overhead),
            "tree_cut_overhead_iters must never change total_simplex_iters"
        );

        // Corroborate at the gate level: an unrelated component (RINS) sees
        // the identical share ceiling whether or not a huge surcharge has
        // accrued.
        let rins_budget_without = rins_iter_budget(&without_overhead);
        let rins_budget_with = rins_iter_budget(&with_overhead);
        assert_eq!(
            rins_budget_without, rins_budget_with,
            "an unrelated component's own budget must not shift because of \
             separation's surcharge"
        );
    }

    /// SENTINEL (Codex review, P1 fix follow-up): on a small/medium-scale
    /// search, `share * total_simplex_iters` stays far below
    /// `heuristics::SUB_MIP_MAX_LP_ITERS` (480_000) — that constant was
    /// calibrated from a *single* call's own recursive iteration need on
    /// realistic instances, not from 5% of the *parent* search's cumulative
    /// count. `rins_iter_budget`/`rens_iter_budget`/`local_branching_iter_
    /// budget` must floor their ceiling at that constant, or every call is
    /// starved for the whole search on exactly this scale (confirmed by
    /// direct regression: `gt2 --timeout 60` went from a deterministic
    /// 200-node `Optimal` to a non-deterministic ~2500-node `Timeout` when
    /// capped to the raw, unfloored share).
    ///
    /// Sentinel: removing `sub_mip_iter_budget_remaining`'s `.max(super::
    /// heuristics::SUB_MIP_MAX_LP_ITERS)` floor makes each of these return
    /// `40_000` (the raw `0.05 * 1_000_000 - 10_000` share) instead of
    /// `470_000` (`480_000 - 10_000`), failing the assertions.
    #[test]
    fn heuristic_iter_budgets_floor_their_ceiling_at_sub_mip_max_lp_iters() {
        let s = stats_with(|s| {
            s.lp_iters_total = 990_000;
            s.rins_iters = 10_000;
        });
        assert_eq!(rins_iter_budget(&s), 470_000);

        let s = stats_with(|s| {
            s.lp_iters_total = 990_000;
            s.rens_iters = 10_000;
        });
        assert_eq!(rens_iter_budget(&s), 470_000);

        let s = stats_with(|s| {
            s.lp_iters_total = 990_000;
            s.local_branching_iters = 10_000;
        });
        assert_eq!(local_branching_iter_budget(&s), 470_000);
    }

    /// Once `share * total_simplex_iters` genuinely exceeds the
    /// `SUB_MIP_MAX_LP_ITERS` floor (a large/long search), the floor is a
    /// no-op and the raw share governs — the original per-call cap this
    /// fix is meant to provide for exactly that regime.
    #[test]
    fn heuristic_iter_budget_uses_raw_share_once_it_exceeds_the_floor() {
        // total = 20_000_000; 0.05 * 20_000_000 = 1_000_000 > 480_000 floor.
        let s = stats_with(|s| {
            s.lp_iters_total = 19_900_000;
            s.rins_iters = 100_000;
        });
        assert_eq!(rins_iter_budget(&s), 900_000);
    }

    /// SENTINEL: a component already at or beyond its (floored) ceiling has
    /// 0 remaining budget, not a negative/wrapped value —
    /// `sub_mip_iter_budget_remaining` saturates rather than underflowing.
    ///
    /// Sentinel: replacing the `saturating_sub` in
    /// `sub_mip_iter_budget_remaining` with a plain `-` panics (debug) or
    /// wraps to a huge `u64` (release) instead of returning `0`, failing
    /// this assertion either way.
    #[test]
    fn heuristic_iter_budget_saturates_at_zero_when_over_the_floored_ceiling() {
        // total = 20_000_000, share ceiling = 0.05 * 20_000_000 = 1_000_000
        // (> the 480_000 floor, so the floor is a no-op here); rins_iters
        // (1_000_000) is already at that ceiling.
        let s = stats_with(|s| {
            s.lp_iters_total = 19_000_000;
            s.rins_iters = 1_000_000;
        });
        assert_eq!(rins_iter_budget(&s), 0);
    }

    /// SENTINEL: `total_iters == 0` (nothing solved yet) reports `u64::MAX`
    /// (truly unbounded), not merely `SUB_MIP_MAX_LP_ITERS` from the floor —
    /// mirrors `zero_total_iters_always_allows`'s `may_run_*` coverage.
    ///
    /// Sentinel: removing the `total_iters == 0` early-return from
    /// `sub_mip_iter_budget_remaining` makes each of these return
    /// `SUB_MIP_MAX_LP_ITERS` (`0.max(SUB_MIP_MAX_LP_ITERS)`, unaffected by
    /// `component_iters == 0`) instead of `u64::MAX`, failing this assertion.
    #[test]
    fn heuristic_iter_budgets_are_unbounded_at_zero_total_iters() {
        let stats = MipStats::default();
        assert_eq!(rins_iter_budget(&stats), u64::MAX);
        assert_eq!(rens_iter_budget(&stats), u64::MAX);
        assert_eq!(local_branching_iter_budget(&stats), u64::MAX);
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
