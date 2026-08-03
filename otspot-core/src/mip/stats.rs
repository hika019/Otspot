//! Branch-and-bound search statistics.
//!
//! [`MipStats`] instruments the MILP/MIQP driver without changing it: every
//! field is written by the search and read by callers, never the other way
//! round. It also carries the reduction the parallel driver needs
//! ([`MipStats::merge_worker`]), which is what keeps "one search's numbers"
//! meaningful when several workers produced them.
//!
//! # Fields that stop being trustworthy under `threads >= 2`
//! `lp_scale_us_root` / `lp_scale_us_desc` / `lp_scale_calls_*` and the
//! `fallback_*` counters are computed as before/after deltas of *process-wide*
//! atomics (`presolve::scaling`, `simplex::dual_advanced`). With one worker
//! the delta around a node solve is that node's own cost; with several,
//! concurrent workers bump the same atomics inside each other's measurement
//! windows, so the per-node attribution is cross-contaminated and only the
//! search-wide totals stay meaningful. This affects measurement only — no
//! `may_run_*` gate reads these fields (they gate on simplex iteration
//! counts, which are per-`SolverResult` and therefore worker-local). Profile
//! MILP scaling/fallback costs with `threads = 1`.

/// Search statistics returned by [`super::solve_milp_with_stats`] /
/// [`super::solve_miqp_with_stats`].
///
/// Counters and timings instrument the branch-and-bound driver without changing
/// its behaviour.  The timing fields help separate *exploration explosion* (many
/// nodes) from *per-node cost* (slow relaxation solves).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct MipStats {
    /// Relaxation solves performed (root included).
    pub nodes_processed: usize,
    /// Number of integer-variable bound fixings applied by reduced-cost fixing.
    pub rc_vars_fixed: usize,
    /// Maximum branching depth reached.
    pub max_depth_seen: usize,
    /// Nodes discarded by bound/infeasibility before branching.
    pub pruned: usize,
    /// Nodes pruned by bound propagation before the LP/QP solve.
    pub propagation_pruned: usize,
    /// Wall-clock microseconds spent in per-node bound propagation.
    pub node_propagation_us: u64,
    /// Number of incumbent improvements (including the first one found).
    pub incumbent_updates: usize,

    // --- relaxation solve wall-clock timing (milliseconds) ---
    /// Total wall time spent inside relaxation solves across all nodes (ms).
    pub relaxation_time_total_ms: f64,
    /// Wall time for the root node relaxation solve (ms).
    pub relaxation_time_root_ms: f64,
    /// Cumulative wall time for all descendant (non-root) relaxation solves (ms).
    pub relaxation_time_desc_ms: f64,
    /// Cumulative time in solves that returned `Optimal` (ms).
    pub relaxation_time_optimal_ms: f64,
    /// Cumulative time in solves that returned `Infeasible` (ms).
    pub relaxation_time_infeasible_ms: f64,

    /// Cumulative LP presolve microseconds across all nodes (zero when presolve does not reduce).
    pub lp_presolve_us_total: u64,
    /// Cumulative LP solve (simplex) microseconds across all nodes.
    pub lp_solve_us_total: u64,
    /// LP solve microseconds in the root node.
    pub lp_solve_us_root: u64,
    /// Cumulative LP solve microseconds in descendant nodes.
    pub lp_solve_us_desc: u64,
    /// Cumulative LP postsolve microseconds across all nodes.
    pub lp_postsolve_us_total: u64,
    /// Cumulative Ruiz scaling microseconds in root node LP solve.
    pub lp_scale_us_root: u64,
    /// Cumulative Ruiz scaling microseconds in descendant node LP solves.
    pub lp_scale_us_desc: u64,
    /// Number of Ruiz scaling calls in root node LP solve.
    pub lp_scale_calls_root: u64,
    /// Number of Ruiz scaling calls in descendant node LP solves.
    pub lp_scale_calls_desc: u64,
    /// Number of branch-variable selections that invoked strong branching.
    pub strong_branch_calls: usize,
    /// Total candidate variables evaluated by strong branching.
    pub strong_branch_candidates: usize,
    /// Total child relaxation solves launched by strong branching.
    pub strong_branch_lp_solves: usize,
    /// Wall-clock microseconds spent in strong-branching child solves.
    pub strong_branch_us: u64,
    /// Bounded dual fallback count: terminal UB violation outside current repair scope.
    pub fallback_ub_violation_out_of_scope: u64,
    /// Bounded artificial Phase I fallback count: reconciled bound violation.
    pub fallback_phase1_bound_violation: u64,
    /// Eq+UB crash-basis fallback count: crash produced bounded-infeasible start.
    pub fallback_crash_infeasible: u64,

    /// Approximate bytes per node for the bounds clone: `n_vars × 2 × size_of::<f64>()`.
    /// Gives a rough idea of per-node memory traffic regardless of node count.
    pub approx_bounds_bytes_per_node: usize,

    /// Whether the feasibility pump found an initial incumbent before branch-and-bound.
    pub fp_incumbent_found: bool,
    /// Wall-clock microseconds spent in the pre-B&B feasibility pump.
    pub fp_us: u64,
    /// Wall-clock microseconds spent adding root cuts before branch-and-bound.
    pub root_cut_us: u64,
    /// Wall-clock microseconds spent in root bound-tightening probing
    /// (`presolve::tighten_bounds_with_probing`) before branch-and-bound.
    pub root_probing_us: u64,
    /// Wall-clock microseconds spent in root static symmetry breaking
    /// (`symmetry::break_symmetry`) before branch-and-bound.
    pub root_symmetry_us: u64,

    /// Objective of the first trusted (Optimal) root relaxation, i.e. the root LP
    /// bound used to start branch-and-bound. With cuts enabled this reflects the
    /// cut-tightened relaxation, so comparing it against the cuts-off value
    /// isolates root gap closure from downstream node-count noise.
    /// `NEG_INFINITY` when no root relaxation solved to Optimal.
    pub root_lp_bound: f64,

    /// Number of RINS heuristic calls attempted.
    pub rins_calls: usize,
    /// Number of times RINS found an improving incumbent.
    pub rins_improvements: usize,
    /// Number of RENS heuristic calls attempted.
    pub rens_calls: usize,
    /// Number of times RENS found an improving incumbent.
    pub rens_improvements: usize,
    /// Number of local-branching heuristic calls attempted.
    pub local_branching_calls: usize,
    /// Number of times local branching found an improving incumbent.
    pub local_branching_improvements: usize,

    /// Number of conflict clauses learned from infeasible nodes.
    pub conflict_clauses_learned: usize,
    /// Number of nodes pruned by conflict analysis (LP solve skipped).
    pub conflict_pruned: usize,

    /// Number of B&B nodes where in-tree separation produced a cut-tightened
    /// (accepted) relaxation result. Zero when `tree_cuts` is off or no cut
    /// improved a node bound.
    pub tree_cut_rounds: usize,

    // --- B&B time attribution (wall-clock microseconds) ---
    // These, together with `lp_solve_us_total` and `node_propagation_us` above,
    // are meant to add up to (most of) the search wall clock so that "many
    // nodes" vs. "expensive per-node heuristics/separation" can be told apart
    // without re-instrumenting. Measurement only: none of these fields change
    // solver behaviour.
    /// Wall-clock microseconds spent in in-tree cut separation (`separate_tree_cuts`).
    pub tree_cut_us: u64,
    /// Wall-clock microseconds spent in the RINS heuristic, including its sub-MIP solve.
    pub rins_us: u64,
    /// Wall-clock microseconds spent in the RENS heuristic, including its sub-MIP solve.
    pub rens_us: u64,
    /// Wall-clock microseconds spent in the local-branching heuristic, including its sub-MIP solve.
    pub local_branching_us: u64,
    /// Wall-clock microseconds spent selecting the branching variable
    /// (`pick_branch_var`), including any strong-branching child solves
    /// (see `strong_branch_us` for that narrower subset).
    pub branch_select_us: u64,
    /// Wall-clock microseconds spent checking/learning conflict clauses.
    pub conflict_us: u64,
    /// Wall-clock microseconds of B&B loop overhead not attributed to any
    /// other named bucket (pruning checks, dive bookkeeping, queue
    /// operations, node cloning). Computed per node as the residual of that
    /// node's loop-iteration wall time after subtracting every other
    /// explicitly measured bucket touched during the same iteration.
    pub node_loop_other_us: u64,
    /// Cumulative `nodes_processed` reported by RINS/RENS/local-branching
    /// sub-MIP solves: branch-and-bound work that is invisible in the outer
    /// `nodes_processed` count.
    pub sub_mip_nodes_total: u64,

    // --- B&B time attribution (deterministic simplex-iteration counters) ---
    // Phase 1c: `mip::effort`'s `may_run_*` gates are computed from these,
    // not from the `*_us` wall-clock counters above (which stay as pure
    // measurement — see their doc comments). Unlike wall time, simplex
    // iteration counts are reproducible for a fixed input and algorithm
    // path, so gating on them keeps the B&B search itself deterministic.
    /// Cumulative simplex iterations across all node relaxation solves
    /// (root + descendants) in the main B&B loop.
    pub lp_iters_total: u64,
    /// Cumulative simplex iterations spent in strong-branching child solves.
    pub strong_branch_iters: u64,
    /// Cumulative simplex iterations spent in the RINS heuristic, including
    /// its sub-MIP's own recursive total.
    pub rins_iters: u64,
    /// Cumulative simplex iterations spent in the RENS heuristic, including
    /// its sub-MIP's own recursive total.
    pub rens_iters: u64,
    /// Cumulative simplex iterations spent in the local-branching heuristic,
    /// including its sub-MIP's own recursive total.
    pub local_branching_iters: u64,
    /// Cumulative simplex iterations spent in in-tree cut separation
    /// (`separate_tree_cuts`), across all rounds of all attempts.
    pub tree_cut_iters: u64,
    /// Cumulative fixed-cost surcharge (iteration-equivalent units, not real
    /// simplex iterations) `separate_tree_cuts` charges for its own
    /// `build_standard_form`-equivalent construction overhead — see
    /// `cuts::tree_cut_construction_surcharge`. Kept out of `tree_cut_iters`
    /// deliberately: only `effort::may_run_separation` and `effort::
    /// separation_iter_budget` add this to their numerator, so separation's
    /// own gate feels its true per-round cost without inflating
    /// `effort::total_simplex_iters` — the shared denominator every other
    /// `may_run_*` gate (RINS/RENS/local-branching/strong-branching) also
    /// reads. An earlier version charged straight into `tree_cut_iters`,
    /// which inflated that shared total and measurably distorted unrelated
    /// gates: `gt2 --timeout 60` regressed from a deterministic 100-node
    /// `Optimal` to a 3,744-node `Timeout` purely from this cross-component
    /// leak, on a run where separation's own round count *increased*
    /// (14 → 183) rather than decreased.
    pub tree_cut_overhead_iters: u64,
    /// Consecutive in-tree separation attempts (that actually ran at least
    /// one round) yielding zero accepted rounds. Reset to 0 on any accepted
    /// round; once it reaches `effort::SEPARATION_DRY_STREAK_LIMIT`,
    /// `effort::may_run_separation` disables separation for the rest of
    /// this solve.
    pub tree_cut_dry_streak: usize,
}

impl MipStats {
    /// Seed statistics for one parallel B&B worker. Differs from
    /// `Default::default()` only in `root_lp_bound`, whose "unset" value is
    /// `-inf` rather than `0.0` — merging a worker that never touched the
    /// root must not pull a negative root bound up to zero.
    ///
    /// # What "per worker" changes about the counters that gate work
    /// Every `nodes_processed`-keyed interval (RINS / RENS / local-branching /
    /// tree-cut node intervals, `effort::SEPARATION_DRY_STREAK_RESET_NODE_
    /// INTERVAL`) counts this worker's own nodes, and every `effort::may_run_*`
    /// share is measured against its own iteration total. Deliberate: a share
    /// enforced per worker is the same share globally, whereas `threads`
    /// workers racing one shared interval counter would fire it at
    /// unpredictable multiples of the intended spacing.
    ///
    /// The knock-on effects are accepted, not accidental — anything that fired
    /// "once per search" now fires once per worker.
    /// `MipState::rens_first_incumbent_attempted` grants its one guaranteed
    /// pre-incumbent RENS attempt to each worker, and `tree_cut_dry_streak` is
    /// counted and reset per worker so separation backs off independently on
    /// each. Both still bound total effort the same way; neither is a
    /// per-search guarantee any more.
    pub(crate) fn worker_seed() -> Self {
        Self {
            root_lp_bound: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    /// Fold one parallel worker's statistics into the search-wide totals.
    ///
    /// The exhaustive destructure below is deliberate: a new `MipStats` field
    /// fails to compile here until its reduction (sum / max / or) is stated,
    /// so a counter can never be silently dropped from parallel runs.
    pub(crate) fn merge_worker(&mut self, other: &MipStats) {
        let MipStats {
            nodes_processed,
            rc_vars_fixed,
            max_depth_seen,
            pruned,
            propagation_pruned,
            node_propagation_us,
            incumbent_updates,
            relaxation_time_total_ms,
            relaxation_time_root_ms,
            relaxation_time_desc_ms,
            relaxation_time_optimal_ms,
            relaxation_time_infeasible_ms,
            lp_presolve_us_total,
            lp_solve_us_total,
            lp_solve_us_root,
            lp_solve_us_desc,
            lp_postsolve_us_total,
            lp_scale_us_root,
            lp_scale_us_desc,
            lp_scale_calls_root,
            lp_scale_calls_desc,
            strong_branch_calls,
            strong_branch_candidates,
            strong_branch_lp_solves,
            strong_branch_us,
            fallback_ub_violation_out_of_scope,
            fallback_phase1_bound_violation,
            fallback_crash_infeasible,
            approx_bounds_bytes_per_node,
            fp_incumbent_found,
            fp_us,
            root_cut_us,
            root_probing_us,
            root_symmetry_us,
            root_lp_bound,
            rins_calls,
            rins_improvements,
            rens_calls,
            rens_improvements,
            local_branching_calls,
            local_branching_improvements,
            conflict_clauses_learned,
            conflict_pruned,
            tree_cut_rounds,
            tree_cut_us,
            rins_us,
            rens_us,
            local_branching_us,
            branch_select_us,
            conflict_us,
            node_loop_other_us,
            sub_mip_nodes_total,
            lp_iters_total,
            strong_branch_iters,
            rins_iters,
            rens_iters,
            local_branching_iters,
            tree_cut_iters,
            tree_cut_overhead_iters,
            tree_cut_dry_streak,
        } = *other;

        // Counters and wall/iteration budgets: additive across workers.
        self.nodes_processed += nodes_processed;
        self.rc_vars_fixed += rc_vars_fixed;
        self.pruned += pruned;
        self.propagation_pruned += propagation_pruned;
        self.node_propagation_us = self.node_propagation_us.saturating_add(node_propagation_us);
        self.incumbent_updates += incumbent_updates;
        self.relaxation_time_total_ms += relaxation_time_total_ms;
        self.relaxation_time_desc_ms += relaxation_time_desc_ms;
        self.relaxation_time_optimal_ms += relaxation_time_optimal_ms;
        self.relaxation_time_infeasible_ms += relaxation_time_infeasible_ms;
        self.lp_presolve_us_total = self
            .lp_presolve_us_total
            .saturating_add(lp_presolve_us_total);
        self.lp_solve_us_total = self.lp_solve_us_total.saturating_add(lp_solve_us_total);
        self.lp_solve_us_desc = self.lp_solve_us_desc.saturating_add(lp_solve_us_desc);
        self.lp_postsolve_us_total = self
            .lp_postsolve_us_total
            .saturating_add(lp_postsolve_us_total);
        self.lp_scale_us_desc = self.lp_scale_us_desc.saturating_add(lp_scale_us_desc);
        self.lp_scale_calls_desc = self.lp_scale_calls_desc.saturating_add(lp_scale_calls_desc);
        self.strong_branch_calls += strong_branch_calls;
        self.strong_branch_candidates += strong_branch_candidates;
        self.strong_branch_lp_solves += strong_branch_lp_solves;
        self.strong_branch_us = self.strong_branch_us.saturating_add(strong_branch_us);
        self.fallback_ub_violation_out_of_scope = self
            .fallback_ub_violation_out_of_scope
            .saturating_add(fallback_ub_violation_out_of_scope);
        self.fallback_phase1_bound_violation = self
            .fallback_phase1_bound_violation
            .saturating_add(fallback_phase1_bound_violation);
        self.fallback_crash_infeasible = self
            .fallback_crash_infeasible
            .saturating_add(fallback_crash_infeasible);
        self.rins_calls += rins_calls;
        self.rins_improvements += rins_improvements;
        self.rens_calls += rens_calls;
        self.rens_improvements += rens_improvements;
        self.local_branching_calls += local_branching_calls;
        self.local_branching_improvements += local_branching_improvements;
        self.conflict_pruned += conflict_pruned;
        self.tree_cut_rounds += tree_cut_rounds;
        self.tree_cut_us = self.tree_cut_us.saturating_add(tree_cut_us);
        self.rins_us = self.rins_us.saturating_add(rins_us);
        self.rens_us = self.rens_us.saturating_add(rens_us);
        self.local_branching_us = self.local_branching_us.saturating_add(local_branching_us);
        self.branch_select_us = self.branch_select_us.saturating_add(branch_select_us);
        self.conflict_us = self.conflict_us.saturating_add(conflict_us);
        self.node_loop_other_us = self.node_loop_other_us.saturating_add(node_loop_other_us);
        self.sub_mip_nodes_total = self.sub_mip_nodes_total.saturating_add(sub_mip_nodes_total);
        self.lp_iters_total = self.lp_iters_total.saturating_add(lp_iters_total);
        self.strong_branch_iters = self.strong_branch_iters.saturating_add(strong_branch_iters);
        self.rins_iters = self.rins_iters.saturating_add(rins_iters);
        self.rens_iters = self.rens_iters.saturating_add(rens_iters);
        self.local_branching_iters = self
            .local_branching_iters
            .saturating_add(local_branching_iters);
        self.tree_cut_iters = self.tree_cut_iters.saturating_add(tree_cut_iters);
        self.tree_cut_overhead_iters = self
            .tree_cut_overhead_iters
            .saturating_add(tree_cut_overhead_iters);

        // Root-node buckets: exactly one worker processes the (unique, depth-0)
        // root, so every other worker contributes an identity element here.
        self.relaxation_time_root_ms += relaxation_time_root_ms;
        self.lp_solve_us_root = self.lp_solve_us_root.saturating_add(lp_solve_us_root);
        self.lp_scale_us_root = self.lp_scale_us_root.saturating_add(lp_scale_us_root);
        self.lp_scale_calls_root = self.lp_scale_calls_root.saturating_add(lp_scale_calls_root);
        self.root_lp_bound = self.root_lp_bound.max(root_lp_bound);

        // Pre-B&B phases: recorded by the driver, never by a worker.
        self.fp_us = self.fp_us.saturating_add(fp_us);
        self.root_cut_us = self.root_cut_us.saturating_add(root_cut_us);
        self.root_probing_us = self.root_probing_us.saturating_add(root_probing_us);
        self.root_symmetry_us = self.root_symmetry_us.saturating_add(root_symmetry_us);
        self.fp_incumbent_found |= fp_incumbent_found;

        // Per-worker extremes / per-worker constants.
        self.max_depth_seen = self.max_depth_seen.max(max_depth_seen);
        self.approx_bounds_bytes_per_node = self
            .approx_bounds_bytes_per_node
            .max(approx_bounds_bytes_per_node);
        self.conflict_clauses_learned = self.conflict_clauses_learned.max(conflict_clauses_learned);
        self.tree_cut_dry_streak = self.tree_cut_dry_streak.max(tree_cut_dry_streak);
    }
}
