//! Mixed-integer programming (MILP / MIQP) via branch-and-bound.
//!
//! MILP (LP relaxation) and convex MIQP (QP relaxation) share one generic driver
//! (`solve_mip_with_stats`); the per-node solver is abstracted by `Relaxation`.
//! Pruning (`qp::global::pruning`) is reused from the spatial QP B&B. Non-convex
//! MIQP is out of scope and reported as [`SolveStatus::NonConvex`].

pub(crate) mod branch;
pub(crate) mod conflict;
pub(crate) mod cut_pool;
pub(crate) mod cuts;
pub(crate) mod heuristics;
pub(crate) mod node;
pub(crate) mod presolve;
mod problem;
pub(crate) mod queue;
pub(crate) mod symmetry;

pub use problem::{MilpProblem, MipProblemError, MiqpProblem};

use crate::linalg::timeout::deadline_reached;
use crate::options::{MipBranching, MipConfig, SolverOptions, WarmStartBasis};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{ConstraintType, SolveStatus, SolverResult, TimingBreakdown};
use crate::qp::global::pruning::{should_prune, within_gap};
use crate::sparse::CscMatrix;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use branch::{
    branch_bounds, is_integer_feasible, select_branching_variable,
    select_branching_variable_reliability, split_integer_box, strong_branch_candidates,
    widest_splittable_integer, PseudocostState,
};
use node::MipNode;
use queue::{NodeQueue, DIVE_FREQUENCY, DIVE_FREQUENCY_NO_INCUMBENT, MAX_DIVE_DEPTH};

/// A continuous relaxation the MIP branch-and-bound driver can solve over
/// arbitrary variable bounds. MILP uses an LP relaxation; convex MIQP uses a
/// QP one. Branching tightens the bounds, so the same driver works for both —
/// only the relaxation solver differs.
pub(crate) trait Relaxation {
    fn num_vars(&self) -> usize;
    fn root_bounds(&self) -> &[(f64, f64)];
    fn integer_vars(&self) -> &[usize];
    /// Solve the relaxation with `bounds` substituted for the original bounds.
    /// `opts` already has multistart / global_optimization stripped and the
    /// deadline fixed by the driver.
    fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult;
    /// Whether the driver should disable presolve on every B&B *node* solve.
    /// True for MILP: each node re-solves the same LP with only bounds tightened,
    /// so per-node presolve is redundant and its variable renumbering drops the
    /// propagated warm-start basis. False (default) for MIQP, whose IPM relies on
    /// presolve's Ruiz scaling for per-node conditioning.
    fn skip_node_presolve(&self) -> bool {
        false
    }
    /// Whether repeated LP Ruiz scaling can be skipped on root-child relaxations
    /// whose matrix/objective are unchanged and only bounds differ.
    fn can_skip_repeated_lp_scaling(&self) -> bool {
        false
    }
    /// Return constraint data for per-node bound propagation, or `None` to skip it.
    ///
    /// MILP returns `Some((&A, &b, &constraint_types))` so the B&B driver can call
    /// [`presolve::tighten_bounds_at_node`] before each LP solve. MIQP returns `None`
    /// because it relies on per-node Ruiz scaling (presolve) for conditioning instead.
    fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
        None
    }

    /// In-tree cut separation hook. Default: no-op (returns `None`).
    ///
    /// MILP overrides this to re-separate GMI/MIR from the node LP relaxation and
    /// return a cut-tightened result when the node bound improves. Cuts bake in
    /// the node's branching-tightened bounds, so they are valid only within this
    /// node's subtree: separation is **node-local** (a fresh pool per call, never
    /// reused at other nodes) and the cut rows are not propagated to children.
    /// `bounds` are the node's bounds; `res` is its (Optimal) relaxation result.
    fn separate_tree_cuts(
        &self,
        _bounds: &[(f64, f64)],
        _res: &SolverResult,
        _mask: &[bool],
        _opts: &SolverOptions,
        _depth: usize,
        _node_index: usize,
    ) -> Option<SolverResult> {
        None
    }

    /// Run the RINS heuristic: fix integer variables where the LP relaxation and
    /// the incumbent agree and solve a sub-MIP over the remaining variables.
    /// Returns `None` for MIQP (default) or when RINS is disabled.
    fn run_rins(
        &self,
        _x_lp: &[f64],
        _x_inc: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _opts: &SolverOptions,
    ) -> Option<SolverResult> {
        None
    }

    /// Run the RENS heuristic: round a node LP relaxation by fixing integral
    /// components and restricting fractional ones to `{floor, ceil}`, then solve
    /// the small sub-MIP. Returns `None` for MIQP (default) or when disabled.
    fn run_rens(
        &self,
        _x_lp: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _opts: &SolverOptions,
    ) -> Option<SolverResult> {
        None
    }

    /// Run the local-branching heuristic: add a Hamming-distance ≤ k cut on the
    /// binary variables around the incumbent and solve the neighborhood sub-MIP.
    /// Returns `None` for MIQP (default) or when disabled.
    fn run_local_branching(
        &self,
        _x_inc: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _opts: &SolverOptions,
    ) -> Option<SolverResult> {
        None
    }
}

/// Search statistics returned by [`solve_milp_with_stats`] / [`solve_miqp_with_stats`].
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
}

/// Solve a MILP to (relative) ε-optimality via branch-and-bound.
pub fn solve_milp(problem: &MilpProblem, options: &SolverOptions, cfg: &MipConfig) -> SolverResult {
    solve_milp_with_stats(problem, options, cfg).0
}

/// Like [`solve_milp`] but also returns search statistics (test sentinel hook).
///
/// Returns `(NumericalError, default stats)` immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_milp_with_stats(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    if options.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    if let Err(e) = validate_integer_vars(&problem.integer_vars, problem.lp.num_vars) {
        return (
            SolverResult::not_supported(e.to_string()),
            MipStats::default(),
        );
    }
    // Establish a shared deadline before FP so that FP and B&B draw from the same
    // budget.  Without this, each LP in FP gets a fresh `timeout_secs` window and
    // `solve_mip_core` resets the clock again — allowing up to (MAX_FP_ITER + 1)×
    // timeout consumption.  If the caller already set an explicit deadline, honour it.
    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| Instant::now() + Duration::from_secs_f64(s))
    });
    let mut opts_with_dl = options.clone();
    opts_with_dl.deadline = deadline;
    opts_with_dl.timeout_secs = None;

    // MILP-specific root presolve: coefficient propagation tightens integer bounds.
    // Presolve is skipped when there are no integer variables (pure LP fallback is
    // handled inside the generic driver). Non-empty integer_vars with infeasible
    // integer rounding return early here before entering the B&B.
    if !problem.integer_vars.is_empty() {
        let mask = integer_mask(problem.lp.num_vars, &problem.integer_vars);
        // Root presolve: multi-pass propagation + probing tightens integer bounds.
        let mut tightened = problem.lp.bounds.clone();
        let presolve_t0 = std::time::Instant::now();
        let presolve_ok = presolve::tighten_bounds_with_probing(
            &problem.lp.a,
            &problem.lp.b,
            &problem.lp.constraint_types,
            &mut tightened,
            &problem.integer_vars,
            deadline,
        );
        let presolve_ms = presolve_t0.elapsed().as_secs_f64() * 1000.0;
        let problem_bt: MilpProblem = match presolve_ok {
            None => {
                // Infeasibility detected at presolve. Report the presolve time as
                // relaxation_time_infeasible_ms so callers see a nonzero infeasibility cost.
                let stats = MipStats {
                    relaxation_time_infeasible_ms: presolve_ms,
                    ..Default::default()
                };
                return (SolverResult::infeasible(), stats);
            }
            Some(_) if tightened != problem.lp.bounds => {
                let mut lp_bt = problem.lp.clone();
                lp_bt.bounds = tightened;
                MilpProblem {
                    lp: lp_bt,
                    integer_vars: problem.integer_vars.clone(),
                }
            }
            Some(_) => problem.clone(),
        };
        // Static symmetry breaking: append lex-leader ordering rows for orbits
        // of interchangeable binary variables. The rows preserve at least one
        // optimal representative per orbit (objective unchanged) while shrinking
        // the search tree, so the whole downstream pipeline (FP, cuts, B&B) may
        // operate on the reduced symmetric space.
        let problem_bt = if cfg.symmetry {
            symmetry::break_symmetry(&problem_bt)
        } else {
            problem_bt
        };
        // Run the feasibility pump on the original (bound-tightened) LP before
        // augmenting with cuts.  FP must see the unmodified constraint structure
        // so that the LP pump LPs and the final validation both use the original
        // bounds and Le/Ge rows, not the GMI cut rows added below.
        let fp_t0 = Instant::now();
        let fp_inc = heuristics::feasibility_pump::run_feasibility_pump(
            &problem_bt.lp,
            &problem_bt.integer_vars,
            cfg.integer_feas_tol,
            &opts_with_dl,
        );
        let fp_us = fp_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        // Root GMI cuts tighten the LP relaxation without removing any
        // integer-feasible point, so the optimum is unchanged while the tree
        // shrinks. The added rows leave `num_vars` (hence `mask`) untouched.
        let cut_t0 = Instant::now();
        let effective = if cfg.cuts {
            cuts::add_root_cuts(&problem_bt, &opts_with_dl, cfg)
        } else {
            problem_bt
        };
        let root_cut_us = cut_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let (res, mut stats) = solve_mip_core(&effective, &opts_with_dl, cfg, mask, fp_inc);
        stats.fp_us = fp_us;
        stats.root_cut_us = root_cut_us;
        return (res, stats);
    }
    solve_mip_with_stats(problem, &opts_with_dl, cfg)
}

/// Solve a **convex** MIQP to (relative) ε-optimality via branch-and-bound.
///
/// Each node solves a convex QP relaxation (IP-PMM). A non-PSD `Q` (non-convex
/// MIQP) is out of scope: the QP relaxation would not be a valid lower bound, so
/// the solver returns [`SolveStatus::NonConvex`] rather than a silently wrong
/// answer. Use `solve_qp_global` for non-convex continuous QP.
pub fn solve_miqp(problem: &MiqpProblem, options: &SolverOptions, cfg: &MipConfig) -> SolverResult {
    solve_miqp_with_stats(problem, options, cfg).0
}

/// Like [`solve_miqp`] but also returns search statistics (test sentinel hook).
///
/// Returns `(NumericalError, default stats)` immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_miqp_with_stats(
    problem: &MiqpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    if options.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    // Central structural check: `solve_fixed_point` (the all-integer-fixed
    // B&B leaf) indexes `quadratic_constraints[k]` for `k < num_constraints`,
    // so a non-empty vector shorter than `num_constraints` (direct
    // public-field assignment on `problem.qp`, bypassing the setter) would
    // panic. `QpProblem::validate` is the shared source of this invariant.
    if problem.qp.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    if let Err(e) = validate_integer_vars(&problem.integer_vars, problem.qp.num_vars) {
        return (
            SolverResult::not_supported(e.to_string()),
            MipStats::default(),
        );
    }
    if !problem.is_convex() {
        return (nonconvex_result(), MipStats::default());
    }
    // MIQP root presolve: bound tightening via coefficient propagation.
    // `tighten_bounds_linear` ignores Q (quadratic term) and operates on the
    // same linear constraints as MILP, so it is valid for convex MIQP too.
    if !problem.integer_vars.is_empty() {
        let n = problem.qp.num_vars;
        let mask = integer_mask(n, &problem.integer_vars);
        match presolve::tighten_bounds_linear(
            n,
            &problem.qp.a,
            &problem.qp.b,
            &problem.qp.constraint_types,
            &problem.qp.bounds,
            &mask,
        ) {
            None => return (SolverResult::infeasible(), MipStats::default()),
            Some(tightened) if tightened != problem.qp.bounds => {
                let mut qp_bt = problem.qp.clone();
                qp_bt.bounds = tightened;
                let problem_bt = MiqpProblem {
                    qp: qp_bt,
                    integer_vars: problem.integer_vars.clone(),
                };
                return solve_mip_core(&problem_bt, options, cfg, mask, None);
            }
            Some(_) => {
                return solve_mip_core(problem, options, cfg, mask, None);
            }
        }
    }
    solve_mip_with_stats(problem, options, cfg)
}

/// Generic branch-and-bound driver shared by MILP (LP relaxation) and convex
/// MIQP (QP relaxation). The only difference between the two is the relaxation
/// solver, abstracted by [`Relaxation`].
fn solve_mip_with_stats<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    let mask = integer_mask(problem.num_vars(), problem.integer_vars());
    solve_mip_core(problem, options, cfg, mask, None)
}

/// Solve candidate LPs for strong branching and return per-variable scores.
///
/// For each candidate variable `j`, solves down and up child LPs, records the
/// objective improvements as pseudocost observations, and returns the combined
/// branching score keyed by `j`.  Candidates that fail to solve to Optimal are
/// skipped (no pseudocost update, no score entry).
#[allow(clippy::too_many_arguments)]
fn measure_strong_branch_scores<R: Relaxation>(
    problem: &R,
    parent_bounds: &[(f64, f64)],
    parent_sol: &[f64],
    parent_obj: f64,
    candidates: &[usize],
    j_to_k: &HashMap<usize, usize>,
    shared: &SolverOptions,
    parent_warm_start: Option<&WarmStartBasis>,
    pc: &mut PseudocostState,
    stats: &mut MipStats,
) -> HashMap<usize, f64> {
    let mut scores = HashMap::new();
    stats.strong_branch_calls += 1;
    stats.strong_branch_candidates += candidates.len();
    let mut sb_opts = shared.clone();
    sb_opts.warm_start = parent_warm_start.cloned();
    if problem.can_skip_repeated_lp_scaling() {
        sb_opts.use_ruiz_scaling = false;
    }
    let mut sb_retry_opts = sb_opts.clone();
    sb_retry_opts.use_ruiz_scaling = true;
    // Cold variants for children that change the standard-form column layout
    // (infinite→finite bound): the parent basis indices no longer match those
    // columns, so the warm start must be dropped — same guard the real-children
    // path applies. Without this, strong-branch scores for one-sided integer
    // variables are computed from a layout-mismatched warm start.
    let mut sb_opts_cold = sb_opts.clone();
    sb_opts_cold.warm_start = None;
    let mut sb_retry_opts_cold = sb_retry_opts.clone();
    sb_retry_opts_cold.warm_start = None;

    for &j in candidates {
        let v = parent_sol[j];
        let (down_bounds, up_bounds) = branch::branch_bounds(parent_bounds, j, v);
        let (down_o, down_ro) = if bound_layout_changes(parent_bounds, &down_bounds, j) {
            (&sb_opts_cold, &sb_retry_opts_cold)
        } else {
            (&sb_opts, &sb_retry_opts)
        };
        let (up_o, up_ro) = if bound_layout_changes(parent_bounds, &up_bounds, j) {
            (&sb_opts_cold, &sb_retry_opts_cold)
        } else {
            (&sb_opts, &sb_retry_opts)
        };

        let t0 = Instant::now();
        let r_down = solve_relaxation_with_scaling_retry(problem, &down_bounds, down_o, down_ro);
        let r_up = solve_relaxation_with_scaling_retry(problem, &up_bounds, up_o, up_ro);
        stats.strong_branch_lp_solves += 2;
        stats.strong_branch_us = stats
            .strong_branch_us
            .saturating_add(t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);

        let down_ok = matches!(
            r_down.status,
            SolveStatus::Optimal | SolveStatus::Infeasible
        );
        let up_ok = matches!(r_up.status, SolveStatus::Optimal | SolveStatus::Infeasible);

        if !down_ok || !up_ok {
            continue;
        }

        let d_down = if r_down.status == SolveStatus::Infeasible {
            f64::INFINITY
        } else {
            (r_down.objective - parent_obj).max(0.0)
        };
        let d_up = if r_up.status == SolveStatus::Infeasible {
            f64::INFINITY
        } else {
            (r_up.objective - parent_obj).max(0.0)
        };

        // `j` is drawn from `candidates`, which `strong_branch_candidates` builds
        // by iterating the same `integer_vars` slice that `j_to_k` was built
        // from (`j_to_k = integer_vars.iter().enumerate().map(|(k, &j)| (j, k))`),
        // so `j` is always a key of `j_to_k`.
        let &k = j_to_k
            .get(&j)
            .expect("j is drawn from integer_vars, which built j_to_k");
        let f_down = v - v.floor();
        let f_up = v.ceil() - v;
        if d_down.is_finite() && f_down > 1e-12 {
            pc.record_down(k, d_down / f_down);
        }
        if d_up.is_finite() && f_up > 1e-12 {
            pc.record_up(k, d_up / f_up);
        }

        let score = branch::pseudocost_score(d_down, d_up);
        scores.insert(j, score);
    }
    scores
}

fn solve_relaxation_with_scaling_retry<R: Relaxation>(
    problem: &R,
    bounds: &[(f64, f64)],
    fast_opts: &SolverOptions,
    retry_opts: &SolverOptions,
) -> SolverResult {
    let res = problem.solve(bounds, fast_opts);
    if fast_opts.use_ruiz_scaling
        || !needs_scaled_retry(&res)
        || deadline_reached(fast_opts.deadline)
    {
        return res;
    }
    let mut retry = problem.solve(bounds, retry_opts);
    retry.timing_breakdown = combine_timing(res.timing_breakdown, retry.timing_breakdown);
    retry
}

fn needs_scaled_retry(res: &SolverResult) -> bool {
    matches!(
        res.status,
        SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::SuboptimalSolution
            | SolveStatus::Stalled
            | SolveStatus::MaxIterations
    )
}

fn combine_timing(
    first: Option<TimingBreakdown>,
    second: Option<TimingBreakdown>,
) -> Option<TimingBreakdown> {
    match (first, second) {
        (None, None) => None,
        (Some(t), None) | (None, Some(t)) => Some(t),
        (Some(a), Some(b)) => Some(TimingBreakdown {
            presolve_us: a.presolve_us.saturating_add(b.presolve_us),
            solve_us: a.solve_us.saturating_add(b.solve_us),
            postsolve_us: a.postsolve_us.saturating_add(b.postsolve_us),
            ipm_factorize_us: a.ipm_factorize_us.saturating_add(b.ipm_factorize_us),
            ipm_solve_us: a.ipm_solve_us.saturating_add(b.ipm_solve_us),
            ipm_reg_retries: a.ipm_reg_retries.saturating_add(b.ipm_reg_retries),
            ipm_used_iterative: a.ipm_used_iterative || b.ipm_used_iterative,
            postsolve_map_us: a.postsolve_map_us.saturating_add(b.postsolve_map_us),
            postsolve_lsq_us: a.postsolve_lsq_us.saturating_add(b.postsolve_lsq_us),
            postsolve_recovery_us: a
                .postsolve_recovery_us
                .saturating_add(b.postsolve_recovery_us),
            postsolve_refine_us: a.postsolve_refine_us.saturating_add(b.postsolve_refine_us),
            postsolve_krylov_ir_us: a
                .postsolve_krylov_ir_us
                .saturating_add(b.postsolve_krylov_ir_us),
        }),
    }
}

/// Core B&B driver that accepts a precomputed `integer_mask` to avoid
/// recomputing it when the caller (e.g. `solve_milp_with_stats`) already has it.
///
/// `initial_incumbent` is an optional integer-feasible solution found by a
/// pre-B&B heuristic (e.g., feasibility pump). When provided it is adopted as
/// the starting incumbent so B&B can immediately prune nodes whose relaxation
/// bound is already within the gap tolerance.
fn solve_mip_core<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
    mask: Vec<bool>,
    initial_incumbent: Option<SolverResult>,
) -> (SolverResult, MipStats) {
    let mut stats = MipStats {
        approx_bounds_bytes_per_node: problem.num_vars() * 2 * std::mem::size_of::<f64>(),
        root_lp_bound: f64::NEG_INFINITY,
        ..MipStats::default()
    };

    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| Instant::now() + Duration::from_secs_f64(s))
    });
    let mut shared = options.clone();
    shared.deadline = deadline;
    shared.timeout_secs = None;
    shared.multistart = None;
    shared.global_optimization = None;

    if problem.integer_vars().is_empty() {
        return (problem.solve(problem.root_bounds(), &shared), stats);
    }

    shared.recover_warm_start_basis = true;
    shared.use_lp_crash_basis = false;
    shared.warm_start = None;
    if problem.skip_node_presolve() {
        shared.presolve = false;
    }

    let integer_vars = problem.integer_vars().to_vec();
    let j_to_k: HashMap<usize, usize> = integer_vars
        .iter()
        .enumerate()
        .map(|(k, &j)| (j, k))
        .collect();
    let use_reliability = cfg.branching == MipBranching::Reliability;
    let mut pc = PseudocostState::new(integer_vars.len());

    let mut state = MipState::new();
    if let Some(inc) = initial_incumbent {
        if state.consider(&inc) {
            stats.incumbent_updates += 1;
            stats.fp_incumbent_found = true;
        }
    }

    let mut q = NodeQueue::new();
    q.push(MipNode::root(
        problem.root_bounds().to_vec(),
        f64::NEG_INFINITY,
    ));

    let mut open_lb = f64::INFINITY;
    let mut had_open = false;
    let mut proof_uncertain = false;
    let mut deadline_stop = false;
    let mut maxnodes_stop = false;
    let mut unbounded = false;
    let mut root_solved = false;
    let mut nodes_since_dive: usize = 0;
    let mut dive_start_depth: usize = 0;
    let mut conflicts = conflict::ConflictStore::new();
    let root_bounds = problem.root_bounds().to_vec();

    while let Some(mut node) = q.pop() {
        // --- Stop conditions ---
        if deadline_reached(deadline) {
            open_lb = open_lb.min(node.lower_bound);
            had_open = true;
            deadline_stop = true;
            if q.is_diving() {
                q.end_dive();
            }
            break;
        }
        if stats.nodes_processed >= cfg.max_nodes {
            open_lb = open_lb.min(node.lower_bound);
            had_open = true;
            maxnodes_stop = true;
            if q.is_diving() {
                q.end_dive();
            }
            break;
        }

        // --- Dive management: start a dive every DIVE_FREQUENCY best-bound pops ---
        if !q.is_diving() {
            nodes_since_dive += 1;
            let freq = if state.incumbent_obj.is_none() {
                DIVE_FREQUENCY_NO_INCUMBENT
            } else {
                DIVE_FREQUENCY
            };
            if nodes_since_dive >= freq {
                nodes_since_dive = 0;
                dive_start_depth = node.depth;
                q.start_dive();
            }
        }

        // --- Incumbent-bound pruning and conflict pruning ---
        if let Some(inc) = state.incumbent_obj {
            if should_prune(node.lower_bound, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                if q.is_diving() {
                    q.end_dive();
                }
                continue;
            }
        }
        if node.depth > 0 && conflicts.is_conflicted(&node.var_bounds) {
            stats.pruned += 1;
            stats.conflict_pruned += 1;
            continue;
        }

        // --- Per-node bound propagation (MILP only; MIQP returns None) ---
        let propagation_t0 = Instant::now();
        let tightened = if let Some((prop_a, prop_b, prop_ct)) = problem.propagation_data() {
            match presolve::tighten_bounds_at_node(
                problem.num_vars(),
                prop_a,
                prop_b,
                prop_ct,
                &node.var_bounds,
                &mask,
            ) {
                Ok(tb) => Some(tb),
                Err(()) => {
                    stats.node_propagation_us = stats.node_propagation_us.saturating_add(
                        propagation_t0
                            .elapsed()
                            .as_micros()
                            .min(u128::from(u64::MAX)) as u64,
                    );
                    stats.pruned += 1;
                    stats.propagation_pruned += 1;
                    continue;
                }
            }
        } else {
            None
        };
        stats.node_propagation_us = stats.node_propagation_us.saturating_add(
            propagation_t0
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
        );
        let solve_bounds: &[(f64, f64)] = tightened.as_deref().unwrap_or(&node.var_bounds);

        // --- Relaxation solve ---
        let scale_before = crate::presolve::scaling::lp_scale_profile_snapshot();
        let fallback_before = crate::simplex::dual_advanced::fallback_profile_snapshot();
        let t0 = Instant::now();
        let mut node_options = shared.clone();
        if root_solved && problem.can_skip_repeated_lp_scaling() {
            node_options.use_ruiz_scaling = false;
        }
        if let Some(ref ws) = node.warm_start {
            node_options.warm_start = Some(ws.clone());
        }
        let mut res = if node_options.use_ruiz_scaling {
            problem.solve(solve_bounds, &node_options)
        } else {
            let mut retry_options = node_options.clone();
            retry_options.use_ruiz_scaling = true;
            solve_relaxation_with_scaling_retry(
                problem,
                solve_bounds,
                &node_options,
                &retry_options,
            )
        };
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let scale_delta = crate::presolve::scaling::lp_scale_profile_delta(
            scale_before,
            crate::presolve::scaling::lp_scale_profile_snapshot(),
        );
        let fallback_delta = crate::simplex::dual_advanced::fallback_profile_delta(
            fallback_before,
            crate::simplex::dual_advanced::fallback_profile_snapshot(),
        );
        accumulate_node_stats(
            &mut stats,
            elapsed_ms,
            &scale_delta,
            &fallback_delta,
            &res,
            &mut root_solved,
            node.depth,
        );

        // --- Status dispatch ---
        match res.status {
            SolveStatus::Infeasible => {
                stats.pruned += 1;
                if q.is_diving() {
                    q.end_dive();
                }
                conflicts.learn(&node.var_bounds, &root_bounds);
                stats.conflict_clauses_learned = conflicts.len();
                continue;
            }
            SolveStatus::Unbounded => {
                unbounded = true;
                if q.is_diving() {
                    q.end_dive();
                }
                break;
            }
            SolveStatus::Timeout => {
                open_lb = open_lb.min(node.lower_bound);
                had_open = true;
                deadline_stop = true;
                if q.is_diving() {
                    q.end_dive();
                }
                break;
            }
            _ => {}
        }

        // --- In-tree cut separation (gated; MILP overrides, MIQP no-op) ---
        if cfg.tree_cuts && matches!(res.status, SolveStatus::Optimal) && !res.solution.is_empty() {
            if let Some(improved) = problem.separate_tree_cuts(
                solve_bounds,
                &res,
                &mask,
                &node_options,
                node.depth,
                stats.nodes_processed,
            ) {
                stats.tree_cut_rounds += 1;
                res = improved;
            }
        }

        // --- Node outcome: prune / leaf / branch ---
        let trusted = matches!(res.status, SolveStatus::Optimal) && !res.solution.is_empty();
        match process_node_outcome(
            problem,
            &mut node,
            &res,
            trusted,
            &mut state,
            &mut stats,
            cfg,
            &shared,
            &mask,
            &integer_vars,
            &j_to_k,
            &mut pc,
            use_reliability,
            &deadline,
            dive_start_depth,
        ) {
            NodeAction::Skip { end_dive } => {
                if end_dive && q.is_diving() {
                    q.end_dive();
                }
                continue;
            }
            NodeAction::OpenLb {
                node_lb,
                uncertain,
                end_dive,
            } => {
                open_lb = open_lb.min(node_lb);
                had_open = true;
                if uncertain {
                    proof_uncertain = true;
                }
                if end_dive && q.is_diving() {
                    q.end_dive();
                }
                continue;
            }
            NodeAction::PushChildren {
                node_lb,
                down,
                up,
                kind,
                end_dive,
            } => {
                if end_dive && q.is_diving() {
                    q.end_dive();
                }
                match kind {
                    ChildKind::Branched {
                        jb,
                        res_obj,
                        jb_val,
                        down_ws,
                        up_ws,
                    } => {
                        q.push(
                            node.child_branched(down, node_lb, down_ws, jb, false, res_obj, jb_val),
                        );
                        q.push(node.child_branched(up, node_lb, up_ws, jb, true, res_obj, jb_val));
                    }
                    ChildKind::Split => {
                        q.push(node.child(down, node_lb));
                        q.push(node.child(up, node_lb));
                    }
                }
            }
        }
    }

    if q.is_diving() {
        q.end_dive();
    }
    if unbounded {
        return (SolverResult::unbounded(), stats);
    }

    let remaining_lb = match q.best_lower_bound() {
        Some(b) => open_lb.min(b),
        None => open_lb,
    };
    let interrupted = deadline_stop || maxnodes_stop;

    match state.incumbent.take() {
        Some(mut inc) => {
            let inc_obj = state.incumbent_obj.expect("incumbent objective set");
            let proven = !proof_uncertain && within_gap(inc_obj, remaining_lb, cfg.gap_tol);
            inc.solution = round_integers(inc.solution, problem.integer_vars());
            inc.status = if proven {
                let effective_lb = remaining_lb.min(inc_obj);
                let scale = 1.0_f64.max(inc_obj.abs());
                let gap_rel = (inc_obj - effective_lb) / scale;
                inc.bound_gap_cert = Some(BoundGapCertificate::new(
                    inc_obj,
                    effective_lb,
                    gap_rel,
                    cfg.gap_tol,
                ));
                SolveStatus::Optimal
            } else if deadline_stop {
                SolveStatus::Timeout
            } else {
                SolveStatus::SuboptimalSolution
            };
            (inc, stats)
        }
        None => (
            finalize_no_incumbent(interrupted, had_open, q.is_empty(), deadline_stop),
            stats,
        ),
    }
}

// ---------------------------------------------------------------------------
// B&B node processing helpers
// ---------------------------------------------------------------------------

/// Action the B&B loop should take after processing a node's outcome.
enum NodeAction {
    /// Skip node (pruned or integer-feasible leaf). End dive when `end_dive`.
    Skip { end_dive: bool },
    /// Record open lower bound (max-depth or proof-uncertain region).
    /// Mark `proof_uncertain` when `uncertain`. End dive when `end_dive`.
    OpenLb {
        node_lb: f64,
        uncertain: bool,
        end_dive: bool,
    },
    /// Push two child nodes. End dive when `end_dive`.
    PushChildren {
        node_lb: f64,
        down: Vec<(f64, f64)>,
        up: Vec<(f64, f64)>,
        kind: ChildKind,
        end_dive: bool,
    },
}

/// Metadata distinguishing trusted-Optimal branches from non-Optimal bisections.
enum ChildKind {
    /// Trusted (Optimal) branch: children carry warm-start bases and pseudocost metadata.
    Branched {
        jb: usize,
        res_obj: f64,
        /// Parent LP solution value at `jb`; used to compute per-unit pseudocost.
        jb_val: f64,
        down_ws: Option<WarmStartBasis>,
        up_ws: Option<WarmStartBasis>,
    },
    /// Non-Optimal bisection on widest integer interval; no warm-start or metadata.
    Split,
}

/// Accumulate per-node timing and profiling counters into `stats`.
fn accumulate_node_stats(
    stats: &mut MipStats,
    elapsed_ms: f64,
    scale_delta: &crate::presolve::scaling::LpScaleProfileSnapshot,
    fallback_delta: &crate::simplex::dual_advanced::SimplexFallbackSnapshot,
    res: &SolverResult,
    root_solved: &mut bool,
    node_depth: usize,
) {
    stats.nodes_processed += 1;
    stats.max_depth_seen = stats.max_depth_seen.max(node_depth);
    stats.relaxation_time_total_ms += elapsed_ms;
    if !*root_solved {
        stats.relaxation_time_root_ms = elapsed_ms;
        stats.lp_scale_us_root += scale_delta.scale_us;
        stats.lp_scale_calls_root += scale_delta.calls;
        if let Some(tb) = res.timing_breakdown {
            stats.lp_solve_us_root += tb.solve_us;
        }
        *root_solved = true;
    } else {
        stats.relaxation_time_desc_ms += elapsed_ms;
        stats.lp_scale_us_desc += scale_delta.scale_us;
        stats.lp_scale_calls_desc += scale_delta.calls;
        if let Some(tb) = res.timing_breakdown {
            stats.lp_solve_us_desc += tb.solve_us;
        }
    }
    stats.fallback_ub_violation_out_of_scope += fallback_delta.ub_violation_out_of_scope;
    stats.fallback_phase1_bound_violation += fallback_delta.phase1_bound_violation;
    stats.fallback_crash_infeasible += fallback_delta.crash_infeasible;
    match res.status {
        SolveStatus::Optimal => stats.relaxation_time_optimal_ms += elapsed_ms,
        SolveStatus::Infeasible => stats.relaxation_time_infeasible_ms += elapsed_ms,
        _ => {}
    }
    if let Some(tb) = res.timing_breakdown {
        stats.lp_presolve_us_total += tb.presolve_us;
        stats.lp_solve_us_total += tb.solve_us;
        stats.lp_postsolve_us_total += tb.postsolve_us;
    }
}

/// Select the variable to branch on for an Optimal relaxation solution.
///
/// With `use_reliability`, runs strong-branching trials for candidates with
/// insufficient pseudocost observations, then falls back to reliability
/// pseudocost selection. Without it, delegates to the configured heuristic
/// (most-infeasible, least-infeasible, etc.).
#[allow(clippy::too_many_arguments)]
fn pick_branch_var<R: Relaxation>(
    problem: &R,
    node_bounds: &[(f64, f64)],
    sol: &[f64],
    obj: f64,
    mask: &[bool],
    integer_vars: &[usize],
    j_to_k: &HashMap<usize, usize>,
    shared: &SolverOptions,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_warm_start: Option<&WarmStartBasis>,
    pc: &mut PseudocostState,
    stats: &mut MipStats,
    use_reliability: bool,
) -> usize {
    if use_reliability {
        let sb_cands = strong_branch_candidates(sol, mask, integer_vars, cfg.integer_feas_tol, pc);
        let strong_scores = if !sb_cands.is_empty() && !deadline_reached(*deadline) {
            measure_strong_branch_scores(
                problem,
                node_bounds,
                sol,
                obj,
                &sb_cands,
                j_to_k,
                shared,
                parent_warm_start,
                pc,
                stats,
            )
        } else {
            HashMap::new()
        };
        select_branching_variable_reliability(
            sol,
            mask,
            integer_vars,
            cfg.integer_feas_tol,
            pc,
            if strong_scores.is_empty() {
                None
            } else {
                Some(&strong_scores)
            },
        )
        .expect("non-integer-feasible Optimal relaxation has a fractional integer var")
    } else {
        select_branching_variable(sol, mask, cfg.integer_feas_tol, cfg.branching)
            .expect("non-integer-feasible Optimal relaxation has a fractional integer var")
    }
}

/// Attempt a RINS heuristic call and update stats/incumbent when it improves.
fn try_rins<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
    rel_sol: &[f64],
) {
    if !cfg.rins_enabled
        || !stats
            .nodes_processed
            .is_multiple_of(heuristics::rins::RINS_INTERVAL)
        || state.incumbent_obj.is_none()
    {
        return;
    }
    let rins_res = {
        let inc_sol = match state.incumbent {
            Some(ref inc) => &inc.solution,
            None => return,
        };
        stats.rins_calls += 1;
        problem.run_rins(rel_sol, inc_sol, cfg, deadline, opts)
    };
    if let Some(res) = rins_res {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.rins_improvements += 1;
        }
    }
}

/// Attempt a RENS heuristic call on the node LP relaxation and update
/// stats/incumbent when it yields an improving feasible point. Unlike RINS,
/// RENS does not require an existing incumbent (it manufactures one).
fn try_rens<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
    rel_sol: &[f64],
) {
    if !cfg.rens_enabled {
        return;
    }
    let should_try = if state.incumbent_obj.is_none() {
        if !state.rens_first_incumbent_attempted {
            state.rens_first_incumbent_attempted = true;
            true
        } else {
            stats
                .nodes_processed
                .is_multiple_of(heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT)
        }
    } else {
        stats
            .nodes_processed
            .is_multiple_of(heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT)
    };
    if !should_try {
        return;
    }
    stats.rens_calls += 1;
    if let Some(res) = problem.run_rens(rel_sol, cfg, deadline, opts) {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.rens_improvements += 1;
        }
    }
}

/// Attempt a local-branching heuristic call around the current incumbent and
/// update stats/incumbent when it strictly improves.
fn try_local_branching<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
) {
    if !cfg.local_branching_enabled
        || !stats
            .nodes_processed
            .is_multiple_of(heuristics::local_branching::LOCAL_BRANCHING_INTERVAL)
    {
        return;
    }
    let lb_res = {
        let inc_sol = match state.incumbent {
            Some(ref inc) => &inc.solution,
            None => return,
        };
        stats.local_branching_calls += 1;
        problem.run_local_branching(inc_sol, cfg, deadline, opts)
    };
    if let Some(res) = lb_res {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.local_branching_improvements += 1;
        }
    }
}

/// Determine what the B&B loop should do after a node's relaxation result.
///
/// Handles both trusted (Optimal) and non-trusted paths: pseudocost updates,
/// integer-feasibility detection, RINS, max-depth, reduced-cost fixing,
/// branching variable selection, and child bound computation.
#[allow(clippy::too_many_arguments)]
fn process_node_outcome<R: Relaxation>(
    problem: &R,
    node: &mut MipNode,
    res: &SolverResult,
    trusted: bool,
    state: &mut MipState,
    stats: &mut MipStats,
    cfg: &MipConfig,
    shared: &SolverOptions,
    mask: &[bool],
    integer_vars: &[usize],
    j_to_k: &HashMap<usize, usize>,
    pc: &mut PseudocostState,
    use_reliability: bool,
    deadline: &Option<Instant>,
    dive_start_depth: usize,
) -> NodeAction {
    if trusted {
        // Pseudocost update: record per-unit cost (delta / fractionality) so
        // that score() can correctly predict gains at different fractionalities.
        if use_reliability {
            if let Some(jb) = node.branch_var {
                // `node.branch_var` is only ever set (via `child_branched`) to the
                // `jb` returned by `pick_branch_var`, which always returns a `j`
                // that is a key of `j_to_k` (either from `integer_vars` directly,
                // via `select_branching_variable_reliability`, or via the
                // `mask`-gated `select_branching_variable`, where `mask` and
                // `j_to_k` are built from the same `integer_vars`).
                let &k = j_to_k
                    .get(&jb)
                    .expect("node.branch_var is always a key of j_to_k");
                let delta = (res.objective - node.parent_obj).max(0.0);
                let v = node.branch_parent_val;
                if node.branch_up {
                    let f_up = v.ceil() - v;
                    if f_up > 1e-12 {
                        pc.record_up(k, delta / f_up);
                    }
                } else {
                    let f_down = v - v.floor();
                    if f_down > 1e-12 {
                        pc.record_down(k, delta / f_down);
                    }
                }
            }
        }
        if stats.nodes_processed == 1 {
            stats.root_lp_bound = res.objective;
        }
        let node_lb = node.lower_bound.max(res.objective);

        // Post-solve bound pruning.
        if let Some(inc) = state.incumbent_obj {
            if should_prune(node_lb, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                return NodeAction::Skip { end_dive: false };
            }
        }

        // Integer-feasible leaf.
        if is_integer_feasible(&res.solution, mask, cfg.integer_feas_tol) {
            if state.consider(res) {
                stats.incumbent_updates += 1;
            }
            return NodeAction::Skip { end_dive: true };
        }

        // Primal heuristics on this fractional node: RENS rounds the LP point to
        // manufacture an incumbent; RINS and local branching refine an existing one.
        try_rens(problem, stats, state, cfg, deadline, shared, &res.solution);
        try_rins(problem, stats, state, cfg, deadline, shared, &res.solution);
        try_local_branching(problem, stats, state, cfg, deadline, shared);

        // Max-depth limit.
        if node.depth + 1 > cfg.max_depth {
            return NodeAction::OpenLb {
                node_lb,
                uncertain: false,
                end_dive: true,
            };
        }

        // End dive when it has descended MAX_DIVE_DEPTH levels below the pivot.
        let end_dive = node.depth >= dive_start_depth + MAX_DIVE_DEPTH;

        // Reduced-cost fixing: tighten integer-variable bounds for children.
        if let Some(inc_obj) = state.incumbent_obj {
            stats.rc_vars_fixed +=
                reduced_cost_fixing(res, inc_obj, &mut node.var_bounds, problem.integer_vars());
        }

        // Select branching variable and compute child bounds.
        let jb = pick_branch_var(
            problem,
            &node.var_bounds,
            &res.solution,
            res.objective,
            mask,
            integer_vars,
            j_to_k,
            shared,
            cfg,
            deadline,
            res.warm_start_basis.as_ref(),
            pc,
            stats,
            use_reliability,
        );
        let (down, up) = branch_bounds(&node.var_bounds, jb, res.solution[jb]);
        let child_ws = res.warm_start_basis.clone();
        let down_ws = if bound_layout_changes(&node.var_bounds, &down, jb) {
            None
        } else {
            child_ws.clone()
        };
        let up_ws = if bound_layout_changes(&node.var_bounds, &up, jb) {
            None
        } else {
            child_ws
        };
        NodeAction::PushChildren {
            node_lb,
            down,
            up,
            kind: ChildKind::Branched {
                jb,
                res_obj: res.objective,
                jb_val: res.solution[jb],
                down_ws,
                up_ws,
            },
            end_dive,
        }
    } else {
        // Non-Optimal relaxation: bisect on the widest integer interval.
        let node_lb = node.lower_bound;
        match widest_splittable_integer(&node.var_bounds, mask) {
            Some(_) if node.depth + 1 > cfg.max_depth => NodeAction::OpenLb {
                node_lb,
                uncertain: true,
                end_dive: true,
            },
            Some(jb) => {
                let end_dive = node.depth >= dive_start_depth + MAX_DIVE_DEPTH;
                let (down, up) = split_integer_box(&node.var_bounds, jb);
                NodeAction::PushChildren {
                    node_lb,
                    down,
                    up,
                    kind: ChildKind::Split,
                    end_dive,
                }
            }
            None => NodeAction::OpenLb {
                node_lb,
                uncertain: true,
                end_dive: true,
            },
        }
    }
}

/// Classify the terminal status when no integer-feasible incumbent was found.
///
/// `Infeasible` may be claimed **only** when the whole tree was resolved: the
/// queue is empty, there was no budget interruption, and **no region was left
/// unexplored** (`had_open == false`). An unexplored region (a depth/budget limit,
/// or an unsolvable no-interior relaxation that could not be bisected) means we
/// cannot prove infeasibility, so a no-solution status is returned instead — never
/// a silent false `Infeasible`.
fn finalize_no_incumbent(
    interrupted: bool,
    had_open: bool,
    queue_empty: bool,
    deadline_stop: bool,
) -> SolverResult {
    let fully_resolved = !interrupted && !had_open && queue_empty;
    if fully_resolved {
        SolverResult::infeasible()
    } else if deadline_stop {
        no_solution_result(SolveStatus::Timeout)
    } else {
        no_solution_result(SolveStatus::MaxIterations)
    }
}

/// Reject an `integer_vars` index out of range for `num_vars` before it
/// reaches `integer_mask`'s `assert!`.
///
/// `MilpProblem`/`MiqpProblem` are public structs with a `pub integer_vars`
/// field: `new()` validates it via `normalize_integer_vars`, but a caller can
/// build the struct with a literal (bypassing `new()` entirely, all fields
/// `pub`) or mutate `integer_vars` afterward, so `solve_milp`/`solve_miqp`
/// must re-check it themselves at the solve entry rather than trust
/// construction-time validation -- the same defense already applied to
/// `MisocpProblem::integers` and `NonconvexQcqp`'s `integers` parameter
/// (Codex review R3 horizontal expansion, nonconvex.rs:763).
fn validate_integer_vars(integer_vars: &[usize], num_vars: usize) -> Result<(), MipProblemError> {
    if let Some(&j) = integer_vars.iter().find(|&&j| j >= num_vars) {
        return Err(MipProblemError::InvalidIntegerVar { index: j, num_vars });
    }
    Ok(())
}

/// Boolean mask of length `num_vars`; `true` where the variable is integral.
pub(crate) fn integer_mask(num_vars: usize, integer_vars: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; num_vars];
    for &j in integer_vars {
        assert!(
            j < num_vars,
            "integer variable index {} out of range for {} variables",
            j,
            num_vars
        );
        mask[j] = true;
    }
    mask
}

/// A result tagging a non-convex MIQP/MIQCP (non-PSD `Q` or nonconvex
/// quadratic constraint) as out of scope.
fn nonconvex_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::NonConvex(
            "convex MIQP/MIQCP only: Q is not PSD or a quadratic constraint is nonconvex"
                .to_string(),
        ),
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

/// Round the integer components of `sol` to exact integers (relaxation noise removal).
fn round_integers(mut sol: Vec<f64>, integer_vars: &[usize]) -> Vec<f64> {
    for &j in integer_vars {
        assert!(
            j < sol.len(),
            "integer variable index {} out of range for solution length {}",
            j,
            sol.len()
        );
        sol[j] = sol[j].round();
    }
    sol
}

/// Returns `true` when tightening var `j`'s bound changes the standard-form
/// column layout vs the parent. An infinite bound becoming finite (ub: ∞→boxed,
/// or lb: free→lower-bounded) changes the number of structural columns or adds
/// a UB constraint row, making the parent basis index-incompatible.
fn bound_layout_changes(
    parent_bounds: &[(f64, f64)],
    child_bounds: &[(f64, f64)],
    j: usize,
) -> bool {
    let (p_lb, p_ub) = parent_bounds[j];
    let (c_lb, c_ub) = child_bounds[j];
    (p_ub.is_infinite() && c_ub.is_finite()) || (p_lb.is_infinite() && c_lb.is_finite())
}

/// A result carrying no usable solution, tagged with `status`.
fn no_solution_result(status: SolveStatus) -> SolverResult {
    SolverResult {
        status,
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

/// Maximum distance from a variable value to a bound for it to be considered "at" that bound.
///
/// Used in reduced-cost fixing to decide whether `x[j]` is at its lower or upper bound.
/// Matches typical LP primal feasibility tolerances.
const BOUND_AT_TOL: f64 = 1e-8;

/// Apply reduced-cost fixing to variable bounds in preparation for branching.
///
/// For each integer variable `j` where the LP value is at a bound and the
/// reduced cost exceeds the MIP gap, `x[j]` cannot improve on the incumbent
/// by moving away from that bound — so we fix it there:
///
/// - `x[j] ≈ lb[j]` and `rc[j] > gap`:  fix `x[j] = ceil(lb)`
/// - `x[j] ≈ ub[j]` and `-rc[j] > gap`: fix `x[j] = floor(ub)`
///
/// Bounds are rounded to the nearest feasible integer before fixing. If the
/// rounded bounds are inconsistent (`ceil(lb) > floor(ub)`), the fix is skipped.
///
/// `gap = incumbent_obj - lp_result.objective` (must be positive).
/// Returns the number of variables fixed.
pub(crate) fn reduced_cost_fixing(
    lp_result: &SolverResult,
    incumbent_obj: f64,
    node_bounds: &mut [(f64, f64)],
    integer_vars: &[usize],
) -> usize {
    if lp_result.reduced_costs.is_empty() || lp_result.solution.is_empty() {
        return 0;
    }
    assert_eq!(
        lp_result.reduced_costs.len(),
        node_bounds.len(),
        "reduced-cost fixing requires one reduced cost per variable"
    );
    assert_eq!(
        lp_result.solution.len(),
        node_bounds.len(),
        "reduced-cost fixing requires one solution value per variable"
    );
    let gap = incumbent_obj - lp_result.objective;
    if gap <= 0.0 {
        return 0;
    }
    let rc = &lp_result.reduced_costs;
    let x = &lp_result.solution;
    let mut count = 0usize;
    for &j in integer_vars {
        assert!(
            j < node_bounds.len(),
            "integer variable index {} out of range for {} variables",
            j,
            node_bounds.len()
        );
        let (lb, ub) = node_bounds[j];
        if (lb - ub).abs() < BOUND_AT_TOL {
            continue; // already fixed
        }
        let int_lb = lb.ceil();
        let int_ub = ub.floor();
        if int_lb > int_ub + BOUND_AT_TOL {
            continue; // empty integer range after rounding
        }
        let xj = x[j];
        let rcj = rc[j];
        if (xj - lb).abs() <= BOUND_AT_TOL && rcj > gap {
            node_bounds[j] = (int_lb, int_lb);
            count += 1;
        } else if (xj - ub).abs() <= BOUND_AT_TOL && -rcj > gap {
            node_bounds[j] = (int_ub, int_ub);
            count += 1;
        }
    }
    count
}

/// Incumbent (best integer-feasible upper bound) tracking.
struct MipState {
    incumbent: Option<SolverResult>,
    incumbent_obj: Option<f64>,
    rens_first_incumbent_attempted: bool,
}

impl MipState {
    fn new() -> Self {
        Self {
            incumbent: None,
            incumbent_obj: None,
            rens_first_incumbent_attempted: false,
        }
    }

    /// Adopt `res` as the new incumbent if it strictly improves the objective.
    /// Returns `true` when the incumbent changed.
    fn consider(&mut self, res: &SolverResult) -> bool {
        let better = match self.incumbent_obj {
            None => true,
            Some(o) => res.objective < o,
        };
        if better {
            self.incumbent_obj = Some(res.objective);
            self.incumbent = Some(res.clone());
        }
        better
    }
}

#[cfg(test)]
mod tests;
