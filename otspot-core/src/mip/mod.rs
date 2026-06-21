//! Mixed-integer programming (MILP / MIQP) via branch-and-bound.
//!
//! MILP (LP relaxation) and convex MIQP (QP relaxation) share one generic driver
//! (`solve_mip_with_stats`); the per-node solver is abstracted by `Relaxation`.
//! Pruning (`qp::global::pruning`) is reused from the spatial QP B&B. Non-convex
//! MIQP is out of scope and reported as [`SolveStatus::NonConvex`].

pub(crate) mod branch;
pub(crate) mod conflict;
pub(crate) mod cuts;
pub(crate) mod heuristics;
pub(crate) mod node;
pub(crate) mod presolve;
mod problem;
pub(crate) mod queue;

pub use problem::{MilpProblem, MipProblemError, MiqpProblem};

use crate::linalg::timeout::deadline_reached;
use crate::options::{MipBranching, MipConfig, SolverOptions};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
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
    /// Return constraint data for per-node bound propagation, or `None` to skip it.
    ///
    /// MILP returns `Some((&A, &b, &constraint_types))` so the B&B driver can call
    /// [`presolve::tighten_bounds_at_node`] before each LP solve. MIQP returns `None`
    /// because it relies on per-node Ruiz scaling (presolve) for conditioning instead.
    fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
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

    /// Number of conflict clauses learned from infeasible nodes.
    pub conflict_clauses_learned: usize,
    /// Number of nodes pruned by conflict analysis (LP solve skipped).
    pub conflict_pruned: usize,
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
        // Run the feasibility pump on the original (bound-tightened) LP before
        // augmenting with cuts.  FP must see the unmodified constraint structure
        // so that the LP pump LPs and the final validation both use the original
        // bounds and Le/Ge rows, not the GMI cut rows added below.
        let fp_inc = heuristics::feasibility_pump::run_feasibility_pump(
            &problem_bt.lp,
            &problem_bt.integer_vars,
            cfg.integer_feas_tol,
            &opts_with_dl,
        );
        // Root GMI cuts tighten the LP relaxation without removing any
        // integer-feasible point, so the optimum is unchanged while the tree
        // shrinks. The added rows leave `num_vars` (hence `mask`) untouched.
        let effective = if cfg.cuts {
            cuts::add_root_cuts(&problem_bt, &opts_with_dl, cfg)
        } else {
            problem_bt
        };
        return solve_mip_core(&effective, &opts_with_dl, cfg, mask, fp_inc);
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
    pc: &mut PseudocostState,
) -> HashMap<usize, f64> {
    let mut scores = HashMap::new();
    let mut sb_opts = shared.clone();
    sb_opts.warm_start = None;

    for &j in candidates {
        let v = parent_sol[j];
        let (down_bounds, up_bounds) = branch::branch_bounds(parent_bounds, j, v);

        let r_down = problem.solve(&down_bounds, &sb_opts);
        let r_up = problem.solve(&up_bounds, &sb_opts);

        let down_ok = matches!(r_down.status, SolveStatus::Optimal | SolveStatus::Infeasible);
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

        if let Some(&k) = j_to_k.get(&j) {
            if d_down.is_finite() {
                pc.record_down(k, d_down);
            }
            if d_up.is_finite() {
                pc.record_up(k, d_up);
            }
        }

        let f_down = v - v.floor();
        let f_up = v.ceil() - v;
        let score = branch::pseudocost_score(d_down * f_down, d_up * f_up);
        scores.insert(j, score);
    }
    scores
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

    // Hybrid node selection state: counts best-bound pops between dives.
    let mut nodes_since_dive: usize = 0;
    let mut dive_start_depth: usize = 0;

    let mut conflicts = conflict::ConflictStore::new();
    let root_bounds = problem.root_bounds().to_vec();

    while let Some(mut node) = q.pop() {
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

        // Initiate a dive after every DIVE_FREQUENCY best-bound pops.
        // Frequency is higher when no incumbent exists (to find one sooner).
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

        if let Some(inc) = state.incumbent_obj {
            if should_prune(node.lower_bound, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                if q.is_diving() {
                    q.end_dive();
                }
                continue;
            }
        }

        // Conflict pruning: skip LP solve when this node's bounds subsume a
        // known infeasible region. Only applied to non-root nodes (depth > 0)
        // since the root has no branching-induced tightenings.
        if node.depth > 0 && conflicts.is_conflicted(&node.var_bounds) {
            stats.pruned += 1;
            stats.conflict_pruned += 1;
            continue;
        }

        // Per-node bound propagation (MILP only; MIQP skips via None).
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
                    stats.pruned += 1;
                    stats.propagation_pruned += 1;
                    continue;
                }
            }
        } else {
            None
        };
        let solve_bounds: &[(f64, f64)] = tightened.as_deref().unwrap_or(&node.var_bounds);

        let scale_before = crate::presolve::scaling::lp_scale_profile_snapshot();
        let fallback_before = crate::simplex::dual_advanced::fallback_profile_snapshot();
        let t0 = Instant::now();
        let res = if let Some(ref ws) = node.warm_start {
            let mut no = shared.clone();
            no.warm_start = Some(ws.clone());
            problem.solve(solve_bounds, &no)
        } else {
            problem.solve(solve_bounds, &shared)
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

        stats.nodes_processed += 1;
        stats.max_depth_seen = stats.max_depth_seen.max(node.depth);

        stats.relaxation_time_total_ms += elapsed_ms;
        if !root_solved {
            stats.relaxation_time_root_ms = elapsed_ms;
            stats.lp_scale_us_root += scale_delta.scale_us;
            stats.lp_scale_calls_root += scale_delta.calls;
            if let Some(tb) = res.timing_breakdown {
                stats.lp_solve_us_root += tb.solve_us;
            }
            root_solved = true;
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

        match res.status {
            SolveStatus::Infeasible => {
                stats.pruned += 1;
                if q.is_diving() {
                    q.end_dive();
                }
                // Learn a conflict clause from this infeasible node's bounds.
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

        let trusted = matches!(res.status, SolveStatus::Optimal) && !res.solution.is_empty();

        if trusted {
            // Update pseudocost from this node's objective improvement over parent.
            if use_reliability {
                if let Some(jb) = node.branch_var {
                    if let Some(&k) = j_to_k.get(&jb) {
                        let delta = (res.objective - node.parent_obj).max(0.0);
                        if node.branch_up {
                            pc.record_up(k, delta);
                        } else {
                            pc.record_down(k, delta);
                        }
                    }
                }
            }

            if stats.nodes_processed == 1 {
                stats.root_lp_bound = res.objective;
            }
            let node_lb = node.lower_bound.max(res.objective);

            if let Some(inc) = state.incumbent_obj {
                if should_prune(node_lb, Some(inc), cfg.gap_tol) {
                    stats.pruned += 1;
                    continue;
                }
            }

            if is_integer_feasible(&res.solution, &mask, cfg.integer_feas_tol) {
                if state.consider(&res) {
                    stats.incumbent_updates += 1;
                }
                if q.is_diving() {
                    q.end_dive();
                }
                continue; // integer-feasible leaf — nothing to branch on
            }

            // RINS: every RINS_INTERVAL nodes, if an incumbent exists, try to
            // improve it by solving a sub-MIP over the free (non-agreeing) variables.
            if cfg.rins_enabled
                && stats.nodes_processed.is_multiple_of(heuristics::rins::RINS_INTERVAL)
                && state.incumbent_obj.is_some()
            {
                if let Some(ref inc_sol) = state.incumbent {
                    stats.rins_calls += 1;
                    if let Some(rins_res) =
                        problem.run_rins(&res.solution, &inc_sol.solution, cfg, &deadline)
                    {
                        if state.consider(&rins_res) {
                            stats.incumbent_updates += 1;
                            stats.rins_improvements += 1;
                        }
                    }
                }
            }

            if node.depth + 1 > cfg.max_depth {
                open_lb = open_lb.min(node_lb);
                had_open = true;
                if q.is_diving() {
                    q.end_dive();
                }
                continue;
            }

            // Terminate the dive when it has descended MAX_DIVE_DEPTH levels below
            // the pivot.  Children are still pushed, but to the heap (best-bound mode).
            if q.is_diving() && node.depth >= dive_start_depth + MAX_DIVE_DEPTH {
                q.end_dive();
            }

            // Reduced-cost fixing: tighten integer-variable bounds for children.
            if let Some(inc_obj) = state.incumbent_obj {
                let fixed = reduced_cost_fixing(
                    &res,
                    inc_obj,
                    &mut node.var_bounds,
                    problem.integer_vars(),
                );
                stats.rc_vars_fixed += fixed;
            }

            // Select branching variable.
            let jb = if use_reliability {
                // Run strong branching for unreliable candidates.
                let sb_cands =
                    strong_branch_candidates(&res.solution, &mask, &integer_vars, cfg.integer_feas_tol, &pc);
                let strong_scores = if !sb_cands.is_empty() && !deadline_reached(deadline) {
                    measure_strong_branch_scores(
                        problem,
                        &node.var_bounds,
                        &res.solution,
                        res.objective,
                        &sb_cands,
                        &j_to_k,
                        &shared,
                        &mut pc,
                    )
                } else {
                    HashMap::new()
                };
                select_branching_variable_reliability(
                    &res.solution,
                    &mask,
                    &integer_vars,
                    cfg.integer_feas_tol,
                    &pc,
                    if strong_scores.is_empty() {
                        None
                    } else {
                        Some(&strong_scores)
                    },
                )
                .expect("a non-integer-feasible Optimal relaxation has a fractional integer var")
            } else {
                select_branching_variable(
                    &res.solution,
                    &mask,
                    cfg.integer_feas_tol,
                    cfg.branching,
                )
                .expect("a non-integer-feasible Optimal relaxation has a fractional integer var")
            };


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
            q.push(node.child_branched(down, node_lb, down_ws, jb, false, res.objective));
            q.push(node.child_branched(up, node_lb, up_ws, jb, true, res.objective));
        } else {
            match widest_splittable_integer(&node.var_bounds, &mask) {
                Some(_) if node.depth + 1 > cfg.max_depth => {
                    open_lb = open_lb.min(node.lower_bound);
                    had_open = true;
                    proof_uncertain = true;
                    if q.is_diving() {
                        q.end_dive();
                    }
                }
                Some(jb) => {
                    if q.is_diving() && node.depth >= dive_start_depth + MAX_DIVE_DEPTH {
                        q.end_dive();
                    }
                    let (down, up) = split_integer_box(&node.var_bounds, jb);
                    q.push(node.child(down, node.lower_bound));
                    q.push(node.child(up, node.lower_bound));
                }
                None => {
                    open_lb = open_lb.min(node.lower_bound);
                    had_open = true;
                    proof_uncertain = true;
                    if q.is_diving() {
                        q.end_dive();
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
            // Proven optimal only when (a) no unexplored region can beat the incumbent
            // by more than the gap (within_gap on a *valid* lower bound), AND (b) no
            // region was left unresolved by a non-Optimal relaxation. Otherwise we still
            // return the incumbent (possibly suboptimal) but never disguise it as a
            // proven optimum.
            let proven = !proof_uncertain && within_gap(inc_obj, remaining_lb, cfg.gap_tol);
            inc.solution = round_integers(inc.solution, problem.integer_vars());
            inc.status = if proven {
                // Clamp remaining_lb to inc_obj: when the queue is fully drained
                // (no open regions), remaining_lb = ∞ and the effective lower bound
                // equals the incumbent itself (gap = 0).
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

/// Boolean mask of length `num_vars`; `true` where the variable is integral.
pub(crate) fn integer_mask(num_vars: usize, integer_vars: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; num_vars];
    for &j in integer_vars {
        if j < num_vars {
            mask[j] = true;
        }
    }
    mask
}

/// A result tagging a non-convex (non-PSD `Q`) MIQP as out of scope.
fn nonconvex_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::NonConvex(
            "convex MIQP only: Q is not positive semidefinite".to_string(),
        ),
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

/// Round the integer components of `sol` to exact integers (relaxation noise removal).
fn round_integers(mut sol: Vec<f64>, integer_vars: &[usize]) -> Vec<f64> {
    for &j in integer_vars {
        if j < sol.len() {
            sol[j] = sol[j].round();
        }
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
/// - `x[j] ≈ lb[j]` and `rc[j] > gap`:  fix `x[j] = lb[j]`
/// - `x[j] ≈ ub[j]` and `-rc[j] > gap`: fix `x[j] = ub[j]`
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
    let gap = incumbent_obj - lp_result.objective;
    if gap <= 0.0 {
        return 0;
    }
    let rc = &lp_result.reduced_costs;
    let x = &lp_result.solution;
    let mut count = 0usize;
    for &j in integer_vars {
        if j >= node_bounds.len() || j >= rc.len() || j >= x.len() {
            continue;
        }
        let (lb, ub) = node_bounds[j];
        if (lb - ub).abs() < BOUND_AT_TOL {
            continue; // already fixed
        }
        let xj = x[j];
        let rcj = rc[j];
        if (xj - lb).abs() <= BOUND_AT_TOL && rcj > gap {
            node_bounds[j] = (lb, lb);
            count += 1;
        } else if (xj - ub).abs() <= BOUND_AT_TOL && -rcj > gap {
            node_bounds[j] = (ub, ub);
            count += 1;
        }
    }
    count
}

/// Incumbent (best integer-feasible upper bound) tracking.
struct MipState {
    incumbent: Option<SolverResult>,
    incumbent_obj: Option<f64>,
}

impl MipState {
    fn new() -> Self {
        Self {
            incumbent: None,
            incumbent_obj: None,
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
