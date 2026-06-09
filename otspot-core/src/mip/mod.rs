//! Mixed-integer programming (MILP / MIQP) via branch-and-bound.
//!
//! MILP (LP relaxation) and convex MIQP (QP relaxation) share one generic driver
//! (`solve_mip_with_stats`); the per-node solver is abstracted by `Relaxation`.
//! Pruning (`qp::global::pruning`) is reused from the spatial QP B&B. Non-convex
//! MIQP is out of scope and reported as [`SolveStatus::NonConvex`].

pub(crate) mod branch;
pub(crate) mod cuts;
pub(crate) mod heuristics;
pub(crate) mod node;
pub(crate) mod presolve;
mod problem;
pub(crate) mod queue;

pub use problem::{MilpProblem, MipProblemError, MiqpProblem};

use crate::options::{MipConfig, SolverOptions};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::global::pruning::{should_prune, within_gap};
use std::time::{Duration, Instant};

use branch::{
    branch_bounds, is_integer_feasible, select_branching_variable, split_integer_box,
    widest_splittable_integer,
};
use node::MipNode;
use queue::NodeQueue;

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
    /// Maximum branching depth reached.
    pub max_depth_seen: usize,
    /// Nodes discarded by bound/infeasibility before branching.
    pub pruned: usize,
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
    /// Cumulative LP postsolve microseconds across all nodes.
    pub lp_postsolve_us_total: u64,

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
        // Root presolve: coefficient propagation tightens integer bounds. Build the
        // effective problem (bound-tightened clone or the original) so the cut and
        // B&B stages share one code path.
        let problem_bt: MilpProblem = match presolve::tighten_integer_bounds(&problem.lp, &mask) {
            None => return (SolverResult::infeasible(), MipStats::default()),
            Some(tightened) if tightened != problem.lp.bounds => {
                let mut lp_bt = problem.lp.clone();
                lp_bt.bounds = tightened;
                MilpProblem {
                    lp: lp_bt,
                    integer_vars: problem.integer_vars.clone(),
                }
            }
            Some(_) => problem.clone(),
        };
        // Root GMI cuts tighten the LP relaxation without removing any
        // integer-feasible point, so the optimum is unchanged while the tree
        // shrinks. The added rows leave `num_vars` (hence `mask`) untouched.
        let effective = if cfg.cuts {
            cuts::add_root_cuts(&problem_bt, &opts_with_dl, cfg)
        } else {
            problem_bt
        };
        let fp_inc = heuristics::feasibility_pump::run_feasibility_pump(
            &effective.lp,
            &effective.integer_vars,
            cfg.integer_feas_tol,
            &opts_with_dl,
        );
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

    // deadline: prefer an explicit deadline, else derive from timeout_secs.
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

    // Degenerate: no integer variables → pure LP/QP passthrough.
    // Return before applying MIP-specific warm-start mutations so the caller's
    // `warm_start` and `recover_warm_start_basis` settings are preserved.
    if problem.integer_vars().is_empty() {
        return (problem.solve(problem.root_bounds(), &shared), stats);
    }

    // Enable basis recovery so LP solves return warm_start_basis for child nodes.
    shared.recover_warm_start_basis = true;
    shared.warm_start = None;

    let mut state = MipState::new();

    // Seed with an initial incumbent (e.g., from the feasibility pump heuristic).
    if let Some(inc) = initial_incumbent {
        if state.consider(&inc) {
            stats.incumbent_updates += 1;
            stats.fp_incumbent_found = true;
        }
    }

    let mut q = NodeQueue::new();
    // The root carries no valid lower bound yet (−∞): a bound is adopted only from an
    // Optimal relaxation. The loop solves the root uniformly with every other node, so
    // Infeasible / Unbounded / stalling roots are all handled in one place.
    q.push(MipNode::root(
        problem.root_bounds().to_vec(),
        f64::NEG_INFINITY,
    ));

    let mut open_lb = f64::INFINITY; // smallest valid bound over unexplored regions
    let mut had_open = false; // any region left unexplored?
    let mut proof_uncertain = false; // an unexplored region stems from a non-Optimal relaxation
    let mut deadline_stop = false;
    let mut maxnodes_stop = false;
    let mut unbounded = false;
    let mut root_solved = false; // first relaxation solve distinguishes root timing

    while let Some(node) = q.pop() {
        if deadline_hit(&deadline) {
            open_lb = open_lb.min(node.lower_bound);
            had_open = true;
            deadline_stop = true;
            break;
        }
        if stats.nodes_processed >= cfg.max_nodes {
            open_lb = open_lb.min(node.lower_bound);
            had_open = true;
            maxnodes_stop = true;
            break;
        }

        // Fathom by the inherited (parent) bound before spending a relaxation solve.
        if let Some(inc) = state.incumbent_obj {
            if should_prune(node.lower_bound, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                continue;
            }
        }

        let t0 = Instant::now();
        let res = if let Some(ref ws) = node.warm_start {
            let mut no = shared.clone();
            no.warm_start = Some(ws.clone());
            problem.solve(&node.var_bounds, &no)
        } else {
            problem.solve(&node.var_bounds, &shared)
        };
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        stats.nodes_processed += 1;
        stats.max_depth_seen = stats.max_depth_seen.max(node.depth);

        // Accumulate timing.
        stats.relaxation_time_total_ms += elapsed_ms;
        if !root_solved {
            stats.relaxation_time_root_ms = elapsed_ms;
            root_solved = true;
        } else {
            stats.relaxation_time_desc_ms += elapsed_ms;
        }
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
                continue;
            }
            SolveStatus::Unbounded => {
                unbounded = true;
                break;
            }
            SolveStatus::Timeout => {
                open_lb = open_lb.min(node.lower_bound);
                had_open = true;
                deadline_stop = true;
                break;
            }
            _ => {}
        }

        // Trust ONLY an exactly-Optimal relaxation (this includes the fixed-point
        // evaluator, which returns Optimal) as a lower bound. A SuboptimalSolution
        // primal objective is an UPPER bound on the relaxation optimum, NOT a lower
        // bound — using it to fathom would over-prune and could drop the true optimum
        // (the box-only off-diagonal silent-wrong).
        let trusted = matches!(res.status, SolveStatus::Optimal) && !res.solution.is_empty();

        if trusted {
            // Optimal relaxation objective is a valid lower bound for this region.
            // The first processed node is the root (best-bound queue, root pushed
            // first): record its bound to isolate root cut gap closure.
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

            // Integer-feasible Optimal relaxation → incumbent (feasible point, exact UB).
            if is_integer_feasible(&res.solution, &mask, cfg.integer_feas_tol) {
                if state.consider(&res) {
                    stats.incumbent_updates += 1;
                }
                continue; // integer-feasible leaf — nothing to branch on
            }

            if node.depth + 1 > cfg.max_depth {
                open_lb = open_lb.min(node_lb);
                had_open = true;
                continue;
            }

            // Guided branching on the most-fractional integer variable.
            let jb = select_branching_variable(
                &res.solution,
                &mask,
                cfg.integer_feas_tol,
                cfg.branching,
            )
            .expect("a non-integer-feasible Optimal relaxation has a fractional integer var");
            let (down, up) = branch_bounds(&node.var_bounds, jb, res.solution[jb]);
            // Propagate parent basis to children. Skip warm-start when the
            // branching variable's bound type changes (e.g. ub=∞→finite adds
            // a UB row in standard form, invalidating basis indices). The
            // up-branch typically triggers lb-violation and cold-starts anyway.
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
            q.push(node.child_warm(down, node_lb, down_ws));
            q.push(node.child_warm(up, node_lb, up_ws));
        } else {
            // Relaxation did not solve to Optimal: a SuboptimalSolution from an IPM stall
            // (box-only off-diagonal QP), MaxIterations, or NumericalError on a region
            // with no interior (e.g. an equality constraint pins it to a point). Its
            // objective is NOT a valid lower bound, so neither fathom nor tighten by it.
            // Fall back to integer-box bisection so the search still reaches an all-fixed
            // leaf (solved exactly by the fixed-point evaluator), never silently dropping
            // a region. The inherited (valid) parent bound is carried forward.
            match widest_splittable_integer(&node.var_bounds, &mask) {
                Some(_) if node.depth + 1 > cfg.max_depth => {
                    open_lb = open_lb.min(node.lower_bound);
                    had_open = true;
                    proof_uncertain = true;
                }
                Some(jb) => {
                    let (down, up) = split_integer_box(&node.var_bounds, jb);
                    q.push(node.child(down, node.lower_bound));
                    q.push(node.child(up, node.lower_bound));
                }
                None => {
                    // No integer box left to split and the relaxation is unsolvable
                    // (e.g. continuous variables in a no-interior region). The region is
                    // left unexplored without a reliable bound → mark the proof uncertain
                    // so the final status never falsely claims Optimal.
                    open_lb = open_lb.min(node.lower_bound);
                    had_open = true;
                    proof_uncertain = true;
                }
            }
        }
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

fn deadline_hit(deadline: &Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
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
