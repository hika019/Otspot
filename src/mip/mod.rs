//! Mixed-integer programming (MILP / MIQP) via branch-and-bound (#14).
//!
//! # Approach
//! Each B&B node solves the **continuous relaxation** (LP for MILP, convex QP for
//! MIQP) over the node's variable bounds. Integer branching tightens one integer
//! variable's bounds (`x_j <= floor(v)` / `x_j >= ceil(v)`) — the same node
//! mechanism the spatial QP B&B uses for box splitting, so relaxations are solved
//! by swapping the bounds vector. The relaxation objective is a valid lower bound;
//! integer-feasible relaxation solutions are upper-bound incumbents. Fathoming
//! reuses the spatial B&B's relative-gap pruning (`qp::global::pruning`).
//!
//! # Reuse vs. duplication
//! - `pruning::{should_prune, within_gap}` is shared directly (pure `f64` logic).
//! - The best-bound priority queue ([`queue::NodeQueue`]) is an intentional small
//!   duplicate of `qp::global::tree::BBTree` (see `queue.rs` for the rationale).
//! - The driver loop mirrors the proven spatial-B&B structure but the per-node
//!   semantics differ (relaxation-objective lower bound vs. interval bound;
//!   integer-feasible incumbents vs. any feasible point).
//!
//! Phase 1 implements MILP; convex MIQP is added in Phase 2.

pub(crate) mod branch;
pub(crate) mod node;
mod problem;
pub(crate) mod queue;

pub use problem::{MilpProblem, MipProblemError};

use crate::options::{MipConfig, SolverOptions};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::qp::global::pruning::{should_prune, within_gap};
use std::time::{Duration, Instant};

use branch::{branch_bounds, is_integer_feasible, select_branching_variable};
use node::MipNode;
use queue::NodeQueue;

/// Search statistics for sentinel/regression tests (not part of the production API).
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
}

/// Solve a MILP to (relative) ε-optimality via branch-and-bound.
pub fn solve_milp(problem: &MilpProblem, options: &SolverOptions, cfg: &MipConfig) -> SolverResult {
    solve_milp_with_stats(problem, options, cfg).0
}

/// Like [`solve_milp`] but also returns search statistics (test sentinel hook).
pub fn solve_milp_with_stats(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    let mut stats = MipStats::default();
    let mask = problem.integer_mask();

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

    let root_bounds = problem.lp.bounds.clone();
    let root = solve_relaxation(&problem.lp, &root_bounds, &shared);
    stats.nodes_processed = 1;

    // Degenerate: no integer variables → the relaxation is the answer (LP fallback).
    if problem.integer_vars.is_empty() {
        return (root, stats);
    }

    // The root relaxation must solve to optimality to drive a valid bound.
    // Infeasible (→ MILP infeasible), Unbounded (→ MILP unbounded), Timeout /
    // NumericalError / MaxIterations all propagate as-is.
    if !matches!(root.status, SolveStatus::Optimal) {
        return (root, stats);
    }

    // Root already integer-feasible → proven optimal.
    if is_integer_feasible(&root.solution, &mask, cfg.integer_feas_tol) {
        let mut r = root;
        r.solution = round_integers(r.solution, &problem.integer_vars);
        r.status = SolveStatus::Optimal;
        return (r, stats);
    }

    let mut state = MipState::new();
    let mut q = NodeQueue::new();

    // Branch the root on its most-fractional integer variable; children inherit the
    // root relaxation objective as their (valid) lower bound.
    let root_node = MipNode::root(root_bounds.clone(), root.objective);
    let j = select_branching_variable(&root.solution, &mask, cfg.integer_feas_tol, cfg.branching)
        .expect("fractional root must have a branch variable");
    let (down, up) = branch_bounds(&root_bounds, j, root.solution[j]);
    q.push(root_node.child(down, root.objective));
    q.push(root_node.child(up, root.objective));

    // Smallest lower bound over regions left unexplored (depth-limited / interrupted
    // nodes that are not in the queue). Combined with the queue frontier at the end.
    let mut open_lb = f64::INFINITY;
    let mut deadline_stop = false;
    let mut maxnodes_stop = false;
    let mut depth_limited = false;
    let mut unbounded = false;

    while let Some(node) = q.pop() {
        if deadline_hit(&deadline) {
            open_lb = open_lb.min(node.lower_bound);
            deadline_stop = true;
            break;
        }
        if stats.nodes_processed >= cfg.max_nodes {
            open_lb = open_lb.min(node.lower_bound);
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

        let res = solve_relaxation(&problem.lp, &node.var_bounds, &shared);
        stats.nodes_processed += 1;
        stats.max_depth_seen = stats.max_depth_seen.max(node.depth);

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
                deadline_stop = true;
                break;
            }
            _ => {}
        }
        // Remaining: Optimal / MaxIterations / SuboptimalSolution / NumericalError.
        if res.solution.is_empty() {
            // No usable point (e.g. NumericalError with no solution) → discard region.
            open_lb = open_lb.min(node.lower_bound);
            continue;
        }

        // Only an optimally solved relaxation yields a trustworthy lower bound;
        // otherwise keep the parent's valid bound.
        let is_optimal = matches!(res.status, SolveStatus::Optimal);
        let node_lb = if is_optimal {
            node.lower_bound.max(res.objective)
        } else {
            node.lower_bound
        };

        // Fathom by the freshly computed (valid) bound.
        if let Some(inc) = state.incumbent_obj {
            if should_prune(node_lb, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                continue;
            }
        }

        // Incumbents are taken only from optimally solved relaxations (guaranteed
        // feasible + exact objective). A degraded point is never reported as a solution.
        if is_optimal && is_integer_feasible(&res.solution, &mask, cfg.integer_feas_tol) {
            if state.consider(&res) {
                stats.incumbent_updates += 1;
            }
            continue; // integer-feasible leaf — nothing to branch on
        }

        if node.depth + 1 > cfg.max_depth {
            open_lb = open_lb.min(node_lb);
            depth_limited = true;
            continue;
        }

        match select_branching_variable(&res.solution, &mask, cfg.integer_feas_tol, cfg.branching) {
            Some(jb) => {
                let (down, up) = branch_bounds(&node.var_bounds, jb, res.solution[jb]);
                q.push(node.child(down, node_lb));
                q.push(node.child(up, node_lb));
            }
            None => {
                // No fractional integer var but the relaxation was not Optimal, so the
                // integer-feasible point is not trusted as an incumbent. Region stays open.
                open_lb = open_lb.min(node_lb);
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
            let proven = within_gap(inc_obj, remaining_lb, cfg.gap_tol);
            inc.solution = round_integers(inc.solution, &problem.integer_vars);
            inc.status = if proven {
                SolveStatus::Optimal
            } else if deadline_stop {
                SolveStatus::Timeout
            } else {
                // budget (max_nodes / max_depth) exhausted with an unproven incumbent
                SolveStatus::SuboptimalSolution
            };
            (inc, stats)
        }
        None => (
            finalize_no_incumbent(interrupted, depth_limited, q.is_empty(), open_lb, deadline_stop),
            stats,
        ),
    }
}

/// Classify the terminal status when no integer-feasible incumbent was found.
///
/// Infeasibility may be claimed **only** when the whole tree was resolved: the
/// queue is empty, there was no budget interruption, no depth limiting, **and**
/// no region was left open by an untrusted/degraded relaxation (`open_lb` still
/// `+∞`). A node whose relaxation was integer-feasible-but-not-`Optimal`
/// (untrusted, not adopted as incumbent) or returned a degraded status with no
/// usable point lowers `open_lb` to a finite value without setting a flag — that
/// region is genuinely unresolved, so reporting `Infeasible` there would be a
/// silent wrong answer. In that case return a no-solution status instead.
fn finalize_no_incumbent(
    interrupted: bool,
    depth_limited: bool,
    queue_empty: bool,
    open_lb: f64,
    deadline_stop: bool,
) -> SolverResult {
    let fully_resolved =
        !interrupted && !depth_limited && queue_empty && open_lb.is_infinite();
    if fully_resolved {
        SolverResult::infeasible()
    } else if deadline_stop {
        no_solution_result(SolveStatus::Timeout)
    } else {
        no_solution_result(SolveStatus::MaxIterations)
    }
}

/// Solve the relaxation over `bounds` by cloning the LP with swapped bounds.
fn solve_relaxation(lp: &LpProblem, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
    let mut sub = lp.clone();
    sub.bounds = bounds.to_vec();
    crate::lp::solve_lp_with(&sub, opts)
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
        Self { incumbent: None, incumbent_obj: None }
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
