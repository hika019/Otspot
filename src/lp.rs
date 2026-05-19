//! LP-specific entry point.
//!
//! Splits LP from the QP `Q.is_zero` dispatch so that LP-only paths
//! (simplex, future IPM-first / crash / postsolve) are owned by this
//! module. `solve_qp_with(Q=0)` keeps backward compat by forwarding
//! here; the two call sites are distinguishable via `SolverResult.stats.route`.

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveRoute, SolverResult};

/// Solve an LP directly. Sets `result.stats.route = SolveRoute::LpDirect`.
pub fn solve_lp_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let mut result = crate::simplex::solve_with(problem, options);
    result.stats.route = SolveRoute::LpDirect;
    result
}

/// LP entry from `solve_qp_with(Q=0)`. Sets `result.stats.route = SolveRoute::LpForwardedFromQp`.
pub(crate) fn solve_lp_forwarded_from_qp(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let mut result = crate::simplex::solve_with(problem, options);
    result.stats.route = SolveRoute::LpForwardedFromQp;
    result
}
