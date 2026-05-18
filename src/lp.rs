//! LP-specific entry point.
//!
//! Splits LP from the QP `Q.is_zero` dispatch so that LP-only paths
//! (simplex, future IPM-first / crash / postsolve) are owned by this
//! module. `solve_qp_with(Q=0)` keeps backward compat by forwarding
//! here, but the two call sites are distinguishable via telemetry.

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolverResult};

/// LP entry call counters.
///
/// `direct` and `forwarded_from_qp` are tracked separately so a
/// regression of `Model::solve` LP path onto `solve_qp_with` is
/// observable (direct stays 0 while forwarded increases).
pub mod telemetry {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static LP_DIRECT_CALLS: AtomicU64 = AtomicU64::new(0);
    pub(super) static LP_FORWARDED_FROM_QP_CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn lp_direct_calls() -> u64 {
        LP_DIRECT_CALLS.load(Ordering::Relaxed)
    }

    pub fn lp_forwarded_from_qp_calls() -> u64 {
        LP_FORWARDED_FROM_QP_CALLS.load(Ordering::Relaxed)
    }

    pub fn reset() {
        LP_DIRECT_CALLS.store(0, Ordering::Relaxed);
        LP_FORWARDED_FROM_QP_CALLS.store(0, Ordering::Relaxed);
    }
}

pub fn solve_lp_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    telemetry::LP_DIRECT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::simplex::solve_with(problem, options)
}

/// LP entry from `solve_qp_with(Q=0)`. Same computation as
/// `solve_lp_with` but counted separately for telemetry.
pub(crate) fn solve_lp_forwarded_from_qp(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    telemetry::LP_FORWARDED_FROM_QP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::simplex::solve_with(problem, options)
}
