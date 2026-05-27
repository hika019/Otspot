// QpPresolveResult is the canonical early-exit sentinel for QP presolve steps.
// It is large (536 bytes) because it carries the reduced problem state, but it is
// a short-lived computation result never stored long-term; boxing would complicate
// every caller without reducing peak memory.
#![allow(clippy::result_large_err)]
//! QP presolve Phase 1: reductions for `min ½x'Qx + c'x  s.t. Ax ⋄ b, lb ≤ x ≤ ub`
//! plus the postsolve metadata to reverse them. Q is stored as the full
//! symmetric matrix.
//!
//! Steps: fix-var / singleton row / singleton col / empty / activity-redundancy
//! / free-var subst / parallel rows / implied bounds / dual-fixing / final
//! redundancy. Followed by large-coefficient rescaling and optional Ruiz scaling.

mod driver;
mod finalize;
mod helpers;
mod state;
mod steps_basic;
mod steps_bounds;
mod steps_free;
mod steps_parallel;
mod steps_redundancy;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_per_step;

pub use driver::run_qp_presolve_phase1;
pub use state::{QpPresolveResult, QpPresolveStatus};
pub(crate) use state::QpPostsolveStep;
