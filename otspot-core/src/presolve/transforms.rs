//! LP presolve: collects 11 reductions and the inverse metadata for postsolve.
//!
//! Fixpoint loop over `MAX_PRESOLVE_ITER` passes of:
//! 1. Fixed variable (lb == ub)
//! 2. Singleton Eq row → variable value
//!    3a/3b. Empty row / column
//! 4. Redundant row from activity
//! 5. Bounds tightening
//! 6. Doubleton Eq (R6)
//! 7. Free-variable substitution (R15)
//! 8. Free singleton column (R5)
//!    9–11 live in `transforms_dup.rs` (parallel row / dup-dom col / dual fixing).

mod bounds;
mod doubleton;
mod driver;
mod empty_redundant;
mod fixed;
mod free;
mod singleton;
mod state;
mod substitution;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_per_step;

pub use driver::{run_presolve, run_presolve_with_flags};
pub(crate) use state::{PostsolveStep, PresolveState};
pub use state::{PresolveFlags, PresolveResult, PresolveStatus};
