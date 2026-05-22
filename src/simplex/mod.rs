//! Two-phase revised simplex with LU-based basis updates.

pub(crate) mod crash;
pub mod dual;
pub mod dual_advanced;
mod dual_common;
mod entry;
pub mod pricing;
pub(crate) mod primal;
mod standard_form;

pub use entry::{solve, solve_with};
pub(crate) use entry::{guard_lp_optimal, solve_without_presolve, with_lp_guard_disabled};

pub(crate) use primal::{extract_solution, revised_simplex_core, two_phase_simplex};
#[cfg(test)]
pub(crate) use primal::reconcile_final_basis_state;

pub(crate) use standard_form::{
    extract_dual_info, timeout_result_with_incumbent, SimplexOutcome, StandardForm,
};
pub(crate) use standard_form::build_standard_form;
#[cfg(test)]
pub(crate) use standard_form::OrigVarInfo;
pub(crate) use standard_form::{build_bounded_standard_form, scale_upper_bounds, BoundedStandardForm};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dual_advanced;
#[cfg(test)]
mod tests_bounded_form;
