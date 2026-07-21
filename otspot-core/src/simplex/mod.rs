//! Two-phase revised simplex with LU-based basis updates.

pub(crate) mod crash;
pub mod dual;
pub mod dual_advanced;
mod dual_common;
mod entry;
pub mod pricing;
pub(crate) mod primal;
mod standard_form;
mod trace;

#[cfg(test)]
pub(crate) use entry::solve;
pub(crate) use entry::solve_with;
#[cfg(test)]
pub(crate) use entry::solve_without_presolve;

#[cfg(test)]
pub(crate) use primal::reconcile_final_basis_state;
#[cfg(test)]
pub(crate) use primal::OBJ_PROGRESS_RESET_COUNT;
pub(crate) use primal::{
    crossover_dual_from_primal, crossover_dual_from_primal_with_dual_warm_start, extract_solution,
    revised_simplex_core, two_phase_simplex,
};

pub(crate) use standard_form::build_standard_form;
#[cfg(test)]
pub(crate) use standard_form::OrigVarInfo;
pub(crate) use standard_form::{
    build_bounded_standard_form_with_deadline, scale_upper_bounds, BoundedStandardForm,
};
pub(crate) use standard_form::{
    external_stop_requested, extract_dual_info, stall_status, stop_result_with_incumbent,
    stop_status, SimplexOutcome, StandardForm,
};

/// Returns `true` when any basic variable value violates its lower bound.
/// Triggers cold-start fallback in the warm-start path.
#[inline]
pub(crate) fn has_lb_violation(x_b: &[f64], tol: f64) -> bool {
    x_b.iter().any(|&v| v < -tol)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_bounded_form;
#[cfg(test)]
mod tests_dual_advanced;
