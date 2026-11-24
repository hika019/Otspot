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
pub(crate) mod unbounded_ray;
pub(crate) mod warm_start;

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

/// The scaling triple the legacy (non-bounded) simplex paths consume: the
/// deterministic equilibration under `use_ruiz_scaling`, the matrix itself
/// with identity scales otherwise (a clone, since consumers take `&`-typed
/// inputs). Hoisting each branch's construction to the caller is what keeps
/// one copy from being pinned across a window where the caller already holds
/// the unscaled standard form and a deeper pass would build its own working
/// copy. Deadline loss surfaces as `None`; callers mint the same `Timeout`
/// shape the path they feed minted.
pub(crate) fn legacy_scale_triple(
    sf: &StandardForm,
    options: &crate::options::SolverOptions,
) -> Option<crate::presolve::LpScalingResult> {
    if options.use_ruiz_scaling {
        crate::presolve::LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c, options.deadline)
    } else {
        Some((
            sf.a.clone(),
            sf.b.clone(),
            sf.c.clone(),
            vec![1.0; sf.m],
            vec![1.0; sf.n_total],
        ))
    }
}
#[cfg(test)]
pub(crate) use standard_form::OrigVarInfo;
pub(crate) use standard_form::{
    build_bounded_standard_form_with_deadline, scale_upper_bounds, BoundedStandardForm,
};
#[cfg(test)]
pub(crate) use standard_form::{
    build_standard_form_call_count, reset_build_standard_form_call_count,
};
pub(crate) use standard_form::{
    external_stop_requested, extract_dual_info, stall_status, stop_result_with_incumbent,
    stop_status, SimplexOutcome, StandardForm,
};
pub(crate) use warm_start::apply_recovered_warm_start_basis;

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
