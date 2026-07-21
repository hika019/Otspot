//! Bounded dual simplex core (BFRT-aware) consuming `BoundedStandardForm`.
//!
//! Companion to `core.rs`. Unlike the legacy core (which sees variable upper
//! bounds as auxiliary `x_j + s = u_j` rows), this loop keeps the upper bound
//! as a non-basic state flag (`at_upper[j]`) and routes entering-variable
//! selection through `bfrt_select_entering` so non-basic columns whose bound
//! switch absorbs less infeasibility than a normal pivot can flip lb<->ub
//! mid-iter instead of forcing a small dual step.

pub(super) mod extract;
pub(super) mod iterate;
pub(super) mod leaving;
pub(super) mod pricing;
pub(super) mod primal;

#[cfg(test)]
mod tests;

// ── Re-exports for dual_advanced consumers ──────────────────────────────────

pub(crate) use extract::{extract_dual_info_bounded, extract_solution_bounded};
pub(crate) use iterate::iterate;
pub(crate) use primal::phase2_primal_bounded;

// Re-exports for tests (available via `use super::*` in tests.rs)
#[cfg(test)]
use extract::bounded_obj;
#[cfg(test)]
use iterate::ftran_column;
#[cfg(test)]
use leaving::{bland_entering, select_leaving_bland_bounded, select_leaving_bounded, BoundedLeave};
#[cfg(test)]
use pricing::{compute_reduced_costs_into_timed, DEADLINE_CHECK_INTERVAL};

#[cfg(test)]
pub(crate) use extract::set_at_upper_apply_disabled;
#[cfg(test)]
pub(crate) use iterate::set_flip_apply_disabled;
#[cfg(test)]
pub(crate) use pricing::{
    partial_price_cols_scanned, reset_partial_price_cols_scanned, set_partial_price_chunk_override,
    set_partial_price_single_window, RC_DEADLINE_CHECK_COUNT,
};
#[cfg(test)]
pub(crate) use primal::{set_primal_alpha_sv_disabled, set_primal_force_bland};

use super::super::pricing::DualLeavingStrategy;
use super::super::standard_form::{BoundedStandardForm, SimplexOutcome};
use crate::options::SolverOptions;
use crate::sparse::CscMatrix;

// Re-export items that tests obtain via `use super::*` (previously file-level imports)
#[cfg(test)]
use super::super::dual_common::basic_obj;
#[cfg(test)]
use extract::project_reduced_costs_to_active_bounds;

/// Terminal status of the bounded dual simplex iteration.
///
/// Distinct from the shared `SimplexOutcome`: the `UbViolationOutOfScope`
/// variant lets the wiring layer route to the legacy core deterministically
/// without confusing genuine deadlines with "this loop doesn't handle that
/// state". `Timeout`/`SingularBasis` retain their usual meaning.
#[derive(Debug)]
pub(crate) enum BoundedOutcome {
    /// Phase 1 dual optimal (perturbed cost).
    #[allow(dead_code)]
    Optimal(f64, Vec<f64>),
    Unbounded,
    /// Deadline or hard iteration cap. Carries the latest objective.
    Timeout(f64),
    SingularBasis,
    /// `x_B[r] > u_{basis[r]}` reached without an lb-violation candidate.
    #[allow(dead_code)]
    UbViolationOutOfScope {
        row: usize,
    },
}

/// Internal state of the bounded dual simplex iteration.
///
/// Field invariants:
/// - `basis.len() == m`, `x_b.len() == m`.
/// - `at_upper.len() == is_basic.len() == reduced_costs.len() == n_total`.
/// - For each basic column `j in basis`, `is_basic[j] == true`,
///   `at_upper[j] == false` (basic vars have no bound state).
/// - Non-basic vars: `at_upper[j]` indicates current bound; the non-basic
///   value is `0` (lb) or `upper_bounds[j]` (ub).
/// - `x_b[i] = (B^{-1} (b - sum_{k at_upper non-basic} u_k * a_k))[i]`,
///   reflecting the flip-applied effective RHS.
pub(crate) struct BoundedDualState {
    pub(crate) basis: Vec<usize>,
    pub(crate) at_upper: Vec<bool>,
    pub(crate) x_b: Vec<f64>,
    pub(crate) reduced_costs: Vec<f64>,
    pub(crate) is_basic: Vec<bool>,
    pub(crate) iterations: usize,
    /// Cyclic partial-pricing cursor for the bounded primal cores.
    pub(crate) price_start: usize,
}

impl BoundedDualState {
    /// Cold-start state from a `BoundedStandardForm`: slacks in the basis,
    /// every non-basic variable at its lower bound, `x_B = b`, dual feasibility
    /// achieved by cost perturbation `c_j = max(c_j, 0)` evaluated at `y = 0`.
    pub(crate) fn cold(bsf: &BoundedStandardForm, b_scaled: &[f64]) -> Self {
        let m = bsf.m;
        let n_total = bsf.n_total;
        assert_eq!(bsf.initial_basis.len(), m);
        let mut is_basic = vec![false; n_total];
        for &j in bsf.initial_basis.iter() {
            if j < n_total {
                is_basic[j] = true;
            }
        }
        Self {
            basis: bsf.initial_basis.clone(),
            at_upper: vec![false; n_total],
            x_b: b_scaled.to_vec(),
            reduced_costs: vec![0.0; n_total],
            is_basic,
            iterations: 0,
            price_start: 0,
        }
    }
}

/// Entry point: drives a cold-start bounded dual simplex on a Le-only
/// `BoundedStandardForm`.
pub(crate) fn solve_bounded_dual(
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    options: &SolverOptions,
    ubs: &[f64],
    leaving: &mut dyn DualLeavingStrategy,
) -> (BoundedOutcome, BoundedDualState) {
    let state = BoundedDualState::cold(bsf, b);
    iterate::iterate(state, bsf, a, c, options, ubs, leaving)
}

// ── Eq+UB dispatch counter (sentinel tests only) ────────────────────────────

#[cfg(test)]
thread_local! {
    static EQ_UB_DISPATCH_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_eq_ub_dispatch_count() {
    EQ_UB_DISPATCH_COUNT.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn eq_ub_dispatch_count() -> u64 {
    EQ_UB_DISPATCH_COUNT.with(|c| c.get())
}

#[cfg(test)]
pub(super) fn bump_eq_ub_dispatch_count() {
    EQ_UB_DISPATCH_COUNT.with(|c| c.set(c.get().saturating_add(1)));
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn bump_eq_ub_dispatch_count() {}

// ── Augmented bounded primal thin wrappers ──────────────────────────────────

/// Bounded primal simplex Phase I on augmented matrix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded_primal_phase1(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
) -> SimplexOutcome {
    primal::primal_simplex_aug(
        a_aug,
        c_aug,
        ubs_aug,
        n_struct,
        state,
        options,
        iters,
        Some(n_struct),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded_primal_phase2_aug(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
) -> SimplexOutcome {
    primal::primal_simplex_aug(a_aug, c_aug, ubs_aug, n_struct, state, options, iters, None)
}
