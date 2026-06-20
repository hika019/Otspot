//! Solution extraction and objective utilities for bounded simplex.

use super::BoundedDualState;
use crate::problem::LpProblem;
use crate::simplex::standard_form::BoundedStandardForm;
use crate::tolerances::feas_rel_tol;

#[cfg(test)]
thread_local! {
    static AT_UPPER_APPLY_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_at_upper_apply_disabled(v: bool) {
    AT_UPPER_APPLY_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn at_upper_apply_disabled() -> bool {
    AT_UPPER_APPLY_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn at_upper_apply_disabled() -> bool {
    false
}

/// Recover the full primal vector from a bounded dual terminal state.
///
/// Unlike `extract_solution` (which sets every non-basic value to 0), this
/// function accounts for variables that are non-basic **at their upper bound**:
/// - basic j = basis[i]: `x_new[j] = x_b[i] * col_scale[j]`
/// - non-basic at lb (`at_upper[j] = false`): `x_new[j] = 0`
/// - non-basic at ub (`at_upper[j] = true`): `x_new[j] = upper_bounds[j]`
///   (no col_scale needed: `upper_bounds` lives in the pre-Ruiz-scale space,
///   so the scale factors cancel)
///
/// The result is mapped back to original variables via `orig_var_info`
/// with double-double arithmetic for free-variable split cancellation.
pub(crate) fn extract_solution_bounded(
    bsf: &BoundedStandardForm,
    state: &BoundedDualState,
    col_scale: &[f64],
) -> Vec<f64> {
    use twofloat::TwoFloat;
    let mut x_new = vec![0.0f64; bsf.n_shifted];

    for i in 0..bsf.m {
        let j = state.basis[i];
        if j < bsf.n_shifted {
            let scale = col_scale.get(j).copied().unwrap_or(1.0);
            x_new[j] = state.x_b[i] * scale;
        }
    }

    if !at_upper_apply_disabled() {
        for j in 0..bsf.n_shifted {
            if !state.is_basic[j] && state.at_upper[j] {
                x_new[j] = bsf.upper_bounds[j];
            }
        }
    }

    let mut solution = vec![0.0f64; bsf.n_orig];
    for (orig_j, sol_j) in solution.iter_mut().enumerate() {
        let info = &bsf.orig_var_info[orig_j];
        let mut value = TwoFloat::from(info.offset);
        for &(new_idx, coeff) in &info.new_vars {
            value += TwoFloat::new_mul(coeff, x_new[new_idx]);
        }
        *sol_j = f64::from(value);
    }
    solution
}

/// Recover original-problem duals, reduced costs, and slack from a bounded
/// dual terminal state. Mirrors `extract_dual_info` but operates on
/// `BoundedStandardForm` (no UB rows ⇒ `bsf.m == m_orig`).
pub(crate) fn extract_dual_info_bounded(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    y_std: &[f64],
    solution: &[f64],
    row_scale: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_orig = bsf.m;
    let n_orig = bsf.n_orig;

    let mut dual_solution = vec![0.0f64; m_orig];
    for i in 0..m_orig {
        let sign = if bsf.row_negated[i] { -1.0 } else { 1.0 };
        let rs = row_scale.get(i).copied().unwrap_or(1.0);
        dual_solution[i] = sign * rs * y_std[i];
    }

    let mut slack = problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    let mut reduced_costs = problem.c.clone();
    for (j, rc_j) in reduced_costs.iter_mut().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc_j -= dual_solution[row] * vals[k];
            }
        }
    }
    project_reduced_costs_to_active_bounds(problem, solution, &mut reduced_costs);

    (dual_solution, reduced_costs, slack)
}

fn active_at_bound(x: f64, bound: f64) -> bool {
    bound.is_finite() && (x - bound).abs() <= feas_rel_tol() * (1.0 + x.abs() + bound.abs())
}

pub(super) fn project_reduced_costs_to_active_bounds(
    problem: &LpProblem,
    solution: &[f64],
    reduced_costs: &mut [f64],
) {
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(problem.num_vars) {
        if j >= solution.len() {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let at_lb = active_at_bound(solution[j], lb);
        let at_ub = active_at_bound(solution[j], ub);
        *rc = if at_lb && !at_ub {
            rc.max(0.0)
        } else if at_ub && !at_lb {
            rc.min(0.0)
        } else if at_lb && at_ub {
            *rc
        } else {
            0.0
        };
    }
}

/// Objective including non-basic at-upper-bound contributions.
///
/// Full objective: c_B^T x_B + Σ_{j non-basic at ub} c_j · u_j.
/// Invariant: `at_upper[j] ⇒ !is_basic[j]`, maintained by `iterate` /
/// `phase2_primal_bounded`. `debug_assert` traps violations in debug/test
/// builds; release builds rely on callers maintaining the invariant.
pub(super) fn bounded_obj(
    c: &[f64],
    basis: &[usize],
    x_b: &[f64],
    at_upper: &[bool],
    is_basic: &[bool],
    ubs: &[f64],
) -> f64 {
    debug_assert_eq!(
        at_upper.len(),
        is_basic.len(),
        "at_upper/is_basic length mismatch"
    );
    debug_assert_eq!(at_upper.len(), c.len(), "at_upper/c length mismatch");
    debug_assert_eq!(at_upper.len(), ubs.len(), "at_upper/ubs length mismatch");
    debug_assert_eq!(basis.len(), x_b.len(), "basis/x_b length mismatch");
    let basic: f64 = basis.iter().zip(x_b.iter()).map(|(&j, &v)| c[j] * v).sum();
    let at_ub: f64 = at_upper
        .iter()
        .enumerate()
        .filter(|&(_, &flag)| flag)
        .inspect(|&(j, _)| {
            debug_assert!(
                !is_basic[j],
                "invariant at_upper[j] => !is_basic[j] violated at j={j}"
            )
        })
        .map(|(j, _)| c[j] * ubs[j])
        .sum();
    basic + at_ub
}
