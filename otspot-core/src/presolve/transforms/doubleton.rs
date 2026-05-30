//! Step 6 (R6): doubleton-Eq row — eliminate one of the two variables.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use super::substitution::{eliminate_variable_via_eq_row, fill_in_exceeds_budget};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step6_doubleton_equation(
    st: &mut PresolveState,
    new_subst: &mut usize,
) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        if st.constraint_types[i] != ConstraintType::Eq {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() != 2 {
            continue;
        }
        let (j1, a1) = active[0];
        let (j2, a2) = active[1];
        if a1.abs() < ZERO_TOL || a2.abs() < ZERO_TOL {
            continue;
        }
        // Pivot choice: prefer the original-free side (avoids postsolve bound checks);
        // otherwise pick the larger-magnitude coefficient for numerical stability.
        let j1_free =
            st.orig_bounds[j1].0 == f64::NEG_INFINITY && st.orig_bounds[j1].1 == f64::INFINITY;
        let j2_free =
            st.orig_bounds[j2].0 == f64::NEG_INFINITY && st.orig_bounds[j2].1 == f64::INFINITY;
        let (pivot_col, pivot_a, other_col, other_a) = if j1_free && !j2_free {
            (j1, a1, j2, a2)
        } else if j2_free && !j1_free {
            (j2, a2, j1, a1)
        } else if a1.abs() >= a2.abs() {
            (j1, a1, j2, a2)
        } else {
            (j2, a2, j1, a1)
        };
        let (lb_p, ub_p) = st.bounds[pivot_col];
        let (lb_o_old, ub_o_old) = st.bounds[other_col];
        let ratio = pivot_a / other_a;
        let bo = st.b[i] / other_a;
        let (other_lb_impl, other_ub_impl) = if ratio > 0.0 {
            let lo = if ub_p == f64::INFINITY {
                f64::NEG_INFINITY
            } else {
                bo - ratio * ub_p
            };
            let hi = if lb_p == f64::NEG_INFINITY {
                f64::INFINITY
            } else {
                bo - ratio * lb_p
            };
            (lo, hi)
        } else if ratio < 0.0 {
            let lo = if lb_p == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                bo - ratio * lb_p
            };
            let hi = if ub_p == f64::INFINITY {
                f64::INFINITY
            } else {
                bo - ratio * ub_p
            };
            (lo, hi)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        let new_lb_o = lb_o_old.max(other_lb_impl);
        let new_ub_o = ub_o_old.min(other_ub_impl);
        if new_lb_o > new_ub_o + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if fill_in_exceeds_budget(st, i, pivot_col) {
            continue;
        }
        if (new_lb_o - lb_o_old).abs() > ZERO_TOL || (new_ub_o - ub_o_old).abs() > ZERO_TOL {
            st.postsolve_stack.push(PostsolveStep::BoundsTightened);
            st.bounds[other_col] = (new_lb_o, new_ub_o);
        }
        eliminate_variable_via_eq_row(st, i, pivot_col)?;
        *new_subst += 1;
    }
    Ok(())
}
