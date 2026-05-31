//! Steps 7 + 8 (R15 + R5): free-variable substitution and free-singleton-column.

use super::state::{PresolveState, PresolveStatus};
use super::substitution::{eliminate_variable_via_eq_row, fill_in_exceeds_budget};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step7_free_var_substitution(
    st: &mut PresolveState,
    new_subst: &mut usize,
    deadline: Option<std::time::Instant>,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if st.removed_cols[j] {
            continue;
        }
        // Use the original bounds: bounds tightening should not disqualify R15.
        let (orig_lb, orig_ub) = st.orig_bounds[j];
        if orig_lb != f64::NEG_INFINITY || orig_ub != f64::INFINITY {
            continue;
        }
        let col_entries = st.active_col_entries(j);
        if col_entries.is_empty() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for &(i, a_ij) in &col_entries {
            if st.constraint_types[i] != ConstraintType::Eq {
                continue;
            }
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            match best {
                None => best = Some((i, a_ij)),
                Some((_, ba)) => {
                    if a_ij.abs() > ba.abs() {
                        best = Some((i, a_ij));
                    }
                }
            }
        }
        if let Some((piv_row, _)) = best {
            if fill_in_exceeds_budget(st, piv_row, j) {
                continue;
            }
            eliminate_variable_via_eq_row(st, piv_row, j)?;
            *new_subst += 1;
        }
    }
    Ok(())
}

/// Step 8 (R5): x_j is the only active variable in one row and is free on both sides.
/// One-sided-free is intentionally deferred to Step 7 / the IPM — the cost-sign /
/// inequality-direction interaction makes safe Eq promotion subtle.
pub(super) fn step8_free_singleton_col(
    st: &mut PresolveState,
    new_subst: &mut usize,
    deadline: Option<std::time::Instant>,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if st.removed_cols[j] {
            continue;
        }
        let (orig_lb, orig_ub) = st.orig_bounds[j];
        let col_entries = st.active_col_entries(j);
        if col_entries.len() != 1 {
            continue;
        }
        let (i, a_ij) = col_entries[0];
        if a_ij.abs() < ZERO_TOL {
            continue;
        }
        let ct = st.constraint_types[i];
        if orig_lb != f64::NEG_INFINITY || orig_ub != f64::INFINITY {
            continue;
        }
        // Free-on-both-sides: Le/Ge promotion to Eq is unsafe in general, so restrict to Eq.
        if ct != ConstraintType::Eq {
            continue;
        }
        if fill_in_exceeds_budget(st, i, j) {
            continue;
        }
        eliminate_variable_via_eq_row(st, i, j)?;
        *new_subst += 1;
    }
    Ok(())
}
