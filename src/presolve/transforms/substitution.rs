//! Variable-elimination via Eq pivot row, shared by R6 (Step 6 doubleton),
//! R15 (Step 7 free-var), R5 (Step 8 free-singleton-col).
//!
//! Eliminating `x_j` from `piv_row` means substituting
//!   `x_j = (b_piv - Σ_{k≠j} a_{piv,k} x_k) / a_{piv,j}`
//! into every other active row and into the objective.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

/// Conservative fill-in budget: skip the substitution when the new nonzeros
/// would exceed `FILL_IN_FACTOR` times the nonzeros removed.
pub(super) fn fill_in_exceeds_budget(st: &PresolveState, piv_row: usize, j: usize) -> bool {
    const FILL_IN_FACTOR: usize = 3;
    let piv_others_cols: Vec<usize> = st.row_entries[piv_row]
        .iter()
        .filter(|&&(jj, v)| jj != j && !st.removed_cols[jj] && v.abs() >= ZERO_TOL)
        .map(|&(jj, _)| jj)
        .collect();
    let col_j_other_rows: Vec<usize> = st.col_entries[j]
        .iter()
        .filter(|&&(ii, v)| ii != piv_row && !st.removed_rows[ii] && v.abs() >= ZERO_TOL)
        .map(|&(ii, _)| ii)
        .collect();
    let mut new_entries: usize = 0;
    for &i in &col_j_other_rows {
        let existing: std::collections::HashSet<usize> = st.row_entries[i]
            .iter()
            .filter(|&&(_, v)| v.abs() >= ZERO_TOL)
            .map(|&(jj, _)| jj)
            .collect();
        for &k in &piv_others_cols {
            if !existing.contains(&k) {
                new_entries += 1;
            }
        }
    }
    let removed_nnz = 1 + piv_others_cols.len() + col_j_other_rows.len();
    new_entries > FILL_IN_FACTOR * removed_nnz.max(1)
}

pub(super) fn eliminate_variable_via_eq_row(
    st: &mut PresolveState,
    piv_row: usize,
    j: usize,
) -> Result<(), PresolveStatus> {
    debug_assert!(!st.removed_rows[piv_row]);
    debug_assert!(!st.removed_cols[j]);
    debug_assert_eq!(st.constraint_types[piv_row], ConstraintType::Eq);

    let pivot = st.coeff(piv_row, j);
    if pivot.abs() < ZERO_TOL {
        return Ok(());
    }
    let piv_b = st.b[piv_row];

    let piv_others: Vec<(usize, f64)> = st.row_entries[piv_row]
        .iter()
        .filter(|&&(jj, v)| jj != j && !st.removed_cols[jj] && v.abs() >= ZERO_TOL)
        .copied()
        .collect();

    let col_j_others: Vec<(usize, f64)> = st.col_entries[j]
        .iter()
        .filter(|&&(ii, v)| ii != piv_row && !st.removed_rows[ii] && v.abs() >= ZERO_TOL)
        .copied()
        .collect();

    // Dual-recovery snapshot taken before distribution: rows i where x_j is
    // eliminated drop out of col_entries[j] during the loop below, but LIFO
    // postsolve replays the snapshot in the order needed for y_piv.
    let col_orig_entries: Vec<(usize, f64)> = col_j_others.clone();
    let c_orig = st.c[j];

    for (i, a_ij) in col_j_others {
        st.b[i] -= a_ij * (piv_b / pivot);
        for &(k_col, a_pk) in &piv_others {
            let delta = -a_ij * a_pk / pivot;
            st.add_to_entry(i, k_col, delta);
        }
        st.add_to_entry(i, j, -a_ij);
    }

    if c_orig.abs() >= ZERO_TOL {
        st.obj_offset += c_orig * piv_b / pivot;
        for &(k_col, a_pk) in &piv_others {
            st.c[k_col] -= c_orig * a_pk / pivot;
        }
        st.c[j] = 0.0;
    }

    let others_for_postsolve: Vec<(usize, f64)> = piv_others.clone();
    st.postsolve_stack.push(PostsolveStep::LinearSubstitution {
        orig_col: j,
        orig_row: Some(piv_row),
        pivot,
        rhs: piv_b,
        others: others_for_postsolve,
        col_orig_entries,
        c_orig,
    });

    st.removed_rows[piv_row] = true;
    st.removed_cols[j] = true;

    Ok(())
}
