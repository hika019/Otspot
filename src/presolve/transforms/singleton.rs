//! Step 2: singleton-Eq row — a single variable can be solved exactly.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step2_singleton_row(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        if st.constraint_types[i] != ConstraintType::Eq {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() == 1 {
            let (j, a_ij) = active[0];
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let value = st.b[i] / a_ij;
            let (lb, ub) = st.bounds[j];
            if value < lb - ZERO_TOL || value > ub + ZERO_TOL {
                return Err(PresolveStatus::Infeasible);
            }
            let value = value.clamp(lb, ub);
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] && row != i {
                    st.b[row] -= val * value;
                }
            }
            let c_j_snapshot = st.c[j];
            st.obj_offset += c_j_snapshot * value;
            st.removed_cols[j] = true;
            st.removed_rows[i] = true;
            st.postsolve_stack.push(PostsolveStep::SingletonRow {
                orig_row: i,
                orig_col: j,
                value,
            });
        }
    }
    Ok(())
}
