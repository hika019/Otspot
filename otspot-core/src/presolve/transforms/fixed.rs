//! Step 1: fix variables whose lower and upper bounds coincide.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::tolerances::ZERO_TOL;

pub(super) fn step1_fixed_variable(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let (lb, ub) = st.bounds[j];
        if lb > ub + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if (lb - ub).abs() < ZERO_TOL {
            let value = lb;
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] {
                    st.b[row] -= val * value;
                }
            }
            st.obj_offset += st.c[j] * value;
            st.removed_cols[j] = true;
            st.postsolve_stack.push(PostsolveStep::FixedVariable { orig_col: j, value });
        }
    }
    Ok(())
}
