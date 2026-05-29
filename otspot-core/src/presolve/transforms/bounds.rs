//! Step 5: bounds tightening from implied row activity.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::presolve::activity::propagate_row_bounds;
use crate::tolerances::ZERO_TOL;

pub(super) fn step5_bounds_tightening(
    st: &mut PresolveState,
    new_fixed: &mut usize,
) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let ct = st.constraint_types[i];
        let entries = st.active_row_entries(i);
        if entries.is_empty() {
            continue;
        }

        let updates = propagate_row_bounds(
            &entries,
            &st.bounds,
            ct,
            st.b[i],
            None, // LP presolve: no integer rounding
        )
        .ok_or(PresolveStatus::Infeasible)?;

        for (j, new_lb, new_ub) in updates {
            st.postsolve_stack.push(PostsolveStep::BoundsTightened);
            st.bounds[j] = (new_lb, new_ub);
            if (new_lb - new_ub).abs() < ZERO_TOL {
                *new_fixed += 1;
            }
        }
    }
    Ok(())
}
