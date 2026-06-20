//! Step 2b: forcing row — when a row's activity range makes it tight,
//! every variable in the row is forced to its contributing bound.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step2b_forcing_row(
    st: &mut PresolveState,
    deadline: Option<std::time::Instant>,
) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if st.removed_rows[i] {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() < 2 {
            continue;
        }

        let (row_lb, row_ub, lb_fin, ub_fin) =
            crate::presolve::activity::activity_range(&active, &st.bounds, None);
        let rhs = st.b[i];
        let ct = st.constraint_types[i];

        // Detect infeasibility before checking forcing.
        let infeasible = match ct {
            ConstraintType::Le => lb_fin && row_lb > rhs + ZERO_TOL,
            ConstraintType::Ge => ub_fin && row_ub < rhs - ZERO_TOL,
            ConstraintType::Eq => {
                (lb_fin && row_lb > rhs + ZERO_TOL) || (ub_fin && row_ub < rhs - ZERO_TOL)
            }
        };
        if infeasible {
            return Err(PresolveStatus::Infeasible);
        }

        // Forcing: the activity bound equals rhs, so all variables are at their
        // contributing bounds.
        // - Le: a_min >= b - eps (min already reaches rhs)
        // - Ge: a_max <= b + eps (max already reaches rhs)
        // - Eq: a_min >= b - eps (forced from below) OR a_max <= b + eps (forced from above),
        //       but only when the bound approximately equals rhs (infeasibility already excluded).
        let forcing = match ct {
            ConstraintType::Le => lb_fin && row_lb >= rhs - ZERO_TOL,
            ConstraintType::Ge => ub_fin && row_ub <= rhs + ZERO_TOL,
            ConstraintType::Eq => {
                (lb_fin && row_lb >= rhs - ZERO_TOL)
                    || (ub_fin && row_ub <= rhs + ZERO_TOL)
            }
        };

        if !forcing {
            continue;
        }

        // Determine which bound direction forces: for Le (or Eq forced from below)
        // the min-activity achieves the rhs, so each variable is at its lb-contributing
        // bound. For Ge (or Eq forced from above) the max-activity achieves the rhs.
        let force_to_min = match ct {
            ConstraintType::Le => true,
            ConstraintType::Ge => false,
            ConstraintType::Eq => lb_fin && row_lb >= rhs - ZERO_TOL,
        };

        // Check that all contributing bounds are finite.
        let all_finite = active.iter().all(|&(j, a_ij)| {
            let (lb_j, ub_j) = st.bounds[j];
            if force_to_min {
                if a_ij > 0.0 { lb_j.is_finite() } else { ub_j.is_finite() }
            } else {
                if a_ij > 0.0 { ub_j.is_finite() } else { lb_j.is_finite() }
            }
        });
        if !all_finite {
            continue;
        }

        // Collect (col, value, old_lb, old_ub) for each forced variable.
        let fixed_vars: Vec<(usize, f64, f64, f64)> = active
            .iter()
            .map(|&(j, a_ij)| {
                let (lb_j, ub_j) = st.bounds[j];
                let value = if force_to_min {
                    if a_ij > 0.0 { lb_j } else { ub_j }
                } else {
                    if a_ij > 0.0 { ub_j } else { lb_j }
                };
                (j, value, lb_j, ub_j)
            })
            .collect();

        // Fix each variable: substitute out of remaining rows and objective.
        for &(j, value, _, _) in &fixed_vars {
            if st.removed_cols[j] {
                continue;
            }
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] && row != i {
                    st.b[row] -= val * value;
                }
            }
            st.obj_offset += st.c[j] * value;
            st.removed_cols[j] = true;
        }

        st.removed_rows[i] = true;
        st.postsolve_stack.push(PostsolveStep::ForcingRow {
            orig_row: i,
            fixed_vars,
        });
    }
    Ok(())
}
