//! Steps 3 + 4: empty rows/columns and activity-redundant constraints.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step3a_empty_row(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let active_count = st.active_row_entries(i).len();
        if active_count == 0 {
            match st.constraint_types[i] {
                ConstraintType::Eq => {
                    if st.b[i].abs() > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Le => {
                    if st.b[i] < -ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Ge => {
                    if st.b[i] > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
            }
            st.removed_rows[i] = true;
            st.postsolve_stack
                .push(PostsolveStep::EmptyRow { orig_row: i });
        }
    }
    Ok(())
}

pub(super) fn step3b_empty_column(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let active_count = st.active_col_entries(j).len();
        if active_count == 0 {
            let (lb, ub) = st.bounds[j];
            let cj = st.c[j];
            let value = if cj > ZERO_TOL {
                if lb == f64::NEG_INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if lb.is_finite() {
                    lb
                } else {
                    0.0
                }
            } else if cj < -ZERO_TOL {
                if ub == f64::INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if ub.is_finite() {
                    ub
                } else {
                    0.0
                }
            } else if lb.is_finite() {
                lb
            } else if ub.is_finite() {
                ub
            } else {
                0.0
            };
            st.obj_offset += cj * value;
            st.removed_cols[j] = true;
            st.postsolve_stack
                .push(PostsolveStep::EmptyColumn { orig_col: j, value });
        }
    }
    Ok(())
}

pub(super) fn step4_redundant_constraint(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let active_entries = st.active_row_entries(i);
        let (row_lb, row_ub, lb_fin, ub_fin) =
            crate::presolve::activity::activity_range(&active_entries, &st.bounds, None);

        let redundant = match st.constraint_types[i] {
            ConstraintType::Le => ub_fin && row_ub <= st.b[i] + ZERO_TOL,
            ConstraintType::Ge => lb_fin && row_lb >= st.b[i] - ZERO_TOL,
            ConstraintType::Eq => {
                lb_fin && ub_fin && row_lb >= st.b[i] - ZERO_TOL && row_ub <= st.b[i] + ZERO_TOL
            }
        };
        if redundant {
            st.removed_rows[i] = true;
            st.postsolve_stack
                .push(PostsolveStep::RedundantConstraint { orig_row: i });
        }
    }
    Ok(())
}
