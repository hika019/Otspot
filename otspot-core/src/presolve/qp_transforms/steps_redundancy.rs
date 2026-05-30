//! QP presolve activity-redundancy steps (initial pass, final pass with tightened bounds).

use super::helpers::skip_step;
use super::state::{QpPresolveResult, Workspace};
use crate::presolve::activity::activity_range;
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;

/// Step 5: drop constraints dominated by activity range; only strict slack qualifies.
pub(super) fn step5_redundant(
    prob: &QpProblem,
    ws: &mut Workspace,
) -> Result<(), QpPresolveResult> {
    if skip_step(5) {
        return Ok(());
    }
    let m = prob.num_constraints;
    for i in 0..m {
        if ws.removed_rows[i] {
            continue;
        }
        let active_entries: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, _)| !ws.removed_cols[j])
            .copied()
            .collect();
        let (row_lb, row_ub, lb_fin, ub_fin) = activity_range(&active_entries, &ws.bounds, None);

        match prob.constraint_types[i] {
            crate::problem::ConstraintType::Le => {
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    ws.removed_rows[i] = true;
                }
            }
            crate::problem::ConstraintType::Eq => {
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    return Err(QpPresolveResult::infeasible(prob));
                }
                // Eq-tightening intentionally skipped: a scalar y[i] cannot satisfy
                // stationarity for multiple bound-pinned variables simultaneously.
            }
            crate::problem::ConstraintType::Ge => {
                if lb_fin && row_lb > ws.b[i] + ZERO_TOL {
                    ws.removed_rows[i] = true;
                }
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    return Err(QpPresolveResult::infeasible(prob));
                }
            }
        }
    }
    Ok(())
}

/// Step 12: re-apply redundancy / infeasibility once bounds have been tightened.
pub(super) fn step12_redundant_final(
    prob: &QpProblem,
    ws: &mut Workspace,
) -> Result<(), QpPresolveResult> {
    if skip_step(12) {
        return Ok(());
    }
    let m = prob.num_constraints;
    for i in 0..m {
        if ws.removed_rows[i] {
            continue;
        }
        let entries: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
            .copied()
            .collect();
        let (row_lb, row_ub, lb_fin, ub_fin) = activity_range(&entries, &ws.bounds, None);

        match prob.constraint_types[i] {
            crate::problem::ConstraintType::Le => {
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    ws.removed_rows[i] = true;
                }
            }
            crate::problem::ConstraintType::Eq => {
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    return Err(QpPresolveResult::infeasible(prob));
                }
            }
            crate::problem::ConstraintType::Ge => {
                if lb_fin && row_lb > ws.b[i] + ZERO_TOL {
                    ws.removed_rows[i] = true;
                }
                if ub_fin && row_ub < ws.b[i] - ZERO_TOL {
                    return Err(QpPresolveResult::infeasible(prob));
                }
            }
        }
        if !ws.removed_rows[i] && lb_fin && row_lb > ws.b[i] + ZERO_TOL {
            match prob.constraint_types[i] {
                crate::problem::ConstraintType::Le | crate::problem::ConstraintType::Eq => {
                    return Err(QpPresolveResult::infeasible(prob));
                }
                crate::problem::ConstraintType::Ge => {
                    ws.removed_rows[i] = true;
                }
            }
        }
    }
    Ok(())
}
