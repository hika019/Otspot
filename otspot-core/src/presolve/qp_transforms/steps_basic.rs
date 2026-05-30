//! QP presolve steps 1-4: fix variables, singleton row/column, empty row/column.

use super::helpers::{apply_fixed_variable, skip_step};
use super::state::{QpPostsolveStep, QpPresolveResult, Workspace};
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;

/// Step 1: fix variables with `lb == ub`.
pub(super) fn step1_fix_var(prob: &QpProblem, ws: &mut Workspace) -> Result<(), QpPresolveResult> {
    if skip_step(1) {
        return Ok(());
    }
    let n = prob.num_vars;
    for j in 0..n {
        if ws.removed_cols[j] {
            continue;
        }
        let (lb, ub) = ws.bounds[j];
        if lb > ub + ZERO_TOL {
            return Err(QpPresolveResult::infeasible(prob));
        }
        if (lb - ub).abs() < ZERO_TOL {
            let val = lb;
            // Skip substitution that would blow up b — the IPM will handle the
            // tightly-bounded variable instead.
            const LARGE_B_THRESHOLD: f64 = 1e5;
            let max_b_change: f64 = {
                let col_start = prob.a.col_ptr[j];
                let col_end = prob.a.col_ptr[j + 1];
                (col_start..col_end)
                    .filter(|&k| !ws.removed_rows[prob.a.row_ind[k]])
                    .map(|k| (prob.a.values[k] * val).abs())
                    .fold(0.0f64, f64::max)
            };
            if max_b_change > LARGE_B_THRESHOLD {
                continue;
            }
            apply_fixed_variable(j, val, prob, ws);
            ws.removed_cols[j] = true;
            ws.postsolve_stack
                .push(QpPostsolveStep::FixedVar { idx: j, val });
        }
    }
    Ok(())
}

/// Step 2: singleton rows — Eq fixes the variable, Le/Ge tightens bounds.
pub(super) fn step2_singleton_row(
    prob: &QpProblem,
    ws: &mut Workspace,
) -> Result<(), QpPresolveResult> {
    if skip_step(2) {
        return Ok(());
    }
    let m = prob.num_constraints;
    for i in 0..m {
        if ws.removed_rows[i] {
            continue;
        }
        let active: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, _)| !ws.removed_cols[j])
            .copied()
            .collect();
        if active.len() != 1 {
            continue;
        }
        let (j, a_ij) = active[0];
        if a_ij.abs() < ZERO_TOL {
            continue;
        }
        if prob.constraint_types[i] == crate::problem::ConstraintType::Eq {
            let val = ws.b[i] / a_ij;
            let (lb, ub) = ws.bounds[j];
            if val >= lb - ZERO_TOL && val <= ub + ZERO_TOL {
                let val = val.clamp(lb, ub);
                apply_fixed_variable(j, val, prob, ws);
                ws.removed_cols[j] = true;
                ws.removed_rows[i] = true;
                ws.postsolve_stack.push(QpPostsolveStep::SingletonRow {
                    row: i,
                    col: j,
                    val,
                });
            }
            continue;
        }
        let val_raw = ws.b[i] / a_ij;
        let (lb, ub) = ws.bounds[j];

        let val = val_raw.clamp(lb, ub);
        if (val - lb).abs() < ZERO_TOL && (val - ub).abs() < ZERO_TOL {
            apply_fixed_variable(j, val, prob, ws);
            ws.removed_cols[j] = true;
            ws.removed_rows[i] = true;
            ws.postsolve_stack.push(QpPostsolveStep::SingletonRow {
                row: i,
                col: j,
                val,
            });
        }
    }
    Ok(())
}

/// Step 3: singleton columns (only when Q[j,j] is empty so we don't drop a quadratic term).
/// O(n*m) inner scan — caller must respect the deadline.
pub(super) fn step3_singleton_col(
    prob: &QpProblem,
    ws: &mut Workspace,
    deadline: Option<std::time::Instant>,
) -> Result<(), QpPresolveResult> {
    if skip_step(3) {
        return Ok(());
    }
    let n = prob.num_vars;
    let m = prob.num_constraints;
    for j in 0..n {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if ws.removed_cols[j] {
            continue;
        }

        let q_nnz_j = {
            let start = prob.q.col_ptr[j];
            let end = prob.q.col_ptr[j + 1];
            (start..end)
                .filter(|&k| prob.q.values[k].abs() > ZERO_TOL)
                .count()
        };
        if q_nnz_j > 0 {
            continue;
        }

        let active_rows: Vec<usize> = (0..m)
            .filter(|&i| {
                !ws.removed_rows[i]
                    && ws.row_entries[i]
                        .iter()
                        .any(|&(jj, v)| jj == j && v.abs() > ZERO_TOL)
            })
            .collect();

        if active_rows.len() != 1 {
            continue;
        }
        let i = active_rows[0];

        // Only Le rows are safe; Eq/Ge are handled by step 7 or by the solver.
        if prob.constraint_types[i] != crate::problem::ConstraintType::Le {
            continue;
        }

        let a_ij = ws.row_entries[i]
            .iter()
            .find(|&&(jj, _)| jj == j)
            .map(|&(_, v)| v)
            .unwrap_or(0.0);
        if a_ij.abs() < ZERO_TOL {
            continue;
        }

        // Only fix when the objective and constraint relaxation pull the same way;
        // otherwise defer to the IPM.
        let (lb, ub) = ws.bounds[j];
        let val = if ws.c[j] > ZERO_TOL && a_ij > ZERO_TOL {
            if lb == f64::NEG_INFINITY {
                0.0
            } else {
                lb
            }
        } else if ws.c[j] < -ZERO_TOL && a_ij < -ZERO_TOL {
            if ub == f64::INFINITY {
                0.0
            } else {
                ub
            }
        } else if ws.c[j].abs() <= ZERO_TOL {
            if a_ij > ZERO_TOL {
                if lb == f64::NEG_INFINITY {
                    0.0
                } else {
                    lb
                }
            } else if ub == f64::INFINITY {
                0.0
            } else {
                ub
            }
        } else {
            continue;
        };

        apply_fixed_variable(j, val, prob, ws);
        ws.removed_cols[j] = true;
        ws.postsolve_stack
            .push(QpPostsolveStep::FixedVar { idx: j, val });
    }
    Ok(())
}

/// Step 4: empty rows and empty columns. Returns Err for trivial infeasibility/unboundedness.
pub(super) fn step4_empty(prob: &QpProblem, ws: &mut Workspace) -> Result<(), QpPresolveResult> {
    if skip_step(4) {
        return Ok(());
    }
    let n = prob.num_vars;
    let m = prob.num_constraints;

    // Empty rows
    for i in 0..m {
        if ws.removed_rows[i] {
            continue;
        }
        let active_count = ws.row_entries[i]
            .iter()
            .filter(|&&(j, _)| !ws.removed_cols[j])
            .count();
        if active_count == 0 {
            let infeasible = match prob.constraint_types[i] {
                crate::problem::ConstraintType::Le => ws.b[i] < -ZERO_TOL,
                crate::problem::ConstraintType::Ge => ws.b[i] > ZERO_TOL,
                crate::problem::ConstraintType::Eq => ws.b[i].abs() > ZERO_TOL,
            };
            if infeasible {
                return Err(QpPresolveResult::infeasible(prob));
            }
            ws.removed_rows[i] = true;
        }
    }

    // Empty columns (only when A[*,j] and Q[*,j] are both empty)
    for j in 0..n {
        if ws.removed_cols[j] {
            continue;
        }
        let a_nnz = {
            let start = prob.a.col_ptr[j];
            let end = prob.a.col_ptr[j + 1];
            (start..end)
                .filter(|&k| {
                    let row = prob.a.row_ind[k];
                    !ws.removed_rows[row] && prob.a.values[k].abs() > ZERO_TOL
                })
                .count()
        };
        if a_nnz > 0 {
            continue;
        }
        let q_nnz = {
            let start = prob.q.col_ptr[j];
            let end = prob.q.col_ptr[j + 1];
            (start..end)
                .filter(|&k| prob.q.values[k].abs() > ZERO_TOL)
                .count()
        };
        if q_nnz > 0 {
            continue;
        }

        // Pure LP variable: minimise `c_j · x_j` over [lb, ub].
        let (lb, ub) = ws.bounds[j];
        let cj = ws.c[j];
        if cj > ZERO_TOL && !lb.is_finite() {
            return Err(QpPresolveResult::unbounded(prob));
        }
        if cj < -ZERO_TOL && !ub.is_finite() {
            return Err(QpPresolveResult::unbounded(prob));
        }
        let val = if cj > ZERO_TOL {
            lb
        } else if cj < -ZERO_TOL {
            ub
        } else if lb.is_finite() {
            lb
        } else if ub.is_finite() {
            ub
        } else {
            0.0
        };

        ws.obj_offset += cj * val;
        ws.removed_cols[j] = true;
        ws.postsolve_stack
            .push(QpPostsolveStep::EmptyCol { idx: j, val });
    }

    Ok(())
}
