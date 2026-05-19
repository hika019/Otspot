//! QP presolve steps 9–11: singleton-ineq→bound, implied-bound tightening, dual fixing.

use super::helpers::skip_step;
use super::state::{QpPostsolveStep, QpPresolveResult, Workspace};
use crate::presolve::activity::activity_range;
use crate::problem::ConstraintType;
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;

/// Step 9: singleton Le/Ge rows with a single active variable are absorbed into
/// that variable's bounds, shrinking the constraint matrix by one row per match.
///
/// For `a·x ≤ b` (Le):
///   `a > 0` → tighten upper bound: `ub = min(ub, b/a)`
///   `a < 0` → tighten lower bound: `lb = max(lb, b/a)`
/// For `a·x ≥ b` (Ge): opposite directions.
///
/// Infeasibility is detected when the resulting `lb > ub + ZERO_TOL`.
/// The removed row is recorded in the postsolve stack for dual recovery.
pub(super) fn step9_singleton_ineq_to_bound(
    prob: &QpProblem,
    ws: &mut Workspace,
    deadline: Option<std::time::Instant>,
) -> Result<(), QpPresolveResult> {
    if skip_step(9) {
        return Ok(());
    }
    let m = prob.num_constraints;
    for i in 0..m {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if ws.removed_rows[i] {
            continue;
        }
        let ct = prob.constraint_types[i];
        if ct == ConstraintType::Eq {
            continue;
        }
        let active: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
            .copied()
            .collect();
        if active.len() != 1 {
            continue;
        }
        let (j, a_ij) = active[0];
        let rhs = ws.b[i] / a_ij;
        let (lb, ub) = ws.bounds[j];

        let (new_lb, new_ub) = match ct {
            ConstraintType::Le => {
                if a_ij > 0.0 {
                    (lb, ub.min(rhs))
                } else {
                    (lb.max(rhs), ub)
                }
            }
            ConstraintType::Ge => {
                if a_ij > 0.0 {
                    (lb.max(rhs), ub)
                } else {
                    (lb, ub.min(rhs))
                }
            }
            ConstraintType::Eq => unreachable!(),
        };

        if new_lb > new_ub + ZERO_TOL {
            return Err(QpPresolveResult::infeasible(prob));
        }
        ws.bounds[j] = (new_lb, new_ub);
        ws.removed_rows[i] = true;
        ws.postsolve_stack.push(QpPostsolveStep::SingletonIneqToBound { row: i, col: j, a_ij, ct });
    }
    Ok(())
}

/// Step 10: detect infeasibility from implied bounds. Bounds themselves are not
/// mutated; dense rows and pathological implied magnitudes are skipped to avoid
/// KKT blowup. O(m·avg_row²) inner — caller must respect the deadline.
pub(super) fn step10_implied_bounds(
    prob: &QpProblem,
    ws: &mut Workspace,
    deadline: Option<std::time::Instant>,
) -> Result<(), QpPresolveResult> {
    const DENSE_ROW_THRESHOLD: usize = 500;
    const IMPLIED_BOUND_SANITY: f64 = 1e8;
    let mut impl_bounds: Vec<(f64, f64)> = ws.bounds.clone();

    let m = prob.num_constraints;
    for i in 0..m {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if ws.removed_rows[i] {
            continue;
        }
        let entries: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
            .copied()
            .collect();

        if entries.len() > DENSE_ROW_THRESHOLD {
            continue;
        }

        let ct = prob.constraint_types[i];
        let do_le_dir = matches!(
            ct,
            crate::problem::ConstraintType::Le | crate::problem::ConstraintType::Eq
        );
        let do_ge_dir = matches!(
            ct,
            crate::problem::ConstraintType::Ge | crate::problem::ConstraintType::Eq
        );

        for &(j, a_ij) in &entries {
            let (old_lb, old_ub) = impl_bounds[j];
            let (rest_lb, rest_ub, rest_lb_fin, rest_ub_fin) =
                activity_range(&entries, &impl_bounds, Some(j));

            let mut new_lb = old_lb;
            let mut new_ub = old_ub;

            if do_le_dir && rest_lb_fin {
                if a_ij > 0.0 {
                    let implied_ub = (ws.b[i] - rest_lb) / a_ij;
                    if (implied_ub.abs() <= IMPLIED_BOUND_SANITY || !old_ub.is_infinite())
                        && implied_ub < new_ub - ZERO_TOL
                    {
                        new_ub = implied_ub;
                    }
                } else if a_ij < 0.0 {
                    let implied_lb = (ws.b[i] - rest_lb) / a_ij;
                    if (implied_lb.abs() <= IMPLIED_BOUND_SANITY || !old_lb.is_infinite())
                        && implied_lb > new_lb + ZERO_TOL
                    {
                        new_lb = implied_lb;
                    }
                }
            }
            if do_ge_dir && rest_ub_fin {
                if a_ij > 0.0 {
                    let implied_lb = (ws.b[i] - rest_ub) / a_ij;
                    if (implied_lb.abs() <= IMPLIED_BOUND_SANITY || !old_lb.is_infinite())
                        && implied_lb > new_lb + ZERO_TOL
                    {
                        new_lb = implied_lb;
                    }
                } else if a_ij < 0.0 {
                    let implied_ub = (ws.b[i] - rest_ub) / a_ij;
                    if (implied_ub.abs() <= IMPLIED_BOUND_SANITY || !old_ub.is_infinite())
                        && implied_ub < new_ub - ZERO_TOL
                    {
                        new_ub = implied_ub;
                    }
                }
            }

            if (new_lb - old_lb).abs() > ZERO_TOL || (new_ub - old_ub).abs() > ZERO_TOL {
                if new_lb > new_ub + ZERO_TOL {
                    return Err(QpPresolveResult::infeasible(prob));
                }
                impl_bounds[j] = (new_lb, new_ub);
            }
        }
    }
    Ok(())
}

/// Step 11: dual-bounds tightening for isolated LP-style columns (no Q, no A).
pub(super) fn step11_dual_fixing(
    prob: &QpProblem,
    ws: &mut Workspace,
) -> Result<(), QpPresolveResult> {
    if skip_step(11) {
        return Ok(());
    }
    let n = prob.num_vars;
    for j in 0..n {
        if ws.removed_cols[j] {
            continue;
        }
        let q_nnz = {
            let start = prob.q.col_ptr[j];
            let end = prob.q.col_ptr[j + 1];
            (start..end).filter(|&k| prob.q.values[k].abs() > ZERO_TOL).count()
        };
        if q_nnz > 0 {
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

        let (lb, ub) = ws.bounds[j];
        let val = if ws.c[j] > ZERO_TOL {
            if lb.is_finite() {
                lb
            } else {
                continue;
            }
        } else if ws.c[j] < -ZERO_TOL {
            if ub.is_finite() {
                ub
            } else {
                continue;
            }
        } else {
            continue;
        };

        ws.obj_offset += ws.c[j] * val;
        ws.bounds[j] = (val, val);
        ws.removed_cols[j] = true;
        ws.postsolve_stack.push(QpPostsolveStep::EmptyCol { idx: j, val });
    }
    Ok(())
}
