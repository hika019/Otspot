//! QP presolve step 7: free-variable substitution via singleton Eq rows.

use super::helpers::{apply_fixed_variable, col_has_structural_q, skip_step};
use super::state::{QpPostsolveStep, QpPresolveResult, Workspace};
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;

/// Step 7: free-variable substitution via Eq rows (restricted to Q-zero columns so we
/// don't have to update Q with a rank-1 term).
/// O(n*m) inner scan — caller must respect the deadline.
pub(super) fn step7_free_var(
    prob: &QpProblem,
    ws: &mut Workspace,
    deadline: Option<std::time::Instant>,
) -> Result<(), QpPresolveResult> {
    if skip_step(7) {
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
        let (lb, ub) = ws.bounds[j];
        if lb != f64::NEG_INFINITY || ub != f64::INFINITY {
            continue;
        }

        if col_has_structural_q(&prob.q, j) {
            continue;
        }

        // Only Eq singleton rows are eligible.
        let singleton_eq_rows: Vec<(usize, f64)> = (0..m)
            .filter_map(|i| {
                if ws.removed_rows[i] {
                    return None;
                }
                if prob.constraint_types[i] != crate::problem::ConstraintType::Eq {
                    return None;
                }
                let active: Vec<_> = ws.row_entries[i]
                    .iter()
                    .filter(|&&(jj, v)| !ws.removed_cols[jj] && v.abs() > ZERO_TOL)
                    .collect();
                (active.len() == 1 && active[0].0 == j).then_some((i, active[0].1))
            })
            .collect();

        if singleton_eq_rows.is_empty() {
            continue;
        }

        let (i, a_ij) = singleton_eq_rows[0];
        let val = ws.b[i] / a_ij;

        apply_fixed_variable(j, val, prob, ws);
        ws.removed_cols[j] = true;
        ws.removed_rows[i] = true;
        ws.postsolve_stack.push(QpPostsolveStep::SingletonRow {
            row: i,
            col: j,
            val,
        });
    }
    Ok(())
}
