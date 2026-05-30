//! QP presolve step 8: parallel rows (`A[i,*] = α · A[j,*]`).
//!
//! Hash-bucket by `(first_col, sign)`, then pair-test inside each bucket.

use super::helpers::skip_step;
use super::state::{QpPresolveResult, Workspace};
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;
use std::collections::HashMap;

pub(super) fn step8_parallel_row(
    prob: &QpProblem,
    ws: &mut Workspace,
    deadline: Option<std::time::Instant>,
) -> Result<(), QpPresolveResult> {
    if skip_step(8) {
        return Ok(());
    }
    let m = prob.num_constraints;
    let mut row_signature: HashMap<(usize, i8), Vec<usize>> = HashMap::new();
    for i in 0..m {
        if ws.removed_rows[i] {
            continue;
        }
        let active: Vec<(usize, f64)> = ws.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
            .copied()
            .collect();
        if active.is_empty() {
            continue;
        }
        let first_col = active[0].0;
        let sign: i8 = if active[0].1 > 0.0 { 1 } else { -1 };
        row_signature.entry((first_col, sign)).or_default().push(i);
    }

    'groups: for row_group in row_signature.values() {
        if row_group.len() < 2 {
            continue;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break 'groups;
        }
        'outer: for &i1 in row_group {
            if ws.removed_rows[i1] {
                continue;
            }
            let entries1: Vec<(usize, f64)> = ws.row_entries[i1]
                .iter()
                .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
                .copied()
                .collect();
            if entries1.is_empty() {
                continue;
            }

            for &i2 in row_group {
                if i2 == i1 || ws.removed_rows[i2] {
                    continue;
                }
                let entries2: Vec<(usize, f64)> = ws.row_entries[i2]
                    .iter()
                    .filter(|&&(j, v)| !ws.removed_cols[j] && v.abs() > ZERO_TOL)
                    .copied()
                    .collect();
                if entries1.len() != entries2.len() {
                    continue;
                }

                let alpha = entries2[0].1 / entries1[0].1;
                let is_parallel =
                    entries1
                        .iter()
                        .zip(entries2.iter())
                        .all(|((c1, v1), (c2, v2))| {
                            *c1 == *c2 && (v2 - alpha * v1).abs() < ZERO_TOL * (1.0 + v1.abs())
                        });

                if !is_parallel {
                    continue;
                }

                // Only Le-Le, Ge-Ge, or Eq-Eq with α > 0. Mixed-type or α ≤ 0 cases
                // would invert the redundancy direction and are deferred to the solver.
                let t1 = prob.constraint_types[i1];
                let t2 = prob.constraint_types[i2];
                let both_le = matches!(t1, crate::problem::ConstraintType::Le)
                    && matches!(t2, crate::problem::ConstraintType::Le);
                let both_ge = matches!(t1, crate::problem::ConstraintType::Ge)
                    && matches!(t2, crate::problem::ConstraintType::Ge);
                let both_eq = matches!(t1, crate::problem::ConstraintType::Eq)
                    && matches!(t2, crate::problem::ConstraintType::Eq);

                if both_eq && alpha > ZERO_TOL {
                    let eff_b2 = ws.b[i2] / alpha;
                    if (eff_b2 - ws.b[i1]).abs() <= ZERO_TOL * (1.0 + ws.b[i1].abs()) {
                        ws.removed_rows[i2] = true;
                    } else {
                        return Err(QpPresolveResult::infeasible(prob));
                    }
                } else if both_le && alpha > ZERO_TOL {
                    let eff_b2 = ws.b[i2] / alpha;
                    if eff_b2 >= ws.b[i1] - ZERO_TOL {
                        ws.removed_rows[i2] = true;
                    } else {
                        ws.removed_rows[i1] = true;
                        continue 'outer;
                    }
                } else if both_ge && alpha > ZERO_TOL {
                    let eff_b2 = ws.b[i2] / alpha;
                    if eff_b2 <= ws.b[i1] + ZERO_TOL {
                        ws.removed_rows[i2] = true;
                    } else {
                        ws.removed_rows[i1] = true;
                        continue 'outer;
                    }
                }
            }
        }
    }
    Ok(())
}
