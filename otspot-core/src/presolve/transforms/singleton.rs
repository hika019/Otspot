//! Step 2: singleton row — a row with a single active variable.
//!
//! - Eq: `a*x = b` fixes `x = b/a`.
//! - Le: `a*x <= b` tightens an upper or lower bound on `x`.
//! - Ge: `a*x >= b` tightens a lower or upper bound on `x`.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step2_singleton_row(
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
        if active.len() != 1 {
            continue;
        }
        let (j, a_ij) = active[0];
        if a_ij.abs() < ZERO_TOL {
            continue;
        }
        match st.constraint_types[i] {
            ConstraintType::Eq => {
                singleton_eq(st, i, j, a_ij)?;
            }
            ConstraintType::Le | ConstraintType::Ge => {
                singleton_ineq(st, i, j, a_ij)?;
            }
        }
    }
    Ok(())
}

fn singleton_eq(
    st: &mut PresolveState,
    i: usize,
    j: usize,
    a_ij: f64,
) -> Result<(), PresolveStatus> {
    let value = st.b[i] / a_ij;
    let (lb, ub) = st.bounds[j];
    if value < lb - ZERO_TOL || value > ub + ZERO_TOL {
        return Err(PresolveStatus::Infeasible);
    }
    let value = value.clamp(lb, ub);

    // Snapshot dual-recovery data before fix_and_remove mutates state.
    let c_orig = st.c[j];
    let col_orig_entries: Vec<(usize, f64)> = st.col_entries[j]
        .iter()
        .filter(|&&(r, v)| r != i && !st.removed_rows[r] && v.abs() >= ZERO_TOL)
        .copied()
        .collect();

    fix_and_remove(st, i, j, value);
    st.postsolve_stack.push(PostsolveStep::SingletonRow {
        orig_row: i,
        orig_col: j,
        value,
        coeff: a_ij,
        col_orig_entries,
        c_orig,
    });
    Ok(())
}

/// Singleton Le/Ge: `a*x {<=,>=} b` implies a one-sided bound on `x`.
///
/// - Le, a > 0: x <= b/a  (tighten ub)
/// - Le, a < 0: x >= b/a  (tighten lb)
/// - Ge, a > 0: x >= b/a  (tighten lb)
/// - Ge, a < 0: x <= b/a  (tighten ub)
fn singleton_ineq(
    st: &mut PresolveState,
    i: usize,
    j: usize,
    a_ij: f64,
) -> Result<(), PresolveStatus> {
    let implied = st.b[i] / a_ij;
    let (lb, ub) = st.bounds[j];
    let ct = st.constraint_types[i];

    let tightens_ub = (ct == ConstraintType::Le && a_ij > 0.0)
        || (ct == ConstraintType::Ge && a_ij < 0.0);

    if tightens_ub {
        if implied < lb - ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if implied < ub - ZERO_TOL {
            st.bounds[j].1 = implied;
        }
    } else {
        if implied > ub + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if implied > lb + ZERO_TOL {
            st.bounds[j].0 = implied;
        }
    }

    st.removed_rows[i] = true;
    st.postsolve_stack
        .push(PostsolveStep::SingletonInequalityRow {
            orig_row: i,
            orig_col: j,
            coeff: a_ij,
            old_lb: lb,
            old_ub: ub,
        });
    Ok(())
}

/// Substitute a fixed variable value into remaining rows and objective, then
/// mark the column and row as removed.
fn fix_and_remove(st: &mut PresolveState, row: usize, col: usize, value: f64) {
    let col_copy = st.col_entries[col].clone();
    for (r, val) in col_copy {
        if !st.removed_rows[r] && r != row {
            st.b[r] -= val * value;
        }
    }
    st.obj_offset += st.c[col] * value;
    st.removed_cols[col] = true;
    st.removed_rows[row] = true;
}
