//! Step 5: bounds tightening from implied row activity.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::presolve::activity::propagate_row_bounds;
use crate::tolerances::ZERO_TOL;

pub(super) fn step5_bounds_tightening(
    st: &mut PresolveState,
    new_fixed: &mut usize,
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
        let ct = st.constraint_types[i];
        let entries = st.active_row_entries(i);
        if entries.is_empty() {
            continue;
        }

        let updates = propagate_row_bounds(
            &entries, &st.bounds, ct, st.b[i],
            // LP presolve passes int_mask=None; acceptance is unconditional
            // (aggressive tightening, unlike MIP presolve which must not cut off
            // integer solutions).
            None,
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

/// Revert presolve-added variable bounds that a retained constraint row already
/// implies, restoring the original infinite bound.
///
/// Activity bound-tightening (step5) routinely hands a variable a finite implied
/// bound on a side that was originally infinite. That bound is redundant whenever
/// a retained row already forces it — but the un-bounded simplex standard form
/// materializes every finite upper bound as an explicit constraint row, so on a
/// mostly-unbounded LP whose rows imply a bound for nearly every column (e.g.
/// set-partitioning `Σx = 1` ⇒ `x ≤ 1`) the reduced problem gains one row per
/// variable and its per-pivot linear algebra blows up. Reverting these bounds to
/// `±∞` leaves the feasible region unchanged: the retained row still enforces the
/// implication.
///
/// Redundancy is checked with the *original* bounds so the decision is independent
/// of any bound reverted here — a retained row that implies `x_j {≤,≥} v` under the
/// original (looser) bounds implies at least that under the emitted (tighter) ones,
/// so removing the emitted bound never enlarges the feasible region. Fixings
/// (`lb == ub`) and bounds that were finite in the original problem are left
/// untouched, so genuine model bounds are never dropped.
pub(super) fn revert_redundant_added_bounds(st: &mut PresolveState) {
    let n = st.bounds.len();
    let m = st.b.len();
    let mut implied_lb = vec![f64::NEG_INFINITY; n];
    let mut implied_ub = vec![f64::INFINITY; n];

    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let entries = st.active_row_entries(i);
        if entries.is_empty() {
            continue;
        }
        // Original bounds keep the implication independent of the reverts below.
        if let Some(updates) = crate::presolve::activity::propagate_row_bounds(
            &entries,
            &st.orig_bounds,
            st.constraint_types[i],
            st.b[i],
            None,
        ) {
            for (j, nlb, nub) in updates {
                if nub < implied_ub[j] {
                    implied_ub[j] = nub;
                }
                if nlb > implied_lb[j] {
                    implied_lb[j] = nlb;
                }
            }
        }
    }

    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let (orig_lb, orig_ub) = st.orig_bounds[j];
        let (lb, ub) = st.bounds[j];
        // Keep genuine fixings produced by tightening.
        if ub - lb <= ZERO_TOL {
            continue;
        }
        let mut new_lb = lb;
        let mut new_ub = ub;
        if orig_ub == f64::INFINITY && ub.is_finite() && implied_ub[j] <= ub + ZERO_TOL {
            new_ub = f64::INFINITY;
        }
        if orig_lb == f64::NEG_INFINITY && lb.is_finite() && implied_lb[j] >= lb - ZERO_TOL {
            new_lb = f64::NEG_INFINITY;
        }
        st.bounds[j] = (new_lb, new_ub);
    }
}
