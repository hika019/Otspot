//! QP presolve step 8: parallel rows (`A[i,*] = α · A[j,*]`).
//!
//! Hash-bucket by `(first_col, sign)`, then pair-test inside each bucket.

use super::helpers::skip_step;
use super::state::{QpPresolveResult, Workspace};
use crate::options::SolverOptions;
use crate::qp::QpProblem;
use crate::tolerances::ZERO_TOL;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::AtomicBool;

// Test-only observability for the mid-loop stop checks below: counts every
// row examination step8 makes (one per grouping-loop row, one per pair-test
// outer-loop row) and flips `STEP8_CANCEL_SIGNAL` once the configured target
// is reached. Entirely `#[cfg(test)]` (definition and every call site), so it
// has zero footprint in production builds -- mirrors `driver`'s
// `STEPS_EXECUTED_TOTAL`/`CANCEL_SIGNAL` mechanism, giving the sentinels a
// deterministic (work-count based, not wall-clock) stand-in for a real
// expiring deadline.
#[cfg(test)]
thread_local! {
    static STEP8_ROWS_EXAMINED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static STEP8_CANCEL_AFTER_ROWS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static STEP8_CANCEL_SIGNAL: std::sync::Arc<AtomicBool> = std::sync::Arc::new(AtomicBool::new(false));
}

#[cfg(test)]
fn test_record_step8_row_examined() {
    let examined = STEP8_ROWS_EXAMINED.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if STEP8_CANCEL_AFTER_ROWS.with(std::cell::Cell::get) == Some(examined) {
        STEP8_CANCEL_SIGNAL.with(|flag| flag.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

/// See `step8_parallel_row`: stop is honored once per examined row, so the
/// post-deadline work leak per interrupt is one row's scan. Measured on
/// IS_LASSO_300 (n=m=30600): the grouping pass (30600 rows) costs ~25ms and
/// the pair-test loop costs ~0.24ms/row inside the largest bucket (726-row
/// group = 241ms), so a per-row `Instant::now()` adds < 0.1% step overhead
/// while bounding overshoot to ≤ ~1ms. Sampling every K rows would reintroduce
/// K-row leaks for no measurable gain at this rate.
#[inline]
fn step8_stop_requested(opts: &SolverOptions) -> bool {
    opts.external_stop_requested()
}

pub(super) fn step8_parallel_row(
    prob: &QpProblem,
    ws: &mut Workspace,
    opts: &SolverOptions,
) -> Result<(), QpPresolveResult> {
    if skip_step(8) {
        return Ok(());
    }
    let m = prob.num_constraints;
    let mut row_signature: HashMap<(usize, i8), Vec<usize>> = HashMap::new();
    for i in 0..m {
        #[cfg(test)]
        test_record_step8_row_examined();
        if step8_stop_requested(opts) {
            return Ok(());
        }
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

    for row_group in row_signature.values() {
        if row_group.len() < 2 {
            continue;
        }
        'outer: for &i1 in row_group {
            // Pair-testing one row against its whole bucket is O(|bucket|)
            // with a fresh Vec collect per candidate -- up to seconds when the
            // bucket is large (IS_LASSO_300: one 2240-row bucket). A stop
            // requested mid-bucket used to go unnoticed until the next bucket
            // boundary; per-row here bounds the leak to one row's scan
            // (derivation in `step8_stop_requested`). `return`, not `break`:
            // skipping the later groups' row scans matters when the stop is
            // already requested, and the removals kept from earlier groups
            // are all individually proven either way.
            #[cfg(test)]
            test_record_step8_row_examined();
            if step8_stop_requested(opts) {
                return Ok(());
            }
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

#[cfg(test)]
mod stop_sentinels;
