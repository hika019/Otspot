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
mod stop_sentinels {
    //! Deterministic (work-count, not wall-clock) sentinels for the per-row
    //! stop checks inside `step8_parallel_row`. Before this fix the step only
    //! consulted the stop condition at bucket-entry boundaries, so a stop
    //! requested mid-bucket (the shape of IS_LASSO_300's 2240-row bucket,
    //! ~2.3s of unbounded `O(|bucket|)` pair-tests) was ignored until the
    //! *next* bucket start — the dominant deadline-overshoot leak behind the
    //! sporadic `qp_is_lasso_300_real_data_honors_deadline` failures.
    use super::*;
    use crate::problem::ConstraintType;
    use otspot_num::sparse::CscMatrix;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// One bucket (every row's first active column is `x0`, positive), whose
    /// only truly parallel pair sits at the very end of the bucket's row order
    /// (rows `m-2`/`m-1`, `A[m-1] = 2·A[m-2]`, both Le with `b[m-1]/2 > b[m-2]`
    /// → the looser row `m-1` is the one dropped). Every other row pairs-test
    /// as non-parallel (distinct second column / non-proportional values).
    ///
    /// Independent oracle for the uncancelled premise (hand solve of the pair
    /// test): `alpha = A[m-1][0]/A[m-2][0] = 2`, both entries proportional
    /// (`2·(1,2) = (2,4)`), `b[m-1]/alpha = 5/2 = 2.5 ≥ b[m-2] = 1` → remove
    /// row `m-1`; no other row pair shares its column pattern, so exactly one
    /// row is removed.
    const M: usize = 120;

    fn single_bucket_fixture() -> (QpProblem, Workspace) {
        let n = M + 1usize;
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        let mut b = vec![0.0f64; M];
        for i in 0..M {
            let (c1, c2, v1, v2, rhs) = match i {
                _ if i == M - 2 => (0usize, 5usize, 1.0f64, 2.0f64, 1.0f64),
                _ if i == M - 1 => (0usize, 5usize, 2.0f64, 4.0f64, 5.0f64),
                _ => (0usize, i + 1, 1.0f64, 2.0 + 0.001 * (i as f64), 1.0f64),
            };
            a_rows.extend_from_slice(&[i, i]);
            a_cols.extend_from_slice(&[c1, c2]);
            a_vals.extend_from_slice(&[v1, v2]);
            b[i] = rhs;
        }
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, M, n).unwrap();
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();
        // Premise of "one bucket": every row must actually land in the
        // (first_col=0, sign=+1) group the sentinels below rely on.
        for i in 0..M {
            assert_eq!(
                prob.constraint_types[i],
                ConstraintType::Le,
                "fixture is all-Le"
            );
            assert!(ws_row0_first(&prob, i));
        }
        let ws = Workspace::from_problem(&prob);
        (prob, ws)
    }

    fn ws_row0_first(prob: &QpProblem, row: usize) -> bool {
        let start = prob.a.col_ptr()[0];
        let end = prob.a.col_ptr()[1];
        (start..end).any(|k| prob.a.row_ind()[k] == row && prob.a.values()[k] > 0.0)
    }

    fn reset(cancel_after: Option<usize>) -> Arc<std::sync::atomic::AtomicBool> {
        STEP8_ROWS_EXAMINED.with(|c| c.set(0));
        STEP8_CANCEL_AFTER_ROWS.with(|c| c.set(cancel_after));
        STEP8_CANCEL_SIGNAL.with(|flag| {
            flag.store(false, Ordering::Relaxed);
            Arc::clone(flag)
        })
    }

    fn examined() -> usize {
        STEP8_ROWS_EXAMINED.with(|c| c.get())
    }

    fn removed_rows(ws: &Workspace) -> Vec<usize> {
        (0..M).filter(|&i| ws.removed_rows[i]).collect::<Vec<_>>()
    }

    #[test]
    fn step8_per_row_stop_checks_bound_the_mid_bucket_leak() {
        // (1) Fixture premise, uncancelled: exactly the late looser row is
        //     dropped, and every row is examined twice (grouping + pair-test).
        let signal = reset(None);
        let (prob, mut ws) = single_bucket_fixture();
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&signal)),
            ..SolverOptions::default()
        };
        assert!(step8_parallel_row(&prob, &mut ws, &opts).is_ok());
        assert_eq!(
            removed_rows(&ws),
            vec![M - 1],
            "uncancelled run must drop exactly the looser parallel row"
        );
        assert_eq!(
            examined(),
            2 * M,
            "uncancelled run examines each row in the grouping pass and the \
             pair-test pass"
        );

        // (2) Stop flips during the grouping pass: the per-row check there
        //     must stop at that row. Sentinel: reverting the grouping-loop
        //     check lets the pass run to completion (examined = 2M - ... at
        //     least M), so `examined == 5` fails.
        let signal = reset(Some(5));
        let (prob, mut ws) = single_bucket_fixture();
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&signal)),
            ..SolverOptions::default()
        };
        assert!(step8_parallel_row(&prob, &mut ws, &opts).is_ok());
        assert_eq!(
            examined(),
            5,
            "the grouping pass must honor the stop the row it flips on"
        );
        assert!(
            removed_rows(&ws).is_empty(),
            "a stop during grouping must reach no pair-test at all"
        );

        // (3) Stop flips during the pair-test pass, 10 rows into the bucket —
        //     far before the pair sits at outer position M-2 (examined M+M-2).
        //     Sentinel (the actual IS_LASSO_300 leak): reverting the pair-test
        //     per-row check makes the bucket run to completion anyway, so
        //     `examined == M + 10` fails (it becomes 2M) and `removed` becomes
        //     [M-1] instead of empty.
        let signal = reset(Some(M + 10));
        let (prob, mut ws) = single_bucket_fixture();
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&signal)),
            ..SolverOptions::default()
        };
        assert!(step8_parallel_row(&prob, &mut ws, &opts).is_ok());
        assert_eq!(
            examined(),
            M + 10,
            "the pair-test pass must honor the stop the row it flips on, \
             not finish the bucket"
        );
        assert!(
            removed_rows(&ws).is_empty(),
            "the parallel pair at bucket positions {}/{} must never be \
             reached once the stop fired at row 10; got {:?}",
            M - 2,
            M - 1,
            removed_rows(&ws)
        );

        reset(None);
    }
}
