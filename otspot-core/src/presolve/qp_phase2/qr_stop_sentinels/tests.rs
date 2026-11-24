//! Deterministic (work-count, not wall-clock) sentinels for the per-item
//! stop checks in `equality_constraint_qr`'s scan / grouping / pair-test
//! passes. Before this fix those passes only ran under `cancellable`'s
//! before/after wrapper, so a stop arriving mid-pass went unnoticed until
//! the pass (and, for the O(Σ bucket²) pair test, all its buckets) ended
//! -- the same unbounded-leak shape `cancellable`'s doc called out as the
//! deadline-overshoot bound (`qp_is_lasso_300_real_data_honors_deadline`).

// 移設前は qp_phase2.rs 内のインライン module で `use super::*;` だった。
// ファイル分離で一段深くなったため、参照先 (qp_phase2 の item と import 束縛) は
// そのまま `super::super` 経由で同一集合を glob する。
use super::super::*;
use otspot_num::sparse::CscMatrix;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// All 120 rows sit in one Le-Le bucket ((2 nnz, {x0,x1}-pattern, |b|=1));
/// four of them form two exact pairs at the very end of the bucket order:
/// rows 116/117 (`x+y=1` vs `-(x+y)=-1` with equal b, pair kept) and rows
/// 118/119 (identical coefficients and RHS to 116/117, so 118's equality
/// is implied by 116's and both 118 and its partner 119 drop).
///
/// Independent oracle (hand elimination, mirroring the function's own
/// procedure): both pairs give `eq_pos_rows = [116, 118]` with RHS
/// (1, 1); col 0 pivots on row 116 (tie), row 118 annihilates exactly
/// (`b_work = 1 - 1·1 = 0` ≤ tol) → drop 118 + partner 119; pivot 116 and
/// its partner 117 stay. The 116 filler rows have b = +1 with per-row
/// distinct coefficients, so the `b_i + b_j ≈ 0` filter only ever lets
/// them meet rows 117/119 and `is_neg` rejects them; no filler pairs.
const M: usize = 120;
const N: usize = 2;
/// Work items a full pass examines: one per scanned column, per grouped
/// row, per pair-tested candidate row.
const FULL_ITEMS: usize = N + 2 * M;

fn single_bucket_fixture() -> QpProblem {
    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    let mut b = vec![0.0f64; M];
    for i in 0..M {
        let (v0, v1, rhs) = match i {
            116 | 118 => (2.0f64, 3.0f64, 1.0f64),
            117 | 119 => (-2.0f64, -3.0f64, -1.0f64),
            _ => (3.0 + 0.01 * (i as f64), 4.0 + 0.01 * (i as f64), 1.0f64),
        };
        a_rows.extend_from_slice(&[i, i]);
        a_cols.extend_from_slice(&[0, 1]);
        a_vals.extend_from_slice(&[v0, v1]);
        b[i] = rhs;
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, M, N).unwrap();
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], N, N).unwrap();
    QpProblem::new_all_le(
        q,
        vec![0.0; N],
        a,
        b,
        vec![(f64::NEG_INFINITY, f64::INFINITY); N],
    )
    .unwrap()
}

fn reset(cancel_after: Option<usize>) -> Arc<std::sync::atomic::AtomicBool> {
    QR_ROWS_EXAMINED.with(|c| c.set(0));
    QR_CANCEL_AFTER_ROWS.with(|c| c.set(cancel_after));
    QR_CANCEL_SIGNAL.with(|flag| {
        flag.store(false, Ordering::Relaxed);
        Arc::clone(flag)
    })
}

fn examined() -> usize {
    QR_ROWS_EXAMINED.with(|c| c.get())
}

fn run(prob: &QpProblem, cancel_after: Option<usize>) -> (bool, Vec<usize>) {
    let signal = reset(cancel_after);
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&signal)),
        ..SolverOptions::default()
    };
    let mut removed_rows = vec![false; prob.num_constraints];
    let consistent = equality_constraint_qr(prob, &mut removed_rows, &opts);
    let removed = (0..M).filter(|&i| removed_rows[i]).collect::<Vec<_>>();
    (consistent, removed)
}

/// sentinel: equality_constraint_qr の per-item stop チェックを戻した時の張 —
/// cancel 信号を次の pass まで無視すると examined が work-count 上限を超えて fail
#[test]
fn qr_per_item_stop_checks_bound_the_pre_elimination_leak() {
    let prob = single_bucket_fixture();
    // Premise (uncancelled): the duplicate pair drops, all passes run.
    let (consistent, removed) = run(&prob, None);
    assert!(consistent, "fixture is consistent");
    assert_eq!(
        removed,
        vec![118, 119],
        "hand-oracle: only the duplicate equality and its partner drop"
    );
    assert_eq!(examined(), FULL_ITEMS, "uncancelled run examines all");

    // Stop during the column scan. Sentinel: reverting the scan-loop
    // check defers the stop to the grouping loop's first row (item 2),
    // so `examined == 1` fails.
    let (consistent, removed) = run(&prob, Some(1));
    assert!(consistent, "a stop proves nothing, not infeasibility");
    assert_eq!(examined(), 1, "scan must honor the stop immediately");
    assert!(removed.is_empty(), "nothing removed once stopped");

    // Stop during grouping (scan examined N items; row 1 is item N+1...
    // N+2 flips at item 3, i.e. the second grouped row).
    // Sentinel: reverting the grouping check defers to the pair test's
    // first candidate (item N+M+1), so `examined == 3` fails.
    let (consistent, removed) = run(&prob, Some(N + 1));
    assert!(consistent);
    assert_eq!(examined(), N + 1, "grouping must honor the stop there");
    assert!(removed.is_empty());

    // Stop 10 rows into the pair-test pass, long before the real pairs at
    // candidate positions 116/118. Sentinel (the actual leak shape):
    // reverting the pair-test check lets all buckets run to completion,
    // so `examined` becomes FULL_ITEMS and `removed` [118, 119].
    let (consistent, removed) = run(&prob, Some(N + M + 10));
    assert!(consistent);
    assert_eq!(
        examined(),
        N + M + 10,
        "the pair-test pass must honor the stop at the row it flips on, \
         not finish the bucket"
    );
    assert!(removed.is_empty(), "pairs never reached once stopped");

    reset(None);
}
