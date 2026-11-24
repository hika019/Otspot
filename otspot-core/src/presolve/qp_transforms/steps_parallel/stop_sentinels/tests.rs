//! Deterministic (work-count, not wall-clock) sentinels for the per-row
//! stop checks inside `step8_parallel_row`. Before this fix the step only
//! consulted the stop condition at bucket-entry boundaries, so a stop
//! requested mid-bucket (the shape of IS_LASSO_300's 2240-row bucket,
//! ~2.3s of unbounded `O(|bucket|)` pair-tests) was ignored until the
//! *next* bucket start — the dominant deadline-overshoot leak behind the
//! sporadic `qp_is_lasso_300_real_data_honors_deadline` failures.

// 移設前は steps_parallel.rs 内のインライン module で `use super::*;` だった。
// ファイル分離で一段深くなったため、参照先 (steps_parallel の item と import
// 束縛) はそのまま `super::super` 経由で同一集合を glob する。
use super::super::*;
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

/// sentinel: step8 bucket 内 pair-test の行単位 stop チェックを戻した時の張 —
/// cancel 後も行を調べ続けると examined が work-count 上限を超えて fail
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
