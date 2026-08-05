//! QP presolve Phase 2: equality-constraint redundancy elimination, near-zero Q
//! off-diagonal pruning, and row-norm constraint preconditioning.

use super::qp_transforms::{QpPostsolveStep, QpPresolveResult};
use crate::options::SolverOptions;
use crate::qp::QpProblem;
use crate::tolerances::{DROP_TOL, SCALING_SIGMA_FLOOR, ZERO_TOL};
use otspot_num::sparse::CscMatrix;
#[cfg(test)]
use std::sync::atomic::AtomicBool;

// Test-only observability, entirely `#[cfg(test)]` (definition and every
// call site below), so it has zero footprint in production builds. Counts
// how many of `run_qp_presolve_phase2`'s 5 `cancellable`-guarded tail steps
// (q_preserved clone / equality_constraint_qr / CSC rebuild /
// constraint_precond / QpProblem::new) actually ran; mirrors
// `qp_transforms::driver`'s `STEPS_EXECUTED_TOTAL`.
//
// `PHASE2_CANCEL_AFTER_STEPS`/`PHASE2_CANCEL_SIGNAL` piggyback on the same
// counter for a deterministic stand-in for a real cancel/deadline race: once
// the executed count reaches the configured target, `PHASE2_CANCEL_SIGNAL`
// flips, fed to `run_qp_presolve_phase2` via `SolverOptions::cancel_flag`
// (an `Arc` clone of the same `AtomicBool`) -- the exact `cancellable`/
// `external_stop_requested` path a real deadline or `Ctrl-C` takes, without
// racing wall-clock time.
#[cfg(test)]
thread_local! {
    static PHASE2_STEPS_EXECUTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PHASE2_CANCEL_AFTER_STEPS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static PHASE2_CANCEL_SIGNAL: std::sync::Arc<AtomicBool> = std::sync::Arc::new(AtomicBool::new(false));
}

#[cfg(test)]
fn test_record_phase2_step_executed() {
    let executed = PHASE2_STEPS_EXECUTED.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if PHASE2_CANCEL_AFTER_STEPS.with(std::cell::Cell::get) == Some(executed) {
        PHASE2_CANCEL_SIGNAL.with(|flag| flag.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

/// Minimum ratio of rows to columns for equality-constraint QR elimination.
/// Elimination cost is O(mn²) and only pays off in strongly over-determined
/// systems (m > n * ROW_OVERDETERMINED_RATIO).
const ROW_OVERDETERMINED_RATIO: usize = 2;

/// Pivot candidate is treated as zero when its absolute value falls below this
/// threshold during partial-pivot Gaussian elimination.
const PIVOT_CANDIDATE_ZERO_TOL: f64 = 1e-10;

/// A row's max absolute coefficient must exceed `1.0 + SCALE_EXCESS_TOL` to
/// trigger per-row normalisation; rows near 1.0 are left unscaled.
const SCALE_EXCESS_TOL: f64 = 1e-10;

/// Maximum `m * n` product before equality-constraint QR elimination is skipped.
///
/// QR elimination is O(mn²); at 10⁸ the dense multiply already takes ~seconds for
/// typical n~10³ problems.  Problems this size are unlikely to have large numbers of
/// Le-Le equality pairs anyway, so the skip is loss-free in practice.
const QR_SKIP_SIZE_THRESHOLD: usize = 100_000_000;

/// Quantisation factor for RHS hashing in the Le-Le equality pair detector.
///
/// `|b[i]| * RHS_HASH_QUANTIZE` is rounded to `i64` so that rows whose RHS values
/// agree within ~1e-9 relative hash into the same bucket.  Collisions between truly
/// distinct rows are harmless — the exact comparison below the bucket lookup rejects
/// non-pairs.  The value 1e9 was chosen so that the quantisation error (~1e-9) is
/// well below `ZERO_TOL` (1e-12 × scale) for typical RHS magnitudes.
const RHS_HASH_QUANTIZE: f64 = 1e9;

/// Detect Le-Le pairs that form an equality (A\[j,*\] = -A\[i,*\] and b\[j\] = -b\[i\]) and
/// drop redundant equality rows via partial-pivot Gaussian elimination. Only runs when
/// `m > 2n` since the elimination cost is O(mn²).
fn equality_constraint_qr(prob: &QpProblem, removed_rows: &mut [bool], opts: &SolverOptions) {
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    let n = prob.num_vars;
    let m = prob.num_constraints;

    if m * n > QR_SKIP_SIZE_THRESHOLD || m <= n * ROW_OVERDETERMINED_RATIO || n == 0 {
        return;
    }

    // Restrict pair detection to Le rows; pairing Eq/Ge rows with Le would corrupt the problem.
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
    for j in 0..n {
        let start = prob.a.col_ptr()[j];
        let end = prob.a.col_ptr()[j + 1];
        for k in start..end {
            let row = prob.a.row_ind()[k];
            if !removed_rows[row]
                && matches!(
                    prob.constraint_types[row],
                    crate::problem::ConstraintType::Le
                )
            {
                row_entries[row].push((j, prob.a.values()[k]));
            }
        }
    }

    // Hash-bucket rows by (nnz, column-pattern, |b|) so pair candidates stay in small groups.
    let col_pattern_hash = |entries: &[(usize, f64)]| -> u64 {
        let mut h = DefaultHasher::new();
        for &(col, _) in entries {
            col.hash(&mut h);
        }
        h.finish()
    };

    let mut groups: HashMap<(usize, u64, i64), Vec<usize>> = HashMap::new();
    for i in 0..m {
        if removed_rows[i] || row_entries[i].is_empty() {
            continue;
        }
        let ch = col_pattern_hash(&row_entries[i]);
        let bk = (prob.b[i].abs() * RHS_HASH_QUANTIZE).round() as i64;
        groups
            .entry((row_entries[i].len(), ch, bk))
            .or_default()
            .push(i);
    }

    let mut eq_pos_rows: Vec<usize> = Vec::new();
    let mut paired = vec![false; m];
    let mut pair_partner: Vec<usize> = vec![usize::MAX; m];

    for group in groups.values() {
        for &i in group {
            if paired[i] {
                continue;
            }
            for &j in group {
                if j <= i || paired[j] {
                    continue;
                }
                let entries_i = &row_entries[i];
                let entries_j = &row_entries[j];
                let b_i = prob.b[i];

                if (b_i + prob.b[j]).abs() > ZERO_TOL * (1.0 + b_i.abs()) {
                    continue;
                }
                let is_neg = entries_i
                    .iter()
                    .zip(entries_j.iter())
                    .all(|((c1, v1), (c2, v2))| {
                        *c1 == *c2 && (v1 + v2).abs() < ZERO_TOL * (1.0 + v1.abs())
                    });
                if is_neg {
                    eq_pos_rows.push(i);
                    paired[i] = true;
                    paired[j] = true;
                    pair_partner[i] = j;
                    break;
                }
            }
        }
    }

    let m_eq = eq_pos_rows.len();
    if m_eq == 0 {
        return;
    }

    // Dense Aeq (m_eq × n) for partial-pivot Gaussian elimination.
    let mut aeq = vec![vec![0.0f64; n]; m_eq];
    for (row_idx, &orig_row) in eq_pos_rows.iter().enumerate() {
        for &(col, val) in &row_entries[orig_row] {
            aeq[row_idx][col] = val;
        }
    }

    let mut pivot_rows: Vec<bool> = vec![false; m_eq];
    let mut pivot_count = 0usize;
    let mut used_pivot_col = vec![false; n];
    let mut work = aeq.clone();

    for col in 0..n {
        // O(m_eq * n) per column, checked once per outer iteration (cheap
        // next to that cost) -- same gap as
        // `dual_advanced::phase1::farkas_infeasibility_certified`'s probe loop.
        //
        // `return`, not `break`: a row not yet visited hasn't been *proven*
        // dependent on the pivots found so far. The "drop every non-pivot
        // row" pass below is only sound once every column had its chance at
        // a pivot; falling through early would drop rows never shown
        // redundant -- silently relaxing the problem, not just a missed
        // optimization. `return` leaves `removed_rows` at the caller's
        // all-`false` initial state, same as the size-cap check above.
        if opts.external_stop_requested() {
            return;
        }
        let mut max_val = 0.0f64;
        let mut max_row = usize::MAX;
        for row in 0..m_eq {
            if pivot_rows[row] {
                continue;
            }
            let v = work[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_row == usize::MAX || max_val < PIVOT_CANDIDATE_ZERO_TOL || used_pivot_col[col] {
            continue;
        }

        pivot_rows[max_row] = true;
        used_pivot_col[col] = true;
        pivot_count += 1;

        let pivot = work[max_row][col];
        for k in 0..m_eq {
            if k == max_row {
                continue;
            }
            let factor = work[k][col] / pivot;
            if factor.abs() < DROP_TOL {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for c in 0..n {
                let delta = factor * work[max_row][c];
                work[k][c] -= delta;
            }
        }

        if pivot_count >= n {
            break;
        }
    }

    // Drop non-pivot rows (and their Le partners) — O(m_eq) via `pair_partner`.
    for (row_idx, &orig_row) in eq_pos_rows.iter().enumerate() {
        if !pivot_rows[row_idx] {
            removed_rows[orig_row] = true;
            let partner = pair_partner[orig_row];
            if partner != usize::MAX {
                removed_rows[partner] = true;
            }
        }
    }
}

/// Normalise constraint rows by `σ_i = max|A[i,*]|⁻¹` (capped at `SCALING_SIGMA_FLOOR`).
/// Improves KKT-matrix conditioning. Returns per-row scales for dual unscaling.
fn constraint_precond(a: &mut CscMatrix, b: &mut [f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();

    let mut row_max = vec![0.0f64; m];
    for col in 0..n {
        let start = a.col_ptr()[col];
        let end = a.col_ptr()[col + 1];
        for k in start..end {
            let row = a.row_ind()[k];
            let v = a.values()[k].abs();
            if v > row_max[row] {
                row_max[row] = v;
            }
        }
    }

    // SCALING_SIGMA_FLOOR caps the per-stage amplification at 1e3 so total
    // amp (phase1·phase2·Ruiz) stays within the IPM's achievable scaled tolerance.
    let sigmas: Vec<f64> = row_max
        .iter()
        .map(|&mx| {
            if mx > 1.0 + SCALE_EXCESS_TOL {
                (1.0 / mx).max(SCALING_SIGMA_FLOOR)
            } else {
                1.0
            }
        })
        .collect();

    let has_any = sigmas.iter().any(|&s| (s - 1.0).abs() > ZERO_TOL);
    if !has_any {
        return sigmas;
    }

    for col in 0..n {
        let start = a.col_ptr()[col];
        let end = a.col_ptr()[col + 1];
        for k in start..end {
            let row = a.row_ind()[k];
            a.values_mut()[k] *= sigmas[row];
        }
    }

    for i in 0..m {
        b[i] *= sigmas[i];
    }

    sigmas
}

/// Runs `work`, honoring cancellation both before and after -- not just
/// before. `equality_constraint_qr` has checkless fast paths (the top-level
/// size-guard skip, and the `m_eq == 0` early return after its row-scan/
/// pairing pass finds nothing to eliminate) that can still cost real time on
/// a large QP without ever consulting `cancel_flag` internally. Checking only
/// beforehand would let a cancellation requested *during* one of those
/// checkless paths go unnoticed until the caller's next unrelated check (or
/// never, if there isn't one) -- Python's `Model.solve()` signal-poll loop
/// stays blocked in `handle.join()` for all of that.
fn cancellable<T>(opts: &SolverOptions, work: impl FnOnce() -> T) -> Option<T> {
    if opts.external_stop_requested() {
        return None;
    }
    let result = work();
    if opts.external_stop_requested() {
        return None;
    }
    Some(result)
}

/// Run Phase 2 of QP presolve on a Phase-1 result: redundant-equality removal,
/// near-zero Q pruning, and row-norm preconditioning.
pub fn run_qp_presolve_phase2(
    phase1_result: QpPresolveResult,
    opts: &SolverOptions,
) -> QpPresolveResult {
    let prob = &phase1_result.reduced;
    let n = prob.num_vars;
    let m = prob.num_constraints;

    if n == 0 || m == 0 {
        return phase1_result;
    }

    // Coefficient magnitude alone cannot make a Q term semantically zero: the
    // complete objective may simply be expressed in correspondingly small units.
    // Wrapped in `cancellable` (not a bare check-then-clone): this is the
    // former entry check 0821b27d dropped when it added the equality_
    // constraint_qr wrapper below, and Q's clone cost scales with its own
    // nnz, not just m/n, so it deserves the same pre+post guard as every
    // other O(problem-size) step in this function.
    let q_preserved = match cancellable(opts, || prob.q.clone()) {
        Some(q) => q,
        None => return phase1_result,
    };
    #[cfg(test)]
    test_record_phase2_step_executed();

    let removed_rows_phase2 = match cancellable(opts, || {
        let mut removed = vec![false; m];
        equality_constraint_qr(prob, &mut removed, opts);
        removed
    }) {
        Some(r) => r,
        None => return phase1_result,
    };
    #[cfg(test)]
    test_record_phase2_step_executed();

    let any_removed = removed_rows_phase2.iter().any(|&b| b);

    // Reuse the map outside this scope for row_scales / row_map syncing too.
    let new_row_map: Vec<Option<usize>> = {
        let mut map = vec![None; m];
        let mut idx = 0usize;
        for i in 0..m {
            if !removed_rows_phase2[i] {
                map[i] = Some(idx);
                idx += 1;
            }
        }
        map
    };

    // CSC rebuild (or, when nothing was removed, the equivalent plain clone):
    // O(nnz(A)) either way, and previously ran unconditionally once the
    // equality_constraint_qr call above returned `Some` -- a cancellation
    // requested in that exact window went unnoticed until this function's
    // own return. `cancellable`'s `Option<Option<_>>` collapses via
    // `.flatten()`: cancellation (outer `None`) and a failed transactional
    // rebuild (inner `None`) both mean "bail out with `phase1_result`
    // unchanged", so they share the one early-return below.
    let (a_new, b_new) = match cancellable(opts, || {
        if any_removed {
            let m_new = new_row_map.iter().filter(|o| o.is_some()).count();

            let mut trip_rows: Vec<usize> = Vec::new();
            let mut trip_cols: Vec<usize> = Vec::new();
            let mut trip_vals: Vec<f64> = Vec::new();
            for j in 0..n {
                let start = prob.a.col_ptr()[j];
                let end = prob.a.col_ptr()[j + 1];
                for k in start..end {
                    let row = prob.a.row_ind()[k];
                    if let Some(ii) = new_row_map[row] {
                        trip_rows.push(ii);
                        trip_cols.push(j);
                        trip_vals.push(prob.a.values()[k]);
                    }
                }
            }
            let a_out = if trip_rows.is_empty() {
                Some(CscMatrix::new(m_new, n))
            } else {
                // Phase 2 is transactional: a failed rebuild must retain the valid
                // Phase-1 problem, never replace its constraints with a zero matrix.
                CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n).ok()
            };

            let b_out: Vec<f64> = (0..m)
                .filter(|&i| !removed_rows_phase2[i])
                .map(|i| prob.b[i])
                .collect();

            a_out.map(|a| (a, b_out))
        } else {
            Some((prob.a.clone(), prob.b.clone()))
        }
    })
    .flatten()
    {
        Some(pair) => pair,
        None => return phase1_result,
    };
    #[cfg(test)]
    test_record_phase2_step_executed();

    let mut a_precond = a_new;
    let mut b_precond = b_new;
    let sigmas = match cancellable(opts, || constraint_precond(&mut a_precond, &mut b_precond)) {
        Some(s) => s,
        None => return phase1_result,
    };
    #[cfg(test)]
    test_record_phase2_step_executed();

    let reduced_new = match cancellable(opts, || {
        let constraint_types_new: Vec<crate::problem::ConstraintType> = (0..m)
            .filter(|&i| !removed_rows_phase2[i])
            .map(|i| prob.constraint_types[i])
            .collect();
        let c_clone = prob.c.clone();
        let bounds_clone = prob.bounds.clone();
        QpProblem::new(
            q_preserved,
            c_clone,
            a_precond,
            b_precond,
            bounds_clone,
            constraint_types_new,
        )
    }) {
        Some(Ok(p)) => p,
        Some(Err(_)) | None => return phase1_result,
    };
    #[cfg(test)]
    test_record_phase2_step_executed();

    let mut result = QpPresolveResult {
        reduced: reduced_new,
        was_reduced: phase1_result.was_reduced || any_removed,
        ..phase1_result
    };

    // When Phase 2 drops rows, compose the phase1 row_map with new_row_map, and
    // contract any phase1 LargeCoeffRowScale entries to match the new row indexing —
    // otherwise postsolve maps reduced duals through stale indices and applies wrong scales.
    if any_removed {
        for entry in result.row_map.iter_mut() {
            if let Some(phase1_i) = *entry {
                *entry = if phase1_i < new_row_map.len() {
                    new_row_map[phase1_i]
                } else {
                    None
                };
            }
        }
        for step in result.postsolve_stack.steps.iter_mut() {
            if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
                if row_scales.len() == m {
                    let compacted: Vec<f64> = (0..m)
                        .filter(|&i| !removed_rows_phase2[i])
                        .map(|i| row_scales[i])
                        .collect();
                    *row_scales = compacted;
                }
            }
        }
    }

    let has_precond_scaling = sigmas.iter().any(|&s| (s - 1.0).abs() > ZERO_TOL);
    if has_precond_scaling {
        result
            .postsolve_stack
            .push(QpPostsolveStep::LargeCoeffRowScale { row_scales: sigmas });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::qp::QpProblem;
    use otspot_num::sparse::CscMatrix;

    fn make_qp_simple(n: usize, m: usize) -> QpProblem {
        // 対角 Q=2I, c=0, A=I (truncated), b=1, bounds無限
        let q = CscMatrix::from_triplets(
            &(0..n).collect::<Vec<_>>(),
            &(0..n).collect::<Vec<_>>(),
            &vec![2.0; n],
            n,
            n,
        )
        .unwrap();
        let a_m = m.min(n);
        let a = CscMatrix::from_triplets(
            &(0..a_m).collect::<Vec<_>>(),
            &(0..a_m).collect::<Vec<_>>(),
            &vec![1.0; a_m],
            m,
            n,
        )
        .unwrap();
        let b = vec![1.0; m];
        QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap()
    }

    #[test]
    fn test_constraint_precond_scales_large_rows() {
        // A行列の行1の係数が大きい場合にスケールされること
        let n = 2usize;
        let m = 2usize;
        let mut a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1000.0, 1000.0],
            m,
            n,
        )
        .unwrap();
        let mut b = vec![1.0, 1000.0];
        let sigmas = constraint_precond(&mut a, &mut b);
        // 行0: max=1.0 → σ=1.0（変化なし）
        // 行1: max=1000.0 → σ=0.001
        assert!((sigmas[0] - 1.0).abs() < 1e-10, "row0 unchanged");
        assert!(
            (sigmas[1] - 0.001).abs() < 1e-7,
            "row1 scaled: σ={}",
            sigmas[1]
        );
        // b[1] がスケールされていること
        assert!((b[1] - 1.0).abs() < 1e-7, "b[1] scaled: {}", b[1]);
    }

    #[test]
    fn test_run_qp_presolve_phase2_no_crash() {
        let prob = make_qp_simple(3, 2);
        let opts = SolverOptions::default();
        let phase1 = crate::presolve::run_qp_presolve_phase1(&prob, &opts);
        let phase2 = run_qp_presolve_phase2(phase1, &opts);
        assert_eq!(
            phase2.orig_num_vars, 3,
            "orig_num_vars preserved through phase2"
        );
        assert_eq!(
            phase2.orig_num_constraints, 2,
            "orig_num_constraints preserved through phase2"
        );
    }

    /// Preset `cancel_flag=true` must make `run_qp_presolve_phase2` return
    /// `phase1_result` unchanged, the same way an already-expired `deadline`
    /// does -- the entry check used to look at `deadline` only, never
    /// `cancel_flag` (Codex PR #31 review, item 3).
    ///
    /// `num_constraints` alone can't tell this apart from the in-loop check
    /// inside `equality_constraint_qr` catching it one statement later (both
    /// leave it unchanged). What the entry check specifically saves is the
    /// row-entry scan, hash-bucketing/pairing, and dense `aeq`/`work`
    /// allocation `equality_constraint_qr` does *before* its loop starts --
    /// only visible on a large enough problem for that setup to cost real
    /// time, hence reusing the mid-loop test's chain construction
    /// (n=600, 3 copies, m=3594).
    ///
    /// Measured (5 trials): entry check present, ~1-7us; reverted to
    /// `deadline`-only (in-loop check alone still catches it, but only after
    /// paying for the setup), ~7.7-10.3ms. Sentinel: reverting the entry
    /// check confirmed this exceeds the 1ms bound below.
    #[test]
    fn test_run_qp_presolve_phase2_honors_preset_cancel_flag() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let n = 600usize;
        let copies = 3usize;
        let m = 2 * copies * (n - 1);
        let mut trip_rows = Vec::with_capacity(2 * m);
        let mut trip_cols = Vec::with_capacity(2 * m);
        let mut trip_vals = Vec::with_capacity(2 * m);
        let mut b = Vec::with_capacity(m);
        for copy in 0..copies {
            for i in 0..(n - 1) {
                let pos_row = 2 * (copy * (n - 1) + i);
                let neg_row = pos_row + 1;
                trip_rows.push(pos_row);
                trip_cols.push(i);
                trip_vals.push(1.0);
                trip_rows.push(pos_row);
                trip_cols.push(i + 1);
                trip_vals.push(1.0);
                b.push(5.0);
                trip_rows.push(neg_row);
                trip_cols.push(i);
                trip_vals.push(-1.0);
                trip_rows.push(neg_row);
                trip_cols.push(i + 1);
                trip_vals.push(-1.0);
                b.push(-5.0);
            }
        }
        let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n).unwrap();
        let q_idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&q_idx, &q_idx, &vec![2.0; n], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();

        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            presolve: false, // phase1 itself must not reduce either, isolating phase2's own behavior
            ..Default::default()
        };
        let phase1 = crate::presolve::run_qp_presolve_phase1(&prob, &opts);
        let phase1_constraints = phase1.reduced.num_constraints;
        assert_eq!(
            phase1_constraints, m,
            "phase1 (presolve=false) must not reduce"
        );

        let t0 = Instant::now();
        let phase2 = run_qp_presolve_phase2(phase1, &opts);
        let elapsed = t0.elapsed();

        assert_eq!(
            phase2.reduced.num_constraints, phase1_constraints,
            "preset cancel_flag=true must skip phase2's own reduction \
             entirely (num_constraints unchanged from phase1's {phase1_constraints}), \
             not run equality_constraint_qr"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "preset cancel_flag=true took {elapsed:?} to return from \
             run_qp_presolve_phase2 -- expected the entry check to skip \
             equality_constraint_qr's row-scan/pairing/dense-matrix setup \
             entirely (measured ~1-7us), not merely have the in-loop check \
             catch it after paying for that setup (measured ~7.7-10.3ms \
             with the entry check reverted to deadline-only)"
        );
    }

    /// `run_qp_presolve_phase2`'s CSC row-map rebuild, `constraint_precond`,
    /// and `QpProblem::new` validation ran unconditionally once
    /// `equality_constraint_qr`'s `cancellable` wrapper returned `Some`
    /// (Codex PR #31 review follow-up, on top of 0821b27d): none re-checked
    /// `external_stop_requested()`, so a cancellation in that window went
    /// unnoticed until this function's own return.
    ///
    /// Deterministic, not wall-clock: the original version of this sentinel
    /// raced a background thread's `sleep` against a 2,000,000-variable
    /// fixture's measured uncancelled duration and was reproducibly flaky
    /// under `nextest`'s parallel execution (`cancelled=113ms` against a
    /// `<103.7ms` threshold on one contended run, 73/73 moments later
    /// uncontended). `PHASE2_CANCEL_AFTER_STEPS`/`PHASE2_CANCEL_SIGNAL` (doc
    /// comment above `test_record_phase2_step_executed`) instead flip
    /// `cancel_flag` as a synchronous side effect of the `n`th guarded step
    /// completing -- no thread, no sleep, so the flip lands identically
    /// every run.
    ///
    /// For `cancel_after` in `1..=4` (of the 5 guarded steps), asserts
    /// execution stops at exactly that step count and the result is
    /// `phase1_result` unchanged (no unreached step's work leaked out). A
    /// trailing uncancelled run confirms all 5 steps execute normally.
    ///
    /// Sentinel: reverting the `cancellable` wrapping on any of the CSC
    /// rebuild / `constraint_precond` / `QpProblem::new` steps makes
    /// `cancel_after` in `{3, 4}` run past their target to 5, failing
    /// `executed == cancel_after`.
    #[test]
    fn test_run_qp_presolve_phase2_tail_stops_at_cancellation_point() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        // Same n=2/m=6 redundant-equality fixture as
        // `test_equality_constraint_qr_redundant_removal` (2 of the 6 rows
        // are an exact duplicate pair): `any_removed` is genuinely `true`
        // here, so `num_constraints` actually would drop from 6 if the CSC
        // rebuild (step 3) and its downstream steps ran to completion --
        // unlike a fixture with nothing to remove, where "unchanged" would
        // hold trivially regardless of whether the fix works.
        fn phase2_input() -> QpPresolveResult {
            let n = 2usize;
            let m = 6usize;
            let a = CscMatrix::from_triplets(
                &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                &[
                    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
                ],
                m,
                n,
            )
            .unwrap();
            let b = vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0];
            let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
            let prob = QpProblem::new_all_le(
                q,
                vec![0.0; n],
                a,
                b,
                vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            )
            .unwrap();
            QpPresolveResult::no_reduction(&prob)
        }

        for cancel_after in 1..=4usize {
            PHASE2_STEPS_EXECUTED.with(|c| c.set(0));
            PHASE2_CANCEL_AFTER_STEPS.with(|c| c.set(Some(cancel_after)));
            PHASE2_CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

            let phase1_result = phase2_input();
            let orig_num_constraints = phase1_result.reduced.num_constraints;
            let opts = PHASE2_CANCEL_SIGNAL.with(|flag| SolverOptions {
                cancel_flag: Some(Arc::clone(flag)),
                ..SolverOptions::default()
            });
            let phase2 = run_qp_presolve_phase2(phase1_result, &opts);

            let executed = PHASE2_STEPS_EXECUTED.with(|c| c.get());
            assert_eq!(
                executed, cancel_after,
                "cancel_flag flips right after step {cancel_after} of 5 completes, \
                 so run_qp_presolve_phase2 must stop there (not run the remaining \
                 steps), got {executed} steps executed"
            );
            assert_eq!(
                phase2.reduced.num_constraints, orig_num_constraints,
                "cancellation after step {cancel_after} must discard the tail's \
                 side effects entirely, returning phase1_result unchanged"
            );
        }

        PHASE2_STEPS_EXECUTED.with(|c| c.set(0));
        PHASE2_CANCEL_AFTER_STEPS.with(|c| c.set(None));
        let phase2 = run_qp_presolve_phase2(phase2_input(), &SolverOptions::default());
        let executed = PHASE2_STEPS_EXECUTED.with(|c| c.get());
        assert_eq!(
            executed, 5,
            "without cancellation all 5 guarded steps must run, got {executed}"
        );
        assert!(
            phase2.reduced.num_constraints < 6,
            "sanity: uncancelled run must actually remove the redundant pair \
             (num_constraints < 6), confirming the fixture's `any_removed=true` \
             premise the loop above depends on; got {}",
            phase2.reduced.num_constraints
        );
    }

    /// `cancel_flag` firing *mid-elimination* (not preset before
    /// `equality_constraint_qr` starts, which the entry check above already
    /// covers) exercises the in-loop check specifically, and its
    /// correctness requirement: on cancellation, `equality_constraint_qr`
    /// must `return` (abort the whole function), not `break` the column
    /// loop and fall through to "drop every row that never became a pivot".
    /// A row that never got a chance to compete for a pivot (because
    /// elimination stopped partway through the columns) has not been
    /// *proven* linearly dependent on the pivots found so far -- treating
    /// it as redundant anyway would silently drop a real constraint
    /// (relaxing the problem), not just skip an optimization.
    ///
    /// n=600 distinct single-variable equalities (`x_i <= c_i` /
    /// `-x_i <= -c_i`, one Le-Le pair per variable, no duplicates): each
    /// falls into its own presolve pairing group (`nnz=1`, distinct column
    /// per group), so `m_eq = n = 600`, and the O(m_eq * n) dense
    /// elimination per column is O(n^3) total -- large enough to still be
    /// mid-loop when `cancel_flag` fires ~5ms in.
    ///
    /// Sentinel: reverting the in-loop check's `return` back to `break`
    /// makes this test fail (`removed_count` becomes nonzero: the
    /// truncated pivot set makes every not-yet-visited row look
    /// non-pivot/redundant and drops it, even though `equality_constraint_qr`
    /// never proved any of them dependent). Confirmed by reverting and
    /// re-running.
    #[test]
    fn test_equality_constraint_qr_mid_loop_cancel_aborts_without_dropping_rows() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        // Chain structure (x_i + x_{i+1} = 5), not n independent
        // single-variable pairs: the latter makes every row already-diagonal
        // (disjoint columns), so the elimination factor is always exactly
        // zero and skipped -- finishes in microseconds, never catching a
        // mid-loop cancel. Overlapping columns force genuine O(m_eq * n)
        // work per pivot, taking long enough (ms) for the spawned thread's
        // cancel to land mid-elimination. `copies` duplicate chains so
        // `m = 2 * copies * (n-1) > n * ROW_OVERDETERMINED_RATIO` (a single
        // copy alone sits just under `2n`, under the threshold).
        let n = 600usize;
        let copies = 3usize;
        let m = 2 * copies * (n - 1);
        let mut trip_rows = Vec::with_capacity(2 * m);
        let mut trip_cols = Vec::with_capacity(2 * m);
        let mut trip_vals = Vec::with_capacity(2 * m);
        let mut b = Vec::with_capacity(m);
        for copy in 0..copies {
            for i in 0..(n - 1) {
                let pos_row = 2 * (copy * (n - 1) + i);
                let neg_row = pos_row + 1;
                trip_rows.push(pos_row);
                trip_cols.push(i);
                trip_vals.push(1.0);
                trip_rows.push(pos_row);
                trip_cols.push(i + 1);
                trip_vals.push(1.0);
                b.push(5.0);
                trip_rows.push(neg_row);
                trip_cols.push(i);
                trip_vals.push(-1.0);
                trip_rows.push(neg_row);
                trip_cols.push(i + 1);
                trip_vals.push(-1.0);
                b.push(-5.0);
            }
        }
        let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n).unwrap();
        let q_idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&q_idx, &q_idx, &vec![2.0; n], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_setter = Arc::clone(&cancel);
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            cancel_setter.store(true, Ordering::Relaxed);
        });

        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            ..Default::default()
        };
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed, &opts);
        setter.join().unwrap();

        let removed_count = removed.iter().filter(|&&b| b).count();
        assert_eq!(
            removed_count, 0,
            "mid-loop cancellation must abort equality_constraint_qr \
             entirely (removed_rows left at its all-false initial state), \
             not drop rows that were never proven redundant; got {removed_count} removed"
        );
    }

    /// `cancellable` must not treat "cancellation wasn't requested before
    /// `work` started" as sufficient -- `work` itself may be the thing that
    /// makes cancellation true (standing in for `equality_constraint_qr`'s
    /// checkless fast paths, where real wall-clock time passes between the
    /// entry check and the moment the caller finds out). A side-effecting
    /// closure makes this deterministic: no thread, no timing, no race --
    /// reverting the post-`work` check back out (leaving only the entry
    /// check) makes this assert `Some(42)` instead, since nothing before
    /// `work` runs ever observes the flag flipping during it.
    #[test]
    fn test_cancellable_rechecks_after_work_not_just_before() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let cancel = Arc::new(AtomicBool::new(false));
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            ..Default::default()
        };

        let result = cancellable(&opts, || {
            cancel.store(true, Ordering::Relaxed);
            42
        });

        assert_eq!(
            result, None,
            "cancellable must recheck after `work` completes, not only before it starts"
        );
    }

    #[test]
    fn test_cancellable_runs_work_when_never_cancelled() {
        let opts = SolverOptions::default();
        assert_eq!(
            cancellable(&opts, || 42),
            Some(42),
            "cancellable must return work's result when cancellation is never requested"
        );
    }

    #[test]
    fn test_cancellable_skips_work_when_preset_before_call() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            ..Default::default()
        };
        let mut ran = false;
        let result = cancellable(&opts, || {
            ran = true;
            42
        });
        assert_eq!(result, None);
        assert!(
            !ran,
            "cancellable must not run work at all when already cancelled at entry"
        );
    }

    #[test]
    fn test_equality_constraint_qr_redundant_removal() {
        // m=6, n=2: 3 等式制約ペア。うち2つは冗長（同一）。→ 1ペアのみ残す
        // 等式: x+y=1 (redundant pair: 2つ), x-y=0 (1つ)
        // Le 制約として:  x+y<=1, -(x+y)<=-1 × 2, x-y<=0, -(x-y)<=0
        // → m=6 > n*2=4 → QR 適用
        let n = 2usize;
        let m = 6usize;
        // rows 0,1: x+y<=1 と -(x+y)<=-1
        // rows 2,3: x+y<=1 と -(x+y)<=-1 (重複)
        // rows 4,5: x-y<=0 と -(x-y)<=0
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            &[
                1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
            ],
            m,
            n,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0];
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed, &SolverOptions::default());
        // 少なくとも1行が除去されているべき（重複行）
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert!(
            removed_count >= 2,
            "at least one redundant pair removed, got {}",
            removed_count
        );
    }

    #[test]
    fn failed_constraint_rebuild_keeps_phase1_problem_intact() {
        let n = 2usize;
        let m = 5usize;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[0, 1, 0, 1, 0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            m,
            n,
        )
        .unwrap();
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let mut prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            vec![1.0, -1.0, 1.0, -1.0, 5.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();
        // Model a finite-input transform overflow after construction. Row 4 is
        // retained by QR, so rebuilding the reduced A must reject this value.
        let row4_pos = prob.a.row_ind().iter().position(|&row| row == 4).unwrap();
        prob.a.values_mut()[row4_pos] = f64::INFINITY;
        let phase1 = QpPresolveResult::no_reduction(&prob);

        let result = run_qp_presolve_phase2(phase1, &SolverOptions::default());

        assert_eq!(result.reduced.num_constraints, m);
        assert_eq!(result.reduced.a.nnz(), 9);
        assert!(result.reduced.a.values().iter().any(|v| v.is_infinite()));
        assert!(!result.was_reduced);
    }

    /// Sentinel: ROW_OVERDETERMINED_RATIO boundary — m = n*2 skips QR (skip path).
    ///
    /// **Sentinel**: changing ROW_OVERDETERMINED_RATIO from 2 to 1 activates QR at m=2n,
    /// which removes redundant rows → removed_count > 0 → this test FAIL.
    #[test]
    fn equality_constraint_qr_skip_at_boundary_m_eq_2n() {
        // n=2, m=4 = n*ROW_OVERDETERMINED_RATIO: condition `m <= n*2` is true → skip.
        // Even with a redundant Le-Le pair present, nothing is removed.
        let n = 2usize;
        let m = 4usize; // exactly n*ROW_OVERDETERMINED_RATIO
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            m,
            n,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0]; // rows 0,1 and rows 2,3 are the same Le-Le pair
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed, &SolverOptions::default());
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert_eq!(
            removed_count, 0,
            "m=n*ROW_OVERDETERMINED_RATIO: QR is skipped, nothing removed (got {})",
            removed_count
        );
    }

    /// Sentinel: ROW_OVERDETERMINED_RATIO boundary — m = n*2+1 runs QR (run path).
    ///
    /// **Sentinel**: changing ROW_OVERDETERMINED_RATIO from 2 to 3 makes `m <= n*3` true
    /// for m=5, n=2 → skips QR → removed_count = 0 → this test FAIL.
    #[test]
    fn equality_constraint_qr_runs_at_boundary_m_eq_2n_plus_1() {
        // n=2, m=5 = n*ROW_OVERDETERMINED_RATIO + 1: condition `m <= n*2` is false → run.
        let n = 2usize;
        let m = 5usize; // n*ROW_OVERDETERMINED_RATIO + 1
                        // Rows 0,1: x+y<=1 / -(x+y)<=-1  (Le-Le pair 1)
                        // Rows 2,3: same pair (redundant)
                        // Row  4: lone x<=5 (no pair, not removed)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[0, 1, 0, 1, 0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            m,
            n,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0, 5.0];
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; n],
            a,
            b,
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        )
        .unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed, &SolverOptions::default());
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert!(
            removed_count >= 2,
            "m > n*ROW_OVERDETERMINED_RATIO: QR runs and removes redundant rows (got {})",
            removed_count
        );
    }

    #[test]
    fn phase2_preserves_cross_terms_at_every_objective_unit_scale() {
        // Independent oracle for Q=s*[[2,1],[1,2]], c=s*[-3,0]:
        // Qx+c=0 gives x=(2,-1), objective=-3s. The cross term determines
        // both coordinates and is below the former absolute cutoff at s=1e-12.
        for scale in [1.0, 1e-12] {
            let q = CscMatrix::from_triplets(
                &[0, 1, 0, 1],
                &[0, 0, 1, 1],
                &[2.0 * scale, scale, scale, 2.0 * scale],
                2,
                2,
            )
            .unwrap();
            let prob = QpProblem::new_all_le(
                q.clone(),
                vec![-3.0 * scale, 0.0],
                CscMatrix::new(1, 2),
                vec![100.0],
                vec![(-10.0, 10.0); 2],
            )
            .unwrap();
            let phase1 = QpPresolveResult::no_reduction(&prob);
            let result = run_qp_presolve_phase2(phase1, &SolverOptions::default());

            assert_eq!(result.reduced.q.col_ptr(), q.col_ptr());
            assert_eq!(result.reduced.q.row_ind(), q.row_ind());
            assert_eq!(result.reduced.q.values(), q.values());
            assert_eq!(result.reduced.q.nnz(), 4);

            let x = [2.0, -1.0];
            let qx = result.reduced.q.mat_vec_mul(&x).unwrap();
            let objective = 0.5 * x.iter().zip(&qx).map(|(xj, qxj)| xj * qxj).sum::<f64>()
                + result
                    .reduced
                    .c
                    .iter()
                    .zip(x)
                    .map(|(cj, xj)| cj * xj)
                    .sum::<f64>();
            assert!((objective - (-3.0 * scale)).abs() <= scale * 1e-12);
        }
    }
}
