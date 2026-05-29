//! QP presolve Phase 2: equality-constraint redundancy elimination, near-zero Q
//! off-diagonal pruning, and row-norm constraint preconditioning.

use crate::options::SolverOptions;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::{DROP_TOL, ZERO_TOL};
use super::qp_transforms::{QpPresolveResult, QpPostsolveStep};

/// Minimum ratio of rows to columns for equality-constraint QR elimination.
/// Elimination cost is O(mn²) and only pays off in strongly over-determined
/// systems (m > n * ROW_OVERDETERMINED_RATIO).
const ROW_OVERDETERMINED_RATIO: usize = 2;

/// Detect Le-Le pairs that form an equality (A\[j,*\] = -A\[i,*\] and b\[j\] = -b\[i\]) and
/// drop redundant equality rows via partial-pivot Gaussian elimination. Only runs when
/// `m > 2n` since the elimination cost is O(mn²).
pub fn equality_constraint_qr(
    prob: &QpProblem,
    removed_rows: &mut [bool],
) {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let n = prob.num_vars;
    let m = prob.num_constraints;

    const QR_SKIP_SIZE_THRESHOLD: usize = 100_000_000;
    if m * n > QR_SKIP_SIZE_THRESHOLD || m <= n * ROW_OVERDETERMINED_RATIO || n == 0 {
        return;
    }

    // Restrict pair detection to Le rows; pairing Eq/Ge rows with Le would corrupt the problem.
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
    for j in 0..n {
        let start = prob.a.col_ptr[j];
        let end = prob.a.col_ptr[j + 1];
        for k in start..end {
            let row = prob.a.row_ind[k];
            if !removed_rows[row]
                && matches!(prob.constraint_types[row], crate::problem::ConstraintType::Le)
            {
                row_entries[row].push((j, prob.a.values[k]));
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
        // Quantise |b| at 1e-9 so rows agreeing within rounding hash together; collisions
        // are harmless because the real comparison runs below.
        let bk = (prob.b[i].abs() * 1e9).round() as i64;
        groups.entry((row_entries[i].len(), ch, bk)).or_default().push(i);
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
                let is_neg = entries_i.iter().zip(entries_j.iter()).all(|((c1, v1), (c2, v2))| {
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

        if max_row == usize::MAX || max_val < 1e-10 || used_pivot_col[col] {
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

/// Drop Q off-diagonal entries with `|Q[i,j]| < EPS_Q` to improve sparsity.
pub fn near_zero_q_removal(q: &CscMatrix, n: usize) -> CscMatrix {
    const EPS_Q: f64 = 1e-10;

    let mut new_col_ptr = vec![0usize; n + 1];
    let mut new_row_ind: Vec<usize> = Vec::new();
    let mut new_values: Vec<f64> = Vec::new();

    for j in 0..n {
        let start = q.col_ptr[j];
        let end = q.col_ptr[j + 1];
        for k in start..end {
            let row = q.row_ind[k];
            let val = q.values[k];
            if row == j || val.abs() >= EPS_Q {
                new_row_ind.push(row);
                new_values.push(val);
            }
        }
        new_col_ptr[j + 1] = new_row_ind.len();
    }

    CscMatrix {
        nrows: q.nrows,
        ncols: n,
        col_ptr: new_col_ptr,
        row_ind: new_row_ind,
        values: new_values,
    }
}

/// Normalise constraint rows by `σ_i = max|A[i,*]|⁻¹` (capped at `SIGMA_FLOOR`).
/// Improves KKT-matrix conditioning. Returns per-row scales for dual unscaling.
pub fn constraint_precond(
    a: &mut CscMatrix,
    b: &mut [f64],
) -> Vec<f64> {
    let m = a.nrows;
    let n = a.ncols;

    let mut row_max = vec![0.0f64; m];
    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            let v = a.values[k].abs();
            if v > row_max[row] { row_max[row] = v; }
        }
    }

    // SIGMA_FLOOR caps the per-stage amplification at 1e3 so total
    // amp (phase1·phase2·Ruiz) stays within the IPM's achievable scaled tolerance.
    const SIGMA_FLOOR: f64 = 1e-3;
    let sigmas: Vec<f64> = row_max.iter().map(|&mx| {
        if mx > 1.0 + 1e-10 { (1.0 / mx).max(SIGMA_FLOOR) } else { 1.0 }
    }).collect();

    let has_any = sigmas.iter().any(|&s| (s - 1.0).abs() > 1e-12);
    if !has_any {
        return sigmas;
    }

    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            a.values[k] *= sigmas[row];
        }
    }

    for i in 0..m {
        b[i] *= sigmas[i];
    }

    sigmas
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

    if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return phase1_result;
    }

    let q_cleaned = near_zero_q_removal(&prob.q, n);

    let mut removed_rows_phase2 = vec![false; m];
    equality_constraint_qr(prob, &mut removed_rows_phase2);

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

    let (a_new, b_new) = if any_removed {
        let m_new = new_row_map.iter().filter(|o| o.is_some()).count();

        let mut trip_rows: Vec<usize> = Vec::new();
        let mut trip_cols: Vec<usize> = Vec::new();
        let mut trip_vals: Vec<f64> = Vec::new();
        for j in 0..n {
            let start = prob.a.col_ptr[j];
            let end = prob.a.col_ptr[j + 1];
            for k in start..end {
                let row = prob.a.row_ind[k];
                if let Some(ii) = new_row_map[row] {
                    trip_rows.push(ii);
                    trip_cols.push(j);
                    trip_vals.push(prob.a.values[k]);
                }
            }
        }
        let a_out = if trip_rows.is_empty() {
            CscMatrix::new(m_new, n)
        } else {
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n)
                .unwrap_or_else(|_| CscMatrix::new(m_new, n))
        };

        let b_out: Vec<f64> = (0..m)
            .filter(|&i| !removed_rows_phase2[i])
            .map(|i| prob.b[i])
            .collect();

        (a_out, b_out)
    } else {
        (prob.a.clone(), prob.b.clone())
    };

    let mut a_precond = a_new;
    let mut b_precond = b_new;
    let sigmas = constraint_precond(&mut a_precond, &mut b_precond);

    let constraint_types_new: Vec<crate::problem::ConstraintType> = (0..m)
        .filter(|&i| !removed_rows_phase2[i])
        .map(|i| prob.constraint_types[i])
        .collect();
    let c_clone = prob.c.clone();
    let bounds_clone = prob.bounds.clone();
    let reduced_new = match QpProblem::new(q_cleaned, c_clone, a_precond, b_precond, bounds_clone, constraint_types_new) {
        Ok(p) => p,
        Err(_) => return phase1_result,
    };

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

    let has_precond_scaling = sigmas.iter().any(|&s| (s - 1.0).abs() > 1e-12);
    if has_precond_scaling {
        result.postsolve_stack.push(QpPostsolveStep::LargeCoeffRowScale { row_scales: sigmas });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::qp::QpProblem;
    use crate::sparse::CscMatrix;

    fn make_qp_simple(n: usize, m: usize) -> QpProblem {
        // 対角 Q=2I, c=0, A=I (truncated), b=1, bounds無限
        let q = CscMatrix::from_triplets(
            &(0..n).collect::<Vec<_>>(),
            &(0..n).collect::<Vec<_>>(),
            &vec![2.0; n],
            n, n,
        ).unwrap();
        let a_m = m.min(n);
        let a = CscMatrix::from_triplets(
            &(0..a_m).collect::<Vec<_>>(),
            &(0..a_m).collect::<Vec<_>>(),
            &vec![1.0; a_m],
            m, n,
        ).unwrap();
        let b = vec![1.0; m];
        QpProblem::new_all_le(q, vec![0.0; n], a, b, vec![(f64::NEG_INFINITY, f64::INFINITY); n]).unwrap()
    }

    #[test]
    fn test_near_zero_q_removal_removes_small_offdiag() {
        // Q = [[2.0, 1e-15], [1e-15, 2.0]] → 非対角を除去
        let q = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1e-15, 1e-15, 2.0], 2, 2
        ).unwrap();
        let q_clean = near_zero_q_removal(&q, 2);
        // 非対角 (0,1)=(1,0) が除去されている
        let diag_count = q_clean.values.iter().zip(q_clean.row_ind.iter()).filter(|(_, &_r)| {
            // どの列かは不明なのでゼロ化された数を確認
            true
        }).count();
        // 非対角2要素が除去され対角2要素のみ残る
        assert_eq!(q_clean.values.len(), 2, "off-diag removed");
        let _ = diag_count;
    }

    #[test]
    fn test_constraint_precond_scales_large_rows() {
        // A行列の行1の係数が大きい場合にスケールされること
        let n = 2usize;
        let m = 2usize;
        let mut a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1],
            &[1.0, 1.0, 1000.0, 1000.0],
            m, n,
        ).unwrap();
        let mut b = vec![1.0, 1000.0];
        let sigmas = constraint_precond(&mut a, &mut b);
        // 行0: max=1.0 → σ=1.0（変化なし）
        // 行1: max=1000.0 → σ=0.001
        assert!((sigmas[0] - 1.0).abs() < 1e-10, "row0 unchanged");
        assert!((sigmas[1] - 0.001).abs() < 1e-7, "row1 scaled: σ={}", sigmas[1]);
        // b[1] がスケールされていること
        assert!((b[1] - 1.0).abs() < 1e-7, "b[1] scaled: {}", b[1]);
    }

    #[test]
    fn test_run_qp_presolve_phase2_no_crash() {
        let prob = make_qp_simple(3, 2);
        let opts = SolverOptions::default();
        let phase1 = crate::presolve::run_qp_presolve_phase1(&prob, &opts);
        let phase2 = run_qp_presolve_phase2(phase1, &opts);
        assert_eq!(phase2.orig_num_vars, 3, "orig_num_vars preserved through phase2");
        assert_eq!(phase2.orig_num_constraints, 2, "orig_num_constraints preserved through phase2");
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
            &[0,0, 1,1, 2,2, 3,3, 4,4, 5,5],
            &[0,1, 0,1, 0,1, 0,1, 0,1, 0,1],
            &[1.0,1.0, -1.0,-1.0, 1.0,1.0, -1.0,-1.0, 1.0,-1.0, -1.0,1.0],
            m, n,
        ).unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0];
        let q = CscMatrix::from_triplets(&[0,1], &[0,1], &[2.0,2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(q, vec![0.0;n], a, b, vec![(f64::NEG_INFINITY,f64::INFINITY);n]).unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed);
        // 少なくとも1行が除去されているべき（重複行）
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert!(removed_count >= 2, "at least one redundant pair removed, got {}", removed_count);
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
            m, n,
        ).unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0]; // rows 0,1 and rows 2,3 are the same Le-Le pair
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q, vec![0.0; n], a, b, vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        ).unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed);
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
            m, n,
        ).unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0, 5.0];
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(
            q, vec![0.0; n], a, b, vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        ).unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed);
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert!(
            removed_count >= 2,
            "m > n*ROW_OVERDETERMINED_RATIO: QR runs and removes redundant rows (got {})",
            removed_count
        );
    }
}
