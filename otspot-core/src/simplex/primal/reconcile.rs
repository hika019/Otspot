//! Basis reconciliation, crash, feasibility check, and solution extraction.

use super::super::dual_common::compute_dual_vars_into;
use super::StandardForm;
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{feas_rel_tol, PIVOT_STABILITY_THRESHOLD, PIVOT_TOL};
#[cfg(test)]
use std::sync::atomic::Ordering;

/// Maximum partial-revert rounds after crash basis construction.
/// Each round restores artificial columns for rows with negative x_b and
/// re-factorizes; 3 rounds is sufficient for observed mid-scale ill LPs.
const CRASH_REVERT_MAX_ROUNDS: usize = 3;

/// Crash basis: cover artificial rows with structural columns to reduce Phase I pivots.
///
/// Returns `Some((crash_basis, x_b))` on success, `None` if the crash is no better
/// than a cold start or the basis is singular. Returns `None` in the following cases:
/// 1. Crash result equals `cold_basis` (no structural coverage gained)
/// 2. LU factorization fails (crashed basis singular)
/// 3. No structural columns remain after partial revert
///
/// Partial revert: rows where `x_b < -PIVOT_TOL` have their crashed column replaced
/// with the artificial and the basis is re-factorized.
pub(super) fn try_apply_crash(
    a_ext: &CscMatrix,
    m: usize,
    n_shifted: usize,
    n_total: usize,
    b_scaled: &[f64],
    max_etas: usize,
    deadline: Option<std::time::Instant>,
    cold_basis: &[usize],
) -> Option<(Vec<usize>, Vec<f64>)> {
    use super::super::crash;

    // Reconstruct needs_artificial from `cold_basis[i] >= n_total`.
    let needs_artificial: Vec<bool> = cold_basis.iter().map(|&c| c >= n_total).collect();

    let num_art_in = needs_artificial.iter().filter(|&&v| v).count();
    if num_art_in == 0 {
        return None;
    }

    let (mut basis, _, num_art_out) =
        crash::compute_crash_basis(a_ext, b_scaled, m, n_shifted, cold_basis, &needs_artificial);

    if num_art_out == num_art_in {
        return None;
    }

    // partial revert loop: 負 x_b の crashed 行を artif に戻す。
    // 復元不能な行 (= 元 cold basis に artif 候補が無い ub/slack 行) で負成分が
    // 出た場合は crash 全体を放棄 (Phase I/II が x_B >= 0 不変式を回復できないため)。
    let mut x_b = vec![0.0_f64; m];
    let mut crashed_count = num_art_in - num_art_out;
    for round in 0..=CRASH_REVERT_MAX_ROUNDS {
        let mut basis_mgr = match LuBasis::new_timed(a_ext, &basis, max_etas, deadline) {
            Ok(b) => b,
            Err(_) => {
                return None;
            }
        };
        let mut x_b_sv = SparseVec::from_dense(b_scaled);
        basis_mgr.ftran(&mut x_b_sv);
        x_b = x_b_sv.to_dense();

        let mut reverts = 0usize;
        for i in 0..m {
            if x_b[i] >= -PIVOT_TOL {
                continue;
            }
            // 負成分行: 元 cold で artif があれば revert、無ければ crash 放棄。
            if cold_basis[i] >= n_total {
                basis[i] = cold_basis[i];
                reverts += 1;
            } else {
                return None;
            }
        }
        if reverts == 0 {
            break;
        }
        crashed_count = crashed_count.saturating_sub(reverts);
        if crashed_count == 0 {
            return None;
        }
        if round == CRASH_REVERT_MAX_ROUNDS && reverts > 0 {
            return None;
        }
    }

    Some((basis, x_b))
}

/// Defense-in-depth feasibility check.  Per constraint, compare violation to
/// `feas_rel_tol() * (1 + |b_i| + |Ax_i|)` so the gate is scale-invariant.
/// `feas_rel_tol() = sqrt(PIVOT_TOL)` follows from Wilkinson's heuristic
/// (see `tolerances.rs`).
pub(super) fn check_eq_feasibility(problem: &LpProblem, solution: &[f64]) -> bool {
    let tol = feas_rel_tol();
    let mut ax = vec![0.0f64; problem.num_constraints];
    for (j, &sj) in solution.iter().enumerate() {
        let (rows, vals) = problem.a.column(j);
        for (k, &row) in rows.iter().enumerate() {
            ax[row] += vals[k] * sj;
        }
    }
    let mut violated = false;
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut max_row = 0usize;
    let mut max_ct = ConstraintType::Eq;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for (i, ((ax_i, ct), bi)) in ax
        .iter()
        .zip(problem.constraint_types.iter())
        .zip(problem.b.iter())
        .enumerate()
    {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
        };
        let scale = 1.0 + bi.abs() + ax_i.abs();
        let rel = violation / scale;
        if rel > max_rel {
            max_rel = rel;
            max_abs = violation;
            max_row = i;
            max_ct = *ct;
            max_ax = *ax_i;
            max_b = *bi;
        }
        if rel > tol {
            violated = true;
        }
    }
    super::trace_stage(format_args!(
        "feasibility check max_rel={max_rel:.9e} max_abs={max_abs:.9e} row={max_row} ct={max_ct:?} ax={max_ax:.9e} b={max_b:.9e} tol={tol:.9e}"
    ));
    !violated
}

/// Per-row BTRAN pivot for `target_rows` using an existing `basis_mgr`.
///
/// For each row still holding an artificial, issues one BTRAN to find the
/// best non-basic structural column and performs an eta update. Used as the
/// fallback after a failed batch LU, and for any unmatched rows after the
/// batch succeeds.
fn pivot_out_sequential(
    a_ext: &CscMatrix,
    basis: &mut [usize],
    sf: &StandardForm,
    options: &SolverOptions,
    target_rows: &[usize],
    basis_mgr: &mut LuBasis,
    is_basic: &mut [bool],
) {
    let m = basis.len();
    let mut z_dense = vec![0.0_f64; m];
    for &i in target_rows {
        if options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return;
        }
        if basis[i] < sf.n_total {
            continue;
        }
        z_dense.iter_mut().for_each(|v| *v = 0.0);
        z_dense[i] = 1.0;
        #[cfg(test)]
        super::PIVOT_OUT_BTRAN_COUNT.with(|c| c.set(c.get() + 1));
        basis_mgr.btran_dense(&mut z_dense);

        let mut best_j: Option<usize> = None;
        let mut best_abs = PIVOT_TOL;
        for j in 0..sf.n_total {
            if is_basic[j] {
                continue;
            }
            let (rows, vals) = a_ext.column(j);
            let mut d_ij = 0.0_f64;
            for (k, &row) in rows.iter().enumerate() {
                if row < m {
                    d_ij += z_dense[row] * vals[k];
                }
            }
            let abs_d = d_ij.abs();
            if abs_d > best_abs {
                best_abs = abs_d;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            let (col_rows, col_vals) = a_ext.column(j);
            let mut d_sv = SparseVec {
                indices: col_rows.to_vec(),
                values: col_vals.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut d_sv);
            match basis_mgr.update(j, i, &d_sv) {
                Ok(()) => {}
                Err(crate::error::SolverError::SingularBasis { .. }) => return,
                Err(err) => panic!("internal reconciliation eta invariant violated: {err}"),
            }
            is_basic[basis[i]] = false;
            is_basic[j] = true;
            basis[i] = j;
            basis_mgr.refactor_if_needed_timed(a_ext, basis, options.deadline);
        }
    }
}

pub(crate) fn pivot_out_degenerate_artificials(
    a_ext: &CscMatrix,
    basis: &mut [usize],
    x_b: &[f64],
    sf: &StandardForm,
    options: &SolverOptions,
) {
    let m = basis.len();

    // Fast pre-check: skip entirely when Phase I already removed all artificials.
    if !basis
        .iter()
        .zip(x_b.iter())
        .any(|(&col, &val)| col >= sf.n_total && val.abs() < PIVOT_TOL)
    {
        #[cfg(test)]
        super::PIVOT_CLEAN_EARLY_EXIT_COUNT.fetch_add(1, Ordering::Relaxed);
        return;
    }

    #[cfg(test)]
    super::PIVOT_CLEAN_CLEANUP_RAN_COUNT.fetch_add(1, Ordering::Relaxed);

    let basis_before = basis.to_vec();

    // Collect rows that have a degenerate artificial (basis[i] >= n_total, x_b[i] ≈ 0).
    let degen_rows: Vec<usize> = (0..m)
        .filter(|&i| basis[i] >= sf.n_total && x_b[i].abs() < PIVOT_TOL)
        .collect();

    // Build is_basic mask for the current basis.
    let mut is_basic = vec![false; a_ext.ncols];
    for &col in basis.iter() {
        is_basic[col] = true;
    }

    // Build per-row candidate lists in one O(nnz) pass through structural columns.
    // Each entry is (|A[r,j]|, j) for non-basic j < sf.n_total with |A[r,j]| >= PIVOT_TOL.
    let mut row_candidates: Vec<Vec<(f64, usize)>> = vec![Vec::new(); m];
    for j in 0..sf.n_total {
        if is_basic[j] {
            continue;
        }
        let (rows, vals) = a_ext.column(j);
        for (k, &row) in rows.iter().enumerate() {
            if row < m {
                let abs_v = vals[k].abs();
                if abs_v >= PIVOT_TOL {
                    row_candidates[row].push((abs_v, j));
                }
            }
        }
    }
    // Sort each row's candidates descending by |A[r,j]| for numerical stability.
    for cands in row_candidates.iter_mut() {
        cands.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Greedy assignment: each degenerate row claims the largest-|A| unclaimed column.
    // `claimed` tracks columns already in the basis or already assigned this round.
    let mut claimed = is_basic.clone();
    let mut matches: Vec<(usize, usize)> = Vec::with_capacity(degen_rows.len());
    let mut unmatched_rows: Vec<usize> = Vec::new();
    for &r in &degen_rows {
        match row_candidates[r].iter().find(|(_, j)| !claimed[*j]) {
            Some(&(_, j)) => {
                claimed[j] = true;
                matches.push((r, j));
            }
            None => unmatched_rows.push(r),
        }
    }

    if matches.is_empty() {
        // Nothing the batch can do: run sequential for all degenerate rows.
        if let Ok(mut basis_mgr) =
            LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline)
        {
            pivot_out_sequential(
                a_ext,
                basis,
                sf,
                options,
                &degen_rows,
                &mut basis_mgr,
                &mut is_basic,
            );
        }
    } else {
        // Batch: apply all greedy matches, then one full LU instead of num_art incremental updates.
        //
        // Build B_before now for two purposes:
        //  1. Post-batch FTRAN stability check (non-singular ≠ well-conditioned).
        //  2. Sequential fallback if the stability check fails.
        let mut b_before_opt: Option<LuBasis> =
            LuBasis::new_timed(a_ext, &basis_before, options.max_etas, options.deadline).ok();

        // Multi-level batch: each iteration commits as many matches as possible via a
        // single LU, then recurses on the remainder.  For rank-saturated LPs the second
        // iteration commits 0 rows and the loop terminates, skipping O(N) wasted BTRANs.
        //
        // For each iteration:
        //  1. Try the full remaining slice as one batch.
        //  2. If singular: binary-search for the maximum non-singular prefix (O(log N) LU calls).
        //  3. Apply the committed prefix to `basis` and advance the offset.
        //  4. If committed == 0: rank saturated; stop (remaining rows have no valid pivot).
        let mut match_offset = 0usize;
        'batch: loop {
            let slice = &matches[match_offset..];
            if slice.is_empty() {
                break;
            }

            // Build trial basis for the full remaining slice.
            let mut trial_basis = basis.to_vec();
            for &(r, j) in slice {
                trial_basis[r] = j;
            }

            #[cfg(test)]
            super::PIVOT_OUT_BATCH_LU_COUNT.with(|c| c.set(c.get() + 1));

            let committed =
                if LuBasis::new_timed(a_ext, &trial_basis, options.max_etas, options.deadline)
                    .is_ok()
                {
                    // Full slice is non-singular: commit all.
                    slice.len()
                } else {
                    // Binary-search for maximum non-singular prefix in [0, slice.len()-1].
                    // Invariant: lo = largest confirmed-valid prefix length (0 = always valid).
                    let mut lo = 0usize;
                    let mut hi = slice.len().saturating_sub(1);
                    while lo < hi {
                        let mid = lo + (hi - lo).div_ceil(2); // ceiling midpoint ensures progress
                        let mut t = basis.to_vec();
                        for &(r, j) in &slice[..mid] {
                            t[r] = j;
                        }
                        if LuBasis::new_timed(a_ext, &t, options.max_etas, options.deadline).is_ok()
                        {
                            lo = mid;
                        } else {
                            hi = mid - 1;
                        }
                    }

                    lo
                };

            if committed == 0 {
                // No progress in this iteration: all remaining candidate columns are
                // rank-saturated in the current basis span. Sequential BTRAN (in B⁻¹aⱼ
                // space) is attempted below for these rows; it too will typically find
                // no valid pivot in a fully rank-saturated subspace, but the attempt is
                // harmless and ensures correctness in partially-degenerate edge cases.
                break 'batch;
            }

            // Apply the committed prefix.
            for &(r, j) in &slice[..committed] {
                basis[r] = j;
            }
            match_offset += committed;
        }

        // Stability verification: sample up to STABILITY_CHECK_LIMIT committed pivots,
        // FTRAN a_j through B_before, reject the batch if |d[r]|/max|d| <
        // PIVOT_STABILITY_THRESHOLD. The greedy picks columns by raw |A[r,j]|, which can
        // differ from the FTRAN ordering |B⁻¹a_j|[r]: on highly-degenerate LPs a raw-A
        // column may have a near-zero FTRAN entry at row r (nearly in the basis span),
        // yielding a non-singular but ill-conditioned basis that blows up later duals.
        // Sampling n_samples = min(match_offset, LIMIT) bounds cost; the index map
        // idx = k·(match_offset-1)/(n_samples-1) covers [0,match_offset-1] inclusive —
        // a naive k·match_offset/n_samples drops the last pivot once match_offset > LIMIT.
        // b_before_opt is None only on a singular Phase-I basis (never expected); then
        // accept the batch and let the final guard decide.
        const STABILITY_CHECK_LIMIT: usize = 64;
        let batch_stable = match_offset == 0 || b_before_opt.is_none() || {
            let b_lu = b_before_opt.as_mut().unwrap();
            let mut col_dense = vec![0.0_f64; m];
            let mut stable = true;
            let n_samples = match_offset.min(STABILITY_CHECK_LIMIT);
            for k in 0..n_samples {
                let idx = if n_samples == 1 {
                    0
                } else {
                    k * (match_offset - 1) / (n_samples - 1)
                };
                let (r, j) = matches[idx];
                col_dense.iter_mut().for_each(|v| *v = 0.0);
                let (rows, vals) = a_ext.column(j);
                for (p, &row) in rows.iter().enumerate() {
                    if row < m {
                        col_dense[row] = vals[p];
                    }
                }
                b_lu.ftran_dense(&mut col_dense);
                let max_abs = col_dense.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_abs <= PIVOT_TOL || col_dense[r].abs() < PIVOT_STABILITY_THRESHOLD * max_abs
                {
                    stable = false;
                    break;
                }
            }
            stable
        };

        if !batch_stable {
            // Ill-conditioned batch: revert and run sequential for all degenerate rows
            // using B_before (already factorized above, reused here).
            basis.copy_from_slice(&basis_before);
            #[cfg(test)]
            super::PIVOT_OUT_SEQUENTIAL_FALLBACK_COUNT.with(|c| c.set(c.get() + 1));
            if let Some(mut b_lu) = b_before_opt {
                let mut seq_is_basic = vec![false; a_ext.ncols];
                for &col in basis.iter() {
                    seq_is_basic[col] = true;
                }
                pivot_out_sequential(
                    a_ext,
                    basis,
                    sf,
                    options,
                    &degen_rows,
                    &mut b_lu,
                    &mut seq_is_basic,
                );
            }
        } else {
            // Batch stable. Run sequential BTRAN for residual rows that the batch could not
            // commit, then for greedy-unmatched rows.
            //
            // `uncommitted_rows`: rows where greedy found a raw-A candidate but the multi-level
            // batch left them uncommitted. match_offset > 0: some rows committed, remainder
            // rank-saturated in raw-A space. match_offset == 0: all greedy matches failed the
            // batch LU (e.g. a single candidate column already coplanar with the basis); the
            // batch_stable short-circuit applies and sequential BTRAN is attempted for all.
            // `unmatched_rows`: greedy found no candidate; sequential BTRAN is the only option.
            let uncommitted_rows: Vec<usize> = if match_offset > 0 {
                // Some rows were committed by the batch; the remainder are rank-saturated
                // in raw-A space. Sequential BTRAN uses B⁻¹aⱼ strength and may succeed.
                #[cfg(test)]
                super::PIVOT_OUT_UNCOMMITTED_SEQUENTIAL_COUNT
                    .with(|c| c.set(c.get() + matches[match_offset..].len()));
                matches[match_offset..].iter().map(|&(r, _)| r).collect()
            } else {
                // match_offset == 0: no rows committed by the batch (batch_stable
                // short-circuits). All greedy matches failed as a group (single-element
                // slices fail immediately when the candidate column is coplanar with the
                // existing basis). Sequential BTRAN attempts each row individually;
                // it will find no pivot in a fully rank-saturated subspace, but the
                // attempt is correct for partially-degenerate edge cases.
                #[cfg(test)]
                super::PIVOT_OUT_UNCOMMITTED_SEQUENTIAL_COUNT
                    .with(|c| c.set(c.get() + matches.len()));
                matches.iter().map(|&(r, _)| r).collect()
            };
            let has_residual = !unmatched_rows.is_empty() || !uncommitted_rows.is_empty();
            if has_residual {
                if let Ok(mut basis_mgr) =
                    LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline)
                {
                    let mut seq_is_basic = vec![false; a_ext.ncols];
                    for &col in basis.iter() {
                        seq_is_basic[col] = true;
                    }
                    // Combine both residual groups into a single sequential call.
                    let combined_rows: Vec<usize> = uncommitted_rows
                        .iter()
                        .chain(unmatched_rows.iter())
                        .copied()
                        .collect();
                    if !combined_rows.is_empty() {
                        pivot_out_sequential(
                            a_ext,
                            basis,
                            sf,
                            options,
                            &combined_rows,
                            &mut basis_mgr,
                            &mut seq_is_basic,
                        );
                    }
                }
            }
        }
    }

    // Final guard: revert if the resulting basis cannot be factorized.
    if LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline).is_err() {
        basis.copy_from_slice(&basis_before);
    }
}

/// Recompute x_B = B^{-1} b and y = B^{-T} c_B from a fresh LU.
pub(crate) fn reconcile_final_basis_state(
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    basis: &[usize],
    x_b: &mut [f64],
    y: &mut [f64],
    max_etas: usize,
    deadline: Option<std::time::Instant>,
) -> Result<(), crate::error::SolverError> {
    let mut basis_mgr = LuBasis::new_timed(a, basis, max_etas, deadline)?;

    x_b.copy_from_slice(b);
    basis_mgr.ftran_dense(x_b);
    for value in x_b.iter_mut() {
        if value.abs() < 1e-12 {
            *value = 0.0;
        }
    }

    compute_dual_vars_into(c, &mut basis_mgr, basis, y);
    Ok(())
}

/// Map the standard-form basic solution back to original variables, inverting
/// shifts/sign-flips/splits.  `col_scale` is the Ruiz column scale (or empty).
///
/// The recomposition `offset + Σ coeff * x_new[idx]` is accumulated in
/// double-double (TwoFloat) precision because free variables are split as
/// `x = x+ − x-`; when the simplex leaves both components large (e.g. on the
/// order of 1e16) f64 subtraction loses the unit-scale residual entirely.
/// The contract is locked by `tests::test_extract_solution_uses_dd_for_split_variable_cancellation`.
pub(crate) fn extract_solution(
    sf: &StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> Vec<f64> {
    use twofloat::TwoFloat;
    assert!(
        col_scale.is_empty() || col_scale.len() == sf.n_total,
        "col_scale must be empty (identity) or match the standard-form column count"
    );
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            let scale = if col_scale.is_empty() {
                1.0
            } else {
                col_scale[basis[i]]
            };
            x_new[basis[i]] = x_b[i] * scale;
        }
    }

    let mut solution = vec![0.0; sf.n_orig];
    for (j, sol_j) in solution.iter_mut().enumerate() {
        let info = &sf.orig_var_info[j];
        let mut value = TwoFloat::from(info.offset);
        for &(new_idx, coeff) in &info.new_vars {
            value += TwoFloat::new_mul(coeff, x_new[new_idx]);
        }
        *sol_j = f64::from(value);
    }
    solution
}
