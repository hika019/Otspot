//! Basis reconciliation, crash, feasibility check, and solution extraction.

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{feas_rel_tol, PIVOT_TOL};
#[cfg(test)]
use std::sync::atomic::Ordering;
use super::super::dual_common::compute_dual_vars_into;
use super::StandardForm;

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
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * sj;
            }
        }
    }
    let mut violated = false;
    for ((ax_i, ct), bi) in ax
        .iter()
        .zip(problem.constraint_types.iter())
        .zip(problem.b.iter())
    {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
        };
        let scale = 1.0 + bi.abs() + ax_i.abs();
        let rel = violation / scale;
        if rel > tol {
            violated = true;
        }
    }
    !violated
}

pub(super) fn pivot_out_degenerate_artificials(
    a_ext: &CscMatrix,
    basis: &mut [usize],
    x_b: &[f64],
    sf: &StandardForm,
    options: &SolverOptions,
) {
    let m = basis.len();

    // Fast pre-check: skip LU build entirely when Phase I has already pivoted
    // out all artificials. For most problems this is the common case; avoiding
    // two LU factorizations (one here, one for validation) saves significant
    // work — especially for large m.
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

    // Pivot stability uses |(B^{-1} a_j)[i]|, not raw A[i,j], so we need an LU.
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(mgr) => mgr,
        Err(_) => return,
    };

    let mut is_basic = vec![false; a_ext.ncols];
    for &col in basis.iter() {
        is_basic[col] = true;
    }

    // BTRAN-based candidate scan: one BTRAN gives the i-th row of B^{-1}; a
    // sparse dot vs each non-basic column ranks candidates without per-column
    // FTRAN. One FTRAN at the end feeds basis_mgr.update — total cost per
    // artificial ≈ O(m + nnz(A)), vs. O(n_total · FTRAN) for the naive form.
    let mut z_dense = vec![0.0_f64; m];
    for i in 0..m {
        if options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return;
        }
        if basis[i] < sf.n_total || x_b[i].abs() >= PIVOT_TOL {
            continue;
        }

        // z = B^{-T} e_i
        z_dense.iter_mut().for_each(|v| *v = 0.0);
        z_dense[i] = 1.0;
        basis_mgr.btran_dense(&mut z_dense);

        // argmax_j |d[i,j]| over non-basic original columns.
        let mut best_j: Option<usize> = None;
        let mut best_abs = PIVOT_TOL;
        for j in 0..sf.n_total {
            if is_basic[j] {
                continue;
            }
            if let Ok((rows, vals)) = a_ext.get_column(j) {
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
        }

        if let Some(j) = best_j {
            let (col_rows, col_vals) = match a_ext.get_column(j) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let mut d_sv = SparseVec {
                indices: col_rows.to_vec(),
                values: col_vals.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut d_sv);
            is_basic[basis[i]] = false;
            is_basic[j] = true;
            basis[i] = j;
            basis_mgr.update(j, i, &d_sv);
            basis_mgr.refactor_if_needed_timed(a_ext, basis, options.deadline);
        }
    }

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
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            let scale = col_scale.get(basis[i]).copied().unwrap_or(1.0);
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
