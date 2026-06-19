//! Forrest-Tomlin basis update module.
//!
//! Replaces the PFI (Product Form of Inverse, eta file) approach with
//! Forrest-Tomlin column replacement. After the initial LU factorization
//! (via faer), L and U factors are extracted into our own sparse format.
//! Basis updates modify U by column replacement + column shift. The
//! resulting upper Hessenberg subdiagonal entries are eliminated by row
//! operations whose multipliers are stored as compact FT etas (2 entries
//! each), applied between L-solve and U-solve during FTRAN/BTRAN.
//!
//! Key advantage over PFI: FT etas have exactly 2 nonzeros each (rows k
//! and k+1), vs O(nnz_pivot_col) for PFI etas. Total per-solve cost:
//! O(nnz_L + nnz_U + 2 * total_ft_etas) instead of O(nnz_LU + k * avg_eta_nnz).

use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::ZERO_TOL;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::simplicial::{
    factorize_simplicial_numeric_lu, factorize_simplicial_numeric_lu_scratch, SimplicialLu,
};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams};
use faer::sparse::linalg::{LuError, SupernodalThreshold};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use std::time::Instant;

/// A single Forrest-Tomlin row operation: Row(k+1) -= mult * Row(k),
/// applied between L-solve and U-solve.
#[derive(Debug, Clone)]
struct FtEta {
    /// Row index k (the "pivot" row).
    k: usize,
    /// Multiplier: row k+1 -= mult * row k.
    mult: f64,
}

/// Extracted LU factors with Forrest-Tomlin update support.
#[derive(Debug, Clone)]
pub(crate) struct FtFactors {
    pub(crate) n: usize,

    // ---- Permutations from LU factorization ----
    row_perm_fwd: Vec<usize>,
    row_perm_inv: Vec<usize>,

    // ---- L factor (CSC, unit diagonal implicit, read-only after refactor) ----
    l_col_ptr: Vec<usize>,
    l_row_ind: Vec<usize>,
    l_values: Vec<f64>,

    // ---- U factor (column pool, modified by FT updates) ----
    /// u_cols[j] = sorted (row_index, value) pairs for column j of U.
    /// After FT column shifts, U is upper Hessenberg (subdiagonals eliminated
    /// by FT etas stored separately). The stored U is always upper triangular.
    u_cols: Vec<Vec<(usize, f64)>>,

    // ---- Combined column permutation (initial col_perm + FT swaps) ----
    q_fwd: Vec<usize>,
    q_inv: Vec<usize>,

    // ---- FT etas: row operations applied between L and U solves ----
    ft_etas: Vec<FtEta>,

    update_count: usize,
}

impl FtFactors {
    /// Factorize using faer and extract L, U, permutations into our own format.
    pub(crate) fn factorize(
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) -> Result<Self, SolverError> {
        let m = basis.len();
        if m == 0 {
            return Err(SolverError::EmptyInput { context: "basis" });
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        let (col_ptr, row_ind, values) = build_basis_csc(a, basis, m)?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err(SolverError::SingularBasis { step: 0 });
        }

        let a_sym = unsafe {
            SymbolicSparseColMatRef::<usize>::new_unchecked(m, m, &col_ptr, None, &row_ind)
        };

        let symbolic_params = LuSymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        };
        let symbolic_lu = factorize_symbolic_lu(a_sym, symbolic_params)
            .map_err(|_| SolverError::SingularBasis { step: 0 })?;

        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        let col_perm = symbolic_lu.col_perm();
        let (cp_fwd, cp_inv) = col_perm.arrays();
        let col_perm_fwd: Vec<usize> = cp_fwd.to_vec();
        let col_perm_inv: Vec<usize> = cp_inv.to_vec();

        let a_num = SparseColMatRef::<'_, usize, f64>::new(a_sym, &values);
        let mut sim_lu = SimplicialLu::<usize, f64>::new();
        let mut rp_fwd = vec![0usize; m];
        let mut rp_inv = vec![0usize; m];

        let req = factorize_simplicial_numeric_lu_scratch::<usize, f64>(m, m);
        let mut mem = MemBuffer::new(req);
        let stack = MemStack::new(&mut mem);

        factorize_simplicial_numeric_lu(
            &mut rp_fwd, &mut rp_inv, &mut sim_lu, a_num, col_perm, stack,
        )
        .map_err(|e| match e {
            LuError::SymbolicSingular { index } => SolverError::SingularBasis { step: index },
            _ => SolverError::SingularBasis { step: 0 },
        })?;

        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        let l_ref = sim_lu.l_factor_unsorted();
        let u_ref = sim_lu.u_factor_unsorted();

        let mut l_col_ptr = Vec::with_capacity(m + 1);
        let mut l_row_ind = Vec::new();
        let mut l_values = Vec::new();

        for j in 0..m {
            l_col_ptr.push(l_row_ind.len());
            let rows = l_ref.row_idx_of_col_raw(j);
            let vals = l_ref.val_of_col(j);
            let mut entries: Vec<(usize, f64)> = rows
                .iter().zip(vals.iter()).map(|(&r, &v)| (r, v)).collect();
            entries.sort_by_key(|&(r, _)| r);
            for (r, v) in entries {
                l_row_ind.push(r);
                l_values.push(v);
            }
        }
        l_col_ptr.push(l_row_ind.len());

        let mut u_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);
        for j in 0..m {
            let rows = u_ref.row_idx_of_col_raw(j);
            let vals = u_ref.val_of_col(j);
            let mut col_entries: Vec<(usize, f64)> = rows
                .iter().zip(vals.iter()).map(|(&r, &v)| (r, v)).collect();
            col_entries.sort_by_key(|&(r, _)| r);
            u_cols.push(col_entries);
        }

        for j in 0..m {
            let diag = u_col_diag(&u_cols[j], j);
            if !diag.is_finite() || diag.abs() < ZERO_TOL {
                return Err(SolverError::SingularBasis { step: j });
            }
        }

        Ok(FtFactors {
            n: m,
            row_perm_fwd: rp_fwd,
            row_perm_inv: rp_inv,
            l_col_ptr, l_row_ind, l_values,
            u_cols,
            q_fwd: col_perm_fwd,
            q_inv: col_perm_inv,
            ft_etas: Vec::new(),
            update_count: 0,
        })
    }

    // ---- FTRAN: solve B * x = rhs ----

    /// FTRAN dense: P_r → L-solve → FT etas (forward) → U-solve → Q^{-1}.
    pub(crate) fn ftran_dense(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut work = vec![0.0; n];
        for i in 0..n { work[i] = rhs[self.row_perm_fwd[i]]; }
        self.l_solve_fwd(&mut work);
        self.apply_ft_etas_fwd(&mut work);
        self.u_solve_back(&mut work);
        for i in 0..n { rhs[i] = work[self.q_inv[i]]; }
    }

    pub(crate) fn ftran(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.ftran_dense(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    // ---- BTRAN: solve B^T * x = rhs ----

    /// BTRAN dense: Q → U^T-solve → FT etas (backward) → L^T-solve → P_r^{-1}.
    pub(crate) fn btran_dense(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut work = vec![0.0; n];
        for i in 0..n { work[i] = rhs[self.q_fwd[i]]; }
        self.ut_solve_fwd(&mut work);
        self.apply_ft_etas_bwd(&mut work);
        self.lt_solve_back(&mut work);
        for i in 0..n { rhs[i] = work[self.row_perm_inv[i]]; }
    }

    pub(crate) fn btran(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.btran_dense(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    // ---- Triangular solves ----

    fn l_solve_fwd(&self, rhs: &mut [f64]) {
        let n = self.n;
        for j in 0..n {
            let xj = rhs[j];
            if xj == 0.0 { continue; }
            let start = self.l_col_ptr[j];
            let end = self.l_col_ptr[j + 1];
            for k in start..end {
                let i = self.l_row_ind[k];
                if i > j { rhs[i] -= self.l_values[k] * xj; }
            }
        }
    }

    fn lt_solve_back(&self, rhs: &mut [f64]) {
        let n = self.n;
        for j in (0..n).rev() {
            let start = self.l_col_ptr[j];
            let end = self.l_col_ptr[j + 1];
            for k in start..end {
                let i = self.l_row_ind[k];
                if i > j { rhs[j] -= self.l_values[k] * rhs[i]; }
            }
        }
    }

    fn u_solve_back(&self, rhs: &mut [f64]) {
        let n = self.n;
        for j in (0..n).rev() {
            let col = &self.u_cols[j];
            let diag = u_col_diag(col, j);
            rhs[j] /= diag;
            let xj = rhs[j];
            if xj == 0.0 { continue; }
            for &(i, v) in col {
                if i < j { rhs[i] -= v * xj; }
            }
        }
    }

    fn ut_solve_fwd(&self, rhs: &mut [f64]) {
        let n = self.n;
        for j in 0..n {
            let col = &self.u_cols[j];
            let mut diag = 0.0;
            for &(i, v) in col {
                if i < j { rhs[j] -= v * rhs[i]; }
                else if i == j { diag = v; }
            }
            rhs[j] /= diag;
        }
    }

    // ---- FT eta application ----

    /// Apply FT etas forward (FTRAN direction): row(k+1) -= mult * row(k).
    fn apply_ft_etas_fwd(&self, rhs: &mut [f64]) {
        for eta in &self.ft_etas {
            rhs[eta.k + 1] -= eta.mult * rhs[eta.k];
        }
    }

    /// Apply FT etas backward (BTRAN direction): transposed, reverse order.
    /// The transpose of "row(k+1) -= mult * row(k)" is "row(k) -= mult * row(k+1)".
    fn apply_ft_etas_bwd(&self, rhs: &mut [f64]) {
        for eta in self.ft_etas.iter().rev() {
            rhs[eta.k] -= eta.mult * rhs[eta.k + 1];
        }
    }

    // ---- Forrest-Tomlin Update ----

    /// Perform Forrest-Tomlin basis update.
    ///
    /// The spike h = L^{-1} * P_r * a_entering is recovered from d via
    /// h = FT_etas_inv * U * Q * d. After column replacement and shift,
    /// Hessenberg row operations generate new FT etas that restore
    /// virtual upper triangularity.
    pub(crate) fn ft_update(&mut self, leaving_row: usize, pivot_col_dense: &[f64]) {
        let n = self.n;

        // Step 1: Recover spike from pivot_col.
        // FTRAN computes: d = Q^{-1} * U^{-1} * FT * L^{-1} * P_r * a
        // where FT = product of E_k etas (already applied in FTRAN).
        // So: U * Q * d = FT * L^{-1} * P_r * a = spike
        // No additional FT eta application needed (they're in d already).
        let mut permuted_d = vec![0.0; n];
        for j in 0..n {
            permuted_d[j] = pivot_col_dense[self.q_fwd[j]];
        }
        let mut spike = vec![0.0; n];
        for j in 0..n {
            let dj = permuted_d[j];
            if dj == 0.0 { continue; }
            for &(i, v) in &self.u_cols[j] {
                spike[i] += v * dj;
            }
        }

        // Step 2: Find position p of leaving column in U-space
        let p = self.q_inv[leaving_row];

        // Step 3: Replace column p of U with spike
        let mut spike_entries: Vec<(usize, f64)> = Vec::new();
        for (i, &v) in spike.iter().enumerate() {
            if v.abs() > ZERO_TOL { spike_entries.push((i, v)); }
        }
        self.u_cols[p] = spike_entries;

        // Step 4: Move column p to position n-1 (shift)
        if p < n - 1 {
            let spike_col = self.u_cols.remove(p);
            self.u_cols.push(spike_col);

            let old_basis_at_p = self.q_fwd[p];
            for j in p..n - 1 { self.q_fwd[j] = self.q_fwd[j + 1]; }
            self.q_fwd[n - 1] = old_basis_at_p;
            for j in 0..n { self.q_inv[self.q_fwd[j]] = j; }
        }

        // Step 5: Hessenberg elimination with row-swap fallback.
        // After column shift, U has subdiagonal entries at (k+1, k) for
        // k in p..n-2. We eliminate them and store FT etas.
        // When the diagonal U[k,k] is zero but U[k+1,k] is non-zero,
        // swap rows k and k+1 (partial pivoting) instead of elimination.
        if p < n.saturating_sub(1) {
            for k in p..n - 1 {
                let sub_val = find_entry(&self.u_cols[k], k + 1);
                if sub_val.abs() <= ZERO_TOL { continue; }
                let diag_val = u_col_diag(&self.u_cols[k], k);

                if diag_val.abs() <= ZERO_TOL {
                    // Zero diagonal in Hessenberg elimination: the FT
                    // update cannot proceed without a row permutation
                    // that would complicate the factorization. Signal
                    // that a full refactorization is needed by setting
                    // update_count to max. The caller will detect this
                    // via needs_refactor() and trigger refactorization.
                    self.update_count = usize::MAX / 2;
                    return;
                }

                let mult = sub_val / diag_val;
                self.ft_etas.push(FtEta { k, mult });

                // Eliminate: Row(k+1) -= mult * Row(k) across all columns
                for j in k..n {
                    let u_kj = find_entry(&self.u_cols[j], k);
                    if u_kj.abs() <= ZERO_TOL && j != k { continue; }
                    let delta = mult * u_kj;
                    if delta.abs() <= ZERO_TOL && j != k { continue; }
                    update_entry(&mut self.u_cols[j], k + 1, -delta);
                }
                remove_entry(&mut self.u_cols[k], k + 1);
            }
        }

        self.update_count += 1;
    }

    pub(crate) fn update_count(&self) -> usize {
        self.update_count
    }
}

// ---- Column entry helper functions ----

fn find_entry(col: &[(usize, f64)], row: usize) -> f64 {
    match col.binary_search_by_key(&row, |&(r, _)| r) {
        Ok(pos) => col[pos].1,
        Err(_) => 0.0,
    }
}

fn u_col_diag(col: &[(usize, f64)], col_index: usize) -> f64 {
    find_entry(col, col_index)
}

fn update_entry(col: &mut Vec<(usize, f64)>, row: usize, delta: f64) {
    match col.binary_search_by_key(&row, |&(r, _)| r) {
        Ok(pos) => {
            col[pos].1 += delta;
            if col[pos].1.abs() <= ZERO_TOL { col.remove(pos); }
        }
        Err(pos) => {
            if delta.abs() > ZERO_TOL { col.insert(pos, (row, delta)); }
        }
    }
}

fn remove_entry(col: &mut Vec<(usize, f64)>, row: usize) {
    if let Ok(pos) = col.binary_search_by_key(&row, |&(r, _)| r) {
        col.remove(pos);
    }
}

type BasisCscParts = (Vec<usize>, Vec<usize>, Vec<f64>);
fn build_basis_csc(
    a: &CscMatrix, basis: &[usize], m: usize,
) -> Result<BasisCscParts, SolverError> {
    let mut col_ptr = vec![0usize; m + 1];
    let mut row_ind: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut tmp: Vec<(usize, f64)> = Vec::new();

    for (j, &col_idx) in basis.iter().enumerate() {
        if col_idx >= a.ncols() {
            return Err(SolverError::IndexOutOfBounds {
                context: "basis_column", index: col_idx, bound: a.ncols(),
            });
        }
        let (rows, vals) = a.get_column(col_idx).unwrap();
        tmp.clear();
        for (&row, &val) in rows.iter().zip(vals.iter()) {
            if row < m { tmp.push((row, val)); }
        }
        tmp.sort_by_key(|&(r, _)| r);
        for &(r, v) in &tmp { row_ind.push(r); values.push(v); }
        col_ptr[j + 1] = row_ind.len();
    }
    Ok((col_ptr, row_ind, values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::test_utils::*;

    #[test]
    fn test_ft_factorize_identity() {
        let a = CscMatrix::identity(3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        assert_eq!(ft.n, 3);
        assert_eq!(ft.update_count, 0);
    }

    #[test]
    fn test_ft_ftran_identity() {
        let a = CscMatrix::identity(3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        for i in 0..3 {
            let mut rhs = vec![0.0; 3];
            rhs[i] = 1.0;
            let expected = rhs.clone();
            ft.ftran_dense(&mut rhs);
            assert_vec_near(&rhs, &expected, 1e-10);
        }
    }

    #[test]
    fn test_ft_ftran_3x3() {
        let dense = vec![
            vec![2.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        ft.ftran_dense(&mut rhs);
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_ft_btran_3x3() {
        let dense = vec![
            vec![2.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        ft.btran_dense(&mut rhs);
        let bt = a.transpose();
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_ft_ftran_btran_consistency() {
        let dense = vec![
            vec![3.0, 1.0, 0.0], vec![1.0, 4.0, 2.0], vec![0.0, 2.0, 5.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let b = vec![1.0, 2.0, 3.0];
        let c = vec![4.0, 5.0, 6.0];
        let mut x = b.clone();
        ft.ftran_dense(&mut x);
        let mut y = c.clone();
        ft.btran_dense(&mut y);
        let xtc: f64 = x.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        let bty: f64 = b.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!((xtc - bty).abs() < 1e-10, "adjoint: {} vs {}", xtc, bty);
    }

    #[test]
    fn test_ft_update_single() {
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0], vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis = vec![0, 1, 2];
        let mut ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let mut pivot_col = vec![3.0, 1.0, 2.0];
        ft.ftran_dense(&mut pivot_col);
        ft.ft_update(1, &pivot_col);
        let b_new_dense = vec![
            vec![2.0, 3.0, 0.0], vec![1.0, 1.0, 1.0], vec![0.0, 2.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);
        let rhs_orig = vec![5.0, 2.0, 4.0];
        let mut rhs = rhs_orig.clone();
        ft.ftran_dense(&mut rhs);
        let check = b_new.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_ft_update_btran_after_update() {
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0], vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis = vec![0, 1, 2];
        let mut ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let mut pivot_col = vec![3.0, 1.0, 2.0];
        ft.ftran_dense(&mut pivot_col);
        ft.ft_update(1, &pivot_col);
        let b_new_dense = vec![
            vec![2.0, 3.0, 0.0], vec![1.0, 1.0, 1.0], vec![0.0, 2.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);
        let bt = b_new.transpose();
        let rhs_orig = vec![5.0, 2.0, 4.0];
        let mut rhs = rhs_orig.clone();
        ft.btran_dense(&mut rhs);
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_ft_multiple_updates() {
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0, 1.0, 0.5],
            vec![1.0, 3.0, 1.0, 1.0, 2.0, 0.0],
            vec![0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0, 3.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 4, 6);
        let mut basis = vec![0, 1, 2, 3];
        let mut ft = FtFactors::factorize(&a, &basis, None).unwrap();

        // Update 1: enter col 4, leave row 2
        let mut d = vec![0.0; 4];
        let (cr, cv) = a.get_column(4).unwrap();
        for (&r, &v) in cr.iter().zip(cv.iter()) { d[r] = v; }
        ft.ftran_dense(&mut d);
        ft.ft_update(2, &d);
        basis[2] = 4;

        // Verify FTRAN after update 1
        let b1 = build_basis_from_a(&a, &basis, 4);
        let rhs1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x1 = rhs1.clone();
        ft.ftran_dense(&mut x1);
        let check1 = b1.mat_vec_mul(&x1).unwrap();
        assert_vec_near(&check1, &rhs1, 1e-8);

        // Update 2: enter col 5, leave row 0
        let mut d2 = vec![0.0; 4];
        let (cr2, cv2) = a.get_column(5).unwrap();
        for (&r, &v) in cr2.iter().zip(cv2.iter()) { d2[r] = v; }
        ft.ftran_dense(&mut d2);
        ft.ft_update(0, &d2);
        basis[0] = 5;

        // Verify FTRAN after update 2
        let b2 = build_basis_from_a(&a, &basis, 4);
        let rhs2 = vec![2.0, 3.0, 1.0, 5.0];
        let mut x2 = rhs2.clone();
        ft.ftran_dense(&mut x2);
        let check2 = b2.mat_vec_mul(&x2).unwrap();
        assert_vec_near(&check2, &rhs2, 1e-8);

        // Verify BTRAN after both updates
        let bt2 = b2.transpose();
        let mut y2 = rhs2.clone();
        ft.btran_dense(&mut y2);
        let check_bt = bt2.mat_vec_mul(&y2).unwrap();
        assert_vec_near(&check_bt, &rhs2, 1e-8);
    }

    #[test]
    fn test_ft_sparse_ftran_btran() {
        let dense = vec![
            vec![2.0, 1.0, 0.0], vec![1.0, 3.0, 1.0], vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        ft.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();
        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
        let mut rhs_sv2 = SparseVec::from_dense(&rhs_orig);
        ft.btran(&mut rhs_sv2);
        let y = rhs_sv2.to_dense();
        let bt = a.transpose();
        let check2 = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_ft_singular_detection() {
        let dense = vec![
            vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![1.0, 2.0, 3.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let result = FtFactors::factorize(&a, &basis, None);
        assert!(result.is_err(), "Should detect singular matrix");
    }

    #[test]
    fn test_ft_20x20() {
        let n = 20;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i); cols.push(i);
            vals.push(10.0 + (i as f64) * 0.5);
        }
        let off_diag: Vec<(usize, usize, f64)> = vec![
            (0,3,1.5),(0,7,-0.8),(1,5,2.1),(1,12,-1.3),(2,8,0.9),(2,15,-0.4),
            (3,0,-1.2),(3,11,0.7),(4,9,1.8),(4,16,-0.6),(5,1,-0.5),(5,13,1.1),
            (6,2,0.3),(6,14,-1.9),(7,0,0.8),(7,17,-0.7),(8,2,-1.4),(8,18,0.6),
            (9,4,1.0),(9,19,-0.3),
        ];
        for (r, c, v) in &off_diag {
            rows.push(*r); cols.push(*c); vals.push(*v);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let basis: Vec<usize> = (0..n).collect();
        let ft = FtFactors::factorize(&a, &basis, None).unwrap();
        for k in 0..3 {
            let rhs_orig: Vec<f64> = (0..n).map(|i| ((i+k*7)%11) as f64 - 5.0).collect();
            let mut rhs = rhs_orig.clone();
            ft.ftran_dense(&mut rhs);
            let check = a.mat_vec_mul(&rhs).unwrap();
            assert_vec_near(&check, &rhs_orig, 1e-8);
        }
        let bt = a.transpose();
        let rhs_orig: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 3.0).collect();
        let mut rhs = rhs_orig.clone();
        ft.btran_dense(&mut rhs);
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);
    }

    fn build_basis_from_a(a: &CscMatrix, basis: &[usize], m: usize) -> CscMatrix {
        let b_cols: Vec<Vec<f64>> = basis.iter().map(|&col| {
            let (rows, vals) = a.get_column(col).unwrap();
            let mut c = vec![0.0; m];
            for (&r, &v) in rows.iter().zip(vals.iter()) { if r < m { c[r] = v; } }
            c
        }).collect();
        dense_to_csc(
            &(0..m).map(|i| b_cols.iter().map(|c| c[i]).collect::<Vec<f64>>()).collect::<Vec<_>>(),
            m, m,
        )
    }
}

