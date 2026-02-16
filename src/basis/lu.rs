//! Sparse LU factorization with Markowitz threshold pivoting

use std::collections::{HashMap, HashSet};

use crate::sparse::{CscMatrix, SparseLowerCSC, SparseUpperCSR};

const MARKOWITZ_THRESHOLD: f64 = 0.1;
const SINGULAR_TOL: f64 = 1e-12;

/// LU factorization result: P_row * B * P_col^T = L * U
/// L is unit lower triangular (CSC), U is upper triangular (CSR).
#[derive(Debug, Clone)]
pub(crate) struct LuFactorization {
    pub l: SparseLowerCSC,
    pub u: SparseUpperCSR,
    /// Row permutation: permuted position i corresponds to original row p_row[i]
    pub p_row: Vec<usize>,
    /// Column permutation: permuted position j corresponds to original column p_col[j]
    pub p_col: Vec<usize>,
    /// Dimension
    pub n: usize,
}

/// Sparse working matrix for Gaussian elimination.
/// Stores entries in row-major HashMaps with column-to-row index sets.
struct WorkingMatrix {
    row_data: Vec<HashMap<usize, f64>>,
    col_rows: Vec<HashSet<usize>>,
}

impl WorkingMatrix {
    fn new(n: usize) -> Self {
        Self {
            row_data: (0..n).map(|_| HashMap::new()).collect(),
            col_rows: (0..n).map(|_| HashSet::new()).collect(),
        }
    }

    fn insert(&mut self, row: usize, col: usize, val: f64) {
        if val.abs() > SINGULAR_TOL {
            self.row_data[row].insert(col, val);
            self.col_rows[col].insert(row);
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        *self.row_data[row].get(&col).unwrap_or(&0.0)
    }

    fn remove(&mut self, row: usize, col: usize) {
        self.row_data[row].remove(&col);
        self.col_rows[col].remove(&row);
    }
}

impl LuFactorization {
    /// Factorize the basis matrix B (columns of A selected by `basis` indices).
    /// Returns P_row * B * P_col^T = L * U
    pub fn factorize(a: &CscMatrix, basis: &[usize]) -> Result<Self, String> {
        let m = basis.len();
        if m == 0 {
            return Err("Empty basis".to_string());
        }

        // Extract basis columns into WorkingMatrix
        let mut work = WorkingMatrix::new(m);
        for (j, &col_idx) in basis.iter().enumerate() {
            if col_idx >= a.ncols {
                return Err(format!(
                    "Basis column {} out of bounds (ncols={})",
                    col_idx, a.ncols
                ));
            }
            let start = a.col_ptr[col_idx];
            let end = a.col_ptr[col_idx + 1];
            for k in start..end {
                let row = a.row_ind[k];
                if row < m {
                    work.insert(row, j, a.values[k]);
                }
            }
        }

        // Initial column nnz for tie-breaking in Markowitz search
        let initial_col_nnz: Vec<usize> = (0..m).map(|j| work.col_rows[j].len()).collect();

        let mut p_row = vec![0usize; m];
        let mut p_col = vec![0usize; m];
        let mut eliminated_rows = vec![false; m];
        let mut eliminated_cols = vec![false; m];

        // Storage for L and U entries (in original indices)
        let mut l_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        let mut u_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        let mut diag = vec![0.0f64; m];

        for step in 0..m {
            // Find column maximum for threshold test (active entries only)
            let mut col_max: Vec<f64> = vec![0.0; m];
            for j in 0..m {
                if eliminated_cols[j] {
                    continue;
                }
                for &r in &work.col_rows[j] {
                    if eliminated_rows[r] {
                        continue;
                    }
                    let abs_val = work.get(r, j).abs();
                    if abs_val > col_max[j] {
                        col_max[j] = abs_val;
                    }
                }
            }

            // Find pivot with minimum Markowitz count satisfying threshold.
            // Tie-break by initial column nnz (ascending).
            let mut best_pivot: Option<(usize, usize)> = None;
            let mut best_markowitz = usize::MAX;
            let mut best_col_order = usize::MAX;

            for j in 0..m {
                if eliminated_cols[j] {
                    continue;
                }
                if col_max[j] <= SINGULAR_TOL {
                    continue;
                }
                let c_nnz = work.col_rows[j]
                    .iter()
                    .filter(|&&r| !eliminated_rows[r])
                    .count();

                for &r in &work.col_rows[j] {
                    if eliminated_rows[r] {
                        continue;
                    }
                    let abs_val = work.get(r, j).abs();
                    if abs_val <= SINGULAR_TOL {
                        continue;
                    }
                    if abs_val < MARKOWITZ_THRESHOLD * col_max[j] {
                        continue;
                    }
                    let r_nnz = work.row_data[r]
                        .keys()
                        .filter(|&&c| !eliminated_cols[c])
                        .count();
                    let markowitz =
                        r_nnz.saturating_sub(1) * c_nnz.saturating_sub(1);

                    if markowitz < best_markowitz
                        || (markowitz == best_markowitz
                            && initial_col_nnz[j] < best_col_order)
                    {
                        best_markowitz = markowitz;
                        best_col_order = initial_col_nnz[j];
                        best_pivot = Some((r, j));
                    }
                }
            }

            let (pivot_row, pivot_col) = match best_pivot {
                Some(p) => p,
                None => return Err(format!("Singular matrix detected at step {}", step)),
            };

            p_row[step] = pivot_row;
            p_col[step] = pivot_col;
            let pivot_val = work.get(pivot_row, pivot_col);
            diag[step] = pivot_val;

            // Record U entries from pivot row (off-diagonal, active columns only)
            for (&c, &val) in &work.row_data[pivot_row] {
                if eliminated_cols[c] || c == pivot_col {
                    continue;
                }
                u_entries[step].push((c, val));
            }

            // Collect active rows in pivot column (excluding pivot row)
            let active_rows_in_col: Vec<usize> = work.col_rows[pivot_col]
                .iter()
                .filter(|&&r| !eliminated_rows[r] && r != pivot_row)
                .copied()
                .collect();

            // Collect pivot row entries (off-diagonal, active columns) for elimination
            let pivot_row_entries: Vec<(usize, f64)> = work.row_data[pivot_row]
                .iter()
                .filter(|(&c, _)| !eliminated_cols[c] && c != pivot_col)
                .map(|(&c, &v)| (c, v))
                .collect();

            // Eliminate: for each active row with non-zero in pivot column
            for r_i in active_rows_in_col {
                let val_ic = work.get(r_i, pivot_col);
                if val_ic.abs() <= SINGULAR_TOL {
                    continue;
                }
                let multiplier = val_ic / pivot_val;
                l_entries[step].push((r_i, multiplier));

                // Update row r_i: work[r_i][c] -= multiplier * work[pivot_row][c]
                for &(c, u_val) in &pivot_row_entries {
                    let old_val = work.get(r_i, c);
                    let new_val = old_val - multiplier * u_val;
                    if new_val.abs() <= SINGULAR_TOL {
                        work.remove(r_i, c);
                    } else {
                        work.row_data[r_i].insert(c, new_val);
                        work.col_rows[c].insert(r_i);
                    }
                }

                // Remove entry at pivot column
                work.remove(r_i, pivot_col);
            }

            eliminated_rows[pivot_row] = true;
            eliminated_cols[pivot_col] = true;
        }

        // Build inverse permutations
        let mut inv_perm_row = vec![0usize; m];
        let mut inv_perm_col = vec![0usize; m];
        for k in 0..m {
            inv_perm_row[p_row[k]] = k;
            inv_perm_col[p_col[k]] = k;
        }

        // Build SparseLowerCSC from l_entries (convert original → permuted indices)
        let mut l_col_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for step in 0..m {
            for &(orig_row, multiplier) in &l_entries[step] {
                let perm_row = inv_perm_row[orig_row];
                l_col_data[step].push((perm_row, multiplier));
            }
            l_col_data[step].sort_by_key(|&(r, _)| r);
        }

        let mut l_col_ptr = vec![0usize; m + 1];
        let mut l_row_ind = Vec::new();
        let mut l_values = Vec::new();
        for j in 0..m {
            l_col_ptr[j] = l_row_ind.len();
            for &(r, v) in &l_col_data[j] {
                l_row_ind.push(r);
                l_values.push(v);
            }
        }
        l_col_ptr[m] = l_row_ind.len();

        let l = SparseLowerCSC {
            col_ptr: l_col_ptr,
            row_ind: l_row_ind,
            values: l_values,
            n: m,
        };

        // Build SparseUpperCSR from u_entries (convert original → permuted indices)
        let mut u_row_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for step in 0..m {
            for &(orig_col, value) in &u_entries[step] {
                let perm_col = inv_perm_col[orig_col];
                u_row_data[step].push((perm_col, value));
            }
            u_row_data[step].sort_by_key(|&(c, _)| c);
        }

        let mut u_row_ptr = vec![0usize; m + 1];
        let mut u_col_ind = Vec::new();
        let mut u_values = Vec::new();
        for i in 0..m {
            u_row_ptr[i] = u_col_ind.len();
            for &(c, v) in &u_row_data[i] {
                u_col_ind.push(c);
                u_values.push(v);
            }
        }
        u_row_ptr[m] = u_col_ind.len();

        let u = SparseUpperCSR {
            row_ptr: u_row_ptr,
            col_ind: u_col_ind,
            values: u_values,
            diag,
            n: m,
        };

        Ok(LuFactorization {
            l,
            u,
            p_row,
            p_col,
            n: m,
        })
    }
}

/// FTRAN: solve B * x = rhs using LU factors
/// P_row * B * P_col^T = L * U
/// 1. Apply row permutation to rhs
/// 2. Forward substitution: L * z = P_row * rhs
/// 3. Back substitution: U * y = z
/// 4. Apply inverse column permutation: x = P_col^T * y
pub(crate) fn solve_ftran(lu: &LuFactorization, rhs: &mut Vec<f64>) {
    let n = lu.n;

    // Step 1: Apply row permutation
    let orig = rhs.clone();
    for i in 0..n {
        rhs[i] = orig[lu.p_row[i]];
    }

    // Step 2: Sparse forward substitution with L
    lu.l.forward_solve(rhs);

    // Step 3: Sparse backward substitution with U
    lu.u.backward_solve(rhs);

    // Step 4: Apply inverse column permutation
    let y = rhs.clone();
    for i in 0..n {
        rhs[lu.p_col[i]] = y[i];
    }
}

/// BTRAN: solve B^T * x = rhs using LU factors
/// 1. Apply column permutation to rhs
/// 2. Forward substitution with U^T
/// 3. Back substitution with L^T
/// 4. Apply row permutation transpose
pub(crate) fn solve_btran(lu: &LuFactorization, rhs: &mut Vec<f64>) {
    let n = lu.n;

    // Step 1: Apply column permutation
    let orig = rhs.clone();
    for i in 0..n {
        rhs[i] = orig[lu.p_col[i]];
    }

    // Step 2: Forward substitution with U^T
    lu.u.solve_transpose(rhs);

    // Step 3: Back substitution with L^T
    lu.l.solve_transpose(rhs);

    // Step 4: Apply row permutation transpose
    let w = rhs.clone();
    for i in 0..n {
        rhs[lu.p_row[i]] = w[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < tol,
                "Mismatch at index {}: {} vs {} (diff={})",
                i,
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    fn dense_to_csc(dense: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if dense[i][j].abs() > 1e-15 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(dense[i][j]);
                }
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
    }

    #[test]
    fn test_lu_identity() {
        let a = CscMatrix::identity(3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        for i in 0..3 {
            let mut rhs = vec![0.0; 3];
            rhs[i] = 1.0;
            let expected = rhs.clone();
            solve_ftran(&lu, &mut rhs);
            assert_vec_near(&rhs, &expected, 1e-10);
        }
    }

    #[test]
    fn test_lu_3x3() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_4x4_sparse() {
        let dense = vec![
            vec![4.0, 0.0, 1.0, 0.0],
            vec![0.0, 3.0, 0.0, 2.0],
            vec![1.0, 0.0, 5.0, 0.0],
            vec![0.0, 1.0, 0.0, 6.0],
        ];
        let a = dense_to_csc(&dense, 4, 4);
        let basis = vec![0, 1, 2, 3];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let rhs_orig = vec![5.0, 5.0, 6.0, 7.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_btran() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_btran(&lu, &mut rhs);

        let bt = a.transpose();
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_singular_detection() {
        let dense = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let result = LuFactorization::factorize(&a, &basis);
        assert!(result.is_err(), "Should detect singular matrix");
    }

    #[test]
    fn test_lu_markowitz() {
        let dense = vec![
            vec![0.001, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let rhs_orig = vec![1.001, 2.0, 2.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_ftran_btran_consistency() {
        let dense = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 4.0, 2.0],
            vec![0.0, 2.0, 5.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let c = vec![4.0, 5.0, 6.0];

        let mut x = b.clone();
        solve_ftran(&lu, &mut x);

        let mut y = c.clone();
        solve_btran(&lu, &mut y);

        let xtc: f64 = x.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        let bty: f64 = b.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (xtc - bty).abs() < 1e-10,
            "Adjoint property failed: x^T*c={} vs b^T*y={}",
            xtc,
            bty
        );
    }

    // --- New tests for sparse LU ---

    #[test]
    fn test_lu_20x20_sparse() {
        // 20x20 sparse matrix with ~10% density (diagonal dominant)
        let n = 20;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        // Strong diagonal for non-singularity
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(10.0 + (i as f64) * 0.5);
        }

        // Off-diagonal entries
        let off_diag: Vec<(usize, usize, f64)> = vec![
            (0, 3, 1.5), (0, 7, -0.8), (1, 5, 2.1), (1, 12, -1.3),
            (2, 8, 0.9), (2, 15, -0.4), (3, 0, -1.2), (3, 11, 0.7),
            (4, 9, 1.8), (4, 16, -0.6), (5, 1, -0.5), (5, 13, 1.1),
            (6, 2, 0.3), (6, 14, -1.9), (7, 0, 0.8), (7, 17, -0.7),
            (8, 2, -1.4), (8, 18, 0.6), (9, 4, 1.0), (9, 19, -0.3),
            (10, 6, -0.9), (10, 3, 1.7), (11, 3, -0.2), (11, 8, 0.5),
            (12, 1, 0.4), (12, 7, -1.6), (13, 5, -0.8), (13, 10, 1.3),
            (14, 6, 0.7), (14, 11, -1.1), (15, 2, 0.6), (15, 9, -0.5),
            (16, 4, -1.0), (16, 13, 0.9), (17, 7, 1.2), (17, 12, -0.4),
            (18, 8, -0.3), (18, 15, 1.5), (19, 9, 0.8), (19, 14, -0.6),
        ];
        for (r, c, v) in &off_diag {
            rows.push(*r);
            cols.push(*c);
            vals.push(*v);
        }

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let basis: Vec<usize> = (0..n).collect();
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        // Test FTRAN with multiple rhs
        for k in 0..3 {
            let rhs_orig: Vec<f64> = (0..n)
                .map(|i| ((i + k * 7) % 11) as f64 - 5.0)
                .collect();
            let mut rhs = rhs_orig.clone();
            solve_ftran(&lu, &mut rhs);
            let check = a.mat_vec_mul(&rhs).unwrap();
            assert_vec_near(&check, &rhs_orig, 1e-8);
        }

        // Test BTRAN
        let bt = a.transpose();
        let rhs_orig: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 3.0).collect();
        let mut rhs = rhs_orig.clone();
        solve_btran(&lu, &mut rhs);
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_fill_in() {
        // Arrow matrix: dense first row/col causes fill-in
        let n = 8;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        // Row 0: all non-zero
        for j in 0..n {
            rows.push(0);
            cols.push(j);
            vals.push(1.0 + j as f64 * 0.3);
        }
        // Column 0: all non-zero
        for i in 1..n {
            rows.push(i);
            cols.push(0);
            vals.push(0.5 + i as f64 * 0.2);
        }
        // Diagonal (i >= 1)
        for i in 1..n {
            rows.push(i);
            cols.push(i);
            vals.push(10.0 + i as f64);
        }

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let basis: Vec<usize> = (0..n).collect();
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        // Verify L has off-diagonal entries (fill-in)
        assert!(lu.l.values.len() > 0, "L should have off-diagonal entries");

        // Verify correctness
        let rhs_orig: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);

        // BTRAN
        let bt = a.transpose();
        let mut rhs2 = rhs_orig.clone();
        solve_btran(&lu, &mut rhs2);
        let check2 = bt.mat_vec_mul(&rhs2).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_ill_conditioned() {
        // Ill-conditioned 5x5 matrix
        let dense = vec![
            vec![1.0, 0.0, 0.0, 0.0, 1e-4],
            vec![0.0, 1.0, 0.0, 1e-4, 0.0],
            vec![0.0, 0.0, 1e-4, 0.0, 0.0],
            vec![0.0, 1e-4, 0.0, 1.0, 0.0],
            vec![1e-4, 0.0, 0.0, 0.0, 1.0],
        ];
        let a = dense_to_csc(&dense, 5, 5);
        let basis = vec![0, 1, 2, 3, 4];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        let rhs_orig = vec![1.0001, 1.0001, 0.0001, 1.0001, 1.0001];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-6);

        // BTRAN
        let bt = a.transpose();
        let mut rhs2 = rhs_orig.clone();
        solve_btran(&lu, &mut rhs2);
        let check2 = bt.mat_vec_mul(&rhs2).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-6);
    }
}
