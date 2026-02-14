//! Sparse LU factorization with Markowitz threshold pivoting

use crate::sparse::CscMatrix;

/// LU factorization result: PA = LU with column permutation
/// Stores L, U as dense matrices for correctness (sparse optimization deferred to M2 later phase)
#[derive(Debug, Clone)]
pub(crate) struct LuFactorization {
    /// Lower triangular matrix (row-major, m x m). L[i][j] for i >= j.
    /// Diagonal of L is 1.0 (unit lower triangular).
    pub l: Vec<Vec<f64>>,
    /// Upper triangular matrix (row-major, m x m). U[i][j] for i <= j.
    pub u: Vec<Vec<f64>>,
    /// Row permutation: row i of the permuted system corresponds to row p_row[i] of original
    pub p_row: Vec<usize>,
    /// Column permutation: col j of the permuted system corresponds to col p_col[j] of original
    pub p_col: Vec<usize>,
    /// Dimension
    pub n: usize,
}

const MARKOWITZ_THRESHOLD: f64 = 0.1;
const SINGULAR_TOL: f64 = 1e-12;

impl LuFactorization {
    /// Factorize the basis matrix B (columns of A selected by `basis` indices)
    /// Returns P_row * B * P_col^T = L * U
    pub fn factorize(a: &CscMatrix, basis: &[usize]) -> Result<Self, String> {
        let m = basis.len();
        if m == 0 {
            return Err("Empty basis".to_string());
        }

        // Extract basis columns into dense m x m matrix
        let mut b = vec![vec![0.0; m]; m];
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
                    b[row][j] = a.values[k];
                }
            }
        }

        // Initialize permutation vectors (identity)
        let mut p_row: Vec<usize> = (0..m).collect();
        let mut p_col: Vec<usize> = (0..m).collect();

        // Working copy for elimination
        let mut work = b;

        // L matrix: unit lower triangular, stored as multipliers
        let mut l = vec![vec![0.0; m]; m];
        for i in 0..m {
            l[i][i] = 1.0;
        }

        // Gaussian elimination with Markowitz threshold pivoting
        for step in 0..m {
            // Count non-zeros in active submatrix for Markowitz
            let mut row_nnz = vec![0usize; m];
            let mut col_nnz = vec![0usize; m];
            for i in step..m {
                for j in step..m {
                    if work[i][j].abs() > SINGULAR_TOL {
                        row_nnz[i] += 1;
                        col_nnz[j] += 1;
                    }
                }
            }

            // Find column maximum for threshold test
            let mut col_max = vec![0.0f64; m];
            for j in step..m {
                for i in step..m {
                    let abs_val = work[i][j].abs();
                    if abs_val > col_max[j] {
                        col_max[j] = abs_val;
                    }
                }
            }

            // Find pivot with minimum Markowitz count that satisfies threshold
            let mut best_pivot: Option<(usize, usize)> = None;
            let mut best_markowitz = usize::MAX;

            for i in step..m {
                for j in step..m {
                    let abs_val = work[i][j].abs();
                    if abs_val <= SINGULAR_TOL {
                        continue;
                    }
                    // Threshold test: |a_ij| >= threshold * max|a_kj|
                    if abs_val < MARKOWITZ_THRESHOLD * col_max[j] {
                        continue;
                    }
                    let markowitz = (row_nnz[i].saturating_sub(1)) * (col_nnz[j].saturating_sub(1));
                    if markowitz < best_markowitz {
                        best_markowitz = markowitz;
                        best_pivot = Some((i, j));
                    }
                }
            }

            let (pivot_row, pivot_col) = match best_pivot {
                Some(p) => p,
                None => return Err(format!("Singular matrix detected at step {}", step)),
            };

            // Swap rows: step <-> pivot_row
            if pivot_row != step {
                work.swap(step, pivot_row);
                p_row.swap(step, pivot_row);
                // Swap L entries for previously computed columns
                for k in 0..step {
                    let tmp = l[step][k];
                    l[step][k] = l[pivot_row][k];
                    l[pivot_row][k] = tmp;
                }
            }

            // Swap columns: step <-> pivot_col
            if pivot_col != step {
                for i in 0..m {
                    work[i].swap(step, pivot_col);
                }
                p_col.swap(step, pivot_col);
            }

            let pivot_val = work[step][step];

            // Eliminate below pivot
            for i in (step + 1)..m {
                if work[i][step].abs() > SINGULAR_TOL {
                    let multiplier = work[i][step] / pivot_val;
                    l[i][step] = multiplier;
                    work[i][step] = 0.0;
                    for j in (step + 1)..m {
                        work[i][j] -= multiplier * work[step][j];
                    }
                }
            }
        }

        // U is the final work matrix (upper triangular)
        let u = work;

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
/// B * x = rhs => P_row * B * P_col^T * (P_col * x) = P_row * rhs
/// L * U * y = P_row * rhs, where y = P_col * x
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

    // Step 2: Forward substitution (L * z = rhs)
    // L is unit lower triangular
    for i in 0..n {
        for j in 0..i {
            rhs[i] -= lu.l[i][j] * rhs[j];
        }
    }

    // Step 3: Back substitution (U * y = z)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            rhs[i] -= lu.u[i][j] * rhs[j];
        }
        rhs[i] /= lu.u[i][i];
    }

    // Step 4: Apply inverse column permutation
    // p_col maps permuted col index -> original col index
    // We need: x[p_col[i]] = y[i]
    let y = rhs.clone();
    for i in 0..n {
        rhs[lu.p_col[i]] = y[i];
    }
}

/// BTRAN: solve B^T * x = rhs using LU factors
/// B^T * x = rhs => (P_col^T)^T * U^T * L^T * (P_row)^T * x = rhs
/// P_col * U^T * L^T * P_row^T * x = rhs
/// 1. Apply inverse column permutation to rhs: rhs' = P_col^T * rhs (i.e., gather by p_col)
/// Actually let me re-derive:
/// P_row * B * P_col^T = L * U
/// B = P_row^{-1} * L * U * P_col
/// B^T = P_col^T * U^T * L^T * P_row^{-T}
/// B^T * x = rhs
/// P_col^T * U^T * L^T * P_row^{-T} * x = rhs
/// Let w = P_row^{-T} * x, i.e., x = P_row^T * w, i.e., w[i] = x[p_row[i]]
/// P_col^T * U^T * L^T * w = rhs
/// Let v = L^T * w
/// P_col^T * U^T * v = rhs
/// U^T * v = P_col * rhs
/// 1. Apply column permutation to rhs: rhs'[i] = rhs[p_col[i]]
/// 2. Forward substitution with U^T: U^T * v = rhs'
/// 3. Back substitution with L^T: L^T * w = v
/// 4. Apply row permutation transpose: x[p_row[i]] = w[i]
pub(crate) fn solve_btran(lu: &LuFactorization, rhs: &mut Vec<f64>) {
    let n = lu.n;

    // Step 1: Apply column permutation: rhs'[i] = rhs[p_col[i]]
    let orig = rhs.clone();
    for i in 0..n {
        rhs[i] = orig[lu.p_col[i]];
    }

    // Step 2: Forward substitution with U^T (U^T is lower triangular)
    for i in 0..n {
        rhs[i] /= lu.u[i][i];
        for j in (i + 1)..n {
            rhs[j] -= lu.u[i][j] * rhs[i];
        }
    }

    // Step 3: Back substitution with L^T (L^T is upper triangular, unit diagonal)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            rhs[i] -= lu.l[j][i] * rhs[j];
        }
    }

    // Step 4: Apply inverse of P_row^T: x[p_row[i]] = w[i]
    let w = rhs.clone();
    for i in 0..n {
        rhs[lu.p_row[i]] = w[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ: {} vs {}", a.len(), b.len());
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

    /// Helper: build CscMatrix from dense row-major matrix
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

        // FTRAN(e_i) should return e_i
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
        // B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        // Test: B * x = rhs => x = B^{-1} * rhs
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        // Verify: B * x should equal rhs_orig
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_4x4_sparse() {
        // Sparse 4x4 matrix
        // [[4, 0, 1, 0],
        //  [0, 3, 0, 2],
        //  [1, 0, 5, 0],
        //  [0, 1, 0, 6]]
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
        // B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize(&a, &basis).unwrap();

        // BTRAN solves B^T * x = rhs
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_btran(&lu, &mut rhs);

        // Verify: B^T * x should equal rhs_orig
        let bt = a.transpose();
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_singular_detection() {
        // Singular matrix: row 2 = row 0
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
        // Matrix with mixed large/small elements to test Markowitz pivoting stability
        // [[0.001, 1.0, 0.0],
        //  [1.0,   0.0, 1.0],
        //  [0.0,   1.0, 1.0]]
        // Without good pivoting, 0.001 as first pivot would cause large fill-in
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
        // For any non-singular B: if FTRAN gives B*x=b => x, and BTRAN gives B^T*y=c => y,
        // then x^T * c = b^T * y (adjoint property)
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

        // x^T * c should equal b^T * y
        let xtc: f64 = x.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        let bty: f64 = b.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (xtc - bty).abs() < 1e-10,
            "Adjoint property failed: x^T*c={} vs b^T*y={}",
            xtc,
            bty
        );
    }
}
