//! Ruiz equilibration scaling for LP preprocessing
//!
//! Iteratively scales rows and columns of the constraint matrix so that
//! the maximum absolute entry in each row/column is close to 1.

use crate::sparse::CscMatrix;

/// Ruiz equilibration scaler
///
/// Stores cumulative row and column scale factors from the iterative scaling.
pub struct RuizScaler {
    pub row_scale: Vec<f64>,
    pub col_scale: Vec<f64>,
}

impl RuizScaler {
    /// Apply Ruiz equilibration to a matrix, RHS vector, and cost vector.
    ///
    /// Returns `(scaled_matrix, scaled_b, scaled_c, row_scale, col_scale)`.
    ///
    /// The scaled problem satisfies:
    ///   Ã = diag(row_scale) * A * diag(col_scale)
    ///   b̃ = diag(row_scale) * b
    ///   c̃ = diag(col_scale) * c
    ///
    /// Iterates up to 20 rounds, stopping early when max scale change < 1e-4.
    pub fn scale(
        matrix: &CscMatrix,
        b: &[f64],
        c: &[f64],
    ) -> (CscMatrix, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = matrix.nrows;
        let n = matrix.ncols;

        let mut cumul_row = vec![1.0f64; m];
        let mut cumul_col = vec![1.0f64; n];

        let mut a = matrix.clone();
        let mut cur_b = b.to_vec();
        let mut cur_c = c.to_vec();

        for _ in 0..20 {
            // Compute row maximums (iterate over all non-zeros)
            let mut row_max = vec![0.0f64; m];
            for k in 0..a.row_ind.len() {
                let row = a.row_ind[k];
                let v = a.values[k].abs();
                if v > row_max[row] {
                    row_max[row] = v;
                }
            }

            // Compute column maximums
            let mut col_max = vec![0.0f64; n];
            for j in 0..n {
                let start = a.col_ptr[j];
                let end = a.col_ptr[j + 1];
                for k in start..end {
                    let v = a.values[k].abs();
                    if v > col_max[j] {
                        col_max[j] = v;
                    }
                }
            }

            // Compute scale factors: 1/sqrt(max), or 1.0 for empty rows/cols
            let row_factor: Vec<f64> = row_max
                .iter()
                .map(|&mx| if mx > 1e-300 { 1.0 / mx.sqrt() } else { 1.0 })
                .collect();
            let col_factor: Vec<f64> = col_max
                .iter()
                .map(|&mx| if mx > 1e-300 { 1.0 / mx.sqrt() } else { 1.0 })
                .collect();

            // Check convergence: max deviation of factors from 1.0
            let max_change = row_factor
                .iter()
                .chain(col_factor.iter())
                .map(|&f| (f - 1.0).abs())
                .fold(0.0f64, f64::max);

            // Apply scaling to matrix entries: a[i,j] *= row_factor[i] * col_factor[j]
            for j in 0..n {
                let start = a.col_ptr[j];
                let end = a.col_ptr[j + 1];
                for k in start..end {
                    let row = a.row_ind[k];
                    a.values[k] *= row_factor[row] * col_factor[j];
                }
            }

            // Apply scaling to b and c
            for i in 0..m {
                cur_b[i] *= row_factor[i];
            }
            for j in 0..n {
                cur_c[j] *= col_factor[j];
            }

            // Accumulate cumulative scales
            for i in 0..m {
                cumul_row[i] *= row_factor[i];
            }
            for j in 0..n {
                cumul_col[j] *= col_factor[j];
            }

            if max_change < 1e-4 {
                break;
            }
        }

        (a, cur_b, cur_c, cumul_row, cumul_col)
    }

    /// Unscale a primal solution vector.
    ///
    /// If the scaled problem solution is `x̃`, the original solution is:
    ///   `x_j = col_scale[j] * x̃_j`
    pub fn unscale_solution(x: &[f64], col_scale: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(col_scale.iter())
            .map(|(&v, &s)| v * s)
            .collect()
    }

    /// Unscale a dual solution vector.
    ///
    /// If the scaled problem dual is `ỹ`, the original dual is:
    ///   `y_i = row_scale[i] * ỹ_i`
    pub fn unscale_dual(y: &[f64], row_scale: &[f64]) -> Vec<f64> {
        y.iter()
            .zip(row_scale.iter())
            .map(|(&v, &s)| v * s)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dense_csc(data: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if data[i][j].abs() > 1e-15 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(data[i][j]);
                }
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
    }

    #[test]
    fn test_ruiz_scale_identity() {
        // Identity matrix — already balanced, scales should stay near 1
        let a = make_dense_csc(
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            2,
            2,
        );
        let b = vec![1.0, 2.0];
        let c = vec![1.0, 1.0];
        let (_, scaled_b, scaled_c, row_scale, col_scale) = RuizScaler::scale(&a, &b, &c);

        // Scales should be close to 1
        for &s in row_scale.iter().chain(col_scale.iter()) {
            assert!((s - 1.0).abs() < 0.01, "Scale {} far from 1", s);
        }
        // b and c should be unchanged (already balanced)
        for i in 0..2 {
            assert!((scaled_b[i] - b[i] * row_scale[i]).abs() < 1e-10);
            assert!((scaled_c[i] - c[i] * col_scale[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ruiz_scale_unbalanced() {
        // Matrix with very unbalanced entries
        // [[1000, 1], [1, 0.001]]
        let a = make_dense_csc(
            &[vec![1000.0, 1.0], vec![1.0, 0.001]],
            2,
            2,
        );
        let b = vec![1.0, 1.0];
        let c = vec![1.0, 1.0];
        let (scaled_a, _, _, _, _) = RuizScaler::scale(&a, &b, &c);

        // After scaling, all entries should be close to 1 in magnitude
        for k in 0..scaled_a.values.len() {
            assert!(
                scaled_a.values[k].abs() < 2.0,
                "Entry {} too large after scaling",
                scaled_a.values[k]
            );
        }
    }

    #[test]
    fn test_unscale_solution() {
        let x_scaled = vec![2.0, 3.0, 0.5];
        let col_scale = vec![0.5, 2.0, 4.0];
        let x_orig = RuizScaler::unscale_solution(&x_scaled, &col_scale);
        assert!((x_orig[0] - 1.0).abs() < 1e-10);
        assert!((x_orig[1] - 6.0).abs() < 1e-10);
        assert!((x_orig[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_unscale_dual() {
        let y_scaled = vec![1.0, 4.0];
        let row_scale = vec![2.0, 0.5];
        let y_orig = RuizScaler::unscale_dual(&y_scaled, &row_scale);
        assert!((y_orig[0] - 2.0).abs() < 1e-10);
        assert!((y_orig[1] - 2.0).abs() < 1e-10);
    }
}
