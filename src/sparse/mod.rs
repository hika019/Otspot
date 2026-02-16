//! Sparse matrix and vector data structures (CSC format)

use std::collections::HashMap;

/// Sparse vector representation (index-value pairs, sorted by index)
#[derive(Debug, Clone)]
pub struct SparseVec {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
    pub len: usize, // logical length
}

const EPS: f64 = 1e-12;

impl SparseVec {
    /// Create an empty sparse vector of given logical length
    pub fn new(len: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            len,
        }
    }

    /// Create SparseVec from dense slice, keeping only non-zero entries (|v| > EPS)
    pub fn from_dense(dense: &[f64]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > EPS {
                indices.push(i);
                values.push(v);
            }
        }
        Self {
            indices,
            values,
            len: dense.len(),
        }
    }

    /// Convert to dense vector
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.len];
        for (k, &idx) in self.indices.iter().enumerate() {
            dense[idx] = self.values[k];
        }
        dense
    }

    /// Get value at index (0.0 if not present)
    pub fn get(&self, idx: usize) -> f64 {
        match self.indices.binary_search(&idx) {
            Ok(pos) => self.values[pos],
            Err(_) => 0.0,
        }
    }

    /// Set value at index. If val is near zero, remove the entry.
    pub fn set(&mut self, idx: usize, val: f64) {
        match self.indices.binary_search(&idx) {
            Ok(pos) => {
                if val.abs() <= EPS {
                    self.indices.remove(pos);
                    self.values.remove(pos);
                } else {
                    self.values[pos] = val;
                }
            }
            Err(pos) => {
                if val.abs() > EPS {
                    self.indices.insert(pos, idx);
                    self.values.insert(pos, val);
                }
            }
        }
    }

    /// self += alpha * other
    pub fn axpy(&mut self, alpha: f64, other: &SparseVec) {
        // Use dense conversion for correctness
        let mut dense = self.to_dense();
        for (k, &idx) in other.indices.iter().enumerate() {
            if idx < dense.len() {
                dense[idx] += alpha * other.values[k];
            }
        }
        let result = SparseVec::from_dense(&dense);
        self.indices = result.indices;
        self.values = result.values;
    }

    /// Dot product with another sparse vector
    pub fn dot(&self, other: &SparseVec) -> f64 {
        let mut result = 0.0;
        let (mut i, mut j) = (0, 0);
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                result += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if self.indices[i] < other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    /// Dot product with a dense vector
    pub fn dot_dense(&self, dense: &[f64]) -> f64 {
        let mut result = 0.0;
        for (k, &idx) in self.indices.iter().enumerate() {
            if idx < dense.len() {
                result += self.values[k] * dense[idx];
            }
        }
        result
    }
}

/// Compressed Sparse Column (CSC) matrix format
#[derive(Debug, Clone)]
pub struct CscMatrix {
    /// Column pointers (length ncols + 1)
    pub col_ptr: Vec<usize>,
    /// Row indices for each non-zero element
    pub row_ind: Vec<usize>,
    /// Values for each non-zero element
    pub values: Vec<f64>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
}

impl CscMatrix {
    /// Create a new empty CSC matrix
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            col_ptr: vec![0; ncols + 1],
            row_ind: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Create CSC matrix from COO (coordinate) format triplets
    /// If multiple values exist for the same (row, col), they are summed
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, String> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err("Triplet arrays must have same length".to_string());
        }

        // Accumulate values for duplicate (row, col) pairs
        let mut map: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..rows.len() {
            let r = rows[i];
            let c = cols[i];
            let v = vals[i];

            if r >= nrows {
                return Err(format!("Row index {} out of bounds (nrows={})", r, nrows));
            }
            if c >= ncols {
                return Err(format!("Col index {} out of bounds (ncols={})", c, ncols));
            }

            *map.entry((r, c)).or_insert(0.0) += v;
        }

        // Convert to sorted triplets
        let mut triplets: Vec<(usize, usize, f64)> = map
            .into_iter()
            .filter(|(_, v)| v.abs() > 1e-15) // Filter near-zero values
            .map(|((r, c), v)| (c, r, v)) // Sort by column first, then row
            .collect();
        triplets.sort_by_key(|&(c, r, _)| (c, r));

        // Build CSC format
        let mut col_ptr = vec![0; ncols + 1];
        let mut row_ind = Vec::new();
        let mut values = Vec::new();

        let mut current_col = 0;
        for (c, r, v) in triplets {
            // Fill col_ptr for empty columns
            while current_col < c {
                current_col += 1;
                col_ptr[current_col] = row_ind.len();
            }
            row_ind.push(r);
            values.push(v);
        }

        // Fill remaining col_ptr entries
        while current_col < ncols {
            current_col += 1;
            col_ptr[current_col] = row_ind.len();
        }

        Ok(Self {
            col_ptr,
            row_ind,
            values,
            nrows,
            ncols,
        })
    }

    /// Transpose the matrix (returns new CSC matrix)
    pub fn transpose(&self) -> Self {
        // Transpose CSC -> CSR of original -> CSC of transpose
        // Collect triplets and rebuild
        let mut triplets = Vec::new();
        for col in 0..self.ncols {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for idx in start..end {
                let row = self.row_ind[idx];
                let val = self.values[idx];
                triplets.push((row, col, val));
            }
        }

        // Build transposed matrix (swap nrows/ncols, swap row/col in triplets)
        let rows: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let cols: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
        let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

        Self::from_triplets(&rows, &cols, &vals, self.ncols, self.nrows)
            .expect("Transpose should never fail on valid matrix")
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn mat_vec_mul(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.ncols {
            return Err(format!(
                "Vector length {} does not match ncols {}",
                x.len(),
                self.ncols
            ));
        }

        let mut y = vec![0.0; self.nrows];
        for (col, &x_val) in x.iter().enumerate() {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for idx in start..end {
                let row = self.row_ind[idx];
                let a_val = self.values[idx];
                y[row] += a_val * x_val;
            }
        }
        Ok(y)
    }

    /// Get the non-zero elements of column j
    /// Returns (row_indices, values) slices
    pub fn get_column(&self, j: usize) -> Result<(&[usize], &[f64]), String> {
        if j >= self.ncols {
            return Err(format!("Column index {} out of bounds (ncols={})", j, self.ncols));
        }
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        Ok((&self.row_ind[start..end], &self.values[start..end]))
    }

    /// Create n x n identity matrix in CSC format
    pub fn identity(n: usize) -> Self {
        let col_ptr: Vec<usize> = (0..=n).collect();
        let row_ind: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        Self {
            col_ptr,
            row_ind,
            values,
            nrows: n,
            ncols: n,
        }
    }
}

/// Compressed Sparse Row (CSR) matrix format
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Row pointers (length nrows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero element
    pub col_ind: Vec<usize>,
    /// Values for each non-zero element
    pub values: Vec<f64>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
}

impl CsrMatrix {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, String> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err("Triplet arrays must have same length".to_string());
        }

        let mut map: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..rows.len() {
            if rows[i] >= nrows {
                return Err(format!("Row index {} out of bounds (nrows={})", rows[i], nrows));
            }
            if cols[i] >= ncols {
                return Err(format!("Col index {} out of bounds (ncols={})", cols[i], ncols));
            }
            *map.entry((rows[i], cols[i])).or_insert(0.0) += vals[i];
        }

        let mut triplets: Vec<(usize, usize, f64)> = map
            .into_iter()
            .filter(|(_, v)| v.abs() > 1e-15)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        let mut row_ptr = vec![0; nrows + 1];
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut current_row = 0;
        for (r, c, v) in triplets {
            while current_row < r {
                current_row += 1;
                row_ptr[current_row] = col_ind.len();
            }
            col_ind.push(c);
            values.push(v);
        }
        while current_row < nrows {
            current_row += 1;
            row_ptr[current_row] = col_ind.len();
        }

        Ok(Self {
            row_ptr,
            col_ind,
            values,
            nrows,
            ncols,
        })
    }

    pub fn get_row(&self, i: usize) -> Result<(&[usize], &[f64]), String> {
        if i >= self.nrows {
            return Err(format!("Row index {} out of bounds (nrows={})", i, self.nrows));
        }
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        Ok((&self.col_ind[start..end], &self.values[start..end]))
    }

    pub fn from_csc(csc: &CscMatrix) -> Self {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..csc.ncols {
            let start = csc.col_ptr[j];
            let end = csc.col_ptr[j + 1];
            for k in start..end {
                rows.push(csc.row_ind[k]);
                cols.push(j);
                vals.push(csc.values[k]);
            }
        }
        Self::from_triplets(&rows, &cols, &vals, csc.nrows, csc.ncols)
            .expect("Conversion from valid CSC should never fail")
    }
}

/// Sparse unit lower triangular matrix in CSC format.
/// Diagonal is implicitly 1.0 (not stored).
/// Column j has entries only at rows i > j.
#[derive(Debug, Clone)]
pub(crate) struct SparseLowerCSC {
    pub col_ptr: Vec<usize>,
    pub row_ind: Vec<usize>,
    pub values: Vec<f64>,
    pub n: usize,
}

impl SparseLowerCSC {
    /// Forward substitution: solve L * x = b (in-place)
    pub fn forward_solve(&self, rhs: &mut [f64]) {
        for j in 0..self.n {
            let x_j = rhs[j];
            if x_j == 0.0 {
                continue;
            }
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            for k in start..end {
                rhs[self.row_ind[k]] -= self.values[k] * x_j;
            }
        }
    }

    /// Solve L^T * x = b (in-place). L^T is unit upper triangular.
    pub fn solve_transpose(&self, rhs: &mut [f64]) {
        for j in (0..self.n).rev() {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * rhs[self.row_ind[k]];
            }
            rhs[j] -= sum;
        }
    }
}

/// Sparse upper triangular matrix in CSR format.
/// Diagonal stored separately; off-diagonal entries at columns j > i for row i.
#[derive(Debug, Clone)]
pub(crate) struct SparseUpperCSR {
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<usize>,
    pub values: Vec<f64>,
    pub diag: Vec<f64>,
    pub n: usize,
}

impl SparseUpperCSR {
    /// Backward substitution: solve U * x = b (in-place)
    pub fn backward_solve(&self, rhs: &mut [f64]) {
        for i in (0..self.n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * rhs[self.col_ind[k]];
            }
            rhs[i] = (rhs[i] - sum) / self.diag[i];
        }
    }

    /// Solve U^T * x = b (in-place). U^T is lower triangular.
    pub fn solve_transpose(&self, rhs: &mut [f64]) {
        for i in 0..self.n {
            rhs[i] /= self.diag[i];
            let x_i = rhs[i];
            if x_i == 0.0 {
                continue;
            }
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                rhs[self.col_ind[k]] -= self.values[k] * x_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_triplets_basic() {
        // 3x3 matrix:
        // [1.0  0.0  2.0]
        // [0.0  3.0  0.0]
        // [4.0  0.0  5.0]
        let rows = vec![0, 2, 1, 0, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let vals = vec![1.0, 4.0, 3.0, 2.0, 5.0];

        let mat = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();

        assert_eq!(mat.nrows, 3);
        assert_eq!(mat.ncols, 3);
        assert_eq!(mat.nnz(), 5);

        // Check column 0: [1.0 at row 0, 4.0 at row 2]
        let (row_idx, values) = mat.get_column(0).unwrap();
        assert_eq!(row_idx, &[0, 2]);
        assert_eq!(values, &[1.0, 4.0]);

        // Check column 1: [3.0 at row 1]
        let (row_idx, values) = mat.get_column(1).unwrap();
        assert_eq!(row_idx, &[1]);
        assert_eq!(values, &[3.0]);

        // Check column 2: [2.0 at row 0, 5.0 at row 2]
        let (row_idx, values) = mat.get_column(2).unwrap();
        assert_eq!(row_idx, &[0, 2]);
        assert_eq!(values, &[2.0, 5.0]);
    }

    #[test]
    fn test_from_triplets_duplicate_entries() {
        // Same (row, col) appears twice -> values should be summed
        let rows = vec![0, 0, 1];
        let cols = vec![0, 0, 1];
        let vals = vec![1.0, 2.0, 3.0];

        let mat = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();

        // Column 0: row 0 should have 1.0 + 2.0 = 3.0
        let (row_idx, values) = mat.get_column(0).unwrap();
        assert_eq!(row_idx, &[0]);
        assert_eq!(values, &[3.0]);

        // Column 1: row 1 should have 3.0
        let (row_idx, values) = mat.get_column(1).unwrap();
        assert_eq!(row_idx, &[1]);
        assert_eq!(values, &[3.0]);
    }

    #[test]
    fn test_transpose() {
        // 2x3 matrix:
        // [1.0  2.0  0.0]
        // [0.0  0.0  3.0]
        let rows = vec![0, 0, 1];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];

        let mat = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        let mat_t = mat.transpose();

        // Transposed should be 3x2:
        // [1.0  0.0]
        // [2.0  0.0]
        // [0.0  3.0]
        assert_eq!(mat_t.nrows, 3);
        assert_eq!(mat_t.ncols, 2);
        assert_eq!(mat_t.nnz(), 3);

        // Check column 0: [1.0 at row 0, 2.0 at row 1]
        let (row_idx, values) = mat_t.get_column(0).unwrap();
        assert_eq!(row_idx, &[0, 1]);
        assert_eq!(values, &[1.0, 2.0]);

        // Check column 1: [3.0 at row 2]
        let (row_idx, values) = mat_t.get_column(1).unwrap();
        assert_eq!(row_idx, &[2]);
        assert_eq!(values, &[3.0]);

        // Double transpose should return to original
        let mat_tt = mat_t.transpose();
        assert_eq!(mat_tt.nrows, mat.nrows);
        assert_eq!(mat_tt.ncols, mat.ncols);
        assert_eq!(mat_tt.row_ind, mat.row_ind);
        assert_eq!(mat_tt.col_ptr, mat.col_ptr);
        assert_eq!(mat_tt.values, mat.values);
    }

    #[test]
    fn test_mat_vec_mul() {
        // 3x3 matrix:
        // [1.0  0.0  2.0]
        // [0.0  3.0  0.0]
        // [4.0  0.0  5.0]
        let rows = vec![0, 2, 1, 0, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let vals = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let mat = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = mat.mat_vec_mul(&x).unwrap();

        // Expected: [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 0*3, 4*1 + 0*2 + 5*3]
        //         = [7.0, 6.0, 19.0]
        assert_eq!(y.len(), 3);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 19.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_vec_mul_dimension_mismatch() {
        let mat = CscMatrix::identity(3);
        let x = vec![1.0, 2.0]; // Wrong size
        let result = mat.mat_vec_mul(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity() {
        let id = CscMatrix::identity(4);
        assert_eq!(id.nrows, 4);
        assert_eq!(id.ncols, 4);
        assert_eq!(id.nnz(), 4);

        // Each column should have exactly one entry at its own row
        for j in 0..4 {
            let (row_idx, values) = id.get_column(j).unwrap();
            assert_eq!(row_idx, &[j]);
            assert_eq!(values, &[1.0]);
        }

        // Identity * vector = vector
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = id.mat_vec_mul(&x).unwrap();
        assert_eq!(y, x);
    }

    #[test]
    fn test_empty_matrix() {
        let mat = CscMatrix::from_triplets(&[], &[], &[], 2, 3).unwrap();
        assert_eq!(mat.nrows, 2);
        assert_eq!(mat.ncols, 3);
        assert_eq!(mat.nnz(), 0);

        // All columns should be empty
        for j in 0..3 {
            let (row_idx, values) = mat.get_column(j).unwrap();
            assert_eq!(row_idx.len(), 0);
            assert_eq!(values.len(), 0);
        }

        // mat_vec_mul should return zero vector
        let y = mat.mat_vec_mul(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn test_get_column_out_of_bounds() {
        let mat = CscMatrix::identity(3);
        let result = mat.get_column(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_triplets_out_of_bounds() {
        // Row index out of bounds
        let result = CscMatrix::from_triplets(&[0, 3], &[0, 0], &[1.0, 2.0], 3, 2);
        assert!(result.is_err());

        // Column index out of bounds
        let result = CscMatrix::from_triplets(&[0, 0], &[0, 2], &[1.0, 2.0], 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_triplets_mismatched_lengths() {
        let result = CscMatrix::from_triplets(&[0, 1], &[0], &[1.0, 2.0], 2, 2);
        assert!(result.is_err());
    }

    // ---- SparseVec tests ----

    #[test]
    fn test_sparse_vec_from_dense_to_dense() {
        let dense = vec![1.0, 0.0, 0.0, 3.5, 0.0, -2.0];
        let sv = SparseVec::from_dense(&dense);
        assert_eq!(sv.len, 6);
        assert_eq!(sv.indices, vec![0, 3, 5]);
        assert_eq!(sv.values, vec![1.0, 3.5, -2.0]);

        let back = sv.to_dense();
        assert_eq!(back, dense);
    }

    #[test]
    fn test_sparse_vec_get_set() {
        let mut sv = SparseVec::new(5);
        assert_eq!(sv.get(0), 0.0);
        assert_eq!(sv.get(4), 0.0);

        sv.set(2, 7.0);
        sv.set(4, -1.0);
        assert_eq!(sv.get(2), 7.0);
        assert_eq!(sv.get(4), -1.0);
        assert_eq!(sv.get(3), 0.0);

        // Overwrite
        sv.set(2, 3.0);
        assert_eq!(sv.get(2), 3.0);

        // Remove by setting to zero
        sv.set(2, 0.0);
        assert_eq!(sv.get(2), 0.0);
        assert_eq!(sv.indices, vec![4]);
    }

    #[test]
    fn test_sparse_vec_dot() {
        let a = SparseVec::from_dense(&[1.0, 0.0, 3.0, 0.0]);
        let b = SparseVec::from_dense(&[2.0, 5.0, 4.0, 0.0]);
        // 1*2 + 0*5 + 3*4 + 0*0 = 14
        assert!((a.dot(&b) - 14.0).abs() < 1e-10);

        // Dot with dense
        let dense = vec![2.0, 5.0, 4.0, 0.0];
        assert!((a.dot_dense(&dense) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_vec_axpy() {
        let mut a = SparseVec::from_dense(&[1.0, 0.0, 3.0]);
        let b = SparseVec::from_dense(&[0.0, 2.0, 1.0]);
        a.axpy(2.0, &b);
        // a = [1, 0, 3] + 2*[0, 2, 1] = [1, 4, 5]
        let dense = a.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-10);
        assert!((dense[1] - 4.0).abs() < 1e-10);
        assert!((dense[2] - 5.0).abs() < 1e-10);
    }

    // ---- CsrMatrix tests ----

    #[test]
    fn test_csr_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mat = CsrMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        assert_eq!(mat.nrows, 3);
        assert_eq!(mat.ncols, 3);
        assert_eq!(mat.nnz(), 5);

        let (ci, v) = mat.get_row(0).unwrap();
        assert_eq!(ci, &[0, 2]);
        assert_eq!(v, &[1.0, 2.0]);

        let (ci, v) = mat.get_row(1).unwrap();
        assert_eq!(ci, &[1]);
        assert_eq!(v, &[3.0]);

        let (ci, v) = mat.get_row(2).unwrap();
        assert_eq!(ci, &[0, 2]);
        assert_eq!(v, &[4.0, 5.0]);
    }

    #[test]
    fn test_csr_from_csc() {
        let rows = vec![0, 2, 1, 0, 2];
        let cols = vec![0, 0, 1, 2, 2];
        let vals = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let csc = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        let csr = CsrMatrix::from_csc(&csc);

        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 3);
        assert_eq!(csr.nnz(), 5);

        let (ci, v) = csr.get_row(0).unwrap();
        assert_eq!(ci, &[0, 2]);
        assert_eq!(v, &[1.0, 2.0]);

        let (ci, v) = csr.get_row(1).unwrap();
        assert_eq!(ci, &[1]);
        assert_eq!(v, &[3.0]);

        let (ci, v) = csr.get_row(2).unwrap();
        assert_eq!(ci, &[0, 2]);
        assert_eq!(v, &[4.0, 5.0]);
    }

    // ---- SparseLowerCSC tests ----

    #[test]
    fn test_sparse_lower_forward_solve() {
        // L = [[1, 0, 0], [2, 1, 0], [3, 4, 1]]
        // CSC: col 0 has (row 1, 2.0), (row 2, 3.0); col 1 has (row 2, 4.0); col 2 empty
        let l = SparseLowerCSC {
            col_ptr: vec![0, 2, 3, 3],
            row_ind: vec![1, 2, 2],
            values: vec![2.0, 3.0, 4.0],
            n: 3,
        };
        // Solve Lx = [1, 4, 18]
        // x[0] = 1, x[1] = 4 - 2*1 = 2, x[2] = 18 - 3*1 - 4*2 = 7
        let mut rhs = vec![1.0, 4.0, 18.0];
        l.forward_solve(&mut rhs);
        assert!((rhs[0] - 1.0).abs() < 1e-10);
        assert!((rhs[1] - 2.0).abs() < 1e-10);
        assert!((rhs[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_lower_solve_transpose() {
        // L = [[1, 0, 0], [2, 1, 0], [3, 4, 1]]
        // L^T = [[1, 2, 3], [0, 1, 4], [0, 0, 1]]
        // Solve L^T x = [11, 9, 1]: x[2]=1, x[1]=9-4*1=5, x[0]=11-2*5-3*1=11-10-3=-2
        let l = SparseLowerCSC {
            col_ptr: vec![0, 2, 3, 3],
            row_ind: vec![1, 2, 2],
            values: vec![2.0, 3.0, 4.0],
            n: 3,
        };
        let mut rhs = vec![11.0, 9.0, 1.0];
        l.solve_transpose(&mut rhs);
        assert!((rhs[0] - (-2.0)).abs() < 1e-10);
        assert!((rhs[1] - 5.0).abs() < 1e-10);
        assert!((rhs[2] - 1.0).abs() < 1e-10);
    }

    // ---- SparseUpperCSR tests ----

    #[test]
    fn test_sparse_upper_backward_solve() {
        // U = [[2, 1, 3], [0, 4, 2], [0, 0, 5]]
        // CSR (off-diag): row 0 has (col 1, 1.0), (col 2, 3.0); row 1 has (col 2, 2.0); row 2 empty
        let u = SparseUpperCSR {
            row_ptr: vec![0, 2, 3, 3],
            col_ind: vec![1, 2, 2],
            values: vec![1.0, 3.0, 2.0],
            diag: vec![2.0, 4.0, 5.0],
            n: 3,
        };
        // Solve Ux = [11, 10, 5]: x[2]=5/5=1, x[1]=(10-2*1)/4=2, x[0]=(11-1*2-3*1)/2=3
        let mut rhs = vec![11.0, 10.0, 5.0];
        u.backward_solve(&mut rhs);
        assert!((rhs[0] - 3.0).abs() < 1e-10);
        assert!((rhs[1] - 2.0).abs() < 1e-10);
        assert!((rhs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_upper_solve_transpose() {
        // U = [[2, 1, 3], [0, 4, 2], [0, 0, 5]]
        // U^T = [[2, 0, 0], [1, 4, 0], [3, 2, 5]]
        // Solve U^T x = [6, 9, 20]: x[0]=6/2=3, x[1]=(9-1*3)/4=1.5, x[2]=(20-3*3-2*1.5)/5=(20-9-3)/5=1.6
        let u = SparseUpperCSR {
            row_ptr: vec![0, 2, 3, 3],
            col_ind: vec![1, 2, 2],
            values: vec![1.0, 3.0, 2.0],
            diag: vec![2.0, 4.0, 5.0],
            n: 3,
        };
        let mut rhs = vec![6.0, 9.0, 20.0];
        u.solve_transpose(&mut rhs);
        assert!((rhs[0] - 3.0).abs() < 1e-10);
        assert!((rhs[1] - 1.5).abs() < 1e-10);
        assert!((rhs[2] - 1.6).abs() < 1e-10);
    }
}
