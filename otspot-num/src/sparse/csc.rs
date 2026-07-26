use super::compress::build_compressed_format;
use super::view::CscMatrixView;
use crate::SolverError;

/// 列圧縮形式（CSC: Compressed Sparse Column）の疎行列
///
/// 非ゼロ要素を列単位で格納する疎行列フォーマット。
/// 列ポインタ・行インデックス・値の3配列で表現される。
///
/// # フォーマット詳細
///
/// 列 `j` の非ゼロ要素は `values[col_ptr[j]..col_ptr[j+1]]` に格納され、
/// 対応する行インデックスは `row_ind[col_ptr[j]..col_ptr[j+1]]` に入る。
/// 各列の行インデックスは昇順にソートされている。
#[derive(Debug, Clone)]
pub struct CscMatrix {
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_ind: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl CscMatrixView for CscMatrix {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    #[inline]
    fn row_ind(&self) -> &[usize] {
        &self.row_ind
    }

    #[inline]
    fn values(&self) -> &[f64] {
        &self.values
    }
}

impl CscMatrix {
    /// 空の CSC 行列を生成する
    ///
    /// すべての要素がゼロの (nrows × ncols) 行列として初期化される。
    ///
    /// # 引数
    /// - `nrows`: 行数
    /// - `ncols`: 列数
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            col_ptr: vec![0; ncols + 1],
            row_ind: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    #[inline]
    pub fn row_ind(&self) -> &[usize] {
        &self.row_ind
    }

    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable view of the stored non-zero values (structure is preserved).
    ///
    /// This is the encapsulated path for in-place value scaling; callers must
    /// not change the length or ordering, only the coefficients.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Build a matrix from already-validated CSC arrays.
    ///
    /// The encapsulated replacement for direct struct-literal construction.
    /// `debug_assert`s enforce the CSC invariants in test/dev builds; release
    /// builds trust the caller (matching the previous struct-literal usage).
    pub fn from_raw_parts(
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<usize>,
        row_ind: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(col_ptr.len(), ncols + 1, "col_ptr length must be ncols+1");
        debug_assert_eq!(
            row_ind.len(),
            values.len(),
            "row_ind/values length mismatch"
        );
        debug_assert_eq!(
            col_ptr.first().copied(),
            Some(0),
            "col_ptr head must be zero"
        );
        debug_assert_eq!(
            col_ptr.last().copied(),
            Some(values.len()),
            "col_ptr tail must equal nnz"
        );
        debug_assert!(
            col_ptr.windows(2).all(|w| w[0] <= w[1]),
            "col_ptr must be monotone non-decreasing"
        );
        debug_assert!(
            col_ptr
                .windows(2)
                .all(|w| { row_ind[w[0]..w[1]].windows(2).all(|rows| rows[0] < rows[1]) }),
            "row indices within each column must be strictly ascending"
        );
        debug_assert!(
            row_ind.iter().all(|&r| r < nrows),
            "row index out of bounds for nrows={nrows}"
        );
        Self {
            col_ptr,
            row_ind,
            values,
            nrows,
            ncols,
        }
    }

    /// Returns a new matrix with all non-zero values multiplied by `factor`.
    pub fn scale_values(&self, factor: f64) -> Self {
        Self {
            col_ptr: self.col_ptr.clone(),
            row_ind: self.row_ind.clone(),
            values: self.values.iter().map(|&v| v * factor).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// 行優先の密行列 (`Vec<Vec<f64>>`, `nrows` 行 × `ncols` 列) に展開する。
    ///
    /// 小規模な行列 (conic bridge の QCQP/SOCP 変換など、O(n²) メモリが許容できる
    /// サイズ) 向け。大規模疎行列には使わないこと。
    pub fn to_dense_rows(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; self.ncols]; self.nrows];
        for j in 0..self.ncols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                d[self.row_ind[k]][j] = self.values[k];
            }
        }
        d
    }

    /// 各行の∞ノルム（行ごとの最大絶対値）を一括計算する: O(nnz)
    ///
    /// CSC格式では行方向アクセスが非効率だが、全非ゼロ要素を1回走査して
    /// 各行の最大絶対値を収集することで O(nnz) で完了する。
    pub fn row_infinity_norms(&self) -> Vec<f64> {
        let mut norms = vec![0.0_f64; self.nrows];
        for (&val, &row) in self.values.iter().zip(self.row_ind.iter()) {
            let abs_val = val.abs();
            if abs_val > norms[row] {
                norms[row] = abs_val;
            }
        }
        norms
    }

    /// Builds a CSC matrix from COO triplets.
    ///
    /// Duplicate `(row, col)` entries are summed; results with `|v| ≤ DROP_TOL` are dropped.
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, SolverError> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err(SolverError::DimensionMismatch {
                field: "triplet_arrays",
                expected: rows.len(),
                got: vals.len(),
            });
        }
        for (i, &v) in vals.iter().enumerate() {
            if !v.is_finite() {
                return Err(SolverError::NonFiniteCoefficient {
                    field: "matrix",
                    index: i,
                });
            }
        }
        // CSC: 主軸=列、副軸=行
        let (col_ptr, row_ind, values) = build_compressed_format(ncols, nrows, cols, rows, vals)?;
        Ok(Self {
            col_ptr,
            row_ind,
            values,
            nrows,
            ncols,
        })
    }

    /// 転置行列を生成する（新しい CSC 行列として返す）
    ///
    /// 元の行列の行と列を入れ替えた行列を返す。
    /// counting sort を使用するため O(nnz) の計算量となる。
    pub fn transpose(&self) -> Self {
        let nnz = self.nnz();
        // Transposed matrix: (ncols x nrows)
        // Step 1: count nnz per row of original (= nnz per col of transposed)
        let mut row_count = vec![0usize; self.nrows];
        for &r in &self.row_ind {
            row_count[r] += 1;
        }

        // Step 2: prefix sum to build col_ptr of transposed matrix
        let mut col_ptr = vec![0usize; self.nrows + 1];
        for r in 0..self.nrows {
            col_ptr[r + 1] = col_ptr[r] + row_count[r];
        }

        // Step 3: scatter non-zeros into transposed positions
        // Process columns 0..ncols in order; for each (row, col, val) in original,
        // write col as row_ind of transposed at position pos[row].
        // Since col increases monotonically, row_ind within each transposed column
        // is written in ascending order — no extra sort needed.
        let mut row_ind = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut pos = col_ptr[..self.nrows].to_vec();

        for col in 0..self.ncols {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for k in start..end {
                let row = self.row_ind[k];
                let p = pos[row];
                row_ind[p] = col;
                values[p] = self.values[k];
                pos[row] += 1;
            }
        }

        Self {
            col_ptr,
            row_ind,
            values,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Matrix-vector product y = A * x. O(nnz).
    pub fn mat_vec_mul(&self, x: &[f64]) -> Result<Vec<f64>, SolverError> {
        if x.len() != self.ncols {
            return Err(SolverError::DimensionMismatch {
                field: "vector",
                expected: self.ncols,
                got: x.len(),
            });
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

    /// Returns `(row_indices, values)` slices for column `j`; both are sorted by row index.
    #[inline]
    pub fn get_column(&self, j: usize) -> Result<(&[usize], &[f64]), SolverError> {
        if j >= self.ncols {
            return Err(SolverError::IndexOutOfBounds {
                context: "column",
                index: j,
                bound: self.ncols,
            });
        }
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        Ok((&self.row_ind[start..end], &self.values[start..end]))
    }

    /// Returns `(row_indices, values)` slices for column `j`; both are sorted by row index.
    ///
    /// Panics if `j >= ncols`; callers must guarantee a valid column index by construction
    /// invariant. Use [`Self::get_column`] when `j` is not provably in-bounds.
    #[inline]
    pub fn column(&self, j: usize) -> (&[usize], &[f64]) {
        assert!(
            j < self.ncols,
            "column {j} out of bounds (ncols={})",
            self.ncols
        );
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        (&self.row_ind[start..end], &self.values[start..end])
    }

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
    fn test_column_matches_get_column_in_bounds() {
        let mat = CscMatrix::identity(3);
        for j in 0..3 {
            let (rows, vals) = mat.column(j);
            let (exp_rows, exp_vals) = mat.get_column(j).unwrap();
            assert_eq!(rows, exp_rows);
            assert_eq!(vals, exp_vals);
        }
    }

    #[test]
    #[should_panic(expected = "column 3 out of bounds (ncols=3)")]
    fn test_column_out_of_bounds_panics() {
        let mat = CscMatrix::identity(3);
        let _ = mat.column(3);
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

    /// Sentinel: non-finite values in triplets must be rejected at construction.
    /// Removing the finiteness check turns Err into Ok → assertion fails (no-op fail).
    #[test]
    fn test_sentinel_triplet_non_finite_rejected() {
        let r = CscMatrix::from_triplets(&[0], &[0], &[f64::NAN], 1, 1);
        assert!(r.is_err(), "NaN in triplet vals must be rejected");
        let r = CscMatrix::from_triplets(&[0], &[0], &[f64::INFINITY], 1, 1);
        assert!(r.is_err(), "+Inf in triplet vals must be rejected");
        let r = CscMatrix::from_triplets(&[0], &[0], &[f64::NEG_INFINITY], 1, 1);
        assert!(r.is_err(), "-Inf in triplet vals must be rejected");
        // Finite values still accepted.
        let r = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1);
        assert!(r.is_ok(), "finite value must still be accepted");
    }

    /// Sentinel: `col_ptr` length must be `ncols + 1`. debug-assertions-only
    /// (see `SparseVec::from_raw_parts` sentinels for why `debug_assert!`
    /// invariants can't be tested under `--release`).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "col_ptr length must be ncols+1")]
    fn test_sentinel_from_raw_parts_rejects_wrong_col_ptr_length() {
        let _ = CscMatrix::from_raw_parts(1, 2, vec![0, 1], vec![0], vec![1.0]);
    }

    /// Sentinel: `col_ptr` must start at zero.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "col_ptr head must be zero")]
    fn test_sentinel_from_raw_parts_rejects_nonzero_col_ptr_head() {
        let _ = CscMatrix::from_raw_parts(1, 1, vec![1, 1], vec![], vec![]);
    }

    /// Sentinel: `col_ptr` tail must equal `nnz` (`values.len()`).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "col_ptr tail must equal nnz")]
    fn test_sentinel_from_raw_parts_rejects_wrong_col_ptr_tail() {
        let _ = CscMatrix::from_raw_parts(1, 1, vec![0, 2], vec![0], vec![1.0]);
    }

    /// Sentinel: `row_ind`/`values` length mismatch must be rejected.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "row_ind/values length mismatch")]
    fn test_sentinel_from_raw_parts_rejects_row_values_length_mismatch() {
        let _ = CscMatrix::from_raw_parts(1, 1, vec![0, 2], vec![0, 0], vec![1.0]);
    }

    /// Sentinel: `col_ptr` must be monotone non-decreasing (a decreasing
    /// entry would mean a column claims a negative nnz count).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "col_ptr must be monotone non-decreasing")]
    fn test_sentinel_from_raw_parts_rejects_non_monotone_col_ptr() {
        // ncols=3 so col_ptr has a genuine middle entry (length ncols+1=4);
        // head=0 and tail=2=nnz are both satisfied, isolating the dip at
        // index 1->2 (2 -> 1) as the only violated invariant.
        let _ = CscMatrix::from_raw_parts(2, 3, vec![0, 2, 1, 2], vec![0, 1], vec![1.0, 2.0]);
    }

    /// Sentinel: row indices within a single column must be strictly
    /// ascending (the `column()`/`get_column()` contract assumes this).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "row indices within each column must be strictly ascending")]
    fn test_sentinel_from_raw_parts_rejects_unsorted_rows_within_column() {
        let _ = CscMatrix::from_raw_parts(2, 1, vec![0, 2], vec![1, 0], vec![1.0, 2.0]);
    }

    /// Sentinel: an out-of-bounds row index (`>= nrows`) must be rejected.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "row index out of bounds for nrows=2")]
    fn test_sentinel_from_raw_parts_rejects_out_of_bounds_row() {
        let _ = CscMatrix::from_raw_parts(2, 1, vec![0, 1], vec![2], vec![1.0]);
    }

    /// A well-formed matrix must still construct cleanly (positive control:
    /// proves the sentinels above are exercising `from_raw_parts`, not some
    /// unconditional panic).
    #[test]
    fn test_from_raw_parts_accepts_well_formed_matrix() {
        let m = CscMatrix::from_raw_parts(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m.nnz(), 3);
    }
}
