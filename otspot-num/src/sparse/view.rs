//! Sparse-matrix contracts.
//!
//! The trait is intentionally read-only.  Algorithms consume a stable CSC view
//! while ownership and construction stay with the concrete matrix crate.  This
//! is the seam used to migrate the legacy `otspot_core::sparse::CscMatrix`
//! without copying matrices or creating a core → numeric → core dependency.

use crate::NumericError;

/// Read-only compressed-sparse-column matrix.
pub trait CscMatrixView {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn col_ptr(&self) -> &[usize];
    fn row_ind(&self) -> &[usize];
    fn values(&self) -> &[f64];

    /// Return one valid column.  Out-of-range access is a programming error:
    /// callers with untrusted indices must validate before entering kernels.
    fn column(&self, column: usize) -> (&[usize], &[f64]) {
        assert!(
            column < self.ncols(),
            "column {column} out of bounds (ncols={})",
            self.ncols()
        );
        let start = self.col_ptr()[column];
        let end = self.col_ptr()[column + 1];
        (&self.row_ind()[start..end], &self.values()[start..end])
    }
}

/// Validate CSC structural invariants and coefficient finiteness.
pub fn validate_csc<M: CscMatrixView + ?Sized>(matrix: &M) -> Result<(), NumericError> {
    let ptr = matrix.col_ptr();
    let rows = matrix.row_ind();
    let values = matrix.values();

    if ptr.len() != matrix.ncols() + 1 {
        return Err(NumericError::DimensionMismatch {
            field: "col_ptr",
            expected: matrix.ncols() + 1,
            got: ptr.len(),
        });
    }
    if rows.len() != values.len() {
        return Err(NumericError::DimensionMismatch {
            field: "row_ind/values",
            expected: rows.len(),
            got: values.len(),
        });
    }
    if ptr.first().copied() != Some(0) {
        return Err(NumericError::InvalidSparseStructure {
            message: "col_ptr[0] must be zero",
        });
    }
    if ptr.last().copied() != Some(values.len()) {
        return Err(NumericError::InvalidSparseStructure {
            message: "col_ptr[ncols] must equal nnz",
        });
    }

    for (j, window) in ptr.windows(2).enumerate() {
        let start = window[0];
        let end = window[1];
        if start > end || end > values.len() {
            return Err(NumericError::InvalidSparseStructure {
                message: "col_ptr must be monotone and within nnz",
            });
        }
        let mut previous = None;
        for position in start..end {
            let row = rows[position];
            if row >= matrix.nrows() {
                return Err(NumericError::IndexOutOfBounds {
                    context: "row",
                    index: row,
                    bound: matrix.nrows(),
                });
            }
            if previous.is_some_and(|p| p >= row) {
                let _ = j;
                return Err(NumericError::InvalidSparseStructure {
                    message: "row indices in each column must be strictly increasing",
                });
            }
            previous = Some(row);
            if !values[position].is_finite() {
                return Err(NumericError::NonFinite {
                    field: "values",
                    index: position,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Matrix {
        rows: usize,
        cols: usize,
        ptr: Vec<usize>,
        index: Vec<usize>,
        values: Vec<f64>,
    }

    impl CscMatrixView for Matrix {
        fn nrows(&self) -> usize {
            self.rows
        }
        fn ncols(&self) -> usize {
            self.cols
        }
        fn col_ptr(&self) -> &[usize] {
            &self.ptr
        }
        fn row_ind(&self) -> &[usize] {
            &self.index
        }
        fn values(&self) -> &[f64] {
            &self.values
        }
    }

    #[test]
    fn validates_canonical_csc() {
        let m = Matrix {
            rows: 2,
            cols: 2,
            ptr: vec![0, 1, 2],
            index: vec![0, 1],
            values: vec![1.0, 2.0],
        };
        assert_eq!(validate_csc(&m), Ok(()));
        assert_eq!(m.column(1), (&[1][..], &[2.0][..]));
    }

    #[test]
    fn rejects_unsorted_or_duplicate_rows() {
        let m = Matrix {
            rows: 2,
            cols: 1,
            ptr: vec![0, 2],
            index: vec![1, 1],
            values: vec![1.0, 2.0],
        };
        assert!(matches!(
            validate_csc(&m),
            Err(NumericError::InvalidSparseStructure { .. })
        ));
    }
}
