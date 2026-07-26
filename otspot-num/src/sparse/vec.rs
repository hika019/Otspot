use crate::ZERO_TOL;

/// 疎ベクトル（インデックス・値のペアリスト、インデックスで昇順ソート済み）
///
/// ゼロでない要素のみをインデックスと値のペアで保持する。
/// `indices` は常に昇順にソートされており、二分探索による O(log n) アクセスが可能。
/// ゼロ近傍の値（絶対値が `ZERO_TOL` 以下）は自動的に除去される。
#[derive(Debug, Clone)]
pub struct SparseVec {
    pub(crate) indices: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) len: usize,
}

impl SparseVec {
    /// Builds a vector from already-sorted `(index, value)` arrays (e.g. a CSC
    /// column extracted via [`crate::sparse::CscMatrix::column`]).
    ///
    /// The encapsulated replacement for direct struct-literal construction.
    /// `debug_assert`s enforce the invariants (matching length, ascending
    /// sorted indices, all indices in bounds) in test/dev builds; release
    /// builds trust the caller (matching the previous struct-literal usage).
    pub fn from_raw_parts(indices: Vec<usize>, values: Vec<f64>, len: usize) -> Self {
        debug_assert_eq!(
            indices.len(),
            values.len(),
            "indices/values length mismatch"
        );
        debug_assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "indices must be strictly ascending"
        );
        debug_assert!(
            indices.last().is_none_or(|&i| i < len),
            "index out of bounds for len={len}"
        );
        Self {
            indices,
            values,
            len,
        }
    }

    /// Creates a `SparseVec` from a dense slice, dropping entries with `|v| ≤ ZERO_TOL`.
    pub fn from_dense(dense: &[f64]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > ZERO_TOL {
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

    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.len];
        for (k, &idx) in self.indices.iter().enumerate() {
            dense[idx] = self.values[k];
        }
        dense
    }

    /// Writes to a pre-allocated buffer (zero-fills first). Avoids heap allocation in hot loops.
    pub fn to_dense_into(&self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = 0.0;
        }
        for (k, &idx) in self.indices.iter().enumerate() {
            buf[idx] = self.values[k];
        }
    }

    /// Non-zero entry indices, ascending order.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Non-zero entry values, aligned with [`Self::indices`].
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Dense dimension of the vector — i.e. the length it would have if
    /// expanded via [`Self::to_dense`]. This is *not* the number of stored
    /// non-zero entries (use `indices().len()` for that); mirrors
    /// `CscMatrix::nrows`/`ncols`, not the `Vec::len` convention.
    #[inline]
    pub fn dim(&self) -> usize {
        self.len
    }

    /// Clears all stored non-zero entries in place, leaving [`Self::dim`] unchanged.
    #[inline]
    pub fn clear(&mut self) {
        self.indices.clear();
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_from_raw_parts_matches_from_dense() {
        // Independent oracle: build the same non-trivial vector two ways
        // (from_dense's own scan vs. the raw-parts constructor with the
        // already-known sparse pattern) and require identical output.
        let dense = vec![1.0, 0.0, 0.0, 3.5, 0.0, -2.0];
        let via_dense = SparseVec::from_dense(&dense);
        let via_raw = SparseVec::from_raw_parts(vec![0, 3, 5], vec![1.0, 3.5, -2.0], 6);
        assert_eq!(via_raw.to_dense(), via_dense.to_dense());
        assert_eq!(via_raw.to_dense(), dense);
    }

    #[test]
    fn test_from_raw_parts_empty() {
        let sv = SparseVec::from_raw_parts(vec![], vec![], 4);
        assert_eq!(sv.to_dense(), vec![0.0; 4]);
    }

    /// Sentinel: a length mismatch between `indices` and `values` must be
    /// rejected. Removing the `debug_assert_eq!` makes this test fail to
    /// panic (no-op fail) under the default (debug-assertions-on) profile.
    /// `debug_assert!` compiles out under `--release` (debug-assertions off),
    /// so the invariant genuinely cannot fire there; gate the test itself on
    /// `cfg(debug_assertions)` rather than asserting a panic that can't happen.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "indices/values length mismatch")]
    fn test_sentinel_from_raw_parts_rejects_length_mismatch() {
        let _ = SparseVec::from_raw_parts(vec![0, 1], vec![1.0], 3);
    }

    /// Sentinel: unsorted (or duplicate) indices must be rejected — the
    /// binary-search contract documented on the type requires strictly
    /// ascending order. debug-assertions-only, see above.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "indices must be strictly ascending")]
    fn test_sentinel_from_raw_parts_rejects_unsorted_indices() {
        let _ = SparseVec::from_raw_parts(vec![2, 1], vec![1.0, 2.0], 3);
    }

    /// Sentinel: an out-of-bounds index (`>= len`) must be rejected.
    /// debug-assertions-only, see above.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "index out of bounds for len=2")]
    fn test_sentinel_from_raw_parts_rejects_out_of_bounds_index() {
        let _ = SparseVec::from_raw_parts(vec![0, 2], vec![1.0, 2.0], 2);
    }

    /// `indices()`/`values()`/`dim()` must agree with the fields they expose.
    /// Independent oracle: re-derive `dim()` from `to_dense().len()` (a path
    /// that never touches the `len` field directly) rather than comparing the
    /// accessor against the field it wraps.
    #[test]
    fn test_accessors_match_dense_expansion() {
        let dense = vec![1.0, 0.0, 0.0, 3.5, 0.0, -2.0];
        let sv = SparseVec::from_dense(&dense);
        assert_eq!(sv.indices(), &[0, 3, 5]);
        assert_eq!(sv.values(), &[1.0, 3.5, -2.0]);
        assert_eq!(sv.dim(), sv.to_dense().len());
        assert_eq!(sv.dim(), dense.len());
    }

    /// Sentinel: `clear()` must drop every stored non-zero (indices and
    /// values both empty) while leaving `dim()` — the logical dimension —
    /// unchanged. Reverting `clear()` to a no-op leaves `indices()`
    /// non-empty, failing the first assertion.
    #[test]
    fn test_sentinel_clear_empties_entries_keeps_dim() {
        let mut sv = SparseVec::from_dense(&[1.0, 0.0, 2.0, 0.0, 3.0]);
        assert!(!sv.indices().is_empty());
        sv.clear();
        assert!(sv.indices().is_empty(), "clear() must drop indices");
        assert!(sv.values().is_empty(), "clear() must drop values");
        assert_eq!(sv.dim(), 5, "clear() must not change dim()");
        assert_eq!(sv.to_dense(), vec![0.0; 5]);
    }
}
