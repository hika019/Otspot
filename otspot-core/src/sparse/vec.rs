use crate::tolerances::ZERO_TOL;

/// 疎ベクトル（インデックス・値のペアリスト、インデックスで昇順ソート済み）
///
/// ゼロでない要素のみをインデックスと値のペアで保持する。
/// `indices` は常に昇順にソートされており、二分探索による O(log n) アクセスが可能。
/// ゼロ近傍の値（絶対値が `ZERO_TOL` 以下）は自動的に除去される。
#[derive(Debug, Clone)]
pub struct SparseVec {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
    pub len: usize,
}

impl SparseVec {
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
}
