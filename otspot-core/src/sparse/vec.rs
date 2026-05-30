use crate::tolerances::ZERO_TOL;

/// 疎ベクトル（インデックス・値のペアリスト、インデックスで昇順ソート済み）
///
/// ゼロでない要素のみをインデックスと値のペアで保持する。
/// `indices` は常に昇順にソートされており、二分探索による O(log n) アクセスが可能。
/// ゼロ近傍の値（絶対値が `ZERO_TOL` 以下）は自動的に除去される。
#[derive(Debug, Clone)]
pub struct SparseVec {
    /// 非ゼロ要素のインデックス（昇順ソート済み）
    pub indices: Vec<usize>,
    /// 非ゼロ要素の値（`indices` と同じ順序）
    pub values: Vec<f64>,
    /// 論理的な長さ（ゼロ要素を含む全体の次元数）
    pub len: usize, // logical length
}

impl SparseVec {
    /// 密ベクトルから疎ベクトルを生成する
    ///
    /// 絶対値が ZERO_TOL（1e-12）を超える要素のみを保持し、残りは捨てる。
    /// インデックスは元の配列の位置順（昇順）で格納される。
    ///
    /// # 引数
    /// - `dense`: 変換元の密ベクトル（スライス）
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

    /// 疎ベクトルを密ベクトルに変換する
    ///
    /// 非ゼロ要素を対応するインデックスに配置し、残りはゼロで埋める。
    /// 返却ベクトルの長さは `self.len` と等しい。
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.len];
        for (k, &idx) in self.indices.iter().enumerate() {
            dense[idx] = self.values[k];
        }
        dense
    }

    /// 事前確保済みバッファに密ベクトルを書き込む
    ///
    /// `buf` を一旦ゼロクリアしてから非ゼロ要素を書き込む。
    /// ヒープ割り当てを行わないため、反復ループ内での再利用に適する。
    ///
    /// # 引数
    /// - `buf`: 書き込み先バッファ（長さ >= `self.len` であること）
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
