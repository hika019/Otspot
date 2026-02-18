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
    /// 指定した論理長の空疎ベクトルを生成する
    ///
    /// 非ゼロ要素は含まない（すべてゼロ）状態で初期化される。
    ///
    /// # 引数
    /// - `len`: ベクトルの論理的な長さ（次元数）
    pub fn new(len: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            len,
        }
    }

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

    /// 指定インデックスの値を取得する
    ///
    /// インデックスが非ゼロ要素として存在しない場合は `0.0` を返す。
    /// 内部では二分探索を使用するため O(log n) で動作する。
    ///
    /// # 引数
    /// - `idx`: 取得するインデックス
    pub fn get(&self, idx: usize) -> f64 {
        match self.indices.binary_search(&idx) {
            Ok(pos) => self.values[pos],
            Err(_) => 0.0,
        }
    }

    /// 指定インデックスに値をセットする
    ///
    /// `val` の絶対値が ZERO_TOL 以下の場合、そのインデックスを非ゼロリストから削除する
    /// （ゼロとみなす）。既存のエントリがない場合は挿入し、ある場合は上書きする。
    /// ソート順を維持するため、挿入位置は二分探索で決定する。
    ///
    /// # 引数
    /// - `idx`: セットするインデックス
    /// - `val`: セットする値（ZERO_TOL 以下なら削除）
    pub fn set(&mut self, idx: usize, val: f64) {
        match self.indices.binary_search(&idx) {
            Ok(pos) => {
                if val.abs() <= ZERO_TOL {
                    self.indices.remove(pos);
                    self.values.remove(pos);
                } else {
                    self.values[pos] = val;
                }
            }
            Err(pos) => {
                if val.abs() > ZERO_TOL {
                    self.indices.insert(pos, idx);
                    self.values.insert(pos, val);
                }
            }
        }
    }

    /// AXPY 演算: `self += alpha * other`
    ///
    /// 内部では一旦密ベクトルに展開して演算し、結果を再び疎ベクトルに変換する。
    /// 正確性を優先した実装（疎・疎のマージより若干コストが高い）。
    ///
    /// # 引数
    /// - `alpha`: スカラー倍率
    /// - `other`: 加算する疎ベクトル
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

    /// 別の疎ベクトルとの内積を計算する
    ///
    /// 両ベクトルのインデックスリストをマージソート的に走査し、
    /// 一致するインデックスの積を加算する。計算量は O(nnz_a + nnz_b)。
    ///
    /// # 引数
    /// - `other`: 内積を取る相手の疎ベクトル
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

    /// 密ベクトルとの内積を計算する
    ///
    /// 疎ベクトルの非ゼロ要素のインデックスのみを参照するため、
    /// 密ベクトルとの積でも O(nnz) で動作する。
    ///
    /// # 引数
    /// - `dense`: 内積を取る相手の密ベクトル（スライス）
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
}
