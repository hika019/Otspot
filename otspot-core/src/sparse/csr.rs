use super::compress::build_compressed_format;
use super::csc::CscMatrix;
use crate::error::SolverError;

/// 行圧縮形式（CSR: Compressed Sparse Row）の疎行列
///
/// 非ゼロ要素を行単位で格納する疎行列フォーマット。
/// 行ポインタ・列インデックス・値の3配列で表現される。
///
/// # フォーマット詳細
///
/// 行 `i` の非ゼロ要素は `values[row_ptr[i]..row_ptr[i+1]]` に格納され、
/// 対応する列インデックスは `col_ind[row_ptr[i]..row_ptr[i+1]]` に入る。
/// 各行の列インデックスは昇順にソートされている。
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// 行ポインタ配列（長さ: nrows + 1）
    /// `row_ptr[i]` は行 i の最初の非ゼロ要素の位置を示す
    pub row_ptr: Vec<usize>,
    /// 各非ゼロ要素の列インデックス
    pub col_ind: Vec<usize>,
    /// 各非ゼロ要素の値
    pub values: Vec<f64>,
    /// 行数
    pub nrows: usize,
    /// 列数
    pub ncols: usize,
}

impl CsrMatrix {
    /// 非ゼロ要素の総数を返す
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// COO（座標形式）のトリプレットから CSR 行列を構築する
    ///
    /// 同一 (row, col) への重複エントリは自動的に加算される。
    /// ゼロ近傍の結果値（絶対値 DROP_TOL 以下）は格納しない。
    ///
    /// # 引数
    /// - `rows`: 各エントリの行インデックス
    /// - `cols`: 各エントリの列インデックス
    /// - `vals`: 各エントリの値
    /// - `nrows`: 行列の行数
    /// - `ncols`: 行列の列数
    ///
    /// # エラー
    /// - `rows`、`cols`、`vals` の長さが異なる場合
    /// - 行/列インデックスが範囲外の場合
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, SolverError> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err(SolverError::DimensionMismatch { field: "triplet_arrays", expected: rows.len(), got: vals.len() });
        }
        // CSR: 主軸=行、副軸=列
        let (row_ptr, col_ind, values) =
            build_compressed_format(nrows, ncols, rows, cols, vals)?;
        Ok(Self { row_ptr, col_ind, values, nrows, ncols })
    }

    /// 行 i の非ゼロ要素を取得する
    ///
    /// 列インデックス配列と値配列のスライスを返す。両スライスの長さは等しく、
    /// 列インデックスは昇順にソートされている。
    ///
    /// # 引数
    /// - `i`: 取得する行インデックス（0-based）
    ///
    /// # 戻り値
    /// - `Ok((col_indices, values))`: 行 i の列インデックスと値のスライスペア
    /// - `Err`: `i` が範囲外の場合
    pub fn get_row(&self, i: usize) -> Result<(&[usize], &[f64]), SolverError> {
        if i >= self.nrows {
            return Err(SolverError::IndexOutOfBounds { context: "row", index: i, bound: self.nrows });
        }
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        Ok((&self.col_ind[start..end], &self.values[start..end]))
    }

    /// CSC 行列を CSR 行列に変換する
    ///
    /// 直接変換アルゴリズムを使用する。
    /// Pass 1: 各行の非ゼロ要素数を数え、prefix sum で row_ptr を構築する。
    /// Pass 2: 列を昇順に走査して col_ind/values を埋める。
    /// 列を昇順で処理するため、各行の col_ind は自動的にソート済みとなる。
    /// 計算量は O(nnz)（トリプレット経由の O(nnz log nnz) より高速）。
    ///
    /// # 引数
    /// - `csc`: 変換元の CSC 行列
    pub fn from_csc(csc: &CscMatrix) -> Self {
        let nnz = csc.nnz();
        let nrows = csc.nrows;
        let ncols = csc.ncols;

        // Pass 1: 各行の要素数をカウントし、prefix sum で row_ptr を構築する
        let mut row_ptr = vec![0usize; nrows + 1];
        for &r in &csc.row_ind {
            row_ptr[r + 1] += 1;
        }
        for i in 0..nrows {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Pass 2: 列を昇順に走査して col_ind/values を配置する
        // cur[i] = 行 i の次の書き込み位置
        let mut col_ind = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut cur = row_ptr[..nrows].to_vec();

        for j in 0..ncols {
            let start = csc.col_ptr[j];
            let end = csc.col_ptr[j + 1];
            for k in start..end {
                let r = csc.row_ind[k];
                let pos = cur[r];
                col_ind[pos] = j;
                values[pos] = csc.values[k];
                cur[r] += 1;
            }
        }

        Self {
            row_ptr,
            col_ind,
            values,
            nrows,
            ncols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
