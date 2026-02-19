/// CSC 形式の疎単位下三角行列
///
/// 対角要素は暗黙的に 1.0（格納しない）。
/// 列 j には行インデックス i > j の要素のみが存在する（下三角部分）。
///
/// LU 分解の L 因子として使用される。前進代入（forward substitution）と
/// その転置解法（L^T x = b）をサポートする。
#[derive(Debug, Clone)]
pub(crate) struct SparseLowerCSC {
    /// 列ポインタ配列（長さ: n + 1）
    pub col_ptr: Vec<usize>,
    /// 各非ゼロ要素の行インデックス（対角以下の行のみ）
    pub row_ind: Vec<usize>,
    /// 各非ゼロ要素の値（対角要素は含まない）
    pub values: Vec<f64>,
    /// 行列のサイズ（n×n）
    pub n: usize,
}

impl SparseLowerCSC {
    /// 前進代入: L * x = b を解く（インプレース）
    ///
    /// 対角要素が暗黙的に 1.0 の単位下三角行列に対して前進代入を行う。
    /// 解 x は `rhs` に上書きされる。
    ///
    /// # 引数
    /// - `rhs`: 入力時は右辺ベクトル b、終了時は解 x（インプレース更新）
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

    /// L^T * x = b を解く（インプレース）
    ///
    /// L^T は単位上三角行列となる。後退代入を用いて解く。
    /// 解 x は `rhs` に上書きされる。
    ///
    /// # 引数
    /// - `rhs`: 入力時は右辺ベクトル b、終了時は解 x（インプレース更新）
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

/// CSR 形式の疎上三角行列
///
/// 対角要素は `diag` 配列に別途格納する。
/// 行 i には列インデックス j > i の要素のみが `row_ptr`/`col_ind`/`values` に入る（対角除く上三角部分）。
///
/// LU 分解の U 因子として使用される。後退代入（backward substitution）と
/// その転置解法（U^T x = b）をサポートする。
#[derive(Debug, Clone)]
pub(crate) struct SparseUpperCSR {
    /// 行ポインタ配列（長さ: n + 1）。対角除く上三角要素を格納
    pub row_ptr: Vec<usize>,
    /// 各非ゼロ要素の列インデックス（j > i のもののみ）
    pub col_ind: Vec<usize>,
    /// 各非ゼロ要素の値（対角要素は含まない）
    pub values: Vec<f64>,
    /// 対角要素の値（長さ: n）
    pub diag: Vec<f64>,
    /// 行列のサイズ（n×n）
    pub n: usize,
}

impl SparseUpperCSR {
    /// 後退代入: U * x = b を解く（インプレース）
    ///
    /// 上三角行列に対して後退代入を行う。解 x は `rhs` に上書きされる。
    ///
    /// # 引数
    /// - `rhs`: 入力時は右辺ベクトル b、終了時は解 x（インプレース更新）
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

    /// U^T * x = b を解く（インプレース）
    ///
    /// U^T は下三角行列となる。前進代入を用いて解く。
    /// 解 x は `rhs` に上書きされる。
    ///
    /// # 引数
    /// - `rhs`: 入力時は右辺ベクトル b、終了時は解 x（インプレース更新）
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
