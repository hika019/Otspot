//! 疎行列・疎ベクトルデータ構造（CSC/CSR フォーマット）
//!
//! このモジュールは数値線形代数で広く使われる疎行列フォーマットを提供する。
//! ゼロ要素を省略して格納することで、大規模疎行列のメモリ効率と演算効率を確保する。
//!
//! # 提供するデータ構造
//!
//! - [`SparseVec`]: インデックス・値ペアのリスト形式による疎ベクトル
//! - [`CscMatrix`]: 列圧縮形式（CSC: Compressed Sparse Column）疎行列
//! - [`CsrMatrix`]: 行圧縮形式（CSR: Compressed Sparse Row）疎行列
//! - `SparseLowerCSC`: CSC形式の疎単位下三角行列（LU分解の L 因子用）
//! - `SparseUpperCSR`: CSR形式の疎上三角行列（LU分解の U 因子用）
//!
//! # 疎行列フォーマットの概要
//!
//! **CSC（列圧縮形式）**: 列ポインタ配列 `col_ptr`、行インデックス配列 `row_ind`、
//! 値配列 `values` の3配列で表現する。列単位のアクセスや列ベクトル演算が高速で、
//! LU 分解などの直接法ソルバで広く使われる。
//!
//! **CSR（行圧縮形式）**: 行ポインタ配列 `row_ptr`、列インデックス配列 `col_ind`、
//! 値配列 `values` の3配列で表現する。行単位のアクセスや行列ベクトル積が高速で、
//! 共役勾配法などの反復法ソルバでよく使われる。
//!
//! **COO（座標形式、入力専用）**: 行・列・値のトリプレットで非ゼロ要素を列挙する形式。
//! `from_triplets` メソッドを通じて CSC/CSR へ変換できる。
//! 同一 (行, 列) への重複エントリは自動的に加算される。

use std::collections::HashMap;
use crate::tolerances::*;

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
    /// 列ポインタ配列（長さ: ncols + 1）
    /// `col_ptr[j]` は列 j の最初の非ゼロ要素の位置を示す
    pub col_ptr: Vec<usize>,
    /// 各非ゼロ要素の行インデックス
    pub row_ind: Vec<usize>,
    /// 各非ゼロ要素の値
    pub values: Vec<f64>,
    /// 行数
    pub nrows: usize,
    /// 列数
    pub ncols: usize,
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

    /// 非ゼロ要素の総数を返す
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// COO（座標形式）のトリプレットから CSC 行列を構築する
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
    ) -> Result<Self, String> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err("Triplet arrays must have same length".to_string());
        }
        // CSC: 主軸=列、副軸=行
        let (col_ptr, row_ind, values) =
            build_compressed_format(ncols, nrows, cols, rows, vals)?;
        Ok(Self { col_ptr, row_ind, values, nrows, ncols })
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

    /// 行列ベクトル積を計算する: y = A * x
    ///
    /// CSC 形式の列走査を利用して O(nnz) で計算する。
    ///
    /// # 引数
    /// - `x`: 入力ベクトル（長さ: ncols）
    ///
    /// # 戻り値
    /// - `Ok(y)`: 結果ベクトル（長さ: nrows）
    /// - `Err`: `x` の長さが `ncols` と一致しない場合
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

    /// 列 j の非ゼロ要素を取得する
    ///
    /// 行インデックス配列と値配列のスライスを返す。両スライスの長さは等しく、
    /// 行インデックスは昇順にソートされている。
    ///
    /// # 引数
    /// - `j`: 取得する列インデックス（0-based）
    ///
    /// # 戻り値
    /// - `Ok((row_indices, values))`: 列 j の行インデックスと値のスライスペア
    /// - `Err`: `j` が範囲外の場合
    pub fn get_column(&self, j: usize) -> Result<(&[usize], &[f64]), String> {
        if j >= self.ncols {
            return Err(format!("Column index {} out of bounds (ncols={})", j, self.ncols));
        }
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        Ok((&self.row_ind[start..end], &self.values[start..end]))
    }

    /// n×n 単位行列を CSC 形式で生成する
    ///
    /// 対角要素が 1.0 で、非対角要素がゼロの正方行列を返す。
    ///
    /// # 引数
    /// - `n`: 行列のサイズ（n×n）
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
    ) -> Result<Self, String> {
        if rows.len() != cols.len() || rows.len() != vals.len() {
            return Err("Triplet arrays must have same length".to_string());
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
    pub fn get_row(&self, i: usize) -> Result<(&[usize], &[f64]), String> {
        if i >= self.nrows {
            return Err(format!("Row index {} out of bounds (nrows={})", i, self.nrows));
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

/// COO トリプレットを圧縮形式（CSC/CSR 共通）に変換する内部ヘルパー
///
/// `major_indices` が主軸（CSC では列、CSR では行）、`minor_indices` が副軸。
/// 重複エントリの加算・境界チェック・DROP_TOL フィルタリング・ソートを行い、
/// `(major_ptr, minor_ind, values)` を返す。
fn build_compressed_format(
    n_major: usize,
    n_minor: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    vals: &[f64],
) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>), String> {
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..major_indices.len() {
        let maj = major_indices[i];
        let min = minor_indices[i];
        if maj >= n_major {
            return Err(format!("Major index {} out of bounds (n_major={})", maj, n_major));
        }
        if min >= n_minor {
            return Err(format!("Minor index {} out of bounds (n_minor={})", min, n_minor));
        }
        *map.entry((maj, min)).or_insert(0.0) += vals[i];
    }

    let mut triplets: Vec<(usize, usize, f64)> = map
        .into_iter()
        .filter(|(_, v)| v.abs() > DROP_TOL)
        .map(|((maj, min), v)| (maj, min, v))
        .collect();
    triplets.sort_by_key(|&(maj, min, _)| (maj, min));

    let mut major_ptr = vec![0; n_major + 1];
    let mut minor_ind = Vec::new();
    let mut values = Vec::new();

    let mut current_major = 0;
    for (maj, min, v) in triplets {
        while current_major < maj {
            current_major += 1;
            major_ptr[current_major] = minor_ind.len();
        }
        minor_ind.push(min);
        values.push(v);
    }
    while current_major < n_major {
        current_major += 1;
        major_ptr[current_major] = minor_ind.len();
    }

    Ok((major_ptr, minor_ind, values))
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
