//! KKTシステムソルバー
//!
//! Active Set法の各反復で解くEQPサブ問題のKKTシステムを構築・因子分解する。
//! NC1修正済み: KKT行列 = [Q, A_W^T; A_W, 0]（Qそのもの、2Qではない）
//!
//! KKTシステム:
//! ```text
//! [Q    A_W^T] [d]   [-(Qx + c)]
//! [A_W  0    ] [λ] = [0        ]
//! ```

use crate::basis::lu;
use crate::error::SolverError;
use crate::sparse::CscMatrix;

/// KKTシステムソルバー
///
/// KKT行列を LU 分解して保持し、右辺ベクトルに対してソルブを行う。
pub(crate) struct KktSolver {
    lu: lu::LuFactorization,
    n: usize,
    w: usize,
}

impl KktSolver {
    /// KKT行列を構築してLU分解する
    ///
    /// # 引数
    /// - `q`: n×n 目的関数二次項行列
    /// - `a_active`: w×n 活性制約行列（元の制約行列の活性行のみ抽出済み）
    ///
    /// # エラー
    /// KKT行列が特異な場合はエラーを返す
    pub fn new(q: &CscMatrix, a_active: &CscMatrix) -> Result<Self, SolverError> {
        let n = q.ncols;
        let w = a_active.nrows;
        let size = n + w;

        let kkt = build_kkt_matrix(q, a_active, n, w)?;
        let basis: Vec<usize> = (0..size).collect();
        let lu = lu::LuFactorization::factorize(&kkt, &basis)?;

        Ok(KktSolver { lu, n, w })
    }

    /// KKTシステムを解いて探索方向 d とラグランジュ乗数 λ を返す
    ///
    /// rhs = [-(Qx + c); 0] を解いて [d; λ] を得る
    ///
    /// # 引数
    /// - `grad`: 勾配ベクトル Qx + c（長さ n）
    ///
    /// # 戻り値
    /// - `d`: 探索方向（長さ n）
    /// - `lambda`: ラグランジュ乗数（長さ w）
    pub fn solve(&self, grad: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
        let size = self.n + self.w;
        let mut rhs = vec![0.0f64; size];
        for i in 0..self.n {
            rhs[i] = -grad[i];
        }
        // rhs[n..] = 0 (equality constraints, already set)

        lu::solve_ftran(&self.lu, &mut rhs);

        let d = rhs[..self.n].to_vec();
        let lambda = rhs[self.n..].to_vec();
        Ok((d, lambda))
    }

    /// 活性集合変更後にKKT行列を再構築して再因子分解する
    #[allow(dead_code)]
    pub fn update(&mut self, q: &CscMatrix, a_active: &CscMatrix) -> Result<(), SolverError> {
        let n = q.ncols;
        let w = a_active.nrows;
        let size = n + w;

        let kkt = build_kkt_matrix(q, a_active, n, w)?;
        let basis: Vec<usize> = (0..size).collect();
        self.lu = lu::LuFactorization::factorize(&kkt, &basis)?;
        self.n = n;
        self.w = w;
        Ok(())
    }
}

/// KKT行列 K = [Q, A_W^T; A_W, 0] を CscMatrix として構築する
///
/// NC1修正: 左上ブロックは Q（2Qではない）
fn build_kkt_matrix(
    q: &CscMatrix,
    a_active: &CscMatrix,
    n: usize,
    w: usize,
) -> Result<CscMatrix, SolverError> {
    let size = n + w;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // 左上ブロック: Q (n×n)
    for col in 0..n {
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        for k in start..end {
            rows.push(q.row_ind[k]);
            cols.push(col);
            vals.push(q.values[k]);
        }
    }

    // 右上・左下ブロック: A_W^T と A_W（A_W は w×n）
    // A_W[i, j] = val → K[n+i, j] = val (左下) and K[j, n+i] = val (右上)
    for col_j in 0..n {
        let start = a_active.col_ptr[col_j];
        let end = a_active.col_ptr[col_j + 1];
        for k in start..end {
            let row_i = a_active.row_ind[k];
            let val = a_active.values[k];
            // 左下: A_W[row_i, col_j] → K[n+row_i, col_j]
            rows.push(n + row_i);
            cols.push(col_j);
            vals.push(val);
            // 右上: A_W^T[col_j, n+row_i] → K[col_j, n+row_i]
            rows.push(col_j);
            cols.push(n + row_i);
            vals.push(val);
        }
    }

    // 右下ブロック: 0 (w×w) — エントリなし

    CscMatrix::from_triplets(&rows, &cols, &vals, size, size)
}

/// 活性制約行列を構築する
///
/// 元の制約行列 a (m×n) から active_set に対応する行を抽出して
/// w×n の CscMatrix を返す。
pub(crate) fn extract_active_rows(
    a: &CscMatrix,
    active_set: &[usize],
) -> Result<CscMatrix, SolverError> {
    let n = a.ncols;
    let w = active_set.len();

    if w == 0 {
        return Ok(CscMatrix::new(0, n));
    }

    // active_set[k] → 行番号の逆引きマップ
    let mut row_to_active: Vec<Option<usize>> = vec![None; a.nrows];
    for (k, &row_idx) in active_set.iter().enumerate() {
        if row_idx >= a.nrows {
            return Err(SolverError::IndexOutOfBounds {
                context: "active_set_row",
                index: row_idx,
                bound: a.nrows,
            });
        }
        row_to_active[row_idx] = Some(k);
    }

    let mut col_ptrs: Vec<usize> = vec![0; n + 1];

    // 各列を走査してアクティブ行のエントリを抽出
    let mut entries: Vec<(usize, usize, f64)> = Vec::new(); // (col, active_row, val)
    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let orig_row = a.row_ind[k];
            if let Some(active_row) = row_to_active[orig_row] {
                entries.push((col, active_row, a.values[k]));
            }
        }
    }

    // CSC形式で構築
    let mut col_counts = vec![0usize; n];
    for &(col, _, _) in &entries {
        col_counts[col] += 1;
    }
    for col in 0..n {
        col_ptrs[col + 1] = col_ptrs[col] + col_counts[col];
    }

    let nnz = entries.len();
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    let mut pos = col_ptrs.clone();
    pos.pop();

    for (col, active_row, val) in entries {
        let idx = pos[col];
        row_ind[idx] = active_row;
        values[idx] = val;
        pos[col] += 1;
    }

    // 各列内の行インデックスをソート
    for col in 0..n {
        let start = col_ptrs[col];
        let end = col_ptrs[col + 1];
        let slice_ri = &mut row_ind[start..end];
        let slice_v = &mut values[start..end];
        // インデックスと値を共に昇順ソート
        let mut pairs: Vec<(usize, f64)> =
            slice_ri.iter().zip(slice_v.iter()).map(|(&r, &v)| (r, v)).collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        for (i, (r, v)) in pairs.into_iter().enumerate() {
            slice_ri[i] = r;
            slice_v[i] = v;
        }
    }

    Ok(CscMatrix { col_ptr: col_ptrs, row_ind, values, nrows: w, ncols: n })
}

/// 勾配 Qx + c を計算する
pub(crate) fn compute_gradient(q: &CscMatrix, x: &[f64], c: &[f64]) -> Vec<f64> {
    let mut grad = c.to_vec();
    // Qx を加算（疎行列×ベクトル）
    for (col, &xj) in x.iter().enumerate() {
        if xj.abs() < 1e-15 {
            continue;
        }
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        for k in start..end {
            grad[q.row_ind[k]] += q.values[k] * xj;
        }
    }
    grad
}

/// 目的関数値 1/2 x^T Q x + c^T x を計算する
pub(crate) fn compute_objective(q: &CscMatrix, x: &[f64], c: &[f64]) -> f64 {
    let mut qx_dot = 0.0f64;
    // x^T Q x = sum over (i,j): x[i] * Q[i,j] * x[j]
    for (col, &xj) in x.iter().enumerate() {
        if xj.abs() < 1e-15 {
            continue;
        }
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        for k in start..end {
            qx_dot += x[q.row_ind[k]] * q.values[k] * xj;
        }
    }
    let cx: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
    0.5 * qx_dot + cx
}
