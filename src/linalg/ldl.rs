//! QDLDL疎LDL^T分解（対称正定値行列）
//!
//! OSQPのQDLDL実装（C言語版）を参考にしたRust移植。
//! 入力行列は上三角のみCSC形式で与える。

use crate::sparse::CscMatrix;
use std::time::Instant;

/// LDL分解エラー
#[derive(Debug)]
pub enum LdlError {
    /// 対角要素がゼロまたは負（特異行列または不定行値）
    SingularOrIndefinite,
    /// deadline を超過した（タイムアウト）
    DeadlineExceeded,
}

/// LDL^T分解の結果
///
/// A = L * D * L^T
/// - L: 単位下三角行列（対角は1、非ゼロのみCSCで格納）
/// - D: 対角行列（Vecとして格納）
/// - Dinv: D^{-1}の対角要素
#[allow(non_snake_case)]
pub struct LdlFactorization {
    pub L: CscMatrix,
    pub D: Vec<f64>,
    pub Dinv: Vec<f64>,
}

impl LdlFactorization {
    /// LDL^T x = b を解く
    ///
    /// 前方代入(L) → 対角スケール(D) → 後方代入(L^T) の3ステップで解く。
    ///
    /// # 引数
    /// - `rhs`: 右辺ベクトル（長さ n）
    /// - `sol`: 解ベクトルの出力先（長さ n）
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.D.len();
        assert_eq!(rhs.len(), n, "rhs length mismatch");
        assert_eq!(sol.len(), n, "sol length mismatch");

        let mut y = rhs.to_vec();

        // 前方代入: L y = rhs (単位下三角、列指向)
        // 列jを処理: y[i] -= L[i,j] * y[j]  (i > j)
        for j in 0..n {
            let start = self.L.col_ptr[j];
            let end = self.L.col_ptr[j + 1];
            for k in start..end {
                let i = self.L.row_ind[k]; // i > j
                y[i] -= self.L.values[k] * y[j];
            }
        }

        // 対角スケール: y = D^{-1} * y
        for (i, y_i) in y.iter_mut().enumerate() {
            *y_i *= self.Dinv[i];
        }

        // 後方代入: L^T x = y (列指向)
        // 列jを逆順処理: y[j] -= L[i,j] * y[i]  (i > j)
        for j in (0..n).rev() {
            let start = self.L.col_ptr[j];
            let end = self.L.col_ptr[j + 1];
            for k in start..end {
                let i = self.L.row_ind[k]; // i > j
                y[j] -= self.L.values[k] * y[i];
            }
        }

        sol.copy_from_slice(&y);
    }
}

/// 対称正定値疎行列のLDL^T分解を実行する
///
/// # 引数
/// - `mat`: 上三角のみCSC形式の対称行列（row_ind[k] <= j for all k in col j）
///
/// # 戻り値
/// - `Ok(LdlFactorization)`: 分解成功
/// - `Err(LdlError::SingularOrIndefinite)`: 対角要素がゼロまたは負
///
/// # アルゴリズム
/// QDLDLアルゴリズム（自然順序、AMD不使用）。
/// 密ワークスペース y[0..n] を使い列jを順次処理する。
pub fn factorize(mat: &CscMatrix) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    assert_eq!(n, mat.ncols, "Matrix must be square");

    let mut d_vec = vec![0.0f64; n];
    let mut dinv = vec![0.0f64; n];

    // 密ワークスペース: y[k] = 列jの処理中の累積値
    // y[j] = D[j]の累積（対角）
    // y[k] (k < j) = D[k] * L[j,k] の累積（消去で使用）
    let mut y = vec![0.0f64; n];

    // L_cols[k] = 列kの非ゼロエントリ: (行インデックスj, 値L[j,k])  j > k 昇順
    let mut l_cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

    for j in 0..n {
        // 行列Aの上三角列jを y に展開
        for idx in mat.col_ptr[j]..mat.col_ptr[j + 1] {
            let i = mat.row_ind[idx];
            let v = mat.values[idx];
            if i == j {
                y[j] = v; // 対角要素
            } else if i < j {
                // 上三角要素 A[i,j] = A[j,i]（対称）
                // → 行jの消去に使う初期値
                y[i] = v;
            }
            // i > j は上三角違反のため無視
        }

        // k = 0..j-1 を昇順に処理して列jのL, Dを計算
        for k in 0..j {
            if y[k] == 0.0 {
                continue; // 疎: スキップ
            }

            // L[j,k] = y[k] / D[k]
            let l_jk = y[k] * dinv[k];

            // D[j] -= L[j,k]^2 * D[k] = l_jk * y[k]
            y[j] -= l_jk * y[k];

            // 列kのLエントリを記録
            l_cols[k].push((j, l_jk));

            // フィルイン更新: 列kの既存エントリ L[m,k] で m < j
            // y[m] -= L[m,k] * y[k]
            // （今push した (j, l_jk) は末尾にあり m=j>=j でbreakされる）
            for &(m, l_mk) in &l_cols[k] {
                if m >= j {
                    break; // l_cols[k] は昇順: これ以降も m >= j
                }
                y[m] -= l_mk * y[k];
            }
        }

        // D[j] が確定
        d_vec[j] = y[j];

        // ゼロ・負・NaNのチェック
        if !d_vec[j].is_finite() || d_vec[j] <= 0.0 {
            // ワークスペースをクリアして返す
            y[..=j].fill(0.0);
            return Err(LdlError::SingularOrIndefinite);
        }
        dinv[j] = 1.0 / d_vec[j];

        // ワークスペースをクリア（次の j のために）
        y[..=j].fill(0.0);
    }

    // l_cols を CscMatrix に変換
    let nnz: usize = l_cols.iter().map(|c| c.len()).sum();
    let mut col_ptr = vec![0usize; n + 1];
    for k in 0..n {
        col_ptr[k + 1] = col_ptr[k] + l_cols[k].len();
    }
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for k in 0..n {
        let start = col_ptr[k];
        for (idx, &(row, val)) in l_cols[k].iter().enumerate() {
            row_ind[start + idx] = row;
            values[start + idx] = val;
        }
    }

    let l_mat = CscMatrix { col_ptr, row_ind, values, nrows: n, ncols: n };

    Ok(LdlFactorization { L: l_mat, D: d_vec, Dinv: dinv })
}

/// deadline 付き LDL^T 分解。外側ループ 1000 列ごとに deadline を確認し、
/// 超過した場合は `Err(LdlError::DeadlineExceeded)` を返す。
///
/// # cmd_171: timeout audit fix
/// n >= 10_000 の LDL 因子化は秒単位を要しうる。T1/T2 チェックは呼び出し元で
/// 実施されるが、factorize 自体が 34 秒ハングした前例 (cmd_170) があるため
/// 関数内部でも deadline を確認する。
pub fn factorize_with_deadline(mat: &CscMatrix, deadline: Option<Instant>) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    assert_eq!(n, mat.ncols, "Matrix must be square");

    let mut d_vec = vec![0.0f64; n];
    let mut dinv = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    let mut l_cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

    for j in 0..n {
        // cmd_171: timeout audit fix — 1000列ごとにdeadlineチェック
        if j % 1000 == 0 {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Err(LdlError::DeadlineExceeded);
                }
            }
        }

        for idx in mat.col_ptr[j]..mat.col_ptr[j + 1] {
            let i = mat.row_ind[idx];
            let v = mat.values[idx];
            if i == j {
                y[j] = v;
            } else if i < j {
                y[i] = v;
            }
        }

        for k in 0..j {
            if y[k] == 0.0 {
                continue;
            }
            let l_jk = y[k] * dinv[k];
            y[j] -= l_jk * y[k];
            l_cols[k].push((j, l_jk));
            for &(m, l_mk) in &l_cols[k] {
                if m >= j {
                    break;
                }
                y[m] -= l_mk * y[k];
            }
        }

        d_vec[j] = y[j];
        if !d_vec[j].is_finite() || d_vec[j] <= 0.0 {
            y[..=j].fill(0.0);
            return Err(LdlError::SingularOrIndefinite);
        }
        dinv[j] = 1.0 / d_vec[j];
        y[..=j].fill(0.0);
    }

    let nnz: usize = l_cols.iter().map(|c| c.len()).sum();
    let mut col_ptr = vec![0usize; n + 1];
    for k in 0..n {
        col_ptr[k + 1] = col_ptr[k] + l_cols[k].len();
    }
    let mut row_ind = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for k in 0..n {
        let start = col_ptr[k];
        for (idx, &(row, val)) in l_cols[k].iter().enumerate() {
            row_ind[start + idx] = row;
            values[start + idx] = val;
        }
    }

    let l_mat = CscMatrix { col_ptr, row_ind, values, nrows: n, ncols: n };
    Ok(LdlFactorization { L: l_mat, D: d_vec, Dinv: dinv })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 上三角CSC行列をCOOから構築するヘルパー（テスト用）
    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        // entriesは (row, col, val) で row <= col のみ
        // 列ごとにグループ化してCSC構築
        let mut cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(row, col, val) in entries {
            assert!(row <= col, "upper triangle only");
            cols[col].push((row, val));
        }
        // 各列内をrow昇順にソート
        for c in cols.iter_mut() {
            c.sort_by_key(|&(r, _)| r);
        }
        let nnz: usize = cols.iter().map(|c| c.len()).sum();
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + cols[j].len();
        }
        let mut row_ind = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        for j in 0..n {
            let start = col_ptr[j];
            for (idx, &(row, val)) in cols[j].iter().enumerate() {
                row_ind[start + idx] = row;
                values[start + idx] = val;
            }
        }
        CscMatrix { col_ptr, row_ind, values, nrows: n, ncols: n }
    }

    /// L D L^T を密行列として復元し、元のAと比較する
    fn reconstruct_ldlt(fac: &LdlFactorization, n: usize) -> Vec<Vec<f64>> {
        // L（単位下三角）を密行列に展開
        let mut l_dense = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            l_dense[i][i] = 1.0; // 対角 = 1
        }
        for k in 0..n {
            let start = fac.L.col_ptr[k];
            let end = fac.L.col_ptr[k + 1];
            for idx in start..end {
                let i = fac.L.row_ind[idx];
                l_dense[i][k] = fac.L.values[idx];
            }
        }

        // L * D * L^T を計算
        let mut a_rec = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l_dense[i][k] * fac.D[k] * l_dense[j][k];
                }
                a_rec[i][j] = sum;
            }
        }
        a_rec
    }

    #[test]
    fn test_ldl_3x3() {
        // A = [[4,1,0],[1,3,2],[0,2,5]]
        let mat = upper_tri_csc(3, &[
            (0, 0, 4.0),
            (0, 1, 1.0), (1, 1, 3.0),
            (1, 2, 2.0), (2, 2, 5.0),
        ]);

        let fac = factorize(&mat).expect("factorize failed");

        // D の正当性確認
        let eps = 1e-10;
        assert!((fac.D[0] - 4.0).abs() < eps, "D[0]={}", fac.D[0]);
        assert!((fac.D[1] - 11.0 / 4.0).abs() < eps, "D[1]={}", fac.D[1]);
        assert!((fac.D[2] - 39.0 / 11.0).abs() < eps, "D[2]={}", fac.D[2]);

        // L D L^T = A を確認
        let a_rec = reconstruct_ldlt(&fac, 3);
        let a_orig = [[4.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 2.0, 5.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((a_rec[i][j] - a_orig[i][j]).abs() < eps,
                    "A[{i},{j}]: expected {}, got {}", a_orig[i][j], a_rec[i][j]);
            }
        }

        // Ax = b を解いてresidual確認
        let b = [1.0f64, 2.0, 3.0];
        let mut x = [0.0f64; 3];
        fac.solve(&b, &mut x);

        // residual = |A*x - b|
        let ax0 = 4.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2];
        let ax2 = 2.0 * x[1] + 5.0 * x[2];
        assert!((ax0 - b[0]).abs() < 1e-10, "residual[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < 1e-10, "residual[1]={}", (ax1 - b[1]).abs());
        assert!((ax2 - b[2]).abs() < 1e-10, "residual[2]={}", (ax2 - b[2]).abs());
    }

    #[test]
    fn test_ldl_singular() {
        // A = [[1,0],[0,0]] → 特異行列
        // 上三角: 対角(0,0)=1のみ。(1,1)=0は非零でないので格納しない
        let mat = CscMatrix {
            col_ptr: vec![0, 1, 1], // col0: 1エントリ, col1: 0エントリ
            row_ind: vec![0],
            values: vec![1.0],
            nrows: 2,
            ncols: 2,
        };
        let result = factorize(&mat);
        assert!(
            matches!(result, Err(LdlError::SingularOrIndefinite)),
            "Expected SingularOrIndefinite"
        );
    }

    #[test]
    fn test_ldl_identity() {
        // A = I_4 → L = I（対角成分は格納しない）, D = ones
        let n = 4;
        let entries: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 1.0)).collect();
        let mat = upper_tri_csc(n, &entries);

        let fac = factorize(&mat).expect("factorize failed");

        // D = ones
        for i in 0..n {
            assert!((fac.D[i] - 1.0).abs() < 1e-14, "D[{}]={}", i, fac.D[i]);
        }
        // L は非ゼロなし（単位行列の単位下三角部は対角のみ、対角は格納しない）
        assert_eq!(fac.L.nnz(), 0, "L should have no stored entries for identity");

        // solve(b) = b
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0f64; n];
        fac.solve(&b, &mut x);
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-14, "x[{}]={}", i, x[i]);
        }
    }

    #[test]
    fn test_ldl_solve() {
        // A = [[5,2],[2,3]], b=[1,1]
        // A^{-1} = 1/11 * [[3,-2],[-2,5]]
        // x = A^{-1}*b = [1/11, 3/11]
        let mat = upper_tri_csc(2, &[
            (0, 0, 5.0), (0, 1, 2.0),
            (1, 1, 3.0),
        ]);

        let fac = factorize(&mat).expect("factorize failed");

        let b = [1.0f64, 1.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);

        let eps = 1e-10;
        assert!((x[0] - 1.0 / 11.0).abs() < eps, "x[0]={}", x[0]);
        assert!((x[1] - 3.0 / 11.0).abs() < eps, "x[1]={}", x[1]);
    }
}
