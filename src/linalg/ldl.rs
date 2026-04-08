//! faer high-level LDL^T wrapper for sparse linear systems (cmd_275)
//!
//! Uses faer's high-level Cholesky API (SymbolicCholesky + SupernodalThreshold::AUTO)
//! to automatically select simplicial or supernodal factorization based on matrix structure.
//! Banded/sparse matrices (LISWET etc.) → simplicial; dense fill (AUG2D etc.) → supernodal.
//!
//! Public API (backward-compatible):
//! - `LdlFactorization`        — positive definite, no AMD
//! - `LdlFactorizationAmd`     — quasidefinite, with AMD permutation
//! - `LdlError`
//! - `factorize`
//! - `factorize_with_deadline`
//! - `factorize_quasidefinite_with_cached_perm`
//! - `factorize_quasidefinite_with_amd`

use crate::linalg::amd::{amd_with_deadline, inv_permute_vec, permute_sym_upper, permute_vec};
use crate::sparse::CscMatrix;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::reborrow::*;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use std::sync::mpsc;
use std::time::Instant;

/// LDL分解エラー
#[non_exhaustive]
#[derive(Debug)]
pub enum LdlError {
    /// 行列が特異または不定（faer regularization でも処理不能な場合）
    SingularOrIndefinite,
    /// deadline を超過した
    DeadlineExceeded,
}

// ── LdlFactorization (positive definite, no AMD) ──────────────────────────────

/// faer high-level LDL^T factorization for positive definite matrices (no AMD).
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
pub struct LdlFactorization {
    symbolic: SymbolicCholesky<usize>,
    l_values: Vec<f64>,
    n: usize,
}

impl LdlFactorization {
    /// LDL^T x = b を解く。sol に解を書き込む。
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        sol.copy_from_slice(rhs);
        let mut mem = MemBuffer::new(
            self.symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
        );
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat =
            faer::MatMut::from_column_major_slice_mut(sol, self.n, 1);
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            sol_mat.rb_mut(),
            faer::Par::Seq,
            stack,
        );
    }
}

// ── LdlFactorizationAmd (quasidefinite, with AMD permutation) ─────────────────

/// faer high-level LDL^T factorization for quasidefinite matrices with AMD ordering.
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
pub struct LdlFactorizationAmd {
    symbolic: SymbolicCholesky<usize>,
    l_values: Vec<f64>,
    /// AMD permutation: perm[k] = original index of reordered index k
    perm: Vec<usize>,
    n: usize,
}

impl LdlFactorizationAmd {
    /// L 因子の非ゼロ数を返す（デバッグ用）
    pub fn nnz_l(&self) -> usize {
        self.symbolic.len_val()
    }

    /// Symbolic を再利用して数値因子化のみを再実行する（高速パス）。
    ///
    /// mat のスパースパターンが初回と同一であることが前提。
    /// IPM の反復内でスパースパターンが変わらない場合に使用する。
    /// Note: deadline check is performed before factorization starts.
    /// Factorization itself may exceed the deadline for large matrices.
    pub fn refactorize_numeric(
        &mut self,
        mat: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), LdlError> {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Err(LdlError::DeadlineExceeded);
            }
        }
        let n = mat.nrows;
        let (new_col_ptr, new_row_ind, new_values) =
            permute_sym_upper(n, &mat.col_ptr, &mat.row_ind, &mat.values, &self.perm);
        let perm_mat = CscMatrix {
            col_ptr: new_col_ptr,
            row_ind: new_row_ind,
            values: new_values,
            nrows: n,
            ncols: n,
        };
        let a_upper = csc_upper_to_faer_upper(&perm_mat);
        let signs = extract_diagonal_signs(&perm_mat);
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: 1e-8,
            dynamic_regularization_epsilon: 1e-13,
        };
        let mut mem = MemBuffer::new(StackReq::any_of(&[
            self.symbolic.factorize_numeric_ldlt_scratch::<f64>(faer::Par::Seq, Default::default()),
            self.symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
        ]));
        let stack = MemStack::new(&mut mem);
        let mut new_l_values = vec![0.0f64; self.symbolic.len_val()];
        self.symbolic
            .factorize_numeric_ldlt(
                &mut new_l_values,
                a_upper.rb(),
                faer::Side::Upper,
                regularization,
                faer::Par::Seq,
                stack,
                Default::default(),
            )
            .map_err(|_| LdlError::SingularOrIndefinite)?;
        self.l_values = new_l_values;
        Ok(())
    }

    /// AMD 付き LDL^T x = b を解く。
    ///
    /// 1. 右辺を前方置換: b_p[k] = rhs[perm[k]]
    /// 2. (PAP^T) x_p = b_p を faer で解く（simplicial/supernodal は AUTO 選択）
    /// 3. 解を逆置換: sol[perm[k]] = x_p[k]
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.n;
        let b_p = permute_vec(rhs, &self.perm);
        let mut x_p = b_p;

        let mut mem = MemBuffer::new(
            self.symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
        );
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat =
            faer::MatMut::from_column_major_slice_mut(&mut x_p, n, 1);
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            sol_mat.rb_mut(),
            faer::Par::Seq,
            stack,
        );

        let x = inv_permute_vec(&x_p, &self.perm);
        sol.copy_from_slice(&x);
    }

    /// Symbolic を再利用して数値因子化のみを実行する（スレッドなし版）。
    ///
    /// # スレッドなし設計の理由
    /// `SymbolicSupernodalCholesky<I>` が `Clone` 未実装のためスレッドへの渡しが困難。
    /// `Arc<SymbolicCholesky>` 化は将来タスク。
    ///
    /// # deadline 超過リスク
    /// スレッドなし実装のため LDLT 計算中は deadline チェックができない（現行と同等）。
    ///
    /// # 引数
    /// - `mat`: スパースパターンが初回と同一の CSC 行列（values のみ更新済み）
    /// - `deadline`: タイムアウト期限（None の場合は無制限で実行）
    pub fn refactorize_numeric_threaded(
        &mut self,
        mat: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), LdlError> {
        // deadline 事前チェック
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Err(LdlError::DeadlineExceeded);
            }
        }
        // スレッドなしで直接呼ぶ（SymbolicSupernodalCholesky<I> が Clone 未実装のため）
        self.refactorize_numeric(mat, None)
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// 上三角 CscMatrix を faer SparseColMat（上三角）に変換する。
///
/// 上三角エントリ (i, j) with i ≤ j をそのまま faer 形式に変換する。
/// 高レベル API に Side::Upper で渡すために使用する。
fn csc_upper_to_faer_upper(mat: &CscMatrix) -> SparseColMat<usize, f64> {
    let n = mat.nrows;
    let mut triplets = Vec::with_capacity(mat.values.len());
    for j in 0..n {
        for k in mat.col_ptr[j]..mat.col_ptr[j + 1] {
            let i = mat.row_ind[k];
            let v = mat.values[k];
            triplets.push(Triplet::new(i, j, v));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets)
        .expect("csc_upper_to_faer_upper: failed to build SparseColMat")
}

/// 対角要素の符号ベクトルを抽出する（LdltRegularization sign-aware 用）。
///
/// 対角が負なら -1、それ以外は +1。
fn extract_diagonal_signs(mat: &CscMatrix) -> Vec<i8> {
    let n = mat.nrows;
    let mut signs = vec![1i8; n];
    for (j, sign) in signs.iter_mut().enumerate() {
        for k in mat.col_ptr[j]..mat.col_ptr[j + 1] {
            if mat.row_ind[k] == j {
                if mat.values[k] < 0.0 {
                    *sign = -1;
                }
                break;
            }
        }
    }
    signs
}

/// 高レベル API で SymbolicCholesky を計算する（SupernodalThreshold::AUTO）。
///
/// Side::Upper で渡すため a_upper は上三角 faer SparseColMat であること。
/// SymmetricOrdering::Identity: AMD は外部で処理済みのため faer 内部では行わない。
fn build_symbolic_hl(
    a_upper: &SparseColMat<usize, f64>,
) -> Result<SymbolicCholesky<usize>, LdlError> {
    factorize_symbolic_cholesky(
        a_upper.symbolic(),
        faer::Side::Upper,
        SymmetricOrdering::Identity,
        CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            ..Default::default()
        },
    )
    .map_err(|_| LdlError::SingularOrIndefinite)
}

/// 高レベル API で symbolic + numeric 因子化を実行する共通処理。
///
/// `signs`: Some → quasidefinite sign-aware regularization
///          None → sign-unaware regularization (positive definite 向け)
fn do_numeric_factorize(
    mat: &CscMatrix,
    signs: Option<&[i8]>,
) -> Result<(SymbolicCholesky<usize>, Vec<f64>), LdlError> {
    let a_upper = csc_upper_to_faer_upper(mat);
    let symbolic = build_symbolic_hl(&a_upper)?;

    let regularization = LdltRegularization {
        dynamic_regularization_signs: signs,
        dynamic_regularization_delta: 1e-8,
        dynamic_regularization_epsilon: 1e-13,
    };

    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        symbolic.factorize_numeric_ldlt_scratch::<f64>(faer::Par::Seq, Default::default()),
        symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
    ]));
    let stack = MemStack::new(&mut mem);

    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a_upper.rb(),
            faer::Side::Upper,
            regularization,
            faer::Par::Seq,
            stack,
            Default::default(),
        )
        .map_err(|_| LdlError::SingularOrIndefinite)?;

    Ok((symbolic, l_values))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// 正定値疎行列の LDL^T 分解を実行する。
pub fn factorize(mat: &CscMatrix) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    let (symbolic, l_values) = do_numeric_factorize(mat, None)?;
    Ok(LdlFactorization { symbolic, l_values, n })
}

/// deadline 付き正定値疎行列の LDL^T 分解。
///
/// deadline チェックは factorize 前のみ実施
/// （faer は mid-factorization キャンセルを持たないため）。
/// Note: deadline check is performed before factorization starts.
/// Factorization itself may exceed the deadline for large matrices.
pub fn factorize_with_deadline(
    mat: &CscMatrix,
    deadline: Option<Instant>,
) -> Result<LdlFactorization, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    factorize(mat)
}

/// AMD キャッシュ済み置換付き quasidefinite LDL^T 分解。
///
/// `mat`: 元の（未置換の）augmented KKT 行列（上三角 CSC）
/// `perm`: 事前計算済み AMD 置換ベクトル（perm[k] = 元インデックス）
/// `deadline`: factorize 前チェックのみ（mid-factorization 未対応）
pub fn factorize_quasidefinite_with_cached_perm(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
) -> Result<LdlFactorizationAmd, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = mat.nrows;
    let (new_col_ptr, new_row_ind, new_values) =
        permute_sym_upper(n, &mat.col_ptr, &mat.row_ind, &mat.values, perm);
    let perm_mat = CscMatrix {
        col_ptr: new_col_ptr,
        row_ind: new_row_ind,
        values: new_values,
        nrows: n,
        ncols: n,
    };
    let signs = extract_diagonal_signs(&perm_mat);
    let (symbolic, l_values) = do_numeric_factorize(&perm_mat, Some(&signs))?;
    Ok(LdlFactorizationAmd { symbolic, l_values, perm: perm.to_vec(), n })
}

/// AMD キャッシュ済み置換付き quasidefinite LDL^T 分解（スレッド版）。
///
/// 因子化をバックグラウンドスレッドで実行し recv_timeout でタイムアウト制限する。
/// タイムアウト後もスレッドは実行を継続する（read-only 計算、いずれ完了する）。
///
/// 短期対処。cmd_403(Inexact IPM)で根本解消予定。
pub fn factorize_quasidefinite_with_cached_perm_threaded(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
) -> Result<LdlFactorizationAmd, LdlError> {
    let remaining = match deadline {
        None => return factorize_quasidefinite_with_cached_perm(mat, perm, None),
        Some(d) => {
            let now = Instant::now();
            if now >= d {
                return Err(LdlError::DeadlineExceeded);
            }
            d - now
        }
    };
    let mat_owned = mat.clone();
    let perm_owned = perm.to_vec();
    let (tx, rx) = mpsc::channel::<Result<LdlFactorizationAmd, LdlError>>();
    std::thread::spawn(move || {
        // 短期対処。cmd_403(Inexact IPM)で根本解消予定。
        // タイムアウト後もスレッドは実行を継続する（read-only 計算）。
        let _ = tx.send(factorize_quasidefinite_with_cached_perm(&mat_owned, &perm_owned, None));
    });
    match rx.recv_timeout(remaining) {
        Ok(result) => result,
        Err(_) => Err(LdlError::DeadlineExceeded),
    }
}

/// AMD 再順序化付き quasidefinite LDL^T 分解（AMD を内部で計算）。
#[allow(dead_code)]
pub fn factorize_quasidefinite_with_amd(
    mat: &CscMatrix,
    deadline: Option<Instant>,
) -> Result<LdlFactorizationAmd, LdlError> {
    let n = mat.nrows;
    let perm = amd_with_deadline(n, &mat.col_ptr, &mat.row_ind, deadline);
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    factorize_quasidefinite_with_cached_perm(mat, &perm, deadline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use faer::sparse::linalg::cholesky::{simplicial, supernodal};

    /// 上三角 CSC 行列をエントリリストから構築するヘルパー
    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let mut cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(row, col, val) in entries {
            assert!(row <= col, "upper triangle only: row={row} col={col}");
            cols[col].push((row, val));
        }
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

    #[test]
    fn test_factorize_pd_3x3_solve() {
        // A = [[4,1,0],[1,3,2],[0,2,5]] — positive definite
        let mat = upper_tri_csc(3, &[
            (0, 0, 4.0), (0, 1, 1.0),
            (1, 1, 3.0), (1, 2, 2.0),
            (2, 2, 5.0),
        ]);
        let fac = factorize(&mat).expect("factorize failed");
        let b = [1.0f64, 2.0, 3.0];
        let mut x = [0.0f64; 3];
        fac.solve(&b, &mut x);
        // residual |Ax - b|
        let ax0 = 4.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2];
        let ax2 = 2.0 * x[1] + 5.0 * x[2];
        let eps = 1e-8;
        assert!((ax0 - b[0]).abs() < eps, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < eps, "r[1]={}", (ax1 - b[1]).abs());
        assert!((ax2 - b[2]).abs() < eps, "r[2]={}", (ax2 - b[2]).abs());
    }

    #[test]
    fn test_factorize_pd_identity() {
        // A = I_4
        let n = 4;
        let entries: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 1.0)).collect();
        let mat = upper_tri_csc(n, &entries);
        let fac = factorize(&mat).expect("factorize failed");
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0f64; n];
        fac.solve(&b, &mut x);
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-10, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn test_factorize_with_deadline_ok() {
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
        // deadline far in the future
        let deadline = Some(Instant::now() + std::time::Duration::from_secs(60));
        let fac = factorize_with_deadline(&mat, deadline).expect("should succeed");
        let b = [1.0f64, 0.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);
        let ax0 = 2.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1];
        assert!((ax0 - b[0]).abs() < 1e-10, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < 1e-10, "r[1]={}", (ax1 - b[1]).abs());
    }

    #[test]
    fn test_factorize_with_deadline_expired() {
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (1, 1, 3.0)]);
        let deadline = Some(Instant::now() - std::time::Duration::from_millis(1));
        let result = factorize_with_deadline(&mat, deadline);
        assert!(
            matches!(result, Err(LdlError::DeadlineExceeded)),
            "Expected DeadlineExceeded"
        );
    }

    #[test]
    fn test_quasidefinite_2x2_identity_perm() {
        // quasidefinite: [[3,1],[1,-2]] — D[0]>0, D[1]<0
        let mat = upper_tri_csc(2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 1, -2.0)]);
        let perm = vec![0usize, 1]; // identity permutation
        let fac = factorize_quasidefinite_with_cached_perm(&mat, &perm, None)
            .expect("quasidefinite factorize failed");
        let b = [1.0f64, 2.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);
        // residual: [[3,1],[1,-2]] * x = b
        let ax0 = 3.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] - 2.0 * x[1];
        let eps = 1e-8;
        assert!((ax0 - b[0]).abs() < eps, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < eps, "r[1]={}", (ax1 - b[1]).abs());
    }

    #[test]
    fn test_quasidefinite_with_amd() {
        // 5x5 quasidefinite: Q=diag(1,2), A=[[1,0],[0,1],[1,1]], δ=1e-4
        let delta = 1e-4f64;
        let mat = upper_tri_csc(5, &[
            (0, 0, 1.0 + delta), (1, 1, 2.0 + delta),
            (2, 2, -delta), (3, 3, -delta), (4, 4, -delta),
            (0, 2, 1.0), (1, 3, 1.0), (0, 4, 1.0), (1, 4, 1.0),
        ]);
        let fac = factorize_quasidefinite_with_amd(&mat, None)
            .expect("quasidefinite_with_amd failed");
        let b = [1.0f64, 2.0, 0.5, -0.5, 1.0];
        let mut x = [0.0f64; 5];
        fac.solve(&b, &mut x);
        // Full matrix for residual check (symmetric)
        let full: &[(usize, usize, f64)] = &[
            (0, 0, 1.0 + delta), (1, 1, 2.0 + delta),
            (2, 2, -delta), (3, 3, -delta), (4, 4, -delta),
            (0, 2, 1.0), (2, 0, 1.0),
            (1, 3, 1.0), (3, 1, 1.0),
            (0, 4, 1.0), (4, 0, 1.0),
            (1, 4, 1.0), (4, 1, 1.0),
        ];
        let mut r = [0.0f64; 5];
        for &(row, col, val) in full {
            r[row] += val * x[col];
        }
        let res: f64 = r.iter().zip(b.iter()).map(|(&ri, &bi)| (ri - bi).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1e-8, "residual={res:.3e}");
    }

    #[test]
    fn test_nnz_l_reasonable() {
        // nnz_l should be positive for a non-trivial matrix
        // (supernodal stores diagonal + fill-in, upper bound = n*(n+1)/2)
        let n = 3usize;
        let mat = upper_tri_csc(n, &[
            (0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0), (1, 2, 2.0), (2, 2, 5.0),
        ]);
        let perm = vec![0, 1, 2];
        let fac = factorize_quasidefinite_with_cached_perm(&mat, &perm, None)
            .expect("factorize failed");
        let nnz = fac.nnz_l();
        // supernodal len_val() includes internal storage (may exceed lower-tri count)
        assert!(nnz > 0, "nnz_l should be positive for non-trivial matrix");
    }

    /// 診断テスト: LISWET類似帯状行列でのsupernodal vs simplicial比較
    ///
    /// LISWET KKT行列に近似した帯状対称行列(n=2000, band=2)を生成し、
    /// supernodalのスーパーノード数・len_val・factorize時間 vs simplicial時間
    /// を計測して報告する。
    #[test]
    fn diag_banded_supernode_vs_simplicial() {
        // 帯状対称正定値行列: n×n, band=2 (LISWETのKKT近似)
        let n = 2000usize;
        let band = 2usize;

        // 下三角 triplets (faer下三角形式)
        let mut lo_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for i in 0..n {
            lo_triplets.push(Triplet::new(i, i, 4.0));
            if i + 1 < n { lo_triplets.push(Triplet::new(i + 1, i, -1.0)); }
            if i + 2 < n { lo_triplets.push(Triplet::new(i + 2, i, -0.5)); }
        }
        let a_lower = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &lo_triplets)
            .expect("build lower failed");

        let a_nnz = a_lower.compute_nnz();
        let a_upper_sym = a_lower.rb().transpose().symbolic().to_col_major()
            .expect("transpose failed");

        // etree / col_counts (共通)
        let mut etree_buf = vec![0isize; n];
        let mut col_counts_buf = vec![0usize; n];
        {
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree_buf, &mut col_counts_buf, a_upper_sym.rb(), stack,
            );
        }

        // Supernodal: (usize::MAX, 1.0) = 現在の実装
        let relax_all: &[(usize, f64)] = &[(usize::MAX, 1.0)];
        // Supernodal: DEFAULT_RELAX
        let relax_default: &[(usize, f64)] = &[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)];

        for (label, relax) in [
            ("supernodal(relax=ALL)", relax_all),
            ("supernodal(relax=DEFAULT)", relax_default),
        ] {
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            // etree/col_counts を再取得（stackが消費されるため）
            let mut etree = etree_buf.clone();
            let mut col_counts = col_counts_buf.clone();
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree, &mut col_counts, a_upper_sym.rb(), stack,
            );

            let mut mem2 = MemBuffer::new(supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n));
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym = supernodal::factorize_supernodal_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
                faer::sparse::linalg::SymbolicSupernodalParams { relax: Some(relax) },
            ).expect("symbolic failed");
            let sym_t = t0.elapsed();

            let n_sn = sym.n_supernodes();
            let len_val = sym.len_val();
            let begin = sym.supernode_begin();
            let end = sym.supernode_end();
            let sizes: Vec<usize> = (0..n_sn).map(|i| end[i] - begin[i]).collect();
            let max_sn = sizes.iter().max().copied().unwrap_or(0);
            let avg_sn = sizes.iter().sum::<usize>() as f64 / n_sn as f64;

            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: 1e-8,
                dynamic_regularization_epsilon: 1e-13,
            };
            let mut mem3 = MemBuffer::new(StackReq::any_of(&[
                supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
                    &sym, faer::Par::Seq, Default::default()),
                sym.solve_in_place_scratch::<f64>(n, faer::Par::Seq),
            ]));
            let stack3 = MemStack::new(&mut mem3);
            let mut l_values = vec![0.0f64; sym.len_val()];
            let t1 = Instant::now();
            supernodal::factorize_supernodal_numeric_ldlt::<usize, f64>(
                &mut l_values, a_lower.rb(), regularization, &sym,
                faer::Par::Seq, stack3, Default::default(),
            ).expect("numeric failed");
            let num_t = t1.elapsed();

            println!(
                "[{label}] n={n}, band={band}: n_supernodes={n_sn}, len_val={len_val}, \
                 max_sn={max_sn}, avg_sn={avg_sn:.1}, sym={sym_t:.3?}, num={num_t:.3?}"
            );
        }

        // Simplicial
        {
            let mut etree = etree_buf.clone();
            let mut col_counts = col_counts_buf.clone();
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree, &mut col_counts, a_upper_sym.rb(), stack,
            );

            let mut mem2 = MemBuffer::new(simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n));
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym_s = simplicial::factorize_simplicial_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
            ).expect("simplicial symbolic failed");
            let sym_t = t0.elapsed();

            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: 1e-8,
                dynamic_regularization_epsilon: 1e-13,
            };
            let l_nnz = sym_s.len_val();
            let mut l_values = vec![0.0f64; l_nnz];
            let mut mem3 = MemBuffer::new(
                simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(n)
            );
            let stack3 = MemStack::new(&mut mem3);
            let t1 = Instant::now();
            simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
                &mut l_values, a_lower.rb(), regularization, &sym_s, stack3,
            ).expect("simplicial numeric failed");
            let num_t = t1.elapsed();

            println!(
                "[simplicial] n={n}, band={band}: l_nnz={l_nnz}, sym={sym_t:.3?}, num={num_t:.3?}"
            );
        }
        let _ = band; // suppress unused warning
    }
}
