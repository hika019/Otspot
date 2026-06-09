//! Forrest-Tomlin LU solve インフラ (Phase 2a)
//!
//! Phase 2a: 自前の可変 U 表現と FT-aware solve を構築し、
//! 更新ゼロの状態で `LuFactorization` との solve 一致を確認する。
//! Phase 2b でこの土台の上に FT rank-1 更新を実装する。
//!
//! ## solve 順序
//! - FTRAN: `x = P_c⁻¹ · U⁻¹ · ft_etas · L0⁻¹ · P_r · rhs`
//! - BTRAN: `x = P_r⁻¹ · L0⁻ᵀ · ft_etas^ᵀ · U⁻ᵀ · P_c · rhs`

use super::eta::EtaMatrix;
use super::lu::LuFactorization;
use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use faer::sparse::SparseColMatRef;

/// 可変 U 行列 (CSC, 行 index 昇順, 対角ポインタ保持)。
///
/// Phase 2b の FT rank-1 更新が列を書き換える対象。
/// `diag_ptr[j]` は列 j の U[j,j] の row_ind/values 上の絶対インデックス。
#[derive(Debug, Clone)]
pub(crate) struct MutableU {
    pub(crate) n: usize,
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_ind: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) diag_ptr: Vec<usize>,
}

impl MutableU {
    /// faer の U 因子 (行 index 未ソート) から構築する。列内行 index を昇順にソートする。
    pub(crate) fn from_faer(n: usize, u_ref: &SparseColMatRef<'_, usize, f64>) -> Self {
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_ind_all: Vec<usize> = Vec::new();
        let mut values_all: Vec<f64> = Vec::new();
        let mut diag_ptr = vec![usize::MAX; n];
        let mut tmp: Vec<(usize, f64)> = Vec::new();

        for j in 0..n {
            tmp.clear();
            for (row, &val) in u_ref
                .row_idx_of_col(j)
                .zip(u_ref.val_of_col(j).iter())
            {
                tmp.push((row, val));
            }
            tmp.sort_unstable_by_key(|&(r, _)| r);

            let base = row_ind_all.len();
            for (k, &(row, val)) in tmp.iter().enumerate() {
                if row == j {
                    diag_ptr[j] = base + k;
                }
                row_ind_all.push(row);
                values_all.push(val);
            }
            col_ptr[j + 1] = row_ind_all.len();
        }

        debug_assert!(
            diag_ptr.iter().all(|&p| p != usize::MAX),
            "U factor is missing diagonal entry — basis may be singular"
        );

        Self {
            n,
            col_ptr,
            row_ind: row_ind_all,
            values: values_all,
            diag_ptr,
        }
    }

    /// 後退代入: `U · y = rhs` の解を rhs に in-place 上書き。
    pub(crate) fn backward_sub(&self, y: &mut [f64]) {
        for j in (0..self.n).rev() {
            y[j] /= self.values[self.diag_ptr[j]];
            let yj = y[j];
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let row = self.row_ind[k];
                if row < j {
                    y[row] -= self.values[k] * yj;
                }
            }
        }
    }

    /// 前進代入: `U^T · y = rhs` の解を rhs に in-place 上書き。
    pub(crate) fn forward_sub_transpose(&self, y: &mut [f64]) {
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let row = self.row_ind[k];
                if row < j {
                    y[j] -= self.values[k] * y[row];
                }
            }
            y[j] /= self.values[self.diag_ptr[j]];
        }
    }
}

/// faer の L 因子 (unit lower triangular) の前進代入。
/// 対角は unit = 1 のため row == j エントリをスキップする。
fn forward_sub_l(n: usize, l_ref: &SparseColMatRef<'_, usize, f64>, y: &mut [f64]) {
    for j in 0..n {
        let yj = y[j];
        for (row, &val) in l_ref
            .row_idx_of_col(j)
            .zip(l_ref.val_of_col(j).iter())
        {
            if row > j {
                y[row] -= val * yj;
            }
        }
    }
}

/// faer の L^T (unit upper triangular) の後退代入。
/// 対角は unit = 1 のため除算不要。
fn backward_sub_lt(n: usize, l_ref: &SparseColMatRef<'_, usize, f64>, y: &mut [f64]) {
    for j in (0..n).rev() {
        for (row, &val) in l_ref
            .row_idx_of_col(j)
            .zip(l_ref.val_of_col(j).iter())
        {
            if row > j {
                y[j] -= val * y[row];
            }
        }
        // unit lower diagonal = 1: 除算不要
    }
}

/// Forrest-Tomlin LU solve 構造体。
///
/// Phase 2a では更新ゼロで `LuFactorization` と同一の solve 結果を保証する。
/// Phase 2b で `update()` に FT rank-1 更新ロジックを実装する。
pub(crate) struct FtLu {
    pub(crate) n: usize,
    pub(crate) lu0: LuFactorization,
    pub(crate) u_mat: MutableU,
    row_perm_fwd: Vec<usize>,
    row_perm_inv: Vec<usize>,
    col_perm_fwd: Vec<usize>,
    /// FT eta 列。Phase 2a は空。Phase 2b で FT 更新時に追加する。
    ft_etas: Vec<EtaMatrix>,
}

impl FtLu {
    pub(crate) fn new(a: &CscMatrix, basis: &[usize]) -> Result<Self, SolverError> {
        let lu0 = LuFactorization::factorize_timed(a, basis, None)?;
        let n = lu0.n;

        let (row_perm_fwd, row_perm_inv) = {
            let rp = lu0.row_perm();
            let (fwd, inv) = rp.arrays();
            (fwd.to_vec(), inv.to_vec())
        };
        let col_perm_fwd = {
            let cp = lu0.col_perm();
            cp.arrays().0.to_vec()
        };
        let u_mat = {
            let u_ref = lu0.u_factor();
            MutableU::from_faer(n, &u_ref)
        };

        Ok(Self {
            n,
            lu0,
            u_mat,
            row_perm_fwd,
            row_perm_inv,
            col_perm_fwd,
            ft_etas: Vec::new(),
        })
    }

    /// FTRAN (dense): `B · x = rhs` を in-place で解く。
    pub(crate) fn ftran(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|p| rhs[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut y);
        }
        super::eta::apply_ftran(&self.ft_etas, &mut y);
        self.u_mat.backward_sub(&mut y);
        for j in 0..n {
            rhs[self.col_perm_fwd[j]] = y[j];
        }
    }

    /// BTRAN (dense): `B^T · x = rhs` を in-place で解く。
    pub(crate) fn btran(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|j| rhs[self.col_perm_fwd[j]]).collect();
        self.u_mat.forward_sub_transpose(&mut y);
        super::eta::apply_btran(&self.ft_etas, &mut y);
        {
            let l = self.lu0.l_factor();
            backward_sub_lt(n, &l, &mut y);
        }
        for i in 0..n {
            rhs[i] = y[self.row_perm_inv[i]];
        }
    }

    /// FTRAN (sparse wrapper)。
    pub(crate) fn ftran_sparse(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.ftran(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    /// BTRAN (sparse wrapper)。
    pub(crate) fn btran_sparse(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.btran(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    /// ピボット更新 (Phase 2b で実装)。
    // Phase 2b で実装: u_mat の該当列を FT rank-1 更新し ft_etas に追加する
    #[allow(dead_code)]
    pub(crate) fn update(
        &mut self,
        _entering_col: usize,
        _leaving_row: usize,
    ) -> Result<(), SolverError> {
        unimplemented!("FT 更新は Phase 2b で実装する")
    }

    /// テスト用: U 対角を 1.0 に固定した ftran (no-op pivot sentinel 確認用)。
    #[cfg(test)]
    fn ftran_unit_pivot(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|p| rhs[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut y);
        }
        let mut broken = self.u_mat.clone();
        for j in 0..n {
            broken.values[broken.diag_ptr[j]] = 1.0;
        }
        broken.backward_sub(&mut y);
        for j in 0..n {
            rhs[self.col_perm_fwd[j]] = y[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::lu::{solve_btran, solve_ftran, LuFactorization};
    use crate::basis::test_utils::{assert_vec_near, dense_to_csc};
    use crate::sparse::CscMatrix;

    /// 決定論的な LCG で n×n の対角優位疎行列を生成する (非特異性を対角優位で保証)。
    fn gen_matrix(n: usize, seed: u64) -> CscMatrix {
        let mut lcg = seed;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(6.0 + (i as f64 * 0.7 + seed as f64 * 0.1).sin().abs() * 2.0);
        }
        for _ in 0..(n * 2) {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = ((lcg >> 33) as usize) % n;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((lcg >> 33) as usize) % n;
            if i != j {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((lcg >> 32) as f64 / u32::MAX as f64 - 0.5) * 0.8;
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    fn gen_rhs(n: usize, seed: u64) -> Vec<f64> {
        let mut lcg = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((lcg >> 32) as f64 / u32::MAX as f64) * 10.0 - 5.0
            })
            .collect()
    }

    /// sentinel: FtLu.ftran が LuFactorization と 1e-10 内で一致し、B·x=rhs 残差 < 1e-10。
    /// 8 seed × 5 rhs、サイズ 10/20/30。
    #[test]
    fn test_ftlu_ftran_matches_lu() {
        let configs: &[(usize, &[u64])] = &[
            (10, &[1, 2, 3]),
            (20, &[10, 20, 30]),
            (30, &[100, 200]),
        ];
        for &(n, seeds) in configs {
            for &seed in seeds {
                let a = gen_matrix(n, seed);
                let basis: Vec<usize> = (0..n).collect();
                let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();
                let ft = FtLu::new(&a, &basis).unwrap();

                for rhs_seed in 0..5u64 {
                    let rhs_orig = gen_rhs(n, seed * 100 + rhs_seed);

                    let mut rhs_lu = rhs_orig.clone();
                    solve_ftran(&lu, &mut rhs_lu);

                    let mut rhs_ft = rhs_orig.clone();
                    ft.ftran(&mut rhs_ft);

                    let max_diff: f64 = rhs_lu
                        .iter()
                        .zip(rhs_ft.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_diff < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: ftran diff={max_diff:.2e}"
                    );

                    let check = a.mat_vec_mul(&rhs_ft).unwrap();
                    let residual: f64 = check
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        residual < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: B·x=rhs residual={residual:.2e}"
                    );
                }
            }
        }
    }

    /// sentinel: FtLu.btran が LuFactorization と 1e-10 内で一致し、B^T·x=rhs 残差 < 1e-10。
    #[test]
    fn test_ftlu_btran_matches_lu() {
        let configs: &[(usize, &[u64])] = &[
            (10, &[1, 2, 3]),
            (20, &[10, 20, 30]),
            (30, &[100, 200]),
        ];
        for &(n, seeds) in configs {
            for &seed in seeds {
                let a = gen_matrix(n, seed);
                let basis: Vec<usize> = (0..n).collect();
                let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();
                let ft = FtLu::new(&a, &basis).unwrap();

                for rhs_seed in 0..5u64 {
                    let rhs_orig = gen_rhs(n, seed * 100 + rhs_seed + 500);

                    let mut rhs_lu = rhs_orig.clone();
                    solve_btran(&lu, &mut rhs_lu);

                    let mut rhs_ft = rhs_orig.clone();
                    ft.btran(&mut rhs_ft);

                    let max_diff: f64 = rhs_lu
                        .iter()
                        .zip(rhs_ft.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_diff < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: btran diff={max_diff:.2e}"
                    );

                    let bt = a.transpose();
                    let check = bt.mat_vec_mul(&rhs_ft).unwrap();
                    let residual: f64 = check
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        residual < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: B^T·x=rhs residual={residual:.2e}"
                    );
                }
            }
        }
    }

    /// sentinel: MutableU の非ゼロ構造が faer u_factor() と一致し、diag_ptr が正しい。
    #[test]
    fn test_ftlu_u_representation_matches_faer() {
        for (n, seed) in [(5usize, 1u64), (10, 42), (20, 99)] {
            let a = gen_matrix(n, seed);
            let basis: Vec<usize> = (0..n).collect();
            let ft = FtLu::new(&a, &basis).unwrap();
            let u_ref = ft.lu0.u_factor();

            for j in 0..n {
                let mut faer_col: Vec<(usize, f64)> = u_ref
                    .row_idx_of_col(j)
                    .zip(u_ref.val_of_col(j).iter())
                    .map(|(r, &v)| (r, v))
                    .collect();
                faer_col.sort_by_key(|&(r, _)| r);

                let start = ft.u_mat.col_ptr[j];
                let end = ft.u_mat.col_ptr[j + 1];
                let mu_col: Vec<(usize, f64)> = (start..end)
                    .map(|k| (ft.u_mat.row_ind[k], ft.u_mat.values[k]))
                    .collect();

                assert_eq!(
                    faer_col.len(),
                    mu_col.len(),
                    "n={n} seed={seed} col={j}: nnz mismatch"
                );
                for (f, m) in faer_col.iter().zip(mu_col.iter()) {
                    assert_eq!(f.0, m.0, "n={n} seed={seed} col={j}: row mismatch");
                    assert!(
                        (f.1 - m.1).abs() < 1e-15,
                        "n={n} seed={seed} col={j}: val mismatch {:.2e} vs {:.2e}",
                        f.1,
                        m.1
                    );
                }

                let diag_idx = ft.u_mat.diag_ptr[j];
                assert_eq!(
                    ft.u_mat.row_ind[diag_idx], j,
                    "n={n} seed={seed}: diag_ptr[{j}] points to row {} not {j}",
                    ft.u_mat.row_ind[diag_idx]
                );
            }
        }
    }

    /// no-op sentinel: U 対角を 1.0 に固定すると B·x=rhs 残差が爆発する。
    /// backward_sub の対角除算コードパスが必須であることを実機確認。
    #[test]
    fn test_ftlu_no_op_pivot_identity_residual_explodes() {
        let dense = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 2.0],
            vec![0.0, 2.0, 5.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let ft = FtLu::new(&a, &[0, 1, 2]).unwrap();
        let rhs_orig = vec![5.0, 6.0, 9.0];

        // 正常: residual < 1e-10
        let mut rhs_ok = rhs_orig.clone();
        ft.ftran(&mut rhs_ok);
        let check_ok = a.mat_vec_mul(&rhs_ok).unwrap();
        let residual_ok: f64 = check_ok
            .iter()
            .zip(rhs_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            residual_ok < 1e-10,
            "correct ftran residual={residual_ok:.2e}"
        );

        // pivot=1 固定: residual が有意に大きい (no-op で fail する設計)
        let mut rhs_broken = rhs_orig.clone();
        ft.ftran_unit_pivot(&mut rhs_broken);
        let check_broken = a.mat_vec_mul(&rhs_broken).unwrap();
        let residual_broken: f64 = check_broken
            .iter()
            .zip(rhs_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            residual_broken > 1e-3,
            "no-op pivot should explode residual, got={residual_broken:.2e}"
        );
    }

    /// 既存テストケースとの整合確認 (3x3 dense / sparse wrapper)。
    #[test]
    fn test_ftlu_small_matrices() {
        let dense3 = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a3 = dense_to_csc(&dense3, 3, 3);
        let ft3 = FtLu::new(&a3, &[0, 1, 2]).unwrap();
        let rhs = vec![3.0, 5.0, 3.0];

        // FTRAN
        let mut x = rhs.clone();
        ft3.ftran(&mut x);
        let check = a3.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs, 1e-10);

        // BTRAN
        let mut y = rhs.clone();
        ft3.btran(&mut y);
        let bt = a3.transpose();
        let check_bt = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check_bt, &rhs, 1e-10);

        // sparse wrapper
        let mut sv = SparseVec::from_dense(&rhs);
        ft3.ftran_sparse(&mut sv);
        let x_sp = sv.to_dense();
        let check_sp = a3.mat_vec_mul(&x_sp).unwrap();
        assert_vec_near(&check_sp, &rhs, 1e-10);
    }
}
