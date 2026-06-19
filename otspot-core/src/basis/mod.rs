//! 改訂単体法（Revised Simplex）の基底管理モジュール
//!
//! 基底行列 B の LU 分解を管理し、FTRAN・BTRAN ソルブと
//! Forrest-Tomlin 基底更新および定期的な再因子分解をサポートする。

pub(crate) mod eta;
pub(crate) mod lu;
pub(crate) mod refactor;
pub(crate) mod ft;

#[cfg(test)]
pub(crate) mod test_utils;

use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use std::time::Instant;

/// 改訂単体法の基底管理トレイト
///
/// 基底行列 B の LU 分解を管理し、FTRAN・BTRAN ソルブ、
/// ピボット更新、再因子分解インターフェースを提供する。
pub(crate) trait BasisManager: Send {
    /// FTRAN: B * x = rhs を解く。結果は `rhs` に上書きされる
    fn ftran(&mut self, rhs: &mut SparseVec);

    /// BTRAN: B^T * x = rhs を解く。結果は `rhs` に上書きされる
    fn btran(&mut self, rhs: &mut SparseVec);

    /// FTRAN（dense版）: B * x = rhs を解く。`rhs` は dense スライスのままで完結する。
    fn ftran_dense(&mut self, rhs: &mut [f64]);

    /// BTRAN（dense版）: B^T * x = rhs を解く。`rhs` は dense スライスのままで完結する。
    fn btran_dense(&mut self, rhs: &mut [f64]);

    /// ピボット後の基底更新: `entering_col` が `leaving_row` を置き換える
    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec);
}

/// Forrest-Tomlin 基底更新付きの LU 分解ベース基底管理構造体
///
/// 初期因子分解後は Forrest-Tomlin column replacement により逐次更新し、
/// 蓄積された更新数が閾値を超えると全再因子分解（refactoring）を行う。
pub(crate) struct LuBasis {
    ft: ft::FtFactors,
    max_updates: usize,
    basis_indices: Vec<usize>,
    /// 再因子分解が特異基底（SingularBasis）により失敗した場合 true。
    /// DeadlineExceeded では false のまま。
    /// 呼び出し元はこのフラグを確認して適切な outcome を返すこと。
    pub(crate) singular_basis: bool,
    /// 再因子分解が失敗した場合 true（SingularBasis または DeadlineExceeded）。
    /// 呼び出し元はこのフラグを確認してsolverを安全に打ち切ること。
    pub(crate) refactor_failed: bool,
}

impl LuBasis {
    #[cfg(test)]
    pub fn new(a: &CscMatrix, basis: &[usize], max_etas: usize) -> Result<Self, SolverError> {
        Self::new_timed(a, basis, max_etas, None)
    }

    pub fn new_timed(
        a: &CscMatrix,
        basis: &[usize],
        max_etas: usize,
        deadline: Option<std::time::Instant>,
    ) -> Result<Self, SolverError> {
        let ft_factors = ft::FtFactors::factorize(a, basis, deadline)?;
        let effective_max = if max_etas == 0 {
            crate::options::default_max_etas(basis.len())
        } else {
            max_etas
        };
        Ok(Self {
            ft: ft_factors,
            max_updates: effective_max,
            basis_indices: basis.to_vec(),
            singular_basis: false,
            refactor_failed: false,
        })
    }

    /// 再因子分解が必要かどうかを返す（FT更新数ベース）
    pub(crate) fn needs_refactor(&self) -> bool {
        self.ft.update_count() >= self.max_updates
    }

    /// 蓄積された FT 更新の数を返す
    pub(crate) fn eta_count(&self) -> usize {
        self.ft.update_count()
    }

    /// 強制的に基底行列を再因子分解する。
    pub(crate) fn force_refactor_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        match ft::FtFactors::factorize(a, basis, deadline) {
            Ok(new_ft) => {
                self.ft = new_ft;
                self.basis_indices = basis.to_vec();
            }
            Err(crate::error::SolverError::SingularBasis { .. }) => {
                self.singular_basis = true;
                self.refactor_failed = true;
            }
            Err(_) => {
                self.refactor_failed = true;
            }
        }
    }

    /// 必要であれば deadline 付きで基底行列を再因子分解する。
    pub(crate) fn refactor_if_needed_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        if self.needs_refactor() {
            self.force_refactor_timed(a, basis, deadline);
        }
    }
}

impl BasisManager for LuBasis {
    fn ftran(&mut self, rhs: &mut SparseVec) {
        self.ft.ftran(rhs);
    }

    fn btran(&mut self, rhs: &mut SparseVec) {
        self.ft.btran(rhs);
    }

    fn ftran_dense(&mut self, rhs: &mut [f64]) {
        self.ft.ftran_dense(rhs);
    }

    fn btran_dense(&mut self, rhs: &mut [f64]) {
        self.ft.btran_dense(rhs);
    }

    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec) {
        let dense = pivot_col.to_dense();
        self.ft.ft_update(leaving_row, &dense);
        self.basis_indices[leaving_row] = entering_col;
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_lu_basis_ftran_btran() {
        // B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 50).unwrap();

        // FTRAN test
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        // Verify B * x = rhs_orig
        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);

        // BTRAN test
        let mut rhs_sv2 = SparseVec::from_dense(&rhs_orig);
        lb.btran(&mut rhs_sv2);
        let y = rhs_sv2.to_dense();

        // Verify B^T * y = rhs_orig
        let bt = a.transpose();
        let check2 = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_update() {
        // A = [[2,1,0,3],[1,3,1,1],[0,1,2,2]]
        // Initial basis = {0,1,2} → B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 50).unwrap();

        // Simulate: entering col 3, leaving row 1
        let entering_col_dense = vec![3.0, 1.0, 2.0];
        let mut pivot_sv = SparseVec::from_dense(&entering_col_dense);
        lb.ftran(&mut pivot_sv);

        lb.update(3, 1, &pivot_sv);

        // New basis = {0, 3, 2} → B_new = [[2,3,0],[1,1,1],[0,2,2]]
        let rhs_orig = vec![5.0, 2.0, 4.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        let b_new_dense = vec![
            vec![2.0, 3.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 2.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);
        let check = b_new.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_refactor_after_max_updates() {
        // 3x5 with extra columns for proper basis changes
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0, 0.5],
            vec![1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0, 2.0, 1.0],
        ];
        let a = dense_to_csc(&dense, 3, 5);
        let mut basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 2).unwrap();

        assert!(!lb.needs_refactor());
        assert_eq!(lb.eta_count(), 0);

        // Update 1: enter col 3, leave row 1
        let mut d1 = vec![3.0, 1.0, 2.0];
        lb.ftran_dense(&mut d1);
        lb.update(3, 1, &SparseVec::from_dense(&d1));
        basis[1] = 3;

        // If zero-diagonal forced early refactor, needs_refactor is true
        // Otherwise eta_count increases normally
        if !lb.needs_refactor() {
            // Update 2: enter col 4, leave row 2
            let mut d2 = vec![0.5, 2.0, 1.0];
            lb.ftran_dense(&mut d2);
            lb.update(4, 2, &SparseVec::from_dense(&d2));
            basis[2] = 4;
        }
        assert!(lb.needs_refactor(), "Should need refactor after updates");

        // Refactor resets
        lb.refactor_if_needed_timed(&a, &basis, None);
        assert!(!lb.needs_refactor());
        assert_eq!(lb.eta_count(), 0);

        // FTRAN should still work
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        lb.ftran_dense(&mut rhs);
        // Build expected B from current basis
        let b_cols: Vec<Vec<f64>> = basis.iter().map(|&col| {
            let (rows, vals) = a.get_column(col).unwrap();
            let mut c = vec![0.0; 3];
            for (&r, &v) in rows.iter().zip(vals.iter()) { c[r] = v; }
            c
        }).collect();
        let b_dense: Vec<Vec<f64>> = (0..3).map(|i| b_cols.iter().map(|c| c[i]).collect()).collect();
        let b = dense_to_csc(&b_dense, 3, 3);
        let check = b.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_refactor() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 2).unwrap();

        // Do 2 identity-updates to trigger refactor
        let mut p1 = SparseVec::from_dense(&[1.0, 0.0, 0.0]);
        lb.ftran(&mut p1);
        lb.update(0, 0, &p1);
        let mut p2 = SparseVec::from_dense(&[0.0, 1.0, 0.0]);
        lb.ftran(&mut p2);
        lb.update(1, 1, &p2);
        assert!(lb.needs_refactor());

        lb.refactor_if_needed_timed(&a, &basis, None);
        assert!(!lb.needs_refactor());
        assert_eq!(lb.eta_count(), 0);

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }
}
