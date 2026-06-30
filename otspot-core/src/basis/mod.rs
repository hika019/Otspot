//! 改訂単体法（Revised Simplex）の基底管理モジュール
//!
//! 基底行列 B の LU 分解を管理し、FTRAN・BTRAN ソルブと
//! ピボット更新（eta ファイル）および定期的な再因子分解をサポートする。

pub(crate) mod eta;
pub(crate) mod lu;
pub(crate) mod refactor;

#[cfg(test)]
pub(crate) mod test_utils;

use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use faer::dyn_stack::MemBuffer;
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
    /// sparse 変換を介さないため、常に dense な入力（c_B 等）に対して高速。
    fn ftran_dense(&mut self, rhs: &mut [f64]);

    /// BTRAN（dense版）: B^T * x = rhs を解く。`rhs` は dense スライスのままで完結する。
    /// sparse 変換を介さないため、常に dense な入力（c_B 等）に対して高速。
    fn btran_dense(&mut self, rhs: &mut [f64]);

    /// ピボット後の基底更新: `entering_col` が `leaving_row` を置き換える
    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec);
}

/// eta ファイル更新付きの LU 分解ベース基底管理構造体
///
/// 初期因子分解後は eta ファイルにより逐次更新し、
/// 蓄積誤差が閾値を超えると全再因子分解（refactoring）を行う。
pub(crate) struct LuBasis {
    lu: lu::LuFactorization,
    eta_file: eta::EtaFile,
    basis_indices: Vec<usize>,
    /// 再因子分解が特異基底（SingularBasis）により失敗した場合 true。
    /// DeadlineExceeded では false のまま。
    /// 呼び出し元はこのフラグを確認して適切な outcome を返すこと。
    pub(crate) singular_basis: bool,
    /// 再因子分解が失敗した場合 true（SingularBasis または DeadlineExceeded）。
    /// 呼び出し元はこのフラグを確認してsolverを安全に打ち切ること。
    pub(crate) refactor_failed: bool,
    solve_scratch: MemBuffer,
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
        let lu = lu::LuFactorization::factorize_timed(a, basis, deadline)?;
        // max_etas == 0 を auto と解釈し m から動的計算 (CLAUDE.md 固定値排除)。
        let effective_max_etas = if max_etas == 0 {
            crate::options::default_max_etas(basis.len())
        } else {
            max_etas
        };
        let solve_scratch = lu::make_solve_scratch(&lu);
        Ok(Self {
            lu,
            eta_file: eta::EtaFile::new(effective_max_etas),
            basis_indices: basis.to_vec(),
            singular_basis: false,
            refactor_failed: false,
            solve_scratch,
        })
    }

    /// 再因子分解が必要かどうかを返す（eta蓄積数ベース）
    pub(crate) fn needs_refactor(&self) -> bool {
        self.eta_file.needs_refactor()
    }

    /// 蓄積された eta 行列の数を返す
    pub(crate) fn eta_count(&self) -> usize {
        self.eta_file.etas.len()
    }

    /// eta ファイルをクリアして強制的に基底行列を再因子分解する。
    ///
    /// 数値的に不安定なピボット（|pivot| / max_col が極めて小さい）の場合に
    /// 呼び出し元が使用する。成功すれば eta クリア + LU 更新、失敗は `refactor_failed = true`。
    pub(crate) fn force_refactor_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        match refactor::refactor_timed(a, basis, deadline) {
            Ok(new_lu) => {
                self.lu = new_lu;
                self.eta_file.etas.clear();
                self.basis_indices = basis.to_vec();
            }
            Err(crate::error::SolverError::SingularBasis { .. }) => {
                self.singular_basis = true;
                self.refactor_failed = true;
            }
            Err(_) => {
                // DeadlineExceeded など
                self.refactor_failed = true;
            }
        }
    }

    /// 数値安定性を検査し、必要であれば deadline 付きで基底行列を再因子分解する。
    ///
    /// # timeout audit fix
    /// refactor_if_needed の deadline 対応版。O(m²〜m³) の LU 再因子分解に
    /// deadline を渡すことで大規模 Simplex でのハングを防止する。
    /// 特異基底または deadline 超過どちらの場合も `refactor_failed = true` を設定する。
    pub(crate) fn refactor_if_needed_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        if self.eta_file.needs_refactor() {
            match refactor::refactor_timed(a, basis, deadline) {
                Ok(new_lu) => {
                    self.lu = new_lu;
                    self.eta_file.etas.clear();
                    self.basis_indices = basis.to_vec();
                }
                Err(crate::error::SolverError::SingularBasis { .. }) => {
                    // 特異基底: SingularBasis フラグを立て、呼び出し元が NumericalError を返せるようにする
                    self.singular_basis = true;
                    self.refactor_failed = true;
                }
                Err(_) => {
                    // DeadlineExceeded など: refactor_failed のみ
                    self.refactor_failed = true;
                }
            }
        }
    }
}

impl BasisManager for LuBasis {
    fn ftran(&mut self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        lu::solve_ftran(&self.lu, &mut dense);
        eta::apply_ftran(&self.eta_file.etas, &mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    fn btran(&mut self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        eta::apply_btran(&self.eta_file.etas, &mut dense);
        lu::solve_btran(&self.lu, &mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    fn ftran_dense(&mut self, rhs: &mut [f64]) {
        lu::solve_ftran_cached(&self.lu, rhs, &mut self.solve_scratch);
        eta::apply_ftran(&self.eta_file.etas, rhs);
    }

    fn btran_dense(&mut self, rhs: &mut [f64]) {
        eta::apply_btran(&self.eta_file.etas, rhs);
        lu::solve_btran_cached(&self.lu, rhs, &mut self.solve_scratch);
    }

    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec) {
        let eta = eta::add_eta_sparse(pivot_col, leaving_row);
        self.eta_file.etas.push(eta);
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
        // Step 1: FTRAN the entering column to get pivot column in basis space
        let entering_col_dense = vec![3.0, 1.0, 2.0]; // column 3 of A
        let mut pivot_sv = SparseVec::from_dense(&entering_col_dense);
        lb.ftran(&mut pivot_sv);

        // Step 2: Update basis
        lb.update(3, 1, &pivot_sv);

        // New basis = {0, 3, 2} → B_new = [[2,3,0],[1,1,1],[0,2,2]]
        // Verify FTRAN on new basis
        let rhs_orig = vec![5.0, 2.0, 4.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        // Verify B_new * x = rhs_orig
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
    fn test_lu_basis_refactor_after_50_etas() {
        // max_etas=50 で 50個のeta蓄積後にrefactorが発動すること
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 50).unwrap();

        // 初期状態ではrefactor不要
        assert!(
            !lb.eta_file.needs_refactor(),
            "Initially should not need refactor"
        );

        // 50個のetaを追加（max_etas=50 → needs_refactor() が true になる）
        for i in 0..50 {
            let r = i % 3;
            let mut pivot = vec![0.0f64, 0.0, 0.0];
            pivot[r] = 1.0;
            lb.eta_file.etas.push(eta::add_eta(&pivot, r));
        }
        assert!(
            lb.eta_file.needs_refactor(),
            "50 etas with max_etas=50 should trigger refactor"
        );

        // refactor後: etaクリア、ftran/btran正常動作
        lb.refactor_if_needed_timed(&a, &basis, None);
        assert!(
            !lb.eta_file.needs_refactor(),
            "After refactor, should not need refactor"
        );
        assert_eq!(
            lb.eta_file.etas.len(),
            0,
            "Etas should be cleared after refactor"
        );

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();
        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);

        // btranも確認
        let bt = a.transpose();
        let mut rhs_sv2 = SparseVec::from_dense(&rhs_orig);
        lb.btran(&mut rhs_sv2);
        let y = rhs_sv2.to_dense();
        let check2 = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_refactor() {
        // Test that refactor_if_needed works correctly
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis, 50).unwrap();

        // Set max_etas to 2 for easy testing
        lb.eta_file.max_etas = 2;

        // Add dummy etas to trigger refactor
        lb.eta_file.etas.push(eta::add_eta(&[1.0, 0.0, 0.0], 0));
        lb.eta_file.etas.push(eta::add_eta(&[0.0, 1.0, 0.0], 1));
        assert!(lb.eta_file.needs_refactor());

        // Refactor should reset etas
        lb.refactor_if_needed_timed(&a, &basis, None);
        assert!(!lb.eta_file.needs_refactor());
        assert_eq!(lb.eta_file.etas.len(), 0);

        // FTRAN should still work correctly after refactor
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }
}
