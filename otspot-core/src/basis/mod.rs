//! 改訂単体法（Revised Simplex）の基底管理モジュール
//!
//! 基底行列 B の LU 分解を管理し、FTRAN・BTRAN ソルブと
//! ピボット更新（eta ファイル）および定期的な再因子分解をサポートする。

pub(crate) mod eta;
pub(crate) mod ft;
pub(crate) mod lu;
pub(crate) mod refactor;

#[cfg(test)]
pub(crate) mod test_utils;

use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use std::time::Instant;
pub(crate) use ft::FtLu;

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
        Ok(Self {
            lu,
            eta_file: eta::EtaFile::new(effective_max_etas),
            basis_indices: basis.to_vec(),
            singular_basis: false,
            refactor_failed: false,
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
        lu::solve_ftran(&self.lu, rhs);
        eta::apply_ftran(&self.eta_file.etas, rhs);
    }

    fn btran_dense(&mut self, rhs: &mut [f64]) {
        eta::apply_btran(&self.eta_file.etas, rhs);
        lu::solve_btran(&self.lu, rhs);
    }

    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec) {
        let eta = eta::add_eta_sparse(pivot_col, leaving_row);
        self.eta_file.etas.push(eta);
        self.basis_indices[leaving_row] = entering_col;
    }
}

// ---- FtLuBasis ----------------------------------------------------------

/// `FtLu` をラップし `BasisMgr` 経由で simplex に接続する構造体。
///
/// 小 pivot や FT 操作蓄積による再分解シグナル、および
/// `singular_basis` / `refactor_failed` フラグを管理する。
pub(crate) struct FtLuBasis {
    ft: FtLu,
    /// FT 操作数の上限 (LuBasis の max_etas に相当)。
    max_ft_ops: usize,
    /// 再因子分解が SingularBasis で失敗した場合 true。
    pub(crate) singular_basis: bool,
    /// 再因子分解が失敗した場合 true (SingularBasis / DeadlineExceeded)。
    pub(crate) refactor_failed: bool,
}

impl FtLuBasis {
    fn new_timed(
        a: &CscMatrix,
        basis: &[usize],
        max_etas: usize,
        deadline: Option<Instant>,
    ) -> Result<Self, SolverError> {
        let ft = FtLu::new_timed(a, basis, deadline)?;
        let effective = if max_etas == 0 {
            crate::options::default_max_etas(basis.len())
        } else {
            max_etas
        };
        Ok(Self {
            ft,
            max_ft_ops: effective,
            singular_basis: false,
            refactor_failed: false,
        })
    }

    /// eta 蓄積または FtLu 内部フラグによる再分解要求。
    pub(crate) fn needs_refactor(&self) -> bool {
        self.ft.needs_refactor() || self.ft.ft_op_count() >= self.max_ft_ops
    }

    /// 蓄積 FT 操作数 (LuBasis::eta_count() 相当)。
    pub(crate) fn eta_count(&self) -> usize {
        self.ft.ft_op_count()
    }

    /// 条件付き再因子分解 (needs_refactor() が true の場合のみ実行)。
    pub(crate) fn refactor_if_needed_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        if self.needs_refactor() {
            self.do_refactor(a, basis, deadline);
        }
    }

    /// 強制再因子分解。
    pub(crate) fn force_refactor_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        self.do_refactor(a, basis, deadline);
    }

    fn do_refactor(&mut self, a: &CscMatrix, basis: &[usize], deadline: Option<Instant>) {
        match FtLu::new_timed(a, basis, deadline) {
            Ok(new_ft) => {
                self.ft = new_ft;
                self.singular_basis = false;
                self.refactor_failed = false;
            }
            Err(SolverError::SingularBasis { .. }) => {
                self.singular_basis = true;
                self.refactor_failed = true;
            }
            Err(_) => {
                self.refactor_failed = true;
            }
        }
    }
}

impl BasisManager for FtLuBasis {
    fn ftran(&mut self, rhs: &mut SparseVec) {
        self.ft.ftran_sparse(rhs);
    }

    fn btran(&mut self, rhs: &mut SparseVec) {
        self.ft.btran_sparse(rhs);
    }

    fn ftran_dense(&mut self, rhs: &mut [f64]) {
        self.ft.ftran(rhs);
    }

    fn btran_dense(&mut self, rhs: &mut [f64]) {
        self.ft.btran(rhs);
    }

    /// FT 更新: `pivot_col` は無視し、FtLu が内部で spike を再計算する。
    /// `FtLu::update` が `Err` を返した場合、`needs_refactor` フラグが立ち、
    /// 後続の `refactor_if_needed_timed` 呼び出しで再分解される。
    fn update(&mut self, entering_col: usize, leaving_row: usize, _pivot_col: &SparseVec) {
        let _ = self.ft.update(entering_col, leaving_row);
    }
}

// ---- BasisMgr enum -------------------------------------------------------

/// 基底管理バックエンドの選択 enum。
///
/// `Lu`: 従来の eta ファイル付き LU (既定・完全不変)。
/// `Ft`: Bartels-Golub-Reid FT-LU 更新 (実験的・`use_ft_basis=true` 時)。
pub(crate) enum BasisMgr {
    Lu(Box<LuBasis>),
    Ft(Box<FtLuBasis>),
}

impl BasisMgr {
    /// 構築: `use_ft` が `false` なら `LuBasis`、`true` なら `FtLuBasis`。
    pub(crate) fn new_timed(
        a: &CscMatrix,
        basis: &[usize],
        max_etas: usize,
        deadline: Option<Instant>,
        use_ft: bool,
    ) -> Result<Self, SolverError> {
        if use_ft {
            Ok(BasisMgr::Ft(Box::new(FtLuBasis::new_timed(a, basis, max_etas, deadline)?)))
        } else {
            Ok(BasisMgr::Lu(Box::new(LuBasis::new_timed(a, basis, max_etas, deadline)?)))
        }
    }

    /// 再分解が必要か。
    pub(crate) fn needs_refactor(&self) -> bool {
        match self {
            BasisMgr::Lu(lb) => lb.needs_refactor(),
            BasisMgr::Ft(fb) => fb.needs_refactor(),
        }
    }

    /// 蓄積 eta/ft-op 数 (新鮮な因子分解後は 0)。
    pub(crate) fn eta_count(&self) -> usize {
        match self {
            BasisMgr::Lu(lb) => lb.eta_count(),
            BasisMgr::Ft(fb) => fb.eta_count(),
        }
    }

    /// 条件付き再因子分解。
    pub(crate) fn refactor_if_needed_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        match self {
            BasisMgr::Lu(lb) => lb.refactor_if_needed_timed(a, basis, deadline),
            BasisMgr::Ft(fb) => fb.refactor_if_needed_timed(a, basis, deadline),
        }
    }

    /// 強制再因子分解。
    pub(crate) fn force_refactor_timed(
        &mut self,
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) {
        match self {
            BasisMgr::Lu(lb) => lb.force_refactor_timed(a, basis, deadline),
            BasisMgr::Ft(fb) => fb.force_refactor_timed(a, basis, deadline),
        }
    }

    /// 特異基底フラグ。
    pub(crate) fn singular_basis(&self) -> bool {
        match self {
            BasisMgr::Lu(lb) => lb.singular_basis,
            BasisMgr::Ft(fb) => fb.singular_basis,
        }
    }

    /// 再因子分解失敗フラグ。
    pub(crate) fn refactor_failed(&self) -> bool {
        match self {
            BasisMgr::Lu(lb) => lb.refactor_failed,
            BasisMgr::Ft(fb) => fb.refactor_failed,
        }
    }
}

impl BasisManager for BasisMgr {
    fn ftran(&mut self, rhs: &mut SparseVec) {
        match self {
            BasisMgr::Lu(lb) => lb.ftran(rhs),
            BasisMgr::Ft(fb) => fb.ftran(rhs),
        }
    }

    fn btran(&mut self, rhs: &mut SparseVec) {
        match self {
            BasisMgr::Lu(lb) => lb.btran(rhs),
            BasisMgr::Ft(fb) => fb.btran(rhs),
        }
    }

    fn ftran_dense(&mut self, rhs: &mut [f64]) {
        match self {
            BasisMgr::Lu(lb) => lb.ftran_dense(rhs),
            BasisMgr::Ft(fb) => fb.ftran_dense(rhs),
        }
    }

    fn btran_dense(&mut self, rhs: &mut [f64]) {
        match self {
            BasisMgr::Lu(lb) => lb.btran_dense(rhs),
            BasisMgr::Ft(fb) => fb.btran_dense(rhs),
        }
    }

    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec) {
        match self {
            BasisMgr::Lu(lb) => lb.update(entering_col, leaving_row, pivot_col),
            BasisMgr::Ft(fb) => fb.update(entering_col, leaving_row, pivot_col),
        }
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

    // ── BasisMgr::Ft (FtLuBasis) integration tests ───────────────────────

    /// FTRAN and BTRAN via BasisMgr::Ft must agree with BasisMgr::Lu on the same
    /// matrix and RHS.
    #[test]
    fn ft_basis_mgr_ftran_btran_matches_lu() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0usize, 1, 2];

        let mut bm_lu = BasisMgr::new_timed(&a, &basis, 50, None, false).unwrap();
        let mut bm_ft = BasisMgr::new_timed(&a, &basis, 50, None, true).unwrap();

        let rhs = vec![3.0, 5.0, 3.0];

        // FTRAN
        let mut sv_lu = SparseVec::from_dense(&rhs);
        bm_lu.ftran(&mut sv_lu);
        let x_lu = sv_lu.to_dense();

        let mut sv_ft = SparseVec::from_dense(&rhs);
        bm_ft.ftran(&mut sv_ft);
        let x_ft = sv_ft.to_dense();

        assert_vec_near(&x_ft, &x_lu, 1e-10);

        // Verify B * x = rhs
        let check = a.mat_vec_mul(&x_ft).unwrap();
        assert_vec_near(&check, &rhs, 1e-10);

        // BTRAN
        let mut sv_lu2 = SparseVec::from_dense(&rhs);
        bm_lu.btran(&mut sv_lu2);
        let y_lu = sv_lu2.to_dense();

        let mut sv_ft2 = SparseVec::from_dense(&rhs);
        bm_ft.btran(&mut sv_ft2);
        let y_ft = sv_ft2.to_dense();

        assert_vec_near(&y_ft, &y_lu, 1e-10);

        // Verify B^T * y = rhs
        let bt = a.transpose();
        let check_bt = bt.mat_vec_mul(&y_ft).unwrap();
        assert_vec_near(&check_bt, &rhs, 1e-10);
    }

    /// After one pivot update, BasisMgr::Ft FTRAN must still satisfy B_new * x = rhs.
    #[test]
    fn ft_basis_mgr_update_correct() {
        // A = [[2,1,0,3],[1,3,1,1],[0,1,2,2]], initial basis [0,1,2]
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis_init = vec![0usize, 1, 2];
        let mut bm = BasisMgr::new_timed(&a, &basis_init, 50, None, true).unwrap();

        // Pivot: entering col=3, leaving row=1. pivot_col ignored by FtLuBasis.
        let dummy_pivot = SparseVec { indices: vec![], values: vec![], len: 3 };
        bm.update(3, 1, &dummy_pivot);

        // New basis = [0, 3, 2].
        // B_new = [[2,3,0],[1,1,1],[0,2,2]].
        let b_new_dense = vec![
            vec![2.0, 3.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 2.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);

        let rhs = vec![5.0, 2.0, 4.0];
        let mut sv = SparseVec::from_dense(&rhs);
        bm.ftran(&mut sv);
        let x = sv.to_dense();

        let check = b_new.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs, 1e-9);
    }

    /// After accumulating `max_ft_ops` FT operations, `needs_refactor()` must
    /// return `true`; calling `refactor_if_needed_timed` must reset it and leave
    /// FTRAN correct on the new basis.
    #[test]
    fn ft_basis_mgr_needs_refactor_refactor_path() {
        // Dense 3x4 matrix; initial basis = [0,1,2].
        // Update enter=3, leave=0 → bump case (spike has entries below row 0)
        // → ft_ops appended. With max_etas=1, needs_refactor() becomes true.
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis_init = vec![0usize, 1, 2];
        let mut bm = BasisMgr::new_timed(&a, &basis_init, 1, None, true).unwrap();
        assert!(!bm.needs_refactor(), "fresh FT basis must not need refactor");

        let dummy_pivot = SparseVec { indices: vec![], values: vec![], len: 3 };
        bm.update(3, 0, &dummy_pivot);

        // After the update: ft_op_count >= 1 == max_ft_ops OR ft.needs_refactor set.
        assert!(
            bm.needs_refactor(),
            "after reaching max_ft_ops, needs_refactor() must be true"
        );

        let new_basis = vec![3usize, 1, 2];
        bm.refactor_if_needed_timed(&a, &new_basis, None);
        assert!(!bm.refactor_failed(), "refactor must succeed on non-singular basis");
        assert!(!bm.needs_refactor(), "after refactor, needs_refactor() must be false");

        // FTRAN must still be correct for the new basis B_new = [[3,1,0],[1,3,1],[2,1,2]].
        let b_new_dense = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![2.0, 1.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);
        let rhs = vec![4.0, 7.0, 5.0];
        let mut sv = SparseVec::from_dense(&rhs);
        bm.ftran(&mut sv);
        let x = sv.to_dense();
        let check = b_new.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs, 1e-9);
    }

    /// Solve a small LP with equality constraints using `use_ft_basis=true` and
    /// verify the objective matches the LU-backend result.
    ///
    /// LP: min 2x + 3y  s.t. x + y = 4, x,y >= 0  → optimal obj = 8 (x=4, y=0).
    #[test]
    fn ft_basis_lp_eq_obj_matches_lu() {
        use crate::options::SolverOptions;
        use crate::problem::{ConstraintType, LpProblem, SolveStatus};
        use crate::sparse::CscMatrix;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![2.0, 3.0],
            a,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let opts_lu = SolverOptions { use_ft_basis: false, ..SolverOptions::default() };
        let opts_ft = SolverOptions { use_ft_basis: true,  ..SolverOptions::default() };

        let r_lu = crate::lp::solve_lp_with(&lp, &opts_lu);
        let r_ft = crate::lp::solve_lp_with(&lp, &opts_ft);

        assert_eq!(r_lu.status, SolveStatus::Optimal);
        assert_eq!(r_ft.status, SolveStatus::Optimal);
        assert!(
            (r_ft.objective - r_lu.objective).abs() < 1e-6,
            "FT obj {} != LU obj {}",
            r_ft.objective, r_lu.objective
        );
    }

    /// Solve an LP with upper bounds using `use_ft_basis=true`; objective must
    /// match the LU result.
    ///
    /// LP: min x + 2y  s.t. x + y <= 5, x <= 3, x,y >= 0
    /// → optimal: x=3, y=0, obj=3.
    #[test]
    fn ft_basis_lp_ub_obj_matches_lu() {
        use crate::options::SolverOptions;
        use crate::problem::{ConstraintType, LpProblem, SolveStatus};
        use crate::sparse::CscMatrix;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 3.0), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let opts_lu = SolverOptions { use_ft_basis: false, ..SolverOptions::default() };
        let opts_ft = SolverOptions { use_ft_basis: true,  ..SolverOptions::default() };

        let r_lu = crate::lp::solve_lp_with(&lp, &opts_lu);
        let r_ft = crate::lp::solve_lp_with(&lp, &opts_ft);

        assert_eq!(r_lu.status, SolveStatus::Optimal);
        assert_eq!(r_ft.status, SolveStatus::Optimal);
        assert!(
            (r_ft.objective - r_lu.objective).abs() < 1e-6,
            "FT obj {} != LU obj {}",
            r_ft.objective, r_lu.objective
        );
    }
}
