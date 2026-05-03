//! KKT 線形系 `K · u = rhs` を解くための統一 trait。
//!
//! IPM / IPPMM の Newton step で繰り返し現れる「対称鞍点 K に対する線形解」を
//! 抽象化する。直接法 (`DirectLdl`) と反復法 (`PreconditionedMinres`、別 commit)
//! が同じインターフェイスを実装し、問題特性に応じた dispatcher が選択する。
//!
//! 設計方針:
//! - **deadline は呼び出し側の責任**: solver 内部で deadline を検査し、超過時は
//!   `KktError::DeadlineExceeded` を返す (上位で best-so-far フォールバック)。
//! - **factorize と solve を分離**: 1 度因子化したら同じ K で複数 RHS を効率的に
//!   解ける (predictor/corrector 用)。`refactor` で K の値が変わったときに再計算。
//! - **失敗の種類を区別**: deadline 超過 / 数値特異 / メモリ予算超過 を区別して
//!   返し、上位の dispatcher がフォールバック判断に使う。
//!
//! 直接法と反復法の dispatcher は別モジュールで実装する (本 trait は機構のみ)。
//!
//! 設計時点 (2026-05-04, feature/kkt-iterative-2026-05-04) の用途想定:
//! - Maros 138 PASS 数 ≥ 136 を退行させずに、QPLIB_9008 (n=1M) の OOM kill を
//!   回避することが直接の目標。
//! - 汎用的に「行列の疎構造とメモリ予算」で direct/iterative を自動選択する基盤。

use crate::sparse::CscMatrix;
use std::time::Instant;

/// `KktSolver::solve` / `refactor` が返すエラー。
///
/// 上位の dispatcher は `WouldExceedMemory` / `SingularOrIndefinite` を見て
/// 反復法へフォールバック判断する。`DeadlineExceeded` はフォールバックしても
/// 残時間がない可能性が高いため、上位で Timeout として伝搬される想定。
#[non_exhaustive]
#[derive(Debug)]
pub enum KktError {
    /// 因子化が deadline 内に完了しなかった。
    DeadlineExceeded,
    /// 行列が特異または不定値で因子化に失敗した。
    /// 規則化 (δ 増加) で救う場合あり、上位責任で retry。
    SingularOrIndefinite,
    /// 因子化が利用可能メモリ予算を超過すると判定された
    /// (symbolic 段階の L_nnz 推定で検出)。反復法フォールバックの主トリガ。
    WouldExceedMemory,
    /// 反復法が最大反復数内に収束しなかった (反復法バックエンド固有)。
    DidNotConverge,
}

impl std::fmt::Display for KktError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KktError::DeadlineExceeded => write!(f, "KKT solver: deadline exceeded"),
            KktError::SingularOrIndefinite => write!(f, "KKT solver: singular or indefinite"),
            KktError::WouldExceedMemory => write!(f, "KKT solver: would exceed memory budget"),
            KktError::DidNotConverge => write!(f, "KKT solver: did not converge"),
        }
    }
}

impl std::error::Error for KktError {}

/// 対称鞍点 KKT 系 `K · u = rhs` を解くソルバの抽象化。
///
/// 実装は `factorize` (or `refactor`) で K を準備し、`solve` で 1 つ以上の
/// 右辺について解を返す。
pub trait KktSolver: Send {
    /// 1 つの右辺に対して `K · u = rhs` を解いて `sol` に書き込む。
    ///
    /// `deadline` は反復法のときだけ意味を持つ (直接法は事前因子化済みなので
    /// solve は速い)。実装は反復中に `Instant::now() >= deadline` を確認して
    /// 早期終了することを推奨。
    fn solve(
        &mut self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError>;

    /// 行列 K を更新して再因子化する (Newton step ごとに K の値が変わる前提)。
    ///
    /// 直接法: AMD + symbolic + numeric を実行。
    /// 反復法: K の参照を更新し、必要なら前処理を再構築。
    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError>;

    /// 行列の次元 (= n + m for KKT 鞍点系)。
    fn dim(&self) -> usize;
}

/// 既存の `factorize_quasidefinite_with_amd` を `KktSolver` trait に適合させる薄い wrapper。
///
/// 振る舞いは現行 LDL 経路と完全に同じ。trait abstraction を入れるだけで
/// IPM / IPPMM のロジックは不変であることを担保する (commit #1 の責務)。
pub struct DirectLdl {
    factor: Option<crate::linalg::ldl::LdlFactorizationAmd>,
    n: usize,
}

impl DirectLdl {
    /// 空の (まだ因子化されていない) `DirectLdl` を作る。`refactor` で使用可能になる。
    pub fn new(n: usize) -> Self {
        Self { factor: None, n }
    }

    /// 即座に行列 K を渡して因子化する (最初の Newton step 用ヘルパ)。
    pub fn from_matrix(k: &CscMatrix, deadline: Option<Instant>) -> Result<Self, KktError> {
        let mut s = Self::new(k.nrows);
        s.refactor(k, deadline)?;
        Ok(s)
    }

    /// 因子化済みの L_nnz を返す (fill-in 計測用、診断のみ)。
    /// 因子化前は 0。
    pub fn l_nnz(&self) -> usize {
        self.factor.as_ref().map_or(0, |f| f.nnz_l())
    }
}

impl KktSolver for DirectLdl {
    fn solve(
        &mut self,
        rhs: &[f64],
        sol: &mut [f64],
        _deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        // 直接法は事前因子化済みなので solve は速い (ms 単位)。
        // deadline チェックは省略 (1 solve が deadline を消費することは現実的に無い)。
        let factor = self.factor.as_ref().ok_or(KktError::SingularOrIndefinite)?;
        factor.solve(rhs, sol);
        Ok(())
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.n || k.ncols != self.n {
            return Err(KktError::SingularOrIndefinite);
        }
        match crate::linalg::ldl::factorize_quasidefinite_with_amd(k, deadline) {
            Ok(f) => {
                self.factor = Some(f);
                Ok(())
            }
            Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                self.factor = None;
                Err(KktError::DeadlineExceeded)
            }
            Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                self.factor = None;
                Err(KktError::SingularOrIndefinite)
            }
        }
    }

    fn dim(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// DirectLdl: 単純な 2x2 quasidefinite で因子化 + solve が動くこと
    #[test]
    fn directldl_2x2_solve_matches_hand_calc() {
        // K = [ 2  1 ]
        //     [ 1 -1 ]   (quasidefinite: top SPD, bottom -SPD)
        // 上三角 CSC: (0,0)=2, (0,1)=1, (1,1)=-1
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver = DirectLdl::from_matrix(&k, None).expect("factorize");
        // K · u = [3, 0]^T → 解析解 u = K^{-1} [3, 0]^T
        // det(K) = 2*(-1) - 1*1 = -3
        // K^{-1} = (1/-3) * [-1 -1; -1 2] = [1/3 1/3; 1/3 -2/3]
        // u = [1, 1]^T (行 1: 1/3*3 + 1/3*0 = 1; 行 2: 1/3*3 + (-2/3)*0 = 1)
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve");
        assert!((sol[0] - 1.0).abs() < 1e-10, "u[0]≈1, got {}", sol[0]);
        assert!((sol[1] - 1.0).abs() < 1e-10, "u[1]≈1, got {}", sol[1]);
        assert_eq!(solver.dim(), 2);
        assert!(solver.l_nnz() > 0, "L should have nonzeros after factorize");
    }

    /// DirectLdl: refactor で別の K に切替えて再 solve できること
    #[test]
    fn directldl_refactor_changes_k() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = DirectLdl::from_matrix(&k1, None).expect("factorize 1");
        let mut sol = vec![0.0; 2];

        // K1 · u = [2, -1]^T → u = [1, 1]^T
        solver.solve(&[2.0, -1.0], &mut sol, None).expect("solve 1");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);

        solver.refactor(&k2, None).expect("refactor");
        // K2 · u = [4, -2]^T → u = [1, 1]^T (K2 = 2 * K1)
        solver.solve(&[4.0, -2.0], &mut sol, None).expect("solve 2");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);
    }

    /// DirectLdl: 過去に経過した deadline を渡すと DeadlineExceeded を返すこと
    #[test]
    fn directldl_past_deadline_returns_deadline_exceeded() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let past = Instant::now() - std::time::Duration::from_secs(1);
        let result = DirectLdl::from_matrix(&k, Some(past));
        assert!(
            matches!(result, Err(KktError::DeadlineExceeded)),
            "past deadline should yield DeadlineExceeded, got {:?}",
            result.err()
        );
    }

    /// DirectLdl: 次元不整合の K を渡すと Err
    #[test]
    fn directldl_dim_mismatch_returns_err() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let mut solver = DirectLdl::new(3); // 3x3 を期待しているが渡される K は 2x2
        let result = solver.refactor(&k, None);
        assert!(result.is_err(), "dim mismatch should yield Err");
    }

    /// trait object として `Box<dyn KktSolver>` 経由で動くこと
    /// (将来 dispatcher が DirectLdl と MINRES を同じ Box に詰めるため)
    #[test]
    fn kkt_solver_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver: Box<dyn KktSolver> = Box::new(
            DirectLdl::from_matrix(&k, None).expect("factorize"),
        );
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve via trait");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);
    }
}
