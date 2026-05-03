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

/// 1 つの線形解 (LDL の L 値配列) が消費してよい既定メモリ予算 (バイト)。
///
/// 4 GiB を既定とする (現代的なマシンの RAM 16〜64 GB の 1/4〜1/16 に相当、
/// 同時に他プロセスが動いていてもクラッシュしない安全マージン)。
///
/// **これは「問題サイズの heuristic threshold」ではなくシステム resource policy**:
/// n=10 でも n=10M でも同じ式 `L_nnz × 16B vs budget` で判定する。問題依存の
/// マジックナンバー (例: `n+m ≥ 100k`) は使わない。
///
/// 環境変数 `KKT_MEMORY_BUDGET_BYTES` で上書き可。CI / 制約環境では小さく、
/// hi-mem サーバでは大きく設定できる。
const DEFAULT_MEMORY_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// LDL の L 値 1 entry あたりのバイト数。f64 = 8B + row index usize = 8B (上限見積り)。
const BYTES_PER_L_ENTRY: usize = 16;

/// 現在の memory budget (バイト) を返す。env > default の優先順位。
pub fn memory_budget_bytes() -> usize {
    std::env::var("KKT_MEMORY_BUDGET_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MEMORY_BUDGET_BYTES)
}

/// memory budget を L_nnz 上限に変換する。
pub fn max_l_nnz_from_budget() -> usize {
    memory_budget_bytes() / BYTES_PER_L_ENTRY
}

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

/// 既存の `factorize_quasidefinite_with_amd_budget` を `KktSolver` trait に
/// 適合させる薄い wrapper。
///
/// `max_l_nnz` を `None` で構築すると budget チェック無しで現行と完全互換。
/// `Some` で構築すると symbolic 段階で L_nnz が超過したとき
/// `KktError::WouldExceedMemory` を返し、上位の dispatcher が反復法へ
/// フォールバックする判断材料にする。
pub struct DirectLdl {
    factor: Option<crate::linalg::ldl::LdlFactorizationAmd>,
    n: usize,
    /// L_nnz 上限。`None` のとき budget チェックなし (現行互換)。
    max_l_nnz: Option<usize>,
}

impl DirectLdl {
    /// 空の (まだ因子化されていない) `DirectLdl` を作る。`refactor` で使用可能になる。
    /// budget チェックなし版。
    pub fn new(n: usize) -> Self {
        Self { factor: None, n, max_l_nnz: None }
    }

    /// memory budget を持つ `DirectLdl` を作る。symbolic 段階で L_nnz が
    /// 超過したら `refactor` が `WouldExceedMemory` を返す。
    pub fn with_budget(n: usize, max_l_nnz: usize) -> Self {
        Self { factor: None, n, max_l_nnz: Some(max_l_nnz) }
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
        match crate::linalg::ldl::factorize_quasidefinite_with_amd_budget(
            k, deadline, self.max_l_nnz,
        ) {
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
            Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                self.factor = None;
                Err(KktError::WouldExceedMemory)
            }
        }
    }

    fn dim(&self) -> usize {
        self.n
    }
}

// ── Iterative backend: PreconditionedMinres ──────────────────────────────────────

/// 対角前処理付き MINRES による反復解法。LDL の fill-in を回避する代替経路。
///
/// 前処理 M は K の対角の絶対値: `M[i,i] = max(|K[i,i]|, MIN_DIAG)`。
/// 不定値 K (鞍点系) で SPD M を構成するために絶対値を取る (Jacobi-style)。
/// 高度な前処理 (block-diagonal、constraint preconditioner) は将来課題。
///
/// メモリ: K の保持 (nnz × 16B) + 数本のワークベクトル (n × 8B × 数本) のみ。
/// fill-in は無いため、QPLIB_9008 級 (n=1M) でも GB 級メモリは不要。
pub struct PreconditionedMinres {
    /// K 行列の所有コピー (mat-vec 用)
    k: CscMatrix,
    /// M^{-1} の対角値 (= 1.0 / max(|K[i,i]|, MIN_DIAG))
    m_inv_diag: Vec<f64>,
    /// 最大反復数 (収束しない場合のフォールバック)
    max_iter: usize,
    /// 相対許容差
    tol: f64,
}

/// 前処理対角の最小値 (これ未満は MIN_DIAG にクランプして M^{-1} を有限化)
const MIN_DIAG: f64 = 1e-12;

impl PreconditionedMinres {
    /// 行列 K を持つ MINRES ソルバを構築する。
    ///
    /// `max_iter` のデフォルトは `2 * n` (= Krylov 部分空間が完全に満たされる
    /// 上限、CG のような有限終了の保証は無いが実用上十分)。
    /// `tol` はデフォルト `1e-9` (IPM Newton step の収束精度として標準)。
    pub fn new(k: CscMatrix) -> Self {
        let n = k.nrows;
        let m_inv_diag = compute_jacobi_inv_diag(&k);
        Self {
            k,
            m_inv_diag,
            max_iter: 2 * n,
            tol: 1e-9,
        }
    }

    /// 反復回数上限と相対許容差をカスタム指定する。
    pub fn with_params(k: CscMatrix, max_iter: usize, tol: f64) -> Self {
        let m_inv_diag = compute_jacobi_inv_diag(&k);
        Self { k, m_inv_diag, max_iter, tol }
    }
}

/// K の対角絶対値から Jacobi 前処理 M^{-1} を構築する。
fn compute_jacobi_inv_diag(k: &CscMatrix) -> Vec<f64> {
    let n = k.nrows;
    let mut diag_abs = vec![MIN_DIAG; n];
    for j in 0..n {
        for k_idx in k.col_ptr[j]..k.col_ptr[j + 1] {
            if k.row_ind[k_idx] == j {
                let v = k.values[k_idx].abs();
                diag_abs[j] = v.max(MIN_DIAG);
                break;
            }
        }
    }
    diag_abs.iter().map(|&d| 1.0 / d).collect()
}

impl KktSolver for PreconditionedMinres {
    fn solve(
        &mut self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        // 解の初期推定はゼロ (warm-start サポートは将来課題)
        for s in sol.iter_mut() { *s = 0.0; }
        let k = &self.k;
        let m_inv = &self.m_inv_diag;
        let stats = crate::linalg::minres::pminres(
            |v, y| crate::linalg::minres::matvec_sym_upper(k, v, y),
            |r, z| {
                for i in 0..r.len() {
                    z[i] = r[i] * m_inv[i];
                }
            },
            rhs,
            sol,
            self.tol,
            self.max_iter,
            || deadline.is_some_and(|d| Instant::now() >= d),
        );
        if stats.converged {
            Ok(())
        } else if deadline.is_some_and(|d| Instant::now() >= d) {
            Err(KktError::DeadlineExceeded)
        } else {
            Err(KktError::DidNotConverge)
        }
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        _deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.dim() || k.ncols != self.dim() {
            return Err(KktError::SingularOrIndefinite);
        }
        // K の値が変わったので前処理を更新。fill-in は無いのでこれだけで OK。
        self.m_inv_diag = compute_jacobi_inv_diag(k);
        self.k = k.clone();
        Ok(())
    }

    fn dim(&self) -> usize {
        self.k.nrows
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

    /// memory budget を持つ DirectLdl: 巨大 L_nnz を要求する K で WouldExceedMemory を返す
    #[test]
    fn directldl_with_tight_budget_returns_would_exceed_memory() {
        // 5x5 quasidefinite (上3 SPD, 下2 -SPD)
        // K = diag(1,1,1,-1,-1) + 少しの off-diag で fill-in
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 0, 1, 2, 3, 3, 4],
            &[0, 1, 1, 2, 2, 2, 3, 4, 4],
            &[1.0, 0.1, 1.0, 0.1, 0.1, 1.0, -1.0, 0.1, -1.0],
            5, 5,
        ).unwrap();
        // budget = 1 entry のみ → 必ず超過
        let mut solver = DirectLdl::with_budget(5, 1);
        let result = solver.refactor(&k, None);
        assert!(
            matches!(result, Err(KktError::WouldExceedMemory)),
            "tight budget (1 entry) should trigger WouldExceedMemory, got {:?}",
            result.err()
        );
    }

    /// memory budget が None (= 既定の現行互換モード) では budget 超過を起こさない
    #[test]
    fn directldl_without_budget_is_unconstrained() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver = DirectLdl::new(2);  // budget なし
        solver.refactor(&k, None).expect("no-budget refactor should always succeed for valid K");
    }

    /// memory_budget_bytes() は env 上書きが効くこと
    #[test]
    fn memory_budget_env_override() {
        // SAFETY: 並列テストでも他テストが同じ env を読まないため安全 (本テストでセット&クリア)
        // テスト並列実行で環境変数が干渉する可能性があるが、一意な値で観測する
        let unique_value = "12345678";
        std::env::set_var("KKT_MEMORY_BUDGET_BYTES", unique_value);
        let observed = memory_budget_bytes();
        std::env::remove_var("KKT_MEMORY_BUDGET_BYTES");
        assert_eq!(observed, 12345678, "env override should be respected");
        // 削除後は default に戻る
        let after_unset = memory_budget_bytes();
        assert_eq!(after_unset, 4 * 1024 * 1024 * 1024, "default = 4 GiB");
    }

    /// max_l_nnz_from_budget は budget をバイト→entry 数に変換する
    #[test]
    fn max_l_nnz_from_budget_conversion() {
        std::env::set_var("KKT_MEMORY_BUDGET_BYTES", "1600");  // 1600 / 16 = 100 entries
        let l = max_l_nnz_from_budget();
        std::env::remove_var("KKT_MEMORY_BUDGET_BYTES");
        assert_eq!(l, 100);
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

    /// PreconditionedMinres: 対称不定値 2x2 で正解を返す
    #[test]
    fn minres_kkt_2x2_indefinite() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver = PreconditionedMinres::new(k);
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("MINRES solve");
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
        assert_eq!(solver.dim(), 2);
    }

    /// PreconditionedMinres: 5x5 quasidef、DirectLdl と数値一致
    #[test]
    fn minres_kkt_5x5_matches_direct_ldl() {
        let entries = [
            (0, 0, 4.0), (0, 1, 0.5), (1, 1, 4.0), (1, 2, 0.5), (2, 2, 4.0),
            (0, 3, 0.3),
            (3, 3, -2.0), (3, 4, 0.4), (4, 4, -2.0),
        ];
        let rows: Vec<usize> = entries.iter().map(|(r, _, _)| *r).collect();
        let cols: Vec<usize> = entries.iter().map(|(_, c, _)| *c).collect();
        let vals: Vec<f64> = entries.iter().map(|(_, _, v)| *v).collect();
        let k = CscMatrix::from_triplets(&rows, &cols, &vals, 5, 5).unwrap();
        let b = vec![1.0, 2.0, -1.0, 0.5, -0.5];

        let mut x_ldl = vec![0.0; 5];
        let mut ldl_solver = DirectLdl::from_matrix(&k, None).unwrap();
        ldl_solver.solve(&b, &mut x_ldl, None).unwrap();

        let mut x_minres = vec![0.0; 5];
        let mut minres_solver = PreconditionedMinres::new(k);
        minres_solver.solve(&b, &mut x_minres, None).expect("MINRES solve");

        for i in 0..5 {
            assert!(
                (x_ldl[i] - x_minres[i]).abs() < 1e-6,
                "x[{}]: LDL={}, MINRES={}", i, x_ldl[i], x_minres[i]
            );
        }
    }

    /// PreconditionedMinres: refactor で K を更新できる
    #[test]
    fn minres_kkt_refactor() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = PreconditionedMinres::new(k1);
        let mut sol = vec![0.0; 2];

        solver.solve(&[2.0, -1.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);

        solver.refactor(&k2, None).expect("refactor");
        solver.solve(&[4.0, -2.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
    }

    /// PreconditionedMinres: 過去 deadline で DeadlineExceeded
    #[test]
    fn minres_kkt_past_deadline() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let mut solver = PreconditionedMinres::new(k);
        let mut sol = vec![0.0; 2];
        let past = Instant::now() - std::time::Duration::from_secs(1);
        let result = solver.solve(&[1.0, 1.0], &mut sol, Some(past));
        assert!(
            matches!(result, Err(KktError::DeadlineExceeded)),
            "past deadline should yield DeadlineExceeded, got {:?}", result.err()
        );
    }

    /// PreconditionedMinres: trait object として動く (DirectLdl と同じ box に詰められる)
    #[test]
    fn minres_kkt_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver: Box<dyn KktSolver> = Box::new(PreconditionedMinres::new(k));
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve via trait");
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
    }
}
