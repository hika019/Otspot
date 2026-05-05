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
///
/// **&self vs &mut self 設計**:
/// - `solve` は `&self`: predictor/corrector で同じ factorization に対し複数 RHS を
///   解く際、IPM 側が `&fac` を持ち回って複数関数に渡せるようにするため。MINRES の
///   reverberation 状態 (Lanczos vectors) は呼び出し毎にローカル確保。
/// - `refactor` は `&mut self`: cached factorization を置き換えるため。
pub trait KktSolver: Send {
    /// 1 つの右辺に対して `K · u = rhs` を解いて `sol` に書き込む。
    ///
    /// `deadline` は反復法のときだけ意味を持つ (直接法は事前因子化済みなので
    /// solve は速い)。実装は反復中に `Instant::now() >= deadline` を確認して
    /// 早期終了することを推奨。
    fn solve(
        &self,
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
        &self,
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
/// 前処理 M は問題に応じて 2 種類選択可能 (`PreconditionerKind`):
/// - `Jacobi`: M[i,i] = |K[i,i]|。最も簡単、汎用、対称鞍点系では弱い。
/// - `BlockDiag { n_top }`: 鞍点 K = [Q+R, A^T; A, -S] のブロック構造を活かし、
///   下部の対角を S の Schur 補集合で近似 (`sum_r A[i,r]²/(Q+R)[r,r] + (Σ_y+δ_y)[i]`)。
///   オフ対角の影響を反映するため Jacobi より大幅に効果的 (cond ~1e10 → ~1e5)。
///
/// メモリ: K の保持 (nnz × 16B) + 数本のワークベクトル (n × 8B × 数本) のみ。
/// fill-in は無いため、QPLIB_9008 級 (n=1M) でも GB 級メモリは不要。
pub struct PreconditionedMinres {
    /// K 行列の所有コピー (mat-vec 用)
    k: CscMatrix,
    /// M^{-1} の対角値
    m_inv_diag: Vec<f64>,
    /// 前処理の種類 (refactor 時に同じ kind で再構築するため保持)
    kind: PreconditionerKind,
    /// 最大反復数 (収束しない場合のフォールバック)
    max_iter: usize,
    /// 相対許容差
    tol: f64,
}

/// 前処理対角の最小値 (これ未満は MIN_DIAG にクランプして M^{-1} を有限化)
const MIN_DIAG: f64 = 1e-12;

/// 前処理の種類。`BlockDiag { n_top }` は鞍点 K = [top n_top × n_top; bottom m × m]
/// の構造を活かす。
#[derive(Debug, Clone, Copy)]
pub enum PreconditionerKind {
    /// 単純な K の対角絶対値 (汎用、対称鞍点系では弱い)
    Jacobi,
    /// ブロック対角 Jacobi: 下部対角を Schur 補集合の対角で近似
    BlockDiag { n_top: usize },
}

/// IPM の Newton 系内側解で使う inexact Newton forcing term η
/// (Eisenstat-Walker 1996 / Wright IPM §11.7)。
/// Newton 系 K·dx = r の内側解を ||K·dx − r|| ≤ η·||r|| まで許容する。
/// 標準的 IPM 文献では η = 0.1 〜 0.5 が推奨範囲。η = 0.1 は安全側の選択。
///
/// MINRES のデフォルト tol = 1e-9 は f64 機械精度近くまで内側 solve していたが、
/// IPM の外側収束にはこの精度は不要 (Newton ステップは外側残差に応じて段階的にしか
/// 進まない)。η = 0.1 で n=1M 級問題の MINRES iter 数を典型 100x 削減できる。
///
/// 単独 MINRES API (`PreconditionedMinres::new` / `with_block_diag`) のデフォルトは
/// 1e-9 を維持 (汎用線形ソルバとしての精度仕様)。本定数は IPM 経路の dispatcher
/// (`factorize_kkt_with_cached_perm`) からのみ使われる。
pub(crate) const MINRES_INEXACT_NEWTON_ETA: f64 = 0.1;

impl PreconditionedMinres {
    /// 単純 Jacobi 前処理付き MINRES を構築する (旧 `new` 互換)。
    pub fn new(k: CscMatrix) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: 2 * n, tol: 1e-9 }
    }

    /// 鞍点 K 用のブロック対角前処理付き MINRES を構築する。
    ///
    /// `n_top` は K の上部ブロック (Q+R) のサイズ。下部ブロックは
    /// `S = A·diag(Q+R)⁻¹·A^T + Σ_y+δ_y` の対角を近似計算する。
    ///
    /// 計算量: O(nnz(K))。fill-in なし。
    pub fn with_block_diag(k: CscMatrix, n_top: usize) -> Self {
        let kind = PreconditionerKind::BlockDiag { n_top };
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: 2 * n, tol: 1e-9 }
    }

    /// IPM Newton 系用 (inexact Newton tol η = 0.1)。鞍点ブロック対角前処理。
    /// `factorize_kkt_with_cached_perm` の MINRES フォールバックから使う想定。
    pub fn with_block_diag_inexact(k: CscMatrix, n_top: usize) -> Self {
        let kind = PreconditionerKind::BlockDiag { n_top };
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: 2 * n, tol: MINRES_INEXACT_NEWTON_ETA }
    }

    /// IPM Newton 系用 (inexact Newton tol η = 0.1)。Jacobi 前処理 (n_top 不明時)。
    pub fn new_inexact(k: CscMatrix) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: 2 * n, tol: MINRES_INEXACT_NEWTON_ETA }
    }

    /// 反復回数上限と相対許容差をカスタム指定する (Jacobi 前処理)。
    pub fn with_params(k: CscMatrix, max_iter: usize, tol: f64) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        Self { k, m_inv_diag, kind, max_iter, tol }
    }
}

/// 前処理 M^{-1} の対角を計算する (kind に応じて Jacobi または BlockDiag)。
fn compute_inv_diag(k: &CscMatrix, kind: PreconditionerKind) -> Vec<f64> {
    match kind {
        PreconditionerKind::Jacobi => compute_jacobi_inv_diag(k),
        PreconditionerKind::BlockDiag { n_top } => compute_block_diag_inv(k, n_top),
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

/// 鞍点 K のブロック対角前処理:
///   M_top[j]    = max(|K[j,j]|, MIN_DIAG)     for j < n_top
///   M_bottom[i] = sum_r A[i,r]² / M_top[r] + |K[n_top+i, n_top+i]|
///                                            for i = 0..(n - n_top)
///
/// K は対称上三角 CSC で格納されている前提。A^T entries は K の上三角の
/// (row r, col n_top+i) with r < n_top に格納。
fn compute_block_diag_inv(k: &CscMatrix, n_top: usize) -> Vec<f64> {
    let n_total = k.nrows;
    debug_assert!(n_top <= n_total);
    let m_bot = n_total - n_top;

    // 1. 上部ブロックの対角 |K[j,j]| (j < n_top)
    let mut top_diag = vec![MIN_DIAG; n_top];
    for j in 0..n_top {
        for k_idx in k.col_ptr[j]..k.col_ptr[j + 1] {
            if k.row_ind[k_idx] == j {
                top_diag[j] = k.values[k_idx].abs().max(MIN_DIAG);
                break;
            }
        }
    }

    // 2. 下部ブロックの Schur 補集合の対角:
    //    S_diag[i] = sum_r A[i,r]² / top_diag[r] + |K[n_top+i, n_top+i]|
    //    K の col n_top+i を走査し、row r < n_top の entry が A^T 値 (= A[i, r]).
    //    row == col のとき K の下部ブロック対角 (= -(Σ_y+δ_y)).
    let mut bot_diag = vec![MIN_DIAG; m_bot];
    for i in 0..m_bot {
        let col = n_top + i;
        let mut accum = 0.0_f64;
        for k_idx in k.col_ptr[col]..k.col_ptr[col + 1] {
            let r = k.row_ind[k_idx];
            let val = k.values[k_idx];
            if r < n_top {
                // A^T entry: K[r, col] = A[i, r]
                accum += (val * val) / top_diag[r];
            } else if r == col {
                // 下部ブロック対角 (負値)。絶対値で正の寄与に。
                accum += val.abs();
            }
            // else: r in [n_top, col), 下部 -S off-diagonal, ignore
        }
        bot_diag[i] = accum.max(MIN_DIAG);
    }

    // 3. M^{-1} = [1/top_diag; 1/bot_diag] を結合
    let mut m_inv = Vec::with_capacity(n_total);
    m_inv.extend(top_diag.iter().map(|&d| 1.0 / d));
    m_inv.extend(bot_diag.iter().map(|&d| 1.0 / d));
    m_inv
}

impl KktSolver for PreconditionedMinres {
    fn solve(
        &self,
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
        if std::env::var("MINRES_TRACE").ok().as_deref() == Some("1") {
            eprintln!("MINRES_SOLVE n={} iters={} max_iter={} tol={:.1e} resid={:.3e} conv={} kind={:?}",
                k.nrows, stats.iters, self.max_iter, self.tol, stats.residual_estimate, stats.converged, self.kind);
        }
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
        // 元の前処理 kind (Jacobi or BlockDiag) を維持して再構築する。
        self.m_inv_diag = compute_inv_diag(k, self.kind);
        self.k = k.clone();
        Ok(())
    }

    fn dim(&self) -> usize {
        self.k.nrows
    }
}

// ── KktFactor: 直接法 / 反復法を統一した最小 API でラップする「factor 互換」型 ─────

/// 既存の `LdlFactorizationAmd` 互換の `solve(&self, rhs, sol)` API を持ちつつ、
/// 内部で直接法 / 反復法を切り替えられる factor 型。
///
/// `LdlFactorizationAmd` を直接持ち回っている既存の IPM コード (predictor/corrector/
/// gondzio/IR) を最小変更で MINRES 対応にする目的で追加。trait オブジェクトでは
/// なく enum dispatch で書くことで、既存コードの `&LdlFactorizationAmd` を
/// `&KktFactor` に置換するだけで済む。
pub enum KktFactor {
    /// 直接法 (LDL, f64) で因子化済み
    Direct(crate::linalg::ldl::LdlFactorizationAmd),
    /// 直接法 (LDL, TwoFloat ≈ 106 bit) で因子化済み。
    /// f64 の precision floor (cond × 2.2e-16) を超える ill-conditioned 系
    /// (QPILOTNO cond 5e13 / QPLIB_10034 cond 1e7×amp 1170) 用。
    DirectDd(crate::linalg::ldl_dd::LdlFactorizationDdAmd),
    /// 反復法 (MINRES + Jacobi) で K の値を保持
    Iterative(PreconditionedMinres),
}

impl KktFactor {
    /// `K · sol = rhs` を解いて `sol` に書き込む。LDL 互換の infallible API。
    ///
    /// 反復法側のエラー (収束失敗等) は内部で握り潰し、best-effort 解を返す。
    /// (既存 `LdlFactorizationAmd::solve` も数値破綻時は NaN を含む解を返すだけで
    /// 破綻自体を return しない infallible API のため、整合)
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        self.solve_with_deadline(rhs, sol, None);
    }

    /// deadline を伝搬する版 `solve`。反復法経路では deadline 超過で
    /// 早期 break する (巨大問題で MINRES が無限ループに陥るのを防ぐ)。
    /// 直接法経路では deadline は無視 (LDL solve は十分速い)。
    pub fn solve_with_deadline(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) {
        match self {
            KktFactor::Direct(ldl) => ldl.solve(rhs, sol),
            KktFactor::DirectDd(ldl_dd) => ldl_dd.solve(rhs, sol),
            KktFactor::Iterative(minres) => {
                // MINRES は Result を返すが、IPM 内部では最大努力 best-effort で十分。
                // deadline で早期 break しても sol には反復途中の状態が入る。
                let _ = minres.solve(rhs, sol, deadline);
            }
        }
    }

    /// 直接法/反復法のどちらが使われているか診断する
    pub fn is_iterative(&self) -> bool {
        matches!(self, KktFactor::Iterative(_))
    }

    /// DD precision LDL を使っているか診断する
    pub fn is_dd(&self) -> bool {
        matches!(self, KktFactor::DirectDd(_))
    }
}

/// 既存の `factorize_quasidefinite_with_cached_perm` 互換シグネチャで、内部的に
/// budget 超過時に MINRES へフォールバックする factor を返す。
///
/// IPM の既存 LDL retry ループから 1:1 で差し替え可能なインターフェイス。
/// (置換目安: `ldl::factorize_quasidefinite_with_cached_perm(K, perm, deadline)`
///  → `factorize_kkt_with_cached_perm(K, perm, deadline, max_l_nnz_from_budget(), Some(n))`)
///
/// `n_top`: 鞍点 K = [Q+R, A^T; A, -S] の上部ブロックサイズ (= n)。`Some(n)` のとき
/// MINRES フォールバック側のブロック対角前処理が有効化される (Jacobi より大幅に
/// 効果的)。`None` のときは Jacobi 前処理 (旧互換)。
pub fn factorize_kkt_with_cached_perm(
    k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
) -> Result<KktFactor, KktError> {
    // env IPM_DD_LDL=1 が設定されているときは TwoFloat (≈106 bit) で因子化する。
    // f64 LDL の forward error = cond × 2.2e-16 が eps を超える ill-cond 系
    // (QPILOTNO cond 5e13 / QPLIB_10034 cond 1e7×amp 1170) 用。
    // budget チェックは f64 経路と同等 (DD は係数行列構造が同じなので同じ AMD/etree)。
    if std::env::var("IPM_DD_LDL").ok().as_deref() == Some("1") {
        // f64 で symbolic を済ませて budget 判定し、超過なら MINRES、超過しないなら DD numeric。
        match crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget(
            k, perm, deadline, Some(max_l_nnz),
        ) {
            Ok(_) => {
                // budget 内 → DD で再因子化
                match crate::linalg::ldl_dd::factorize_quasidefinite_with_cached_perm_dd(
                    k, perm, deadline,
                ) {
                    Ok(f) => {
                        if std::env::var("IPM_DD_LDL_TRACE").ok().as_deref() == Some("1") {
                            eprintln!(
                                "IPM_DD_LDL_TRACE factorize OK n={} L_nnz={}",
                                k.nrows,
                                f.nnz_l()
                            );
                        }
                        return Ok(KktFactor::DirectDd(f));
                    }
                    Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                        return Err(KktError::DeadlineExceeded);
                    }
                    Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                        return Err(KktError::SingularOrIndefinite);
                    }
                    Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                        // DD path は budget チェックなし。到達不可だが安全に MINRES へ。
                    }
                }
            }
            Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                // budget 超過 → DD でも同じく超過する。MINRES へフォールバック (下のロジック流用)。
            }
            Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                return Err(KktError::DeadlineExceeded);
            }
            Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                return Err(KktError::SingularOrIndefinite);
            }
        }
        // budget 超過時のみここに来る → MINRES
        let minres = match n_top {
            Some(n) if n <= k.nrows => PreconditionedMinres::with_block_diag_inexact(k.clone(), n),
            _ => PreconditionedMinres::new_inexact(k.clone()),
        };
        return Ok(KktFactor::Iterative(minres));
    }

    match crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget(
        k, perm, deadline, Some(max_l_nnz),
    ) {
        Ok(f) => Ok(KktFactor::Direct(f)),
        Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
            // budget 超過: MINRES にフォールバック (deadline は呼び出し側に伝搬しないが、
            // MINRES は solve 時に deadline 引数で制御される)。
            // IPM/IPPMM の Newton 系内側解として使うため inexact Newton tol η=0.1 を採用
            // (Wright IPM §11.7 / Eisenstat-Walker)。1e-9 は外側 IPM 収束に過剰精度。
            let minres = match n_top {
                Some(n) if n <= k.nrows => PreconditionedMinres::with_block_diag_inexact(k.clone(), n),
                _ => PreconditionedMinres::new_inexact(k.clone()),
            };
            Ok(KktFactor::Iterative(minres))
        }
        Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => Err(KktError::DeadlineExceeded),
        Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
            Err(KktError::SingularOrIndefinite)
        }
    }
}

// ── Auto dispatcher: try direct, fall back to iterative on memory budget ────────

/// 最後に使用したバックエンド種別 (診断 / 検証用)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KktBackend {
    /// 直接法 (LDL)
    Direct,
    /// 反復法 (MINRES + Jacobi)
    Iterative,
}

/// 自動切替 dispatcher: 直接法 (LDL) を試し、メモリ予算超過時は反復法 (MINRES) に
/// フォールバックする。
///
/// 判定:
/// - `refactor` 内で DirectLdl を `max_l_nnz_from_budget()` 制限付きで試行
/// - `WouldExceedMemory` を観測したら以降の `refactor` は反復法を使う
///   (一度 budget を超えた問題は次 Newton step でも同じく超えるため、
///    direct path を再度試みても無駄)
/// - 数値特異 / deadline 切れは上位に伝搬 (反復法でも救えない)
///
/// fallback 後も deadline を共有する: direct で消費した時間は反復法の budget
/// から自動的に減る (両方 deadline 引数で同じ Instant を見る)。
///
/// 「分岐は問題サイズの heuristic ではなく実測 L_nnz vs system memory の
/// 比較」という設計のため、Maros 138 (小〜中問題) は direct 経路で性能維持、
/// QPLIB_9008 (n=1M) は iterative 経路で OOM 回避という両立が自動で実現する。
pub struct AutoKktSolver {
    n: usize,
    /// 直接法バックエンド。一度 WouldExceedMemory が出たら `None` にして以降スキップ
    direct: Option<DirectLdl>,
    /// 反復法バックエンド。lazy init (direct が成功している間は不要)
    iterative: Option<PreconditionedMinres>,
    /// 最後に使ったバックエンド (`solve` でどちらを呼ぶか決める)
    last_used: Option<KktBackend>,
}

impl AutoKktSolver {
    /// 既定の memory budget (`KKT_MEMORY_BUDGET_BYTES` env または 4 GiB) で
    /// dispatcher を構築する。
    pub fn new(n: usize) -> Self {
        Self {
            n,
            direct: Some(DirectLdl::with_budget(n, max_l_nnz_from_budget())),
            iterative: None,
            last_used: None,
        }
    }

    /// memory budget を明示指定するコンストラクタ (主にテスト用)。
    pub fn with_budget(n: usize, max_l_nnz: usize) -> Self {
        Self {
            n,
            direct: Some(DirectLdl::with_budget(n, max_l_nnz)),
            iterative: None,
            last_used: None,
        }
    }

    /// 最後に使われたバックエンド種別を返す (診断用)。
    pub fn last_backend(&self) -> Option<KktBackend> {
        self.last_used
    }
}

impl KktSolver for AutoKktSolver {
    fn solve(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        match self.last_used {
            Some(KktBackend::Direct) => self
                .direct
                .as_ref()
                .ok_or(KktError::SingularOrIndefinite)?
                .solve(rhs, sol, deadline),
            Some(KktBackend::Iterative) => self
                .iterative
                .as_ref()
                .ok_or(KktError::SingularOrIndefinite)?
                .solve(rhs, sol, deadline),
            None => Err(KktError::SingularOrIndefinite),
        }
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.n || k.ncols != self.n {
            return Err(KktError::SingularOrIndefinite);
        }

        // Phase 1: direct を試す (まだ disable されていなければ)。
        if let Some(direct) = self.direct.as_mut() {
            match direct.refactor(k, deadline) {
                Ok(()) => {
                    self.last_used = Some(KktBackend::Direct);
                    return Ok(());
                }
                Err(KktError::WouldExceedMemory) => {
                    // 一度 budget 超えた問題は今後も超えるので direct を永久 disable。
                    self.direct = None;
                }
                Err(e) => return Err(e),
            }
        }

        // Phase 2: iterative にフォールバック (deadline は共有なので残時間で動く)。
        if self.iterative.is_none() {
            self.iterative = Some(PreconditionedMinres::new(k.clone()));
        } else {
            self.iterative.as_mut().unwrap().refactor(k, deadline)?;
        }
        self.last_used = Some(KktBackend::Iterative);
        Ok(())
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

    /// AutoKktSolver: budget 十分なら direct を選ぶ
    #[test]
    fn auto_uses_direct_when_budget_sufficient() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        // budget 10000 entries (2x2 では絶対に超過しない)
        let mut solver = AutoKktSolver::with_budget(2, 10000);
        solver.refactor(&k, None).expect("refactor");
        assert_eq!(solver.last_backend(), Some(KktBackend::Direct));

        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve");
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }

    /// AutoKktSolver: budget 不足なら iterative にフォールバック
    #[test]
    fn auto_falls_back_to_iterative_when_budget_exceeded() {
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[4.0, 0.5, 4.0, 0.5, 4.0, 0.3, -2.0, 0.4, -2.0],
            5, 5,
        ).unwrap();
        // budget 1 entry → 必ず超過 → iterative にフォールバック
        let mut solver = AutoKktSolver::with_budget(5, 1);
        solver.refactor(&k, None).expect("refactor (iterative)");
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative));

        let b = vec![1.0, 2.0, -1.0, 0.5, -0.5];
        let mut sol = vec![0.0; 5];
        solver.solve(&b, &mut sol, None).expect("solve");
        // LDL 解と比較
        let factor = crate::linalg::ldl::factorize_quasidefinite_with_amd(&k, None).unwrap();
        let mut sol_ldl = vec![0.0; 5];
        factor.solve(&b, &mut sol_ldl);
        for i in 0..5 {
            assert!(
                (sol[i] - sol_ldl[i]).abs() < 1e-6,
                "auto[{}]={} vs ldl[{}]={}", i, sol[i], i, sol_ldl[i]
            );
        }
    }

    /// AutoKktSolver: 一度 budget 超過したら以降の refactor も iterative を維持
    #[test]
    fn auto_remembers_iterative_after_first_overflow() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = AutoKktSolver::with_budget(2, 1);  // 必ず超過
        solver.refactor(&k1, None).unwrap();
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative));
        solver.refactor(&k2, None).unwrap();
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative),
            "should stay iterative after first overflow");
    }

    /// factorize_kkt_with_cached_perm: budget 十分なら Direct を返す
    #[test]
    fn factorize_kkt_chooses_direct_when_budget_sufficient() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let perm = crate::linalg::amd::amd_with_deadline(2, &k.col_ptr, &k.row_ind, None);
        let factor = factorize_kkt_with_cached_perm(&k, &perm, None, 1000, None)
            .expect("factor should succeed");
        assert!(matches!(factor, KktFactor::Direct(_)));
        assert!(!factor.is_iterative());

        let mut sol = vec![0.0; 2];
        factor.solve(&[3.0, 0.0], &mut sol);
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }

    /// factorize_kkt_with_cached_perm: budget 不足なら Iterative を返す
    #[test]
    fn factorize_kkt_chooses_iterative_when_budget_exceeded() {
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2], &[0, 1, 1, 2, 2],
            &[4.0, 0.5, 4.0, 0.5, -2.0], 3, 3
        ).unwrap();
        let perm = crate::linalg::amd::amd_with_deadline(3, &k.col_ptr, &k.row_ind, None);
        let factor = factorize_kkt_with_cached_perm(&k, &perm, None, 1, None)
            .expect("factor should succeed (fallback)");
        assert!(matches!(factor, KktFactor::Iterative(_)));
        assert!(factor.is_iterative());

        let b = vec![1.0, 2.0, -1.0];
        let mut sol = vec![0.0; 3];
        factor.solve(&b, &mut sol);
        // LDL 直接で解いた解と比較。dispatcher は inexact Newton tol η=0.1 を使うため
        // 厳密一致はしない。η * ||b|| の許容で確認。
        let factor_ldl = crate::linalg::ldl::factorize_quasidefinite_with_amd(&k, None).unwrap();
        let mut sol_ldl = vec![0.0; 3];
        factor_ldl.solve(&b, &mut sol_ldl);
        let b_inf = b.iter().map(|v: &f64| v.abs()).fold(0.0_f64, f64::max);
        let tol = MINRES_INEXACT_NEWTON_ETA * b_inf;
        for i in 0..3 {
            assert!(
                (sol[i] - sol_ldl[i]).abs() < tol.max(1e-6),
                "MINRES[{}]={} vs LDL[{}]={} (tol={})", i, sol[i], i, sol_ldl[i], tol
            );
        }
    }

    /// AutoKktSolver: trait object として動く (IPM 配線で使う形)
    #[test]
    fn auto_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver: Box<dyn KktSolver> = Box::new(AutoKktSolver::new(2));
        solver.refactor(&k, None).unwrap();
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }
}
