//! ソルバー設定パラメータモジュール
//!
//! [`SolverOptions`] を通じてシンプレックス法の動作を制御する。
//! 許容誤差・反復上限・リファクタリング頻度などを一元管理する。
//!
//! ## ソルバー固有オプション
//!
//! ソルバー固有パラメータは各サブ構造体で管理する:
//! - ADMM: [`SolverOptions::admm`] ([`AdmmOptions`])
//! - IPM: [`SolverOptions::ipm`] ([`IpmOptions`])
//! - Active Set: [`SolverOptions::active_set`] ([`ActiveSetOptions`])

use crate::tolerances::*;
use std::sync::{
    atomic::AtomicBool,
    Arc,
};
use std::time::Instant;

/// QP ソルバー選択
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpSolverChoice {
    /// Auto: n < qp_solver_threshold → Active Set, n >= threshold → IPM→ADMMフォールバック
    #[default]
    Auto,
    /// 強制 ADMM
    Admm,
    /// 強制 Active Set
    ActiveSet,
    /// 強制 IPM (内点法)
    Ipm,
    /// 強制 IPM→AS クロスオーバー
    IpmCrossover,
    /// 強制 MINRES+制約前処理 IPM (n > LDL_THRESHOLD でも強制使用)
    IpmMinres,
    /// 強制 IPM Schur complement パス（--features parallel なしでも動作）
    IpmSchur,
}

/// シンプレックス法の選択
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplexMethod {
    /// デフォルト: warm-startの有無で自動選択
    #[default]
    Auto,
    /// 強制的にPrimal Simplex
    Primal,
    /// 強制的にDual Simplex
    Dual,
}

/// warm-start用の基底情報
///
/// 前の最適解から基底情報を保持し、次のLP求解時にDual Simplexの
/// 初期基底として使用する。SQP統合時の主要インターフェース。
#[derive(Debug, Clone)]
pub struct WarmStartBasis {
    /// 基底変数のインデックスリスト（標準形の列番号、長さ = m）
    pub basis: Vec<usize>,
    /// 基底変数の値 x_B（長さ = m）
    /// warm-start時、新しいRHSで再計算されるため、古い値でもよい
    pub x_b: Vec<f64>,
}

/// QP ソルバーの収束精度を抽象化する列挙型
///
/// 各ソルバーは `Tolerance` を内部の収束基準に変換して使用する。
/// ユーザーは ADMM の `eps_abs/eps_rel` や IPM の `eps` を意識する必要がない。
///
/// ## 内部翻訳テーブル
///
/// | Tolerance | ADMM eps_abs/eps_rel | IPM eps |
/// |-----------|---------------------|---------|
/// | High      | 1e-8                | 1e-10   |
/// | Medium    | 1e-6                | 1e-8    |
/// | Fast      | 1e-3                | 1e-6    |
/// | Custom(v) | v                   | v       |
///
/// `Medium` はデフォルト値（Gurobi と同等の精度水準 `eps=1e-6`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// 高精度: 精密な解が必要な研究・検証用途向け
    High,
    /// 中精度（デフォルト）: 汎用的な実用問題向け (ADMM: 1e-6, IPM: 1e-8)
    Medium,
    /// 高速: 計算速度優先、精度を緩める (ADMM: 1e-3, IPM: 1e-6)
    Fast,
    /// カスタム: 各ソルバーの収束基準に直接使用する数値を指定
    Custom(f64),
}

/// ADMM（Alternating Direction Method of Multipliers）固有オプション
///
/// [`SolverOptions::admm`] フィールドに設定する。
#[derive(Debug, Clone)]
pub struct AdmmOptions {
    /// 最大反復回数（None = デフォルト: 10000）
    pub max_iter: Option<usize>,
    /// 絶対収束tolerance（デフォルト: 1e-6）
    pub eps_abs: f64,
    /// 相対収束tolerance（デフォルト: 1e-6）
    pub eps_rel: f64,
    /// ペナルティパラメータ ρ 初期値（デフォルト: 0.1）
    pub rho: f64,
    /// 近接正則化パラメータ σ（デフォルト: 1e-6）
    pub sigma: f64,
    /// 過緩和係数 α（デフォルト: 1.6）
    pub alpha: f64,
    /// x-update のソルバー選択
    /// None = Auto（A^T*A 推定充填率 > 10% または n > LDL_MAX_N のとき自動的に CG を選択）
    /// Some(true) = 強制 CG
    /// Some(false) = 強制 LDL
    pub use_cg: Option<bool>,
}

impl Default for AdmmOptions {
    fn default() -> Self {
        Self {
            max_iter: None,
            eps_abs: 1e-6,
            eps_rel: 1e-6,
            rho: 0.1,
            sigma: 1e-6,
            alpha: 1.6,
            use_cg: None,
        }
    }
}

/// IPM（内点法）固有オプション
///
/// [`SolverOptions::ipm`] フィールドに設定する。
#[derive(Debug, Clone)]
pub struct IpmOptions {
    /// 最大反復数（デフォルト: 100）
    pub max_iter: usize,
    /// 収束 tolerance（デフォルト: 1e-8）
    pub eps: f64,
    /// 近接正則化下限 δ_min（デフォルト: 1e-8）
    pub delta_min: f64,
    /// 近接正則化初期値 δ_p（デフォルト: 1e-6）
    pub delta_p_init: f64,
    /// 近接正則化初期値 δ_d（デフォルト: 1e-6）
    pub delta_d_init: f64,
}

impl Default for IpmOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            eps: 1e-8,
            delta_min: 1e-8,
            delta_p_init: 1e-6,
            delta_d_init: 1e-6,
        }
    }
}

/// Active Set 法固有オプション
///
/// [`SolverOptions::active_set`] フィールドに設定する。
#[derive(Debug, Clone, Default)]
pub struct ActiveSetOptions {
    /// 最大反復回数（None = 自動: 100*(m+n)+1000）
    /// 設定した場合、グローバルの `max_iterations` より優先される。
    pub max_iter: Option<usize>,
}

/// ソルバーの動作設定
///
/// 許容誤差・反復上限・リファクタリング頻度などを制御する。
/// `Default` でtolerance.rsの標準値が設定される。
///
/// ## ソルバー固有パラメータ
///
/// `admm`・`ipm`・`active_set` フィールドの各サブ構造体を使用すること。
/// 旧来のフラット形式（`sigma`, `rho` 等）は非推奨。
#[derive(Debug, Clone)]
pub struct SolverOptions {
    // --- 共通設定 ---
    /// シンプレックス法の最適性・実行可能性判定の閾値（デフォルト: 1e-8）
    pub primal_tol: f64,
    /// 最大反復回数（None = 自動計算: 100*(m+n)+1000）
    pub max_iterations: Option<usize>,
    /// eta ファイルの最大保持数（リファクタリング閾値）
    pub max_etas: usize,
    /// 解の微小値クランプ閾値（デフォルト: 1e-14）
    pub clamp_tol: f64,
    /// シンプレックス法の選択（デフォルト: Auto）
    pub simplex_method: SimplexMethod,
    /// 双対実行可能性の閾値（デフォルト: PIVOT_TOL = 1e-8）
    pub dual_tol: f64,
    /// warm-start基底情報（Noneの場合はコールドスタート）
    pub warm_start: Option<WarmStartBasis>,
    /// Presolve有効/無効（デフォルト: true）
    pub presolve: bool,
    /// 並列Active Set実行数（parallel feature有効時のみ使用。デフォルト4）
    pub parallel_runs: usize,
    /// タイムアウト時間（秒）。None の場合は無制限（デフォルト: None）
    pub timeout_secs: Option<f64>,
    /// 並列ワーカー間共有のキャンセルフラグ（内部使用）
    pub(crate) cancel_flag: Option<Arc<AtomicBool>>,
    /// タイムアウト期限（内部使用。qp_solve_impl の先頭で timeout_secs から計算）
    pub(crate) deadline: Option<Instant>,

    // --- QP solver 自動切替 ---
    /// QP solver 選択（デフォルト: Auto）
    pub qp_solver: QpSolverChoice,
    /// QP 自動切替の閾値（デフォルト: 10_000）
    pub qp_solver_threshold: usize,

    // --- Ruiz スケーリング ---
    /// ADMM/IPM 実行前に Ruiz equilibration スケーリングを適用する（デフォルト: true）
    pub use_ruiz_scaling: bool,

    // --- 収束精度抽象化 ---
    /// 収束精度の抽象レベル（None の場合は admm.eps_abs/eps_rel, ipm.eps を直接使用）
    ///
    /// Some(_) の場合、各ソルバーはこの設定から eps を計算して使用する。
    /// admm.eps_abs/eps_rel および ipm.eps の設定は無視される。
    pub tolerance: Option<Tolerance>,

    // --- ソルバー固有オプション ---
    /// ADMM 固有オプション
    pub admm: AdmmOptions,
    /// IPM 固有オプション
    pub ipm: IpmOptions,
    /// Active Set 固有オプション
    pub active_set: ActiveSetOptions,

    // --- 旧 ADMM 固有パラメータ（非推奨: `admm.*` を使用すること）---
    /// # Deprecated
    /// `admm.sigma` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.sigma` instead")]
    pub sigma: f64,
    /// # Deprecated
    /// `admm.rho` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.rho` instead")]
    pub rho: f64,
    /// # Deprecated
    /// `admm.alpha` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.alpha` instead")]
    pub alpha: f64,
    /// # Deprecated
    /// `admm.eps_abs` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.eps_abs` instead")]
    pub eps_abs: f64,
    /// # Deprecated
    /// `admm.eps_rel` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.eps_rel` instead")]
    pub eps_rel: f64,
    /// # Deprecated
    /// `admm.max_iter` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.max_iter` instead")]
    pub max_iter_admm: Option<usize>,
    /// # Deprecated
    /// `admm.use_cg` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `admm.use_cg` instead")]
    pub admm_use_cg: Option<bool>,

    // --- 旧 IPM 固有パラメータ（非推奨: `ipm.*` を使用すること）---
    /// # Deprecated
    /// `ipm.max_iter` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `ipm.max_iter` instead")]
    pub max_iter_ipm: usize,
    /// # Deprecated
    /// `ipm.eps` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `ipm.eps` instead")]
    pub eps_ipm: f64,
    /// # Deprecated
    /// `ipm.delta_p_init` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `ipm.delta_p_init` instead")]
    pub delta_p: f64,
    /// # Deprecated
    /// `ipm.delta_d_init` を使用すること。
    #[deprecated(since = "0.1.0", note = "use `ipm.delta_d_init` instead")]
    pub delta_d: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        #[allow(deprecated)]
        Self {
            primal_tol: PIVOT_TOL, // 1e-8
            max_iterations: None,  // auto
            max_etas: 50,
            clamp_tol: 1e-14,
            simplex_method: SimplexMethod::Auto,
            dual_tol: PIVOT_TOL,
            warm_start: None,
            presolve: true,
            parallel_runs: 4,
            timeout_secs: None,
            cancel_flag: None,
            deadline: None,
            qp_solver: QpSolverChoice::Auto,
            qp_solver_threshold: 10_000,
            use_ruiz_scaling: true,
            tolerance: None,
            admm: AdmmOptions::default(),
            ipm: IpmOptions::default(),
            active_set: ActiveSetOptions::default(),
            // Deprecated fields (backward compat)
            sigma: 1e-6,
            rho: 0.1,
            alpha: 1.6,
            eps_abs: 1e-6,
            eps_rel: 1e-6,
            max_iter_admm: None,
            admm_use_cg: None,
            max_iter_ipm: 100,
            eps_ipm: 1e-8,
            delta_p: 1e-6,
            delta_d: 1e-6,
        }
    }
}

impl SolverOptions {
    /// ADMM の eps_abs を取得（tolerance が Some の場合は変換して返す）
    pub fn admm_eps_abs(&self) -> f64 {
        match self.tolerance {
            Some(Tolerance::High)      => 1e-8,
            Some(Tolerance::Medium)    => 1e-6,
            Some(Tolerance::Fast)      => 1e-3,
            Some(Tolerance::Custom(v)) => v,
            None => self.admm.eps_abs,
        }
    }

    /// ADMM の eps_rel を取得（tolerance が Some の場合は変換して返す）
    pub fn admm_eps_rel(&self) -> f64 {
        match self.tolerance {
            Some(Tolerance::High)      => 1e-8,
            Some(Tolerance::Medium)    => 1e-6,
            Some(Tolerance::Fast)      => 1e-3,
            Some(Tolerance::Custom(v)) => v,
            None => self.admm.eps_rel,
        }
    }

    /// IPM の eps を取得（tolerance が Some の場合は変換して返す）
    pub fn ipm_eps(&self) -> f64 {
        match self.tolerance {
            Some(Tolerance::High)      => 1e-10,
            Some(Tolerance::Medium)    => 1e-8,
            Some(Tolerance::Fast)      => 1e-6,
            Some(Tolerance::Custom(v)) => v,
            None => self.ipm.eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance 翻訳メソッドが正しい値を返すことを確認する
    #[test]
    fn test_tolerance_translation() {
        // High
        let opts_high = SolverOptions { tolerance: Some(Tolerance::High), ..Default::default() };
        assert_eq!(opts_high.admm_eps_abs(), 1e-8, "High: admm_eps_abs");
        assert_eq!(opts_high.admm_eps_rel(), 1e-8, "High: admm_eps_rel");
        assert_eq!(opts_high.ipm_eps(), 1e-10, "High: ipm_eps");

        // Medium
        let opts_med = SolverOptions { tolerance: Some(Tolerance::Medium), ..Default::default() };
        assert_eq!(opts_med.admm_eps_abs(), 1e-6, "Medium: admm_eps_abs");
        assert_eq!(opts_med.admm_eps_rel(), 1e-6, "Medium: admm_eps_rel");
        assert_eq!(opts_med.ipm_eps(), 1e-8, "Medium: ipm_eps");

        // Fast
        let opts_fast = SolverOptions { tolerance: Some(Tolerance::Fast), ..Default::default() };
        assert_eq!(opts_fast.admm_eps_abs(), 1e-3, "Fast: admm_eps_abs");
        assert_eq!(opts_fast.admm_eps_rel(), 1e-3, "Fast: admm_eps_rel");
        assert_eq!(opts_fast.ipm_eps(), 1e-6, "Fast: ipm_eps");

        // Custom
        let opts_custom = SolverOptions { tolerance: Some(Tolerance::Custom(1e-5)), ..Default::default() };
        assert_eq!(opts_custom.admm_eps_abs(), 1e-5, "Custom: admm_eps_abs");
        assert_eq!(opts_custom.admm_eps_rel(), 1e-5, "Custom: admm_eps_rel");
        assert_eq!(opts_custom.ipm_eps(), 1e-5, "Custom: ipm_eps");

        // None → 各フィールドのデフォルト値を返す
        let opts_none = SolverOptions::default();
        assert_eq!(opts_none.admm_eps_abs(), 1e-6, "None: admm_eps_abs (default)");
        assert_eq!(opts_none.admm_eps_rel(), 1e-6, "None: admm_eps_rel (default)");
        assert_eq!(opts_none.ipm_eps(), 1e-8, "None: ipm_eps (default)");
    }
}
