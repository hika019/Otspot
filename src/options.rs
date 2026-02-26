//! ソルバー設定パラメータモジュール
//!
//! [`SolverOptions`] を通じてシンプレックス法の動作を制御する。
//! 許容誤差・反復上限・リファクタリング頻度などを一元管理する。

use crate::tolerances::*;
use std::sync::{
    atomic::AtomicBool,
    Arc,
};
use std::time::Instant;

/// QP ソルバー選択
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpSolverChoice {
    /// Auto: n > qp_solver_threshold のとき ADMM を選択、さもなくば Active Set
    #[default]
    Auto,
    /// 強制 ADMM
    Admm,
    /// 強制 Active Set
    ActiveSet,
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

/// ソルバーの動作設定
///
/// 許容誤差・反復上限・リファクタリング頻度などを制御する。
/// `Default` でtolerance.rsの標準値が設定される。
#[derive(Debug, Clone)]
pub struct SolverOptions {
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

    // --- ADMM固有パラメータ ---
    /// ADMM近接正則化パラメータ σ（デフォルト: 1e-6）
    pub sigma: f64,
    /// ADMMペナルティパラメータ ρ 初期値（デフォルト: 0.1）
    pub rho: f64,
    /// ADMM過緩和係数 α（デフォルト: 1.6）
    pub alpha: f64,
    /// ADMM絶対収束tolerance（デフォルト: 1e-3）
    pub eps_abs: f64,
    /// ADMM相対収束tolerance（デフォルト: 1e-3）
    pub eps_rel: f64,
    /// ADMMの最大反復回数（None = デフォルト: 10000）
    pub max_iter_admm: Option<usize>,
    /// ADMM x-update のソルバー選択
    /// None = Auto（A^T*A 推定充填率 > 10% または n > LDL_MAX_N のとき自動的に CG を選択）
    /// Some(true) = 強制 CG
    /// Some(false) = 強制 LDL
    pub admm_use_cg: Option<bool>,

    // --- QP solver 自動切替 ---
    /// QP solver 選択（デフォルト: Auto）
    /// Auto: n > qp_solver_threshold のとき ADMM を選択
    /// Admm: 強制 ADMM
    /// ActiveSet: 強制 Active Set
    pub qp_solver: QpSolverChoice,
    /// QP 自動切替の閾値（デフォルト: 10_000）
    pub qp_solver_threshold: usize,

    // --- Ruiz スケーリング ---
    /// ADMM 実行前に Ruiz equilibration スケーリングを適用する（デフォルト: true）
    /// false のとき、スケーリングをスキップして従来通りに動作する。
    pub use_ruiz_scaling: bool,

    // --- IPM固有パラメータ ---
    /// IPM 最大反復数（デフォルト: 100）
    pub max_iter_ipm: usize,
    /// IPM 収束 tolerance（デフォルト: 1e-8）
    pub eps_ipm: f64,
    /// IP-PMM 近接正則化初期値 δ_p（デフォルト: 1e-6）
    pub delta_p: f64,
    /// IP-PMM 近接正則化初期値 δ_d（デフォルト: 1e-6）
    pub delta_d: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
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
            sigma: 1e-6,
            rho: 0.1,
            alpha: 1.6,
            eps_abs: 1e-3,
            eps_rel: 1e-3,
            max_iter_admm: None,
            admm_use_cg: None,
            qp_solver: QpSolverChoice::Auto,
            qp_solver_threshold: 10_000,
            use_ruiz_scaling: true,
            max_iter_ipm: 100,
            eps_ipm: 1e-8,
            delta_p: 1e-6,
            delta_d: 1e-6,
        }
    }
}
