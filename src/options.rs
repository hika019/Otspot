//! ソルバー設定パラメータモジュール
//!
//! [`SolverOptions`] を通じてシンプレックス法の動作を制御する。
//! 許容誤差・反復上限・リファクタリング頻度などを一元管理する。

use crate::tolerances::*;

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
        }
    }
}
