//! ソルバー設定パラメータモジュール
//!
//! [`SolverOptions`] を通じてシンプレックス法の動作を制御する。
//! 許容誤差・反復上限・リファクタリング頻度などを一元管理する。
//!
//! ## ソルバー固有オプション
//!
//! ソルバー固有パラメータは各サブ構造体で管理する:
//! - IPM: [`SolverOptions::ipm`] ([`IpmOptions`])

use crate::tolerances::*;
use std::sync::{
    atomic::AtomicBool,
    Arc,
};
use std::time::Instant;

/// QP ソルバー選択
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpSolverChoice {
    /// Concurrent: 全ソルバー並列実行、最速の解を返す（デフォルト）
    #[default]
    Concurrent,
    /// 強制 IPM (内点法)
    Ipm,
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
/// ユーザーは IPM の `eps` を意識する必要がない。
///
/// ## 内部翻訳テーブル
///
/// | Tolerance | IPM eps |
/// |-----------|---------|
/// | High      | 1e-8    |
/// | Medium    | 1e-6    |
/// | Fast      | 1e-6    |
/// | Custom(v) | v       |
///
/// `Medium` はデフォルト値（Gurobi と同等の精度水準 `eps=1e-6`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// 高精度: 精密な解が必要な研究・検証用途向け
    High,
    /// 中精度（デフォルト）: 汎用的な実用問題向け (IPM: 1e-6)
    Medium,
    /// 高速: 計算速度優先、精度を緩める (IPM: 1e-6)
    Fast,
    /// カスタム: 各ソルバーの収束基準に直接使用する数値を指定
    Custom(f64),
}

/// IPM（内点法）固有オプション
///
/// [`SolverOptions::ipm`] フィールドに設定する。
#[derive(Debug, Clone)]
pub struct IpmOptions {
    /// 最大反復数（デフォルト: 1000）。IPMの安全弁。timeout が主ガード。
    pub max_iter: usize,
    /// 収束 tolerance（デフォルト: 1e-8）
    pub eps: f64,
    /// 近接正則化下限 δ_min（デフォルト: 1e-8）
    pub delta_min: f64,
    /// 近接正則化初期値 δ_p（デフォルト: 1e-6）
    pub delta_p_init: f64,
    /// 近接正則化初期値 δ_d（デフォルト: 1e-6）
    pub delta_d_init: f64,
    /// Gondzio多重修正子の最大corrector数（デフォルト: 3）
    /// PARAM: 根拠=Gondzio(1997)推奨値(2-5) | 要検証=大規模問題
    pub max_correctors: usize,
}

impl Default for IpmOptions {
    fn default() -> Self {
        Self {
            max_iter: usize::MAX,
            eps: 1e-8,
            delta_min: 1e-8,
            delta_p_init: 1e-6,
            delta_d_init: 1e-6,
            max_correctors: 3,
        }
    }
}

/// ソルバーの動作設定
///
/// 許容誤差・反復上限・リファクタリング頻度などを制御する。
/// `Default` でtolerance.rsの標準値が設定される。
///
/// ## ソルバー固有パラメータ
///
/// `ipm` フィールドのサブ構造体を使用すること。
#[derive(Debug, Clone)]
pub struct SolverOptions {
    // --- 共通設定 ---
    /// シンプレックス法の最適性・実行可能性判定の閾値（デフォルト: 1e-8）
    pub primal_tol: f64,
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
    /// タイムアウト時間（秒）。None の場合は無制限（デフォルト: None）
    pub timeout_secs: Option<f64>,
    /// 並列ワーカー間共有のキャンセルフラグ（内部使用）
    pub(crate) cancel_flag: Option<Arc<AtomicBool>>,
    /// タイムアウト期限（内部使用。solve の先頭で timeout_secs から計算）
    pub(crate) deadline: Option<Instant>,

    // --- QP solver 選択 ---
    /// QP solver 選択（デフォルト: Concurrent）
    pub qp_solver: QpSolverChoice,

    // --- Ruiz スケーリング ---
    /// IPM 実行前に Ruiz equilibration スケーリングを適用する（デフォルト: true）
    pub use_ruiz_scaling: bool,

    // --- 収束精度抽象化 ---
    /// 収束精度の抽象レベル（None の場合は ipm.eps を直接使用）
    ///
    /// Some(_) の場合、各ソルバーはこの設定から eps を計算して使用する。
    /// ipm.eps の設定は無視される。
    pub tolerance: Option<Tolerance>,

    // --- ソルバー固有オプション ---
    /// IPM 固有オプション
    pub ipm: IpmOptions,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            primal_tol: PIVOT_TOL, // 1e-8
            max_etas: 50,
            clamp_tol: 1e-14,
            simplex_method: SimplexMethod::Auto,
            dual_tol: PIVOT_TOL,
            warm_start: None,
            presolve: true,
            timeout_secs: None,
            cancel_flag: None,
            deadline: None,
            qp_solver: QpSolverChoice::Concurrent,
            use_ruiz_scaling: true,
            tolerance: None,
            ipm: IpmOptions::default(),
        }
    }
}

impl SolverOptions {
    /// IPM の eps を取得（tolerance が Some の場合は変換して返す）
    pub fn ipm_eps(&self) -> f64 {
        match self.tolerance {
            Some(Tolerance::High)      => 1e-8,
            Some(Tolerance::Medium)    => 1e-6,
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
        assert_eq!(opts_high.ipm_eps(), 1e-8, "High: ipm_eps");

        // Medium
        let opts_med = SolverOptions { tolerance: Some(Tolerance::Medium), ..Default::default() };
        assert_eq!(opts_med.ipm_eps(), 1e-6, "Medium: ipm_eps");

        // Fast
        let opts_fast = SolverOptions { tolerance: Some(Tolerance::Fast), ..Default::default() };
        assert_eq!(opts_fast.ipm_eps(), 1e-6, "Fast: ipm_eps");

        // Custom
        let opts_custom = SolverOptions { tolerance: Some(Tolerance::Custom(1e-5)), ..Default::default() };
        assert_eq!(opts_custom.ipm_eps(), 1e-5, "Custom: ipm_eps");

        // None → ipm.eps のデフォルト値を返す
        let opts_none = SolverOptions::default();
        assert_eq!(opts_none.ipm_eps(), 1e-8, "None: ipm_eps (default)");
    }
}
