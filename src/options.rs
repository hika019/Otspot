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

/// シンプレックス法の選択
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplexMethod {
    /// デフォルト: warm-startの有無で自動選択
    #[default]
    Auto,
    /// 強制的にPrimal Simplex
    Primal,
    /// 強制的にDual Simplex
    Dual,
    /// 産業品質Dual Simplex（dual_advanced/モジュール）
    DualAdvanced,
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

/// QP IP-PMM の内部 interior point 引継ぎ情報。
///
/// 前ノードの最適 (x, y, μ) を次ノードの central path 出発点として使う。
/// LP の [`WarmStartBasis`] は basis index、QP は central path 上の点で
/// 共存しないため別 struct で持つ。
///
/// 規約:
/// - `x` 長さ = n (primal)
/// - `y` 長さ = m (元制約 dual、ユーザー符号規約。Ge は内部で反転される)
/// - `mu` = sᵀy/m_ineq に相当する barrier parameter。
///   再帰的 B&B では parent の final μ をそのまま渡す想定。
///
/// 入口で interior 補正される (μ floor / x bound margin / y positivity)
/// ため境界値や 0 を渡しても安全に IPM が起動する。
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub mu: f64,
}

/// LP 用拡張 warm start (#15 速度改善 F1)。
///
/// 既存 [`WarmStartBasis`] は (basis index, x_B) のみ。LP の Simplex/Dual 入口で
/// 外部 solver 由来の (x, y, basis) を受け取って simplex に着地させるための
/// 上位構造体。
///
/// 規約:
/// - `basis` 長さ = m_ext (standard form 行数)、各値 < n_total (standard form 列数)。
///   一致しない場合は silent SKIP せず eprintln + drop。
/// - `x_orig` 長さ = problem.num_vars (元変数空間)
/// - `y_orig` 長さ = problem.num_constraints (元制約空間、ユーザー符号)
///
/// `basis` のみ与えれば既存 dual simplex warm start と同等。`x_orig`/`y_orig` は
/// 将来 IPM crossover や presolve 整合に使う slot。
#[derive(Debug, Clone)]
pub struct LpWarmStart {
    pub basis: Vec<usize>,
    pub x_orig: Option<Vec<f64>>,
    pub y_orig: Option<Vec<f64>>,
}

/// Multi-start サンプリング戦略 (#5 Phase 2)。
///
/// IPM は inertia 補正下で「最寄り KKT 点」へ収束するため、非凸 QP では
/// 出発点を変えると到達する local optimum が変わる。出発点生成方法の選択。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartStrategy {
    /// 各変数を box bounds 内で独立一様サンプリング (LCG)。
    RandomBox,
    /// Latin Hypercube Sampling: 各次元を `n_starts` strata に分割し列ごと permutation。
    /// box 全域被覆性が pure random より高い。
    LatinHypercube,
}

/// Multi-start local search (#5 Phase 2) **user-facing** config。
///
/// `n_starts` 個の初期点で IPM を解き、最良の objective を採用する。
/// 非凸 QP の脱出率向上 + Phase 3 spatial B&B の上界 (incumbent) 供給。
///
/// **user 指定 (pub field、explicit input)**:
/// - `n_starts`: 探索並列度 (大域最適 hit 確率に直結)
/// - `seed`: 再現性 (同 seed = 同 random init = 同 best)
/// - `strategy`: 探索戦略 (RandomBox / LatinHypercube)
///
/// **内部勝手に決める (user に見せない)**:
/// - 乱数 algorithm (LCG 1664525/1013904223)
/// - 無限境界のサンプリング半径 (`MULTISTART_UNBOUNDED_RANGE = 10.0`)
/// - thread 並列度 (`SolverOptions::threads` から `min(n_starts, threads)` 自動分配)
/// - 各 inner solve の `threads = 1` 強制 (faer 多重並列化を抑止)
///
/// 規約:
/// - `n_starts == 1` は cold solve 1 回のみ (= 既存挙動)
/// - `n_starts >= 2` で start #0 = cold、#1..#n_starts = random initial (warm_start_qp.x 注入)
/// - 全 start で deadline を共有 (timeout_secs / deadline は最初に固定)
#[derive(Debug, Clone)]
pub struct MultiStartConfig {
    /// 初期点数 (cold #0 + random #1..#n_starts)。1 で multistart 無効化、
    /// 2 以上で並列 escape 探索。default=1 (= 既存挙動)。
    pub n_starts: usize,
    /// 乱数 seed。default=`DEFAULT_MULTISTART_SEED` (= 0xC0FFEE_DEADBEEF) で
    /// deterministic test 環境を保護。bench で variance 取るときは user が変える。
    /// `0` は内部で 1 に補正 (LCG state=0 固着回避)。
    pub seed: u64,
    /// サンプリング戦略。default=`RandomBox` (= 各次元独立一様)。
    pub strategy: StartStrategy,
}

/// `MultiStartConfig::seed` の default。固定値で deterministic test を保護。
/// magic 根拠: 任意の非零値で良い。0xC0FFEE_DEADBEEF は識別性のためのフォーク値。
pub const DEFAULT_MULTISTART_SEED: u64 = 0x_00C0_FFEE_DEAD_BEEF;

/// 分枝戦略 (#6 Phase 3 spatial B&B)。
///
/// `MaxViolation`: 現 box midpoint から x* が最も離れた連続変数を選び、x*[j] で
/// 2 子に分割する。Phase 3 唯一の戦略。Phase 4 以降に strong branching 追加予定。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    MaxViolation,
}

/// `GlobalOptimizationConfig` defaults。
///
/// - `DEFAULT_GLOBAL_GAP_TOL = 1e-3`: 相対 ε-optimal gap。Phase 3 (interval lower
///   bound) は緩い下界しか出ないため 1e-6 級まで詰めると node 爆発。Phase 4 (α-BB)
///   で 1e-6 へ tighten 想定。
/// - `DEFAULT_GLOBAL_MAX_DEPTH = 20`: tree depth 上限。2^20 ≈ 100 万 node 安全装置。
/// - `DEFAULT_GLOBAL_MAX_NODES = 10_000`: 探索 node 上限。Phase 3 は 1 node ≈ 1 IPM
///   solve なので n=50 で 10k node ≈ wall 数十秒の budget。
pub const DEFAULT_GLOBAL_GAP_TOL: f64 = 1e-3;
pub const DEFAULT_GLOBAL_MAX_DEPTH: usize = 20;
pub const DEFAULT_GLOBAL_MAX_NODES: usize = 10_000;

/// Spatial Branch-and-Bound 設定 (#6 / #7 非凸 QP 大域最適化)。
///
/// **user 指定** ε-optimal global solve のパラメータ。`SolverOptions::global_optimization`
/// に注入し、`solve_qp_global` から参照される。`solve_qp_with` の dispatch 対象には**ならない**
/// (= 明示呼び出し)。誤って global path に倒すと既存 QP user の wall が桁違いに増える
/// リスクを抑える。
///
/// 規約:
/// - `gap_tol > 0`: 相対 gap (= |UB - LB| / max(1, |UB|))
/// - `max_depth >= 1`: 0 は root 1 回のみ
/// - `max_nodes >= 1`: 0 は root も解かない
/// - `use_alpha_bb`: true で Phase 4 α-BB underestimator を下界に使う (default)。
///   false にすると Phase 3 の interval-arithmetic bound に戻す (退化/比較用)。
#[derive(Debug, Clone)]
pub struct GlobalOptimizationConfig {
    pub gap_tol: f64,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub branching: BranchingStrategy,
    pub use_alpha_bb: bool,
}

impl Default for GlobalOptimizationConfig {
    fn default() -> Self {
        Self {
            gap_tol: DEFAULT_GLOBAL_GAP_TOL,
            max_depth: DEFAULT_GLOBAL_MAX_DEPTH,
            max_nodes: DEFAULT_GLOBAL_MAX_NODES,
            branching: BranchingStrategy::MaxViolation,
            use_alpha_bb: true,
        }
    }
}

impl Default for MultiStartConfig {
    fn default() -> Self {
        Self {
            n_starts: 1,
            seed: DEFAULT_MULTISTART_SEED,
            strategy: StartStrategy::RandomBox,
        }
    }
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
#[non_exhaustive]
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
    /// 最大反復数（デフォルト: usize::MAX = 上限なし）。timeout が主ガード。
    pub max_iter: usize,
    /// 収束 tolerance（デフォルト: 1e-6）
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
            eps: 1e-6,
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
    /// QP IP-PMM の interior point warm start (B&B node 間引継ぎ用)
    pub warm_start_qp: Option<QpWarmStart>,
    /// LP 拡張 warm start (#15)。`warm_start` より優先される。
    /// `basis` のみ与えれば既存挙動と同等で、`x_orig`/`y_orig` は将来 IPM crossover 用。
    pub warm_start_lp: Option<LpWarmStart>,
    /// LP cold start 時 simplex crash basis を適用する (#15)。
    /// warm_start / warm_start_lp が Some なら無視される。
    pub use_lp_crash_basis: bool,
    /// Presolve有効/無効（デフォルト: true）
    pub presolve: bool,
    /// タイムアウト時間（秒）。None の場合は無制限（デフォルト: None）
    pub timeout_secs: Option<f64>,
    /// 並列ワーカー間共有のキャンセルフラグ（内部使用）
    pub(crate) cancel_flag: Option<Arc<AtomicBool>>,
    /// タイムアウト期限（内部使用。solve の先頭で timeout_secs から計算）
    pub(crate) deadline: Option<Instant>,

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

    /// Multi-start local search (#5 Phase 2)。`None` (default) は無効 = 既存挙動。
    /// `Some(_)` かつ `n_starts >= 2` の場合 `solve_qp_with` は内部で
    /// `solve_qp_multistart` に委譲する (再入防止のため委譲時は None に剥がす)。
    pub multistart: Option<MultiStartConfig>,

    /// Phase 3 spatial Branch-and-Bound (#6 非凸 QP 大域最適化) の設定。
    /// `None` (default) は無効。`Some(_)` のとき `solve_qp_global` から参照される。
    /// **`solve_qp_with` は dispatch しない**: 明示呼び出し限定 (誤って既存 user の wall を桁違いに伸ばさないため)。
    pub global_optimization: Option<GlobalOptimizationConfig>,

    /// **user 指定** 全 solver 共通の thread 上限 (LP / QP / 非凸 multistart 全て)。
    ///
    /// default = 1 (シリアル、既存挙動完全保護、bench worker と多重化しない)。
    ///
    /// **現状の実効範囲 (2026-05-18 時点)**:
    /// - multistart 時の並列度 = `min(n_starts, threads)` を内部で自動分配 = **実効並列**
    /// - 各 inner solve は `threads = 1` 強制 (二重並列化抑止)
    /// - **単発 LP/QP solve では現状 no-op** (faer 内部は `Par::Seq` hardcode、
    ///   per-call parallelism 配線は future work)
    /// - 単発 solve threads option 指定しても **値は受理されるが効果ゼロ**
    ///
    /// CLAUDE.md cpu800% 上限考慮、`bench_parallel.sh --jobs N × threads=1`
    /// と (#31 完了後) `--jobs 1 × threads=N` のいずれも合計 N CPU で動かせる設計。
    pub threads: usize,
}

/// max_etas の auto 計算: m に応じた動的設定 (CLAUDE.md ベンチ tuning 値排除)。
/// 小規模 (m<1000) は 20、大規模では m/50。
///
/// 旧 m/100 は dfl001 級 (m=12857) で max_etas=128、refactor 1 回 720ms × 69 = 50s
/// が timeout の主因 (Task #6/9 観測)。m/50 で refactor 頻度を半減、per-iter eta cost
/// 増加とのトレードオフで dfl001 改善を狙う (eta cost は per-iter ~50us 程度の増)。
pub fn default_max_etas(m: usize) -> usize {
    (m / 50).max(20)
}

/// Phase I retry 上限: revised_simplex_core が同じ basis で無限ループに入る
/// 退化問題用の安全装置。
pub const MAX_PHASE1_RETRIES: usize = 8;

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            primal_tol: PIVOT_TOL, // 1e-8
            // max_etas: 0 = auto (default_max_etas(m) で m から計算、各 simplex 入口で適用)
            max_etas: 0,
            clamp_tol: 1e-14,
            simplex_method: SimplexMethod::Auto,
            dual_tol: PIVOT_TOL,
            warm_start: None,
            warm_start_qp: None,
            warm_start_lp: None,
            use_lp_crash_basis: true,
            presolve: true,
            timeout_secs: None,
            cancel_flag: None,
            deadline: None,
            use_ruiz_scaling: true,
            tolerance: None,
            ipm: IpmOptions::default(),
            multistart: None,
            global_optimization: None,
            threads: 1,
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
        assert_eq!(opts_none.ipm_eps(), 1e-6, "None: ipm_eps (default)");
    }
}
