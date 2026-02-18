//! 数値許容値の一元管理モジュール
//!
//! ソルバー全体で使用する数値定数をここに集約する。
//! 定数値を変更する場合はこのファイルのみを編集すればよい。

/// 構造的ゼロ判定の閾値（絶対値がこれ以下の値をゼロとみなす）
pub const ZERO_TOL: f64 = 1e-12;

/// シンプレックス法の最適性・実行可能性判定の閾値
pub const PIVOT_TOL: f64 = 1e-8;

/// 行列構築時の微小値除去の閾値
pub const DROP_TOL: f64 = 1e-15;

/// LU 特異性判定の閾値
pub const SINGULAR_TOL: f64 = 1e-12;

/// Markowitz ピボット選択のしきい値
pub const MARKOWITZ_THRESHOLD: f64 = 0.1;
