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

/// シンプレックス法のピボット安定性しきい値
///
/// FTRAN 後の入基列で |d[leaving_row]| / max(|d|) < PIVOT_STABILITY_THRESHOLD の場合、
/// ピボット前に LU を再因子分解して eta 蓄積による数値誤差をリセットする。
/// 値: 最大列エントリの 1% 未満のピボットを「不安定」と判定。
pub const PIVOT_STABILITY_THRESHOLD: f64 = 0.01;

/// 比率テストにおける近ゼロピボット検出の絶対閾値
///
/// 退化ステップ (step ≈ 0) で Bland 則が d < NEAR_ZERO_PIVOT_GUARD のピボット行を選んだ場合、
/// eta ファイルに inv_pivot = 1/d >> 1 の巨大エントリが生じて後続 FTRAN を汚染する。
/// この閾値以上の d を持つ代替行が存在すればそちらへオーバーライドする。
/// PIVOT_TOL の 100 倍 — 比率テストの最小適格値 (PIVOT_TOL) の直上にある
/// 事実上ゼロな d を捕捉する。
pub const NEAR_ZERO_PIVOT_GUARD: f64 = PIVOT_TOL * 100.0;
