//! IPM 初期点計算
//!
//! Mehrotra heuristic に基づく starting point の計算。

/// 内点法の初期点を計算する
///
/// 初期点: x = 0, s_i = max(1, |b_ext_i| + 1)（s > 0 保証）, y_i = 1
///
/// # 引数
/// - `n`: 変数の次元
/// - `b_ext`: 拡張制約の右辺ベクトル
///
/// # 戻り値
/// `(x, s, y)` — 主変数, スラック変数, 双対変数
pub(crate) fn compute_initial_point(n: usize, b_ext: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_ext = b_ext.len();
    let x = vec![0.0f64; n];
    let s: Vec<f64> = b_ext.iter().map(|&bi| 1.0_f64.max(bi.abs() + 1.0)).collect();
    let y = vec![1.0f64; m_ext];
    (x, s, y)
}
