//! IPM 初期点計算
//!
//! Mehrotra heuristic に基づく starting point の計算。

/// 内点法の初期点を計算する
///
/// 不等式行: x = 0, s_i = max(1, |b_ext_i| + 1)（s > 0 保証）, y_i = 1
/// 等式行: s_i = 0（スラックなし）, y_i = 0（自由変数）
///
/// # 引数
/// - `n`: 変数の次元
/// - `b_ext`: 拡張制約の右辺ベクトル
/// - `is_eq_ext`: 各行が等式制約かどうか
///
/// # 戻り値
/// `(x, s, y)` — 主変数, スラック変数, 双対変数
pub(crate) fn compute_initial_point(n: usize, b_ext: &[f64], is_eq_ext: &[bool]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_ext = b_ext.len();
    let x = vec![0.0f64; n];
    let mut s = Vec::with_capacity(m_ext);
    let mut y = Vec::with_capacity(m_ext);
    for i in 0..m_ext {
        if is_eq_ext[i] {
            s.push(0.0);
            y.push(0.0);
        } else {
            s.push(1.0_f64.max(b_ext[i].abs() + 1.0));
            y.push(1.0);
        }
    }
    (x, s, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_initial_point_empty() {
        let (x, s, y) = compute_initial_point(0, &[], &[]);
        assert!(x.is_empty());
        assert!(s.is_empty());
        assert!(y.is_empty());
    }

    #[test]
    fn test_compute_initial_point_basic() {
        // b_ext[0]=0.0 → s=max(1, 0+1)=1.0
        // b_ext[1]=0.5 → s=max(1, 0.5+1)=1.5
        let (x, s, y) = compute_initial_point(2, &[0.0, 0.5], &[false, false]);
        assert_eq!(x, vec![0.0, 0.0]);
        assert!((s[0] - 1.0).abs() < 1e-15, "s[0]={}", s[0]);
        assert!((s[1] - 1.5).abs() < 1e-15, "s[1]={}", s[1]);
        assert_eq!(y, vec![1.0, 1.0]);
    }

    #[test]
    fn test_compute_initial_point_negative_and_large() {
        // b_ext[0]=2.0 → s=max(1, 3.0)=3.0
        // b_ext[1]=-3.0 → s=max(1, |-3|+1)=4.0
        // b_ext[2]=0.0 → s=1.0
        let (_x, s, _y) = compute_initial_point(3, &[2.0, -3.0, 0.0], &[false, false, false]);
        assert!((s[0] - 3.0).abs() < 1e-15, "s[0]={}", s[0]);
        assert!((s[1] - 4.0).abs() < 1e-15, "s[1]={}", s[1]);
        assert!((s[2] - 1.0).abs() < 1e-15, "s[2]={}", s[2]);
        assert!(s.iter().all(|&v| v >= 1.0), "all s >= 1.0 violated");
    }

    #[test]
    fn test_compute_initial_point_size_separation() {
        // n=5変数, b_ext=3要素（制約数3）
        let b_ext = [1.0, 2.0, 3.0];
        let (x, s, y) = compute_initial_point(5, &b_ext, &[false, false, false]);
        assert_eq!(x.len(), 5, "x.len() should be n=5");
        assert_eq!(s.len(), 3, "s.len() should be m_ext=3");
        assert_eq!(y.len(), 3, "y.len() should be m_ext=3");
        assert!(x.iter().all(|&v| v == 0.0));
        assert!(y.iter().all(|&v| v == 1.0));
    }
}
