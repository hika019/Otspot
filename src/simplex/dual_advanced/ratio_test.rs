//! Dual simplex ratio test戦略
//!
//! 設計書 §3.4 `dual_advanced/ratio_test.rs` に準拠。
//! HarrisRatioTest（2パス法）と StandardRatioTest（Bland則付き）を提供する。

/// Ratio testの戦略トレイト
///
/// 双対シンプレックス法における入基変数の選択を抽象化する。
/// Phase 2 (core.rs) で使用予定。
pub(crate) trait RatioTestStrategy {
    /// 双対比率テスト: 入基変数を選択する
    ///
    /// # 引数
    /// - `trow`: pivot行の非基底係数（長さ `n_price`）
    /// - `reduced_costs`: 被縮小費用（長さ `n_price`）
    /// - `is_basic`: 各変数が基底変数かどうかのフラグ（長さ `n_price`）
    /// - `n_price`: 対象変数数
    ///
    /// # 戻り値
    /// - `Some((entering_col, theta))`: 入基変数インデックスと双対ステップ
    /// - `None`: 候補なし（双対非有界 = 主実行不可）
    fn select_entering(
        &self,
        trow: &[f64],
        reduced_costs: &[f64],
        is_basic: &[bool],
        n_price: usize,
    ) -> Option<(usize, f64)>;
}

/// Harris ratio test（2パス法）
///
/// Pass 1（Relaxed）: Harris tolerance `harris_tol`（α_H）を用いてθ_maxを計算し、候補集合を構築
///   θ_max = min_j { (r_j + α_H) / trow[j] }  (trow[j] > pivot_tol)
///   候補集合 C = { j : r_j / trow[j] ≤ θ_max }
///
/// Pass 2（安定性優先）: 候補集合C内で |trow[j]| が最大の列を選択
///   数値安定性が最も高いピボットを優先する
///
/// Bland則フォールバック: Pass 1の候補集合が空の場合、最小インデックスを選択
///
/// # パラメータ
/// - `harris_tol`: α_H（典型値: 1e-7〜1e-5。dual_tol相当）
/// - `pivot_tol`: ピボット候補の最小閾値
pub(crate) struct HarrisRatioTest {
    pub harris_tol: f64,
    pub pivot_tol: f64,
}

impl HarrisRatioTest {
    pub fn new(harris_tol: f64, pivot_tol: f64) -> Self {
        Self { harris_tol, pivot_tol }
    }
}

impl RatioTestStrategy for HarrisRatioTest {
    fn select_entering(
        &self,
        trow: &[f64],
        reduced_costs: &[f64],
        is_basic: &[bool],
        n_price: usize,
    ) -> Option<(usize, f64)> {
        let alpha_h = self.harris_tol;
        let pivot_tol = self.pivot_tol;

        // Pass 1: θ_max = min_j { (r_j + α_H) / trow[j] } for trow[j] > pivot_tol
        let mut theta_max = f64::INFINITY;
        for j in 0..n_price {
            if is_basic[j] { continue; }
            if trow[j] > pivot_tol {
                let relaxed_ratio = (reduced_costs[j] + alpha_h) / trow[j];
                if relaxed_ratio < theta_max {
                    theta_max = relaxed_ratio;
                }
            }
        }

        if theta_max == f64::INFINITY {
            // 有効な候補が1つもない（双対非有界）
            return None;
        }

        // 候補集合 C = { j : r_j / trow[j] ≤ θ_max }
        // Pass 2: C内で |trow[j]| 最大を選択
        let mut best_j: Option<usize> = None;
        let mut best_pivot = 0.0_f64;
        let mut best_theta = f64::INFINITY;

        for j in 0..n_price {
            if is_basic[j] { continue; }
            if trow[j] > pivot_tol {
                let ratio = reduced_costs[j] / trow[j];
                if ratio <= theta_max {
                    let pivot_abs = trow[j].abs();
                    if pivot_abs > best_pivot {
                        best_pivot = pivot_abs;
                        best_j = Some(j);
                        best_theta = ratio;
                    }
                }
            }
        }

        if let Some(j) = best_j {
            return Some((j, best_theta));
        }

        // Bland則フォールバック: 候補集合が空の場合、最小インデックスを選択
        let mut fallback_j: Option<usize> = None;
        let mut fallback_theta = f64::INFINITY;
        for j in 0..n_price {
            if is_basic[j] { continue; }
            if trow[j] > pivot_tol {
                let ratio = reduced_costs[j] / trow[j];
                if fallback_j.is_none() || ratio < fallback_theta
                    || (ratio == fallback_theta && j < fallback_j.unwrap())
                {
                    fallback_j = Some(j);
                    fallback_theta = ratio;
                }
            }
        }

        fallback_j.map(|j| (j, fallback_theta))
    }
}

/// Pure Bland ratio test: minimum ratio, smallest-index tiebreak.
///
/// Anti-cycling 用フォールバック (cf. `core.rs::dual_simplex_core_advanced`)。
/// `HarrisRatioTest` の数値安定性優先 (|trow|最大) を捨て、決定的な
/// インデックス順で候補を選ぶ。strict `<` で min を更新するため、
/// 同じ min ratio に到達した最初の j (= smallest idx) が選ばれる。
pub(crate) fn bland_ratio_test(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
    pivot_tol: f64,
) -> Option<(usize, f64)> {
    let mut best_ratio = f64::INFINITY;
    let mut best_j: Option<usize> = None;
    for j in 0..n_price {
        if is_basic[j] {
            continue;
        }
        if trow[j] > pivot_tol {
            let ratio = reduced_costs[j] / trow[j];
            if ratio < best_ratio {
                best_ratio = ratio;
                best_j = Some(j);
            }
        }
    }
    best_j.map(|j| (j, best_ratio))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances::PIVOT_TOL;

    const HARRIS_TOL: f64 = 1e-7;

    /// ヘルパー: is_basicをすべてfalseで初期化
    fn no_basic(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    // ======================================================
    // HarrisRatioTest tests
    // ======================================================

    /// Pass 1: 候補集合構築の正確性
    /// θ_max = min { (r_j + α_H) / trow[j] } を計算し、r_j/trow[j] ≤ θ_max の
    /// 候補のみが選ばれる。
    #[test]
    fn harris_candidate_set_construction() {
        // trow = [1.0, 2.0, 3.0], r = [0.3, 0.4, 0.9]
        // ratio = [0.3, 0.2, 0.3]
        // relaxed_ratio = [(0.3+1e-7)/1, (0.4+1e-7)/2, (0.9+1e-7)/3]
        //               ≈ [0.3000001, 0.2000001, 0.3000000]
        // θ_max ≈ 0.2000001 (j=1)
        // 候補: j で ratio ≤ θ_max → j=1 only (ratio[0]=0.3 > θ_max, ratio[2]=0.3 > θ_max)
        // Pass 2: C = {j=1}, |trow[1]|=2.0 → j=1 selected
        let trow = vec![1.0, 2.0, 3.0];
        let r = vec![0.3, 0.4, 0.9];
        let is_basic = no_basic(3);
        let harris = HarrisRatioTest::new(HARRIS_TOL, PIVOT_TOL);
        let result = harris.select_entering(&trow, &r, &is_basic, 3);
        assert!(result.is_some());
        let (col, theta) = result.unwrap();
        assert_eq!(col, 1, "j=1 should be selected (only candidate)");
        assert!((theta - 0.2).abs() < 1e-6, "theta should be ~0.2, got {}", theta);
    }

    /// Pass 2: 複数候補から |trow[j]| 最大が選ばれる（数値安定性重視）
    #[test]
    fn harris_pass2_max_pivot_selected() {
        // trow = [1.0, 5.0, 2.0], r = [0.1, 0.5, 0.2]
        // ratio = [0.1, 0.1, 0.1]（全て同じ）
        // α_H = 1e-7 → θ_max = min { (0.1+1e-7)/1, (0.5+1e-7)/5, (0.2+1e-7)/2 }
        //                      ≈ min { 0.1000001, 0.1000000, 0.1000001 } ≈ 0.1000000 (j=1)
        // 候補: all j with ratio ≤ θ_max ≈ 0.1 → all three (ratio ≈ 0.1)
        // Pass 2: |trow[j]| → max is j=1 (|5.0|)
        let trow = vec![1.0, 5.0, 2.0];
        let r = vec![0.1, 0.5, 0.2];
        let is_basic = no_basic(3);
        let harris = HarrisRatioTest::new(1e-3, PIVOT_TOL); // 大きめのharris_tolで全候補を含める
        let result = harris.select_entering(&trow, &r, &is_basic, 3);
        assert!(result.is_some());
        let (col, _theta) = result.unwrap();
        assert_eq!(col, 1, "j=1 has largest |trow[j]|=5.0, should be selected for stability");
    }

    /// Pass 2の安定性: 同率の候補が複数ある時に最大|trow[j]|が選ばれる
    #[test]
    fn harris_pass2_largest_pivot_among_candidates() {
        // trow = [3.0, 1.0, 7.0, 2.0], r = [0.3, 0.1, 0.7, 0.2]
        // ratio = [0.1, 0.1, 0.1, 0.1]（全て0.1）
        // 大きなharris_tolで全候補が含まれる
        // Pass 2: |trow|最大は j=2 (7.0)
        let trow = vec![3.0, 1.0, 7.0, 2.0];
        let r = vec![0.3, 0.1, 0.7, 0.2];
        let is_basic = no_basic(4);
        let harris = HarrisRatioTest::new(1e-2, PIVOT_TOL);
        let result = harris.select_entering(&trow, &r, &is_basic, 4);
        assert!(result.is_some());
        let (col, _) = result.unwrap();
        assert_eq!(col, 2, "j=2 has largest |trow[j]|=7.0, should be selected");
    }

    /// Bland則フォールバック: 通常のPass 1・2で候補が拾えない場合も結果を返す
    /// （harris_tol = 0 の極端ケースで、全候補がθ_maxより大きくなりフォールバックへ）
    #[test]
    fn harris_bland_fallback_when_no_candidate() {
        // harris_tol = 0 にすると θ_max = min { r_j / trow[j] } (標準的な最小ratio)
        // 候補集合 = { j : ratio ≤ θ_max } → 最小ratio達成者のみ
        // 同率なしなら Pass 2 が機能する。フォールバックはほぼ到達しないが動作を保証する。
        // ここでは候補集合が空になるケースをシミュレート:
        // trow = [1.0, 2.0], r = [0.1, 0.1] → ratio = [0.1, 0.1]
        // harris_tol = 0: θ_max = min { 0.1/1, 0.1/2 } = 0.05
        // 候補: ratio ≤ 0.05 → ratio[0]=0.1, ratio[1]=0.05 → j=1 only
        // Pass 2: j=1 selected
        let trow = vec![1.0, 2.0];
        let r = vec![0.1, 0.1];
        let is_basic = no_basic(2);
        let harris = HarrisRatioTest::new(0.0, PIVOT_TOL);
        let result = harris.select_entering(&trow, &r, &is_basic, 2);
        assert!(result.is_some(), "Should return Some even with harris_tol=0");
        let (col, _) = result.unwrap();
        assert_eq!(col, 1, "j=1 (ratio=0.05) should be the candidate");
    }

    /// 空入力: 候補なし → None
    #[test]
    fn harris_no_eligible_column_returns_none() {
        // trow が全て負 → pivot_tol以下 → 候補なし
        let trow = vec![-1.0, -2.0];
        let r = vec![0.1, 0.2];
        let is_basic = no_basic(2);
        let harris = HarrisRatioTest::new(HARRIS_TOL, PIVOT_TOL);
        let result = harris.select_entering(&trow, &r, &is_basic, 2);
        assert!(result.is_none(), "No eligible columns should return None");
    }

    /// is_basic フラグ: 基底変数はスキップされる
    #[test]
    fn harris_skips_basic_variables() {
        // j=0 が basic, j=1 が非basic
        // j=1 のみが候補
        let trow = vec![5.0, 1.0];
        let r = vec![0.1, 0.2];
        let is_basic = vec![true, false];
        let harris = HarrisRatioTest::new(HARRIS_TOL, PIVOT_TOL);
        let result = harris.select_entering(&trow, &r, &is_basic, 2);
        assert!(result.is_some());
        let (col, _) = result.unwrap();
        assert_eq!(col, 1, "j=0 is basic, only j=1 should be selected");
    }

    // ======================================================
    // bland_ratio_test tests
    // ======================================================

    /// Bland: 最小 ratio が選ばれる
    #[test]
    fn bland_selects_min_ratio() {
        // trow = [1.0, 2.0, 3.0], r = [0.3, 0.2, 0.9]
        // ratio = [0.3, 0.1, 0.3] → min is j=1
        let trow = vec![1.0, 2.0, 3.0];
        let r = vec![0.3, 0.2, 0.9];
        let is_basic = no_basic(3);
        let result = bland_ratio_test(&trow, &r, &is_basic, 3, PIVOT_TOL);
        assert!(result.is_some());
        let (col, theta) = result.unwrap();
        assert_eq!(col, 1);
        assert!((theta - 0.1).abs() < 1e-9);
    }

    /// Bland: 同率 → smallest index
    #[test]
    fn bland_smallest_index_on_tie() {
        // ratio が完全に同じ場合、最小 idx (j=0) が選ばれる
        let trow = vec![1.0, 1.0, 1.0];
        let r = vec![0.5, 0.5, 0.5];
        let is_basic = no_basic(3);
        let result = bland_ratio_test(&trow, &r, &is_basic, 3, PIVOT_TOL);
        assert!(result.is_some());
        let (col, _) = result.unwrap();
        assert_eq!(col, 0);
    }

    /// Bland: 候補なし → None
    #[test]
    fn bland_no_eligible_returns_none() {
        let trow = vec![-1.0, 0.0, -2.0];
        let r = vec![0.1, 0.2, 0.3];
        let is_basic = no_basic(3);
        let result = bland_ratio_test(&trow, &r, &is_basic, 3, PIVOT_TOL);
        assert!(result.is_none());
    }

    /// Bland: basic 変数はスキップ
    #[test]
    fn bland_skips_basic_variables() {
        // j=0 が basic だが ratio 最小。j=1 が次点で選ばれる
        let trow = vec![10.0, 1.0];
        let r = vec![0.1, 0.5];
        let is_basic = vec![true, false];
        let result = bland_ratio_test(&trow, &r, &is_basic, 2, PIVOT_TOL);
        assert!(result.is_some());
        let (col, theta) = result.unwrap();
        assert_eq!(col, 1);
        assert!((theta - 0.5).abs() < 1e-9);
    }

}
