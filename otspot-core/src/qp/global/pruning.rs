//! 枝刈り判定 (Phase 3 spatial B&B)。
//!
//! 「node の lower bound が現 incumbent (best upper bound) に gap_tol 以内まで
//! 接近している」 = 当該 subtree から incumbent を超える改善は見込めない → prune。
//!
//! gap は相対 = |UB - LB| / max(1, |UB|)。incumbent ≈ 0 でも安定。
//! 絶対 gap で判定すると |UB| ≫ 1 で誤って tight 化、|UB| ≪ 1 で誤って loose 化する。

/// node が ε-optimal 圏内か (= prune 可)。
/// `incumbent` None (= まだ feasible 解見つかってない) は prune できない。
pub(crate) fn should_prune(node_lower_bound: f64, incumbent: Option<f64>, gap_tol: f64) -> bool {
    match incumbent {
        None => false,
        Some(inc) => within_gap(inc, node_lower_bound, gap_tol),
    }
}

/// (incumbent - lower_bound) / max(1, |incumbent|) <= gap_tol。
/// 「incumbent から見て lb が gap_tol 以内まで上がっている」 = 改善余地 ≤ gap_tol。
///
/// `incumbent` / `lower_bound` いずれかが非有限 (+inf sentinel / NaN 等、真の
/// 値を表さない) のときは無条件に `false`。IEEE754 では
/// `(+inf - finite) <= gap_tol * +inf` が `+inf <= +inf` に評価され真になり
/// (incumbent 側)、対称に `(finite - (+inf)) = -inf <= gap_tol * scale` も
/// 常に真になる (lower_bound 側)。前者は「有効な incumbent が一つもない」、
/// 後者は「この node/region の下界が一度も確定していない (bound 計算が
/// objective の有限性を見ずに +inf を fold した)」ことを、どちらも
/// 「最適性証明済み」と取り違える false-Optimal の入口になるため、両方とも
/// ここで閉じる。呼び出し元 (`should_prune` の 2 経路、`within_gap` の直接呼び出し
/// による証明判定) は `false` を「証明不能、探索続行」の既存経路にそのまま
/// フォールバックできる — 追加のブックキーピングは不要 (mip::process_node_
/// outcome / qp::global::solve_qp_global_with_stats の該当 sentinel で検証済み)。
pub(crate) fn within_gap(incumbent: f64, lower_bound: f64, gap_tol: f64) -> bool {
    if !incumbent.is_finite() || !lower_bound.is_finite() {
        return false;
    }
    let scale = 1.0_f64.max(incumbent.abs());
    (incumbent - lower_bound) <= gap_tol * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_prune_without_incumbent() {
        assert!(!should_prune(-100.0, None, 1e-3));
        assert!(!should_prune(0.0, None, 1e-3));
    }

    #[test]
    fn prune_when_lower_bound_within_gap_of_incumbent() {
        // incumbent=-5, lb=-5.001, gap=0.001 / max(1,5) = 2e-4 < 1e-3 → prune
        assert!(should_prune(-5.001, Some(-5.0), 1e-3));
    }

    #[test]
    fn no_prune_when_lower_bound_below_incumbent_minus_gap() {
        // incumbent=-5, lb=-10, gap=5 / 5 = 1.0 > 1e-3 → not prune
        assert!(!should_prune(-10.0, Some(-5.0), 1e-3));
    }

    #[test]
    fn gap_uses_relative_scale_for_large_incumbent() {
        // incumbent=1e6, lb=1e6 - 100. abs gap=100, rel=100/1e6=1e-4 < 1e-3 → prune
        assert!(should_prune(1e6 - 100.0, Some(1e6), 1e-3));
        // 同じ abs gap でも incumbent=10 では rel=10 → not prune
        assert!(!should_prune(10.0 - 100.0, Some(10.0), 1e-3));
    }

    #[test]
    fn gap_clamps_to_unit_scale_near_zero_incumbent() {
        // incumbent=0.001, lb=0, abs gap=0.001, scale=max(1,0.001)=1, rel=1e-3 → ok at tol=1e-3
        assert!(should_prune(0.0, Some(0.001), 1e-3));
        // incumbent=0.001, lb=-0.5, rel=0.501 / 1 = 0.501 → no prune
        assert!(!should_prune(-0.5, Some(0.001), 1e-3));
    }

    /// SENTINEL: a `+inf` incumbent (the "no valid incumbent" sentinel value used
    /// by `SolverResult::infeasible()` / `numerical_error()` / `timeout()`, and by
    /// extension any accidentally-adopted non-finite incumbent) must never be
    /// reported as within the gap of a finite lower bound.
    ///
    /// Without the `is_finite()` guard, IEEE754 makes `(inf - lb) <= gap_tol * inf`
    /// evaluate to `inf <= inf` = true for ANY finite `lb`, which is exactly the
    /// false-Optimal mechanism this guards against: `finalize_mip_result` would
    /// treat a poisoned/absent incumbent as gap-proven and stamp a bogus
    /// `BoundGapCertificate`. Revert the `is_finite()` check to see this fail.
    #[test]
    fn within_gap_rejects_infinite_incumbent_against_finite_lower_bound() {
        assert!(!within_gap(f64::INFINITY, 5.0, 1e-6));
        assert!(!within_gap(f64::INFINITY, -1e9, 1e-6));
        assert!(!within_gap(f64::INFINITY, 0.0, 1.0));
    }

    /// SENTINEL companion: `should_prune` (used by both the MIP driver's node
    /// pruning and the nonconvex QP spatial B&B) must not treat a `+inf`
    /// incumbent as license to prune every remaining finite-bound node — that
    /// would silently truncate the search to nothing while believing it is
    /// optimal.
    #[test]
    fn should_prune_rejects_infinite_incumbent() {
        assert!(!should_prune(5.0, Some(f64::INFINITY), 1e-6));
        assert!(!should_prune(0.0, Some(f64::INFINITY), 1e-6));
    }

    /// `within_gap` must also reject a NaN incumbent (same non-finite family;
    /// NaN comparisons are false either way in the old formula, but the guard
    /// should be the single source of truth rather than relying on that
    /// incidental property).
    #[test]
    fn within_gap_rejects_nan_incumbent() {
        assert!(!within_gap(f64::NAN, 5.0, 1e-6));
    }

    /// SENTINEL (P0, symmetric counterpart of `within_gap_rejects_infinite_
    /// incumbent_against_finite_lower_bound`): a `+inf` `lower_bound` must
    /// never be reported as within the gap of ANY finite incumbent.
    ///
    /// Without the `lower_bound.is_finite()` guard, IEEE754 makes
    /// `(finite - +inf) = -inf <= gap_tol * scale` evaluate to `-inf <= finite`
    /// = true unconditionally — a node/region whose lower bound was never
    /// actually established (e.g. a "trusted" relaxation result folded a
    /// corrupt `+inf` objective into `node_lb` before any leaf-level
    /// finiteness check ran) would be reported as gap-closed against *any*
    /// incumbent, including one found elsewhere in a completely different,
    /// never-explored part of the tree. Revert the `lower_bound.is_finite()`
    /// check to see this fail.
    #[test]
    fn within_gap_rejects_infinite_lower_bound_against_finite_incumbent() {
        assert!(!within_gap(0.0, f64::INFINITY, 1e-6));
        assert!(!within_gap(-5.0, f64::INFINITY, 1e-6));
        assert!(!within_gap(1e9, f64::INFINITY, 1.0));
    }

    /// SENTINEL companion: `should_prune` must not treat a `+inf` node lower
    /// bound as gap-closed against a finite incumbent either — a node whose
    /// own bound computation produced `+inf` (never a genuine dual bound) is
    /// not proof its subtree is exhausted, and silently pruning it here
    /// discards that subtree without any open-region bookkeeping.
    #[test]
    fn should_prune_rejects_infinite_lower_bound() {
        assert!(!should_prune(f64::INFINITY, Some(0.0), 1e-6));
        assert!(!should_prune(f64::INFINITY, Some(-5.0), 1e-6));
    }

    /// `within_gap` must also reject a NaN `lower_bound` (same non-finite
    /// family as the incumbent-side NaN sentinel above).
    #[test]
    fn within_gap_rejects_nan_lower_bound() {
        assert!(!within_gap(0.0, f64::NAN, 1e-6));
    }
}
