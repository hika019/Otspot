//! 枝刈り判定 (Phase 3 spatial B&B)。
//!
//! 「node の lower bound が現 incumbent (best upper bound) に gap_tol 以内まで
//! 接近している」 = 当該 subtree から incumbent を超える改善は見込めない → prune。
//!
//! gap は相対 = |UB - LB| / max(1, |UB|)。incumbent ≈ 0 でも安定。
//! 絶対 gap で判定すると |UB| ≫ 1 で誤って tight 化、|UB| ≪ 1 で誤って loose 化する。

/// node が ε-optimal 圏内か (= prune 可)。
/// `incumbent` None (= まだ feasible 解見つかってない) は prune できない。
pub(crate) fn should_prune(
    node_lower_bound: f64,
    incumbent: Option<f64>,
    gap_tol: f64,
) -> bool {
    match incumbent {
        None => false,
        Some(inc) => within_gap(inc, node_lower_bound, gap_tol),
    }
}

/// (incumbent - lower_bound) / max(1, |incumbent|) <= gap_tol。
/// 「incumbent から見て lb が gap_tol 以内まで上がっている」 = 改善余地 ≤ gap_tol。
pub(crate) fn within_gap(incumbent: f64, lower_bound: f64, gap_tol: f64) -> bool {
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
}
