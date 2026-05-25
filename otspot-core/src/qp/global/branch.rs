//! 分枝戦略 (#6 Phase 3 spatial B&B)。
//!
//! MaxViolation: x*[j] が現 box midpoint から最も離れた変数 j* を選び、x*[j] で
//! 2 子に分割する (= 後続 IPM local solve がより細かな box で別 local optimum を
//! 引き出す可能性が高い変数を優先)。
//!
//! 無限境界を持つ変数 (lb / ub = ±∞) は分枝対象外: split point が定義できない。
//! 大域最適化では caller が事前に box 化 (= 全 var に有限 bound) を保証する責務。

use super::node::BBNode;
use crate::options::QpWarmStart;

/// 分枝判定下限: IPM 解 noise (eps=1e-6) や桁落ちを分枝 trigger にしない最小 box 幅。
/// 根拠: IPM tolerance 1e-6 の数倍以上、bound 退化判定 FX_TOL (= 1e-12) より上。
pub(crate) const MIN_BRANCH_BOX_WIDTH: f64 = 1e-6;

/// **MaxViolation** = spatial B&B 文脈での variable selection。
///
/// 連続空間 B&B では純粋な「|x[j] - mid| / width」だけでは concave-symmetric QP の
/// 退化が捕まらない: cold IPM が saddle y=0 に固着し dev_y=0 が永続、never-branched
/// な変数の subbox が縮まず大域 corner に到達しない。
///
/// よって score = (width, dev) **lexicographic** を採用:
///   1. primary: box width 最大 (= 探索余地最大、対称 saddle var 強制分割)
///   2. secondary tie-break: dev = |x[j] - mid| / width (= IPM 解情報を活用)
///   3. tertiary: 変数 index 小 (FIFO、決定論性)
///
/// 結果として「max-violation の概念を spatial B&B に適合させた」branching。
/// Phase 4 で strong branching 等が入る際の baseline。
///
/// 返り値 None: 全変数の box が too narrow (= width <= MIN_BRANCH_BOX_WIDTH) → 分枝不能。
pub(crate) fn select_branching_variable(node: &BBNode, x: &[f64]) -> Option<usize> {
    let n = node.var_bounds.len();
    debug_assert_eq!(n, x.len(), "x length must match node bounds");
    let mut best: Option<(usize, f64, f64)> = None; // (j, width, dev)
    for j in 0..n {
        let (lb, ub) = node.var_bounds[j];
        if !lb.is_finite() || !ub.is_finite() {
            continue;
        }
        let width = ub - lb;
        if width <= MIN_BRANCH_BOX_WIDTH {
            continue;
        }
        let mid = 0.5 * (lb + ub);
        let xj = x[j].clamp(lb, ub);
        let dev = (xj - mid).abs() / width;
        let better = match best {
            None => true,
            Some((_, bw, bd)) => {
                // lexicographic (width, dev): width 大優先、同 width は dev 大優先。
                width > bw || (width == bw && dev > bd)
            }
        };
        if better {
            best = Some((j, width, dev));
        }
    }
    best.map(|(j, _, _)| j)
}

/// Split node: var j を split_at で 2 子に分割。
/// split_at が [lb_j+ε, ub_j-ε] interior 外なら midpoint に fall back
/// (= 縮退分割を避ける、片側 child が parent と同 box になる事故を防ぐ)。
pub(crate) fn split_node(
    parent: &BBNode,
    j: usize,
    split_at: f64,
    parent_warm: Option<QpWarmStart>,
) -> (BBNode, BBNode) {
    let (lb, ub) = parent.var_bounds[j];
    let mid = 0.5 * (lb + ub);
    let s = if split_at > lb + MIN_BRANCH_BOX_WIDTH && split_at < ub - MIN_BRANCH_BOX_WIDTH {
        split_at
    } else {
        mid
    };
    let mut left = parent.var_bounds.clone();
    left[j].1 = s;
    let mut right = parent.var_bounds.clone();
    right[j].0 = s;
    // 子の lower_bound 初期値 = 親 lb (分枝で下界が悪化することはない)。
    // 実 lb は bound::interval_quadratic_bounds で再計算される。
    (
        parent.child(left, parent.lower_bound, parent_warm.clone()),
        parent.child(right, parent.lower_bound, parent_warm),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_widest_box_primary_then_dev_tiebreak() {
        // widths: var0=10, var1=2, var2=100 → widest=var2 (primary)
        let n = BBNode::root(vec![(0.0, 10.0), (-1.0, 1.0), (0.0, 100.0)], 0.0);
        let x = vec![9.0, 0.9, 50.0];
        let j = select_branching_variable(&n, &x).expect("should branch");
        assert_eq!(j, 2, "widest box (var2 width=100) chosen primary");
    }

    #[test]
    fn dev_breaks_ties_among_equal_widths() {
        // 全 width = 2、dev: var0=0.45, var1=0.0, var2=0.4 → var0
        let n = BBNode::root(vec![(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)], 0.0);
        let x = vec![0.9, 0.0, 0.8];
        let j = select_branching_variable(&n, &x).expect("should branch");
        assert_eq!(j, 0, "dev tiebreak picks var0 with highest |x-mid|/width");
    }

    #[test]
    fn skips_infinite_bounds() {
        let n = BBNode::root(
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, 1.0)],
            0.0,
        );
        let x = vec![0.0, 0.9];
        let j = select_branching_variable(&n, &x).expect("should branch");
        assert_eq!(j, 1, "infinite-bound var skipped");
    }

    #[test]
    fn returns_none_when_all_too_narrow() {
        let n = BBNode::root(vec![(0.0, 1e-10), (5.0, 5.0)], 0.0);
        let x = vec![0.0, 5.0];
        assert!(select_branching_variable(&n, &x).is_none());
    }

    #[test]
    fn split_at_interior_uses_provided_point() {
        let p = BBNode::root(vec![(0.0, 10.0)], -1.0);
        let (l, r) = split_node(&p, 0, 3.0, None);
        assert_eq!(l.var_bounds[0], (0.0, 3.0));
        assert_eq!(r.var_bounds[0], (3.0, 10.0));
        assert_eq!(l.depth, 1);
        assert_eq!(r.depth, 1);
    }

    #[test]
    fn split_at_boundary_falls_back_to_midpoint() {
        let p = BBNode::root(vec![(0.0, 10.0)], -1.0);
        // split_at = 0.0 (lb 上) → midpoint = 5.0 にフォールバック
        let (l, r) = split_node(&p, 0, 0.0, None);
        assert_eq!(l.var_bounds[0], (0.0, 5.0));
        assert_eq!(r.var_bounds[0], (5.0, 10.0));
    }

    #[test]
    fn widest_box_dominates_even_when_other_var_has_high_dev() {
        // 退化 saddle (var0 で x=corner, var1 で x=mid) でも widest を優先 →
        // concave-symmetric 問題で全 var を順番に分割できる
        let n = BBNode::root(vec![(-1.0, 0.0), (-1.0, 1.0)], 0.0);
        // var0 width=1, x=-1 corner (dev=0.5); var1 width=2, x=0 midpoint (dev=0)
        let x = vec![-1.0, 0.0];
        let j = select_branching_variable(&n, &x).expect("must branch");
        assert_eq!(j, 1, "widest var (1) must beat narrower high-dev var (0)");
    }

    #[test]
    fn split_propagates_warm_to_both_children() {
        let p = BBNode::root(vec![(0.0, 4.0)], 0.0);
        let warm = QpWarmStart {
            x: vec![2.0],
            y: vec![],
            mu: 1e-6,
        };
        let (l, r) = split_node(&p, 0, 2.0, Some(warm));
        assert!(l.warm.is_some());
        assert!(r.warm.is_some());
    }
}
