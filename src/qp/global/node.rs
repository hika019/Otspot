//! Spatial B&B のノード (#6 Phase 3)。
//!
//! ノード = 1 つの box subproblem。`var_bounds` は親 problem の bound を
//! 分枝により狭めた領域。`lower_bound` は親から継承 or 自前 interval bound の
//! 最大値。`warm` は親解 (#12 IPM warm start) を子に引き継ぐためのスナップショット。

use crate::options::QpWarmStart;

#[derive(Debug, Clone)]
pub(crate) struct BBNode {
    pub var_bounds: Vec<(f64, f64)>,
    pub lower_bound: f64,
    pub depth: usize,
    pub warm: Option<QpWarmStart>,
}

impl BBNode {
    pub fn root(var_bounds: Vec<(f64, f64)>, lower_bound: f64) -> Self {
        Self {
            var_bounds,
            lower_bound,
            depth: 0,
            warm: None,
        }
    }

    pub fn child(
        &self,
        new_bounds: Vec<(f64, f64)>,
        lower_bound: f64,
        warm: Option<QpWarmStart>,
    ) -> Self {
        Self {
            var_bounds: new_bounds,
            lower_bound,
            depth: self.depth + 1,
            warm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_node_depth_zero_no_warm() {
        let r = BBNode::root(vec![(0.0, 1.0), (-1.0, 1.0)], -5.0);
        assert_eq!(r.depth, 0);
        assert!(r.warm.is_none());
        assert_eq!(r.var_bounds.len(), 2);
        assert_eq!(r.lower_bound, -5.0);
    }

    #[test]
    fn child_increments_depth_and_propagates_warm() {
        let p = BBNode::root(vec![(0.0, 1.0)], -3.0);
        let warm = QpWarmStart {
            x: vec![0.5],
            y: vec![],
            mu: 1e-6,
        };
        let c = p.child(vec![(0.0, 0.5)], -2.5, Some(warm.clone()));
        assert_eq!(c.depth, 1);
        let w = c.warm.expect("warm propagated");
        assert_eq!(w.x, vec![0.5]);
        assert_eq!(c.lower_bound, -2.5);
    }
}
