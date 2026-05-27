//! MILP/MIQP branch-and-bound node.
//!
//! A node is one relaxation subproblem: the parent variable bounds tightened by
//! integer branching (`x_j <= floor(v)` / `x_j >= ceil(v)`). `lower_bound` is the
//! parent relaxation objective (a valid lower bound on every descendant, since
//! tightening bounds can only raise the minimum). `depth` is the branching depth.

/// Per-variable `(lower, upper)` bounds for a relaxation subproblem.
pub(crate) type VarBounds = Vec<(f64, f64)>;

#[derive(Debug, Clone)]
pub(crate) struct MipNode {
    /// Variable bounds for this subproblem's relaxation.
    pub var_bounds: VarBounds,
    /// Lower bound inherited from the parent relaxation objective.
    pub lower_bound: f64,
    /// Branching depth (root = 0).
    pub depth: usize,
}

impl MipNode {
    pub fn root(var_bounds: VarBounds, lower_bound: f64) -> Self {
        Self { var_bounds, lower_bound, depth: 0 }
    }

    pub fn child(&self, new_bounds: VarBounds, lower_bound: f64) -> Self {
        Self { var_bounds: new_bounds, lower_bound, depth: self.depth + 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_has_depth_zero() {
        let r = MipNode::root(vec![(0.0, 1.0), (0.0, 3.0)], -2.0);
        assert_eq!(r.depth, 0);
        assert_eq!(r.lower_bound, -2.0);
        assert_eq!(r.var_bounds.len(), 2);
    }

    #[test]
    fn child_increments_depth() {
        let r = MipNode::root(vec![(0.0, 3.0)], -2.0);
        let c = r.child(vec![(0.0, 1.0)], -1.0);
        assert_eq!(c.depth, 1);
        assert_eq!(c.var_bounds, vec![(0.0, 1.0)]);
        assert_eq!(c.lower_bound, -1.0);
    }
}
