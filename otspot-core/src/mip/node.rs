//! MILP/MIQP branch-and-bound node.
//!
//! A node is one relaxation subproblem: the parent variable bounds tightened by
//! integer branching (`x_j <= floor(v)` / `x_j >= ceil(v)`). `lower_bound` is the
//! parent relaxation objective (a valid lower bound on every descendant, since
//! tightening bounds can only raise the minimum). `depth` is the branching depth.

use crate::options::WarmStartBasis;

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
    /// LP basis for warm-starting the child LP. `None` at root, after MIQP
    /// nodes, or when the standard-form layout changes (bound-type mismatch).
    /// The up-branch commonly falls through to cold start (lb-violation check).
    pub warm_start: Option<WarmStartBasis>,
}

impl MipNode {
    pub fn root(var_bounds: VarBounds, lower_bound: f64) -> Self {
        Self { var_bounds, lower_bound, depth: 0, warm_start: None }
    }

    pub fn child(&self, new_bounds: VarBounds, lower_bound: f64) -> Self {
        self.child_warm(new_bounds, lower_bound, None)
    }

    /// Creates a child node with an explicit warm-start basis.
    ///
    /// `warm_start = None` triggers cold start. The simplex also falls back to
    /// cold start when any basic variable violates its lower bound (typical on
    /// the up-branch) or when presolve reduces the child LP.
    pub fn child_warm(
        &self,
        new_bounds: VarBounds,
        lower_bound: f64,
        warm_start: Option<WarmStartBasis>,
    ) -> Self {
        Self { var_bounds: new_bounds, lower_bound, depth: self.depth + 1, warm_start }
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
