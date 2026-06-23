//! Conflict analysis for branch-and-bound.
//!
//! When a node is found infeasible, the branching decisions that led there are
//! recorded as a *conflict clause*: a set of bound tightenings that together
//! cause infeasibility. Future nodes whose bounds subsume a stored clause are
//! pruned without solving the relaxation.

/// Maximum number of conflict clauses retained.
pub(crate) const MAX_CONFLICTS: usize = 1000;

/// Which bound was tightened by a branching decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundType {
    Lower,
    Upper,
}

/// A single conflict clause: all items must hold simultaneously for the clause
/// to trigger. Each item is `(variable index, bound direction, bound value)`.
#[derive(Debug, Clone)]
pub(crate) struct ConflictClause {
    pub items: Vec<(usize, BoundType, f64)>,
}

/// Store of learned conflict clauses. Bounded to [`MAX_CONFLICTS`] entries;
/// new clauses are silently dropped when the store is full.
#[derive(Debug, Default)]
pub(crate) struct ConflictStore {
    clauses: Vec<ConflictClause>,
}

impl ConflictStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Learn a conflict clause from an infeasible node.
    ///
    /// Compares `node_bounds` against `root_bounds` and records the bounds that
    /// were tightened by branching. Empty clauses (no tightening vs. root) are
    /// not stored since they would prune every future node.
    pub(crate) fn learn(&mut self, node_bounds: &[(f64, f64)], root_bounds: &[(f64, f64)]) {
        if self.clauses.len() >= MAX_CONFLICTS {
            return;
        }
        debug_assert_eq!(
            node_bounds.len(),
            root_bounds.len(),
            "bounds length mismatch in conflict learning"
        );
        let mut items = Vec::new();
        for (j, (&(node_lb, node_ub), &(root_lb, root_ub))) in
            node_bounds.iter().zip(root_bounds.iter()).enumerate()
        {
            if node_lb > root_lb + f64::EPSILON {
                items.push((j, BoundType::Lower, node_lb));
            }
            if node_ub < root_ub - f64::EPSILON {
                items.push((j, BoundType::Upper, node_ub));
            }
        }
        if !items.is_empty() {
            self.clauses.push(ConflictClause { items });
        }
    }

    /// Returns `true` when `node_bounds` subsumes at least one stored conflict
    /// clause, meaning this node is dominated by a known infeasible region and
    /// can be pruned without solving the relaxation.
    pub(crate) fn is_conflicted(&self, node_bounds: &[(f64, f64)]) -> bool {
        self.clauses.iter().any(|clause| {
            clause.items.iter().all(|&(j, bound_type, value)| {
                let Some(&(lb, ub)) = node_bounds.get(j) else {
                    return false;
                };
                match bound_type {
                    BoundType::Lower => lb >= value - f64::EPSILON,
                    BoundType::Upper => ub <= value + f64::EPSILON,
                }
            })
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.clauses.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(pairs: &[(f64, f64)]) -> Vec<(f64, f64)> {
        pairs.to_vec()
    }

    #[test]
    fn learn_records_tightened_lb() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        let node = bounds(&[(5.0, 10.0), (0.0, 10.0)]);
        store.learn(&node, &root);
        assert_eq!(store.len(), 1);
        assert_eq!(store.clauses[0].items.len(), 1);
        assert_eq!(store.clauses[0].items[0], (0, BoundType::Lower, 5.0));
    }

    #[test]
    fn learn_records_tightened_ub() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        let node = bounds(&[(0.0, 3.0), (0.0, 10.0)]);
        store.learn(&node, &root);
        assert_eq!(store.len(), 1);
        assert_eq!(store.clauses[0].items[0], (0, BoundType::Upper, 3.0));
    }

    #[test]
    fn learn_records_both_directions() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        let node = bounds(&[(2.0, 7.0), (0.0, 10.0)]);
        store.learn(&node, &root);
        assert_eq!(store.clauses[0].items.len(), 2);
    }

    #[test]
    fn learn_skips_empty_clause() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0)]);
        let node = bounds(&[(0.0, 10.0)]);
        store.learn(&node, &root);
        assert_eq!(store.len(), 0, "empty clause must not be stored");
    }

    #[test]
    fn learn_respects_max_conflicts_cap() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0)]);
        for i in 0..MAX_CONFLICTS {
            let node = bounds(&[((i + 1) as f64, 10.0)]);
            store.learn(&node, &root);
        }
        assert_eq!(store.len(), MAX_CONFLICTS);
        let node = bounds(&[(MAX_CONFLICTS as f64 + 1.0, 10.0)]);
        store.learn(&node, &root);
        assert_eq!(store.len(), MAX_CONFLICTS, "must not exceed MAX_CONFLICTS");
    }

    #[test]
    fn is_conflicted_prunes_node_matching_clause() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        let infeasible = bounds(&[(5.0, 10.0), (0.0, 3.0)]);
        store.learn(&infeasible, &root);

        assert!(store.is_conflicted(&bounds(&[(5.0, 10.0), (0.0, 3.0)])));

        // Tighter bounds also match
        assert!(store.is_conflicted(&bounds(&[(6.0, 10.0), (0.0, 2.0)])));
    }

    #[test]
    fn is_conflicted_does_not_prune_relaxed_node() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        let infeasible = bounds(&[(5.0, 10.0), (0.0, 3.0)]);
        store.learn(&infeasible, &root);

        // Only one of the two tightenings
        assert!(!store.is_conflicted(&bounds(&[(5.0, 10.0), (0.0, 10.0)])));
        // Looser lb
        assert!(!store.is_conflicted(&bounds(&[(4.0, 10.0), (0.0, 3.0)])));
    }

    #[test]
    fn is_conflicted_empty_store_returns_false() {
        let store = ConflictStore::new();
        assert!(!store.is_conflicted(&bounds(&[(5.0, 10.0)])));
    }

    #[test]
    fn is_conflicted_matches_any_clause() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0), (0.0, 10.0)]);
        store.learn(&bounds(&[(5.0, 10.0), (0.0, 10.0)]), &root);
        store.learn(&bounds(&[(0.0, 10.0), (0.0, 3.0)]), &root);

        assert!(store.is_conflicted(&bounds(&[(5.0, 10.0), (0.0, 10.0)])));
        assert!(store.is_conflicted(&bounds(&[(0.0, 10.0), (0.0, 3.0)])));
        // Matches neither clause alone
        assert!(!store.is_conflicted(&bounds(&[(4.0, 10.0), (0.0, 4.0)])));
    }

    /// Sentinel: removing the `is_empty()` guard in `learn` causes empty clauses
    /// to be stored, and `is_conflicted` would then prune every node.
    #[test]
    fn empty_clause_does_not_prune_nodes() {
        let mut store = ConflictStore::new();
        let root = bounds(&[(0.0, 10.0)]);
        let node_no_change = bounds(&[(0.0, 10.0)]);
        store.learn(&node_no_change, &root);
        // Even with an empty clause stored (if sentinel fails), a real node must not be pruned
        // because that would cause over-pruning. We verify by checking that an arbitrary node
        // is not flagged as conflicted when no tightening was recorded.
        assert!(!store.is_conflicted(&bounds(&[(3.0, 7.0)])));
    }
}
