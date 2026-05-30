//! Best-bound-first priority queue for MILP/MIQP branch-and-bound.
//!
//! `pop` returns the node with the smallest `lower_bound` (best-bound search);
//! ties break FIFO (smaller insertion `seq` first) for determinism.
//!
//! ## Why this duplicates `qp::global::tree::BBTree`
//! This is an intentional small duplicate of the spatial QP B&B
//! queue — **not** an oversight. The two queues are byte-for-byte similar but the
//! node payloads differ (`MipNode` vs `BBNode`, the latter carries a QP-specific
//! warm start). Generalizing `BBTree` over a `lower_bound()` trait would require
//! editing the live non-convex QP B&B while MILP/MIQP is still being brought up,
//! risking a silent regression there. Unifying the two into one generic engine is
//! deferred to a dedicated refactor that will prove non-convex-QP invariance with
//! a no-op sentinel. Until then this ~40-line copy keeps the new feature isolated.

use super::node::MipNode;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Entry {
    node: MipNode,
    seq: u64,
}

impl Eq for Entry {}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.node.lower_bound == other.node.lower_bound && self.seq == other.seq
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so the smallest lower_bound pops first,
        // and break ties by smaller seq (FIFO).
        other
            .node
            .lower_bound
            .partial_cmp(&self.node.lower_bound)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct NodeQueue {
    heap: BinaryHeap<Entry>,
    next_seq: u64,
}

impl NodeQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
        }
    }

    pub fn push(&mut self, node: MipNode) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(Entry { node, seq });
    }

    pub fn pop(&mut self) -> Option<MipNode> {
        self.heap.pop().map(|e| e.node)
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Smallest `lower_bound` still queued (= global lower bound on the
    /// unexplored region). `None` when the queue is empty.
    pub fn best_lower_bound(&self) -> Option<f64> {
        self.heap.peek().map(|e| e.node.lower_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(lb: f64, depth: usize) -> MipNode {
        let mut n = MipNode::root(vec![(0.0, 1.0)], lb);
        n.depth = depth;
        n
    }

    #[test]
    fn pops_smallest_lower_bound_first() {
        let mut q = NodeQueue::new();
        q.push(node(3.0, 0));
        q.push(node(-1.0, 0));
        q.push(node(7.0, 0));
        assert_eq!(q.pop().unwrap().lower_bound, -1.0);
        assert_eq!(q.pop().unwrap().lower_bound, 3.0);
        assert_eq!(q.pop().unwrap().lower_bound, 7.0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn ties_break_fifo() {
        let mut q = NodeQueue::new();
        for i in 0..5 {
            q.push(node(0.0, i));
        }
        for i in 0..5 {
            assert_eq!(q.pop().unwrap().depth, i, "expected FIFO order on ties");
        }
    }

    #[test]
    fn best_lower_bound_tracks_top() {
        let mut q = NodeQueue::new();
        assert!(q.best_lower_bound().is_none());
        q.push(node(5.0, 0));
        q.push(node(1.5, 0));
        q.push(node(3.0, 0));
        assert_eq!(q.best_lower_bound(), Some(1.5));
        q.pop();
        assert_eq!(q.best_lower_bound(), Some(3.0));
    }
}
