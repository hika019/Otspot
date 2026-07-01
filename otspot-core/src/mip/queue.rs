//! Best-bound-first priority queue with depth-first diving for MILP/MIQP B&B.
//!
//! Normal mode: `pop` returns the node with the smallest `lower_bound`.
//! Dive mode: `pop` returns nodes LIFO from a dive stack (depth-first).
//! On `end_dive`, remaining dive-stack nodes are flushed back to the heap.
//!
//! ## Why this duplicates `qp::global::tree::BBTree`
//! This is an intentional small duplicate of the spatial QP B&B queue — **not**
//! an oversight. The two queues are byte-for-byte similar but the node payloads
//! differ (`MipNode` vs `BBNode`). Generalizing over a `lower_bound()` trait would
//! require editing the live non-convex QP B&B while MILP/MIQP is still being brought
//! up, risking a silent regression there. Unifying is deferred to a dedicated
//! refactor that will prove non-convex-QP invariance with a no-op sentinel.

use super::node::MipNode;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Dive one depth-first path every this many best-bound pops (when an incumbent exists).
pub(crate) const DIVE_FREQUENCY: usize = 5;
/// Dive frequency used when no incumbent has been found yet (more aggressive).
pub(crate) const DIVE_FREQUENCY_NO_INCUMBENT: usize = 2;
/// Maximum branching depth (relative to the dive pivot) before a dive is terminated.
pub(crate) const MAX_DIVE_DEPTH: usize = 30;

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
    /// LIFO stack used during depth-first diving phases.
    dive_stack: Vec<MipNode>,
    next_seq: u64,
    diving: bool,
}

impl NodeQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            dive_stack: Vec::new(),
            next_seq: 0,
            diving: false,
        }
    }

    /// Push a node. In dive mode the node goes on the LIFO dive stack;
    /// otherwise it is enqueued on the best-bound heap.
    pub fn push(&mut self, node: MipNode) {
        if self.diving {
            self.dive_stack.push(node);
        } else {
            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Entry { node, seq });
        }
    }

    /// Pop the next node to process.
    ///
    /// In dive mode: LIFO from the dive stack. When the last node is popped from
    /// the dive stack (leaving it empty), diving ends eagerly so that subsequent
    /// `push` calls and `is_diving` checks immediately reflect normal best-bound
    /// mode. Falls back to the best-bound heap when the dive stack is empty.
    pub fn pop(&mut self) -> Option<MipNode> {
        if self.diving {
            if let Some(node) = self.dive_stack.pop() {
                if self.dive_stack.is_empty() {
                    // Last dive node popped — return to best-bound immediately.
                    self.diving = false;
                }
                return Some(node);
            }
            // Stack was already empty; normalise state and fall through.
            self.diving = false;
        }
        self.heap.pop().map(|e| e.node)
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty() && self.dive_stack.is_empty()
    }

    /// Smallest `lower_bound` in the heap (= global lower bound over unexplored regions).
    /// `None` when the heap is empty.
    pub fn best_lower_bound(&self) -> Option<f64> {
        self.heap.peek().map(|e| e.node.lower_bound)
    }

    /// Enter depth-first diving mode. Subsequent `push`/`pop` calls operate on the
    /// LIFO dive stack.
    pub fn start_dive(&mut self) {
        self.diving = true;
    }

    /// Leave diving mode. All remaining dive-stack nodes are flushed back to the
    /// best-bound heap so no subproblem is lost.
    pub fn end_dive(&mut self) {
        self.diving = false;
        for node in self.dive_stack.drain(..) {
            let seq = self.next_seq;
            self.next_seq += 1;
            self.heap.push(Entry { node, seq });
        }
    }

    /// `true` when currently in depth-first diving mode.
    pub fn is_diving(&self) -> bool {
        self.diving
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

    // --- diving mode ---

    #[test]
    fn dive_mode_pops_lifo() {
        let mut q = NodeQueue::new();
        q.push(node(1.0, 0)); // heap
        q.start_dive();
        q.push(node(5.0, 1)); // dive_stack
        q.push(node(6.0, 2)); // dive_stack (pushed last → pops first)
        assert!(q.is_diving());

        // LIFO: last pushed comes out first.
        assert_eq!(q.pop().unwrap().depth, 2, "LIFO: depth-2 node first");
        assert_eq!(q.pop().unwrap().depth, 1, "LIFO: depth-1 node second");
        // dive_stack now empty → auto-end dive, fall back to heap.
        assert!(!q.is_diving(), "auto-end when dive_stack exhausted");
        assert_eq!(q.pop().unwrap().lower_bound, 1.0, "fallback to heap");
        assert!(q.pop().is_none());
    }

    #[test]
    fn end_dive_flushes_stack_to_heap_in_best_bound_order() {
        let mut q = NodeQueue::new();
        q.push(node(5.0, 0)); // heap
        q.start_dive();
        q.push(node(1.0, 1)); // dive_stack
        q.push(node(3.0, 2)); // dive_stack
        q.end_dive();

        assert!(!q.is_diving());
        // After flush: heap contains lb=5.0, lb=1.0, lb=3.0.
        // Best-bound ordering: 1.0 < 3.0 < 5.0.
        assert_eq!(q.pop().unwrap().lower_bound, 1.0);
        assert_eq!(q.pop().unwrap().lower_bound, 3.0);
        assert_eq!(q.pop().unwrap().lower_bound, 5.0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn end_dive_on_empty_stack_is_noop() {
        let mut q = NodeQueue::new();
        q.push(node(2.0, 0));
        q.start_dive();
        q.end_dive(); // nothing in dive_stack to flush
        assert!(!q.is_diving());
        assert_eq!(q.pop().unwrap().lower_bound, 2.0);
    }

    #[test]
    fn is_empty_requires_both_structures_empty() {
        let mut q = NodeQueue::new();
        assert!(q.is_empty());

        q.start_dive();
        q.push(node(1.0, 1)); // on dive_stack
        assert!(!q.is_empty());

        q.end_dive(); // flushed to heap
        assert!(!q.is_empty());

        q.pop();
        assert!(q.is_empty());
    }

    #[test]
    fn best_lower_bound_excludes_dive_stack() {
        let mut q = NodeQueue::new();
        q.push(node(3.0, 0)); // heap
        q.start_dive();
        q.push(node(0.5, 1)); // dive_stack — lower bound, but NOT in heap
                              // best_lower_bound reflects only the heap.
        assert_eq!(q.best_lower_bound(), Some(3.0));
        q.end_dive();
        // After flush both are in heap; 0.5 becomes the best.
        assert_eq!(q.best_lower_bound(), Some(0.5));
    }

    #[test]
    fn start_dive_end_dive_roundtrip() {
        let mut q = NodeQueue::new();
        assert!(!q.is_diving());
        q.start_dive();
        assert!(q.is_diving());
        q.end_dive();
        assert!(!q.is_diving());
    }

    #[test]
    fn dive_depth_first_ordering_within_tree() {
        // Simulate a 3-level dive: root pops best-bound, then two levels of children
        // are pushed depth-first. Verify the depth-first pop ordering: deepest child first.
        let mut q = NodeQueue::new();
        // Seed heap with one pivot node.
        q.push(node(0.0, 0));

        // Pop pivot (best-bound).
        let pivot = q.pop().unwrap();
        assert_eq!(pivot.depth, 0);

        // Start dive; push pivot's two children.
        q.start_dive();
        let child_a = pivot.child(vec![(0.0, 1.0)], 0.5); // depth 1
        let child_b = pivot.child(vec![(0.0, 1.0)], 0.5); // depth 1
        q.push(child_a);
        q.push(child_b);

        // Pop child_b (LIFO), push its child (depth 2).
        let cb = q.pop().unwrap();
        assert_eq!(cb.depth, 1);
        q.push(cb.child(vec![(0.0, 1.0)], 1.0)); // depth 2

        // Next pop is the depth-2 grandchild (deepest first).
        let grandchild = q.pop().unwrap();
        assert_eq!(
            grandchild.depth, 2,
            "depth-first: grandchild before sibling"
        );

        // Remaining: child_a (depth 1) in dive_stack.
        let sibling = q.pop().unwrap();
        assert_eq!(
            sibling.depth, 1,
            "sibling of child_b explored after grandchild"
        );
    }
}
