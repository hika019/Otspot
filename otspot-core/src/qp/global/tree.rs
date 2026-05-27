//! Best-bound first priority queue for spatial B&B (Phase 3)。
//!
//! `pop` の度に lower_bound 最小の node を返す = best-bound strategy。
//! 同 lower bound は FIFO (seq 小優先) で決定論的に。
//!
//! `std::collections::BinaryHeap` (max-heap) を `Reverse` 順序で wrap。
//! O(log n) push/pop。Phase 3 想定 ≤ 10k node では十分。

use super::node::BBNode;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Entry {
    node: BBNode,
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
        // max-heap を min-heap として使うため lower_bound 逆順、tie は seq 逆順。
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

pub(crate) struct BBTree {
    heap: BinaryHeap<Entry>,
    next_seq: u64,
    pushed: usize,
    popped: usize,
}

impl BBTree {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            pushed: 0,
            popped: 0,
        }
    }
    pub fn push(&mut self, node: BBNode) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.pushed += 1;
        self.heap.push(Entry { node, seq });
    }
    pub fn pop(&mut self) -> Option<BBNode> {
        let r = self.heap.pop().map(|e| e.node);
        if r.is_some() {
            self.popped += 1;
        }
        r
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    #[cfg(test)]
    pub fn pushed(&self) -> usize {
        self.pushed
    }
    #[cfg(test)]
    pub fn popped(&self) -> usize {
        self.popped
    }
    /// 残 node の中で最小 lower bound (= 未探索領域の global lb)。
    /// None = queue 空 = 未探索領域なし (= incumbent が global optimal)。
    pub fn best_lower_bound(&self) -> Option<f64> {
        self.heap.peek().map(|e| e.node.lower_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pop_returns_smallest_lower_bound_first() {
        let mut t = BBTree::new();
        t.push(BBNode::root(vec![(0.0, 1.0)], 3.0));
        t.push(BBNode::root(vec![(0.0, 1.0)], -1.0));
        t.push(BBNode::root(vec![(0.0, 1.0)], 7.0));
        let n1 = t.pop().unwrap();
        assert_eq!(n1.lower_bound, -1.0);
        let n2 = t.pop().unwrap();
        assert_eq!(n2.lower_bound, 3.0);
        let n3 = t.pop().unwrap();
        assert_eq!(n3.lower_bound, 7.0);
        assert!(t.pop().is_none());
    }

    #[test]
    fn tie_break_fifo_seq() {
        let mut t = BBTree::new();
        for i in 0..5 {
            let mut n = BBNode::root(vec![(0.0, 1.0)], 0.0);
            n.depth = i;
            t.push(n);
        }
        for i in 0..5 {
            assert_eq!(t.pop().unwrap().depth, i, "expected FIFO order");
        }
    }

    #[test]
    fn best_lower_bound_matches_top() {
        let mut t = BBTree::new();
        assert!(t.best_lower_bound().is_none());
        t.push(BBNode::root(vec![], 5.0));
        t.push(BBNode::root(vec![], 1.5));
        t.push(BBNode::root(vec![], 3.0));
        assert_eq!(t.best_lower_bound(), Some(1.5));
        t.pop();
        assert_eq!(t.best_lower_bound(), Some(3.0));
    }

    #[test]
    fn pushed_popped_counters_track() {
        let mut t = BBTree::new();
        for _ in 0..3 {
            t.push(BBNode::root(vec![], 0.0));
        }
        t.pop();
        t.pop();
        assert_eq!(t.pushed(), 3);
        assert_eq!(t.popped(), 2);
    }
}
