//! Cut pool for in-tree MILP separation.
//!
//! Stores cutting-plane rows for one B&B node and chooses cuts to add each round.
//! The pool is **node-local**: in-tree GMI/MIR cuts bake in branching-tightened
//! bounds and are valid only within that node's subtree. Reusing them globally
//! can remove feasible integer points, so [`super::cuts::separate_tree_cuts`]
//! creates a fresh pool per node and ages cuts only across that node's rounds.
//!
//! Filters keep it useful and bounded: violated cuts tighten the relaxation;
//! near-parallel cuts are skipped to avoid redundant basis pressure; and stale
//! cuts are evicted after several unselected rounds.

use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

/// Minimum amount by which `x_star` must violate a cut for it to be added.
/// Below this the cut is numerically already satisfied and cannot tighten the
/// relaxation; matches the feasibility band used by root separation.
const VIOLATION_TOL: f64 = 1e-6;

/// Maximum cosine similarity between two cut normals for both to be retained.
/// Above this the cuts are effectively parallel: the second adds no independent
/// face but doubles the basis pressure, so it is dropped (degeneracy guard).
const MAX_PARALLEL_COSINE: f64 = 0.999;

/// Rounds a cut may go unselected before eviction. Caps pool lifetime so stale
/// rows do not accumulate across a node's separation rounds; 10 comfortably
/// outlives the per-node round budget while still reclaiming dead cuts.
const MAX_UNUSED_ROUNDS: usize = 10;

/// Hard cap on stored cuts. Bounds memory and per-node LP cost on pathological
/// instances; when exceeded the least-recently-used (oldest) cuts are evicted.
const MAX_POOL_SIZE: usize = 512;

/// A single cut row `coeffs · x {>=,<=} rhs` over the original variable space.
#[derive(Clone)]
pub(crate) struct Cut {
    pub coeffs: Vec<f64>,
    pub rhs: f64,
    pub sense: ConstraintType,
}

impl Cut {
    /// Signed violation by `x_star`: positive ⇒ the point breaks the cut.
    fn violation(&self, x_star: &[f64]) -> f64 {
        let lhs: f64 = self.coeffs.iter().zip(x_star).map(|(&g, &x)| g * x).sum();
        match self.sense {
            ConstraintType::Ge => self.rhs - lhs,
            ConstraintType::Le => lhs - self.rhs,
            // Equality cuts are never produced by separation; treat as no violation.
            ConstraintType::Eq => 0.0,
        }
    }
}

/// A pooled cut with its cached norm and idle-round counter.
struct PooledCut {
    cut: Cut,
    norm: f64,
    rounds_unused: usize,
}

/// Persistent store of generated cuts shared across the B&B search.
pub(crate) struct CutPool {
    cuts: Vec<PooledCut>,
}

impl CutPool {
    pub(crate) fn new() -> Self {
        Self { cuts: Vec::new() }
    }

    /// Number of cuts currently held (test/diagnostic hook).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Run one separation round against the node LP point `x_star`.
    ///
    /// 1. Age every stored cut by one round.
    /// 2. Ingest each `candidate` that `x_star` violates and that is not
    ///    near-parallel to an already-stored cut.
    /// 3. Select the violated cuts as a mutually-orthogonal set (greedy by
    ///    violation), resetting their age (they are "used" this round).
    /// 4. Evict cuts idle beyond [`MAX_UNUSED_ROUNDS`] and enforce [`MAX_POOL_SIZE`].
    ///
    /// Returns the selected cuts to append to the node subproblem.
    pub(crate) fn separate_round(&mut self, candidates: Vec<Cut>, x_star: &[f64]) -> Vec<Cut> {
        for pc in &mut self.cuts {
            pc.rounds_unused += 1;
        }

        for cut in candidates {
            let norm = l2_norm(&cut.coeffs);
            if norm <= ZERO_TOL {
                continue;
            }
            if cut.violation(x_star) <= VIOLATION_TOL {
                continue;
            }
            let parallel = self
                .cuts
                .iter()
                .any(|pc| cosine(&pc.cut.coeffs, pc.norm, &cut.coeffs, norm) > MAX_PARALLEL_COSINE);
            if parallel {
                continue;
            }
            self.cuts.push(PooledCut {
                cut,
                norm,
                rounds_unused: 0,
            });
        }

        // Candidate indices violated by x_star, strongest violation first.
        let mut order: Vec<usize> = (0..self.cuts.len())
            .filter(|&i| self.cuts[i].cut.violation(x_star) > VIOLATION_TOL)
            .collect();
        order.sort_by(|&a, &b| {
            self.cuts[b]
                .cut
                .violation(x_star)
                .total_cmp(&self.cuts[a].cut.violation(x_star))
        });

        let mut selected: Vec<usize> = Vec::new();
        for i in order {
            let ni = self.cuts[i].norm;
            let orthogonal = selected.iter().all(|&s| {
                cosine(
                    &self.cuts[s].cut.coeffs,
                    self.cuts[s].norm,
                    &self.cuts[i].cut.coeffs,
                    ni,
                ) <= MAX_PARALLEL_COSINE
            });
            if orthogonal {
                selected.push(i);
            }
        }

        for &i in &selected {
            self.cuts[i].rounds_unused = 0;
        }
        let out: Vec<Cut> = selected.iter().map(|&i| self.cuts[i].cut.clone()).collect();

        self.evict();
        out
    }

    /// Drop idle cuts and enforce the size cap (oldest evicted first).
    fn evict(&mut self) {
        self.cuts.retain(|pc| pc.rounds_unused <= MAX_UNUSED_ROUNDS);
        if self.cuts.len() > MAX_POOL_SIZE {
            self.cuts.sort_by_key(|pc| pc.rounds_unused);
            self.cuts.truncate(MAX_POOL_SIZE);
        }
    }
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// |a·b| / (‖a‖‖b‖); 1.0 ⇒ parallel, 0.0 ⇒ orthogonal. Norms are precomputed.
fn cosine(a: &[f64], norm_a: f64, b: &[f64], norm_b: f64) -> f64 {
    if norm_a <= ZERO_TOL || norm_b <= ZERO_TOL {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    (dot / (norm_a * norm_b)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ge(coeffs: Vec<f64>, rhs: f64) -> Cut {
        Cut {
            coeffs,
            rhs,
            sense: ConstraintType::Ge,
        }
    }

    #[test]
    fn ge_violation_sign() {
        // x >= 1 with x_star = 0 is violated by +1.
        let c = ge(vec![1.0], 1.0);
        assert!((c.violation(&[0.0]) - 1.0).abs() < 1e-12);
        // satisfied point gives negative violation.
        assert!(c.violation(&[2.0]) < 0.0);
    }

    #[test]
    fn le_violation_sign() {
        let c = Cut {
            coeffs: vec![1.0],
            rhs: 1.0,
            sense: ConstraintType::Le,
        };
        // x <= 1 with x_star = 3 is violated by +2.
        assert!((c.violation(&[3.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn satisfied_candidate_is_rejected() {
        let mut pool = CutPool::new();
        // x >= 1 but x_star = 5 satisfies it → not added, not selected.
        let out = pool.separate_round(vec![ge(vec![1.0], 1.0)], &[5.0]);
        assert!(out.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn violated_candidate_is_selected() {
        let mut pool = CutPool::new();
        let out = pool.separate_round(vec![ge(vec![1.0, 0.0], 1.0)], &[0.0, 0.0]);
        assert_eq!(out.len(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn parallel_cut_filtered() {
        let mut pool = CutPool::new();
        // Two near-parallel violated cuts; only one survives ingestion.
        let cuts = vec![ge(vec![1.0, 0.0], 1.0), ge(vec![2.0, 0.0], 2.0)];
        let out = pool.separate_round(cuts, &[0.0, 0.0]);
        assert_eq!(out.len(), 1, "parallel duplicate must be filtered");
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn orthogonal_cuts_both_selected() {
        let mut pool = CutPool::new();
        let cuts = vec![ge(vec![1.0, 0.0], 1.0), ge(vec![0.0, 1.0], 1.0)];
        let out = pool.separate_round(cuts, &[0.0, 0.0]);
        assert_eq!(out.len(), 2, "orthogonal cuts are independent");
    }

    #[test]
    fn aging_evicts_unused_cut() {
        let mut pool = CutPool::new();
        // Add a cut violated at x_star=0.
        pool.separate_round(vec![ge(vec![1.0], 1.0)], &[0.0]);
        assert_eq!(pool.len(), 1);
        // Now repeatedly separate at a point that satisfies it (never selected);
        // after MAX_UNUSED_ROUNDS+1 idle rounds it is evicted.
        for _ in 0..=MAX_UNUSED_ROUNDS {
            pool.separate_round(vec![], &[5.0]);
        }
        assert_eq!(pool.len(), 0, "idle cut must age out");
    }

    #[test]
    fn selected_cut_age_resets() {
        let mut pool = CutPool::new();
        pool.separate_round(vec![ge(vec![1.0], 1.0)], &[0.0]);
        // Keep selecting it (still violated) past the aging horizon; must persist.
        for _ in 0..(MAX_UNUSED_ROUNDS + 5) {
            let out = pool.separate_round(vec![], &[0.0]);
            assert_eq!(out.len(), 1);
        }
        assert_eq!(pool.len(), 1, "repeatedly-used cut must not age out");
    }
}
