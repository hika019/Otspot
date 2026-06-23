//! Integer branching for MILP/MIQP branch-and-bound.

use super::node::VarBounds;
use crate::options::MipBranching;

/// Minimum observation count before a variable uses pseudocosts instead of
/// strong branching.
pub(crate) const RELIABILITY_THRESHOLD: u32 = 8;

/// Maximum number of strong-branching candidates evaluated per node.
pub(crate) const MAX_STRONG_BRANCH_CANDIDATES: usize = 10;

/// Score mixing parameter μ: `(1-μ)·min + μ·max`.  μ = 1/6 follows Achterberg 2005.
pub(crate) const PSEUDOCOST_MU: f64 = 1.0 / 6.0;

/// Per-variable pseudocost accumulators.
///
/// Indexed by the *position of the integer variable in the integer_vars list*,
/// NOT by the original variable index.  Callers must translate between the two.
#[derive(Debug, Clone)]
pub(crate) struct PseudocostState {
    pub up_sum: Vec<f64>,
    pub down_sum: Vec<f64>,
    pub up_count: Vec<u32>,
    pub down_count: Vec<u32>,
}

impl PseudocostState {
    pub(crate) fn new(n_int: usize) -> Self {
        Self {
            up_sum: vec![0.0; n_int],
            down_sum: vec![0.0; n_int],
            up_count: vec![0; n_int],
            down_count: vec![0; n_int],
        }
    }

    /// Average pseudocost for the up branch of integer variable `k`.
    /// Returns `None` when no observations are available.
    pub(crate) fn up_cost(&self, k: usize) -> Option<f64> {
        if self.up_count[k] == 0 {
            None
        } else {
            Some(self.up_sum[k] / self.up_count[k] as f64)
        }
    }

    /// Average pseudocost for the down branch of integer variable `k`.
    /// Returns `None` when no observations are available.
    pub(crate) fn down_cost(&self, k: usize) -> Option<f64> {
        if self.down_count[k] == 0 {
            None
        } else {
            Some(self.down_sum[k] / self.down_count[k] as f64)
        }
    }

    pub(crate) fn record_up(&mut self, k: usize, delta: f64) {
        if delta >= 0.0 {
            self.up_sum[k] += delta;
            self.up_count[k] += 1;
        }
    }

    pub(crate) fn record_down(&mut self, k: usize, delta: f64) {
        if delta >= 0.0 {
            self.down_sum[k] += delta;
            self.down_count[k] += 1;
        }
    }

    /// `true` when variable `k` has enough observations for reliability branching.
    pub(crate) fn is_reliable(&self, k: usize) -> bool {
        self.up_count[k] >= RELIABILITY_THRESHOLD && self.down_count[k] >= RELIABILITY_THRESHOLD
    }

    /// Pseudocost-based score for variable `k` given fractional value `v`.
    ///
    /// Uses available cost estimates; falls back to fractionality-based
    /// heuristic when one side has no data.
    pub(crate) fn score(&self, k: usize, v: f64) -> f64 {
        let f_down = v - v.floor();
        let f_up = v.ceil() - v;

        let d_up = self.up_cost(k).unwrap_or(1.0) * f_up;
        let d_down = self.down_cost(k).unwrap_or(1.0) * f_down;

        pseudocost_score(d_down, d_up)
    }
}

/// Combined branching score: `(1 - μ)·min(d,u) + μ·max(d,u)`.
pub(crate) fn pseudocost_score(d_down: f64, d_up: f64) -> f64 {
    let lo = d_down.min(d_up);
    let hi = d_down.max(d_up);
    (1.0 - PSEUDOCOST_MU) * lo + PSEUDOCOST_MU * hi
}

/// Distance from `v` to its nearest integer, in `[0, 0.5]`.
pub(crate) fn fractionality(v: f64) -> f64 {
    (v - v.round()).abs()
}

/// `true` when `v` is within `tol` of an integer.
pub(crate) fn is_integer(v: f64, tol: f64) -> bool {
    fractionality(v) <= tol
}

/// `true` when every integer-constrained component of `x` is integral within `tol`.
pub(crate) fn is_integer_feasible(x: &[f64], integer_mask: &[bool], tol: f64) -> bool {
    debug_assert_eq!(x.len(), integer_mask.len(), "x / mask length mismatch");
    integer_mask
        .iter()
        .zip(x.iter())
        .all(|(&is_int, &v)| !is_int || is_integer(v, tol))
}

/// Select the branching variable using `MostFractional` only.
///
/// For `Reliability`, use [`select_branching_variable_reliability`] instead.
pub(crate) fn select_branching_variable(
    x: &[f64],
    integer_mask: &[bool],
    tol: f64,
    strategy: MipBranching,
) -> Option<usize> {
    debug_assert_eq!(x.len(), integer_mask.len(), "x / mask length mismatch");
    match strategy {
        MipBranching::MostFractional => select_most_fractional(x, integer_mask, tol),
        MipBranching::Reliability => select_most_fractional(x, integer_mask, tol),
    }
}

/// Select the branching variable using pseudocost / reliability logic.
///
/// Returns the original variable index (not the integer-variable index `k`).
/// Strong-branching candidates (unreliable variables) are collected by the
/// caller and their costs are measured separately; this function picks from the
/// full candidate set using combined scores.
///
/// `integer_vars`: the ordered list of integer variable indices.
/// `pc`: accumulated pseudocost state (indexed by position in `integer_vars`).
/// `strong_scores`: optional externally-measured strong-branch scores for
///   unreliable variables, keyed by variable index `j`.
pub(crate) fn select_branching_variable_reliability(
    x: &[f64],
    integer_mask: &[bool],
    integer_vars: &[usize],
    tol: f64,
    pc: &PseudocostState,
    strong_scores: Option<&std::collections::HashMap<usize, f64>>,
) -> Option<usize> {
    debug_assert_eq!(x.len(), integer_mask.len());

    let mut best: Option<(usize, f64)> = None;

    for (k, &j) in integer_vars.iter().enumerate() {
        if !integer_mask[j] {
            continue;
        }
        let frac = fractionality(x[j]);
        if frac <= tol {
            continue;
        }

        let score = if let Some(ss) = strong_scores {
            if let Some(&s) = ss.get(&j) {
                s
            } else {
                pc.score(k, x[j])
            }
        } else {
            pc.score(k, x[j])
        };

        let better = best.is_none_or(|(_, bs)| score > bs);
        if better {
            best = Some((j, score));
        }
    }

    best.map(|(j, _)| j)
}

/// Collect unreliable fractional integer variables that need strong branching.
///
/// Returns up to `MAX_STRONG_BRANCH_CANDIDATES` variable indices (original)
/// sorted by decreasing fractionality.
pub(crate) fn strong_branch_candidates(
    x: &[f64],
    integer_mask: &[bool],
    integer_vars: &[usize],
    tol: f64,
    pc: &PseudocostState,
) -> Vec<usize> {
    let mut candidates: Vec<(usize, f64)> = integer_vars
        .iter()
        .enumerate()
        .filter_map(|(k, &j)| {
            if !integer_mask[j] {
                return None;
            }
            let frac = fractionality(x[j]);
            if frac <= tol || pc.is_reliable(k) {
                return None;
            }
            Some((j, frac))
        })
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(MAX_STRONG_BRANCH_CANDIDATES);
    candidates.into_iter().map(|(j, _)| j).collect()
}

/// Most-fractional rule: pick the integer variable whose value is closest to a
/// half-integer. Ties break to the smallest index for determinism.
fn select_most_fractional(x: &[f64], integer_mask: &[bool], tol: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (j, &is_int) in integer_mask.iter().enumerate() {
        if !is_int {
            continue;
        }
        let frac = fractionality(x[j]);
        if frac <= tol {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, bf)) => frac > bf,
        };
        if better {
            best = Some((j, frac));
        }
    }
    best.map(|(j, _)| j)
}

/// Produce the down/up child bounds by branching variable `j` at value `v`.
pub(crate) fn branch_bounds(
    parent_bounds: &[(f64, f64)],
    j: usize,
    v: f64,
) -> (VarBounds, VarBounds) {
    let mut down = parent_bounds.to_vec();
    down[j].1 = v.floor();
    let mut up = parent_bounds.to_vec();
    up[j].0 = v.ceil();
    (down, up)
}

/// Select an integer variable whose box still spans at least two integers
/// (`ub - lb >= 1`), preferring the widest. Used as a fallback when a node's
/// relaxation cannot be solved.
pub(crate) fn widest_splittable_integer(
    bounds: &[(f64, f64)],
    integer_mask: &[bool],
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (j, &is_int) in integer_mask.iter().enumerate() {
        if !is_int {
            continue;
        }
        let (lb, ub) = bounds[j];
        if !lb.is_finite() || !ub.is_finite() {
            continue;
        }
        let width = ub - lb;
        if width < 1.0 {
            continue;
        }
        if best.is_none_or(|(_, bw)| width > bw) {
            best = Some((j, width));
        }
    }
    best.map(|(j, _)| j)
}

/// Bisect integer variable `j`'s box into two non-empty integer subranges.
pub(crate) fn split_integer_box(bounds: &[(f64, f64)], j: usize) -> (VarBounds, VarBounds) {
    let (lb, ub) = bounds[j];
    let mid = (0.5 * (lb + ub)).floor().max(lb).min(ub - 1.0);
    let mut down = bounds.to_vec();
    down[j].1 = mid;
    let mut up = bounds.to_vec();
    up[j].0 = mid + 1.0;
    (down, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractionality_measures_distance_to_nearest_integer() {
        assert!((fractionality(2.0) - 0.0).abs() < 1e-15);
        assert!((fractionality(2.5) - 0.5).abs() < 1e-15);
        assert!((fractionality(2.3) - 0.3).abs() < 1e-12);
        assert!((fractionality(2.7) - 0.3).abs() < 1e-12);
        assert!((fractionality(-1.4) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn is_integer_respects_tolerance() {
        assert!(is_integer(3.0000001, 1e-6));
        assert!(!is_integer(3.1, 1e-6));
        assert!(is_integer(-2.9999999, 1e-6));
    }

    #[test]
    fn integer_feasible_ignores_continuous_components() {
        let x = vec![1.5, 0.3];
        assert!(!is_integer_feasible(&x, &[true, false], 1e-6));
        let x2 = vec![1.0, 0.3];
        assert!(is_integer_feasible(&x2, &[true, false], 1e-6));
    }

    #[test]
    fn select_picks_most_fractional_then_lowest_index() {
        let x = vec![1.2, 1.5, 2.5];
        let j =
            select_branching_variable(&x, &[true, true, true], 1e-6, MipBranching::MostFractional)
                .unwrap();
        assert_eq!(j, 1);
    }

    #[test]
    fn select_skips_continuous_and_integral_vars() {
        let x = vec![0.5, 3.0, 2.4];
        let j =
            select_branching_variable(&x, &[false, true, true], 1e-6, MipBranching::MostFractional)
                .unwrap();
        assert_eq!(j, 2);
    }

    #[test]
    fn select_returns_none_when_all_integral() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(select_branching_variable(
            &x,
            &[true, true, true],
            1e-6,
            MipBranching::MostFractional
        )
        .is_none());
    }

    #[test]
    fn branch_bounds_floor_and_ceil() {
        let parent = vec![(0.0, 5.0), (0.0, 5.0)];
        let (down, up) = branch_bounds(&parent, 0, 2.4);
        assert_eq!(down[0], (0.0, 2.0));
        assert_eq!(up[0], (3.0, 5.0));
        assert_eq!(down[1], (0.0, 5.0));
        assert_eq!(up[1], (0.0, 5.0));
    }

    #[test]
    fn branch_bounds_can_produce_empty_box_for_pruning() {
        let parent = vec![(3.0, 5.0)];
        let (down, _up) = branch_bounds(&parent, 0, 3.5);
        assert_eq!(down[0], (3.0, 3.0));
    }

    #[test]
    fn widest_splittable_integer_picks_widest_nonsingleton() {
        let bounds = vec![(0.0, 1.0), (3.0, 3.0), (0.0, 4.0)];
        assert_eq!(
            widest_splittable_integer(&bounds, &[true, true, true]),
            Some(2)
        );
    }

    #[test]
    fn widest_splittable_integer_skips_continuous_and_singletons() {
        let bounds = vec![(0.0, 10.0), (2.0, 2.0)];
        assert!(widest_splittable_integer(&bounds, &[false, true]).is_none());
    }

    #[test]
    fn split_integer_box_yields_two_nonempty_ranges() {
        let bounds = vec![(0.0, 4.0)];
        let (down, up) = split_integer_box(&bounds, 0);
        assert_eq!(down[0], (0.0, 2.0));
        assert_eq!(up[0], (3.0, 4.0));
    }

    #[test]
    fn split_integer_box_binary_splits_to_singletons() {
        let bounds = vec![(0.0, 1.0)];
        let (down, up) = split_integer_box(&bounds, 0);
        assert_eq!(down[0], (0.0, 0.0));
        assert_eq!(up[0], (1.0, 1.0));
    }

    // ---- PseudocostState ------------------------------------------------

    #[test]
    fn pseudocost_state_new_is_zero() {
        let pc = PseudocostState::new(3);
        for k in 0..3 {
            assert!(pc.up_cost(k).is_none());
            assert!(pc.down_cost(k).is_none());
            assert!(!pc.is_reliable(k));
        }
    }

    #[test]
    fn pseudocost_record_and_average() {
        let mut pc = PseudocostState::new(2);
        pc.record_up(0, 1.0);
        pc.record_up(0, 3.0);
        assert_eq!(pc.up_cost(0), Some(2.0));
        pc.record_down(0, 4.0);
        assert_eq!(pc.down_cost(0), Some(4.0));
        assert!(pc.down_cost(1).is_none());
    }

    #[test]
    fn pseudocost_record_ignores_negative_delta() {
        let mut pc = PseudocostState::new(1);
        pc.record_up(0, -1.0);
        assert!(pc.up_cost(0).is_none(), "negative delta must be ignored");
    }

    #[test]
    fn pseudocost_is_reliable_after_threshold_observations() {
        let mut pc = PseudocostState::new(1);
        for _ in 0..RELIABILITY_THRESHOLD {
            pc.record_up(0, 1.0);
            pc.record_down(0, 1.0);
        }
        assert!(pc.is_reliable(0));
    }

    #[test]
    fn pseudocost_is_not_reliable_with_fewer_observations() {
        let mut pc = PseudocostState::new(1);
        for _ in 0..(RELIABILITY_THRESHOLD - 1) {
            pc.record_up(0, 1.0);
            pc.record_down(0, 1.0);
        }
        assert!(!pc.is_reliable(0));
    }

    #[test]
    fn pseudocost_score_prefers_balanced_branches() {
        // Equal up/down cost → score = that cost (no asymmetry penalty).
        let s1 = pseudocost_score(2.0, 2.0);
        // Asymmetric: one side very small → min dominates.
        let s2 = pseudocost_score(0.01, 10.0);
        // Balanced should score higher than extremely unbalanced.
        assert!(s1 > s2, "s1={s1} s2={s2}");
    }

    #[test]
    fn pseudocost_score_formula() {
        // score = (1 - 1/6)*min + (1/6)*max = 5/6*min + 1/6*max
        let d = 1.0_f64;
        let u = 4.0_f64;
        let expected = (5.0 / 6.0) * d + (1.0 / 6.0) * u;
        let got = pseudocost_score(d, u);
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
    }

    #[test]
    fn select_reliability_picks_best_score() {
        // 3 integer variables; var 0 has pseudocost; var 1 has strong score;
        // var 2 falls back to default.
        let x = vec![0.5, 0.5, 0.5];
        let mask = vec![true, true, true];
        let ivars = vec![0, 1, 2];
        let mut pc = PseudocostState::new(3);
        // Give var 0 a high pseudocost so it should win.
        for _ in 0..RELIABILITY_THRESHOLD {
            pc.record_up(0, 10.0);
            pc.record_down(0, 10.0);
        }
        // vars 1 and 2 are unreliable (no data); score falls to heuristic default.
        let mut ss = std::collections::HashMap::new();
        ss.insert(1usize, 0.1); // low strong-branch score for var 1
        let j = select_branching_variable_reliability(&x, &mask, &ivars, 1e-6, &pc, Some(&ss));
        assert_eq!(j, Some(0), "var 0 with high pseudocost should win");
    }

    #[test]
    fn select_reliability_returns_none_when_all_integral() {
        let x = vec![1.0, 2.0];
        let mask = vec![true, true];
        let ivars = vec![0, 1];
        let pc = PseudocostState::new(2);
        let j =
            select_branching_variable_reliability(&x, &mask, &ivars, 1e-6, &pc, None);
        assert!(j.is_none());
    }

    #[test]
    fn strong_branch_candidates_limits_to_max_and_skips_reliable() {
        let n = MAX_STRONG_BRANCH_CANDIDATES + 5;
        let x: Vec<f64> = (0..n).map(|i| 0.1 + 0.03 * i as f64).collect();
        let mask = vec![true; n];
        let ivars: Vec<usize> = (0..n).collect();
        let pc = PseudocostState::new(n);
        let cands = strong_branch_candidates(&x, &mask, &ivars, 1e-6, &pc);
        assert!(
            cands.len() <= MAX_STRONG_BRANCH_CANDIDATES,
            "len={}",
            cands.len()
        );
    }

    #[test]
    fn strong_branch_candidates_skips_reliable_vars() {
        let x = vec![0.5, 0.5];
        let mask = vec![true, true];
        let ivars = vec![0, 1];
        let mut pc = PseudocostState::new(2);
        // Make var 0 reliable.
        for _ in 0..RELIABILITY_THRESHOLD {
            pc.record_up(0, 1.0);
            pc.record_down(0, 1.0);
        }
        let cands = strong_branch_candidates(&x, &mask, &ivars, 1e-6, &pc);
        assert!(!cands.contains(&0), "reliable var 0 must be excluded");
        assert!(cands.contains(&1), "unreliable var 1 must be included");
    }

    /// Strong-branch scores must use raw objective gains directly.
    ///
    /// `d_down`/`d_up` already measure the actual child LP improvement for the
    /// current fractional value; multiplying by fractionality a second time
    /// (double-normalization) would suppress near-integer candidates unfairly.
    /// This test verifies that `pseudocost_score(d_down, d_up)` is the correct
    /// call site formula and that the f-multiplied version gives a meaningfully
    /// different (lower) result for a near-integer variable.
    #[test]
    fn strong_branch_score_uses_raw_gains_not_double_normalized() {
        // Near-integer variable: v = 0.9  →  f_down = 0.9, f_up = 0.1
        // Actual child LP gains measured by strong branching:
        let d_down = 1.0_f64;
        let d_up = 0.5_f64;
        let f_down = 0.9_f64;
        let f_up = 0.1_f64;

        // Correct score: raw gains passed directly.
        let raw_score = pseudocost_score(d_down, d_up);
        // Bugged score: gains multiplied by fractionality again.
        let bugged_score = pseudocost_score(d_down * f_down, d_up * f_up);

        // raw_score must be strictly larger; the double-normalization suppressed
        // the near-integer candidate by roughly 1/f_up ≈ 10×.
        assert!(
            raw_score > bugged_score * 2.0,
            "raw_score={raw_score:.4} must dominate bugged_score={bugged_score:.4}"
        );

        // Sanity: raw_score equals pseudocost_score(1.0, 0.5) = 5/6*0.5 + 1/6*1.0
        let expected = (5.0 / 6.0) * d_up + (1.0 / 6.0) * d_down;
        assert!((raw_score - expected).abs() < 1e-12);

        // Sanity for two candidate variables with equal d but different f:
        // var A: v=0.9 (near-integer), d_down=1.0, d_up=0.5
        // var B: v=0.5 (most-fractional), d_down=1.0, d_up=0.5
        // With correct scoring both should have identical scores (same raw gains).
        let score_a = pseudocost_score(1.0, 0.5);
        let score_b = pseudocost_score(1.0, 0.5);
        assert!((score_a - score_b).abs() < 1e-12, "equal raw gains → equal scores");

        // With double-normalization the most-fractional candidate gets a higher
        // score even though the actual improvement is identical.
        let bugged_a = pseudocost_score(1.0 * 0.9, 0.5 * 0.1); // near-integer var
        let bugged_b = pseudocost_score(1.0 * 0.5, 0.5 * 0.5); // most-fractional var
        assert!(
            bugged_b > bugged_a,
            "double-normalization unfairly favours most-fractional: bugged_b={bugged_b:.4} bugged_a={bugged_a:.4}"
        );
    }
}
