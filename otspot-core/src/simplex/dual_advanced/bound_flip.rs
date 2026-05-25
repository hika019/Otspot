//! Bound-Flipping Ratio Test (BFRT, Maros 2003 §3.7).
//!
//! Classical Harris ratio test stops at the smallest breakpoint
//! θ* = min { r_j / α_j : α_j > 0 } and selects the corresponding entering
//! column. BFRT exploits non-basic variables that carry a *finite* upper bound
//! u_j: when the dual step θ would push some r_k below 0, the corresponding
//! variable k can be **flipped to its upper bound** instead of becoming the
//! entering variable. The basis is unchanged for k (only its non-basic value
//! switches from 0 to u_k), so the dual step can continue past θ_k to the next
//! breakpoint. The chosen θ is the one that maximises the cumulative dual
//! objective improvement.
//!
//! References: Maros (2003), "Computational Techniques of the Simplex Method",
//! §7.6 (dual simplex with bounded variables); HiGHS / CPLEX / Gurobi all use
//! a variant of this algorithm. Reported pivot reductions: 30-60 % on
//! bound-rich LPs (pilot87, pds-*).
//!
//! ## Sign convention
//!
//! Matches `HarrisRatioTest` in `ratio_test.rs`:
//! - leaving row r has `x_B[r] < 0`
//! - `trow[j] = (B^{-1} a_j)[r]` (the dual coefficient on column j)
//! - we look at columns with `trow[j] > pivot_tol` (Harris-compatible)
//! - reduced cost `r_j ≥ 0` is the dual feasibility margin
//! - breakpoint `θ_j = r_j / trow[j] ≥ 0`
//! - dual step magnitude is bounded by the cumulative residual `|x_B[r]|`
//!   minus the contribution of variables already flipped
//!   (`u_k * trow[k]` per flip)
//!
//! A variable currently at upper bound (`at_upper[j] = true`) participates
//! symmetrically: its breakpoint comes from `-r_j / -trow[j]` and a flip
//! returns it to its lower bound.

use std::cell::Cell;

/// Smallest |Δθ| considered a real breakpoint advance. Below this we treat
/// successive breakpoints as a tie and prefer the larger |pivot| for numerical
/// stability — same rationale as Harris pass 2 in `HarrisRatioTest`.
///
/// Magnitude rationale: PIVOT_TOL (1e-8) is the canonical "numerically zero"
/// boundary; BFRT inherits it so tied-ratio handling is consistent across
/// strategies. Lowering risks selecting an unstable pivot; raising risks
/// merging genuinely distinct breakpoints and inflating the dual step.
pub(crate) const BFRT_TIE_TOL: f64 = 1e-8;

// Process-global probe counter incremented every time BFRT *successfully*
// returns a result that includes ≥ 1 bound flip. Sentinel tests read this to
// prove wiring is live — Harris-equivalent calls leave it at 0.
// `Cell` + `thread_local!` keeps the hook cheap (no atomic on hot path) and
// test-friendly (each `#[test]` thread sees an independent counter).
thread_local! {
    static BFRT_FLIP_INVOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Reset the per-thread BFRT flip counter. Test-only helper.
pub fn reset_bfrt_flip_invocations() {
    BFRT_FLIP_INVOCATIONS.with(|c| c.set(0));
}

/// Read the per-thread BFRT flip counter.
pub fn bfrt_flip_invocations() -> u64 {
    BFRT_FLIP_INVOCATIONS.with(|c| c.get())
}

pub(super) fn bump_bfrt_flip_invocations() {
    BFRT_FLIP_INVOCATIONS.with(|c| c.set(c.get().saturating_add(1)));
}

/// Per-column metadata for BFRT.
#[derive(Debug, Clone, Copy)]
pub struct ColBound {
    /// Upper bound of the variable in shifted form (lb = 0 always). `f64::INFINITY`
    /// means unbounded above (degenerates to Harris for this column).
    pub upper: f64,
    /// `true` if the variable is currently non-basic at its upper bound;
    /// `false` if at its lower bound (= 0). Basic variables: value is
    /// irrelevant (caller skips them via `is_basic`).
    pub at_upper: bool,
}

/// Outcome of the BFRT ratio test.
#[derive(Debug, Clone)]
pub struct BfrtResult {
    /// Entering column.
    pub entering_col: usize,
    /// Dual step magnitude (= breakpoint of the entering column).
    pub theta: f64,
    /// Columns that should switch bound (flip lb↔ub) before the entering
    /// column enters the basis. The basis itself is unchanged for these.
    pub flips: Vec<usize>,
}

/// 4-step BFRT (Maros 2003).
///
/// 1. Enumerate breakpoints `θ_j = r_j / α_j` for compatible columns.
///    A column is *compatible* if it can absorb a positive dual step:
///    - at lower bound (`at_upper[j] = false`) with `trow[j] > pivot_tol`
///    - at upper bound (`at_upper[j] = true`) with `trow[j] < -pivot_tol`
///      (the breakpoint is then `(-r_j) / (-trow[j])` ≥ 0)
/// 2. Sort breakpoints by θ ascending.
/// 3. Walk breakpoints; track remaining residual `R = |x_B[r]| − Σ u_k |α_k|`.
///    Each crossing flips column k and consumes `u_k · |α_k|` of residual.
///    Stop as soon as R ≤ 0 — the column at which R first becomes ≤ 0 is the
///    entering column (the basis must pivot here, no flip past).
/// 4. Return (entering, θ, flips).
///
/// **Tie handling** (memo: `feedback_test_multi_data_pattern`): when several
/// breakpoints fall within `BFRT_TIE_TOL` of the chosen θ, prefer the one
/// with the largest |pivot| as the entering column for numerical stability
/// (mirrors Harris pass 2).
///
/// Returns `None` when no compatible column exists (dual unbounded → primal
/// infeasible). Returns `Some(BfrtResult)` with `flips.is_empty()` and a
/// Harris-equivalent θ when *all* compatible columns have infinite upper bound
/// (i.e., the LP has no exploitable bounded structure) — this keeps the
/// wrapper drop-in.
pub fn bfrt_select_entering(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    bounds: &[ColBound],
    n_price: usize,
    pivot_tol: f64,
    leaving_residual: f64,
) -> Option<BfrtResult> {
    debug_assert!(trow.len() >= n_price);
    debug_assert!(reduced_costs.len() >= n_price);
    debug_assert!(is_basic.len() >= n_price);
    debug_assert!(bounds.len() >= n_price);

    // Step 1: collect compatible breakpoints.
    // Each entry: (theta, j, |pivot|, weight) where weight = u_j * |trow[j]|
    // is the residual consumed if we cross this breakpoint (= flip variable j).
    // For infinite upper bound the column cannot be flipped (no other bound to
    // move to), so we set weight = +∞ which forces the walk to stop at it.
    let mut breaks: Vec<(f64, usize, f64, f64)> = Vec::new();
    for j in 0..n_price {
        if is_basic[j] {
            continue;
        }
        let a = trow[j];
        let r = reduced_costs[j];
        let b = &bounds[j];
        let (theta, abs_pivot) = if !b.at_upper && a > pivot_tol {
            (r / a, a.abs())
        } else if b.at_upper && a < -pivot_tol {
            // r_j ≤ 0 at upper bound; -r/-a = r/a but both sign-flipped → positive.
            ((-r) / (-a), a.abs())
        } else {
            continue;
        };
        let weight = if b.upper.is_finite() {
            b.upper * abs_pivot
        } else {
            f64::INFINITY
        };
        breaks.push((theta, j, abs_pivot, weight));
    }

    if breaks.is_empty() {
        return None;
    }

    // Step 2: sort by theta ascending. Stable sort keeps deterministic
    // behavior across breakpoints with identical θ.
    breaks.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Step 3: walk breakpoints, tracking residual.
    let residual_target = leaving_residual.abs();
    let mut residual = residual_target;
    let mut flips: Vec<usize> = Vec::new();
    let mut entering_idx: usize = 0;
    let mut found = false;
    for (k, &(_theta, _j, _abs_pivot, weight)) in breaks.iter().enumerate() {
        // Residual after passing this breakpoint = residual - weight.
        // If residual would go ≤ 0, this breakpoint is the entering column.
        if residual <= weight {
            entering_idx = k;
            found = true;
            break;
        }
        residual -= weight;
        // Columns crossed but not selected as entering = flip candidates.
        // Only finite-upper columns can flip; infinite-upper columns would
        // have weight = +∞ and the loop would have broken above.
    }

    if !found {
        // Residual never absorbed — all compatible columns are bounded and
        // their combined slack still cannot cover the leaving violation.
        // Standard Maros: pick the last breakpoint as entering (the dual step
        // is capped there by infeasibility detection in the caller).
        entering_idx = breaks.len() - 1;
    }

    // Step 4: tie-aware entering selection. Among breakpoints within
    // BFRT_TIE_TOL of the chosen θ, prefer the largest |pivot|. Tied
    // breakpoints earlier in the sort still count as flips of preceding
    // variables (we already consumed their weight from the residual).
    let chosen_theta = breaks[entering_idx].0;
    let mut best_idx = entering_idx;
    let mut best_pivot = breaks[entering_idx].2;
    for (k, &(theta, _j, abs_pivot, _w)) in breaks.iter().enumerate().skip(entering_idx + 1) {
        if (theta - chosen_theta).abs() > BFRT_TIE_TOL {
            break;
        }
        if abs_pivot > best_pivot {
            best_pivot = abs_pivot;
            best_idx = k;
        }
    }
    // Tied losers in [entering_idx, best_idx) are flips iff they have a
    // finite upper bound — an ∞-upper column has no second bound to flip to
    // and downstream wiring would corrupt state if it tried.
    for k in entering_idx..best_idx {
        let col = breaks[k].1;
        if bounds[col].upper.is_finite() {
            flips.push(col);
        }
    }
    // Flips that occurred during the residual walk (before entering_idx).
    let mut flips_pre: Vec<usize> = (0..entering_idx).map(|k| breaks[k].1).collect();
    flips_pre.append(&mut flips);
    let flips = flips_pre;

    if !flips.is_empty() {
        bump_bfrt_flip_invocations();
    }

    Some(BfrtResult {
        entering_col: breaks[best_idx].1,
        theta: breaks[best_idx].0,
        flips,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances::PIVOT_TOL;

    fn lb_bounds(uppers: &[f64]) -> Vec<ColBound> {
        uppers
            .iter()
            .map(|&u| ColBound { upper: u, at_upper: false })
            .collect()
    }

    fn no_basic(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    /// Harris-equivalence: no flippable structure (all uppers infinite) → BFRT
    /// must reproduce Harris' choice (the smallest breakpoint).
    #[test]
    fn bfrt_no_finite_upper_matches_harris() {
        let trow = vec![1.0, 2.0, 3.0];
        let r = vec![0.3, 0.4, 0.9];
        // breakpoints: 0.3, 0.2, 0.3 → Harris picks j=1 (θ=0.2)
        let bounds = lb_bounds(&[f64::INFINITY; 3]);
        let result =
            bfrt_select_entering(&trow, &r, &no_basic(3), &bounds, 3, PIVOT_TOL, 100.0).unwrap();
        assert_eq!(result.entering_col, 1);
        assert!((result.theta - 0.2).abs() < 1e-9);
        assert!(result.flips.is_empty(), "no finite uppers → no flips");
    }

    /// 2-flip example: a small leading breakpoint absorbs a small slice of
    /// residual; BFRT should flip past it and pick a later entering with
    /// larger θ.
    #[test]
    fn bfrt_flips_past_small_breakpoint() {
        // breakpoints: j=0: θ=0.1 (u=1, |α|=1, weight=1)
        //              j=1: θ=0.5 (u=1, |α|=2, weight=2)
        //              j=2: θ=1.0 (u=∞, weight=∞)
        // leaving residual = 1.5 → flipping j=0 absorbs 1, residual=0.5
        //                          0.5 ≤ weight(j=1)=2 → entering j=1, θ=0.5
        // Harris would have picked j=0 with θ=0.1.
        let trow = vec![1.0, 2.0, 0.5];
        let r = vec![0.1, 1.0, 0.5];
        let bounds = vec![
            ColBound { upper: 1.0, at_upper: false },
            ColBound { upper: 1.0, at_upper: false },
            ColBound { upper: f64::INFINITY, at_upper: false },
        ];
        let res = bfrt_select_entering(&trow, &r, &no_basic(3), &bounds, 3, PIVOT_TOL, 1.5).unwrap();
        assert_eq!(res.entering_col, 1, "BFRT should skip j=0 and pick j=1");
        assert!((res.theta - 0.5).abs() < 1e-9);
        assert_eq!(res.flips, vec![0], "j=0 must be marked as a flip");
    }

    /// Multi-flip: 3 small bounded breakpoints + one infinite. Residual
    /// large enough to absorb all 3 flips → entering at the infinite-upper
    /// breakpoint.
    #[test]
    fn bfrt_flips_three_then_enters_at_infinite() {
        // j=0: θ=0.1 weight=1 (u=1, |α|=1)
        // j=1: θ=0.2 weight=2 (u=2, |α|=1)
        // j=2: θ=0.3 weight=3 (u=3, |α|=1)
        // j=3: θ=0.4 weight=∞ (u=∞, |α|=1)
        // residual=10 → flips=[0,1,2] (consume 6), entering=j=3
        let trow = vec![1.0, 1.0, 1.0, 1.0];
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let bounds = vec![
            ColBound { upper: 1.0, at_upper: false },
            ColBound { upper: 2.0, at_upper: false },
            ColBound { upper: 3.0, at_upper: false },
            ColBound { upper: f64::INFINITY, at_upper: false },
        ];
        let res = bfrt_select_entering(&trow, &r, &no_basic(4), &bounds, 4, PIVOT_TOL, 10.0).unwrap();
        assert_eq!(res.entering_col, 3);
        assert_eq!(res.flips, vec![0, 1, 2]);
        assert!((res.theta - 0.4).abs() < 1e-9);
    }

    /// at_upper case: a column currently at its upper bound contributes a
    /// negative `trow` and a non-positive reduced cost. The breakpoint is
    /// still positive (= r/a with both signs flipped); flipping returns the
    /// variable to its lower bound.
    #[test]
    fn bfrt_handles_at_upper_columns() {
        // j=0 at upper, trow=-1, r=-0.2 → θ = (-(-0.2))/(-(-1)) = 0.2, weight = 1
        // j=1 at lower, trow=2, r=0.6 → θ=0.3, weight = ∞
        // residual=0.5 → flip j=0 (consume 1, but residual=0.5 ≤ 1 → entering=j=0?)
        // Wait: residual=0.5, weight(j=0)=1, residual ≤ weight → entering=j=0
        // So *no* flips, entering=j=0 at θ=0.2. Test the at_upper sign math.
        let trow = vec![-1.0, 2.0];
        let r = vec![-0.2, 0.6];
        let bounds = vec![
            ColBound { upper: 1.0, at_upper: true },
            ColBound { upper: f64::INFINITY, at_upper: false },
        ];
        let res = bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 0.5).unwrap();
        assert_eq!(res.entering_col, 0);
        assert!((res.theta - 0.2).abs() < 1e-9);
        assert!(res.flips.is_empty());
    }

    /// No compatible column → None (dual unbounded → primal infeasible).
    #[test]
    fn bfrt_returns_none_when_no_compatible_column() {
        let trow = vec![-1.0, -2.0];
        let r = vec![0.1, 0.2];
        // all at lower bound but trow < 0 → none compatible
        let bounds = lb_bounds(&[1.0, 1.0]);
        let res = bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 1.0);
        assert!(res.is_none());
    }

    /// Tie-breaking: two breakpoints within BFRT_TIE_TOL, prefer larger |pivot|.
    #[test]
    fn bfrt_tie_prefers_larger_pivot() {
        // j=0: trow=1, r=0.1 → θ=0.1, |pivot|=1, weight=u*|α|=∞
        // j=1: trow=5, r=0.5 → θ=0.1, |pivot|=5, weight=∞
        // residual=10 → enters at first weight=∞, but with tie → pick j=1 (largest |pivot|)
        let trow = vec![1.0, 5.0];
        let r = vec![0.1, 0.5];
        let bounds = lb_bounds(&[f64::INFINITY, f64::INFINITY]);
        let res = bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
        assert_eq!(res.entering_col, 1, "larger |pivot| should win the tie");
        // Reviewer P1: tie-zone losers must not be pushed as flips when their
        // upper bound is infinite — there is no other bound to flip to, and a
        // downstream caller iterating flips blindly would corrupt state.
        assert!(
            res.flips.iter().all(|&f| bounds[f].upper.is_finite()),
            "flips must not contain infinite-upper columns: {:?}",
            res.flips,
        );
    }

    /// Reviewer P1 regression: when *all* tie-zone candidates have infinite
    /// upper, the loser cannot be marked as a flip. Minimal reproduction —
    /// independent of the tie-breaker outcome.
    #[test]
    fn bfrt_tie_excludes_infinite_upper() {
        let trow = vec![1.0, 5.0];
        let r = vec![0.1, 0.5];
        let bounds = lb_bounds(&[f64::INFINITY, f64::INFINITY]);
        let res = bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
        assert!(
            res.flips.iter().all(|&f| bounds[f].upper.is_finite()),
            "no infinite-upper flips even on tie, got: {:?}",
            res.flips,
        );
    }

    /// Reviewer P1 regression: mixed tie zone (one finite, one infinite). The
    /// finite-upper tie loser is still a legitimate flip; the infinite one
    /// must be filtered out.
    #[test]
    fn bfrt_tie_filters_only_infinite_upper() {
        // j=0: trow=1, r=0.1, u=1   → θ=0.1, |pivot|=1, weight=1
        // j=1: trow=5, r=0.5, u=∞   → θ=0.1, |pivot|=5, weight=∞
        // residual=10 → walk: residual(10) > weight(1) → flip j=0, residual=9
        //               at k=1, weight=∞ → entering=j=1 (already chosen)
        // No tie loop swap needed (entering is already the larger-|pivot|).
        // The finite j=0 is a real walk-flip; that path is unaffected by the fix.
        let trow = vec![1.0, 5.0];
        let r = vec![0.1, 0.5];
        let bounds = vec![
            ColBound { upper: 1.0, at_upper: false },
            ColBound { upper: f64::INFINITY, at_upper: false },
        ];
        let res = bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
        assert_eq!(res.entering_col, 1);
        assert_eq!(res.flips, vec![0], "finite-upper walk-flip must survive");
        assert!(
            res.flips.iter().all(|&f| bounds[f].upper.is_finite()),
            "no infinite-upper in flips: {:?}",
            res.flips,
        );
    }

    /// Skip basic columns.
    #[test]
    fn bfrt_skips_basic_columns() {
        let trow = vec![5.0, 1.0];
        let r = vec![0.1, 0.5];
        let bounds = lb_bounds(&[f64::INFINITY, f64::INFINITY]);
        let is_basic = vec![true, false];
        let res = bfrt_select_entering(&trow, &r, &is_basic, &bounds, 2, PIVOT_TOL, 10.0).unwrap();
        assert_eq!(res.entering_col, 1);
    }

    /// Probe counter: increments only when a real flip occurs.
    #[test]
    fn bfrt_flip_counter_increments_only_when_flipping() {
        reset_bfrt_flip_invocations();

        // Case 1: no flips → counter stays 0
        let bounds = lb_bounds(&[f64::INFINITY; 2]);
        let _ = bfrt_select_entering(
            &[1.0, 2.0],
            &[0.3, 0.4],
            &no_basic(2),
            &bounds,
            2,
            PIVOT_TOL,
            10.0,
        );
        assert_eq!(bfrt_flip_invocations(), 0);

        // Case 2: a real flip → counter increments by 1
        let bounds = vec![
            ColBound { upper: 1.0, at_upper: false },
            ColBound { upper: f64::INFINITY, at_upper: false },
        ];
        let _ = bfrt_select_entering(
            &[1.0, 1.0],
            &[0.1, 0.5],
            &no_basic(2),
            &bounds,
            2,
            PIVOT_TOL,
            5.0,
        );
        assert_eq!(bfrt_flip_invocations(), 1);

        // Case 3: another flip → counter = 2
        let _ = bfrt_select_entering(
            &[1.0, 1.0],
            &[0.1, 0.5],
            &no_basic(2),
            &bounds,
            2,
            PIVOT_TOL,
            5.0,
        );
        assert_eq!(bfrt_flip_invocations(), 2);
    }

    /// Stress: many breakpoints, mix of bounded and infinite. BFRT should
    /// reach a strictly larger θ than Harris would.
    #[test]
    fn bfrt_beats_harris_on_bounded_chain() {
        // 10 bounded columns at θ = 0.01, 0.02, ..., 0.10, each weight=1
        // 1 infinite column at θ = 1.0
        // residual = 5 → flip first 5 bounded, enter at the 6th bounded (θ=0.06)
        let mut trow = Vec::new();
        let mut r = Vec::new();
        let mut bounds = Vec::new();
        for k in 1..=10 {
            trow.push(1.0);
            r.push(0.01 * k as f64);
            bounds.push(ColBound { upper: 1.0, at_upper: false });
        }
        trow.push(1.0);
        r.push(1.0);
        bounds.push(ColBound { upper: f64::INFINITY, at_upper: false });

        let res = bfrt_select_entering(&trow, &r, &no_basic(11), &bounds, 11, PIVOT_TOL, 5.0).unwrap();
        // residual=5, walk: after 4 flips residual=1, at k=4 residual(1) ≤ weight(1)
        // → entering=j=4 (0-indexed, the 5th column), θ=0.05, flips=[0,1,2,3]
        assert_eq!(res.entering_col, 4);
        assert!((res.theta - 0.05).abs() < 1e-9);
        assert_eq!(res.flips.len(), 4);
        // Harris-equivalent θ would be 0.01 (the smallest breakpoint).
        assert!(res.theta > 0.01 * 4.0, "BFRT must beat Harris by ≥ 4x here");
    }
}
