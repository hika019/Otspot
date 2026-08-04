//! Bound-Flipping Ratio Test (BFRT, Maros 2003 §3.7 / §7.6).
//!
//! 古典 Harris は最小 breakpoint で停止するが、BFRT は finite upper bound を持つ
//! 非基底変数を upper bound に flip することで dual step を次の breakpoint まで
//! 延長する。基底は変わらず (非基底値が 0 → u_k に切替) 累積 dual obj 改善を
//! 最大化する θ を選ぶ。bound-rich LP (pilot87, pds-*) で pivot 30-60% 削減。
//!
//! 符号規約は `HarrisRatioTest` (ratio_test.rs) と同じ: leaving row r で
//! `x_B[r] < 0`, `trow[j] = (B^{-1} a_j)[r]`, `θ_j = r_j / trow[j] ≥ 0`、
//! dual step は累積残差 `|x_B[r]|` − Σ flip 寄与 `u_k·trow[k]` で bounded。
//! `at_upper[j] = true` の変数は `−r_j / −trow[j]` で対称に参加する。

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

// BFRT flip probe counter — sentinel tests verify wiring is live.
// `Cell` + `thread_local!`: no atomic on hot path, isolated per `#[test]` thread.
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

/// 4-step BFRT (Maros 2003): (1) enumerate breakpoints `θ_j = r_j / α_j` for
/// compatible columns (at lower with `trow > pivot_tol`, at upper with
/// `trow < -pivot_tol`); (2) sort by θ ascending; (3) walk while tracking
/// residual `R = |x_B[r]| − Σ u_k|α_k|`, flipping each crossing and stopping
/// when R ≤ 0 (entering column = the one that brings R ≤ 0); (4) return
/// `(entering, θ, flips)`.
///
/// **Bound-feasibility invariant.** Step 3 stops at the first breakpoint whose
/// flip capacity `weight_k = u_k·|α_k|` covers the residual still to absorb.
/// That test *is* the entering column's own box constraint: its primal step is
/// `R / |α_k|`, so `R ≤ weight_k ⟺ step ≤ u_k`. Every column this function
/// returns as `entering_col` therefore satisfies `R / |α_entering| ≤
/// u_entering`, where `R = |leaving_residual| − Σ_{k ∈ flips} u_k·|α_k|` — the
/// one exception is the residual-not-absorbable fallback below, where *no*
/// column satisfies it.
///
/// Ties within `BFRT_TIE_TOL` of the chosen θ prefer largest |pivot| (Harris
/// pass 2), restricted to candidates that keep the invariant above. Returns
/// `None` if no compatible column (dual unbounded → primal infeasible);
/// returns Harris-equivalent θ with empty `flips` when no finite upper bound
/// exists (drop-in wrapper).
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
        if theta < -pivot_tol {
            continue;
        }
        let theta = theta.max(0.0);
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
    let mut residual = leaving_residual.abs();
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
        // is capped there by infeasibility detection in the caller). The last
        // breakpoint is not a flip, so hand back the capacity the walk just
        // charged against it: `residual` must stay "what the entering column
        // has to absorb" in both branches.
        entering_idx = breaks.len() - 1;
        residual += breaks[entering_idx].3;
    }

    // Step 4: tie-aware entering selection. Among breakpoints within
    // BFRT_TIE_TOL of the chosen θ, prefer the largest |pivot| (Harris pass 2)
    // — but only among candidates that can absorb `residual` themselves.
    //
    // Swapping the entering column leaves the flip set `0..entering_idx`
    // untouched, so the replacement inherits the same `residual` and must pass
    // the very test Step 3 used to stop: `residual ≤ weight_k`, i.e. its primal
    // step `residual / |α_k|` stays inside its own upper bound. Skipping this
    // check lets a tied column with a large pivot but a tiny upper bound enter
    // far past its own bound (the tie only equalises θ, never capacity).
    //
    // Candidates *before* `entering_idx` are excluded structurally rather than
    // by this test: un-flipping one to make it entering hands back the
    // `weight_k` it already absorbed, so it would have to absorb `residual +
    // weight_k ≤ weight_k`, i.e. `residual ≤ 0`. The walk only ever subtracts
    // when `residual > weight`, so a non-empty flip prefix leaves `residual >
    // 0` strictly — no consumed candidate is ever eligible.
    let chosen_theta = breaks[entering_idx].0;
    let mut best_idx = entering_idx;
    let mut best_pivot = breaks[entering_idx].2;
    for (k, &(theta, _j, abs_pivot, weight)) in breaks.iter().enumerate().skip(entering_idx + 1) {
        if (theta - chosen_theta).abs() > BFRT_TIE_TOL {
            break;
        }
        if residual <= weight && abs_pivot > best_pivot {
            best_pivot = abs_pivot;
            best_idx = k;
        }
    }
    // Flips that occurred during the residual walk (before entering_idx).
    // Candidates tied at the selected theta are not crossed by the dual step:
    // the algorithm stops on that breakpoint and only the chosen column enters.
    // Marking same-theta losers as flips changes the primal RHS without a
    // corresponding step past their breakpoint and violates A*x=b.
    let flips: Vec<usize> = (0..entering_idx).map(|k| breaks[k].1).collect();

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
            .map(|&u| ColBound {
                upper: u,
                at_upper: false,
            })
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
            ColBound {
                upper: 1.0,
                at_upper: false,
            },
            ColBound {
                upper: 1.0,
                at_upper: false,
            },
            ColBound {
                upper: f64::INFINITY,
                at_upper: false,
            },
        ];
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(3), &bounds, 3, PIVOT_TOL, 1.5).unwrap();
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
            ColBound {
                upper: 1.0,
                at_upper: false,
            },
            ColBound {
                upper: 2.0,
                at_upper: false,
            },
            ColBound {
                upper: 3.0,
                at_upper: false,
            },
            ColBound {
                upper: f64::INFINITY,
                at_upper: false,
            },
        ];
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(4), &bounds, 4, PIVOT_TOL, 10.0).unwrap();
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
            ColBound {
                upper: 1.0,
                at_upper: true,
            },
            ColBound {
                upper: f64::INFINITY,
                at_upper: false,
            },
        ];
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 0.5).unwrap();
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

    /// Independent oracle for the bound-feasibility invariant, recomputed from
    /// the *inputs and the returned result only* — it never reads
    /// `bfrt_select_entering`'s internals. The entering column has to absorb
    /// whatever the reported flips left over, and its primal step is that
    /// residual divided by its own pivot.
    fn implied_entering_step(
        trow: &[f64],
        bounds: &[ColBound],
        leaving_residual: f64,
        res: &BfrtResult,
    ) -> f64 {
        let absorbed: f64 = res
            .flips
            .iter()
            .map(|&k| bounds[k].upper * trow[k].abs())
            .sum();
        (leaving_residual.abs() - absorbed) / trow[res.entering_col].abs()
    }

    struct TieCase {
        name: &'static str,
        trow: Vec<f64>,
        r: Vec<f64>,
        bounds: Vec<ColBound>,
        leaving_residual: f64,
        expect_entering: usize,
        expect_flips: Vec<usize>,
        /// Hand-computed primal step of the expected entering column.
        expect_step: f64,
    }

    fn boxed(upper: f64, at_upper: bool) -> ColBound {
        ColBound { upper, at_upper }
    }

    /// SENTINEL + property oracle: the entering column returned by BFRT must
    /// never be pushed past its *own* upper bound.
    ///
    /// The Step 3 walk stops at the first breakpoint whose flip capacity
    /// `u_k·|α_k|` covers the remaining residual, which is exactly the
    /// statement `residual / |α_k| ≤ u_k`. The Step 4 tie-break then replaces
    /// that column with the largest-|pivot| column of the tie band — and a tie
    /// equalises only θ, never capacity, so an unfiltered swap can hand the
    /// residual to a column whose upper bound is orders of magnitude too small
    /// to hold it. Cases A/C below are that bug (present since the tie-break
    /// was introduced, independent of scan direction); B/D confirm the
    /// largest-pivot preference is *kept* whenever the swap is legitimate.
    ///
    /// All ratios are exact powers of two (1/16/64, 1/128) so `r_j / α_j` is
    /// bit-identical across the tie — decimal fractions that are only
    /// *mathematically* equal (0.3/3.0 vs 0.1/1.0) do not round to the same
    /// f64 and silently dissolve the tie the test means to build.
    ///
    /// No-op / revert-fail proof: dropping the `residual <= weight` guard from
    /// Step 4 makes cases A, C and F select the small-bound column instead
    /// (`entering_col` 1 instead of 0) and their `expect_step` assertions fail.
    #[test]
    fn bfrt_entering_never_exceeds_its_own_upper_bound() {
        // Case F's second breakpoint sits inside BFRT_TIE_TOL of the first
        // without being bit-equal to it (band membership, not exact equality).
        const NEAR_TIE: f64 = 4e-9;
        const {
            assert!(NEAR_TIE < BFRT_TIE_TOL, "case F must stay inside the band");
        }

        let cases = [
            TieCase {
                // θ = 0.25 for both; weights 64 and 1/128·16 = 0.125.
                // Walk: 32 ≤ 64 → entering = j0. j1 ties with |pivot| 16 > 1
                // but can only absorb 0.125 of the 32 residual.
                name: "A: forward tie, larger pivot lacks capacity",
                trow: vec![1.0, 16.0],
                r: vec![0.25, 4.0],
                bounds: vec![boxed(64.0, false), boxed(0.0078125, false)],
                leaving_residual: -32.0,
                expect_entering: 0,
                expect_flips: vec![],
                expect_step: 32.0,
            },
            TieCase {
                // Same tie, but j1's bound now covers the residual
                // (4·16 = 64 ≥ 32) → the largest-pivot preference applies.
                name: "B: forward tie, larger pivot has capacity",
                trow: vec![1.0, 16.0],
                r: vec![0.25, 4.0],
                bounds: vec![boxed(64.0, false), boxed(4.0, false)],
                leaving_residual: -32.0,
                expect_entering: 1,
                expect_flips: vec![],
                expect_step: 2.0,
            },
            TieCase {
                // at_upper tie candidate (negative pivot, non-positive reduced
                // cost): θ = (-r)/(-α) = 0.25, same capacity shortfall as A.
                name: "C: at_upper tie candidate lacks capacity",
                trow: vec![1.0, -16.0],
                r: vec![0.25, -4.0],
                bounds: vec![boxed(64.0, false), boxed(0.0078125, true)],
                leaving_residual: -32.0,
                expect_entering: 0,
                expect_flips: vec![],
                expect_step: 32.0,
            },
            TieCase {
                name: "D: at_upper tie candidate has capacity",
                trow: vec![1.0, -16.0],
                r: vec![0.25, -4.0],
                bounds: vec![boxed(64.0, false), boxed(4.0, true)],
                leaving_residual: -32.0,
                expect_entering: 1,
                expect_flips: vec![],
                expect_step: 2.0,
            },
            TieCase {
                // Three-way exact tie at θ = 0.2, weight 1 each. The walk
                // spends j0 and j1 as flips (2.5 → 1.5 → 0.5) and stops at j2.
                // j1 carries the largest |pivot| of the band but is already
                // consumed: making it entering would hand back its own weight,
                // forcing it to absorb 0.5 + 1 = 1.5 > 1 = weight. No consumed
                // candidate can ever pass that test, so the walk's choice stands.
                name: "E: consumed flip is never swapped in as entering",
                trow: vec![1.0, 8.0, 2.0],
                r: vec![0.2, 1.6, 0.4],
                bounds: vec![boxed(1.0, false), boxed(0.125, false), boxed(0.5, false)],
                leaving_residual: -2.5,
                expect_entering: 2,
                expect_flips: vec![0, 1],
                expect_step: 0.25,
            },
            TieCase {
                name: "F: near-tie inside BFRT_TIE_TOL, larger pivot lacks capacity",
                trow: vec![1.0, 16.0],
                r: vec![0.25, 16.0 * (0.25 + NEAR_TIE)],
                bounds: vec![boxed(64.0, false), boxed(0.0078125, false)],
                leaving_residual: -32.0,
                expect_entering: 0,
                expect_flips: vec![],
                expect_step: 32.0,
            },
        ];

        for case in &cases {
            let n = case.trow.len();
            let res = bfrt_select_entering(
                &case.trow,
                &case.r,
                &no_basic(n),
                &case.bounds,
                n,
                PIVOT_TOL,
                case.leaving_residual,
            )
            .unwrap_or_else(|| panic!("{}: expected a breakpoint", case.name));

            assert_eq!(
                res.entering_col, case.expect_entering,
                "{}: entering_col",
                case.name
            );
            assert_eq!(res.flips, case.expect_flips, "{}: flips", case.name);

            let step = implied_entering_step(&case.trow, &case.bounds, case.leaving_residual, &res);
            assert!(
                (step - case.expect_step).abs() < 1e-12,
                "{}: implied step {step} != hand-computed {}",
                case.name,
                case.expect_step
            );
            assert!(
                step <= case.bounds[res.entering_col].upper,
                "{}: entering column {} steps to {step}, past its own upper bound {}",
                case.name,
                res.entering_col,
                case.bounds[res.entering_col].upper,
            );
        }
    }

    /// The tie band's exactness premise for the E case above, asserted
    /// separately so a future edit that perturbs the literals cannot silently
    /// turn the three-way tie into three distinct breakpoints.
    #[test]
    fn bfrt_tie_fixture_ratios_are_bit_exact() {
        for (r, a) in [(0.25, 1.0), (4.0, 16.0)] {
            assert_eq!(r / a, 0.25, "power-of-two ratio must be bit-exact");
        }
        for (r, a) in [(0.2, 1.0), (1.6, 8.0), (0.4, 2.0)] {
            assert_eq!(r / a, 0.2, "power-of-two ratio must be bit-exact");
        }
        for (u, a) in [(1.0, 1.0), (0.125, 8.0), (0.5, 2.0)] {
            assert_eq!(u * a, 1.0, "power-of-two weight must be bit-exact");
        }
        assert_eq!(0.0078125_f64 * 16.0, 0.125);
        assert_eq!(4.0_f64 * 16.0, 64.0);
    }

    /// Documented exception to the bound-feasibility invariant: when the
    /// combined flip capacity of *every* compatible column still cannot cover
    /// the leaving violation, no column can absorb the residual and BFRT falls
    /// back to the last breakpoint (Maros; the caller's infeasibility
    /// detection caps the step). Pinned here so the Step 4 capacity guard is
    /// not silently widened into "return None" — that maps straight to
    /// `SolveStatus::Infeasible` in `finish_bounded`.
    #[test]
    fn bfrt_falls_back_to_last_breakpoint_when_capacity_is_exhausted() {
        // weights 1 + 1 = 2 < residual 10 → the walk never stops.
        let trow = vec![1.0, 1.0];
        let r = vec![0.25, 0.5];
        let bounds = lb_bounds(&[1.0, 1.0]);
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, -10.0).unwrap();
        assert_eq!(res.entering_col, 1, "fallback = last breakpoint");
        assert_eq!(res.flips, vec![0]);
        let step = implied_entering_step(&trow, &bounds, -10.0, &res);
        assert!(
            step > bounds[res.entering_col].upper,
            "this fixture is the exhausted-capacity case by construction: \
             step {step} must exceed upper {}",
            bounds[res.entering_col].upper
        );
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
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
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

    #[test]
    fn bfrt_tie_loser_at_selected_breakpoint_is_not_flip() {
        let trow = vec![1.0, 5.0];
        let r = vec![0.1, 0.5];
        let bounds = lb_bounds(&[1.0, 1.0]);
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 0.5).unwrap();
        assert_eq!(res.entering_col, 1);
        assert!(
            res.flips.is_empty(),
            "same-theta losers are not crossed and must not flip"
        );
    }

    #[test]
    fn bfrt_skips_negative_breakpoints() {
        let trow = vec![1.0, 2.0];
        let r = vec![-1.0, 1.0];
        let bounds = lb_bounds(&[f64::INFINITY, f64::INFINITY]);
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 1.0).unwrap();
        assert_eq!(res.entering_col, 1);
        assert!((res.theta - 0.5).abs() < 1e-12);
    }

    /// Reviewer P1 regression: when *all* tie-zone candidates have infinite
    /// upper, the loser cannot be marked as a flip. Minimal reproduction —
    /// independent of the tie-breaker outcome.
    #[test]
    fn bfrt_tie_excludes_infinite_upper() {
        let trow = vec![1.0, 5.0];
        let r = vec![0.1, 0.5];
        let bounds = lb_bounds(&[f64::INFINITY, f64::INFINITY]);
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
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
            ColBound {
                upper: 1.0,
                at_upper: false,
            },
            ColBound {
                upper: f64::INFINITY,
                at_upper: false,
            },
        ];
        let res =
            bfrt_select_entering(&trow, &r, &no_basic(2), &bounds, 2, PIVOT_TOL, 10.0).unwrap();
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
            ColBound {
                upper: 1.0,
                at_upper: false,
            },
            ColBound {
                upper: f64::INFINITY,
                at_upper: false,
            },
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
            bounds.push(ColBound {
                upper: 1.0,
                at_upper: false,
            });
        }
        trow.push(1.0);
        r.push(1.0);
        bounds.push(ColBound {
            upper: f64::INFINITY,
            at_upper: false,
        });

        let res =
            bfrt_select_entering(&trow, &r, &no_basic(11), &bounds, 11, PIVOT_TOL, 5.0).unwrap();
        // residual=5, walk: after 4 flips residual=1, at k=4 residual(1) ≤ weight(1)
        // → entering=j=4 (0-indexed, the 5th column), θ=0.05, flips=[0,1,2,3]
        assert_eq!(res.entering_col, 4);
        assert!((res.theta - 0.05).abs() < 1e-9);
        assert_eq!(res.flips.len(), 4);
        // Harris-equivalent θ would be 0.01 (the smallest breakpoint).
        assert!(res.theta > 0.01 * 4.0, "BFRT must beat Harris by ≥ 4x here");
    }
}
