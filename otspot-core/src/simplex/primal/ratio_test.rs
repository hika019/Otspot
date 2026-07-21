//! Harris ratio test helpers for the feasibility-preserving leaving-row selection.

use crate::tolerances::PIVOT_TOL;

/// Maximum step that keeps every basic variable ≥ −`tol`:
///   `θ = min_{i: d[i]>floor} (x_b[i] + tol) / d[i]`.
/// `INFINITY` when no row is eligible (unbounded direction).
pub(super) fn bound_tolerance_step(x_b: &[f64], d: &[f64], m: usize, floor: f64, tol: f64) -> f64 {
    let mut theta = f64::INFINITY;
    for i in 0..m {
        if d[i] > floor {
            let t = (x_b[i] + tol) / d[i];
            if t < theta {
                theta = t;
            }
        }
    }
    theta
}

/// Update the running best leaving candidate: largest pivot `|d[i]|`, ties in
/// `|d[i]|` (within `PIVOT_TOL`) broken by Bland's rule (smallest basic index).
fn relax_best(
    leaving: &mut Option<usize>,
    best_pivot_abs: &mut f64,
    i: usize,
    d_abs: f64,
    basis: &[usize],
) {
    if d_abs > *best_pivot_abs + PIVOT_TOL {
        *best_pivot_abs = d_abs;
        *leaving = Some(i);
    } else if (d_abs - *best_pivot_abs).abs() <= PIVOT_TOL {
        match *leaving {
            None => *leaving = Some(i),
            Some(prev) if basis[i] < basis[prev] => *leaving = Some(i),
            _ => {}
        }
    }
}

/// Pick the leaving row with the largest pivot `|d[i]|` among rows whose ratio
/// `x_b[i]/d[i]` does not exceed `theta` (the Harris tie-band); ties in `|d[i]|`
/// break by Bland's rule (smallest basic index, anti-cycling). Returns the row,
/// or `None`.
///
/// Phase I artificial preference: when `art_threshold = Some(t)` and the tie-band
/// contains an artificial basic variable (`basis[i] >= t`), the leaving row is
/// chosen among the artificial rows only. Driving artificials out first is the
/// standard HiGHS/GLPK Phase I min-ratio preference: it shrinks Σartificial
/// faster on degenerate vertices where structural rows otherwise take the tie and
/// leave artificials stranded at near-zero (tiny-step) pivots. `theta` is
/// unchanged, so the pivot stays feasibility-preserving (every tie-band row keeps
/// basics `≥ −feas_tol`); only the choice among equally-eligible rows differs.
pub(super) fn max_pivot_within(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    theta: f64,
    art_threshold: Option<usize>,
) -> Option<usize> {
    let mut leaving_any: Option<usize> = None;
    let mut best_any = 0.0f64;
    let mut leaving_art: Option<usize> = None;
    let mut best_art = 0.0f64;
    for i in 0..m {
        if d[i] > floor {
            let ratio = x_b[i] / d[i];
            if ratio <= theta {
                let d_abs = d[i].abs();
                relax_best(&mut leaving_any, &mut best_any, i, d_abs, basis);
                if art_threshold.is_some_and(|t| basis[i] >= t) {
                    relax_best(&mut leaving_art, &mut best_art, i, d_abs, basis);
                }
            }
        }
    }
    // Prefer an artificial leaving row when Phase I supplies a threshold and at
    // least one artificial sits in the tie-band; otherwise the largest-pivot row.
    if leaving_art.is_some() {
        leaving_art
    } else {
        leaving_any
    }
}

/// Harris ratio test (Pass 2), **feasibility-preserving**.
///
/// Computes `θ = min_{i: d[i]>floor} (x_b[i] + feas_tol) / d[i]` and picks
/// the largest-pivot row within `θ` (Bland tie-break). Keeps all pivot-eligible
/// basics at `≥ −feas_tol`; a row in `[−feas_tol, 0)` can give a small negative
/// step, and the optimality backstop returns Timeout rather than false-Optimal.
///
/// `feas_tol` = `options.primal_tol`. Returns `None` for an unbounded direction.
/// `art_threshold = Some(t)` enables the Phase I artificial leaving preference
/// (see `max_pivot_within`); `None` leaves the largest-pivot rule unchanged.
pub(super) fn select_leaving_feasibility_preserving(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    feas_tol: f64,
    art_threshold: Option<usize>,
) -> Option<usize> {
    let theta = bound_tolerance_step(x_b, d, m, floor, feas_tol);
    if !theta.is_finite() {
        return None;
    }
    max_pivot_within(x_b, d, basis, m, floor, theta, art_threshold)
}

#[cfg(test)]
mod ratio_test_feasibility_tests {
    //! Sentinels for the feasibility-preserving Harris ratio test
    //! (`select_leaving_feasibility_preserving`).
    //!
    //! The leaving row must be chosen so the pivot step keeps every basic value
    //! ≥ −feas_tol, with the violation bounded by `feas_tol` independent of the
    //! pivot magnitude `d[i]`. The previous absolute-ratio window
    //! `min_ratio + PIVOT_TOL` let a binding row overshoot by `PIVOT_TOL·d[i]`,
    //! which for large `d[i]` (ill-scaled columns) exceeded any clamp and
    //! cascaded into primal infeasibility (pilot87 Phase II).

    use super::select_leaving_feasibility_preserving;
    use crate::tolerances::PIVOT_TOL;

    /// Apply the pivot for a chosen leaving row and return the minimum basic
    /// value afterwards (the feasibility witness).
    fn min_basic_after_pivot(x_b: &[f64], d: &[f64], leaving: usize) -> f64 {
        let step = x_b[leaving] / d[leaving];
        let mut min_v = f64::INFINITY;
        for i in 0..x_b.len() {
            let v = if i == leaving {
                step
            } else {
                x_b[i] - d[i] * step
            };
            if v < min_v {
                min_v = v;
            }
        }
        min_v
    }

    /// Reference implementation of the OLD absolute-ratio window
    /// (`min_ratio + PIVOT_TOL`, max |d|, Bland tie-break). Used only to prove
    /// the no-op: reverting the production helper to this rule reintroduces the
    /// feasibility breach this sentinel guards against.
    fn old_absolute_window_leaving(x_b: &[f64], d: &[f64], basis: &[usize], floor: f64) -> usize {
        let m = x_b.len();
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if d[i] > floor {
                min_ratio = min_ratio.min(x_b[i] / d[i]);
            }
        }
        let window = min_ratio + PIVOT_TOL;
        let mut leaving = None;
        let mut best = 0.0f64;
        for i in 0..m {
            if d[i] > floor && x_b[i] / d[i] <= window {
                let da = d[i].abs();
                if da > best + PIVOT_TOL {
                    best = da;
                    leaving = Some(i);
                } else if (da - best).abs() <= PIVOT_TOL {
                    match leaving {
                        None => leaving = Some(i),
                        Some(p) if basis[i] < basis[p] => leaving = Some(i),
                        _ => {}
                    }
                }
            }
        }
        leaving.unwrap()
    }

    /// Sentinel (no-op proof): an ill-scaled tie where the absolute-ratio window
    /// breaches feasibility but the bound-tolerance helper does not.
    ///
    /// Two rows share a huge pivot (|d|=1e6). Row 0 has the true min ratio
    /// (x_b=1e-9) and row 1 is far from it (x_b=1e-3). The absolute window
    /// `min_ratio+PIVOT_TOL` admits BOTH and, on the |d| tie, Bland picks row 1
    /// (lower basis index). Its step 1e-9 then drives row 0 to ≈ −1e-3 ≪ −tol.
    /// The helper's bound-tolerance step admits only row 0, so no basic value
    /// drops below −feas_tol.
    ///
    /// Reverting the helper to the absolute window makes it return row 1 →
    /// `assert_eq!(leaving, 0)` and the feasibility assertion both FAIL.
    #[test]
    fn bound_tolerance_blocks_ill_scaled_overshoot() {
        let x_b = [1e-9, 1e-3];
        let d = [1e6, 1e6];
        let basis = [5usize, 3usize];
        let feas_tol = PIVOT_TOL;
        let floor = PIVOT_TOL;

        let leaving = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            floor,
            feas_tol,
            None,
        )
        .expect("eligible leaving row exists");
        assert_eq!(
            leaving, 0,
            "helper must pick the true-min-ratio row 0, not the far row 1"
        );
        let min_basic = min_basic_after_pivot(&x_b, &d, leaving);
        assert!(
            min_basic >= -feas_tol,
            "helper pivot must keep basics ≥ −feas_tol; got {min_basic}"
        );

        // No-op proof: the old absolute-ratio window picks row 1 and breaches.
        let old_leaving = old_absolute_window_leaving(&x_b, &d, &basis, floor);
        assert_eq!(old_leaving, 1, "old window picks the far row (Bland tie)");
        let old_min_basic = min_basic_after_pivot(&x_b, &d, old_leaving);
        assert!(
            old_min_basic < -feas_tol,
            "old window must breach feasibility (proves the sentinel bites); got {old_min_basic}"
        );
        assert!(
            old_min_basic < -1e-4,
            "breach magnitude ∝ d[i]; expected ≈ −1e-3, got {old_min_basic}"
        );
    }

    /// Stability is preserved: when several rows leave safely within the
    /// bound-tolerance window, the helper still selects the largest pivot.
    #[test]
    fn picks_largest_pivot_within_window() {
        // Both rows are at the degenerate vertex (x_b ≈ 0), so both are within
        // θ. Row 1 has the larger pivot and must be chosen for stability.
        let x_b = [0.0, 0.0];
        let d = [0.5, 2.0];
        let basis = [7usize, 4usize];
        let leaving = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            None,
        )
        .expect("eligible leaving row exists");
        assert_eq!(leaving, 1, "must pick the larger pivot |d|=2.0 (row 1)");
    }

    /// Phase I artificial preference (ON vs OFF): when the tie-band holds both a
    /// structural and an artificial row, `Some(threshold)` drives the artificial
    /// out first, while `None` keeps the largest-pivot (structural) choice. θ —
    /// hence the step bound and feasibility — is identical for both: the
    /// preference only reorders equally eligible rows, never the min ratio.
    #[test]
    fn phase1_prefers_artificial_within_tie_band() {
        // Both rows at the degenerate vertex (x_b = 0) ⇒ both inside θ.
        // Row 0 structural (basis 3), larger pivot |d| = 2.0.
        // Row 1 artificial (basis 10 ≥ threshold 10), smaller pivot |d| = 0.5.
        let x_b = [0.0, 0.0];
        let d = [2.0, 0.5];
        let basis = [3usize, 10usize];
        let threshold = 10usize;

        let off = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            None,
        )
        .expect("eligible leaving row exists");
        assert_eq!(off, 0, "OFF: largest pivot wins (structural row 0)");

        let on = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            Some(threshold),
        )
        .expect("eligible leaving row exists");
        assert_eq!(on, 1, "ON: artificial row 1 leaves first");

        // Feasibility preserved for either choice (θ, hence step bound, unchanged).
        assert!(min_basic_after_pivot(&x_b, &d, off) >= -PIVOT_TOL);
        assert!(min_basic_after_pivot(&x_b, &d, on) >= -PIVOT_TOL);
    }

    /// No-op proof: an artificial row OUTSIDE the tie-band (ratio > θ) is never
    /// preferred. The preference only reorders within θ and never changes the min
    /// ratio, so `Some(threshold)` matches `None` when no artificial is in-band.
    #[test]
    fn phase1_artificial_outside_tie_band_is_noop() {
        // Row 0 structural, true min ratio 0. Row 1 artificial but at ratio 1 ≫ θ.
        let x_b = [0.0, 1.0];
        let d = [1.0, 1.0];
        let basis = [3usize, 10usize];
        let threshold = 10usize;

        let off = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            None,
        );
        let on = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            Some(threshold),
        );
        assert_eq!(on, Some(0), "out-of-band artificial must NOT be selected");
        assert_eq!(on, off, "preference is a no-op outside the tie-band");
    }

    /// No eligible row (all directions ≤ floor) ⇒ unbounded ⇒ None.
    #[test]
    fn no_eligible_row_is_unbounded() {
        let x_b = [3.0, 4.0];
        let d = [-1.0, 0.0];
        let basis = [0usize, 1usize];
        let leaving = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
            None,
        );
        assert!(
            leaving.is_none(),
            "no positive direction ⇒ unbounded ⇒ None"
        );
    }
}
