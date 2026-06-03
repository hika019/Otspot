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

/// Pick the leaving row with the largest pivot `|d[i]|` among rows whose ratio
/// `x_b[i]/d[i]` does not exceed `theta`; ties in `|d[i]|` break by Bland's rule
/// (smallest basic index, anti-cycling). Returns the row, or `None`.
pub(super) fn max_pivot_within(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    theta: f64,
) -> Option<usize> {
    let mut leaving: Option<usize> = None;
    let mut best_pivot_abs = 0.0f64;
    for i in 0..m {
        if d[i] > floor {
            let ratio = x_b[i] / d[i];
            if ratio <= theta {
                let d_abs = d[i].abs();
                if d_abs > best_pivot_abs + PIVOT_TOL {
                    best_pivot_abs = d_abs;
                    leaving = Some(i);
                } else if (d_abs - best_pivot_abs).abs() <= PIVOT_TOL {
                    match leaving {
                        None => leaving = Some(i),
                        Some(prev) if basis[i] < basis[prev] => leaving = Some(i),
                        _ => {}
                    }
                }
            }
        }
    }
    leaving
}

/// Harris ratio test (Pass 2), **feasibility-preserving**.
///
/// The leaving step is bounded by the variable-tolerance maximum step
///   `θ = min_{i: d[i]>floor} (x_b[i] + feas_tol) / d[i]`,
/// and among rows within `θ` we take the largest pivot `|d[i]|` (Bland
/// tie-break). For a leaving row with `x_b ≥ 0` this keeps every pivot-eligible
/// basic value (`d[i] > floor`) at `≥ −feas_tol` independent of `d[i]`. A
/// leaving row inside the `[−feas_tol, 0)` band gives a small negative step that
/// can transiently breach `−feas_tol`; the optimality backstop (exact
/// `x_b = B⁻¹b` recheck) then returns an honest Timeout, never false-Optimal.
///
/// The predecessor's absolute *ratio* window `min_ratio + ε` overshot by
/// `ε·d[i]` — unbounded for ill-scaled columns (pilot87: `d[i] ≈ 1.3e6` turned
/// `ε = 1e-8` into a 0.013 breach), producing an `x_b < 0` basis, negative
/// ratios, and a wandering objective instead of convergence.
///
/// `feas_tol` = `options.primal_tol`. Returns `None` for an unbounded direction.
pub(super) fn select_leaving_feasibility_preserving(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    feas_tol: f64,
) -> Option<usize> {
    let theta = bound_tolerance_step(x_b, d, m, floor, feas_tol);
    if !theta.is_finite() {
        return None;
    }
    max_pivot_within(x_b, d, basis, m, floor, theta)
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
            let v = if i == leaving { step } else { x_b[i] - d[i] * step };
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

        let leaving =
            select_leaving_feasibility_preserving(&x_b, &d, &basis, x_b.len(), floor, feas_tol)
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
        let leaving =
            select_leaving_feasibility_preserving(&x_b, &d, &basis, x_b.len(), PIVOT_TOL, PIVOT_TOL)
                .expect("eligible leaving row exists");
        assert_eq!(leaving, 1, "must pick the larger pivot |d|=2.0 (row 1)");
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
        );
        assert!(leaving.is_none(), "no positive direction ⇒ unbounded ⇒ None");
    }
}
