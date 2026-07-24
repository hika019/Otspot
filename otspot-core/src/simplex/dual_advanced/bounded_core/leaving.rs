//! Leaving-variable selection (ratio tests) for bounded primal simplex.

use crate::sparse::CscMatrix;
use crate::tolerances::PIVOT_TOL;

/// Outcome of the bounded (two-sided) ratio test.
#[cfg_attr(test, derive(Debug))]
pub(super) enum BoundedLeave {
    /// Entering variable reaches its own opposite bound before any basic
    /// variable; flip it without a basis change (step = `ub_q`).
    Flip,
    /// Basic variable in `row` leaves at its lower (`at_ub = false`) or upper
    /// (`at_ub = true`) bound; `step` is the primal step length.
    Pivot { row: usize, at_ub: bool, step: f64 },
    /// No basic variable blocks the step and the entering bound is infinite.
    Unbounded,
}

/// Running best leaving candidate for the two-sided Harris ratio test: largest
/// pivot `|eff|`, ties (within `PIVOT_TOL`) broken by Bland's rule (smallest
/// basic index). Tracks the bound side and step so the chosen row carries them.
#[derive(Default)]
struct LeaveCand {
    row: Option<usize>,
    at_ub: bool,
    step: f64,
    best_pivot_abs: f64,
}

impl LeaveCand {
    fn relax(&mut self, i: usize, pivot_abs: f64, at_ub: bool, step: f64, basis: &[usize]) {
        if pivot_abs > self.best_pivot_abs + PIVOT_TOL {
            self.best_pivot_abs = pivot_abs;
            self.row = Some(i);
            self.at_ub = at_ub;
            self.step = step;
        } else if (pivot_abs - self.best_pivot_abs).abs() <= PIVOT_TOL {
            match self.row {
                None => {
                    self.row = Some(i);
                    self.at_ub = at_ub;
                    self.step = step;
                }
                Some(prev) if basis[i] < basis[prev] => {
                    self.row = Some(i);
                    self.at_ub = at_ub;
                    self.step = step;
                }
                _ => {}
            }
        }
    }
}

/// Two-sided Harris ratio test for the bounded primal cores.
///
/// Pass 1: feasibility-preserving step `θ = min_i (room_i + feas_tol) / |eff_i|`
/// (capped by `ub_q`). Pass 2: among rows with true ratio ≤ θ, pick the
/// largest pivot `|eff_i|` (Bland tie-break). Largest-pivot selection keeps
/// the basis well-conditioned under degeneracy.
///
/// Phase I artificial preference: when `art_threshold = Some(t)`, artificials
/// in the tie-band are preferred as the leaving variable (standard HiGHS/GLPK
/// Phase I rule — avoids stranding artificials on degenerate vertices).
pub(super) fn select_leaving_bounded(
    alpha: &[f64],
    dir: f64,
    x_b: &[f64],
    basis: &[usize],
    ubs: &[f64],
    ub_q: f64,
    m: usize,
    floor: f64,
    feas_tol: f64,
    art_threshold: Option<usize>,
) -> BoundedLeave {
    let mut theta = f64::INFINITY;
    let mut min_true = f64::INFINITY;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        if eff > floor {
            theta = theta.min((xi + feas_tol) / eff);
            min_true = min_true.min(xi / eff);
        } else if eff < -floor && ub_i.is_finite() {
            let neg = -eff;
            theta = theta.min((ub_i - xi + feas_tol) / neg);
            min_true = min_true.min((ub_i - xi) / neg);
        }
    }

    // Entering bound binds strictly first → flip (preserves "pivot on ties",
    // never flips past a degenerate blocking row whose true ratio is 0).
    if ub_q.is_finite() && ub_q < min_true {
        return BoundedLeave::Flip;
    }
    // Never step past the entering variable's own bound.
    if ub_q.is_finite() {
        theta = theta.min(ub_q);
    }
    if !theta.is_finite() {
        return BoundedLeave::Unbounded;
    }

    // Pass 2: among rows with true ratio ≤ θ, take the largest pivot (Bland
    // tie-break). `best_art` tracks the same over artificial rows only; when an
    // artificial sits in the tie-band it is preferred (Phase I, see above).
    let mut best = LeaveCand::default();
    let mut best_art = LeaveCand::default();
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        let (true_ratio, at_ub, pivot_abs) = if eff > floor {
            (xi / eff, false, eff)
        } else if eff < -floor && ub_i.is_finite() {
            ((ub_i - xi) / (-eff), true, -eff)
        } else {
            continue;
        };
        if true_ratio <= theta {
            let step = true_ratio.max(0.0);
            best.relax(i, pivot_abs, at_ub, step, basis);
            if art_threshold.is_some_and(|t| basis[i] >= t) {
                best_art.relax(i, pivot_abs, at_ub, step, basis);
            }
        }
    }

    let chosen = if best_art.row.is_some() {
        best_art
    } else {
        best
    };
    match chosen.row {
        Some(row) => BoundedLeave::Pivot {
            row,
            at_ub: chosen.at_ub,
            step: chosen.step,
        },
        None => BoundedLeave::Unbounded,
    }
}

/// Practical Bland leaving: minimum-ratio within a `PIVOT_TOL` tolerance band,
/// ties broken by smallest basic-variable index.
///
/// Used by `primal_simplex_aug` once a degenerate stall triggers anti-cycling.
/// Unlike `select_leaving_bounded` (largest-pivot Harris, chosen for LU
/// conditioning), this selects the smallest-basis-index row among those whose
/// ratio lies in `[min_ratio, min_ratio + PIVOT_TOL]`. Paired with Bland
/// entering (smallest improving column index) it breaks degenerate cycling in
/// practice.
///
/// The `PIVOT_TOL` band deviates from strict Bland: exact Bland finiteness
/// requires the strict minimum ratio; the band can admit additional candidates
/// beyond that strict minimum, so the theoretical finiteness guarantee is
/// weakened in proportion to `PIVOT_TOL`. Conditioning is sacrificed
/// deliberately; Bland mode is a transient escape, not the steady-state pricing.
pub(super) fn select_leaving_bland_bounded(
    alpha: &[f64],
    dir: f64,
    x_b: &[f64],
    basis: &[usize],
    ubs: &[f64],
    ub_q: f64,
    m: usize,
    floor: f64,
) -> BoundedLeave {
    let mut min_ratio = f64::INFINITY;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        if eff > floor {
            min_ratio = min_ratio.min(xi / eff);
        } else if eff < -floor && ub_i.is_finite() {
            min_ratio = min_ratio.min((ub_i - xi) / (-eff));
        }
    }

    if ub_q.is_finite() && ub_q < min_ratio {
        return BoundedLeave::Flip;
    }
    if ub_q.is_finite() {
        min_ratio = min_ratio.min(ub_q);
    }
    if !min_ratio.is_finite() {
        return BoundedLeave::Unbounded;
    }

    // Among rows achieving the minimum ratio (within PIVOT_TOL), Bland selects
    // the smallest basic-variable index — never the largest pivot.
    let mut leaving: Option<usize> = None;
    let mut leaving_at_ub = false;
    let mut chosen_step = 0.0f64;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        let (true_ratio, at_ub) = if eff > floor {
            (xi / eff, false)
        } else if eff < -floor && ub_i.is_finite() {
            ((ub_i - xi) / (-eff), true)
        } else {
            continue;
        };
        if true_ratio <= min_ratio + PIVOT_TOL {
            match leaving {
                None => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                    chosen_step = true_ratio.max(0.0);
                }
                Some(prev) if basis[i] < basis[prev] => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                    chosen_step = true_ratio.max(0.0);
                }
                _ => {}
            }
        }
    }

    match leaving {
        Some(row) => BoundedLeave::Pivot {
            row,
            at_ub: leaving_at_ub,
            step: chosen_step,
        },
        None => BoundedLeave::Unbounded,
    }
}

/// Bland entering for `primal_simplex_aug`: the smallest structural-column
/// index whose reduced cost is improving. Scanning from index 0 (rather than the
/// Devex / partial-pricing order) is what gives Bland its anti-cycling guarantee.
/// Reduced cost is recomputed directly from the current duals `y`; artificials
/// `[n_struct, n_aug)` are never priced.
pub(super) fn bland_entering(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    at_upper: &[bool],
    y: &[f64],
    n_struct: usize,
    floor: f64,
) -> Option<usize> {
    for j in 0..n_struct {
        if is_basic[j] {
            continue;
        }
        let (rows, vals) = a.column(j);
        let mut rc = c[j];
        for (k, &row) in rows.iter().enumerate() {
            rc -= vals[k] * y[row];
        }
        let violation = if at_upper[j] { rc } else { -rc };
        if violation > floor {
            return Some(j);
        }
    }
    None
}
