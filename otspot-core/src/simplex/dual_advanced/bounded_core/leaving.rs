//! Leaving-variable selection (ratio tests) for bounded primal simplex.

use crate::tolerances::PIVOT_TOL;
use otspot_num::sparse::CscMatrix;

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

/// Relative tolerance for the Bland leaving-rule tie test.
///
/// Textbook Bland's-rule finite-termination proofs require selecting, among
/// rows achieving the *exact* minimum ratio, the smallest basic-variable
/// index — and the entering rule (`bland_entering`) to pick the exact
/// smallest-index column with any nonzero improving reduced cost. Neither
/// precondition holds exactly here: `bland_entering` accepts a column once
/// its violation exceeds `floor` (`PIVOT_TOL`, not zero), and this leaving
/// rule admits a *band* (`min_ratio + tie_band`, not exact equality) so two
/// independently-perturbed rows that are mathematically tied but not
/// bit-identical (LU roundoff) both count. So this tightened band is **not**
/// a restored formal finite-termination guarantee — it is a large reduction
/// in the false-tie window relative to the former `PIVOT_TOL`-sized absolute
/// band, empirically confirmed to eliminate the specific cycle observed on
/// pk1 (MIPLIB): one B&B node LP revisited 592 distinct bases for
/// ~5,000,000 consecutive 100%-degenerate pivots in bland mode before this
/// fix. The actual backstop against any *residual* cycle this band doesn't
/// prevent is [`super::primal::obj_plateau_should_bail`]'s objective-plateau bail,
/// which needs no such proof — it only needs to detect non-improvement.
///
/// `1e-9` relative is ~4 orders of magnitude above `f64::EPSILON` (headroom
/// for the roundoff case above) and ~4 orders of magnitude *below*
/// `PIVOT_TOL` (1e-8 absolute — sized for rejecting near-zero pivot
/// *elements*, a different quantity, and far too loose for a ratio-tie test).
const BLAND_TIE_REL_TOL: f64 = 1e-9;

/// Practical Bland leaving: minimum-ratio within [`BLAND_TIE_REL_TOL`] of the
/// exact minimum, ties broken by smallest basic-variable index.
///
/// Used by `primal_simplex_aug` once a degenerate stall triggers anti-cycling.
/// Unlike `select_leaving_bounded` (largest-pivot Harris, chosen for LU
/// conditioning), this selects the smallest-basis-index row among those tied
/// for `min_ratio`. Paired with Bland entering (smallest improving column
/// index) this substantially narrows the cycling window Bland's rule is
/// meant to close — see [`BLAND_TIE_REL_TOL`] for why it is *not* a restored
/// formal guarantee, and [`super::primal::obj_plateau_should_bail`] for the backstop
/// that does not depend on one.
///
/// The step actually taken is always the exact `min_ratio`, never the chosen
/// row's own `true_ratio`: within the (tiny) tie band a non-selected row's
/// ratio can differ from `min_ratio` by up to the tolerance, and stepping to
/// a tied-but-not-minimal ratio would leave the true minimizer's row slightly
/// primal-infeasible.
///
/// **Invariant / error bound (Codex review, P1, documented not fixed — see
/// below for why).** When the *chosen* row is itself not the exact
/// minimizer (`true_ratio > min_ratio`, only admitted because it is within
/// `tie_band` of it — see [`BLAND_TIE_REL_TOL`]), stepping by `min_ratio`
/// leaves that row's variable short of the bound it is declared nonbasic at
/// (`at_ub`) by exactly `|eff_i| * (true_ratio - min_ratio) <= |eff_i| *
/// tie_band`: the row ends up strictly *inside* the box, never past it (a
/// one-line derivation: for the `at_ub` branch, `x_new = x_i - eff_i *
/// min_ratio = ub_i + eff_i * (true_ratio - min_ratio)`, and `eff_i < 0`
/// there, so `x_new <= ub_i`; the lower-bound branch is symmetric with the
/// bound at `0`).
///
/// This is the safe side of a tradeoff intrinsic to a *banded* Bland
/// tie-break: the alternative (stepping by the chosen row's own
/// `true_ratio` instead of `min_ratio`) would instead push the true
/// minimizer's row (a *different* row, whose exact ratio is `min_ratio`)
/// *past* its own bound by the same gap — a genuine primal infeasibility
/// introduced by this ratio test itself, which is strictly worse than a
/// nonbasic variable that merely has not yet reached the bound it is
/// labelled at. Neither choice is exact once the tie band admits more than
/// one row; this one cannot manufacture infeasibility.
///
/// The residual is transient, not accumulated pivot over pivot: each
/// LU-rebuild checkpoint recomputes the full `x_b` from `basis` and
/// `at_upper` directly (`reconcile_bounded_terminal_state` in
/// `dual_advanced::mod`), independent of the incremental step chain, and its
/// `BoundedTerminalReconcile::BoundViolation` outcome is the backstop that
/// catches an actual (non-transient) excursion beyond `options.primal_tol`
/// — which this bounded-short residual, by construction, is not.
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
    // No `min_ratio = min_ratio.min(ub_q)` here (P3-1): reaching this point
    // means NOT(ub_q.is_finite() && ub_q < min_ratio), i.e. either `ub_q` is
    // infinite (the `.min` would be skipped anyway) or `ub_q >= min_ratio`
    // (the `.min` would be a no-op) — the entering variable's own bound can
    // never tighten `min_ratio` once the Flip check above has run.
    if !min_ratio.is_finite() {
        return BoundedLeave::Unbounded;
    }

    // Among rows achieving the exact minimum ratio (within BLAND_TIE_REL_TOL,
    // absorbing floating-point noise only), Bland selects the smallest
    // basic-variable index — never the largest pivot.
    let tie_band = min_ratio.abs().max(1.0) * BLAND_TIE_REL_TOL;
    let mut leaving: Option<usize> = None;
    let mut leaving_at_ub = false;
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
        if true_ratio <= min_ratio + tie_band {
            match leaving {
                None => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                }
                Some(prev) if basis[i] < basis[prev] => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                }
                _ => {}
            }
        }
    }

    match leaving {
        Some(row) => BoundedLeave::Pivot {
            row,
            at_ub: leaving_at_ub,
            // `step` is always `min_ratio`, not `row`'s own `true_ratio` when
            // `row` was picked from within the tie band — see this function's
            // doc ("Invariant / error bound") for the resulting bounded,
            // never-past-the-bound short-of-bound residual.
            step: min_ratio.max(0.0),
        },
        // P3-1: unreachable, not a defensive fallback. `min_ratio` is finite
        // here (checked above) and was computed as the minimum of the exact
        // same per-row formula this second pass recomputes over the exact
        // same (alpha/x_b/basis/ubs are unchanged between passes) inputs, so
        // the row that achieved it in pass 1 satisfies `true_ratio <=
        // min_ratio + tie_band` bit-for-bit in pass 2 — `leaving` is always
        // `Some`. A future change that could make this fire would be a
        // correctness bug, not a legitimate Unbounded case; panicking makes
        // that loud instead of silently reporting an unbounded ray.
        None => unreachable!(
            "min_ratio is finite, so its achieving row must be found again \
             in this identical second pass"
        ),
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
