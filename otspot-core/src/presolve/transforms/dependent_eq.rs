//! Step 12: removal of linearly dependent equality rows (rank-based).
//!
//! Analogue of HiGHS `removeDependentEquations` (`HPresolveDependent.cpp`):
//! when an Eq row of the *current presolve system* lies in the span of the
//! other surviving Eq rows and its RHS carries the same linear combination,
//! the row is redundant for the feasible set and is removed, shrinking the
//! row dimension the revised simplex LU-factors per pivot. Detection runs
//! incremental sparse Gauss elimination with threshold (max-abs) column
//! pivoting over the active Eq rows in ascending-nnz order; each independent
//! row is normalized to pivot = 1 and registered under its pivot column, and
//! a candidate that reduces to numerical zero is *dependent*.
//!
//! Two invariants keep the sweep correct and finite (violating them was
//! the pair of P1 soundness bugs fixed here): registration never truncates
//! witness coefficients — dropping a below-tolerance entry corrupts the
//! registered span, so a genuinely independent row reduces to zero and is
//! removed, flipping a bounded LP to Unbounded
//! (`step12_exact_witness_registration_solves_f3_repro_optimal_zero`) — and
//! elimination reduces against every pivot column whose coefficient is
//! nonzero at all, so supports stay pivot-free and re-elimination chains
//! move strictly forward (termination). The RHS consistency test is bounded
//! by a rounding-error model, not a fraction of the accumulated scale
//! (`rhs_rounding_bound`; `step12_rhs_rounding_model_rejects_inconsistent_1e12_rhs`).
//! Ambiguous rows fail open (kept) unless `removal_effect_bounded` proves
//! them harmless; dependency is proven on the *current* system and
//! transported back through the LIFO postsolve chain.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;

/// Relative threshold above which a working-row entry can serve as a pivot
/// (`|v| > DEP_EQ_RANK_REL_TOL * running row max`), i.e. the numerical-zero
/// cut of the *rank decision* (`|v| <= tol` counts as zero for
/// independence).
///
/// Matched to `PROP_TOL` (1e-9) in `transforms_dominance.rs`, the closest
/// sibling rule (parallel-row removal), and sits between `ZERO_TOL` (1e-12,
/// structural zero) and `PIVOT_TOL` (1e-8, the simplex acceptance tolerance):
/// a removal justified at 1e-9 relative perturbation stays invisible to the
/// downstream feasibility gates. Raising this constant makes removal *more*
/// aggressive, so the fail-open direction is protected by keeping it small.
/// This tolerance gates the rank decision only: elimination against an
/// existing pivot column is triggered by any nonzero (module doc invariant
/// 2), registration truncates nothing (invariant 1), and candidates whose
/// verdict exact arithmetic could reverse are fail-open via
/// `removal_effect_bounded`.
const DEP_EQ_RANK_REL_TOL: f64 = 1e-9;

/// Slack factor on the Higham gamma_K bound used by `rhs_rounding_bound`
/// (`|accumulated rounding| <= DEP_EQ_RHS_ERR_CHAIN_SLACK * K * f64::EPSILON
/// * scale`).
///
/// `b_acc` is a signed left-to-right sum of K+1 floating-point products
/// (`b - a_1*r_1 - ... - a_K*r_K`); the classic gamma_K model
/// (`K*eps / (1 - K*eps) * sum|terms|`) bounds its rounding by `K*eps` times
/// the accumulated product scale up to the `1 - K*eps` denominator, which
/// stays above 3/4 for every chain this sweep can build (K is at most the
/// number of Eq rows, and `K*eps <= 1e-9` needs ~1e16 rows to fail). A
/// factor of 4 dominates the model by one order of magnitude while still
/// landing ~2 orders *below* the residual of the smallest true
/// inconsistency the pass must catch on the same scale (on the F2 repro LP:
/// bound ~7e-3 vs a real inconsistency of 1.0).
const DEP_EQ_RHS_ERR_CHAIN_SLACK: f64 = 4.0;

/// Absolute floor added to the `rhs_rounding_bound` threshold.
///
/// A pure underflow cover: products `a_i * b_i` whose exact value is nonzero
/// can round to 0.0 once they drop below the f64 normal/subnormal range,
/// making the *tracked* accumulated scale miss that rounding step.
/// [`f64::MIN_POSITIVE`] is far above any such lost contribution yet far
/// below any representable data magnitude, so the floor never admits a real
/// inconsistency — it only keeps the bound nonzero when the whole chain is
/// at zero scale (homogeneous rows, where `b_acc` is exactly 0 anyway).
const DEP_EQ_RHS_ERR_ABS_FLOOR: f64 = f64::MIN_POSITIVE;

/// Relative threshold for the bound-aware residual-effect test (see
/// `removal_effect_bounded`): removal is admissible only when every
/// surviving residual coefficient `a_j` satisfies
/// `|a_j| * (ub_j - lb_j) <= DEP_EQ_BOUND_EFFECT_REL_TOL * (1 + coef_scale)`
/// with `coef_scale` the candidate's *coefficient* elimination scale
/// (running max |w| plus the sum of |elimination ratios|, an upper bound on
/// the `sum |alpha_i * a_i| products` the matrix reduction performed —
/// witness coefficients are at most 1 in magnitude because the pivot is the
/// max-abs entry).
///
/// The bound term upper-bounds the two-sided violation the removed row can
/// suffer at a reduced-system solution, so a term at this scale leaves the
/// removed row satisfied to within a 1e-9 relative perturbation of the row's
/// own elimination scale, which the postsolve acceptance then verifies
/// against the downstream 1e-5..1e-4 feasibility gates. The scale base
/// matters: the original form used the accumulated RHS scale, which is
/// unrelated to matrix magnitude — with `b` around 1e12 it inflated the
/// limit to ~4000 coefficient-units, the same
/// tolerance-vs-accumulated-witness-scale class as the F2 bug fixed in
/// `rhs_rounding_bound` (the module doc's sentinels pin both). Raising this
/// constant makes removal *more* aggressive; the fail-open direction is
/// protected by keeping it small.
const DEP_EQ_BOUND_EFFECT_REL_TOL: f64 = 1e-9;

/// Per-row fill budget for the elimination working row, as a multiple of the
/// candidate row's original nnz. Beyond it the rank verdict for that row is
/// abandoned (row kept). Mirrors the intent of the Markowitz guard in
/// `substitution.rs`: cap dense-cascade work so presolve cannot spend its
/// deadline budget on rank discovery with no guaranteed payoff.
const DEP_EQ_MAX_WORK_FACTOR: usize = 8;

/// Absolute cap on the elimination working-row nnz regardless of the
/// candidate row's original nnz (long rows must not launder an unbounded
/// per-elimination update budget through the factor above).
const DEP_EQ_ABS_WORK_CAP: usize = 4096;

/// Sentinel-visible fill guard: true when a working row of `touched_nnz`
/// entries (built from a candidate row with `orig_nnz` nonzeros) has
/// exhausted its elimination budget.
pub(super) fn work_over_budget(touched_nnz: usize, orig_nnz: usize) -> bool {
    touched_nnz > (DEP_EQ_MAX_WORK_FACTOR * orig_nnz).min(DEP_EQ_ABS_WORK_CAP)
}

/// Upper bound on `|b_acc|` explainable purely by accumulation rounding,
/// for a candidate that performed `rhs_ops` RHS updates over an accumulated
/// product scale `rhs_scale = |b_0| + sum |ratio_i * rhs_i|`.
///
/// This is the rounding-error model the RHS consistency test must use: a
/// truly dependent row cancels its RHS through the same elimination chain,
/// so what survives in `b_acc` is the rounding of summing `rhs_ops + 1`
/// signed products left-to-right in f64 — at most a few ulps of the sum of
/// their magnitudes (gamma_K bound, slack documented on
/// `DEP_EQ_RHS_ERR_CHAIN_SLACK`), plus the underflow floor
/// (`DEP_EQ_RHS_ERR_ABS_FLOOR`). A residual *above* this bound cannot be
/// rounding; it is a genuine RHS inconsistency (the simplex stage must see
/// the row to declare Infeasible) or numerically ambiguous rank — both keep
/// the row, the fail-open side of the feasibility-preservation contract.
fn rhs_rounding_bound(rhs_ops: usize, rhs_scale: f64) -> f64 {
    DEP_EQ_RHS_ERR_CHAIN_SLACK * (rhs_ops as f64) * f64::EPSILON * rhs_scale
        + DEP_EQ_RHS_ERR_ABS_FLOOR
}

/// One registered independent row, normalized to pivot = 1.
struct BasisRow {
    /// Support entries `(col, val)`; the pivot entry is `(pivot_col, 1.0)`.
    /// Two guarantees (module doc): every surviving nonzero coefficient is
    /// stored (exact span witness, never truncated), and no entry sits on a
    /// column that was already a pivot column at registration time (all such
    /// coefficients were eliminated, not dropped).
    entries: Vec<(usize, f64)>,
    /// The candidate's RHS carried through the same normalizations.
    rhs: f64,
}

/// Outcome of reducing one candidate Eq row against the registered basis.
enum Verdict {
    /// Matrix entries fully reduced to numerical zero (the pivot-column
    /// invariant means every pivot-column coefficient is gone and no other
    /// column exceeds the rank tolerance); carries the RHS residual
    /// magnitude, the accumulated RHS product scale, the number of RHS
    /// updates that scale was accumulated over, the coefficient-side
    /// elimination scale (see `removal_effect_bounded`), and the surviving
    /// sub-tolerance residual coefficients `(col, val)` (exactly-zero
    /// entries are omitted — they cannot contribute a bound effect).
    Dependent {
        rhs_residual: f64,
        rhs_scale: f64,
        rhs_ops: usize,
        coef_scale: f64,
        residual: Vec<(usize, f64)>,
    },
    /// Row is independent: normalized support entries and pivot column.
    /// `entries` holds *every* surviving nonzero coefficient divided by the
    /// pivot (correctly rounded, no truncation at any tolerance): dropping
    /// small coefficients here corrupts the registered span and lets a
    /// later, genuinely independent row reduce to zero against the mangled
    /// witness (`step12_exact_witness_registration_solves_f3_repro_optimal_zero`).
    Independent {
        pivot_col: usize,
        entries: Vec<(usize, f64)>,
        rhs: f64,
    },
    /// Elimination exhausted the fill budget; verdict unknown (keep row).
    OverBudget,
}

/// Dense working row with stamp-based lazy reset, reused across candidates.
struct ElimTracker {
    /// Dense accumulator, length = number of variables.
    w: Vec<f64>,
    /// Per-column membership stamp (generation counter trick).
    stamp: Vec<usize>,
    /// Columns currently in the working row (incl. numerically-zero ones).
    touched: Vec<usize>,
    generation: usize,
    /// Max |w| ever observed for the current candidate (tolerance scale).
    w_scale: f64,
    /// RHS carried through the elimination chain.
    b_acc: f64,
    /// Sum of |ratio| * |basis rhs| magnitudes (input scale of
    /// `rhs_rounding_bound`).
    b_scale: f64,
    /// Number of RHS updates performed for the current candidate
    /// (`rhs_ops` of the rounding model).
    elim_ops: usize,
    /// Sum of |elimination ratios|; with `w_scale` it upper-bounds the
    /// `sum |alpha_i * a_i|` scale of the matrix reduction (coefficients of
    /// registered rows are <= 1 in magnitude because the pivot is the max
    /// absolute entry).
    ratio_abs_sum: f64,
}

impl ElimTracker {
    fn new(n: usize) -> Self {
        ElimTracker {
            w: vec![0.0; n],
            stamp: vec![0; n],
            touched: Vec::new(),
            generation: 0,
            w_scale: 0.0,
            b_acc: 0.0,
            b_scale: 0.0,
            elim_ops: 0,
            ratio_abs_sum: 0.0,
        }
    }

    #[inline]
    fn touch(&mut self, col: usize) {
        if self.stamp[col] != self.generation {
            self.stamp[col] = self.generation;
            self.w[col] = 0.0;
            self.touched.push(col);
        }
    }

    /// Reduce `entries` (candidate row) against the registered basis rows.
    fn reduce(
        &mut self,
        entries: &[(usize, f64)],
        rhs: f64,
        pivot_at: &[usize],
        basis: &[BasisRow],
    ) -> Verdict {
        self.generation += 1;
        self.touched.clear();
        self.w_scale = 0.0;
        self.elim_ops = 0;
        self.ratio_abs_sum = 0.0;
        self.b_acc = rhs;
        self.b_scale = rhs.abs();
        for &(col, val) in entries {
            self.touch(col);
            self.w[col] = val;
            self.w_scale = self.w_scale.max(val.abs());
        }
        let budget_guard = entries.len();

        loop {
            let mut eliminated = false;
            let mut k = 0usize;
            while k < self.touched.len() {
                let col = self.touched[k];
                k += 1;
                let slot = pivot_at[col];
                // Any *nonzero* coefficient on a pivot column gets
                // eliminated (module doc invariant 2): below-tolerance
                // coefficients are numerically irrelevant to the rank but
                // must not sit on a registered witness's support, and
                // eliminating them is a harmless tiny-ratio update. The
                // Dependent/Independent distinction stays tol-gated in the
                // max-abs scan below.
                if slot == usize::MAX || self.w[col] == 0.0 {
                    continue;
                }
                eliminated = true;
                self.elim_ops += 1;
                let ratio = self.w[col];
                self.ratio_abs_sum += ratio.abs();
                let basis_row = &basis[slot];
                for &(bc, bv) in &basis_row.entries {
                    self.touch(bc);
                    let nv = self.w[bc] - ratio * bv;
                    self.w[bc] = nv;
                    self.w_scale = self.w_scale.max(nv.abs());
                }
                // Pivot entry is normalized to 1.0: the subtraction above
                // cancels w[col] up to rounding; pin it to exact zero.
                self.w[col] = 0.0;
                self.b_acc -= ratio * basis_row.rhs;
                self.b_scale += ratio.abs() * basis_row.rhs.abs();
                if work_over_budget(self.touched.len(), budget_guard) {
                    return Verdict::OverBudget;
                }
            }
            if !eliminated {
                break;
            }
        }

        let tol = DEP_EQ_RANK_REL_TOL * self.w_scale;
        // Termination: a registered support excludes the pivot columns that
        // existed at its insertion (module doc invariant 2), so a hit that
        // re-opens a pivot column always sits on a strictly later pivot than
        // the elimination that injected it, and the re-elimination chain
        // advances through the basis finitely. At loop exit no pivot column
        // carries a coefficient, so the max-abs scan can only pick a
        // non-pivot column.
        let mut pivot_col = usize::MAX;
        for &col in &self.touched {
            let v = self.w[col].abs();
            if v > tol && (pivot_col == usize::MAX || v > self.w[pivot_col].abs()) {
                pivot_col = col;
            }
        }
        if pivot_col == usize::MAX {
            let mut residual = Vec::new();
            for &col in &self.touched {
                if self.w[col] != 0.0 {
                    residual.push((col, self.w[col]));
                }
            }
            return Verdict::Dependent {
                rhs_residual: self.b_acc.abs(),
                rhs_scale: self.b_scale,
                rhs_ops: self.elim_ops,
                coef_scale: self.w_scale + self.ratio_abs_sum,
                residual,
            };
        }
        let pv = self.w[pivot_col];
        // Exact registration (module doc invariant 1): keep every surviving
        // nonzero coefficient, however far below the rank tolerance — the
        // registered row is the span witness later candidates are reduced
        // against, and truncating it corrupts that span. Only the correctly
        // rounded division by the pivot is applied; nothing is dropped, so
        // no witness ever needs an "approximate" marker.
        let mut norm = Vec::with_capacity(self.touched.len());
        for &col in &self.touched {
            let v = self.w[col];
            if v != 0.0 {
                norm.push((col, v / pv));
            }
        }
        Verdict::Independent {
            pivot_col,
            entries: norm,
            rhs: self.b_acc / pv,
        }
    }
}

/// Bound-aware admissibility test for a numerically-zero candidate row.
///
/// After elimination the candidate row is `a^T x = r` with every `|a_j|`
/// below the rank tolerance. Removing it lets a reduced-system solution
/// violate that row by `a^T x' - r`, which postsolve and the simplex
/// feasibility gates see two-sided; the objective direction is not known
/// here, so no one-sided (sign-exploiting) argument can cap the violation
/// from above or below. For a coefficient `a_j` on a variable with both
/// bounds finite, the variable's contribution to `|a^T x' - r|` is at most
/// `|a_j| * (ub_j - lb_j)` (the row is satisfied at some point of the box —
/// the original system's solution — so the deviation from that point is
/// bounded by the box's image width). If a bound is missing, that side of
/// the violation is unbounded: a small `|a_j|` is then not evidence of
/// harmlessness, because the variable can exploit it without limit (the
/// 1e-10-on-free-`w` case turns a bounded LP into an unbounded one). Hence
/// every nonzero residual coefficient must have a finite slack range with
/// the term below the tolerance, else the row is kept (fail-open). Uses
/// *current* bounds: earlier steps only tighten them soundly, so a tighter
/// range can only make admissible removals visible.
///
/// `limit` is relative to `coef_scale`, the candidate's coefficient-side
/// elimination scale: the residual `a_j` are coefficients and carry no
/// meaning against an RHS-derived scale (see
/// `DEP_EQ_BOUND_EFFECT_REL_TOL` for the class fix).
fn removal_effect_bounded(st: &PresolveState, residual: &[(usize, f64)], coef_scale: f64) -> bool {
    let limit = DEP_EQ_BOUND_EFFECT_REL_TOL * (1.0 + coef_scale);
    residual.iter().all(|&(col, a)| {
        let (lb, ub) = st.bounds[col];
        lb.is_finite() && ub.is_finite() && a.abs() * (ub - lb) <= limit
    })
}

pub(super) fn step12_dependent_equations(
    st: &mut PresolveState,
    deadline: Option<std::time::Instant>,
) -> Result<(), PresolveStatus> {
    // Only Eq rows are candidates: an inequality row that happens to lie in
    // the span of other rows may still bind through its slack, so it is never
    // removed here. Empty Eq rows belong to step 3a.
    let mut candidates: Vec<(usize, Vec<(usize, f64)>)> = (0..st.b.len())
        .filter(|&i| !st.removed_rows[i] && st.constraint_types[i] == ConstraintType::Eq)
        .map(|i| (i, st.active_row_entries(i)))
        .filter(|(_, entries)| !entries.is_empty())
        .collect();
    if candidates.len() < 2 {
        return Ok(());
    }
    // Ascending nnz gives cheap early pivots and cheaper later eliminations.
    candidates.sort_by_key(|(row, entries)| (entries.len(), *row));

    let n = st.bounds.len();
    let mut tracker = ElimTracker::new(n);
    let mut pivot_at: Vec<usize> = vec![usize::MAX; n];
    let mut basis: Vec<BasisRow> = Vec::new();

    for (row, entries) in candidates {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        match tracker.reduce(&entries, st.b[row], &pivot_at, &basis) {
            Verdict::Dependent {
                rhs_residual,
                rhs_scale,
                rhs_ops,
                coef_scale,
                residual,
            } => {
                if rhs_residual <= rhs_rounding_bound(rhs_ops, rhs_scale)
                    && removal_effect_bounded(st, &residual, coef_scale)
                {
                    st.removed_rows[row] = true;
                    st.postsolve_stack
                        .push(PostsolveStep::DependentEquation { orig_row: row });
                }
                // Residual above the rounding-error bound: the row is
                // genuinely inconsistent (or the rank is numerically
                // ambiguous) — keep it (fail-open); the simplex stage
                // reports Infeasible with its own tolerances. NOTE: both
                // P1 sentinels named in the module doc were reachable
                // end-to-end only because a reduced-LP
                // Infeasible/Unbounded verdict propagates without postsolve
                // acceptance (see the propagation site in
                // `simplex/entry.rs`); acceptance redesign is deliberately
                // out of scope, so this transform carries the guarantee.
            }
            Verdict::Independent {
                pivot_col,
                entries,
                rhs,
            } => {
                pivot_at[pivot_col] = basis.len();
                basis.push(BasisRow { entries, rhs });
            }
            // Keep the row and do not register it: a partially reduced row
            // could re-feed earlier pivot columns and break the sweep's
            // termination argument.
            Verdict::OverBudget => {}
        }
    }
    Ok(())
}
