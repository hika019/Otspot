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
//! a candidate that reduces to numerical zero is *dependent*. The sweep loop
//! terminates because a registered row's support never contains columns that
//! were already pivot columns at its insertion time, so re-elimination chains
//! move strictly forward in insertion order.
//!
//! Dependency is proven on the *current* system (steps 1/2b/6-8 fold removed
//! columns into the remaining rows' `b` and coefficients), so it transports
//! back through the LIFO postsolve chain without referencing removed columns.
//! Every ambiguous outcome keeps the row (fail-open — feasibility preservation):
//! a dependent row whose surviving residual coefficients cannot be shown
//! harmless against the variables' finite slack ranges is not removed (see
//! `removal_effect_bounded`); a dependent row with inconsistent RHS is not
//! removed and not reported infeasible; a fill-aborted row is kept and not
//! registered in the basis. See the tolerance constants below.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;

/// Relative threshold below which a working-row entry is numerically zero
/// during rank elimination (`|v| <= DEP_EQ_RANK_REL_TOL * running row max`).
///
/// Matched to `PROP_TOL` (1e-9) in `transforms_dominance.rs`, the closest
/// sibling rule (parallel-row removal), and sits between `ZERO_TOL` (1e-12,
/// structural zero) and `PIVOT_TOL` (1e-8, the simplex acceptance tolerance):
/// a removal justified at 1e-9 relative perturbation stays invisible to the
/// downstream feasibility gates. Raising this constant makes removal *more*
/// aggressive, so the fail-open direction is protected by keeping it small.
const DEP_EQ_RANK_REL_TOL: f64 = 1e-9;

/// Relative threshold for the RHS consistency test on a detected-dependent
/// row (`|b_acc| <= DEP_EQ_RHS_REL_TOL * (1 + accumulated elimination scale)`).
///
/// A genuinely dependent row also cancels its RHS through the same
/// elimination chain, leaving only rounding of the same order as the matrix
/// residuals — hence the same magnitude as `DEP_EQ_RANK_REL_TOL`. A residual
/// above it means either true inconsistency (then the simplex stage must see
/// the row to declare Infeasible) or numerically ambiguous rank; both cases
/// keep the row, which is the fail-open side of the feasibility-preservation
/// contract.
const DEP_EQ_RHS_REL_TOL: f64 = 1e-9;

/// Relative threshold for the bound-aware residual-effect test (see
/// `removal_effect_bounded`): removal is admissible only when every
/// surviving residual coefficient `a_j` satisfies
/// `|a_j| * (ub_j - lb_j) <= DEP_EQ_BOUND_EFFECT_REL_TOL * (1 + rhs_scale)`.
///
/// Same magnitude and rationale as `DEP_EQ_RHS_REL_TOL`: the bound term
/// upper-bounds the two-sided violation the removed row can suffer at a
/// reduced-system solution, so a term at this scale leaves the removed row
/// satisfied to within the same slack the RHS test already tolerates, far
/// below the downstream 1e-5..1e-4 feasibility gates. Raising it makes
/// removal *more* aggressive; the fail-open direction is protected by
/// keeping it small.
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

/// One registered independent row, normalized to pivot = 1.
struct BasisRow {
    /// Support entries `(col, val)`; the pivot entry is `(pivot_col, 1.0)`.
    entries: Vec<(usize, f64)>,
    /// The candidate's RHS carried through the same normalizations.
    rhs: f64,
}

/// Outcome of reducing one candidate Eq row against the registered basis.
enum Verdict {
    /// Matrix entries fully reduced to numerical zero; carries the RHS
    /// residual magnitude, the accumulated RHS elimination scale, and the
    /// surviving residual coefficients `(col, val)` that were pinned to
    /// numerical zero (exactly-zero entries are omitted — they cannot
    /// contribute a bound effect).
    Dependent {
        rhs_residual: f64,
        rhs_scale: f64,
        residual: Vec<(usize, f64)>,
    },
    /// Row is independent: normalized support entries and pivot column.
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
    /// Sum of |ratio| * |basis rhs| magnitudes (RHS tolerance scale).
    b_scale: f64,
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
        self.b_acc = rhs;
        self.b_scale = rhs.abs();
        for &(col, val) in entries {
            self.touch(col);
            self.w[col] = val;
            self.w_scale = self.w_scale.max(val.abs());
        }
        let budget_guard = entries.len();

        loop {
            let tol = DEP_EQ_RANK_REL_TOL * self.w_scale;
            let mut eliminated = false;
            let mut k = 0usize;
            while k < self.touched.len() {
                let col = self.touched[k];
                k += 1;
                let slot = pivot_at[col];
                if slot == usize::MAX || self.w[col].abs() <= tol {
                    continue;
                }
                eliminated = true;
                let ratio = self.w[col];
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
                residual,
            };
        }
        let pv = self.w[pivot_col];
        let mut norm = Vec::with_capacity(self.touched.len());
        for &col in &self.touched {
            let v = self.w[col];
            if v.abs() > tol {
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
fn removal_effect_bounded(st: &PresolveState, residual: &[(usize, f64)], rhs_scale: f64) -> bool {
    let limit = DEP_EQ_BOUND_EFFECT_REL_TOL * (1.0 + rhs_scale);
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
                residual,
            } => {
                if rhs_residual <= DEP_EQ_RHS_REL_TOL * (1.0 + rhs_scale)
                    && removal_effect_bounded(st, &residual, rhs_scale)
                {
                    st.removed_rows[row] = true;
                    st.postsolve_stack
                        .push(PostsolveStep::DependentEquation { orig_row: row });
                }
                // Inconsistent RHS: keep the row (fail-open) — the simplex
                // stage reports Infeasible with its own tolerances.
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
