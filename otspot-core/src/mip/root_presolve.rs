//! Root-only structural presolve for MILP: MIP-safe reductions applied **once**
//! at the root, with the whole branch-and-bound run on the reduced matrix and
//! only the final solution mapped back (SCIP-style presolve/B&B integration).

//! # Why this exists as a separate path from the LP-layer presolve
//! The per-node LP path deliberately runs *without* the LP-layer structural
//! presolve (`MilpProblem::skip_node_presolve`): every node re-solves the same
//! matrix with swapped bounds, so re-running presolve per node only renumbers
//! columns and kills the propagated warm-start basis. That argument applies to
//! *per-node re-execution* only. Nothing in the MIP pipeline performed any
//! structural reduction at all — bound propagation/probing never deletes a
//! row or column. This module supplies that missing root reduction: it runs
//! once, before symmetry/FP/cuts/B&B, and the downstream stages all operate on
//! the reduced matrix with unchanged warm-start semantics (node LPs share the
//! one reduced matrix, so the no-presolve warm-start contract is preserved).

//! # MIP-safety contract
//! Every rule must be a **feasibility- and objective-preserving bijection on
//! integer feasible points**, in both directions, so that:
//! - reduced infeasible ⇒ original infeasible (and original infeasible ⇒
//!   reduced infeasible),
//! - reduced unbounded ⇒ original unbounded,
//! - the postsolved incumbent is integer-feasible for the *original* problem
//!   with the same objective value.
//!
//! Rules implemented: fixed-column removal, empty/redundant row removal,
//! forcing rows (activity bound pinned to the rhs), singleton rows (equality
//! with an integrality/divisibility guard), parallel/duplicate row merging
//! (with contradiction detection), empty-column removal, and doubleton
//! substitution restricted to the coarse integrality-class split below.

//! # Doubleton substitution — coarse integrality split
//! SCIP's quadruple rule classifies each row member as continuous / dynamic
//! (integer but currently continuous-like) / integer / static. This is the
//! documented coarse safe subset: the eliminated variable `x2` of an equality
//! `a1·x1 + a2·x2 = b` must have lock 1 (occur only in that row, so removal
//! touches no other row, the postsolve dependency graph is acyclic by
//! construction, and density growth is zero). Then:
//! - `x2` **continuous**: always eligible — the reconstructed value
//!   `(b − a1·x1)/a2` is real and its feasibility is exactly what the induced
//!   bound tightening on `x1` encodes; no integrality constraint can break.
//! - `x2` **integer**: eligible only under the unimodular split — `x1` is also
//!   integer (all row members share the integer class), `|a2| = 1` and `a1`,
//!   `b` are integer-valued within [`ROOT_PRESOLVE_INT_DATA_TOL`]. Then
//!   `(b − a1·x1)/a2` is integer for every integer `x1`, i.e. the elimination
//!   is a bijection on integer points. Non-±1 pivots are rejected (they would
//!   relax the lattice and can shift the integer optimum).

//! # Failure handling
//! Provable reductions that reach a contradiction return [`ReduceOutcome::Infeasible`]
//! — sound by the bijection contract. Any rule with numerically inconclusive
//! input (infinite activity, non-finite values, divisibility tolerance breach
//! on data) simply does not fire (fail-open: the row/column is kept). The
//! caller additionally verifies the postsolved solution against the *original*
//! problem and falls back to a full unreduced re-run if verification fails, so
//! a hypothetical unsound reduction cannot corrupt an optimality proof.

use super::presolve::propagation_pass;
use super::stats::MipStats;
use crate::mip::problem::MilpProblem;
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::tolerances::INT_ROUND_TOL;
use otspot_num::linalg::timeout::deadline_reached;
use otspot_num::sparse::CscMatrix;
use std::collections::HashMap;
use std::time::Instant;

/// Maximum structural-reduction rounds. Bound propagation converges within
/// `mip::presolve::MAX_PRESOLVE_ROUNDS` (10); a few extra passes cover the
/// cascades the structural rules add (each removal exposes new fixed columns,
/// forcing rows, and duplicate groups). Reaching the cap stops the loop with
/// the reductions applied so far (still sound) and records it in
/// [`RootPresolveInfo::hit_round_cap`].
const ROOT_PRESOLVE_MAX_ROUNDS: usize = 12;

/// Relative width tolerance for treating a **continuous** `[lb, ub]` as
/// pinned: a column is fixed when `ub − lb ≤ tol · (1 + max(|lb|,|ub|))`.
/// The relative term keeps large-magnitude MPS bounds (data like 1e6 with
/// exact pinning) from being mis-read as floating-point slack of 1e-9.
/// Integer columns are NOT judged by this test — a width of 1 at 1e9 fits
/// inside `tol · scale` while the interval still holds two integers, and
/// pinning then deletes an integer point (PR #89 finding F1). Integer
/// pinning uses [`unique_integer_in_bounds`] instead.
const ROOT_PRESOLVE_PIN_TOL: f64 = 1e-9;

/// ULP budget for the integer pinning uniqueness test
/// ([`unique_integer_in_bounds`]): the interval is widened by
/// `ROOT_PRESOLVE_INT_UNIQUE_ULPS · f64::EPSILON · max(|lb|,|ub|)`, i.e.
/// 1–2 ulps of the bound magnitude per unit. `ceil` and `floor` are
/// exact on f64, so this eps only pays for the widening subtractions
/// themselves. Any magnitude-scaled *decimal* tolerance (1e-9-class) is
/// ~7 orders larger than ulp at 1e9 and would accept intervals spanning
/// two integers.
const ROOT_PRESOLVE_INT_UNIQUE_ULPS: f64 = 4.0;

/// Absolute part of the activity-vs-rhs margin used by redundant-row and
/// infeasibility detection: `tol = ABS + REL · scale` where `scale` is the
/// sum of finite activity contributions plus `|rhs|`. Row activity sums
/// over `nnz` terms, so a pure absolute tolerance would flag rows with
/// large coefficients as "equal" when their floating-point accumulation
/// error exceeds it. This is also the cap on genuine forcing slack —
/// forcing decisions compare against the ABSOLUTE term only (see
/// `row_pass`), never the inflated `tol`.
const ROOT_PRESOLVE_ACT_ABS_TOL: f64 = 1e-9;
/// Relative part of [`ROOT_PRESOLVE_ACT_ABS_TOL`]'s margin: a real
/// rounding-error bound for the activity sum, `64 · f64::EPSILON ·
/// Σ|a_j · bound_j|`-class (the γ_k ≤ k·2⁻⁵³/Σ bound for rows of tens of
/// terms with ~2 digits of safety). The previous 1e-9 looked like parse
/// noise allowance but is ≈ 1.0 at 1e9-scale rows — ~7 orders above the
/// ulp of the accumulated sum (ulp(2e9) ≈ 2.4e-7) — and let merely
/// redundant rows read as forcing (PR #89 finding F2). Parse-noise
/// tolerance belongs to data comparisons (`merge_pass` rhs bit checks),
/// not to activity-vs-rhs decisions.
const ROOT_PRESOLVE_ACT_REL_TOL: f64 = 64.0 * f64::EPSILON;

/// Minimum `|a2|` accepted as a substitution pivot. Dividing by a smaller
/// pivot amplifies round-off into the reconstructed value and into the
/// induced bound tightening; such rows keep their original form instead
/// (same magnitude floor the simplex uses for pivots, `PIVOT_TOL`-class).
const ROOT_PRESOLVE_SUBST_MIN_PIVOT: f64 = 1e-8;

/// Safety loosening applied to the bound induced on `x1` by eliminating
/// `x2` before integer rounding: `new_ub += margin · (1 + |new_ub|)`
/// (symmetrically for `lb`). The induced bound is exact arithmetic on exact
/// MPS data; the margin only pays for f64 round-off in the division and
/// cannot cut off a true optimum by more than the solver's own tolerances.
const ROOT_PRESOLVE_SUBST_BOUND_MARGIN: f64 = 1e-9;

/// Tolerance for testing whether a coefficient / rhs is integer-valued when
/// the unimodular doubleton guard requires integer row data. MPS coefficients
/// parse exactly for integer literals; 1e-6 absorbs parse/round-trip noise
/// while still rejecting genuinely fractional data (1e-3-level) by orders of
/// magnitude.
const ROOT_PRESOLVE_INT_DATA_TOL: f64 = 1e-6;

/// Residual tolerance for verifying a postsolved solution against the
/// *original* problem: a row passes when the violation is within
/// `ABS + REL · (|rhs| + Σ|a_ij·x_j|)`. Set well above the solver's own LP
/// feasibility tolerance so a legitimately-tolerant incumbent is not
/// rejected, yet far below what an actual mapping bug produces (O(1) or
/// larger residuals on the rows the map got wrong).
const ROOT_PRESOLVE_VERIFY_ABS_TOL: f64 = 1e-6;
/// Relative part of [`ROOT_PRESOLVE_VERIFY_ABS_TOL`].
const ROOT_PRESOLVE_VERIFY_REL_TOL: f64 = 1e-7;

/// Reduction counters returned with every [`ReduceOutcome`]. Reported in
/// `MipStats` so "root presolve actually ran" is measurable, never silent.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RootPresolveInfo {
    /// Rows removed (empty / redundant / forcing-resolved / merged / substituted).
    pub rows_removed: usize,
    /// Columns removed (fixed / empty / substituted).
    pub cols_removed: usize,
    /// Doubleton substitutions applied.
    pub substitutions: usize,
    /// Reduction rounds completed before the fixpoint or a stop trigger.
    pub rounds: usize,
    /// The round loop stopped at [`ROOT_PRESOLVE_MAX_ROUNDS`] with work left.
    pub hit_round_cap: bool,
    /// The deadline stopped the loop before the fixpoint.
    pub hit_deadline: bool,
}

/// Substitution record: eliminated column `x2 = (rhs − a1 · x1) / a2`.
#[derive(Debug, Clone, Copy)]
struct SubstRecord {
    a2: f64,
    /// Row rhs at substitution time (later rhs updates cannot affect this row
    /// — it was removed together with `x2`).
    rhs: f64,
    /// The surviving partner column (original index space).
    x1: usize,
    a1: f64,
}

/// Maps a reduced-space solution back to the original variable space.
///
/// `kept[j]` is the original index of reduced column `j`; removed columns are
/// either pinned to a recorded constant (fixed / forcing / empty-column rules)
/// or reconstructible from a [`SubstRecord`]. Substitution dependencies are
/// acyclic by the lock-1 discipline (an eliminated variable's row contains no
/// other lock-1-eliminated variable of the same row), but `construct` still
/// memoizes and cycle-guards as a fail-closed belt.
#[derive(Debug, Clone)]
pub(crate) struct PostsolveMap {
    kept: Vec<usize>,
    pinned: Vec<Option<f64>>,
    subst: HashMap<usize, SubstRecord>,
    integer: Vec<bool>,
}

impl PostsolveMap {
    /// Reconstruct the full original-space solution from the reduced-space one.
    ///
    /// `int_feas_tol` is the caller's integer-feasibility tolerance: kept
    /// integer columns are rounded to the nearest integer within it; a kept
    /// integer column further away is reported as `None` (fail-closed — the
    /// caller then falls back to the unreduced run).
    pub(crate) fn construct(&self, reduced_x: &[f64], int_feas_tol: f64) -> Option<Vec<f64>> {
        if reduced_x.len() != self.kept.len() {
            return None;
        }
        let n = self.pinned.len();
        let mut out = vec![f64::NAN; n];
        for (j, kept_j) in self.kept.iter().enumerate() {
            let mut v = reduced_x[j];
            if self.integer[*kept_j] {
                let r = v.round();
                if (v - r).abs() > int_feas_tol.max(INT_ROUND_TOL) {
                    return None;
                }
                v = r;
            }
            out[*kept_j] = v;
        }
        for j in 0..n {
            if !self.resolve(j, &mut out)? {
                return None;
            }
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// Resolve one original-space column into `out[j]` (pinned constant,
    /// substitution reconstruction, or an already-written kept value).
    /// Memoized recursion with a visiting set for a fail-closed cycle guard.
    fn resolve(&self, j: usize, out: &mut [f64]) -> Option<bool> {
        if !out[j].is_nan() {
            return Some(true);
        }
        if let Some(v) = self.pinned[j] {
            out[j] = v;
            return Some(true);
        }
        let rec = *self.subst.get(&j)?;
        // Cycle guard: mark with +inf; a revisit sees a non-NaN value and the
        // final all-finite check in `construct` rejects the whole mapping.
        out[j] = f64::INFINITY;
        if !self.resolve(rec.x1, out)? {
            return Some(false);
        }
        let x1 = out[rec.x1];
        if !x1.is_finite() {
            return Some(false);
        }
        out[j] = (rec.rhs - rec.a1 * x1) / rec.a2;
        Some(true)
    }

    /// Map a reduced-space [`SolverResult`] back to the original problem and
    /// verify it *independently* before it is published.
    ///
    /// - solution present: reconstruct, then re-check rows / bounds /
    ///   integrality via [`verify_original_solution`] and recompute the
    ///   original-space objective (must agree with the reduced-space value
    ///   the search reported — the bijection preserves objective values).
    /// - no solution (Infeasible / no-incumbent Timeout): returned
    ///   unchanged; reduced infeasible ⇒ original infeasible holds through
    ///   the bijective reduction contract.
    /// - fully pinned (`kept` empty): the pinned point is the only feasible
    ///   point of the reduced space; it replaces the empty solution after
    ///   verification.
    ///
    /// `None` means the mapping or verification failed; the caller then
    /// falls back to an unreduced re-run (fail-closed: a reduction bug can
    /// never masquerade as an optimality proof).
    pub(crate) fn publish(
        &self,
        res: &SolverResult,
        orig: &MilpProblem,
        int_feas_tol: f64,
    ) -> Option<SolverResult> {
        let mut out = res.clone();
        if self.kept.is_empty() {
            // Fully pinned: nothing was solved; the caller's placeholder
            // carries status/objective, the pinned point replaces the empty
            // solution and is verified below exactly like a mapped one.
            out.solution = self.construct(&[], int_feas_tol)?;
        } else if !res.solution.is_empty() {
            out.solution = self.construct(&res.solution, int_feas_tol)?;
        } else {
            return Some(out);
        }
        if !verify_original_solution(orig, &out.solution, int_feas_tol) {
            return None;
        }
        let obj: f64 = orig
            .lp
            .c
            .iter()
            .zip(out.solution.iter())
            .map(|(cj, xj)| cj * xj)
            .sum::<f64>()
            + orig.lp.obj_offset;
        let scale = obj.abs().max(res.objective.abs()).max(1.0);
        if (obj - res.objective).abs()
            > ROOT_PRESOLVE_VERIFY_ABS_TOL + ROOT_PRESOLVE_VERIFY_REL_TOL * scale
        {
            return None;
        }
        out.objective = obj;
        Some(out)
    }
}

/// Result of [`reduce`].
pub(crate) enum ReduceOutcome {
    /// The reduction sequence proved the problem infeasible. Sound in the
    /// original space because every applied rule is a bijection on integer
    /// feasible points (see module docs).
    Infeasible { info: RootPresolveInfo },
    /// No reduction could be applied (problem unchanged) or the state became
    /// numerically unusable before any reduction. The caller keeps the
    /// original problem untouched.
    NotApplied { info: RootPresolveInfo },
    /// Reduced problem plus the postsolve map. When `kept` is empty every
    /// column was pinned: the solution is fully determined and `reduced` is an
    /// empty (0-var) MILP — the caller must use [`PostsolveMap::construct`]
    /// with an empty slice rather than branch-and-bound.
    Reduced {
        /// Boxed: the variant otherwise dwarfs the others (clippy
        /// large_enum_variant) — NotApplied/Infeasible carry only counters.
        reduced: Box<MilpProblem>,
        map: PostsolveMap,
        info: RootPresolveInfo,
    },
}

/// One mutable row in the reduction state. Entries are keyed by *original*
/// column indices; removed columns are filtered out lazily at round boundaries.
#[derive(Debug, Clone)]
struct RowState {
    entries: Vec<(usize, f64)>,
    ctype: ConstraintType,
    rhs: f64,
    alive: bool,
}

/// Mutable reduction state (original index space throughout).
struct RedState {
    rows: Vec<RowState>,
    bounds: Vec<(f64, f64)>,
    c: Vec<f64>,
    obj_delta: f64,
    integer: Vec<bool>,
    alive_col: Vec<bool>,
    pinned: Vec<Option<f64>>,
    subst: HashMap<usize, SubstRecord>,
    /// Row ids per column (superset over time — rows are only removed).
    col_rows: Vec<Vec<usize>>,
    info: RootPresolveInfo,
}

fn pinned_point(lb: f64, ub: f64) -> f64 {
    if lb.is_finite() {
        lb
    } else {
        ub
    }
}

fn is_pinned(lb: f64, ub: f64) -> bool {
    // Both sides must be finite: an infinite bound is never "pinned"
    // (the width-vs-scale test alone degenerates because
    // `inf <= tol · inf` holds in floating point, which would pin a free
    // column to its finite endpoint and can cut off an unbounded ray).
    if !lb.is_finite() || !ub.is_finite() {
        return false;
    }
    if lb > ub {
        return false;
    }
    (ub - lb) <= ROOT_PRESOLVE_PIN_TOL * (1.0 + lb.abs().max(ub.abs()))
}

/// The unique integer inside `[lb − eps, ub + eps]`, or `None` when the
/// interval contains no integer or more than one. This is the sound
/// integer-pinning criterion: removing the column is a bijection on
/// integer points exactly when its integer-feasible set is a single
/// point, which a width-vs-scale test cannot decide at large magnitude
/// (F1: `[1e9, 1e9 + 1]` passes `1e-9 · scale` yet holds two integers;
/// pinning loses one of them). The eps is ULP-scale
/// ([`ROOT_PRESOLVE_INT_UNIQUE_ULPS`]), so exact integer data decides
/// exactly and parse noise below a few ulps still pins as before.
fn unique_integer_in_bounds(lb: f64, ub: f64) -> Option<f64> {
    if !lb.is_finite() || !ub.is_finite() || lb > ub {
        return None;
    }
    let eps = ROOT_PRESOLVE_INT_UNIQUE_ULPS * f64::EPSILON * lb.abs().max(ub.abs());
    let lo = (lb - eps).ceil();
    let hi = (ub + eps).floor();
    (lo == hi).then_some(lo)
}

fn activity_margin(scale: f64) -> f64 {
    ROOT_PRESOLVE_ACT_ABS_TOL + ROOT_PRESOLVE_ACT_REL_TOL * scale
}

/// Min/max activity of a row over the current box. `None` when the needed
/// side is unbounded (∞ coefficients — the caller skips activity rules).
fn row_activity_bounds(entries: &[(usize, f64)], bounds: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut min_act = 0.0;
    let mut max_act = 0.0;
    for &(j, a) in entries {
        let (lb, ub) = bounds[j];
        let (lo, hi) = if a > 0.0 { (lb, ub) } else { (ub, lb) };
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        min_act += a * if a > 0.0 { lb } else { ub };
        max_act += a * if a > 0.0 { ub } else { lb };
    }
    Some((min_act, max_act))
}

/// Integer value inside `[lb, ub]`, preferring the one nearest 0; `None` when
/// the interval contains no integer (infeasible for an integer column).
fn integer_in_bounds(lb: f64, ub: f64) -> Option<f64> {
    let ceil_lb = lb.ceil();
    let floor_ub = ub.floor();
    if ceil_lb <= floor_ub + ROOT_PRESOLVE_PIN_TOL {
        // Clamp 0 into the interval, then round toward the valid range.
        let t = 0.0_f64.clamp(lb, ub);
        let r = t.round();
        if r >= ceil_lb - ROOT_PRESOLVE_PIN_TOL && r <= floor_ub + ROOT_PRESOLVE_PIN_TOL {
            return Some(r);
        }
        return Some(if r < ceil_lb { ceil_lb } else { floor_ub });
    }
    if (lb - floor_ub).abs() <= ROOT_PRESOLVE_PIN_TOL {
        return Some(floor_ub);
    }
    if (ceil_lb - ub).abs() <= ROOT_PRESOLVE_PIN_TOL {
        return Some(ceil_lb);
    }
    None
}

/// Continuous value inside `[lb, ub]` nearest 0 (handles infinite sides).
fn continuous_in_bounds(lb: f64, ub: f64) -> Option<f64> {
    match (lb.is_finite(), ub.is_finite()) {
        (true, true) => Some(0.0_f64.clamp(lb, ub)),
        (true, false) => Some(lb),
        (false, true) => Some(ub),
        (false, false) => Some(0.0),
    }
}

fn is_integer_valued(v: f64) -> bool {
    v.is_finite() && (v - v.round()).abs() <= ROOT_PRESOLVE_INT_DATA_TOL
}

impl RedState {
    fn new(problem: &MilpProblem) -> Self {
        let n = problem.lp.num_vars;
        let m = problem.lp.num_constraints;
        let mut rows: Vec<RowState> = (0..m)
            .map(|i| RowState {
                entries: Vec::new(),
                ctype: problem.lp.constraint_types[i],
                rhs: problem.lp.b[i],
                alive: true,
            })
            .collect();
        let mut col_rows = vec![Vec::new(); n];
        for j in 0..n {
            for k in problem.lp.a.col_ptr()[j]..problem.lp.a.col_ptr()[j + 1] {
                let i = problem.lp.a.row_ind()[k];
                let v = problem.lp.a.values()[k];
                rows[i].entries.push((j, v));
                col_rows[j].push(i);
            }
        }
        let mut integer = vec![false; n];
        for &j in &problem.integer_vars {
            integer[j] = true;
        }
        RedState {
            rows,
            bounds: problem.lp.bounds.clone(),
            c: problem.lp.c.clone(),
            obj_delta: 0.0,
            integer,
            alive_col: vec![true; n],
            pinned: vec![None; n],
            subst: HashMap::new(),
            col_rows,
            info: RootPresolveInfo::default(),
        }
    }

    /// Remove column `j` after pinning it to `v`: rhs updates and objective
    /// offset only (row entries are filtered lazily).
    fn pin_column(&mut self, j: usize, v: f64) {
        for &i in &self.col_rows[j] {
            if self.rows[i].alive {
                if let Some(&(_, a)) = self.rows[i].entries.iter().find(|&&(k, _)| k == j) {
                    self.rows[i].rhs -= a * v;
                }
            }
        }
        self.obj_delta += self.c[j] * v;
        self.alive_col[j] = false;
        self.pinned[j] = Some(v);
        self.info.cols_removed += 1;
    }

    /// Propagate bounds over alive rows, then pin and remove fixed columns.
    /// Returns `None` on infeasibility; on success reports whether anything
    /// changed. Dead columns are filtered out of the propagated rows: their
    /// contributions were already folded into the rhs (pinned) or moved into
    /// the partner's bound (substituted), so propagating their stale bounds
    /// would double-count them.
    fn propagate_and_fix(&mut self) -> Option<(bool, bool)> {
        // Build the alive-row LP snapshot for the propagation pass: dead
        // columns are filtered out (their contributions were already folded
        // into the rhs or moved into the partner's bound).
        let mut alive_entries: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.rows.len());
        let mut ctypes: Vec<ConstraintType> = Vec::with_capacity(self.rows.len());
        let mut rhs: Vec<f64> = Vec::with_capacity(self.rows.len());
        for r in self.rows.iter().filter(|r| r.alive) {
            alive_entries.push(
                r.entries
                    .iter()
                    .filter(|(j, _)| self.alive_col[*j])
                    .copied()
                    .collect(),
            );
            ctypes.push(r.ctype);
            rhs.push(r.rhs);
        }
        let mut bounds = self.bounds.clone();
        let tightened =
            propagation_pass(&alive_entries, &ctypes, &rhs, &mut bounds, &self.integer)?;
        let mut changed = tightened > 0;
        if bounds != self.bounds {
            self.bounds = bounds;
        }
        let mut pinned_any = false;
        for j in 0..self.bounds.len() {
            if !self.alive_col[j] {
                continue;
            }
            let (lb, ub) = self.bounds[j];
            if self.integer[j] {
                // Integer columns pin only on a UNIQUE integer in the
                // interval (F1): the continuous width-vs-scale test would
                // pin `[1e9, 1e9 + 1]` and delete an integer point. An
                // interval containing no integer is left alone —
                // `propagation_pass` already ceils/floors integer bounds
                // and reports the empty-interval infeasibility there
                // (fail-open here: keeping a column is always sound).
                if let Some(v) = unique_integer_in_bounds(lb, ub) {
                    self.pin_column(j, v);
                    pinned_any = true;
                }
                continue;
            }
            if is_pinned(lb, ub) {
                self.pin_column(j, pinned_point(lb, ub));
                pinned_any = true;
            }
        }
        changed |= pinned_any;
        Some((changed, false))
    }

    /// Endpoints attaining `which` (min or max) activity, as bound updates.
    /// Returns `None` when some member cannot safely be pinned to its
    /// endpoint — an infinite endpoint (defensive: the caller's activity
    /// check already rejects those) or a fractional endpoint on an integer
    /// column. The caller must then keep the row: removing a forcing row
    /// without pinning every member is not a sound reduction.
    fn force_endpoints(
        &mut self,
        entries: &[(usize, f64)],
        at_min: bool,
        changed: &mut bool,
    ) -> Option<()> {
        for &(j, a) in entries {
            if !self.alive_col[j] {
                continue;
            }
            let (lb, ub) = self.bounds[j];
            let endpoint = if (a > 0.0) == at_min { lb } else { ub };
            if !endpoint.is_finite() {
                return None;
            }
            if self.integer[j] && !is_integer_valued(endpoint) {
                return None;
            }
            // Already at the single point? Compare exactly: the continuous
            // `is_pinned` scale test would call `[1e9, 1e9 + 1]` "pinned"
            // and let the row be removed while this member keeps a full
            // unit of width (F1 family). Writing equal bounds again is
            // harmless, so this is purely the `changed` bookkeeping skip.
            if lb != ub {
                self.bounds[j] = (endpoint, endpoint);
                *changed = true;
            }
        }
        Some(())
    }

    /// Row rules: empty / activity infeasible / redundant / forcing /
    /// singleton equality. Returns `Some((changed, infeasible))`.
    fn row_pass(&mut self) -> (bool, bool) {
        let mut changed = false;
        for i in 0..self.rows.len() {
            if !self.rows[i].alive {
                continue;
            }
            self.rows[i]
                .entries
                .retain(|(j, _)| self.alive_col[*j] && *j < self.bounds.len());
            let entries = self.rows[i].entries.clone();
            let rhs = self.rows[i].rhs;
            let scale = rhs.abs()
                + entries
                    .iter()
                    .map(|&(j, a)| a * self.bounds[j].0.abs().max(self.bounds[j].1.abs()))
                    .fold(
                        0.0_f64,
                        |acc, v| if v.is_finite() { acc + v.abs() } else { acc },
                    );
            let tol = activity_margin(scale);
            if entries.is_empty() {
                let ok = match self.rows[i].ctype {
                    ConstraintType::Le => rhs >= -tol,
                    ConstraintType::Ge => rhs <= tol,
                    ConstraintType::Eq => rhs.abs() <= tol,
                };
                if !ok {
                    return (changed, true);
                }
                self.rows[i].alive = false;
                self.info.rows_removed += 1;
                changed = true;
                continue;
            }
            // Singleton equality pins its variable outright (works even with
            // unbounded partners where the activity rule below is blind).
            if entries.len() == 1 && matches!(self.rows[i].ctype, ConstraintType::Eq) {
                let (j, a) = entries[0];
                if a.abs() < ROOT_PRESOLVE_SUBST_MIN_PIVOT {
                    continue;
                }
                let v = rhs / a;
                if self.integer[j] {
                    let r = v.round();
                    if (v - r).abs() > ROOT_PRESOLVE_INT_DATA_TOL * (1.0 + v.abs()) {
                        return (changed, true);
                    }
                    let (lb, ub) = self.bounds[j];
                    if r < lb - ROOT_PRESOLVE_INT_DATA_TOL || r > ub + ROOT_PRESOLVE_INT_DATA_TOL {
                        return (changed, true);
                    }
                    self.bounds[j] = (r, r);
                } else {
                    self.bounds[j] = (v, v);
                }
                self.rows[i].alive = false;
                self.info.rows_removed += 1;
                changed = true;
                continue;
            }
            let Some((min_act, max_act)) = row_activity_bounds(&entries, &self.bounds) else {
                continue;
            };
            match self.rows[i].ctype {
                ConstraintType::Le => {
                    if min_act > rhs + tol {
                        return (changed, true);
                    }
                    if max_act <= rhs + tol {
                        // Row implied over the box. Pinning is only sound
                        // for GENUINE forcing: the real slack `rhs −
                        // min_act` below the absolute activity tolerance
                        // means no feasible point can lift any member off
                        // its min-activity endpoint. The scaled `tol`
                        // bounds accumulation error, not forcing slack:
                        // at 1e9-scale rows it is O(1)-wide and would
                        // "force" endpoints of a merely redundant row,
                        // deleting integer points (F2).
                        let forcing = rhs - min_act <= ROOT_PRESOLVE_ACT_ABS_TOL;
                        if !forcing || self.force_endpoints(&entries, true, &mut changed).is_some()
                        {
                            self.rows[i].alive = false;
                            self.info.rows_removed += 1;
                            changed = true;
                        }
                    }
                }
                ConstraintType::Ge => {
                    if max_act < rhs - tol {
                        return (changed, true);
                    }
                    if min_act >= rhs - tol {
                        // Mirror of the Le forcing rule above.
                        let forcing = max_act - rhs <= ROOT_PRESOLVE_ACT_ABS_TOL;
                        if !forcing
                            || self
                                .force_endpoints(&entries, false, &mut changed)
                                .is_some()
                        {
                            self.rows[i].alive = false;
                            self.info.rows_removed += 1;
                            changed = true;
                        }
                    }
                }
                ConstraintType::Eq => {
                    if min_act > rhs + tol || max_act < rhs - tol {
                        return (changed, true);
                    }
                    // Per-side forcing (F2's equality analogue): pin only
                    // when one extreme of the activity window coincides
                    // with the rhs within the ABSOLUTE tolerance — then
                    // every feasible point attains that extreme, so every
                    // member sits at its endpoint for it. Pinning the
                    // opposite extreme (the old scale-inflated test, which
                    // fired whenever both sides were within `tol`) puts
                    // the members at an endpoint whose activity misses
                    // the rhs by up to 2·tol and drops feasible points.
                    let lo_force = (min_act - rhs).abs() <= ROOT_PRESOLVE_ACT_ABS_TOL;
                    let hi_force = (max_act - rhs).abs() <= ROOT_PRESOLVE_ACT_ABS_TOL;
                    if lo_force || hi_force {
                        // Pin the extreme that equals the rhs; if both do
                        // (window collapsed onto it), prefer the closer.
                        let at_min = lo_force && rhs - min_act <= max_act - rhs;
                        if self
                            .force_endpoints(&entries, at_min, &mut changed)
                            .is_some()
                        {
                            self.rows[i].alive = false;
                            self.info.rows_removed += 1;
                            changed = true;
                        }
                    } else if max_act <= rhs + tol {
                        // The ≤ side holds over the whole box, so it is
                        // the redundant half of the equality; keep the
                        // ≥ side (the only one that can cut the box).
                        // (The pre-fix form set Le here, keeping the
                        // satisfied side and silently dropping the
                        // binding constraint — same root cause, F2-adjacent.)
                        self.rows[i].ctype = ConstraintType::Ge;
                        changed = true;
                    } else if min_act >= rhs - tol {
                        self.rows[i].ctype = ConstraintType::Le;
                        changed = true;
                    }
                }
            }
        }
        (changed, false)
    }

    /// Doubleton substitution over alive equality rows with exactly two live
    /// entries. See module docs for the safety split.
    fn doubleton_pass(&mut self) -> (bool, bool) {
        let mut locks = vec![0usize; self.bounds.len()];
        for r in self.rows.iter().filter(|r| r.alive) {
            for &(j, _) in &r.entries {
                locks[j] += 1;
            }
        }
        let mut changed = false;
        for i in 0..self.rows.len() {
            if !self.rows[i].alive || !matches!(self.rows[i].ctype, ConstraintType::Eq) {
                continue;
            }
            let entries: Vec<(usize, f64)> = self.rows[i]
                .entries
                .iter()
                .copied()
                .filter(|(j, _)| self.alive_col[*j])
                .collect();
            if entries.len() != 2 {
                continue;
            }
            let rhs = self.rows[i].rhs;
            let (p0, p1) = (entries[0], entries[1]);
            // First eligible orientation wins (`break` below): orientation 0
            // is tried, and only a guard rejection (`continue`) switches to
            // the mirror orientation.
            for (target, partner) in [(p0, p1), (p1, p0)] {
                let (j2, a2) = target;
                let (j1, a1) = partner;
                if locks[j2] != 1 || a2.abs() < ROOT_PRESOLVE_SUBST_MIN_PIVOT {
                    continue;
                }
                if self.integer[j2] {
                    let unimodular = self.integer[j1]
                        && (a2.abs() - 1.0).abs() <= ROOT_PRESOLVE_INT_DATA_TOL
                        && is_integer_valued(a1)
                        && is_integer_valued(rhs);
                    if !unimodular {
                        continue;
                    }
                }
                if self.eliminate_doubleton(j2, a2, rhs, j1, a1) {
                    self.rows[i].alive = false;
                    self.info.rows_removed += 1;
                    self.info.substitutions += 1;
                    changed = true;
                }
                break;
            }
        }
        (changed, false)
    }

    /// Apply one substitution: induced bound tightening on `x1`, objective
    /// transfer, column removal with a [`SubstRecord`]. Returns false only
    /// when the induced tightening would prove infeasibility past tolerance
    /// margins (caller keeps the row — fail-open).
    fn eliminate_doubleton(&mut self, j2: usize, a2: f64, rhs: f64, j1: usize, a1: f64) -> bool {
        // Induced bounds on x1 from a1·x1 + a2·x2 = rhs with x2 ∈ [lb2, ub2]:
        //   a2 > 0:  rhs − a2·ub2 ≤ a1·x1 ≤ rhs − a2·lb2
        //   a2 < 0:  rhs − a2·lb2 ≤ a1·x1 ≤ rhs − a2·ub2
        // (then divide by a1, flipping the side when a1 < 0).
        let (lb2, ub2) = self.bounds[j2];
        let (l1, u1) = self.bounds[j1];
        let (mut nl1, mut nu1) = (l1, u1);
        for (bound2, is_lower_for_x1) in [(lb2, false), (ub2, true)] {
            if !bound2.is_finite() || a1 == 0.0 {
                continue;
            }
            // The bound2 that caps x1 from above/below depends on sign(a2).
            let hi_side = if a2 > 0.0 {
                !is_lower_for_x1
            } else {
                is_lower_for_x1
            };
            let cap = (rhs - a2 * bound2) / a1;
            let m = ROOT_PRESOLVE_SUBST_BOUND_MARGIN * (1.0 + cap.abs());
            if a1 > 0.0 {
                // cap is the direct bound on x1.
                if hi_side {
                    nu1 = nu1.min(cap + m);
                } else {
                    nl1 = nl1.max(cap - m);
                }
            } else {
                // Division by a1 < 0 flips: cap bounds x1 on the other side.
                if hi_side {
                    nl1 = nl1.max(cap - m);
                } else {
                    nu1 = nu1.min(cap + m);
                }
            }
        }
        if nl1 > nu1 {
            return false;
        }
        let (nl1, nu1) = if self.integer[j1] {
            match integer_in_bounds(nl1, nu1) {
                Some(_) => (nl1.floor(), nu1.ceil()),
                None => return false,
            }
        } else {
            (nl1, nu1)
        };
        if nl1 != l1 || nu1 != u1 {
            self.bounds[j1] = (nl1, nu1);
        }
        // Objective transfer: c2·x2 = c2·(rhs − a1·x1)/a2 = c2·rhs/a2 −
        // (c2·a1/a2)·x1, so the constant goes to the offset and the x1 part
        // to the partner's objective coefficient.
        let c2 = self.c[j2];
        if c2 != 0.0 {
            self.obj_delta += c2 * rhs / a2;
            self.c[j1] -= c2 * a1 / a2;
        }
        self.alive_col[j2] = false;
        self.info.cols_removed += 1;
        self.subst.insert(
            j2,
            SubstRecord {
                a2,
                rhs,
                x1: j1,
                a1,
            },
        );
        true
    }

    /// Parallel/duplicate row merging on identical *live* entry patterns
    /// (dead columns filtered out first: their contributions are already
    /// folded into the rhs, and a stale coefficient that differs between two
    /// rows would otherwise hide a genuine duplicate).
    ///
    /// Non-kept rows are dropped only on *exact* arithmetic: dropping the
    /// weaker of two same-sense rows is plain redundancy with no tolerance
    /// involved. Contradictions are declared infeasible only when the rhs
    /// gap exceeds the activity margin; an Eq-vs-Eq gap *inside* the margin
    /// is ambiguous (genuine near-contradiction vs floating-point noise),
    /// and the merge then fails open — every Eq row is kept for the LP and
    /// B&B to judge with their own tolerances, instead of the merge picking
    /// a winner that could be the wrong side of a real contradiction.
    fn merge_pass(&mut self) -> (bool, bool) {
        let mut groups: HashMap<Vec<(usize, u64)>, Vec<usize>> = HashMap::new();
        for (i, r) in self.rows.iter().enumerate() {
            if !r.alive {
                continue;
            }
            let key: Vec<(usize, u64)> = r
                .entries
                .iter()
                .filter(|(j, _)| self.alive_col[*j])
                .map(|(j, v)| (*j, v.to_bits()))
                .collect();
            if key.is_empty() {
                continue;
            }
            groups.entry(key).or_default().push(i);
        }
        let mut changed = false;
        for members in groups.values() {
            if members.len() < 2 {
                continue;
            }
            // Contradiction check across Le/Ge pairs, best-of within a type.
            let best_le = members
                .iter()
                .filter(|&&i| matches!(self.rows[i].ctype, ConstraintType::Le))
                .copied()
                .min_by(|a, b| self.rows[*a].rhs.total_cmp(&self.rows[*b].rhs));
            let best_ge = members
                .iter()
                .filter(|&&i| matches!(self.rows[i].ctype, ConstraintType::Ge))
                .copied()
                .max_by(|a, b| self.rows[*a].rhs.total_cmp(&self.rows[*b].rhs));
            let eqs: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&i| matches!(self.rows[i].ctype, ConstraintType::Eq))
                .collect();
            if let (Some(le), Some(ge)) = (best_le, best_ge) {
                let scale = self.rows[le].rhs.abs().max(self.rows[ge].rhs.abs()) + 1.0;
                if self.rows[le].rhs < self.rows[ge].rhs - activity_margin(scale) {
                    return (changed, true);
                }
            }
            // Eq rows collapse to one only when all their rhs bits agree.
            // Extremes (min vs max) are compared, not adjacent pairs, so a
            // group of three with the ends contradicting is caught. Beyond
            // the margin: infeasible. Inside it with differing bits: keep
            // every Eq row (fail-open, see the doc comment).
            let min_eq = eqs
                .iter()
                .copied()
                .min_by(|&a, &b| self.rows[a].rhs.total_cmp(&self.rows[b].rhs));
            let max_eq = eqs
                .iter()
                .copied()
                .max_by(|&a, &b| self.rows[a].rhs.total_cmp(&self.rows[b].rhs));
            let mut eqs_collapse = true;
            if let (Some(mn), Some(mx)) = (min_eq, max_eq) {
                let (r_min, r_max) = (self.rows[mn].rhs, self.rows[mx].rhs);
                if r_min.to_bits() != r_max.to_bits() {
                    let scale = r_min.abs().max(r_max.abs()) + 1.0;
                    if r_max - r_min > activity_margin(scale) {
                        return (changed, true);
                    }
                    eqs_collapse = false;
                }
            }
            let mut keep: Vec<usize> = [best_le, best_ge].into_iter().flatten().collect();
            if eqs_collapse {
                keep.extend(eqs.first().copied());
            } else {
                keep.extend(eqs.iter().copied());
            }
            for &i in members {
                if !keep.contains(&i) && self.rows[i].alive {
                    self.rows[i].alive = false;
                    self.info.rows_removed += 1;
                    changed = true;
                }
            }
        }
        (changed, false)
    }

    /// Remove columns with no alive-row entries and exactly-zero objective
    /// coefficient (free constant: any box value attains the same objective).
    fn empty_col_pass(&mut self) -> (bool, bool) {
        let mut used = vec![false; self.bounds.len()];
        for r in self.rows.iter().filter(|r| r.alive) {
            for &(j, _) in &r.entries {
                if self.alive_col[j] {
                    used[j] = true;
                }
            }
        }
        let mut changed = false;
        for j in 0..self.bounds.len() {
            if !self.alive_col[j] || used[j] || self.c[j] != 0.0 {
                continue;
            }
            let (lb, ub) = self.bounds[j];
            let v = if self.integer[j] {
                match integer_in_bounds(lb, ub) {
                    Some(v) => v,
                    None => return (changed, true),
                }
            } else {
                match continuous_in_bounds(lb, ub) {
                    Some(v) => v,
                    None => return (changed, true),
                }
            };
            self.pin_column(j, v);
            changed = true;
        }
        (changed, false)
    }
}

/// Run the root structural reduction. See module docs for the safety contract.
pub(crate) fn reduce(problem: &MilpProblem, deadline: Option<Instant>) -> ReduceOutcome {
    let mut st = RedState::new(problem);
    let mut infeasible = false;
    for round in 0..ROOT_PRESOLVE_MAX_ROUNDS {
        if deadline_reached(deadline) {
            st.info.hit_deadline = true;
            st.info.rounds = round;
            return finalize(problem, st);
        }
        st.info.rounds = round + 1;
        let mut changed_round = false;
        match st.propagate_and_fix() {
            None => {
                infeasible = true;
                break;
            }
            Some((c, _)) => changed_round |= c,
        }
        let (c, inf) = st.row_pass();
        infeasible |= inf;
        changed_round |= c;
        if infeasible {
            break;
        }
        let (c, inf) = st.doubleton_pass();
        infeasible |= inf;
        changed_round |= c;
        if infeasible {
            break;
        }
        let (c, inf) = st.merge_pass();
        infeasible |= inf;
        changed_round |= c;
        if infeasible {
            break;
        }
        let (c, inf) = st.empty_col_pass();
        infeasible |= inf;
        if inf {
            break;
        }
        changed_round |= c;
        if !changed_round {
            return finalize(problem, st);
        }
    }
    if infeasible {
        st.info.hit_round_cap = false;
        return ReduceOutcome::Infeasible { info: st.info };
    }
    st.info.hit_round_cap = st.info.rounds >= ROOT_PRESOLVE_MAX_ROUNDS;
    finalize(problem, st)
}

/// Build the reduced [`MilpProblem`] + [`PostsolveMap`], or report
/// [`ReduceOutcome::NotApplied`] when nothing was removed.
fn finalize(problem: &MilpProblem, st: RedState) -> ReduceOutcome {
    let info = st.info;
    if info.rows_removed == 0 && info.cols_removed == 0 {
        return ReduceOutcome::NotApplied { info };
    }
    let kept: Vec<usize> = (0..st.alive_col.len())
        .filter(|&j| st.alive_col[j])
        .collect();
    let mut pos = vec![usize::MAX; st.alive_col.len()];
    for (new_j, &orig_j) in kept.iter().enumerate() {
        pos[orig_j] = new_j;
    }
    let alive_rows: Vec<&RowState> = st.rows.iter().filter(|r| r.alive).collect();
    let mut rows_v = Vec::new();
    let mut cols_v = Vec::new();
    let mut vals_v = Vec::new();
    for (i, r) in alive_rows.iter().enumerate() {
        for &(j, a) in &r.entries {
            if st.alive_col[j] && a != 0.0 {
                rows_v.push(i);
                cols_v.push(pos[j]);
                vals_v.push(a);
            }
        }
    }
    let a = match CscMatrix::from_triplets(&rows_v, &cols_v, &vals_v, alive_rows.len(), kept.len())
    {
        Ok(m) => m,
        // Numerically unusable reduction (should not happen — inputs already
        // validated). Fail-open: keep the original problem.
        Err(_) => return ReduceOutcome::NotApplied { info },
    };
    let c: Vec<f64> = kept.iter().map(|&j| st.c[j]).collect();
    let bounds: Vec<(f64, f64)> = kept.iter().map(|&j| st.bounds[j]).collect();
    let integer_vars: Vec<usize> = kept
        .iter()
        .enumerate()
        .filter(|(_, &j)| st.integer[j])
        .map(|(new_j, _)| new_j)
        .collect();
    let ctypes: Vec<ConstraintType> = alive_rows.iter().map(|r| r.ctype).collect();
    let rhs: Vec<f64> = alive_rows.iter().map(|r| r.rhs).collect();
    let mut lp = match LpProblem::new_general(c, a, rhs, ctypes, bounds, problem.lp.name.clone()) {
        Ok(l) => l,
        Err(_) => return ReduceOutcome::NotApplied { info },
    };
    lp.obj_offset = problem.lp.obj_offset + st.obj_delta;
    let reduced = match MilpProblem::new(lp, integer_vars) {
        Ok(m) => m,
        Err(_) => return ReduceOutcome::NotApplied { info },
    };
    ReduceOutcome::Reduced {
        reduced: Box::new(reduced),
        map: PostsolveMap {
            kept,
            pinned: st.pinned,
            subst: st.subst,
            integer: st.integer,
        },
        info,
    }
}

/// Early-return variant of [`apply`]: `(result, stats)` that
/// `solve_milp_with_stats` must return directly (proved-infeasible or the
/// fully-pinned point). Boxed for the same reason as `MipSearchEarlyReturn`.
type ApplyEarlyReturn = Box<(SolverResult, MipStats)>;

/// Successful [`apply`] outcome: the problem the rest of the root pipeline
/// continues on, plus the postsolve bookkeeping.
pub(crate) struct RootPresolveStep {
    /// Reduced problem, or the original one when nothing was removed / the
    /// flag is off.
    pub problem: MilpProblem,
    /// Postsolve map when the reduction fired; the final incumbent must go
    /// through [`PostsolveMap::publish`] before being returned.
    pub postsolve: Option<PostsolveMap>,
    /// Wall-clock microseconds of the reduce() call (true zero when the
    /// flag is off — nothing ran).
    pub rp_us: u64,
    /// Reduction counters (also reported when an early return happened).
    pub info: RootPresolveInfo,
}

/// Run the root structural presolve stage of `solve_milp_with_stats`.
///
/// `Err(early)` means the caller must return that `(result, stats)`
/// immediately: either the reduction proved infeasibility (sound in the
/// original space by the bijection contract), or every column got pinned
/// (the reduced search space is empty — the pinned point is published
/// through the same verification path, with the unreduced fallback if
/// verification fails). The infeasible return attributes its cost to
/// `relaxation_time_infeasible_ms` plus the root buckets, mirroring the
/// bound-probing infeasible return.
#[allow(clippy::too_many_arguments)] // root-stage glue: timings are threaded through
pub(crate) fn apply(
    input: MilpProblem,
    orig: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
    deadline: Option<Instant>,
    probing_ms: f64,
    probing_us: u64,
) -> Result<RootPresolveStep, ApplyEarlyReturn> {
    let t0 = Instant::now();
    let outcome = if cfg.root_presolve {
        reduce(&input, deadline)
    } else {
        ReduceOutcome::NotApplied {
            info: RootPresolveInfo::default(),
        }
    };
    // With the flag off nothing ran, so the bucket stays a true zero (not a
    // measurement of the skipped branch — that would add jitter noise to
    // every disabled run).
    let elapsed = if cfg.root_presolve {
        t0.elapsed()
    } else {
        std::time::Duration::ZERO
    };
    let rp_ms = elapsed.as_secs_f64() * 1000.0;
    let rp_us = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
    let info = match &outcome {
        ReduceOutcome::Infeasible { info }
        | ReduceOutcome::NotApplied { info }
        | ReduceOutcome::Reduced { info, .. } => *info,
    };
    let counters = |m: &mut MipStats| {
        m.root_presolve_us = rp_us;
        m.root_presolve_rows_removed = info.rows_removed;
        m.root_presolve_cols_removed = info.cols_removed;
        m.root_presolve_substitutions = info.substitutions;
    };
    let (problem, postsolve) = match outcome {
        ReduceOutcome::Infeasible { .. } => {
            let mut stats = MipStats {
                relaxation_time_infeasible_ms: probing_ms + rp_ms,
                root_probing_us: probing_us,
                ..Default::default()
            };
            counters(&mut stats);
            return Err(Box::new((SolverResult::infeasible(), stats)));
        }
        ReduceOutcome::NotApplied { .. } => (input, None),
        ReduceOutcome::Reduced { reduced, map, .. } => (*reduced, Some(map)),
    };
    // Fully pinned: the reduced search space is empty — the pinned point is
    // the only candidate, so symmetry/FP/cuts/B&B are all skipped.
    if postsolve.is_some() && problem.lp.num_vars == 0 {
        let placeholder = SolverResult {
            status: SolveStatus::Optimal,
            objective: problem.lp.obj_offset,
            ..SolverResult::default()
        };
        let mut carried = MipStats {
            root_probing_us: probing_us,
            ..MipStats::default()
        };
        counters(&mut carried);
        let mapped = postsolve
            .as_ref()
            .and_then(|m| m.publish(&placeholder, orig, cfg.integer_feas_tol));
        return Err(Box::new(match mapped {
            Some(res) => (res, carried),
            None => rerun_without_root_presolve(orig, options, cfg, &carried),
        }));
    }
    Ok(RootPresolveStep {
        problem,
        postsolve,
        rp_us,
        info,
    })
}

/// Fail-closed escape hatch of `solve_milp_with_stats`: the postsolved
/// solution of the root structural presolve failed its independent check
/// against the original problem (or its reconstruction), so the search is
/// repeated with the reduction disabled.
///
/// Attribution: `carried` holds the root-phase buckets (and reduction
/// counters) of the discarded reduced attempt — they are folded into the
/// fallback's stats so the spent time is not hidden, and
/// `MipStats::root_presolve_fallbacks` records the event. The discarded
/// attempt's per-node search time deliberately stays unattributed (it
/// belongs to a result that was thrown away; counting it as search work
/// would distort the node-keyed effort gates of the surviving run).
pub(crate) fn rerun_without_root_presolve(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
    carried: &MipStats,
) -> (SolverResult, MipStats) {
    let mut off_cfg = cfg.clone();
    off_cfg.root_presolve = false;
    let (res, mut stats) = super::solve_milp_with_stats(problem, options, &off_cfg);
    stats.root_probing_us = stats
        .root_probing_us
        .saturating_add(carried.root_probing_us);
    stats.root_presolve_us = stats
        .root_presolve_us
        .saturating_add(carried.root_presolve_us);
    stats.root_symmetry_us = stats
        .root_symmetry_us
        .saturating_add(carried.root_symmetry_us);
    stats.fp_us = stats.fp_us.saturating_add(carried.fp_us);
    stats.root_cut_us = stats.root_cut_us.saturating_add(carried.root_cut_us);
    stats.root_presolve_rows_removed += carried.root_presolve_rows_removed;
    stats.root_presolve_cols_removed += carried.root_presolve_cols_removed;
    stats.root_presolve_substitutions += carried.root_presolve_substitutions;
    stats.root_presolve_fallbacks += 1;
    (res, stats)
}

/// Verify a postsolved original-space solution against the *original* problem:
/// all rows (with magnitude-relative tolerances), bounds, and integrality.
/// Used before publishing a postsolved result; a failure triggers the
/// caller's unreduced fallback, so a reduction bug can never masquerade as an
/// optimality proof.
pub(crate) fn verify_original_solution(orig: &MilpProblem, x: &[f64], int_feas_tol: f64) -> bool {
    if x.len() != orig.lp.num_vars {
        return false;
    }
    for (j, &(lb, ub)) in orig.lp.bounds.iter().enumerate() {
        let slack = ROOT_PRESOLVE_VERIFY_ABS_TOL + ROOT_PRESOLVE_VERIFY_REL_TOL * x[j].abs();
        if x[j] < lb - slack || x[j] > ub + slack {
            return false;
        }
        if orig.integer_vars.contains(&j) {
            let r = x[j].round();
            if (x[j] - r).abs() > int_feas_tol.max(ROOT_PRESOLVE_VERIFY_ABS_TOL) {
                return false;
            }
        }
    }
    if orig.lp.num_constraints == 0 {
        return true;
    }
    let lhs = match orig.lp.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Row scale: |rhs| + Σ|a_ij·x_j| (magnitude-relative violation budget).
    // The matrix is CSC, so the per-row sums accumulate through a column walk.
    let mut scale = vec![0.0_f64; orig.lp.num_constraints];
    for (i, s) in scale.iter_mut().enumerate() {
        *s = orig.lp.b[i].abs();
    }
    for j in 0..orig.lp.num_vars {
        for k in orig.lp.a.col_ptr()[j]..orig.lp.a.col_ptr()[j + 1] {
            let i = orig.lp.a.row_ind()[k];
            scale[i] += orig.lp.a.values()[k].abs() * x[j].abs();
        }
    }
    for i in 0..orig.lp.num_constraints {
        let tol = ROOT_PRESOLVE_VERIFY_ABS_TOL + ROOT_PRESOLVE_VERIFY_REL_TOL * scale[i];
        let d = lhs[i] - orig.lp.b[i];
        let ok = match orig.lp.constraint_types[i] {
            ConstraintType::Le => d <= tol,
            ConstraintType::Ge => d >= -tol,
            ConstraintType::Eq => d.abs() <= tol,
        };
        if !ok {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_milp(
        c: Vec<f64>,
        triplets: &[(usize, usize, f64)],
        m: usize,
        b: Vec<f64>,
        ctypes: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
        ints: Vec<usize>,
    ) -> MilpProblem {
        let n = c.len();
        let mut rows = Vec::with_capacity(triplets.len());
        let mut cols = Vec::with_capacity(triplets.len());
        let mut vals = Vec::with_capacity(triplets.len());
        for &(r, j, v) in triplets {
            rows.push(r);
            cols.push(j);
            vals.push(v);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();
        MilpProblem::new(lp, ints).unwrap()
    }

    /// Duplicate equality rows with bit-identical rhs collapse to one row.
    ///
    /// Sentinel: refusing the collapse whenever any rhs bits differ (the
    /// fail-open rule taken too far) leaves `rows_removed` at 0 → FAIL.
    #[test]
    fn duplicate_eq_rows_with_identical_rhs_collapse() {
        // Three-member rows keep the doubleton rule out of this fixture so
        // the counter measures the merge alone.
        let p = build_milp(
            vec![1.0, 0.0, 0.0],
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
            ],
            2,
            vec![3.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
            vec![0, 1, 2],
        );
        match reduce(&p, None) {
            ReduceOutcome::Reduced { info, reduced, .. } => {
                assert_eq!(info.rows_removed, 1, "one duplicate Eq must go");
                assert_eq!(reduced.lp.num_constraints, 1);
            }
            ReduceOutcome::Infeasible { .. } => panic!("identical rows are not infeasible"),
            ReduceOutcome::NotApplied { .. } => panic!("duplicate collapse must fire"),
        }
    }

    /// Two parallel Eq rows whose rhs differ by *less than the activity
    /// margin* are arithmetically ambiguous (parse noise vs a genuine
    /// near-contradiction): the merge keeps both rows (fail-open) instead of
    /// picking a winner or declaring infeasibility.
    ///
    /// Sentinel: the pre-fix merge dropped every Eq but the first here,
    /// turning this into `Reduced { rows_removed: 1, .. }` → FAIL.
    #[test]
    fn eq_merge_rhs_gap_inside_margin_is_fail_open() {
        let p = build_milp(
            vec![1.0, 1.0],
            &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)],
            2,
            vec![1.0, 1.0 + 1e-12],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(-100.0, 100.0), (-100.0, 100.0)],
            vec![0, 1],
        );
        match reduce(&p, None) {
            ReduceOutcome::NotApplied { .. } => {}
            ReduceOutcome::Infeasible { .. } => {
                panic!("sub-margin rhs gap must not declare infeasibility")
            }
            ReduceOutcome::Reduced { info, .. } => panic!(
                "sub-margin rhs gap must keep both Eq rows, got rows_removed={}",
                info.rows_removed
            ),
        }
    }

    /// The same pair with a rhs gap far beyond the margin is a genuine
    /// parallel-Eq contradiction and must be reported infeasible (sound
    /// through the bijection contract).
    ///
    /// Sentinel: relaxing the contradiction test to fail-open unconditionally
    /// returns NotApplied here → FAIL.
    #[test]
    fn duplicate_eq_rows_contradiction_is_infeasible() {
        let p = build_milp(
            vec![1.0, 1.0],
            &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)],
            2,
            vec![1.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(-100.0, 100.0), (-100.0, 100.0)],
            vec![0, 1],
        );
        assert!(
            matches!(reduce(&p, None), ReduceOutcome::Infeasible { .. }),
            "parallel Eq rows with rhs 1 and 3 are infeasible"
        );
    }

    /// Two rows that only become parallel after a shared column is pinned
    /// (with *different* coefficients there, so the raw patterns differ)
    /// must still merge on their live entries — the merge key filters dead
    /// columns itself rather than relying on pass order for the filter.
    #[test]
    fn merge_uses_live_entries_after_pinned_column() {
        // rows: r0 x0+x2+2·x1 ≤ 12, r1 x0+x2+3·x1 ≤ 13, r2 x1 = 5 (singleton).
        let p = build_milp(
            vec![1.0, 0.0, 1.0],
            &[
                (0, 0, 1.0),
                (0, 2, 1.0),
                (0, 1, 2.0),
                (1, 0, 1.0),
                (1, 2, 1.0),
                (1, 1, 3.0),
                (2, 1, 1.0),
            ],
            3,
            vec![12.0, 13.0, 5.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Eq],
            vec![(-10.0, 5.0), (0.0, 10.0), (-10.0, 5.0)],
            vec![0, 1, 2],
        );
        match reduce(&p, None) {
            ReduceOutcome::Reduced { info, reduced, .. } => {
                assert_eq!(info.cols_removed, 1, "x1 pinned by its singleton Eq");
                assert_eq!(
                    info.rows_removed, 2,
                    "singleton row + the merged weaker duplicate"
                );
                assert_eq!(reduced.lp.num_constraints, 1, "only x0 + x2 ≤ -2 survives");
            }
            other => {
                let kind = match other {
                    ReduceOutcome::Infeasible { .. } => "Infeasible",
                    ReduceOutcome::NotApplied { .. } => "NotApplied",
                    ReduceOutcome::Reduced { .. } => unreachable!("matched above"),
                };
                panic!("expected a reduction, got {kind}")
            }
        }
    }

    /// Doubleton substitution under the unimodular guard on an integer lock-1
    /// target: the eliminated column is reconstructed exactly on integer
    /// points and its objective contribution moves to offset + partner cost.
    ///
    /// Sentinel: dropping the unimodular guard or the objective transfer in
    /// `eliminate_doubleton` breaks the reconstruction or the offset → FAIL.
    #[test]
    fn doubleton_substitution_reconstructs_integer_point() {
        // min x0 + 2·x1  s.t.  x0 + x1 = 4;  x0 in [0,3] int, x1 in [0,5] int.
        let p = build_milp(
            vec![1.0, 2.0],
            &[(0, 0, 1.0), (0, 1, 1.0)],
            1,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 5.0)],
            vec![0, 1],
        );
        match reduce(&p, None) {
            ReduceOutcome::Reduced {
                reduced, map, info, ..
            } => {
                assert_eq!(info.substitutions, 1);
                assert_eq!(reduced.lp.num_vars, 1);
                assert!((reduced.lp.obj_offset - 4.0).abs() < 1e-9);
                let x = map.construct(&[1.0], 1e-6).expect("reconstruction");
                assert_eq!(x, vec![3.0, 1.0]);
                assert!(verify_original_solution(&p, &x, 1e-6));
            }
            _ => panic!("unimodular doubleton must fire"),
        }
    }

    /// Every column pinned ⇒ the reduced problem has zero variables and the
    /// postsolve map reconstructs the single feasible point from pins alone.
    ///
    /// Sentinel: skipping pin-column bookkeeping (or the empty-`kept`
    /// reconstruction) loses the pinned values → FAIL.
    #[test]
    fn fully_pinned_reduction_reconstructs_the_only_point() {
        // x1 = 5; x0 − x1 ≥ 0; x0 ≤ 5 ⇒ x0 = 5.
        let p = build_milp(
            vec![1.0, 1.0],
            &[(0, 1, 1.0), (1, 0, 1.0), (1, 1, -1.0), (2, 0, 1.0)],
            3,
            vec![5.0, 0.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Ge, ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
            vec![0, 1],
        );
        match reduce(&p, None) {
            ReduceOutcome::Reduced { reduced, map, .. } => {
                assert_eq!(reduced.lp.num_vars, 0);
                let x = map.construct(&[], 1e-6).expect("pinned reconstruction");
                assert_eq!(x, vec![5.0, 5.0]);
                assert!(verify_original_solution(&p, &x, 1e-6));
            }
            _ => panic!("pinning must empty the problem"),
        }
    }

    /// A column whose only finite side is one endpoint of an infinite
    /// interval is never "pinned": `ub − lb ≤ tol · scale` degenerates for
    /// infinite bounds (`inf ≤ tol · inf` is true in floating point), and
    /// treating it as pinned would pin a free column and cut off its
    /// unbounded ray (a real regression found by the unbounded sentinel
    /// during wiring).
    ///
    /// Sentinel: dropping the finite-side requirement in `is_pinned`
    /// reproduces `Reduced` (column pinned to 0) → FAIL.
    #[test]
    fn infinite_bound_column_is_never_pinned() {
        // max x via min -x, x integer in [0, +inf), no rows at all.
        let p = build_milp(
            vec![-1.0],
            &[],
            0,
            vec![],
            vec![],
            vec![(0.0, f64::INFINITY)],
            vec![0],
        );
        match reduce(&p, None) {
            ReduceOutcome::NotApplied { .. } => {}
            ReduceOutcome::Reduced { info, .. } => panic!(
                "an infinitely bounded column must not be pinned, got cols_removed={}",
                info.cols_removed
            ),
            ReduceOutcome::Infeasible { .. } => panic!("an unbounded box is feasible"),
        }
    }

    /// Solve with default options/config (root presolve ON) for the
    /// end-to-end review sentinels below.
    fn solve_default(p: &MilpProblem) -> SolverResult {
        let (res, _stats) =
            crate::mip::solve_milp_with_stats(p, &SolverOptions::default(), &MipConfig::default());
        res
    }

    /// An integer interval holding two integers is never pinned: removing
    /// the column must be a bijection on integer points, and `[1e9,
    /// 1e9 + 1]` has two of them.
    ///
    /// Pre-fix, the width-vs-scale pin test (`1e-9 · (1 + |b|) ≈ 1.0` at
    /// this magnitude) pinned the column to 1e9 and published Optimal
    /// obj −1e15 — the true optimum is −1e6·(1e9+1), a loss of exactly
    /// 1e6 (PR #89 finding F1).
    ///
    /// Sentinel: restoring width-vs-scale integer pinning reproduces the
    /// pinned point and the 1e6 objective loss → FAIL.
    #[test]
    fn wide_integer_interval_at_large_magnitude_is_not_pinned() {
        // min −1e6·x, x integer in [1e9, 1e9+1], no rows.
        let p = build_milp(
            vec![-1e6],
            &[],
            0,
            vec![],
            vec![],
            vec![(1e9, 1e9 + 1.0)],
            vec![0],
        );
        assert!(
            matches!(reduce(&p, None), ReduceOutcome::NotApplied { .. }),
            "an integer interval with two integers must not reduce"
        );
        let res = solve_default(&p);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert_eq!(res.objective, -1e6 * (1e9_f64 + 1.0));
        assert_eq!(res.solution, vec![1e9 + 1.0]);

        // Control: a single-integer interval (sub-ulp parse slack) must
        // still pin — the fix is uniqueness, not "never pin".
        let q = build_milp(
            vec![-1e6],
            &[],
            0,
            vec![],
            vec![],
            vec![(1e9, 1e9 + 1e-9)],
            vec![0],
        );
        match reduce(&q, None) {
            ReduceOutcome::Reduced { reduced, .. } => assert_eq!(reduced.lp.num_vars, 0),
            _ => panic!("a single-integer interval must pin"),
        }
        let res = solve_default(&q);
        assert_eq!(res.objective, -1e15);
        assert_eq!(res.solution, vec![1e9]);
    }

    /// A row that is merely redundant at 1e9 scale must not force its
    /// members to endpoints.
    ///
    /// Pre-fix, forcing tested `min_act ≥ rhs − tol` with the scale-
    /// inflated margin (`tol ≈ 4.000000009` for this row), so a real
    /// slack of 4 — ten orders above the row's fp accumulation error
    /// (ulp(2e9) ≈ 2.4e-7) — read as forcing: both columns pinned to 1e9
    /// and Optimal obj −2e15 published, a 4e6 objective loss (PR #89
    /// finding F2). Forcing now compares the real slack against the
    /// absolute activity tolerance only.
    ///
    /// Sentinel: restoring the scale-relative forcing margin reproduces
    /// the pins and the 4e6 loss → FAIL.
    #[test]
    fn redundant_large_magnitude_row_does_not_force_endpoints() {
        // min −1e6·(x+y) s.t. x+y ≤ rhs; x, y integer in [1e9, 1e9+2].
        // max activity is 2e9+4: any rhs ≥ that is redundant, never forcing.
        let mk = |rhs: f64| {
            build_milp(
                vec![-1e6, -1e6],
                &[(0, 0, 1.0), (0, 1, 1.0)],
                1,
                vec![rhs],
                vec![ConstraintType::Le],
                vec![(1e9, 1e9 + 2.0), (1e9, 1e9 + 2.0)],
                vec![0, 1],
            )
        };
        for rhs in [2e9 + 4.0, 2e9 + 8.0] {
            let p = mk(rhs);
            match reduce(&p, None) {
                ReduceOutcome::Reduced { map, .. } => {
                    assert!(
                        map.pinned.iter().all(Option::is_none),
                        "no member of a redundant row may be pinned (rhs = {rhs})"
                    );
                }
                ReduceOutcome::NotApplied { .. } => {}
                ReduceOutcome::Infeasible { .. } => panic!("the row is redundant, not infeasible"),
            }
            let res = solve_default(&p);
            assert_eq!(res.status, SolveStatus::Optimal);
            assert_eq!(res.objective, -2e6 * (1e9_f64 + 2.0));
            assert_eq!(res.solution, vec![1e9 + 2.0, 1e9 + 2.0]);
        }
    }

    /// An equality row pins members to the extreme whose activity EQUALS
    /// the rhs (within the absolute forcing tolerance), and a one-side-
    /// redundant equality weakens to the BINDING side.
    ///
    /// Pre-fix, an Eq row whose activity window sat within the scaled
    /// `tol` of the rhs on both sides pinned every member to the side
    /// picked by `max_act − rhs ≤ rhs − min_act` — at large magnitude
    /// that is the extreme whose activity misses the rhs by up to 2·tol
    /// (window [2e9, 2e9+4] with rhs 2e9+4 pinned the MIN endpoints, an
    /// equality violation of 4 and a 4e6 objective loss). The one-side
    /// weakening also kept the side it had just proven redundant,
    /// dropping the binding constraint (usually rescued by the
    /// verification fallback, so it degraded to a re-run rather than a
    /// wrong answer). The rewrite is a regression guard: propagation
    /// resolves these rows first, which is why no wrong-published
    /// counterexample exists for this branch today.
    ///
    /// Sentinel: restoring the old side selection / weakening changes
    /// the reduced structure or the published point → FAIL.
    #[test]
    fn equality_row_forces_only_the_side_equal_to_rhs() {
        // x + y = 10, x in [0,8] int, y in [0,2] int, z = 3: the max
        // activity equals the rhs, so the only feasible point is (8, 2).
        let p = build_milp(
            vec![1.0, 0.0, 0.0],
            &[(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0)],
            2,
            vec![10.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 8.0), (0.0, 2.0), (0.0, 5.0)],
            vec![0, 1, 2],
        );
        let res = solve_default(&p);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert_eq!(res.objective, 8.0);
        assert_eq!(res.solution, vec![8.0, 2.0, 3.0]);

        let mk = |rhs: f64, c: f64| {
            build_milp(
                vec![c, c, 0.0],
                &[(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0)],
                2,
                vec![rhs, 3.0],
                vec![ConstraintType::Eq, ConstraintType::Eq],
                vec![(1e9, 1e9 + 2.0), (1e9, 1e9 + 2.0), (0.0, 5.0)],
                vec![0, 1, 2],
            )
        };
        // Upper extreme reaches 2e9+4 exactly: x = y = 1e9+2 is forced.
        let res = solve_default(&mk(2e9 + 4.0, -1e6));
        assert_eq!(res.status, SolveStatus::Optimal);
        assert_eq!(res.objective, -2e6 * (1e9_f64 + 2.0));
        assert_eq!(res.solution, vec![1e9 + 2.0, 1e9 + 2.0, 3.0]);
        // Lower extreme reaches 2e9 exactly: x = y = 1e9 is forced.
        let res = solve_default(&mk(2e9, 1e6));
        assert_eq!(res.status, SolveStatus::Optimal);
        assert_eq!(res.objective, 2e6 * 1e9);
        assert_eq!(res.solution, vec![1e9, 1e9, 3.0]);
    }

    /// `verify_original_solution` rejects a wrong point (row violation) —
    /// the independent check the fail-closed fallback relies on.
    ///
    /// Sentinel: any tolerance slop that admits O(1) row violations → FAIL.
    #[test]
    fn verify_original_solution_rejects_row_violation() {
        let p = build_milp(
            vec![1.0, 1.0],
            &[(0, 0, 1.0), (0, 1, 1.0)],
            1,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 5.0)],
            vec![0, 1],
        );
        assert!(verify_original_solution(&p, &[3.0, 1.0], 1e-6));
        assert!(!verify_original_solution(&p, &[3.0, 2.0], 1e-6));
        assert!(
            !verify_original_solution(&p, &[3.5, 0.5], 1e-6),
            "fractional point on integer columns must be rejected"
        );
    }
}
