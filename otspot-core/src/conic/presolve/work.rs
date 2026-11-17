//! Sparse reduction workspace for the tier-1 conic presolve.
//!
//! Holds the mutable row/column form of `A` and `G` plus the variable-bound
//! layer, and applies the three tier-1 reductions:
//!
//! * **domain identity absorption** — a singleton orthant row `g·x_j + s = h`
//!   is an encoded variable bound (`x_j >= h/g` for `g < 0`, `<=` otherwise;
//!   the CBF `VAR L+`/`L-`/`L=` encoding); it is lifted into the bound layer.
//!   Bounds on variables that survive re-enter the reduced problem as rows
//!   (the IPM only consumes rows at this tier — native bound handling is the
//!   tier-2 formulation work the research report excludes); bounds on an
//!   eliminated variable are carried through the substitution as
//!   *transformed* orthant rows (pivot path) or dropped with dual recovery
//!   (fixed-variable path, where the bound is satisfied at the fixed value).
//! * **fixed-variable elimination** — `lb == ub` substitutes the constant
//!   into every surviving row (`b`, `h`) and removes the column.
//! * **equality substitution** — a pivot `a_ij` of an equality row eliminates
//!   that row and column `j`, distributing the pivot row into the other `A`
//!   rows, *all* `G` rows, `b`/`h` and `c`.
//!
//! The LIFO postsolve stack recorded here is replayed by [`super::postsolve`]
//! to map primal/dual/ray/Farkas certificates back to the original space.
//! The design mirrors the LP presolve (`crate::presolve::transforms`), from
//! which the Markowitz-style fill guard, the lazy stale-entry handling and
//! the dual-recovery column snapshots are adapted to the conic row layout.

use super::postsolve::{AbsorbedSnapshot, ConicPostsolveStep};
use super::postsolve::{BoundLayer, InsertOutcome};
use super::{REDUCED_DIM_MIN_COLS, REDUCED_DIM_MIN_ROWS};
use crate::conic::ConicProblem;
use crate::tolerances::ZERO_TOL;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Max magnitude of a substitution multiplier `coef / pivot`, of the pivot
/// value `rhs / pivot`, or of a fixed-variable contribution `|a·val|`. Direct
/// port of the LP contract (`steps_basic.rs:26-37`, research report contract
/// (c)): beyond this the coefficient amplification the original space would
/// see outruns what the eps-tighten bound compensates, so the substitution is
/// skipped and the IPM handles the variable instead.
const SUBST_ABS_GUARD: f64 = 1e5;

/// Max number of active nonzeros in a pivot row considered for equality
/// substitution. Bounds the `O(col_count * row_len)` distribution work per
/// elimination; rows above this are left to the IPM.
const PIVOT_ROW_NNZ_CAP: usize = 128;

/// Markowitz fill budget: an equality substitution is accepted when its
/// Markowitz fill `(row_len-1) * col_count` stays within
/// `FILL_RATIO_BUDGET` times the nonzeros it removes. The LP simplex guard
/// uses a ratio of 1 (sparsity of one factorization dominates there); for the
/// conic route the dominant cost is the KKT dimension `N` (dense-equivalent
/// memory `~8N^2`, research report §2 measured), so paying some fill to shrink
/// `N` is profitable. [`MAX_NNZ_GROWTH`] caps the total blow-up this looser
/// budget could otherwise cascade into.
const FILL_RATIO_BUDGET: usize = 8;

/// Stop substituting once the workspace nonzero appends reach
/// `MAX_NNZ_GROWTH` times the initial nnz. Substitutions are an
/// equivalence-preserving partial state, so stopping early is always safe —
/// it only gives up reduction depth, never correctness.
const MAX_NNZ_GROWTH: f64 = 4.0;

/// Merges a stale-tolerant sparse vector `(index, value)` into canonical form:
/// one entry per index with the summed value, structural zeros dropped. All
/// workspace reads go through this so append-only updates behave like proper
/// coefficient additions.
fn merged(entries: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut sorted: Vec<(usize, f64)> = entries.to_vec();
    sorted.sort_unstable_by_key(|&(i, _)| i);
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(sorted.len());
    for (i, v) in sorted {
        match out.last_mut() {
            Some((pi, pv)) if *pi == i => *pv += v,
            _ => out.push((i, v)),
        }
    }
    out.retain(|(_, v)| v.abs() > ZERO_TOL && v.is_finite());
    out
}

/// One `G` row: `entries·x + s_k = rhs` with `s_k` in the cone component
/// indicated by `soc_block`.
struct GRow {
    entries: Vec<(usize, f64)>,
    rhs: f64,
    /// `None` = orthant row (absorbable, reorderable within the orthant
    /// block); `Some(b)` = row of SOC block `b` (frozen contiguous group —
    /// tier-1 never removes or reorders SOC rows).
    soc_block: Option<usize>,
    /// Original row id in the input problem (`None` for rows materialized
    /// from input `var_bounds`, which have no original row to map back to).
    orig: Option<usize>,
    /// Original row inf-norm (eps-tighten growth tracker basis).
    orig_norm: f64,
}

/// Row of the equality matrix `A` (every row is an equality).
struct ARow {
    entries: Vec<(usize, f64)>,
    rhs: f64,
    orig_norm: f64,
}

/// Mutable reduction workspace; workspace indices are internal, `orig`
/// fields carry original row identity through transformations so the
/// postsolve maps are exact.
pub(super) struct Workspace {
    pub(super) n: usize,
    c: Vec<f64>,
    a_rows: Vec<ARow>,
    a_removed: Vec<bool>,
    /// Column index into `A`: `a_cols[j] = [(a_row, value)]` (append-only).
    a_cols: Vec<Vec<(usize, f64)>>,
    g_rows: Vec<GRow>,
    g_removed: Vec<bool>,
    g_cols: Vec<Vec<(usize, f64)>>,
    removed_cols: Vec<bool>,
    /// Bound layer: absorbed domain rows per column (duals recovered at the
    /// column's elimination, see `Eliminated` in `super::postsolve`).
    bounds: Vec<BoundLayer>,
    pub(super) stack: Vec<ConicPostsolveStep>,
    nnz_initial: usize,
    nnz_appended: usize,
    /// Max row-norm growth ratio over transformed rows (eps-tighten input).
    pub(super) growth: f64,
}

impl Workspace {
    /// Builds the workspace from the (already materialized) problem. `g_soc`
    /// and `g_orig` are per-`G`-row cone-block / original-row annotations of
    /// the *materialized* layout produced by `super::materialize_bounds`.
    pub(super) fn build(
        problem: &ConicProblem,
        g_soc: &[Option<usize>],
        g_orig: &[Option<usize>],
    ) -> Workspace {
        let n = problem.n();
        let p = problem.p();
        let m = problem.h.len();
        let mut a_rows: Vec<ARow> = (0..p)
            .map(|_| ARow {
                entries: Vec::new(),
                rhs: 0.0,
                orig_norm: 0.0,
            })
            .collect();
        let mut a_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in 0..n {
            for k in problem.a.col_ptr()[col]..problem.a.col_ptr()[col + 1] {
                let row = problem.a.row_ind()[k];
                let val = problem.a.values()[k];
                a_rows[row].entries.push((col, val));
                a_rows[row].orig_norm = a_rows[row].orig_norm.max(val.abs());
                a_cols[col].push((row, val));
            }
        }
        for i in 0..p {
            a_rows[i].rhs = problem.b[i];
        }
        let mut g_rows: Vec<GRow> = (0..m)
            .map(|_| GRow {
                entries: Vec::new(),
                rhs: 0.0,
                soc_block: None,
                orig: None,
                orig_norm: 0.0,
            })
            .collect();
        let mut g_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in 0..n {
            for k in problem.g.col_ptr()[col]..problem.g.col_ptr()[col + 1] {
                let row = problem.g.row_ind()[k];
                let val = problem.g.values()[k];
                g_rows[row].entries.push((col, val));
                g_rows[row].orig_norm = g_rows[row].orig_norm.max(val.abs());
                g_cols[col].push((row, val));
            }
        }
        for k in 0..m {
            g_rows[k].rhs = problem.h[k];
            g_rows[k].soc_block = g_soc[k];
            g_rows[k].orig = g_orig[k];
        }
        Workspace {
            n,
            c: problem.c.clone(),
            a_rows,
            a_removed: vec![false; p],
            a_cols,
            g_rows,
            g_removed: vec![false; m],
            g_cols,
            removed_cols: vec![false; n],
            bounds: vec![BoundLayer::default(); n],
            stack: Vec::new(),
            nnz_initial: problem.a.nnz() + problem.g.nnz(),
            nnz_appended: 0,
            growth: 1.0,
        }
    }

    /// Active merged entries of `A` row `i` (removed columns dropped).
    fn a_active(&self, i: usize) -> Vec<(usize, f64)> {
        merged(&self.a_rows[i].entries)
            .into_iter()
            .filter(|(j, _)| !self.removed_cols[*j])
            .collect()
    }

    /// Active merged entries of `G` row `k`.
    fn g_active(&self, k: usize) -> Vec<(usize, f64)> {
        merged(&self.g_rows[k].entries)
            .into_iter()
            .filter(|(j, _)| !self.removed_cols[*j])
            .collect()
    }

    /// Active merged `A` entries of column `j` (removed rows dropped).
    fn a_col(&self, j: usize) -> Vec<(usize, f64)> {
        merged(&self.a_cols[j])
            .into_iter()
            .filter(|(i, _)| !self.a_removed[*i])
            .collect()
    }

    /// Active merged `G` entries of column `j`.
    fn g_col(&self, j: usize) -> Vec<(usize, f64)> {
        merged(&self.g_cols[j])
            .into_iter()
            .filter(|(k, _)| !self.g_removed[*k])
            .collect()
    }

    /// Appends an update to `A` (lazy merge on read).
    fn add_to_a(&mut self, i: usize, j: usize, delta: f64) {
        if delta == 0.0 {
            return;
        }
        self.a_rows[i].entries.push((j, delta));
        self.a_cols[j].push((i, delta));
        self.nnz_appended += 1;
    }

    /// Appends an update to `G`.
    fn add_to_g(&mut self, k: usize, j: usize, delta: f64) {
        if delta == 0.0 {
            return;
        }
        self.g_rows[k].entries.push((j, delta));
        self.g_cols[j].push((k, delta));
        self.nnz_appended += 1;
    }

    /// Row-norm growth bookkeeping for the eps-tighten sigma.
    fn note_growth(&mut self, norm: f64, orig_norm: f64) {
        if orig_norm > 0.0 && norm > self.growth * orig_norm {
            self.growth = norm / orig_norm;
        }
    }

    /// Recomputes growth trackers for the rows touched during one
    /// elimination (called on the merged, post-update row data).
    fn track_touched_growth(&mut self, a_rows_touched: &[usize], g_rows_touched: &[usize]) {
        for &i in a_rows_touched {
            let norm = merged(&self.a_rows[i as usize].entries)
                .into_iter()
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);
            self.note_growth(norm, self.a_rows[i as usize].orig_norm);
        }
        for &k in g_rows_touched {
            let norm = merged(&self.g_rows[k].entries)
                .into_iter()
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);
            self.note_growth(norm, self.g_rows[k].orig_norm);
        }
    }

    fn n_active_cols(&self) -> usize {
        self.n - self.removed_cols.iter().filter(|&&r| r).count()
    }

    /// Remove equality rows that reduced to `0 = 0` (recorded as
    /// `RedundantEqRow`, dual `0` — the row's multiplier is a free coordinate
    /// in the original KKT system once the row holds identically, and leaving
    /// an all-zero row in the reduced `A` would singularize the KKT
    /// factorization). A row reduced to `0 = rhs` with `rhs != 0`, or a `G`
    /// row reduced to `s_k = rhs` with `rhs < 0`, means the *original*
    /// problem is infeasible — but its original-space Farkas pair would need
    /// the substitution chain's dual replay (not just this row), so the tier
    /// declines the reduction instead of shipping an unprovable certificate:
    /// the caller then solves the original data, which the IPM's own
    /// infeasibility machinery handles exactly as before presolve existed.
    /// Returns `(removed, decline)`.
    pub(super) fn simplify_zero_rows(&mut self) -> (usize, bool) {
        for i in 0..self.a_rows.len() {
            if self.a_removed[i] || !self.a_active(i).is_empty() {
                continue;
            }
            let rhs = self.a_rows[i].rhs;
            if rhs.abs() > ZERO_TOL * (1.0 + rhs.abs()) {
                return (0, true);
            }
            self.a_removed[i] = true;
            self.stack
                .push(ConicPostsolveStep::RedundantEqRow { orig_row: i });
            return (1, false);
        }
        for k in 0..self.g_rows.len() {
            if self.g_removed[k] || !self.g_active(k).is_empty() {
                continue;
            }
            let rhs = self.g_rows[k].rhs;
            if rhs < -ZERO_TOL * (1.0 + rhs.abs()) {
                return (0, true);
            }
        }
        (0, false)
    }

    // Finalize-side accessors (row data stays private to this module).
    pub(super) fn a_active_pub(&self, i: usize) -> Vec<(usize, f64)> {
        self.a_active(i)
    }
    pub(super) fn a_row_count(&self) -> usize {
        self.a_rows.len()
    }
    pub(super) fn a_row_removed(&self, i: usize) -> bool {
        self.a_removed[i]
    }
    pub(super) fn g_row_count(&self) -> usize {
        self.g_rows.len()
    }
    pub(super) fn g_row_removed(&self, k: usize) -> bool {
        self.g_removed[k]
    }
    pub(super) fn g_active_pub(&self, k: usize) -> Vec<(usize, f64)> {
        self.g_active(k)
    }
    pub(super) fn a_row_rhs(&self, i: usize) -> f64 {
        self.a_rows[i].rhs
    }
    pub(super) fn g_row_rhs(&self, k: usize) -> f64 {
        self.g_rows[k].rhs
    }
    pub(super) fn g_row_orig(&self, k: usize) -> Option<usize> {
        self.g_rows[k].orig
    }
    pub(super) fn g_row_soc(&self, k: usize) -> Option<usize> {
        self.g_rows[k].soc_block
    }
    pub(super) fn soc_block_dims(&self) -> Vec<usize> {
        let mut dims: Vec<usize> = Vec::new();
        for row in &self.g_rows {
            if let Some(bi) = row.soc_block {
                if dims.len() <= bi {
                    dims.resize(bi + 1, 0);
                }
                dims[bi] += 1;
            }
        }
        dims
    }
    pub(super) fn col_removed(&self, j: usize) -> bool {
        self.removed_cols[j]
    }
    /// Bounds still held for live columns (finalize re-materializes them as
    /// orthant rows — the tier-1 IPM consumes rows only).
    pub(super) fn active_bound_layers(&self) -> Vec<(usize, AbsorbedSnapshot)> {
        let mut out = Vec::new();
        for j in 0..self.n {
            if self.removed_cols[j] {
                continue;
            }
            for snap in self.bounds[j].peek() {
                out.push((j, snap));
            }
        }
        out
    }
    pub(super) fn c_of(&self, j: usize) -> f64 {
        self.c[j]
    }

    fn nnz_budget_exceeded(&self) -> bool {
        self.nnz_initial + self.nnz_appended > (self.nnz_initial as f64 * MAX_NNZ_GROWTH) as usize
    }
}

/// Workspace-side reduction passes (see module docs).
impl Workspace {
    /// One sweep of domain identity absorption. Returns `(conflict,
    /// changed)`: `Some(..)` when the absorbed bounds of some column prove
    /// `lb > ub` (Farkas-ready via the two witness bound rows); `changed`
    /// counts rows absorbed or dropped during the sweep.
    pub(super) fn absorb_sweep(&mut self) -> (Option<super::InfeasProof>, usize) {
        let mut conflict = None;
        let mut changed = 0usize;
        for k in 0..self.g_rows.len() {
            if self.g_removed[k] || self.g_rows[k].soc_block.is_some() {
                continue;
            }
            let active = self.g_active(k);
            if active.len() != 1 {
                continue;
            }
            let (j, g) = active[0];
            let h = self.g_rows[k].rhs;
            if !g.is_finite() || !h.is_finite() || g.abs() <= ZERO_TOL {
                continue;
            }
            let val = h / g;
            let is_lower = g < 0.0;
            let snap = AbsorbedSnapshot {
                g_row: k,
                orig_row: self.g_rows[k].orig,
                g_coef: g,
                rhs: h,
                is_lower,
            };
            match self.bounds[j].insert(snap, val, is_lower) {
                InsertOutcome::Conflict(opposite) => {
                    self.g_removed[k] = true;
                    let _ = j;
                    conflict = Some(super::InfeasProof::BoundPair {
                        witness: [snap, opposite],
                    });
                    break;
                }
                InsertOutcome::Absorbed { superseded } => {
                    self.g_removed[k] = true;
                    if let Some(old) = superseded {
                        if let Some(orig) = old.orig_row {
                            self.stack
                                .push(ConicPostsolveStep::DroppedBound { orig_row: orig });
                        }
                    }
                    changed += 1;
                }
                InsertOutcome::RedundantSelf => {
                    // The bound was already tighter (or equal): this row is
                    // implied by the surviving one and drops with dual `0` —
                    // `z = 0` on an implied row keeps every KKT/certificate
                    // equation the original problem has to satisfy.
                    self.g_removed[k] = true;
                    if let Some(orig) = snap.orig_row {
                        self.stack
                            .push(ConicPostsolveStep::DroppedBound { orig_row: orig });
                    }
                    changed += 1;
                }
            }
        }
        (conflict, changed)
    }

    /// One sweep of fixed-variable elimination (`lb == ub` in the bound
    /// layer), honoring [`SUBST_ABS_GUARD`]; returns columns eliminated.
    pub(super) fn fix_sweep(&mut self) -> usize {
        let mut fixed = 0usize;
        for j in 0..self.n {
            if self.removed_cols[j] || self.n_active_cols() <= REDUCED_DIM_MIN_COLS {
                continue;
            }
            let Some(val) = self.bounds[j].fixed_value() else {
                continue;
            };
            let guard_ok = self
                .a_col(j)
                .iter()
                .all(|&(_, a)| (a * val).abs() <= SUBST_ABS_GUARD)
                && self
                    .g_col(j)
                    .iter()
                    .all(|&(_, g)| (g * val).abs() <= SUBST_ABS_GUARD);
            if !guard_ok {
                continue;
            }
            self.eliminate_col(j, None, val);
            fixed += 1;
        }
        fixed
    }

    /// Contract (c) guard for a pivot: every multiplier `coef / pivot` and
    /// the eliminated value `rhs / pivot` must stay within `SUBST_ABS_GUARD`.
    fn pivot_guards_fail(&self, i: usize, j: usize, pivot: f64, rhs: f64) -> bool {
        if (rhs / pivot).abs() > SUBST_ABS_GUARD {
            return true;
        }
        (self.a_col(j).into_iter().chain(self.g_col(j))).any(|(r, v)| {
            if r == i {
                false
            } else {
                (v / pivot).abs() > SUBST_ABS_GUARD
            }
        })
    }

    /// Best (lowest Markowitz fill) admissible pivot in `A` row `i`.
    fn best_pivot_in_row(&self, i: usize) -> Option<(usize, f64, usize)> {
        let row = self.a_active(i);
        if row.is_empty() || row.len() > PIVOT_ROW_NNZ_CAP {
            return None;
        }
        let rhs = self.a_rows[i].rhs;
        let mut best: Option<(usize, f64, usize)> = None;
        for &(j, pivot) in &row {
            if pivot.abs() <= ZERO_TOL || self.removed_cols[j] {
                continue;
            }
            let a_other = self.a_col(j).into_iter().filter(|(r, _)| *r != i).count();
            let g_count = self.g_col(j).len();
            let fill = (row.len() - 1) * (a_other + g_count);
            let removed_nnz = 1 + (row.len() - 1) + a_other + g_count;
            if fill > FILL_RATIO_BUDGET * removed_nnz.max(1) {
                continue;
            }
            if self.pivot_guards_fail(i, j, pivot, rhs) {
                continue;
            }
            if best.is_none_or(|(_, _, f)| fill < f) {
                best = Some((j, pivot, fill));
            }
        }
        best
    }

    /// Equality-substitution sweep driven by a lazy Markowitz priority queue:
    /// each `A` row carries its cheapest admissible pivot; rows touched by an
    /// elimination are version-bumped and recomputed when they surface again.
    /// One pass per queue round (absorption/fix passes re-run between full
    /// rounds at the driver level).
    pub(super) fn substitute_sweep(&mut self) -> usize {
        if self.n_active_cols() <= REDUCED_DIM_MIN_COLS || self.a_rows.is_empty() {
            return 0;
        }
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        for i in 0..self.a_rows.len() {
            if let Some((_, _, fill)) = self.best_pivot_in_row(i) {
                heap.push(Reverse((fill, i)));
            }
        }
        let mut eliminated = 0;
        // Each accepted elimination removes exactly one active column, so the
        // loop terminates after at most `n` pops that pass every guard;
        // stale queue entries are simply recomputed and rejected.
        while let Some(Reverse((_, i))) = heap.pop() {
            if self.a_removed[i] {
                continue;
            }
            let Some((j, pivot, _)) = self.best_pivot_in_row(i) else {
                continue;
            };
            if self.n_active_cols() <= REDUCED_DIM_MIN_COLS
                || self.active_a_rows() <= REDUCED_DIM_MIN_ROWS
                || self.nnz_budget_exceeded()
            {
                break;
            }
            let rhs = self.a_rows[i].rhs;
            let touched_a: Vec<usize> = self
                .a_col(j)
                .into_iter()
                .map(|(r, _)| r)
                .filter(|&r| r != i)
                .collect();
            self.eliminate_col(j, Some((i, pivot, rhs)), 0.0);
            for r in touched_a {
                if let Some((_, _, fill)) = self.best_pivot_in_row(r) {
                    heap.push(Reverse((fill, r)));
                }
            }
            eliminated += 1;
        }
        eliminated
    }

    fn active_a_rows(&self) -> usize {
        self.a_rows.len() - self.a_removed.iter().filter(|&&r| r).count()
    }

    /// Core elimination of column `j`:
    /// * pivot path: `via = Some((piv_row, pivot, rhs))` — also removes the
    ///   equality row, distributes it into every other `A` row and every
    ///   active `G` row, folds `c_j` into the objective;
    /// * fixed path: `via = None`, `fixed = val` — pure constant substitution.
    ///
    /// Absorbed bounds of `j` follow the path: the pivot path re-materializes
    /// each bound row as a transformed active orthant row (slack coefficient
    /// stays `+1` for either pivot sign, dual rides along unchanged via the
    /// preserved original row id); the fixed path drops them for dual recovery
    /// at replay (`Eliminated` step).
    fn eliminate_col(&mut self, j: usize, via: Option<(usize, f64, f64)>, fixed: f64) {
        let (piv_row, pivot, rhs) = match via {
            Some(t) => (Some(t.0), t.1, t.2),
            None => (None, 1.0, fixed),
        };
        let others: Vec<(usize, f64)> = match piv_row {
            Some(r) => self
                .a_active(r)
                .into_iter()
                .filter(|(c, _)| *c != j)
                .collect(),
            None => Vec::new(),
        };
        let col_a: Vec<(usize, f64)> = self
            .a_col(j)
            .into_iter()
            .filter(|(r, _)| Some(*r) != piv_row)
            .collect();
        let col_g: Vec<(usize, f64)> = self.g_col(j);
        let c_orig = self.c[j];

        for (r, a_rj) in &col_a {
            self.a_rows[*r].rhs -= a_rj * (rhs / pivot);
            for &(k_col, a_pk) in &others {
                let delta = -a_rj * a_pk / pivot;
                self.add_to_a(*r, k_col, delta);
            }
            self.add_to_a(*r, j, -a_rj);
        }
        for (k, g_kj) in &col_g {
            self.g_rows[*k].rhs -= g_kj * (rhs / pivot);
            for &(k_col, a_pk) in &others {
                let delta = -g_kj * a_pk / pivot;
                self.add_to_g(*k, k_col, delta);
            }
            self.add_to_g(*k, j, -g_kj);
        }
        if c_orig != 0.0 {
            for &(k_col, a_pk) in &others {
                self.c[k_col] -= c_orig * a_pk / pivot;
            }
            self.c[j] = 0.0;
        }
        self.track_touched_growth(
            &col_a.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
            &col_g.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        );

        // Bounds of the eliminated column.
        let bound_rows: Vec<AbsorbedSnapshot> = self.bounds[j].take();
        if piv_row.is_some() {
            for snap in &bound_rows {
                // Bound `g·x_j + s = h` with `x_j = (rhs − Σ a·x)/pivot`
                // becomes `(−g·a_pk/pivot)·x_k + s = h − g·rhs/pivot`.
                let mut entries: Vec<(usize, f64)> = others
                    .iter()
                    .map(|&(col, a_pk)| (col, -snap.g_coef * a_pk / pivot))
                    .collect();
                entries.retain(|(_, v)| v.abs() > ZERO_TOL);
                if entries.is_empty() {
                    // Constant-true/false row after substitution: `s = h'`
                    // with `h' >= 0` here (a violated bound would have been
                    // caught as a fixed-value conflict). Keep it as an
                    // active zero-entry orthant row: it still carries its
                    // original identity and dual.
                    self.g_rows[snap.g_row].entries.clear();
                    self.g_rows[snap.g_row].rhs = snap.rhs - snap.g_coef * rhs / pivot;
                } else {
                    self.g_rows[snap.g_row].entries = entries;
                    self.g_rows[snap.g_row].rhs = snap.rhs - snap.g_coef * rhs / pivot;
                }
                self.g_removed[snap.g_row] = false;
                for (col, v) in self.g_rows[snap.g_row].entries.clone() {
                    self.g_cols[col].push((snap.g_row, v));
                    self.nnz_appended += 1;
                }
            }
        }

        if let Some(p) = piv_row {
            self.a_removed[p] = true;
        }
        self.removed_cols[j] = true;
        self.stack.push(ConicPostsolveStep::Eliminated {
            col: j,
            piv_row,
            pivot,
            rhs,
            others,
            col_a,
            col_g,
            c_orig,
            bounds: bound_rows,
        });
    }
}
