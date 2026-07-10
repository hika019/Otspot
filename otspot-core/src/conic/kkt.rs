//! Sparse augmented KKT system for the conic interior-point method: the
//! Newton system per predictor/corrector step, unknowns `(dx, dy, dz)`:
//!
//! ```text
//! [ delta_p I    A^T        G^T   ] [dx]   [ -rx          ]
//! [   A        -delta_d I    0    ] [dy] = [ -ry          ]
//! [   G           0        -W^2   ] [dz]   [ -rz - W*rc   ]
//! ```
//!
//! Regularization signs (`+delta_p`/`-delta_d`) follow
//! `qp::ipm_core::kkt::build_augmented_system`. Quasidefinite system with
//! `Q = 0` (QCQP quadratics bridge into SOC blocks upstream) and `Sigma`
//! generalised to the block-diagonal NT operator `W^2`, left fully
//! unregularized (see `KktSkeleton::materialize` for why). SOCs at or above
//! `cone::SOC_BORDER_MIN_DIM` replace `W^2` with an `O(d)` rank-1
//! border (`cone::visit_border_pattern`); see [`build_rhs`] for the RHS
//! derivation.

use super::cone::{self, Blocks, Scaling};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{
    factorize_kkt_pre_permuted_cached_par, factorize_kkt_with_cached_perm_par, KktConfig,
    KktFactor, KktSolver, PreconditionedMinres,
};
use crate::sparse::CscMatrix;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Sparse mat-vec helpers (no A/G densification).
// ---------------------------------------------------------------------------

/// `A x` (length `a.nrows()`).
pub(super) fn spmv(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.nrows()];
    for j in 0..a.ncols() {
        let xj = x[j];
        if xj != 0.0 {
            for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
                out[a.row_ind()[k]] += a.values()[k] * xj;
            }
        }
    }
    out
}

/// `A^T y` (length `a.ncols()`).
pub(super) fn spmtv(a: &CscMatrix, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.ncols()];
    for (j, out_j) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
            s += a.values()[k] * y[a.row_ind()[k]];
        }
        *out_j = s;
    }
    out
}

/// `|A| |x|` (elementwise-abs mat-vec), length `a.nrows()`. Used by the
/// dual-infeasibility (improving ray) certificate's scale-invariant residual.
pub(super) fn spmv_abs(a: &CscMatrix, x_abs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.nrows()];
    for j in 0..a.ncols() {
        let xj = x_abs[j];
        if xj != 0.0 {
            for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
                out[a.row_ind()[k]] += a.values()[k].abs() * xj;
            }
        }
    }
    out
}

/// Accumulates `|A|^T |y|` into `acc` (length `a.ncols()`). Used by the
/// primal-infeasibility (Farkas) certificate's scale-invariant residual,
/// which sums the contribution from both `A` and `G`.
pub(super) fn spmtv_abs_accum(a: &CscMatrix, y_abs: &[f64], acc: &mut [f64]) {
    for (j, acc_j) in acc.iter_mut().enumerate() {
        let mut s = 0.0;
        for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
            s += a.values()[k].abs() * y_abs[a.row_ind()[k]];
        }
        *acc_j += s;
    }
}

// ---------------------------------------------------------------------------
// Static regularization ladder (mirrors `qp::ipm_core::ippmm::state` /
// `qp::ipm_core::ippmm::factorize::probe_ldl_health`).
// ---------------------------------------------------------------------------

/// Initial static regularization. Matches the magnitude of the (now-removed)
/// dense-KKT path's `reg` constant.
const REG_DELTA_INIT: f64 = 1e-10;
/// Growth factor on a failed factorization or unhealthy sanity probe.
/// Matches `qp::ipm_core::ippmm::state::LDL_REG_GROWTH`.
const REG_GROWTH: f64 = 10.0;
/// Regularization ceiling for the retry ladder. Matches
/// `qp::ipm_core::ippmm::state::LDL_REG_CEILING`.
const REG_CEILING: f64 = 1.0;
/// Retries before falling back to an identity permutation. Matches
/// `qp::ipm_core::ippmm::state::LDL_REG_RETRY_MAX`.
const REG_RETRY_MAX: u32 = 10;
/// Sanity-probe relative-residual tolerance. Matches
/// `qp::ipm_core::ippmm::factorize`'s `LDL_HEALTH_REL_TOL`.
const LDL_HEALTH_REL_TOL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Static structure + per-iteration value cache.
// ---------------------------------------------------------------------------

enum SlotTag {
    DxDiag(usize),
    DyDiag(usize),
    W2(usize),
    /// Dynamic entry of the rank-1-border representation (Phase 3b):
    /// indexes into `Scaling::border_values`'s output, which visits the
    /// same entries in the same order as `cone::visit_border_pattern`.
    Border(usize),
}

/// Static sparsity + slot bookkeeping for the augmented KKT matrix, built
/// once from the problem's `A`, `G`, and cone block dimensions (all
/// iteration-invariant). `materialize` rewrites only the dynamic slots
/// (`W^2`/border block entries + the `dx`/`dy` regularization diagonals)
/// each iteration; the `A^T`/`G^T` values and the border corners (`+1`/`-1`,
/// always exactly those constants regardless of NT scaling) are copied once
/// and never touched again.
struct KktSkeleton {
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    static_values: Vec<f64>,
    w2_slots: Vec<usize>,
    border_slots: Vec<usize>,
    dx_diag_slot: Vec<usize>,
    dy_diag_slot: Vec<usize>,
    n: usize,
    p: usize,
    m: usize,
    /// Number of border-enabled second-order cones (`cone::Blocks::n_border`);
    /// contributes one `aux_u` column (next to `dx`) and one `aux_v` column
    /// (next to `dz`) each. Column layout: `[0,n)=dx`, `[n,n_e)=aux_u`
    /// (`n_e = n + n_border`), `[n_e,n_e+p)=dy`, `[n_e+p,n_e+p+m)=dz`,
    /// `[n_e+p+m,total)=aux_v`.
    n_border: usize,
}

impl KktSkeleton {
    /// Size of the positive-definite half (`dx` plus every `aux_u`).
    fn n_e(&self) -> usize {
        self.n + self.n_border
    }

    fn total(&self) -> usize {
        self.n_e() + self.p + self.m + self.n_border
    }

    fn materialize(&self, sc: &Scaling, blk: &Blocks, delta_p: f64, delta_d: f64) -> CscMatrix {
        let mut values = self.static_values.clone();
        let w2vals = sc.w2_values_col_major(blk);
        debug_assert_eq!(w2vals.len(), self.w2_slots.len());
        // `W^2` is left fully unregularized: unlike the QP `Sigma` block
        // (genuinely zero for an inactive inequality, hence its `-delta_d`
        // floor), `W^2` is always strictly positive by construction (`s`,
        // `z` stay in the strict cone interior via fraction-to-boundary), so
        // no floor is needed for quasidefiniteness -- and adding one, even
        // one deliberately smaller than `faer`'s own internal clamp
        // (`crate::linalg::ldl`'s `LDLT_REG_EPSILON`/`LDLT_REG_DELTA`), only
        // ever hurts: a conflicting orthant/SOC pair's `W^2_ii = s_i/z_i`
        // needs to keep shrinking without a floor for the Newton direction
        // to keep amplifying `z_i` toward a Farkas ray (measured: a fixed
        // floor here reproduces the pre-fix plateau in
        // `socp_degenerate_fixed_var_infeasible_gets_certificate`). When the
        // true value underflows `faer`'s own clamp threshold, the MINRES
        // escalation in `factorize_with_retry` (which factors the raw
        // matrix, not an LDL of it) picks up the slack.
        for (k, &slot) in self.w2_slots.iter().enumerate() {
            values[slot] = -w2vals[k];
        }
        let border_vals = sc.border_values(blk);
        debug_assert_eq!(border_vals.len(), self.border_slots.len());
        // Same unregularized-by-design rationale as `W^2` above: these are
        // the identical `-eta^2` diagonal / `eta*sqrt(2)*wbar` couplings
        // underlying the dense block, just re-expressed sparsely.
        for (k, &slot) in self.border_slots.iter().enumerate() {
            values[slot] = border_vals[k];
        }
        for &slot in &self.dx_diag_slot {
            values[slot] = delta_p;
        }
        for &slot in &self.dy_diag_slot {
            values[slot] = -delta_d;
        }
        let total = self.total();
        CscMatrix {
            col_ptr: self.col_ptr.clone(),
            row_ind: self.row_ind.clone(),
            values,
            nrows: total,
            ncols: total,
        }
    }

    /// Applies a symmetric permutation to the static skeleton (mirrors
    /// `qp::ipm_core::kkt::AugmentedKktCache::permute`), remapping every
    /// dynamic slot into the permuted index space so `materialize` on the
    /// result needs no further permutation work per iteration.
    fn permute(&self, perm: &[usize]) -> KktSkeleton {
        let total = self.total();
        debug_assert_eq!(perm.len(), total);

        let mut inv_perm = vec![0usize; total];
        for (k, &i) in perm.iter().enumerate() {
            inv_perm[i] = k;
        }

        let nnz = self.row_ind.len();
        // (new_row, new_col, orig_slot)
        let mut entries: Vec<(usize, usize, usize)> = Vec::with_capacity(nnz);
        for col in 0..total {
            let new_col = inv_perm[col];
            for s in self.col_ptr[col]..self.col_ptr[col + 1] {
                let row = self.row_ind[s];
                let new_row = inv_perm[row];
                let (r, c) = if new_row <= new_col {
                    (new_row, new_col)
                } else {
                    (new_col, new_row)
                };
                entries.push((r, c, s));
            }
        }
        entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut new_of_orig = vec![usize::MAX; nnz];
        let mut new_col_ptr = Vec::with_capacity(total + 1);
        let mut new_row_ind = Vec::with_capacity(nnz);
        let mut new_static_values = Vec::with_capacity(nnz);
        new_col_ptr.push(0);
        let mut cur_col = 0usize;
        for &(r, c, orig_slot) in &entries {
            while cur_col < c {
                new_col_ptr.push(new_row_ind.len());
                cur_col += 1;
            }
            let new_slot = new_row_ind.len();
            new_row_ind.push(r);
            new_static_values.push(self.static_values[orig_slot]);
            new_of_orig[orig_slot] = new_slot;
        }
        while cur_col < total {
            new_col_ptr.push(new_row_ind.len());
            cur_col += 1;
        }

        let remap =
            |slots: &[usize]| -> Vec<usize> { slots.iter().map(|&s| new_of_orig[s]).collect() };
        KktSkeleton {
            col_ptr: new_col_ptr,
            row_ind: new_row_ind,
            static_values: new_static_values,
            w2_slots: remap(&self.w2_slots),
            border_slots: remap(&self.border_slots),
            dx_diag_slot: remap(&self.dx_diag_slot),
            dy_diag_slot: remap(&self.dy_diag_slot),
            n: self.n,
            p: self.p,
            m: self.m,
            n_border: self.n_border,
        }
    }
}

/// Builds the static (unpermuted) KKT skeleton from the problem's `A`
/// (`p x n`), `G` (`m x n`), and cone block dimensions. Column layout:
/// `[0,n)` = `dx`, `[n,n_e)` = `aux_u` (`n_e = n + n_border`), `[n_e,n_e+p)`
/// = `dy`, `[n_e+p,n_e+p+m)` = `dz`, `[n_e+p+m,total)` = `aux_v` (Phase 3b;
/// one `aux_u`/`aux_v` pair per second-order cone at or above
/// `cone::SOC_BORDER_MIN_DIM`, discarded from the solve after use -- see
/// `solve_dir`; placement rationale in the module doc comment).
fn build_skeleton(a: &CscMatrix, g: &CscMatrix, blk: &Blocks, n: usize, p: usize) -> KktSkeleton {
    let m = blk.dim();
    let n_border = blk.n_border();
    let n_e = n + n_border;
    let total = n_e + p + m + n_border;
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(g.ncols(), n);
    debug_assert_eq!(a.nrows(), p);
    debug_assert_eq!(g.nrows(), m);

    let mut col_entries: Vec<Vec<(usize, f64, Option<SlotTag>)>> =
        (0..total).map(|_| Vec::new()).collect();

    for j in 0..n {
        col_entries[j].push((j, 0.0, Some(SlotTag::DxDiag(j))));
    }
    // `aux_u` corners: static `+1`, purely diagonal within `[dx, aux_u]`
    // (no coupling to `dx` or to other `aux_u`s -- only to `dz` rows, added
    // below from the `dz`-column side since `aux_u`'s column index is
    // always smaller).
    for bi in 0..blk.soc.len() {
        if let Some(idx) = blk.border_idx(bi) {
            let col = n + idx;
            col_entries[col].push((col, 1.0, None));
        }
    }
    for j in 0..n {
        for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
            let row_k = a.row_ind()[k];
            col_entries[n_e + row_k].push((j, a.values()[k], None));
        }
    }
    for k in 0..p {
        col_entries[n_e + k].push((n_e + k, 0.0, Some(SlotTag::DyDiag(k))));
    }
    for j in 0..n {
        for k in g.col_ptr()[j]..g.col_ptr()[j + 1] {
            let row_i = g.row_ind()[k];
            col_entries[n_e + p + row_i].push((j, g.values()[k], None));
        }
    }
    let mut w2_count = 0usize;
    cone::visit_w2_pattern(blk, |r, c, _is_diag| {
        col_entries[n_e + p + c].push((n_e + p + r, 0.0, Some(SlotTag::W2(w2_count))));
        w2_count += 1;
    });

    // Rank-1-border representation (Phase 3b): dynamic diagonal/coupling
    // entries via the shared visitor (kept in lockstep with
    // `Scaling::border_values` by construction, both driven by the same
    // per-SOC iteration order). `aux_u` sits *before* `dz` (column `n+idx`
    // < any `dz` row), so its coupling is stored from the `dz` column's
    // side (row = `aux_u`'s smaller index); `aux_v` sits *after* `dz`
    // (column `n_e+p+m+idx` > any `dz` row), so its coupling is stored from
    // its own column's side (row = `dz`'s smaller index) -- both are valid
    // upper-triangular (`row <= col`) placements of the same symmetric
    // entry. The two static corners (`+1`/`-1`, exact constants regardless
    // of NT scaling) never change, so get no `SlotTag`, mirroring how
    // `A`/`G`'s own structural entries above get `None`.
    let mut border_count = 0usize;
    cone::visit_border_pattern(blk, |r, kind| {
        let dz_row = n_e + p + r;
        match kind {
            cone::BorderEntryKind::Diag => {
                col_entries[dz_row].push((dz_row, 0.0, Some(SlotTag::Border(border_count))));
            }
            cone::BorderEntryKind::CouplingU(idx) => {
                let u_col = n + idx;
                col_entries[dz_row].push((u_col, 0.0, Some(SlotTag::Border(border_count))));
            }
            cone::BorderEntryKind::CouplingV(idx) => {
                let v_col = n_e + p + m + idx;
                col_entries[v_col].push((dz_row, 0.0, Some(SlotTag::Border(border_count))));
            }
        }
        border_count += 1;
    });
    // `aux_v` corners: static `-1`.
    for bi in 0..blk.soc.len() {
        if let Some(idx) = blk.border_idx(bi) {
            let col = n_e + p + m + idx;
            col_entries[col].push((col, -1.0, None));
        }
    }

    for entries in col_entries.iter_mut() {
        entries.sort_by_key(|&(r, _, _)| r);
    }

    let nnz: usize = col_entries.iter().map(|v| v.len()).sum();
    let mut col_ptr = Vec::with_capacity(total + 1);
    let mut row_ind = Vec::with_capacity(nnz);
    let mut static_values = Vec::with_capacity(nnz);
    let mut dx_diag_slot = vec![0usize; n];
    let mut dy_diag_slot = vec![0usize; p];
    let mut w2_slots = vec![0usize; w2_count];
    let mut border_slots = vec![0usize; border_count];
    col_ptr.push(0);
    for entries in col_entries.into_iter() {
        for (row, val, tag) in entries {
            let slot = row_ind.len();
            row_ind.push(row);
            static_values.push(val);
            match tag {
                Some(SlotTag::DxDiag(j)) => dx_diag_slot[j] = slot,
                Some(SlotTag::DyDiag(k)) => dy_diag_slot[k] = slot,
                Some(SlotTag::W2(idx)) => w2_slots[idx] = slot,
                Some(SlotTag::Border(idx)) => border_slots[idx] = slot,
                None => {}
            }
        }
        col_ptr.push(row_ind.len());
    }

    KktSkeleton {
        col_ptr,
        row_ind,
        static_values,
        w2_slots,
        border_slots,
        dx_diag_slot,
        dy_diag_slot,
        n,
        p,
        m,
        n_border,
    }
}

/// Which rung of [`factorize_with_retry`]'s escalation ladder succeeded on
/// the previous outer IPM iteration. Ill-conditioning evolves continuously
/// along the central path (the NT scaling changes a bounded amount per
/// step), so the level that worked last iteration is the best starting guess
/// for this one -- without it, a trajectory pinned at the equilibration
/// rung (e.g. a diverging Farkas-ray run) would re-walk all
/// [`REG_RETRY_MAX`] failing ladder rungs every iteration.
enum EscalationHint {
    /// The f64 ladder succeeded at this `delta`. The next call starts one
    /// rung *below* (floored at [`REG_DELTA_INIT`]) so the ladder can decay
    /// back to minimal regularization as conditioning recovers.
    Ladder(f64),
    /// The Jacobi-equilibration rung succeeded; try it first next call.
    Equilibrated,
}

/// Reusable state across all outer IPM iterations: the static skeleton
/// (unpermuted, used for the sanity probe's `K*sol` check and the identity
/// fallback), its AMD-permuted counterpart (used for the fast factorization
/// path), the cached symbolic Cholesky (sparsity-pattern-invariant across
/// regularization retries and outer iterations), and the previous
/// iteration's successful escalation rung.
pub(super) struct KktCaches {
    base: KktSkeleton,
    perm: Vec<usize>,
    permuted: KktSkeleton,
    symbolic: Option<Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>>,
    /// Size of the positive-definite half (`dx` plus every `aux_u`; see
    /// `KktSkeleton::n_e`), used as the MINRES block-diagonal
    /// preconditioner's `n_top`.
    n_top: usize,
    hint: EscalationHint,
}

pub(super) fn build_kkt_caches(
    a: &CscMatrix,
    g: &CscMatrix,
    blk: &Blocks,
    n: usize,
    p: usize,
    deadline: Option<Instant>,
) -> KktCaches {
    let base = build_skeleton(a, g, blk, n, p);
    let perm = amd_pinned_aux(&base, deadline);
    let permuted = base.permute(&perm);
    let n_top = base.n_e();
    KktCaches {
        base,
        perm,
        permuted,
        symbolic: None,
        n_top,
        hint: EscalationHint::Ladder(REG_DELTA_INIT),
    }
}

/// AMD ordering with every auxiliary (border) column excluded from AMD's
/// input and pinned, in original order, to the *end* of the permutation
/// (`aux_u` block then `aux_v` block).
///
/// A correctness requirement, not an optimization: faer's AMD declares any
/// node of degree `> 10*sqrt(total)` a "dense node" and defers all such
/// nodes past every sparse node. `aux_u`'s column is dense in its cone's
/// `d` rows, so past that threshold AMD reshuffles the `dx`/`dz`
/// elimination order it couples -- measured (`n=m=d`, `p=1`): the healthy
/// `dx`-before-`dz` order flips from 202/202 to 0/202 at `d=202`, degrading
/// Newton directions to `O(1)` garbage failing the health probe. Pinning
/// the aux columns last processes `[dx, dy, dz]` first and applies the
/// rank-1 corrections as final Schur updates -- textbook bordered-system
/// elimination, `O(d)` fill per aux column (`conic_border_l_fill_stays_
/// linear` measures `4.0*d` at `d=100,000`). Sentinel (verified by revert):
/// `conic_kkt_direction_matches_dense_schur_oracle` case
/// `single_large_soc_border` (d=300, past the cutoff) fails without pinning.
fn amd_pinned_aux(base: &KktSkeleton, deadline: Option<Instant>) -> Vec<usize> {
    let total = base.total();
    if base.n_border == 0 {
        return amd_with_deadline(total, &base.col_ptr, &base.row_ind, deadline);
    }
    let n_e = base.n_e();
    let core_end = n_e + base.p + base.m;
    // Core (non-aux) node list: `dx` `[0,n)`, `dy`/`dz` `[n_e, core_end)`.
    // `aux_u` `[n, n_e)` and `aux_v` `[core_end, total)` are excluded.
    let mut core_of_full = vec![usize::MAX; total];
    let mut full_of_core = Vec::with_capacity(total - 2 * base.n_border);
    for i in (0..base.n).chain(n_e..core_end) {
        core_of_full[i] = full_of_core.len();
        full_of_core.push(i);
    }
    let n_core = full_of_core.len();

    // Sub-pattern over core nodes only (upper triangle is preserved:
    // core index order is monotone in the full index order).
    let mut sub_col_ptr = Vec::with_capacity(n_core + 1);
    let mut sub_row_ind = Vec::new();
    sub_col_ptr.push(0);
    for &col in &full_of_core {
        for k in base.col_ptr[col]..base.col_ptr[col + 1] {
            let row = base.row_ind[k];
            if core_of_full[row] != usize::MAX {
                sub_row_ind.push(core_of_full[row]);
            }
        }
        sub_col_ptr.push(sub_row_ind.len());
    }

    let sub_perm = amd_with_deadline(n_core, &sub_col_ptr, &sub_row_ind, deadline);
    let mut perm = Vec::with_capacity(total);
    perm.extend(sub_perm.iter().map(|&k| full_of_core[k]));
    perm.extend(base.n..n_e); // aux_u block
    perm.extend(core_end..total); // aux_v block
    debug_assert_eq!(perm.len(), total);
    perm
}

/// Factorizes the augmented KKT matrix for the current NT scaling `sc`,
/// growing the static `dx`/`dy` regularization on factorization failure or
/// an unhealthy sanity probe (`probe_rhs`, the real affine-step RHS) up to
/// [`REG_RETRY_MAX`] times (mirrors
/// `qp::ipm_core::ippmm::factorize::factorize_kkt_with_retry`'s retry
/// ladder), then escalating through Jacobi equilibration, DD-LDL, and
/// MINRES-on-the-raw-matrix, and finally re-using the smallest-regularization
/// AMD-ordered factorization as a last resort. Returns `None` only on
/// deadline expiry or total failure of every rung.
pub(super) fn factorize_with_retry(
    caches: &mut KktCaches,
    sc: &Scaling,
    blk: &Blocks,
    probe_rhs: &[f64],
    deadline: Option<Instant>,
    kkt_cfg: &KktConfig,
) -> Option<ConicFactor> {
    // A trajectory pinned at the equilibration rung skips the (known-failing)
    // ladder. On equilibration failure the hint is reset and the full ladder
    // below still runs, so this is purely a fast path, never a narrowing.
    if matches!(caches.hint, EscalationHint::Equilibrated) {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        if let Some(f) = try_equilibrated(caches, sc, blk, probe_rhs, deadline, kkt_cfg) {
            return Some(f);
        }
        caches.hint = EscalationHint::Ladder(REG_DELTA_INIT);
    }

    let mut delta = match caches.hint {
        EscalationHint::Ladder(d) => (d / REG_GROWTH).max(REG_DELTA_INIT),
        EscalationHint::Equilibrated => REG_DELTA_INIT,
    };
    for _ in 0..REG_RETRY_MAX {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let pre_permuted = caches.permuted.materialize(sc, blk, delta, delta);
        let unpermuted = caches.base.materialize(sc, blk, delta, delta);
        match factorize_kkt_pre_permuted_cached_par(
            &pre_permuted,
            &unpermuted,
            &caches.perm,
            deadline,
            kkt_cfg,
            Some(caches.n_top),
            caches.symbolic.clone(),
            faer::Par::Seq,
        ) {
            Ok(f) => {
                if caches.symbolic.is_none() {
                    caches.symbolic = f.symbolic_arc();
                }
                let healthy = f.is_iterative() || probe_kkt_health(&f, &unpermuted, probe_rhs);
                if healthy {
                    caches.hint = EscalationHint::Ladder(delta);
                    return Some(ConicFactor::direct(f));
                }
            }
            Err(crate::linalg::kkt_solver::KktError::DeadlineExceeded) => return None,
            Err(_) => {}
        }
        let next = (delta * REG_GROWTH).min(REG_CEILING);
        if next == delta {
            break;
        }
        delta = next;
    }

    // Jacobi-equilibration escalation (see [`try_equilibrated`]).
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return None;
    }
    if let Some(f) = try_equilibrated(caches, sc, blk, probe_rhs, deadline, kkt_cfg) {
        caches.hint = EscalationHint::Equilibrated;
        return Some(f);
    }
    // Ladder and equilibration both failed: remember the ceiling so the next
    // iteration reaches the deeper rungs after ~2 ladder tries instead of
    // re-walking every rung. (No hints exist for DD/MINRES/last-resort: the
    // cheap-and-exact equilibration attempt should always precede them.)
    caches.hint = EscalationHint::Ladder(REG_CEILING);

    // DD-LDL (TwoFloat, ~106-bit) escalation. The f64 ladder above only grows
    // `delta` on the `dx`/`dy` diagonals -- irrelevant when the true
    // ill-conditioning lives in `W^2` (deliberately unregularized, see the
    // module doc), e.g. a conflicting orthant/SOC pair where `s_i` has
    // shrunk enough that `f64`'s ~16 digits can no longer resolve `W^2_ii`
    // against the rest of the matrix. Still health-probed against the same
    // clamp thresholds as the f64 path (`crate::linalg::ldl_dd`'s
    // `EPSILON`/`DELTA`), so once `W^2_ii` underflows that shared
    // threshold, extra mantissa bits alone do not recover it (confirmed:
    // `socp_degenerate_fixed_var_infeasible_gets_certificate` shows the
    // same premature plateau under DD as under f64). Materialized at
    // REG_DELTA_INIT, not the ladder's final `delta`: a larger `dx`/`dy`
    // regularization buys nothing here (see the last-resort comment
    // below) and would only perturb the system away from the exact one
    // MINRES needs.
    let unpermuted = caches
        .base
        .materialize(sc, blk, REG_DELTA_INIT, REG_DELTA_INIT);
    if !kkt_cfg.dd_ldl {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let dd_cfg = KktConfig {
            dd_ldl: true,
            ..*kkt_cfg
        };
        if let Ok(f) = factorize_kkt_with_cached_perm_par(
            &unpermuted,
            &caches.perm,
            deadline,
            &dd_cfg,
            Some(caches.n_top),
            faer::Par::Seq,
        ) {
            if probe_kkt_health(&f, &unpermuted, probe_rhs) {
                return Some(ConicFactor::direct(f));
            }
        }
    }

    // MINRES-on-the-raw-matrix escalation. Both LDL backends factor the
    // matrix by elimination, and both clamp any pivot below a fixed
    // magnitude threshold regardless of precision (see above) -- a
    // structural limitation of no-pivot elimination, not of `f64` precision.
    // MINRES instead works directly with the exact stored matrix entries via
    // mat-vec products, so it is not subject to that clamp; the
    // block-diagonal preconditioner (`n_top = caches.n_top`, the `[dx, aux_u]`
    // PD group) keeps it effective on this saddle-point structure. Tight
    // tolerance + iterative refinement since this is a last-resort accurate
    // solve, not the loose inexact-Newton search direction used elsewhere.
    const MINRES_LAST_RESORT_TOL: f64 = 1e-10;
    const MINRES_LAST_RESORT_IR: usize = 2;
    let minres = PreconditionedMinres::with_block_diag_inexact(
        unpermuted,
        caches.n_top,
        MINRES_LAST_RESORT_TOL,
        MINRES_LAST_RESORT_IR,
    );
    let mut probe_sol = vec![0.0; caches.base.total()];
    let minres_result = minres.solve(probe_rhs, &mut probe_sol, deadline);
    if minres_result.is_ok() {
        return Some(ConicFactor::direct(KktFactor::Iterative(minres)));
    }

    // Last resort: re-use the AMD-ordered factorization at the *smallest*
    // tried regularization (`REG_DELTA_INIT`), accepted despite failing the
    // health probe. Deliberately not an identity-permutation fallback (as
    // `qp::ipm_core::ippmm::factorize` uses): AMD's order here eliminates
    // the `dz` block before `dx` (matches the Schur-complement elimination
    // direction), whereas an identity order eliminates `dx` first, whose
    // *own* regularization then bleeds into `dz`'s diagonal via the Schur
    // update, corrupting exactly the tiny `W^2` entries this whole ladder
    // exists to protect. A larger `delta` on `dx`/`dy` doesn't change `dz`'s
    // own diagonal (see the module doc comment), so re-trying with a bigger
    // regularization buys nothing; the smallest one keeps the AMD-ordered
    // elimination closest to exact.
    let unpermuted = caches
        .base
        .materialize(sc, blk, REG_DELTA_INIT, REG_DELTA_INIT);
    let pre_permuted = caches
        .permuted
        .materialize(sc, blk, REG_DELTA_INIT, REG_DELTA_INIT);
    match factorize_kkt_pre_permuted_cached_par(
        &pre_permuted,
        &unpermuted,
        &caches.perm,
        deadline,
        kkt_cfg,
        Some(caches.n_top),
        caches.symbolic.clone(),
        faer::Par::Seq,
    ) {
        Ok(f) => Some(ConicFactor::direct(f)),
        Err(_) => None,
    }
}

/// Jacobi (symmetric diagonal) equilibration escalation rung. `faer`'s LDL
/// clamps any pivot below a *fixed* magnitude regardless of the rest of
/// the matrix (see the module doc); a conflicting orthant/SOC pair's
/// `W^2_ii` can underflow that threshold while other rows stay `O(1)`.
/// Rescaling `K -> D K D` with `D_i = 1/sqrt(|K_ii|)` (applied to every
/// row) normalises every diagonal to exactly `+-1`, so no diagonal is
/// ever "small" relative to the fixed clamp regardless of the original
/// dynamic range -- the tiny value survives instead as a large but
/// finite off-diagonal, which the clamp does not touch. Solving `(D K
/// D) x' = D rhs` then recovering `x = D x'`
/// ([`ConicFactor::equilibrated`]) is exact, not approximate -- unlike
/// DD/MINRES, which hit the identical premature plateau (measured:
/// `socp_degenerate_fixed_var_infeasible_gets_certificate`). Reuses
/// `caches.perm`/`caches.symbolic`: equilibration only rescales values,
/// the sparsity pattern is unchanged.
fn try_equilibrated(
    caches: &KktCaches,
    sc: &Scaling,
    blk: &Blocks,
    probe_rhs: &[f64],
    deadline: Option<Instant>,
    kkt_cfg: &KktConfig,
) -> Option<ConicFactor> {
    let unpermuted = caches
        .base
        .materialize(sc, blk, REG_DELTA_INIT, REG_DELTA_INIT);
    let (scaled, d) = equilibrate(&unpermuted);
    let (perm_col_ptr, perm_row_ind, perm_values) = crate::linalg::amd::permute_sym_upper(
        scaled.nrows(),
        scaled.col_ptr(),
        scaled.row_ind(),
        scaled.values(),
        &caches.perm,
    );
    let pre_permuted_scaled = CscMatrix {
        col_ptr: perm_col_ptr,
        row_ind: perm_row_ind,
        values: perm_values,
        nrows: scaled.nrows(),
        ncols: scaled.ncols(),
    };
    if let Ok(f) = factorize_kkt_pre_permuted_cached_par(
        &pre_permuted_scaled,
        &scaled,
        &caches.perm,
        deadline,
        kkt_cfg,
        Some(caches.n_top),
        caches.symbolic.clone(),
        faer::Par::Seq,
    ) {
        let scaled_rhs: Vec<f64> = probe_rhs.iter().zip(&d).map(|(&r, &di)| r * di).collect();
        let healthy = f.is_iterative() || probe_kkt_health(&f, &scaled, &scaled_rhs);
        if healthy {
            return Some(ConicFactor::equilibrated(f, d));
        }
    }
    None
}

/// Symmetric diagonal equilibration `K -> D K D`, `D_i = 1/sqrt(|K_ii|)`.
/// `mat` must be upper-triangular with every diagonal entry present. Every
/// conic KKT diagonal is nonzero by construction (`dx`'s `delta_p > 0`,
/// `dy`'s `-delta_d < 0`, `dz`'s `-W^2_ii < 0`), but `W^2_ii = s_i/z_i` can
/// underflow to an exact `0.0` (or a subnormal whose reciprocal overflows)
/// in `f64` on extreme trajectories -- those rows fall back to `D_i = 1`
/// (no scaling) rather than poisoning the whole matrix with `inf`/`NaN`.
/// Returns the scaled matrix and `D`.
fn equilibrate(mat: &CscMatrix) -> (CscMatrix, Vec<f64>) {
    let n = mat.nrows();
    let mut diag = vec![0.0_f64; n];
    for col in 0..n {
        for k in mat.col_ptr()[col]..mat.col_ptr()[col + 1] {
            if mat.row_ind()[k] == col {
                diag[col] = mat.values()[k];
                break;
            }
        }
    }
    let d: Vec<f64> = diag
        .iter()
        .map(|&v| {
            let s = v.abs();
            // `1.0/s` guards subnormal `s` whose reciprocal overflows even
            // though `1/sqrt(s)` alone would still be finite: the scaled
            // diagonal is computed as `v * d_i * d_i = v / s`, so `1/s` is
            // the quantity that must stay finite.
            if s > 0.0 && (1.0 / s).is_finite() {
                1.0 / s.sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let mut values = mat.values().to_vec();
    for col in 0..n {
        for k in mat.col_ptr()[col]..mat.col_ptr()[col + 1] {
            let row = mat.row_ind()[k];
            values[k] *= d[row] * d[col];
        }
    }
    let scaled = CscMatrix {
        col_ptr: mat.col_ptr().to_vec(),
        row_ind: mat.row_ind().to_vec(),
        values,
        nrows: n,
        ncols: n,
    };
    (scaled, d)
}

/// A [`KktFactor`] together with an optional symmetric diagonal
/// equilibration (see [`equilibrate`]). `solve` transparently rescales the
/// RHS/solution around the wrapped factor's own (possibly-equilibrated)
/// solve; callers never need to know whether equilibration was used.
pub(super) struct ConicFactor {
    factor: KktFactor,
    scale: Option<Vec<f64>>,
}

impl ConicFactor {
    /// `nnz(L)` of the direct LDL factor (`None` for the DD / MINRES
    /// backends). Test-only: used by the `O(d)`-fill fence for the
    /// border representation (`conic_border_l_fill_stays_linear`).
    #[cfg(test)]
    pub(super) fn nnz_l(&self) -> Option<usize> {
        match &self.factor {
            KktFactor::Direct(f) => Some(f.nnz_l()),
            _ => None,
        }
    }

    fn direct(factor: KktFactor) -> Self {
        Self {
            factor,
            scale: None,
        }
    }

    fn equilibrated(factor: KktFactor, d: Vec<f64>) -> Self {
        Self {
            factor,
            scale: Some(d),
        }
    }

    pub(super) fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        match &self.scale {
            None => self.factor.solve(rhs, sol),
            Some(d) => {
                let scaled_rhs: Vec<f64> = rhs.iter().zip(d).map(|(&r, &di)| r * di).collect();
                let mut scaled_sol = vec![0.0; rhs.len()];
                self.factor.solve(&scaled_rhs, &mut scaled_sol);
                for (o, (&s, &di)) in sol.iter_mut().zip(scaled_sol.iter().zip(d)) {
                    *o = s * di;
                }
            }
        }
    }
}

/// (A) `||K*sol-rhs|| / ||rhs|| <= LDL_HEALTH_REL_TOL` (LDL breakdown sanity,
/// eps-independent) (B) `||sol||_inf / ||rhs||_inf <= 1/eps_machine` (cond(K)
/// stays within f64 range). Mirrors
/// `qp::ipm_core::ippmm::factorize::probe_ldl_health`.
fn probe_kkt_health(f: &KktFactor, mat: &CscMatrix, rhs: &[f64]) -> bool {
    let dim = mat.nrows();
    let rhs_inf = rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if rhs_inf <= 0.0 || !rhs_inf.is_finite() {
        return true;
    }
    let mut sol = vec![0.0_f64; dim];
    f.solve(rhs, &mut sol);

    let mut kx = vec![0.0_f64; dim];
    for col in 0..mat.ncols() {
        for k in mat.col_ptr()[col]..mat.col_ptr()[col + 1] {
            let row = mat.row_ind()[k];
            let val = mat.values()[k];
            kx[row] += val * sol[col];
            if row != col {
                kx[col] += val * sol[row];
            }
        }
    }
    let resid_inf = (0..dim)
        .map(|i| (rhs[i] - kx[i]).abs())
        .fold(0.0_f64, f64::max);
    let rel_resid = resid_inf / rhs_inf;
    let sol_inf = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let amplification = sol_inf / rhs_inf;
    let f64_precision_ceiling = 1.0 / f64::EPSILON;
    rel_resid.is_finite()
        && rel_resid <= LDL_HEALTH_REL_TOL
        && amplification.is_finite()
        && amplification <= f64_precision_ceiling
}

/// Builds the augmented-system RHS `(-rx, 0, -ry, -rz - W*rc, 0)` (column
/// layout `[dx, aux_u, dy, dz, aux_v]`) for a given complementarity target
/// `rc` (`-lambda` for the affine direction, `jdiv(lambda, target)` for the
/// corrector). Derivation of the `dz` row: scaled complementarity is
/// `W^{-1} ds + W dz = rc` (arrow-form, matching `jdiv`/`jprod`), and the
/// conic residual is `ds = -rz - G dx`; substituting gives `-W^2 dz + G dx
/// = -rz - W rc`, i.e. the RHS above (`ds` is recovered from the scaled
/// complementarity as `W (rc - W dz)`, see [`solve_dir`]). Shared by
/// [`solve_dir`] (the real solve) and
/// [`factorize_with_retry`]'s sanity probe, so the probe exercises the
/// exact system the outer solver depends on. Both auxiliary (border) row
/// ranges are always zero: they are not physical unknowns, just a device
/// for keeping `-W^2` sparse (see `cone::visit_border_pattern`), so they
/// carry no forcing term.
pub(super) fn build_rhs(
    sc: &Scaling,
    blk: &Blocks,
    n: usize,
    p: usize,
    m: usize,
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    rc: &[f64],
) -> Vec<f64> {
    let w_rc = sc.apply_w(blk, rc);
    let n_e = n + blk.n_border();
    let total = n_e + p + m + blk.n_border();
    let mut rhs = vec![0.0; total];
    for i in 0..n {
        rhs[i] = -rx[i];
    }
    for i in 0..p {
        rhs[n_e + i] = -ry[i];
    }
    for i in 0..m {
        rhs[n_e + p + i] = -rz[i] - w_rc[i];
    }
    rhs
}

/// Solves the augmented system for one predictor/corrector direction. The
/// border auxiliary unknowns (`aux_u` at `sol[n..n_e]`, `aux_v` at
/// `sol[n_e+p+m..]`, see `cone::visit_border_pattern`) are solved alongside
/// `dx`/`dy`/`dz` but discarded -- meaningless outside the linear solve.
///
/// `ds` is recovered per cone type: the primal-residual form `ds = -rz - G dx`
/// and the scaled-complementarity form `ds = W (rc - W dz)` (from
/// `W^{-1} ds + W dz = rc`) agree at the exact solution but differ by the solve
/// residual, and each is unstable in a different regime (detailed at the two
/// branches below).
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_dir(
    factor: &ConicFactor,
    g: &CscMatrix,
    sc: &Scaling,
    blk: &Blocks,
    n: usize,
    p: usize,
    m: usize,
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    rc: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let rhs = build_rhs(sc, blk, n, p, m, rx, ry, rz, rc);
    let n_e = n + blk.n_border();
    let total = n_e + p + m + blk.n_border();
    let mut sol = vec![0.0; total];
    factor.solve(&rhs, &mut sol);
    let dx = sol[0..n].to_vec();
    let dy = sol[n_e..n_e + p].to_vec();
    let dz = sol[n_e + p..n_e + p + m].to_vec();

    // Orthant rows: scaled-complementarity form. The primal form sign-flips
    // when a bound's slack collapses (`W^2 = s/z -> 0`, as at the CBLIB `*_w`
    // permanently-active bounds): `rz` and `G dx` are then same-order but their
    // true difference sits below the solve's error floor (primal-form `*_w`
    // still hit `MaxIterations` even from the data-scaled start).
    let w_dz = sc.apply_w(blk, &dz);
    let scaled_residual: Vec<f64> = rc
        .iter()
        .zip(&w_dz)
        .map(|(rc_i, wdz_i)| rc_i - wdz_i)
        .collect();
    let mut ds = sc.apply_w(blk, &scaled_residual);
    // SOC rows: primal-residual form, contracting `rz` exactly
    // (`rz_new = (1 - alpha) rz`). The complementarity form applies the dense
    // `W` twice per block, amplifying the solve residual with dimension --
    // measured to diverge on the border-represented `d ~ 1e5` cone whose `dz`
    // is itself exact (`conic_kkt_direction_matches_dense_schur_oracle`).
    if blk.l < m {
        let gdx = spmv(g, &dx);
        for i in blk.l..m {
            ds[i] = -rz[i] - gdx[i];
        }
    }
    (dx, dy, dz, ds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentinel for [`equilibrate`]'s zero/subnormal-diagonal guard:
    /// reverting the guard (plain `1/sqrt(|K_ii|)`) makes `d[1]` infinite
    /// (zero diagonal) and the scaled `[2][2]` entry non-finite (subnormal
    /// diagonal whose reciprocal overflows), failing the assertions below.
    #[test]
    fn equilibrate_guards_zero_and_subnormal_diagonals() {
        // Upper-tri 4x4 CSC: diagonals [1e-10, 0.0, 5e-324 (subnormal), 4.0]
        // plus one off-diagonal (row 0, col 3).
        let mat = CscMatrix {
            col_ptr: vec![0, 1, 2, 3, 5],
            row_ind: vec![0, 1, 2, 0, 3],
            values: vec![1e-10, 0.0, 5e-324, -1.0, 4.0],
            nrows: 4,
            ncols: 4,
        };
        let (scaled, d) = equilibrate(&mat);
        for (i, &di) in d.iter().enumerate() {
            assert!(di.is_finite() && di > 0.0, "d[{i}]={di}");
        }
        assert_eq!(d[1], 1.0, "zero diagonal must fall back to no scaling");
        assert_eq!(d[2], 1.0, "subnormal diagonal must fall back to no scaling");
        for (k, &v) in scaled.values().iter().enumerate() {
            assert!(v.is_finite(), "scaled values[{k}]={v}");
        }
        // Healthy diagonals normalize to +-1 (up to sqrt rounding); the
        // guarded rows keep their original (unscaled) diagonal values.
        assert!(
            (scaled.values()[0] - 1.0).abs() < 1e-12,
            "{}",
            scaled.values()[0]
        );
        assert!(
            (scaled.values()[4] - 1.0).abs() < 1e-12,
            "{}",
            scaled.values()[4]
        );
        assert_eq!(scaled.values()[1], 0.0);
        assert_eq!(scaled.values()[2], 5e-324);
    }
}
