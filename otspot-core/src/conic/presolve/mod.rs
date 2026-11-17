//! Tier-1 presolve for the conic route (`min c^T x  s.t.  A x = b,  G x + s =
//! h,  s ∈ K`).
//!
//! Closes the gap reported by the conic-presolve research note (no presolve
//! stage existed at all: `solve_socp` = validate → Ruiz equilibration → IPM).
//! The tier implemented here is **equality substitution + fixed-variable
//! elimination + domain identity-row absorption** into the variable-bound
//! layer ([`crate::conic::ConicProblem::var_bounds`]), with the full
//! postsolve/certificate contract of report §5: LIFO postsolve stack,
//! original-space re-verification of primal/dual/ray/Farkas payloads,
//! presolve-scale eps tightening, and the LP `steps_basic.rs` coefficient
//! guard ported as a named constant.
//!
//! Explicitly **out of tier here** (report §5 boundary, stated honestly):
//! variable-cone / native-bound IPM formulation work (tier 2), block-angular
//! decomposition and iterative solves (tier 3), and generic lindep /
//! redundant-row removal — it needs a rank/nullspace machinery that does not
//! fall out of the postsolve-stack design, so it is omitted rather than
//! approximated. Surviving domain bounds are re-materialized as orthant rows
//! (the IPM consumes rows only), which is why the `*_w` memory classes are
//! *not* rescued by this tier; the reduction that does land here is the
//! equality-row/column elimination that dominates the `nql` / `chainsing` /
//! `qssp` shape.

mod postsolve;
mod work;

#[cfg(test)]
mod tests;

pub(crate) use postsolve::VERIFY_KKT_SLACK;

use crate::conic::{ConeSpec, ConicProblem, ConicResult};
use postsolve::{ConicPostsolveStep, Maps};
use work::Workspace;

/// Never reduce below this many live columns: the IPM, its starting-point
/// heuristics and the KKT factorization are only exercised end-to-end with a
/// real variable block, so a fully reduced 0/1-column "problem" is better
/// solved unreduced than speculatively.
pub const REDUCED_DIM_MIN_COLS: usize = 2;

/// Floor on live equality rows. Zero is legal for the IPM (the MISOCP node
/// relaxations run with no equalities), so only [`REDUCED_DIM_MIN_COLS`] is
/// load-bearing; this documents the intended stopping rule at the boundary
/// where substitution has consumed every usable pivot row.
pub const REDUCED_DIM_MIN_ROWS: usize = 0;

/// Minimum eps-tighten factor the conic presolve will request (research
/// report contract (b), mirroring `tighten_ipm_eps_for_presolve_scale`):
/// tightening below `1e-3 × tol` asks the IPM for three-plus orders beyond
/// the user target, where factorization noise dominates the benefit and hard
/// instances stall; the re-verification gate + fallback — not an unbounded
/// tighten — is the protection against amplification.
pub const EPS_SIGMA_CLAMP: f64 = 1e-3;

/// Noise floor for the tightened conic IPM tolerance. The conic KKT is a
/// quasidefinite sparse LDL^T whose backward error scales with `kappa` times
/// machine eps; a relative residual target below `1e-12` is unreachable noise
/// on CBLIB-scale data (and stays far above the `1e-16` mantissa edge, so
/// the IPM's divergence checks keep seeing a meaningful loop).
pub const CONIC_PRESOLVE_EPS_FLOOR: f64 = 1e-12;

/// Cap on presolve driver rounds (absorption → fixing → substitution). Each
/// changing round strictly removes at least one row/column, so the cap only
/// guards against a logic bug cycling, not against slow convergence; on
/// exceeding it the current (always-equivalent) partial reduction is simply
/// finalized.
const MAX_PRESOLVE_ROUNDS: usize = 64;

/// Public dimensional summary of a reduction (unit-level RED→GREEN evidence).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReductionStats {
    /// Original variable count.
    pub n0: usize,
    /// Reduced variable count.
    pub n1: usize,
    /// Original equality rows.
    pub p0: usize,
    /// Reduced equality rows.
    pub p1: usize,
    /// Original conic rows.
    pub m0: usize,
    /// Reduced conic rows.
    pub m1: usize,
    /// Original KKT dimension (see [`kkt_dim`]).
    pub kkt0: usize,
    /// Reduced KKT dimension.
    pub kkt1: usize,
    /// Residual-eps tightening factor in `(0, 1]` (see [`tightened_tol`]).
    pub sigma: f64,
    /// Domain identity rows absorbed into the bound layer.
    pub m_domain_absorbed: usize,
    /// Columns eliminated by fixed-value substitution.
    pub n_fixed: usize,
    /// Columns eliminated by equality pivoting.
    pub n_substituted: usize,
}

/// KKT dimension the conic factorization sees for `problem`: `n + p + m`.
/// (The SOC rank-1 border trick only engages at cone dim ≥ 256 and adds its
/// aux rows there; no CBLIB bench class reaches it, and adding them here
/// would misreport the study's dim table — see report §1.2 item 3.)
pub fn kkt_dim(problem: &ConicProblem) -> usize {
    problem.n() + problem.p() + problem.m()
}

/// A domain-bound conflict proven during absorption (`lb > ub` for column
/// `col`) with the two absorbed witness rows. `z = 1/|g_k|` on both witness
/// rows and `y = 0` satisfies `A^T y + G^T z = 0` and `b^T y + h^T z =
/// ub − lb < 0` exactly, so the certificate is emitted directly in the
/// *original* space and re-verified by `verify_infeas_cert`.
pub(crate) enum InfeasProof {
    /// Two absorbed bound rows pinning `lb > ub`.
    BoundPair {
        /// The two witness bound rows (insertion order).
        witness: [postsolve::AbsorbedSnapshot; 2],
    },
}

/// Original-space Farkas pair for an [`InfeasProof`] (`None` if a witness
/// lacks original row identity, e.g. rows materialized from input bounds).
pub(crate) fn infeas_certificate(
    proof: &InfeasProof,
    problem: &ConicProblem,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let y = vec![0.0; problem.p()];
    let mut z = vec![0.0; problem.m()];
    match proof {
        InfeasProof::BoundPair { witness, .. } => {
            for w in witness {
                let orig = w.orig_row?;
                z[orig] = 1.0 / w.g_coef.abs();
            }
        }
    }
    Some((y, z))
}

/// A completed tier-1 reduction: reduced problem, index maps, LIFO stack,
/// conflict flag, eps-tighten factor. [`ConicReduction::reduce`] returns
/// `Option<ConicReduction>`: `None` means "solve as-is" (nothing reduced, or
/// a proven conflict whose witnesses lack original row identity).
pub struct ConicReduction {
    reduced: Option<ConicProblem>,
    maps: Option<Maps>,
    stack: Vec<ConicPostsolveStep>,
    conflict: Option<InfeasProof>,
    stats: ReductionStats,
}

/// Tightened IPM tolerance for a reduction with factor `sigma` (contract (b):
/// counteract residual amplification of the postsolve mapping, following the
/// QP route's `tighten_ipm_eps_for_presolve_scale`).
pub fn tightened_tol(user_tol: f64, sigma: f64) -> f64 {
    let s = sigma.clamp(EPS_SIGMA_CLAMP, 1.0);
    (user_tol * s).max(CONIC_PRESOLVE_EPS_FLOOR)
}

impl ConicReduction {
    /// Runs the tier-1 reductions on `problem`.
    pub fn reduce(problem: &ConicProblem) -> Option<ConicReduction> {
        let (mat, g_soc, g_orig) = materialize_bounds(problem);
        let mut ws = Workspace::build(&mat, &g_soc, &g_orig);
        let stats0 = ReductionStats {
            n0: problem.n(),
            n1: problem.n(),
            p0: problem.p(),
            p1: problem.p(),
            m0: problem.m(),
            m1: problem.m(),
            kkt0: kkt_dim(problem),
            kkt1: kkt_dim(problem),
            sigma: 1.0,
            m_domain_absorbed: 0,
            n_fixed: 0,
            n_substituted: 0,
        };
        let mut conflict = None;
        let mut rounds = 0usize;
        let mut stats = stats0;
        loop {
            rounds += 1;
            let (conf, absorbed) = ws.absorb_sweep();
            stats.m_domain_absorbed += absorbed;
            if let Some(c) = conf {
                conflict = Some(c);
                break;
            }
            let (zero, decline) = ws.simplify_zero_rows();
            if decline {
                return None;
            }
            let fixed = ws.fix_sweep();
            let substed = ws.substitute_sweep();
            stats.n_fixed += fixed;
            stats.n_substituted += substed;
            if absorbed + zero + fixed + substed == 0 || rounds >= MAX_PRESOLVE_ROUNDS {
                break;
            }
        }
        let sigma = (1.0 / ws.growth).min(1.0);
        if let Some(conf) = conflict {
            // Witnesses must carry original row identity for an
            // original-space certificate; otherwise decline (caller solves).
            if infeas_certificate(&conf, problem).is_none() {
                return None;
            }
            stats.sigma = sigma;
            return Some(ConicReduction {
                reduced: None,
                maps: None,
                stack: Vec::new(),
                conflict: Some(conf),
                stats,
            });
        }
        if ws.stack.is_empty() {
            return None;
        }
        let stack = std::mem::take(&mut ws.stack);
        let (reduced, maps) = finalize(&ws);
        stats.sigma = (1.0 / ws.growth).min(1.0);
        stats.n1 = reduced.n();
        stats.p1 = reduced.p();
        stats.m1 = reduced.m();
        stats.kkt1 = kkt_dim(&reduced);
        Some(ConicReduction {
            reduced: Some(reduced),
            maps: Some(maps),
            stack,
            conflict: None,
            stats,
        })
    }

    /// The reduced problem (`None` on the conflict path).
    pub fn reduced(&self) -> Option<&ConicProblem> {
        self.reduced.as_ref()
    }

    /// Dimensional summary (RED→GREEN assertion surface).
    pub fn stats(&self) -> ReductionStats {
        self.stats
    }

    /// `Some` when presolve proved the problem infeasible.
    pub(crate) fn conflict(&self) -> Option<&InfeasProof> {
        self.conflict.as_ref()
    }

    /// Maps the proof to an original-space Farkas pair.
    pub(crate) fn conflict_certificate(
        &self,
        problem: &ConicProblem,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        infeas_certificate(self.conflict.as_ref()?, problem)
    }

    /// Maps a reduced-space solve back to the original problem and
    /// re-verifies it there (`postsolve` module); `None` = verification
    /// failed, caller must re-solve the original problem unreduced.
    pub(crate) fn postsolve(
        &self,
        problem: &ConicProblem,
        reduced: &ConicProblem,
        res: &ConicResult,
        tol_gate: f64,
    ) -> Option<ConicResult> {
        let maps = self.maps.as_ref()?;
        postsolve::postsolve_result(problem, reduced, maps, &self.stack, res, tol_gate)
    }
}

/// Materializes input `var_bounds` into orthant rows appended to the
/// problem's *orthant* block so the IPM (rows only at tier 1) sees an
/// equivalent problem. Returns the problem plus per-`G`-row annotations
/// (`soc_block`, `original row`). Empty `var_bounds` — every site that does
/// not opt in — keeps the problem structurally identical; only the
/// annotations are computed.
pub(crate) fn materialize_bounds(
    problem: &ConicProblem,
) -> (ConicProblem, Vec<Option<usize>>, Vec<Option<usize>>) {
    let l0 = problem.cone.l;
    let m0 = problem.m();
    let mut extras: Vec<(usize, f64, f64)> = Vec::new();
    for (j, &(lb, ub)) in problem.var_bounds.iter().enumerate() {
        if lb.is_finite() {
            extras.push((j, -1.0, -lb));
        }
        if ub.is_finite() {
            extras.push((j, 1.0, ub));
        }
    }
    let shift = extras.len();
    // New G row order: [orthant 0..l0][extras][original SOC rows l0..m0].
    // SOC rows keep their relative order (they only shift down by `shift`).
    let m_new = m0 + shift;
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_new];
    for col in 0..problem.g.ncols() {
        for k in problem.g.col_ptr()[col]..problem.g.col_ptr()[col + 1] {
            let row = problem.g.row_ind()[k];
            let new_row = if row < l0 { row } else { row + shift };
            row_entries[new_row].push((col, problem.g.values()[k]));
        }
    }
    let mut h: Vec<f64> = Vec::with_capacity(m_new);
    h.extend_from_slice(&problem.h[..l0]);
    for (idx, &(col, gcoef, rhs)) in extras.iter().enumerate() {
        row_entries[l0 + idx].push((col, gcoef));
        h.push(rhs);
    }
    h.extend_from_slice(&problem.h[l0..]);
    let mut g_soc: Vec<Option<usize>> = vec![None; l0 + shift];
    for (bi, &dim) in problem.cone.soc.iter().enumerate() {
        g_soc.extend(std::iter::repeat(Some(bi)).take(dim));
    }
    let mut g_orig: Vec<Option<usize>> = (0..l0).map(Some).collect();
    g_orig.extend(std::iter::repeat(None).take(shift));
    g_orig.extend((l0..m0).map(Some));
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for (r, entries) in row_entries.iter().enumerate() {
        for &(col, v) in entries {
            rows.push(r);
            cols.push(col);
            vals.push(v);
        }
    }
    let g =
        otspot_num::sparse::CscMatrix::from_triplets(&rows, &cols, &vals, m_new, problem.g.ncols())
            .expect("bound materialization keeps G structurally valid");
    let mat = ConicProblem {
        c: problem.c.clone(),
        a: problem.a.clone(),
        b: problem.b.clone(),
        g,
        h,
        cone: ConeSpec {
            l: l0 + shift,
            soc: problem.cone.soc.clone(),
        },
        var_bounds: Vec::new(),
    };
    (mat, g_soc, g_orig)
}

/// Builds the reduced problem + original-index maps from the workspace.
/// Orthant survivors are emitted in deterministic original-row order, then
/// each SOC block keeps its frozen row order (tier-1 never drops SOC rows —
/// asserted).
fn finalize(ws: &Workspace) -> (ConicProblem, Maps) {
    let col_map: Vec<usize> = (0..ws.n).filter(|j| !ws.col_removed(*j)).collect();
    let mut new_col_of = vec![usize::MAX; ws.n];
    for (new, &orig) in col_map.iter().enumerate() {
        new_col_of[orig] = new;
    }
    let a_map: Vec<usize> = (0..ws.a_row_count())
        .filter(|i| !ws.a_row_removed(*i))
        .collect();
    let mut a_rows_t: Vec<usize> = Vec::new();
    let mut a_cols_t: Vec<usize> = Vec::new();
    let mut a_vals_t: Vec<f64> = Vec::new();
    let mut b: Vec<f64> = Vec::with_capacity(a_map.len());
    for (new, &i) in a_map.iter().enumerate() {
        for (col, v) in ws.a_active_pub(i) {
            if ws.col_removed(col) {
                continue;
            }
            a_rows_t.push(new);
            a_cols_t.push(new_col_of[col]);
            a_vals_t.push(v);
        }
        b.push(ws.a_row_rhs(i));
    }
    let mut orthants: Vec<usize> = (0..ws.g_row_count())
        .filter(|k| !ws.g_row_removed(*k) && ws.g_row_soc(*k).is_none())
        .collect();
    orthants.sort_by_key(|k| ws.g_row_orig(*k).unwrap_or(usize::MAX));
    let soc_blocks = ws.soc_block_dims();
    let mut block_rows: Vec<Vec<usize>> = vec![Vec::new(); soc_blocks.len()];
    for k in 0..ws.g_row_count() {
        if !ws.g_row_removed(k) {
            if let Some(bi) = ws.g_row_soc(k) {
                block_rows[bi].push(k);
            }
        }
    }
    for block in &mut block_rows {
        block.sort_by_key(|k| ws.g_row_orig(*k).unwrap_or(usize::MAX));
    }
    for (bi, block) in block_rows.iter().enumerate() {
        assert_eq!(
            block.len(),
            soc_blocks[bi],
            "tier-1 presolve must never drop SOC rows"
        );
    }
    let g_order: Vec<usize> = orthants
        .iter()
        .chain(block_rows.iter().flatten())
        .copied()
        .collect();
    // Surviving bound-layer entries re-materialize as orthant rows (the
    // tier-1 IPM consumes rows only; native bound consumption is tier 2).
    let bound_extras: Vec<(usize, postsolve::AbsorbedSnapshot)> = ws.active_bound_layers();
    let mut g_rows_t: Vec<usize> = Vec::new();
    let mut g_cols_t: Vec<usize> = Vec::new();
    let mut g_vals_t: Vec<f64> = Vec::new();
    // G row layout must match the positional cone spec: first the `l`
    // orthant rows (surviving workspace orthants, then the re-materialized
    // bound-layer rows), then the SOC blocks in original order.
    let n_orth = orthants.len() + bound_extras.len();
    let m_new = n_orth + soc_blocks.iter().sum::<usize>();
    let mut h = vec![0.0; m_new];
    let mut g_map: Vec<Option<usize>> = vec![None; m_new];
    let mut new_of: Vec<usize> = vec![usize::MAX; ws.g_row_count()];
    for (idx, &k) in orthants.iter().enumerate() {
        new_of[k] = idx;
    }
    for (bi, block) in block_rows.iter().enumerate() {
        for (t, &k) in block.iter().enumerate() {
            new_of[k] = n_orth + soc_blocks[..bi].iter().sum::<usize>() + t;
        }
    }
    for &k in &g_order {
        let new = new_of[k];
        h[new] = ws.g_row_rhs(k);
        g_map[new] = ws.g_row_orig(k);
        for (col, v) in ws.g_active_pub(k) {
            if ws.col_removed(col) {
                continue;
            }
            g_rows_t.push(new);
            g_cols_t.push(new_col_of[col]);
            g_vals_t.push(v);
        }
    }
    for (off, &(col, snap)) in bound_extras.iter().enumerate() {
        let new = orthants.len() + off;
        g_rows_t.push(new);
        g_cols_t.push(new_col_of[col]);
        g_vals_t.push(snap.g_coef);
        h[new] = snap.rhs;
        g_map[new] = snap.orig_row;
    }
    let a = otspot_num::sparse::CscMatrix::from_triplets(
        &a_rows_t,
        &a_cols_t,
        &a_vals_t,
        b.len(),
        col_map.len(),
    )
    .expect("finalize keeps A structurally valid");
    let g = otspot_num::sparse::CscMatrix::from_triplets(
        &g_rows_t,
        &g_cols_t,
        &g_vals_t,
        h.len(),
        col_map.len(),
    )
    .expect("finalize keeps G structurally valid");
    let reduced = ConicProblem {
        c: col_map.iter().map(|&j| ws.c_of(j)).collect(),
        a,
        b,
        g,
        h,
        cone: ConeSpec {
            l: orthants.len() + bound_extras.len(),
            soc: soc_blocks,
        },
        var_bounds: Vec::new(),
    };
    (
        reduced,
        Maps {
            col: col_map,
            a: a_map,
            g: g_map,
        },
    )
}
