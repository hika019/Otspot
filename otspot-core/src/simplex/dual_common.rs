//! Shared simplex primitives (primal + dual).
//!
//! `compute_dual_vars` / `compute_reduced_costs` were duplicated byte-for-byte
//! between `dual.rs` and `dual_advanced/core.rs`; the inline reduced-cost loop
//! in `primal.rs::revised_simplex_core` was a third copy. Three sites meant any
//! future correction (e.g. dual reconstruction drift) had to land thrice — the
//! kind of bug-magnet the DRY audit flagged.
//!
//! `_into` variants let the primal core reuse pre-allocated buffers across the
//! hot-loop; the allocating wrappers are kept for dual paths that refresh
//! reduced costs less frequently.
//!
//! `basic_obj` consolidates `c_B^T x_B`, repeated 19× across the simplex tree
//! for objective reporting on Optimal/Timeout/SingularBasis exits.

use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use otspot_num::linalg::parallelism::solver_thread_pool;
use otspot_num::sparse::{CscMatrix, SparseVec};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// Minimum priced-column nnz for one full reduced-cost sweep before the
/// column-parallel dispatch branch is taken.
///
/// Grounds (p2p7_P6 probe, `OTSPOT_P6_PROBE=1` direct measurement): the
/// legacy full sweep costs ~14.6 ms/pass on rail2586 (~25k nnz/pass,
/// ~1.7 ns/nnz once scattered `y` reads dominate) while a rayon task spawn
/// is single-digit µs. Below this threshold the dispatch is the same order
/// as the kernel itself — the exact regime where the P6 window-scan verdict
/// measured expected benefit ≤ 0. Small LPs (afiro: 48 nnz/pass) and window
/// sizes (d6cube: 3.3k nnz/call) stay strictly on the serial branch.
const PARALLEL_SCAN_MIN_NNZ: usize = 20_000;

/// Chunks created per worker thread by the parallel sweep branch.
///
/// Columns have highly uneven nnz (network LPs mix nnz=1 and nnz>500
/// columns), so one chunk per thread leaves stragglers. A 4× finer chunk
/// grid feeds rayon's work stealing at an order that absorbs the imbalance;
/// chunking never affects results because every column is written into its
/// own `rc_out` slot with an unchanged intra-column dot order.
const PARALLEL_SCAN_CHUNKS_PER_THREAD: usize = 4;

/// Test-only: number of times the parallel sweep branch actually dispatched.
/// Sentinels use it to prove (a) budget 1 never leaves the serial branch and
/// (b) the end-to-end plumbing really reaches the sweep with budget ≥ 2.
#[cfg(test)]
static RC_PARALLEL_DISPATCH_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
pub(crate) fn reset_rc_parallel_dispatch_count() {
    RC_PARALLEL_DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn rc_parallel_dispatch_count() -> u64 {
    RC_PARALLEL_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// y = B^{-T} c_B written into the caller's buffer. `y_out.len()` is the basis
/// dimension m; the caller owns the allocation so a hot loop can reuse it.
pub(super) fn compute_dual_vars_into(
    c: &[f64],
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    y_out: &mut [f64],
) {
    debug_assert_eq!(y_out.len(), basis.len());
    for (i, slot) in y_out.iter_mut().enumerate() {
        *slot = c[basis[i]];
    }
    basis_mgr.btran_dense(y_out);
}

/// y = B^{-T} c_B. Allocating wrapper; callers in cold paths (or those that
/// only need y once) use this. Hot loops should use `_into` + a reused buffer.
pub(super) fn compute_dual_vars(
    c: &[f64],
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    m: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    debug_assert_eq!(basis.len(), m);
    compute_dual_vars_into(c, basis_mgr, basis, &mut y);
    y
}

/// r_j = c_j − y^T a_j written into `rc_out`. Basic columns are zeroed; the
/// caller must pre-size `rc_out` to `n_price` and supply a `y_buf` of length m
/// (reused across iterations). `y_buf` is overwritten on every call.
///
/// `threads` is the `SolverOptions::threads` budget for the column sweep
/// after `y` is complete. Budget 1 (the default) runs the identical serial
/// loop; ≥ 2 may split whole columns across a dedicated pool — see
/// [`scan_reduced_costs`] for why any split stays bit-exact.
pub(super) fn compute_reduced_costs_into(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    basis: &[usize],
    y_buf: &mut [f64],
    rc_out: &mut [f64],
    threads: usize,
) {
    debug_assert_eq!(rc_out.len(), n_price);
    compute_dual_vars_into(c, basis_mgr, basis, y_buf);
    scan_reduced_costs(a, c, is_basic, n_price, y_buf, rc_out, threads);
}

/// The column sweep of `r_j = c_j − y^T a_j` for an already-computed `y`.
///
/// Serial/parallel split: `threads <= 1` or a sweep below
/// [`PARALLEL_SCAN_MIN_NNZ`] runs [`scan_reduced_cost_range`] over the whole
/// slice — byte-for-byte the loop this function has always run. Only a
/// dedicated-pool dispatch (never the global rayon pool, mirroring
/// `with_solver_pool`'s under-use-over-overrun fallback policy) may parallelize.
///
/// `#[inline(always)]` is load-bearing: without it the extracted sweep loop
/// cost the default threads=1 path ~+3% wall on rail507/osa-60/pds-30 (diag
/// A/B 77.7s → 80.0s with plain extraction; pds-30's status even flipped past
/// the bench deadline). Inlining restores the pre-extraction code shape in
/// the caller, which is the whole point of the "budget 1 = byte-identical,
/// speed-identical code path" contract.
#[inline(always)]
fn scan_reduced_costs(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    n_price: usize,
    y_buf: &[f64],
    rc_out: &mut [f64],
    threads: usize,
) {
    debug_assert_eq!(rc_out.len(), n_price);
    let scan_nnz = a.col_ptr()[n_price].saturating_sub(a.col_ptr()[0]);
    if threads > 1 && n_price > 1 && scan_nnz >= PARALLEL_SCAN_MIN_NNZ {
        if let Some(pool) = solver_thread_pool(threads) {
            pool.install(|| {
                let chunks = (threads * PARALLEL_SCAN_CHUNKS_PER_THREAD).min(n_price);
                let chunk_cols = n_price.div_ceil(chunks);
                rc_out
                    .par_chunks_mut(chunk_cols)
                    .enumerate()
                    .for_each(|(chunk_idx, rc_slice)| {
                        scan_reduced_cost_range(
                            a,
                            c,
                            is_basic,
                            y_buf,
                            rc_slice,
                            chunk_idx * chunk_cols,
                        );
                    });
            });
            #[cfg(test)]
            RC_PARALLEL_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    scan_reduced_cost_range(a, c, is_basic, y_buf, rc_out, 0);
}

/// Contiguous-column part of the sweep: writes `rc_out[0..len]` for global
/// columns `[col_base, col_base + len)`. This is THE loop — serial callers
/// pass the whole slice at `col_base = 0`, parallel callers pass disjoint
/// chunks. Bit-exactness of the split rests on two properties of this body:
/// each output slot is written exactly once, and the dot product for column
/// `j` accumulates `y[row_k] * val_k` in column-storage order, which chunk
/// boundaries cannot reorder. Any change to the accumulation order here
/// changes every LP trajectory — it is guarded by
/// `full_sweep_matches_reference_addition_order`.
///
/// `#[inline(always)]`: also required for the speed-identical default path —
/// see [`scan_reduced_costs`] on why the loop must end up inside the caller.
#[inline(always)]
fn scan_reduced_cost_range(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    y_buf: &[f64],
    rc_out: &mut [f64],
    col_base: usize,
) {
    for (offset, rc_slot) in rc_out.iter_mut().enumerate() {
        let j = col_base + offset;
        if is_basic[j] {
            *rc_slot = 0.0;
            continue;
        }
        let (rows, vals) = a.column(j);
        let mut ya = 0.0;
        for (k, &row) in rows.iter().enumerate() {
            ya += y_buf[row] * vals[k];
        }
        *rc_slot = c[j] - ya;
    }
}

/// r_j = c_j − y^T a_j with y = B^{-T} c_B. Basic columns are skipped
/// (r_j ≡ 0). Allocating wrapper used by dual paths that refresh r only after
/// a refactor. `threads` is forwarded to [`compute_reduced_costs_into`].
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_reduced_costs(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    basis: &[usize],
    threads: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    let mut reduced_costs = vec![0.0f64; n_price];
    compute_reduced_costs_into(
        a,
        c,
        basis_mgr,
        is_basic,
        n_price,
        basis,
        &mut y,
        &mut reduced_costs,
        threads,
    );
    reduced_costs
}

/// Standard-form dual infeasibility from reduced costs: maximum negative
/// reduced-cost violations over nonbasic priced columns. In the primal core all
/// nonbasic variables are at their lower bound, so dual feasibility is
/// `r_j >= 0`; the magnitude, not the count, is the progress signal.
pub(super) fn reduced_cost_dual_infeasibility(
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
) -> f64 {
    let limit = n_price.min(reduced_costs.len()).min(is_basic.len());
    let mut infeas = 0.0;
    for j in 0..limit {
        if is_basic[j] {
            continue;
        }
        let rc = reduced_costs[j];
        if !rc.is_finite() {
            return f64::INFINITY;
        }
        let viol = (-rc).max(0.0);
        if viol > infeas {
            infeas = viol;
        }
    }
    infeas
}

/// Convert a `SimplexOutcome` to a `SolverResult`.
///
/// Single implementation shared by all three simplex dispatch paths:
/// - warm-start dual (`dual.rs`): `dual_unbounded_is_infeasible = true`
/// - cold-start primal phase-II (`dual.rs`): `dual_unbounded_is_infeasible = false`
/// - advanced dual warm-start (`dual_advanced`): `dual_unbounded_is_infeasible = true`
///
/// The only semantic difference between callers is the `Unbounded` arm:
/// dual simplex yields dual-unbounded ⇒ primal-infeasible; primal simplex
/// yields a genuinely unbounded objective.
#[allow(clippy::too_many_arguments)]
pub(super) fn outcome_to_result(
    outcome: SimplexOutcome,
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    row_scale: &[f64],
    dual_unbounded_is_infeasible: bool,
    options: &crate::options::SolverOptions,
) -> SolverResult {
    match outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, basis, x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: basis.to_vec(),
                x_b: x_b.to_vec(),
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => {
            if dual_unbounded_is_infeasible {
                SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: f64::INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            } else {
                SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            }
        }
        SimplexOutcome::Timeout(obj) | SimplexOutcome::Stalled(obj) => {
            // Clock-recheck, not variant trust: `external_stop_requested` is
            // monotonic, so a genuine deadline/cancel stop still classifies as
            // Timeout; any internal dead-end that minted `Timeout` (or a future
            // unswept mint) is honestly downgraded to a stall status.
            let solution = extract_solution(sf, basis, x_b, col_scale);
            let status = super::stop_status(!solution.is_empty(), options);
            SolverResult {
                status,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

// Test-only observability for `lp_unbounded_ray_verified`'s two cancellation
// checkpoints (loop-top and pre-accept). Mirrors `presolve::qp_phase2`'s
// `PHASE2_STEPS_EXECUTED`/`PHASE2_CANCEL_AFTER_STEPS`/`PHASE2_CANCEL_SIGNAL`
// pattern: entirely `#[cfg(test)]` (definitions and every call site below),
// zero footprint in production builds.
//
// `cancel_flag` is monotonic (never reset once `true`), so a plain pass/fail
// on the function's own boolean return cannot distinguish which checkpoint
// caught it: whichever runs *after* the flip catches it regardless of which
// one(s) are present (Codex PR #31 re-review: reverting either checkpoint
// alone left both original sentinels passing, since the other, untouched
// checkpoint still caught the monotonic flag -- only reverting both failed).
// These counters make each checkpoint's own hit count independently
// observable, so a test can assert precisely how many times it ran.
#[cfg(test)]
thread_local! {
    static RAY_LOOP_TOP_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RAY_CANCEL_AFTER_LOOP_TOP_HITS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static RAY_PRE_ACCEPT_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RAY_CANCEL_AFTER_PRE_ACCEPT_HITS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static RAY_CANCEL_SIGNAL: std::sync::Arc<AtomicBool> =
        std::sync::Arc::new(AtomicBool::new(false));
    // Plain counter (no cancel-trigger side effect): records whether the
    // LU factorization + initial BTRAN setup ran at all, so a test can tell
    // "the entry check returned before setup" apart from "setup ran, then
    // iteration 0's own loop-top check caught a preset flag" -- both leave
    // `RAY_LOOP_TOP_HITS` at 0 (its recorder only fires *after* that check
    // passes), so it alone cannot distinguish them.
    static RAY_SETUP_REACHED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn test_record_ray_loop_top_hit() {
    let hits = RAY_LOOP_TOP_HITS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if RAY_CANCEL_AFTER_LOOP_TOP_HITS.with(std::cell::Cell::get) == Some(hits) {
        RAY_CANCEL_SIGNAL.with(|flag| flag.store(true, Ordering::Relaxed));
    }
}

#[cfg(test)]
fn test_record_ray_pre_accept_hit() {
    let hits = RAY_PRE_ACCEPT_HITS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if RAY_CANCEL_AFTER_PRE_ACCEPT_HITS.with(std::cell::Cell::get) == Some(hits) {
        RAY_CANCEL_SIGNAL.with(|flag| flag.store(true, Ordering::Relaxed));
    }
}

/// Verify an LP `Unbounded` exit against a re-derived recession ray (symmetric
/// to the Phase-I Farkas gate: unverified ray ⇒ honest Stalled).
///
/// Eta drift can falsely read `B⁻¹a_q ≤ 0`; this rebuilds a clean LU and
/// confirms column `q < n_enter` is improving (`r_q < −dual_tol`) AND unbounded:
///   - structural/slack rows: `(B⁻¹a_q)[i] ≤ ray_floor` (each finite UB is a
///     slack row, so a real UB limit surfaces as a positive component);
///   - artificial rows (`basis[i] ≥ n_enter`): `|(B⁻¹a_q)[i]| ≤ ray_floor`
///     (a `< 0` entry increases the artificial off 0 ⇒ not an original-LP ray).
///
/// `ray_floor = EPSILON·max(1,‖B⁻¹a_q‖∞)` — machine-noise scale, not `PIVOT_TOL`
/// (a real positive pivot is a leaving row; the looser tol leaks false Unbounded).
/// `n_enter` excludes Big-M artificials; pure-slack paths pass `n_enter = n_price`.
/// Empty witness ⇒ Timeout.
pub(super) fn lp_unbounded_ray_verified(
    a: &CscMatrix,
    basis: &[usize],
    c: &[f64],
    m: usize,
    n_price: usize,
    n_enter: usize,
    options: &SolverOptions,
) -> bool {
    // Entry check (Codex PR #31 re-review): `LuBasis::new_timed` only
    // consults `options.deadline`, not `cancel_flag`, and the BTRAN just
    // below runs unconditionally either way. Without this, a cancel_flag
    // preset before this function is ever called still pays for a full LU
    // factorization + BTRAN before the loop's own checks get a chance.
    if options.external_stop_requested() {
        return false;
    }
    let mut basis_mgr = match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(_) => return false,
    };
    let mut in_basis = vec![false; n_price];
    for &col in basis {
        if col < n_price {
            in_basis[col] = true;
        }
    }
    // y = B⁻ᵀ c_B
    let mut y: Vec<f64> = basis.iter().map(|&col| c[col]).collect();
    basis_mgr.btran_dense(&mut y);
    #[cfg(test)]
    RAY_SETUP_REACHED.with(|c| c.set(c.get() + 1));

    for q in 0..n_enter {
        // O(n_enter) FTRAN solves + O(nnz) ray checks: same cost profile as
        // `dual_advanced::phase1::farkas_infeasibility_certified`'s Strategy 2
        // per-row loop, and needs the identical per-iteration recheck (an
        // external stop must not wait out a loop whose own cost scales with
        // problem size).
        if options.external_stop_requested() {
            return false;
        }
        #[cfg(test)]
        test_record_ray_loop_top_hit();
        if in_basis[q] {
            continue;
        }
        let (rows, vals) = a.column(q);
        let rc = c[q]
            - rows
                .iter()
                .zip(vals.iter())
                .map(|(&r, &v)| v * y[r])
                .sum::<f64>();
        if rc >= -options.dual_tol {
            continue;
        }
        let mut d_sv = SparseVec::from_raw_parts(rows.to_vec(), vals.to_vec(), m);
        basis_mgr.ftran(&mut d_sv);
        // A recession ray needs every basic structural/slack component ≤ 0
        // (increasing x_q from lb=0, ub=∞ never drives a basic var below its lb).
        // The floor is machine-noise scale (`EPSILON·max(1,‖d‖∞)`), matching the
        // simplex's last-chance ratio test: any *real* positive pivot — even
        // below `PIVOT_TOL` (0 < d_i < 1e-8) — is a genuine leaving row, so the
        // direction is bounded, NOT a ray. `PIVOT_TOL` here would re-admit those
        // and leak a false-Unbounded. Basic artificials (basis[i] ≥ n_enter) need
        // the stricter |d_i| ≤ floor: d_i < 0 there *increases* the artificial
        // off 0, feasible only in the augmented system, not the original LP — not
        // a recession ray. Allowing it leaks a false-Unbounded whenever a
        // degenerate artificial lingers in the Phase-II basis (the symptom this
        // gate prevents).
        let d = d_sv.to_dense();
        let scale = d.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
        let ray_floor = f64::EPSILON * scale;
        let ray_ok = d.iter().enumerate().all(|(i, &di)| {
            if basis[i] >= n_enter {
                di.abs() <= ray_floor
            } else {
                di <= ray_floor
            }
        });
        if ray_ok {
            // An external stop requested while this FTRAN/ray check ran must
            // not let a coincidentally-already-computed verdict slip past it:
            // every caller unconditionally maps `true` here to a hard
            // `SolveStatus::Unbounded`, bypassing the honest Stalled/Timeout
            // path an external stop is supposed to take (same class of bug as
            // the Farkas gates' pre-`return true` recheck).
            #[cfg(test)]
            test_record_ray_pre_accept_hit();
            if options.external_stop_requested() {
                return false;
            }
            return true;
        }
    }
    false
}

/// c_B^T x_B = Σ c[basis[i]] · x_B[i]. Shared between primal/dual simplex for
/// objective reporting on Optimal / Timeout / SingularBasis exits.
pub(super) fn basic_obj(c: &[f64], basis: &[usize], x_b: &[f64]) -> f64 {
    debug_assert_eq!(basis.len(), x_b.len());
    basis.iter().zip(x_b.iter()).map(|(&j, &v)| c[j] * v).sum()
}

/// Verify primal feasibility on a fresh exact `x_b = B^{-1} b_rhs` before
/// minting the `Optimal` outcome for a dual-feasible candidate — mirrors
/// `primal::core::revised_simplex_core`'s pre-Optimal guard. Eta-file drift
/// accumulated across many incremental pivots can leave `x_b` reporting
/// every basic variable as (barely) feasible when a numerically clean
/// recomputation would show otherwise (bug-hunt P1). Shared by
/// `dual::dual_simplex_core` and `dual_advanced::core::
/// dual_simplex_core_advanced`, whose non-basic variables are always at
/// their lower bound (no per-variable upper bound / at_upper accounting —
/// that is `bounded_core::iterate`'s own concern).
pub(super) fn mint_optimal_after_fresh_reverify(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    b_rhs: &[f64],
    basis: &[usize],
    basis_mgr: &mut LuBasis,
    m: usize,
    options: &SolverOptions,
) -> SimplexOutcome {
    basis_mgr.force_refactor_timed(a, basis, options.deadline);
    if basis_mgr.refactor_failed {
        if basis_mgr.singular_basis {
            return SimplexOutcome::SingularBasis;
        }
        return SimplexOutcome::Timeout(basic_obj(c, basis, x_b));
    }
    x_b.copy_from_slice(b_rhs);
    basis_mgr.ftran_dense(x_b);
    for v in x_b.iter_mut() {
        if v.abs() < options.clamp_tol {
            *v = 0.0;
        }
    }
    let obj = basic_obj(c, basis, x_b);
    let min_basic = x_b.iter().copied().fold(f64::INFINITY, f64::min);
    if min_basic < -options.primal_tol {
        return SimplexOutcome::Stalled(obj);
    }
    SimplexOutcome::Optimal(obj, compute_dual_vars(c, basis_mgr, basis, m))
}

/// Anti-cycling: enter Bland mode after K = `(NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN)`
/// consecutive no-progress iterations.
pub(super) const NO_PROGRESS_TRIGGER_FACTOR: usize = 3;
pub(super) const NO_PROGRESS_MIN: usize = 100;

/// Relative improvement threshold for progress detection:
/// improvement is counted only when `best - current > |best| * NO_PROGRESS_REL_EPS`.
pub(super) const NO_PROGRESS_REL_EPS: f64 = 1e-12;

/// Returns `true` when `current` is strictly better than `best` by more than
/// the relative noise floor `best.abs().max(floor) * NO_PROGRESS_REL_EPS`.
///
/// `floor = 0.0`: noise floor scales with `|best|` (use in dual anti-cycling).
/// `floor = 1.0`: guards against noise resets when `best ≈ 0`, preventing
/// sub-eps improvements from counting as real progress (primal Phase I).
pub(super) fn made_progress_with_floor(best: f64, current: f64, floor: f64) -> bool {
    best - current > best.abs().max(floor) * NO_PROGRESS_REL_EPS
}

/// Periodic deadline-check interval inside the m-BTRAN gamma loop.
/// Checked every `GAMMA_DEADLINE_CHECK_INTERVAL` rows; large enough to
/// amortize the `Instant::now()` syscall but small enough to catch an
/// expired deadline before a full O(m²) budget overrun on large warm-starts.
const GAMMA_DEADLINE_CHECK_INTERVAL: usize = 10;

/// γ_i = ||(B^{-1})_{i,:}||² for each basis row i via m BTRANs.
///
/// Used by DSE leaving strategies at warm-start and after refactor to set
/// the exact initial weights. Cost O(m²); called once per warm-start solve
/// or refactor boundary.
///
/// Returns `None` when `deadline` is `Some` and the deadline expires inside
/// the loop (checked every `GAMMA_DEADLINE_CHECK_INTERVAL` rows), or when
/// `cancel_flag` is `Some` and the flag is set. Callers must propagate this
/// as a `Timeout` outcome so the solver stays within its time budget on large
/// warm-start solves.
pub(super) fn recompute_gamma_truth(
    basis_mgr: &mut LuBasis,
    m: usize,
    deadline: Option<std::time::Instant>,
    cancel_flag: Option<&AtomicBool>,
) -> Option<Vec<f64>> {
    let mut gamma_truth = vec![0.0f64; m];
    let mut e_i = vec![0.0f64; m];
    let mut rho_i = vec![0.0f64; m];
    for i in 0..m {
        if i % GAMMA_DEADLINE_CHECK_INTERVAL == 0 {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return None;
            }
            if cancel_flag.is_some_and(|f| f.load(Ordering::Relaxed)) {
                return None;
            }
        }
        e_i.iter_mut().for_each(|v| *v = 0.0);
        e_i[i] = 1.0;
        let mut sv = SparseVec::from_dense(&e_i);
        basis_mgr.btran(&mut sv);
        sv.to_dense_into(&mut rho_i);
        gamma_truth[i] = rho_i.iter().map(|&v| v * v).sum();
    }
    Some(gamma_truth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::LuBasis;
    use otspot_num::sparse::CscMatrix;

    /// P2 pin: user-facing classification must clock-recheck, never trust the
    /// `SimplexOutcome` variant. A `Timeout(_)` outcome reaching
    /// `outcome_to_result` with budget untouched (unswept internal dead-end
    /// mint, e.g. an exotic refactor failure) must NOT surface as
    /// `SolveStatus::Timeout`; with an incumbent it is `SuboptimalSolution`.
    /// `external_stop_requested` is monotonic, so a genuine deadline stop
    /// still classifies as Timeout (second assert).
    ///
    /// no-op proof: reverting `outcome_to_result` to variant-trust
    /// (`Timeout(obj) => status: SolveStatus::Timeout`) fails the first assert.
    #[test]
    fn outcome_to_result_clock_rechecks_timeout_variant() {
        use crate::problem::{LpProblem, SolveStatus};
        use crate::simplex::build_standard_form;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new(vec![1.0, 1.0], a, vec![4.0]).unwrap();
        let sf = build_standard_form(&lp);
        let basis = sf.initial_basis.clone();
        let x_b = sf.b.clone();
        let col_scale = vec![1.0; sf.n_total];
        let row_scale = vec![1.0; sf.m];

        let fresh = crate::options::SolverOptions::default();
        let r = outcome_to_result(
            SimplexOutcome::Timeout(0.0),
            &sf,
            &lp,
            &basis,
            &x_b,
            &col_scale,
            &row_scale,
            false,
            &fresh,
        );
        assert_eq!(
            r.status,
            SolveStatus::SuboptimalSolution,
            "Timeout variant with budget left must clock-recheck to a stall status"
        );

        let expired = crate::options::SolverOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..Default::default()
        };
        let r = outcome_to_result(
            SimplexOutcome::Timeout(0.0),
            &sf,
            &lp,
            &basis,
            &x_b,
            &col_scale,
            &row_scale,
            false,
            &expired,
        );
        assert_eq!(
            r.status,
            SolveStatus::Timeout,
            "expired deadline stays Timeout"
        );

        // Stalled variant with an expired clock is honestly Timeout as well
        // (the stop the caller observes IS the budget).
        let r = outcome_to_result(
            SimplexOutcome::Stalled(0.0),
            &sf,
            &lp,
            &basis,
            &x_b,
            &col_scale,
            &row_scale,
            false,
            &expired,
        );
        assert_eq!(r.status, SolveStatus::Timeout);
    }

    /// A = [I_m | extras]. Extra column (m + k) is a single +2.0 at row (k mod m)
    /// so r_{m+k} = c_{m+k} − 2·c[k mod m] under the identity basis.
    fn make_identity_plus(n: usize, m: usize) -> (CscMatrix, Vec<f64>, Vec<usize>) {
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for j in 0..m {
            rows.push(j);
            cols.push(j);
            vals.push(1.0);
        }
        for j in m..n {
            rows.push((j - m) % m);
            cols.push(j);
            vals.push(2.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let basis: Vec<usize> = (0..m).collect();
        let c: Vec<f64> = (0..n).map(|j| (j as f64) + 1.0).collect();
        (a, c, basis)
    }

    #[test]
    fn dual_vars_identity_basis_returns_c_b() {
        let m = 4;
        let (a, c, basis) = make_identity_plus(m + 3, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y = compute_dual_vars(&c, &mut bm, &basis, m);
        for i in 0..m {
            assert!(
                (y[i] - c[i]).abs() < 1e-12,
                "y[{}] = {} expected {}",
                i,
                y[i],
                c[i]
            );
        }
    }

    /// `_into` must agree byte-for-byte with the allocating wrapper. Multiple
    /// invocations into the same buffer must not leave stale state.
    #[test]
    fn dual_vars_into_matches_allocating_and_is_reuse_safe() {
        let m = 4;
        let (a, c, basis) = make_identity_plus(m + 3, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y_alloc = compute_dual_vars(&c, &mut bm, &basis, m);

        let mut y_into = vec![999.0f64; m];
        compute_dual_vars_into(&c, &mut bm, &basis, &mut y_into);
        for i in 0..m {
            assert!((y_into[i] - y_alloc[i]).abs() < 1e-14);
        }
        // Second call into the same buffer (stale sentinel left from above) must
        // still produce the canonical answer — covers buffer-reuse correctness.
        for slot in y_into.iter_mut() {
            *slot = -42.0;
        }
        compute_dual_vars_into(&c, &mut bm, &basis, &mut y_into);
        for i in 0..m {
            assert!((y_into[i] - y_alloc[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn reduced_costs_identity_basis_match_closed_form() {
        let m = 3;
        let n = m + 3;
        let (a, c, basis) = make_identity_plus(n, m);
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let r = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis, 1);

        for j in 0..m {
            assert_eq!(r[j], 0.0);
        }
        for j in m..n {
            let expected = c[j] - 2.0 * c[(j - m) % m];
            assert!(
                (r[j] - expected).abs() < 1e-12,
                "r[{}] = {} expected {}",
                j,
                r[j],
                expected
            );
        }
    }

    /// `_into` variant must produce identical reduced costs and must zero
    /// basic-column slots even when the caller's buffer holds stale non-zero
    /// values (primal hot loop reuses `rc_vec` across iterations).
    #[test]
    fn reduced_costs_into_matches_allocating_and_clears_basic_slots() {
        let m = 3;
        let n = m + 3;
        let (a, c, basis) = make_identity_plus(n, m);
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();

        let r_alloc = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis, 1);

        let mut y_buf = vec![0.0f64; m];
        // Pre-fill rc_out with garbage to ensure basic-slot zeroing happens.
        let mut rc_out = vec![123.456f64; n];
        compute_reduced_costs_into(
            &a,
            &c,
            &mut bm,
            &is_basic,
            n,
            &basis,
            &mut y_buf,
            &mut rc_out,
            1,
        );
        for j in 0..n {
            assert!((rc_out[j] - r_alloc[j]).abs() < 1e-14, "j={}", j);
        }
        for j in 0..m {
            assert_eq!(rc_out[j], 0.0, "basic slot {} not zeroed", j);
        }
    }

    #[test]
    fn reduced_costs_zero_cost_yields_zero_vector() {
        let m = 3;
        let n = m + 2;
        let (a, _c, basis) = make_identity_plus(n, m);
        let c = vec![0.0f64; n];
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let r = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis, 1);
        for &rj in &r {
            assert!(rj.abs() < 1e-14, "r = {:?} should be all zero", r);
        }
    }

    /// Permuted basis: confirm `c[basis[i]]` indexing is honoured end-to-end.
    /// With B = P (a permutation of I), y^T a_{basis[i]} must equal c[basis[i]].
    #[test]
    fn dual_vars_permuted_basis_uses_basis_indexing() {
        let m = 3;
        let n = m;
        let rows: Vec<usize> = (0..m).collect();
        let cols: Vec<usize> = (0..m).collect();
        let vals: Vec<f64> = vec![1.0; m];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let basis = vec![2usize, 0, 1];
        let c = vec![10.0, 20.0, 30.0];
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y = compute_dual_vars(&c, &mut bm, &basis, m);

        for i in 0..m {
            let (rs, vs) = a.column(basis[i]);
            let mut dot = 0.0;
            for (k, &row) in rs.iter().enumerate() {
                dot += y[row] * vs[k];
            }
            assert!(
                (dot - c[basis[i]]).abs() < 1e-12,
                "y^T a_{{basis[{}]}} = {} expected {}",
                i,
                dot,
                c[basis[i]]
            );
        }
    }

    #[test]
    fn basic_obj_identity_basis() {
        let m = 4;
        let (_a, c, basis) = make_identity_plus(m + 2, m);
        // x_B = [1, 2, 3, 4], c[basis[i]] = c[i] = i+1 → obj = 1·1 + 2·2 + 3·3 + 4·4 = 30
        let x_b = vec![1.0, 2.0, 3.0, 4.0];
        let obj = basic_obj(&c, &basis, &x_b);
        let expected: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
        assert!((obj - expected).abs() < 1e-14);
        assert!((obj - 30.0).abs() < 1e-14);
    }

    /// basic_obj on a permuted basis must respect `basis[i]` indexing.
    #[test]
    fn basic_obj_permuted_basis() {
        let basis = vec![2usize, 0, 1];
        let c = vec![10.0, 20.0, 30.0];
        let x_b = vec![1.0, 2.0, 3.0];
        // obj = c[2]·x_b[0] + c[0]·x_b[1] + c[1]·x_b[2] = 30+20+60 = 110
        let obj = basic_obj(&c, &basis, &x_b);
        assert!((obj - 110.0).abs() < 1e-14);
    }

    /// Empty basis edge case: must return 0.0, not panic.
    #[test]
    fn basic_obj_empty_basis() {
        let c = vec![1.0, 2.0, 3.0];
        let basis: Vec<usize> = vec![];
        let x_b: Vec<f64> = vec![];
        assert_eq!(basic_obj(&c, &basis, &x_b), 0.0);
    }

    /// Negative x_B (dual simplex pre-feasible state) must accumulate signs
    /// correctly — not abs-summed by accident.
    #[test]
    fn basic_obj_negative_x_b_signs_preserved() {
        let c = vec![1.0, 2.0, 3.0];
        let basis = vec![0usize, 1, 2];
        let x_b = vec![-1.0, 2.0, -3.0];
        // obj = -1 + 4 + -9 = -6
        assert!((basic_obj(&c, &basis, &x_b) - (-6.0)).abs() < 1e-14);
    }

    /// An already-expired deadline must short-circuit the BTRAN loop immediately
    /// (checked at i=0) and return `None`.
    ///
    /// no-op proof: removing the deadline check inside the loop (reverting
    /// `recompute_gamma_truth` to a `-> Vec<f64>` that ignores `deadline`) makes
    /// this test fail — it would return `Some(...)` instead of `None`.
    #[test]
    fn dse_expired_deadline_during_gamma_loop_returns_timeout() {
        let m = 4;
        let (a, _c, basis) = make_identity_plus(m + 2, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        // deadline 1 second in the past — already expired before the first BTRAN
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let result = recompute_gamma_truth(&mut bm, m, Some(past), None);
        assert!(
            result.is_none(),
            "expired deadline must short-circuit the BTRAN loop and return None; \
             got Some (deadline check is missing or not firing at i=0)"
        );
        // No-deadline call must always succeed.
        let result_no_dl = recompute_gamma_truth(&mut bm, m, None, None);
        assert!(
            result_no_dl.is_some(),
            "None deadline must always return Some"
        );
    }

    /// A pre-set cancel_flag must abort the BTRAN loop and return `None`.
    ///
    /// no-op proof: removing the cancel_flag check inside the loop makes this
    /// test return `Some(...)` instead of `None` → test fails.
    #[test]
    fn recompute_gamma_truth_cancellation_during_sweep() {
        use std::sync::Arc;
        let m = 4;
        let (a, _c, basis) = make_identity_plus(m + 2, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        // Pre-set cancel flag: already true before the sweep starts.
        let flag = Arc::new(AtomicBool::new(true));
        let result = recompute_gamma_truth(&mut bm, m, None, Some(&flag));
        assert!(
            result.is_none(),
            "pre-set cancel_flag must abort the BTRAN sweep and return None; \
             got Some (cancel_flag check is missing or not firing)"
        );
        // Cleared flag: sweep must complete.
        flag.store(false, Ordering::Relaxed);
        let result_ok = recompute_gamma_truth(&mut bm, m, None, Some(&flag));
        assert!(
            result_ok.is_some(),
            "cleared cancel_flag must allow sweep to complete"
        );
        // No flag at all: must always succeed.
        let result_no_flag = recompute_gamma_truth(&mut bm, m, None, None);
        assert!(
            result_no_flag.is_some(),
            "None cancel_flag must always return Some"
        );
    }

    /// `made_progress_with_floor(_, _, 0.0)` must return true iff improvement
    /// exceeds the relative noise floor `|best| * NO_PROGRESS_REL_EPS`.
    ///
    /// no-op proof: stubbing `made_progress_with_floor` to always return `false`
    /// makes the second and third assertions fail → test fails.
    #[test]
    fn made_progress_threshold_boundary() {
        // Clear improvement far above noise floor.
        assert!(
            made_progress_with_floor(1.0, 0.0, 0.0),
            "1.0 → 0.0 is clear improvement; must return true"
        );
        // Improvement exactly at the noise floor is NOT counted as progress.
        let eps = NO_PROGRESS_REL_EPS;
        let best = 1.0_f64;
        let at_boundary = best - best.abs() * eps;
        assert!(
            !made_progress_with_floor(best, at_boundary, 0.0),
            "improvement == eps * |best| is not strictly above threshold; must return false"
        );
        // Improvement just above the noise floor IS counted.
        let above_boundary = best - best.abs() * eps - 1e-15;
        assert!(
            made_progress_with_floor(best, above_boundary, 0.0),
            "improvement slightly above eps * |best| must return true"
        );
        // No improvement at all.
        assert!(
            !made_progress_with_floor(1.0, 1.0, 0.0),
            "no improvement must return false"
        );
        // best == 0, current == 0: no improvement → false.
        assert!(
            !made_progress_with_floor(0.0, 0.0, 0.0),
            "best == 0, current == 0: no improvement must return false"
        );
    }

    /// `made_progress_with_floor`: floor=1.0 keeps near-zero `best` from treating
    /// sub-eps values as noise-free progress; floor=0.0 uses `|best|` as the scale.
    ///
    /// no-op proof: stubbing `made_progress_with_floor` to always return `false`
    /// means `best_obj` in primal Phase I never updates → `OBJ_PROGRESS_RESET_COUNT`
    /// stays 0 → `b2_obj_progress_reset_fires_on_improving_objective` FAIL.
    #[test]
    fn made_progress_with_floor_protects_near_zero_best() {
        // floor=1.0: improvement < floor*eps = 1e-12 is rejected even when best ≈ 0.
        assert!(
            !made_progress_with_floor(0.0, -0.5e-12, 1.0),
            "near-zero best, floor=1.0: improvement below floor*eps must be rejected"
        );
        // best=0, current<0 (improvement), floor=0.0:
        //   0 - (-0.5e-12) = 0.5e-12 > 0.0 * eps = 0 → true.
        //   Note: best==0 does NOT imply false when floor=0; only best==current==0 gives false.
        assert!(
            made_progress_with_floor(0.0, -0.5e-12, 0.0),
            "best=0, current=-0.5e-12, floor=0: any positive improvement must return true"
        );
        // floor=1.0 with improvement above the floor*eps threshold.
        assert!(
            made_progress_with_floor(0.0, -1.5e-12, 1.0),
            "floor=1.0, improvement > floor*eps must pass"
        );
        // floor=0.0, typical values: improvement above |best|*eps.
        assert!(
            made_progress_with_floor(1.0, 0.0, 0.0),
            "floor=0.0, best=1.0, current=0.0: clear improvement must return true"
        );
    }

    /// LOAD-BEARING sentinel for the verified-Unbounded gate. Pins both directions:
    /// a genuine recession ray must verify (else true-Unbounded → false Timeout),
    /// and a bounded basis must NOT (else an eta-drift false-Unbounded survives).
    /// Also pins the `n_enter` exclusion (artificials never enter a ray).
    ///
    /// no-op proof: a gate hard-wired to `true` fails the bounded assertion; one
    /// hard-wired to `false` fails the genuine-ray assertion.
    #[test]
    fn lp_unbounded_ray_verified_distinguishes_genuine_bounded_and_artificial() {
        let opts = crate::options::SolverOptions::default();

        // Genuine unbounded: A=[[-1, 1]], basis={col1}, c=[-1,0]. col0 improving
        // (r0=-1) with B⁻¹a_0=[-1] ≤ 0 ⇒ no leaving row ⇒ unbounded ray.
        let a_unb = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, 1.0], 1, 2).unwrap();
        assert!(
            lp_unbounded_ray_verified(&a_unb, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "genuine unbounded ray must verify (else true-Unbounded demoted to Timeout)"
        );

        // Bounded: A=[[1, 1]], same basis/c. col0 improving but B⁻¹a_0=[1] > 0
        // ⇒ leaving row exists ⇒ no recession ray.
        let a_bnd = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_bnd, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "bounded LP must NOT verify a ray (else eta-drift false-Unbounded survives)"
        );

        // n_enter exclusion: A=[[-1, 1, -1]], basis={col1}, c=[0,0,-1], n_enter=2.
        // Only col2 is an unbounded direction, but it is ≥ n_enter (artificial) ⇒
        // excluded ⇒ no verified ray.
        let a_art =
            CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[-1.0, 1.0, -1.0], 1, 3).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_art, &[1], &[0.0, 0.0, -1.0], 1, 3, 2, &opts),
            "a column ≥ n_enter (artificial) must be excluded from the recession ray"
        );

        // CONCERN B verification: bounded direction with 0 < pivot < PIVOT_TOL must NOT verify.
        let a_smallpiv = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[5e-9, 1.0], 1, 2).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_smallpiv, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "CONCERN B: a small positive pivot (0<d<PIVOT_TOL) is a leaving row ⇒ bounded, not a ray"
        );
    }

    /// LOAD-BEARING sentinel for the basic-artificial ray check. A direction that
    /// drives no structural variable below its lower bound but *increases* a basic
    /// artificial off 0 is feasible only in the augmented system, not the original
    /// LP — so it is NOT a recession ray and must be rejected.
    ///
    /// Setup (m=2, cols [struct0 | struct1 | artificial2], n_enter=2, n_price=3):
    ///   A col0=(row0,1), col1=(row1,-1), col2=(row1,1); basis={col0, col2}=I, c=[0,-1,0].
    ///   Entering col1 (rc=-1) gives d=B⁻¹a_1=[0,-1]: row0 structural d=0 ≤ floor (ok),
    ///   row1 is the basic artificial (basis[1]=2 ≥ n_enter=2) with d=-1 < 0 ⇒ the
    ///   artificial grows to +t ⇒ not an original-LP ray.
    ///
    /// no-op proof: dropping the `basis[i] ≥ n_enter ⇒ |d_i| ≤ floor` branch (i.e.
    /// reverting to the plain `d_i ≤ floor` test) admits this direction as a ray and
    /// this assertion flips to a false-Unbounded → the test fails.
    #[test]
    fn lp_unbounded_ray_verified_rejects_ray_that_increases_basic_artificial() {
        let opts = crate::options::SolverOptions::default();
        let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 1, 2], &[1.0, -1.0, 1.0], 2, 3).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a, &[0, 2], &[0.0, -1.0, 0.0], 2, 3, 2, &opts),
            "a direction that increases a basic artificial off 0 must NOT verify as a recession ray"
        );
    }

    /// `lp_unbounded_ray_verified` had no cancellation check anywhere (Codex
    /// PR #31 review follow-up, same class as the Farkas gates already fixed
    /// in this crate): all 4 production callers (`dual.rs`, `dual_advanced/
    /// pipeline.rs`, `primal/mod.rs`'s `gate_phase2_unbounded`, `dual_advanced/
    /// phase1.rs`) unconditionally map a bare `true` here to `SolveStatus::
    /// Unbounded`.
    ///
    /// Deterministic, not a wall-clock race (Codex PR #31 re-review):
    /// `cancel_flag` is monotonic, so "did `verified` come back `false`"
    /// cannot distinguish which checkpoint caught it -- whichever runs
    /// after the flip catches it regardless of presence. Reverting either
    /// checkpoint alone left the original race-based sentinels passing
    /// (reviewer's 4 experiments); only reverting both failed. These two
    /// tests instead assert each checkpoint's own hit count (see
    /// `RAY_LOOP_TOP_HITS`/`RAY_PRE_ACCEPT_HITS`'s doc comment above).
    ///
    /// Targets the loop-top checkpoint. Fixture: `m = 1`, column 0 the
    /// identity basis, `DECOYS = 10` empty non-ray columns (`rc = 0 >=
    /// -dual_tol` skips each via `continue`, so `ray_ok` never computes and
    /// `RAY_PRE_ACCEPT_HITS` stays 0, isolating this test's signal to
    /// loop-top). `cancel_flag` flips right after loop-top hit
    /// `CANCEL_AFTER`, a mock-clock stand-in for a real expiring
    /// deadline/cancel mid-scan.
    ///
    /// Sentinel: reverting the loop-top check makes `RAY_LOOP_TOP_HITS`
    /// climb to all `DECOYS`, failing `hits == CANCEL_AFTER`.
    #[test]
    fn lp_unbounded_ray_verified_loop_top_check_stops_scan_at_cancellation_point() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        const DECOYS: usize = 10;
        const CANCEL_AFTER: usize = 3;

        // Column 0 = identity basis; columns 1..=DECOYS = empty decoys, no
        // genuine ray anywhere.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, DECOYS + 1).unwrap();
        let c = vec![0.0_f64; DECOYS + 1];

        RAY_LOOP_TOP_HITS.with(|c| c.set(0));
        RAY_PRE_ACCEPT_HITS.with(|c| c.set(0));
        RAY_CANCEL_AFTER_LOOP_TOP_HITS.with(|c| c.set(Some(CANCEL_AFTER)));
        RAY_CANCEL_AFTER_PRE_ACCEPT_HITS.with(|c| c.set(None));
        RAY_CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

        let opts = RAY_CANCEL_SIGNAL.with(|flag| SolverOptions {
            cancel_flag: Some(Arc::clone(flag)),
            ..SolverOptions::default()
        });
        let verified = lp_unbounded_ray_verified(&a, &[0], &c, 1, DECOYS + 1, DECOYS + 1, &opts);

        RAY_CANCEL_AFTER_LOOP_TOP_HITS.with(|c| c.set(None));
        let loop_top_hits = RAY_LOOP_TOP_HITS.with(std::cell::Cell::get);
        let pre_accept_hits = RAY_PRE_ACCEPT_HITS.with(std::cell::Cell::get);

        assert!(!verified, "no genuine ray exists in this fixture");
        assert_eq!(
            pre_accept_hits, 0,
            "no column here ever satisfies ray_ok, so the pre-accept checkpoint \
             must never fire -- confirms this test's signal is isolated to the \
             loop-top checkpoint"
        );
        assert_eq!(
            loop_top_hits,
            CANCEL_AFTER,
            "cancel_flag flips right after loop-top hit {CANCEL_AFTER}, so the \
             loop-top check must stop the scan there (not continue through the \
             remaining {} decoys), got {loop_top_hits} hits",
            DECOYS - CANCEL_AFTER
        );
    }

    /// Companion to the loop-top sentinel above, targeting the pre-accept
    /// checkpoint specifically: a cancellation arriving exactly when
    /// `ray_ok` first becomes `true` for some column, before the function
    /// commits to `return true` for it.
    ///
    /// Fixture: `m = 1`, column 0 the identity basis, column 1 a genuine ray
    /// (`a[0][1] = -1`, `c[1] = -1`, same shape as
    /// `lp_unbounded_ray_verified_distinguishes_genuine_bounded_and_
    /// artificial`'s minimal case) -- reached immediately after column 0
    /// (`in_basis[0]` skips it before any FTRAN). `RAY_CANCEL_AFTER_PRE_
    /// ACCEPT_HITS = Some(1)` flips `cancel_flag` as a side effect of
    /// `ray_ok` becoming `true` for the *first* (only) candidate, landing
    /// squarely between that candidate's own loop-top check (already passed,
    /// flag still `false` then) and the accept decision.
    ///
    /// Sentinel: reverting the `if options.external_stop_requested() {
    /// return false; }` right before `return true` (after `ray_ok` is
    /// computed) makes `verified` come back `true` instead of `false` --
    /// `RAY_PRE_ACCEPT_HITS` reaching 1 is unaffected either way (it's
    /// recorded before the reverted line), so only the boolean discriminates
    /// this specific checkpoint, which is exactly what reverting removes.
    #[test]
    fn lp_unbounded_ray_verified_pre_accept_check_rejects_ray_at_cancellation_point() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, -1.0], 1, 2).unwrap();
        let c = vec![0.0_f64, -1.0];

        RAY_LOOP_TOP_HITS.with(|c| c.set(0));
        RAY_PRE_ACCEPT_HITS.with(|c| c.set(0));
        RAY_CANCEL_AFTER_LOOP_TOP_HITS.with(|c| c.set(None));
        RAY_CANCEL_AFTER_PRE_ACCEPT_HITS.with(|c| c.set(Some(1)));
        RAY_CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

        let opts = RAY_CANCEL_SIGNAL.with(|flag| SolverOptions {
            cancel_flag: Some(Arc::clone(flag)),
            ..SolverOptions::default()
        });
        let verified = lp_unbounded_ray_verified(&a, &[0], &c, 1, 2, 2, &opts);

        RAY_CANCEL_AFTER_PRE_ACCEPT_HITS.with(|c| c.set(None));
        let loop_top_hits = RAY_LOOP_TOP_HITS.with(std::cell::Cell::get);
        let pre_accept_hits = RAY_PRE_ACCEPT_HITS.with(std::cell::Cell::get);

        assert_eq!(
            pre_accept_hits, 1,
            "column 1's ray_ok must genuinely evaluate true (reaching the \
             pre-accept checkpoint) for this test to exercise anything -- \
             confirms the fixture's premise, independent of cancellation"
        );
        assert_eq!(
            loop_top_hits, 2,
            "both column 0 (in_basis, skipped) and column 1 (the ray) must \
             reach the loop-top checkpoint normally before cancel_flag flips -- \
             confirms the flip lands at the pre-accept checkpoint specifically, \
             not any loop-top check (RAY_CANCEL_AFTER_LOOP_TOP_HITS is unset)"
        );
        assert!(
            !verified,
            "cancel_flag flips as a side effect of ray_ok becoming true for \
             column 1's own pre-accept checkpoint -- must prevent accepting \
             this otherwise-valid ray"
        );
    }

    /// Companion to the two checkpoint sentinels above, for the entry check
    /// (Codex PR #31 re-review): `LuBasis::new_timed` only consults
    /// `options.deadline`, not `cancel_flag`, so a preset `cancel_flag` (no
    /// deadline set) previously still paid for a full LU factorization and
    /// the `y = B⁻ᵀc_B` BTRAN before the loop's own checks got a chance.
    ///
    /// `RAY_SETUP_REACHED`, not `RAY_LOOP_TOP_HITS`, is the discriminator:
    /// iteration 0's own loop-top check *also* catches a preset flag (this
    /// fixture's single non-basic column is q=0), so `RAY_LOOP_TOP_HITS`
    /// stays 0 whether the entry check runs or not -- it cannot tell "entry
    /// check returned early" apart from "setup ran, then iteration 0 caught
    /// it anyway". `RAY_SETUP_REACHED` only increments *after* the LU
    /// factorization and initial BTRAN both complete, so it is 0 only when
    /// the entry check actually prevented that setup from running at all.
    ///
    /// Sentinel: reverting the new entry check makes `verified` come back
    /// `true` (this fixture's column 1 is a genuine ray) and
    /// `RAY_SETUP_REACHED` read 1, not 0.
    #[test]
    fn lp_unbounded_ray_verified_entry_check_skips_lu_setup_when_preset() {
        use std::sync::Arc;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, -1.0], 1, 2).unwrap();
        let c = vec![0.0_f64, -1.0];

        RAY_SETUP_REACHED.with(|c| c.set(0));

        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            ..SolverOptions::default()
        };
        let verified = lp_unbounded_ray_verified(&a, &[0], &c, 1, 2, 2, &opts);

        let setup_reached = RAY_SETUP_REACHED.with(std::cell::Cell::get);
        assert_eq!(
            setup_reached, 0,
            "preset cancel_flag=true must return from the entry check before \
             LuBasis::new_timed/btran_dense ever run -- `RAY_SETUP_REACHED` is \
             the discriminator here, not `RAY_LOOP_TOP_HITS`: iteration 0's \
             own loop-top check *also* catches a preset flag and would leave \
             RAY_LOOP_TOP_HITS at 0 even with the entry check reverted (setup \
             would have run to completion first), got {setup_reached}"
        );
        assert!(
            !verified,
            "preset cancel_flag=true must reject this otherwise-genuine ray"
        );
    }

    // ====================================================================
    // wave-2 (perf/w2-lp-fullsweep-par): column-parallel full sweep tests
    // ====================================================================

    /// Deterministic xorshift64* LCG — fixtures must reproduce byte-for-byte
    /// on every run so bitwise sentinels are stable across machines.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
        fn signed_unit(&mut self) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            if self.next_u64() & 1 == 0 {
                0.5 + u
            } else {
                -(0.5 + u)
            }
        }
    }

    /// Random standard-form-shaped sweep fixture: identity basis (cols
    /// `0..m` basic), nonbasic cols `m..n` with random support, plus
    /// deliberately pathological shapes — the random block never touches
    /// row `m - 1` (that row is covered by its own basis column only), the
    /// last `n_empty_cols` nonbasic columns have no entries at all (rc must
    /// be exactly `c[j]`), and every column pair `(10k, 10k+1)` above
    /// `dup_first` duplicates support + values with equal cost (Dantzig
    /// ties, index order decides).
    fn make_sweep_fixture(
        seed: u64,
        m: usize,
        n: usize,
        n_empty_cols: usize,
        max_col_nnz: usize,
    ) -> (CscMatrix, Vec<f64>, Vec<usize>, Vec<bool>) {
        let mut rng = Lcg(seed);
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for j in 0..m {
            rows.push(j);
            cols.push(j);
            vals.push(1.0);
        }
        let live_rows = m - 1; // random blocks never touch row m-1
        let dup_first = m + 10;
        let mut j = m;
        while j < n - n_empty_cols {
            let duplicate = j >= dup_first && j % 10 == 1;
            if duplicate {
                // Exact copy of column j-1's stored support and values.
                let prev_start = cols.iter().position(|&col| col == j - 1).unwrap();
                let prev_end = cols.iter().rposition(|&col| col == j - 1).unwrap() + 1;
                for k in prev_start..prev_end {
                    let r = rows[k];
                    let v = vals[k];
                    rows.push(r);
                    cols.push(j);
                    vals.push(v);
                }
                j += 1;
                continue;
            }
            let nnz = 1 + rng.below(max_col_nnz);
            let mut picked: Vec<usize> = Vec::with_capacity(nnz);
            for _ in 0..nnz {
                let r = rng.below(live_rows);
                if !picked.contains(&r) {
                    picked.push(r);
                    rows.push(r);
                    cols.push(j);
                    vals.push(rng.signed_unit());
                }
            }
            j += 1;
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let basis: Vec<usize> = (0..m).collect();
        // Pair members (10k, 10k+1) share the cost formula (k/10 is equal
        // for both), so a duplicated support yields an exact rc tie.
        let c: Vec<f64> = (0..n)
            .map(|k| {
                if k >= dup_first && k % 10 <= 1 {
                    0.25 * ((k / 10) as f64 + 1.0)
                } else {
                    rng.signed_unit()
                }
            })
            .collect();
        let is_basic: Vec<bool> = (0..n).map(|k| k < m).collect();
        (a, c, basis, is_basic)
    }

    fn bits_eq(lhs: &[f64], rhs: &[f64]) -> bool {
        lhs.len() == rhs.len()
            && lhs
                .iter()
                .zip(rhs.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The full sweep under any parallel budget must equal the serial full
    /// sweep bit-for-bit, on a gate-open fixture (`min_nnz` far above
    /// `PARALLEL_SCAN_MIN_NNZ` by construction: 4000 cols × avg ~6 nnz) that
    /// contains empty rows, empty columns, and tied duplicate columns.
    ///
    /// The oracle is the serial full sweep itself: per-column outputs have
    /// no cross-column dependence and each intra-column dot keeps column
    /// storage order, so a correct split cannot change a single bit.
    ///
    /// Sentinel: giving the parallel branch a wrong chunk base offset
    /// (`chunk_idx` instead of `chunk_idx * chunk_cols` in
    /// `scan_reduced_costs`) or reordering the accumulation in
    /// `scan_reduced_cost_range` (`rows.iter().rev()`) fails the bitwise
    /// comparison for odd-size chunks / tie fixtures.
    #[test]
    fn full_sweep_parallel_matches_serial_bitwise_across_budgets() {
        let m = 96usize;
        let n = 4_200usize;
        let (a, c, basis, is_basic) = make_sweep_fixture(0xC0FFEE, m, n, 37, 12);
        assert!(
            a.nnz() >= PARALLEL_SCAN_MIN_NNZ,
            "fixture premise: gate must be open, nnz={}",
            a.nnz()
        );
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();

        let mut y = vec![0.0f64; m];
        let mut rc_serial = vec![0.0f64; n];
        compute_reduced_costs_into(
            &a,
            &c,
            &mut bm,
            &is_basic,
            n,
            &basis,
            &mut y,
            &mut rc_serial,
            1,
        );

        for budget in [2usize, 3, 8] {
            reset_rc_parallel_dispatch_count();
            let mut y2 = vec![0.0f64; m];
            let mut rc_par = vec![-1.0f64; n];
            compute_reduced_costs_into(
                &a,
                &c,
                &mut bm,
                &is_basic,
                n,
                &basis,
                &mut y2,
                &mut rc_par,
                budget,
            );
            assert!(
                rc_parallel_dispatch_count() > 0,
                "budget={budget} on a gate-open fixture must exercise the \
                 parallel branch (plumbing sentinel; budget 1 must NOT)"
            );
            assert!(bits_eq(&y, &y2), "budget={budget}: y diverged");
            assert!(
                bits_eq(&rc_serial, &rc_par),
                "budget={budget}: parallel sweep differs from serial bitwise"
            );
        }
    }

    /// Independent recomputation oracle: the expected `r_j` is rebuilt in
    /// the test with a plain dense loop over `a.column(j)` in storage order,
    /// NOT by calling the production scan. This pins the accumulation order
    /// itself — chunk plumbing that merely permutes columns but changes the
    /// dot order would pass the serial-vs-parallel test above only if both
    /// branches shared the new order, and would fail here.
    #[test]
    fn full_sweep_matches_reference_addition_order() {
        let m = 96usize;
        let n = 4_200usize;
        let (a, c, basis, is_basic) = make_sweep_fixture(0xBADF00D, m, n, 15, 12);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y = compute_dual_vars(&c, &mut bm, &basis, m);

        let mut y_buf = y.clone();
        let mut rc_par = vec![0.0f64; n];
        compute_reduced_costs_into(
            &a,
            &c,
            &mut bm,
            &is_basic,
            n,
            &basis,
            &mut y_buf,
            &mut rc_par,
            4,
        );

        for j in 0..n {
            let expected = if is_basic[j] {
                0.0
            } else {
                let (rows, vals) = a.column(j);
                let mut ya = 0.0f64;
                for (k, &row) in rows.iter().enumerate() {
                    ya += y[row] * vals[k];
                }
                c[j] - ya
            };
            assert_eq!(
                rc_par[j].to_bits(),
                expected.to_bits(),
                "j={j} parallel rc {} != reference {} (bitwise)",
                rc_par[j],
                expected
            );
        }
        // Pathological shapes really are in the fixture: empty columns must
        // return c[j] exactly, and row m-1 must never appear in any column.
        for j in n - 15..n {
            assert!(a.column(j).0.is_empty(), "col {j} should be empty");
            assert_eq!(rc_par[j], c[j], "empty col rc must be c[j]");
        }
    }

    /// Non-finite `y` (NaN / ±inf from numerical damage) must propagate
    /// through the parallel sweep exactly as through the serial one, down to
    /// NaN payload bits. `y_buf` is injected directly via the internal scan
    /// entry so the fixture does not need a numerically-singular basis.
    #[test]
    fn full_sweep_nonfinite_y_matches_serial_bitwise() {
        let m = 96usize;
        let n = 4_200usize;
        let (a, c, _basis, is_basic) = make_sweep_fixture(0x5EED, m, n, 21, 12);
        let mut y_bad = vec![0.0f64; m];
        y_bad[3] = f64::NAN;
        y_bad[7] = f64::INFINITY;
        y_bad[11] = f64::NEG_INFINITY;
        y_bad[5] = -0.0;
        // Row m-1 is touched only by its own (basic) identity column: garbage
        // there must NOT leak into any nonbasic rc through the split.
        y_bad[m - 1] = f64::NAN;

        let mut rc_serial = vec![0.0f64; n];
        scan_reduced_costs(&a, &c, &is_basic, n, &y_bad, &mut rc_serial, 1);
        reset_rc_parallel_dispatch_count();
        let mut rc_par = vec![0.0f64; n];
        scan_reduced_costs(&a, &c, &is_basic, n, &y_bad, &mut rc_par, 2);
        assert!(rc_parallel_dispatch_count() > 0);
        assert!(
            bits_eq(&rc_serial, &rc_par),
            "non-finite y must propagate identically (NaN payloads included)"
        );
        // Columns touching row 3 (NaN row) must actually be NaN — otherwise
        // the fixture injects damage no column sees and the test proves 0.
        let touches_nan_row = (0..n).any(|j| a.column(j).0.contains(&3));
        assert!(touches_nan_row, "fixture premise: NaN row must be hit");
    }

    /// Gate contract: budget 1 never dispatches regardless of size; budgets
    /// ≥ 2 below `PARALLEL_SCAN_MIN_NNZ` never dispatch either; the same
    /// budget on a gate-open sweep does dispatch.
    #[test]
    fn sweep_gate_keeps_small_and_default_budget_scans_serial() {
        let m = 48usize;
        let n_small = 300usize; // ~3 nnz/col: far below the gate
        let (a_small, c_small, basis_small, is_basic_small) =
            make_sweep_fixture(0x1234, m, n_small, 5, 4);
        let mut bm_small = LuBasis::new(&a_small, &basis_small, 32).unwrap();
        assert!(
            a_small.nnz() < PARALLEL_SCAN_MIN_NNZ,
            "fixture premise: small sweep below gate, nnz={}",
            a_small.nnz()
        );

        reset_rc_parallel_dispatch_count();
        let mut y = vec![0.0f64; m];
        let mut rc = vec![0.0f64; n_small];
        compute_reduced_costs_into(
            &a_small,
            &c_small,
            &mut bm_small,
            &is_basic_small,
            n_small,
            &basis_small,
            &mut y,
            &mut rc,
            8,
        );
        assert_eq!(
            rc_parallel_dispatch_count(),
            0,
            "sub-gate sweeps must stay serial even with budget 8"
        );

        let (a_big, c_big, basis_big, is_basic_big) = make_sweep_fixture(0x1235, m, 4_200, 5, 12);
        let mut bm_big = LuBasis::new(&a_big, &basis_big, 32).unwrap();
        reset_rc_parallel_dispatch_count();
        let mut rc_big = vec![0.0f64; 4_200];
        compute_reduced_costs_into(
            &a_big,
            &c_big,
            &mut bm_big,
            &is_basic_big,
            4_200,
            &basis_big,
            &mut y,
            &mut rc_big,
            1,
        );
        assert_eq!(
            rc_parallel_dispatch_count(),
            0,
            "default budget 1 must run the serial branch even gate-open \
             (no-op proof: swapping the `threads > 1` condition to `> 0` \
             would make the default path touch the pool)"
        );
    }

    /// End-to-end plumbing proof + trajectory bit-identity: the published
    /// result of a real `solve` must be bitwise identical between budgets
    /// 1/2/8 on the primal core (full sweep every iteration, gate open at
    /// this size) AND on the default Auto route. For the primal core the
    /// parallel branch must actually have been reached (dispatch counter) —
    /// otherwise the budget never left the QP/MIP paths and the feature is
    /// a silent no-op. Auto is only held to result identity: its window
    /// pricing is deliberately NOT parallelized (p2p7_P6 verdict).
    ///
    /// Sentinel: passing a hard-coded `1` instead of `options.threads` at
    /// the primal call site keeps results equal but fails the dispatch-count
    /// assertion.
    #[test]
    fn solve_lp_parallel_budget_matches_serial_publication_bitwise() {
        use crate::options::SimplexMethod;

        // Priced nnz ≈ 400 rows × 60 entries + 400 slacks > gate.
        let lp = make_random_lp_for_route(0xA11CE, 400, 800, 60);
        let mut results = Vec::new();
        for (method, budget) in [
            (SimplexMethod::Primal, 1usize),
            (SimplexMethod::Primal, 2),
            (SimplexMethod::Primal, 8),
            (SimplexMethod::Auto, 1),
            (SimplexMethod::Auto, 2),
        ] {
            let opts = SolverOptions {
                presolve: false,
                simplex_method: method,
                threads: budget,
                max_iters: Some(3_000),
                ..Default::default()
            };
            reset_rc_parallel_dispatch_count();
            let r = crate::lp::solve_lp_with(&lp, &opts);
            if matches!(method, SimplexMethod::Primal) {
                if budget > 1 {
                    assert!(
                        rc_parallel_dispatch_count() > 0,
                        "{method:?} budget={budget}: plumbing dead — the LP \
                         route never reached the parallel sweep branch"
                    );
                } else {
                    assert_eq!(rc_parallel_dispatch_count(), 0);
                }
            }
            results.push((method, budget, r));
        }
        // Identity contract is WITHIN a method (routes legitimately differ
        // from each other; budgets must not).
        for method in [SimplexMethod::Primal, SimplexMethod::Auto] {
            let group: Vec<_> = results.iter().filter(|(m, _, _)| *m == method).collect();
            let reference = &group[0].2;
            for (_, budget, r) in &group[1..] {
                assert_eq!(r.status, reference.status, "{method:?}/{budget}");
                assert_eq!(r.iterations, reference.iterations, "{method:?}/{budget}");
                assert_eq!(
                    r.objective.to_bits(),
                    reference.objective.to_bits(),
                    "{method:?}/budget={budget}: objective diverged"
                );
                assert!(
                    bits_eq(&r.solution, &reference.solution),
                    "{method:?}/budget={budget}: solution diverged"
                );
            }
        }
    }

    /// `cancel_flag` must stop a budget-2 solve exactly like a budget-1
    /// solve: the parallel sweep introduced no un-cancellable region and no
    /// hang (preset flag → both stop at the first loop-top check).
    #[test]
    fn solve_lp_parallel_budget_honors_preset_cancel_flag() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let lp = make_random_lp_for_route(0xBEEF, 400, 800, 60);
        for budget in [1usize, 2] {
            let opts = SolverOptions {
                presolve: false,
                threads: budget,
                cancel_flag: Some(Arc::new(AtomicBool::new(true))),
                ..Default::default()
            };
            let r = crate::lp::solve_lp_with(&lp, &opts);
            assert_eq!(
                r.status,
                crate::problem::SolveStatus::Timeout,
                "budget={budget} preset cancel must Timeout"
            );
        }
    }

    /// Preset `deadline` + budget 2 must report Timeout (the pool must not
    /// keep the solve alive past the external budget).
    #[test]
    fn solve_lp_parallel_budget_honors_expired_deadline() {
        let lp = make_random_lp_for_route(0xDEAD, 400, 800, 60);
        let opts = SolverOptions {
            presolve: false,
            threads: 2,
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..Default::default()
        };
        let r = crate::lp::solve_lp_with(&lp, &opts);
        assert_eq!(
            r.status,
            crate::problem::SolveStatus::Timeout,
            "expired deadline must Timeout at budget 2"
        );
    }

    /// Random-but-deterministic LP big enough that its standard-form priced
    /// matrix opens the parallel gate (`m * density` entries plus slacks),
    /// with mixed-sign costs so Phase II really pivots (a positive-cost LP
    /// would terminate at the zero BFS with a single sweep).
    fn make_random_lp_for_route(
        seed: u64,
        m: usize,
        n: usize,
        density: usize,
    ) -> crate::problem::LpProblem {
        use crate::problem::ConstraintType;
        let mut rng = Lcg(seed);
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for _ in 0..density {
                let j = rng.below(n);
                rows.push(i);
                cols.push(j);
                vals.push(rng.signed_unit());
            }
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let c: Vec<f64> = (0..n).map(|_| rng.signed_unit()).collect();
        let b: Vec<f64> = (0..m).map(|_| 1.0 + rng.below(50) as f64).collect();
        crate::problem::LpProblem::new_general(
            c,
            a,
            b,
            vec![ConstraintType::Le; m],
            vec![(0.0, 10.0); n],
            None,
        )
        .unwrap()
    }
}
