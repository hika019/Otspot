//! Revised simplex core iteration loop.

use super::super::dual_common::{
    basic_obj, compute_reduced_costs_into, made_progress_with_floor,
    reduced_cost_dual_infeasibility, NO_PROGRESS_MIN,
};
use super::super::pricing::PricingStrategy;
use super::super::trace::IterTrace;
use super::super::SimplexOutcome;
use super::ratio_test::select_leaving_feasibility_preserving;
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::qp::certificate::LP_CERT_TOL;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{PIVOT_STABILITY_THRESHOLD, PIVOT_TOL};
use std::sync::atomic::Ordering;

fn cleanup_trace_enabled() -> bool {
    std::env::var_os("OTSPOT_CLEANUP_TRACE_DETAIL").is_some()
}

fn cleanup_trace_every() -> usize {
    std::env::var("OTSPOT_CLEANUP_TRACE_EVERY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1000)
}

fn top_reduced_cost_violation(
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
) -> Option<(usize, f64)> {
    let limit = n_price.min(reduced_costs.len()).min(is_basic.len());
    let mut best: Option<(usize, f64)> = None;
    for j in 0..limit {
        if is_basic[j] {
            continue;
        }
        let rc = reduced_costs[j];
        if !rc.is_finite() {
            return Some((j, rc));
        }
        let viol = (-rc).max(0.0);
        if best.is_none_or(|(_, best_viol)| viol > best_viol) {
            best = Some((j, viol));
        }
    }
    best
}

fn made_cleanup_df_progress(best: f64, current: f64, target_df: f64) -> bool {
    best - current > best.abs().max(target_df) * target_df
}

/// Primal Phase I cycling early-bail (klein3 origin)。`cold_start_dual` の
/// Primal Phase I は Bland switch を持たず無限 cycle で half-deadline を焼く。
/// `Timeout` 早期 return で Big-M (`dual_simplex_core_advanced`、Bland + lex
/// perturbation あり) に残時間を譲る。
///
/// `K = max(BAIL_TRIGGER_FACTOR · m, BAIL_TRIGGER_MIN)`、AND 条件で発火:
/// (1) Phase I obj `cᵀx_B` が K 連続未改善 + (2) pivot step ≈ 0 が K' 連続。
/// AND が真 cycling (klein3) と slow-but-progressing (forplan) を切り分ける —
/// forplan は step > 0 で counter reset、klein3 は step ≈ 0 で両 counter trip。
///
/// `enable_phase1_cycling_bail` gate: Primal Phase I のみ `true`、Phase II は
/// obj plateau が optimum 近接の signal なので `false`。
const BAIL_TRIGGER_FACTOR: usize = 10;
const BAIL_TRIGGER_MIN: usize = 5_000;
/// Step-plateau threshold K'. Set to K / `STEP_BAIL_RATIO` so a single
/// non-degenerate pivot within any K'-iter window refutes cycling. Smaller
/// than K because step ≈ 0 is a stronger per-iter signature than obj
/// plateau (which can also come from f64 noise on real decrements), so
/// fewer consecutive occurrences are required.
const STEP_BAIL_RATIO: usize = 10;

/// Revised simplex core with on-demand LU refactor.
///
/// `enable_phase1_cycling_bail` arms the obj+step plateau early-bail; pass
/// `true` only from Primal Phase I.
///
/// `art_threshold = Some(t)` enables Phase I artificial leaving preference:
/// basis columns `>= t` leave first within the Harris tie-band. Pass `None` from
/// Phase II / non-artificial solves.
///
/// `enable_cleanup_stall_bail` is for crossover cleanup only. Shared primal
/// solves must keep searching because a flat Phase II objective can mean "near
/// optimum". Cleanup may keep the best recovered vertex when degenerate pivots
/// stop improving reduced-cost dual infeasibility; Phase I artificial removal
/// also counts as progress.
///
/// `cleanup_target_df` restores the best certificate seen and returns once the
/// target is met; ordinary simplex callers pass `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn revised_simplex_core<P: PricingStrategy>(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    b_rhs: &[f64],
    basis: &mut [usize],
    m: usize,
    n_cols: usize,
    n_price: usize,
    pricing: &mut P,
    options: &SolverOptions,
    iter_count_out: &mut usize,
    enable_phase1_cycling_bail: bool,
    art_threshold: Option<usize>,
    enable_cleanup_stall_bail: bool,
    cleanup_target_df: Option<f64>,
) -> SimplexOutcome {
    let max_iter = usize::MAX; // timeout is the real guard
    let mut basis_mgr = match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // Buffers reused each iteration. y_dense / rc_vec are filled in-place by
    // compute_reduced_costs_into; d_dense is the FTRAN result for the entering
    // column. Per-iter allocation matters: revised simplex commonly runs
    // 10^4–10^6 iterations on real LPs.
    let mut y_dense = vec![0.0f64; m];
    let mut d_dense = vec![0.0f64; m];
    let mut rc_vec = vec![0.0f64; n_price];

    // eta-update can silently accept a pivot that makes B numerically singular;
    // the loss is only visible at the next fresh LU. On detection we revert to
    // `basis_snapshot` (the last basis a fresh LU accepted) and switch the ratio
    // test to a column-relative pivot floor to prevent re-introducing the same
    // singularity. `blocked_at_basis` records entering columns that triggered a
    // revert so pricing skips them until the next clean refactor.
    let mut blocked_at_basis: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut consecutive_blocks: usize = 0;
    let max_consecutive_blocks: usize = m;
    let mut stable_mode: bool = false;
    let mut basis_snapshot: Vec<usize> = basis.to_vec();

    // Phase I cycling early-bail state.
    let obj_bail_trigger = (BAIL_TRIGGER_FACTOR * m).max(BAIL_TRIGGER_MIN);
    let step_bail_trigger = obj_bail_trigger / STEP_BAIL_RATIO;
    // Charnes perturbation bound: x_b[i] ≤ PIVOT_TOL * m for a degenerate basis,
    // so O(1) leaving-direction d[leaving] → step ≤ PIVOT_TOL * m.
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    // Initialize from the actual starting objective so progress_eps is finite
    // from iteration 1.  f64::INFINITY would make progress_eps = ∞ and the
    // improvement condition `current + ∞ < ∞` always false, causing the
    // obj-progress counter to increment even on genuinely improving iterations.
    let mut best_obj: f64 = basic_obj(c, basis, x_b);
    let mut iters_since_obj_progress: usize = 0;
    let mut iters_since_step_progress: usize = 0;
    // Crossover cleanup has no Bland switch in the primal core. Treat K as
    // no-progress patience: if reduced-cost infeasibility cannot improve for
    // K = max(m, 100) consecutive iterations, assume a degenerate cleanup
    // cycle and bail to the caller's best-vertex handoff. The global deadline
    // remains the hard backstop.
    let cleanup_stall_trigger = m.max(NO_PROGRESS_MIN);
    let cleanup_trace_detail = cleanup_trace_enabled();
    let cleanup_trace_every = cleanup_trace_every();
    let mut cleanup_best_df: Option<f64> = None;
    let mut cleanup_best_basis: Vec<usize> = basis.to_vec();
    let mut cleanup_best_x_b: Vec<f64> = x_b.to_vec();
    let mut cleanup_best_artificials =
        art_threshold.map(|t| basis.iter().copied().filter(|&col| col >= t).count());
    let mut cleanup_iters_since_progress: usize = 0;
    // Cycle detection: when a basis repeats, block the entering variable that
    // contributed to the cycle for m/2 iterations. This forces a different
    // pricing path without changing the numerical method, breaking the cycle.
    let mut cycle_basis_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let cycle_block_duration: usize = m / 2;
    let mut cycle_block_col: Option<usize> = None;
    let mut cycle_block_remaining: usize = 0;
    let mut trace = IterTrace::new("primal-revised");

    for _iter in 0..max_iter {
        *iter_count_out = iter_count_out.saturating_add(1);
        let timed_out = options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }

        if let Some(t) = trace.as_mut() {
            let obj = basic_obj(c, basis, x_b);
            t.log(*iter_count_out, obj, basis, false);
        }

        // y = B^{-T} c_B, then r_j = c_j − y^T a_j for non-basic j. Both steps
        // are shared with the dual paths (see `dual_common`).
        compute_reduced_costs_into(
            a,
            c,
            &mut basis_mgr,
            &is_basic,
            n_price,
            basis,
            &mut y_dense,
            &mut rc_vec,
        );
        if enable_cleanup_stall_bail {
            let current_df = reduced_cost_dual_infeasibility(&rc_vec, &is_basic, n_price);
            if cleanup_trace_detail && (*iter_count_out).is_multiple_of(cleanup_trace_every) {
                let top = top_reduced_cost_violation(&rc_vec, &is_basic, n_price);
                #[allow(clippy::print_stderr)]
                {
                    eprintln!(
                        "[cleanup-detail] iter={} df={:.9e} best={:.9e} top={:?} no_prog={}",
                        *iter_count_out,
                        current_df,
                        cleanup_best_df.unwrap_or(f64::INFINITY),
                        top,
                        cleanup_iters_since_progress
                    );
                }
            }
            let progress_target = cleanup_target_df.unwrap_or(LP_CERT_TOL);
            let (df_progress, new_best_df) = match cleanup_best_df {
                Some(best_df) => (
                    made_cleanup_df_progress(best_df, current_df, progress_target),
                    current_df < best_df,
                ),
                None => (true, true),
            };
            if new_best_df {
                cleanup_best_df = Some(current_df);
                cleanup_best_basis.copy_from_slice(basis);
                cleanup_best_x_b.copy_from_slice(x_b);
            }
            let artificial_progress = if let (Some(threshold), Some(best_artificials)) =
                (art_threshold, cleanup_best_artificials.as_mut())
            {
                let current_artificials = basis
                    .iter()
                    .copied()
                    .filter(|&col| col >= threshold)
                    .count();
                if current_artificials < *best_artificials {
                    *best_artificials = current_artificials;
                    cleanup_best_basis.copy_from_slice(basis);
                    cleanup_best_x_b.copy_from_slice(x_b);
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if cleanup_target_df.is_some_and(|target| current_df <= target) {
                basis.copy_from_slice(&cleanup_best_basis);
                x_b.copy_from_slice(&cleanup_best_x_b);
                let obj = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }

            if df_progress || artificial_progress {
                cleanup_iters_since_progress = 0;
            } else {
                cleanup_iters_since_progress = cleanup_iters_since_progress.saturating_add(1);
                if cleanup_iters_since_progress >= cleanup_stall_trigger {
                    if cleanup_trace_detail {
                        let fresh_df = match LuBasis::new_timed(
                            a,
                            basis,
                            options.max_etas,
                            options.deadline,
                        ) {
                            Ok(mut fresh_mgr) => {
                                let mut y_fresh = vec![0.0f64; m];
                                let mut rc_fresh = vec![0.0f64; n_price];
                                compute_reduced_costs_into(
                                    a,
                                    c,
                                    &mut fresh_mgr,
                                    &is_basic,
                                    n_price,
                                    basis,
                                    &mut y_fresh,
                                    &mut rc_fresh,
                                );
                                reduced_cost_dual_infeasibility(&rc_fresh, &is_basic, n_price)
                            }
                            Err(_) => f64::NAN,
                        };
                        #[allow(clippy::print_stderr)]
                        {
                            eprintln!(
                                "[cleanup-detail] bail iter={} current_df={:.9e} best_df={:.9e} fresh_df={:.9e} top={:?}",
                                *iter_count_out,
                                current_df,
                                cleanup_best_df.unwrap_or(f64::INFINITY),
                                fresh_df,
                                top_reduced_cost_violation(&rc_vec, &is_basic, n_price)
                            );
                        }
                    }
                    basis.copy_from_slice(&cleanup_best_basis);
                    x_b.copy_from_slice(&cleanup_best_x_b);
                    for v in is_basic.iter_mut() {
                        *v = false;
                    }
                    for &col in basis.iter() {
                        is_basic[col] = true;
                    }
                    let obj = basic_obj(c, basis, x_b);
                    if let Some(t) = trace.as_mut() {
                        t.log_stall_bail(*iter_count_out, obj, cleanup_stall_trigger);
                    }
                    return SimplexOutcome::Timeout(obj);
                }
            }
        }
        // Masking RC of blocked columns prevents pricing from re-selecting an
        // entering column known to produce a singular basis from `basis_snapshot`.
        for &j in &blocked_at_basis {
            if j < n_price {
                rc_vec[j] = 0.0;
            }
        }

        // Temporarily mask the blocked column to force a different pricing path
        // when a cycle was recently detected. The mask is lifted after m/2 iters.
        if cycle_block_remaining > 0 {
            cycle_block_remaining -= 1;
            if let Some(col) = cycle_block_col {
                if col < n_price {
                    rc_vec[col] = 0.0;
                }
            }
            if cycle_block_remaining == 0 {
                cycle_block_col = None;
            }
        }

        let entering_col = match pricing.select_entering(&rc_vec, n_price) {
            None => {
                // Optimal (dual-feasible). Verify primal feasibility on a fresh
                // exact x_b = B⁻¹b: a leaving row in the [−primal_tol, 0) band can
                // leave the basis slightly infeasible. If a basic variable is still
                // below −primal_tol (Phase II only), the declared optimum is not a
                // true feasible vertex — return an honest Timeout incumbent rather
                // than a false-Optimal. Phase I feasibility is reconciled by its
                // caller.
                basis_mgr.force_refactor_timed(a, basis, options.deadline);
                if basis_mgr.refactor_failed {
                    // Cannot recompute x_b to verify the vertex; never claim
                    // Optimal on a stale x_b.
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
                let obj: f64 = basic_obj(c, basis, x_b);
                if !enable_phase1_cycling_bail {
                    let min_basic = x_b.iter().copied().fold(f64::INFINITY, f64::min);
                    if min_basic < -options.primal_tol {
                        return SimplexOutcome::Stalled(obj);
                    }
                }
                return SimplexOutcome::Optimal(obj, y_dense.clone());
            }
            Some(j) => j,
        };

        // FTRAN: d = B^{-1} a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        // Save inf-norm of original column for the corruption check below.
        let orig_col_norm = col_vals
            .iter()
            .cloned()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        let mut d_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        d_sv.to_dense_into(&mut d_dense);

        // Refactor on FTRAN corruption: |d|_∞ > 1e12 · |a_q|_∞ or inf/NaN
        // signals eta-accumulated blow-up; reset and recompute d.
        {
            let d_max_abs = d_dense.iter().cloned().fold(0.0f64, |acc, v| {
                if v.is_finite() {
                    acc.max(v.abs())
                } else {
                    f64::INFINITY
                }
            });
            let d_corrupt =
                !d_max_abs.is_finite() || (orig_col_norm > 0.0 && d_max_abs > 1e12 * orig_col_norm);
            if d_corrupt && basis_mgr.eta_count() > 0 {
                basis_mgr.force_refactor_timed(a, basis, options.deadline);
                if basis_mgr.refactor_failed {
                    if basis_mgr.singular_basis {
                        blocked_at_basis.insert(entering_col);
                        consecutive_blocks += 1;
                        if consecutive_blocks > max_consecutive_blocks {
                            return SimplexOutcome::SingularBasis;
                        }
                        stable_mode = true;
                        if !revert_to_snapshot(
                            a,
                            basis,
                            x_b,
                            b_rhs,
                            &basis_snapshot,
                            &mut is_basic,
                            &mut basis_mgr,
                            options,
                        ) {
                            return SimplexOutcome::SingularBasis;
                        }
                        continue;
                    } else {
                        // refactor_failed && !singular ⇔ refactor_timed hit the
                        // deadline (basis/mod.rs) — an external stop, hence Timeout.
                        let obj: f64 = basic_obj(c, basis, x_b);
                        return SimplexOutcome::Timeout(obj);
                    }
                }
                let (cr2, cv2) = a.get_column(entering_col).unwrap();
                d_sv = SparseVec {
                    indices: cr2.to_vec(),
                    values: cv2.to_vec(),
                    len: m,
                };
                basis_mgr.ftran(&mut d_sv);
                d_sv.to_dense_into(&mut d_dense);
                basis_snapshot.copy_from_slice(basis);
            }
        }
        let d = &d_dense;

        // Harris 2-pass ratio test (feasibility-preserving, see
        // `select_leaving_feasibility_preserving`). Pass 1 below derives the
        // pivot eligibility floor (`effective_floor`) and detects an unbounded
        // direction; the leaving row is then chosen by the bound-tolerance
        // helper so the step cannot push any pivot-eligible basic value below
        // −primal_tol (the solve's primal feasibility tolerance).
        //
        // When `stable_mode` is on, eligibility uses a column-relative pivot
        // floor (~1% of |d|_∞) instead of the absolute PIVOT_TOL — necessary
        // after a singular-basis revert, since the absolute floor admits pivots
        // that recreate the same singularity. The fallback to PIVOT_TOL when
        // no row clears the relative floor preserves unboundedness sensitivity.
        let max_d_abs = d.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        let stable_floor = if stable_mode {
            (PIVOT_STABILITY_THRESHOLD * max_d_abs).max(PIVOT_TOL)
        } else {
            PIVOT_TOL
        };

        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if d[i] > stable_floor {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio {
                    min_ratio = ratio;
                }
            }
        }

        let mut effective_floor = stable_floor;
        if !min_ratio.is_finite() && stable_mode {
            for i in 0..m {
                if d[i] > PIVOT_TOL {
                    let ratio = x_b[i] / d[i];
                    if ratio < min_ratio {
                        min_ratio = ratio;
                    }
                }
            }
            effective_floor = PIVOT_TOL;
        }
        if !min_ratio.is_finite() {
            // Last-chance fallback before declaring Unbounded: allow pivots above
            // machine-noise scale. With heavily scaled models, true candidates can
            // sit below PIVOT_TOL; rejecting them here causes false Unbounded.
            let tiny_floor = f64::EPSILON * max_d_abs.max(1.0);
            for i in 0..m {
                if d[i] > tiny_floor {
                    let ratio = x_b[i] / d[i];
                    if ratio < min_ratio {
                        min_ratio = ratio;
                    }
                }
            }
            effective_floor = tiny_floor;
        }

        if !min_ratio.is_finite() {
            return SimplexOutcome::Unbounded;
        }

        // Feasibility-preserving ratio test (δ = primal_tol): the leaving step
        // keeps x_b ≥ −primal_tol, preventing the absolute-window cascade.
        let leaving_row = match select_leaving_feasibility_preserving(
            x_b,
            d,
            basis,
            m,
            effective_floor,
            options.primal_tol,
            art_threshold,
        ) {
            None => return SimplexOutcome::Unbounded,
            Some(i) => i,
        };

        let step = x_b[leaving_row] / d[leaving_row];
        for i in 0..m {
            x_b[i] -= d[i] * step;
        }
        x_b[leaving_row] = step;

        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }
        let leaving_col = basis[leaving_row];

        if let Some(t) = trace.as_mut() {
            t.note_pivot(step.abs(), step_zero_threshold);
        }

        pricing.update_weights(&basis_mgr, entering_col, leaving_col, leaving_row, d);

        is_basic[leaving_col] = false;
        is_basic[entering_col] = true;
        basis[leaving_row] = entering_col;

        // Cycle detection: if the new basis was seen before, re-apply Charnes
        // perturbation to break the degenerate cycle. Near-zero x_b values are
        // perturbed to unique small positives so the ratio test sees distinct
        // step sizes, preventing the exact-tie sequence that causes cycling.
        {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            basis.len().hash(&mut h);
            basis.hash(&mut h);
            let bhash = h.finish();
            if !cycle_basis_hashes.insert(bhash) {
                // Repeated basis detected. Phase I and Phase II differ:
                //
                // Phase I (enable_phase1_cycling_bail=true): Charnes perturbation
                // on near-zero x_b rows only — degenerate artificials produce
                // zero-step pivots and ratio-test ties; nudging them to unique
                // positives breaks the tie without disturbing non-degenerate rows.
                //
                // Phase II (false): NO x_b perturbation — it starts primal-feasible
                // (x_b ≥ 0), and Charnes shifts the objective by O(c_max·eps·m),
                // knocking the solve off near-optimal regions (pilot87: 312 → 467,
                // 100k+ iters to recover). Instead reset the Devex weights; the
                // column block below forces a different entering variable.
                if enable_phase1_cycling_bail {
                    let eps = PIVOT_TOL * (m as f64).max(1.0);
                    for (i, v) in x_b.iter_mut().enumerate() {
                        if v.abs() < step_zero_threshold {
                            *v = eps * (i as f64 + 1.0);
                        } else {
                            #[cfg(test)]
                            super::CYCLE_DETECT_NONDEGEN_PRESERVED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                } else {
                    // Phase II: reset Devex weights to force a different pricing
                    // direction. No x_b perturbation → no objective disruption.
                    pricing.reset_weights(n_cols);
                }
                // Also block the entering column briefly to force path divergence.
                cycle_block_col = Some(entering_col);
                cycle_block_remaining = cycle_block_duration;
                cycle_basis_hashes.clear();
            }
        }

        // Small pivot would blow up the eta inverse-pivot factor; refactor
        // instead of accumulating another eta.
        let pivot_unstable = d[leaving_row].abs() < PIVOT_STABILITY_THRESHOLD * max_d_abs
            && basis_mgr.eta_count() > 0;

        if pivot_unstable {
            basis_mgr.force_refactor_timed(a, basis, options.deadline);
        } else {
            basis_mgr.update(entering_col, leaving_row, &d_sv);
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
        }

        if basis_mgr.refactor_failed {
            if basis_mgr.singular_basis {
                blocked_at_basis.insert(entering_col);
                consecutive_blocks += 1;

                if consecutive_blocks > max_consecutive_blocks {
                    return SimplexOutcome::SingularBasis;
                }

                stable_mode = true;
                if !revert_to_snapshot(
                    a,
                    basis,
                    x_b,
                    b_rhs,
                    &basis_snapshot,
                    &mut is_basic,
                    &mut basis_mgr,
                    options,
                ) {
                    return SimplexOutcome::SingularBasis;
                }
                continue;
            } else {
                // refactor_failed && !singular ⇔ deadline during refactor_timed
                // (basis/mod.rs) — an external stop, hence Timeout.
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
        }

        // Snapshot the basis once a fresh LU accepts it; entries previously
        // blocked may now be safe.
        if basis_mgr.eta_count() == 0 {
            basis_snapshot.copy_from_slice(basis);
            if !blocked_at_basis.is_empty() {
                blocked_at_basis.clear();
                consecutive_blocks = 0;
            }
        }

        // Cycling early-bail. Trigger requires (a) `c^T x_B`
        // plateau for `obj_bail_trigger` iters AND (b) step ≈ 0 for
        // `step_bail_trigger` iters AND (c) Phase I caller. Either signal
        // alone is insufficient: forplan-style Phase I (slow real progress)
        // has step > 0 and resets (b); a Phase II near the optimum sees
        // obj plateau but is gated off by (c).
        let current_obj: f64 = basic_obj(c, basis, x_b);
        if made_progress_with_floor(best_obj, current_obj, 1.0) {
            best_obj = current_obj;
            iters_since_obj_progress = 0;
            #[cfg(test)]
            super::OBJ_PROGRESS_RESET_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            iters_since_obj_progress = iters_since_obj_progress.saturating_add(1);
        }
        if step.abs() > step_zero_threshold {
            iters_since_step_progress = 0;
        } else {
            iters_since_step_progress = iters_since_step_progress.saturating_add(1);
        }
        if enable_phase1_cycling_bail
            && iters_since_obj_progress >= obj_bail_trigger
            && iters_since_step_progress >= step_bail_trigger
        {
            return SimplexOutcome::Stalled(current_obj);
        }
    }

    let obj: f64 = basic_obj(c, basis, x_b);
    SimplexOutcome::Timeout(obj)
}

/// Restore `basis_snapshot` and rebuild `x_b = B^{-1} b` from a fresh LU.
/// `false` ⇒ snapshot factors as singular (treat as fatal SingularBasis).
fn revert_to_snapshot(
    a: &CscMatrix,
    basis: &mut [usize],
    x_b: &mut [f64],
    b_rhs: &[f64],
    basis_snapshot: &[usize],
    is_basic: &mut [bool],
    basis_mgr: &mut LuBasis,
    options: &SolverOptions,
) -> bool {
    basis.copy_from_slice(basis_snapshot);
    for v in is_basic.iter_mut() {
        *v = false;
    }
    for &col in basis.iter() {
        is_basic[col] = true;
    }
    match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
        Ok(mut mgr) => {
            // Recompute x_B; carrying eta drift could leave a slack negative.
            x_b.copy_from_slice(b_rhs);
            mgr.ftran_dense(x_b);
            for v in x_b.iter_mut() {
                if v.abs() < options.clamp_tol {
                    *v = 0.0;
                }
            }
            *basis_mgr = mgr;
            true
        }
        Err(_) => false,
    }
}
