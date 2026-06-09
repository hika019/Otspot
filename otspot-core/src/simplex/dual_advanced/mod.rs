//! Dual Simplex 法（dual.rs 拡張版）
//!
//! 既存dual.rs（warm-start基盤）を拡張し、Harris ratio test、
//! Dual Steepest Edge、Big-M Phase Iを備えたDual Simplexを提供する。
//!

use super::dual_common::outcome_to_result;
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving};
use super::{build_bounded_standard_form_with_deadline, scale_upper_bounds, BoundedStandardForm};
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::basis::{BasisManager, LuBasis};
use crate::options::{DualPricing, SolverOptions, WarmStartBasis};
use crate::presolve::LpEquilibration;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::DROP_TOL;
use bounded_core::{
    bounded_primal_phase1, bounded_primal_phase2_aug, bump_eq_ub_dispatch_count,
    extract_dual_info_bounded, extract_solution_bounded, iterate as bounded_iterate,
    phase2_primal_bounded, solve_bounded_dual, BoundedDualState, BoundedOutcome,
};

pub mod bound_flip;
mod bounded_core;
mod core;
mod phase1;
pub mod ratio_test;
mod steepest_edge;

fn deadline_expired(deadline: Option<std::time::Instant>) -> bool {
    deadline.is_some_and(|d| std::time::Instant::now() >= d)
}

/// Applies deterministic per-row upward jitter to `x_B` with magnitude `mag`.
/// Row 0 always has frac=0 (Knuth PRNG: `0 * SPREAD_MULT = 0`), so the first
/// element is never modified; the jitter remains upward-only and keeps the
/// starting point primal-feasible (`x_B ≥ 0`).
fn perturb_x_b_with_mag(x_b: &mut [f64], mag: f64) {
    /// Odd 32-bit multiplier (Knuth LCG) spreading per-row jitter deterministically.
    const SPREAD_MULT: u64 = 2_654_435_761;
    for (i, v) in x_b.iter_mut().enumerate() {
        let frac = ((i as u64).wrapping_mul(SPREAD_MULT) & 0xFFFF_FFFF) as f64 / 4_294_967_296.0;
        *v += mag * (v.abs() + 1.0) * frac;
    }
}

// Test-only thread-local: when Some(mag), `maybe_perturb_initial_xb` applies
// the jitter with that explicit magnitude, bypassing the env-gated OnceLock.
// Set and restored via ScopedDisable; never leaks across test boundaries.
#[cfg(test)]
thread_local! {
    static PERTURB_MAG_OVERRIDE: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

/// Set the per-thread perturbation magnitude override (test use only).
#[cfg(test)]
pub(crate) fn set_perturb_mag_override(v: Option<f64>) {
    PERTURB_MAG_OVERRIDE.with(|c| c.set(v));
}

/// Apply the production jitter formula to `x_b` with explicit `mag`.
/// Bypasses the env-gated `OnceLock`; used by sentinels that need a
/// before/after-reconcile comparison at a known magnitude.
#[cfg(test)]
pub(crate) fn test_apply_perturb_with_mag(x_b: &mut [f64], mag: f64) {
    perturb_x_b_with_mag(x_b, mag);
}

/// Env-gated (`OTSPOT_PERTURB=1`) anti-degeneracy experiment: perturb the initial
/// basic values by a small per-row deterministic jitter. Because the bounded
/// primal core carries `x_B` forward incrementally, this is equivalent to an RHS
/// perturbation `b ← b + ξ` that breaks degenerate ties (`B⁻¹b` avoids exact
/// zeros) without touching the iteration logic. The terminal `reconcile`
/// recomputes `x_B = B⁻¹b` against the *true* `b` and validates feasibility, so a
/// perturbation that pushes the optimal basis infeasible falls back safely rather
/// than returning a wrong answer. Off by default — measurement only, not yet
/// bench-validated across the full LP suite.
fn maybe_perturb_initial_xb(x_b: &mut [f64]) {
    #[cfg(test)]
    if let Some(test_mag) = PERTURB_MAG_OVERRIDE.with(|c| c.get()) {
        perturb_x_b_with_mag(x_b, test_mag);
        return;
    }

    use std::sync::OnceLock;
    const PERTURB_MAG_DEFAULT: f64 = 1e-7;
    static MAG: OnceLock<Option<f64>> = OnceLock::new();
    let mag = *MAG.get_or_init(|| {
        let on = std::env::var("OTSPOT_PERTURB")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        if !on {
            return None;
        }
        let m = std::env::var("OTSPOT_PERTURB_MAG")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&v| v > 0.0)
            .unwrap_or(PERTURB_MAG_DEFAULT);
        Some(m)
    });
    let Some(mag) = mag else { return };
    perturb_x_b_with_mag(x_b, mag);
}

fn timeout_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        ..Default::default()
    }
}

fn bounded_obj_from_state(c: &[f64], ubs: &[f64], state: &BoundedDualState) -> f64 {
    let basic: f64 = state
        .basis
        .iter()
        .zip(state.x_b.iter())
        .map(|(&j, &v)| c[j] * v)
        .sum();
    let at_ub: f64 = state
        .at_upper
        .iter()
        .enumerate()
        .filter(|&(j, &flag)| flag && !state.is_basic[j])
        .map(|(j, _)| c[j] * ubs[j])
        .sum();
    basic + at_ub
}

enum BoundedTerminalReconcile {
    Optimal(f64),
    Timeout(f64),
    BoundViolation,
    MatrixAccessError,
    SingularBasis,
}

fn reconcile_bounded_terminal_state(
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    ubs: &[f64],
    state: &mut BoundedDualState,
    options: &SolverOptions,
) -> BoundedTerminalReconcile {
    let mut rhs = b.to_vec();
    for (j, &at_ub) in state.at_upper.iter().enumerate() {
        if state.is_basic[j] || !at_ub {
            continue;
        }
        let ub = ubs[j];
        if !ub.is_finite() {
            continue;
        }
        let Ok((rows, vals)) = a.get_column(j) else {
            return BoundedTerminalReconcile::MatrixAccessError;
        };
        for (k, &row) in rows.iter().enumerate() {
            rhs[row] -= vals[k] * ub;
        }
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(crate::error::SolverError::DeadlineExceeded) => {
                return BoundedTerminalReconcile::Timeout(bounded_obj_from_state(c, ubs, state));
            }
            Err(_) => return BoundedTerminalReconcile::SingularBasis,
        };
    let mut x_b_sv = SparseVec::from_dense(&rhs);
    basis_mgr.ftran(&mut x_b_sv);
    state.x_b = x_b_sv.to_dense();
    for v in state.x_b.iter_mut() {
        if v.abs() < options.clamp_tol {
            *v = 0.0;
        }
    }

    for (i, &x_i) in state.x_b.iter().enumerate() {
        if x_i < -options.primal_tol {
            return BoundedTerminalReconcile::BoundViolation;
        }
        let ub = ubs[state.basis[i]];
        if ub.is_finite() && x_i > ub + options.primal_tol {
            return BoundedTerminalReconcile::BoundViolation;
        }
    }
    let obj = bounded_obj_from_state(c, ubs, state);
    BoundedTerminalReconcile::Optimal(obj)
}

/// Builds a [`DualLeavingStrategy`] from `pricing`; DSE initialises *m* weights to 1.
fn make_leaving_strategy(pricing: DualPricing, m: usize) -> Box<dyn DualLeavingStrategy> {
    match pricing {
        DualPricing::MostInfeasible => Box::new(MostInfeasibleLeaving),
        DualPricing::SteepestEdge => Box::new(steepest_edge::DualSteepestEdgeLeaving::new(m)),
    }
}

/// Returns `true` when the given warm basis is dual-feasible under cost vector `c`,
/// i.e. all reduced costs r_j = c_j − y^T a_j ≥ −dual_tol for non-basic j.
///
/// A basis optimal for LP1 may be dual-infeasible if only `c` changes (not `b`).
/// Passing a dual-infeasible basis to the dual simplex causes it to exit as
/// "Optimal" (no lb-violations in x_B) with a wrong objective value.
fn warm_basis_is_dual_feasible(
    a: &crate::sparse::CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    dual_tol: f64,
) -> bool {
    let rc =
        super::dual_common::compute_reduced_costs(a, c, basis_mgr, is_basic, n_price, m, basis);
    rc.iter()
        .enumerate()
        .all(|(j, &r)| is_basic[j] || r >= -dual_tol)
}

/// Dual Simplex強化版エントリポイント
///
/// warm-start提供時: 基底からx_Bを再計算し、dual_simplex_core_advancedを実行
/// cold-start (Le-only): コスト摂動でDual実行可能性を確保し、Harris ratio testで最適化
/// cold-start (Ge/Eq含む): dual::two_phase_dual_simplexにフォールバック
pub(crate) fn solve_dual_advanced(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    if deadline_expired(options.deadline) {
        return timeout_result();
    }
    // Bounded path: problems with finite upper bounds use BFRT-aware iteration.
    // Two sub-paths gated on the BSF shape:
    //   - Le-only (num_artificial == 0)        → `try_bounded` (dual BFRT then primal).
    //   - Has artificials (num_artificial > 0) → `try_bounded_phase1_eq` (augmented
    //     primal Phase I+II); preserves m (no UB-row blow-up) and dispatch is
    //     skipped via thread-local hook in tests for no-op proofs.
    // Le / Ge / Eq rows are handled uniformly: the constraint sense only sets the
    // standard-form slack sign and `needs_artificial`, both already baked into
    // `bsf`. Phase I minimises Σ artificials independent of the original sense, so
    // Ge needs no special path. A spurious "Optimal" is still caught by
    // `guard_lp_optimal` at the entry, so opening Ge cannot return a wrong answer.
    if !bounded_dispatch_disabled() && problem.bounds.iter().any(|&(_, ub)| ub.is_finite()) {
        let Some(bsf) = build_bounded_standard_form_with_deadline(problem, options.deadline) else {
            return timeout_result();
        };
        if bsf.num_artificial == 0 {
            if let Some(result) = try_bounded(&bsf, problem, options) {
                return result;
            }
            // UbViolationOutOfScope → fall through to legacy path
        } else if let Some(result) = try_bounded_phase1_eq(&bsf, problem, options) {
            return result;
            // Phase I infeasibility undecided → fall through to legacy path
        }
    }

    let m = sf.m;
    let Some((a, b, c, row_scale, col_scale)) =
        LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c, options.deadline)
    else {
        return timeout_result();
    };

    if let Some(warm) = &options.warm_start {
        // Warm start: 提供された基底でx_Bを新しいRHSから再計算
        if warm.basis.len() == m && warm.basis.iter().all(|&idx| idx < sf.n_total) {
            let mut basis = warm.basis.clone();

            match LuBasis::new_timed(&a, &basis, options.max_etas, options.deadline) {
                Ok(mut basis_mgr) => {
                    // x_B = B^{-1} b_new (FTRANで計算)
                    let mut x_b_sv = SparseVec::from_dense(&b);
                    basis_mgr.ftran(&mut x_b_sv);
                    let mut x_b = x_b_sv.to_dense();

                    // Guard: dual simplex requires r_j ≥ 0 for all non-basic j.
                    // A basis optimal for LP1 is dual-infeasible when only c changes,
                    // causing dual simplex to exit as Optimal with wrong objective.
                    // Fall through to cold start if the basis is dual-infeasible.
                    let is_basic: Vec<bool> = {
                        let mut v = vec![false; sf.n_total];
                        for &j in &basis {
                            v[j] = true;
                        }
                        v
                    };
                    if !warm_basis_is_dual_feasible(
                        &a,
                        &c,
                        &mut basis_mgr,
                        &basis,
                        &is_basic,
                        sf.n_total,
                        m,
                        options.dual_tol,
                    ) {
                        // dual infeasible under new c → cold start
                    } else {
                        let mut leaving = make_leaving_strategy(options.dual_pricing, m);
                        let mut total_iters: usize = 0;
                        let outcome = core::dual_simplex_core_advanced(
                            &a,
                            &mut x_b,
                            &c,
                            &mut basis,
                            m,
                            sf.n_total,
                            sf.n_total,
                            false, // warm-start: classical Bland, no fallback ⇒ never yield
                            options,
                            leaving.as_mut(),
                            &mut total_iters,
                        );

                        let mut result = outcome_to_result(
                            outcome, sf, problem, &basis, &x_b, &col_scale, &row_scale, true,
                        );
                        result.iterations = total_iters;
                        return result;
                    }
                }
                Err(_) => {
                    // 基底が特異 → cold-startにフォールバック
                }
            }
        }
    }

    // cold-start: Le-only問題（人工変数不要）はHarris dual simplexを使用
    if sf.num_artificial == 0 {
        return cold_start_advanced(sf, problem, options, &a, &b, &c, &row_scale, &col_scale);
    }

    // Cold-start with Ge/Eq constraints: try the standard two-phase simplex
    // path first. On feasible-but-degenerate LPs (d6cube/pds-10 class), Big-M
    // Phase I can spend the whole budget in its augmented Phase II even though
    // the regular primal path quickly finds a feasible incumbent and may finish.
    //
    // Big-M remains the fallback for cases where primal Phase I never produced
    // a feasible incumbent, or where primal reported Infeasible without a
    // verifiable Farkas ray.
    let primal_result = super::dual::two_phase_dual_simplex(sf, problem, options);
    match primal_result.status {
        SolveStatus::Timeout if primal_result.solution.is_empty() => {
            let bigm_result =
                phase1::big_m_cold_start(sf, problem, options, &a, &b, &c, &row_scale, &col_scale);
            if bigm_result.status == SolveStatus::Timeout {
                let mut r = primal_result;
                r.iterations = r.iterations.saturating_add(bigm_result.iterations);
                r
            } else {
                bigm_result
            }
        }
        SolveStatus::Infeasible if !primal_result.dual_solution.is_empty() => primal_result,
        SolveStatus::Infeasible => {
            let bigm_result =
                phase1::big_m_cold_start(sf, problem, options, &a, &b, &c, &row_scale, &col_scale);
            if bigm_result.status == SolveStatus::Timeout {
                SolverResult {
                    status: SolveStatus::Timeout,
                    iterations: primal_result
                        .iterations
                        .saturating_add(bigm_result.iterations),
                    ..primal_result
                }
            } else {
                bigm_result
            }
        }
        _ => primal_result,
    }
}

// ── Bounded (BFRT) path ───────────────────────────────────────────────────────

#[cfg(test)]
thread_local! {
    static BOUNDED_DISPATCH_DISABLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_bounded_dispatch_disabled(v: bool) {
    BOUNDED_DISPATCH_DISABLE.with(|c| c.set(v));
}

fn bounded_dispatch_disabled() -> bool {
    #[cfg(test)]
    {
        BOUNDED_DISPATCH_DISABLE.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Try to solve a Le-only bounded LP via the BFRT-aware dual+primal path.
///
/// Returns `Some(result)` on success or definite failure (Infeasible / Timeout /
/// NumericalError). Returns `None` when `BoundedOutcome::UbViolationOutOfScope`
/// is reached, signalling the caller to fall back to the legacy path.
fn try_bounded(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> Option<SolverResult> {
    if deadline_expired(options.deadline) {
        return Some(timeout_result());
    }
    let Some((a, b, c, row_scale, col_scale)) =
        LpEquilibration::scale_with_deadline(&bsf.a, &bsf.b, &bsf.c, options.deadline)
    else {
        return Some(timeout_result());
    };
    let ubs = scale_upper_bounds(&bsf.upper_bounds, &col_scale);
    // total_iters is always assigned before read (warm branch overwrites before
    // passing &mut to finish_bounded; cold path overwrites before return).
    let mut total_iters: usize;

    // Warm start: reuse a previously-saved bounded-path basis when the index
    // space matches (basis.len() == bsf.m, all indices < bsf.n_total). Warm
    // starts from the legacy path have basis.len() == sf.m > bsf.m when UBs
    // are present, so they fall through to cold start automatically.
    if let Some(warm) = &options.warm_start {
        if warm.basis.len() == bsf.m && warm.basis.iter().all(|&idx| idx < bsf.n_total) {
            if let Ok(mut basis_mgr) =
                LuBasis::new_timed(&a, &warm.basis, options.max_etas, options.deadline)
            {
                let mut x_b_sv = SparseVec::from_dense(&b);
                basis_mgr.ftran(&mut x_b_sv);
                let x_b = x_b_sv.to_dense();
                // Bounded path warm-start: has_lb_violation 時は cold-start fallback。
                // Reason: bounded_core::iterate の BFRT は lower-bound 列のみ選択、
                // sign flip equivalent なし → lb_violation は repair 不可、cycle→timeout (codex review #175)。
                // 真因対処は #190 (WarmStartBasis.at_upper field 追加 + bounded core repair algorithm)。
                //
                // Also fall through when the warm basis is dual-infeasible under
                // the new cost vector c: the dual simplex would exit immediately as
                // Optimal with a wrong objective value.
                let has_lb_violation = super::has_lb_violation(&x_b, options.primal_tol);
                let is_basic_bounded: Vec<bool> = {
                    let mut v = vec![false; bsf.n_total];
                    for &j in &warm.basis {
                        v[j] = true;
                    }
                    v
                };
                if !has_lb_violation
                    && warm_basis_is_dual_feasible(
                        &a,
                        &c,
                        &mut basis_mgr,
                        &warm.basis,
                        &is_basic_bounded,
                        bsf.n_total,
                        bsf.m,
                        options.dual_tol,
                    )
                {
                    let state = BoundedDualState {
                        basis: warm.basis.clone(),
                        at_upper: vec![false; bsf.n_total],
                        x_b,
                        reduced_costs: vec![0.0; bsf.n_total],
                        is_basic: is_basic_bounded,
                        iterations: 0,
                        price_start: 0,
                    };
                    let mut leaving = make_leaving_strategy(options.dual_pricing, bsf.m);
                    let (dual_out, dual_state) =
                        bounded_iterate(state, bsf, &a, &c, options, &ubs, leaving.as_mut());
                    total_iters = dual_state.iterations;
                    let result = finish_bounded(
                        dual_out,
                        dual_state,
                        bsf,
                        &a,
                        &b,
                        &c,
                        &row_scale,
                        &col_scale,
                        &ubs,
                        problem,
                        options,
                        &mut total_iters,
                    );
                    if result.is_some() {
                        return result;
                    }
                    // UbViolationOutOfScope → cold start
                } // dual-infeasibility: fall through to cold start
            }
            // Singular warm basis → cold start
        }
    }

    // Cold start.
    let mut leaving = make_leaving_strategy(options.dual_pricing, bsf.m);
    let (dual_out, dual_state) =
        solve_bounded_dual(bsf, &a, &b, &c, options, &ubs, leaving.as_mut());
    total_iters = dual_state.iterations;
    finish_bounded(
        dual_out,
        dual_state,
        bsf,
        &a,
        &b,
        &c,
        &row_scale,
        &col_scale,
        &ubs,
        problem,
        options,
        &mut total_iters,
    )
}

/// Build the augmented matrix `[bsf.a | I_art]` for the Eq+UB Phase I path.
/// Returns `(a_aug, art_col_of_row, n_art)`. `art_col_of_row[i] = Some(col)` iff
/// `needs_artificial[i]`; `col` indexes into `[bsf.n_total, n_aug)`.
fn build_a_aug_for_eq(
    bsf: &BoundedStandardForm,
    a_scaled: &CscMatrix,
    needs_artificial: &[bool],
) -> (CscMatrix, Vec<Option<usize>>, usize) {
    let m = bsf.m;
    let n_total = bsf.n_total;
    let mut art_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut n_art = 0usize;
    for i in 0..m {
        if needs_artificial[i] {
            art_col_of_row[i] = Some(n_total + n_art);
            n_art += 1;
        }
    }
    let n_aug = n_total + n_art;

    let mut trip_rows: Vec<usize> = Vec::with_capacity(a_scaled.nnz() + n_art);
    let mut trip_cols: Vec<usize> = Vec::with_capacity(a_scaled.nnz() + n_art);
    let mut trip_vals: Vec<f64> = Vec::with_capacity(a_scaled.nnz() + n_art);
    for j in 0..n_total {
        let (rows, vals) = a_scaled.get_column(j).unwrap();
        for (k, &row) in rows.iter().enumerate() {
            let v = vals[k];
            if v.abs() > DROP_TOL {
                trip_rows.push(row);
                trip_cols.push(j);
                trip_vals.push(v);
            }
        }
    }
    for (i, col_opt) in art_col_of_row.iter().enumerate() {
        if let Some(col) = col_opt {
            trip_rows.push(i);
            trip_cols.push(*col);
            trip_vals.push(1.0);
        }
    }

    let a_aug = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_aug)
        .expect("augmented matrix construction must succeed (deduplicated by build)");
    (a_aug, art_col_of_row, n_art)
}

/// Initial basic values `x_B = B^{-1} b` for the slack/artificial starting
/// basis. That basis is structurally diagonal — each slack column and each
/// artificial column has its single nonzero at its own row — but after Ruiz
/// scaling the Le-slack diagonals are non-unit, so `x_B[i] = b[i] / diag_i`,
/// not `b[i]`. Artificial columns have `diag = 1`. Negative results are clamped
/// to 0: the bounded primal starts inside the box and a feasible slack/art
/// basis yields `x_B ≥ 0` (the clamp is a no-op there and only guards roundoff).
fn diag_basis_initial_x_b(a_aug: &CscMatrix, basis: &[usize], b: &[f64]) -> Vec<f64> {
    let m = basis.len();
    let mut x_b = vec![0.0f64; m];
    for i in 0..m {
        let (rows, vals) = a_aug.get_column(basis[i]).unwrap();
        let mut diag = 0.0f64;
        for (k, &r) in rows.iter().enumerate() {
            if r == i {
                diag = vals[k];
                break;
            }
        }
        debug_assert!(
            diag != 0.0,
            "slack/artificial starting basis must be diagonal (nonzero pivot at its own row)"
        );
        let v = b[i] / diag;
        x_b[i] = if v < 0.0 { 0.0 } else { v };
    }
    x_b
}

/// Eq+UB cold start: augment with artificials, run primal Phase I (minimise
/// sum of artificials), then primal Phase II on the augmented matrix with
/// pricing restricted to structural columns.
///
/// Returns `None` to signal "fall through to legacy" when Phase I evidence is
/// inconclusive (e.g. Phase I Timeout on a possibly-feasible LP) — Big-M /
/// primal Phase I in the legacy path may still resolve it.
fn try_bounded_phase1_eq(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> Option<SolverResult> {
    if deadline_expired(options.deadline) {
        return Some(timeout_result());
    }
    let Some((a, b, c, row_scale, col_scale)) =
        LpEquilibration::scale_with_deadline(&bsf.a, &bsf.b, &bsf.c, options.deadline)
    else {
        return Some(timeout_result());
    };
    let ubs = scale_upper_bounds(&bsf.upper_bounds, &col_scale);

    bump_eq_ub_dispatch_count();

    // Optional crash basis: replaces artificial placeholders with structural
    // columns where a primal-feasible pivot exists. Returns identity-equivalent
    // (basis_pre = bsf.initial_basis, needs_artificial = bsf.needs_artificial)
    // when disabled. Crash is applied opportunistically; if x_b ≥ 0 fails after
    // FTRAN, we fall back to identity (no recursion: re-derive on the spot).
    let identity_state = || {
        let (a_aug, art_col_of_row, n_art) = build_a_aug_for_eq(bsf, &a, &bsf.needs_artificial);
        let n_aug = bsf.n_total + n_art;
        let mut ubs_aug = vec![f64::INFINITY; n_aug];
        ubs_aug[..bsf.n_total].copy_from_slice(&ubs);
        let mut basis: Vec<usize> = bsf.initial_basis.clone();
        for (i, col_opt) in art_col_of_row.iter().enumerate() {
            if let Some(col) = col_opt {
                basis[i] = *col;
            }
        }
        let mut is_basic = vec![false; n_aug];
        for &j in &basis {
            is_basic[j] = true;
        }
        let x_b = diag_basis_initial_x_b(&a_aug, &basis, &b);
        (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b)
    };

    let (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b) = if options.use_lp_crash_basis {
        let (basis_pre, needs_artificial, _n_art_post) = super::crash::compute_crash_basis(
            &a,
            &b,
            bsf.m,
            bsf.n_shifted,
            &bsf.initial_basis,
            &bsf.needs_artificial,
        );
        let (a_aug, art_col_of_row, n_art) = build_a_aug_for_eq(bsf, &a, &needs_artificial);
        let n_aug = bsf.n_total + n_art;
        let mut ubs_aug = vec![f64::INFINITY; n_aug];
        ubs_aug[..bsf.n_total].copy_from_slice(&ubs);
        let mut basis: Vec<usize> = basis_pre;
        for (i, col_opt) in art_col_of_row.iter().enumerate() {
            if let Some(col) = col_opt {
                basis[i] = *col;
            }
        }
        let mut is_basic = vec![false; n_aug];
        for &j in &basis {
            is_basic[j] = true;
        }
        // Compute x_B = B^{-1} b. Skip the FTRAN when crash made no change
        // (basis is the slack/art identity, so x_B = b directly).
        let needs_ftran = needs_artificial
            .iter()
            .zip(bsf.needs_artificial.iter())
            .any(|(post, orig)| post != orig);
        let x_b = if needs_ftran {
            match LuBasis::new_timed(&a_aug, &basis, options.max_etas, options.deadline) {
                Ok(mut bm) => {
                    let mut sv = SparseVec::from_dense(&b);
                    bm.ftran(&mut sv);
                    let xb = sv.to_dense();
                    // Reject both lb (xb < 0) and ub (xb > ubs_aug[basis[i]])
                    // violations — primal_simplex_aug starts inside the box and
                    // cannot repair an initial UB overshoot.
                    let infeasible = xb.iter().enumerate().any(|(i, &v)| {
                        if v < -options.primal_tol {
                            return true;
                        }
                        let ub_i = ubs_aug[basis[i]];
                        ub_i.is_finite() && v > ub_i + options.primal_tol
                    });
                    if infeasible {
                        // Crash yielded a bounded-infeasible RHS. The user
                        // explicitly enabled crash; the legacy path (crash +
                        // Big-M, UB rows expanded) can absorb violations the
                        // bounded primal start cannot. Hand off rather than
                        // silently dropping crash's iter savings.
                        return None;
                    }
                    xb
                }
                Err(_) => {
                    return run_phase1_then_phase2(
                        bsf,
                        problem,
                        options,
                        identity_state,
                        &b,
                        &c,
                        &row_scale,
                        &col_scale,
                    );
                }
            }
        } else {
            // Crash made no change → basis is still the slack/art identity
            // (diagonal); use the scaled-diagonal solve, not a raw b.clone().
            diag_basis_initial_x_b(&a_aug, &basis, &b)
        };
        (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b)
    } else {
        identity_state()
    };

    run_phase1_then_phase2(
        bsf,
        problem,
        options,
        move || (a_aug, art_col_of_row, ubs_aug, basis, is_basic, x_b),
        &b,
        &c,
        &row_scale,
        &col_scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_phase1_then_phase2<F>(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    state_factory: F,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> Option<SolverResult>
where
    F: FnOnce() -> (
        CscMatrix,
        Vec<Option<usize>>,
        Vec<f64>,
        Vec<usize>,
        Vec<bool>,
        Vec<f64>,
    ),
{
    fn mark_eq_ub_path(mut r: SolverResult) -> SolverResult {
        r.stats.bounded_eq_ub_path = true;
        r
    }

    let (a_aug, art_col_of_row, mut ubs_aug, basis, is_basic, mut x_b) = state_factory();
    let n_aug = a_aug.ncols;
    maybe_perturb_initial_xb(&mut x_b);
    let mut state = BoundedDualState {
        basis,
        at_upper: vec![false; n_aug],
        x_b,
        reduced_costs: vec![0.0; n_aug],
        is_basic,
        iterations: 0,
        price_start: 0,
    };

    // Phase I: minimise sum of artificials. Structural cost = 0.
    let mut c_p1 = vec![0.0f64; n_aug];
    for col in art_col_of_row.iter().flatten() {
        c_p1[*col] = 1.0;
    }
    let mut iters: usize = 0;
    let p1_out = bounded_primal_phase1(
        &a_aug,
        &c_p1,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p1_out {
        SimplexOutcome::SingularBasis => {
            return Some(mark_eq_ub_path(SolverResult::numerical_error()));
        }
        SimplexOutcome::Unbounded => {
            // Phase I obj = sum of artificials ≥ 0, bounded below. Should not
            // happen; treat as "no decision" and fall through to legacy.
            return None;
        }
        SimplexOutcome::Timeout(_) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            return Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Timeout,
                objective: bsf.obj_offset,
                solution,
                iterations: iters,
                ..Default::default()
            }));
        }
        SimplexOutcome::Optimal(_, _) => {
            let art_sum = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p1, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(_) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::BoundViolation => return None,
                BoundedTerminalReconcile::MatrixAccessError
                | BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            if art_sum > options.primal_tol {
                // Phase I converged with residual artificials → infeasible.
                let mut r = SolverResult::infeasible();
                r.iterations = iters;
                return Some(mark_eq_ub_path(r));
            }
        }
    }

    // Pin artificials to ub = 0 for Phase II. Phase I leaves residual
    // artificials basic at 0 (art_sum ≤ primal_tol); with ub = ∞ a structural
    // pivot whose column has a nonzero entry in an artificial's row could push
    // it positive (its increasing direction is otherwise unbounded), silently
    // violating Ax = b. Pinning ub = 0 makes the ratio test cap such a step at
    // 0, driving the artificial out by a degenerate pivot instead. Required for
    // the multi-artificial regime (e.g. pds-20: 27827 artificials) where the
    // single-artificial optimality argument does not generalise.
    for col in art_col_of_row.iter().flatten() {
        ubs_aug[*col] = 0.0;
    }

    // Phase II: minimise true objective on augmented matrix; artificials cost 0
    // so any that remain basic at 0 do not bias the objective. They are never
    // priced (n_struct = bsf.n_total) so they cannot re-enter.
    let mut c_p2 = vec![0.0f64; n_aug];
    c_p2[..bsf.n_total].copy_from_slice(c);
    let p2_out = bounded_primal_phase2_aug(
        &a_aug,
        &c_p2,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p2_out {
        SimplexOutcome::Optimal(_, y) => {
            let pre_reconcile_x_b = state.x_b.clone();
            let obj = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p2, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(obj) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: obj + bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::BoundViolation => {
                    state.x_b = pre_reconcile_x_b;
                    let obj = bounded_obj_from_state(&c_p2, &ubs_aug, &state);
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: obj + bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::MatrixAccessError
                | BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            // Warm-start basis only when no artificial remains basic; else the
            // legacy warm path would index outside `bsf.n_total`.
            let ws = if state.basis.iter().all(|&j| j < bsf.n_total) {
                Some(WarmStartBasis {
                    basis: state.basis.clone(),
                    x_b: state.x_b.clone(),
                })
            } else {
                None
            };
            Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: ws,
                iterations: iters,
                ..Default::default()
            }))
        }
        SimplexOutcome::Unbounded => Some(mark_eq_ub_path(SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            iterations: iters,
            ..Default::default()
        })),
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                iterations: iters,
                ..Default::default()
            }))
        }
        SimplexOutcome::SingularBasis => Some(mark_eq_ub_path(SolverResult::numerical_error())),
    }
}

/// Convert a `BoundedOutcome` from the dual phase into a `SolverResult`,
/// running Phase 2 primal on `Optimal`. Returns `None` for
/// `UbViolationOutOfScope` so the caller can fall back to the legacy path.
#[allow(clippy::too_many_arguments)]
fn finish_bounded(
    dual_out: BoundedOutcome,
    dual_state: BoundedDualState,
    bsf: &BoundedStandardForm,
    a: &crate::sparse::CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
    ubs: &[f64],
    problem: &LpProblem,
    options: &SolverOptions,
    total_iters: &mut usize,
) -> Option<SolverResult> {
    match dual_out {
        BoundedOutcome::UbViolationOutOfScope { .. } => None,
        BoundedOutcome::Unbounded => Some(SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        }),
        BoundedOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &dual_state, col_scale);
            Some(SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: *total_iters,
                ..Default::default()
            })
        }
        BoundedOutcome::SingularBasis => Some(SolverResult::numerical_error()),
        BoundedOutcome::Optimal(_, _) => {
            let (p2_out, mut p2_state) =
                phase2_primal_bounded(bsf, dual_state, a, c, options, total_iters, ubs);
            let p2_out = match p2_out {
                SimplexOutcome::Optimal(_, y) => {
                    let pre_reconcile_x_b = p2_state.x_b.clone();
                    match reconcile_bounded_terminal_state(a, b, c, ubs, &mut p2_state, options) {
                        BoundedTerminalReconcile::Optimal(obj) => SimplexOutcome::Optimal(obj, y),
                        BoundedTerminalReconcile::Timeout(obj) => SimplexOutcome::Timeout(obj),
                        BoundedTerminalReconcile::BoundViolation => {
                            p2_state.x_b = pre_reconcile_x_b;
                            let obj = bounded_obj_from_state(c, ubs, &p2_state);
                            SimplexOutcome::Timeout(obj)
                        }
                        BoundedTerminalReconcile::MatrixAccessError
                        | BoundedTerminalReconcile::SingularBasis => SimplexOutcome::SingularBasis,
                    }
                }
                other => other,
            };
            Some(finish_bounded_phase2(
                p2_out,
                p2_state,
                bsf,
                col_scale,
                row_scale,
                problem,
                *total_iters,
            ))
        }
    }
}

fn finish_bounded_phase2(
    out: SimplexOutcome,
    state: BoundedDualState,
    bsf: &BoundedStandardForm,
    col_scale: &[f64],
    row_scale: &[f64],
    problem: &LpProblem,
    total_iters: usize,
) -> SolverResult {
    match out {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: state.basis,
                x_b: state.x_b,
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        },
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

/// Le-only cold startでHarris Dual Simplexを使用する
///
/// dual.rs::cold_start_dual と同じ構造だが、Phase 1で dual_simplex_core_advanced
/// （Harris ratio test + LuBasis::needs_refactor）を使用する。
#[allow(clippy::too_many_arguments)]
fn cold_start_advanced(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &crate::sparse::CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;

    // Le-only: スラック基底 B=I, x_B = b ≥ 0（標準形変換後）
    let mut basis = sf.initial_basis.clone();
    let mut x_b = b.to_vec();

    // コスト摂動: c̃_j = max(c_j, 0) → スラック基底（y=0）で r̃_j = c̃_j ≥ 0 → 双対実行可能
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut leaving = make_leaving_strategy(options.dual_pricing, m);

    // Phase 1: Harris dual simplexで主実行可能性を修復
    // Le-onlyでb≥0の場合、x_B=b≥0なので即座に終了（0反復）
    let mut total_iters: usize = 0;
    let phase1_outcome = core::dual_simplex_core_advanced(
        a,
        &mut x_b,
        &c_perturbed,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        false, // Le-only cold-start: classical Bland, no fallback ⇒ never yield
        options,
        leaving.as_mut(),
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            // 双対非有界 = 主実行不可
            return SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            };
        }
        SimplexOutcome::Timeout(_) => {
            return super::timeout_result_with_incumbent(
                sf,
                problem,
                &basis,
                &x_b,
                col_scale,
                total_iters,
            );
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {
            // Phase 1完了: x_B ≥ 0 (主実行可能)
        }
    }

    // Phase 2: 元のコストで主実行可能点からPrimal Simplexで最適化
    use super::pricing::SteepestEdgePricing;
    let mut pricing = SteepestEdgePricing::new(sf.n_total);
    let phase2_outcome = super::revised_simplex_core(
        a,
        &mut x_b,
        c,
        b,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        &mut pricing,
        options,
        &mut total_iters,
        false,
    );

    // Phase 2はPrimalなのでUnbounded=主非有界
    // (result.iterations は match の後で set)
    let mut result = match phase2_outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
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
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        },
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    };
    result.iterations = total_iters;
    result
}

// ── Wiring sentinels ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{SolverOptions, WarmStartBasis};
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, reset_bfrt_flip_invocations,
    };
    use crate::simplex::standard_form::build_standard_form;

    /// min -x0 - x1, x0+x1 ≤ 6, x0-x1 ≤ 2, 0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 4
    /// Known optimal: x0=4, x1=2, obj=-6.
    fn lp_2x2_boxed() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a =
            CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![6.0, 2.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 4.0)],
            None,
        )
        .unwrap()
    }

    /// min -x0 - 3*x1, x0+x1 ≤ 5, 0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 2.
    /// Pricing scores: x1=3 > x0=1, so x1 enters first. The ratio test gives
    /// min_step=5 but ub_x1=2 < 5, triggering a Phase 2 primal BFRT flip.
    /// After the flip, x0 enters the basis at value 3.
    /// Optimal: x0=3 (basic), x1=2 (non-basic at ub), obj=-3-6=-9.
    fn lp_flip_trigger() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![-1.0, -3.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 2.0)],
            None,
        )
        .unwrap()
    }

    fn lp_no_ub() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn bound_violation_timeout_objective_uses_restored_incumbent() {
        use crate::sparse::CscMatrix;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let c = vec![10.0, 1.0];
        let ubs = vec![3.0, 10.0];
        let mut state = BoundedDualState {
            basis: vec![1],
            at_upper: vec![true, false],
            x_b: vec![0.5],
            reduced_costs: vec![0.0, 0.0],
            is_basic: vec![false, true],
            iterations: 0,
            price_start: 0,
        };
        let pre_reconcile_x_b = state.x_b.clone();

        assert!(matches!(
            reconcile_bounded_terminal_state(
                &a,
                &b,
                &c,
                &ubs,
                &mut state,
                &SolverOptions::default()
            ),
            BoundedTerminalReconcile::BoundViolation
        ));
        let invalid_obj = bounded_obj_from_state(&c, &ubs, &state);
        state.x_b = pre_reconcile_x_b;
        let timeout_obj = bounded_obj_from_state(&c, &ubs, &state);
        let restored_solution = [ubs[0], state.x_b[0]];
        let recomputed_obj: f64 = c
            .iter()
            .zip(restored_solution.iter())
            .map(|(&c_j, &x_j)| c_j * x_j)
            .sum();

        assert_ne!(invalid_obj, timeout_obj);
        assert_eq!(timeout_obj, recomputed_obj);
    }

    /// **Flip > 0 sentinel**: solving a boxed LP via `solve_dual_advanced`
    /// must exercise at least one Phase 2 primal BFRT flip (entering variable
    /// hits its upper bound before any basis row leaves).
    ///
    /// No-op proof: `bfrt_wiring_flip_count_positive_noop_proof` verifies that
    /// disabling the bounded dispatch makes flip count = 0.
    #[test]
    fn bfrt_wiring_flip_count_positive() {
        let lp = lp_flip_trigger();
        let sf = build_standard_form(&lp);
        reset_bfrt_flip_invocations();
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        let flips = bfrt_flip_invocations();
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "expected Optimal, got {:?}",
            result.status
        );
        assert!(
            (result.objective - (-9.0)).abs() < 1e-5,
            "expected obj=-9, got {:.6e}",
            result.objective
        );
        assert!(
            flips > 0,
            "bfrt_wiring_flip_count_positive: flip count = 0, bounded path not exercised"
        );
    }

    /// **No-op proof**: disabling bounded dispatch causes flip count = 0.
    /// This sentinel must FAIL whenever the bounded path is bypassed.
    #[test]
    fn bfrt_wiring_flip_count_positive_noop_proof() {
        let lp = lp_flip_trigger();
        let sf = build_standard_form(&lp);
        let _guard = crate::ScopedDisable::new(
            || set_bounded_dispatch_disabled(true),
            || set_bounded_dispatch_disabled(false),
        );
        reset_bfrt_flip_invocations();
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        let flips_disabled = bfrt_flip_invocations();
        assert_eq!(
            flips_disabled, 0,
            "noop proof: expected 0 flips with bounded dispatch disabled, got {flips_disabled}"
        );
        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// **Multi-pattern coverage**: three LP shapes all reach Optimal.
    /// Pattern 2 (flip-trigger, finite UBs) asserts flip count > 0 as a
    /// load-bearing sentinel — fails if bounded dispatch is bypassed.
    #[test]
    fn bfrt_wiring_multi_pattern_correct() {
        // Pattern 1: 2x2 boxed — bounded path, Phase 2 converges without BFRT flip.
        {
            let lp = lp_2x2_boxed();
            let sf = build_standard_form(&lp);
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            assert_eq!(r.status, SolveStatus::Optimal, "pattern 1 status");
            assert!(
                (r.objective - (-6.0)).abs() < 1e-5,
                "pattern 1 obj={}",
                r.objective
            );
        }
        // Pattern 2: flip-trigger LP — entering variable hits its UB before leaving
        // row. Flip count > 0 confirms the BFRT flip path in Phase 2 is reachable.
        {
            let lp = lp_flip_trigger();
            let sf = build_standard_form(&lp);
            reset_bfrt_flip_invocations();
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            let flips = bfrt_flip_invocations();
            assert_eq!(r.status, SolveStatus::Optimal, "pattern 2 status");
            assert!(
                (r.objective - (-9.0)).abs() < 1e-5,
                "pattern 2 obj={}",
                r.objective
            );
            assert!(
                flips > 0,
                "pattern 2: flip count = 0, bounded path not exercised"
            );
        }
        // Pattern 3: no UBs → legacy path, no flip assertion.
        {
            let lp = lp_no_ub();
            let sf = build_standard_form(&lp);
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            assert_eq!(r.status, SolveStatus::Optimal, "pattern 3 status");
        }
    }

    /// **Ge + UB bounded path (sentinel):** a `Ge`-constrained LP with finite
    /// upper bounds must solve via the bounded Phase I path, not the legacy
    /// UB-row-expanded path. The bounded path emits a `warm_start_basis`; the
    /// legacy `two_phase_dual_simplex` cold path does not.
    ///
    /// No-op proof: re-adding the `!has_ge` gate in `solve_dual_advanced` routes
    /// this LP to the legacy path, which returns `warm_start_basis = None`,
    /// failing the `is_some()` assertion. The objective check guards correctness
    /// of the opened path (a wrong dual sign would be demoted by guard_lp_optimal
    /// upstream, but here we assert the raw Optimal value directly).
    #[test]
    fn ge_with_ub_solves_via_bounded_path_with_warm_basis() {
        use crate::sparse::CscMatrix;
        // min x + y  s.t.  x + y >= 3 (Ge),  0 <= x,y <= 4.
        // Optimal: x + y = 3 (any split), obj = 3.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 4.0), (0.0, 4.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal, "Ge+UB status: {:?}", r.status);
        assert!(
            (r.objective - 3.0).abs() < 1e-6,
            "Ge+UB obj={} expected 3",
            r.objective
        );
        assert!(
            r.warm_start_basis.is_some(),
            "Ge+UB must route to the bounded path (emits warm_start_basis); \
             None means it fell back to the legacy path (has_ge gate re-added)"
        );
    }

    /// **Ge infeasible (sentinel):** a Ge-constrained LP whose Ge bound exceeds the
    /// variable's upper bound must solve to Infeasible via the bounded path.
    ///
    /// LP: min x  s.t. x ≥ 5 (Ge),  0 ≤ x ≤ 3.
    /// The Ge constraint requires x ≥ 5 but the UB forces x ≤ 3; infeasible.
    /// The finite UB causes the bounded Phase I path to be taken.
    ///
    /// No-op proof: if Phase I incorrectly declares artificials = 0 for an
    /// infeasible Ge system, the solver returns Optimal or SuboptimalSolution,
    /// failing the Infeasible assertion.
    #[test]
    fn ge_infeasible_bounded_path_returns_infeasible() {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 3.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(
            r.status,
            SolveStatus::Infeasible,
            "Ge infeasible LP (x>=5, UB=3) must be Infeasible, got {:?} (obj={:.6e})",
            r.status,
            r.objective
        );
    }

    /// **Ge at-upper-bound (sentinel):** in a Ge LP whose optimal places a
    /// variable non-basic at its upper bound, `extract_solution_bounded` must
    /// apply the `at_upper` correction so the solution and objective are correct.
    ///
    /// LP: min -x  s.t. x ≥ 2 (Ge),  0 ≤ x ≤ 3.
    /// Optimal: x = 3 (at UB, non-basic), obj = -3.
    ///
    /// No-op proof: `bounded_core::set_at_upper_apply_disabled(true)` causes
    /// `extract_solution_bounded` to leave x at 0 instead of its upper bound,
    /// yielding obj = 0 ≠ -3 and failing the objective assertion.
    #[test]
    fn ge_at_upper_bound_solution_extracted_correctly() {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 3.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "Ge at-UB LP status: {:?}",
            r.status
        );
        assert!(
            (r.objective - (-3.0)).abs() < 1e-6,
            "Ge at-UB obj={:.6e} expected -3 (x should be at UB=3); \
             disabled at_upper correction would yield obj=0",
            r.objective
        );
        assert!(
            r.solution.first().copied().unwrap_or(0.0) > 2.5,
            "Ge at-UB solution[0]={:.6e} expected ≈3 (at upper bound)",
            r.solution.first().copied().unwrap_or(0.0)
        );
    }

    /// **Ge + Eq + Le mixed (sentinel):** a LP with all three constraint kinds
    /// (multiple artificials) must solve correctly via the bounded Phase I path.
    ///
    /// LP: min x + y + z
    ///       x + y ≥ 2  (Ge)
    ///       x     = 1  (Eq)
    ///       y + z ≤ 3  (Le)
    ///       0 ≤ x,y,z ≤ 4
    ///
    /// Optimal (by hand): x=1 (Eq), y=1 (Ge binding, minimize y+z with z≥0),
    /// z=0, obj=2.
    ///
    /// No-op proof: if Ge or Eq artificials are mishandled (wrong Phase I
    /// placement or incorrect basis injection), the solver returns a wrong
    /// status or objective, failing the assertions below.
    #[test]
    fn ge_eq_le_mixed_types_solve_correctly() {
        use crate::sparse::CscMatrix;
        use crate::test_kkt::assert_solver_invariants_lp;
        // rows=[0,0,1,2,2], cols=[0,1,0,1,2]:
        //   Row 0 (Ge): x + y >= 2
        //   Row 1 (Eq): x     = 1
        //   Row 2 (Le): y + z <= 3
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2, 2],
            &[0, 1, 0, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            3,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![2.0, 1.0, 3.0],
            vec![ConstraintType::Ge, ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "Ge+Eq+Le mixed LP must be Optimal, got {:?}",
            r.status
        );
        assert!(
            (r.objective - 2.0).abs() < 1e-6,
            "Ge+Eq+Le obj={:.6e} expected 2.0",
            r.objective
        );
        assert_solver_invariants_lp(&r, &lp);
    }

    /// **Ge dual sign (sentinel):** for a min LP with a binding Ge constraint,
    /// `dual_solution[i]` must be ≥ 0 (LP simplex convention: Ge dual ≥ 0).
    ///
    /// LP: min x + y  s.t. x + y ≥ 3 (Ge),  0 ≤ x,y ≤ 4.
    /// Optimal: x + y = 3 (binding), obj = 3.
    /// KKT stationarity: 1 - 1·y0 = 0 ⟹ y0 = 1 > 0. ✓
    ///
    /// No-op proof: if `extract_dual_info_bounded` applies the wrong negation
    /// for a `row_negated` Ge row, `dual_solution[0]` becomes ≤ −1, failing
    /// both the sign assertion and `assert_solver_invariants_lp` dual-sign check.
    #[test]
    fn ge_dual_sign_nonnegative_in_min_problem() {
        use crate::sparse::CscMatrix;
        use crate::test_kkt::assert_solver_invariants_lp;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 4.0), (0.0, 4.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(r.status, SolveStatus::Optimal, "Ge dual status: {:?}", r.status);
        assert!(
            (r.objective - 3.0).abs() < 1e-6,
            "Ge dual obj={:.6e} expected 3",
            r.objective
        );
        assert!(
            !r.dual_solution.is_empty(),
            "dual_solution must be non-empty for Ge dual sign check"
        );
        assert!(
            r.dual_solution[0] >= -1e-6,
            "Ge dual (LP simplex convention) must be ≥ 0 for binding Ge in min LP, \
             got {:.6e}; wrong row_negated sign in extract_dual_info_bounded would yield ≤ -1",
            r.dual_solution[0]
        );
        // Quantitative check: KKT stationarity gives y0 = 1 for this LP.
        assert!(
            r.dual_solution[0] > 0.5,
            "Ge dual expected ≈1.0 (KKT stationarity: 1 - y0 = 0), got {:.6e}",
            r.dual_solution[0]
        );
        assert_solver_invariants_lp(&r, &lp);
    }

    /// **P2-B** — Warm start is accepted even when the warm basis has
    /// lb-violations after a b-perturbation (legacy path, no finite UBs).
    ///
    /// LP: min -3x0 - x1, x0+x1≤4, x0≤3, x1≤2, x0,x1 ≥ 0.
    /// Cold optimal: x0=3, x1=1, obj=-10. Warm basis = {x0, x1, s2}.
    /// Perturb b=[1,3,2]: B⁻¹·[1,3,2] = [3, -2, 4] → lb-violation at x1.
    /// After guard removal (#175) the dual simplex repairs x1 and converges
    /// to the perturbed-LP optimal x0=1, x1=0, obj=-3.
    ///
    /// If `has_lb_violation` were re-added to the legacy path, the warm
    /// solve would fall through to cold start and still produce Optimal.
    /// The definitive iteration-level sentinel is
    /// `dse_iter_count_matches_or_beats_most_infeasible` in
    /// `tests/diag_dse_pivot_selection.rs`.
    #[test]
    fn legacy_warm_start_lb_violation_repairs_and_converges() {
        use crate::sparse::CscMatrix;
        const OBJ_TOL: f64 = 1e-5;

        // No finite UBs → legacy dual path.
        // LP: min -3x0 - x1, x0+x1≤b[0], x0≤b[1], x1≤b[2], x0,x1≥0
        let make_lp = |b: Vec<f64>| {
            LpProblem::new_general(
                vec![-3.0, -1.0],
                CscMatrix::from_triplets(&[0, 0, 1, 2], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 3, 2)
                    .unwrap(),
                b,
                vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
                vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
                None,
            )
            .unwrap()
        };

        // Cold solve: b=[4,3,2], optimal x0=3, x1=1, obj=-10.
        let lp_orig = make_lp(vec![4.0, 3.0, 2.0]);
        let sf_orig = build_standard_form(&lp_orig);
        let r_cold = solve_dual_advanced(&sf_orig, &lp_orig, &SolverOptions::default());
        assert_eq!(
            r_cold.status,
            SolveStatus::Optimal,
            "cold: {:?}",
            r_cold.status
        );
        assert!(
            (r_cold.objective - (-10.0)).abs() < OBJ_TOL,
            "cold obj={:.6e} expected -10",
            r_cold.objective
        );
        let warm = r_cold
            .warm_start_basis
            .expect("cold solve must return warm_start_basis");

        // Perturbed LP: b=[1,3,2]. Warm basis has x1=-2 (lb-violation).
        // Dual simplex must repair and converge to x0=1, x1=0, obj=-3.
        let lp_p = make_lp(vec![1.0, 3.0, 2.0]);
        let sf_p = build_standard_form(&lp_p);
        let r_warm = solve_dual_advanced(
            &sf_p,
            &lp_p,
            &SolverOptions {
                warm_start: Some(warm),
                ..SolverOptions::default()
            },
        );
        assert_eq!(
            r_warm.status,
            SolveStatus::Optimal,
            "warm re-solve: {:?} — guard still present?",
            r_warm.status
        );
        assert!(
            (r_warm.objective - (-3.0)).abs() < OBJ_TOL,
            "warm re-solve obj={:.6e} expected -3",
            r_warm.objective
        );

        // Consistency: cold re-solve agrees.
        let r_cold_p = solve_dual_advanced(&sf_p, &lp_p, &SolverOptions::default());
        assert_eq!(r_cold_p.status, SolveStatus::Optimal);
        assert!(
            (r_cold_p.objective - r_warm.objective).abs() < OBJ_TOL,
            "warm {:.6e} != cold {:.6e}",
            r_warm.objective,
            r_cold_p.objective
        );
    }

    /// Warm start from a bounded-path solve is accepted and reused.
    /// Uses the flip-trigger LP so that the cold solve exercises the BFRT flip
    /// path and flip count > 0 becomes a load-bearing sentinel.
    #[test]
    fn bfrt_wiring_warm_start_reuse() {
        let lp = lp_flip_trigger();
        let sf = build_standard_form(&lp);
        reset_bfrt_flip_invocations();
        let r1 = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        let flips = bfrt_flip_invocations();
        assert_eq!(r1.status, SolveStatus::Optimal);
        assert!(
            flips > 0,
            "warm_start_reuse cold solve: flip count = 0, bounded path not exercised"
        );
        let ws = r1
            .warm_start_basis
            .expect("bounded path must return warm_start_basis");
        let r2 = solve_dual_advanced(
            &sf,
            &lp,
            &SolverOptions {
                warm_start: Some(ws),
                ..SolverOptions::default()
            },
        );
        assert_eq!(
            r2.status,
            SolveStatus::Optimal,
            "warm restart: {:?}",
            r2.status
        );
        assert!(
            (r2.objective - r1.objective).abs() < 1e-5,
            "warm restart obj drift: {} vs {}",
            r2.objective,
            r1.objective
        );
    }

    /// **Sentinel**: warm-start with a basis that is dual-infeasible under the
    /// new cost vector must NOT return the wrong objective.
    ///
    /// LP1: `min x0+x1, x0+x1 ≤ 3, x0,x1 ≥ 0` → optimal basis {slack}, obj=0.
    /// LP2: `min -x0-x1, x0+x1 ≤ 3` — same structure, c flipped.
    /// The warm basis {slack} has x_B=[3] ≥ 0 (no lb-violation), but r_x0=r_x1=-1
    /// (dual infeasible under LP2's cost). Without the guard, dual simplex exits
    /// immediately as Optimal with obj=0 (WRONG). With the guard, falls through to
    /// cold start → obj=-3 (correct).
    ///
    /// no-op proof: if `warm_basis_is_dual_feasible` always returns `true` (guard
    /// is a no-op), the dual simplex warm-start uses the dual-infeasible basis and
    /// returns obj≈0 instead of -3 → assertion fails.
    #[test]
    fn warm_start_dual_infeasible_cost_change_falls_through_to_cold_start() {
        use crate::sparse::CscMatrix;
        const OBJ_TOL: f64 = 1e-6;

        // LP1: min x0+x1, x0+x1 ≤ 3, x0,x1 ≥ 0.
        // No finite UBs → legacy dual path.
        let make_lp = |c: Vec<f64>| {
            let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
            LpProblem::new_general(
                c,
                a,
                vec![3.0],
                vec![ConstraintType::Le],
                vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
                None,
            )
            .unwrap()
        };

        // Cold solve LP1: optimal basis is {slack=col 2}, x_B=[3], obj=0.
        let lp1 = make_lp(vec![1.0, 1.0]);
        let sf1 = build_standard_form(&lp1);
        let r1 = solve_dual_advanced(&sf1, &lp1, &SolverOptions::default());
        assert_eq!(
            r1.status,
            SolveStatus::Optimal,
            "LP1 cold solve: {:?}",
            r1.status
        );
        assert!(
            r1.objective.abs() < OBJ_TOL,
            "LP1 obj={:.6e} expected 0",
            r1.objective
        );
        let ws = r1
            .warm_start_basis
            .expect("LP1 must return warm_start_basis");

        // Warm-solve LP2: min -x0-x1 (cost flipped). The LP1 optimal warm basis
        // {slack} is dual-infeasible: r_x0=r_x1=-1 < 0 under LP2's cost.
        // Guard must fall through to cold start → correct obj=-3.
        let lp2 = make_lp(vec![-1.0, -1.0]);
        let sf2 = build_standard_form(&lp2);
        let r2 = solve_dual_advanced(
            &sf2,
            &lp2,
            &SolverOptions {
                warm_start: Some(ws),
                ..SolverOptions::default()
            },
        );
        assert_eq!(
            r2.status,
            SolveStatus::Optimal,
            "LP2 warm-solve status: {:?} (expected Optimal)",
            r2.status
        );
        assert!(
            (r2.objective - (-3.0)).abs() < OBJ_TOL,
            "LP2 warm-solve obj={:.6e} expected -3 (got 0 = guard missing)",
            r2.objective
        );

        // Consistency: cold re-solve of LP2 must agree.
        let r2_cold = solve_dual_advanced(&sf2, &lp2, &SolverOptions::default());
        assert_eq!(r2_cold.status, SolveStatus::Optimal);
        assert!(
            (r2_cold.objective - r2.objective).abs() < OBJ_TOL,
            "cold {:.6e} != warm {:.6e}",
            r2_cold.objective,
            r2.objective
        );
    }

    /// Sentinel: Ge/Eq cold-start (primal-first dispatch) must solve optimally.
    ///
    /// dc658d4 changed dispatch order: primal is tried first for Ge/Eq problems.
    /// Big-M is the fallback only when primal fails (Timeout with empty solution).
    /// This sentinel validates the end-to-end correctness of the new dispatch.
    ///
    /// no-op proof: removing the Ge/Eq dispatch branch (e.g. routing all Ge/Eq
    /// to Big-M and skipping primal) causes `Optimal` with wrong objective when
    /// Big-M stalls; a strict obj check would catch that.
    #[test]
    fn ge_eq_cold_start_primal_first_dispatch_solves_optimally() {
        use crate::sparse::CscMatrix;
        const OBJ_TOL: f64 = 1e-6;

        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);

        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());

        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - 3.0).abs() < OBJ_TOL,
            "Ge/Eq LP should have obj=3.0, got {:.6e}",
            result.objective
        );
    }

    /// Sentinel: warm basis from a previously-bounded Optimal solve must not mask
    /// a genuinely-infeasible next LP. LP2 has `num_artificial != 0` so the bounded
    /// dispatch gate is bypassed and the warm leg is routed through the legacy path;
    /// this guards that the Farkas-Infeasible return is preserved.
    ///
    /// no-op proof: replacing the `Infeasible` return in `two_phase_dual_simplex`
    /// with `Optimal` causes this assertion to FAIL.
    #[test]
    fn warm_basis_from_bounded_dispatch_does_not_mask_farkas_infeasibility() {
        use crate::sparse::CscMatrix;
        const OBJ_TOL: f64 = 1e-6;

        let make_lp = |b_rhs: f64| {
            let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
            LpProblem::new_general(
                vec![-1.0, -3.0],
                a,
                vec![b_rhs],
                vec![ConstraintType::Le],
                vec![(0.0, 4.0), (0.0, 2.0)],
                None,
            )
            .unwrap()
        };

        // Cold solve LP1 via bounded dispatch (Le-only, finite UBs).
        let lp1 = make_lp(5.0);
        let sf1 = build_standard_form(&lp1);
        let r1 = solve_dual_advanced(&sf1, &lp1, &SolverOptions::default());
        assert_eq!(r1.status, SolveStatus::Optimal, "LP1 cold: {:?}", r1.status);
        assert!(
            (r1.objective - (-9.0)).abs() < OBJ_TOL,
            "LP1 obj={:.6e} expected -9",
            r1.objective
        );
        let warm = r1
            .warm_start_basis
            .expect("bounded cold solve must return warm_start_basis");

        // LP2: x0+x1 ≤ -1 is infeasible since x0,x1 ≥ 0.
        let lp2 = make_lp(-1.0);
        let sf2 = build_standard_form(&lp2);
        let r2 = solve_dual_advanced(
            &sf2,
            &lp2,
            &SolverOptions {
                warm_start: Some(warm),
                ..SolverOptions::default()
            },
        );
        assert_eq!(
            r2.status,
            SolveStatus::Infeasible,
            "LP2 (x0+x1 ≤ -1, finite UBs) must be Infeasible; got {:?}",
            r2.status
        );

        // Cold solve of LP2 must also return Infeasible.
        let r2_cold = solve_dual_advanced(&sf2, &lp2, &SolverOptions::default());
        assert_eq!(
            r2_cold.status,
            SolveStatus::Infeasible,
            "LP2 cold: expected Infeasible, got {:?}",
            r2_cold.status
        );
    }

    #[test]
    fn warm_start_with_expired_deadline_returns_timeout() {
        let lp = lp_2x2_boxed();
        let sf = build_standard_form(&lp);
        let options = SolverOptions {
            warm_start: Some(WarmStartBasis {
                basis: sf.initial_basis.clone(),
                x_b: vec![],
            }),
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
            ..SolverOptions::default()
        };

        let result = solve_dual_advanced(&sf, &lp, &options);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    // ── Eq + UB Phase I (new path) sentinels ─────────────────────────────────

    /// `min  x0 + x1  s.t.  x0 + x1 = 3,  0 ≤ x0 ≤ 2,  0 ≤ x1 ≤ 2`.
    /// Unique optimal: x0=2 (at UB), x1=1, obj=3. Eq row forces an artificial;
    /// the finite UB on x0 means the Eq+UB Phase I path is dispatched and the
    /// at-upper state on x0 is exercised end-to-end.
    fn lp_eq_with_finite_ubs() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 2.0), (0.0, 2.0)],
            None,
        )
        .unwrap()
    }

    /// **Dispatch sentinel**: solving an Eq+UB LP via `solve_dual_advanced`
    /// must increment `eq_ub_dispatch_count` (the new path was taken).
    ///
    /// No-op proof: `eq_ub_dispatch_noop_proof` disables the bounded dispatch
    /// hook and asserts the counter stays at 0.
    #[test]
    fn eq_ub_dispatch_count_positive() {
        let lp = lp_eq_with_finite_ubs();
        let sf = build_standard_form(&lp);
        bounded_core::reset_eq_ub_dispatch_count();
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal, "expected Optimal");
        assert!(
            (result.objective - 3.0).abs() < 1e-6,
            "expected obj=3, got {:.6e}",
            result.objective
        );
        let count = bounded_core::eq_ub_dispatch_count();
        assert!(
            count > 0,
            "eq_ub_dispatch_count_positive: counter = 0, new path not exercised"
        );
    }

    /// **No-op proof**: disabling bounded dispatch keeps the counter at 0 and
    /// the legacy path still produces the correct answer.
    #[test]
    fn eq_ub_dispatch_noop_proof() {
        let lp = lp_eq_with_finite_ubs();
        let sf = build_standard_form(&lp);
        let _guard = crate::ScopedDisable::new(
            || set_bounded_dispatch_disabled(true),
            || set_bounded_dispatch_disabled(false),
        );
        bounded_core::reset_eq_ub_dispatch_count();
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        let count = bounded_core::eq_ub_dispatch_count();
        assert_eq!(
            count, 0,
            "noop proof: expected 0 dispatches with bounded disabled, got {count}"
        );
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective - 3.0).abs() < 1e-6);
    }

    /// **Optimality + at-upper correctness** under the new path. x0 sits at
    /// its upper bound (=2) in the optimum; if the at-upper accounting in
    /// `extract_solution_bounded` were broken, x0 would come back as 0 and
    /// `c·x = 1` (not 3).
    #[test]
    fn eq_ub_phase1_solution_recovers_at_upper_variable() {
        let lp = lp_eq_with_finite_ubs();
        let sf = build_standard_form(&lp);
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.solution.len(), 2);
        // x0 at UB = 2, x1 = 1.
        assert!(
            (result.solution[0] - 2.0).abs() < 1e-6,
            "expected x0=2, got {:.6e}",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 1.0).abs() < 1e-6,
            "expected x1=1, got {:.6e}",
            result.solution[1]
        );
    }

    /// Infeasible Eq+UB: the Phase I path must declare Infeasible without
    /// falling back when `min sum(artificials) > 0`.
    ///
    /// `x0 + x1 = 10, 0 ≤ x0 ≤ 2, 0 ≤ x1 ≤ 2` — sum is bounded by 4, cannot
    /// reach 10.
    #[test]
    fn eq_ub_phase1_detects_infeasibility() {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 2.0), (0.0, 2.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "expected Infeasible, got {:?}",
            result.status
        );
    }

    /// Multi-pattern coverage of the Eq+UB Phase I path:
    /// - Eq + boxed at corner (already covered by lp_eq_with_finite_ubs)
    /// - Eq with one half-bounded variable
    /// - Mixed Le + Eq with finite UBs
    #[test]
    fn eq_ub_phase1_multi_pattern_correct() {
        use crate::sparse::CscMatrix;
        // Pattern A: Eq + one half-bounded var.
        // min x0 + x1, x0 + x1 = 5, 0 ≤ x0 ≤ 3, 0 ≤ x1 (no ub)
        // Optimal: x0=3 (or any split summing to 5), obj=5.
        {
            let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
            let lp = LpProblem::new_general(
                vec![1.0, 1.0],
                a,
                vec![5.0],
                vec![ConstraintType::Eq],
                vec![(0.0, 3.0), (0.0, f64::INFINITY)],
                None,
            )
            .unwrap();
            let sf = build_standard_form(&lp);
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            assert_eq!(r.status, SolveStatus::Optimal, "pattern A");
            assert!(
                (r.objective - 5.0).abs() < 1e-6,
                "pattern A obj={}",
                r.objective
            );
        }
        // Pattern B: Le + Eq + finite UBs.
        // min -x0 - x1, x0 + x1 = 4, x0 ≤ 5, 0 ≤ x0 ≤ 3, 0 ≤ x1 ≤ 3
        // Optimal: x0=3, x1=1, obj=-4. (Equivalent: x0=1, x1=3.)
        {
            let a =
                CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
                    .unwrap();
            let lp = LpProblem::new_general(
                vec![-1.0, -1.0],
                a,
                vec![5.0, 4.0],
                vec![ConstraintType::Le, ConstraintType::Eq],
                vec![(0.0, 3.0), (0.0, 3.0)],
                None,
            )
            .unwrap();
            let sf = build_standard_form(&lp);
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            assert_eq!(r.status, SolveStatus::Optimal, "pattern B");
            assert!(
                (r.objective - (-4.0)).abs() < 1e-6,
                "pattern B obj={}",
                r.objective
            );
        }
    }

    /// **Direct sentinel** for the scaled-slack initial value (codex review
    /// P1). The starting slack/artificial basis is diagonal, but after Ruiz
    /// scaling the Le-slack diagonal is not 1, so `x_B = B^{-1} b` must divide
    /// by that diagonal — `b.clone()` is wrong whenever `diag ≠ 1`. Here the
    /// basis columns have diagonals 2 and 4, so `x_B` must be `[5, 3]`, not
    /// `[10, 12]`. Returns the raw RHS (fails) if the division is dropped.
    #[test]
    fn diag_basis_initial_x_b_divides_by_diagonal() {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 4.0], 2, 2).unwrap();
        let x_b = diag_basis_initial_x_b(&a, &[0, 1], &[10.0, 12.0]);
        assert!(
            (x_b[0] - 5.0).abs() < 1e-12,
            "row0: expected b/diag=5, got {}",
            x_b[0]
        );
        assert!(
            (x_b[1] - 3.0).abs() < 1e-12,
            "row1: expected b/diag=3, got {}",
            x_b[1]
        );
    }

    // ── P2-b: perturbation sentinels ─────────────────────────────────────────

    /// Degenerate 2-Eq + UB LP for perturbation sentinels.
    ///
    /// `min x0+x1+x2`, `x0+x1=1` (Eq), `x1+x2=1` (Eq), `0 ≤ xᵢ ≤ 1`.
    /// Unique optimal: `x0=0, x1=1, x2=0`, `obj=1`. Two Eq rows → two initial
    /// artificials (crash disabled), so `perturb_x_b_with_mag` affects row 1
    /// (frac₁ ≈ 0.618 ≠ 0).
    fn lp_degenerate_2eq() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
        )
        .unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![1.0, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap()
    }

    /// Sentinel (P2-b): perturbation ON (mag ∈ {1e-7, 1e-2}) and OFF reach
    /// the same optimal status, objective, and solution.
    ///
    /// `maybe_perturb_initial_xb` jitters the initial `x_B` before Phase I of
    /// the Eq+UB path. `reconcile_bounded_terminal_state` then recomputes
    /// `x_B = B⁻¹b` (true `b`) at the end of Phase I and Phase II, removing
    /// the jitter before the solution is reported.
    ///
    /// No-op proof: see `perturb_noop_proof_reconcile_removes_perturbation`.
    #[test]
    fn perturb_on_off_same_optimal_degenerate_eq_ub() {
        let lp = lp_degenerate_2eq();
        let sf = build_standard_form(&lp);
        const OBJ_TOL: f64 = 1e-6;

        // Reference: perturbation OFF.
        let opts_off = SolverOptions {
            use_lp_crash_basis: false,
            ..SolverOptions::default()
        };
        let r_off = solve_dual_advanced(&sf, &lp, &opts_off);
        assert_eq!(
            r_off.status,
            SolveStatus::Optimal,
            "OFF: expected Optimal, got {:?}",
            r_off.status
        );
        assert!(
            (r_off.objective - 1.0).abs() < OBJ_TOL,
            "OFF: obj={:.9e} expected 1.0",
            r_off.objective
        );
        assert_eq!(r_off.solution.len(), 3);
        assert!((r_off.solution[1] - 1.0).abs() < OBJ_TOL, "OFF: x1={:.6e}", r_off.solution[1]);

        for &mag in &[1e-7_f64, 1e-2_f64] {
            let _g = crate::ScopedDisable::new(
                || set_perturb_mag_override(Some(mag)),
                || set_perturb_mag_override(None),
            );
            let opts_on = SolverOptions {
                use_lp_crash_basis: false,
                ..SolverOptions::default()
            };
            let r_on = solve_dual_advanced(&sf, &lp, &opts_on);
            assert_eq!(
                r_on.status,
                SolveStatus::Optimal,
                "mag={mag:.0e}: expected Optimal, got {:?}",
                r_on.status
            );
            assert!(
                (r_on.objective - r_off.objective).abs() < OBJ_TOL,
                "mag={mag:.0e}: perturbed obj={:.9e} != off obj={:.9e} (reconcile did not remove perturbation)",
                r_on.objective,
                r_off.objective
            );
            assert!(
                (r_on.solution[1] - 1.0).abs() < OBJ_TOL,
                "mag={mag:.0e}: x1={:.6e} expected 1.0",
                r_on.solution[1]
            );
        }
    }

    /// No-op proof (P2-b): applying perturbation to a known-optimal `x_B`
    /// yields a wrong objective *before* reconcile, and the true objective
    /// *after* reconcile.
    ///
    /// Design: if this test's first `assert!` (perturbation changes obj) were
    /// removed and `reconcile_bounded_terminal_state` were also skipped, the
    /// second `assert!` (reconciled obj = true) would fail — proving that
    /// reconcile is the mechanism that removes the perturbation.
    ///
    /// System: `A = diag(1,1)`, `b = [2, 3]`, `c = [1, 1]`, `ubs = [∞, ∞]`.
    /// Optimal basis = {x0, x1}, `x_B = [2, 3]`, `obj = 5`.
    /// After `perturb_x_b_with_mag(x_b, 0.01)`: row 0 unchanged (frac₀=0),
    /// row 1 shifted by ≈ 0.0247 → `obj ≈ 5.025 ≠ 5`.
    #[test]
    fn perturb_noop_proof_reconcile_removes_perturbation() {
        use crate::sparse::CscMatrix;
        const TRUE_OBJ: f64 = 5.0;
        const OBJ_TOL: f64 = 1e-6;
        const MAG: f64 = 0.01;

        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let b = vec![2.0_f64, 3.0];
        let c = vec![1.0_f64, 1.0];
        let ubs = vec![f64::INFINITY; 2];

        // Optimal state with PERTURBED x_b (simulates pre-reconcile state).
        let mut x_b = vec![2.0_f64, 3.0];
        test_apply_perturb_with_mag(&mut x_b, MAG);
        // frac[0]=0 → x_b[0] unchanged; frac[1]≈0.618 → x_b[1] ≈ 3.025.

        let mut state = BoundedDualState {
            basis: vec![0, 1],
            at_upper: vec![false, false],
            x_b: x_b.clone(),
            reduced_costs: vec![0.0; 2],
            is_basic: vec![true, true],
            iterations: 0,
            price_start: 0,
        };

        // Before reconcile: perturbed x_b gives wrong objective.
        let before_obj = bounded_obj_from_state(&c, &ubs, &state);
        assert!(
            (before_obj - TRUE_OBJ).abs() > OBJ_TOL,
            "no-op proof precondition: perturbation must change obj \
             ({before_obj:.9e} vs {TRUE_OBJ:.9e}); if this fails, \
             `perturb_x_b_with_mag` is broken"
        );

        // After reconcile: x_b = B⁻¹b (true b) → correct objective.
        let after_obj = match reconcile_bounded_terminal_state(
            &a,
            &b,
            &c,
            &ubs,
            &mut state,
            &SolverOptions::default(),
        ) {
            BoundedTerminalReconcile::Optimal(obj) => obj,
            _ => panic!("reconcile must succeed on a valid optimal state"),
        };
        assert!(
            (after_obj - TRUE_OBJ).abs() < OBJ_TOL,
            "reconcile must restore true obj: expected {TRUE_OBJ:.9e}, got {after_obj:.9e}"
        );
    }

    /// Integration smoke: mixed Eq+Le with a large Le coefficient solves
    /// correctly through the new path. Ruiz usually equilibrates the slack
    /// diagonal back to ≈1 here, so this is coverage rather than a strict
    /// no-op proof of the division (that is the direct test above).
    ///
    /// LP: `min -x1`, `x0 - x1 = 0` (Eq → artificial → new path),
    /// `100 x1 ≤ 100` (Le), `0 ≤ x0 ≤ 2`, `0 ≤ x1 ≤ 2`. Optimum: x0=x1=1, obj=-1.
    #[test]
    fn eq_ub_phase1_scaled_le_slack_feasible() {
        use crate::sparse::CscMatrix;
        // row 0 Eq: x0 - x1 = 0 ; row 1 Le: 100 x1 <= 100
        let a =
            CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, -1.0, 100.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, -1.0],
            a,
            vec![0.0, 100.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 2.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        bounded_core::reset_eq_ub_dispatch_count();
        let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        assert!(
            bounded_core::eq_ub_dispatch_count() > 0,
            "new Eq+UB path not dispatched"
        );
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "feasible LP misreported as {:?} (scaled-slack init bug)",
            r.status
        );
        assert!(
            (r.objective - (-1.0)).abs() < 1e-6,
            "expected obj=-1, got {:.6e}",
            r.objective
        );
        // Eq constraint x0 - x1 = 0 must hold on the returned solution.
        assert!(
            (r.solution[0] - r.solution[1]).abs() < 1e-6,
            "Eq x0-x1=0 violated: x0={}, x1={}",
            r.solution[0],
            r.solution[1]
        );
    }

    /// **BFRT flip count > 0 under the new path** — confirms the at-upper
    /// flip path inside `primal_simplex_aug` is reachable. Eq forces an
    /// artificial; the structural entering variable must hit its UB before
    /// the artificial leaves at lb to trigger a flip.
    ///
    /// LP: `min -3 x0 - x1`, `x0 + x1 = 4`, `0 ≤ x0 ≤ 2`, `0 ≤ x1 ≤ 3`.
    /// Optimal: x0=2 (UB), x1=2, obj=-8. x0 enters first (largest |rc|);
    /// ratio test: leaving via artificial at row 0 gives step = 4 (artificial
    /// hits lb at 4), but x0's UB = 2 < 4 → flip x0 to UB. Then x1 enters
    /// and pushes the artificial out at step=2.
    ///
    /// Crash is disabled so the new path runs from identity start — with
    /// crash on, the crash basis would put a structural column in the row
    /// and overshoot its UB, triggering the legacy fall-through.
    #[test]
    fn eq_ub_phase1_bfrt_flip_count_positive() {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![-3.0, -1.0],
            a,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 2.0), (0.0, 3.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let opts = SolverOptions {
            use_lp_crash_basis: false,
            ..SolverOptions::default()
        };
        reset_bfrt_flip_invocations();
        let r = solve_dual_advanced(&sf, &lp, &opts);
        let flips = bfrt_flip_invocations();
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!(
            (r.objective - (-8.0)).abs() < 1e-6,
            "expected obj=-8, got {:.6e}",
            r.objective
        );
        assert!(
            flips > 0,
            "eq_ub_phase1_bfrt_flip_count_positive: flip count = 0, BFRT not exercised in new path"
        );
    }

    // P1-hypothesis: degenerate Phase I leaves a basic artificial at 0; in
    // Phase II a structural entering column has a NEGATIVE eta in the artificial
    // row, ratio-test skips it (ub=∞), and the step grows the artificial back
    // positive — yielding a solution violating the Eq constraint.
    //
    // LP: min -x1; x0-x1=0 (Eq, b=0); x1≤3 (Le); 0≤x0≤2, 0≤x1≤3.
    // True optimum: x0=x1=2, obj=-2. Bug returns x0=0, x1=3, obj=-3 (Eq=-3≠0).
    // Asserts (a) Eq satisfaction and (b) correct objective.
    #[test]
    fn p1_hypothesis_art_goes_positive_in_phase2() {
        use crate::sparse::CscMatrix;
        // x0 - x1 = 0 (row 0 Eq), x1 <= 3 (row 1 Le)
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, -1.0, 1.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, -1.0],
            a,
            vec![0.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 3.0)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        // Disable crash so identity-start is used (crash might sidestep the
        // degenerate artificial).
        let opts = SolverOptions {
            use_lp_crash_basis: false,
            ..SolverOptions::default()
        };
        // Verify the new path is taken.
        bounded_core::reset_eq_ub_dispatch_count();
        let result = solve_dual_advanced(&sf, &lp, &opts);
        let dispatch_count = bounded_core::eq_ub_dispatch_count();
        assert!(
            dispatch_count > 0,
            "new Eq+UB path was NOT dispatched (count=0), test is not exercising the new code"
        );
        // Must be Optimal.
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "expected Optimal, got {:?}",
            result.status
        );
        assert_eq!(result.solution.len(), 2, "solution length wrong");
        let x0 = result.solution[0];
        let x1 = result.solution[1];
        // Eq constraint: x0 - x1 = 0 must hold.
        assert!(
            (x0 - x1).abs() < 1e-6,
            "Eq constraint violated: x0={x0:.6e}, x1={x1:.6e}, diff={:.6e}",
            (x0 - x1).abs()
        );
        // True optimal: x0 = x1 = 2, obj = -2.
        assert!(
            (result.objective - (-2.0)).abs() < 1e-6,
            "expected obj=-2, got {:.6e}",
            result.objective
        );
    }
}
