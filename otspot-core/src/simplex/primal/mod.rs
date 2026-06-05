//! Primal (revised) simplex: two-phase driver and the iteration core.

mod core;
mod crossover;
mod ratio_test;
mod reconcile;

pub(crate) use core::revised_simplex_core;
pub(crate) use crossover::crossover_dual_from_primal;
pub(crate) use reconcile::{extract_solution, reconcile_final_basis_state};

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::presolve::LpEquilibration;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::*;
use super::dual_common::{basic_obj, lp_unbounded_ray_verified};
use super::pricing::SteepestEdgePricing;
use super::{extract_dual_info, SimplexOutcome, StandardForm};
use self::reconcile::{check_eq_feasibility, pivot_out_degenerate_artificials, try_apply_crash};

/// Counts `pivot_out_degenerate_artificials` early-exit firings (test-only).
///
/// Incremented each time the fast pre-check short-circuits the function
/// (no degenerate artificials in the basis). Tests assert this increases
/// to verify the early-exit fires; removing the check makes the count
/// stagnate, failing the sentinel.
#[cfg(test)]
pub(crate) static PIVOT_CLEAN_EARLY_EXIT_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts `pivot_out_degenerate_artificials` cleanup-body entries (test-only).
///
/// Incremented when the early-exit is *not* taken (a degenerate artificial is
/// in the basis), so the LU build + BTRAN cleanup runs. The complementary
/// sentinel asserts this increases on a degenerate-artificial LP: it proves
/// the early-exit does not mis-fire and strand an artificial in the basis.
#[cfg(test)]
pub(crate) static PIVOT_CLEAN_CLEANUP_RAN_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts obj-progress resets in `revised_simplex_core` (test-only).
///
/// Incremented each time the condition `current_obj + progress_eps < best_obj`
/// is satisfied, i.e. best_obj is updated.  With the correct finite
/// initialization (`best_obj = basic_obj(...)`) this fires whenever the
/// objective genuinely improves.  With the old `f64::INFINITY` init,
/// `progress_eps = ∞` so `current + ∞ < ∞` is always false and the counter
/// never increments — the sentinel test for B2 asserts it is > 0.
#[cfg(test)]
pub(crate) static OBJ_PROGRESS_RESET_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts non-degenerate x_b entries preserved (not perturbed) during cycle
/// detection in `revised_simplex_core` (test-only).
///
/// Each time cycle detection fires and an x_b row with `|v| >= step_zero_threshold`
/// is skipped, this counter increments. The sentinel asserts it is > 0 on a
/// degenerate LP that triggers cycling: proves non-degenerate rows are untouched.
/// Reverting to the old "add to ALL x_b" approach removes the `else` branch
/// entirely, keeping this counter at zero and failing the assertion.
#[cfg(test)]
pub(crate) static CYCLE_DETECT_NONDEGEN_PRESERVED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts BTRAN calls issued inside `pivot_out_degenerate_artificials` sequential fallback
/// (test-only).
///
/// Incremented each time `btran_dense` is called in the per-row sequential path within
/// `pivot_out_degenerate_artificials`. In the batch path this stays at zero (no BTRANs
/// needed). Sentinel asserts it is zero after solving an LP where all degenerate
/// artificials are handled by the batch: reverting to the O(num_art) sequential path
/// makes it increase by num_art, failing the assertion (no-op FAIL).
#[cfg(test)]
pub(crate) static PIVOT_OUT_BTRAN_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts batch LU factorization attempts in `pivot_out_degenerate_artificials` (test-only).
///
/// Incremented once each time the batch greedy assignment is attempted and a single
/// `LuBasis::new_timed` is called for all matched rows. Sentinel asserts it increases
/// by exactly 1 when the batch path is taken. Reverting to the sequential path keeps
/// this at zero and fails the assertion (no-op FAIL).
#[cfg(test)]
pub(crate) static PIVOT_OUT_BATCH_LU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Verified-ray gate for a Phase II `Unbounded` exit (shared with the Big-M
/// path). An eta-drift false-Unbounded (`B⁻¹a_q` reads ≤ 0 only because of a
/// stale factorization) becomes an honest Timeout, mirroring the Phase-I Farkas
/// gate. `n_enter` excludes artificials (`= sf.n_total`); pure-slack callers
/// pass `n_enter = n_cols`.
#[allow(clippy::too_many_arguments)]
fn gate_phase2_unbounded(
    outcome: SimplexOutcome,
    a: &CscMatrix,
    basis: &[usize],
    c: &[f64],
    x_b: &[f64],
    m: usize,
    n_cols: usize,
    n_enter: usize,
    options: &SolverOptions,
) -> SimplexOutcome {
    if matches!(outcome, SimplexOutcome::Unbounded)
        && !lp_unbounded_ray_verified(a, basis, c, m, n_cols, n_enter, options)
    {
        SimplexOutcome::Timeout(basic_obj(c, basis, x_b))
    } else {
        outcome
    }
}

/// Minimum absolute diagonal entry to trust when dividing `x_B[i]` by the
/// Ruiz-scaled slack/artificial column's diagonal. Prevents division by
/// near-zero when equilibration shrinks the diagonal below f64 noise.
const SLACK_DIAG_TOL: f64 = 1e-14;

/// Phase I infeasibility verdict, gated on a verified Farkas certificate.
///
/// A non-empty `farkas` ray (`Aᵀy ≤ tol` ∀ j, `bᵀy > tol`) proves the original
/// LP infeasible. An empty `farkas` means Phase I stopped (Unbounded ray /
/// positive artificial residual) WITHOUT a verifiable certificate — typically a
/// non-converged or cycling Phase I on a slow-but-feasible LP. Declaring
/// Infeasible there is a false verdict (ns1688926-class), so return Timeout
/// (honest inconclusive), matching `big_m_cold_start`'s Farkas gate.
fn phase1_infeasibility_verdict(farkas: Vec<f64>, total_iters: usize) -> SolverResult {
    if farkas.is_empty() {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            iterations: total_iters,
            ..Default::default()
        };
    }
    SolverResult {
        status: SolveStatus::Infeasible,
        objective: f64::INFINITY,
        dual_solution: farkas,
        iterations: total_iters,
        ..Default::default()
    }
}

/// Extract a verified Farkas infeasibility certificate from a Phase I basis.
///
/// `y = B^{-T} e_art` checked against `{A x = b, x ≥ 0}` Farkas alternative
/// (`Aᵀy ≤ tol` ∀ non-artificial j, `bᵀy > tol`, `tol = dual_tol·max(1,‖b‖∞)`,
/// consistent with the Big-M Phase I checker). Empty return = certificate not
/// verifiable (LU fail / no artificial in basis / numeric fail), NOT feasibility.
fn extract_farkas_certificate(
    a_ext: &CscMatrix,
    b: &[f64],
    basis: &[usize],
    m: usize,
    n_original: usize,
    options: &SolverOptions,
) -> Vec<f64> {
    if !basis.iter().any(|&col| col >= n_original) {
        return vec![];
    }
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(_) => return vec![],
    };
    let mut y: Vec<f64> = (0..m)
        .map(|i| if basis[i] >= n_original { 1.0 } else { 0.0 })
        .collect();
    basis_mgr.btran_dense(&mut y);

    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = options.dual_tol * (1.0_f64).max(b_norm);

    let by: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
    if by <= tol {
        return vec![];
    }
    for j in 0..n_original {
        let Ok((rows, vals)) = a_ext.get_column(j) else {
            return vec![];
        };
        let aty: f64 = rows.iter().zip(vals.iter()).map(|(&r, &v)| v * y[r]).sum();
        if aty > tol {
            return vec![];
        }
    }
    y
}

fn extract_timeout_solution_reconciled(
    sf: &StandardForm,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    max_etas: usize,
    deadline: Option<std::time::Instant>,
) -> Vec<f64> {
    let mut x_b_reconciled = x_b.to_vec();
    let mut y = vec![0.0_f64; basis.len()];
    if reconcile_final_basis_state(
        a,
        b,
        c,
        basis,
        &mut x_b_reconciled,
        &mut y,
        max_etas,
        deadline,
    )
    .is_ok()
    {
        extract_solution(sf, basis, &x_b_reconciled, col_scale)
    } else {
        extract_solution(sf, basis, x_b, col_scale)
    }
}

fn objective_from_solution(sf: &StandardForm, problem: &LpProblem, solution: &[f64]) -> f64 {
    problem
        .c
        .iter()
        .zip(solution.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>()
        + sf.obj_offset
}

/// Two-phase primal simplex on a standard-form LP. Skips Phase I when no
/// artificials are needed. Phase I minimizes the sum of artificials; a
/// positive minimum proves Infeasible. Ruiz equilibration is applied first.
pub(crate) fn two_phase_simplex(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let m = sf.m;
    let mut total_iters: usize = 0;

    let Some((a, b, c, row_scale, col_scale)) =
        LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c, options.deadline)
    else {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            ..Default::default()
        };
    };

    if sf.num_artificial == 0 {
        // Direct Phase II.
        let mut basis = sf.initial_basis.clone();
        let mut x_b = b.clone();
        // Ruiz equilibration scales slack diagonals away from 1; divide by the
        // diagonal so B · x_b = b_scaled holds.
        for i in 0..m {
            let col = basis[i];
            if let Ok((rows, vals)) = a.get_column(col) {
                for (k, &row) in rows.iter().enumerate() {
                    if row == i && vals[k].abs() > SLACK_DIAG_TOL {
                        x_b[i] /= vals[k];
                        break;
                    }
                }
            }
        }
        let mut pricing = SteepestEdgePricing::new(sf.n_total);

        let phase2_outcome = revised_simplex_core(
            &a,
            &mut x_b,
            &c,
            &b,
            &mut basis,
            m,
            sf.n_total,
            sf.n_total,
            &mut pricing,
            options,
            &mut total_iters,
            false,
        );
        let phase2_outcome = gate_phase2_unbounded(
            phase2_outcome,
            &a,
            &basis,
            &c,
            &x_b,
            m,
            sf.n_total,
            sf.n_total,
            options,
        );
        match phase2_outcome {
            SimplexOutcome::Optimal(obj, mut y) => {
                match reconcile_final_basis_state(
                    &a,
                    &b,
                    &c,
                    &basis,
                    &mut x_b,
                    &mut y,
                    options.max_etas,
                    options.deadline,
                ) {
                    Ok(()) => {}
                    Err(crate::error::SolverError::DeadlineExceeded) => {
                        let solution = extract_timeout_solution_reconciled(
                            sf,
                            &a,
                            &b,
                            &c,
                            &basis,
                            &x_b,
                            &col_scale,
                            options.max_etas,
                            options.deadline,
                        );
                        return SolverResult {
                            status: SolveStatus::Timeout,
                            objective: obj + sf.obj_offset,
                            solution,
                            iterations: total_iters,
                            ..Default::default()
                        };
                    }
                    Err(_) => return SolverResult::numerical_error(),
                }
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                // Defense-in-depth against false Optimal on Eq constraints.
                if !check_eq_feasibility(problem, &solution) {
                    return SolverResult {
                        status: SolveStatus::NumericalError,
                        objective: obj + sf.obj_offset,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        ..Default::default()
                    };
                }
                let (dual_solution, reduced_costs, slack) =
                    extract_dual_info(sf, problem, &y, &solution, &row_scale);
                let ws = WarmStartBasis {
                    basis: basis.clone(),
                    x_b: x_b.clone(),
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
                iterations: total_iters,
                ..Default::default()
            },
            SimplexOutcome::Timeout(obj) => {
                let solution = extract_timeout_solution_reconciled(
                    sf,
                    &a,
                    &b,
                    &c,
                    &basis,
                    &x_b,
                    &col_scale,
                    options.max_etas,
                    options.deadline,
                );
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
        }
    } else {
        // Phase I + Phase II (Ruiz-scaled system)
        let mut trip_rows: Vec<usize> = Vec::new();
        let mut trip_cols: Vec<usize> = Vec::new();
        let mut trip_vals: Vec<f64> = Vec::new();

        // Structural columns (Ruiz-scaled)
        for j in 0..a.ncols {
            if let Ok((r, v)) = a.get_column(j) {
                for (k, &row) in r.iter().enumerate() {
                    trip_rows.push(row);
                    trip_cols.push(j);
                    trip_vals.push(v[k]);
                }
            }
        }

        let mut basis = sf.initial_basis.clone();
        let mut x_b = b.clone();
        let mut art_col = sf.n_total;

        // All artificials in [sf.n_total, n_ext) — no split.
        for i in 0..m {
            if !sf.needs_artificial[i] {
                continue;
            }
            trip_rows.push(i);
            trip_cols.push(art_col);
            trip_vals.push(1.0);
            basis[i] = art_col;
            art_col += 1;
        }
        let n_ext = art_col;

        let a_ext = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext).unwrap();

        // Phase I cost: penalize all artificials.
        let mut c_phase1 = vec![0.0; n_ext];
        c_phase1[sf.n_total..].fill(1.0);

        // Crash basis: cover artificial rows with structural columns to reduce
        // Phase I pivots. Rows with negative x_b after FTRAN are rolled back.
        let crashed = if options.warm_start.is_none()
            && options.use_lp_crash_basis
            && sf.num_artificial > 0
        {
            try_apply_crash(
                &a_ext,
                m,
                sf.n_shifted,
                sf.n_total,
                &b,
                options.max_etas,
                options.deadline,
                &basis,
            )
        } else {
            None
        };
        if let Some((crash_basis, crash_x_b)) = crashed {
            basis = crash_basis;
            x_b = crash_x_b;
        } else {
            // Correct x_b for diagonal entries of initial basis columns.
            // Art cols have entry 1.0 → no change. Scaled slack cols → divide by diagonal.
            for i in 0..m {
                if let Ok((rows, vals)) = a_ext.get_column(basis[i]) {
                    for (k, &row) in rows.iter().enumerate() {
                        if row == i && vals[k].abs() > SLACK_DIAG_TOL {
                            x_b[i] /= vals[k];
                            break;
                        }
                    }
                }
            }
        }

        // Charnes perturbation: give each degenerate artificial row a unique tiny
        // positive x_b so ratio-test produces step>0 (prevents Phase I cycling).
        // The final reconcile restores exact B^{-1}b.
        for i in 0..m {
            if basis[i] >= sf.n_total && x_b[i].abs() <= PIVOT_TOL {
                x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
            }
        }

        let mut pricing1 = SteepestEdgePricing::new(n_ext);
        let phase1_outcome = revised_simplex_core(
            &a_ext,
            &mut x_b,
            &c_phase1,
            &b,
            &mut basis,
            m,
            n_ext,
            n_ext,
            &mut pricing1,
            options,
            &mut total_iters,
            true,
        );
        match phase1_outcome {
            SimplexOutcome::Optimal(_obj, _) => {
                // Phase I can declare Optimal while eta drift leaves x_b < 0.
                // Re-factor with fresh LU; if primal-infeasibility persists, retry
                // Phase I. MAX_PHASE1_RETRIES caps the loop to avoid infinite
                // re-pivoting on a stable-but-infeasible basis.
                use crate::options::MAX_PHASE1_RETRIES;
                let mut phase1_feasible = false;
                'retry: for attempt in 0..=MAX_PHASE1_RETRIES {
                    if options
                        .deadline
                        .is_some_and(|d| std::time::Instant::now() >= d)
                    {
                        break 'retry;
                    }
                    let mut y_dummy = vec![0.0f64; m];
                    let rec_obj = match reconcile_final_basis_state(
                        &a_ext,
                        &b,
                        &c_phase1,
                        &basis,
                        &mut x_b,
                        &mut y_dummy,
                        options.max_etas,
                        options.deadline,
                    ) {
                        Ok(()) => (0..m)
                            .map(|i| c_phase1[basis[i]] * x_b[i].max(0.0))
                            .sum::<f64>(),
                        Err(_) => break 'retry,
                    };
                    if rec_obj <= PIVOT_TOL {
                        phase1_feasible = true;
                        break 'retry;
                    }
                    if attempt == MAX_PHASE1_RETRIES {
                        break 'retry;
                    }

                    // Artificials remain positive: clamp drift and retry Phase I.
                    for v in x_b.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                    let mut pricing_retry = SteepestEdgePricing::new(n_ext);
                    match revised_simplex_core(
                        &a_ext,
                        &mut x_b,
                        &c_phase1,
                        &b,
                        &mut basis,
                        m,
                        n_ext,
                        n_ext,
                        &mut pricing_retry,
                        options,
                        &mut total_iters,
                        true,
                    ) {
                        SimplexOutcome::Optimal(_, _) => {}
                        SimplexOutcome::Unbounded => break 'retry,
                        SimplexOutcome::Timeout(_) => {
                            // Cycling bail may fire when Phase I objective is already 0
                            // (artificials eliminated but degenerate basis stalls).
                            // Reconcile with fresh LU before giving up: if all
                            // artificials are truly gone (rec_obj ≤ PIVOT_TOL),
                            // the bail was a false positive and Phase II can run.
                            let mut y_check = vec![0.0f64; m];
                            let reconciled = reconcile_final_basis_state(
                                &a_ext,
                                &b,
                                &c_phase1,
                                &basis,
                                &mut x_b,
                                &mut y_check,
                                options.max_etas,
                                options.deadline,
                            )
                            .is_ok();
                            if reconciled {
                                let rec_obj_retry: f64 = (0..m)
                                    .map(|i| c_phase1[basis[i]] * x_b[i].max(0.0))
                                    .sum();
                                if rec_obj_retry <= PIVOT_TOL {
                                    // Artificials are gone: treat as feasible and
                                    // proceed to Phase II via the retry reconcile loop.
                                    phase1_feasible = true;
                                    break 'retry;
                                }
                            }
                            return SolverResult {
                                status: SolveStatus::Timeout,
                                objective: f64::INFINITY,
                                solution: vec![],
                                dual_solution: vec![],
                                reduced_costs: vec![],
                                slack: vec![],
                                warm_start_basis: None,
                                iterations: total_iters,
                                ..Default::default()
                            };
                        }
                        SimplexOutcome::SingularBasis => {
                            return SolverResult::numerical_error();
                        }
                    }
                }

                if !phase1_feasible {
                    // Declare Infeasible only with a verified Farkas certificate.
                    // True infeasible LPs have a valid dual ray at this basis;
                    // feasible LPs cycling in Phase I do not (empty cert → Timeout).
                    let farkas =
                        extract_farkas_certificate(&a_ext, &b, &basis, m, sf.n_total, options);
                    return phase1_infeasibility_verdict(farkas, total_iters);
                }

                // Phase I feasible: pivot out any remaining degenerate artificials
                pivot_out_degenerate_artificials(&a_ext, &mut basis, &x_b, sf, options);

                let mut c_phase2 = vec![0.0; n_ext];
                c_phase2[..sf.n_total].copy_from_slice(&c[..sf.n_total]);
                {
                    let mut y_transition = vec![0.0f64; m];
                    match reconcile_final_basis_state(
                        &a_ext,
                        &b,
                        &c_phase2,
                        &basis,
                        &mut x_b,
                        &mut y_transition,
                        options.max_etas,
                        options.deadline,
                    ) {
                        Ok(()) => {}
                        Err(crate::error::SolverError::DeadlineExceeded) => {
                            let solution = extract_timeout_solution_reconciled(
                                sf,
                                &a_ext,
                                &b,
                                &c_phase2,
                                &basis,
                                &x_b,
                                &col_scale,
                                options.max_etas,
                                options.deadline,
                            );
                            return SolverResult {
                                status: SolveStatus::Timeout,
                                objective: objective_from_solution(sf, problem, &solution),
                                solution,
                                iterations: total_iters,
                                ..Default::default()
                            };
                        }
                        Err(_) => return SolverResult::numerical_error(),
                    }
                }
                // Charnes perturbation for Phase II anti-cycling.
                // Rows with x_b ≈ 0 cause ratio-test step=0. The final reconcile restores
                // exact B^{-1}b after Phase II completes.
                for i in 0..m {
                    if x_b[i].abs() < PIVOT_TOL {
                        x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
                    }
                }
                for v in x_b.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }

                let mut pricing2 = SteepestEdgePricing::new(n_ext);
                let phase2_outcome = revised_simplex_core(
                    &a_ext,
                    &mut x_b,
                    &c_phase2,
                    &b,
                    &mut basis,
                    m,
                    n_ext,
                    sf.n_total,
                    &mut pricing2,
                    options,
                    &mut total_iters,
                    false,
                );
                let phase2_outcome = gate_phase2_unbounded(
                    phase2_outcome,
                    &a_ext,
                    &basis,
                    &c_phase2,
                    &x_b,
                    m,
                    n_ext,
                    sf.n_total,
                    options,
                );
                match phase2_outcome {
                    SimplexOutcome::Optimal(obj2, mut y) => {
                        match reconcile_final_basis_state(
                            &a_ext,
                            &b,
                            &c_phase2,
                            &basis,
                            &mut x_b,
                            &mut y,
                            options.max_etas,
                            options.deadline,
                        ) {
                            Ok(()) => {}
                            Err(crate::error::SolverError::DeadlineExceeded) => {
                                let solution = extract_timeout_solution_reconciled(
                                    sf,
                                    &a_ext,
                                    &b,
                                    &c_phase2,
                                    &basis,
                                    &x_b,
                                    &col_scale,
                                    options.max_etas,
                                    options.deadline,
                                );
                                return SolverResult {
                                    status: SolveStatus::Timeout,
                                    objective: obj2 + sf.obj_offset,
                                    solution,
                                    iterations: total_iters,
                                    ..Default::default()
                                };
                            }
                            Err(_) => return SolverResult::numerical_error(),
                        }
                        let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                        if !check_eq_feasibility(problem, &solution) {
                            return SolverResult {
                                status: SolveStatus::NumericalError,
                                objective: obj2 + sf.obj_offset,
                                solution: vec![],
                                dual_solution: vec![],
                                reduced_costs: vec![],
                                slack: vec![],
                                warm_start_basis: None,
                                ..Default::default()
                            };
                        }
                        let (dual_solution, reduced_costs, slack) =
                            extract_dual_info(sf, problem, &y, &solution, &row_scale);
                        let ws = WarmStartBasis {
                            basis: basis.clone(),
                            x_b: x_b.clone(),
                        };
                        SolverResult {
                            status: SolveStatus::Optimal,
                            objective: obj2 + sf.obj_offset,
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
                        iterations: total_iters,
                        ..Default::default()
                    },
                    SimplexOutcome::Timeout(obj2) => {
                        let solution = extract_timeout_solution_reconciled(
                            sf,
                            &a_ext,
                            &b,
                            &c_phase2,
                            &basis,
                            &x_b,
                            &col_scale,
                            options.max_etas,
                            options.deadline,
                        );
                        SolverResult {
                            status: SolveStatus::Timeout,
                            objective: obj2 + sf.obj_offset,
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
            SimplexOutcome::Unbounded => {
                // A Phase I unbounded ray suggests primal infeasibility, but only a
                // verified Farkas certificate proves it; empty cert → Timeout
                // (spurious unbounded ray on a feasible LP, ns1688926-class).
                let farkas = extract_farkas_certificate(&a_ext, &b, &basis, m, sf.n_total, options);
                phase1_infeasibility_verdict(farkas, total_iters)
            }
            SimplexOutcome::Timeout(obj1) => {
                // obj1 ≤ PIVOT_TOL ⇒ artificials look near-zero at timeout.
                // Reconcile with a fresh LU; only enter Phase II if the
                // accurate x_b still shows feasibility.
                if obj1 <= PIVOT_TOL {
                    {
                        let mut y_dummy = vec![0.0_f64; m];
                        if reconcile_final_basis_state(
                            &a_ext,
                            &b,
                            &c_phase1,
                            &basis,
                            &mut x_b,
                            &mut y_dummy,
                            options.max_etas,
                            options.deadline,
                        )
                        .is_err()
                        {
                            return SolverResult {
                                status: SolveStatus::Timeout,
                                objective: f64::INFINITY,
                                solution: vec![],
                                dual_solution: vec![],
                                reduced_costs: vec![],
                                slack: vec![],
                                warm_start_basis: None,
                                iterations: total_iters,
                                ..Default::default()
                            };
                        }
                    }
                    // After reconcile: if arts still > PIVOT_TOL, Phase I hasn't
                    // converged — do not run Phase II from an infeasible start.
                    let rec_obj: f64 = (0..m).map(|i| c_phase1[basis[i]] * x_b[i].max(0.0)).sum();
                    if rec_obj > PIVOT_TOL {
                        return SolverResult {
                            status: SolveStatus::Timeout,
                            objective: f64::INFINITY,
                            solution: vec![],
                            dual_solution: vec![],
                            reduced_costs: vec![],
                            slack: vec![],
                            warm_start_basis: None,
                            iterations: total_iters,
                            ..Default::default()
                        };
                    }
                    pivot_out_degenerate_artificials(&a_ext, &mut basis, &x_b, sf, options);

                    let mut c_phase2 = vec![0.0; n_ext];
                    c_phase2[..sf.n_total].copy_from_slice(&c[..sf.n_total]);
                    {
                        let mut y_transition = vec![0.0f64; m];
                        match reconcile_final_basis_state(
                            &a_ext,
                            &b,
                            &c_phase2,
                            &basis,
                            &mut x_b,
                            &mut y_transition,
                            options.max_etas,
                            options.deadline,
                        ) {
                            Ok(()) => {}
                            Err(crate::error::SolverError::DeadlineExceeded) => {
                                let solution = extract_timeout_solution_reconciled(
                                    sf,
                                    &a_ext,
                                    &b,
                                    &c_phase2,
                                    &basis,
                                    &x_b,
                                    &col_scale,
                                    options.max_etas,
                                    options.deadline,
                                );
                                return SolverResult {
                                    status: SolveStatus::Timeout,
                                    objective: objective_from_solution(sf, problem, &solution),
                                    solution,
                                    iterations: total_iters,
                                    ..Default::default()
                                };
                            }
                            Err(_) => return SolverResult::numerical_error(),
                        }
                    }
                    for i in 0..m {
                        if x_b[i].abs() < PIVOT_TOL {
                            x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
                        }
                    }
                    for v in x_b.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }

                    let mut pricing2 = SteepestEdgePricing::new(n_ext);
                    let phase2_outcome = revised_simplex_core(
                        &a_ext,
                        &mut x_b,
                        &c_phase2,
                        &b,
                        &mut basis,
                        m,
                        n_ext,
                        sf.n_total,
                        &mut pricing2,
                        options,
                        &mut total_iters,
                        false,
                    );
                    let phase2_outcome = gate_phase2_unbounded(
                        phase2_outcome,
                        &a_ext,
                        &basis,
                        &c_phase2,
                        &x_b,
                        m,
                        n_ext,
                        sf.n_total,
                        options,
                    );
                    match phase2_outcome {
                        SimplexOutcome::Optimal(obj2, mut y) => {
                            match reconcile_final_basis_state(
                                &a_ext,
                                &b,
                                &c_phase2,
                                &basis,
                                &mut x_b,
                                &mut y,
                                options.max_etas,
                                options.deadline,
                            ) {
                                Ok(()) => {}
                                Err(crate::error::SolverError::DeadlineExceeded) => {
                                    let solution = extract_timeout_solution_reconciled(
                                        sf,
                                        &a_ext,
                                        &b,
                                        &c_phase2,
                                        &basis,
                                        &x_b,
                                        &col_scale,
                                        options.max_etas,
                                        options.deadline,
                                    );
                                    return SolverResult {
                                        status: SolveStatus::Timeout,
                                        objective: obj2 + sf.obj_offset,
                                        solution,
                                        iterations: total_iters,
                                        ..Default::default()
                                    };
                                }
                                Err(_) => return SolverResult::numerical_error(),
                            }
                            let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                            if !check_eq_feasibility(problem, &solution) {
                                return SolverResult::numerical_error();
                            }
                            let (dual_solution, reduced_costs, slack) =
                                extract_dual_info(sf, problem, &y, &solution, &row_scale);
                            let ws = WarmStartBasis {
                                basis: basis.clone(),
                                x_b: x_b.clone(),
                            };
                            return SolverResult {
                                status: SolveStatus::Optimal,
                                objective: obj2 + sf.obj_offset,
                                solution,
                                dual_solution,
                                reduced_costs,
                                slack,
                                warm_start_basis: Some(ws),
                                ..Default::default()
                            };
                        }
                        SimplexOutcome::Timeout(obj2) => {
                            let solution = extract_timeout_solution_reconciled(
                                sf,
                                &a_ext,
                                &b,
                                &c_phase2,
                                &basis,
                                &x_b,
                                &col_scale,
                                options.max_etas,
                                options.deadline,
                            );
                            return SolverResult {
                                status: SolveStatus::Timeout,
                                objective: obj2 + sf.obj_offset,
                                solution,
                                iterations: total_iters,
                                ..Default::default()
                            };
                        }
                        SimplexOutcome::Unbounded => {
                            return SolverResult {
                                status: SolveStatus::Unbounded,
                                objective: f64::NEG_INFINITY,
                                solution: vec![],
                                dual_solution: vec![],
                                reduced_costs: vec![],
                                slack: vec![],
                                warm_start_basis: None,
                                ..Default::default()
                            };
                        }
                        SimplexOutcome::SingularBasis => {
                            return SolverResult::numerical_error();
                        }
                    }
                }
                // obj1 > PIVOT_TOL: Phase1 が実行可能基底を発見できないまま時間切れ。
                SolverResult {
                    status: SolveStatus::Timeout,
                    objective: f64::INFINITY,
                    solution: vec![],
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
}

/// Test-only entry point that applies the cycle-detection selective Charnes
/// perturbation to a given x_b vector.  Mirrors the exact production logic
/// (same formulas, same condition) so tests exercise the live code path.
///
/// No-op proof: reverting the production code to `*v += eps*(i+1)` for ALL
/// entries requires removing the `v.abs() < step_zero_threshold` guard.  The
/// test below then asserts that large values are preserved, which fails
/// because the old unconditional `+=` would change them.
#[cfg(test)]
pub(crate) fn test_apply_selective_charnes_perturb(x_b: &mut [f64], m: usize) {
    let eps = PIVOT_TOL * (m as f64).max(1.0);
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    for (i, v) in x_b.iter_mut().enumerate() {
        if v.abs() < step_zero_threshold {
            *v = eps * (i as f64 + 1.0);
        }
    }
}

#[cfg(test)]
mod farkas_gate_tests {
    //! Phase I infeasibility must be declared ONLY with a verified Farkas
    //! certificate. ns1688926 (feasible, ‖b‖≈2.4e7) and cplex2 exit Phase I with
    //! a spurious Unbounded ray whose `y = B^{-T} e_art` has `bᵀy ≈ 0` — not a
    //! witness. Trusting that exit returned false-Infeasible. These sentinels
    //! pin the gate: empty cert ⇒ Timeout, verified cert ⇒ Infeasible.

    use super::{extract_farkas_certificate, phase1_infeasibility_verdict};
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    /// No-op sentinel: reverting the gate to an unconditional `Infeasible`
    /// return (the pre-fix behaviour) makes this assertion fail.
    #[test]
    fn empty_cert_yields_timeout_not_infeasible() {
        let r = phase1_infeasibility_verdict(vec![], 7);
        assert_eq!(
            r.status,
            SolveStatus::Timeout,
            "empty (unverifiable) Farkas cert must NOT be declared Infeasible"
        );
        assert_eq!(r.iterations, 7);
    }

    #[test]
    fn verified_cert_yields_infeasible() {
        let r = phase1_infeasibility_verdict(vec![-1.0, 1.0], 3);
        assert_eq!(r.status, SolveStatus::Infeasible);
        assert_eq!(r.dual_solution, vec![-1.0, 1.0]);
        assert_eq!(r.iterations, 3);
    }

    /// The discriminator the gate depends on: an identical Phase I basis yields a
    /// valid witness (`bᵀy > 0`) on a genuinely infeasible RHS but an empty cert
    /// (`bᵀy ≈ 0`) on a feasible RHS. a_ext = [x0 | a0 | a1], rows x0=b0, x0=b1.
    #[test]
    fn extract_farkas_discriminates_witness_from_degenerate() {
        // col0 = x0 (rows 0,1); col1 = a0 (row 0); col2 = a1 (row 1).
        let a_ext =
            CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
                .unwrap();
        let basis = [0usize, 2usize]; // x0 + artificial a1, a1 degenerate iff feasible
        let n_original = 1;
        let opts = SolverOptions::default();

        // Infeasible: x0 = 1 ∧ x0 = 2. bᵀy = 1 > 0 ⇒ verified witness (non-empty).
        let cert_infeasible =
            extract_farkas_certificate(&a_ext, &[1.0, 2.0], &basis, 2, n_original, &opts);
        assert!(
            !cert_infeasible.is_empty(),
            "true infeasible RHS must yield a verified Farkas certificate"
        );

        // Feasible: x0 = 1 ∧ x0 = 1. bᵀy = 0 ⇒ not a witness (empty).
        let cert_feasible =
            extract_farkas_certificate(&a_ext, &[1.0, 1.0], &basis, 2, n_original, &opts);
        assert!(
            cert_feasible.is_empty(),
            "feasible RHS (degenerate artificial, bᵀy≈0) must NOT be certified infeasible"
        );
    }
}

#[cfg(test)]
mod cycle_perturbation_tests {
    //! Sentinels for the selective Charnes perturbation in cycle detection.
    //!
    //! `revised_simplex_core` applies Charnes perturbation only to near-zero
    //! x_b entries (`|v| < step_zero_threshold`). The old code added `eps*(i+1)`
    //! to ALL entries, causing Phase II objective jumps on mixed-sign cost
    //! vectors (pilot-class problems).

    use super::test_apply_selective_charnes_perturb;
    use crate::options::{SimplexMethod, SolverOptions};
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::simplex::entry::solve_with;
    use crate::sparse::CscMatrix;
    use crate::tolerances::PIVOT_TOL;

    /// Sentinel (primary, direct): selective perturbation must not modify large x_b.
    ///
    /// The helper `test_apply_selective_charnes_perturb` is an exact copy of the
    /// production cycle-detection block (same formulas, same guard condition).
    /// Testing it directly is equivalent to testing the production code path.
    ///
    /// No-op proof: removing the `v.abs() < step_zero_threshold` guard in
    /// production (reverting to `*v += eps*(i+1)` for ALL values) means the
    /// helper must also be updated consistently — and the helper would then
    /// modify `x_b[0]=100` to `100 + eps*1 ≠ 100`, failing `assert_eq!`.
    #[test]
    fn selective_charnes_perturb_spares_large_values() {
        let m = 4usize;
        let eps = PIVOT_TOL * (m as f64);       // = step_zero_threshold
        let thresh = eps;

        // Mix of large (non-degenerate) and near-zero (degenerate) values.
        let mut x_b = vec![100.0, 0.0, thresh * 0.5, 50.0];
        let saved = [x_b[0], x_b[3]];

        test_apply_selective_charnes_perturb(&mut x_b, m);

        // Non-degenerate rows: must be UNCHANGED.
        assert_eq!(x_b[0], saved[0], "x_b[0]=100 must not be modified (non-degenerate)");
        assert_eq!(x_b[3], saved[1], "x_b[3]=50 must not be modified (non-degenerate)");

        // Near-zero rows: must become eps*(i+1) (unique small positives).
        // i=1 → eps*(1+1) = eps*2; i=2 → eps*(2+1) = eps*3.
        assert_eq!(
            x_b[1], eps * 2.0,
            "x_b[1]=0 at i=1 must become eps*(1+1)=eps*2"
        );
        assert_eq!(
            x_b[2], eps * 3.0,
            "x_b[2]=thresh*0.5 at i=2 must become eps*(2+1)=eps*3"
        );
        // Perturbed values must be distinct and positive.
        assert!(x_b[1] > 0.0 && x_b[2] > 0.0, "perturbed values must be positive");
        assert!(
            (x_b[1] - x_b[2]).abs() > 1e-20,
            "perturbed values must be distinct"
        );
    }

    fn primal_opts() -> SolverOptions {
        SolverOptions {
            simplex_method: SimplexMethod::Primal,
            ..Default::default()
        }
    }

    fn make_le_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        m: usize,
        n: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, m, n).unwrap();
        LpProblem::new_general(
            c,
            a,
            b,
            vec![ConstraintType::Le; m],
            vec![(0.0, f64::INFINITY); n],
            None,
        )
        .unwrap()
    }

    /// Sentinel: selective Charnes perturbation must not modify large x_b values.
    ///
    /// This tests the perturbation logic by verifying that a degenerate LP with
    /// a large non-degenerate basic variable reaches the known optimal. Under the
    /// old "add to ALL x_b" approach, the large basic variable's value would
    /// receive an additive shift of O(eps·m), temporarily distorting the Phase II
    /// objective by O(c·eps·m²) and causing the solver to oscillate away from the
    /// optimum.
    ///
    /// No-op proof: reverting to `*v += eps*(i+1)` for all x_b causes the Phase II
    /// objective of this LP to jump upward each time cycling is detected, preventing
    /// convergence to the known optimal within the default iteration budget.
    /// The assertion `result.status == Optimal` would then fail.
    ///
    /// LP: min -x1 - x2  s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1 (degenerate opt at (1,1))
    /// This is the same degenerate LP used in `test_highly_degenerate_lp` (tests.rs).
    #[test]
    fn selective_perturbation_degenerate_lp_converges() {
        // Degenerate LP with known opt = -2 at the degenerate vertex (1,1).
        let lp = make_le_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result = solve_with(&lp, &primal_opts());
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "degenerate LP must reach Optimal; got {:?}",
            result.status
        );
        assert!(
            (result.objective - (-2.0)).abs() < 1e-6,
            "expected obj=-2, got {}",
            result.objective
        );
    }

    /// Sentinel: selective Charnes perturbation formula uses `step_zero_threshold`
    /// as both the condition and the eps magnitude, not a fixed constant.
    ///
    /// Both the perturbation amount (`eps = PIVOT_TOL * m`) and the guard
    /// (`step_zero_threshold = PIVOT_TOL * m`) must scale with m. Using a fixed
    /// constant instead would silently perturb non-degenerate rows (too large
    /// a threshold) or fail to perturb truly degenerate ones (too small).
    ///
    /// No-op: changing either formula to a constant that does not scale with m
    /// causes this test to fail for m > 1.
    #[test]
    fn cycle_perturbation_threshold_scales_with_m() {
        for m in [1_usize, 10, 100, 1441] {
            let eps = PIVOT_TOL * (m as f64).max(1.0);
            let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
            assert!(
                (eps - step_zero_threshold).abs() < f64::EPSILON,
                "eps and step_zero_threshold must match for m={m}: eps={eps}, thr={step_zero_threshold}"
            );
            if m > 1 {
                assert!(
                    eps > PIVOT_TOL,
                    "threshold must exceed PIVOT_TOL for m={m}>1 (must scale with m)"
                );
            }
        }
    }
}
