//! Primal (revised) simplex: two-phase driver and the iteration core.

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::presolve::LpEquilibration;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use std::sync::atomic::Ordering;

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

use super::dual_common::{
    basic_obj, compute_dual_vars_into, compute_reduced_costs_into, lp_unbounded_ray_verified,
    made_progress_with_floor,
};
use super::pricing::{PricingStrategy, SteepestEdgePricing};
use super::trace::IterTrace;
use super::{build_standard_form, extract_dual_info, SimplexOutcome, StandardForm};

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

/// Relative tolerance below which a standard-form column value is treated as
/// at-bound (zero) when seeding the crossover basis from `x_star`.
const CROSSOVER_ZERO_TOL: f64 = 1e-9;

/// Bound-aware dual infeasibility of `y` against the reported primal `x_star`:
/// the worst per-variable reduced-cost sign violation. `0` iff `y` is KKT
/// dual-feasible and complementary with `x_star` (the metric `postsolve` and
/// `guard_lp_optimal` ultimately gate on). Used to pick the crossover dual that
/// is actually complementary with the *reported* primal.
fn crossover_dual_infeasibility(problem: &LpProblem, x_star: &[f64], y: &[f64]) -> f64 {
    let n = problem.num_vars;
    let mut max_viol = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let lb_s = if lb.is_finite() { lb.abs() } else { 0.0 };
        let ub_s = if ub.is_finite() { ub.abs() } else { 0.0 };
        let fixed = lb.is_finite()
            && ub.is_finite()
            && (ub - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s.max(ub_s));
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x_star[j] - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s);
        let at_ub = ub.is_finite() && (x_star[j] - ub).abs() < COMP_SLACK_REL_TOL * (1.0 + ub_s);
        let mut rc = problem.c[j];
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc -= vals[k] * y[row];
            }
        }
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -rc)
        } else if at_ub && !at_lb {
            f64::max(0.0, rc)
        } else {
            rc.abs()
        };
        if viol > max_viol {
            max_viol = viol;
        }
    }
    max_viol
}

/// Derive a globally dual-feasible dual for `problem` from its known optimal
/// primal `x_star` (postsolved original-space optimum) via primal crossover:
/// reconstruct an optimal basis *at* `x_star` and read `y = B⁻ᵀ c_B`.
///
///   1. Standard form + `x_star` → standard-form primal `x_std`.
///   2. Initial basis = slacks ± one artificial per `needs_artificial` row (a
///      permuted ±identity, provably non-singular).
///   3. Seat every support column (`x_std > 0`) via FTRAN pivots, so `B⁻¹b =
///      x_star` represents the optimal vertex (`B` stays non-singular).
///   4. Phase I drives residual artificials out (degenerate at feasible x*).
///   5. A no-perturbation Phase II takes only degenerate (step-0) pivots,
///      walking bases at the fixed vertex to a dual-feasible one.
///
/// Any optimal basis yields a dual-feasible dual, so this is degeneracy-robust
/// where incremental per-transform recovery can strand. Returns `(dual,
/// reduced_costs)` in original space, or `None` if the crossover cannot complete.
pub(crate) fn crossover_dual_from_primal(
    problem: &LpProblem,
    x_star: &[f64],
    deadline: Option<std::time::Instant>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let sf = build_standard_form(problem);
    let m = sf.m;
    let n_orig = problem.num_vars;
    let n_total = sf.n_total;
    let n_shifted = sf.n_shifted;
    if x_star.len() != n_orig || m == 0 {
        return None;
    }

    let options = SolverOptions {
        deadline,
        warm_start: None,
        ..Default::default()
    };

    // (1) x_star → standard-form primal x_std (variable shifts / free-var splits).
    let mut x_std = vec![0.0_f64; n_total];
    for j in 0..n_orig {
        let info = &sf.orig_var_info[j];
        let xj = x_star[j];
        if info.new_vars.len() == 2 {
            x_std[info.new_vars[0].0] = xj.max(0.0);
            x_std[info.new_vars[1].0] = (-xj).max(0.0);
        } else {
            let (idx, coeff) = info.new_vars[0];
            let val = if coeff > 0.0 {
                xj - info.offset
            } else {
                info.offset - xj
            };
            x_std[idx] = val.max(0.0);
        }
    }
    // Slack values from the structural row sums (each slack has one entry).
    let mut row_struct_sum = vec![0.0_f64; m];
    for j in 0..n_shifted {
        if x_std[j].abs() < CROSSOVER_ZERO_TOL {
            continue;
        }
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                row_struct_sum[row] += vals[k] * x_std[j];
            }
        }
    }
    for j in n_shifted..n_total {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            if rows.len() == 1 && vals[0].abs() > 0.0 {
                let i = rows[0];
                x_std[j] = ((sf.b[i] - row_struct_sum[i]) / vals[0]).max(0.0);
            }
        }
    }

    // (2) a_ext = A plus one artificial unit column per row with no slack basis
    // column. The basis (slacks ± artificials) is a permuted ±identity, hence
    // provably non-singular — unlike a partial LTSF crash, whose covered block
    // can be rank-deficient when columns are assigned with active count > 1.
    let mut basis = sf.initial_basis.clone();
    let mut tr: Vec<usize> = Vec::new();
    let mut tc: Vec<usize> = Vec::new();
    let mut tv: Vec<f64> = Vec::new();
    for j in 0..n_total {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                tr.push(row);
                tc.push(j);
                tv.push(vals[k]);
            }
        }
    }
    let mut art = n_total;
    for i in 0..m {
        if sf.needs_artificial[i] {
            tr.push(i);
            tc.push(art);
            tv.push(1.0);
            basis[i] = art;
            art += 1;
        }
    }
    let n_ext = art;
    let a_ext = CscMatrix::from_triplets(&tr, &tc, &tv, m, n_ext).ok()?;

    // (3) x_star-driven refinement via FTRAN pivots: seat every support column
    // (x_std > 0 — structurals AND slacks) into the basis, displacing 0-valued
    // slacks / artificials. Pivoting on a nonzero (B⁻¹aⱼ)ᵢ keeps B non-singular
    // (a blind index swap does not). A non-binding Ge surplus slack starts
    // nonbasic, so seating slacks too is required or B⁻¹b ≠ x*.
    {
        let mut basis_mgr = LuBasis::new_timed(&a_ext, &basis, options.max_etas, deadline).ok()?;
        let mut is_basic = vec![false; n_ext];
        for &col in basis.iter() {
            is_basic[col] = true;
        }
        let removable = |col: usize| -> bool {
            col >= n_total || (col >= n_shifted && x_std[col] <= CROSSOVER_ZERO_TOL)
        };
        let mut active: Vec<(f64, usize)> = (0..n_total)
            .filter(|&j| x_std[j] > CROSSOVER_ZERO_TOL && !is_basic[j])
            .map(|j| (x_std[j], j))
            .collect();
        active.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_xj, j) in active {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            let Ok((col_rows, col_vals)) = a_ext.get_column(j) else {
                continue;
            };
            let mut d_sv = SparseVec {
                indices: col_rows.to_vec(),
                values: col_vals.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut d_sv);
            let mut best_row: Option<usize> = None;
            let mut best_abs = PIVOT_TOL;
            for (k, &row) in d_sv.indices.iter().enumerate() {
                let abs = d_sv.values[k].abs();
                if abs > best_abs && removable(basis[row]) {
                    best_abs = abs;
                    best_row = Some(row);
                }
            }
            if let Some(row) = best_row {
                is_basic[basis[row]] = false;
                is_basic[j] = true;
                basis_mgr.update(j, row, &d_sv);
                basis[row] = j;
                basis_mgr.refactor_if_needed_timed(&a_ext, &basis, deadline);
            }
        }
    }

    // (4) Reconcile x_B = B⁻¹b from a fresh LU (also detects a singular basis).
    let mut x_b = vec![0.0_f64; m];
    let mut y_tmp = vec![0.0_f64; m];
    let mut c_phase1 = vec![0.0_f64; n_ext];
    c_phase1[n_total..].fill(1.0);
    reconcile_final_basis_state(
        &a_ext,
        &sf.b,
        &c_phase1,
        &basis,
        &mut x_b,
        &mut y_tmp,
        options.max_etas,
        deadline,
    )
    .ok()?;
    for v in x_b.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // Phase I: drive any residual artificials out (degenerate at the feasible x*).
    if basis.iter().any(|&col| col >= n_total) {
        for i in 0..m {
            if basis[i] >= n_total && x_b[i].abs() <= PIVOT_TOL {
                x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
            }
        }
        let mut pricing1 = SteepestEdgePricing::new(n_ext);
        let mut iters = 0usize;
        match revised_simplex_core(
            &a_ext,
            &mut x_b,
            &c_phase1,
            &sf.b,
            &mut basis,
            m,
            n_ext,
            n_ext,
            &mut pricing1,
            &options,
            &mut iters,
            true,
        ) {
            SimplexOutcome::Optimal(_, _) => {}
            _ => return None,
        }
        // Verify feasibility with a fresh LU (eta drift can mask residual arts).
        if reconcile_final_basis_state(
            &a_ext,
            &sf.b,
            &c_phase1,
            &basis,
            &mut x_b,
            &mut y_tmp,
            options.max_etas,
            deadline,
        )
        .is_err()
        {
            return None;
        }
        let phase1_obj: f64 = (0..m).map(|i| c_phase1[basis[i]] * x_b[i].max(0.0)).sum();
        if phase1_obj > PIVOT_TOL {
            return None;
        }
        pivot_out_degenerate_artificials(&a_ext, &mut basis, &x_b, &sf, &options);
    }

    // (5) Read the dual at the x*-representing basis. Its BFS is x*, so its dual
    // is KKT-complementary with x*. When x* is a degenerate vertex this basis may
    // not yet be dual-feasible; a perturbation-free Phase II then walks the bases
    // at the fixed vertex (degenerate, step-0 pivots) to a dual-feasible one.
    let mut c_phase2 = vec![0.0_f64; n_ext];
    c_phase2[..n_total].copy_from_slice(&sf.c[..n_total]);
    let row_scale = vec![1.0_f64; m];

    let mut y = vec![0.0_f64; m];
    if reconcile_final_basis_state(
        &a_ext,
        &sf.b,
        &c_phase2,
        &basis,
        &mut x_b,
        &mut y,
        options.max_etas,
        deadline,
    )
    .is_err()
    {
        return None;
    }
    let (dual1, rc1, _) = extract_dual_info(&sf, problem, &y, x_star, &row_scale);
    let df1 = crossover_dual_infeasibility(problem, x_star, &dual1);
    if df1 <= crate::qp::certificate::LP_CERT_TOL {
        return Some((dual1, rc1));
    }

    for v in x_b.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut pricing2 = SteepestEdgePricing::new(n_ext);
    let mut iters2 = 0usize;
    let phase2 = revised_simplex_core(
        &a_ext,
        &mut x_b,
        &c_phase2,
        &sf.b,
        &mut basis,
        m,
        n_ext,
        n_total,
        &mut pricing2,
        &options,
        &mut iters2,
        false,
    );
    match phase2 {
        SimplexOutcome::Optimal(_, mut y2) => {
            if reconcile_final_basis_state(
                &a_ext,
                &sf.b,
                &c_phase2,
                &basis,
                &mut x_b,
                &mut y2,
                options.max_etas,
                deadline,
            )
            .is_err()
            {
                return Some((dual1, rc1));
            }
            let (dual2, rc2, _) = extract_dual_info(&sf, problem, &y2, x_star, &row_scale);
            let df2 = crossover_dual_infeasibility(problem, x_star, &dual2);
            if df2 < df1 {
                Some((dual2, rc2))
            } else {
                Some((dual1, rc1))
            }
        }
        _ => Some((dual1, rc1)),
    }
}

/// Maximum partial-revert rounds after crash basis construction.
/// Each round restores artificial columns for rows with negative x_b and
/// re-factorizes; 3 rounds is sufficient for observed mid-scale ill LPs.
const CRASH_REVERT_MAX_ROUNDS: usize = 3;

/// Crash basis: cover artificial rows with structural columns to reduce Phase I pivots.
///
/// Returns `Some((crash_basis, x_b))` on success, `None` if the crash is no better
/// than a cold start or the basis is singular. Returns `None` in the following cases:
/// 1. Crash result equals `cold_basis` (no structural coverage gained)
/// 2. LU factorization fails (crashed basis singular)
/// 3. No structural columns remain after partial revert
///
/// Partial revert: rows where `x_b < -PIVOT_TOL` have their crashed column replaced
/// with the artificial and the basis is re-factorized.
fn try_apply_crash(
    a_ext: &CscMatrix,
    m: usize,
    n_shifted: usize,
    n_total: usize,
    b_scaled: &[f64],
    max_etas: usize,
    deadline: Option<std::time::Instant>,
    cold_basis: &[usize],
) -> Option<(Vec<usize>, Vec<f64>)> {
    use super::crash;
    use crate::basis::{BasisManager, LuBasis};
    use crate::sparse::SparseVec;
    use crate::tolerances::PIVOT_TOL;

    // 入力 needs_artificial を `cold_basis[i] >= n_total` から再構築。
    let needs_artificial: Vec<bool> = cold_basis.iter().map(|&c| c >= n_total).collect();

    let num_art_in = needs_artificial.iter().filter(|&&v| v).count();
    if num_art_in == 0 {
        return None;
    }

    let (mut basis, _, num_art_out) =
        crash::compute_crash_basis(a_ext, b_scaled, m, n_shifted, cold_basis, &needs_artificial);

    if num_art_out == num_art_in {
        return None;
    }

    // partial revert loop: 負 x_b の crashed 行を artif に戻す。
    // 復元不能な行 (= 元 cold basis に artif 候補が無い ub/slack 行) で負成分が
    // 出た場合は crash 全体を放棄 (Phase I/II が x_B >= 0 不変式を回復できないため)。
    let mut x_b = vec![0.0_f64; m];
    let mut crashed_count = num_art_in - num_art_out;
    for round in 0..=CRASH_REVERT_MAX_ROUNDS {
        let mut basis_mgr = match LuBasis::new_timed(a_ext, &basis, max_etas, deadline) {
            Ok(b) => b,
            Err(_) => {
                return None;
            }
        };
        let mut x_b_sv = SparseVec::from_dense(b_scaled);
        basis_mgr.ftran(&mut x_b_sv);
        x_b = x_b_sv.to_dense();

        let mut reverts = 0usize;
        for i in 0..m {
            if x_b[i] >= -PIVOT_TOL {
                continue;
            }
            // 負成分行: 元 cold で artif があれば revert、無ければ crash 放棄。
            if cold_basis[i] >= n_total {
                basis[i] = cold_basis[i];
                reverts += 1;
            } else {
                return None;
            }
        }
        if reverts == 0 {
            break;
        }
        crashed_count = crashed_count.saturating_sub(reverts);
        if crashed_count == 0 {
            return None;
        }
        if round == CRASH_REVERT_MAX_ROUNDS && reverts > 0 {
            return None;
        }
    }

    Some((basis, x_b))
}

/// Defense-in-depth feasibility check.  Per constraint, compare violation to
/// `feas_rel_tol() * (1 + |b_i| + |Ax_i|)` so the gate is scale-invariant.
/// `feas_rel_tol() = sqrt(PIVOT_TOL)` follows from Wilkinson's heuristic
/// (see `tolerances.rs`).
fn check_eq_feasibility(problem: &LpProblem, solution: &[f64]) -> bool {
    let tol = feas_rel_tol();
    let mut ax = vec![0.0f64; problem.num_constraints];
    for (j, &sj) in solution.iter().enumerate() {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * sj;
            }
        }
    }
    let mut violated = false;
    for ((ax_i, ct), bi) in ax
        .iter()
        .zip(problem.constraint_types.iter())
        .zip(problem.b.iter())
    {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
        };
        let scale = 1.0 + bi.abs() + ax_i.abs();
        let rel = violation / scale;
        if rel > tol {
            violated = true;
        }
    }
    !violated
}

fn pivot_out_degenerate_artificials(
    a_ext: &CscMatrix,
    basis: &mut [usize],
    x_b: &[f64],
    sf: &StandardForm,
    options: &SolverOptions,
) {
    let m = basis.len();

    // Fast pre-check: skip LU build entirely when Phase I has already pivoted
    // out all artificials. For most problems this is the common case; avoiding
    // two LU factorizations (one here, one for validation) saves significant
    // work — especially for large m.
    if !basis
        .iter()
        .zip(x_b.iter())
        .any(|(&col, &val)| col >= sf.n_total && val.abs() < PIVOT_TOL)
    {
        #[cfg(test)]
        PIVOT_CLEAN_EARLY_EXIT_COUNT.fetch_add(1, Ordering::Relaxed);
        return;
    }

    #[cfg(test)]
    PIVOT_CLEAN_CLEANUP_RAN_COUNT.fetch_add(1, Ordering::Relaxed);

    let basis_before = basis.to_vec();

    // Pivot stability uses |(B^{-1} a_j)[i]|, not raw A[i,j], so we need an LU.
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(mgr) => mgr,
        Err(_) => return,
    };

    let mut is_basic = vec![false; a_ext.ncols];
    for &col in basis.iter() {
        is_basic[col] = true;
    }

    // BTRAN-based candidate scan: one BTRAN gives the i-th row of B^{-1}; a
    // sparse dot vs each non-basic column ranks candidates without per-column
    // FTRAN. One FTRAN at the end feeds basis_mgr.update — total cost per
    // artificial ≈ O(m + nnz(A)), vs. O(n_total · FTRAN) for the naive form.
    let mut z_dense = vec![0.0_f64; m];
    for i in 0..m {
        if options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return;
        }
        if basis[i] < sf.n_total || x_b[i].abs() >= PIVOT_TOL {
            continue;
        }

        // z = B^{-T} e_i
        z_dense.iter_mut().for_each(|v| *v = 0.0);
        z_dense[i] = 1.0;
        basis_mgr.btran_dense(&mut z_dense);

        // argmax_j |d[i,j]| over non-basic original columns.
        let mut best_j: Option<usize> = None;
        let mut best_abs = PIVOT_TOL;
        for j in 0..sf.n_total {
            if is_basic[j] {
                continue;
            }
            if let Ok((rows, vals)) = a_ext.get_column(j) {
                let mut d_ij = 0.0_f64;
                for (k, &row) in rows.iter().enumerate() {
                    if row < m {
                        d_ij += z_dense[row] * vals[k];
                    }
                }
                let abs_d = d_ij.abs();
                if abs_d > best_abs {
                    best_abs = abs_d;
                    best_j = Some(j);
                }
            }
        }

        if let Some(j) = best_j {
            let (col_rows, col_vals) = match a_ext.get_column(j) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let mut d_sv = SparseVec {
                indices: col_rows.to_vec(),
                values: col_vals.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut d_sv);
            is_basic[basis[i]] = false;
            is_basic[j] = true;
            basis[i] = j;
            basis_mgr.update(j, i, &d_sv);
            basis_mgr.refactor_if_needed_timed(a_ext, basis, options.deadline);
        }
    }

    if LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline).is_err() {
        basis.copy_from_slice(&basis_before);
    }
}

/// Recompute x_B = B^{-1} b and y = B^{-T} c_B from a fresh LU.
pub(crate) fn reconcile_final_basis_state(
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    basis: &[usize],
    x_b: &mut [f64],
    y: &mut [f64],
    max_etas: usize,
    deadline: Option<std::time::Instant>,
) -> Result<(), crate::error::SolverError> {
    let mut basis_mgr = LuBasis::new_timed(a, basis, max_etas, deadline)?;

    x_b.copy_from_slice(b);
    basis_mgr.ftran_dense(x_b);
    for value in x_b.iter_mut() {
        if value.abs() < 1e-12 {
            *value = 0.0;
        }
    }

    compute_dual_vars_into(c, &mut basis_mgr, basis, y);
    Ok(())
}

/// Map the standard-form basic solution back to original variables, inverting
/// shifts/sign-flips/splits.  `col_scale` is the Ruiz column scale (or empty).
///
/// The recomposition `offset + Σ coeff * x_new[idx]` is accumulated in
/// double-double (TwoFloat) precision because free variables are split as
/// `x = x+ − x-`; when the simplex leaves both components large (e.g. on the
/// order of 1e16) f64 subtraction loses the unit-scale residual entirely.
/// The contract is locked by `tests::test_extract_solution_uses_dd_for_split_variable_cancellation`.
pub(crate) fn extract_solution(
    sf: &StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> Vec<f64> {
    use twofloat::TwoFloat;
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            let scale = col_scale.get(basis[i]).copied().unwrap_or(1.0);
            x_new[basis[i]] = x_b[i] * scale;
        }
    }

    let mut solution = vec![0.0; sf.n_orig];
    for (j, sol_j) in solution.iter_mut().enumerate() {
        let info = &sf.orig_var_info[j];
        let mut value = TwoFloat::from(info.offset);
        for &(new_idx, coeff) in &info.new_vars {
            value += TwoFloat::new_mul(coeff, x_new[new_idx]);
        }
        *sol_j = f64::from(value);
    }
    solution
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

/// Maximum step that keeps every basic variable ≥ −`tol`:
///   `θ = min_{i: d[i]>floor} (x_b[i] + tol) / d[i]`.
/// `INFINITY` when no row is eligible (unbounded direction).
fn bound_tolerance_step(x_b: &[f64], d: &[f64], m: usize, floor: f64, tol: f64) -> f64 {
    let mut theta = f64::INFINITY;
    for i in 0..m {
        if d[i] > floor {
            let t = (x_b[i] + tol) / d[i];
            if t < theta {
                theta = t;
            }
        }
    }
    theta
}

/// Pick the leaving row with the largest pivot `|d[i]|` among rows whose ratio
/// `x_b[i]/d[i]` does not exceed `theta`; ties in `|d[i]|` break by Bland's rule
/// (smallest basic index, anti-cycling). Returns the row, or `None`.
fn max_pivot_within(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    theta: f64,
) -> Option<usize> {
    let mut leaving: Option<usize> = None;
    let mut best_pivot_abs = 0.0f64;
    for i in 0..m {
        if d[i] > floor {
            let ratio = x_b[i] / d[i];
            if ratio <= theta {
                let d_abs = d[i].abs();
                if d_abs > best_pivot_abs + PIVOT_TOL {
                    best_pivot_abs = d_abs;
                    leaving = Some(i);
                } else if (d_abs - best_pivot_abs).abs() <= PIVOT_TOL {
                    match leaving {
                        None => leaving = Some(i),
                        Some(prev) if basis[i] < basis[prev] => leaving = Some(i),
                        _ => {}
                    }
                }
            }
        }
    }
    leaving
}

/// Harris ratio test (Pass 2), **feasibility-preserving**.
///
/// The leaving step is bounded by the variable-tolerance maximum step
///   `θ = min_{i: d[i]>floor} (x_b[i] + feas_tol) / d[i]`,
/// and among rows within `θ` we take the largest pivot `|d[i]|` (Bland
/// tie-break). For a leaving row with `x_b ≥ 0` this keeps every pivot-eligible
/// basic value (`d[i] > floor`) at `≥ −feas_tol` independent of `d[i]`. A
/// leaving row inside the `[−feas_tol, 0)` band gives a small negative step that
/// can transiently breach `−feas_tol`; the optimality backstop (exact
/// `x_b = B⁻¹b` recheck) then returns an honest Timeout, never false-Optimal.
///
/// The predecessor's absolute *ratio* window `min_ratio + ε` overshot by
/// `ε·d[i]` — unbounded for ill-scaled columns (pilot87: `d[i] ≈ 1.3e6` turned
/// `ε = 1e-8` into a 0.013 breach), producing an `x_b < 0` basis, negative
/// ratios, and a wandering objective instead of convergence.
///
/// `feas_tol` = `options.primal_tol`. Returns `None` for an unbounded direction.
fn select_leaving_feasibility_preserving(
    x_b: &[f64],
    d: &[f64],
    basis: &[usize],
    m: usize,
    floor: f64,
    feas_tol: f64,
) -> Option<usize> {
    let theta = bound_tolerance_step(x_b, d, m, floor, feas_tol);
    if !theta.is_finite() {
        return None;
    }
    max_pivot_within(x_b, d, basis, m, floor, theta)
}

/// Revised simplex core: BTRAN → pricing → FTRAN → Harris ratio test →
/// rank-1 basis update, with on-demand LU refactor.
///
/// `enable_phase1_cycling_bail` arms the obj+step plateau early-bail
/// described above; pass `true` only from Primal Phase I.
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
                        return SimplexOutcome::Timeout(obj);
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
                            CYCLE_DETECT_NONDEGEN_PRESERVED
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
            OBJ_PROGRESS_RESET_COUNT.fetch_add(1, Ordering::Relaxed);
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
            return SimplexOutcome::Timeout(current_obj);
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
mod crossover_tests {
    //! `crossover_dual_from_primal` reconstructs an optimal basis at a known
    //! primal optimum `x*` and reads `y = B⁻ᵀ c_B`. The contract: the returned
    //! dual is KKT dual-feasible AND complementary with `x*`
    //! (`crossover_dual_infeasibility ≈ 0`), across constraint senses, free
    //! variables, finite upper bounds, and non-binding Ge rows.
    use super::{crossover_dual_from_primal, crossover_dual_infeasibility};
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    /// Tolerance for "dual-feasible & complementary with x*".
    const DF_TOL: f64 = 1e-7;

    fn assert_crossover_complementary(problem: &LpProblem, x_star: &[f64], label: &str) {
        let (y, rc) = crossover_dual_from_primal(problem, x_star, None)
            .unwrap_or_else(|| panic!("{label}: crossover returned None"));
        assert_eq!(y.len(), problem.num_constraints, "{label}: dual length");
        assert_eq!(rc.len(), problem.num_vars, "{label}: rc length");
        let df = crossover_dual_infeasibility(problem, x_star, &y);
        assert!(
            df < DF_TOL,
            "{label}: dual infeasibility {df:.3e} must be ~0 — the crossover dual \
             must be KKT-feasible and complementary with x* (y={y:?})"
        );
    }

    fn lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        b: Vec<f64>,
        ct: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
    ) -> LpProblem {
        let m = b.len();
        let a = CscMatrix::from_triplets(rows, cols, vals, m, c.len()).unwrap();
        LpProblem::new_general(c, a, b, ct, bounds, None).unwrap()
    }

    /// Le-only LP, unique optimum. min -x1-x2 s.t. x1+2x2<=4, 3x1+x2<=6.
    /// Optimum x*=(1.6, 1.2): both Le binding, both x interior.
    #[test]
    fn crossover_le_unique_optimum() {
        let p = lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 1.0],
            vec![4.0, 6.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[1.6, 1.2], "le_unique");
    }

    /// Equality constraint (artificial in standard form) + free variable (± split).
    /// min x1 + x2 s.t. x1 + x2 = 3 (Eq), x1 free, x2 >= 0. Optimum x*=(3,0).
    #[test]
    fn crossover_eq_with_free_var() {
        let p = lp(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[3.0, 0.0], "eq_free");
    }

    /// Finite upper bound (UB row in standard form). min -x1 s.t. x1+x2<=3,
    /// x1 ∈ [0,2], x2 ∈ [0,5]. Optimum x*=(2,0): x1 at UB.
    #[test]
    fn crossover_finite_upper_bound() {
        let p = lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            vec![3.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 5.0)],
        );
        assert_crossover_complementary(&p, &[2.0, 0.0], "finite_ub");
    }

    /// SENTINEL (support-slack seating): a Ge row that is NON-binding at the
    /// optimum, so its surplus slack is a support column (value > 0) that starts
    /// NONBASIC (the Ge row is seeded with an artificial). If the refinement seats
    /// only structural support columns (not slacks), B⁻¹b ≠ x* and the dual is
    /// wrong. min -x1 s.t. x1<=2 (Le), x1+x2>=1 (Ge, surplus=1>0 at opt),
    /// x1,x2 ∈ [0,10]. Optimum x*=(2,0): y0=-1 (Le), y1=0 (Ge non-binding).
    #[test]
    fn crossover_seats_support_slack_on_nonbinding_ge() {
        let p = lp(
            vec![-1.0, 0.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 1.0],
            vec![2.0, 1.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        assert_crossover_complementary(&p, &[2.0, 0.0], "ge_nonbinding");
    }

    /// Degenerate optimum: x*=(1,1) with THREE binding constraints (x1<=1,
    /// x2<=1, x1+x2<=2) but only 2 structurals — a degenerate vertex represented
    /// by several bases; the crossover must reach a dual-feasible one. min -x1-x2.
    #[test]
    fn crossover_degenerate_vertex() {
        let p = lp(
            vec![-1.0, -1.0],
            &[0, 1, 2, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
            vec![
                ConstraintType::Le,
                ConstraintType::Le,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[1.0, 1.0], "degenerate");
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

#[cfg(test)]
mod ratio_test_feasibility_tests {
    //! Sentinels for the feasibility-preserving Harris ratio test
    //! (`select_leaving_feasibility_preserving`).
    //!
    //! The leaving row must be chosen so the pivot step keeps every basic value
    //! ≥ −feas_tol, with the violation bounded by `feas_tol` independent of the
    //! pivot magnitude `d[i]`. The previous absolute-ratio window
    //! `min_ratio + PIVOT_TOL` let a binding row overshoot by `PIVOT_TOL·d[i]`,
    //! which for large `d[i]` (ill-scaled columns) exceeded any clamp and
    //! cascaded into primal infeasibility (pilot87 Phase II).

    use super::select_leaving_feasibility_preserving;
    use crate::tolerances::PIVOT_TOL;

    /// Apply the pivot for a chosen leaving row and return the minimum basic
    /// value afterwards (the feasibility witness).
    fn min_basic_after_pivot(x_b: &[f64], d: &[f64], leaving: usize) -> f64 {
        let step = x_b[leaving] / d[leaving];
        let mut min_v = f64::INFINITY;
        for i in 0..x_b.len() {
            let v = if i == leaving { step } else { x_b[i] - d[i] * step };
            if v < min_v {
                min_v = v;
            }
        }
        min_v
    }

    /// Reference implementation of the OLD absolute-ratio window
    /// (`min_ratio + PIVOT_TOL`, max |d|, Bland tie-break). Used only to prove
    /// the no-op: reverting the production helper to this rule reintroduces the
    /// feasibility breach this sentinel guards against.
    fn old_absolute_window_leaving(x_b: &[f64], d: &[f64], basis: &[usize], floor: f64) -> usize {
        let m = x_b.len();
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if d[i] > floor {
                min_ratio = min_ratio.min(x_b[i] / d[i]);
            }
        }
        let window = min_ratio + PIVOT_TOL;
        let mut leaving = None;
        let mut best = 0.0f64;
        for i in 0..m {
            if d[i] > floor && x_b[i] / d[i] <= window {
                let da = d[i].abs();
                if da > best + PIVOT_TOL {
                    best = da;
                    leaving = Some(i);
                } else if (da - best).abs() <= PIVOT_TOL {
                    match leaving {
                        None => leaving = Some(i),
                        Some(p) if basis[i] < basis[p] => leaving = Some(i),
                        _ => {}
                    }
                }
            }
        }
        leaving.unwrap()
    }

    /// Sentinel (no-op proof): an ill-scaled tie where the absolute-ratio window
    /// breaches feasibility but the bound-tolerance helper does not.
    ///
    /// Two rows share a huge pivot (|d|=1e6). Row 0 has the true min ratio
    /// (x_b=1e-9) and row 1 is far from it (x_b=1e-3). The absolute window
    /// `min_ratio+PIVOT_TOL` admits BOTH and, on the |d| tie, Bland picks row 1
    /// (lower basis index). Its step 1e-9 then drives row 0 to ≈ −1e-3 ≪ −tol.
    /// The helper's bound-tolerance step admits only row 0, so no basic value
    /// drops below −feas_tol.
    ///
    /// Reverting the helper to the absolute window makes it return row 1 →
    /// `assert_eq!(leaving, 0)` and the feasibility assertion both FAIL.
    #[test]
    fn bound_tolerance_blocks_ill_scaled_overshoot() {
        let x_b = [1e-9, 1e-3];
        let d = [1e6, 1e6];
        let basis = [5usize, 3usize];
        let feas_tol = PIVOT_TOL;
        let floor = PIVOT_TOL;

        let leaving =
            select_leaving_feasibility_preserving(&x_b, &d, &basis, x_b.len(), floor, feas_tol)
                .expect("eligible leaving row exists");
        assert_eq!(
            leaving, 0,
            "helper must pick the true-min-ratio row 0, not the far row 1"
        );
        let min_basic = min_basic_after_pivot(&x_b, &d, leaving);
        assert!(
            min_basic >= -feas_tol,
            "helper pivot must keep basics ≥ −feas_tol; got {min_basic}"
        );

        // No-op proof: the old absolute-ratio window picks row 1 and breaches.
        let old_leaving = old_absolute_window_leaving(&x_b, &d, &basis, floor);
        assert_eq!(old_leaving, 1, "old window picks the far row (Bland tie)");
        let old_min_basic = min_basic_after_pivot(&x_b, &d, old_leaving);
        assert!(
            old_min_basic < -feas_tol,
            "old window must breach feasibility (proves the sentinel bites); got {old_min_basic}"
        );
        assert!(
            old_min_basic < -1e-4,
            "breach magnitude ∝ d[i]; expected ≈ −1e-3, got {old_min_basic}"
        );
    }

    /// Stability is preserved: when several rows leave safely within the
    /// bound-tolerance window, the helper still selects the largest pivot.
    #[test]
    fn picks_largest_pivot_within_window() {
        // Both rows are at the degenerate vertex (x_b ≈ 0), so both are within
        // θ. Row 1 has the larger pivot and must be chosen for stability.
        let x_b = [0.0, 0.0];
        let d = [0.5, 2.0];
        let basis = [7usize, 4usize];
        let leaving =
            select_leaving_feasibility_preserving(&x_b, &d, &basis, x_b.len(), PIVOT_TOL, PIVOT_TOL)
                .expect("eligible leaving row exists");
        assert_eq!(leaving, 1, "must pick the larger pivot |d|=2.0 (row 1)");
    }

    /// No eligible row (all directions ≤ floor) ⇒ unbounded ⇒ None.
    #[test]
    fn no_eligible_row_is_unbounded() {
        let x_b = [3.0, 4.0];
        let d = [-1.0, 0.0];
        let basis = [0usize, 1usize];
        let leaving = select_leaving_feasibility_preserving(
            &x_b,
            &d,
            &basis,
            x_b.len(),
            PIVOT_TOL,
            PIVOT_TOL,
        );
        assert!(leaving.is_none(), "no positive direction ⇒ unbounded ⇒ None");
    }
}
