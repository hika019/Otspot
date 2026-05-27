//! Primal (revised) simplex: two-phase driver and the iteration core.

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::presolve::RuizScaler;
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

use super::dual_common::{basic_obj, compute_dual_vars_into, compute_reduced_costs_into};
use super::pricing::{PricingStrategy, SteepestEdgePricing};
use super::{StandardForm, SimplexOutcome, extract_dual_info};

/// Minimum absolute diagonal entry to trust when dividing `x_B[i]` by the
/// Ruiz-scaled slack/artificial column's diagonal. Prevents division by
/// near-zero when equilibration shrinks the diagonal below f64 noise.
const SLACK_DIAG_TOL: f64 = 1e-14;

/// Attempt to extract a verified Farkas infeasibility certificate from a
/// Phase I basis that could not be driven to feasibility.
///
/// Constructs `y = B^{-T} e_art` (the Phase I dual, where `e_art[i] = 1` for
/// artificial basis columns) and checks the Farkas alternative for the
/// standard-form LP `{Ax = b, x ≥ 0}`:
///
///   A^T y ≤ tol  (for all non-artificial columns j < n_original)
///   b^T y > tol
///
/// Returns `y` if both conditions hold, or an empty Vec if the certificate
/// cannot be verified (LU failure, no artificials in basis, or numeric check
/// failed). An empty return does NOT mean the LP is feasible — it means this
/// basis cannot provide a Farkas proof; the caller should re-verify via Big-M
/// rather than blindly trusting the unverified Infeasible verdict.
///
/// Tolerance: `dual_tol * max(1, ‖b‖∞)` — consistent with the Big-M Phase I
/// Farkas checker so both paths discriminate at the same numeric threshold.
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
    let mut basis_mgr = match LuBasis::new(a_ext, basis, options.max_etas) {
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
        let Ok((rows, vals)) = a_ext.get_column(j) else { return vec![]; };
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
    if reconcile_final_basis_state(a, b, c, basis, &mut x_b_reconciled, &mut y, max_etas, deadline).is_ok() {
        extract_solution(sf, basis, &x_b_reconciled, col_scale)
    } else {
        extract_solution(sf, basis, x_b, col_scale)
    }
}

/// Two-phase primal simplex on a standard-form LP. Skips Phase I when no
/// artificials are needed. Phase I minimizes the sum of artificials; a
/// positive minimum proves Infeasible. Ruiz equilibration is applied first.
pub(crate) fn two_phase_simplex(sf: &StandardForm, problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = sf.m;
    let mut total_iters: usize = 0;

    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&sf.a, &sf.b, &sf.c);

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

        match revised_simplex_core(&a, &mut x_b, &c, &b, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options, &mut total_iters, false)
        {
            SimplexOutcome::Optimal(obj, mut y) => {
                match reconcile_final_basis_state(&a, &b, &c, &basis, &mut x_b, &mut y, options.max_etas, options.deadline) {
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
                        return SolverResult { status: SolveStatus::Timeout, objective: obj + sf.obj_offset, solution, iterations: total_iters, ..Default::default() };
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
                let ws = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
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
            SimplexOutcome::SingularBasis => {
                SolverResult::numerical_error()
            }
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
            if !sf.needs_artificial[i] { continue; }
            trip_rows.push(i);
            trip_cols.push(art_col);
            trip_vals.push(1.0);
            basis[i] = art_col;
            art_col += 1;
        }
        let n_ext = art_col;

        let a_ext =
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext).unwrap();

        // Phase I cost: penalize all artificials.
        let mut c_phase1 = vec![0.0; n_ext];
        c_phase1[sf.n_total..].fill(1.0);

        // Crash basis: cover artificial rows with structural columns to reduce
        // Phase I pivots. Rows with negative x_b after FTRAN are rolled back.
        let crashed = if options.warm_start.is_none()
            && options.use_lp_crash_basis
            && sf.num_artificial > 0
        {
            try_apply_crash(&a_ext, m, sf.n_shifted, sf.n_total, &b, options.max_etas, &basis)
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
                    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        break 'retry;
                    }
                    let mut y_dummy = vec![0.0f64; m];
                    let rec_obj = match reconcile_final_basis_state(
                        &a_ext, &b, &c_phase1, &basis, &mut x_b, &mut y_dummy,
                        options.max_etas, options.deadline,
                    ) {
                        Ok(()) => {
                            (0..m).map(|i| c_phase1[basis[i]] * x_b[i].max(0.0)).sum::<f64>()
                        }
                        Err(_) => break 'retry,
                    };
                    if rec_obj <= PIVOT_TOL { phase1_feasible = true; break 'retry; }
                    if attempt == MAX_PHASE1_RETRIES { break 'retry; }

                    // Artificials remain positive: clamp drift and retry Phase I.
                    for v in x_b.iter_mut() {
                        if *v < 0.0 { *v = 0.0; }
                    }
                    let mut pricing_retry = SteepestEdgePricing::new(n_ext);
                    match revised_simplex_core(
                        &a_ext, &mut x_b, &c_phase1, &b, &mut basis,
                        m, n_ext, n_ext, &mut pricing_retry, options,
                        &mut total_iters, true,
                    ) {
                        SimplexOutcome::Optimal(_, _) => {}
                        SimplexOutcome::Unbounded => break 'retry,
                        SimplexOutcome::Timeout(obj1) => {
                            return SolverResult {
                                status: SolveStatus::Timeout,
                                objective: obj1 + sf.obj_offset,
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
                    // Extract and verify a Farkas certificate from the final Phase I basis.
                    // True infeasible LPs have a valid dual ray at this basis (LP duality);
                    // feasible LPs cycling in Phase I do not. The certificate is stored in
                    // dual_solution so `solve_dual_advanced` can gate Big-M re-verification:
                    // certified → trust; uncertified → pilot87-class false-Infeasible → Big-M.
                    let farkas = extract_farkas_certificate(
                        &a_ext, &b, &basis, m, sf.n_total, options,
                    );
                    return SolverResult {
                        status: SolveStatus::Infeasible,
                        objective: 0.0,
                        solution: vec![],
                        dual_solution: farkas,
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        iterations: total_iters,
                        ..Default::default()
                    };
                }

                // Phase I feasible: pivot out any remaining degenerate artificials
                pivot_out_degenerate_artificials(&a_ext, &mut basis, &x_b, sf, options);

                let mut c_phase2 = vec![0.0; n_ext];
                c_phase2[..sf.n_total].copy_from_slice(&c[..sf.n_total]);
                {
                    let mut y_transition = vec![0.0f64; m];
                    match reconcile_final_basis_state(
                        &a_ext, &b, &c_phase2, &basis, &mut x_b, &mut y_transition,
                        options.max_etas, options.deadline,
                    ) {
                        Ok(()) => {}
                        Err(crate::error::SolverError::DeadlineExceeded) => {
                            let solution = extract_timeout_solution_reconciled(
                                sf, &a_ext, &b, &c_phase2, &basis, &x_b, &col_scale,
                                options.max_etas, options.deadline,
                            );
                            return SolverResult { status: SolveStatus::Timeout, objective: sf.obj_offset, solution, iterations: total_iters, ..Default::default() };
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
                    if *v < 0.0 { *v = 0.0; }
                }

                let mut pricing2 = SteepestEdgePricing::new(n_ext);
                match revised_simplex_core(
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
                ) {
                    SimplexOutcome::Optimal(obj2, mut y) => {
                        match reconcile_final_basis_state(&a_ext, &b, &c_phase2, &basis, &mut x_b, &mut y, options.max_etas, options.deadline) {
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
                                return SolverResult { status: SolveStatus::Timeout, objective: obj2 + sf.obj_offset, solution, iterations: total_iters, ..Default::default() };
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
                        let ws = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
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
                    SimplexOutcome::SingularBasis => {
                        SolverResult::numerical_error()
                    }
                }
            }
            SimplexOutcome::Unbounded => {
                // Phase I unbounded direction implies primal infeasibility. Attempt to
                // extract a Farkas certificate from the current basis (same discriminator
                // as the !phase1_feasible path).
                let farkas = extract_farkas_certificate(
                    &a_ext, &b, &basis, m, sf.n_total, options,
                );
                SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: farkas,
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    iterations: total_iters,
                    ..Default::default()
                }
            },
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
                                objective: obj1 + sf.obj_offset,
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
                    let rec_obj: f64 = (0..m)
                        .map(|i| c_phase1[basis[i]] * x_b[i].max(0.0))
                        .sum();
                    if rec_obj > PIVOT_TOL {
                        return SolverResult {
                            status: SolveStatus::Timeout,
                            objective: obj1 + sf.obj_offset,
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
                            &a_ext, &b, &c_phase2, &basis, &mut x_b, &mut y_transition,
                            options.max_etas, options.deadline,
                        ) {
                            Ok(()) => {}
                            Err(crate::error::SolverError::DeadlineExceeded) => {
                                let solution = extract_timeout_solution_reconciled(
                                    sf, &a_ext, &b, &c_phase2, &basis, &x_b, &col_scale,
                                    options.max_etas, options.deadline,
                                );
                                return SolverResult { status: SolveStatus::Timeout, objective: sf.obj_offset, solution, iterations: total_iters, ..Default::default() };
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
                        if *v < 0.0 { *v = 0.0; }
                    }

                    let mut pricing2 = SteepestEdgePricing::new(n_ext);
                    match revised_simplex_core(
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
                    ) {
                        SimplexOutcome::Optimal(obj2, mut y) => {
                            match reconcile_final_basis_state(&a_ext, &b, &c_phase2, &basis, &mut x_b, &mut y, options.max_etas, options.deadline) {
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
                                    return SolverResult { status: SolveStatus::Timeout, objective: obj2 + sf.obj_offset, solution, iterations: total_iters, ..Default::default() };
                                }
                                Err(_) => return SolverResult::numerical_error(),
                            }
                            let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                            if !check_eq_feasibility(problem, &solution) {
                                return SolverResult::numerical_error();
                            }
                            let (dual_solution, reduced_costs, slack) =
                                extract_dual_info(sf, problem, &y, &solution, &row_scale);
                            let ws = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
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
                    objective: obj1 + sf.obj_offset,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    iterations: total_iters,
                    ..Default::default()
                }
            }
            SimplexOutcome::SingularBasis => {
                SolverResult::numerical_error()
            }
        }
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
    cold_basis: &[usize],
) -> Option<(Vec<usize>, Vec<f64>)> {
    use crate::basis::{BasisManager, LuBasis};
    use crate::sparse::SparseVec;
    use super::crash;
    use crate::tolerances::PIVOT_TOL;

    // 入力 needs_artificial を `cold_basis[i] >= n_total` から再構築。
    let needs_artificial: Vec<bool> = cold_basis.iter().map(|&c| c >= n_total).collect();

    let num_art_in = needs_artificial.iter().filter(|&&v| v).count();
    if num_art_in == 0 {
        return None;
    }

    let (mut basis, _, num_art_out) = crash::compute_crash_basis(
        a_ext, b_scaled, m, n_shifted, cold_basis, &needs_artificial,
    );

    if num_art_out == num_art_in {
        return None;
    }

    // partial revert loop: 負 x_b の crashed 行を artif に戻す。
    // 復元不能な行 (= 元 cold basis に artif 候補が無い ub/slack 行) で負成分が
    // 出た場合は crash 全体を放棄 (Phase I/II が x_B >= 0 不変式を回復できないため)。
    let mut x_b = vec![0.0_f64; m];
    let mut crashed_count = num_art_in - num_art_out;
    for round in 0..=CRASH_REVERT_MAX_ROUNDS {
        let mut basis_mgr = match LuBasis::new(a_ext, &basis, max_etas) {
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
    for ((ax_i, ct), bi) in ax.iter().zip(problem.constraint_types.iter()).zip(problem.b.iter()) {
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
    if !basis.iter().zip(x_b.iter()).any(|(&col, &val)| col >= sf.n_total && val.abs() < PIVOT_TOL) {
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
        if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
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
pub(crate) fn extract_solution(sf: &StandardForm, basis: &[usize], x_b: &[f64], col_scale: &[f64]) -> Vec<f64> {
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

/// Primal Phase I cycling early-bail. klein3 observation: with
/// Ge/Eq constraints, `cold_start_dual` (dual.rs) falls back to Primal
/// `two_phase_simplex` whose Phase I cycles indefinitely (no Bland switch),
/// burning the whole `solve_dual_advanced` half-deadline before Big-M can
/// start. The bail returns `Timeout` with empty solution, which lets
/// `solve_dual_advanced` invoke Big-M (`dual_simplex_core_advanced` does
/// have a Bland switch + lex perturbation) with the remaining deadline.
///
/// `K = max(BAIL_TRIGGER_FACTOR * m, BAIL_TRIGGER_MIN)`. Tuned so klein3
/// (m ≈ 88, iter rate ≈ 3300/s) bails in well under 1 s; slow-but-
/// progressing LPs that decrease the objective at least every K pivots
/// stay unaffected.
///
/// The bail fires only when **both** signatures are observed within the
/// window: the Phase I objective `c^T x_B` does not improve for K
/// consecutive iters, **and** the pivot step is essentially zero (degenerate)
/// for K' consecutive iters. The AND condition is what distinguishes true
/// cycling (klein3) from slow-but-progressing Phase I (forplan): forplan's
/// Phase I pivots have step > 0 (real basis transitions reducing arts)
/// even when individual obj decrements fall below `NO_PROGRESS_REL_EPS`,
/// so the step counter resets and bail does not fire. klein3's degenerate
/// cycling exhibits step ≈ 0 on every pivot (Charnes-perturbed values are
/// the only nonzero contribution and they cancel), so both counters trip.
///
/// Bail is also gated on `enable_phase1_cycling_bail`. Callers pass `true`
/// only for Primal Phase I (where Big-M is a meaningful fall-back); Phase II
/// and all dual-driven Phase II calls pass `false` because at Phase II
/// entry the primal incumbent is already feasible and an obj plateau there
/// signals proximity to the optimum, not cycling.
const BAIL_TRIGGER_FACTOR: usize = 10;
const BAIL_TRIGGER_MIN: usize = 5_000;
/// Step-plateau threshold K'. Set to K / `STEP_BAIL_RATIO` so a single
/// non-degenerate pivot within any K'-iter window refutes cycling. Smaller
/// than K because step ≈ 0 is a stronger per-iter signature than obj
/// plateau (which can also come from f64 noise on real decrements), so
/// fewer consecutive occurrences are required.
const STEP_BAIL_RATIO: usize = 10;
/// `current + best * REL_EPS < best`: relative threshold above f64 noise
/// (~1e-15) that filters degenerate step≈0 "non-progress" from real moves.
const NO_PROGRESS_REL_EPS: f64 = 1e-12;
/// `step` magnitudes at or below this are treated as degenerate (step ≈ 0).
/// Sized to cover the Charnes perturbation upper bound: `x_b[i]` after
/// perturbation is at most `PIVOT_TOL * m`, and a pivot whose `d[leaving]`
/// is O(1) yields `step <= PIVOT_TOL * m`. We pad by a factor for the
/// general O(1/|d|) blow-up case.
const STEP_DEGENERATE_FACTOR: f64 = 1.0;

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
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
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
    let step_zero_threshold = PIVOT_TOL * STEP_DEGENERATE_FACTOR * (m as f64).max(1.0);
    // Initialize from the actual starting objective so progress_eps is finite
    // from iteration 1.  f64::INFINITY would make progress_eps = ∞ and the
    // improvement condition `current + ∞ < ∞` always false, causing the
    // obj-progress counter to increment even on genuinely improving iterations.
    let mut best_obj: f64 = basic_obj(c, basis, x_b);
    let mut iters_since_obj_progress: usize = 0;
    let mut iters_since_step_progress: usize = 0;

    for _iter in 0..max_iter {
        *iter_count_out = iter_count_out.saturating_add(1);
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }

        // y = B^{-T} c_B, then r_j = c_j − y^T a_j for non-basic j. Both steps
        // are shared with the dual paths (see `dual_common`).
        compute_reduced_costs_into(
            a, c, &mut basis_mgr, &is_basic, n_price, basis, &mut y_dense, &mut rc_vec,
        );
        // Masking RC of blocked columns prevents pricing from re-selecting an
        // entering column known to produce a singular basis from `basis_snapshot`.
        for &j in &blocked_at_basis {
            if j < n_price {
                rc_vec[j] = 0.0;
            }
        }

        let entering_col = match pricing.select_entering(&rc_vec, n_price) {
            None => {
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Optimal(obj, y_dense.clone());
            }
            Some(j) => j,
        };

        // FTRAN: d = B^{-1} a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        // Save inf-norm of original column for the corruption check below.
        let orig_col_norm = col_vals.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
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
                if v.is_finite() { acc.max(v.abs()) } else { f64::INFINITY }
            });
            let d_corrupt = !d_max_abs.is_finite()
                || (orig_col_norm > 0.0 && d_max_abs > 1e12 * orig_col_norm);
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
                            a, basis, x_b, b_rhs, &basis_snapshot,
                            &mut is_basic, &mut basis_mgr, options,
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
                d_sv = SparseVec { indices: cr2.to_vec(), values: cv2.to_vec(), len: m };
                basis_mgr.ftran(&mut d_sv);
                d_sv.to_dense_into(&mut d_dense);
                basis_snapshot.copy_from_slice(basis);
            }
        }
        let d = &d_dense;

        // Harris 2-pass ratio test. Pass 2 selects max |d[i]| within
        // `min_ratio + PIVOT_TOL` and breaks ties by Bland's rule.
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

        let effective_floor = if min_ratio.is_finite() {
            stable_floor
        } else if stable_mode {
            for i in 0..m {
                if d[i] > PIVOT_TOL {
                    let ratio = x_b[i] / d[i];
                    if ratio < min_ratio { min_ratio = ratio; }
                }
            }
            PIVOT_TOL
        } else {
            PIVOT_TOL
        };

        if !min_ratio.is_finite() {
            return SimplexOutcome::Unbounded;
        }

        let harris_window = min_ratio + PIVOT_TOL;
        let mut leaving: Option<usize> = None;
        let mut best_pivot_abs = 0.0f64;
        for i in 0..m {
            if d[i] > effective_floor {
                let ratio = x_b[i] / d[i];
                if ratio <= harris_window {
                    let d_abs = d[i].abs();
                    if d_abs > best_pivot_abs + PIVOT_TOL {
                        best_pivot_abs = d_abs;
                        leaving = Some(i);
                    } else if (d_abs - best_pivot_abs).abs() <= PIVOT_TOL {
                        // tie: Bland's rule
                        match leaving {
                            None => leaving = Some(i),
                            Some(prev) if basis[i] < basis[prev] => leaving = Some(i),
                            _ => {}
                        }
                    }
                }
            }
        }

        let leaving_row = match leaving {
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

        pricing.update_weights(&basis_mgr, entering_col, leaving_col, d);

        is_basic[leaving_col] = false;
        is_basic[entering_col] = true;
        basis[leaving_row] = entering_col;

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
                    a, basis, x_b, b_rhs, &basis_snapshot,
                    &mut is_basic, &mut basis_mgr, options,
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
        let progress_eps = best_obj.abs().max(1.0) * NO_PROGRESS_REL_EPS;
        if current_obj + progress_eps < best_obj {
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
    for v in is_basic.iter_mut() { *v = false; }
    for &col in basis.iter() {
        is_basic[col] = true;
    }
    match LuBasis::new(a, basis, options.max_etas) {
        Ok(mut mgr) => {
            // Recompute x_B; carrying eta drift could leave a slack negative.
            x_b.copy_from_slice(b_rhs);
            mgr.ftran_dense(x_b);
            for v in x_b.iter_mut() {
                if v.abs() < options.clamp_tol { *v = 0.0; }
            }
            *basis_mgr = mgr;
            true
        }
        Err(_) => false,
    }
}
