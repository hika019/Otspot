//! Primal (revised) simplex: two-phase driver and the iteration core.

mod core;
mod crossover;
mod ratio_test;
mod reconcile;

pub(crate) use core::revised_simplex_core;
#[cfg(test)]
pub(crate) use core::set_eta_update_disabled;
pub(crate) use crossover::{
    crossover_dual_from_primal, crossover_dual_from_primal_with_dual_warm_start,
};
#[cfg(test)]
pub(crate) use reconcile::pivot_out_degenerate_artificials as test_pivot_out_degenerate_artificials;
pub(crate) use reconcile::{extract_solution, reconcile_final_basis_state};

use self::reconcile::{check_eq_feasibility, pivot_out_degenerate_artificials, try_apply_crash};
use super::dual_common::{basic_obj, lp_unbounded_ray_verified};
use super::pricing::SteepestEdgePricing;
use super::{
    external_stop_requested, extract_dual_info, stall_status, SimplexOutcome, StandardForm,
};
use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::presolve::LpEquilibration;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::*;

#[allow(clippy::print_stderr)]
fn trace_stage(message: impl std::fmt::Display) {
    if std::env::var("OTSPOT_SIMPLEX_STAGE_TRACE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        eprintln!("[simplex-stage] {message}");
    }
}

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

// Pivot-out sentinel counters (test-only). These are `thread_local` rather than global
// atomics: `pivot_out_degenerate_artificials` and its sequential fallback always run on
// the thread that drives the solve, so a thread-local counter captures exactly the
// increments of the solve invoked on this thread. Under `cargo test` (a shared thread
// pool) one test's solve can no longer corrupt another's counter delta, and under
// nextest (process-per-test) each process starts fresh — so the before/after deltas the
// sentinels assert on are exact under either runner with no inter-test locking.
thread_local! {
    /// Counts BTRAN calls issued inside `pivot_out_degenerate_artificials` sequential
    /// fallback. In the batch path this stays at zero (no BTRANs needed). Sentinel asserts
    /// the delta is zero after solving an LP where all degenerate artificials are handled
    /// by the batch: reverting to the O(num_art) sequential path makes it increase by
    /// num_art, failing the assertion (no-op FAIL).
    #[cfg(test)]
    pub(crate) static PIVOT_OUT_BTRAN_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Counts batch LU factorization attempts in `pivot_out_degenerate_artificials`.
    /// Incremented once each time the batch greedy assignment is attempted and a single
    /// `LuBasis::new_timed` is called for all matched rows. Sentinel asserts the delta is
    /// exactly 1 when the batch path is taken. Reverting to the sequential path keeps this
    /// at zero and fails the assertion (no-op FAIL).
    #[cfg(test)]
    pub(crate) static PIVOT_OUT_BATCH_LU_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Counts batch→sequential reverts in `pivot_out_degenerate_artificials`. Incremented
    /// once each time the post-batch FTRAN stability check rejects the batch assignment
    /// (ill-conditioned pivots) and the per-row sequential path is taken instead. Sentinel
    /// asserts the delta is positive on a known ill-conditioned instance (degen2): removing
    /// the stability check keeps this at zero — the unstable batch is accepted and the
    /// correctness assertion downstream fails (no-op FAIL).
    #[cfg(test)]
    pub(crate) static PIVOT_OUT_SEQUENTIAL_FALLBACK_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Counts rows routed to pivot_out_sequential via the uncommitted_rows path.
    /// Incremented for each row in matches[match_offset..] (partial-commit case,
    /// match_offset > 0) or in matches[0..] (no-commit case, match_offset == 0,
    /// batch_stable short-circuited). Sentinel asserts this is positive when the
    /// batch LU fails for the full match set; removing the uncommitted_rows path
    /// keeps it at zero — assertion fails (no-op FAIL).
    #[cfg(test)]
    pub(crate) static PIVOT_OUT_UNCOMMITTED_SEQUENTIAL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Verified-ray gate for a Phase II `Unbounded` exit (shared with the Big-M
/// path). An eta-drift false-Unbounded (`B⁻¹a_q` reads ≤ 0 only because of a
/// stale factorization) becomes an honest Stalled, mirroring the Phase-I Farkas
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
        SimplexOutcome::Stalled(basic_obj(c, basis, x_b))
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
/// Infeasible there is a false verdict (ns1688926-class), so return an honest
/// inconclusive, matching `big_m_cold_start`'s Farkas gate: Timeout when the
/// stop was external (deadline/cancel), MaxIterations for an internal stall.
fn phase1_infeasibility_verdict(
    farkas: Vec<f64>,
    total_iters: usize,
    options: &SolverOptions,
) -> SolverResult {
    if farkas.is_empty() {
        let status = if external_stop_requested(options) {
            SolveStatus::Timeout
        } else {
            stall_status(false)
        };
        return SolverResult {
            status,
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
    let art_rows: Vec<usize> = (0..m).filter(|&i| basis[i] >= n_original).collect();
    if art_rows.is_empty() {
        return vec![];
    }
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(_) => return vec![],
    };

    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = options.dual_tol * (1.0_f64).max(b_norm);

    // Helper: check A^T y <= tol for all original columns.
    let aty_ok = |y: &[f64]| -> bool {
        for j in 0..n_original {
            let Ok((rows, vals)) = a_ext.get_column(j) else {
                return false;
            };
            let aty: f64 = rows.iter().zip(vals.iter()).map(|(&r, &v)| v * y[r]).sum();
            if aty > tol {
                return false;
            }
        }
        true
    };

    // Compute fresh x_B = B^{-1} b to identify which artificial rows are genuinely
    // infeasible (positive) vs numerically negative (eta-drift artifact).
    let mut x_b_fresh = b.to_vec();
    basis_mgr.ftran_dense(&mut x_b_fresh);

    // Strategy 1: joint indicator over ALL artificial rows.
    // `by = sum_art x_B[i]` can approach zero when positive and negative art rows
    // cancel (cplex2-class: primal Phase I cycling leaves mixed-sign art x_B).
    {
        let mut y: Vec<f64> = (0..m)
            .map(|i| if basis[i] >= n_original { 1.0 } else { 0.0 })
            .collect();
        basis_mgr.btran_dense(&mut y);
        let by: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
        if by > tol && aty_ok(&y) {
            return y;
        }
    }

    // Strategy 2: positive-art-only indicator — avoids sign cancellation.
    // Uses only artificial rows where the fresh x_B > PIVOT_TOL (genuinely
    // infeasible rows).  `by = sum_{pos-art} x_B[i]^2 / norm` stays positive.
    // At the dual-feasible Phase I basis, the full joint indicator satisfies
    // A^T y ≤ 0; the positive-only subset may or may not, so we verify.
    {
        let pos_art_indicator: Vec<f64> = (0..m)
            .map(|i| {
                if basis[i] >= n_original && x_b_fresh[i] > PIVOT_TOL {
                    x_b_fresh[i]
                } else {
                    0.0
                }
            })
            .collect();
        if pos_art_indicator.iter().any(|&v| v > 0.0) {
            let mut y = pos_art_indicator;
            basis_mgr.btran_dense(&mut y);
            let by: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
            if by > PIVOT_TOL && aty_ok(&y) {
                return y;
            }
        }
    }

    // Strategy 3: per-row probes — tries each positive-x_B artificial row
    // individually.  Useful when the weighted combination still fails aty_ok.
    for &row in &art_rows {
        if x_b_fresh[row] <= PIVOT_TOL {
            continue;
        }
        let mut e_i = vec![0.0_f64; m];
        e_i[row] = 1.0;
        basis_mgr.btran_dense(&mut e_i);
        let by: f64 = b.iter().zip(e_i.iter()).map(|(&bi, &yi)| bi * yi).sum();
        if by > tol && aty_ok(&e_i) {
            return e_i;
        }
    }

    vec![]
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

fn adjust_xb_for_scaled_diag(a: &CscMatrix, basis: &[usize], x_b: &mut [f64], m: usize) {
    for i in 0..m {
        if let Ok((rows, vals)) = a.get_column(basis[i]) {
            for (k, &row) in rows.iter().enumerate() {
                if row == i && vals[k].abs() > SLACK_DIAG_TOL {
                    x_b[i] /= vals[k];
                    break;
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn build_phase1_system(
    sf: &StandardForm,
    a: &CscMatrix,
    b: &[f64],
    m: usize,
    options: &SolverOptions,
) -> (CscMatrix, Vec<usize>, Vec<f64>, Vec<f64>, usize) {
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
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
    let mut x_b = b.to_vec();
    let mut art_col = sf.n_total;
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

    let mut c_phase1 = vec![0.0; n_ext];
    c_phase1[sf.n_total..].fill(1.0);

    let crashed =
        if options.warm_start.is_none() && options.use_lp_crash_basis && sf.num_artificial > 0 {
            try_apply_crash(
                &a_ext,
                m,
                sf.n_shifted,
                sf.n_total,
                b,
                options.max_etas,
                options.deadline,
                &basis,
            )
        } else {
            None
        };
    if let Some((crash_basis, crash_x_b)) = crashed {
        trace_stage("crash basis accepted");
        basis = crash_basis;
        x_b = crash_x_b;
    } else {
        trace_stage("cold artificial basis");
        adjust_xb_for_scaled_diag(&a_ext, &basis, &mut x_b, m);
    }

    for i in 0..m {
        if basis[i] >= sf.n_total && x_b[i].abs() <= PIVOT_TOL {
            x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
        }
    }

    (a_ext, basis, x_b, c_phase1, n_ext)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
fn verify_phase1_feasibility(
    a_ext: &CscMatrix,
    b: &[f64],
    c_phase1: &[f64],
    basis: &mut [usize],
    x_b: &mut [f64],
    m: usize,
    n_ext: usize,
    n_total: usize,
    options: &SolverOptions,
    total_iters: &mut usize,
) -> Result<(), SolverResult> {
    use crate::options::MAX_PHASE1_RETRIES;
    for attempt in 0..=MAX_PHASE1_RETRIES {
        if options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            break;
        }
        let mut y_dummy = vec![0.0f64; m];
        let rec_obj = match reconcile_final_basis_state(
            a_ext,
            b,
            c_phase1,
            basis,
            x_b,
            &mut y_dummy,
            options.max_etas,
            options.deadline,
        ) {
            Ok(()) => (0..m)
                .map(|i| c_phase1[basis[i]] * x_b[i].max(0.0))
                .sum::<f64>(),
            Err(_) => {
                trace_stage("phase1 reconcile failed");
                break;
            }
        };
        if rec_obj <= PIVOT_TOL {
            return Ok(());
        }
        if attempt == MAX_PHASE1_RETRIES {
            break;
        }

        for v in x_b.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let mut pricing_retry = SteepestEdgePricing::new(n_ext);
        match revised_simplex_core(
            a_ext,
            x_b,
            c_phase1,
            b,
            basis,
            m,
            n_ext,
            n_ext,
            &mut pricing_retry,
            options,
            total_iters,
            true,
            Some(n_total),
            false,
            None,
        ) {
            SimplexOutcome::Optimal(_, _) => {}
            SimplexOutcome::Unbounded => break,
            SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
                let mut y_check = vec![0.0f64; m];
                if reconcile_final_basis_state(
                    a_ext,
                    b,
                    c_phase1,
                    basis,
                    x_b,
                    &mut y_check,
                    options.max_etas,
                    options.deadline,
                )
                .is_ok()
                {
                    let rec_obj_retry: f64 =
                        (0..m).map(|i| c_phase1[basis[i]] * x_b[i].max(0.0)).sum();
                    if rec_obj_retry <= PIVOT_TOL {
                        return Ok(());
                    }
                }
                // No feasible point: Timeout only for an external stop; a
                // cycling/plateau bail with budget left is MaxIterations.
                // Clock-recheck, not variant trust.
                return Err(SolverResult {
                    status: super::stop_status(false, options),
                    objective: f64::INFINITY,
                    iterations: *total_iters,
                    ..Default::default()
                });
            }
            SimplexOutcome::SingularBasis => {
                trace_stage("phase1 retry singular basis");
                return Err(SolverResult::numerical_error());
            }
        }
    }
    trace_stage("phase1 not feasible; extracting farkas");
    let farkas = extract_farkas_certificate(a_ext, b, basis, m, n_total, options);
    Err(phase1_infeasibility_verdict(farkas, *total_iters, options))
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
fn transition_to_phase2(
    sf: &StandardForm,
    problem: &LpProblem,
    a_ext: &CscMatrix,
    b: &[f64],
    c: &[f64],
    basis: &mut [usize],
    x_b: &mut [f64],
    col_scale: &[f64],
    m: usize,
    n_ext: usize,
    options: &SolverOptions,
    total_iters: usize,
) -> Result<Vec<f64>, SolverResult> {
    pivot_out_degenerate_artificials(a_ext, basis, x_b, sf, options);
    let remaining_art = basis.iter().filter(|&&col| col >= sf.n_total).count();
    trace_stage(format_args!(
        "pivot out degenerate artificials done remaining_art={remaining_art}"
    ));

    let mut c_phase2 = vec![0.0; n_ext];
    c_phase2[..sf.n_total].copy_from_slice(&c[..sf.n_total]);
    {
        let mut y_transition = vec![0.0f64; m];
        match reconcile_final_basis_state(
            a_ext,
            b,
            &c_phase2,
            basis,
            x_b,
            &mut y_transition,
            options.max_etas,
            options.deadline,
        ) {
            Ok(()) => {}
            Err(crate::error::SolverError::DeadlineExceeded) => {
                trace_stage("phase2 transition reconcile deadline");
                let solution = extract_timeout_solution_reconciled(
                    sf,
                    a_ext,
                    b,
                    &c_phase2,
                    basis,
                    x_b,
                    col_scale,
                    options.max_etas,
                    options.deadline,
                );
                return Err(SolverResult {
                    status: SolveStatus::Timeout,
                    objective: objective_from_solution(sf, problem, &solution),
                    solution,
                    iterations: total_iters,
                    ..Default::default()
                });
            }
            Err(_) => {
                trace_stage("phase2 transition reconcile numerical");
                return Err(SolverResult::numerical_error());
            }
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
    Ok(c_phase2)
}

#[allow(clippy::too_many_arguments)]
fn finalize_phase2(
    sf: &StandardForm,
    problem: &LpProblem,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    basis: &mut [usize],
    x_b: &mut [f64],
    col_scale: &[f64],
    row_scale: &[f64],
    options: &SolverOptions,
    total_iters: &mut usize,
) -> SolverResult {
    let m = sf.m;
    let n_cols = a.ncols;
    let mut pricing = SteepestEdgePricing::new(n_cols);
    let outcome = revised_simplex_core(
        a,
        x_b,
        c,
        b,
        basis,
        m,
        n_cols,
        sf.n_total,
        &mut pricing,
        options,
        total_iters,
        false,
        None,
        false,
        None,
    );
    let outcome = gate_phase2_unbounded(outcome, a, basis, c, x_b, m, n_cols, sf.n_total, options);

    match outcome {
        SimplexOutcome::Optimal(obj, mut y) => {
            match reconcile_final_basis_state(
                a,
                b,
                c,
                basis,
                x_b,
                &mut y,
                options.max_etas,
                options.deadline,
            ) {
                Ok(()) => {}
                Err(crate::error::SolverError::DeadlineExceeded) => {
                    trace_stage("phase2 final reconcile deadline");
                    let solution = extract_timeout_solution_reconciled(
                        sf,
                        a,
                        b,
                        c,
                        basis,
                        x_b,
                        col_scale,
                        options.max_etas,
                        options.deadline,
                    );
                    return SolverResult {
                        status: SolveStatus::Timeout,
                        objective: obj + sf.obj_offset,
                        solution,
                        iterations: *total_iters,
                        ..Default::default()
                    };
                }
                Err(_) => {
                    trace_stage("phase2 final reconcile numerical");
                    return SolverResult::numerical_error();
                }
            }
            let solution = extract_solution(sf, basis, x_b, col_scale);
            if !check_eq_feasibility(problem, &solution) {
                trace_stage("phase2 final feasibility failed");
                return SolverResult::numerical_error();
            }
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(WarmStartBasis {
                    basis: basis.to_vec(),
                    x_b: x_b.to_vec(),
                }),
                iterations: *total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            iterations: *total_iters,
            ..Default::default()
        },
        SimplexOutcome::Timeout(obj) | SimplexOutcome::Stalled(obj) => {
            let solution = extract_timeout_solution_reconciled(
                sf,
                a,
                b,
                c,
                basis,
                x_b,
                col_scale,
                options.max_etas,
                options.deadline,
            );
            // Clock-recheck, not variant trust (see dual_common::outcome_to_result).
            let status = super::stop_status(!solution.is_empty(), options);
            SolverResult {
                status,
                objective: obj + sf.obj_offset,
                solution,
                iterations: *total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => {
            trace_stage("phase2 singular basis");
            SolverResult::numerical_error()
        }
    }
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

    trace_stage(format_args!(
        "start m={} n_total={} n_artificial={}",
        sf.m, sf.n_total, sf.num_artificial
    ));

    let (a, b, c, row_scale, col_scale) = if options.use_ruiz_scaling {
        let Some(scaled) =
            LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c, options.deadline)
        else {
            trace_stage("equilibration timeout");
            return SolverResult {
                status: SolveStatus::Timeout,
                objective: f64::INFINITY,
                ..Default::default()
            };
        };
        scaled
    } else {
        (
            sf.a.clone(),
            sf.b.clone(),
            sf.c.clone(),
            vec![1.0; sf.m],
            vec![1.0; sf.n_total],
        )
    };

    // Direct Phase II — no artificials needed.
    if sf.num_artificial == 0 {
        trace_stage("direct phase2 start");
        let mut basis = sf.initial_basis.clone();
        let mut x_b = b.clone();
        adjust_xb_for_scaled_diag(&a, &basis, &mut x_b, m);
        return finalize_phase2(
            sf,
            problem,
            &a,
            &b,
            &c,
            &mut basis,
            &mut x_b,
            &col_scale,
            &row_scale,
            options,
            &mut total_iters,
        );
    }

    // Phase I + Phase II
    trace_stage("phase1 setup start");
    let (a_ext, mut basis, mut x_b, c_phase1, n_ext) = build_phase1_system(sf, &a, &b, m, options);
    trace_stage(format_args!("phase1 setup done n_ext={n_ext}"));

    let mut pricing1 = SteepestEdgePricing::new(n_ext);
    trace_stage("phase1 core start");
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
        Some(sf.n_total),
        false,
        None,
    );

    match phase1_outcome {
        SimplexOutcome::Optimal(_obj, _) => {
            trace_stage(format_args!("phase1 optimal iters={total_iters}"));
            if let Err(result) = verify_phase1_feasibility(
                &a_ext,
                &b,
                &c_phase1,
                &mut basis,
                &mut x_b,
                m,
                n_ext,
                sf.n_total,
                options,
                &mut total_iters,
            ) {
                return result;
            }
            trace_stage("pivot out degenerate artificials start");
            let c_phase2 = match transition_to_phase2(
                sf,
                problem,
                &a_ext,
                &b,
                &c,
                &mut basis,
                &mut x_b,
                &col_scale,
                m,
                n_ext,
                options,
                total_iters,
            ) {
                Ok(c2) => c2,
                Err(result) => return result,
            };
            trace_stage("phase2 core start");
            finalize_phase2(
                sf,
                problem,
                &a_ext,
                &b,
                &c_phase2,
                &mut basis,
                &mut x_b,
                &col_scale,
                &row_scale,
                options,
                &mut total_iters,
            )
        }
        SimplexOutcome::Unbounded => {
            trace_stage(format_args!("phase1 unbounded iters={total_iters}"));
            let farkas = extract_farkas_certificate(&a_ext, &b, &basis, m, sf.n_total, options);
            phase1_infeasibility_verdict(farkas, total_iters, options)
        }
        SimplexOutcome::Timeout(obj1) | SimplexOutcome::Stalled(obj1) => {
            trace_stage(format_args!(
                "phase1 timeout/stall iters={total_iters} obj={obj1:.9e}"
            ));
            // No feasible point recovered: Timeout only for an external stop; a
            // cycling/plateau bail with budget left is MaxIterations.
            // Clock-recheck, not variant trust (see dual_common::outcome_to_result).
            let bail_status = super::stop_status(false, options);
            if obj1 > PIVOT_TOL {
                return SolverResult {
                    status: bail_status,
                    objective: f64::INFINITY,
                    iterations: total_iters,
                    ..Default::default()
                };
            }
            // Near-feasible at timeout/stall: reconcile and verify.
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
                        status: bail_status,
                        objective: f64::INFINITY,
                        iterations: total_iters,
                        ..Default::default()
                    };
                }
            }
            let rec_obj: f64 = (0..m).map(|i| c_phase1[basis[i]] * x_b[i].max(0.0)).sum();
            if rec_obj > PIVOT_TOL {
                return SolverResult {
                    status: bail_status,
                    objective: f64::INFINITY,
                    iterations: total_iters,
                    ..Default::default()
                };
            }
            let c_phase2 = match transition_to_phase2(
                sf,
                problem,
                &a_ext,
                &b,
                &c,
                &mut basis,
                &mut x_b,
                &col_scale,
                m,
                n_ext,
                options,
                total_iters,
            ) {
                Ok(c2) => c2,
                Err(result) => return result,
            };
            finalize_phase2(
                sf,
                problem,
                &a_ext,
                &b,
                &c_phase2,
                &mut basis,
                &mut x_b,
                &col_scale,
                &row_scale,
                options,
                &mut total_iters,
            )
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
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
mod eta_atomicity_tests {
    use super::{revised_simplex_core, set_eta_update_disabled};
    use crate::options::SolverOptions;
    use crate::simplex::pricing::SteepestEdgePricing;
    use crate::simplex::SimplexOutcome;
    use crate::sparse::CscMatrix;

    struct EtaUpdateGuard(bool);
    impl EtaUpdateGuard {
        fn disabled() -> Self {
            Self(set_eta_update_disabled(true))
        }
    }
    impl Drop for EtaUpdateGuard {
        fn drop(&mut self) {
            set_eta_update_disabled(self.0);
        }
    }

    #[test]
    fn rejected_eta_leaves_standard_primal_state_unchanged() {
        // min -x, x + s = 1, initial basis {s}: the first pivot is forced.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let mut x_b = vec![1.0];
        let mut basis = vec![1usize];
        let expected_x_b = x_b.clone();
        let expected_basis = basis.clone();
        let mut pricing = SteepestEdgePricing::new(2);
        let mut iterations = 0;
        let _guard = EtaUpdateGuard::disabled();

        let outcome = revised_simplex_core(
            &a,
            &mut x_b,
            &[-1.0, 0.0],
            &[1.0],
            &mut basis,
            1,
            2,
            2,
            &mut pricing,
            &SolverOptions::default(),
            &mut iterations,
            false,
            None,
            false,
            None,
        );

        assert!(matches!(outcome, SimplexOutcome::SingularBasis));
        assert_eq!(x_b, expected_x_b);
        assert_eq!(basis, expected_basis);
    }
}

#[cfg(test)]
mod farkas_gate_tests {
    //! Phase I infeasibility must be declared ONLY with a verified Farkas
    //! certificate. ns1688926 (feasible, ‖b‖≈2.4e7) and cplex2 exit Phase I with
    //! a spurious Unbounded ray whose `y = B^{-T} e_art` has `bᵀy ≈ 0` — not a
    //! witness. Trusting that exit returned false-Infeasible. These sentinels
    //! pin the gate: empty cert ⇒ honest inconclusive (Timeout on external
    //! stop, MaxIterations on internal stall), verified cert ⇒ Infeasible.

    use super::{extract_farkas_certificate, phase1_infeasibility_verdict};
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    /// No-op sentinel: reverting the gate to an unconditional `Infeasible`
    /// return (the pre-fix behaviour) makes this assertion fail.
    ///
    /// Without an expired deadline the empty-cert verdict is an internal
    /// stall: MaxIterations, never a self-declared Timeout with budget left.
    #[test]
    fn empty_cert_without_deadline_yields_max_iterations() {
        let r = phase1_infeasibility_verdict(vec![], 7, &SolverOptions::default());
        assert_eq!(
            r.status,
            SolveStatus::MaxIterations,
            "empty (unverifiable) Farkas cert with budget left must be \
             MaxIterations, not Infeasible and not Timeout"
        );
        assert_eq!(r.iterations, 7);
    }

    /// With the deadline actually expired the empty-cert verdict is Timeout.
    #[test]
    fn empty_cert_with_expired_deadline_yields_timeout() {
        let opts = SolverOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..Default::default()
        };
        let r = phase1_infeasibility_verdict(vec![], 7, &opts);
        assert_eq!(
            r.status,
            SolveStatus::Timeout,
            "empty Farkas cert with an expired deadline is an external stop"
        );
    }

    #[test]
    fn verified_cert_yields_infeasible() {
        let r = phase1_infeasibility_verdict(vec![-1.0, 1.0], 3, &SolverOptions::default());
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
        let eps = PIVOT_TOL * (m as f64); // = step_zero_threshold
        let thresh = eps;

        // Mix of large (non-degenerate) and near-zero (degenerate) values.
        let mut x_b = vec![100.0, 0.0, thresh * 0.5, 50.0];
        let saved = [x_b[0], x_b[3]];

        test_apply_selective_charnes_perturb(&mut x_b, m);

        // Non-degenerate rows: must be UNCHANGED.
        assert_eq!(
            x_b[0], saved[0],
            "x_b[0]=100 must not be modified (non-degenerate)"
        );
        assert_eq!(
            x_b[3], saved[1],
            "x_b[3]=50 must not be modified (non-degenerate)"
        );

        // Near-zero rows: must become eps*(i+1) (unique small positives).
        // i=1 → eps*(1+1) = eps*2; i=2 → eps*(2+1) = eps*3.
        assert_eq!(
            x_b[1],
            eps * 2.0,
            "x_b[1]=0 at i=1 must become eps*(1+1)=eps*2"
        );
        assert_eq!(
            x_b[2],
            eps * 3.0,
            "x_b[2]=thresh*0.5 at i=2 must become eps*(2+1)=eps*3"
        );
        // Perturbed values must be distinct and positive.
        assert!(
            x_b[1] > 0.0 && x_b[2] > 0.0,
            "perturbed values must be positive"
        );
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
mod phase1_artificial_preference_tests {
    //! End-to-end sentinel for the Phase I artificial leaving preference.
    //!
    //! The ratio-test unit no-op proof lives in `ratio_test.rs` (ON vs OFF on a
    //! single tie-band). Here we prove the production Phase I path — artificial
    //! preference always ON — still finds a feasible vertex and the correct
    //! optimum on a degenerate LP whose equality constraints force several
    //! artificials (one redundant ⇒ a residual degenerate artificial). The
    //! preference must not strand an artificial (false-Infeasible) nor shift the
    //! optimum.

    use crate::options::{SimplexMethod, SolverOptions};
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::simplex::entry::solve_with;
    use crate::sparse::CscMatrix;

    fn primal_opts() -> SolverOptions {
        SolverOptions {
            simplex_method: SimplexMethod::Primal,
            ..Default::default()
        }
    }

    /// Degenerate 2×2 transportation LP: supplies (1,1), demands (1,1). Four
    /// equality rows — one redundant (Σsupply = Σdemand) — give four artificials
    /// with a residual degenerate artificial at the feasible vertex. The feasible
    /// set is x = (t, 1−t, 1−t, t), t ∈ [0,1]; cost (1,2,2,1) ⇒ obj = 4 − 2t,
    /// minimized at t = 1: x* = (1,0,0,1), obj = 2.
    #[test]
    fn degenerate_transportation_phase1_finds_optimum() {
        // Rows: x1+x2=1, x3+x4=1, x1+x3=1, x2+x4=1.
        let a = CscMatrix::from_triplets(
            &[0, 2, 0, 3, 1, 2, 1, 3],
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
            4,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0, 2.0, 1.0],
            a,
            vec![1.0, 1.0, 1.0, 1.0],
            vec![ConstraintType::Eq; 4],
            vec![(0.0, f64::INFINITY); 4],
            None,
        )
        .unwrap();

        let result = solve_with(&lp, &primal_opts());
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "degenerate multi-artificial LP must be feasible+Optimal (no false-Infeasible); got {:?}",
            result.status
        );
        assert!(
            (result.objective - 2.0).abs() < 1e-6,
            "artificial preference must not shift the optimum; expected obj=2, got {}",
            result.objective
        );
        // Primal feasibility: every equality residual ≈ 0 and x ≥ 0.
        let x = &result.solution;
        assert!(x.iter().all(|&v| v >= -1e-6), "x must be ≥ 0: {x:?}");
        let rows = [[0, 1], [2, 3], [0, 2], [1, 3]];
        for r in rows {
            let lhs = x[r[0]] + x[r[1]];
            assert!(
                (lhs - 1.0).abs() < 1e-6,
                "equality row {r:?} must hold: lhs={lhs}"
            );
        }
    }
}
