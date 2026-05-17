//! Primal (revised) simplex: two-phase driver and the iteration core.

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::presolve::RuizScaler;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use std::sync::atomic::Ordering;

use super::pricing::{PricingStrategy, SteepestEdgePricing};
use super::{StandardForm, SimplexOutcome, extract_dual_info};

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
                    if row == i && vals[k].abs() > 1e-14 {
                        x_b[i] /= vals[k];
                        break;
                    }
                }
            }
        }
        let mut pricing = SteepestEdgePricing::new(sf.n_total);

        match revised_simplex_core(&a, &mut x_b, &c, &b, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options, &mut total_iters)
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
                if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                    eprintln!("[NE-TRACE] primal.rs:159 Direct-Phase-II SingularBasis (no Phase I)");
                }
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

        // Correct x_b for diagonal entries of initial basis columns.
        // Art cols have entry 1.0 → no change. Scaled slack cols → divide by diagonal.
        for i in 0..m {
            if let Ok((rows, vals)) = a_ext.get_column(basis[i]) {
                for (k, &row) in rows.iter().enumerate() {
                    if row == i && vals[k].abs() > 1e-14 {
                        x_b[i] /= vals[k];
                        break;
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
                        &mut total_iters,
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
                            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                                eprintln!("[NE-TRACE] primal.rs:294 Phase-I retry SingularBasis (attempt={}, total_iters={})", attempt, total_iters);
                            }
                            return SolverResult::numerical_error();
                        }
                    }
                }

                if !phase1_feasible {
                    return SolverResult {
                        status: SolveStatus::Infeasible,
                        objective: 0.0,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
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
                            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                                eprintln!("[NE-TRACE] primal.rs Phase-II Optimal-but-Eq-violated NumericalError (total_iters={})", total_iters);
                            }
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
                        if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                            eprintln!("[NE-TRACE] primal.rs:440 Phase-II SingularBasis (total_iters={})", total_iters);
                        }
                        SolverResult::numerical_error()
                    }
                }
            }
            SimplexOutcome::Unbounded => { SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            } },
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
                            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                                eprintln!("[NE-TRACE] primal.rs Phase-II-after-Phase-I-Timeout Timeout (total_iters={}, obj2={:.6e})", total_iters, obj2);
                            }
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
                            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                                eprintln!("[NE-TRACE] primal.rs:615 Phase-II-after-Phase-I-Timeout SingularBasis (total_iters={})", total_iters);
                            }
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
                if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                    eprintln!("[NE-TRACE] primal.rs:630 Phase-I SingularBasis (total_iters={})", total_iters);
                }
                SolverResult::numerical_error()
            }
        }
    }
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
    let mut max_abs_viol = 0.0_f64;
    let mut max_rel_viol = 0.0_f64;
    let mut worst_row = (0usize, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut violated = false;
    for (i, ((ax_i, ct), bi)) in ax.iter().zip(problem.constraint_types.iter()).zip(problem.b.iter()).enumerate() {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
        };
        let scale = 1.0 + bi.abs() + ax_i.abs();
        let rel = violation / scale;
        if violation > max_abs_viol {
            max_abs_viol = violation;
            worst_row = (i, *ax_i, *bi, violation);
        }
        if rel > max_rel_viol { max_rel_viol = rel; }
        if rel > tol {
            violated = true;
        }
    }
    if std::env::var("DUMP_CHECK_EQ").ok().as_deref() == Some("1") {
        eprintln!("[check_eq] name={:?} m={} max_abs_viol={:.3e} max_rel_viol={:.3e} tol={:.3e} worst_row=(i={}, ax={:.6e}, b={:.6e}, viol={:.3e}) PASS={}",
            problem.name, problem.num_constraints, max_abs_viol, max_rel_viol, tol,
            worst_row.0, worst_row.1, worst_row.2, worst_row.3, !violated);
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
    let m = basis.len();
    let mut basis_mgr = LuBasis::new_timed(a, basis, max_etas, deadline)?;

    x_b.copy_from_slice(b);
    basis_mgr.ftran_dense(x_b);
    for value in x_b.iter_mut() {
        if value.abs() < 1e-12 {
            *value = 0.0;
        }
    }

    for i in 0..m {
        y[i] = c[basis[i]];
    }
    basis_mgr.btran_dense(y);
    Ok(())
}

/// Map the standard-form basic solution back to original variables, inverting
/// shifts/sign-flips/splits.  `col_scale` is the Ruiz column scale (or empty).
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
            value = value + TwoFloat::new_mul(coeff, x_new[new_idx]);
        }
        *sol_j = f64::from(value);
    }
    solution
}

/// Revised simplex core: BTRAN → pricing → FTRAN → Harris ratio test →
/// rank-1 basis update, with on-demand LU refactor.
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
) -> SimplexOutcome {
    let max_iter = usize::MAX; // timeout is the real guard
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // Buffers reused each iteration.
    let mut c_b = vec![0.0f64; m];
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

    for _iter in 0..max_iter {
        *iter_count_out = iter_count_out.saturating_add(1);
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }

        // y = BTRAN(c_B); c_B is dense so use btran_dense.
        for i in 0..m {
            c_b[i] = c[basis[i]];
        }
        y_dense.copy_from_slice(&c_b);
        basis_mgr.btran_dense(&mut y_dense);
        let y = &y_dense;

        for j in 0..n_price {
            if is_basic[j] {
                rc_vec[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut ya = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                ya += y[row] * vals[k];
            }
            rc_vec[j] = c[j] - ya;
        }
        // Masking RC of blocked columns prevents pricing from re-selecting an
        // entering column known to produce a singular basis from `basis_snapshot`.
        for &j in &blocked_at_basis {
            if j < n_price {
                rc_vec[j] = 0.0;
            }
        }

        let entering_col = match pricing.select_entering(&rc_vec, n_price) {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
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
                        let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
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
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
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
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
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
