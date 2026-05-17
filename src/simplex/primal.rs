//! 主シンプレックス法（Primal Simplex）の実装
//!
//! 2相シンプレックス法（Phase I + Phase II）と改訂シンプレックス法コアを提供する。
//! このモジュールは `mod.rs` から分離された primal simplex 専用モジュール。

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

/// 2相シンプレックス法で標準形LPを解く
///
/// - 人工変数が不要な場合: Phase II に直接進む
/// - 人工変数が必要な場合: Phase I で実行可能基底を求めてから Phase II を実行
///
/// Phase I の目的関数は人工変数の和の最小化。
/// 最小値がゼロより大きければ元問題は実行不可能（Infeasible）。
/// Ruiz equilibration スケーリングを適用してから解く。
pub(crate) fn two_phase_simplex(sf: &StandardForm, problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = sf.m;
    // task #19: simplex 反復回数を SolverResult.iterations に伝播するため、
    // すべての core 呼び出しで out-param 経由で累積する。
    let mut total_iters: usize = 0;

    // Apply Ruiz equilibration scaling to the standard form
    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&sf.a, &sf.b, &sf.c);

    if sf.num_artificial == 0 {
        // Direct Phase II
        let mut basis = sf.initial_basis.clone();
        let mut x_b = b.clone();
        // Correct x_b for Ruiz-scaled diagonal initial basis: x_b[i] = b_scaled[i] / B[i, basis[i]]
        // After Ruiz equilibration, slack basis columns have scaled diagonal entries != 1.0,
        // so the naive x_b = b_scaled is inconsistent with B * x_b = b_scaled.
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
                        return SolverResult { status: SolveStatus::Timeout, objective: obj + sf.obj_offset, solution, ..Default::default() };
                    }
                    Err(_) => return SolverResult::numerical_error(),
                }
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                // Eq制約 feasibility check — 偽 Optimal 返却を防ぐ defense-in-depth
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
                // Reconcile + retry loop for Phase I premature termination.
                //
                // Primal Phase I sometimes reaches a state that is dual-feasible (all r_j ≥ 0)
                // but primal-infeasible (x_b < 0 for some rows) due to LU eta drift.
                // In that case primal simplex declares "Optimal" incorrectly.
                // After fresh-LU reconcile reveals primal infeasibility, we switch to
                // DUAL SIMPLEX to restore primal feasibility (dual simplex starts from
                // dual-feasible, fixes primal infeasibility). Then re-check.
                // Phase I retry: 安全装置として上限を残す (MAX_PHASE1_RETRIES 撤廃すると
                // revised_simplex_core が「同じ basis で Optimal を返し続ける」無限ループに
                // 入るケースで bandm/beaconfd 等が TIMEOUT 化したため revert)。
                // TODO: 「同じ basis を繰り返したら abort」の progress 検出を実装し、
                //       MAX_PHASE1_RETRIES に依存しない収束判定に置換する。
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

                    // Positive artificials remain: clamp any negative drift, retry Phase I.
                    for v in x_b.iter_mut() {
                        if *v < 0.0 { *v = 0.0; }
                    }
                    let mut pricing_retry = SteepestEdgePricing::new(n_ext);
                    match revised_simplex_core(
                        &a_ext, &mut x_b, &c_phase1, &b, &mut basis,
                        m, n_ext, n_ext, &mut pricing_retry, options,
                        &mut total_iters,
                    ) {
                        SimplexOutcome::Optimal(_, _) => {} // check reconcile next iteration
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
                            return SolverResult { status: SolveStatus::Timeout, objective: sf.obj_offset, solution, ..Default::default() };
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
                                return SolverResult { status: SolveStatus::Timeout, objective: obj2 + sf.obj_offset, solution, ..Default::default() };
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
                // obj1 ≤ PIVOT_TOL means non-degen arts appear near-zero at timeout.
                // Reconcile with fresh LU to get accurate x_b = B^{-1}b, then
                // re-verify feasibility. If reconcile shows arts > PIVOT_TOL, Phase I
                // didn't truly achieve feasibility → return Timeout (not run Phase II).
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
                                ..Default::default()
                            };
                        }
                    }
                    // Post-reconcile feasibility check: if arts still > PIVOT_TOL,
                    // Phase I hasn't converged. Return Timeout rather than running
                    // Phase II from an infeasible starting point (which yields OBJ_MISMATCH).
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
                                return SolverResult { status: SolveStatus::Timeout, objective: sf.obj_offset, solution, ..Default::default() };
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
                                    return SolverResult { status: SolveStatus::Timeout, objective: obj2 + sf.obj_offset, solution, ..Default::default() };
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

/// 制約 feasibility を確認する（defense-in-depth）。
///
/// `solution` は problem の変数空間の解ベクトル。各制約 i について
/// `violation = |Ax_i − b_i|` (Eq) / `max(0, Ax_i − b_i)` (Le) /
/// `max(0, b_i − Ax_i)` (Ge) を求め、**relative** に比較する:
///
///   violation > `feas_rel_tol()` * (1 + |b_i| + |Ax_i|)  →  false
///
/// **旧実装** (`FEASIBILITY_TOL = 1e-4` の絶対閾値) は scale 依存:
///   - bore3d (|b|≈O(1)) の 100-単位違反 → relative 100 で問題なく検出
///   - cycle/d6cube (|b|≫1) の **真の Optimal** にも relative 1e-7 程度の
///     `|Ax-b|` 揺らぎがあり、絶対 1e-4 で誤 reject (NumericalError 化)
///
/// relative 化により scale 非依存となり、bore3d 級異常は引き続き検出し、
/// cycle/d6cube の false NumericalError は解消する。
/// `feas_rel_tol() = sqrt(PIVOT_TOL)` は magic ではなく Wilkinson 経験則から
/// 構造的に導出（`tolerances.rs` 参照）。
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
    let mut worst_row = (0usize, 0.0_f64, 0.0_f64, 0.0_f64); // row, ax, b, viol
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

    // Build LuBasis from the Phase I final basis to enable pivot selection.
    // |d[i]| = |(B^{-1} a_j)[i]| measures the numerical stability of a candidate
    // entering pivot; raw A[i,j] would ignore basis transformation.
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(mgr) => mgr,
        Err(_) => return, // singular basis: leave artificials as-is
    };

    let mut is_basic = vec![false; a_ext.ncols];
    for &col in basis.iter() {
        is_basic[col] = true;
    }

    // BTRAN-based candidate scan (replaces the prior O(n_artificial × n_total)
    // FTRAN-per-column loop):
    //   For each degenerate artificial row i:
    //     1. z = B^{-T} e_i   (one BTRAN — gives the i-th row of B^{-1})
    //     2. d[i, j] = z · A_{:,j} for every non-basic j (sparse dot, O(nnz(A_j)))
    //     3. choose argmax_j |d[i, j]| > PIVOT_TOL, FTRAN that one column to
    //        obtain the d-vector required by basis_mgr.update.
    // Per artificial cost: 1 BTRAN + sum_j nnz(A_{:,j}) + 1 FTRAN
    //                    ≈ O(m + nnz(A)) instead of O(n_total × FTRAN).
    // osa-60 (n_total ≈ 243k, n_artificial = 11): ~30k× speedup vs. the prior
    // formulation (verified via tests/diag_ken18_osa60.rs).
    let mut z_dense = vec![0.0_f64; m];
    for i in 0..m {
        // Coarse deadline guard: with O(m + nnz(A)) per artificial the function
        // is fast enough that this should rarely fire — kept as a safety net.
        if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        if basis[i] < sf.n_total || x_b[i].abs() >= PIVOT_TOL {
            continue;
        }

        // Step 1: z = B^{-T} e_i
        z_dense.iter_mut().for_each(|v| *v = 0.0);
        z_dense[i] = 1.0;
        basis_mgr.btran_dense(&mut z_dense);

        // Step 2: scan all non-basic original columns; track argmax |d|.
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
            // Step 3: one FTRAN on the chosen column to feed basis_mgr.update.
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

    // Final singularity check: restore on failure
    if LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline).is_err() {
        basis.copy_from_slice(&basis_before);
    }
}

/// 最終基底に対して x_B = B^{-1}b, y = B^{-T}c_B を再計算する。
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

    // x_B = B^{-1} b (b は dense なので ftran_dense で sparse 変換を省略)
    x_b.copy_from_slice(b);
    basis_mgr.ftran_dense(x_b);
    for value in x_b.iter_mut() {
        if value.abs() < 1e-12 {
            *value = 0.0;
        }
    }

    // y = B^{-T} c_B (c_b は常に dense なので btran_dense で sparse 変換を省略)
    for i in 0..m {
        y[i] = c[basis[i]];
    }
    basis_mgr.btran_dense(y);
    Ok(())
}

/// 最適基底解から元の変数への解ベクトルを復元する
///
/// 標準形の最適解を、変数変換（オフセット・係数）を逆適用して
/// 元問題の変数値に変換する。
/// `col_scale` はRuizスケーリングの列スケール因子。スケーリングを行わない場合は空スライスを渡す。
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

// --- Core Revised Simplex ---

/// 改訂シンプレックス法のコアアルゴリズム
///
/// LU分解を用いて基底行列を管理し、以下の手順を繰り返す:
///
/// 1. **BTRAN**: 双対変数 `y = B^{-T} c_B` を計算
/// 2. **価格付け（Pricing）**: PricingStrategyにより入基変数を選択
/// 3. **FTRAN**: ピボット列 `d = B^{-1} a_j` を計算
/// 4. **比率テスト**: Bland則を用いて離基変数を選択（退化サイクルの防止）
/// 5. **基底更新**: `x_B` の更新と基底行列のrank-1更新
///
/// # 引数
///
/// * `a` - 制約行列（CSC疎行列形式）
/// * `x_b` - 現在の基底解ベクトル（更新される）
/// * `c` - 目的関数係数ベクトル
/// * `basis` - 基底変数インデックスリスト（更新される）
/// * `m` - 制約数
/// * `n_cols` - 全列数
/// * `n_price` - 価格付けの対象列数（人工変数を除く場合に `n_total` を指定）
/// * `pricing` - 価格付け戦略（DantzigPricing または SteepestEdgePricing）
/// * `options` - ソルバー設定（反復上限・eta 保持数・クランプ閾値を含む）
///
/// # 戻り値
///
/// [`SimplexOutcome::Optimal`] — 最適目的関数値と双対変数、または [`SimplexOutcome::Unbounded`]
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
    let max_iter = usize::MAX; // timeout が実質的なガード（max_iterations廃止）
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj); // DeadlineExceeded 等
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // Pre-allocate reusable buffers to avoid heap allocation inside the iteration loop
    let mut c_b = vec![0.0f64; m];
    let mut y_dense = vec![0.0f64; m];
    let mut d_dense = vec![0.0f64; m];
    // Pre-allocate RC vector (reused each iteration) for pricing strategy
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
        // タイムアウト・キャンセルチェック
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }

        // 1. Dual variables: y = BTRAN(c_B)
        // c_b は常に dense なので btran_dense で sparse 変換を省略する
        for i in 0..m {
            c_b[i] = c[basis[i]];
        }
        y_dense.copy_from_slice(&c_b);
        basis_mgr.btran_dense(&mut y_dense);
        let y = &y_dense;

        // 2. Compute reduced costs for all pricing candidates
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

        // 3. Select entering variable via pricing strategy
        let entering_col = match pricing.select_entering(&rc_vec, n_price) {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Optimal(obj, y_dense.clone());
            }
            Some(j) => j,
        };

        // 4. FTRAN: pivot column d = B^{-1} * a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        // 元の列 inf-ノルムを保存（borrow は d_sv の .to_vec() で終了）
        let orig_col_norm = col_vals.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        let mut d_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        d_sv.to_dense_into(&mut d_dense);

        // 4b. FTRAN 安定性チェック: eta 蓄積による数値誤差が d を汚染していないか確認する。
        //     汚染シグナル: d 値が inf/NaN、または d の最大値が元の列ノルムの 1e12 倍超。
        //     汚染検出時は即時再因子分解でリセット後に d を再計算する。
        {
            let d_max_abs = d_dense.iter().cloned().fold(0.0f64, |acc, v| {
                if v.is_finite() { acc.max(v.abs()) } else { f64::INFINITY }
            });
            // 期待される d のノルムは原則として原列ノルムの数倍以内。
            // 1e12 倍超は eta 蓄積による数値爆発とみなす（または inf/NaN）。
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
                // 新鮮な LU で d を再計算
                let (cr2, cv2) = a.get_column(entering_col).unwrap();
                d_sv = SparseVec { indices: cr2.to_vec(), values: cv2.to_vec(), len: m };
                basis_mgr.ftran(&mut d_sv);
                d_sv.to_dense_into(&mut d_dense);
                // force_refactor が成功 → snapshot 更新
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

        // 6. Update x_b
        let step = x_b[leaving_row] / d[leaving_row];
        for i in 0..m {
            x_b[i] -= d[i] * step;
        }
        x_b[leaving_row] = step;

        // Clamp near-zero to prevent drift
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }
        // 7. Get leaving column index before updating basis
        let leaving_col = basis[leaving_row];

        // 8. Update pricing weights
        pricing.update_weights(&basis_mgr, entering_col, leaving_col, d);

        // 9. Update basis tracking
        is_basic[leaving_col] = false;
        is_basic[entering_col] = true;

        // 10. Update basis index and check pivot stability
        basis[leaving_row] = entering_col;

        // A relatively small pivot blows up the eta inverse-pivot factor and
        // contaminates subsequent FTRAN/BTRAN; refactor instead of accumulating
        // another eta. `max_d_abs` is already computed for the ratio test.
        let pivot_unstable = d[leaving_row].abs() < PIVOT_STABILITY_THRESHOLD * max_d_abs
            && basis_mgr.eta_count() > 0;

        if pivot_unstable {
            // 不安定ピボット: eta を追加せず直ちに再因子分解（新 basis で LU をリセット）
            basis_mgr.force_refactor_timed(a, basis, options.deadline);
        } else {
            // 通常ピボット: eta で逐次更新
            basis_mgr.update(entering_col, leaving_row, &d_sv);

            // 11. Refactor if needed
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
        }

        if basis_mgr.refactor_failed {
            if basis_mgr.singular_basis {
                blocked_at_basis.insert(entering_col);
                consecutive_blocks += 1;

                if consecutive_blocks > max_consecutive_blocks {
                    // No stable pivot from `basis_snapshot` after m attempts.
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

        // After a fresh LU has accepted the current basis, snapshot it and
        // clear the per-snapshot blocklist; entries that recreated singularity
        // earlier may now be safe.
        if basis_mgr.eta_count() == 0 {
            basis_snapshot.copy_from_slice(basis);
            if !blocked_at_basis.is_empty() {
                blocked_at_basis.clear();
                consecutive_blocks = 0;
            }
        }
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    // Unreachable with max_iter = usize::MAX (timeout is the real guard).
    SimplexOutcome::Timeout(obj)
}

/// Restore `basis_snapshot` and rebuild `x_b = B^{-1} b` from a fresh LU.
///
/// Returns `false` only if the snapshot itself factors as singular — which
/// implies the basis became singular before any fresh LU could accept it.
/// Caller treats `false` as fatal SingularBasis.
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
            // Recomputing x_B avoids the eta-induced drift that, if carried in
            // a stale snapshot, can leave a slack at a negative (infeasible) value.
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
