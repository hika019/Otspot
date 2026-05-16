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

        match revised_simplex_core(&a, &mut x_b, &c, &b, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options)
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
                use crate::options::MAX_PHASE1_RETRIES;
                let mut phase1_feasible = false;
                'retry: for attempt in 0..=MAX_PHASE1_RETRIES {
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
                        SimplexOutcome::SingularBasis => return SolverResult::numerical_error(),
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
                            ..Default::default()
                        }
                    }
                    SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
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
                        SimplexOutcome::SingularBasis => return SolverResult::numerical_error(),
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
            SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
        }
    }
}

/// Eq 制約の feasibility を確認する（案D: defense-in-depth）
///
/// `solution` は problem の変数空間の解ベクトル。
/// Eq 制約の violation が `FEASIBILITY_TOL` を超える場合 false を返す。
/// bore3d の 100 単位違反のような明らかな誤 Optimal を検出し NumericalError として返すために使う。
fn check_eq_feasibility(problem: &LpProblem, solution: &[f64]) -> bool {
    const FEASIBILITY_TOL: f64 = 1e-4;
    let mut ax = vec![0.0f64; problem.num_constraints];
    for (j, &sj) in solution.iter().enumerate() {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * sj;
            }
        }
    }
    for ((ax_i, ct), bi) in ax.iter().zip(problem.constraint_types.iter()).zip(problem.b.iter()) {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
        };
        if violation > FEASIBILITY_TOL {
            return false;
        }
    }
    true
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

    // Build LuBasis from the Phase I final basis to enable FTRAN-based pivot selection.
    // Using FTRAN (d = B^{-1} a_j) for pivot element selection instead of the raw matrix
    // entry A[i,j] prevents ill-conditioned bases: raw entries ignore basis transformation,
    // while |d[i]| directly measures the numerical stability of the entering pivot.
    let mut basis_mgr = match LuBasis::new_timed(a_ext, basis, options.max_etas, options.deadline) {
        Ok(mgr) => mgr,
        Err(_) => return, // singular basis: leave artificials as-is
    };

    let mut is_basic = vec![false; a_ext.ncols];
    for &col in basis.iter() {
        is_basic[col] = true;
    }

    for i in 0..m {
        // deadline チェック: pds-20 等の大規模で artificial 多数の場合、
        // m_artificial × n_total 回 FTRAN で 1000 秒予算を簡単に消費する。
        // 各 i 反復先頭で deadline 検査し、超過なら未処理 artificial を残して return。
        if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        if basis[i] < sf.n_total || x_b[i].abs() >= PIVOT_TOL {
            continue;
        }

        // Find non-basic original column j maximising |d[i]| = |(B^{-1} a_j)[i]|.
        // This is the true pivot element after the current basis transformation.
        let mut best_j = None;
        let mut best_abs = PIVOT_TOL;
        let mut best_d_sv = SparseVec::new(m);

        for j in 0..sf.n_total {
            if is_basic[j] {
                continue;
            }
            // 内側ループでも 1024 回ごとに deadline 検査 (pds-20 では n_total ≈ 33,798)。
            if j & 0x3ff == 0 && options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return;
            }
            if let Ok((rows, vals)) = a_ext.get_column(j) {
                let mut col_dense = vec![0.0_f64; m];
                for (k, &row) in rows.iter().enumerate() {
                    if row < m {
                        col_dense[row] = vals[k];
                    }
                }
                let mut d_sv = SparseVec::from_dense(&col_dense);
                basis_mgr.ftran(&mut d_sv);
                let d_i = d_sv.get(i).abs();
                if d_i > best_abs {
                    best_abs = d_i;
                    best_j = Some(j);
                    best_d_sv = d_sv;
                }
            }
        }

        if let Some(j) = best_j {
            is_basic[basis[i]] = false;
            is_basic[j] = true;
            basis[i] = j;
            // Rank-1 LU update: subsequent FTRANs use the updated (post-pivot) basis.
            basis_mgr.update(j, i, &best_d_sv);
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
    _b: &[f64],
    basis: &mut [usize],
    m: usize,
    n_cols: usize,
    n_price: usize,
    pricing: &mut P,
    options: &SolverOptions,
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

    for _iter in 0..max_iter {
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
                        return SimplexOutcome::SingularBasis;
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
            }
        }
        let d = &d_dense;

        // 5. Ratio test (Bland's rule for ties, with degenerate near-zero pivot avoidance)
        //
        // Phase II: skip rows where basis[i] >= n_price (artificial variables in basis).
        //
        // Primary: Bland's rule (smallest basis index among ties) for anti-cycling.
        // Override: on a degenerate step (step≈0), if Bland's selects a row with
        //   d[row] < NEAR_ZERO_PIVOT_GUARD (near-machine-zero), prefer the Bland's-selected
        //   row among rows where d[row] >= NEAR_ZERO_PIVOT_GUARD.
        //   Degenerate steps don't change x_B so any tied leaving row is primal-feasible.
        //   This prevents near-zero eta entries (inv_pivot → huge) that corrupt FTRAN.
        //   Using an absolute threshold preserves Bland's anti-cycling guarantee for all
        //   non-near-zero pivots (a relative threshold fires too broadly and breaks Bland's).
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;
        let mut stable_leaving: Option<usize> = None;

        for i in 0..m {
            if d[i] > PIVOT_TOL {
                let ratio = x_b[i] / d[i];
                // Bland's rule (primary tie-breaking)
                if ratio < min_ratio - PIVOT_TOL {
                    min_ratio = ratio;
                    leaving = Some(i);
                } else if (ratio - min_ratio).abs() < PIVOT_TOL {
                    if let Some(prev) = leaving {
                        if basis[i] < basis[prev] {
                            leaving = Some(i);
                        }
                    }
                }
                // Track non-near-zero degenerate candidates for near-zero override.
                // Use Bland's rule (smallest basis index) to maintain anti-cycling.
                if ratio <= PIVOT_TOL && d[i] >= NEAR_ZERO_PIVOT_GUARD {
                    match stable_leaving {
                        None => stable_leaving = Some(i),
                        Some(prev) if basis[i] < basis[prev] => stable_leaving = Some(i),
                        _ => {}
                    }
                }
            }
        }

        // On a degenerate step: if Bland's selected a near-zero pivot (d < NEAR_ZERO_PIVOT_GUARD),
        // switch to a non-near-zero alternative to prevent singular eta entries.
        if let (Some(bland), Some(stable)) = (leaving, stable_leaving) {
            if min_ratio <= PIVOT_TOL && d[bland] < NEAR_ZERO_PIVOT_GUARD {
                leaving = Some(stable);
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

        // ピボット安定性チェック: |d[leaving_row]| / max(d) が閾値未満の場合、
        // eta の inv_pivot が大きくなりすぎて FTRAN/BTRAN が数値爆発する。
        // その場合は eta を追加せず即時再因子分解でリセットする（eta 蓄積誤差を防ぐ）。
        let max_d_abs = d.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
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
                return SimplexOutcome::SingularBasis;
            } else {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
        }
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    // max_iter=usize::MAX のためここには事実上到達しない
    SimplexOutcome::Timeout(obj)
}
