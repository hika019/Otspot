//! Dual Simplex 法（dual.rs 拡張版）
//!
//! 既存dual.rs（warm-start基盤）を拡張し、Harris ratio test、
//! Dual Steepest Edge、Big-M Phase Iを備えたDual Simplexを提供する。
//!

use super::dual_common::outcome_to_result;
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving};
use super::{build_bounded_standard_form, scale_upper_bounds, BoundedStandardForm};
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::basis::{BasisManager, LuBasis};
use crate::options::{DualPricing, SolverOptions, WarmStartBasis};
use crate::presolve::LpEquilibration;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::SparseVec;
use bounded_core::{
    extract_dual_info_bounded, extract_solution_bounded, iterate as bounded_iterate,
    phase2_primal_bounded, solve_bounded_dual, BoundedDualState, BoundedOutcome,
};

pub mod bound_flip;
mod bounded_core;
mod core;
mod phase1;
pub mod ratio_test;
mod steepest_edge;

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
    // Bounded path: problems with finite upper bounds use BFRT-aware iteration.
    // Gate: Le-only (num_artificial == 0) and dispatch not disabled by test hook.
    if !bounded_dispatch_disabled() && problem.bounds.iter().any(|&(_, ub)| ub.is_finite()) {
        let bsf = build_bounded_standard_form(problem);
        if bsf.num_artificial == 0 {
            if let Some(result) = try_bounded(&bsf, problem, options) {
                return result;
            }
            // UbViolationOutOfScope → fall through to legacy path
        }
    }

    let m = sf.m;
    let (a, b, c, row_scale, col_scale) = LpEquilibration::scale(&sf.a, &sf.b, &sf.c);

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

    // Cold-start with Ge/Eq constraints: run Primal first with the *full*
    // user budget; only fall back to Big-M Phase I if Primal had no
    // feasible incumbent (Phase I cycled on infeasibility — klein3 case).
    // When Primal returned with a non-empty solution the LP is feasible,
    // so Big-M's "Timeout + artificials left → Infeasible" heuristic would
    // wrongly flip the verdict (observed on d6cube, pds-10).
    //
    // `revised_simplex_core` has a no-progress early-bail so a
    // Primal Phase I cycle returns Timeout in O(K) pivots, leaving the
    // remaining budget to Big-M. A defensive half-deadline split previously
    // stacked with `phase1::big_m_cold_start`'s own inner split, producing
    // wall ≈ 0.75 × user_budget for slow-but-progressing LPs
    // (neos / rail2586 / rail4284). Removed — slow Primal now honors
    // the full budget and returns its incumbent, cycling Primal still bails
    // quickly via the early-bail.
    let primal_result = super::dual::two_phase_dual_simplex(sf, problem, options);
    match primal_result.status {
        SolveStatus::Timeout if primal_result.solution.is_empty() => {
            let bigm_result =
                phase1::big_m_cold_start(sf, problem, options, &a, &b, &c, &row_scale, &col_scale);
            if bigm_result.status == SolveStatus::Timeout {
                // Both phases timed out: sum iterations for observability.
                let mut r = primal_result;
                r.iterations = r.iterations.saturating_add(bigm_result.iterations);
                r
            } else {
                bigm_result
            }
        }
        // Primal returned a Farkas-certified Infeasible (dual_solution is the ray).
        // True infeasible LPs (galenet/ex72a/forest6) provide a valid Farkas proof
        // at the final Phase I basis; no Big-M re-verification needed.
        SolveStatus::Infeasible if !primal_result.dual_solution.is_empty() => primal_result,
        SolveStatus::Infeasible => {
            // Uncertified Infeasible: primal Phase I could not produce a Farkas proof.
            // pilot87-class: feasible LP cycling in Phase I → Big-M is the arbiter.
            //   - Big-M Optimal/feasible → pilot87-class false-Infeasible resolved
            //   - Big-M Infeasible (certified via Farkas) → true infeasible confirmed
            //   - Big-M Timeout → inconclusive; return Timeout, not the unverified Infeasible
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
    let (a, b, c, row_scale, col_scale) = LpEquilibration::scale(&bsf.a, &bsf.b, &bsf.c);
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
        &c,
        &row_scale,
        &col_scale,
        &ubs,
        problem,
        options,
        &mut total_iters,
    )
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
            let (p2_out, p2_state) =
                phase2_primal_bounded(bsf, dual_state, a, c, options, total_iters, ubs);
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
    use crate::options::SolverOptions;
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
}
