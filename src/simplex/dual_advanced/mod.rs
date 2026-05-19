//! 産業品質Dual Simplex法
//!
//! 既存dual.rs（warm-start基盤）を拡張し、Harris ratio test、
//! Dual Steepest Edge、Big-M Phase Iを備えた高性能Dual Simplexを提供する。
//!
//! 設計書 §3.2 に準拠。

use crate::basis::{BasisManager, LuBasis};
use crate::options::{DualPricing, SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::presolve::RuizScaler;
use crate::sparse::SparseVec;
use super::{StandardForm, SimplexOutcome, extract_solution, extract_dual_info};
use super::{build_bounded_standard_form, scale_upper_bounds, BoundedStandardForm};
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving};
use bounded_core::{
    BoundedDualState, BoundedOutcome, extract_solution_bounded, extract_dual_info_bounded,
    solve_bounded_dual, phase2_primal_bounded, iterate as bounded_iterate,
};

mod bounded_core;
mod core;
mod phase1;
pub mod ratio_test;
mod steepest_edge;
pub mod bound_flip;

/// `options.dual_pricing` から DualLeavingStrategy を組み立てる。
/// DSE 経路は m 個の重みを new() で初期化する (γ_i = 1, 識別基底想定)。
fn make_leaving_strategy(pricing: DualPricing, m: usize) -> Box<dyn DualLeavingStrategy> {
    match pricing {
        DualPricing::MostInfeasible => Box::new(MostInfeasibleLeaving),
        DualPricing::SteepestEdge => Box::new(steepest_edge::DualSteepestEdgeLeaving::new(m)),
    }
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
    if !bounded_dispatch_disabled()
        && problem.bounds.iter().any(|&(_, ub)| ub.is_finite())
    {
        let bsf = build_bounded_standard_form(problem);
        if bsf.num_artificial == 0 {
            if let Some(result) = try_bounded(&bsf, problem, options) {
                return result;
            }
            // UbViolationOutOfScope → fall through to legacy path
        }
    }

    let m = sf.m;
    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&sf.a, &sf.b, &sf.c);

    if let Some(warm) = &options.warm_start {
        // Warm start: 提供された基底でx_Bを新しいRHSから再計算
        if warm.basis.len() == m && warm.basis.iter().all(|&idx| idx < sf.n_total) {
            let mut basis = warm.basis.clone();

            match LuBasis::new(&a, &basis, options.max_etas) {
                Ok(mut basis_mgr) => {
                    // x_B = B^{-1} b_new (FTRANで計算)
                    let mut x_b_sv = SparseVec::from_dense(&b);
                    basis_mgr.ftran(&mut x_b_sv);
                    let mut x_b = x_b_sv.to_dense();

                    let mut leaving = make_leaving_strategy(options.dual_pricing, m);
                    let mut total_iters: usize = 0;
                    let outcome = core::dual_simplex_core_advanced(
                        &a, &mut x_b, &c, &mut basis, m, sf.n_total, options,
                        leaving.as_mut(),
                        &mut total_iters,
                    );

                    let mut result = outcome_to_result(
                        outcome, sf, problem, &basis, &x_b, &col_scale, &row_scale,
                        true, // dual_unbounded → Infeasible
                    );
                    result.iterations = total_iters;
                    return result;
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
            let bigm_result = phase1::big_m_cold_start(
                sf, problem, options, &a, &b, &c, &row_scale, &col_scale,
            );
            if bigm_result.status == SolveStatus::Timeout {
                // Phase Primal と Phase Big-M 両方 Timeout: 全体 iter 数 (sum) を
                // observability として保持。primal_result を base にして iter のみ加算。
                let mut r = primal_result;
                r.iterations = r.iterations.saturating_add(bigm_result.iterations);
                r
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
    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&bsf.a, &bsf.b, &bsf.c);
    let ubs = scale_upper_bounds(&bsf.upper_bounds, &col_scale);
    // total_iters is always assigned before read (warm branch overwrites before
    // passing &mut to finish_bounded; cold path overwrites before return).
    let mut total_iters: usize;

    // Warm start: reuse a previously-saved bounded-path basis when the index
    // space matches (basis.len() == bsf.m, all indices < bsf.n_total). Warm
    // starts from the legacy path have basis.len() == sf.m > bsf.m when UBs
    // are present, so they fall through to cold start automatically.
    if let Some(warm) = &options.warm_start {
        if warm.basis.len() == bsf.m
            && warm.basis.iter().all(|&idx| idx < bsf.n_total)
        {
            if let Ok(mut basis_mgr) = LuBasis::new(&a, &warm.basis, options.max_etas) {
                let mut x_b_sv = SparseVec::from_dense(&b);
                basis_mgr.ftran(&mut x_b_sv);
                let x_b = x_b_sv.to_dense();
                let mut is_basic = vec![false; bsf.n_total];
                for &j in &warm.basis {
                    is_basic[j] = true;
                }
                let at_upper = if warm.at_upper.len() == bsf.n_total {
                    warm.at_upper.clone()
                } else {
                    vec![false; bsf.n_total]
                };
                let state = BoundedDualState {
                    basis: warm.basis.clone(),
                    at_upper,
                    x_b,
                    reduced_costs: vec![0.0; bsf.n_total],
                    is_basic,
                    iterations: 0,
                };
                let (dual_out, dual_state) =
                    bounded_iterate(state, bsf, &a, &c, options, &ubs);
                total_iters = dual_state.iterations;
                let result = finish_bounded(
                    dual_out, dual_state, bsf, &a, &c, &row_scale, &col_scale, &ubs,
                    problem, options, &mut total_iters,
                );
                if result.is_some() {
                    return result;
                }
                // UbViolationOutOfScope from warm start → cold start
            }
            // Singular warm basis → cold start
        }
    }

    // Cold start.
    let (dual_out, dual_state) =
        solve_bounded_dual(bsf, &a, &b, &c, options, &ubs);
    total_iters = dual_state.iterations;
    finish_bounded(
        dual_out, dual_state, bsf, &a, &c, &row_scale, &col_scale, &ubs,
        problem, options, &mut total_iters,
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
            objective: 0.0,
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
            let (p2_out, p2_state) = phase2_primal_bounded(
                bsf, dual_state, a, c, options, total_iters, ubs,
            );
            Some(finish_bounded_phase2(p2_out, p2_state, bsf, col_scale, row_scale, problem, *total_iters))
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
                at_upper: state.at_upper,
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
        a, &mut x_b, &c_perturbed, &mut basis, m, sf.n_total, options,
        leaving.as_mut(),
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            // 双対非有界 = 主実行不可
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
        SimplexOutcome::Timeout(_) => {
            return super::timeout_result_with_incumbent(sf, problem, &basis, &x_b, col_scale, total_iters);
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
        a, &mut x_b, c, &b, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options,
        &mut total_iters, false,
    );

    // Phase 2はPrimalなのでUnbounded=主非有界
    // (result.iterations は match の後で set)
    let mut result = match phase2_outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis { basis: basis.to_vec(), x_b: x_b.to_vec(), at_upper: vec![] };
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
            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                eprintln!("[NE-TRACE] dual_advanced cold_start Phase-2 Timeout (total_iters={}, obj={:.6e})", total_iters, obj);
            }
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
        SimplexOutcome::SingularBasis => {
            if std::env::var("DUMP_NE_TRACE").ok().as_deref() == Some("1") {
                eprintln!("[NE-TRACE] dual_advanced cold_start Phase-2 SingularBasis (total_iters={})", total_iters);
            }
            SolverResult::numerical_error()
        }
    };
    result.iterations = total_iters;
    result
}

/// SimplexOutcome → SolverResult 変換
///
/// `dual_unbounded_is_infeasible`: trueの場合、Unbounded = 双対非有界 = 主実行不可
#[allow(clippy::too_many_arguments)]
fn outcome_to_result(
    outcome: SimplexOutcome,
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    row_scale: &[f64],
    dual_unbounded_is_infeasible: bool,
) -> SolverResult {
    match outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, basis, x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis { basis: basis.to_vec(), x_b: x_b.to_vec(), at_upper: vec![] };
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
        SimplexOutcome::Unbounded => {
            if dual_unbounded_is_infeasible {
                // 双対非有界 = 主実行不可
                SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            } else {
                SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            }
        }
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution(sf, basis, x_b, col_scale);
            // iterations は呼び出し側 (solve_dual_advanced) で total_iters を上書き
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
}

// ── Wiring sentinels ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::options::SolverOptions;
    use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, reset_bfrt_flip_invocations,
    };
    use crate::simplex::standard_form::build_standard_form;

    /// min -x0 - x1, x0+x1 ≤ 6, x0-x1 ≤ 2, 0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 4
    /// Known optimal: x0=4, x1=2, obj=-6.
    fn lp_2x2_boxed() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![-1.0, -1.0], a, vec![6.0, 2.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 4.0)],
            None,
        ).unwrap()
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
            vec![-1.0, -3.0], a, vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 2.0)],
            None,
        ).unwrap()
    }

    fn lp_no_ub() -> LpProblem {
        use crate::sparse::CscMatrix;
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 2.0], a, vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap()
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
        assert_eq!(result.status, SolveStatus::Optimal,
            "expected Optimal, got {:?}", result.status);
        assert!((result.objective - (-9.0)).abs() < 1e-5,
            "expected obj=-9, got {:.6e}", result.objective);
        assert!(flips > 0,
            "bfrt_wiring_flip_count_positive: flip count = 0, bounded path not exercised");
    }

    /// **No-op proof**: disabling bounded dispatch causes flip count = 0.
    /// This sentinel must FAIL whenever the bounded path is bypassed.
    #[test]
    fn bfrt_wiring_flip_count_positive_noop_proof() {
        let lp = lp_flip_trigger();
        let sf = build_standard_form(&lp);
        set_bounded_dispatch_disabled(true);
        reset_bfrt_flip_invocations();
        let result = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
        let flips_disabled = bfrt_flip_invocations();
        set_bounded_dispatch_disabled(false);
        assert_eq!(flips_disabled, 0,
            "noop proof: expected 0 flips with bounded dispatch disabled, got {flips_disabled}");
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
            assert!((r.objective - (-6.0)).abs() < 1e-5, "pattern 1 obj={}", r.objective);
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
            assert!((r.objective - (-9.0)).abs() < 1e-5, "pattern 2 obj={}", r.objective);
            assert!(flips > 0, "pattern 2: flip count = 0, bounded path not exercised");
        }
        // Pattern 3: no UBs → legacy path, no flip assertion.
        {
            let lp = lp_no_ub();
            let sf = build_standard_form(&lp);
            let r = solve_dual_advanced(&sf, &lp, &SolverOptions::default());
            assert_eq!(r.status, SolveStatus::Optimal, "pattern 3 status");
        }
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
        assert!(flips > 0,
            "warm_start_reuse cold solve: flip count = 0, bounded path not exercised");
        let ws = r1.warm_start_basis.expect("bounded path must return warm_start_basis");
        let r2 = solve_dual_advanced(
            &sf, &lp,
            &SolverOptions { warm_start: Some(ws), ..SolverOptions::default() },
        );
        assert_eq!(r2.status, SolveStatus::Optimal, "warm restart: {:?}", r2.status);
        assert!((r2.objective - r1.objective).abs() < 1e-5,
            "warm restart obj drift: {} vs {}", r2.objective, r1.objective);
    }
}
