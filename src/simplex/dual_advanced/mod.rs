//! 産業品質Dual Simplex法
//!
//! 既存dual.rs（warm-start基盤）を拡張し、Harris ratio test、
//! Dual Steepest Edge、Big-M Phase Iを備えた高性能Dual Simplexを提供する。
//!
//! 設計書 §3.2 に準拠。

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::presolve::RuizScaler;
use crate::sparse::SparseVec;
use super::{StandardForm, SimplexOutcome, extract_solution, extract_dual_info};
use super::pricing::MostInfeasibleLeaving;

mod core;
mod phase1;
pub mod ratio_test;
mod steepest_edge;
mod bound_flip;

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

                    let leaving = MostInfeasibleLeaving;
                    let mut total_iters: usize = 0;
                    let outcome = core::dual_simplex_core_advanced(
                        &a, &mut x_b, &c, &mut basis, m, sf.n_total, options, &leaving,
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

    // cold-start: Ge/Eq制約を含む問題:
    //   1. まず既存 dual::two_phase_dual_simplex (= Primal fallback) を deadline の
    //      半分で試す。多くの infeasible LP は Primal Phase I で短時間 Infeasible
    //      検出できる (klein1, klein2 など)。
    //   2. Primal が Timeout を返した場合 (klein3 のような degenerate cycling)、
    //      残り半分の deadline で Big-M Phase I (phase1.rs, task #11) に fallback。
    //      Big-M は cycling を検出して Infeasibility 推定する。
    //   3. Primal が NumericalError / Infeasible / Optimal などを返したら
    //      そのまま返却 (regression を避ける)。
    //
    // この順序により klein1/2 は従来通り高速、klein3 は Big-M で対処可能。
    let primal_options = clone_options_with_half_deadline(options);
    let primal_result = super::dual::two_phase_dual_simplex(sf, problem, &primal_options);
    match primal_result.status {
        SolveStatus::Timeout => {
            // klein3 級 cycling 疑い → Big-M Phase I で Infeasibility 推定。
            let bigm_result = phase1::big_m_cold_start(
                sf, problem, options, &a, &b, &c, &row_scale, &col_scale,
            );
            // Big-M も Timeout なら Primal の結果を返す (incumbent 保持)
            if bigm_result.status == SolveStatus::Timeout {
                primal_result
            } else {
                bigm_result
            }
        }
        _ => primal_result,
    }
}

/// SolverOptions のクローンを返し、deadline がある場合は残り時間の半分に縮める。
fn clone_options_with_half_deadline(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    if let Some(d) = options.deadline {
        let now = std::time::Instant::now();
        let remaining = d.saturating_duration_since(now);
        o.deadline = Some(now + remaining / 2);
    }
    o
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

    let leaving = MostInfeasibleLeaving;

    // Phase 1: Harris dual simplexで主実行可能性を修復
    // Le-onlyでb≥0の場合、x_B=b≥0なので即座に終了（0反復）
    let mut total_iters: usize = 0;
    let phase1_outcome = core::dual_simplex_core_advanced(
        a, &mut x_b, &c_perturbed, &mut basis, m, sf.n_total, options, &leaving,
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
            return super::timeout_result_with_incumbent(sf, problem, &basis, &x_b, col_scale);
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
        &mut total_iters,
    );

    // Phase 2はPrimalなのでUnbounded=主非有界
    // (result.iterations は match の後で set)
    let mut result = match phase2_outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis { basis: basis.to_vec(), x_b: x_b.to_vec() };
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
            let ws = WarmStartBasis { basis: basis.to_vec(), x_b: x_b.to_vec() };
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
