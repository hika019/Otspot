//! Dual Simplex法の実装
//!
//! warm-start基盤を主目的とした双対単体法。
//! Primal Simplex（mod.rs）と基底管理（basis/）を共有する。
//!
//! # アルゴリズム概要
//!
//! Dual SimplexはPrimal Simplexと「何を維持し、何を修復するか」が逆転する:
//! - **維持**: 双対実行可能性（r_j ≥ 0）
//! - **修復**: 主実行可能性（x_B ≥ 0）
//!
//! # 主要ユースケース
//!
//! SQP等で制約RHSのみが変化する場合、前の最適基底を用いてwarm-startすることで
//! O(少数反復)で新しい最適解に到達できる。

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::presolve::RuizScaler;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use super::{StandardForm, SimplexOutcome, extract_solution, extract_dual_info, timeout_result_with_incumbent};
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving, SteepestEdgePricing};
use std::sync::atomic::Ordering;

/// Dual Simplex法の2相実装エントリポイント
///
/// - **warm-start**: 提供された基底を使用してx_Bを再計算し、Dual Simplexで修復
/// - **コールドスタート**: コスト摂動でDual実行可能性を確保後、Primal Phase IIで最適化
pub(crate) fn two_phase_dual_simplex(
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
                Ok(basis_mgr) => {
                    // x_B = B^{-1} b_new (FTRANで計算)
                    let mut x_b_sv = SparseVec::from_dense(&b);
                    basis_mgr.ftran(&mut x_b_sv);
                    let mut x_b = x_b_sv.to_dense();

                    let outcome = dual_simplex_core(
                        &a, &mut x_b, &c, &mut basis, m, sf.n_total, options,
                    );

                    // Dual SimplexではUnbounded=双対非有界=主実行不可
                    return warm_outcome_to_result(
                        outcome, sf, problem, &basis, &x_b, &col_scale, &row_scale,
                    );
                }
                Err(_) => {
                    // 基底が特異 → コールドスタートにフォールバック
                }
            }
        }
    }

    // コールドスタート
    cold_start_dual(sf, problem, options, &a, &b, &c, &row_scale, &col_scale)
}

/// コールドスタートでのDual Simplex
///
/// コスト摂動でDual実行可能性を確保し、Dual Phase I（主実行可能性の修復）後に
/// Primal Phase II（目的関数の最適化）を実行する。
#[allow(clippy::too_many_arguments)]
fn cold_start_dual(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;

    // 人工変数が必要な問題はPrimal Simplexにフォールバック
    // (Ge/Eq制約でスラック基底が特異になるため)
    if sf.num_artificial > 0 {
        return super::two_phase_simplex(sf, problem, options);
    }

    // Le-only問題: スラック基底 B=I, x_B = b ≥ 0 (標準形変換後)
    let mut basis = sf.initial_basis.clone();
    let mut x_b = b.to_vec();

    // コスト摂動: c̃_j = max(c_j, 0) → r̃_j = c̃_j ≥ 0 (双対実行可能)
    // スラック基底でy=0なのでr̃_j = c̃_j
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    // Dual Phase I: 主実行可能性を修復
    // Le-onlyでb≥0の場合、x_B=b≥0なので即座に終了（0反復）
    let phase1_outcome = dual_simplex_core(
        a, &mut x_b, &c_perturbed, &mut basis, m, sf.n_total, options,
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
            return timeout_result_with_incumbent(sf, problem, &basis, &x_b, col_scale);
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {
            // Phase I完了: x_B ≥ 0 (主実行可能)
        }
    }

    // Primal Phase II: 元のコストで最適化（主実行可能点から）
    let mut pricing = SteepestEdgePricing::new(sf.n_total);
    let phase2_outcome = super::revised_simplex_core(
        a, &mut x_b, c, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options,
    );

    // Phase IIはPrimalなのでUnbounded=主非有界
    primal_outcome_to_result(
        phase2_outcome, sf, problem, &basis, &x_b, col_scale, row_scale,
    )
}

/// Dual Simplex用のSimplexOutcome→SolverResult変換
/// (Unbounded = 双対非有界 = 主実行不可)
fn warm_outcome_to_result(
    outcome: SimplexOutcome,
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    row_scale: &[f64],
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
        SimplexOutcome::Unbounded => SolverResult {
            // 双対非有界 = 主実行不可
            status: SolveStatus::Infeasible,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        },
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

/// Primal Simplex用のSimplexOutcome→SolverResult変換
/// (Unbounded = 主非有界)
fn primal_outcome_to_result(
    outcome: SimplexOutcome,
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    row_scale: &[f64],
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

/// Dual Simplexコアアルゴリズム
///
/// 双対実行可能性（r_j ≥ 0）を維持しながら、主実行可能性（x_B ≥ 0）を反復的に修復する。
///
/// # 前提条件
/// - 呼び出し前に双対実行可能性が確保されていること（warm-startまたはコスト摂動）
///
/// # 戻り値
/// - `Optimal`: 主実行可能性達成（最適）
/// - `Unbounded`: 双対比率テストで候補なし（双対非有界 = 主実行不可）
/// - `Timeout`: refactor_failed（数値障害）またはmax_iter到達（事実上不到達）
fn dual_simplex_core(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    basis: &mut [usize],
    m: usize,
    n_price: usize,
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
            return SimplexOutcome::Timeout(obj);
        }
    };

    // 基底追跡用フラグ
    let mut is_basic = vec![false; n_price];
    for &b in basis.iter() {
        if b < n_price {
            is_basic[b] = true;
        }
    }

    // 初期被縮小費用: r_j = c_j - y^T a_j (y = B^{-T} c_B)
    let mut reduced_costs = compute_reduced_costs(a, c, &basis_mgr, &is_basic, n_price, m, basis);

    let leaving_strategy = MostInfeasibleLeaving;
    let mut rho_dense = vec![0.0f64; m];
    let mut trow = vec![0.0f64; n_price];
    let mut alpha_dense = vec![0.0f64; m];

    for _iter in 0..max_iter {
        // タイムアウト・キャンセルチェック
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }

        // Step 1: 離基変数選択 - 最も主実行不可な基底変数
        let leaving_row = match leaving_strategy.select_leaving(x_b, options.primal_tol) {
            None => {
                // 全て x_B[i] ≥ -ε → 主実行可能 → 最適
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                let y = compute_dual_vars(c, &basis_mgr, basis, m);
                return SimplexOutcome::Optimal(obj, y);
            }
            Some(p) => p,
        };

        // Step 2: BTRAN: ρ = B^{-T} e_p
        let mut e_p = vec![0.0f64; m];
        e_p[leaving_row] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&e_p);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut rho_dense);

        // Step 3: PRICE: trow[j] = ρ^T a_j (非基底列のみ)
        for j in 0..n_price {
            if is_basic[j] {
                trow[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut dot = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                dot += rho_dense[row] * vals[k];
            }
            trow[j] = dot;
        }

        // Step 4: 双対比率テスト: 入基変数選択
        let (entering_col, theta) = match dual_ratio_test(
            &trow, &reduced_costs, &is_basic, n_price, PIVOT_TOL,
        ) {
            None => {
                // 候補なし: 双対非有界 = 主実行不可
                return SimplexOutcome::Unbounded;
            }
            Some(result) => result,
        };

        // Step 5: FTRAN: α = B^{-1} a_q (入基変数のピボット列)
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut alpha_sv);
        alpha_sv.to_dense_into(&mut alpha_dense);

        // Step 6: ピボット要素の数値安定性チェック
        let pivot_element = alpha_dense[leaving_row];
        if pivot_element.abs() < PIVOT_TOL {
            // 数値的に不安定 → refactorして被縮小費用を再計算
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return SimplexOutcome::SingularBasis;
                }
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs =
                compute_reduced_costs(a, c, &basis_mgr, &is_basic, n_price, m, basis);
            continue;
        }

        // Step 7: x_Bの更新
        // step = x_B[p] / α[p] (負の値)
        let step = x_b[leaving_row] / pivot_element;
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[leaving_row] = step;

        // 微小値クランプ（数値ドリフト防止）
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // Step 8: 被縮小費用の更新
        // r_j_new = r_j - θ * trow[j] for all non-basic j
        let leaving_col = basis[leaving_row];
        for j in 0..n_price {
            if !is_basic[j] {
                reduced_costs[j] -= theta * trow[j];
                // 入基変数qのr_q_new = r_q - θ * trow[q] = 0 (数学的に保証)
            }
        }
        // 離基変数の被縮小費用: r_{leaving_col} = -θ
        // (trow[leaving_col] = 1 は数学的に保証される)
        if leaving_col < n_price {
            reduced_costs[leaving_col] = -theta;
        }

        // Step 9: 基底追跡の更新
        if leaving_col < n_price {
            is_basic[leaving_col] = false;
        }
        is_basic[entering_col] = true;

        // Step 10: 基底マネージャの更新（eta追加）
        basis_mgr.update(entering_col, leaving_row, &alpha_sv);
        basis[leaving_row] = entering_col;

        // Step 11: 必要に応じてrefactor + 被縮小費用リセット
        if basis_mgr_needs_refactor_approx(_iter) {
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return SimplexOutcome::SingularBasis;
                }
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
            // refactor後に被縮小費用を再計算（数値誤差リセット）
            reduced_costs =
                compute_reduced_costs(a, c, &basis_mgr, &is_basic, n_price, m, basis);
        }
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    // max_iter=usize::MAX のためここには事実上到達しない
    SimplexOutcome::Timeout(obj)
}

/// refactor_if_neededの呼び出し頻度を制御するヘルパー
///
/// LuBasisのneeds_refactorは内部でチェックするが、
/// compute_reduced_costsは追加コストがあるため毎反復は呼ばない。
#[inline]
fn basis_mgr_needs_refactor_approx(iter: usize) -> bool {
    // 50反復ごと、またはeta閾値到達時
    iter % 50 == 49
}

/// 初期被縮小費用を計算する
///
/// r_j = c_j - y^T a_j (y = B^{-T} c_B)
fn compute_reduced_costs(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &LuBasis,
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    basis: &[usize],
) -> Vec<f64> {
    // y = B^{-T} c_B
    let c_b: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
    let mut y_sv = SparseVec::from_dense(&c_b);
    basis_mgr.btran(&mut y_sv);
    let y = y_sv.to_dense();

    // r_j = c_j - y^T a_j
    let mut reduced_costs = vec![0.0f64; n_price];
    for j in 0..n_price {
        if is_basic[j] {
            continue;
        }
        let (rows, vals) = a.get_column(j).unwrap();
        let mut ya = 0.0;
        for (k, &row) in rows.iter().enumerate() {
            ya += y[row] * vals[k];
        }
        reduced_costs[j] = c[j] - ya;
    }
    reduced_costs
}

/// 双対変数を計算する: y = B^{-T} c_B
fn compute_dual_vars(
    c: &[f64],
    basis_mgr: &LuBasis,
    basis: &[usize],
    m: usize,
) -> Vec<f64> {
    let c_b: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
    let mut y_sv = SparseVec::from_dense(&c_b);
    basis_mgr.btran(&mut y_sv);
    y_sv.to_dense()
}

/// 双対比率テスト: 入基変数を選択する
///
/// x_B[p] < 0（下限違反）の場合: trow[j] > pivot_tol の列が候補
/// θ = min_{j: trow[j] > ε} { r_j / trow[j] }
///
/// # 戻り値
/// - `Some((entering_col, theta))`: 入基変数のインデックスと双対ステップサイズ
/// - `None`: 候補なし（双対非有界 = 主実行不可）
fn dual_ratio_test(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
    pivot_tol: f64,
) -> Option<(usize, f64)> {
    let mut min_ratio = f64::INFINITY;
    let mut entering = None;

    for j in 0..n_price {
        if is_basic[j] { continue; }

        // x_B[p] < 0 → trow[j] > 0 の列が候補
        if trow[j] > pivot_tol {
            let ratio = reduced_costs[j] / trow[j];
            if ratio < min_ratio - pivot_tol {
                min_ratio = ratio;
                entering = Some(j);
            } else if (ratio - min_ratio).abs() <= pivot_tol {
                // Bland's rule: 同率なら列インデックスが小さい方
                if let Some(prev_j) = entering {
                    if j < prev_j {
                        entering = Some(j);
                    }
                }
            }
        }
    }

    entering.map(|j| (j, min_ratio))
}

#[cfg(test)]
mod tests {
    use crate::options::{SimplexMethod, SolverOptions};
    use crate::problem::{LpProblem, SolveStatus};
    use crate::simplex::solve_with;
    use crate::sparse::CscMatrix;
    use crate::tolerances::PIVOT_TOL;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    /// Le制約のみの基本LP（c ≥ 0）: スラック基底が双対実行可能 → 即座に最適
    #[test]
    fn test_dual_basic_nonneg_cost() {
        // min x1 + 2*x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3
        // 最適解: x1=0, x2=0, obj=0 (c ≥ 0 なので原点が最適)
        let lp = make_lp(
            vec![1.0, 2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            result.objective.abs() < PIVOT_TOL,
            "Expected obj=0.0, got {}",
            result.objective
        );
    }

    /// Primal SimplexとDual Simplexが同じ最適解を返すことを検証する
    #[test]
    fn test_dual_matches_primal() {
        // min -x1 - 2*x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3
        // 最適解: x1=1, x2=3, obj=-7
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        let primal_opts = SolverOptions {
            simplex_method: SimplexMethod::Primal,
            ..SolverOptions::default()
        };
        let dual_opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };

        let result_p = solve_with(&lp, &primal_opts);
        let result_d = solve_with(&lp, &dual_opts);

        assert_eq!(result_p.status, SolveStatus::Optimal);
        assert_eq!(result_d.status, SolveStatus::Optimal);
        assert!(
            (result_p.objective - result_d.objective).abs() < 1e-6,
            "Primal obj={}, Dual obj={}",
            result_p.objective,
            result_d.objective
        );
    }

    /// warm-start: RHSのみ変更した場合に正しい最適解が得られることを検証する
    #[test]
    fn test_dual_warm_start_rhs_change() {
        // LP1: min -x1 - 2*x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3
        // 最適解: x1=1, x2=3, obj=-7
        let lp1 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(result1.warm_start_basis.is_some());

        // LP2: 同じ問題、RHSのみ変更 b=[5, 3, 3]
        // 最適解: x1=2, x2=3, obj=-8
        let lp2 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 3.0, 3.0],
        );

        // コールドスタートで解く（正解確認用）
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // warm-startで解く（Dual Simplex）
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);

        assert_eq!(result2_warm.status, SolveStatus::Optimal, "Warm start should be Optimal");
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "Warm start obj={}, Cold start obj={}",
            result2_warm.objective,
            result2_cold.objective
        );
    }

    /// SimplexMethod::Dualオプションが正しく動作することを検証
    #[test]
    fn test_dual_simplex_method_option() {
        // min -x1 - x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3
        // 最適解: x1=1, x2=3 or x1=3, x2=1, obj=-4
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < PIVOT_TOL,
            "Expected obj=-4.0, got {}",
            result.objective
        );
    }

    /// warm-start後の被縮小費用が全て非負（双対実行可能性の維持）
    #[test]
    fn test_dual_warm_start_preserves_dual_feasibility() {
        // LP1: min x1 + x2 s.t. x1+x2 ≤ 6, x1 ≤ 4, x2 ≤ 4
        // 最適解: x1=0, x2=0, obj=0 (c ≥ 0 → 原点が最適)
        let lp1 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![6.0, 4.0, 4.0],
        );

        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(result1.warm_start_basis.is_some());

        // LP2: RHSのみ変更 b=[5, 3, 3]（同じ問題構造）
        let lp2 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 3.0, 3.0],
        );

        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result2 = solve_with(&lp2, &opts_warm);

        assert_eq!(result2.status, SolveStatus::Optimal);
        // 双対変数（被縮小費用）が全て非負（最小化問題で最適性条件）
        for &rc in &result2.reduced_costs {
            assert!(
                rc >= -PIVOT_TOL,
                "Reduced cost {rc} should be ≥ 0 at optimality (dual feasibility)"
            );
        }
    }

    #[test]
    fn test_dual_simplex_timeout() {
        // コールドスタートLP、deadlineを過去に設定してTimeout確認
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// BUG-SX-002 (dual): LuBasis::new Err → 偽Optimal廃止
    /// dual_simplex_core で特異初期基底を渡し、Optimal 以外が返ることを確認。
    /// SingularBasis が発生した場合は SingularBasis を返す（Timeout ではなく）。
    #[test]
    fn test_sx002_dual_lu_basis_err_should_return_timeout() {
        use crate::simplex::dual::dual_simplex_core;
        use crate::simplex::SimplexOutcome;

        // 2×2 行列: 列0 = [1; 0]
        // basis = [0, 0] → B = [[1, 1]; [0, 0]] → rank 1 → 特異 → LuBasis::new Err
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let mut x_b = vec![1.0, 0.0];
        let mut basis = vec![0usize, 0]; // 同一列 → 特異基底
        let opts = SolverOptions::default();
        let outcome = dual_simplex_core(
            &a, &mut x_b, &c, &mut basis, 2, 2, &opts,
        );
        assert!(
            !matches!(outcome, SimplexOutcome::Optimal(..)),
            "BUG-SX-002 (dual): LuBasis::new Err 時は Optimal を返してはならない"
        );
        // 特異基底は SingularBasis または Timeout を返す（Optimal は不可）
        assert!(
            matches!(outcome, SimplexOutcome::Timeout(..) | SimplexOutcome::SingularBasis),
            "BUG-SX-002 (dual): LuBasis::new Err 時は Timeout または SingularBasis を返すべき"
        );
    }
}
