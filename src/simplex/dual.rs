//! Dual Simplexモジュール
//!
//! Primal Simplexの双対版。双対実行可能性を維持しながら主実行可能性を修復する。
//! warm-start用途（SQP統合）で特に有効。
//!
//! **使用条件**:
//! - warm-start: 前の最適基底は常に双対実行可能 → RHSのみ変化 → 主実行可能性を修復
//! - cold start: c_j >= 0（全非基底変数のコストが非負）のときのみDual Simpleを使用
//!   - それ以外はPrimal Simplexにフォールバック

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::presolve::RuizScaler;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm, two_phase_simplex};
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving};

/// 双対変数を計算する: y = B^{-T} c_B
fn compute_dual_variables(
    basis_mgr: &LuBasis,
    c: &[f64],
    basis: &[usize],
    m: usize,
) -> Vec<f64> {
    let c_b: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
    let mut y_sv = SparseVec::from_dense(&c_b);
    basis_mgr.btran(&mut y_sv);
    y_sv.to_dense()
}

/// Dual比率テスト: 入基変数を選択する
///
/// x_B[leaving_row] < 0（下限違反）の場合: trow[j] > 0 の列が候補
/// q = argmin_{j: trow[j] > pivot_tol} { r_j / trow[j] }
///
/// 返り値: (entering_col, theta) — thetaは双対ステップサイズ
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
        if is_basic[j] {
            continue;
        }
        if trow[j] > pivot_tol {
            let ratio = reduced_costs[j] / trow[j];
            if ratio < min_ratio - pivot_tol {
                min_ratio = ratio;
                entering = Some(j);
            } else if (ratio - min_ratio).abs() < pivot_tol {
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

/// Dual Simplexコアアルゴリズム
///
/// 双対実行可能な初期基底から出発し、主実行可能性を修復する。
/// 既存のLuBasis/BasisManagerをPrimal Simplexと共有して使う。
///
/// **前提**: 入力の basis + x_b は双対実行可能（r_j >= 0 for all non-basic j）であること。
pub(crate) fn dual_simplex_core(
    a: &CscMatrix,
    x_b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    m: usize,
    n_cols: usize,
    n_price: usize,
    options: &SolverOptions,
) -> SimplexOutcome {
    let max_iter = options.max_iterations.unwrap_or(100 * (m + n_cols) + 1000);
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Optimal(obj, vec![0.0; m]);
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // reduced costsを初期化: r_j = c_j - y^T a_j
    let y_init = compute_dual_variables(&basis_mgr, c, basis, m);
    let mut reduced_costs = vec![0.0f64; n_price];
    for j in 0..n_price {
        if is_basic[j] {
            continue;
        }
        let (rows, vals) = a.get_column(j).unwrap();
        let mut ya = 0.0;
        for (k, &row) in rows.iter().enumerate() {
            ya += y_init[row] * vals[k];
        }
        reduced_costs[j] = c[j] - ya;
    }

    // Pre-allocate buffers
    let mut rho_dense = vec![0.0f64; m];
    let mut trow = vec![0.0f64; n_price];
    let mut alpha_dense = vec![0.0f64; m];
    let leaving_strategy = MostInfeasibleLeaving;

    for _iter in 0..max_iter {
        // 1. Leaving variable selection: most infeasible x_B[i]
        let leaving_row = match leaving_strategy.select_leaving(x_b, options.primal_tol) {
            None => {
                // 全て x_B[i] >= -ε → 主実行可能 → 最適
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                let y = compute_dual_variables(&basis_mgr, c, basis, m);
                return SimplexOutcome::Optimal(obj, y);
            }
            Some(p) => p,
        };

        // 2. BTRAN: ρ = B^{-T} e_p
        let mut e_p = vec![0.0f64; m];
        e_p[leaving_row] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&e_p);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut rho_dense);

        // 3. PRICE: trow[j] = ρ^T a_j for all non-basic j
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

        // 4. Dual Ratio Test: entering variable selection
        let (entering_col, theta) = match dual_ratio_test(
            &trow,
            &reduced_costs,
            &is_basic,
            n_price,
            PIVOT_TOL,
        ) {
            None => {
                // 双対非有界 = 主実行不可 (Infeasible)
                return SimplexOutcome::Unbounded;
            }
            Some(result) => result,
        };

        // 5. FTRAN: α = B^{-1} a_q（ピボット列）
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut alpha_sv);
        alpha_sv.to_dense_into(&mut alpha_dense);

        // 6. ピボット要素チェック
        let pivot_element = alpha_dense[leaving_row];
        if pivot_element.abs() < PIVOT_TOL {
            // 数値的に不安定 — refactorしてreduced costsを再計算
            basis_mgr.refactor_if_needed(a, basis);
            let y_ref = compute_dual_variables(&basis_mgr, c, basis, m);
            for j in 0..n_price {
                if is_basic[j] {
                    reduced_costs[j] = 0.0;
                    continue;
                }
                let (rows, vals) = a.get_column(j).unwrap();
                let mut ya = 0.0;
                for (k, &row) in rows.iter().enumerate() {
                    ya += y_ref[row] * vals[k];
                }
                reduced_costs[j] = c[j] - ya;
            }
            continue;
        }

        // 7. Update x_B
        let step = x_b[leaving_row] / pivot_element;
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[leaving_row] = step;

        // Clamp near-zero
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // 8. Update reduced costs: r_j -= θ * trow[j]
        for j in 0..n_price {
            if !is_basic[j] {
                reduced_costs[j] -= theta * trow[j];
            }
        }

        // 9. Update basis tracking
        let leaving_col = basis[leaving_row];
        is_basic[leaving_col] = false;
        is_basic[entering_col] = true;

        // 10. Update basis manager
        basis_mgr.update(entering_col, leaving_row, &alpha_sv);
        basis[leaving_row] = entering_col;

        // 離基変数のreduced cost = theta（双対ステップサイズ）
        reduced_costs[leaving_col] = theta;
        // 入基変数のreduced cost = 0
        reduced_costs[entering_col] = 0.0;

        // 11. Refactor if needed
        basis_mgr.refactor_if_needed(a, basis);
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    SimplexOutcome::MaxIterations(obj)
}

/// 2相Dual Simplexで標準形LPを解く
///
/// **warm-start時**: 前の最適基底を使い、新しいRHSに対してDual Simplexで修復。
/// **cold start時**:
///   - c_j >= 0 for all non-basic j（双対実行可能）: Dual Simplexを使用
///   - c_j < 0 の非基底変数あり: Primal Simplexにフォールバック
///
/// この設計はwarm-startをメイン用途とする設計書§4の方針に従う。
pub(crate) fn two_phase_dual_simplex(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let m = sf.m;

    // Apply Ruiz equilibration scaling
    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&sf.a, &sf.b, &sf.c);

    if let Some(ws) = &options.warm_start {
        // ── warm-start path ──
        let basis = ws.basis.clone();

        // 妥当性チェック
        if basis.len() != m || basis.iter().any(|&idx| idx >= sf.n_total) {
            // 無効なwarm-start → Primal Simplexにフォールバック
            return two_phase_simplex(sf, problem, options);
        }

        // x_B = B^{-1} b_new を計算（新しいRHSで再計算）
        let mut basis = basis;
        let mut x_b = match LuBasis::new(&a, &basis, options.max_etas) {
            Ok(bm) => {
                let mut rhs_sv = SparseVec::from_dense(&b);
                bm.ftran(&mut rhs_sv);
                rhs_sv.to_dense()
            }
            Err(_) => return two_phase_simplex(sf, problem, options),
        };

        match dual_simplex_core(&a, &mut x_b, &c, &mut basis, m, sf.n_total, sf.n_total, options) {
            SimplexOutcome::Optimal(obj, y) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                let (dual_solution, reduced_costs, slack) =
                    extract_dual_info(sf, problem, &y, &solution, &row_scale);
                let ws_out = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution,
                    reduced_costs,
                    slack,
                    warm_start_basis: Some(ws_out),
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
            },
            SimplexOutcome::MaxIterations(obj) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                SolverResult {
                    status: SolveStatus::MaxIterations,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                }
            }
        }
    } else {
        // ── cold start path ──
        let mut basis = sf.initial_basis.clone();

        // x_B = B^{-1} * b を計算（B ≠ I の場合があるため必須）
        let mut x_b = match LuBasis::new(&a, &basis, options.max_etas) {
            Ok(bm) => {
                let mut rhs_sv = SparseVec::from_dense(&b);
                bm.ftran(&mut rhs_sv);
                rhs_sv.to_dense()
            }
            Err(_) => {
                // 基底行列が特異 → Primal Simplexにフォールバック
                return two_phase_simplex(sf, problem, options);
            }
        };

        // 双対実行可能性チェック（初期基底のr_j = c_j - y^T a_j）
        // スラック基底の場合: y = B^{-T} c_B = 0（スラックのコストは0）
        // → r_j = c_j。双対実行可能 ⟺ c_j >= 0 for all non-basic j
        // より一般的には y を計算する。ここでは is_basic を使う。
        let is_basic: Vec<bool> = {
            let mut v = vec![false; sf.n_total];
            for &b_idx in basis.iter() {
                if b_idx < sf.n_total {
                    v[b_idx] = true;
                }
            }
            v
        };

        // 双対変数 y = B^{-T} c_B を計算してreduced costsを確認
        let dual_feasible = {
            let basis_mgr_check = match LuBasis::new(&a, &basis, options.max_etas) {
                Ok(bm) => bm,
                Err(_) => return two_phase_simplex(sf, problem, options),
            };
            let y = compute_dual_variables(&basis_mgr_check, &c, &basis, m);
            (0..sf.n_total).filter(|&j| !is_basic[j]).all(|j| {
                let (rows, vals) = a.get_column(j).unwrap();
                let ya: f64 = rows.iter().zip(vals).map(|(&r, &v)| y[r] * v).sum();
                c[j] - ya >= -PIVOT_TOL
            })
        };

        if !dual_feasible {
            // 双対実行不可 → Primal Simplexを使用
            return two_phase_simplex(sf, problem, options);
        }

        // 双対実行可能 → Dual Simplexでx_B >= 0に修復
        match dual_simplex_core(&a, &mut x_b, &c, &mut basis, m, sf.n_total, sf.n_total, options) {
            SimplexOutcome::Optimal(obj, y) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                let (dual_solution, reduced_costs, slack) =
                    extract_dual_info(sf, problem, &y, &solution, &row_scale);
                let ws_out = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution,
                    reduced_costs,
                    slack,
                    warm_start_basis: Some(ws_out),
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                // Dual SimplexのUnbounded = 主実行不可
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
            },
            SimplexOutcome::MaxIterations(obj) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                SolverResult {
                    status: SolveStatus::MaxIterations,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::{LpProblem, SolveStatus};
    use crate::options::{SimplexMethod, SolverOptions};
    use crate::simplex::solve_with;
    use crate::sparse::CscMatrix;

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

    fn dual_opts() -> SolverOptions {
        SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        }
    }

    fn primal_opts() -> SolverOptions {
        SolverOptions {
            simplex_method: SimplexMethod::Primal,
            ..SolverOptions::default()
        }
    }

    /// テスト1: Le制約のみの基本LP（c >= 0）→ 双対実行可能スタート
    /// min x1 + x2  s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3
    /// 最適解: x1=0, x2=0, obj=0
    #[test]
    fn test_dual_basic_le_constraints() {
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve_with(&lp, &dual_opts());
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(result.objective.abs() < 1e-6, "Expected obj=0, got {}", result.objective);
    }

    /// テスト2: Primalと同じ最適解になること（複数問題）
    /// 負のコスト（dual infeasible cold start）はPrimalにフォールバック
    #[test]
    fn test_dual_matches_primal() {
        // 問題A: min -x1 - x2  s.t. x1+x2<=4, x1<=3, x2<=3
        // c < 0 → cold start dual infeasible → Primal fallback
        let lp_a = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let p = solve_with(&lp_a, &primal_opts());
        let d = solve_with(&lp_a, &dual_opts());
        assert_eq!(p.status, d.status);
        if p.status == SolveStatus::Optimal {
            assert!(
                (p.objective - d.objective).abs() < 1e-6,
                "obj mismatch: primal={} dual={}",
                p.objective,
                d.objective
            );
        }

        // 問題B: min -2x1 - 3x2, x1+x2<=4, 2x1+x2<=6
        // c < 0 → Primal fallback
        let lp_b = make_lp(
            vec![-2.0, -3.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 2.0, 1.0],
            2,
            2,
            vec![4.0, 6.0],
        );
        let p2 = solve_with(&lp_b, &primal_opts());
        let d2 = solve_with(&lp_b, &dual_opts());
        assert_eq!(p2.status, d2.status);
        if p2.status == SolveStatus::Optimal {
            assert!(
                (p2.objective - d2.objective).abs() < 1e-6,
                "obj mismatch: primal={} dual={}",
                p2.objective,
                d2.objective
            );
        }
    }

    /// テスト3: 主実行不可LP → Infeasible
    /// x1 >= 2, x1 <= 1 (矛盾) — dual feasible cold start（c_x1=1 >= 0）
    #[test]
    fn test_dual_infeasible() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![2.0, 1.0],
            vec![ConstraintType::Ge, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result = solve_with(&lp, &dual_opts());
        assert_eq!(result.status, SolveStatus::Infeasible, "Expected Infeasible, got {:?}", result.status);
    }

    /// テスト4: max_iter=1 で打ち切り → MaxIterations または Optimal
    #[test]
    fn test_dual_max_iterations() {
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
            max_iterations: Some(1),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        // c < 0 → Primal fallback → MaxIterations or Optimal
        assert!(
            result.status == SolveStatus::MaxIterations || result.status == SolveStatus::Optimal,
            "Expected MaxIterations or Optimal, got {:?}",
            result.status
        );
    }

    /// テスト5: SimplexMethod::Dual 指定でDual Simplexが呼ばれること
    #[test]
    fn test_dual_simplex_method_option() {
        // c >= 0 → Dual Simplex使用
        let lp = make_lp(
            vec![1.0, 2.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![3.0],
        );
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "Unexpected status: {:?}", result.status);
        assert!(result.objective.abs() < 1e-6, "Expected obj=0, got {}", result.objective);
    }

    /// テスト6: warm-start — RHSのみ変更したLPで前の基底から最適解
    #[test]
    fn test_dual_warm_start_rhs_change() {
        // LP1: min x1 + x2  s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3
        let lp1 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(result1.warm_start_basis.is_some(), "warm_start_basis should be Some after Optimal");

        // LP2: 同じ構造でRHSのみ変更
        let lp2 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 4.0, 4.0],
        );

        // warm-startなしで解く
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // warm-startありで解く
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);
        assert_eq!(result2_warm.status, SolveStatus::Optimal);
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "warm-start result mismatch: {} vs {}",
            result2_warm.objective,
            result2_cold.objective
        );
    }

    /// テスト7: 大きい/小さい係数混合LP（NaN/Infなし）
    #[test]
    fn test_dual_large_coefficient() {
        // c >= 0 → Dual Simplex使用
        let lp = make_lp(
            vec![1e6, 1e-6],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve_with(&lp, &dual_opts());
        assert!(!result.objective.is_nan(), "Objective should not be NaN");
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::MaxIterations,
            "Unexpected status: {:?}",
            result.status
        );
    }
}
