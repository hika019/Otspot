//! Dual Simplexコアループ（強化版）
//!
//! 設計書 §3.3 `dual_simplex_core_advanced()` に準拠。
//! DualLeavingStrategy / RatioTestStrategy を差し替え可能な形で実装する。
//!
//! Phase 2 実装範囲:
//! - MostInfeasibleLeaving によるleaving選択
//! - HarrisRatioTest による ratio test
//! - LuBasis::needs_refactor() ベースの refactor判定（50反復固定廃止）
//! - DSE重み更新（3j）: Phase 3でno-opからDSEに差替予定

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use super::ratio_test::{RatioTestStrategy, HarrisRatioTest};
use super::super::SimplexOutcome;
use super::super::pricing::DualLeavingStrategy;
use std::sync::atomic::Ordering;

/// 被縮小費用を計算する: r_j = c_j - y^T a_j（y = B^{-T} c_B）
fn compute_reduced_costs(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    basis: &[usize],
) -> Vec<f64> {
    // y = B^{-T} c_B (c_b は常に dense なので btran_dense で sparse 変換を省略)
    let mut y: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
    basis_mgr.btran_dense(&mut y);

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
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    m: usize,
) -> Vec<f64> {
    // c_b は常に dense なので btran_dense で sparse 変換を省略
    let mut y: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
    basis_mgr.btran_dense(&mut y);
    y
}

/// Dual Simplexコアループ（強化版）
///
/// 設計書 §3.3 の `dual_simplex_core_advanced()` を実装する。
///
/// # 引数
/// - `a`: 制約行列（Ruizスケーリング済み）
/// - `x_b`: 基底変数値ベクトル（mutable、反復ごとに更新）
/// - `c`: 目的関数係数（スケーリング済み）
/// - `basis`: 基底インデックス配列（mutable）
/// - `m`: 制約数
/// - `n_price`: price対象の列数
/// - `options`: ソルバーオプション
/// - `leaving`: 離基変数選択戦略（DualLeavingStrategy実装: MostInfeasibleLeaving or DualSteepestEdge）
///
/// # 戻り値
/// SimplexOutcome (Optimal/Unbounded/Timeout)
#[allow(clippy::too_many_arguments)]
pub(crate) fn dual_simplex_core_advanced(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    basis: &mut [usize],
    m: usize,
    n_price: usize,
    options: &SolverOptions,
    leaving: &dyn DualLeavingStrategy,
) -> SimplexOutcome {
    // Step 1: LuBasis初期化
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

    // Step 2: 初期被縮小費用計算: r_j = c_j - y^T a_j
    let mut reduced_costs =
        compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);

    // Harris ratio test（Phase 2ではデフォルト）
    let ratio_tester = HarrisRatioTest::new(options.dual_tol, PIVOT_TOL);

    let mut rho_dense = vec![0.0f64; m];
    let mut trow = vec![0.0f64; n_price];
    let mut alpha_dense = vec![0.0f64; m];

    // Step 3: 反復ループ
    loop {
        // 3a: タイムアウト/キャンセルチェック
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }

        // 3b: 離基変数選択（leaving.select_leaving()）
        let leaving_row = match leaving.select_leaving(x_b, options.primal_tol) {
            None => {
                // 全て x_B[i] ≥ -ε → 主実行可能 → 最適
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                let y = compute_dual_vars(c, &mut basis_mgr, basis, m);
                return SimplexOutcome::Optimal(obj, y);
            }
            Some(p) => p,
        };

        // 3c: BTRAN: ρ = B^{-T} e_p
        let mut e_p = vec![0.0f64; m];
        e_p[leaving_row] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&e_p);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut rho_dense);

        // 3d: PRICE: trow[j] = ρ^T a_j（非基底列のみ）
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

        // 3e: Harris ratio test → entering_col, theta
        let (entering_col, theta) = match ratio_tester.select_entering(
            &trow,
            &reduced_costs,
            &is_basic,
            n_price,
        ) {
            None => {
                // 候補なし: 双対非有界 = 主実行不可
                return SimplexOutcome::Unbounded;
            }
            Some(result) => result,
        };

        // 3f: FTRAN: α = B^{-1} a_q
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut alpha_sv);
        alpha_sv.to_dense_into(&mut alpha_dense);

        // 3g: ピボット要素安定性チェック（|α[p]| < pivot_tolerance → refactorまたはskip）
        let pivot_element = alpha_dense[leaving_row];
        if pivot_element.abs() < PIVOT_TOL {
            // 数値的に不安定 → refactorして被縮小費用を再計算
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
            continue;
        }

        // 3h: x_B更新 + 微小値クランプ
        let step = x_b[leaving_row] / pivot_element;
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[leaving_row] = step;
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // 3i: 被縮小費用の増分更新
        // r_j_new = r_j - θ * trow[j]（非基底変数全て）
        let leaving_col = basis[leaving_row];
        for j in 0..n_price {
            if !is_basic[j] {
                reduced_costs[j] -= theta * trow[j];
            }
        }
        // 離基変数の被縮小費用: r_{leaving_col} = -θ
        if leaving_col < n_price {
            reduced_costs[leaving_col] = -theta;
        }

        // 3j: DSE重み更新（Phase 2ではno-op）
        // TODO Phase 3: DualSteepestEdge::update_weights() をここで呼び出す

        // 基底追跡更新
        if leaving_col < n_price {
            is_basic[leaving_col] = false;
        }
        is_basic[entering_col] = true;

        // 3k: 基底更新（LuBasis::update）
        basis_mgr.update(entering_col, leaving_row, &alpha_sv);
        basis[leaving_row] = entering_col;

        // 3l: refactor判定（LuBasis::needs_refactor）+ 必要なら refactor + 被縮小費用再計算
        // needs_refactor()でeta蓄積数ベースに判定（50反復固定廃止）
        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return SimplexOutcome::SingularBasis;
                }
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
            // refactor後は被縮小費用の数値誤差をリセット
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
        }
    }
}
