//! Dual Simplexコアループ（強化版）
//!
//! 設計書 §3.3 `dual_simplex_core_advanced()` に準拠。
//! DualLeavingStrategy / RatioTestStrategy を差し替え可能な形で実装する。
//!
//! 実装範囲:
//! - DualLeavingStrategy 経由のleaving選択 (MostInfeasibleLeaving / DualSteepestEdgeLeaving)
//! - HarrisRatioTest による ratio test
//! - LuBasis::needs_refactor() ベースの refactor判定
//! - DSE 重み更新 (3j): `leaving.needs_sigma()` が true のとき σ = B^{-1} ρ_p
//!   を計算し `leaving.after_pivot(...)` で γ を rank-1 更新

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use super::ratio_test::{RatioTestStrategy, HarrisRatioTest, bland_ratio_test};
use super::super::SimplexOutcome;
use super::super::dual_common::{basic_obj, compute_dual_vars, compute_reduced_costs};
use super::super::pricing::DualLeavingStrategy;
use std::sync::atomic::Ordering;

/// No-progress 判定で Bland's rule に切り替える反復数:
///   `K = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN)`
/// m 個の basis 変数を全置換するのに最低 m iter 必要、3 倍は実用的安全マージン
/// (典型的 cycle 長は m 以下)。下限 100 は m が小さい問題の degenerate
/// な揺り戻しを許容する床。
const NO_PROGRESS_TRIGGER_FACTOR: usize = 3;
const NO_PROGRESS_MIN: usize = 100;

/// 進歩判定の相対閾値: `best - current > best * REL_EPS` のとき改善とみなす。
/// 1e-12 は f64 の数値ノイズ (~1e-15) より十分大きく、有意な改善のみ拾う。
const NO_PROGRESS_REL_EPS: f64 = 1e-12;

// Bland's rule leaving と progress metric は `DualLeavingStrategy::bland_leaving`
// と `::progress_metric` (default + strategy override) に委譲した。

/// Lex 摂動: bland_mode 起動時に reduced_costs (non-basic) と x_b
/// (basis values) の両方に `eps * (1 + i/n) * scale` を加算し degeneracy を
/// 解消、Bland's rule の有限終了を保証する。
///
/// **reduced_costs 摂動が cycle 解消の本体**: klein3 観測の 2-cycle は entering
/// ratio test の tie (j=43 / j=63 が ほぼ同じ ratio を持ち is_basic 切替で交互
/// 選択) が原因。reduced_costs を positive 値で線形に摂動 → ratio test の tie
/// 解消 → 一意 entering。**c (objective) を直接摂動すると dual feasibility を
/// 破る** (推奨されない経路) ので reduced_costs 経路を選択。
/// **x_b 摂動は補助**: leaving の degeneracy 解消用。
///
/// 摂動は positive 加算なので `r_j > 0` を保ち dual feasibility 維持。
/// 注意: refactor 後 (`needs_refactor()` で reduced_costs 再計算) は摂動が
/// 失われるため、Bland mode 中も逐次再注入が必要 (実装側で対応)。
///
/// `LEX_PERTURB_REL = 1e-4` は reduced_costs / x_b スケールに対する相対摂動。
/// 1e-6 では cycle 行の reduced_cost diff (~1e-3 オーダー) を上回れず tie 残存。
/// 1e-3 では Phase 1 の Infeasible 判定境界に影響しうるため間を取った。
const LEX_PERTURB_REL: f64 = 1e-4;

/// γ_i = ||(B^{-1})_{i,:}||² 真値再計算 (m BTRAN). DSE warm-start init と
/// refactor 後の drift wipe で同一手順を踏むため、2 caller 重複を helper 化。
/// O(m²) cost (m BTRAN). Caller は `leaving.set_initial_gamma(...)` に渡す。
fn recompute_gamma_truth(basis_mgr: &mut LuBasis, m: usize) -> Vec<f64> {
    let mut gamma_truth = vec![0.0f64; m];
    let mut e_i = vec![0.0f64; m];
    let mut rho_i = vec![0.0f64; m];
    for i in 0..m {
        e_i.iter_mut().for_each(|v| *v = 0.0);
        e_i[i] = 1.0;
        let mut sv = SparseVec::from_dense(&e_i);
        basis_mgr.btran(&mut sv);
        sv.to_dense_into(&mut rho_i);
        gamma_truth[i] = rho_i.iter().map(|&v| v * v).sum();
    }
    gamma_truth
}

/// reduced_costs (non-basic only) と x_b に lex 摂動を加える。
fn apply_lex_perturbation(
    reduced_costs: &mut [f64],
    is_basic: &[bool],
    x_b: &mut [f64],
    m: usize,
) {
    let scale_r = reduced_costs
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let scale_x = x_b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
    let base_r = LEX_PERTURB_REL * scale_r;
    let base_x = LEX_PERTURB_REL * scale_x;
    let n_price = reduced_costs.len();
    for (j, slot) in reduced_costs.iter_mut().enumerate() {
        if !is_basic[j] {
            *slot += base_r * (1.0 + (j as f64) / (n_price as f64));
        }
    }
    for (i, slot) in x_b.iter_mut().enumerate() {
        *slot += base_x * (1.0 + (i as f64) / (m as f64));
    }
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
/// - `leaving`: 離基変数選択戦略（DualLeavingStrategy実装。`&mut` でDSE等の状態更新を許容）
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
    leaving: &mut dyn DualLeavingStrategy,
    iter_count_out: &mut usize,
) -> SimplexOutcome {
    // Step 1: LuBasis初期化
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = basic_obj(c, basis, x_b);
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
    // σ = B^{-1} ρ_p, allocated only when the strategy needs it (DSE).
    // Stateless strategies skip the extra FTRAN entirely; sigma_dense
    // stays empty and `after_pivot` ignores its argument.
    let needs_sigma = leaving.needs_sigma();
    let mut sigma_dense = if needs_sigma {
        vec![0.0f64; m]
    } else {
        Vec::new()
    };

    // DSE init: γ_i = ||(B^{-1})_{i,:}||² via m BTRANs. Required for warm-start
    // bases (B ≠ I) — γ = 1 is only correct at the identity basis, and the σ
    // cross-term in update_after_pivot would otherwise dominate and clamp γ
    // to FLOOR within a few pivots (verified on textbook degenerate LP).
    // Cost: O(m²). Acceptable: one-shot per warm-start solve.
    if needs_sigma {
        let gamma_truth = recompute_gamma_truth(&mut basis_mgr, m);
        leaving.set_initial_gamma(&gamma_truth);
    }

    // Anti-cycling state: progress_metric が K iter 改善なし → Bland fallback。
    // 一度 bland_mode に入ったら戻さない (再 cycle 防止)。progress_metric は
    // leaving strategy が提供し、auxiliary objective (Big-M Phase I の人工変数
    // 残存 etc.) も含む。global `sum_neg` だと初期 `x_B ≥ 0` の Big-M Phase I で
    // `best = 0` → threshold = 0 → 必ず no-progress と判定されて誤発火する。
    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    let mut best_infeas = leaving.progress_metric(x_b, basis);
    let mut iters_since_progress: usize = 0;
    let mut bland_mode = false;

    // Step 3: 反復ループ
    loop {
        *iter_count_out = iter_count_out.saturating_add(1);
        // 3a: タイムアウト/キャンセルチェック
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }

        // 3b: 離基変数選択
        // bland_mode では leaving.bland_leaving (strategy-aware Bland) に切り替え。
        // 通常時は leaving.select_leaving を使用。
        let leaving_pick = if bland_mode {
            leaving.bland_leaving(x_b, options.primal_tol, basis)
        } else {
            leaving.select_leaving(x_b, options.primal_tol, basis)
        };
        let leaving_row = match leaving_pick {
            None => {
                // 全て x_B[i] ≥ -ε → 主実行可能 → 最適
                let obj: f64 = basic_obj(c, basis, x_b);
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

        // 3e: ratio test → entering_col, theta
        // bland_mode では pure Bland (min ratio + smallest idx tiebreak)。
        let ratio_pick = if bland_mode {
            bland_ratio_test(&trow, &reduced_costs, &is_basic, n_price, PIVOT_TOL)
        } else {
            ratio_tester.select_entering(&trow, &reduced_costs, &is_basic, n_price)
        };
        let (entering_col, theta) = match ratio_pick {
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
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
            if bland_mode {
                apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m);
            }
            leaving.after_refactor(m);
            continue;
        }

        // 3f': DSE-only — σ = B^{-1} ρ_p (extra FTRAN for the γ rank-1 update).
        // Computed *before* the basis update so it refers to the old B^{-1}.
        if needs_sigma {
            let mut sigma_sv = SparseVec::from_dense(&rho_dense);
            basis_mgr.ftran(&mut sigma_sv);
            sigma_sv.to_dense_into(&mut sigma_dense);
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

        // 3j: DSE 重み更新 (stateless strategies は no-op)。
        // σ は needs_sigma 経路でのみ valid、stateless 側は &[] が渡る。
        leaving.after_pivot(leaving_row, &alpha_dense, &sigma_dense, pivot_element);

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
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            // refactor後は被縮小費用の数値誤差をリセット
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
            if bland_mode {
                apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m);
            }
            // Stateful weight 戦略 (DSE) は drift をここでリセット +
            // refactor 後の真の γ を m BTRAN で再計算 (initial init と同じ理由)。
            leaving.after_refactor(m);
            if needs_sigma {
                let gamma_truth = recompute_gamma_truth(&mut basis_mgr, m);
                leaving.set_initial_gamma(&gamma_truth);
            }
        }

        // 3m: 進歩観測 → no-progress なら Bland mode へ遷移
        if !bland_mode {
            let current = leaving.progress_metric(x_b, basis);
            let threshold = best_infeas * (1.0 - NO_PROGRESS_REL_EPS);
            if current < threshold {
                best_infeas = current;
                iters_since_progress = 0;
            } else {
                iters_since_progress = iters_since_progress.saturating_add(1);
                if iters_since_progress >= k_trigger {
                    bland_mode = true;
                    apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m);
                }
            }
        }
    }
}
