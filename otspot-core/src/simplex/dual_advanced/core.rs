//! Dual Simplexコアループ（強化版）
//!
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
use super::super::dual_common::{basic_obj, compute_dual_vars, compute_reduced_costs, recompute_gamma_truth};
use super::super::pricing::DualLeavingStrategy;
use std::sync::atomic::Ordering;

/// No-progress 判定で Bland's rule に切り替える反復数:
///   `K = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN)`
/// m 個の basis 変数を全置換するのに最低 m iter 必要、3 倍は実用的安全マージン
/// (典型的 cycle 長は m 以下)。下限 100 は m が小さい問題の degenerate
/// な揺り戻しを許容する床。
const NO_PROGRESS_TRIGGER_FACTOR: usize = 3;
const NO_PROGRESS_MIN: usize = 100;

/// Bland mode hard iteration cap factor: after `BLAND_ITER_CAP_FACTOR * n_price`
/// iterations in Bland mode, bail with Timeout so the caller can run a Farkas
/// infeasibility check rather than cycling indefinitely (e.g. klein3 class).
const BLAND_ITER_CAP_FACTOR: usize = 10;

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

/// reduced_costs (non-basic only) と、オプションで x_b に lex 摂動を加える。
///
/// `perturb_x = true` は bland_mode 初回エントリ時のみ指定する。refactor 後の
/// 再注入では `false` を渡し x_b 摂動をスキップする。x_b は refactor で
/// リセットされないため、毎回加算すると正帰還で発散する (B4 バグ)。
/// reduced_costs は refactor で新計算されるため毎回再注入が必要。
fn apply_lex_perturbation(
    reduced_costs: &mut [f64],
    is_basic: &[bool],
    x_b: &mut [f64],
    m: usize,
    perturb_x: bool,
) {
    let scale_r = reduced_costs
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let base_r = LEX_PERTURB_REL * scale_r;
    let n_price = reduced_costs.len();
    for (j, slot) in reduced_costs.iter_mut().enumerate() {
        if !is_basic[j] {
            *slot += base_r * (1.0 + (j as f64) / (n_price as f64));
        }
    }
    if perturb_x {
        let scale_x = x_b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        let base_x = LEX_PERTURB_REL * scale_x;
        for (i, slot) in x_b.iter_mut().enumerate() {
            *slot += base_x * (1.0 + (i as f64) / (m as f64));
        }
    }
}

/// Dual Simplexコアループ（強化版）
///
/// Dual Simplex コアループを実装する。
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
    let mut bland_start_iter: usize = 0;

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

        // Bland mode hard cap: if we have iterated > BLAND_ITER_CAP_FACTOR * n_price
        // iterations in Bland mode, bail so the caller can run Farkas infeasibility
        // check (catches klein3-class cycling that produces corrupt basis state).
        if bland_mode
            && *iter_count_out - bland_start_iter > BLAND_ITER_CAP_FACTOR * n_price
        {
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

        // 3d': lb-violation 方向補正。
        //
        // 通常の双対 simplex 比率テストは trow[j] > 0 を入基候補とする (ub 違反方向)。
        // ウォームスタート時の lb 違反 (x_b[r] < 0) を修復するには入基変数の「離基行への
        // 影響が x_b[r] を増加させる方向」、すなわち trow[j] < 0 を選ばなければならない。
        // trow を符号反転することで既存の比率テスト実装をそのまま再利用する。
        //
        // 被縮小費用更新と離基変数の r 値も整合的に符号反転する (3i 参照)。
        let lb_violation = x_b[leaving_row] < 0.0;
        if lb_violation {
            for t in trow[..n_price].iter_mut() {
                *t = -*t;
            }
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
                // rc は refactor で再計算済み; x_b は初回エントリ時に摂動済みなので再注入しない。
                apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, false);
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
        // lb-violation: 離基変数は lb で退出 → reduced cost は正 (+theta)。
        // ub-violation: 離基変数は ub で退出 → reduced cost は負 (-theta)。
        if leaving_col < n_price {
            reduced_costs[leaving_col] = if lb_violation { theta } else { -theta };
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
                // rc は再計算済み; x_b は初回エントリ時に摂動済みなので再注入しない。
                apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, false);
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
                    bland_start_iter = *iter_count_out;
                    // 初回エントリ: rc と x_b の両方を摂動する。
                    apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B4 sentinel: `apply_lex_perturbation` の `perturb_x=false` 時は x_b を変更しない。
    ///
    /// B4 バグ: 旧実装では refactor 後も x_b を毎回摂動していたため、
    /// `scale_x = max|x_b|` が摂動量に比例して増大し正帰還で発散した。
    ///
    /// 修正後: `perturb_x=false` では x_b は変更されない (rc のみ再摂動)。
    /// このテストを戻す (perturb_x パラメータ削除) と x_b が複数回摂動されて
    /// 各 assert が FAIL する (no-op FAIL)。
    ///
    /// 2 種類のデータパターン:
    ///   - Pattern A: 全変数非基底、x_b スケール = 1.0 (min floor)
    ///   - Pattern B: x_b に大きな値を含む (scale_x が自明でない)
    #[test]
    fn b4_perturb_x_false_does_not_modify_xb() {
        let is_basic = vec![false, false, false, false];

        // Pattern A: x_b = [0.5, 1.0, 0.3]
        {
            let mut rc = vec![1.0, 2.0, 3.0, 0.5];
            let mut x_b = vec![0.5_f64, 1.0, 0.3];
            let m = x_b.len();

            // 初回エントリ (perturb_x=true): x_b が変化する
            apply_lex_perturbation(&mut rc, &is_basic, &mut x_b, m, true);
            let x_b_after_entry = x_b.clone();
            assert_ne!(x_b_after_entry, vec![0.5, 1.0, 0.3], "initial entry must perturb x_b");

            // refactor 後 (perturb_x=false): x_b は変化しない
            let mut rc2 = vec![1.0, 2.0, 3.0, 0.5];
            apply_lex_perturbation(&mut rc2, &is_basic, &mut x_b, m, false);
            assert_eq!(x_b, x_b_after_entry,
                "Pattern A: x_b must not change with perturb_x=false (B4 バグ復帰で FAIL)");

            // 2 回目 refactor (perturb_x=false): 同じく変化なし
            let mut rc3 = vec![1.0, 2.0, 3.0, 0.5];
            apply_lex_perturbation(&mut rc3, &is_basic, &mut x_b, m, false);
            assert_eq!(x_b, x_b_after_entry,
                "Pattern A: x_b must remain stable across repeated perturb_x=false calls");
        }

        // Pattern B: x_b に大きな値
        {
            let mut rc = vec![100.0, 200.0, 50.0, 1.0];
            let mut x_b = vec![1000.0_f64, 500.0, 750.0];
            let m = x_b.len();

            apply_lex_perturbation(&mut rc, &is_basic, &mut x_b, m, true);
            let x_b_after_entry = x_b.clone();
            assert_ne!(x_b_after_entry, vec![1000.0, 500.0, 750.0], "initial entry must perturb x_b");

            // 3 回連続 refactor: x_b は不変
            for call_n in 0..3 {
                let mut rc_n = vec![100.0, 200.0, 50.0, 1.0];
                apply_lex_perturbation(&mut rc_n, &is_basic, &mut x_b, m, false);
                assert_eq!(x_b, x_b_after_entry,
                    "Pattern B: x_b must not grow after refactor call #{} (B4 バグ復帰で FAIL)", call_n);
            }
        }
    }
}
