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

use super::super::dual_common::{
    basic_obj, compute_dual_vars, made_progress_with_floor, recompute_gamma_truth, NO_PROGRESS_MIN,
    NO_PROGRESS_TRIGGER_FACTOR,
};
use super::super::pricing::DualLeavingStrategy;
use super::super::trace::IterTrace;
use super::super::SimplexOutcome;
use super::ratio_test::{bland_ratio_test, HarrisRatioTest, RatioTestStrategy};
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Lex 摂動 (bland_mode 起動時): reduced_costs (non-basic) と x_b に
/// `eps·(1+i/n)·scale` を加算し ratio test の tie を解消、Bland's rule の有限終了
/// を保証する。reduced_costs 摂動が cycle 解消の本体 (klein3 観測 2-cycle 起因)、
/// x_b 摂動は leaving degeneracy 補助。`c` 直接摂動は dual feasibility を破るため
/// 不採用。positive 加算で `r_j > 0` を保つ。refactor 後は摂動が失われるため
/// 逐次再注入する。
///
/// `1e-4` の bench evidence: 1e-6 は cycle 行 reduced_cost diff (~1e-3) を上回れず
/// tie 残存、1e-3 は Phase 1 Infeasible 判定境界に影響する。
const LEX_PERTURB_REL: f64 = 1e-4;

#[derive(Clone, Copy, Debug, Default)]
struct LexPerturbStats {
    delta: f64,
    effect: f64,
}

#[derive(Default)]
struct BasisCycleDetector {
    seen: HashMap<Vec<usize>, usize>,
}

impl BasisCycleDetector {
    fn repeated(&mut self, iter: usize, basis: &[usize]) -> bool {
        self.seen.insert(basis.to_vec(), iter).is_some()
    }

    fn clear(&mut self) {
        self.seen.clear();
    }
}

fn collect_bland_ratio_candidates(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
    pivot_tol: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut candidates = Vec::new();
    let mut ratios = Vec::new();
    for j in 0..n_price {
        if is_basic[j] || trow[j] <= pivot_tol {
            continue;
        }
        let ratio = reduced_costs[j] / trow[j];
        if ratio >= pivot_tol {
            candidates.push(j);
            ratios.push(ratio);
        }
    }
    (candidates, ratios)
}

fn collect_harris_ratio_candidates(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
    harris_tol: f64,
    pivot_tol: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut theta_max = f64::INFINITY;
    for j in 0..n_price {
        if is_basic[j] || trow[j] <= pivot_tol {
            continue;
        }
        let relaxed_ratio = (reduced_costs[j] + harris_tol) / trow[j];
        if relaxed_ratio < theta_max {
            theta_max = relaxed_ratio;
        }
    }
    if !theta_max.is_finite() {
        return (Vec::new(), Vec::new());
    }
    let mut candidates = Vec::new();
    let mut ratios = Vec::new();
    for j in 0..n_price {
        if is_basic[j] || trow[j] <= pivot_tol {
            continue;
        }
        let ratio = reduced_costs[j] / trow[j];
        if ratio <= theta_max {
            candidates.push(j);
            ratios.push(ratio);
        }
    }
    (candidates, ratios)
}

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
) -> LexPerturbStats {
    let scale_r = reduced_costs
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let base_r = LEX_PERTURB_REL * scale_r;
    let n_price = reduced_costs.len();
    let mut max_rc_delta = 0.0_f64;
    for (j, slot) in reduced_costs.iter_mut().enumerate() {
        if !is_basic[j] {
            let delta = base_r * (1.0 + (j as f64) / (n_price as f64));
            *slot += delta;
            max_rc_delta = max_rc_delta.max(delta.abs());
        }
    }
    let mut max_x_delta = 0.0_f64;
    if perturb_x {
        let scale_x = x_b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        let base_x = LEX_PERTURB_REL * scale_x;
        for (i, slot) in x_b.iter_mut().enumerate() {
            let delta = base_x * (1.0 + (i as f64) / (m as f64));
            *slot += delta;
            max_x_delta = max_x_delta.max(delta.abs());
        }
        return LexPerturbStats {
            delta: base_r,
            effect: max_rc_delta.max(max_x_delta),
        };
    }
    LexPerturbStats {
        delta: base_r,
        effect: max_rc_delta,
    }
}

#[inline]
fn deadline_expired(deadline: Option<std::time::Instant>) -> bool {
    deadline.is_some_and(|d| std::time::Instant::now() >= d)
}

fn compute_reduced_costs_timed(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    basis: &[usize],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<f64>> {
    if deadline_expired(deadline) {
        return None;
    }
    let y = compute_dual_vars(c, basis_mgr, basis, m);
    let mut reduced_costs = vec![0.0f64; n_price];
    for j in 0..n_price {
        if deadline_expired(deadline) {
            return None;
        }
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
    Some(reduced_costs)
}

/// Dual simplex core loop (advanced variant).
///
/// - `leaving`: leaving-variable selection strategy (`&mut` to allow DSE weight updates).
/// - `n_enter`: number of leading columns eligible to *enter* the basis. Columns
///   `[n_enter, n_price)` may start basic and leave, but the ratio test never
///   re-selects them. Big-M Phase I passes `n_enter = n_total` so artificials,
///   once driven out, cannot re-enter — this is what makes artificial removal
///   monotone (each Priority-2 pivot replaces an artificial with a structural
///   column) and forbids the degenerate artificial↔artificial swap cycle
///   (nug08-3rd). Callers without artificials pass `n_enter = n_price`.
/// - `yield_on_stall`: when true, a Bland-mode no-progress stall returns `Timeout`
///   so the caller can hand off to an alternative method. ONLY the Big-M Phase I
///   caller passes `true` — its Priority-2 artificial-removal is a non-standard
///   leaving rule for which Bland gives no finite-termination guarantee, and it
///   has a primal fallback (`two_phase_dual_simplex`) to yield to. Standard
///   callers (warm-start / Le-only cold-start) pass `false`: their leaving rule
///   is classical Bland with guaranteed finite termination and they have NO
///   fallback, so yielding would turn a solvable LP into a spurious Timeout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dual_simplex_core_advanced(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    basis: &mut [usize],
    m: usize,
    n_price: usize,
    n_enter: usize,
    yield_on_stall: bool,
    options: &SolverOptions,
    leaving: &mut dyn DualLeavingStrategy,
    iter_count_out: &mut usize,
) -> SimplexOutcome {
    debug_assert!(n_enter <= n_price);
    // Step 1: LuBasis初期化
    let mut basis_mgr = match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
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
    let mut reduced_costs = match compute_reduced_costs_timed(
        a,
        c,
        &mut basis_mgr,
        &is_basic,
        n_price,
        m,
        basis,
        options.deadline,
    ) {
        Some(rc) => rc,
        None => {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }
    };

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
        match recompute_gamma_truth(
            &mut basis_mgr,
            m,
            options.deadline,
            options.cancel_flag.as_deref(),
        ) {
            None => {
                let obj = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            Some(gamma_truth) => leaving.set_initial_gamma(&gamma_truth),
        }
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
    let mut cycle_detector = BasisCycleDetector::default();
    let mut no_candidate_refreshed_basis: Option<Vec<usize>> = None;
    let mut rejected_pivot_basis: Option<Vec<usize>> = None;
    let mut rejected_pivot_row: Option<usize> = None;
    let mut rejected_entering = vec![false; n_price];
    let mut trace = IterTrace::new("dual-advanced");

    // Step 3: 反復ループ
    loop {
        *iter_count_out = iter_count_out.saturating_add(1);
        // 3a: タイムアウト/キャンセルチェック
        let timed_out = options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }

        if let Some(t) = trace.as_mut() {
            let obj = basic_obj(c, basis, x_b);
            t.log(*iter_count_out, obj, basis, bland_mode);
        }

        if !bland_mode && cycle_detector.repeated(*iter_count_out, basis) {
            bland_mode = true;
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
        // rejected_entering は (basis, leaving_row) ペアにスコープされる。
        // 違うペアに移行したらリセット。
        if rejected_pivot_basis.as_deref() != Some(&*basis)
            || rejected_pivot_row != Some(leaving_row)
        {
            rejected_pivot_basis = None;
            rejected_pivot_row = None;
            rejected_entering.fill(false);
        }
        let mut masked_is_basic: Vec<bool>;
        let price_excluded: &[bool] = if rejected_entering.iter().any(|&v| v) {
            masked_is_basic = is_basic.clone();
            for (j, blocked) in rejected_entering.iter().enumerate().take(n_enter) {
                if *blocked {
                    masked_is_basic[j] = true;
                }
            }
            &masked_is_basic
        } else {
            &is_basic
        };
        // 3c: BTRAN: ρ = B^{-T} e_p
        let mut e_p = vec![0.0f64; m];
        e_p[leaving_row] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&e_p);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut rho_dense);

        // 3d: PRICE: trow[j] = ρ^T a_j（非基底列のみ）
        for j in 0..n_price {
            if deadline_expired(options.deadline) {
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
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
        // lb 違反 (x_b[r] < 0) を修復するには入基変数の「離基行への影響が x_b[r] を
        // 増加させる方向」、すなわち trow[j] < 0 を選ぶ。trow を符号反転して既存の
        // 比率テスト実装を再利用する。被縮小費用と離基変数の r 値も整合反転する (3i)。
        //
        // 適用条件は `x_b[r] < 0 ∧ allows_lb_repair ∧ 離基変数が人工でない`:
        // - Big-M Phase I では人工駆出 (Priority 2) pivot が構造行を大きく lb 違反
        //   させる (beaconfd: x_b ≈ −9329)。これは sign-flip で正しく repair する
        //   (旧 blanket false は逆方向 pivot で 2-cycle を起こしていた)。
        // - ただし離基変数自身が人工 (`basis[r] >= n_enter`) で負値の場合は repair
        //   しない。人工の負値は sign-flip すると人工を basis に留めたまま値だけ
        //   0 へ寄せ続け、478 連続 pivot → Phase II cycle に陥る (sierra)。人工は
        //   標準方向で駆出する (n_enter により再入基も禁止済 → 単調に basis から消える)。
        //
        // Completeness trade-off (codex P2-#2): a feasible Eq system can need a
        // *valid* lb-repair on an artificial-leaving row. Suppressing it here makes
        // bare Big-M abandon such cases (e.g. `-2x+y=1, -2x+2y=3` with crash off).
        // We keep the guard — distinguishing a valid artificial lb-repair from
        // sierra's chase is fragile and a wrong split re-opens the 478-pivot cycle —
        // and rely on the primal fallback (`two_phase_dual_simplex`) to recover,
        // which it does (verified Optimal even crash-off). Default config solves it
        // directly via the crash basis. See `big_m_phase1_artificial_lb_repair_edge_*`.
        let mut lb_violation =
            x_b[leaving_row] < 0.0 && leaving.allows_lb_repair() && basis[leaving_row] < n_enter;
        let artificial_lb_violation =
            x_b[leaving_row] < 0.0 && leaving.allows_lb_repair() && basis[leaving_row] >= n_enter;
        if lb_violation {
            for t in trow[..n_price].iter_mut() {
                *t = -*t;
            }
        }

        // 3e: ratio test → entering_col, theta
        // bland_mode では pure Bland (min ratio + smallest idx tiebreak)。
        let (mut candidate_indices, mut candidate_ratios, mut ratio_pick) = if bland_mode {
            let (indices, ratios) =
                collect_bland_ratio_candidates(&trow, &reduced_costs, price_excluded, n_enter, PIVOT_TOL);
            let pick = bland_ratio_test(&trow, &reduced_costs, price_excluded, n_enter, PIVOT_TOL);
            (indices, ratios, pick)
        } else {
            let (indices, ratios) = collect_harris_ratio_candidates(
                &trow,
                &reduced_costs,
                price_excluded,
                n_enter,
                ratio_tester.harris_tol,
                ratio_tester.pivot_tol,
            );
            let pick = ratio_tester.select_entering(&trow, &reduced_costs, price_excluded, n_enter);
            (indices, ratios, pick)
        };
        if ratio_pick.is_none() && artificial_lb_violation {
            // 人工変数の lb-repair 方向で候補なし→ 標準方向で再試行。
            // 人工の負値は lb-repair 側で駆出できないケース (maros/pilot 等) があり、
            // 反転して再び ratio test することで構造列への pivot を試みる。
            for t in trow[..n_price].iter_mut() {
                *t = -*t;
            }
            lb_violation = true;
            (candidate_indices, candidate_ratios, ratio_pick) = if bland_mode {
                let (indices, ratios) = collect_bland_ratio_candidates(
                    &trow,
                    &reduced_costs,
                    price_excluded,
                    n_enter,
                    PIVOT_TOL,
                );
                let pick =
                    bland_ratio_test(&trow, &reduced_costs, price_excluded, n_enter, PIVOT_TOL);
                (indices, ratios, pick)
            } else {
                let (indices, ratios) = collect_harris_ratio_candidates(
                    &trow,
                    &reduced_costs,
                    price_excluded,
                    n_enter,
                    ratio_tester.harris_tol,
                    ratio_tester.pivot_tol,
                );
                let pick =
                    ratio_tester.select_entering(&trow, &reduced_costs, price_excluded, n_enter);
                (indices, ratios, pick)
            };
        }
        if let Some(t) = trace.as_mut() {
            t.log_ratio_test(
                &candidate_indices,
                &candidate_ratios,
                ratio_pick.map(|(j, _)| j),
                bland_mode,
            );
        }

        let (entering_col, theta) = match ratio_pick {
            None => {
                // rejected_entering が active → 全候補を除外した状態; fallback なし
                if rejected_entering.iter().any(|&v| v) {
                    let obj: f64 = basic_obj(c, basis, x_b);
                    return SimplexOutcome::Timeout(obj);
                }
                // 候補なしは dual-unbounded の証明候補だが、Bland 長走では
                // eta/rc drift が全候補を負 ratio 側へ押し出すことがある。
                // 同一基底につき一度だけ fresh LU + rc で再試行し、
                // 真の候補なしだけを Unbounded として返す。
                if no_candidate_refreshed_basis.as_deref() != Some(&*basis) {
                    no_candidate_refreshed_basis = Some(basis.to_vec());
                    basis_mgr.force_refactor_timed(a, basis, options.deadline);
                    if basis_mgr.refactor_failed {
                        if basis_mgr.singular_basis {
                            return SimplexOutcome::SingularBasis;
                        }
                        let obj: f64 = basic_obj(c, basis, x_b);
                        return SimplexOutcome::Timeout(obj);
                    }
                    reduced_costs = match compute_reduced_costs_timed(
                        a,
                        c,
                        &mut basis_mgr,
                        &is_basic,
                        n_price,
                        m,
                        basis,
                        options.deadline,
                    ) {
                        Some(rc) => rc,
                        None => {
                            let obj: f64 = basic_obj(c, basis, x_b);
                            return SimplexOutcome::Timeout(obj);
                        }
                    };
                    if bland_mode {
                        let stats =
                            apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, false);
                        if let Some(t) = trace.as_mut() {
                            t.log_lex_perturbation(stats.delta, stats.effect);
                        }
                    }
                    leaving.after_refactor(m);
                    continue;
                }
                // fresh factorization でも候補なし: 双対非有界 = 主実行不可
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
            // 数値的に不安定。refactor/recompute だけでは同じ候補を再選択して
            // period=1 停滞になるため、この (basis, leaving_row) の間だけ
            // この entering_col を候補から除外する。
            rejected_pivot_basis = Some(basis.to_vec());
            rejected_pivot_row = Some(leaving_row);
            if entering_col < rejected_entering.len() {
                rejected_entering[entering_col] = true;
            }
            basis_mgr.force_refactor_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs = match compute_reduced_costs_timed(
                a,
                c,
                &mut basis_mgr,
                &is_basic,
                n_price,
                m,
                basis,
                options.deadline,
            ) {
                Some(rc) => rc,
                None => {
                    let obj: f64 = basic_obj(c, basis, x_b);
                    return SimplexOutcome::Timeout(obj);
                }
            };
            if bland_mode {
                // rc は refactor で再計算済み; x_b は初回エントリ時に摂動済みなので再注入しない。
                let stats = apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, false);
                if let Some(t) = trace.as_mut() {
                    t.log_lex_perturbation(stats.delta, stats.effect);
                }
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
            if deadline_expired(options.deadline) {
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
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
        no_candidate_refreshed_basis = None;
        rejected_pivot_basis = None;
        rejected_pivot_row = None;
        rejected_entering.fill(false);

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
            reduced_costs = match compute_reduced_costs_timed(
                a,
                c,
                &mut basis_mgr,
                &is_basic,
                n_price,
                m,
                basis,
                options.deadline,
            ) {
                Some(rc) => rc,
                None => {
                    let obj: f64 = basic_obj(c, basis, x_b);
                    return SimplexOutcome::Timeout(obj);
                }
            };
            if bland_mode {
                // rc は再計算済み; x_b は初回エントリ時に摂動済みなので再注入しない。
                let stats = apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, false);
                if let Some(t) = trace.as_mut() {
                    t.log_lex_perturbation(stats.delta, stats.effect);
                }
            }
            leaving.after_refactor(m);
        }

        // 3m: 進歩観測 (bland_mode 中も継続)。
        // - !bland_mode で K iter 改善なし → Bland mode へ遷移 (lex 摂動 + counter リセット)。
        // - bland_mode でも K iter 改善なし:
        //   * `yield_on_stall` (Big-M のみ): Priority-2 駆出規則は Bland の有限終了
        //     保証外で、bland でも発散しうる (degen3)。Timeout を返し呼出側 primal
        //     fallback (`two_phase_dual_simplex`) に譲る。固定 cap でなく多手法 handoff。
        //   * それ以外 (warm / Le-only cold-start): leaving は古典 Bland で有限終了が
        //     保証され、fallback も無い。yield すると可解 LP を偽 Timeout 化するため
        //     yield せず Bland を継続する (counter のみリセット)。
        let current = leaving.progress_metric(x_b, basis);
        if made_progress_with_floor(best_infeas, current, 0.0) {
            best_infeas = current;
            iters_since_progress = 0;
            if !bland_mode {
                cycle_detector.clear();
            }
        } else {
            iters_since_progress = iters_since_progress.saturating_add(1);
            if iters_since_progress >= k_trigger {
                if bland_mode {
                    if yield_on_stall {
                        let obj: f64 = basic_obj(c, basis, x_b);
                        return SimplexOutcome::Timeout(obj);
                    }
                    // Standard caller: keep iterating; classical Bland terminates.
                    iters_since_progress = 0;
                } else {
                    bland_mode = true;
                    iters_since_progress = 0;
                    // 初回エントリ: rc と x_b の両方を摂動する。
                    let stats =
                        apply_lex_perturbation(&mut reduced_costs, &is_basic, x_b, m, true);
                    if let Some(t) = trace.as_mut() {
                        t.log_lex_perturbation(stats.delta, stats.effect);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::pricing::MostInfeasibleLeaving;
    use super::*;
    use crate::options::SolverOptions;
    use crate::sparse::CscMatrix;

    #[test]
    fn basis_cycle_detector_flags_repeated_basis_and_clear_resets() {
        let mut detector = BasisCycleDetector::default();
        assert!(!detector.repeated(1, &[1, 2, 3]));
        assert!(!detector.repeated(2, &[1, 3, 2]));
        assert!(
            detector.repeated(3, &[1, 2, 3]),
            "revisiting an earlier basis must trigger Bland/lex anti-cycling"
        );
        detector.clear();
        assert!(
            !detector.repeated(4, &[1, 2, 3]),
            "real progress clears the cycle window"
        );
    }

    /// Sentinel (P0 proof): lb-violation + sign-flipped ratio test None
    /// = dual-simplex infeasibility proof → must return `SimplexOutcome::Unbounded`.
    ///
    /// LP: min 0, s.t. x + s = -1, x,s ≥ 0. Infeasible (no feasible x,s ≥ 0 with x+s = -1).
    /// Warm basis {s}: x_b = [-1] → lb-violation at row 0.
    /// Sign-flip: trow[x] = -1 → no ratio-test candidate → Unbounded (= caller Infeasible).
    ///
    /// no-op proof: restoring the fb410eb fallback (trow restore + retry with ub direction)
    /// finds trow[x]=1 > 0, pivot proceeds with step = -1/1 = -1 → x_b never reaches 0,
    /// iteration cycles until the hard cap fires → Timeout, not Unbounded → test FAILS.
    #[test]
    fn warm_start_infeasible_basis_returns_unbounded_not_cycle() {
        // A = [[1, 1]] (x, s columns), b = [-1], c = [0, 0], basis = [1] (s basic).
        // B = [[1]] (s column), B^{-1} = [[1]], x_b = B^{-1} b = [-1].
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let c = vec![0.0_f64, 0.0];
        let mut basis = vec![1usize]; // s in basis
        let mut x_b = vec![-1.0_f64]; // lb-violation
        let opts = SolverOptions::default();
        let mut leaving = MostInfeasibleLeaving;
        let mut iters = 0usize;

        let outcome = dual_simplex_core_advanced(
            &a,
            &mut x_b,
            &c,
            &mut basis,
            1,
            2,
            2,
            false,
            &opts,
            &mut leaving,
            &mut iters,
        );
        assert!(
            matches!(outcome, SimplexOutcome::Unbounded),
            "warm-start lb-violation with no lb-repair candidate must yield Unbounded \
             (dual infeasibility proof); got {outcome:?}. \
             If Timeout: fallback retry was restored — the no-op proof triggered."
        );
    }

    /// LOAD-BEARING sentinel (n_enter re-entry ban). The entering ratio test must
    /// never select a column with index >= `n_enter`. This is the mechanism that
    /// stops the Big-M Phase I artificial↔artificial swap cycle (nug08-3rd): once
    /// an artificial leaves it cannot re-enter, so each Priority-2 pivot replaces
    /// it with a structural column (monotone).
    ///
    /// Construction (1 row, 4 cols, a = [1,1,1,-1], basis = {col 0}, x_B = [-1]):
    /// the lb-violation sign-flip leaves only col 3 with a positive `trow`. The
    /// leaving variable is col 0 (< n_enter in both cases), so the *lb-repair*
    /// decision is identical — only the *entering* exclusion differs:
    ///   - n_enter = 4: col 3 enters → pivot → x_B = [1] ≥ 0 → Optimal.
    ///   - n_enter = 2: col 3 (idx 3 ≥ 2) is excluded, no other candidate → Unbounded.
    ///
    /// no-op proof: reverting core to pass `n_price` to the ratio test makes the
    /// n_enter=2 run also admit col 3 → Optimal, not Unbounded → this FAILS.
    #[test]
    fn n_enter_excludes_high_index_columns_from_entering() {
        // a = [1, 1, 1, -1], single row.
        let a =
            CscMatrix::from_triplets(&[0, 0, 0, 0], &[0, 1, 2, 3], &[1.0, 1.0, 1.0, -1.0], 1, 4)
                .unwrap();
        let c = vec![0.0_f64; 4];
        let opts = SolverOptions::default();

        let run = |n_enter: usize| {
            let mut basis = vec![0usize];
            let mut x_b = vec![-1.0_f64];
            let mut leaving = MostInfeasibleLeaving;
            let mut iters = 0usize;
            dual_simplex_core_advanced(
                &a, &mut x_b, &c, &mut basis, 1, 4, n_enter, false, &opts, &mut leaving, &mut iters,
            )
        };

        assert!(
            matches!(run(4), SimplexOutcome::Optimal(_, _)),
            "n_enter=4 (col 3 enterable) must reach Optimal"
        );
        assert!(
            matches!(run(2), SimplexOutcome::Unbounded),
            "n_enter=2 must EXCLUDE col 3 (idx >= n_enter) from entering → Unbounded; \
             got non-Unbounded ⇒ core passes n_price not n_enter (artificial re-entry ban reverted)"
        );
    }

    /// LOAD-BEARING sentinel: the stall→Timeout yield must fire ONLY when the
    /// caller opts in (`yield_on_stall = true`, i.e. Big-M, which has a fallback).
    /// A standard caller (`false`) must keep iterating — classical Bland
    /// terminates and there is no fallback, so yielding would turn a solvable LP
    /// into a spurious Timeout (codex concern A).
    ///
    /// `AlwaysStallLeaving` forces an endless degenerate pivot loop with a
    /// constant progress metric, so the core enters bland_mode and never makes
    /// progress. With a short deadline:
    ///   - yield_on_stall=true  → Timeout at the stall (~2·k_trigger iters).
    ///   - yield_on_stall=false → Timeout only at the deadline (far more iters).
    ///
    /// no-op proof: if the yield ignores the flag (always yields), the no-yield
    /// run stops at the same ~2·k_trigger point ⇒ `iters_noyield ≈ iters_yield`
    /// ⇒ the `>` assertion fails.
    #[test]
    fn yield_on_stall_gated_by_caller_flag() {
        use super::super::super::pricing::DualLeavingStrategy;

        struct AlwaysStallLeaving;
        impl DualLeavingStrategy for AlwaysStallLeaving {
            fn select_leaving(&mut self, _x_b: &[f64], _t: f64, _b: &[usize]) -> Option<usize> {
                Some(0)
            }
            fn bland_leaving(&mut self, _x_b: &[f64], _t: f64, _b: &[usize]) -> Option<usize> {
                Some(0)
            }
            fn progress_metric(&mut self, _x_b: &[f64], _b: &[usize]) -> f64 {
                1.0
            }
            // Suppress lb-repair so a negative x_B[0] is never sign-flip-repaired:
            // every column has a[0]>0 ⇒ trow stays positive (no Unbounded), the
            // forced negative leaving never resolves (no Optimal) ⇒ perpetual stall.
            fn allows_lb_repair(&self) -> bool {
                false
            }
        }

        // m=1, three columns all with a[0]=1. Forcing row 0 to leave with x_B[0]<0
        // and no lb-repair keeps x_B[0] negative forever (trow > 0 each iter), so
        // the core enters bland_mode and never makes progress.
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
        let c = vec![0.0_f64, 0.0, 0.0];
        let run = |yield_on_stall: bool| {
            let opts = SolverOptions {
                max_etas: 1,
                deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(300)),
                ..SolverOptions::default()
            };
            let mut basis = vec![2usize];
            let mut x_b = vec![-1.0_f64];
            let mut leaving = AlwaysStallLeaving;
            let mut iters = 0usize;
            let out = dual_simplex_core_advanced(
                &a, &mut x_b, &c, &mut basis, 1, 3, 3, yield_on_stall, &opts, &mut leaving,
                &mut iters,
            );
            (out, iters)
        };

        let (out_yield, iters_yield) = run(true);
        let (out_noyield, iters_noyield) = run(false);
        assert!(
            matches!(out_yield, SimplexOutcome::Timeout(_)),
            "yield path must Timeout (stall-yield)"
        );
        assert!(
            matches!(out_noyield, SimplexOutcome::Timeout(_)),
            "no-yield path must Timeout (deadline), not stall-yield"
        );
        assert!(
            iters_noyield > iters_yield.saturating_mul(4),
            "yield_on_stall=false must NOT stall-yield: it ran only {iters_noyield} iters \
             vs the yield path's {iters_yield}; near-equal ⇒ the flag is ignored and a \
             standard caller would be wrongly Timeout-ed"
        );
    }

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
            assert_ne!(
                x_b_after_entry,
                vec![0.5, 1.0, 0.3],
                "initial entry must perturb x_b"
            );

            // refactor 後 (perturb_x=false): x_b は変化しない
            let mut rc2 = vec![1.0, 2.0, 3.0, 0.5];
            apply_lex_perturbation(&mut rc2, &is_basic, &mut x_b, m, false);
            assert_eq!(
                x_b, x_b_after_entry,
                "Pattern A: x_b must not change with perturb_x=false (B4 バグ復帰で FAIL)"
            );

            // 2 回目 refactor (perturb_x=false): 同じく変化なし
            let mut rc3 = vec![1.0, 2.0, 3.0, 0.5];
            apply_lex_perturbation(&mut rc3, &is_basic, &mut x_b, m, false);
            assert_eq!(
                x_b, x_b_after_entry,
                "Pattern A: x_b must remain stable across repeated perturb_x=false calls"
            );
        }

        // Pattern B: x_b に大きな値
        {
            let mut rc = vec![100.0, 200.0, 50.0, 1.0];
            let mut x_b = vec![1000.0_f64, 500.0, 750.0];
            let m = x_b.len();

            apply_lex_perturbation(&mut rc, &is_basic, &mut x_b, m, true);
            let x_b_after_entry = x_b.clone();
            assert_ne!(
                x_b_after_entry,
                vec![1000.0, 500.0, 750.0],
                "initial entry must perturb x_b"
            );

            // 3 回連続 refactor: x_b は不変
            for call_n in 0..3 {
                let mut rc_n = vec![100.0, 200.0, 50.0, 1.0];
                apply_lex_perturbation(&mut rc_n, &is_basic, &mut x_b, m, false);
                assert_eq!(
                    x_b, x_b_after_entry,
                    "Pattern B: x_b must not grow after refactor call #{} (B4 バグ復帰で FAIL)",
                    call_n
                );
            }
        }
    }

    /// #202 sentinel: `set_initial_gamma` must be invoked exactly once
    /// (cold-start). The per-refactor m-BTRAN re-init was redundant — the
    /// rank-1 update is exact (Forrest-Goldfarb 1992) and `after_refactor`
    /// handles CEILING-flagged drift via identity reset.
    ///
    /// no-op proof: restoring the `recompute_gamma_truth` block after the
    /// `needs_refactor` branch makes the call counter > 1 (1 cold + N refactors).
    #[test]
    fn recompute_gamma_after_refactor_not_called() {
        use super::super::super::pricing::DualLeavingStrategy;
        use std::cell::Cell;

        struct CountingDseLeaving<'a> {
            init_calls: &'a Cell<usize>,
            refactor_calls: &'a Cell<usize>,
        }
        impl<'a> DualLeavingStrategy for CountingDseLeaving<'a> {
            fn select_leaving(
                &mut self,
                x_b: &[f64],
                primal_tol: f64,
                _basis: &[usize],
            ) -> Option<usize> {
                let mut best_row = None;
                let mut max_violation = primal_tol;
                for (i, &val) in x_b.iter().enumerate() {
                    if val < -max_violation {
                        max_violation = -val;
                        best_row = Some(i);
                    }
                }
                best_row
            }
            fn needs_sigma(&self) -> bool {
                true
            }
            fn after_refactor(&mut self, _m: usize) {
                self.refactor_calls.set(self.refactor_calls.get() + 1);
            }
            fn set_initial_gamma(&mut self, _gamma_truth: &[f64]) {
                self.init_calls.set(self.init_calls.get() + 1);
            }
        }

        // m=3 LP with 3 lb-violated rows + 3 non-zero non-basic cols + 3 slacks.
        // A · x = b with x ≥ 0:
        //   -x0 - x1                + s0           = -1
        //         -x1 - x2               + s1      = -2
        //   -x0       - x2                    + s2 = -1
        // c = [4, 5, 6, 0, 0, 0]: r_j ≥ 0 at slack basis → dual-feasible.
        // max_etas=1 forces refactor after every successful pivot.
        let rows = [0, 2, 0, 1, 1, 2, 0, 1, 2];
        let cols = [0, 0, 1, 1, 2, 2, 3, 4, 5];
        let vals = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 6).unwrap();
        let c = vec![4.0, 5.0, 6.0, 0.0, 0.0, 0.0];
        let mut basis = vec![3usize, 4, 5];
        let mut x_b = vec![-1.0_f64, -2.0, -1.0];
        let opts = SolverOptions {
            max_etas: 1,
            ..SolverOptions::default()
        };
        let init_calls = Cell::new(0);
        let refactor_calls = Cell::new(0);
        let mut leaving = CountingDseLeaving {
            init_calls: &init_calls,
            refactor_calls: &refactor_calls,
        };
        let mut iters = 0usize;
        let _ = dual_simplex_core_advanced(
            &a,
            &mut x_b,
            &c,
            &mut basis,
            3,
            6,
            6,
            false,
            &opts,
            &mut leaving,
            &mut iters,
        );

        assert!(
            refactor_calls.get() >= 1,
            "test data must trigger at least one refactor; got {} (raise max_etas trigger?)",
            refactor_calls.get(),
        );
        assert_eq!(
            init_calls.get(),
            1,
            "set_initial_gamma must fire exactly once (cold-start only); got {} \
             across {} refactor(s) — recompute_gamma_truth re-introduced after refactor?",
            init_calls.get(),
            refactor_calls.get(),
        );
    }
}
