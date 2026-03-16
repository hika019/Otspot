//! IPM ステップ計算（Mehrotra predictor-corrector）
//!
//! - メインループ (`solve_qp_ipm_inner`)
//! - 制約なし QP (`solve_unconstrained`)
//! - fraction-to-boundary
//! - ユーティリティ

use crate::linalg::ldl;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::kkt::{
    build_augmented_system, build_extended_constraints,
    build_schur_complement,
    collect_part1_diag_indices, collect_part3_diag_indices,
    collect_q_diag_base, update_augmented_values,
    norm_inf, spmtv, spmv, spmv_q, KktCache,
};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::ldl::LdlFactorizationAmd;
use super::init::compute_initial_point;

// ---------------------------------------------------------------------------
// Infeasibility / Unboundedness 検出
// ---------------------------------------------------------------------------

/// Gondzio corrector 後のステップ方向 (Δx, Δy) から実行不能または非有界を検出する。
///
/// MIN_DIR_NORM より小さい方向ベクトルでの検出は行わない（収束時の誤検知防止）。
///
/// **実行不能 (Primal Infeasibility)**:
/// ||Δy_orig||_inf > MIN_DIR_NORM かつ:
///   ① ||A_orig^T * Δy_orig|| / max(1, ||Δy||) < ε_inf
///   ② b_orig · Δy_orig / max(1, ||Δy||) < -ε_inf
///
/// **非有界 (Dual Infeasibility / Unboundedness)**:
/// ||Δx||_inf > MIN_DIR_NORM かつ:
///   ③ c · Δx / max(1, ||Δx||) < -ε_inf  (LP: Q=0)
///      ||(Q*Δx + c)|| / max(1, ||Δx||) < ε_inf  (QP: Q≠0)
///   ④ ||A_orig * Δx|| / max(1, ||Δx||) < ε_inf
fn check_infeasible_or_unbounded(
    dx: &[f64],
    dy: &[f64],
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_orig: usize,
    m_ext: usize,
    iter: usize,
) -> Option<SolveStatus> {
    const EPS_INF: f64 = 1e-8;
    const MIN_ITER: usize = 5;
    /// 収束時の偽陽性防止: 方向ベクトルが MIN_DIR_NORM 以下は検出スキップ。
    /// 収束時は Δx→0, Δy→0 なので norm=max(1,||Δ||)=1 となり比率が偶然ε未満になる。
    const MIN_DIR_NORM: f64 = 1e-3;

    if iter < MIN_ITER {
        return None;
    }

    let n = dx.len();

    // --- Primal Infeasibility check ---
    // ||Δy_orig|| が MIN_DIR_NORM より小さければスキップ（収束時偽陽性防止）。
    if m_orig > 0 {
        let dy_orig = &dy[..m_orig];
        let norm_dy_inf = norm_inf(dy_orig);
        if norm_dy_inf > MIN_DIR_NORM {
            let norm_dy = norm_dy_inf.max(1.0);
            // ① A_orig^T * Δy_orig: a_ext は CSC, 行インデックス < m_orig のエントリのみ使用
            let mut at_dy = vec![0.0f64; n];
            for (j, at_dy_j) in at_dy.iter_mut().enumerate() {
                for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                    let row = a_ext.row_ind[ptr];
                    if row < m_orig {
                        *at_dy_j += a_ext.values[ptr] * dy_orig[row];
                    }
                }
            }
            let cond_a = norm_inf(&at_dy) / norm_dy < EPS_INF;
            // ② b_orig · Δy_orig
            let b_dy: f64 = problem.b.iter().zip(dy_orig.iter()).map(|(&bi, &dyi)| bi * dyi).sum();
            let cond_b = b_dy / norm_dy < -EPS_INF;
            if cond_a && cond_b {
                return Some(SolveStatus::Infeasible);
            }
        }
    }

    // --- Dual Infeasibility / Unboundedness check ---
    // m_orig=0 かつ m_ext>0 の場合（境界制約のみの問題）はチェック全体をスキップ。
    // 等式制約なし問題は通常 bounded のため偽陽性回避を優先する。
    if m_orig == 0 && m_ext > 0 {
        return None;
    }
    // ||Δx|| が MIN_DIR_NORM より小さければスキップ（収束時偽陽性防止）。
    let norm_dx_inf = norm_inf(dx);
    if norm_dx_inf <= MIN_DIR_NORM {
        return None;
    }
    let norm_dx = norm_dx_inf.max(1.0);

    // ③ 目的関数方向条件: LP(Q=0) → c·Δx < -ε*norm_dx; QP(Q≠0) → ||Q*Δx+c||/norm_dx < ε
    let is_lp = problem.q.values.iter().all(|&v| v == 0.0);
    let cond_obj = if is_lp {
        let c_dx: f64 = problem.c.iter().zip(dx.iter()).map(|(&ci, &dxi)| ci * dxi).sum();
        c_dx / norm_dx < -EPS_INF
    } else {
        let mut qdx = vec![0.0f64; n];
        spmv_q(&problem.q, dx, &mut qdx);
        let qdx_plus_c_norm: f64 = qdx.iter().zip(problem.c.iter())
            .map(|(&qi, &ci)| (qi + ci).abs())
            .fold(0.0_f64, f64::max);
        qdx_plus_c_norm / norm_dx < EPS_INF
    };
    if !cond_obj {
        return None;
    }

    // ④ ||A_orig * Δx|| / norm_dx < ε
    if m_orig > 0 {
        let mut a_dx = vec![0.0f64; m_orig];
        for (j, &dxj) in dx.iter().enumerate() {
            for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                let row = a_ext.row_ind[ptr];
                if row < m_orig {
                    a_dx[row] += a_ext.values[ptr] * dxj;
                }
            }
        }
        if norm_inf(&a_dx) / norm_dx >= EPS_INF {
            return None;
        }
    }

    Some(SolveStatus::Unbounded)
}

// ---------------------------------------------------------------------------
// fraction-to-boundary
// ---------------------------------------------------------------------------

/// α = min(1, τ · min_i { -v_i / Δv_i }  for Δv_i < 0 )
pub(crate) fn fraction_to_boundary(v: &[f64], dv: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for (&vi, &dvi) in v.iter().zip(dv.iter()) {
        if dvi < 0.0 {
            let step = tau * vi / (-dvi);
            if step < alpha {
                alpha = step;
            }
        }
    }
    alpha
}

// ---------------------------------------------------------------------------
// IPM 内部ソルバー
// ---------------------------------------------------------------------------

/// IPM内部ソルバー（Ruizスケーリング適用済みproblemを受け取る）
///
/// augmented KKT system + LDLT（DirectLDL一本化）
pub(crate) fn solve_qp_ipm_inner(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);

    // T1: 処理前タイムアウトチェック
    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築
    let (a_ext, b_ext, m_ext, m_orig, _n_lb) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 初期点
    let (mut x, mut s, mut y) = compute_initial_point(n, &b_ext);

    // 作業バッファ
    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];

    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];
    // AMD permutation キャッシュ（augmented system のスパースパターンは反復間で不変）
    let mut amd_perm_cache: Option<Vec<usize>> = None;
    // KKT 差分更新キャッシュ（方式 D: values 直接更新 + refactorize_numeric 再利用）
    let mut kkt_cache: Option<KktCache> = None;
    let mut fac_cache: Option<LdlFactorizationAmd> = None;

    let mut status = SolveStatus::Timeout;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    for iter in 0..options.ipm.max_iter {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // 残差計算
        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = s^T y / m_ext（相補性ギャップ）
        let mu: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum::<f64>() / m_ext as f64;

        // 最終残差を更新（収束・MaxIterations・Timeout いずれの場合も最後の値を保持）
        final_residuals = Some((norm_inf(&r_p), norm_inf(&r_d), mu));

        // 収束判定: 混合許容誤差 eps_abs + eps_rel * norm (Gurobi方式)
        // prim: ||r_p|| < eps * (1 + norm_b), dual: ||r_d|| < eps * (1 + norm_c)
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        if norm_inf(&r_d) < eps * (1.0 + norm_c)
            && norm_inf(&r_p) < eps * (1.0 + norm_b)
            && mu < eps
        {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // μ が正則化下限以下まで収縮し、残差も eps 水準以下なら SuboptimalSolution
        // （delta_min バイアスにより完全収束不能だが実用精度に達している状態を検出）
        // 閾値 = max(eps*(1+norm), delta_min*10): 正則化限界(~delta_min)の10倍をフロアとする
        // delta_min*100(旧)はpv_retry時にeps連動を阻害しDTOC3退行を引き起こした[cmd_441]
        let thr_d = (eps * (1.0 + norm_c)).max(options.ipm.delta_min * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(options.ipm.delta_min * 10.0);
        if mu < options.ipm.delta_min * 1e-2
            && norm_inf(&r_d) < thr_d
            && norm_inf(&r_p) < thr_p
        {
            status = SolveStatus::SuboptimalSolution;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = options.ipm.delta_min.max(options.ipm.delta_p_init * mu);
        let delta_d = options.ipm.delta_min.max(options.ipm.delta_d_init * mu);

        // Σ = diag(s_i / y_i)（両パスで共通）
        // y→0 のとき si/yi→Inf になる場合がある。faerはInf値を行列要素として処理できないため
        // sigma_max = 1/delta_min でクリップ（ippmm.rs L278-284と同等）
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter())
            .map(|(&si, &yi)| {
                let v = si / yi;
                if v.is_finite() { v } else { sigma_max }
            })
            .collect();

        // ===== LDLパス: augmented system + factorize_quasidefinite_with_deadline =====

        // T2: 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // augmented KKT行列構築 + factorize（delta_p/delta_d リトライ最大10回, 上限1e0）
        // 方式 D: 初回は full 構築 + symbolic/numeric 全因子化、2反復目以降は values 差分更新
        //         + refactorize_numeric（symbolic 再利用）で O(nnz log nnz) → O(n + m_ext) に削減。
        // AMD permutation はスパースパターン不変なので初回のみ計算してキャッシュ
        let mut delta_p_retry = delta_p;
        let mut delta_d_retry = delta_d;
        let mut retry_timeout = false;
        'retry: for _retry in 0..10 {
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                retry_timeout = true;
                break 'retry;
            }
            if let Some(cache) = kkt_cache.as_mut() {
                // 2反復目以降: values のみ O(n + m_ext) で更新（高速パス）
                update_augmented_values(cache, &sigma_vec, delta_p_retry, delta_d_retry);
                let f = fac_cache.as_mut().unwrap();
                match f.refactorize_numeric_threaded(&cache.mat, timeout_ctx.deadline) {
                    Ok(()) => {
                        break 'retry;
                    }
                    Err(ldl::LdlError::DeadlineExceeded) => {
                        status = SolveStatus::Timeout;
                        final_iter = iter;
                        retry_timeout = true;
                        break 'retry;
                    }
                    Err(_) => {
                        // SingularOrIndefinite → delta_p 増加してリトライ
                        if delta_p_retry >= 1e0 {
                            // GQ-01: 高速パス全リトライ失敗時 fac_cache を無効化
                            // → M-02 チェックで NumericalError を返す（stale cache で solve しない）
                            fac_cache = None;
                            break 'retry;
                        }
                        delta_p_retry = (delta_p_retry * 10.0).min(1e0);
                        delta_d_retry = (delta_d_retry * 10.0).min(1e0);
                        // symbolic は delta_p/delta_d 変更で無効にならないのでキャッシュ維持
                        continue;
                    }
                }
            } else {
                // 初回: KKT 行列を full 構築し、インデックスを収集
                let aug_mat = build_augmented_system(
                    &problem.q, &a_ext, &sigma_vec, delta_p_retry, delta_d_retry,
                );
                // 初回のみ AMD permutation を計算してキャッシュ
                if amd_perm_cache.is_none() {
                    // 第1防御: full AMD（primal/dual交互消去でfill-in最小化）
                    amd_perm_cache = Some(amd_with_deadline(aug_mat.nrows, &aug_mat.col_ptr, &aug_mat.row_ind, timeout_ctx.deadline));
                }
                let perm = amd_perm_cache.as_ref().unwrap();
                // Part 1/3 のインデックスを収集して KktCache を構築
                let part1_idx = collect_part1_diag_indices(&aug_mat, n);
                let part3_idx = collect_part3_diag_indices(&aug_mat, n, m_ext);
                let q_diag_base = collect_q_diag_base(&problem.q, n);
                kkt_cache = Some(KktCache {
                    mat: aug_mat,
                    part1_diag_idx: part1_idx,
                    q_diag_base,
                    part3_diag_idx: part3_idx,
                    part1_updated_idx: (0..n).collect(),
                });
                // 初回は symbolic + numeric の全因子化
                match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                    &kkt_cache.as_ref().unwrap().mat, perm, timeout_ctx.deadline
                ) {
                    Ok(f) => { fac_cache = Some(f); break 'retry; }
                    Err(ldl::LdlError::DeadlineExceeded) => {
                        status = SolveStatus::Timeout;
                        final_iter = iter;
                        retry_timeout = true;
                        break 'retry;
                    }
                    Err(_) => {
                        kkt_cache = None; // 初回失敗時はキャッシュをリセット
                        if delta_p_retry >= 1e0 { break 'retry; }
                        delta_p_retry = (delta_p_retry * 10.0).min(1e0);
                        delta_d_retry = (delta_d_retry * 10.0).min(1e0);
                    }
                }
            }
        }
        // retry ループ後: Timeout が発生した場合は外ループを抜ける
        if retry_timeout {
            break;
        }
        // 第3防御: Identity fallback — 全リトライ失敗時に identity perm + 大きな delta で再試行
        // amd_perm_cache を無効化し、次の反復で block AMD が再計算されるようにする
        if fac_cache.is_none() {
            amd_perm_cache = None;
            drop(kkt_cache.take()); // 旧キャッシュを解放し、次行で再構築
            let delta_fallback = 1e-2_f64.max(delta_p_retry).max(delta_d_retry);
            let aug_mat_fb = build_augmented_system(
                &problem.q, &a_ext, &sigma_vec, delta_fallback, delta_fallback,
            );
            let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
            let part1_idx = collect_part1_diag_indices(&aug_mat_fb, n);
            let part3_idx = collect_part3_diag_indices(&aug_mat_fb, n, m_ext);
            let q_diag_base = collect_q_diag_base(&problem.q, n);
            kkt_cache = Some(KktCache {
                mat: aug_mat_fb,
                part1_diag_idx: part1_idx,
                q_diag_base,
                part3_diag_idx: part3_idx,
                part1_updated_idx: (0..n).collect(),
            });
            match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                &kkt_cache.as_ref().unwrap().mat, &identity_perm, timeout_ctx.deadline
            ) {
                Ok(f) => { fac_cache = Some(f); }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    // identity fallback も失敗 → fac_cache は None のまま → M-02
                }
            }
        }

        // M-02: fac_cache が None なら全リトライ失敗 → NumericalError
        if fac_cache.is_none() {
            return numerical_error_result(n);
        }
        let fac = fac_cache.as_ref().unwrap();

        // augmented system の RHS: [r_d; r_p_mod]（size = n + m_ext）
        let total = n + m_ext;
        let mut rhs = vec![0.0f64; total];
        let mut sol = vec![0.0f64; total];

        // --- Predictor ---
        let r_c_pred: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
        let r_p_mod_pred: Vec<f64> = r_p.iter().zip(r_c_pred.iter()).zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();

        rhs[..n].copy_from_slice(&r_d);
        rhs[n..].copy_from_slice(&r_p_mod_pred);
        fac.solve(&rhs, &mut sol);
        // augmented system: sol[..n]=dx_pred（未使用）, sol[n..]=dy_pred
        let dy_pred = sol[n..].to_vec();

        let mut ds_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }

        let alpha_s_pred = fraction_to_boundary(&s, &ds_pred, super::TAU);
        let alpha_y_pred = fraction_to_boundary(&y, &dy_pred, super::TAU);
        let alpha_pred = alpha_s_pred.min(alpha_y_pred);
        let mu_aff: f64 = s.iter().zip(y.iter()).zip(ds_pred.iter()).zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| (si + alpha_pred * dsi) * (yi + alpha_pred * dyi))
            .sum::<f64>() / m_ext as f64;
        let sigma_center = if mu > 1e-15 { (mu_aff / mu).powi(3).min(1.0) } else { 0.0 };

        // --- Corrector ---
        let r_c_corr: Vec<f64> = s.iter().zip(y.iter()).zip(ds_pred.iter()).zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| sigma_center * mu - si * yi - dsi * dyi).collect();
        let r_p_mod_corr: Vec<f64> = r_p.iter().zip(r_c_corr.iter()).zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();

        rhs[..n].copy_from_slice(&r_d);
        rhs[n..].copy_from_slice(&r_p_mod_corr);
        fac.solve(&rhs, &mut sol);
        dx.copy_from_slice(&sol[..n]);
        dy.copy_from_slice(&sol[n..]);

        for i in 0..m_ext {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }

                // α: fraction-to-boundary (corrector)
        let alpha_s = fraction_to_boundary(&s, &ds, super::TAU);
        let alpha_y = fraction_to_boundary(&y, &dy, super::TAU);
        let alpha = alpha_s.min(alpha_y);

        // ========== Gondzio Multiple Centrality Correctors (Augmented path) ==========
        let mut alpha = alpha;
        if alpha < 0.999 {
            let mut alpha_prev = alpha;
            for _k in 0..options.ipm.max_correctors {
                // (1) 目標step sizeとμ
                let alpha_target = (alpha_prev + super::BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
                let mu_target: f64 = s.iter().zip(y.iter()).zip(ds.iter().zip(dy.iter()))
                    .map(|((&si, &yi), (&dsi, &dyi))| {
                        (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                    })
                    .sum::<f64>() / m_ext as f64;
                let mu_target = mu_target.max(0.0);

                // (2) 各complementarity pairの目標範囲
                let target_lo = super::GAMMA_L * mu_target;
                let target_hi = super::GAMMA_U * mu_target;

                // (3) Gondzio corrector RHS構築
                //     v_i = (s_i + α·ds_i)(y_i + α·dy_i) を[target_lo, target_hi]に射影
                let mut r_c_gondzio = vec![0.0f64; m_ext];
                for i in 0..m_ext {
                    let si_new = s[i] + alpha_prev * ds[i];
                    let yi_new = y[i] + alpha_prev * dy[i];
                    let v_i = si_new * yi_new;
                    let v_target = if v_i < target_lo {
                        target_lo - v_i
                    } else if v_i > target_hi {
                        target_hi - v_i
                    } else {
                        0.0
                    };
                    r_c_gondzio[i] = r_c_corr[i] + v_target;
                }

                // (4) 修正RHS構築 & LDL因子再利用solve
                let r_p_mod_gondzio: Vec<f64> = r_p.iter().zip(r_c_gondzio.iter()).zip(y.iter())
                    .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();
                rhs[..n].copy_from_slice(&r_d);
                rhs[n..].copy_from_slice(&r_p_mod_gondzio);
                fac.solve(&rhs, &mut sol);
                let dx_new = sol[..n].to_vec();
                let dy_new = sol[n..].to_vec();
                let ds_new: Vec<f64> = (0..m_ext)
                    .map(|i| r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i])
                    .collect();

                // (5) 新しいstep sizeを計算
                let alpha_s_new = fraction_to_boundary(&s, &ds_new, super::TAU);
                let alpha_y_new = fraction_to_boundary(&y, &dy_new, super::TAU);
                let alpha_new = alpha_s_new.min(alpha_y_new);

                // (6) 改善判定: 改善なしならbreak
                if alpha_new < alpha_prev + super::ALPHA_IMPROVE_THRESHOLD {
                    break;
                }

                // (7) 改善あり → 方向を更新
                dx.copy_from_slice(&dx_new);
                dy.copy_from_slice(&dy_new);
                ds.copy_from_slice(&ds_new);
                alpha_prev = alpha_new;
            }
            alpha = alpha_prev;
        }
        // ========== Gondzio Correctors End ==========

        // Infeasibility / Unboundedness 検出（augmented パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter,
        ) {
            status = infeas_status;
            final_iter = iter;
            break;
        }

        // 変数更新
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_ext {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }
    }

    // 目的関数値
    spmv_q(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = y[..m_orig].to_vec();
    let bound_duals = y[m_orig..m_ext].to_vec();

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,
        active_set: vec![],
        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// IPM Schur complement 内部ソルバー
// ---------------------------------------------------------------------------

/// Schur complement LDL パスを使う IPM 内部ソルバー
///
/// n <= LDL_THRESHOLD 専用。n > LDL_THRESHOLD の場合は `solve_qp_ipm_inner` に委譲。
pub(crate) fn solve_qp_ipm_schur_inner(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let n = problem.num_vars;

    // n > LDL_THRESHOLD → Schur は非効率なので augmented に委譲
    if n > super::LDL_THRESHOLD {
        return solve_qp_ipm_inner(problem, options);
    }

    let timeout_ctx = TimeoutCtx::from_options(options);

    // T1: 処理前タイムアウトチェック
    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築
    let (a_ext, b_ext, m_ext, m_orig, _n_lb) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 初期点
    let (mut x, mut s, mut y) = compute_initial_point(n, &b_ext);

    // 作業バッファ
    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];

    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];

    let mut status = SolveStatus::Timeout;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    for iter in 0..options.ipm.max_iter {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // 残差計算
        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = s^T y / m_ext（相補性ギャップ）
        let mu: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum::<f64>() / m_ext as f64;

        // 最終残差を更新（収束・MaxIterations・Timeout いずれの場合も最後の値を保持）
        final_residuals = Some((norm_inf(&r_p), norm_inf(&r_d), mu));

        // 収束判定: 混合許容誤差 eps_abs + eps_rel * norm (Gurobi方式)
        // prim: ||r_p|| < eps * (1 + norm_b), dual: ||r_d|| < eps * (1 + norm_c)
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        if norm_inf(&r_d) < eps * (1.0 + norm_c)
            && norm_inf(&r_p) < eps * (1.0 + norm_b)
            && mu < eps
        {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // μ が正則化下限以下まで収縮し、残差も eps 水準以下なら SuboptimalSolution
        // （augmented パスと同一設計: 閾値 = max(eps*(1+norm), delta_min*10)）[cmd_441]
        let thr_d = (eps * (1.0 + norm_c)).max(options.ipm.delta_min * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(options.ipm.delta_min * 10.0);
        if mu < options.ipm.delta_min * 1e-2
            && norm_inf(&r_d) < thr_d
            && norm_inf(&r_p) < thr_p
        {
            status = SolveStatus::SuboptimalSolution;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = options.ipm.delta_min.max(options.ipm.delta_p_init * mu);
        let delta_d = options.ipm.delta_min.max(options.ipm.delta_d_init * mu);

        // Σ = diag(s_i / y_i),  D = Σ + δ_d
        // y→0 のとき si/yi→Inf になる場合がある。sigma_max = 1/delta_min でクリップ（ippmm.rs同等）
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter())
            .map(|(&si, &yi)| {
                let v = si / yi;
                if v.is_finite() { v } else { sigma_max }
            })
            .collect();
        let d_vec: Vec<f64> = sigma_vec.iter().map(|&sg| sg + delta_d).collect();
        let d_inv: Vec<f64> = d_vec.iter().map(|&d| 1.0 / d).collect();

        // ===== LDLパス: Schur complement を明示構築して LDL 分解 =====

        // T2: LDL 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // δ_p を ×10 ずつ増やして最大10回リトライ（上限1e0）
        let mut delta_p_retry = delta_p;
        let mut fac_opt = None;
        for _retry in 0..10 {
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                break;
            }
            let m_mat_retry = match build_schur_complement(&problem.q, &a_ext, &d_inv, delta_p_retry, &timeout_ctx.cancel) {
                Some(m) => m,
                None => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
            };
            match ldl::factorize_with_deadline_threaded(&m_mat_retry, timeout_ctx.deadline) {
                Ok(f) => { fac_opt = Some(f); break; }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    if delta_p_retry >= 1e0 { break; } // 上限到達→あきらめ
                    delta_p_retry = (delta_p_retry * 10.0).min(1e0);
                }
            }
        }
        if status == SolveStatus::Timeout && fac_opt.is_none() {
            break;
        }
        let fac = match fac_opt {
            Some(f) => f,
            None => return numerical_error_result(n),
        };

        // --- Predictor ---
        let r_c_pred: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
        let r_p_mod_pred: Vec<f64> = r_p.iter().zip(r_c_pred.iter()).zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();
        let tmp_pred: Vec<f64> = r_p_mod_pred.iter().zip(d_inv.iter()).map(|(&ri, &di)| ri * di).collect();
        let mut atmp = vec![0.0f64; n];
        spmtv(&a_ext, &tmp_pred, &mut atmp);
        let rhs_x_pred: Vec<f64> = r_d.iter().zip(atmp.iter()).map(|(&rdi, &ai)| rdi + ai).collect();
        let mut dx_pred = vec![0.0f64; n];
        fac.solve(&rhs_x_pred, &mut dx_pred);

        let mut a_dx_pred = vec![0.0f64; m_ext];
        spmv(&a_ext, &dx_pred, &mut a_dx_pred);
        let mut dy_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            dy_pred[i] = d_inv[i] * (a_dx_pred[i] - r_p_mod_pred[i]);
        }
        let mut ds_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }

        let alpha_s_pred = fraction_to_boundary(&s, &ds_pred, super::TAU);
        let alpha_y_pred = fraction_to_boundary(&y, &dy_pred, super::TAU);
        let alpha_pred = alpha_s_pred.min(alpha_y_pred);
        let mu_aff: f64 = s.iter().zip(y.iter()).zip(ds_pred.iter()).zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| (si + alpha_pred * dsi) * (yi + alpha_pred * dyi))
            .sum::<f64>() / m_ext as f64;
        let sigma_center = if mu > 1e-15 { (mu_aff / mu).powi(3).min(1.0) } else { 0.0 };

        // --- Corrector ---
        let r_c_corr: Vec<f64> = s.iter().zip(y.iter()).zip(ds_pred.iter()).zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| sigma_center * mu - si * yi - dsi * dyi).collect();
        let r_p_mod_corr: Vec<f64> = r_p.iter().zip(r_c_corr.iter()).zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();
        let tmp_corr: Vec<f64> = r_p_mod_corr.iter().zip(d_inv.iter()).map(|(&ri, &di)| ri * di).collect();
        let mut atmp_corr = vec![0.0f64; n];
        spmtv(&a_ext, &tmp_corr, &mut atmp_corr);
        let rhs_x_corr: Vec<f64> = r_d.iter().zip(atmp_corr.iter()).map(|(&rdi, &ai)| rdi + ai).collect();
        fac.solve(&rhs_x_corr, &mut dx);

        let mut a_dx_corr = vec![0.0f64; m_ext];
        spmv(&a_ext, &dx, &mut a_dx_corr);
        for i in 0..m_ext {
            dy[i] = d_inv[i] * (a_dx_corr[i] - r_p_mod_corr[i]);
        }
        for i in 0..m_ext {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }

        // α: fraction-to-boundary (corrector)
        let alpha_s = fraction_to_boundary(&s, &ds, super::TAU);
        let alpha_y = fraction_to_boundary(&y, &dy, super::TAU);
        let alpha = alpha_s.min(alpha_y);

        // ========== Gondzio Multiple Centrality Correctors (Schur path) ==========
        let mut alpha = alpha;
        if alpha < 0.999 {
            let mut alpha_prev = alpha;
            for _k in 0..options.ipm.max_correctors {
                // (1) 目標step sizeとμ
                let alpha_target = (alpha_prev + super::BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
                let mu_target: f64 = s.iter().zip(y.iter()).zip(ds.iter().zip(dy.iter()))
                    .map(|((&si, &yi), (&dsi, &dyi))| {
                        (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                    })
                    .sum::<f64>() / m_ext as f64;
                let mu_target = mu_target.max(0.0);

                // (2) 各complementarity pairの目標範囲
                let target_lo = super::GAMMA_L * mu_target;
                let target_hi = super::GAMMA_U * mu_target;

                // (3) Gondzio corrector RHS構築
                let mut r_c_gondzio = vec![0.0f64; m_ext];
                for i in 0..m_ext {
                    let si_new = s[i] + alpha_prev * ds[i];
                    let yi_new = y[i] + alpha_prev * dy[i];
                    let v_i = si_new * yi_new;
                    let v_target = if v_i < target_lo {
                        target_lo - v_i
                    } else if v_i > target_hi {
                        target_hi - v_i
                    } else {
                        0.0
                    };
                    r_c_gondzio[i] = r_c_corr[i] + v_target;
                }

                // (4) Schur版: 修正RHS構築 & LDL因子再利用solve
                let r_p_mod_gondzio: Vec<f64> = r_p.iter().zip(r_c_gondzio.iter()).zip(y.iter())
                    .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();
                let tmp_gon: Vec<f64> = r_p_mod_gondzio.iter().zip(d_inv.iter())
                    .map(|(&ri, &di)| ri * di).collect();
                let mut atmp_gon = vec![0.0f64; n];
                spmtv(&a_ext, &tmp_gon, &mut atmp_gon);
                let rhs_x_gon: Vec<f64> = r_d.iter().zip(atmp_gon.iter())
                    .map(|(&rdi, &ai)| rdi + ai).collect();
                let mut dx_new = vec![0.0f64; n];
                fac.solve(&rhs_x_gon, &mut dx_new);

                let mut a_dx_gon = vec![0.0f64; m_ext];
                spmv(&a_ext, &dx_new, &mut a_dx_gon);
                let mut dy_new = vec![0.0f64; m_ext];
                for i in 0..m_ext {
                    dy_new[i] = d_inv[i] * (a_dx_gon[i] - r_p_mod_gondzio[i]);
                }
                let ds_new: Vec<f64> = (0..m_ext)
                    .map(|i| r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i])
                    .collect();

                // (5) 新しいstep sizeを計算
                let alpha_s_new = fraction_to_boundary(&s, &ds_new, super::TAU);
                let alpha_y_new = fraction_to_boundary(&y, &dy_new, super::TAU);
                let alpha_new = alpha_s_new.min(alpha_y_new);

                // (6) 改善判定: 改善なしならbreak
                if alpha_new < alpha_prev + super::ALPHA_IMPROVE_THRESHOLD {
                    break;
                }

                // (7) 改善あり → 方向を更新
                dx.copy_from_slice(&dx_new);
                dy.copy_from_slice(&dy_new);
                ds.copy_from_slice(&ds_new);
                alpha_prev = alpha_new;
            }
            alpha = alpha_prev;
        }
        // ========== Gondzio Correctors End ==========

        // Infeasibility / Unboundedness 検出（Schur パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter,
        ) {
            status = infeas_status;
            final_iter = iter;
            break;
        }

        // 変数更新
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_ext {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }
    }

    // 目的関数値
    spmv_q(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = y[..m_orig].to_vec();
    let bound_duals = y[m_orig..m_ext].to_vec();

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,
        active_set: vec![],
        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 制約なし QP
// ---------------------------------------------------------------------------

/// 制約なし QP を解く: Qx = -c（Q が PD でない場合は δ_p I で正則化）
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_unconstrained(problem: &QpProblem, timeout_ctx: &TimeoutCtx) -> SolverResult {
    let n = problem.num_vars;

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if n == 0 {
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        };
    }

    let delta_p = 1e-7;
    let mut triplet_rows: Vec<usize> = Vec::new();
    let mut triplet_cols: Vec<usize> = Vec::new();
    let mut triplet_vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];

    for col in 0..n {
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            let row = problem.q.row_ind[k];
            if row <= col {
                triplet_rows.push(row);
                triplet_cols.push(col);
                let v = problem.q.values[k] + if row == col { delta_p } else { 0.0 };
                triplet_vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            triplet_rows.push(i);
            triplet_cols.push(i);
            triplet_vals.push(delta_p);
        }
    }

    let q_reg = CscMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n, n)
        .unwrap();

    match ldl::factorize(&q_reg) {
        Ok(fac) => {
            let rhs: Vec<f64> = problem.c.iter().map(|&ci| -ci).collect();
            let mut x = vec![0.0f64; n];
            fac.solve(&rhs, &mut x);

            let mut qx = vec![0.0f64; n];
            spmv_q(&problem.q, &x, &mut qx);
            let objective = 0.5
                * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
                + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

            SolverResult {
                status: SolveStatus::Optimal,
                objective,
                solution: x,
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 1,
                ..Default::default()
            }
        }
        Err(_) => numerical_error_result(n),
    }
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

pub(crate) fn timeout_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
        ..Default::default()
    }
}

pub(crate) fn numerical_error_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::check_infeasible_or_unbounded;
    use crate::problem::SolveStatus;
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    /// STEP-T1: iter < MIN_ITER(=5) の場合 None が返ること
    #[test]
    fn test_iter_guard() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // MIN_DIR_NORM を超える大きさだが iter ガードが先
        let dy: Vec<f64> = vec![];
        // iter=4 < MIN_ITER=5 → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 4),
            None,
            "STEP-T1: iter < MIN_ITER は None であること"
        );
    }

    /// STEP-T2: ||Δx||_inf <= MIN_DIR_NORM(=1e-3) の場合 None が返ること
    #[test]
    fn test_min_dir_norm_guard() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![5e-4]; // ||dx||_inf = 5e-4 <= MIN_DIR_NORM = 1e-3
        let dy: Vec<f64> = vec![];
        // 収束時偽陽性防止: dx が小さすぎる → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10),
            None,
            "STEP-T2: ||dx||_inf <= MIN_DIR_NORM は None であること"
        );
    }

    /// STEP-T3: Farkas dual ray 条件を満たすベクトルで Infeasible 判定を確認
    ///
    /// A_orig = 0 (1x2 ゼロ行列), b = [-1], dy_orig = [2.0]
    /// ① ||A^T * dy_orig|| = 0 < ε ✓
    /// ② b · dy_orig = -2 < -ε ✓
    /// → Infeasible
    #[test]
    fn test_primal_infeasible_farkas() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let c = vec![1.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap(); // 1x2 ゼロ行列
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap();
        let dx = vec![1e-10, 1e-10]; // 非常に小さい → dual チェックはスキップ
        let dy = vec![2.0]; // norm = 2.0 > MIN_DIR_NORM
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 1, 1, 10),
            Some(SolveStatus::Infeasible),
            "STEP-T3: Farkas ray 条件 → Infeasible であること"
        );
    }

    /// STEP-T4: LP (Q=0) で c·Δx < 0 条件の Unbounded 判定を確認
    ///
    /// n=1, m_orig=0: c=[-1], dx=[1.0] → c·dx/norm_dx = -1 < -ε → Unbounded
    #[test]
    fn test_dual_infeasible_lp() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap(); // Q=0 (LP)
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(); // 制約なし
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // c·dx = -1 < -ε, m_ext=0 なので dual guard は無効
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10),
            Some(SolveStatus::Unbounded),
            "STEP-T4: LP dual infeasibility → Unbounded であること"
        );
    }
}

