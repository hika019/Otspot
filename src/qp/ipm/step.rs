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
    norm_inf, spmtv, spmv, spmv_q,
};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::ldl::LdlFactorizationAmd;
use super::init::compute_initial_point;

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

    let mut status = SolveStatus::MaxIterations;
    let mut final_iter = options.ipm.max_iter;

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

        // 収束判定
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let dual_res = norm_inf(&r_d) / norm_c;
        let prim_res = norm_inf(&r_p) / norm_b;

        if dual_res < options.ipm_eps() && prim_res < options.ipm_eps() && mu < options.ipm_eps() {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = options.ipm.delta_min.max(options.ipm.delta_p_init * mu);
        let delta_d = options.ipm.delta_min.max(options.ipm.delta_d_init * mu);

        // Σ = diag(s_i / y_i)（両パスで共通）
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| si / yi).collect();

        // ===== LDLパス: augmented system + factorize_quasidefinite_with_deadline =====

        // T2: 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // augmented KKT行列構築 + factorize（delta_p リトライ最大4回）
        // AMD permutation はスパースパターン不変なので初回のみ計算してキャッシュ
        let mut delta_p_retry = delta_p;
        let mut fac_opt: Option<LdlFactorizationAmd> = None;
        for _retry in 0..4 {
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                break;
            }
            let aug_mat = build_augmented_system(
                &problem.q, &a_ext, &sigma_vec, delta_p_retry, delta_d,
            );
            // 初回のみ AMD permutation を計算してキャッシュ
            if amd_perm_cache.is_none() {
                amd_perm_cache = Some(amd_with_deadline(aug_mat.nrows, &aug_mat.col_ptr, &aug_mat.row_ind, timeout_ctx.deadline));
            }
            let perm = amd_perm_cache.as_ref().unwrap();
            match ldl::factorize_quasidefinite_with_cached_perm(&aug_mat, perm, timeout_ctx.deadline) {
                Ok(f) => { fac_opt = Some(f); break; }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => { delta_p_retry *= 10.0; }
            }
        }
        if status == SolveStatus::Timeout {
            break;
        }
        let fac = match fac_opt {
            Some(f) => f,
            None => return numerical_error_result(n),
        };

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

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals: vec![],
        active_set: vec![],
        iterations: final_iter,
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

    let mut status = SolveStatus::MaxIterations;
    let mut final_iter = options.ipm.max_iter;

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

        // 収束判定
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let dual_res = norm_inf(&r_d) / norm_c;
        let prim_res = norm_inf(&r_p) / norm_b;

        if dual_res < options.ipm_eps() && prim_res < options.ipm_eps() && mu < options.ipm_eps() {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = options.ipm.delta_min.max(options.ipm.delta_p_init * mu);
        let delta_d = options.ipm.delta_min.max(options.ipm.delta_d_init * mu);

        // Σ = diag(s_i / y_i),  D = Σ + δ_d
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| si / yi).collect();
        let d_vec: Vec<f64> = sigma_vec.iter().map(|&sg| sg + delta_d).collect();
        let d_inv: Vec<f64> = d_vec.iter().map(|&d| 1.0 / d).collect();

        // ===== LDLパス: Schur complement を明示構築して LDL 分解 =====

        // T2: LDL 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // δ_p を ×10 ずつ増やして最大4回リトライ
        let mut delta_p_retry = delta_p;
        let mut fac_opt = None;
        for _retry in 0..4 {
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
            match ldl::factorize_with_deadline(&m_mat_retry, timeout_ctx.deadline) {
                Ok(f) => { fac_opt = Some(f); break; }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => { delta_p_retry *= 10.0; }
            }
        }
        if status == SolveStatus::Timeout {
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

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals: vec![],
        active_set: vec![],
        iterations: final_iter,
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

