//! IPM ステップ計算（Mehrotra predictor-corrector）
//!
//! - メインループ (`solve_qp_ipm_inner`)
//! - 制約なし QP (`solve_unconstrained`)
//! - fraction-to-boundary
//! - ユーティリティ

use crate::linalg::cg::{pcg_solve, CgWorkspace};
use crate::linalg::ldl;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::kkt::{
    build_augmented_system, build_extended_constraints,
    compute_jacobi_precond_ipm, mv_ipm_apply, norm_inf, spmtv, spmv, spmv_q,
};
#[cfg(feature = "parallel")]
use super::kkt::{build_constraint_schur, build_schur_complement};
#[cfg(feature = "parallel")]
use crate::linalg::ldl::LdlFactorizationAmd;
#[cfg(feature = "parallel")]
use crate::linalg::minres::{pminres_solve, MinresWorkspace};
#[cfg(feature = "parallel")]
use std::time::Instant;
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
/// n <= LDL_THRESHOLD: augmented KKT system + LDLT（D01-b/c）
/// n >  LDL_THRESHOLD: Matrix-Free PCG（Jacobi 前処理）でSchur complementを求解（D01-d: CGパスは変更しない）
pub(crate) fn solve_qp_ipm_inner(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let n = problem.num_vars;
    let use_cg = n > super::LDL_THRESHOLD;
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
    let mut cg_ws_opt: Option<CgWorkspace> = if use_cg { Some(CgWorkspace::new(n)) } else { None };

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

        if !use_cg {
            // ===== LDLパス: augmented system + factorize_quasidefinite_with_deadline =====

            // T2: 因子化前タイムアウトチェック
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                break;
            }

            // augmented KKT行列構築 + factorize（delta_p リトライ最大4回）
            let mut delta_p_retry = delta_p;
            let mut fac_opt = None;
            for _retry in 0..4 {
                if timeout_ctx.should_stop() {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                let aug_mat = build_augmented_system(
                    &problem.q, &a_ext, &sigma_vec, delta_p_retry, delta_d,
                );
                match ldl::factorize_quasidefinite_with_deadline(&aug_mat, timeout_ctx.deadline) {
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
        } else {
            // ===== CGパス: Matrix-Free PCG で Schur complement を求解（D01-d: 変更しない）=====
            let d_vec: Vec<f64> = sigma_vec.iter().map(|&sg| sg + delta_d).collect();
            let d_inv: Vec<f64> = d_vec.iter().map(|&d| 1.0 / d).collect();
            let m_inv = compute_jacobi_precond_ipm(&problem.q, &a_ext, &d_inv, delta_p);
            let cg_ws = cg_ws_opt.as_mut().unwrap();

            // T2: タイムアウトチェック
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                break;
            }

            // --- Predictor ---
            let r_c_pred: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
            let r_p_mod_pred: Vec<f64> = r_p.iter().zip(r_c_pred.iter()).zip(y.iter())
                .map(|((&rpi, &rci), &yi)| rpi - rci / yi).collect();
            let tmp_pred: Vec<f64> = r_p_mod_pred.iter().zip(d_inv.iter()).map(|(&ri, &di)| ri * di).collect();
            let mut atmp = vec![0.0f64; n];
            spmtv(&a_ext, &tmp_pred, &mut atmp);
            let rhs_x_pred: Vec<f64> = r_d.iter().zip(atmp.iter()).map(|(&rdi, &ai)| rdi + ai).collect();
            let mut dx_pred = vec![0.0f64; n];
            {
                let mut kv = |v: &[f64], o: &mut [f64]| {
                    mv_ipm_apply(&problem.q, &a_ext, &d_inv, delta_p, v, o);
                };
                let cg_result = pcg_solve(
                    &mut kv, &m_inv, &rhs_x_pred, &mut dx_pred,
                    super::CG_MAX_ITER, super::CG_TOL, cg_ws,
                    timeout_ctx.deadline, Some(&timeout_ctx.cancel),
                );
                if cg_result.timed_out {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
            }

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
            {
                let mut kv = |v: &[f64], o: &mut [f64]| {
                    mv_ipm_apply(&problem.q, &a_ext, &d_inv, delta_p, v, o);
                };
                let cg_result = pcg_solve(
                    &mut kv, &m_inv, &rhs_x_corr, &mut dx,
                    super::CG_MAX_ITER, super::CG_TOL, cg_ws,
                    timeout_ctx.deadline, Some(&timeout_ctx.cancel),
                );
                if cg_result.timed_out {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
            }

            let mut a_dx_corr = vec![0.0f64; m_ext];
            spmv(&a_ext, &dx, &mut a_dx_corr);
            for i in 0..m_ext {
                dy[i] = d_inv[i] * (a_dx_corr[i] - r_p_mod_corr[i]);
            }
            for i in 0..m_ext {
                ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
            }
        }

        // α: fraction-to-boundary (corrector)
        let alpha_s = fraction_to_boundary(&s, &ds, super::TAU);
        let alpha_y = fraction_to_boundary(&y, &dy, super::TAU);
        let alpha = alpha_s.min(alpha_y);

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
#[cfg(feature = "parallel")]
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

// ---------------------------------------------------------------------------
// MINRES + 制約前処理 IPM バリアント
// ---------------------------------------------------------------------------

/// MINRES の最大反復数（制約前処理付き）
#[cfg(feature = "parallel")]
const MINRES_MAX_ITER: usize = 50;

/// Q + δ_p·I の上三角 CSC を構築するヘルパー
#[cfg(feature = "parallel")]
#[allow(clippy::needless_range_loop)]
fn build_q_delta(q: &CscMatrix, delta_p: f64, n: usize) -> CscMatrix {
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k] + if row == col { delta_p } else { 0.0 };
                rows.push(row);
                cols.push(col);
                vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            rows.push(i);
            cols.push(i);
            vals.push(delta_p);
        }
    }
    if rows.is_empty() {
        CscMatrix::new(n, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }
}

/// augmented KKT 系を MINRES + 制約前処理（M_cp）で解くヘルパー
///
/// K = [Q+δ_p I, A^T; A, -D]  (対称不定値)
/// M_cp = [(Q+δ_p I)^{-1}, 0; 0, S^{-1}]  (ブロック対角 SPD)
///
/// `rhs` = [r_d (n-dim); r_p_mod (m_ext-dim)]
/// `x`   = [dx (n-dim);  dy (m_ext-dim)] （初期値を 0 に設定してから呼ぶこと）
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn solve_kkt_minres_constraint_precond(
    n: usize,
    m_ext: usize,
    q: &CscMatrix,
    q_fac: &LdlFactorizationAmd,
    s_fac: &LdlFactorizationAmd,
    a_ext: &CscMatrix,
    d_vec: &[f64],
    delta_p: f64,
    rhs: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
    deadline: Option<Instant>,
    cancel: &std::sync::atomic::AtomicBool,
    ws: &mut MinresWorkspace,
) -> crate::linalg::minres::MinresResult {
    // A^T * v2 の一時バッファ（kv_op 用）
    let mut tmp_atv = vec![0.0f64; n];

    // K * v 演算: K = [Q+δI, A^T; A, -D]
    let mut kv_op = |v: &[f64], out: &mut [f64]| {
        let (v1, v2) = v.split_at(n);
        let (out1, out2) = out.split_at_mut(n);
        // out1 = Q*v1 + δ_p*v1 + A^T*v2
        spmv_q(q, v1, out1);
        for i in 0..n {
            out1[i] += delta_p * v1[i];
        }
        spmtv(a_ext, v2, &mut tmp_atv);
        for i in 0..n {
            out1[i] += tmp_atv[i];
        }
        // out2 = A*v1 - D*v2
        spmv(a_ext, v1, out2);
        for i in 0..m_ext {
            out2[i] -= d_vec[i] * v2[i];
        }
    };

    // M_cp^{-1} * v 演算: ブロック対角 [(Q+δI)^{-1}, 0; 0, S^{-1}]
    let mut precond_op = |v: &[f64], out: &mut [f64]| {
        let (v1, v2) = v.split_at(n);
        let (out1, out2) = out.split_at_mut(n);
        q_fac.solve(v1, out1);
        s_fac.solve(v2, out2);
    };

    pminres_solve(
        &mut kv_op,
        &mut precond_op,
        rhs,
        x,
        max_iter,
        tol,
        ws,
        deadline,
        Some(cancel),
    )
}

/// MINRES + 制約前処理を使う IPM 内部ソルバー（n > LDL_THRESHOLD 専用）
///
/// n <= LDL_THRESHOLD の場合は `solve_qp_ipm_inner`（augmented LDL パス）に委譲。
/// S 構築失敗・MINRES 収束失敗の場合は CG+Jacobi パスに自動フォールバック。
#[cfg(feature = "parallel")]
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_qp_ipm_minres_inner(
    problem: &QpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let n = problem.num_vars;

    // n <= LDL_THRESHOLD: augmented LDL パスが最適
    if n <= super::LDL_THRESHOLD {
        return solve_qp_ipm_inner(problem, options);
    }

    let timeout_ctx = TimeoutCtx::from_options(options);

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    let (a_ext, b_ext, m_ext, m_orig, _n_lb) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

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

    // MINRES バッファ（n+m_ext サイズ）
    let total = n + m_ext;
    let mut minres_ws = MinresWorkspace::new(total);
    let mut minres_rhs = vec![0.0f64; total];
    let mut minres_sol = vec![0.0f64; total];

    // CG フォールバック用
    let mut cg_ws = CgWorkspace::new(n);
    let mut use_cg = false; // true になったら全反復で CG を使用

    let mut status = SolveStatus::MaxIterations;
    let mut final_iter = options.ipm.max_iter;

    'main_loop: for iter in 0..options.ipm.max_iter {
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

        // μ = s^T y / m_ext
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
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| si / yi).collect();
        let d_vec: Vec<f64> = sigma_vec.iter().map(|&sg| sg + delta_d).collect();
        let d_inv: Vec<f64> = d_vec.iter().map(|&d| 1.0 / d).collect();

        // T2: タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // ===== MINRES パス（use_cg=false の間のみ試行）=====
        let mut minres_solved = false;

        if !use_cg {
            // Q + δ_p·I を構築して AMD LDL 分解
            let q_delta = build_q_delta(&problem.q, delta_p, n);
            let q_fac_opt = ldl::factorize_with_amd(&q_delta).ok();

            if let Some(q_fac) = q_fac_opt {
                // S = D + A(Q+δI)^{-1}A^T を構築
                let s_mat_opt =
                    build_constraint_schur(&q_fac, &a_ext, &d_vec, &timeout_ctx.cancel);

                if let Some(s_mat) = s_mat_opt {
                    if let Ok(s_fac) = ldl::factorize_with_amd(&s_mat) {
                        // --- Predictor ---
                        let r_c_pred: Vec<f64> =
                            s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
                        let r_p_mod_pred: Vec<f64> = r_p
                            .iter()
                            .zip(r_c_pred.iter())
                            .zip(y.iter())
                            .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
                            .collect();

                        minres_rhs[..n].copy_from_slice(&r_d);
                        minres_rhs[n..].copy_from_slice(&r_p_mod_pred);
                        minres_sol.iter_mut().for_each(|v| *v = 0.0);

                        let pred_res = solve_kkt_minres_constraint_precond(
                            n, m_ext, &problem.q, &q_fac, &s_fac, &a_ext, &d_vec,
                            delta_p, &minres_rhs, &mut minres_sol,
                            MINRES_MAX_ITER, super::CG_TOL,
                            timeout_ctx.deadline, &timeout_ctx.cancel,
                            &mut minres_ws,
                        );

                        if pred_res.timed_out {
                            status = SolveStatus::Timeout;
                            final_iter = iter;
                            break 'main_loop;
                        }

                        if pred_res.converged {
                            let dy_pred: Vec<f64> = minres_sol[n..].to_vec();
                            let ds_pred: Vec<f64> = r_c_pred
                                .iter()
                                .zip(sigma_vec.iter())
                                .zip(dy_pred.iter())
                                .zip(y.iter())
                                .map(|(((&rci, &sgi), &dyi), &yi)| rci / yi - sgi * dyi)
                                .collect();

                            let alpha_s_pred = fraction_to_boundary(&s, &ds_pred, super::TAU);
                            let alpha_y_pred = fraction_to_boundary(&y, &dy_pred, super::TAU);
                            let alpha_pred = alpha_s_pred.min(alpha_y_pred);
                            let mu_aff: f64 = s
                                .iter()
                                .zip(y.iter())
                                .zip(ds_pred.iter())
                                .zip(dy_pred.iter())
                                .map(|(((&si, &yi), &dsi), &dyi)| {
                                    (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
                                })
                                .sum::<f64>()
                                / m_ext as f64;
                            let sigma_center = if mu > 1e-15 {
                                (mu_aff / mu).powi(3).min(1.0)
                            } else {
                                0.0
                            };

                            // --- Corrector ---
                            let r_c_corr: Vec<f64> = s
                                .iter()
                                .zip(y.iter())
                                .zip(ds_pred.iter())
                                .zip(dy_pred.iter())
                                .map(|(((&si, &yi), &dsi), &dyi)| {
                                    sigma_center * mu - si * yi - dsi * dyi
                                })
                                .collect();
                            let r_p_mod_corr: Vec<f64> = r_p
                                .iter()
                                .zip(r_c_corr.iter())
                                .zip(y.iter())
                                .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
                                .collect();

                            minres_rhs[..n].copy_from_slice(&r_d);
                            minres_rhs[n..].copy_from_slice(&r_p_mod_corr);
                            minres_sol.iter_mut().for_each(|v| *v = 0.0);

                            let corr_res = solve_kkt_minres_constraint_precond(
                                n, m_ext, &problem.q, &q_fac, &s_fac, &a_ext, &d_vec,
                                delta_p, &minres_rhs, &mut minres_sol,
                                MINRES_MAX_ITER, super::CG_TOL,
                                timeout_ctx.deadline, &timeout_ctx.cancel,
                                &mut minres_ws,
                            );

                            if corr_res.timed_out {
                                status = SolveStatus::Timeout;
                                final_iter = iter;
                                break 'main_loop;
                            }

                            dx.copy_from_slice(&minres_sol[..n]);
                            dy.copy_from_slice(&minres_sol[n..]);
                            for i in 0..m_ext {
                                ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
                            }
                            minres_solved = true;
                        }
                        // pred_res.converged == false: fall through to CG
                    }
                }
                // s_mat_opt == None: fall through to CG
            }
            // q_fac_opt == None: fall through to CG

            if !minres_solved {
                use_cg = true;
            }
        }

        // ===== CG+Jacobi フォールバック =====
        if !minres_solved {
            let m_inv = compute_jacobi_precond_ipm(&problem.q, &a_ext, &d_inv, delta_p);

            // --- Predictor ---
            let r_c_pred: Vec<f64> =
                s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
            let r_p_mod_pred: Vec<f64> = r_p
                .iter()
                .zip(r_c_pred.iter())
                .zip(y.iter())
                .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
                .collect();
            let tmp_pred: Vec<f64> = r_p_mod_pred
                .iter()
                .zip(d_inv.iter())
                .map(|(&ri, &di)| ri * di)
                .collect();
            let mut atmp = vec![0.0f64; n];
            spmtv(&a_ext, &tmp_pred, &mut atmp);
            let rhs_x_pred: Vec<f64> = r_d
                .iter()
                .zip(atmp.iter())
                .map(|(&rdi, &ai)| rdi + ai)
                .collect();
            let mut dx_pred = vec![0.0f64; n];
            {
                let mut kv = |v: &[f64], o: &mut [f64]| {
                    mv_ipm_apply(&problem.q, &a_ext, &d_inv, delta_p, v, o);
                };
                let cg_result = pcg_solve(
                    &mut kv, &m_inv, &rhs_x_pred, &mut dx_pred,
                    super::CG_MAX_ITER, super::CG_TOL, &mut cg_ws,
                    timeout_ctx.deadline, Some(&timeout_ctx.cancel),
                );
                if cg_result.timed_out {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
            }

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
            let mu_aff: f64 = s
                .iter()
                .zip(y.iter())
                .zip(ds_pred.iter())
                .zip(dy_pred.iter())
                .map(|(((&si, &yi), &dsi), &dyi)| {
                    (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
                })
                .sum::<f64>()
                / m_ext as f64;
            let sigma_center = if mu > 1e-15 { (mu_aff / mu).powi(3).min(1.0) } else { 0.0 };

            // --- Corrector ---
            let r_c_corr: Vec<f64> = s
                .iter()
                .zip(y.iter())
                .zip(ds_pred.iter())
                .zip(dy_pred.iter())
                .map(|(((&si, &yi), &dsi), &dyi)| sigma_center * mu - si * yi - dsi * dyi)
                .collect();
            let r_p_mod_corr: Vec<f64> = r_p
                .iter()
                .zip(r_c_corr.iter())
                .zip(y.iter())
                .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
                .collect();
            let tmp_corr: Vec<f64> = r_p_mod_corr
                .iter()
                .zip(d_inv.iter())
                .map(|(&ri, &di)| ri * di)
                .collect();
            let mut atmp_corr = vec![0.0f64; n];
            spmtv(&a_ext, &tmp_corr, &mut atmp_corr);
            let rhs_x_corr: Vec<f64> = r_d
                .iter()
                .zip(atmp_corr.iter())
                .map(|(&rdi, &ai)| rdi + ai)
                .collect();
            {
                let mut kv = |v: &[f64], o: &mut [f64]| {
                    mv_ipm_apply(&problem.q, &a_ext, &d_inv, delta_p, v, o);
                };
                let cg_result = pcg_solve(
                    &mut kv, &m_inv, &rhs_x_corr, &mut dx,
                    super::CG_MAX_ITER, super::CG_TOL, &mut cg_ws,
                    timeout_ctx.deadline, Some(&timeout_ctx.cancel),
                );
                if cg_result.timed_out {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
            }

            let mut a_dx_corr = vec![0.0f64; m_ext];
            spmv(&a_ext, &dx, &mut a_dx_corr);
            for i in 0..m_ext {
                dy[i] = d_inv[i] * (a_dx_corr[i] - r_p_mod_corr[i]);
            }
            for i in 0..m_ext {
                ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
            }
        }

        // α: fraction-to-boundary (corrector)
        let alpha_s = fraction_to_boundary(&s, &ds, super::TAU);
        let alpha_y = fraction_to_boundary(&y, &dy, super::TAU);
        let alpha = alpha_s.min(alpha_y);

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
