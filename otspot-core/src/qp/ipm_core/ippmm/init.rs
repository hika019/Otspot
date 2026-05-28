//! 初期点 (x, s, y) の構築: warm start 経路 + Mehrotra cold-start 射影。

use super::state::{warm_bound_margin, WARM_BOUND_REL_MARGIN};
use super::warm_start::apply_qp_warm_start;
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{factorize_kkt_with_cached_perm_par, KktConfig};
use crate::linalg::timeout::TimeoutCtx;
use crate::tolerances::UNDERFLOW_GUARD;
use faer::Par;
use crate::options::SolverOptions;
use crate::qp::ipm_core::kkt::build_augmented_system;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

pub(super) struct InitialPoint {
    pub(super) x: Vec<f64>,
    pub(super) s: Vec<f64>,
    pub(super) y: Vec<f64>,
    /// warm start 由来の μ。None なら cold start。
    pub(super) warm_mu: Option<f64>,
}

pub(super) fn build_initial_point(
    problem: &QpProblem,
    options: &SolverOptions,
    a_ext: &CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    m_orig: usize,
    m_ext: usize,
    m_ineq: usize,
    timeout_ctx: &TimeoutCtx,
    par: Par,
) -> InitialPoint {
    let n = problem.num_vars;

    // 巨大 |ub|=1e11 で midpoint 起点だと pf が抜けないため、0 が bounds 内なら 0 を優先。
    let x0: Vec<f64> = problem
        .bounds
        .iter()
        .map(|&(lb, ub)| {
            let lb_fin = lb.is_finite();
            let ub_fin = ub.is_finite();
            let zero_in_bounds = (!lb_fin || lb <= 0.0) && (!ub_fin || ub >= 0.0);
            if zero_in_bounds {
                0.0
            } else if lb_fin && ub_fin {
                (lb + ub) / 2.0
            } else if lb_fin {
                lb + 1.0
            } else if ub_fin {
                ub - 1.0
            } else {
                0.0
            }
        })
        .collect();

    // s0 = b_ext − A_ext·x0。等式行は s=0、不等式行は s≥1 にクランプ。
    let mut ax0 = vec![0.0f64; m_ext];
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax0[a_ext.row_ind[k]] += a_ext.values[k] * x0[col];
        }
    }
    let s0: Vec<f64> = b_ext
        .iter()
        .zip(ax0.iter())
        .enumerate()
        .map(|(i, (&bi, &axi))| {
            if is_eq_ext[i] { 0.0 } else { (bi - axi).max(1.0) }
        })
        .collect();
    let y0: Vec<f64> = (0..m_ext)
        .map(|i| if is_eq_ext[i] { 0.0 } else { 1.0 })
        .collect();

    let mut x = x0.clone();
    let mut s = s0;
    let mut y = y0;

    // warm start が渡されていれば Mehrotra init を skip し、interior 補正のみ適用する。
    let warm_mu = if let Some(ws) = options.warm_start_qp.as_ref() {
        apply_qp_warm_start(
            ws, problem, a_ext, b_ext, is_eq_ext, m_orig, m_ext,
            &mut x, &mut y, &mut s,
        )
    } else {
        None
    };

    if warm_mu.is_none() {
        let kkt_cfg = KktConfig {
            dd_ldl: options.ipm.dd_ldl,
            minres_ir: options.ipm.effective_minres_ir(),
            max_l_nnz: options.ipm.effective_max_l_nnz(),
        };
        mehrotra_cold_init(
            problem, a_ext, b_ext, is_eq_ext, m_ext, m_ineq,
            &ax0, timeout_ctx, par, &kkt_cfg,
            &mut x, &mut s, &mut y,
        );
    }

    InitialPoint { x, s, y, warm_mu }
}

/// Mehrotra 1992 標準初期点 (Wright §5.1): 全制約射影 + δ_s/δ_y 正補正 + Σ 均一化補正。
/// 等式のみの射影だと |b|≈1e11 級で s0 が膨張し K matrix が暴走する。
fn mehrotra_cold_init(
    problem: &QpProblem,
    a_ext: &CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    m_ext: usize,
    _m_ineq: usize,
    ax0: &[f64],
    timeout_ctx: &TimeoutCtx,
    par: Par,
    kkt_cfg: &KktConfig,
    x: &mut [f64],
    s: &mut [f64],
    y: &mut [f64],
) {
    let n = problem.num_vars;

    let r_p: Vec<f64> = b_ext.iter().zip(ax0.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    let r_p_inf = r_p.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if r_p_inf > 1e-6 && !timeout_ctx.should_stop() {
        let q_zero = CscMatrix::new(n, n);
        let sigma_zero = vec![0.0_f64; m_ext];
        let k_init = build_augmented_system(&q_zero, a_ext, &sigma_zero, 1.0, 1.0);
        let perm_init = amd_with_deadline(
            k_init.nrows, &k_init.col_ptr, &k_init.row_ind, timeout_ctx.deadline,
        );
        if let Ok(fac_init) = factorize_kkt_with_cached_perm_par(
            &k_init, &perm_init, timeout_ctx.deadline, kkt_cfg, Some(n), par,
        ) {
            let mut rhs_init = vec![0.0_f64; n + m_ext];
            rhs_init[n..(m_ext + n)].copy_from_slice(&r_p[..m_ext]);
            let mut sol_init = vec![0.0_f64; n + m_ext];
            fac_init.solve(&rhs_init, &mut sol_init);
            let dx_inf = sol_init[..n].iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            if dx_inf.is_finite() && dx_inf < 1e15 {
                for j in 0..n {
                    let x_new = x[j] + sol_init[j];
                    let (lb, ub) = problem.bounds[j];
                    x[j] = match (lb.is_finite(), ub.is_finite()) {
                        (true, true) => {
                            let range = ub - lb;
                            let margin = range * WARM_BOUND_REL_MARGIN;
                            if range > 2.0 * margin {
                                x_new.clamp(lb + margin, ub - margin)
                            } else {
                                0.5 * (lb + ub)
                            }
                        }
                        (true, false) => x_new.max(lb + warm_bound_margin(lb)),
                        (false, true) => x_new.min(ub - warm_bound_margin(ub)),
                        (false, false) => x_new,
                    };
                }
            }
        }
    }

    let mut ax_new = vec![0.0_f64; m_ext];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax_new[a_ext.row_ind[k]] += a_ext.values[k] * x[col];
        }
    }
    let s_hat: Vec<f64> = b_ext.iter().zip(ax_new.iter()).enumerate()
        .map(|(i, (&bi, &axi))| if is_eq_ext[i] { 0.0 } else { bi - axi })
        .collect();
    let y_hat: Vec<f64> = (0..m_ext)
        .map(|i| if is_eq_ext[i] { 0.0 } else { 1.0 })
        .collect();

    let s_min_ineq = s_hat.iter().zip(is_eq_ext.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
        .fold(f64::INFINITY, f64::min);
    let y_min_ineq = y_hat.iter().zip(is_eq_ext.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
        .fold(f64::INFINITY, f64::min);
    let delta_s = (-1.5 * s_min_ineq).max(0.0) + 1.0;
    let delta_y = (-1.5 * y_min_ineq).max(0.0) + 1.0;

    let s_pos: Vec<f64> = s_hat.iter().enumerate()
        .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_s })
        .collect();
    let y_pos: Vec<f64> = y_hat.iter().enumerate()
        .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_y })
        .collect();

    let sy_sum: f64 = s_pos.iter().zip(y_pos.iter()).map(|(&si, &yi)| si * yi).sum();
    let s_sum_pos: f64 = s_pos.iter().sum();
    let y_sum_pos: f64 = y_pos.iter().sum();
    let delta_s_corr = if y_sum_pos > UNDERFLOW_GUARD { sy_sum / (2.0 * y_sum_pos) } else { 0.0 };
    let delta_y_corr = if s_sum_pos > UNDERFLOW_GUARD { sy_sum / (2.0 * s_sum_pos) } else { 0.0 };

    for i in 0..m_ext {
        s[i] = if is_eq_ext[i] { 0.0 } else { s_pos[i] + delta_s_corr };
        y[i] = if is_eq_ext[i] { 0.0 } else { y_pos[i] + delta_y_corr };
    }

}
