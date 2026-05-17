//! Predictor-Corrector-Gondzio 共通ループ部品。

use crate::linalg::kkt_solver::KktFactor;
use crate::sparse::CscMatrix;
use super::common::fraction_to_boundary_masked;
use super::{TAU, BETA_GONDZIO, GAMMA_L, GAMMA_U, ALPHA_IMPROVE_THRESHOLD};

/// 標準 f64 IR の反復上限。3 を超えると LDL 精度限界で利得が出ないため。
pub(crate) const IR_MAX_ITERS: usize = 3;

/// DD 残差経路の IR 上限 (LDL precision 限界を反復で突破するため拡張)。
pub(crate) const IR_MAX_ITERS_DD: usize = 10;

/// Dekker/Knuth two-sum: a + b = s + e (s, e は f64、|e| ≤ ulp(s)/2)。
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// FMA-based two-product: a * b = hi + lo (真値一致)。
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    (hi, lo)
}

/// double-double 残差 r = rhs − K·sol。LDL precision floor を Wilkinson IR で eps_residual まで詰める。
fn compute_residual_dd(aug_mat: &CscMatrix, sol: &[f64], rhs: &[f64], out: &mut [f64]) {
    let n = sol.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(out.len(), n);

    let mut hi = vec![0.0_f64; n];
    let mut lo = vec![0.0_f64; n];

    for i in 0..n {
        hi[i] = rhs[i];
    }

    for col in 0..aug_mat.ncols {
        let xv_c = sol[col];
        for ptr in aug_mat.col_ptr[col]..aug_mat.col_ptr[col + 1] {
            let row = aug_mat.row_ind[ptr];
            let val = aug_mat.values[ptr];
            let (p_hi, p_lo) = two_prod(val, xv_c);
            let (s, e1) = two_sum(hi[row], -p_hi);
            let (s2, e2) = two_sum(lo[row], e1 - p_lo);
            hi[row] = s;
            lo[row] = s2 + e2;

            if row != col {
                let xv_r = sol[row];
                let (p_hi2, p_lo2) = two_prod(val, xv_r);
                let (s3, e3) = two_sum(hi[col], -p_hi2);
                let (s4, e4) = two_sum(lo[col], e3 - p_lo2);
                hi[col] = s3;
                lo[col] = s4 + e4;
            }
        }
    }

    for i in 0..n {
        out[i] = hi[i] + lo[i];
    }
}

/// aug_mat は fac に factorize された対称上三角 CSC である必要がある。
pub(crate) fn solve_with_iterative_refinement(
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    rhs: &[f64],
    sol: &mut [f64],
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) {
    let n = sol.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(aug_mat.nrows, n);
    debug_assert_eq!(aug_mat.ncols, n);

    fac.solve_with_deadline(rhs, sol, deadline);

    if max_iters == 0 {
        return;
    }
    // 反復 backend は自分で tol まで収束するため IR を被せると deadline 浪費。
    if fac.is_iterative() {
        return;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }

    let use_dd_residual = std::env::var("IR_DD").ok().as_deref() == Some("1");
    let max_iters = if use_dd_residual { max_iters.max(IR_MAX_ITERS_DD) } else { max_iters };

    // 大型 (n+m_ext > 100k) は IR overhead が deadline を圧迫するため skip。
    const IR_SKIP_LARGE_THRESHOLD: usize = 100_000;
    if n > IR_SKIP_LARGE_THRESHOLD {
        return;
    }

    let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
    let use_dd_residual = std::env::var("IR_DD").ok().as_deref() == Some("1");
    let resid_skip_threshold = if use_dd_residual {
        rhs_inf * 1e-30
    } else {
        rhs_inf * 1e-13
    };

    let mut kx = vec![0.0_f64; n];
    let mut residual = vec![0.0_f64; n];
    let mut correction = vec![0.0_f64; n];

    let trace_ir = std::env::var("IR_TRACE").ok().as_deref() == Some("1");
    if trace_ir {
        let sol_inf_initial = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        eprintln!("IR_START n={} rhs_inf={:.3e} sol_inf={:.3e} thr={:.3e} dd={}",
            n, rhs_inf, sol_inf_initial, resid_skip_threshold, use_dd_residual);
    }
    for _ir_iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        if use_dd_residual {
            compute_residual_dd(aug_mat, sol, rhs, &mut residual);
        } else {
            for v in kx.iter_mut() {
                *v = 0.0;
            }
            for col in 0..aug_mat.ncols {
                for ptr in aug_mat.col_ptr[col]..aug_mat.col_ptr[col + 1] {
                    let row = aug_mat.row_ind[ptr];
                    let val = aug_mat.values[ptr];
                    kx[row] += val * sol[col];
                    if row != col {
                        kx[col] += val * sol[row];
                    }
                }
            }
            for i in 0..n {
                residual[i] = rhs[i] - kx[i];
            }
        }

        let mut resid_inf = 0.0_f64;
        for i in 0..n {
            resid_inf = resid_inf.max(residual[i].abs());
        }
        if resid_inf <= resid_skip_threshold {
            if trace_ir { eprintln!("IR iter={} EXIT_resid_small resid_inf={:.3e}", _ir_iter, resid_inf); }
            return;
        }

        for v in correction.iter_mut() {
            *v = 0.0;
        }
        fac.solve_with_deadline(&residual, &mut correction, deadline);

        // Backtrack guard: NaN/Inf protection
        let any_bad = correction.iter().any(|v| !v.is_finite());
        if any_bad {
            if trace_ir { eprintln!("IR iter={} EXIT_nan resid_inf={:.3e}", _ir_iter, resid_inf); }
            return;
        }

        // correction が sol を超える = LDL 精度限界の虚偽補正 → IR を打ち切る。
        let correction_inf = correction.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let sol_inf = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        if trace_ir {
            eprintln!("IR iter={} resid_inf={:.3e} correction_inf={:.3e} sol_inf={:.3e}", _ir_iter, resid_inf, correction_inf, sol_inf);
        }
        if correction_inf > sol_inf {
            if trace_ir { eprintln!("IR iter={} EXIT_correction_too_large", _ir_iter); }
            return;
        }

        for i in 0..n {
            sol[i] += correction[i];
        }
    }
}

pub(crate) struct PredictorResult {
    pub dy_pred: Vec<f64>,
    pub ds_pred: Vec<f64>,
    pub sigma_center: f64,
}

/// Σ = diag(s_i / y_i)。等式行は 0、nan/inf は sigma_max でクランプ。
pub(crate) fn compute_sigma_vec(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    sigma_max: f64,
) -> Vec<f64> {
    s.iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                let v = si / yi;
                if v.is_finite() { v } else { sigma_max }
            }
        })
        .collect()
}

/// Schur 経由で KKT step を解く: S·dx = r_d + AᵀD⁻¹r_p、続いて dy = D⁻¹(A·dx − r_p)。
/// active constraints で |D⁻¹| 増幅による cancellation が df を発散させるため、
/// dy の back-substitution は TwoFloat (DD ≈106 bit) で計算する。
pub(crate) fn solve_kkt_via_schur(
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    r_d: &[f64],
    r_p_mod: &[f64],
    dx_out: &mut [f64],
    dy_out: &mut [f64],
) {
    use super::kkt::spmtv;
    use twofloat::TwoFloat;

    let n = r_d.len();
    let m_ext = r_p_mod.len();

    let mut d_inv_rp = vec![0.0_f64; m_ext];
    for i in 0..m_ext {
        d_inv_rp[i] = d_inv[i] * r_p_mod[i];
    }
    let mut at_d_inv_rp = vec![0.0_f64; n];
    spmtv(a_ext, &d_inv_rp, &mut at_d_inv_rp);
    let rhs_s: Vec<f64> = r_d
        .iter()
        .zip(at_d_inv_rp.iter())
        .map(|(&r, &v)| r + v)
        .collect();

    s_fac.solve(&rhs_s, dx_out);
    let _ = s_mat;
    let _ = n;

    let zero_dd = TwoFloat::from(0.0);
    let mut a_dx_dd: Vec<TwoFloat> = vec![zero_dd; m_ext];
    for col in 0..n {
        let cs = a_ext.col_ptr[col];
        let ce = a_ext.col_ptr[col + 1];
        let dx_col = dx_out[col];
        for k in cs..ce {
            let row = a_ext.row_ind[k];
            let v = a_ext.values[k];
            a_dx_dd[row] = a_dx_dd[row] + TwoFloat::new_mul(v, dx_col);
        }
    }
    for i in 0..m_ext {
        let diff_dd = a_dx_dd[i] - TwoFloat::from(r_p_mod[i]);
        let scaled = diff_dd * TwoFloat::from(d_inv[i]);
        dy_out[i] = f64::from(scaled);
    }
}


#[allow(clippy::too_many_arguments)]
pub(crate) fn predictor_step_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    mu: f64,
) -> PredictorResult {
    let r_c_pred: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi })
        .collect();

    let r_p_mod_pred: Vec<f64> = r_primal
        .iter()
        .zip(r_c_pred.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    let mut dx = vec![0.0_f64; n];
    let mut dy_pred = vec![0.0_f64; m_ext];
    solve_kkt_via_schur(s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_pred, &mut dx, &mut dy_pred);

    let mut ds_pred = vec![0.0_f64; m_ext];
    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds_pred[i] = 0.0;
        } else {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }
    }

    let alpha_s_pred = fraction_to_boundary_masked(s, &ds_pred, TAU, is_eq_ext);
    let alpha_y_pred = fraction_to_boundary_masked(y, &dy_pred, TAU, is_eq_ext);
    let alpha_pred = alpha_s_pred.min(alpha_y_pred);

    let mu_aff: f64 = if m_ineq > 0 {
        s.iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .enumerate()
            .filter(|&(i, _)| !is_eq_ext[i])
            .map(|(_, (((&si, &yi), &dsi), &dyi))| {
                (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
            })
            .sum::<f64>()
            / m_ineq as f64
    } else {
        0.0
    };

    let sigma_center = if mu > 1e-15 {
        (mu_aff / mu).powi(3).min(1.0)
    } else {
        0.0
    };

    PredictorResult {
        dy_pred,
        ds_pred,
        sigma_center,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector_step_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    pred: &PredictorResult,
    mu: f64,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> (f64, Vec<f64>) {
    let r_c_corr: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .zip(pred.ds_pred.iter())
        .zip(pred.dy_pred.iter())
        .enumerate()
        .map(|(i, (((&si, &yi), &dsi), &dyi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                pred.sigma_center * mu - si * yi - dsi * dyi
            }
        })
        .collect();

    let r_p_mod_corr: Vec<f64> = r_primal
        .iter()
        .zip(r_c_corr.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    solve_kkt_via_schur(s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_corr, dx, dy);
    let _ = n;

    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds[i] = 0.0;
        } else {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }
    }

    let alpha_s = fraction_to_boundary_masked(s, ds, TAU, is_eq_ext);
    let alpha_y = fraction_to_boundary_masked(y, dy, TAU, is_eq_ext);
    let alpha = alpha_s.min(alpha_y);

    (alpha, r_c_corr)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gondzio_correctors_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    r_c_corr: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> f64 {
    let mut alpha_prev = alpha_init;
    for _k in 0..max_correctors {
        let alpha_target = (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
        let mu_target: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(ds.iter().zip(dy.iter()))
                .enumerate()
                .filter(|&(i, _)| !is_eq_ext[i])
                .map(|(_, ((&si, &yi), (&dsi, &dyi)))| {
                    (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                })
                .sum::<f64>()
                / m_ineq as f64
        } else {
            0.0
        };
        let mu_target = mu_target.max(0.0);

        let target_lo = GAMMA_L * mu_target;
        let target_hi = GAMMA_U * mu_target;

        let mut r_c_gondzio = vec![0.0_f64; m_ext];
        for i in 0..m_ext {
            if is_eq_ext[i] {
                r_c_gondzio[i] = 0.0;
                continue;
            }
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

        let r_p_mod_gondzio: Vec<f64> = r_primal
            .iter()
            .zip(r_c_gondzio.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        let mut dx_new = vec![0.0_f64; n];
        let mut dy_new = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_gondzio, &mut dx_new, &mut dy_new,
        );

        let ds_new: Vec<f64> = (0..m_ext)
            .map(|i| {
                if is_eq_ext[i] {
                    0.0
                } else {
                    r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i]
                }
            })
            .collect();

        let alpha_s_new = fraction_to_boundary_masked(s, &ds_new, TAU, is_eq_ext);
        let alpha_y_new = fraction_to_boundary_masked(y, &dy_new, TAU, is_eq_ext);
        let alpha_new = alpha_s_new.min(alpha_y_new);

        if alpha_new <= alpha_prev * ALPHA_IMPROVE_THRESHOLD {
            break;
        }
        alpha_prev = alpha_new;
        dx.copy_from_slice(&dx_new);
        dy.copy_from_slice(&dy_new);
        ds.copy_from_slice(&ds_new);
    }
    alpha_prev
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predictor_step(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    mu: f64,
    deadline: Option<std::time::Instant>,
) -> PredictorResult {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    let r_c_pred: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi })
        .collect();

    let r_p_mod_pred: Vec<f64> = r_primal
        .iter()
        .zip(r_c_pred.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    rhs[..n].copy_from_slice(r_dual);
    rhs[n..].copy_from_slice(&r_p_mod_pred);
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

    let dy_pred = sol[n..].to_vec();

    let mut ds_pred = vec![0.0f64; m_ext];
    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds_pred[i] = 0.0;
        } else {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }
    }

    let alpha_s_pred = fraction_to_boundary_masked(s, &ds_pred, TAU, is_eq_ext);
    let alpha_y_pred = fraction_to_boundary_masked(y, &dy_pred, TAU, is_eq_ext);
    let alpha_pred = alpha_s_pred.min(alpha_y_pred);

    let mu_aff: f64 = if m_ineq > 0 {
        s.iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .enumerate()
            .filter(|&(i, _)| !is_eq_ext[i])
            .map(|(_, (((&si, &yi), &dsi), &dyi))| {
                (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
            })
            .sum::<f64>()
            / m_ineq as f64
    } else {
        0.0
    };

    let sigma_center = if mu > 1e-15 {
        (mu_aff / mu).powi(3).min(1.0)
    } else {
        0.0
    };

    PredictorResult {
        dy_pred,
        ds_pred,
        sigma_center,
    }
}

/// 戻り値 r_c_corr は続く Gondzio correctors に渡す。
#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector_step(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    pred: &PredictorResult,
    mu: f64,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
) -> (f64, Vec<f64>) {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    let r_c_corr: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .zip(pred.ds_pred.iter())
        .zip(pred.dy_pred.iter())
        .enumerate()
        .map(|(i, (((&si, &yi), &dsi), &dyi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                pred.sigma_center * mu - si * yi - dsi * dyi
            }
        })
        .collect();

    let r_p_mod_corr: Vec<f64> = r_primal
        .iter()
        .zip(r_c_corr.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    rhs[..n].copy_from_slice(r_dual);
    rhs[n..].copy_from_slice(&r_p_mod_corr);
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

    dx.copy_from_slice(&sol[..n]);
    dy.copy_from_slice(&sol[n..]);

    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds[i] = 0.0;
        } else {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }
    }

    let alpha_s = fraction_to_boundary_masked(s, ds, TAU, is_eq_ext);
    let alpha_y = fraction_to_boundary_masked(y, dy, TAU, is_eq_ext);
    let alpha = alpha_s.min(alpha_y);

    (alpha, r_c_corr)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gondzio_correctors(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    r_c_corr: &[f64],
    sigma_vec: &[f64],
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
) -> f64 {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    let mut alpha_prev = alpha_init;
    for _k in 0..max_correctors {
        let alpha_target = (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
        let mu_target: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(ds.iter().zip(dy.iter()))
                .enumerate()
                .filter(|&(i, _)| !is_eq_ext[i])
                .map(|(_, ((&si, &yi), (&dsi, &dyi)))| {
                    (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                })
                .sum::<f64>()
                / m_ineq as f64
        } else {
            0.0
        };
        let mu_target = mu_target.max(0.0);

        let target_lo = GAMMA_L * mu_target;
        let target_hi = GAMMA_U * mu_target;

        let mut r_c_gondzio = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            if is_eq_ext[i] {
                r_c_gondzio[i] = 0.0;
                continue;
            }
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

        let r_p_mod_gondzio: Vec<f64> = r_primal
            .iter()
            .zip(r_c_gondzio.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        rhs[..n].copy_from_slice(r_dual);
        rhs[n..].copy_from_slice(&r_p_mod_gondzio);
        solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

        let dx_new = sol[..n].to_vec();
        let dy_new = sol[n..].to_vec();
        let ds_new: Vec<f64> = (0..m_ext)
            .map(|i| {
                if is_eq_ext[i] {
                    0.0
                } else {
                    r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i]
                }
            })
            .collect();

        let alpha_s_new = fraction_to_boundary_masked(s, &ds_new, TAU, is_eq_ext);
        let alpha_y_new = fraction_to_boundary_masked(y, &dy_new, TAU, is_eq_ext);
        let alpha_new = alpha_s_new.min(alpha_y_new);

        if alpha_new < alpha_prev + ALPHA_IMPROVE_THRESHOLD {
            break;
        }

        dx.copy_from_slice(&dx_new);
        dy.copy_from_slice(&dy_new);
        ds.copy_from_slice(&ds_new);
        alpha_prev = alpha_new;
    }
    alpha_prev
}

/// 等式行は s=0 維持で y のみ更新、不等式行は s,y 両方を 1e-12 floor で更新。
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_variables(
    x: &mut [f64],
    s: &mut [f64],
    y: &mut [f64],
    dx: &[f64],
    ds: &[f64],
    dy: &[f64],
    alpha: f64,
    is_eq_ext: &[bool],
) {
    for i in 0..x.len() {
        x[i] += alpha * dx[i];
    }
    let m_ext = s.len();
    for i in 0..m_ext {
        if is_eq_ext[i] {
            y[i] += alpha * dy[i];
        } else {
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
}

#[cfg(test)]
mod tests {
    use super::{compute_sigma_vec, update_variables, solve_kkt_via_schur};
    use crate::qp::ipm_core::kkt::{build_augmented_system, build_schur_system};
    use crate::linalg::amd::amd_with_deadline;
    use crate::sparse::CscMatrix;

    /// DD 残差は f64 では相殺で消える値を exact に返すこと。
    #[test]
    fn test_dd_residual_precision_vs_f64() {
        use super::compute_residual_dd;
        let n = 2;
        let a = 1.0_f64 + 1e-16;
        let mat = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[a, 1.0],
            n, n,
        ).unwrap();
        let sol = vec![1.0_f64, 1.0_f64];
        let rhs = vec![a, 1.0_f64];
        let mut r_dd = vec![0.0_f64; n];
        compute_residual_dd(&mat, &sol, &rhs, &mut r_dd);
        for &v in &r_dd {
            assert!(v.abs() < 1e-30, "got {:e}", v);
        }
    }

    /// 多制約・等式・不等式混在で σ 幅広い range の下、Schur が augmented と数値一致すること。
    #[test]
    fn test_schur_matches_augmented_realistic() {
        let n = 4;
        let m_ext = 6;

        let q = CscMatrix::from_triplets(
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[2.0, 4.0, 0.5, 1.0],
            n, n,
        ).unwrap();

        // A_ext: 6×4 にいくつかの非ゼロ
        // 行 0: x0 + x1 (eq)
        // 行 1: x2 + x3 (eq)
        // 行 2: x0 (lb)
        // 行 3: x1 (lb)
        // 行 4: -x2 (ub)
        // 行 5: x0 + x3 (mixed)
        let rows = vec![0, 0, 1, 1, 2, 3, 4, 5, 5];
        let cols = vec![0, 1, 2, 3, 0, 1, 2, 0, 3];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
        let a_ext = CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap();

        // sigma: equality は 0、inequality は様々な値 (LISWET 風 dynamic range)
        let sigma_vec = vec![0.0, 0.0, 1e-3, 1e1, 1e3, 5e-2];
        let rho_p = 0.05_f64;
        let delta_d = 0.02_f64;

        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let aug_perm = amd_with_deadline(aug_mat.nrows, &aug_mat.col_ptr, &aug_mat.row_ind, None);
        let aug_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&aug_mat, &aug_perm, None).unwrap());

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm = amd_with_deadline(s_mat.nrows, &s_mat.col_ptr, &s_mat.row_ind, None);
        let s_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&s_mat, &s_perm, None).unwrap());

        let r_d = vec![0.5, -1.0, 0.2, 0.8];
        let r_p_mod = vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6];

        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug);
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            &s_fac, &s_mat, &d_inv, &a_ext, &r_d, &r_p_mod,
            &mut dx_schur, &mut dy_schur,
        );

        eprintln!("dx_aug   = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug   = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            let diff = (dx_aug[i] - dx_schur[i]).abs();
            let scale = dx_aug[i].abs().max(dx_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dx[{}]: aug={}, schur={}, rel_diff={}",
                i, dx_aug[i], dx_schur[i], diff / scale
            );
        }
        for i in 0..m_ext {
            let diff = (dy_aug[i] - dy_schur[i]).abs();
            let scale = dy_aug[i].abs().max(dy_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dy[{}]: aug={}, schur={}, rel_diff={}",
                i, dy_aug[i], dy_schur[i], diff / scale
            );
        }
    }

    /// 2 変数 1 制約で Schur と augmented LDL が同じ (dx, dy) を出すこと。
    #[test]
    fn test_schur_matches_augmented() {
        let n = 2;
        let m_ext = 1;

        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0, 4.0],
            n, n,
        ).unwrap();

        let a_ext = CscMatrix::from_triplets(
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            m_ext, n,
        ).unwrap();

        let sigma_vec = vec![0.5_f64];
        let rho_p = 0.1_f64;
        let delta_d = 0.05_f64;

        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let perm: Vec<usize> = (0..aug_mat.nrows).collect();
        let aug_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&aug_mat, &perm, None).unwrap());

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm: Vec<usize> = amd_with_deadline(s_mat.nrows, &s_mat.col_ptr, &s_mat.row_ind, None);
        let s_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&s_mat, &s_perm, None).unwrap());

        let r_d = vec![1.0, 2.0];
        let r_p_mod = vec![3.0];

        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug);
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            &s_fac, &s_mat, &d_inv, &a_ext, &r_d, &r_p_mod,
            &mut dx_schur, &mut dy_schur,
        );

        eprintln!("dx_aug = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            assert!(
                (dx_aug[i] - dx_schur[i]).abs() < 1e-9,
                "dx[{}]: aug={}, schur={}", i, dx_aug[i], dx_schur[i]
            );
        }
        for i in 0..m_ext {
            assert!(
                (dy_aug[i] - dy_schur[i]).abs() < 1e-9,
                "dy[{}]: aug={}, schur={}", i, dy_aug[i], dy_schur[i]
            );
        }
    }


    #[test]
    fn test_compute_sigma_vec_eq_row_is_zero() {
        let s = vec![2.0, 4.0];
        let y = vec![1.0, 2.0];
        let is_eq_ext = vec![true, false];
        let sigma_max = 1e6_f64;
        let result = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 2.0).abs() < 1e-12, "got {}", result[1]);
    }

    #[test]
    fn test_update_variables_alpha_one() {
        let mut x = vec![1.0, 2.0];
        let mut s = vec![0.5, 0.5];
        let mut y = vec![1.0, 1.0];
        let dx = vec![0.1, 0.2];
        let ds = vec![0.3, -0.6];
        let dy = vec![0.1, 0.1];
        let is_eq_ext = vec![false, false];
        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, 1.0, &is_eq_ext);
        assert!((x[0] - 1.1).abs() < 1e-12);
        assert!((x[1] - 2.2).abs() < 1e-12);
        assert!((s[0] - 0.8).abs() < 1e-12);
        assert_eq!(s[1], 1e-12);
        assert!((y[0] - 1.1).abs() < 1e-12);
        assert!((y[1] - 1.1).abs() < 1e-12);
    }
}
