//! Env-gated 診断トレース (IPPMM_TRACE / IPPMM_ACTIVE_TRACE / IPPMM_SIGMA_DIAG)。

use super::state::PmmState;

pub(super) fn emit_iter_trace(
    iter: usize, mu: f64, nr_p: f64, nr_d: f64,
    pmm: &PmmState, x: &[f64], y: &[f64], reg_limit: f64,
) {
    if std::env::var("IPPMM_TRACE").ok().as_deref() != Some("1") {
        return;
    }
    let prox_d_inf = x.iter().zip(pmm.x_ref.iter())
        .map(|(&xi, &xref)| (pmm.rho * (xi - xref)).abs())
        .fold(0.0_f64, f64::max);
    let prox_p_inf = y.iter().zip(pmm.y_ref.iter())
        .map(|(&yi, &yref)| (pmm.delta * (yi - yref)).abs())
        .fold(0.0_f64, f64::max);
    let diff_x_inf = x.iter().zip(pmm.x_ref.iter())
        .map(|(&xi, &xref)| (xi - xref).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "IPPMM_TRACE iter={:4} mu={:.3e} pf={:.3e} df={:.3e} rho={:.3e} delta={:.3e} prox_d_inf={:.3e} prox_p_inf={:.3e} diff_x_inf={:.3e} reg_limit={:.3e}",
        iter, mu, nr_p, nr_d, pmm.rho, pmm.delta, prox_d_inf, prox_p_inf, diff_x_inf, reg_limit
    );
}

pub(super) fn emit_active_trace(
    iter: usize, m_ineq: usize, s: &[f64], y: &[f64], is_eq_ext: &[bool],
) {
    if std::env::var("IPPMM_ACTIVE_TRACE").ok().as_deref() != Some("1") {
        return;
    }
    let s_inf = s.iter().zip(is_eq_ext.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v.abs()) })
        .fold(0.0_f64, f64::max).max(1e-300);
    let y_inf = y.iter().zip(is_eq_ext.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v.abs()) })
        .fold(0.0_f64, f64::max).max(1e-300);
    let s_small_abs = s.iter().zip(is_eq_ext.iter())
        .filter(|(_, &eq)| !eq).filter(|(&v, _)| v < 1e-6).count();
    let s_small_rel = s.iter().zip(is_eq_ext.iter())
        .filter(|(_, &eq)| !eq).filter(|(&v, _)| v < 1e-6 * s_inf).count();
    let y_large_abs = y.iter().zip(is_eq_ext.iter())
        .filter(|(_, &eq)| !eq).filter(|(&v, _)| v > 1e-6).count();
    let y_large_rel = y.iter().zip(is_eq_ext.iter())
        .filter(|(_, &eq)| !eq).filter(|(&v, _)| v > 1e-6 * y_inf).count();
    eprintln!(
        "IPPMM_ACTIVE iter={:4} m_ineq={} s_inf={:.3e} y_inf={:.3e} s<1e-6={} s<1e-6*smax={} y>1e-6={} y>1e-6*ymax={}",
        iter, m_ineq, s_inf, y_inf, s_small_abs, s_small_rel, y_large_abs, y_large_rel
    );
}

pub(super) fn emit_sigma_diag(
    iter: usize, mu: f64, sigma_vec: &[f64],
    s: &[f64], y: &[f64], is_eq_ext: &[bool],
) {
    if std::env::var("IPPMM_SIGMA_DIAG").ok().as_deref() != Some("1") {
        return;
    }
    let mut sigma_min = f64::INFINITY;
    let mut sigma_max_actual = 0.0_f64;
    let mut s_min = f64::INFINITY;
    let mut s_max = 0.0_f64;
    let mut y_min = f64::INFINITY;
    let mut y_max = 0.0_f64;
    for (i, &sig) in sigma_vec.iter().enumerate() {
        if !is_eq_ext[i] {
            if sig > 0.0 && sig.is_finite() {
                sigma_min = sigma_min.min(sig);
                sigma_max_actual = sigma_max_actual.max(sig);
            }
            if s[i] > 0.0 { s_min = s_min.min(s[i]); s_max = s_max.max(s[i]); }
            if y[i] > 0.0 { y_min = y_min.min(y[i]); y_max = y_max.max(y[i]); }
        }
    }
    eprintln!(
        "IPPMM_SIGMA iter={} mu={:.3e} Σ:[{:.3e},{:.3e}] range={:.3e} s:[{:.3e},{:.3e}] y:[{:.3e},{:.3e}]",
        iter, mu, sigma_min, sigma_max_actual, sigma_max_actual / sigma_min.max(1e-300),
        s_min, s_max, y_min, y_max
    );
}
