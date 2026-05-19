//! Env-gated 診断トレース (IPPMM_TRACE / IPPMM_ACTIVE_TRACE / IPPMM_SIGMA_DIAG)。

use super::state::PmmState;
use crate::problem::SolveStatus;

fn trace_enabled() -> bool {
    std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1")
}

pub(super) fn emit_exit_optimal_main(iter: usize, nr_p_rel: f64, nr_d_rel: f64, rel_gap: f64) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_EXIT iter={} path=Optimal_main pf_rel={:.3e} df_rel={:.3e} rel_gap={:.3e}",
        iter, nr_p_rel, nr_d_rel, rel_gap
    );
}

pub(super) fn emit_exit_nan_guard(
    iter: usize, is_quasi_optimal: bool, best_iter: usize,
    best_score: f64, best_rel_gap: f64, best_residuals: (f64, f64, f64),
) {
    if !trace_enabled() { return; }
    let path_label = if is_quasi_optimal {
        "Optimal_NaN_guard_bestsofar"
    } else {
        "SuboptimalSolution_NaN_guard_diverged_bestsofar"
    };
    eprintln!(
        "IPPMM_EXIT iter={} path={} best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
        iter, path_label, best_iter, best_score, best_rel_gap,
        best_residuals.0, best_residuals.1, best_residuals.2
    );
}

pub(super) fn emit_exit_nan_guard_no_best(iter: usize) {
    if !trace_enabled() { return; }
    eprintln!("IPPMM_EXIT iter={} path=NumericalError_NaN_guard_no_best", iter);
}

pub(super) fn emit_debug_infeas_meta(
    iter: usize, best_score: f64, quality_threshold: f64,
    eps_orig: f64, eps: f64, best_finite: bool, consecutive_infeas: usize,
) {
    if !trace_enabled() { return; }
    eprintln!("IPPMM_DEBUG iter={} best_score={:e} quality_threshold={:e} eps_orig={:e} eps={:e} best_finite={} consecutive_infeas={}",
        iter, best_score, quality_threshold, eps_orig, eps, best_finite, consecutive_infeas);
}

pub(super) fn emit_exit_reject_false_infeas(
    iter: usize, infeas_status: &SolveStatus, best_iter: usize,
    best_score: f64, best_rel_gap: f64, best_residuals: (f64, f64, f64),
) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_EXIT iter={} path=reject_false_{:?}_bestsofar best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
        iter, infeas_status, best_iter, best_score, best_rel_gap,
        best_residuals.0, best_residuals.1, best_residuals.2
    );
}

pub(super) fn emit_debug_infeas_continue(iter: usize, consecutive: usize, min_consecutive: usize) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_DEBUG iter={} infeas trigger #{} (< {}), continue iterating",
        iter, consecutive, min_consecutive
    );
}

pub(super) fn emit_exit_demote_to_suboptimal(
    iter: usize, infeas_status: &SolveStatus, best_iter: usize,
    best_score: f64, best_residuals: (f64, f64, f64), consecutive: usize,
) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_EXIT iter={} path=demote_{:?}_to_suboptimal_bestsofar best_iter={} best_score={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e}) consecutive={}",
        iter, infeas_status, best_iter, best_score,
        best_residuals.0, best_residuals.1, best_residuals.2, consecutive
    );
}

pub(super) fn emit_exit_check_infeas(
    iter: usize, infeas_status: &SolveStatus, best_score: f64, consecutive: usize,
) {
    if !trace_enabled() { return; }
    eprintln!("IPPMM_EXIT iter={} path=check_infeas status={:?} best_score={:.3e} consecutive={}",
        iter, infeas_status, best_score, consecutive);
}

pub(super) fn emit_step_diag(
    iter: usize, alpha: f64, ndx: f64, ndy: f64, nds: f64,
    r_d_pmm: &[f64], r_p_pmm: &[f64],
) {
    if !trace_enabled() { return; }
    let nrdpmm = r_d_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let nrppmm = r_p_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    eprintln!(
        "IPPMM_STEP iter={:4} alpha={:.6e} dx_inf={:.3e} dy_inf={:.3e} ds_inf={:.3e} rdpmm_inf={:.3e} rppmm_inf={:.3e}",
        iter, alpha, ndx, ndy, nds, nrdpmm, nrppmm
    );
}

pub(super) fn emit_exit_alpha_stall(
    iter: usize, alpha_stall_converged: bool, alpha_stall_count: usize, best_iter: usize,
    best_score: f64, best_rel_gap: f64, rho: f64, reg_limit: f64,
    best_residuals: (f64, f64, f64),
) {
    if !trace_enabled() { return; }
    let exit_reason = if alpha_stall_converged { "conv" } else { "deadlock" };
    eprintln!(
        "IPPMM_EXIT iter={} path=Suboptimal_alpha_stall_bestsofar reason={} stall_count={} best_iter={} best_score={:.3e} best_rel_gap={:.3e} rho={:.3e} reg_limit={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
        iter, exit_reason, alpha_stall_count, best_iter, best_score, best_rel_gap,
        rho, reg_limit,
        best_residuals.0, best_residuals.1, best_residuals.2
    );
}

pub(super) fn emit_exit_residual_stall(
    iter: usize, window: usize, last_improve_iter: usize, best_iter: usize,
    best_score: f64, best_rel_gap: f64, best_residuals: (f64, f64, f64),
) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_EXIT iter={} path=Suboptimal_residual_stall_bestsofar window={} last_improve_iter={} best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
        iter, window, last_improve_iter, best_iter, best_score, best_rel_gap,
        best_residuals.0, best_residuals.1, best_residuals.2
    );
}

pub(super) fn emit_exit_timeout_bestsofar(
    best_iter: usize, best_score: f64, best_rel_gap: f64, best_residuals: (f64, f64, f64),
) {
    if !trace_enabled() { return; }
    eprintln!(
        "IPPMM_EXIT path=Timeout_bestsofar_fallback best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
        best_iter, best_score, best_rel_gap,
        best_residuals.0, best_residuals.1, best_residuals.2
    );
}

pub(super) fn emit_prof_summary(
    prof_iters: usize, residual_ns: u128, factor_ns: u128, predcorr_ns: u128,
    gondzio_ns: u128, update_ns: u128, other_ns: u128,
) {
    let total_ns = residual_ns + factor_ns + predcorr_ns + gondzio_ns + update_ns + other_ns;
    let total_ms = total_ns as f64 / 1_000_000.0;
    let frac = |v: u128| -> f64 { 100.0 * v as f64 / total_ns.max(1) as f64 };
    eprintln!(
        "IPM_PROF iters={} total={:.1}ms residual={:.1}ms({:.1}%) factor={:.1}ms({:.1}%) predcorr={:.1}ms({:.1}%) gondzio={:.1}ms({:.1}%) update={:.1}ms({:.1}%)",
        prof_iters, total_ms,
        residual_ns as f64 / 1e6, frac(residual_ns),
        factor_ns as f64 / 1e6, frac(factor_ns),
        predcorr_ns as f64 / 1e6, frac(predcorr_ns),
        gondzio_ns as f64 / 1e6, frac(gondzio_ns),
        update_ns as f64 / 1e6, frac(update_ns),
    );
}

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
