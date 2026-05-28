//! IPM_PROF / IPPMM_SIGMA_DIAG 系の任意診断トレース。

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
