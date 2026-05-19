//! presolve スケーリング由来の残差増幅を打ち消すため IPM 内部 eps を σ_total で厳格化。

use crate::options::SolverOptions;
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::QpPresolveResult;

/// presolve スケーリング (LargeCoeffRowScale × Ruiz E / c·D) で問題が σ 倍に縮むと
/// unscale 時に残差が 1/σ 倍に増幅される。primal 側 e_min × LargeCoeffRowScale と
/// dual 側 c·d_min の小さい方を sigma_total とし、IPM eps を user_eps×σ に厳しくする。
/// noise floor は `ipm_core::IPM_EPS_NOISE_FLOOR` で集約 (scaling.rs::EPS_FLOOR と
/// 共通)。amp 経由の二段 tightening を defeat されない設計。
pub(super) fn tighten_ipm_eps_for_presolve_scale(
    opts: &SolverOptions,
    presolve_result: &QpPresolveResult,
) -> SolverOptions {
    use crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR;
    let mut primal_row_scale_min = 1.0_f64;
    for step in presolve_result.postsolve_stack.steps.iter() {
        if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
            let local_min = row_scales
                .iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .fold(f64::INFINITY, |a, &v| a.min(v));
            if local_min.is_finite() {
                primal_row_scale_min *= local_min;
            }
        }
    }
    let mut dual_col_scale_min = f64::INFINITY;
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let e_min = scaler
            .e
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if e_min.is_finite() {
            primal_row_scale_min *= e_min;
        }
        let d_min = scaler
            .d
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if d_min.is_finite() && scaler.c.is_finite() && scaler.c > 0.0 {
            dual_col_scale_min = scaler.c * d_min;
        }
    }
    let sigma_total = primal_row_scale_min.min(dual_col_scale_min);
    if sigma_total < 1.0 && sigma_total > 0.0 {
        let mut tightened = opts.clone();
        let eps_orig = opts.ipm_eps();
        let eps_scaled = (eps_orig * sigma_total).max(IPM_EPS_NOISE_FLOOR);
        tightened.tolerance = None;
        tightened.ipm.eps = eps_scaled;
        if std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "POST_STAGE [IPM eps tighten] σ_total={:.3e} eps_orig={:.3e} → eps_scaled={:.3e}",
                sigma_total, eps_orig, eps_scaled
            );
        }
        tightened
    } else {
        opts.clone()
    }
}
