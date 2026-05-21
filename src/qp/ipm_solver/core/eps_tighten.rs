//! Presolve-scaling tightening: multiply IPM internal eps by σ_total to counteract
//! residual amplification on unscale.

use crate::options::SolverOptions;
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::QpPresolveResult;

/// Compute `eps_scaled = opts.ipm.eps * sigma_total` and return a clone of `opts` with
/// `ipm.eps = eps_scaled` and `tolerance = None`.
///
/// Uses `opts.ipm.eps` directly (not `opts.ipm_eps()`): when the attempt loop sets
/// `Tolerance::Custom(user_eps)` and `ipm.eps = user_eps/tighten`, calling `ipm_eps()`
/// would return `user_eps`, silently bypassing the tighten factor.
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
        // Use opts.ipm.eps directly: the retry loop sets this to user_eps/tighten.
        // opts.ipm_eps() would return user_eps from Tolerance::Custom, bypassing tightening.
        let eps_orig = opts.ipm.eps;
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

#[cfg(test)]
mod eps_tighten_tests {
    use super::*;
    use crate::options::{IpmOptions, SolverOptions, Tolerance};
    use crate::linalg::ruiz::RuizScaler;
    use crate::presolve::QpPresolveResult;
    use crate::problem::ConstraintType;
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    fn presolve_with_ruiz_e(e_min: f64) -> QpPresolveResult {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            q, vec![0.0], a, vec![1.0],
            vec![(0.0_f64, f64::INFINITY)],
            vec![ConstraintType::Eq],
        ).unwrap();
        let mut pre = QpPresolveResult::no_reduction(&prob);
        // e=[e_min] → primal_row_scale_min=e_min; d=[1], c=1 → dual_col_scale_min=1
        // sigma_total = min(e_min, 1) = e_min  (when e_min < 1)
        pre.ruiz_scaler = Some(RuizScaler { e: vec![e_min], d: vec![1.0], c: 1.0 });
        pre
    }

    /// With Tolerance::Custom(user_eps) set, ipm_eps() returns user_eps (bypasses ipm.eps).
    /// The fix uses opts.ipm.eps directly, preserving the attempt-loop tighten factor.
    /// Reverting to ipm_eps() would produce an output 1000x looser; this test FAILS then.
    #[test]
    fn custom_tolerance_does_not_bypass_attempt_tighten() {
        let user_eps = 1e-4;
        let tighten = 1000.0;
        let sigma = 1e-3;

        let pre = presolve_with_ruiz_e(sigma);
        let mut opts = SolverOptions::default();
        opts.tolerance = Some(Tolerance::Custom(user_eps));
        opts.ipm = IpmOptions { eps: user_eps / tighten, ..IpmOptions::default() };

        let result = tighten_ipm_eps_for_presolve_scale(&opts, &pre);

        // Correct: eps_scaled = (user_eps/tighten) * sigma = 1e-7 * 1e-3 = 1e-10
        let attempt_eps = user_eps / tighten;
        let expected = (attempt_eps * sigma)
            .max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);
        // Pre-fix wrong value: (user_eps) * sigma = 1e-4 * 1e-3 = 1e-7 (1000x looser)
        let pre_fix_wrong = user_eps * sigma;

        assert!(
            (result.ipm.eps - expected).abs() < 1e-30,
            "must use ipm.eps={:.3e} (attempt-tightened), not ipm_eps()={:.3e} (Custom bypass). \
             expected={:.3e}, got={:.3e}",
            attempt_eps, user_eps, expected, result.ipm.eps
        );
        assert!(
            result.ipm.eps < pre_fix_wrong * 0.01,
            "pre-fix would give {:.3e} (1000x looser); got {:.3e}",
            pre_fix_wrong, result.ipm.eps
        );
        assert!(result.tolerance.is_none(), "Custom tolerance must be cleared");
    }

    /// Without Custom tolerance, ipm_eps() == ipm.eps; fix must not change this path.
    #[test]
    fn no_custom_tolerance_unchanged() {
        let ipm_eps = 1e-7;
        let sigma = 1e-3;

        let pre = presolve_with_ruiz_e(sigma);
        let mut opts = SolverOptions::default();
        opts.tolerance = None;
        opts.ipm = IpmOptions { eps: ipm_eps, ..IpmOptions::default() };

        let result = tighten_ipm_eps_for_presolve_scale(&opts, &pre);
        let expected = (ipm_eps * sigma).max(crate::qp::ipm_core::IPM_EPS_NOISE_FLOOR);

        assert!(
            (result.ipm.eps - expected).abs() < 1e-30,
            "without Custom tolerance, output must be ipm.eps*sigma={:.3e}; got {:.3e}",
            expected, result.ipm.eps
        );
    }
}
