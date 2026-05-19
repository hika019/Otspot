//! KKT factorization with 3 段防御 (probe-based regularization retry + identity-perm fallback).

use super::state::{LDL_FALLBACK_DELTA_MIN, LDL_REG_CEILING, LDL_REG_GROWTH, LDL_REG_RETRY_MAX};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{
    factorize_kkt_pre_permuted_cached_par, factorize_kkt_with_cached_perm_par, max_l_nnz_from_budget,
    KktError, KktFactor,
};
use crate::linalg::timeout::TimeoutCtx;
use crate::qp::ipm_core::kkt::{build_schur_system, AugmentedKktCache, PermutedAugmentedKkt};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use faer::Par;

/// 反復間で sparsity 不変な構造を保持する。
pub(super) struct FactorizeCaches {
    pub(super) amd_perm: Option<Vec<usize>>,
    pub(super) aug_permuted: Option<PermutedAugmentedKkt>,
    pub(super) symbolic_cholesky:
        Option<std::sync::Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>>,
}

impl FactorizeCaches {
    pub(super) fn new() -> Self {
        Self {
            amd_perm: None,
            aug_permuted: None,
            symbolic_cholesky: None,
        }
    }
}

pub(super) enum FactorizeOutcome {
    Ok {
        factor: KktFactor,
        aug_mat: CscMatrix,
        d_inv: Option<Vec<f64>>,
        /// 因子化で実際に採用された ρ (regularization retry で持ち上げ済みの値)。
        /// check_infeasible_or_unbounded の delta_p に必要。
        rho_used: f64,
    },
    Timeout,
    Failure,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn factorize_kkt_with_retry(
    problem: &QpProblem,
    a_ext: &CscMatrix,
    aug_cache: &AugmentedKktCache,
    caches: &mut FactorizeCaches,
    sigma_vec: &[f64],
    is_eq_ext: &[bool],
    s: &[f64],
    r_d_pmm: &[f64],
    r_p_pmm: &[f64],
    rho_matrix: f64,
    delta_matrix: f64,
    inertia_correction: f64,
    use_schur: bool,
    timeout_ctx: &TimeoutCtx,
    par: Par,
    n: usize,
    prof: bool,
) -> FactorizeOutcome {
    // 不定 Q 用に rho_retry を inertia_correction で下限し、Q+ρI を初回から PSD に。
    let mut rho_retry = rho_matrix.max(inertia_correction);
    let ldl_reg_ceiling = LDL_REG_CEILING.max(inertia_correction);
    let mut delta_retry = delta_matrix;
    let mut fac_opt: Option<KktFactor> = None;
    let mut aug_mat_opt: Option<CscMatrix> = None;
    let mut d_inv_opt: Option<Vec<f64>> = None;

    for _retry in 0..LDL_REG_RETRY_MAX {
        if timeout_ctx.should_stop() {
            return FactorizeOutcome::Timeout;
        }
        let prof_t_build = if prof { Some(std::time::Instant::now()) } else { None };
        let mat_for_factor = if use_schur {
            let (s_mat, d_inv) = build_schur_system(
                &problem.q, a_ext, sigma_vec, rho_retry, delta_retry,
            );
            d_inv_opt = Some(d_inv);
            s_mat
        } else {
            aug_cache.materialize(sigma_vec, rho_retry, delta_retry)
        };
        if let Some(t) = prof_t_build {
            eprintln!("FACT_PROF section=build n={} nnz={} t={:.3}ms",
                mat_for_factor.nrows, mat_for_factor.values.len(),
                t.elapsed().as_secs_f64() * 1000.0);
        }
        if caches.amd_perm.is_none() {
            caches.amd_perm = Some(amd_with_deadline(
                mat_for_factor.nrows,
                &mat_for_factor.col_ptr,
                &mat_for_factor.row_ind,
                timeout_ctx.deadline,
            ));
        }
        let perm = caches.amd_perm.as_ref().unwrap();
        // Schur / DD-LDL は pre-permuted 未対応なので通常経路へ。
        let dd_ldl = std::env::var("IPM_DD_LDL").ok().as_deref() == Some("1");
        let use_pre_permuted = !use_schur && !dd_ldl;
        if use_pre_permuted && caches.aug_permuted.is_none() {
            caches.aug_permuted = Some(aug_cache.permute(perm));
        }
        let prof_t_factor = if prof { Some(std::time::Instant::now()) } else { None };
        let factor_result = if use_pre_permuted {
            let permuted_cache = caches.aug_permuted.as_ref().unwrap();
            let pre_permuted = permuted_cache.materialize(sigma_vec, rho_retry, delta_retry);
            factorize_kkt_pre_permuted_cached_par(
                &pre_permuted,
                &mat_for_factor,
                perm,
                timeout_ctx.deadline,
                max_l_nnz_from_budget(),
                Some(n),
                caches.symbolic_cholesky.clone(),
                par,
            )
        } else {
            factorize_kkt_with_cached_perm_par(
                &mat_for_factor,
                perm,
                timeout_ctx.deadline,
                max_l_nnz_from_budget(),
                Some(n),
                par,
            )
        };
        if use_pre_permuted && caches.symbolic_cholesky.is_none() {
            if let Ok(ref f) = factor_result {
                caches.symbolic_cholesky = f.symbolic_arc();
            }
        }
        match factor_result {
            Ok(f) => {
                if let Some(t) = prof_t_factor {
                    eprintln!("FACT_PROF section=factorize n={} t={:.3}ms",
                        mat_for_factor.nrows, t.elapsed().as_secs_f64() * 1000.0);
                }
                let prof_t_probe = if prof { Some(std::time::Instant::now()) } else { None };
                // 健全性プローブ: factorize Ok でも cond(K) 大で LDL solve が Newton 方向
                // を central path から外す病理を ||K·sol − rhs||/||rhs|| で直接弾く。
                // iterative backend は LDL 精度概念が無いので skip。
                if !f.is_iterative()
                    && !probe_ldl_health(
                        &f, &mat_for_factor, r_d_pmm, r_p_pmm, s, is_eq_ext, n,
                    )
                {
                    if rho_retry >= ldl_reg_ceiling {
                        break;
                    }
                    rho_retry = (rho_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                    delta_retry = (delta_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                    continue;
                }
                if let Some(t) = prof_t_probe {
                    eprintln!("FACT_PROF section=probe n={} t={:.3}ms",
                        mat_for_factor.nrows, t.elapsed().as_secs_f64() * 1000.0);
                }
                fac_opt = Some(f);
                aug_mat_opt = Some(mat_for_factor);
                break;
            }
            Err(KktError::DeadlineExceeded) => {
                return FactorizeOutcome::Timeout;
            }
            Err(_) => {
                if rho_retry >= ldl_reg_ceiling {
                    break;
                }
                rho_retry = (rho_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                delta_retry = (delta_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
            }
        }
    }

    // 第3防御: identity perm + 大きな delta で再試行。
    if fac_opt.is_none() {
        caches.amd_perm = None;
        let delta_fallback = LDL_FALLBACK_DELTA_MIN.max(rho_retry).max(delta_retry);
        let aug_mat_fb = aug_cache.materialize(sigma_vec, rho_retry, delta_fallback);
        let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
        match factorize_kkt_with_cached_perm_par(
            &aug_mat_fb,
            &identity_perm,
            timeout_ctx.deadline,
            max_l_nnz_from_budget(),
            Some(n),
            par,
        ) {
            Ok(f) => {
                fac_opt = Some(f);
                aug_mat_opt = Some(aug_mat_fb);
            }
            Err(KktError::DeadlineExceeded) => return FactorizeOutcome::Timeout,
            Err(_) => {}
        }
    }

    match (fac_opt, aug_mat_opt) {
        (Some(factor), Some(aug_mat)) => FactorizeOutcome::Ok {
            factor,
            aug_mat,
            d_inv: d_inv_opt,
            rho_used: rho_retry,
        },
        _ => FactorizeOutcome::Failure,
    }
}

/// (A) ||K·sol−rhs||/||rhs|| ≤ 1e-3 (LDL 大破綻 sanity, eps 独立)
/// (B) sol_inf / rhs_inf ≤ 1/ε_machine (cond(K) が f64 域内)
fn probe_ldl_health(
    f: &KktFactor,
    mat_for_factor: &CscMatrix,
    r_d_pmm: &[f64],
    r_p_pmm: &[f64],
    s: &[f64],
    is_eq_ext: &[bool],
    n: usize,
) -> bool {
    let probe_dim = mat_for_factor.nrows;
    let mut probe_rhs = vec![0.0_f64; probe_dim];
    probe_rhs[..n].copy_from_slice(r_d_pmm);
    // 予測子 RHS 下半分: 不等式行は r_p + s、等式行は r_p。
    for (i, slot) in probe_rhs[n..].iter_mut().enumerate() {
        *slot = if is_eq_ext[i] { r_p_pmm[i] } else { r_p_pmm[i] + s[i] };
    }
    let rhs_inf = probe_rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if rhs_inf <= 0.0 || !rhs_inf.is_finite() {
        return true;
    }
    let mut probe_sol = vec![0.0_f64; probe_dim];
    f.solve(&probe_rhs, &mut probe_sol);

    let mut kx = vec![0.0_f64; probe_dim];
    for col in 0..mat_for_factor.ncols {
        let cs = mat_for_factor.col_ptr[col];
        let ce = mat_for_factor.col_ptr[col + 1];
        for ptr in cs..ce {
            let row = mat_for_factor.row_ind[ptr];
            let val = mat_for_factor.values[ptr];
            kx[row] += val * probe_sol[col];
            if row != col {
                kx[col] += val * probe_sol[row];
            }
        }
    }
    let mut resid_inf = 0.0_f64;
    for i in 0..probe_dim {
        let r = (probe_rhs[i] - kx[i]).abs();
        if r > resid_inf { resid_inf = r; }
    }
    let rel_resid = resid_inf / rhs_inf;
    let sol_inf = probe_sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let f64_precision_ceiling = 1.0 / f64::EPSILON;
    let amplification = sol_inf / rhs_inf;
    const LDL_HEALTH_REL_TOL: f64 = 1e-3;
    let unhealthy = !rel_resid.is_finite()
        || rel_resid > LDL_HEALTH_REL_TOL
        || !amplification.is_finite()
        || amplification > f64_precision_ceiling;
    !unhealthy
}

/// Schur auto-detect (probe).
pub(super) fn auto_schur_enabled(
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_ext: usize,
    options: &crate::options::SolverOptions,
    timeout_ctx: &TimeoutCtx,
    par: Par,
) -> bool {
    let explicit_schur = std::env::var("QP_SCHUR").ok().as_deref() == Some("1");
    let auto_schur_disabled = std::env::var("QP_NO_AUTO_SCHUR").ok().as_deref() == Some("1");
    if explicit_schur || auto_schur_disabled {
        return false;
    }
    use crate::qp::ipm_core::kkt::build_augmented_system;
    let probe_sigma: Vec<f64> = vec![1.0; m_ext];
    let probe_rho = options.ipm.delta_min;
    let probe_aug = build_augmented_system(&problem.q, a_ext, &probe_sigma, probe_rho, probe_rho);
    let probe_perm = amd_with_deadline(
        probe_aug.nrows, &probe_aug.col_ptr, &probe_aug.row_ind, timeout_ctx.deadline,
    );
    let probe_result = crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
        &probe_aug, &probe_perm, timeout_ctx.deadline, Some(max_l_nnz_from_budget()), par,
    );
    let exceeds = matches!(probe_result, Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }));
    if exceeds && std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
        eprintln!("IPPMM_AUTO_SCHUR: augmented L_nnz exceeds budget, switching to Schur formulation");
    }
    exceeds
}
