//! KKT factorization with 3 段防御 (probe-based regularization retry + identity-perm fallback).

use super::state::{LDL_FALLBACK_DELTA_MIN, LDL_REG_CEILING, LDL_REG_GROWTH, LDL_REG_RETRY_MAX};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{
    factorize_kkt_pre_permuted_cached_par, factorize_kkt_with_cached_perm_par,
    KktConfig, KktError, KktFactor,
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

// KktFactor is large but FactorizeOutcome is a short-lived per-iteration result;
// boxing would add allocation overhead on the hot IPM path.
#[allow(clippy::large_enum_variant)]
pub(super) enum FactorizeOutcome {
    Ok {
        factor: KktFactor,
        aug_mat: CscMatrix,
        d_inv: Option<Vec<f64>>,
        /// 因子化で実際に採用された ρ (regularization retry で持ち上げ済みの値)。
        /// check_infeasible_or_unbounded の delta_p に必要。
        rho_used: f64,
        /// この 1 iteration で発生した regularization retry 回数 (健全性プローブ失敗含む)。
        retry_count: u32,
        /// 採用された KktFactor が iterative (MINRES) backend か。
        used_iterative: bool,
        /// 数値 LDL 因子化の所要時間 (全 retry 合計、ナノ秒)。
        factorize_ns: u128,
    },
    Timeout,
    Failure,
}

/// factorize_kkt_with_retry の入力一式 (1 iteration 分)。
/// caches は mut 参照で別経路、本 ctx は immutable view のみ束ねる。
pub(super) struct FactorizeContext<'a> {
    pub problem: &'a QpProblem,
    pub a_ext: &'a CscMatrix,
    pub aug_cache: &'a AugmentedKktCache,
    pub sigma_vec: &'a [f64],
    pub is_eq_ext: &'a [bool],
    pub s: &'a [f64],
    pub r_d_pmm: &'a [f64],
    pub r_p_pmm: &'a [f64],
    pub rho_matrix: f64,
    pub delta_matrix: f64,
    pub inertia_correction: f64,
    pub use_schur: bool,
    pub timeout_ctx: &'a TimeoutCtx,
    pub par: Par,
    pub n: usize,
    pub prof: bool,
    /// KKT factorization/MINRES configuration from IpmOptions.
    pub kkt_cfg: KktConfig,
}

pub(super) fn factorize_kkt_with_retry(
    ctx: &FactorizeContext<'_>,
    caches: &mut FactorizeCaches,
) -> FactorizeOutcome {
    // 不定 Q 用に rho_retry を inertia_correction で下限し、Q+ρI を初回から PSD に。
    let mut rho_retry = ctx.rho_matrix.max(ctx.inertia_correction);
    let ldl_reg_ceiling = LDL_REG_CEILING.max(ctx.inertia_correction);
    let mut delta_retry = ctx.delta_matrix;
    let mut fac_opt: Option<KktFactor> = None;
    let mut aug_mat_opt: Option<CscMatrix> = None;
    let mut d_inv_opt: Option<Vec<f64>> = None;

    // 計測変数 (常時収集、prof flag 不要)。
    let mut retry_count: u32 = 0;
    let mut total_factorize_ns: u128 = 0;
    let mut used_iterative = false;

    for _retry in 0..LDL_REG_RETRY_MAX {
        if ctx.timeout_ctx.should_stop() {
            return FactorizeOutcome::Timeout;
        }
        let prof_t_build = if ctx.prof { Some(std::time::Instant::now()) } else { None };
        let mat_for_factor = if ctx.use_schur {
            let (s_mat, d_inv) = build_schur_system(
                &ctx.problem.q, ctx.a_ext, ctx.sigma_vec, rho_retry, delta_retry,
            );
            d_inv_opt = Some(d_inv);
            s_mat
        } else {
            ctx.aug_cache.materialize(ctx.sigma_vec, rho_retry, delta_retry)
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
                ctx.timeout_ctx.deadline,
            ));
        }
        let perm = caches.amd_perm.as_ref().unwrap();
        // Schur / DD-LDL は pre-permuted 未対応なので通常経路へ。
        let use_pre_permuted = !ctx.use_schur && !ctx.kkt_cfg.dd_ldl;
        if use_pre_permuted && caches.aug_permuted.is_none() {
            caches.aug_permuted = Some(ctx.aug_cache.permute(perm));
        }
        let t_factor = std::time::Instant::now();
        let factor_result = if use_pre_permuted {
            let permuted_cache = caches.aug_permuted.as_ref().unwrap();
            let pre_permuted = permuted_cache.materialize(ctx.sigma_vec, rho_retry, delta_retry);
            factorize_kkt_pre_permuted_cached_par(
                &pre_permuted,
                &mat_for_factor,
                perm,
                ctx.timeout_ctx.deadline,
                &ctx.kkt_cfg,
                Some(ctx.n),
                caches.symbolic_cholesky.clone(),
                ctx.par,
            )
        } else {
            factorize_kkt_with_cached_perm_par(
                &mat_for_factor,
                perm,
                ctx.timeout_ctx.deadline,
                &ctx.kkt_cfg,
                Some(ctx.n),
                ctx.par,
            )
        };
        let iter_factorize_ns = t_factor.elapsed().as_nanos();
        if use_pre_permuted && caches.symbolic_cholesky.is_none() {
            if let Ok(ref f) = factor_result {
                caches.symbolic_cholesky = f.symbolic_arc();
            }
        }
        match factor_result {
            Ok(f) => {
                if ctx.prof {
                    eprintln!("FACT_PROF section=factorize n={} t={:.3}ms",
                        mat_for_factor.nrows, iter_factorize_ns as f64 / 1_000_000.0);
                }
                total_factorize_ns += iter_factorize_ns;
                let prof_t_probe = if ctx.prof { Some(std::time::Instant::now()) } else { None };
                // 健全性プローブ: factorize Ok でも cond(K) 大で LDL solve が Newton 方向
                // を central path から外す病理を ||K·sol − rhs||/||rhs|| で直接弾く。
                // iterative backend は LDL 精度概念が無いので skip。
                if !f.is_iterative()
                    && !probe_ldl_health(
                        &f, &mat_for_factor, ctx.r_d_pmm, ctx.r_p_pmm, ctx.s, ctx.is_eq_ext, ctx.n,
                    )
                {
                    retry_count += 1;
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
                used_iterative = f.is_iterative();
                fac_opt = Some(f);
                aug_mat_opt = Some(mat_for_factor);
                break;
            }
            Err(KktError::DeadlineExceeded) => {
                return FactorizeOutcome::Timeout;
            }
            Err(_) => {
                total_factorize_ns += iter_factorize_ns;
                retry_count += 1;
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
        let aug_mat_fb = ctx.aug_cache.materialize(ctx.sigma_vec, rho_retry, delta_fallback);
        let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
        let t_fb = std::time::Instant::now();
        let fb_result = factorize_kkt_with_cached_perm_par(
            &aug_mat_fb,
            &identity_perm,
            ctx.timeout_ctx.deadline,
            &ctx.kkt_cfg,
            Some(ctx.n),
            ctx.par,
        );
        total_factorize_ns += t_fb.elapsed().as_nanos();
        match fb_result {
            Ok(f) => {
                used_iterative = f.is_iterative();
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
            retry_count,
            used_iterative,
            factorize_ns: total_factorize_ns,
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
    use crate::qp::ipm_core::kkt::build_augmented_system;
    let probe_sigma: Vec<f64> = vec![1.0; m_ext];
    let probe_rho = options.ipm.delta_min;
    let probe_aug = build_augmented_system(&problem.q, a_ext, &probe_sigma, probe_rho, probe_rho);
    let probe_perm = amd_with_deadline(
        probe_aug.nrows, &probe_aug.col_ptr, &probe_aug.row_ind, timeout_ctx.deadline,
    );
    let probe_result = crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
        &probe_aug, &probe_perm, timeout_ctx.deadline, Some(options.ipm.effective_max_l_nnz()), par,
    );
    matches!(probe_result, Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }))
}
