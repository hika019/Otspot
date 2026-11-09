//! KKT factorization with 3 段防御 (probe-based regularization retry + identity-perm fallback).

use super::state::{
    LDL_FALLBACK_DELTA_MIN, LDL_REG_CEILING, LDL_REG_GROWTH, LDL_REG_RETRY_MAX, REG_LIMIT_MIN,
};
use crate::qp::diagnostics::with_ippmm_sink;
use crate::qp::ipm_core::kkt::{build_schur_system, AugmentedKktCache, PermutedAugmentedKkt};
use crate::qp::problem::QpProblem;
use faer::Par;
use otspot_num::linalg::amd::amd_with_deadline;
use otspot_num::linalg::kkt_solver::{
    factorize_kkt_pre_permuted_cached_par, factorize_kkt_with_cached_perm_par, KktConfig, KktError,
    KktFactor,
};
use otspot_num::linalg::timeout::TimeoutCtx;
use otspot_num::sparse::CscMatrix;

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
        // 診断: 元順序組み立て (materialize / Schur 構築) を assembly 排他的
        // 計時・count 側計数へ (review R1。置換済みも同窓 — helper doc 参照)。
        let t_asm = std::time::Instant::now();
        let mat_for_factor = if ctx.use_schur {
            let (s_mat, d_inv) = build_schur_system(
                &ctx.problem.q,
                ctx.a_ext,
                ctx.sigma_vec,
                rho_retry,
                delta_retry,
            );
            d_inv_opt = Some(d_inv);
            s_mat
        } else {
            ctx.aug_cache
                .materialize(ctx.sigma_vec, rho_retry, delta_retry)
        };
        let (asm_us, is_schur_build) = (t_asm.elapsed().as_micros() as u64, ctx.use_schur);
        with_ippmm_sink(|s| {
            s.ipm.assembly_us += asm_us;
            if is_schur_build {
                s.counts.kkt_schur_builds += 1;
            } else {
                s.counts.materialize_original_calls += 1;
            }
        });
        if caches.amd_perm.is_none() {
            let t_amd = std::time::Instant::now();
            caches.amd_perm = Some(amd_with_deadline(
                mat_for_factor.nrows(),
                mat_for_factor.col_ptr(),
                mat_for_factor.row_ind(),
                ctx.timeout_ctx.deadline,
            ));
            let amd_us = t_amd.elapsed().as_micros() as u64;
            with_ippmm_sink(|s| s.ipm.assembly_us += amd_us);
        }
        let perm = caches.amd_perm.as_ref().unwrap();
        // Schur / DD-LDL は pre-permuted 未対応なので通常経路へ。
        let use_pre_permuted = !ctx.use_schur && !ctx.kkt_cfg.dd_ldl;
        if use_pre_permuted && caches.aug_permuted.is_none() {
            let t_perm = std::time::Instant::now();
            caches.aug_permuted = Some(ctx.aug_cache.permute(perm));
            let perm_us = t_perm.elapsed().as_micros() as u64;
            with_ippmm_sink(|s| s.ipm.assembly_us += perm_us);
        }
        // 診断 (R1): 置換済み materialize は行列組み立て (helper の doc に
        // 境界定義)。ここで計時・計数した後に factor タイマーを開始する。
        let pre_permuted_opt = if use_pre_permuted {
            Some(materialize_permuted_in_assembly(
                ctx,
                caches,
                rho_retry,
                delta_retry,
            ))
        } else {
            None
        };
        let t_factor = std::time::Instant::now();
        let factor_result = if use_pre_permuted {
            factorize_kkt_pre_permuted_cached_par(
                pre_permuted_opt.as_ref().unwrap(),
                &mat_for_factor,
                perm,
                ctx.timeout_ctx.deadline,
                &ctx.kkt_cfg,
                Some(ctx.n),
                caches.symbolic_cholesky.clone(),
                ctx.par,
            )
        } else {
            // Schur / DD / (DD からの) 通常経路の wrapper は内部で
            // `permute_sym_upper` を実行するため factor_us 窓に置換コピーを
            // 含む (pre-permuted 経路との境界差 — diagnostics.rs doc に明記)。
            factorize_kkt_with_cached_perm_par(
                &mat_for_factor,
                perm,
                ctx.timeout_ctx.deadline,
                &ctx.kkt_cfg,
                Some(ctx.n),
                ctx.par,
            )
        };
        let iter_factorize = t_factor.elapsed();
        let iter_factorize_ns = iter_factorize.as_nanos();
        with_ippmm_sink(|s| {
            s.ipm.factor_us += iter_factorize.as_micros() as u64;
            s.counts.kkt_factorize_attempts += 1;
        });
        if use_pre_permuted && caches.symbolic_cholesky.is_none() {
            if let Ok(ref f) = factor_result {
                caches.symbolic_cholesky = f.symbolic_arc();
            }
        }
        match factor_result {
            Ok(f) => {
                total_factorize_ns += iter_factorize_ns;
                // Sanity probe: even factorize Ok can yield ill-conditioned K; reject via ||K·sol−rhs||/||rhs||.
                // Iterative backend has no LDL precision concept, skip.
                if !f.is_iterative() {
                    // 診断: probe は factorize 呼び出し専有時間 (呼出回数は count 側)。
                    let t_probe = std::time::Instant::now();
                    let probe_ok = probe_ldl_health(
                        &f,
                        &mat_for_factor,
                        ctx.r_d_pmm,
                        ctx.r_p_pmm,
                        ctx.s,
                        ctx.is_eq_ext,
                        ctx.n,
                    );
                    let probe_us = t_probe.elapsed().as_micros() as u64;
                    with_ippmm_sink(|s| {
                        s.ipm.probe_us += probe_us;
                        s.counts.kkt_health_probes += 1;
                    });
                    if !probe_ok {
                        retry_count += 1;
                        if rho_retry >= ldl_reg_ceiling {
                            break;
                        }
                        rho_retry = (rho_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                        delta_retry = (delta_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                        continue;
                    }
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
        let t_asm_fb = std::time::Instant::now();
        let aug_mat_fb = ctx
            .aug_cache
            .materialize(ctx.sigma_vec, rho_retry, delta_fallback);
        let asm_fb_us = t_asm_fb.elapsed().as_micros() as u64;
        let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows()).collect();
        let t_fb = std::time::Instant::now();
        let fb_result = factorize_kkt_with_cached_perm_par(
            &aug_mat_fb,
            &identity_perm,
            ctx.timeout_ctx.deadline,
            &ctx.kkt_cfg,
            Some(ctx.n),
            ctx.par,
        );
        let fb_elapsed = t_fb.elapsed();
        total_factorize_ns += fb_elapsed.as_nanos();
        // 第3防御の構築も元順序 materialize 1 回 + factorize 1 試行 (R1 監査:
        // assembly 窓の計時は上の t_asm_fb で factor 窓 (t_fb) と排他、回数も
        // 構築点で materialize_original_calls に計上 — 置換済みは構築しない)。
        with_ippmm_sink(|s| {
            s.ipm.assembly_us += asm_fb_us;
            s.ipm.factor_us += fb_elapsed.as_micros() as u64;
            s.counts.materialize_original_calls += 1;
            s.counts.kkt_factorize_attempts += 1;
        });
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
            retry_count,
            used_iterative,
            factorize_ns: total_factorize_ns,
        },
        _ => FactorizeOutcome::Failure,
    }
}

/// R1: 置換済み (AMD 順序) cache の materialize。行列組み立てであり数値分解
/// ではないため assembly の排他的計時へ入れ、構築回数を
/// `materialize_permuted_calls` に積む。呼び出し側はこれが返ってから factor
/// タイマーを開始する — これで factor_us 窓は wrapper 内部 (対角符号抽出・
/// CSC→faer 変換・初回のみ symbolic 解析・数値 LDL・MINRES fallback 構築。
/// 境界の詳細は `crate::qp::diagnostics::IpmPhaseTimes::factor_us` doc) と
/// 一致する。
fn materialize_permuted_in_assembly(
    ctx: &FactorizeContext<'_>,
    caches: &FactorizeCaches,
    rho_retry: f64,
    delta_retry: f64,
) -> CscMatrix {
    let permuted_cache = caches
        .aug_permuted
        .as_ref()
        .expect("caller が permute 後にのみ呼ぶ (use_pre_permuted 経路)");
    let t_perm_mat = std::time::Instant::now();
    let pre_permuted = permuted_cache.materialize(ctx.sigma_vec, rho_retry, delta_retry);
    let perm_mat_us = t_perm_mat.elapsed().as_micros() as u64;
    with_ippmm_sink(|s| {
        s.ipm.assembly_us += perm_mat_us;
        s.counts.materialize_permuted_calls += 1;
    });
    pre_permuted
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
    let probe_dim = mat_for_factor.nrows();
    let mut probe_rhs = vec![0.0_f64; probe_dim];
    probe_rhs[..n].copy_from_slice(r_d_pmm);
    // 予測子 RHS 下半分: 不等式行は r_p + s、等式行は r_p。
    for (i, slot) in probe_rhs[n..].iter_mut().enumerate() {
        *slot = if is_eq_ext[i] {
            r_p_pmm[i]
        } else {
            r_p_pmm[i] + s[i]
        };
    }
    let rhs_inf = probe_rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if rhs_inf <= 0.0 || !rhs_inf.is_finite() {
        return true;
    }
    let mut probe_sol = vec![0.0_f64; probe_dim];
    if f.solve(&probe_rhs, &mut probe_sol).is_err() {
        return false;
    }

    let mut kx = vec![0.0_f64; probe_dim];
    for col in 0..mat_for_factor.ncols() {
        let cs = mat_for_factor.col_ptr()[col];
        let ce = mat_for_factor.col_ptr()[col + 1];
        for ptr in cs..ce {
            let row = mat_for_factor.row_ind()[ptr];
            let val = mat_for_factor.values()[ptr];
            kx[row] += val * probe_sol[col];
            if row != col {
                kx[col] += val * probe_sol[row];
            }
        }
    }
    let mut resid_inf = 0.0_f64;
    for i in 0..probe_dim {
        let r = (probe_rhs[i] - kx[i]).abs();
        if r > resid_inf {
            resid_inf = r;
        }
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
pub(crate) fn auto_schur_enabled(
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_ext: usize,
    options: &crate::options::SolverOptions,
    timeout_ctx: &TimeoutCtx,
    par: Par,
) -> bool {
    use crate::qp::ipm_core::kkt::build_augmented_system;
    let probe_sigma: Vec<f64> = vec![1.0; m_ext];
    // adaptive reg_limit の最終 floor (REG_LIMIT_MIN) で probe する。旧実装は
    // 廃止された固定 options.ipm.delta_min (1e-8) を使っていたが、実際の反復では
    // 正則化がそこまで下がりうるため、fill-in 予測が最悪ケースを反映するよう
    // 最小値で probe する。
    let probe_rho = REG_LIMIT_MIN;
    let probe_aug = build_augmented_system(&problem.q, a_ext, &probe_sigma, probe_rho, probe_rho);
    let probe_perm = amd_with_deadline(
        probe_aug.nrows(),
        probe_aug.col_ptr(),
        probe_aug.row_ind(),
        timeout_ctx.deadline,
    );
    let probe_result = otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
        &probe_aug,
        &probe_perm,
        timeout_ctx.deadline,
        Some(options.ipm.effective_max_l_nnz()),
        par,
    );
    matches!(
        probe_result,
        Err(otspot_num::linalg::ldl::LdlError::WouldExceedBudget { .. })
    )
}

#[cfg(test)]
mod r1_materialize_accounting_tests {
    //! R1 sentinel: 置換済み materialize の時間・回数計装。旧実装 (factor
    //! タイマー内で置換済み構築をし、回数を積まない) へ revert すると
    //! `materialize_permuted_calls` 系 assert が fail する。秒数の厳密一致は
    //! 検証しない (review §2: 計測境界はソース確認、実測差は diag 実行で)。

    use super::*;
    use crate::qp::diagnostics::{
        begin_attempt, capture_diagnostics, finish_attempt, AttemptKind, AttemptOutcomeBits,
    };
    use crate::qp::ipm_core::kkt::build_extended_constraints;
    use otspot_num::linalg::timeout::TimeoutCtx;

    struct Fixture {
        problem: QpProblem,
        a_ext: CscMatrix,
        aug_cache: crate::qp::ipm_core::kkt::AugmentedKktCache,
        sigma_vec: Vec<f64>,
        is_eq_ext: Vec<bool>,
        s: Vec<f64>,
        r_d_pmm: Vec<f64>,
        r_p_pmm: Vec<f64>,
    }

    fn fixture(rho_on_diag: bool) -> Fixture {
        // min (x-1)^2 + (y-2)^2 + 3x+y の対角 Q 版 (R1 経路検証用の手計算不要な
        // 小 QP: 健全経路は正則化不要、singular 検証は rho=0 で構成する)。
        let q_diag: f64 = if rho_on_diag { 1.0 } else { 0.0 };
        let q =
            otspot_num::sparse::CscMatrix::from_triplets(&[0, 1], &[0, 1], &[q_diag, q_diag], 2, 2)
                .unwrap();
        let a = otspot_num::sparse::CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![3.0, 1.0],
            a,
            vec![10.0],
            vec![(-10.0, 10.0), (-10.0, 10.0)],
        )
        .unwrap();
        let (a_ext, _b_ext, m_ext, _m_orig, _n_lb, is_eq_ext) =
            build_extended_constraints(&problem);
        let aug_cache = crate::qp::ipm_core::kkt::build_augmented_cache(&problem.q, &a_ext);
        Fixture {
            r_d_pmm: vec![1.0, 1.0],
            problem,
            a_ext,
            aug_cache,
            sigma_vec: vec![1.0; m_ext],
            is_eq_ext,
            s: vec![1.0; m_ext],
            r_p_pmm: vec![0.1; m_ext],
        }
    }

    fn factorize_once(
        fx: &Fixture,
        use_schur: bool,
        dd_ldl: bool,
        rho_matrix: f64,
    ) -> (
        FactorizeOutcome,
        crate::qp::diagnostics::AttemptDiagnostics,
        FactorizeCaches,
    ) {
        run(fx, use_schur, dd_ldl, rho_matrix, rho_matrix == 0.0)
    }

    fn run(
        fx: &Fixture,
        use_schur: bool,
        dd_ldl: bool,
        rho_matrix: f64,
        singular_setup: bool,
    ) -> (
        FactorizeOutcome,
        crate::qp::diagnostics::AttemptDiagnostics,
        FactorizeCaches,
    ) {
        let timeout_ctx = TimeoutCtx::new(None, None, None);
        let mut caches = FactorizeCaches::new();
        let cfg = KktConfig {
            dd_ldl,
            ..KktConfig::default()
        };
        // singular_setup: σ=0・delta=0・rho=0 で正則化成分を全ゼロにし、
        // rank 不足な拡張 A (境界行 5 > n=2) により K を恒久的 singular に
        // する (retry 計数忠実性テスト用)。
        let sigma_vec: Vec<f64> = if singular_setup {
            vec![0.0; fx.sigma_vec.len()]
        } else {
            fx.sigma_vec.clone()
        };
        let ((outcome, rec), _solve_diag) = capture_diagnostics(|| {
            let ctx = FactorizeContext {
                problem: &fx.problem,
                a_ext: &fx.a_ext,
                aug_cache: &fx.aug_cache,
                sigma_vec: &sigma_vec,
                is_eq_ext: &fx.is_eq_ext,
                s: &fx.s,
                r_d_pmm: &fx.r_d_pmm,
                r_p_pmm: &fx.r_p_pmm,
                rho_matrix,
                delta_matrix: if singular_setup { 0.0 } else { 1e-4 },
                inertia_correction: 0.0,
                use_schur,
                timeout_ctx: &timeout_ctx,
                par: faer::Par::Seq,
                n: fx.problem.num_vars,
                kkt_cfg: cfg,
            };
            let start = begin_attempt(AttemptKind::Ladder, false, 1.0);
            let out = factorize_kkt_with_retry(&ctx, &mut caches);
            let rec = finish_attempt(
                start,
                AttemptOutcomeBits {
                    termination: None,
                    iterations: 0,
                    numerical_failure: false,
                    infeasibility_claim: false,
                },
            );
            (out, rec)
        });
        (outcome, rec, caches)
    }

    /// pre-permuted 健全経路: 元順序と置換済みの両構築をカウンタが検知する
    /// (review §2 検証1)。factorize 1 試行 = 元順序 1 + 置換済み 1。
    #[test]
    fn pre_permuted_path_counts_both_materializes() {
        let fx = fixture(true);
        let (outcome, rec, _caches) = factorize_once(&fx, false, false, 1.0);
        assert!(matches!(outcome, FactorizeOutcome::Ok { .. }));
        let c = &rec.counts;
        assert_eq!(c.kkt_factorize_attempts, 1);
        assert_eq!(c.materialize_original_calls, 1, "元順序構築 1 回");
        assert_eq!(c.materialize_permuted_calls, 1, "置換済み構築 1 回 (R1)");
        assert_eq!(c.materialize_total_calls(), 2, "総計は導出値");
        assert_eq!(c.kkt_schur_builds, 0);
        assert_eq!(c.kkt_health_probes, 1, "健全プローブは 1 試行 1 回");
    }

    /// Schur 経路: 実行されていない構築 (元順序/置換済み) が加算されない
    /// (review §2 検証2)。Schur 構築のみが計上される。
    #[test]
    fn schur_path_counts_no_materialize_builds() {
        let fx = fixture(true);
        let (outcome, rec, caches) = factorize_once(&fx, true, false, 1.0);
        assert!(matches!(outcome, FactorizeOutcome::Ok { .. }));
        let c = &rec.counts;
        assert_eq!(c.kkt_schur_builds, 1);
        assert_eq!(
            c.materialize_original_calls, 0,
            "Schur は元順序を構築しない"
        );
        assert_eq!(
            c.materialize_permuted_calls, 0,
            "Schur は置換済みも構築しない"
        );
        assert!(caches.aug_permuted.is_none(), "pre-permuted cache 不構築");
    }

    /// DD-LDL 経路: wrapper が pre-permuted 非対応のため置換済み構築は
    /// 実行されず、元順序のみ計上 (review §2 検証2)。
    #[test]
    fn dd_path_counts_only_original_materialize() {
        let fx = fixture(true);
        let (outcome, rec, caches) = factorize_once(&fx, false, true, 1.0);
        assert!(
            matches!(outcome, FactorizeOutcome::Ok { .. }),
            "DD 健全経路は Ok"
        );
        let c = &rec.counts;
        assert_eq!(c.kkt_factorize_attempts, 1);
        assert_eq!(c.materialize_original_calls, 1);
        assert_eq!(
            c.materialize_permuted_calls, 0,
            "DD は置換済み構築を実行しない"
        );
        assert!(caches.aug_permuted.is_none());
    }

    /// 失敗 retry + identity fallback の回数忠実性: 正則化成分を全ゼロにした
    /// 恒久不良系で、ループ全試行と fallback が「実際に実行された構築」と
    /// 1:1 で計上されることを検証する (review §2 検証2・§2.2.4)。元順序は全
    /// factorize 試行 (loop+fallback) で 1:1、置換済みは loop 分のみ —
    /// fallback が元順序-only 構築であることも同時に pin する。
    #[test]
    fn retry_and_identity_fallback_count_builds_actually_run() {
        // σ=0・delta=0・rho=0 (rho_retry は 0*GROWTH で 0 のまま) で正則化
        // 成分が消え、rank 不足な拡張 A により K が恒久不良。loop の各試行は
        // 数値 LDL 上は Ok だが健全プローブが拒否して retry になる (LISWET7
        // 実測と同じ probe 由来 retry)。identity fallback は delta_fallback
        // (>= LDL_FALLBACK_DELTA_MIN) 入りの元順序再構築で救済し結果は Ok。
        let fx = fixture(false);
        let (outcome, rec, _caches) = factorize_once(&fx, false, false, 0.0);
        let retry_count = match outcome {
            FactorizeOutcome::Ok { retry_count, .. } => retry_count,
            FactorizeOutcome::Timeout | FactorizeOutcome::Failure => {
                panic!("identity fallback で救済される想定 (Timeout/Failure は非期待)")
            }
        };
        let c = &rec.counts;
        // loop 10 試行 + fallback 1 試行 = 11。fallback は元順序のみ構築。
        assert_eq!(retry_count, LDL_REG_RETRY_MAX as u32);
        assert_eq!(c.kkt_factorize_attempts, LDL_REG_RETRY_MAX as u64 + 1);
        assert_eq!(
            c.materialize_original_calls, c.kkt_factorize_attempts,
            "元順序構築は全 factorize 試行と 1:1 (fallback 含む)"
        );
        assert_eq!(
            c.materialize_permuted_calls,
            c.kkt_factorize_attempts - 1,
            "置換済み構築は loop 実行分だけ (fallback 分を数えない)"
        );
        // loop の 10 試行は数値的に Ok を返すが健全プローブが全員拒否
        // (retry の出所は probe 失敗)。fallback 試行にプローブは走らない。
        assert_eq!(
            c.kkt_health_probes, LDL_REG_RETRY_MAX as u64,
            "プローブ拒否 10 回 = retry 回数と一致"
        );
    }
}
