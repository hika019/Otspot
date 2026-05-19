//! IP-PMM 主反復ループ (predictor-corrector + Gondzio + PMM ρ/δ 更新)。

use super::factorize::{
    auto_schur_enabled, factorize_kkt_with_retry, FactorizeCaches, FactorizeOutcome,
};
use super::init::build_initial_point;
use super::state::{
    alpha_stall_eps_for, PmmState, ALPHA_DEADLOCK_N, ALPHA_STALL_N, DELTA_INIT,
    DIRECTION_BLOWUP_THRESHOLD, DUALITY_GAP_TOL, MIN_CONSECUTIVE_INFEAS, MU_ZERO_THRESHOLD,
    PF_FAR_FROM_TARGET_RATIO, PF_HISTORY_LEN, PF_STUCK_RATIO, PMM_IMPROVE_THRESHOLD,
    PMM_SLOW_RATE, PROX_DOMINATE_RATIO, REG_LIMIT_MIN, REG_LIMIT_STEP, RESIDUAL_STALL_REL_DEC,
    RESIDUAL_STALL_WINDOW, RHO_INIT, STEP_REL_CAP,
};
use super::trace::{emit_active_trace, emit_iter_trace, emit_sigma_diag};
use crate::linalg::kkt_solver::inexact_eta_for_eps;
use crate::linalg::parallelism::solver_par_from_threads;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::ipm_core::common::{
    check_infeasible_or_unbounded, numerical_error_result, solve_unconstrained, timeout_result,
};
use crate::qp::ipm_core::kkt::{
    build_extended_constraints, collapse_extended_dual, norm_inf, spmtv, spmv, spmv_q,
};
use crate::qp::ipm_core::solver_loop::{
    compute_sigma_vec, corrector_step, corrector_step_schur, gondzio_correctors,
    gondzio_correctors_schur, predictor_step, predictor_step_schur, update_variables,
};
use crate::qp::problem::QpProblem;

/// IP-PMM 内部ソルバー (Ruiz scaling 後の problem を受け取る)。
pub(crate) fn solve_ippmm_inner(
    problem: &QpProblem,
    options: &SolverOptions,
    eps_orig: f64,
) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);
    let par = solver_par_from_threads(options.threads);

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    let (a_ext, b_ext, m_ext, m_orig, _n_lb, is_eq_ext) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    let eq_count = is_eq_ext.iter().filter(|&&v| v).count();
    let m_ineq = m_ext - eq_count;

    let init = build_initial_point(
        problem, options, &a_ext, &b_ext, &is_eq_ext, m_orig, m_ext, m_ineq,
        &timeout_ctx, par,
    );
    let (mut x, mut s, mut y, warm_mu) = (init.x, init.s, init.y, init.warm_mu);

    let (rho_init, delta_init) = match warm_mu {
        // warm start: μ 規模に揃えた rho/delta で出発し proximal pull を最小化。
        Some(mu) => {
            let v = mu.max(options.ipm.delta_min);
            (v, v)
        }
        None => (RHO_INIT, DELTA_INIT),
    };

    let mut pmm = PmmState {
        x_ref: x.clone(),
        y_ref: y.clone(),
        rho: rho_init,
        delta: delta_init,
        prev_nr_p: f64::INFINITY,
        prev_nr_d: f64::INFINITY,
    };

    // Gershgorin 由来の Q + δ_ic·I PSD 化量。凸 QP では 0。indefinite 判定 (返却 status 用) のみに使う。
    let inertia_correction = crate::qp::ipm_core::kkt::compute_inertia_correction(&problem.q);
    let q_is_indefinite = inertia_correction > 0.0;

    // QP_REG_LIMIT で override 可。
    let default_reg_qp = 5e-8;
    let default_reg_lp = 5e-10;
    let initial_reg_limit = std::env::var("QP_REG_LIMIT").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or_else(|| {
            if problem.q.values.iter().all(|&v| v == 0.0) {
                default_reg_lp
            } else {
                default_reg_qp
            }
        });
    // rank-deficient Q + c≈0 で rho が floor に張り付き proximal 項が df を支配する病理を回避する適応 floor。
    let c_max = problem.c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let allow_adaptive_reg = c_max < 1e-6;
    let mut reg_limit = initial_reg_limit;

    // pf-stagnation trigger (adaptive reg_limit の追加経路、c≠0 問題向け):
    // pf が最近の N 反復で実質改善せず (ratio > THRESHOLD) かつ pf が target から
    // 桁違いに離れている場合、reg_limit を下げて IPM が boundary を探索できる。
    let mut pf_history: Vec<f64> = Vec::with_capacity(PF_HISTORY_LEN);

    // 1 iter 単発の infeasible fire は noise なので、K iter 連続で確定。
    let mut consecutive_infeas_triggers: usize = 0;

    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];
    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];

    // 反復間で sparsity 不変な構造はキャッシュ。
    let aug_cache = crate::qp::ipm_core::kkt::build_augmented_cache(&problem.q, &a_ext);
    let mut factor_caches = FactorizeCaches::new();

    let inexact_eta = inexact_eta_for_eps(eps_orig);

    // augmented LDL が memory budget 超過なら Schur (n×n SPD) に切替。
    let explicit_schur = std::env::var("QP_SCHUR").ok().as_deref() == Some("1");
    let auto_schur = auto_schur_enabled(problem, &a_ext, m_ext, options, &timeout_ctx, par);
    let use_schur = explicit_schur || auto_schur;

    // 終了条件は Some(Optimal) / Some(Timeout) のみ。MaxIterations 経路は除去。
    let mut status: Option<SolveStatus> = None;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    // NaN guard で崩壊解を返さないための best-so-far スナップショット。
    let mut best_score = f64::INFINITY;
    let mut best_x = x.clone();
    let mut best_y = y.clone();
    let mut best_s = s.clone();
    let mut best_iter: usize = 0;
    let mut best_residuals: (f64, f64, f64) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut best_rel_gap: f64 = f64::INFINITY;

    let mut alpha_stall_count: usize = 0;

    let mut last_score_improvement_iter: usize = 0;
    let mut last_score_improvement_value: f64 = f64::INFINITY;

    let prof = std::env::var("IPM_PROF").ok().as_deref() == Some("1");
    let mut prof_iters: usize = 0;
    let mut prof_residual_ns: u128 = 0;

    let mut prof_factor_ns: u128 = 0;
    let mut prof_predcorr_ns: u128 = 0;
    let mut prof_gondzio_ns: u128 = 0;
    let mut prof_update_ns: u128 = 0;
    let prof_other_ns: u128 = 0;

    for iter in 0..options.ipm.max_iter {
        let prof_iter_start = if prof { Some(std::time::Instant::now()) } else { None };
        let mut prof_section_start = prof_iter_start;
        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = sᵀy / m_ineq (等式行除外)
        let mu: f64 = if m_ineq > 0 {
            s.iter().zip(y.iter()).zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>() / m_ineq as f64
        } else {
            0.0
        };

        let nr_p = norm_inf(&r_p);
        let nr_d = norm_inf(&r_d);
        final_residuals = Some((nr_p, nr_d, mu));

        // 符号規約: r_d = -(Qx + c + A^T y) → dual = -0.5 x^T Q x - Σ b_ext·y。
        let qx_dot_x: f64 = qx.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        let c_dot_x: f64 = problem.c.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        let p_obj_s = 0.5 * qx_dot_x + c_dot_x;
        let mut d_lin: f64 = 0.0;
        for i in 0..m_ext {
            d_lin -= b_ext[i] * y[i];
        }
        let d_obj_s = -0.5 * qx_dot_x + d_lin;
        let gap_abs = p_obj_s - d_obj_s;
        let gap_denom = p_obj_s.abs().max(d_obj_s.abs()).max(1.0);
        let rel_gap = gap_abs / gap_denom;

        // mu は dual と同スケール (sᵀy/m) なので ||c|| 大の問題でバイアスしないよう mu/(1+||c||) 正規化。
        let norm_c_bs = norm_inf(&problem.c).max(1.0);
        let norm_b_bs = norm_inf(&b_ext).max(1.0);
        if nr_p.is_finite() && nr_d.is_finite() && mu.is_finite() {
            let score = nr_p / (1.0 + norm_b_bs)
                + nr_d / (1.0 + norm_c_bs)
                + mu.abs() / (1.0 + norm_c_bs);
            if score < best_score {
                best_score = score;
                best_x.copy_from_slice(&x);
                best_y.copy_from_slice(&y);
                best_s.copy_from_slice(&s);
                best_iter = iter;
                best_residuals = (nr_p, nr_d, mu);
                best_rel_gap = rel_gap;
            }
            // residual 停滞検出: best_score が「有意に」減少したら improvement とみなす。
            if score < last_score_improvement_value * (1.0 - RESIDUAL_STALL_REL_DEC) {
                last_score_improvement_iter = iter;
                last_score_improvement_value = score;
            }
        }

        emit_iter_trace(iter, mu, nr_p, nr_d, &pmm, &x, &y, reg_limit);
        emit_active_trace(iter, m_ineq, &s, &y, &is_eq_ext);

        // per-row componentwise relative (bench と同形):
        //   primal: max_i |r_p[i]| / (1 + |ax[i]| + |b_ext[i]|)
        //   dual:   max_j |r_d[j]| / (1 + |qx[j]| + |c[j]| + |aty[j]|)
        let eps = options.ipm_eps();
        let nr_p_rel = {
            let mut m = 0.0_f64;
            for i in 0..m_ext {
                let denom_i = 1.0 + ax[i].abs() + b_ext[i].abs();
                let rel_i = r_p[i].abs() / denom_i;
                if rel_i > m { m = rel_i; }
            }
            m
        };
        let nr_d_rel = {
            let mut m = 0.0_f64;
            for j in 0..n {
                let denom_j = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs();
                let rel_j = r_d[j].abs() / denom_j;
                if rel_j > m { m = rel_j; }
            }
            m
        };

        if std::env::var("IPPMM_OPT_DIAG").ok().as_deref() == Some("1") {
            eprintln!(
                "IPPMM_OPT iter={} pf_rel={:.3e}/eps={:.3e}{} df_rel={:.3e}/eps={:.3e}{} mu={:.3e}/eps={:.3e}{} relgap={:.3e}/tol={:.3e}{}",
                iter,
                nr_p_rel, eps, if nr_p_rel < eps { "✓" } else { "✗" },
                nr_d_rel, eps, if nr_d_rel < eps { "✓" } else { "✗" },
                mu, eps, if mu < eps { "✓" } else { "✗" },
                rel_gap, DUALITY_GAP_TOL, if rel_gap.abs() < DUALITY_GAP_TOL { "✓" } else { "✗" },
            );
        }

        // 残差小・duality gap 大の偽 Optimal (rank-deficient Q + c=0) を弾くため rel_gap も要求。
        if nr_p_rel < eps && nr_d_rel < eps && mu < eps && rel_gap.abs() < DUALITY_GAP_TOL {
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "IPPMM_EXIT iter={} path=Optimal_main pf_rel={:.3e} df_rel={:.3e} rel_gap={:.3e}",
                    iter, nr_p_rel, nr_d_rel, rel_gap
                );
            }
            status = Some(SolveStatus::Optimal);
            final_iter = iter;
            break;
        }

        // Algorithm PEU: primal/dual 改善を独立判定。
        let primal_improved = PMM_IMPROVE_THRESHOLD * pmm.prev_nr_p > nr_p;
        let dual_improved = PMM_IMPROVE_THRESHOLD * pmm.prev_nr_d > nr_d;

        // r_d_pmm = r_d − ρ(x−x_ref), r_p_pmm = r_p − δ(y−y_ref)。
        let rho_prox = pmm.rho;
        let delta_prox = pmm.delta;
        let mut r_d_pmm = r_d.clone();
        let mut r_p_pmm = r_p.clone();
        for i in 0..n {
            r_d_pmm[i] -= rho_prox * (x[i] - pmm.x_ref[i]);
        }
        for i in 0..m_ext {
            r_p_pmm[i] -= delta_prox * (y[i] - pmm.y_ref[i]);
        }

        // Σ = diag(s_i / y_i) (等式行は0)
        let sigma_max = 1.0 / options.ipm.delta_min.max(MU_ZERO_THRESHOLD);
        let sigma_vec = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);

        emit_sigma_diag(iter, mu, &sigma_vec, &s, &y, &is_eq_ext);

        // 正則化は PMM 駆動。mu 依存 floor は使わない。
        let rho_matrix = pmm.rho.max(options.ipm.delta_min);
        let delta_matrix = pmm.delta.max(options.ipm.delta_min);

        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        if let Some(t) = prof_section_start {
            prof_residual_ns += t.elapsed().as_nanos();
            prof_section_start = Some(std::time::Instant::now());
        }

        let factorize_outcome = factorize_kkt_with_retry(
            problem, &a_ext, &aug_cache, &mut factor_caches,
            &sigma_vec, &is_eq_ext, &s, &r_d_pmm, &r_p_pmm,
            rho_matrix, delta_matrix, inertia_correction,
            use_schur, &timeout_ctx, par, n, prof,
        );
        let (mut fac, aug_mat, d_inv_opt, rho_retry) = match factorize_outcome {
            FactorizeOutcome::Ok { factor, aug_mat, d_inv, rho_used } => {
                (factor, aug_mat, d_inv, rho_used)
            }
            FactorizeOutcome::Timeout => {
                status = Some(SolveStatus::Timeout);
                final_iter = iter;
                break;
            }
            FactorizeOutcome::Failure => return numerical_error_result(n),
        };
        // MINRES (iterative) backend のみ user eps 由来 η を反映、Direct/DirectDd では no-op。
        fac.set_iterative_tol(inexact_eta);

        if let Some(t) = prof_section_start {
            prof_factor_ns += t.elapsed().as_nanos();
            prof_section_start = Some(std::time::Instant::now());
        }

        let (pred, alpha, r_c_corr) = if use_schur {
            let d_inv = d_inv_opt.as_ref().expect("d_inv must be set when use_schur");
            let pred = predictor_step_schur(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, &aug_mat, d_inv, &a_ext, n, m_ext, mu,
            );
            let (alpha, r_c_corr) = corrector_step_schur(
                &s, &y, &is_eq_ext,
                &pred, mu,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, &aug_mat, d_inv, &a_ext, n, m_ext,
                &mut dx, &mut dy, &mut ds,
            );
            (pred, alpha, r_c_corr)
        } else {
            let pred = predictor_step(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, &aug_mat, n, m_ext, mu,
                timeout_ctx.deadline,
            );
            let (alpha, r_c_corr) = corrector_step(
                &s, &y, &is_eq_ext,
                &pred, mu,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, &aug_mat, n, m_ext,
                &mut dx, &mut dy, &mut ds,
                timeout_ctx.deadline,
            );
            (pred, alpha, r_c_corr)
        };

        if let Some(t) = prof_section_start {
            prof_predcorr_ns += t.elapsed().as_nanos();
            prof_section_start = Some(std::time::Instant::now());
        }

        let mut alpha = alpha;
        if alpha < 0.999 {
            alpha = if use_schur {
                let d_inv = d_inv_opt.as_ref().expect("d_inv must be set when use_schur");
                gondzio_correctors_schur(
                    &s, &y, &is_eq_ext, m_ineq,
                    &r_d_pmm, &r_p_pmm,
                    &r_c_corr, &sigma_vec, &fac, &aug_mat, d_inv, &a_ext, n, m_ext,
                    options.ipm.max_correctors, alpha,
                    &mut dx, &mut dy, &mut ds,
                    timeout_ctx.deadline,
                )
            } else {
                gondzio_correctors(
                    &s, &y, &is_eq_ext, m_ineq,
                    &r_d_pmm, &r_p_pmm,
                    &r_c_corr, &sigma_vec, &fac, &aug_mat, n, m_ext,
                    options.ipm.max_correctors, alpha,
                    &mut dx, &mut dy, &mut ds,
                    timeout_ctx.deadline,
                )
            };
        }

        let _ = pred;

        // NaN/Inf または finite-but-huge は LDL blow-up とみなし best-so-far で復帰。
        let direction_finite_but_huge = dx.iter().chain(dy.iter()).chain(ds.iter())
            .any(|v| v.is_finite() && v.abs() > DIRECTION_BLOWUP_THRESHOLD);
        if dx.iter().any(|v| !v.is_finite())
            || dy.iter().any(|v| !v.is_finite())
            || ds.iter().any(|v| !v.is_finite())
            || direction_finite_but_huge
        {
            if best_score.is_finite() {
                x.copy_from_slice(&best_x);
                y.copy_from_slice(&best_y);
                s.copy_from_slice(&best_s);
                final_iter = best_iter;
                final_residuals = Some(best_residuals);
                let quality_threshold = 10.0 * eps_orig;
                let combined_quasi = best_score < quality_threshold
                    && best_rel_gap.abs() < DUALITY_GAP_TOL;
                let feasibility_quasi = best_residuals.0 < eps_orig
                    && best_residuals.1 < eps_orig;
                let is_quasi_optimal = combined_quasi || feasibility_quasi;
                let exit_status = if is_quasi_optimal {
                    SolveStatus::Optimal
                } else {
                    SolveStatus::SuboptimalSolution
                };
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
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
                status = Some(exit_status);
            } else {
                final_iter = iter;
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!("IPPMM_EXIT iter={} path=NumericalError_NaN_guard_no_best", iter);
                }
                status = Some(SolveStatus::NumericalError);
            }
            break;
        }

        // check_infeasible_or_unbounded は Newton 方向の Farkas-like 近似なので
        // PMM floor 起因の false-positive がありうる。best-so-far がある間は信用せず、
        // best が無い時のみ Infeasible/Unbounded を確定とみなす。
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter, rho_retry,
        ) {
            consecutive_infeas_triggers += 1;
            let quality_threshold = 10.0 * eps_orig;
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!("IPPMM_DEBUG iter={} best_score={:e} quality_threshold={:e} eps_orig={:e} eps={:e} best_finite={} consecutive_infeas={}", iter, best_score, quality_threshold, eps_orig, eps, best_score.is_finite(), consecutive_infeas_triggers);
            }
            if best_score.is_finite()
                && best_score < quality_threshold
                && best_rel_gap.abs() < DUALITY_GAP_TOL
            {
                x.copy_from_slice(&best_x);
                y.copy_from_slice(&best_y);
                s.copy_from_slice(&best_s);
                final_iter = best_iter;
                final_residuals = Some(best_residuals);
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "IPPMM_EXIT iter={} path=reject_false_{:?}_bestsofar best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
                        iter, infeas_status, best_iter, best_score, best_rel_gap,
                        best_residuals.0, best_residuals.1, best_residuals.2
                    );
                }
                status = Some(SolveStatus::Optimal);
                break;
            }
            // N 連続 fire まで判定保留: PMM floor の false-positive に adaptive reg の猶予を与える。
            if consecutive_infeas_triggers < MIN_CONSECUTIVE_INFEAS {
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "IPPMM_DEBUG iter={} infeas trigger #{} (< {}), continue iterating",
                        iter, consecutive_infeas_triggers, MIN_CONSECUTIVE_INFEAS
                    );
                }
            } else {
                if best_score < quality_threshold {
                    x.copy_from_slice(&best_x);
                    y.copy_from_slice(&best_y);
                    s.copy_from_slice(&best_s);
                    final_iter = best_iter;
                    final_residuals = Some(best_residuals);
                    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "IPPMM_EXIT iter={} path=demote_{:?}_to_suboptimal_bestsofar best_iter={} best_score={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e}) consecutive={}",
                            iter, infeas_status, best_iter, best_score,
                            best_residuals.0, best_residuals.1, best_residuals.2,
                            consecutive_infeas_triggers
                        );
                    }
                    status = Some(SolveStatus::SuboptimalSolution);
                    break;
                }
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!("IPPMM_EXIT iter={} path=check_infeas status={:?} best_score={:.3e} consecutive={}", iter, infeas_status, best_score, consecutive_infeas_triggers);
                }
                status = Some(infeas_status);
                final_iter = iter;
                break;
            }
        } else {
            // 検出器が反応しなかった iter で carry-over count をリセット。
            consecutive_infeas_triggers = 0;
        }

        let ndx = dx.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let ndy = dy.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let nds = ds.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
            let nrdpmm = r_d_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let nrppmm = r_p_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            eprintln!(
                "IPPMM_STEP iter={:4} alpha={:.6e} dx_inf={:.3e} dy_inf={:.3e} ds_inf={:.3e} rdpmm_inf={:.3e} rppmm_inf={:.3e}",
                iter, alpha, ndx, ndy, nds, nrdpmm, nrppmm
            );
        }

        // Trust-region cap: alpha·|dv|_inf ≤ STEP_REL_CAP·max(|v|_inf, 1)。
        // fraction-to-boundary は s,y>0 のみ保護で dx は無制約 → cap で 1 iter 3 桁以上の暴発を抑制。
        let nx_safe = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let ny_safe = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let ns_safe = s.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let alpha_x_cap = if ndx > 0.0 { (STEP_REL_CAP * nx_safe / ndx).min(1.0) } else { 1.0 };
        let alpha_y_cap = if ndy > 0.0 { (STEP_REL_CAP * ny_safe / ndy).min(1.0) } else { 1.0 };
        let alpha_s_cap = if nds > 0.0 { (STEP_REL_CAP * ns_safe / nds).min(1.0) } else { 1.0 };
        let alpha_tr = alpha_x_cap.min(alpha_y_cap).min(alpha_s_cap);
        let alpha = alpha.min(alpha_tr);

        if let Some(t) = prof_section_start {
            prof_gondzio_ns += t.elapsed().as_nanos();
            prof_section_start = Some(std::time::Instant::now());
        }

        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, alpha, &is_eq_ext);

        // alpha=0 持続 = line search 停止 = 数値飽和 / null-space 漂流。best-so-far で復帰。
        if alpha < alpha_stall_eps_for(eps_orig) {
            alpha_stall_count += 1;
        } else {
            alpha_stall_count = 0;
        }
        // 真収束後の停滞のみ早期脱出 (best_score < eps)、マージナル問題は timeout 側に委ねる。
        let alpha_stall_converged = best_score.is_finite() && best_score < eps;
        let alpha_stall_deadlock = alpha_stall_count >= ALPHA_DEADLOCK_N
            && best_score.is_finite()
            && pmm.rho <= reg_limit * 1.01
            && pmm.delta <= reg_limit * 1.01;
        if alpha_stall_count >= ALPHA_STALL_N
            && (alpha_stall_converged || alpha_stall_deadlock)
        {
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            final_iter = best_iter;
            final_residuals = Some(best_residuals);
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                let exit_reason = if alpha_stall_converged { "conv" } else { "deadlock" };
                eprintln!(
                    "IPPMM_EXIT iter={} path=Suboptimal_alpha_stall_bestsofar reason={} stall_count={} best_iter={} best_score={:.3e} best_rel_gap={:.3e} rho={:.3e} reg_limit={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
                    iter, exit_reason, alpha_stall_count, best_iter, best_score, best_rel_gap,
                    pmm.rho, reg_limit,
                    best_residuals.0, best_residuals.1, best_residuals.2
                );
            }
            status = Some(SolveStatus::SuboptimalSolution);
            break;
        }

        // alpha > 0 でも best_score が窓内で改善しない病理向け (alpha-stall と独立)。
        let residual_stall = best_score.is_finite()
            && iter >= last_score_improvement_iter + RESIDUAL_STALL_WINDOW
            && best_score >= eps;
        if residual_stall {
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            final_iter = best_iter;
            final_residuals = Some(best_residuals);
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "IPPMM_EXIT iter={} path=Suboptimal_residual_stall_bestsofar window={} last_improve_iter={} best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
                    iter, RESIDUAL_STALL_WINDOW, last_score_improvement_iter,
                    best_iter, best_score, best_rel_gap,
                    best_residuals.0, best_residuals.1, best_residuals.2
                );
            }
            status = Some(SolveStatus::SuboptimalSolution);
            break;
        }

        // Algorithm PEU Step 0: r = |μ_k − μ_{k+1}| / μ_k (corrector + line search 後の実 μ)。
        let mu_new: f64 = if m_ineq > 0 {
            s.iter().zip(y.iter()).zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>() / m_ineq as f64
        } else {
            0.0
        };
        let r = if mu > MU_ZERO_THRESHOLD || mu_new > MU_ZERO_THRESHOLD {
            (mu - mu_new).abs() / mu.max(mu_new).max(MU_ZERO_THRESHOLD)
        } else {
            0.0
        };

        // 等式 (mu=0) では mu_rate=0.9 で高速減衰、それ以外は r を [0.2, 0.9] で clamp。
        let mu_rate_raw = if mu < MU_ZERO_THRESHOLD && mu_new < MU_ZERO_THRESHOLD { 0.9 } else { r };
        let mu_rate = mu_rate_raw.clamp(0.2, 0.9);

        pf_history.push(nr_p);
        if pf_history.len() > PF_HISTORY_LEN {
            pf_history.remove(0);
        }

        // Adaptive reg_limit: prox が df を支配 (c≈0) または pf が窓内停滞 + target から遠い場合、floor を下げる。
        if (pmm.rho - reg_limit).abs() < reg_limit * 0.01 && reg_limit > REG_LIMIT_MIN {
            let mut should_lower = false;
            if allow_adaptive_reg {
                let prox_d_inf = x.iter().zip(pmm.x_ref.iter())
                    .map(|(&xi, &xref)| (pmm.rho * (xi - xref)).abs())
                    .fold(0.0_f64, f64::max);
                if prox_d_inf > nr_d * PROX_DOMINATE_RATIO && nr_d > 0.0 {
                    should_lower = true;
                }
            }
            if !should_lower
                && pf_history.len() == PF_HISTORY_LEN
                && pf_history[0] > 0.0
                && nr_p > eps_orig * PF_FAR_FROM_TARGET_RATIO
            {
                let ratio = nr_p / pf_history[0];
                if ratio > PF_STUCK_RATIO {
                    should_lower = true;
                }
            }
            if should_lower {
                reg_limit = (reg_limit * REG_LIMIT_STEP).max(REG_LIMIT_MIN);
                pf_history.clear();
            }
        }

        // Algorithm PEU Step 1&2 (OR 判定): どちらか改善があれば δ,ρ 両方を mu_rate で更新。
        let either_improved = primal_improved || dual_improved;
        let force_ref_update = std::env::var("IPPMM_FORCE_REF_UPDATE").ok().as_deref() == Some("1");
        if either_improved || force_ref_update {
            pmm.y_ref.copy_from_slice(&y);
            pmm.x_ref.copy_from_slice(&x);
            pmm.delta = (pmm.delta * (1.0 - mu_rate)).max(reg_limit);
            pmm.rho   = (pmm.rho   * (1.0 - mu_rate)).max(reg_limit);
        } else {
            pmm.delta = (pmm.delta * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
            pmm.rho   = (pmm.rho   * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
        }

        pmm.prev_nr_p = nr_p;
        pmm.prev_nr_d = nr_d;

        if let Some(t) = prof_section_start {
            prof_update_ns += t.elapsed().as_nanos();
        }
        if let Some(t) = prof_iter_start {
            let _ = t.elapsed().as_nanos();
        }
        prof_iters += 1;
    }

    if prof {
        let total_ns = prof_residual_ns + prof_factor_ns + prof_predcorr_ns + prof_gondzio_ns + prof_update_ns + prof_other_ns;
        let total_ms = total_ns as f64 / 1_000_000.0;
        let frac = |v: u128| -> f64 { 100.0 * v as f64 / total_ns.max(1) as f64 };
        eprintln!(
            "IPM_PROF iters={} total={:.1}ms residual={:.1}ms({:.1}%) factor={:.1}ms({:.1}%) predcorr={:.1}ms({:.1}%) gondzio={:.1}ms({:.1}%) update={:.1}ms({:.1}%)",
            prof_iters,
            total_ms,
            prof_residual_ns as f64 / 1e6, frac(prof_residual_ns),
            prof_factor_ns as f64 / 1e6, frac(prof_factor_ns),
            prof_predcorr_ns as f64 / 1e6, frac(prof_predcorr_ns),
            prof_gondzio_ns as f64 / 1e6, frac(prof_gondzio_ns),
            prof_update_ns as f64 / 1e6, frac(prof_update_ns),
        );
    }

    let status = status.unwrap_or(SolveStatus::Timeout);

    // 素の Timeout 経路は発散 x をそのまま返してしまうので best-so-far で上書き。
    if matches!(status, SolveStatus::Timeout | SolveStatus::MaxIterations)
        && best_score.is_finite()
    {
        let norm_b_bs = norm_inf(&b_ext).max(1.0);
        let norm_c_bs = norm_inf(&problem.c).max(1.0);
        let current_score = match final_residuals {
            Some((nr_p, nr_d, mu)) if nr_p.is_finite() && nr_d.is_finite() && mu.is_finite() => {
                nr_p / (1.0 + norm_b_bs) + nr_d / (1.0 + norm_c_bs) + mu.abs()
            }
            _ => f64::INFINITY,
        };
        if best_score < current_score {
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            final_iter = best_iter;
            final_residuals = Some(best_residuals);
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "IPPMM_EXIT path=Timeout_bestsofar_fallback best_iter={} best_score={:.3e} best_rel_gap={:.3e} best=(pf={:.3e},df={:.3e},mu={:.3e})",
                    best_iter, best_score, best_rel_gap,
                    best_residuals.0, best_residuals.1, best_residuals.2
                );
            }
        }
    }

    spmv_q(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = collapse_extended_dual(&y, m_orig, &problem.constraint_types);
    let bound_duals = y[m_orig..].to_vec();

    // 不定 Q の Optimal は慣性修正により局所最適に降格。
    let final_status = if q_is_indefinite && status == SolveStatus::Optimal {
        SolveStatus::LocallyOptimal
    } else {
        status
    };

    SolverResult {
        status: final_status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,

        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        // best-so-far の rel gap。unscale_ipm_result の昇格ゲート用。
        duality_gap_rel: if best_rel_gap.is_finite() { Some(best_rel_gap) } else { None },
        ..Default::default()
    }
}

