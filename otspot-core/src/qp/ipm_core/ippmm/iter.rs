//! IP-PMM 主反復ループ (predictor-corrector + Gondzio + PMM ρ/δ 更新)。

use super::factorize::{
    auto_schur_enabled, factorize_kkt_with_retry, FactorizeCaches, FactorizeContext,
    FactorizeOutcome,
};
use super::init::build_initial_point;
use super::state::{
    alpha_stall_eps_for, PmmState, ADAPTIVE_REG_C_MAX_THRESH, ALPHA_DEADLOCK_N, ALPHA_STALL_N,
    DELTA_INIT, DIRECTION_BLOWUP_THRESHOLD, DUALITY_GAP_TOL, GONDZIO_ALPHA_TRIGGER,
    MIN_CONSECUTIVE_INFEAS, MU_ZERO_THRESHOLD, PF_FAR_FROM_TARGET_RATIO, PF_HISTORY_LEN,
    PF_STUCK_RATIO, PMM_IMPROVE_THRESHOLD, PMM_SLOW_RATE, PROX_DOMINATE_RATIO, REG_LIMIT_INIT_LP,
    REG_LIMIT_INIT_QP, REG_LIMIT_MIN, REG_LIMIT_STEP, RESIDUAL_STALL_REL_DEC,
    RESIDUAL_STALL_WINDOW, RHO_INIT, STEP_REL_CAP,
};
use crate::linalg::kkt_solver::{inexact_eta_for_eps, KktConfig};
use crate::linalg::parallelism::solver_par_from_threads;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::ipm_core::common::{
    check_infeasible_or_unbounded, numerical_error_result, solve_unconstrained, timeout_result,
};
use crate::qp::ipm_core::kkt::{
    build_extended_constraints, collapse_extended_dual, norm_inf, spmtv, spmv,
};
use crate::qp::ipm_core::solver_loop::{
    compute_sigma_vec, corrector_step, corrector_step_schur, gondzio_correctors,
    gondzio_correctors_schur, predictor_step, predictor_step_schur, update_variables,
};
use crate::qp::problem::QpProblem;
use crate::tolerances::any_nonfinite;

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
        && problem
            .bounds
            .iter()
            .all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
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
        problem,
        options,
        &a_ext,
        &b_ext,
        &is_eq_ext,
        m_orig,
        m_ext,
        &timeout_ctx,
        par,
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

    let initial_reg_limit = if problem.q.values.iter().all(|&v| v == 0.0) {
        REG_LIMIT_INIT_LP
    } else {
        REG_LIMIT_INIT_QP
    };
    // rank-deficient Q + c≈0 で rho が floor に張り付き proximal 項が df を支配する病理を回避する適応 floor。
    let c_max = problem.c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let allow_adaptive_reg = c_max < ADAPTIVE_REG_C_MAX_THRESH;
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
    let use_schur = auto_schur_enabled(problem, &a_ext, m_ext, options, &timeout_ctx, par);

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

    let mut total_factorize_ns: u128 = 0;
    let mut total_solve_ns: u128 = 0;
    let mut total_reg_retries: u32 = 0;
    let mut any_iterative = false;

    for iter in 0..options.ipm.max_iter {
        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = sᵀy / m_ineq (等式行除外)
        let mu: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>()
                / m_ineq as f64
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
            let score =
                nr_p / (1.0 + norm_b_bs) + nr_d / (1.0 + norm_c_bs) + mu.abs() / (1.0 + norm_c_bs);
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

        // per-row componentwise relative (bench と同形):
        //   primal: max_i |r_p[i]| / (1 + |ax[i]| + |b_ext[i]|)
        //   dual:   max_j |r_d[j]| / (1 + |qx[j]| + |c[j]| + |aty[j]|)
        let eps = options.ipm_eps();
        let nr_p_rel = {
            let mut m = 0.0_f64;
            for i in 0..m_ext {
                let denom_i = 1.0 + ax[i].abs() + b_ext[i].abs();
                let rel_i = r_p[i].abs() / denom_i;
                if rel_i > m {
                    m = rel_i;
                }
            }
            m
        };
        let nr_d_rel = {
            let mut m = 0.0_f64;
            for j in 0..n {
                let denom_j = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs();
                let rel_j = r_d[j].abs() / denom_j;
                if rel_j > m {
                    m = rel_j;
                }
            }
            m
        };

        // 残差小・duality gap 大の偽 Optimal (rank-deficient Q + c=0) を弾くため rel_gap も要求。
        if nr_p_rel < eps && nr_d_rel < eps && mu < eps && rel_gap.abs() < DUALITY_GAP_TOL {
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

        // 正則化は PMM 駆動。mu 依存 floor は使わない。
        let rho_matrix = pmm.rho.max(options.ipm.delta_min);
        let delta_matrix = pmm.delta.max(options.ipm.delta_min);

        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        let fact_ctx = FactorizeContext {
            problem,
            a_ext: &a_ext,
            aug_cache: &aug_cache,
            sigma_vec: &sigma_vec,
            is_eq_ext: &is_eq_ext,
            s: &s,
            r_d_pmm: &r_d_pmm,
            r_p_pmm: &r_p_pmm,
            rho_matrix,
            delta_matrix,
            inertia_correction,
            use_schur,
            timeout_ctx: &timeout_ctx,
            par,
            n,
            kkt_cfg: KktConfig {
                dd_ldl: options.ipm.dd_ldl,
                minres_ir: options.ipm.effective_minres_ir(),
                max_l_nnz: options.ipm.effective_max_l_nnz(),
            },
        };
        let factorize_outcome = factorize_kkt_with_retry(&fact_ctx, &mut factor_caches);
        let (mut fac, aug_mat, d_inv_opt, rho_retry) = match factorize_outcome {
            FactorizeOutcome::Ok {
                factor,
                aug_mat,
                d_inv,
                rho_used,
                retry_count,
                used_iterative,
                factorize_ns,
            } => {
                total_factorize_ns += factorize_ns;
                total_reg_retries += retry_count;
                any_iterative |= used_iterative;
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

        let t_solve = std::time::Instant::now();
        let (alpha, r_c_corr) = if use_schur {
            let d_inv = d_inv_opt
                .as_ref()
                .expect("d_inv must be set when use_schur");
            let pred = predictor_step_schur(
                &s,
                &y,
                &is_eq_ext,
                m_ineq,
                &r_d_pmm,
                &r_p_pmm,
                &sigma_vec,
                &fac,
                d_inv,
                &a_ext,
                m_ext,
                mu,
                timeout_ctx.deadline,
            );
            corrector_step_schur(
                &s,
                &y,
                &is_eq_ext,
                &pred,
                mu,
                &r_d_pmm,
                &r_p_pmm,
                &sigma_vec,
                &fac,
                d_inv,
                &a_ext,
                m_ext,
                &mut dx,
                &mut dy,
                &mut ds,
                timeout_ctx.deadline,
            )
        } else {
            let pred = predictor_step(
                &s,
                &y,
                &is_eq_ext,
                m_ineq,
                &r_d_pmm,
                &r_p_pmm,
                &sigma_vec,
                &fac,
                &aug_mat,
                n,
                m_ext,
                mu,
                timeout_ctx.deadline,
            );
            corrector_step(
                &s,
                &y,
                &is_eq_ext,
                &pred,
                mu,
                &r_d_pmm,
                &r_p_pmm,
                &sigma_vec,
                &fac,
                &aug_mat,
                n,
                m_ext,
                &mut dx,
                &mut dy,
                &mut ds,
                timeout_ctx.deadline,
            )
        };

        let mut alpha = alpha;
        if alpha < GONDZIO_ALPHA_TRIGGER {
            alpha = if use_schur {
                let d_inv = d_inv_opt
                    .as_ref()
                    .expect("d_inv must be set when use_schur");
                gondzio_correctors_schur(
                    &s,
                    &y,
                    &is_eq_ext,
                    m_ineq,
                    &r_d_pmm,
                    &r_p_pmm,
                    &r_c_corr,
                    &sigma_vec,
                    &fac,
                    d_inv,
                    &a_ext,
                    m_ext,
                    options.ipm.max_correctors,
                    alpha,
                    &mut dx,
                    &mut dy,
                    &mut ds,
                    timeout_ctx.deadline,
                )
            } else {
                gondzio_correctors(
                    &s,
                    &y,
                    &is_eq_ext,
                    m_ineq,
                    &r_d_pmm,
                    &r_p_pmm,
                    &r_c_corr,
                    &sigma_vec,
                    &fac,
                    &aug_mat,
                    n,
                    m_ext,
                    options.ipm.max_correctors,
                    alpha,
                    &mut dx,
                    &mut dy,
                    &mut ds,
                    timeout_ctx.deadline,
                )
            };
        }

        // NaN/Inf または finite-but-huge は LDL blow-up とみなし best-so-far で復帰。
        let direction_finite_but_huge = dx
            .iter()
            .chain(dy.iter())
            .chain(ds.iter())
            .any(|v| v.is_finite() && v.abs() > DIRECTION_BLOWUP_THRESHOLD);
        if any_nonfinite(&dx)
            || any_nonfinite(&dy)
            || any_nonfinite(&ds)
            || direction_finite_but_huge
        {
            if best_score.is_finite() {
                x.copy_from_slice(&best_x);
                y.copy_from_slice(&best_y);
                s.copy_from_slice(&best_s);
                final_iter = best_iter;
                final_residuals = Some(best_residuals);
                let quality_threshold = 10.0 * eps_orig;
                let combined_quasi =
                    best_score < quality_threshold && best_rel_gap.abs() < DUALITY_GAP_TOL;
                let feasibility_quasi = best_residuals.0 < eps_orig && best_residuals.1 < eps_orig;
                let is_quasi_optimal = combined_quasi || feasibility_quasi;
                let exit_status = if is_quasi_optimal {
                    SolveStatus::Optimal
                } else {
                    SolveStatus::SuboptimalSolution
                };
                status = Some(exit_status);
            } else {
                final_iter = iter;
                status = Some(SolveStatus::NumericalError);
            }
            break;
        }

        // check_infeasible_or_unbounded は Newton 方向の Farkas-like 近似なので
        // PMM floor 起因の false-positive がありうる。best-so-far がある間は信用せず、
        // best が無い時のみ Infeasible/Unbounded を確定とみなす。
        if let Some(infeas_status) =
            check_infeasible_or_unbounded(&dx, &dy, problem, &a_ext, m_orig, m_ext, iter, rho_retry)
        {
            consecutive_infeas_triggers += 1;
            let quality_threshold = 10.0 * eps_orig;
            if best_score.is_finite()
                && best_score < quality_threshold
                && best_rel_gap.abs() < DUALITY_GAP_TOL
            {
                x.copy_from_slice(&best_x);
                y.copy_from_slice(&best_y);
                s.copy_from_slice(&best_s);
                final_iter = best_iter;
                final_residuals = Some(best_residuals);
                status = Some(SolveStatus::Optimal);
                break;
            }
            // N 連続 fire まで判定保留: PMM floor の false-positive に adaptive reg の猶予を与える。
            if consecutive_infeas_triggers < MIN_CONSECUTIVE_INFEAS {
            } else {
                if best_score < quality_threshold {
                    x.copy_from_slice(&best_x);
                    y.copy_from_slice(&best_y);
                    s.copy_from_slice(&best_s);
                    final_iter = best_iter;
                    final_residuals = Some(best_residuals);
                    status = Some(SolveStatus::SuboptimalSolution);
                    break;
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

        // Trust-region cap: alpha·|dv|_inf ≤ STEP_REL_CAP·max(|v|_inf, 1)。
        // fraction-to-boundary は s,y>0 のみ保護で dx は無制約 → STEP_REL_CAP (=1e3) で 1 iter 3 桁以上の暴発を抑制。
        let nx_safe = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let ny_safe = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let ns_safe = s.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        let alpha_x_cap = if ndx > 0.0 {
            (STEP_REL_CAP * nx_safe / ndx).min(1.0)
        } else {
            1.0
        };
        let alpha_y_cap = if ndy > 0.0 {
            (STEP_REL_CAP * ny_safe / ndy).min(1.0)
        } else {
            1.0
        };
        let alpha_s_cap = if nds > 0.0 {
            (STEP_REL_CAP * ns_safe / nds).min(1.0)
        } else {
            1.0
        };
        let alpha_tr = alpha_x_cap.min(alpha_y_cap).min(alpha_s_cap);
        let alpha = alpha.min(alpha_tr);

        // predictor/corrector + Gondzio 全体の solve 時間を常時収集。
        total_solve_ns += t_solve.elapsed().as_nanos();

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
        if alpha_stall_count >= ALPHA_STALL_N && (alpha_stall_converged || alpha_stall_deadlock) {
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            final_iter = best_iter;
            final_residuals = Some(best_residuals);
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
            status = Some(SolveStatus::SuboptimalSolution);
            break;
        }

        // Algorithm PEU Step 0: r = |μ_k − μ_{k+1}| / μ_k (corrector + line search 後の実 μ)。
        let mu_new: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>()
                / m_ineq as f64
        } else {
            0.0
        };
        let r = if mu > MU_ZERO_THRESHOLD || mu_new > MU_ZERO_THRESHOLD {
            (mu - mu_new).abs() / mu.max(mu_new).max(MU_ZERO_THRESHOLD)
        } else {
            0.0
        };

        // For equality-only problems (mu≈0) use a fixed fast-decay rate;
        // otherwise clamp the relative mu reduction to a stable range.
        const MU_RATE_EQ: f64 = 0.9;
        const MU_RATE_MIN: f64 = 0.2;
        const MU_RATE_MAX: f64 = 0.9;
        let mu_rate_raw = if mu < MU_ZERO_THRESHOLD && mu_new < MU_ZERO_THRESHOLD {
            MU_RATE_EQ
        } else {
            r
        };
        let mu_rate = mu_rate_raw.clamp(MU_RATE_MIN, MU_RATE_MAX);

        pf_history.push(nr_p);
        if pf_history.len() > PF_HISTORY_LEN {
            pf_history.remove(0);
        }

        // Adaptive reg_limit: prox が df を支配 (c≈0) または pf が窓内停滞 + target から遠い場合、floor を下げる。
        if (pmm.rho - reg_limit).abs() < reg_limit * 0.01 && reg_limit > REG_LIMIT_MIN {
            let mut should_lower = false;
            if allow_adaptive_reg {
                let prox_d_inf = x
                    .iter()
                    .zip(pmm.x_ref.iter())
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

        // PEU Step 1&2: for box-only QPs (m_orig == 0), require both primal AND dual
        // to improve before fast-decreasing δ,ρ (P-G 2021 Algorithm 1 outer-loop intent).
        // Without this, δ collapses via dual-only improvement → dy ≈ r_p/δ blows up.
        // For m_orig > 0, OR logic is kept: the original linear constraints in a_ext
        // contribute a Schur term A_orig(Q+ρI)⁻¹A_origᵀ/δ that must grow as δ→0 to
        // drive primal feasibility. (Bound rows remain in a_ext for both cases; it is
        // the original-constraint Schur term that differentiates the δ requirements.)
        let box_only = m_orig == 0;
        let both_improved = primal_improved && dual_improved;
        let either_improved = primal_improved || dual_improved;
        let use_fast_rate = if box_only {
            both_improved
        } else {
            either_improved
        };
        // Update reference point whenever at least one residual improved.
        if either_improved {
            pmm.y_ref.copy_from_slice(&y);
            pmm.x_ref.copy_from_slice(&x);
        }
        if use_fast_rate {
            pmm.delta = (pmm.delta * (1.0 - mu_rate)).max(reg_limit);
            pmm.rho = (pmm.rho * (1.0 - mu_rate)).max(reg_limit);
        } else {
            pmm.delta = (pmm.delta * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
            pmm.rho = (pmm.rho * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
        }

        pmm.prev_nr_p = nr_p;
        pmm.prev_nr_d = nr_d;
    }

    let status = status.unwrap_or(SolveStatus::Timeout);

    // 素の Timeout 経路は発散 x をそのまま返してしまうので best-so-far で上書き。
    if matches!(status, SolveStatus::Timeout | SolveStatus::MaxIterations) && best_score.is_finite()
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
        }
    }

    spmv(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter()
            .zip(x.iter())
            .map(|(&qi, &xi)| qi * xi)
            .sum::<f64>()
        + problem
            .c
            .iter()
            .zip(x.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>();

    let dual_solution = collapse_extended_dual(&y, m_orig, &problem.constraint_types);
    let bound_duals = y[m_orig..].to_vec();

    // 不定 Q の Optimal は慣性修正により局所最適に降格。
    let final_status = if q_is_indefinite && status == SolveStatus::Optimal {
        SolveStatus::LocallyOptimal
    } else {
        status
    };

    let ipm_timing = crate::problem::TimingBreakdown {
        ipm_factorize_us: (total_factorize_ns / 1_000) as u64,
        ipm_solve_us: (total_solve_ns / 1_000) as u64,
        ipm_reg_retries: total_reg_retries,
        ipm_used_iterative: any_iterative,
        ..Default::default()
    };

    SolverResult {
        status: final_status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,

        iterations: final_iter,
        final_residuals,
        // best-so-far の rel gap。unscale_ipm_result の昇格ゲート用。
        duality_gap_rel: if best_rel_gap.is_finite() {
            Some(best_rel_gap)
        } else {
            None
        },
        timing_breakdown: Some(ipm_timing),
        ..Default::default()
    }
}
