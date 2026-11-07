//! IP-PMM 主反復ループ (predictor-corrector + Gondzio + PMM ρ/δ 更新)。

use super::factorize::{
    auto_schur_enabled, factorize_kkt_with_retry, FactorizeCaches, FactorizeContext,
    FactorizeOutcome,
};
use super::init::build_initial_point;
use super::state::{
    alpha_stall_eps_for, duality_gap, matrix_reg_floor_for_lp, pf_stuck_should_lower_reg_limit,
    ConvergenceMeasures, PmmState, ADAPTIVE_REG_C_MAX_THRESH, ALPHA_DEADLOCK_N, ALPHA_STALL_N,
    DELTA_INIT, DIRECTION_BLOWUP_THRESHOLD, GONDZIO_ALPHA_TRIGGER, INFEAS_DETECTOR_DISTRUST_SCORE,
    MIN_CONSECUTIVE_INFEAS, MU_ZERO_THRESHOLD, PF_HISTORY_LEN, PMM_IMPROVE_THRESHOLD,
    PMM_SLOW_RATE, PROX_DOMINATE_RATIO, REG_LIMIT_INIT_LP, REG_LIMIT_INIT_QP, REG_LIMIT_MIN,
    REG_LIMIT_STEP, RESIDUAL_STALL_REL_DEC, RESIDUAL_STALL_WINDOW, RHO_INIT, SIGMA_MAX_FALLBACK,
    STEP_REL_CAP,
};
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
use faer::Par;
use otspot_num::linalg::kkt_solver::{inexact_eta_for_eps, KktConfig};
use otspot_num::linalg::parallelism::with_solver_pool;
use otspot_num::linalg::timeout::TimeoutCtx;

#[cfg(test)]
thread_local! {
    /// Test-only cancel injection for the Infeasible/Unbounded certificate
    /// commit point below: counts calls to `test_record_infeas_commit`
    /// (invoked right before the consecutive-fire detector's conclusion is
    /// accepted); once the count reaches `CANCEL_AFTER_INFEAS_COMMIT`, flips
    /// `CANCEL_SIGNAL` -- the same shared `AtomicBool` a test threads through
    /// `SolverOptions::cancel_flag` -- so `timeout_ctx.should_stop()`
    /// observes a freshly-fired cancellation exactly at the commit point,
    /// without racing wall-clock time. Mirrors
    /// `presolve::qp_transforms::driver`'s
    /// `STEPS_EXECUTED_TOTAL`/`CANCEL_AFTER_STEPS`/`CANCEL_SIGNAL` pattern.
    /// `pub(super)` so the sibling `ippmm::tests` module can drive it.
    pub(super) static INFEAS_COMMIT_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    pub(super) static CANCEL_AFTER_INFEAS_COMMIT: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    pub(super) static CANCEL_SIGNAL: std::sync::Arc<std::sync::atomic::AtomicBool> =
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
}

#[cfg(test)]
fn test_record_infeas_commit() {
    let count = INFEAS_COMMIT_COUNT.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if CANCEL_AFTER_INFEAS_COMMIT.with(std::cell::Cell::get) == Some(count) {
        CANCEL_SIGNAL.with(|flag| flag.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

/// Consecutive-fire commit stop check, factored out so the call
/// site is a single condition instead of a `#[cfg(test)]` hook line plus the
/// real check.
fn should_stop_before_infeas_commit(timeout_ctx: &TimeoutCtx) -> bool {
    #[cfg(test)]
    test_record_infeas_commit();
    timeout_ctx.should_stop()
}

/// IP-PMM 内部ソルバー (Ruiz scaling 後の problem を受け取る)。
///
/// `SolverOptions::threads` 専用の rayon プールへ solve 全体を閉じ込め、その
/// プールに対応する `Par` を本体へ渡す。faer へ `Par::Rayon(threads)` を渡す
/// だけでは上限にならない (`otspot_num::linalg::parallelism` のモジュール doc
/// に実測値) ため、ここが QP/conic 経路のスレッド上限を成立させる唯一の点。
/// `threads = 1` (既定) はプールを作らずインライン + `Par::Seq` で走るので、
/// 従来の挙動と完全に同一。
pub(crate) fn solve_ippmm_inner(
    problem: &QpProblem,
    options: &SolverOptions,
    eps_orig: f64,
) -> SolverResult {
    with_solver_pool(options.threads, |par| {
        solve_ippmm_inner_confined(problem, options, eps_orig, par)
    })
}

/// Certificate-acceptance contract: `check_infeasible_or_unbounded`'s
/// consecutive-fire detector is a Newton-direction Farkas-like heuristic, not
/// a hard proof. If an external stop (cancel/deadline) is observed at the
/// instant its conclusion (Infeasible/Unbounded, or the best-so-far Stalled
/// downgrade that doubts it) would be accepted, `Timeout` wins -- discarding
/// an unverified heuristic conclusion in favor of the caller's stop request
/// is the conservative choice.
fn solve_ippmm_inner_confined(
    problem: &QpProblem,
    options: &SolverOptions,
    eps_orig: f64,
    par: Par,
) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::new(
        options.deadline,
        options.timeout_secs,
        options.cancel_flag.clone(),
    );

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

    let scales = crate::qp::ipm_core::common::DetectionScales::compute(problem, &a_ext, &b_ext);
    // 受理部が通過した証書の種類 (None = 未受理)。終端の stats 組立で
    // ray / Farkas 各フラグへ分離して記録する (SolveStats の doc 参照)。
    let mut certified: Option<CertKind> = None;

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
        // floor は REG_LIMIT_MIN (adaptive reg_limit が最終的に到達する下限) —
        // 固定 delta_min=1e-8 は tight eps でこの初期値自体が過大な床になる
        // (bug-frontier 実測: LISWET7 系, `matrix_reg_floor_for_lp` の doc 参照)。
        Some(mu) => {
            let v = mu.max(REG_LIMIT_MIN);
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

    let is_lp = problem.q.values().iter().all(|&v| v == 0.0);
    let initial_reg_limit = if is_lp {
        REG_LIMIT_INIT_LP
    } else {
        REG_LIMIT_INIT_QP
    };
    // rank-deficient Q + c≈0 で rho が floor に張り付き proximal 項が df を支配する病理を回避する適応 floor。
    let c_max = problem.c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let allow_adaptive_reg = c_max < ADAPTIVE_REG_C_MAX_THRESH;
    let mut reg_limit = initial_reg_limit;
    // KKT 行列正則化の絶対下限 (reg_limit の適応引下げとは独立)。
    let matrix_reg_floor = matrix_reg_floor_for_lp(is_lp);

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
    let use_schur = options
        .schur_hint
        .unwrap_or_else(|| auto_schur_enabled(problem, &a_ext, m_ext, options, &timeout_ctx, par));

    // ループ内 break は Optimal / Timeout / Stalled / NumericalError / Infeasible /
    // Unbounded を設定する。break せず max_iter を使い切った場合のみ MaxIterations。
    let mut status: Option<SolveStatus> = None;
    let mut iterations_consumed = 0usize;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    // NaN guard で崩壊解を返さないための best-so-far スナップショット。
    let mut best_score = f64::INFINITY;
    let mut best_x = x.clone();
    let mut best_y = y.clone();
    let mut best_s = s.clone();
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

        let gap = duality_gap(&qx, &x, &problem.c, &b_ext, &y);

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
                best_residuals = (nr_p, nr_d, mu);
                best_rel_gap = gap.rel;
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

        let measures = ConvergenceMeasures {
            nr_p_rel,
            nr_d_rel,
            mu,
            gap,
        };
        if measures.converged(eps) {
            // Optimal 打刻は検証済み ray をレースで追い越せない: 直近の
            // Newton 方向へ単発の構造チェックを走らせ、検証 PASS なら
            // 構造判定が勝つ (sentinel: truly_unbounded_delta_5e_8_*、
            // structural_verdict_at_convergence の doc 参照)。
            if let Some((structural, cert)) = structural_verdict_at_convergence(
                iter,
                (&dx, &ds, &dy),
                (problem, &a_ext),
                (m_orig, m_ext),
                (&b_ext, &is_eq_ext, &scales),
            ) {
                certified = cert;
                status = Some(structural);
                break;
            }
            status = Some(SolveStatus::Optimal);
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

        // KKT 行列側の正則化。床は LP 経路のみ (`matrix_reg_floor_for_lp` の doc)。
        let rho_matrix = pmm.rho.max(matrix_reg_floor);
        let delta_matrix = pmm.delta.max(matrix_reg_floor);

        // Σ = diag(s_i / y_i) (等式行は0)
        let sigma_vec = compute_sigma_vec(&s, &y, &is_eq_ext, SIGMA_MAX_FALLBACK);

        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
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
        let (mut fac, aug_mat, d_inv_opt) = match factorize_outcome {
            FactorizeOutcome::Ok {
                factor,
                aug_mat,
                d_inv,
                retry_count,
                used_iterative,
                factorize_ns,
            } => {
                total_factorize_ns += factorize_ns;
                total_reg_retries += retry_count;
                any_iterative |= used_iterative;
                (factor, aug_mat, d_inv)
            }
            FactorizeOutcome::Timeout => {
                status = Some(SolveStatus::Timeout);
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
            // Optimal は収束判定 (nr_p_rel/nr_d_rel/mu/rel_gap) を通った iterate のみが
            // 名乗る。blow-up からの best-so-far 復帰は品質を主張せず Stalled で返し、
            // 実際の品質判定は finalize (prove_optimal) に委ねる。
            if best_score.is_finite() {
                restore_best_so_far(
                    &mut x,
                    &mut y,
                    &mut s,
                    &mut final_residuals,
                    &best_x,
                    &best_y,
                    &best_s,
                    best_residuals,
                );
                status = Some(SolveStatus::Stalled);
            } else {
                status = Some(SolveStatus::NumericalError);
            }
            break;
        }

        // 受理ゲート (count 管理・distrust・独立 ray/Farkas 検証) は helper doc 参照。
        match check_infeas_verdict(
            &mut consecutive_infeas_triggers,
            &mut status,
            &mut certified,
            (&timeout_ctx, best_score, eps_orig),
            (&dx, &ds, &dy),
            (problem, &a_ext),
            (m_orig, m_ext, iter),
            (&b_ext, &is_eq_ext, &scales),
        ) {
            // Break の *status (検証済み判定 / stop 時の Timeout) はゲート側で確定済み。
            InfeasGate::Break => break,
            InfeasGate::Stall => {
                restore_best_so_far(
                    &mut x,
                    &mut y,
                    &mut s,
                    &mut final_residuals,
                    &best_x,
                    &best_y,
                    &best_s,
                    best_residuals,
                );
                status = Some(SolveStatus::Stalled);
                break;
            }
            InfeasGate::Continue => {}
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
        iterations_consumed = iter.saturating_add(1);

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
            final_residuals = Some(best_residuals);
            status = Some(SolveStatus::Stalled);
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
            final_residuals = Some(best_residuals);
            status = Some(SolveStatus::Stalled);
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

        // Adaptive reg_limit: prox が df を支配 (c≈0)、または pf-stagnation
        // (`pf_stuck_should_lower_reg_limit`, state.rs 参照) の場合、floor を
        // 下げて IPM が boundary を探索できるようにする。
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
                && pf_stuck_should_lower_reg_limit(nr_p, pf_history[0], eps_orig)
            {
                should_lower = true;
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

    finalize_ippmm_result(
        problem,
        &b_ext,
        accepted_ray_dir(status.as_ref(), &dx),
        q_is_indefinite,
        options.ipm.max_iter,
        iterations_consumed,
        status,
        final_residuals,
        best_score,
        best_residuals,
        best_rel_gap,
        x,
        y,
        s,
        &best_x,
        &best_y,
        &best_s,
        total_factorize_ns,
        total_solve_ns,
        total_reg_retries,
        any_iterative,
        certified,
    )
}

/// ループ終端後の後処理: 反復予算枯渇の status 確定 (MaxIterations)、Timeout/
/// MaxIterations 到達時の best-so-far への上書き、目的値・双対解の復元、不定 Q の
/// 検出器 fire の受理ゲート (巨大ループからの切り出しであり、受理部契約の
/// 明文化でもある)。
///
/// 動作:
/// 1. fire 回数管理 — fire 無しでカウントリセット、N 連続 fire まで判定保留し
///    (PMM floor の false-positive に adaptive reg の猶予を与える)、N 到達で
///    カウントを戻してゲート評価へ。
/// 2. 受理直前の時間切れ check — `should_stop_before_infeas_commit` 側は best
///    破棄の Timeout 返却を許可しない契約のため、best を Stalled で返す。
/// 3. ray / Farkas の独立検証を先に実行し、成立時に *status と
///    *certified (検証 PASS した証書種類) を設定してループ終了を返す。検証 PASS
///    は eps-free の証明であり、eps 依存の distrust が前借りしてはならない (真に
///    非有界な問題で distrust Stall → finalize の Optimal 打刻、調査台帳
///    /tmp/opencode/eps_diverge_repro/REPORT.md)。
/// 4. distrust が扱うのは未証明の主張のみ: best が quality 圏内なら
///    best-so-far を Stalled で返し、そうでなければ撤回して反復継続 —
///    偽陽性で打ち切るとその後も収束へ進めた solve を早期に奪う (QFORPLAN
///    no-presolve fallback: iter 42 打ち切り comp 6.7e-7、継続なら iter 52 で
///    真最適点 comp 3.9e-8)。後に本物の ray 方向が出れば再受理され得る。
///    修正前は distrust の後のみ検証を走らせていた (検証省略 trace 実測:
///    /tmp/opencode/inv_eps_anom/REPORT.md)。
///
/// ゲートの終了指示。Break = *status 確定済みで break、Stall = best を復元して
/// Stalled で break、Continue = 反復継続 (fire 無し / 保留 / 未証明撤回)。
#[derive(Debug)]
enum InfeasGate {
    Break,
    Stall,
    Continue,
}

/// 検証 PASS 済み受理から ray 証書の方向だけを抽出する (finalize 呼び出し
/// 引数用)。Unbounded 判定のときのみ直近 Newton 方向 dx が受理部検証の
/// 検査対象であり、それ以外の終端 (Timeout 撤回 / Stalled / Infeasible /
/// 非構造判定) は方向を保持しない。受理部 2 箇所はいずれも検証 PASS 後
/// `update_variables` の step 適用前に break するため、break 時点の dx は
/// 検証が検査した方向そのもの (検証の再実行はしない)。
fn accepted_ray_dir(claimed: Option<&SolveStatus>, dx: &[f64]) -> Option<Vec<f64>> {
    // Infeasible の Farkas 証書 dy は拡張系の内部行レイアウト (元行 + 境界行の
    // 並び・符号慣例) に依存するためレイアウト非依存の独立再計算形状が無く
    // 対象外 (`crate::problem::SolverResult::ray_certificate` doc 参照)。
    claimed
        .filter(|s| **s == SolveStatus::Unbounded)
        .map(|_| dx.to_vec())
}

/// 内側受理部が通過した証書の種類。検出器の主張に対応する:
/// `Unbounded` は降下方向 dx の recession ray 証明 (`verify_unbounded_ray`)、
/// `Infeasible` は双対方向 dy の Farkas 代替証明 (`verify_infeasible_farkas`)。
/// 両者は主張する数学的事実が別物なので、`SolveStats` の側でも
/// `unbounded_ray_certified` / `infeasible_farkas_certified` に分離して記録する
/// (共有名は「Unbounded + infeasibility certified」の誤読を生む。PR #71 review)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CertKind {
    Ray,
    Farkas,
}

/// 検証 PASS した検出器主張の status → 証書種類の対応。`verify_detector_claim`
/// は `Unbounded` / `Infeasible` 以外常に false を返すので、この関数は受理
/// 経路では必ず Some になる。`_ => None` arm は検証器契約が将来ずれたときに
/// 存在しない証書を主張しないための保守的ガード。
fn cert_kind_of(claimed: &SolveStatus) -> Option<CertKind> {
    match claimed {
        SolveStatus::Unbounded => Some(CertKind::Ray),
        SolveStatus::Infeasible => Some(CertKind::Farkas),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn check_infeas_verdict(
    consecutive: &mut usize,
    status: &mut Option<SolveStatus>,
    certified: &mut Option<CertKind>,
    (timeout_ctx, best_score, eps_orig): (&TimeoutCtx, f64, f64),
    (dx, ds, dy): (&[f64], &[f64], &[f64]),
    (problem, a_ext): (&QpProblem, &otspot_num::sparse::CscMatrix),
    (m_orig, m_ext, iter): (usize, usize, usize),
    (b_ext, is_eq_ext, scales): (
        &[f64],
        &[bool],
        &crate::qp::ipm_core::common::DetectionScales,
    ),
) -> InfeasGate {
    let Some(infeas_status) =
        check_infeasible_or_unbounded(dx, dy, problem, a_ext, m_orig, m_ext, iter, scales)
    else {
        *consecutive = 0;
        return InfeasGate::Continue;
    };
    *consecutive += 1;
    if *consecutive < MIN_CONSECUTIVE_INFEAS {
        return InfeasGate::Continue;
    }
    *consecutive = 0;
    // stop check は base 同一に best 破棄の Timeout を返す (cancel race sentinel
    // `cancel_race_before_infeasible_certificate_prefers_timeout` が pin する契約)。
    if should_stop_before_infeas_commit(timeout_ctx) {
        *status = Some(SolveStatus::Timeout);
        return InfeasGate::Break;
    }
    // 検証器が審判: PASS は eps-free の証明であり、ヒューリスティックは
    // 証明を棄却できない。distrust は未証明の主張にのみ作用する
    // (revert sentinel: truly_unbounded_delta_1e_9_never_optimal_cross_eps)。
    if verify_detector_claim(
        &infeas_status,
        (dx, ds, dy),
        (problem, a_ext),
        (b_ext, is_eq_ext, scales),
    ) {
        *certified = cert_kind_of(&infeas_status);
        *status = Some(infeas_status);
        return InfeasGate::Break;
    }
    if best_score.is_finite() && best_score < INFEAS_DETECTOR_DISTRUST_SCORE * eps_orig {
        return InfeasGate::Stall;
    }
    InfeasGate::Continue
}

/// 検出器の主張 (Unbounded→ray / Infeasible→Farkas) への独立検証器の判定。
/// 検証器は eps-free・componentwise 後方誤差境界 (common.rs) で、PASS は
/// 本物の ray / Farkas 証書の証明。連続 fire 受理ゲート
/// (`check_infeas_verdict`) と Optimal 打刻直前の構造チェック
/// (`final_structural_check`) で共有する。
fn verify_detector_claim(
    claimed: &SolveStatus,
    (dx, ds, dy): (&[f64], &[f64], &[f64]),
    (problem, a_ext): (&QpProblem, &otspot_num::sparse::CscMatrix),
    (b_ext, is_eq_ext, scales): (
        &[f64],
        &[bool],
        &crate::qp::ipm_core::common::DetectionScales,
    ),
) -> bool {
    match claimed {
        SolveStatus::Unbounded => crate::qp::ipm_core::common::verify_unbounded_ray(
            dx, ds, problem, a_ext, b_ext, is_eq_ext, scales,
        ),
        SolveStatus::Infeasible => crate::qp::ipm_core::common::verify_infeasible_farkas(
            dy, a_ext, b_ext, is_eq_ext, scales,
        ),
        _ => false,
    }
}

/// 収束時 (Optimal 打刻直前) の構造判定受理。
///
/// `final_structural_check` の単発走査に、iter 0 で Newton 方向が未計算のため
/// 走らせないガードと、検証 PASS した構造判定の証書種類 (`cert_kind_of`)
/// 紐付けを畳み込み、`(status, 証書種類)` を返す。呼び出し側は status と
/// `certified` に書き込んで break するだけでよく、受理部 2 系統 (ここ と
/// `check_infeas_verdict`) とも証書の配線は証書種類側の名前関数に一元化する
/// (`CertKind` の doc 参照)。
fn structural_verdict_at_convergence(
    iter: usize,
    (dx, ds, dy): (&[f64], &[f64], &[f64]),
    (problem, a_ext): (&QpProblem, &otspot_num::sparse::CscMatrix),
    (m_orig, m_ext): (usize, usize),
    (b_ext, is_eq_ext, scales): (
        &[f64],
        &[bool],
        &crate::qp::ipm_core::common::DetectionScales,
    ),
) -> Option<(SolveStatus, Option<CertKind>)> {
    let structural = if iter > 0 {
        final_structural_check(
            (dx, ds, dy),
            (problem, a_ext),
            (m_orig, m_ext, iter),
            (b_ext, is_eq_ext, scales),
        )
    } else {
        None
    };
    structural.map(|claimed| {
        let kind = cert_kind_of(&claimed);
        (claimed, kind)
    })
}

/// Optimal 打刻直前の単発構造チェック。
///
/// 収束判定はループ本体で検出器ブロックより先に走るため、loose eps では
/// 真に非有界な問題が検出器の連続 fire 完了前に Optimal で break できる
/// (ray 上の変数に残るコスト係数がそのまま KKT 残差になるため、有限
/// iterate は eps ≥ delta で常に収束を通る。/tmp/opencode/
/// eps_diverge_repro/REPORT.md)。よって Optimal を返す直前に、直近計算済み
/// Newton 方向へ検出器 + 独立検証器を 1 回走らせ、検証 PASS なら構造判定が
/// 勝つ。この 1 回のみ MIN_CONSECUTIVE_INFEAS を迂回する — カウンタは毎反復
/// fire のコスト抑制が目的で、正しさの網は検証器 (偽 ray は PASS 不能) そのもの。
/// コストは Optimal break 1 回につき O(nnz) × 1。
fn final_structural_check(
    (dx, ds, dy): (&[f64], &[f64], &[f64]),
    (problem, a_ext): (&QpProblem, &otspot_num::sparse::CscMatrix),
    (m_orig, m_ext, iter): (usize, usize, usize),
    (b_ext, is_eq_ext, scales): (
        &[f64],
        &[bool],
        &crate::qp::ipm_core::common::DetectionScales,
    ),
) -> Option<SolveStatus> {
    let claimed =
        check_infeasible_or_unbounded(dx, dy, problem, a_ext, m_orig, m_ext, iter, scales)?;
    verify_detector_claim(
        &claimed,
        (dx, ds, dy),
        (problem, a_ext),
        (b_ext, is_eq_ext, scales),
    )
    .then_some(claimed)
}

/// best-so-far iterate (x, y, s) と残差計測を復元する (blow-up 復帰と検出器
/// distrust 復帰の共通化)。
#[allow(clippy::too_many_arguments)]
fn restore_best_so_far(
    x: &mut [f64],
    y: &mut [f64],
    s: &mut [f64],
    final_residuals: &mut Option<(f64, f64, f64)>,
    best_x: &[f64],
    best_y: &[f64],
    best_s: &[f64],
    best_residuals: (f64, f64, f64),
) {
    x.copy_from_slice(best_x);
    y.copy_from_slice(best_y);
    s.copy_from_slice(best_s);
    *final_residuals = Some(best_residuals);
}

/// Optimal→LocallyOptimal 降格、`SolverResult` の組み立てを行う。
#[allow(clippy::too_many_arguments)]
fn finalize_ippmm_result(
    problem: &QpProblem,
    b_ext: &[f64],
    certified_ray: Option<Vec<f64>>,
    q_is_indefinite: bool,
    max_iter: usize,
    iterations_consumed: usize,
    status: Option<SolveStatus>,
    mut final_residuals: Option<(f64, f64, f64)>,
    best_score: f64,
    best_residuals: (f64, f64, f64),
    best_rel_gap: f64,
    mut x: Vec<f64>,
    mut y: Vec<f64>,
    mut s: Vec<f64>,
    best_x: &[f64],
    best_y: &[f64],
    best_s: &[f64],
    total_factorize_ns: u128,
    total_solve_ns: u128,
    total_reg_retries: u32,
    any_iterative: bool,
    certified: Option<CertKind>,
) -> SolverResult {
    // `build_extended_constraints` の 4 番目の返り値は `problem.num_constraints`
    // そのもの (kkt.rs)。導出に置き換えて呼び出し引数を ray ペイロードへ転用した。
    let m_orig = problem.num_constraints;
    let iterations_consumed = if status.is_none() {
        max_iter
    } else {
        iterations_consumed
    };

    // break なしのループ終端 = 反復予算枯渇。Timeout に丸めず MaxIterations で報告する。
    let status = status.unwrap_or(SolveStatus::MaxIterations);

    // 素の Timeout 経路は発散 x をそのまま返してしまうので best-so-far で上書き。
    if matches!(status, SolveStatus::Timeout | SolveStatus::MaxIterations) && best_score.is_finite()
    {
        let norm_b_bs = norm_inf(b_ext).max(1.0);
        let norm_c_bs = norm_inf(&problem.c).max(1.0);
        let current_score = match final_residuals {
            Some((nr_p, nr_d, mu)) if nr_p.is_finite() && nr_d.is_finite() && mu.is_finite() => {
                nr_p / (1.0 + norm_b_bs) + nr_d / (1.0 + norm_c_bs) + mu.abs()
            }
            _ => f64::INFINITY,
        };
        if best_score < current_score {
            x.copy_from_slice(best_x);
            y.copy_from_slice(best_y);
            s.copy_from_slice(best_s);
            final_residuals = Some(best_residuals);
        }
    }

    let mut qx = vec![0.0f64; problem.num_vars];
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

    // ray 証書ペイロードの組立: 検証済み Unbounded のときのみ Some。
    // certified_ray (accepted_ray_dir 経由) は受理部検証の検査方向そのもの、
    // 基点はここへ渡された x (受理 break 時に step 未適用の検証検査 iterate)。
    // certified でない・Unbounded 以外の方向は保持せず廃棄
    // (Some ⇔ 検証済み Unbounded の契約)。
    let ray_certificate =
        if final_status == SolveStatus::Unbounded && certified == Some(CertKind::Ray) {
            certified_ray.map(|direction| crate::problem::RayCertificate {
                direction,
                base_point: x.clone(),
            })
        } else {
            None
        };

    SolverResult {
        status: final_status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,
        ray_certificate,

        iterations: iterations_consumed,
        final_residuals,
        // best-so-far の rel gap (診断用)。
        duality_gap_rel: if best_rel_gap.is_finite() {
            Some(best_rel_gap)
        } else {
            None
        },
        timing_breakdown: Some(ipm_timing),
        stats: crate::problem::SolveStats {
            // 受理した証書の種類ごとに別フラグへ立てる。ray (非有界) と
            // Farkas (実行不可能) は主張する数学的事実が別物なので共有名は
            // 使わない (SolveStats の doc 参照)。下流 (`ipm_solver::core` →
            // `finalize_outcome`) の検証義務ゲートがこれらのフラグを参照する。
            unbounded_ray_certified: matches!(certified, Some(CertKind::Ray)),
            infeasible_farkas_certified: matches!(certified, Some(CertKind::Farkas)),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod gate_ordering_tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::qp::ipm_core::common::DetectionScales;
    use otspot_num::sparse::CscMatrix;

    fn inf() -> f64 {
        f64::INFINITY
    }

    /// U1 (手計算 true-unbounded: min -delta·x2 s.t. x1=1, ray=e2)。
    /// `bound_x2` は x2 ≥ 0 を付与し e2 を退行方向でなくする検出器偽陽性 fixture。
    fn u1(delta: f64, bound_x2: bool) -> QpProblem {
        let x2b = if bound_x2 {
            (0.0, inf())
        } else {
            (-inf(), inf())
        };
        QpProblem::new(
            CscMatrix::new(2, 2),
            vec![0.0, -delta],
            CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap(),
            vec![1.0],
            vec![(-inf(), inf()), x2b],
            vec![ConstraintType::Eq],
        )
        .unwrap()
    }

    struct Fixture {
        problem: QpProblem,
        a_ext: CscMatrix,
        b_ext: Vec<f64>,
        is_eq_ext: Vec<bool>,
        scales: DetectionScales,
        m_orig: usize,
        m_ext: usize,
    }

    fn fixture(delta: f64, bound_x2: bool) -> Fixture {
        let problem = u1(delta, bound_x2);
        let (a_ext, b_ext, m_ext, m_orig, _n_lb, is_eq_ext) = build_extended_constraints(&problem);
        let scales = DetectionScales::compute(&problem, &a_ext, &b_ext);
        Fixture {
            problem,
            a_ext,
            b_ext,
            is_eq_ext,
            scales,
            m_orig,
            m_ext,
        }
    }

    /// 真 ray 方向が検証 PASS したとき、distrust 圏内の best_score (1e-12
    /// < 10·1e-6) でも Stall でなく証明受理 (Break + certified) すること。
    /// Part 1 (検証器を distrust より前へ) の revert で Stall が返り FAIL。
    #[test]
    fn verifier_pass_preempts_distrust_stall() {
        let f = fixture(1e-9, false);
        let dx = vec![0.0, 1.0];
        let ds = vec![0.0; f.m_ext];
        let dy = vec![0.0; f.m_ext];
        let timeout_ctx = TimeoutCtx::new(None, Some(60.0), None);
        let mut consecutive = MIN_CONSECUTIVE_INFEAS - 1;
        let mut status = None;
        let mut certified: Option<CertKind> = None;
        let gate = check_infeas_verdict(
            &mut consecutive,
            &mut status,
            &mut certified,
            (&timeout_ctx, 1e-12, 1e-6),
            (&dx, &ds, &dy),
            (&f.problem, &f.a_ext),
            (f.m_orig, f.m_ext, 6),
            (&f.b_ext, &f.is_eq_ext, &f.scales),
        );
        assert!(
            matches!(gate, InfeasGate::Break),
            "verifier-passed ray must be accepted even when distrust range, got {gate:?}"
        );
        assert_eq!(
            certified,
            Some(CertKind::Ray),
            "accepted Unbounded must carry the ray certificate (not Farkas, not none)"
        );
        assert_eq!(status, Some(SolveStatus::Unbounded));
    }

    /// 検証 FAIL の未証明主張 (境界行により e2 は退行方向でない) は、従来
    /// 通り distrust が Stalled に退ける (distrust の役割は維持される)。
    #[test]
    fn unproven_claim_still_distrust_stalled() {
        let f = fixture(1e-9, true);
        assert_eq!(f.m_ext, 2, "fixture must carry the bound row");
        let dx = vec![0.0, 1.0];
        let ds = vec![0.0; f.m_ext];
        let dy = vec![0.0; f.m_ext];
        let timeout_ctx = TimeoutCtx::new(None, Some(60.0), None);
        let mut consecutive = MIN_CONSECUTIVE_INFEAS - 1;
        let mut status = None;
        let mut certified: Option<CertKind> = None;
        let gate = check_infeas_verdict(
            &mut consecutive,
            &mut status,
            &mut certified,
            (&timeout_ctx, 1e-12, 1e-6),
            (&dx, &ds, &dy),
            (&f.problem, &f.a_ext),
            (f.m_orig, f.m_ext, 6),
            (&f.b_ext, &f.is_eq_ext, &f.scales),
        );
        assert!(
            matches!(gate, InfeasGate::Stall),
            "unproven detector claim within distrust range must stall, got {gate:?}"
        );
        assert_eq!(certified, None);
        assert!(status.is_none());
    }

    /// accepted_ray_dir: Unbounded 受理のときのみ検証検査方向を渡す。
    /// Timeout 撤回 (status=Timeout)、Stalled、Infeasible (Farkas 対象外)、
    /// 非構造終端 (status=None) はすべて None — ペイロードが未受理方向を
    /// 運ぶ経路を unit で封じる (revert: 判定フィルタを外すと fail)。
    #[test]
    fn accepted_ray_dir_only_carries_verified_unbounded_direction() {
        let dx = vec![0.0, 1.0];
        assert_eq!(
            accepted_ray_dir(Some(&SolveStatus::Unbounded), &dx),
            Some(dx.clone()),
            "verified Unbounded acceptance must carry the checked direction"
        );
        for status in [
            SolveStatus::Timeout,
            SolveStatus::Stalled,
            SolveStatus::Infeasible,
            SolveStatus::Optimal,
        ] {
            assert_eq!(
                accepted_ray_dir(Some(&status), &dx),
                None,
                "{status:?} termination must not carry a ray payload"
            );
        }
        assert_eq!(accepted_ray_dir(None, &dx), None);
    }

    /// final_structural_check: 真 ray は Some(Unbounded)、検出器 MIN_ITER 未満と
    /// ゼロ方向は None (Optimal 打刻を允许)。
    #[test]
    fn final_structural_check_accepts_ray_guards_cheap_cases() {
        let f = fixture(1e-9, false);
        let dx = vec![0.0, 1.0];
        let ds = vec![0.0; f.m_ext];
        let dy = vec![0.0; f.m_ext];
        assert_eq!(
            final_structural_check(
                (&dx, &ds, &dy),
                (&f.problem, &f.a_ext),
                (f.m_orig, f.m_ext, 5),
                (&f.b_ext, &f.is_eq_ext, &f.scales),
            ),
            Some(SolveStatus::Unbounded),
            "true ray direction must win the at-convergence structural check"
        );
        assert_eq!(
            final_structural_check(
                (&dx, &ds, &dy),
                (&f.problem, &f.a_ext),
                (f.m_orig, f.m_ext, 4),
                (&f.b_ext, &f.is_eq_ext, &f.scales),
            ),
            None,
            "below detector MIN_ITER the check must yield to Optimal"
        );
        let dz = vec![0.0; 2];
        assert_eq!(
            final_structural_check(
                (&dz, &ds, &dy),
                (&f.problem, &f.a_ext),
                (f.m_orig, f.m_ext, 6),
                (&f.b_ext, &f.is_eq_ext, &f.scales),
            ),
            None,
            "zero direction must yield to Optimal"
        );
    }
}
