//! IP-PMM (Pougkakiotis & Gondzio 2021, DOI 10.1007/s10589-020-00240-9)
//!
//! Augmented KKT (quasi-definite, upper-tri CSC):
//!   K = [(Q + ρI),  Aᵀ ]
//!       [A,        -D  ]   D = Σ + δI, Σ = diag(s/y)
//!
//! PMM update rule (Algorithm PEU §5.1.4):
//!   r = |μ_k − μ_{k+1}| / μ_k (実 μ)
//!   primal_improved = 0.95·prev_nr_p > nr_p  →  y_ref=y, δ *= (1−r),  else δ *= (1−r/3)
//!   dual_improved   = 0.95·prev_nr_d > nr_d  →  x_ref=x, ρ *= (1−r),  else ρ *= (1−r/3)

use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{
    factorize_kkt_pre_permuted_cached, factorize_kkt_with_cached_perm, inexact_eta_for_eps,
    max_l_nnz_from_budget, KktError, KktFactor,
};

use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::kkt::{spmv, spmtv, spmv_q, norm_inf, build_extended_constraints, build_augmented_system, build_schur_system};
use super::common::{check_infeasible_or_unbounded, solve_unconstrained, timeout_result, numerical_error_result};
use super::solver_loop::{
    compute_sigma_vec, predictor_step, corrector_step, gondzio_correctors,
    predictor_step_schur, corrector_step_schur, gondzio_correctors_schur,
    update_variables,
};
use super::kkt::collapse_extended_dual;

/// 論文 §5.1 推奨初期値。
const RHO_INIT: f64 = 8.0;
const DELTA_INIT: f64 = 8.0;

/// warm start safe guard.
/// μ floor: x·y=0 / s=0 を渡された場合に central path から外れないため。
const WARM_MU_MIN: f64 = 1e-8;
/// 両端有限 box では range × WARM_BOUND_REL_MARGIN を interior 余白にとる
/// (cold init の 1% 余白より tighter、warm 値を最大限尊重する)。
const WARM_BOUND_REL_MARGIN: f64 = 1e-6;
/// 半側有限 bound では cold init と同等の絶対 1.0 余白。
const WARM_BOUND_ABS_MARGIN: f64 = 1.0;
/// 不等式行 s, y の boundary 上で σ=s/y が発散するため両側を floor。
const WARM_SY_MIN: f64 = 1e-8;

/// 5% 以上の残差減少を改善とみなす (Gondzio2021 MATLAB)。
const PMM_IMPROVE_THRESHOLD: f64 = 0.95;
const PMM_SLOW_RATE: f64 = 2.0 / 3.0;

/// μ が実質 0 と判定する境界 (機械精度直上)。
const MU_ZERO_THRESHOLD: f64 = 1e-15;

const LDL_REG_RETRY_MAX: usize = 10;
const LDL_REG_GROWTH: f64 = 10.0;
const LDL_REG_CEILING: f64 = 1.0;
const LDL_FALLBACK_DELTA_MIN: f64 = 1e-2;

/// tight eps で正常な小 alpha を stall 扱いしないため eps スケールで閾値を緩める。
fn alpha_stall_eps_for(eps: f64) -> f64 {
    (eps * 1e-2).max(1e-14)
}
const ALPHA_STALL_N: usize = 5;
const ALPHA_DEADLOCK_N: usize = 20;

/// alpha > 0 でも residual が改善しない病理 (n=250k 級) 用の停滞窓。
/// 50 iter は典型収束速度 0.5^50 ≈ 9e-16 を踏まえた観測窓、REL_DEC=1e-3 は数値飽和判定。
const RESIDUAL_STALL_WINDOW: usize = 50;
const RESIDUAL_STALL_REL_DEC: f64 = 1e-3;

struct PmmState {
    x_ref: Vec<f64>,
    y_ref: Vec<f64>,
    rho: f64,
    delta: f64,
    prev_nr_p: f64,
    prev_nr_d: f64,
}

/// warm start から (x, y, s) を初期化し、有効なら μ を返す (none で cold start)。
///
/// 規約:
/// - `ws.x` 長さ n、`ws.y` 長さ m_orig (user 符号、Ge は内部で反転)、`ws.mu` スカラー
/// - bound row dual / slack は 1.0 で cold 初期化 (B&B でも bound multiplier は不安定)
fn apply_qp_warm_start(
    ws: &crate::options::QpWarmStart,
    problem: &crate::qp::problem::QpProblem,
    a_ext: &crate::sparse::CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    m_orig: usize,
    m_ext: usize,
    x: &mut [f64],
    y: &mut [f64],
    s: &mut [f64],
) -> Option<f64> {
    use crate::problem::ConstraintType;
    let n = problem.num_vars;
    if ws.x.len() != n || ws.y.len() != m_orig {
        return None;
    }
    let mu = ws.mu.max(WARM_MU_MIN);

    for j in 0..n {
        let xj = ws.x[j];
        let (lb, ub) = problem.bounds[j];
        x[j] = match (lb.is_finite(), ub.is_finite()) {
            (true, true) => {
                let range = ub - lb;
                let margin = (range * WARM_BOUND_REL_MARGIN).min(WARM_BOUND_ABS_MARGIN);
                if range > 2.0 * margin {
                    xj.clamp(lb + margin, ub - margin)
                } else {
                    0.5 * (lb + ub)
                }
            }
            (true, false) => xj.max(lb + WARM_BOUND_ABS_MARGIN),
            (false, true) => xj.min(ub - WARM_BOUND_ABS_MARGIN),
            (false, false) => xj,
        };
    }

    // 元制約 dual を内部符号 (Ge は -1 倍) に展開。
    for i in 0..m_orig {
        let yi = match problem.constraint_types[i] {
            ConstraintType::Ge => -ws.y[i],
            _ => ws.y[i],
        };
        y[i] = if is_eq_ext[i] { yi } else { yi.max(WARM_SY_MIN) };
    }

    // 自然な slack s = b_ext − A_ext·x (ineq は WARM_SY_MIN で boundary 退避)。
    let mut ax = vec![0.0_f64; m_ext];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax[a_ext.row_ind[k]] += a_ext.values[k] * x[col];
        }
    }
    for i in 0..m_ext {
        if is_eq_ext[i] {
            s[i] = 0.0;
        } else {
            s[i] = (b_ext[i] - ax[i]).max(WARM_SY_MIN);
        }
    }
    // bound 行 dual は中心パス s·y=μ から逆算 (x interior → y≈0、x active → y≈μ/ε 大)。
    // ユーザーが bound_duals を渡さない設計のため central path 関係で推定する。
    for i in m_orig..m_ext {
        y[i] = (mu / s[i]).max(WARM_SY_MIN);
    }

    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
        let x_inf = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let y_inf = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let s_inf = s.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        eprintln!(
            "IPPMM_INIT_WARM: μ={:.3e} |x|_inf={:.3e} |y|_inf={:.3e} |s|_inf={:.3e}",
            mu, x_inf, y_inf, s_inf
        );
    }
    Some(mu)
}

/// IP-PMM 内部ソルバー (Ruiz scaling 後の problem を受け取る)。
pub(crate) fn solve_ippmm_inner(
    problem: &QpProblem,
    options: &SolverOptions,
    eps_orig: f64,
) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);

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

    // 巨大 |ub|=1e11 で midpoint 起点だと pf が抜けないため、0 が bounds 内なら 0 を優先。
    let x0: Vec<f64> = problem
        .bounds
        .iter()
        .map(|&(lb, ub)| {
            let lb_fin = lb.is_finite();
            let ub_fin = ub.is_finite();
            // 0 が bounds 内なら 0 を優先
            let zero_in_bounds = (!lb_fin || lb <= 0.0) && (!ub_fin || ub >= 0.0);
            if zero_in_bounds {
                0.0
            } else if lb_fin && ub_fin {
                (lb + ub) / 2.0
            } else if lb_fin {
                lb + 1.0
            } else if ub_fin {
                ub - 1.0
            } else {
                0.0
            }
        })
        .collect();

    // s0 = b_ext − A_ext·x0。等式行は s=0、不等式行は s≥1 にクランプ。
    let mut ax0 = vec![0.0f64; m_ext];
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax0[a_ext.row_ind[k]] += a_ext.values[k] * x0[col];
        }
    }
    let s0: Vec<f64> = b_ext
        .iter()
        .zip(ax0.iter())
        .enumerate()
        .map(|(i, (&bi, &axi))| {
            if is_eq_ext[i] { 0.0 } else { (bi - axi).max(1.0) }
        })
        .collect();
    let y0: Vec<f64> = (0..m_ext)
        .map(|i| if is_eq_ext[i] { 0.0 } else { 1.0 })
        .collect();

    let mut x = x0.clone();
    let mut s = s0.clone();
    let mut y = y0.clone();

    // warm start が渡されていれば Mehrotra init を skip し、interior 補正のみ適用する。
    // 補正は (i) bound 余白 (ii) μ floor (iii) s,y boundary floor の三段。
    let warm_mu = if let Some(ws) = options.warm_start_qp.as_ref() {
        apply_qp_warm_start(
            ws, problem, &a_ext, &b_ext, &is_eq_ext, m_orig, m_ext,
            &mut x, &mut y, &mut s,
        )
    } else {
        None
    };

    if warm_mu.is_none() {
    // Mehrotra 1992 標準初期点 (Wright §5.1): 全制約射影 + δ_s/δ_y 正補正 + Σ 均一化補正。
    // 等式のみの射影だと |b|≈1e11 級で s0 が膨張し K matrix が暴走する。
    {
        let r_p: Vec<f64> = b_ext.iter().zip(ax0.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();
        let r_p_inf = r_p.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if r_p_inf > 1e-6 && !timeout_ctx.should_stop() {
            let q_zero = CscMatrix::new(n, n);
            let sigma_zero = vec![0.0_f64; m_ext];
            let k_init = build_augmented_system(&q_zero, &a_ext, &sigma_zero, 1.0, 1.0);
            let perm_init = amd_with_deadline(
                k_init.nrows, &k_init.col_ptr, &k_init.row_ind, timeout_ctx.deadline,
            );
            if let Ok(fac_init) = factorize_kkt_with_cached_perm(
                &k_init, &perm_init, timeout_ctx.deadline, max_l_nnz_from_budget(), Some(n),
            ) {
                let mut rhs_init = vec![0.0_f64; n + m_ext];
                for i in 0..m_ext { rhs_init[n + i] = r_p[i]; }
                let mut sol_init = vec![0.0_f64; n + m_ext];
                fac_init.solve(&rhs_init, &mut sol_init);
                let dx_inf = sol_init[..n].iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                if dx_inf.is_finite() && dx_inf < 1e15 {
                    for j in 0..n {
                        let x_new = x[j] + sol_init[j];
                        let (lb, ub) = problem.bounds[j];
                        x[j] = match (lb.is_finite(), ub.is_finite()) {
                            (true, true) => {
                                let range = ub - lb;
                                let raw_margin = (range * 0.01).min(1.0);
                                if raw_margin > 0.0 && range > 2.0 * raw_margin {
                                    x_new.clamp(lb + raw_margin, ub - raw_margin)
                                } else {
                                    0.5 * (lb + ub)
                                }
                            }
                            (true, false) => x_new.max(lb + 1.0),
                            (false, true) => x_new.min(ub - 1.0),
                            (false, false) => x_new,
                        };
                    }
                    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "IPPMM_INIT_PROJ: r_p_inf={:.3e} dx_inf={:.3e} |x|_inf={:.3e}",
                            r_p_inf, dx_inf,
                            x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
                        );
                    }
                }
            }
        }

        let mut ax_new = vec![0.0_f64; m_ext];
        for col in 0..n {
            for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
                ax_new[a_ext.row_ind[k]] += a_ext.values[k] * x[col];
            }
        }
        let s_hat: Vec<f64> = b_ext.iter().zip(ax_new.iter()).enumerate()
            .map(|(i, (&bi, &axi))| if is_eq_ext[i] { 0.0 } else { bi - axi })
            .collect();
        let y_hat: Vec<f64> = (0..m_ext)
            .map(|i| if is_eq_ext[i] { 0.0 } else { 1.0 })
            .collect();

        let s_min_ineq = s_hat.iter().zip(is_eq_ext.iter())
            .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
            .fold(f64::INFINITY, f64::min);
        let y_min_ineq = y_hat.iter().zip(is_eq_ext.iter())
            .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
            .fold(f64::INFINITY, f64::min);
        let delta_s = (-1.5 * s_min_ineq).max(0.0) + 1.0;
        let delta_y = (-1.5 * y_min_ineq).max(0.0) + 1.0;

        let s_pos: Vec<f64> = s_hat.iter().enumerate()
            .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_s })
            .collect();
        let y_pos: Vec<f64> = y_hat.iter().enumerate()
            .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_y })
            .collect();

        let sy_sum: f64 = s_pos.iter().zip(y_pos.iter()).map(|(&si, &yi)| si * yi).sum();
        let s_sum_pos: f64 = s_pos.iter().sum();
        let y_sum_pos: f64 = y_pos.iter().sum();
        let delta_s_corr = if y_sum_pos > 1e-300 { sy_sum / (2.0 * y_sum_pos) } else { 0.0 };
        let delta_y_corr = if s_sum_pos > 1e-300 { sy_sum / (2.0 * s_sum_pos) } else { 0.0 };

        for i in 0..m_ext {
            s[i] = if is_eq_ext[i] { 0.0 } else { s_pos[i] + delta_s_corr };
            y[i] = if is_eq_ext[i] { 0.0 } else { y_pos[i] + delta_y_corr };
        }

        if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
            let s_inf = s.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let y_inf = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            eprintln!(
                "IPPMM_INIT_MEHROTRA: δ_s={:.3e} δ_y={:.3e} δ_s_corr={:.3e} δ_y_corr={:.3e} |s|_inf={:.3e} |y|_inf={:.3e} mu_init={:.3e}",
                delta_s, delta_y, delta_s_corr, delta_y_corr, s_inf, y_inf,
                sy_sum / m_ineq.max(1) as f64
            );
        }
    }
    } // end cold start init

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
    let _ = x0; let _ = y0; let _ = s0;

    // Gershgorin 由来の Q + δ_ic·I PSD 化量。凸 QP では 0。indefinite 判定 (返却 status 用) のみに使う。
    let inertia_correction = super::kkt::compute_inertia_correction(&problem.q);
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
    const REG_LIMIT_MIN: f64 = 1e-14;
    const PROX_DOMINATE_RATIO: f64 = 0.5;
    /// 一度の調整で reg_limit を割る倍率
    const REG_LIMIT_STEP: f64 = 1e-3;

    // pf-stagnation trigger (adaptive reg_limit の追加経路、c≠0 問題向け):
    // pf が最近の N 反復で実質改善せず (ratio > THRESHOLD) かつ pf が target から
    // 桁違いに離れている場合、reg_limit を下げて IPM が boundary を探索できる
    // c≠0 でも floor で boundary に到達できず suboptimal で停滞するケース用。
    // PF_STUCK_RATIO=0.95: 5 iter 連続で 5%未満の改善を停滞と判定。
    const PF_HISTORY_LEN: usize = 5;
    const PF_STUCK_RATIO: f64 = 0.95;
    /// pf > FAR·eps を「まだ収束遠し」と判定する係数。
    const PF_FAR_FROM_TARGET_RATIO: f64 = 1e2;
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
    let mut amd_perm_cache: Option<Vec<usize>> = None;
    let aug_cache = super::kkt::build_augmented_cache(&problem.q, &a_ext);
    let mut aug_permuted_cache: Option<super::kkt::PermutedAugmentedKkt> = None;
    let mut symbolic_cholesky_cache:
        Option<std::sync::Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>> = None;

    let inexact_eta = inexact_eta_for_eps(eps_orig);

    // augmented LDL が memory budget 超過なら Schur (n×n SPD) に切替。
    // augmented MINRES は ill-cond saddle で direction error が発散するため。
    let explicit_schur = std::env::var("QP_SCHUR").ok().as_deref() == Some("1");
    let auto_schur_disabled = std::env::var("QP_NO_AUTO_SCHUR").ok().as_deref() == Some("1");
    let auto_schur = if explicit_schur || auto_schur_disabled {
        false
    } else {
        let probe_sigma: Vec<f64> = vec![1.0; m_ext];
        let probe_rho = options.ipm.delta_min;
        let probe_aug = build_augmented_system(&problem.q, &a_ext, &probe_sigma, probe_rho, probe_rho);
        let probe_perm = amd_with_deadline(
            probe_aug.nrows, &probe_aug.col_ptr, &probe_aug.row_ind, timeout_ctx.deadline,
        );
        let probe_result = crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget(
            &probe_aug, &probe_perm, timeout_ctx.deadline, Some(max_l_nnz_from_budget()),
        );
        let exceeds = matches!(probe_result, Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }));
        if exceeds && std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
            eprintln!("IPPMM_AUTO_SCHUR: augmented L_nnz exceeds budget, switching to Schur formulation");
        }
        exceeds
    };

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
        const DUALITY_GAP_TOL: f64 = 1e-3;

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
            // 「有意」= last_score_improvement_value × (1 - RESIDUAL_STALL_REL_DEC) を下回る。
            // この基準で改善が無いまま RESIDUAL_STALL_WINDOW iter 経過したら停滞と判定。
            if score < last_score_improvement_value * (1.0 - RESIDUAL_STALL_REL_DEC) {
                last_score_improvement_iter = iter;
                last_score_improvement_value = score;
            }
        }

        // Exp M trace [release-safe, env-gated]
        if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
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
        // per-iter active-set count: wrong-basin lock-in を観測する診断 (IPPMM_ACTIVE_TRACE=1)。
        if std::env::var("IPPMM_ACTIVE_TRACE").ok().as_deref() == Some("1") {
            let s_inf = s.iter().zip(is_eq_ext.iter())
                .filter_map(|(&v, &eq)| if eq { None } else { Some(v.abs()) })
                .fold(0.0_f64, f64::max).max(1e-300);
            let y_inf = y.iter().zip(is_eq_ext.iter())
                .filter_map(|(&v, &eq)| if eq { None } else { Some(v.abs()) })
                .fold(0.0_f64, f64::max).max(1e-300);
            let s_small_abs = s.iter().zip(is_eq_ext.iter())
                .filter(|(_, &eq)| !eq)
                .filter(|(&v, _)| v < 1e-6).count();
            let s_small_rel = s.iter().zip(is_eq_ext.iter())
                .filter(|(_, &eq)| !eq)
                .filter(|(&v, _)| v < 1e-6 * s_inf).count();
            let y_large_abs = y.iter().zip(is_eq_ext.iter())
                .filter(|(_, &eq)| !eq)
                .filter(|(&v, _)| v > 1e-6).count();
            let y_large_rel = y.iter().zip(is_eq_ext.iter())
                .filter(|(_, &eq)| !eq)
                .filter(|(&v, _)| v > 1e-6 * y_inf).count();
            eprintln!(
                "IPPMM_ACTIVE iter={:4} m_ineq={} s_inf={:.3e} y_inf={:.3e} s<1e-6={} s<1e-6*smax={} y>1e-6={} y>1e-6*ymax={}",
                iter, m_ineq, s_inf, y_inf, s_small_abs, s_small_rel, y_large_abs, y_large_rel
            );
        }

        // per-row componentwise relative (bench と同形):
        //   primal: max_i |r_p[i]| / (1 + |ax[i]| + |b_ext[i]|)
        //   dual:   max_j |r_d[j]| / (1 + |qx[j]| + |c[j]| + |aty[j]|)
        // OSQP 全体正規化は ||b|| 大の問題で threshold が緩み unscale 後の bench を通せない。
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
        if nr_p_rel < eps
            && nr_d_rel < eps
            && mu < eps
            && rel_gap.abs() < DUALITY_GAP_TOL
        {
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
        // 行列側 (rho_matrix/delta_matrix) と RHS 補正側 (rho_prox/delta_prox) を区別する。
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

        if std::env::var("IPPMM_SIGMA_DIAG").ok().as_deref() == Some("1") {
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

        // 不定 Q 用に rho_retry を inertia_correction で下限し、Q+ρI を初回から PSD に。
        // 必要最低 rho は -λ_min(Q) なので天井も同様に持ち上げる (例: λ_min≈-398 を許す)。
        let mut rho_retry = rho_matrix.max(inertia_correction);
        let ldl_reg_ceiling = LDL_REG_CEILING.max(inertia_correction);
        let mut delta_matrix_retry = delta_matrix;
        let mut fac_opt: Option<KktFactor> = None;
        let mut aug_mat_opt: Option<crate::sparse::CscMatrix> = None;
        let use_schur = explicit_schur || auto_schur;
        let mut d_inv_opt: Option<Vec<f64>> = None;
        for _retry in 0..LDL_REG_RETRY_MAX {
            if timeout_ctx.should_stop() {
                status = Some(SolveStatus::Timeout);
                final_iter = iter;
                break;
            }
            let prof_t_build = if prof { Some(std::time::Instant::now()) } else { None };
            let mat_for_factor = if use_schur {
                let (s_mat, d_inv) = build_schur_system(
                    &problem.q,
                    &a_ext,
                    &sigma_vec,
                    rho_retry,
                    delta_matrix_retry,
                );
                d_inv_opt = Some(d_inv);
                s_mat
            } else {
                aug_cache.materialize(&sigma_vec, rho_retry, delta_matrix_retry)
            };
            if let Some(t) = prof_t_build {
                eprintln!("FACT_PROF section=build n={} nnz={} t={:.3}ms", mat_for_factor.nrows, mat_for_factor.values.len(), t.elapsed().as_secs_f64() * 1000.0);
            }
            if amd_perm_cache.is_none() {
                amd_perm_cache = Some(amd_with_deadline(
                    mat_for_factor.nrows,
                    &mat_for_factor.col_ptr,
                    &mat_for_factor.row_ind,
                    timeout_ctx.deadline,
                ));
            }
            let perm = amd_perm_cache.as_ref().unwrap();
            // Schur / DD-LDL は pre-permuted 未対応なので通常経路へ。
            let dd_ldl = std::env::var("IPM_DD_LDL").ok().as_deref() == Some("1");
            let use_pre_permuted = !use_schur && !dd_ldl;
            if use_pre_permuted && aug_permuted_cache.is_none() {
                aug_permuted_cache = Some(aug_cache.permute(perm));
            }
            let prof_t_factor = if prof { Some(std::time::Instant::now()) } else { None };
            let factor_result = if use_pre_permuted {
                let permuted_cache = aug_permuted_cache.as_ref().unwrap();
                let pre_permuted = permuted_cache.materialize(&sigma_vec, rho_retry, delta_matrix_retry);
                factorize_kkt_pre_permuted_cached(
                    &pre_permuted,
                    &mat_for_factor,
                    perm,
                    timeout_ctx.deadline,
                    max_l_nnz_from_budget(),
                    Some(n),
                    symbolic_cholesky_cache.clone(),
                )
            } else {
                factorize_kkt_with_cached_perm(
                    &mat_for_factor,
                    perm,
                    timeout_ctx.deadline,
                    max_l_nnz_from_budget(),
                    Some(n),
                )
            };
            if use_pre_permuted && symbolic_cholesky_cache.is_none() {
                if let Ok(ref f) = factor_result {
                    symbolic_cholesky_cache = f.symbolic_arc();
                }
            }
            match factor_result {
                Ok(f) => {
                    if let Some(t) = prof_t_factor {
                        eprintln!("FACT_PROF section=factorize n={} t={:.3}ms", mat_for_factor.nrows, t.elapsed().as_secs_f64() * 1000.0);
                    }
                    let prof_t_probe = if prof { Some(std::time::Instant::now()) } else { None };
                    // 健全性プローブ: factorize Ok でも cond(K) 大で LDL solve が Newton 方向
                    // を central path から外す病理を ||K·sol − rhs||/||rhs|| で直接弾く。
                    // iterative backend は LDL 精度概念が無いので skip。
                    if !f.is_iterative() {
                        let probe_dim = mat_for_factor.nrows;
                        let mut probe_rhs = vec![0.0_f64; probe_dim];
                        probe_rhs[..n].copy_from_slice(&r_d_pmm);
                        // 予測子 RHS 下半分: 不等式行は r_p + s、等式行は r_p。
                        for (i, slot) in probe_rhs[n..].iter_mut().enumerate() {
                            *slot = if is_eq_ext[i] { r_p_pmm[i] } else { r_p_pmm[i] + s[i] };
                        }
                        let rhs_inf = probe_rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                        if rhs_inf > 0.0 && rhs_inf.is_finite() {
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
                            // (A) ||K·sol−rhs||/||rhs|| ≤ 1e-3 (LDL 大破綻 sanity, eps 独立)
                            // (B) sol_inf / rhs_inf ≤ 1/ε_machine (cond(K) が f64 域内)
                            const LDL_HEALTH_REL_TOL: f64 = 1e-3;
                            let unhealthy = !rel_resid.is_finite()
                                || rel_resid > LDL_HEALTH_REL_TOL
                                || !amplification.is_finite()
                                || amplification > f64_precision_ceiling;
                            if unhealthy {
                                if rho_retry >= ldl_reg_ceiling {
                                    break;
                                }
                                rho_retry = (rho_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                                delta_matrix_retry = (delta_matrix_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                                continue;
                            }
                        }
                    }
                    if let Some(t) = prof_t_probe {
                        eprintln!("FACT_PROF section=probe n={} t={:.3}ms", mat_for_factor.nrows, t.elapsed().as_secs_f64() * 1000.0);
                    }
                    fac_opt = Some(f);
                    aug_mat_opt = Some(mat_for_factor);
                    break;
                }
                Err(KktError::DeadlineExceeded) => {
                    status = Some(SolveStatus::Timeout);
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    if rho_retry >= ldl_reg_ceiling {
                        break;
                    }
                    rho_retry = (rho_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                    delta_matrix_retry = (delta_matrix_retry * LDL_REG_GROWTH).min(ldl_reg_ceiling);
                }
            }
        }
        if matches!(status, Some(SolveStatus::Timeout)) {
            break;
        }
        // 第3防御: identity perm + 大きな delta で再試行。
        if fac_opt.is_none() {
            amd_perm_cache = None;
            let delta_fallback = LDL_FALLBACK_DELTA_MIN.max(rho_retry).max(delta_matrix_retry);
            let aug_mat_fb = aug_cache.materialize(&sigma_vec, rho_retry, delta_fallback);
            let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
            match factorize_kkt_with_cached_perm(
                &aug_mat_fb,
                &identity_perm,
                timeout_ctx.deadline,
                max_l_nnz_from_budget(),
                Some(n),
            ) {
                Ok(f) => {
                    fac_opt = Some(f);
                    aug_mat_opt = Some(aug_mat_fb);
                }
                Err(KktError::DeadlineExceeded) => {
                    status = Some(SolveStatus::Timeout);
                    final_iter = iter;
                }
                Err(_) => {}
            }
        }
        if matches!(status, Some(SolveStatus::Timeout)) {
            break;
        }
        let mut fac = match fac_opt {
            Some(f) => f,
            None => return numerical_error_result(n),
        };
        // MINRES (iterative) backend のみ user eps 由来 η を反映、Direct/DirectDd では no-op。
        fac.set_iterative_tol(inexact_eta);
        let aug_mat_for_ir = aug_mat_opt
            .as_ref()
            .expect("aug_mat_opt must be set when fac_opt is set");

        if let Some(t) = prof_section_start {
            prof_factor_ns += t.elapsed().as_nanos();
            prof_section_start = Some(std::time::Instant::now());
        }

        let (pred, alpha, r_c_corr) = if use_schur {
            let d_inv = d_inv_opt.as_ref().expect("d_inv must be set when use_schur");
            let pred = predictor_step_schur(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, aug_mat_for_ir, d_inv, &a_ext, n, m_ext, mu,
            );
            let (alpha, r_c_corr) = corrector_step_schur(
                &s, &y, &is_eq_ext,
                &pred, mu,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, aug_mat_for_ir, d_inv, &a_ext, n, m_ext,
                &mut dx, &mut dy, &mut ds,
            );

            (pred, alpha, r_c_corr)
        } else {
            let pred = predictor_step(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, aug_mat_for_ir, n, m_ext, mu,
                timeout_ctx.deadline,
            );
            let (alpha, r_c_corr) = corrector_step(
                &s, &y, &is_eq_ext,
                &pred, mu,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, aug_mat_for_ir, n, m_ext,
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
                    &r_c_corr, &sigma_vec, &fac, aug_mat_for_ir, d_inv, &a_ext, n, m_ext,
                    options.ipm.max_correctors, alpha,
                    &mut dx, &mut dy, &mut ds,
                    timeout_ctx.deadline,
                )
            } else {
                gondzio_correctors(
                    &s, &y, &is_eq_ext, m_ineq,
                    &r_d_pmm, &r_p_pmm,
                    &r_c_corr, &sigma_vec, &fac, aug_mat_for_ir, n, m_ext,
                    options.ipm.max_correctors, alpha,
                    &mut dx, &mut dy, &mut ds,
                    timeout_ctx.deadline,
                )
            };
        }

        let _ = pred;

        // NaN/Inf または finite-but-huge (>1e30) は LDL blow-up とみなし best-so-far で復帰。
        const DIRECTION_BLOWUP_THRESHOLD: f64 = 1e30;
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
            const MIN_CONSECUTIVE_INFEAS: usize = 3;
            if consecutive_infeas_triggers < MIN_CONSECUTIVE_INFEAS {
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "IPPMM_DEBUG iter={} infeas trigger #{} (< {}), continue iterating",
                        iter, consecutive_infeas_triggers, MIN_CONSECUTIVE_INFEAS
                    );
                }
            } else {
                // K 連続 fire: best が quality_threshold 未満なら false-positive 扱いで降格、それ以外は検出器に従う。
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
            // これにより「散発的な fire」は確証なしと判定。
            consecutive_infeas_triggers = 0;
        }

        // step magnitude trace（IPPMM_TRACE=1 のときのみ）
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
        // fraction-to-boundary は s,y>0 のみ保護で dx は無制約 → 1e3 cap で 1 iter 3 桁以上の暴発を抑制 (Wright §5.2)。
        const STEP_REL_CAP: f64 = 1e3;
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
        // best_score < eps が成立しない場合の deadlock gate (rho/delta 両方が floor 付近)。
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
        if (pmm.rho - reg_limit).abs() < reg_limit * 0.01
            && reg_limit > REG_LIMIT_MIN
        {
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


// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-4; // IP-PMM は標準 IPM より tolerance がゆるめでも通ることを確認

    fn close(a: f64, b: f64, name: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name,
            b,
            a,
            (a - b).abs()
        );
    }

    fn default_opts() -> SolverOptions {
        SolverOptions {
            timeout_secs: Some(10.0),
            use_ruiz_scaling: false,
            ..Default::default()
        }
    }

    /// IPPMM-T1: 2変数基本 QP
    /// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ippmm_basic_2d() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T1: status");
        close(result.solution[0], 0.5, "IPPMM-T1: x[0]");
        close(result.solution[1], 0.5, "IPPMM-T1: x[1]");
        close(result.objective, 0.5, "IPPMM-T1: objective");
    }

    /// IPPMM-T2: 制約なし QP
    /// min (x-3)^2 + (y-4)^2  → Q=2I, c=[-6,-8], 制約なし
    /// 期待: x*=3, y*=4, obj=-25
    #[test]
    fn test_ippmm_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T2: status");
        close(result.solution[0], 3.0, "IPPMM-T2: x[0]");
        close(result.solution[1], 4.0, "IPPMM-T2: x[1]");
        close(result.objective, -25.0, "IPPMM-T2: objective");
    }

    /// IPPMM-T3: 等式制約付き QP
    /// min x^2 + y^2  s.t. x + y = 1  (2不等式で表現)
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ippmm_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T3: status");
        close(result.solution[0], 0.5, "IPPMM-T3: x[0]");
        close(result.solution[1], 0.5, "IPPMM-T3: x[1]");
        close(result.objective, 0.5, "IPPMM-T3: objective");
    }

    /// IPPMM-T4: Box 制約付き QP
    /// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
    /// 期待: x*=y*=1, obj=-6
    #[test]
    fn test_ippmm_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T4: status");
        close(result.solution[0], 1.0, "IPPMM-T4: x[0]");
        close(result.solution[1], 1.0, "IPPMM-T4: x[1]");
        close(result.objective, -6.0, "IPPMM-T4: objective");
    }


    /// IPPMM-T5: タイムアウト動作確認
    #[test]
    fn test_ippmm_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(0.0001),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "IPPMM-T5: expected Timeout or Optimal, got {:?}",
            result.status
        );
    }

    /// IPPMM-T-conv1: 等式制約収束確認
    /// min x²+y² s.t. x+y=1 (ConstraintType::Eq)
    /// QpProblem::new() を使用
    /// 期待: 5秒以内にOptimal、x*=y*=0.5
    #[test]
    fn test_ippmm_eq_convergence_check() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
        assert_eq!(result.status, SolveStatus::Optimal, "conv-eq: status");
        close(result.solution[0], 0.5, "conv-eq: x[0]");
        close(result.solution[1], 0.5, "conv-eq: x[1]");
    }

    /// IPPMM-T-conv2: 不等式制約収束確認
    /// min x²+y² s.t. x+y>=1 (Le形式: -x-y <= -1、ConstraintType::Le)
    /// QpProblem::new() を使用
    /// 期待: 5秒以内にOptimal、x*=y*=0.5
    #[test]
    fn test_ippmm_le_convergence_check() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
        assert_eq!(result.status, SolveStatus::Optimal, "conv-le: status");
        close(result.solution[0], 0.5, "conv-le: x[0]");
        close(result.solution[1], 0.5, "conv-le: x[1]");
    }

    /// IPPMM-T-Ge1: Ge制約防御テスト
    /// min x²+y² s.t. x+y≥1 (ConstraintType::Ge)
    /// QpProblem::new() を使用
    /// 期待: 5秒以内にOptimal、x*=y*=0.5
    #[test]
    fn test_ippmm_ge_defensive() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert!(start.elapsed().as_secs_f64() < 6.0, "Test exceeded 6 second wall-clock limit");
        assert_eq!(result.status, SolveStatus::Optimal, "ge-defensive: status");
        close(result.solution[0], 0.5, "ge-defensive: x[0]");
        close(result.solution[1], 0.5, "ge-defensive: x[1]");
    }

    /// IPPMM-T-F1: 空制約退化ケース
    /// min 0.5*(x²+y²) - x - y (Q=I, c=[-1,-1], 制約なし)
    /// 期待: Optimal、x*=y*=1.0
    #[test]
    fn test_ippmm_empty_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::new(0, 2);
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "empty-constraints: status");
        close(result.solution[0], 1.0, "empty-constraints: x[0]");
        close(result.solution[1], 1.0, "empty-constraints: x[1]");
    }

    /// IPPMM-T-F2: 複数等式制約退化ケース
    /// min x²+y²+z² s.t. x+y=1 (Eq), y+z=1 (Eq)
    /// 期待: Optimal、x*=z*=1/3、y*=2/3
    #[test]
    fn test_ippmm_multiple_equality_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
        // A = [[1,1,0],[0,1,1]]
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2, 3,
        ).unwrap();
        let b = vec![1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq, ConstraintType::Eq]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            use_ruiz_scaling: false,
            ..Default::default()
        };
        let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "multi-eq: status");
        close(result.solution[0], 1.0 / 3.0, "multi-eq: x[0]");
        close(result.solution[1], 2.0 / 3.0, "multi-eq: x[1]");
        close(result.solution[2], 1.0 / 3.0, "multi-eq: x[2]");
    }
}
