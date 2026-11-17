//! 初期点 (x, s, y) の構築: warm start 経路 + Mehrotra cold-start 射影。
//!
//! box 化適用版: 境界行は KKT 系に存在せず、スラックは x の従属量
//! (s_t = b_t − a_t x、`BoundBox::slack`)。Mehrotra 射影も box 系
//! (次元 n + m_real、境界対角寄与 d_t = 1/(σ_t+δ) を Q 対角へ畳み込む) で解く
//! — 旧 extended 射影の正確 Schur 補集約と代数同値 (境界行 σ=0・δ=1 の極限)。

use super::state::{warm_bound_margin, WARM_BOUND_REL_MARGIN};
use super::warm_start::apply_qp_warm_start;
use crate::options::SolverOptions;
use crate::qp::ipm_core::kkt::{build_boxed_augmented_cache, BoundBox};
use crate::qp::problem::QpProblem;
use crate::tolerances::UNDERFLOW_GUARD;
use faer::Par;
use otspot_num::linalg::amd::amd_with_deadline;
use otspot_num::linalg::kkt_solver::{factorize_kkt_with_cached_perm_par, KktConfig};
use otspot_num::linalg::timeout::TimeoutCtx;
use otspot_num::sparse::CscMatrix;

pub(super) struct InitialPoint {
    pub(super) x: Vec<f64>,
    /// extended レイアウトの初期スラック。境界行成分は従属導出
    /// (`BoundBox::fill_slacks`) で確定済み。
    pub(super) s_full: Vec<f64>,
    /// extended レイアウト [元行 | lb | ub] の dual (box 化前後で不変)。
    pub(super) y: Vec<f64>,
    /// warm start 由来の μ。None なら cold start。
    pub(super) warm_mu: Option<f64>,
}

/// Mehrotra cold-init の射影 KKT が使う正則化 (ρ=δ=1)。
/// 旧 extended 版 `build_augmented_system(q_zero, a_ext, σ=0, 1.0, 1.0)` と同一値で、
/// 境界行の対角寄与は 1/(0+1) = 1 ずつ x 対角へ入る (soft projection)。
const INIT_PROJECTION_REGULARIZATION: f64 = 1.0;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_initial_point(
    problem: &QpProblem,
    options: &SolverOptions,
    a_ext: &CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    a_base: &CscMatrix,
    b_base: &[f64],
    is_eq_base: &[bool],
    bound: &BoundBox,
    m_orig: usize,
    m_ext: usize,
    timeout_ctx: &TimeoutCtx,
    par: Par,
) -> InitialPoint {
    let n = problem.num_vars;
    let m_real = m_orig;

    // 巨大 |ub|=1e11 で midpoint 起点だと pf が抜けないため、0 が bounds 内なら 0 を優先。
    let x0: Vec<f64> = problem
        .bounds
        .iter()
        .map(|&(lb, ub)| {
            let lb_fin = lb.is_finite();
            let ub_fin = ub.is_finite();
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

    // s0 = b_base − A_base·x0。等式行は s=0、不等式行は s≥1 にクランプ。
    // 境界行スラックは従属量なので状態として持たない (y の初期値のみ構築)。
    let mut ax0 = vec![0.0f64; m_real];
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        for k in a_base.col_ptr()[col]..a_base.col_ptr()[col + 1] {
            ax0[a_base.row_ind()[k]] += a_base.values()[k] * x0[col];
        }
    }
    let s0: Vec<f64> = b_base
        .iter()
        .zip(ax0.iter())
        .enumerate()
        .map(|(i, (&bi, &axi))| {
            if is_eq_base[i] {
                0.0
            } else {
                (bi - axi).max(1.0)
            }
        })
        .collect();
    // y は extended レイアウトのまま (元行: is_eq で 0/1、境界行: cold は 1.0)。
    let mut y0: Vec<f64> = (0..m_real)
        .map(|i| if is_eq_base[i] { 0.0 } else { 1.0 })
        .collect();
    y0.extend(std::iter::repeat_n(1.0, bound.count()));

    let mut x = x0.clone();
    let mut s_real = s0;
    let mut y = y0;

    // warm start が渡されていれば Mehrotra init を skip し、interior 補正のみ適用する。
    // (warm_start は extended レイアウトの scratch s を受け取る: 境界行分は
    //  終端の従属導出で上書きされるため transient。)
    let warm_mu = if let Some(ws) = options.warm_start_qp.as_ref() {
        let mut s_work: Vec<f64> = Vec::with_capacity(m_ext);
        s_work.extend_from_slice(&s_real);
        let mut tail = vec![0.0f64; bound.count()];
        bound.init_slacks(&x, &mut tail);
        s_work.extend(tail);
        let mu = apply_qp_warm_start(
            ws,
            problem,
            a_ext,
            b_ext,
            is_eq_ext,
            m_orig,
            m_ext,
            &mut x,
            &mut y,
            &mut s_work,
        );
        if mu.is_some() {
            s_real.copy_from_slice(&s_work[..m_real]);
        }
        mu
    } else {
        None
    };

    if warm_mu.is_none() {
        let kkt_cfg = KktConfig {
            dd_ldl: options.ipm.dd_ldl,
            minres_ir: options.ipm.effective_minres_ir(),
            max_l_nnz: options.ipm.effective_max_l_nnz(),
        };
        mehrotra_cold_init_boxed(
            problem,
            a_base,
            b_base,
            is_eq_base,
            bound,
            &ax0,
            timeout_ctx,
            par,
            &kkt_cfg,
            &mut x,
            &mut s_real,
            &mut y,
        );
    }

    // 境界スラック: box 化境界行は従属量として最終 x から導出、保持境界行は
    // 最終 x と整合する独立初期値 (以降 ds 積分で進化)。
    let mut s_full: Vec<f64> = Vec::with_capacity(m_ext);
    s_full.extend_from_slice(&s_real);
    let mut s_bound = vec![0.0f64; bound.count()];
    bound.init_slacks(&x, &mut s_bound);
    s_full.extend(s_bound);

    InitialPoint {
        x,
        s_full,
        y,
        warm_mu,
    }
}

/// Mehrotra 1992 標準初期点 (Wright §5.1) の box 化版: 実制約射影 + δ_s/δ_y
/// 正補正 + Σ 均一化補正。等式のみの射影だと |b|≈1e11 級で s0 が膨張し
/// K matrix が暴走する。
///
/// 射影系は box 系 (境界行なし・境界対角寄与畳み込み) で構築する:
/// 旧 extended 射影 [[I, A_extᵀ],[A_ext, −I]] (σ=0・ρ=δ=1) の境界行 Schur
/// 消去と正確に一致する — 境界行の対角寄与 1/(0+1) = 1 が x 対角へ加算され、
/// 右辺は Σ_t a_t·(b_t − a_t x0) として x 側へ移る。解 (dx) は extended 版と
/// 代数同値で、x 更新後の clamp (strict interior 射影) も従来と同じ。
#[allow(clippy::too_many_arguments)]
fn mehrotra_cold_init_boxed(
    problem: &QpProblem,
    a_base: &CscMatrix,
    b_base: &[f64],
    is_eq_base: &[bool],
    bound: &BoundBox,
    ax0: &[f64],
    timeout_ctx: &TimeoutCtx,
    par: Par,
    kkt_cfg: &KktConfig,
    x: &mut [f64],
    s_real: &mut [f64],
    y: &mut [f64],
) {
    let n = problem.num_vars;
    let m_real = b_base.len();
    let n_bound = bound.count();

    let r_p_real: Vec<f64> = b_base
        .iter()
        .zip(ax0.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    // 境界行の r_p は従属スラック定義より構造的 0 だが、射影右辺には
    // b_t − a_t x0 (= 従属スラックの前像) が Schur 前向き代入で乗る。
    let r_p_inf = r_p_real.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if r_p_inf > 1e-6 && !timeout_ctx.should_stop() {
        let q_zero = CscMatrix::new(n, n);
        let init_cache = build_boxed_augmented_cache(&q_zero, a_base, bound);
        let sigma_zero = vec![0.0_f64; m_real + n_bound];
        let k_init = init_cache.materialize(
            &sigma_zero,
            INIT_PROJECTION_REGULARIZATION,
            INIT_PROJECTION_REGULARIZATION,
        );
        let perm_init = amd_with_deadline(
            k_init.nrows(),
            k_init.col_ptr(),
            k_init.row_ind(),
            timeout_ctx.deadline,
        );
        if let Ok(fac_init) = factorize_kkt_with_cached_perm_par(
            &k_init,
            &perm_init,
            timeout_ctx.deadline,
            kkt_cfg,
            Some(n),
            par,
        ) {
            // box 系右辺: x ブロック = Σ_t a_t·1·(b_t − a_t x0) (box 化境界行のみ)、
            // 系行ブロック = [r_p_real | 保持境界行の b_t − a_t x0]。保持行は
            // scatter せず系行右辺として射影に入る (KKT 系の次元は n + m_sys)。
            let kept = bound.kept_rows();
            let m_sys = m_real + kept.len();
            let mut rhs_init = vec![0.0_f64; n + m_sys];
            for t in 0..n_bound {
                let j = bound.var[t];
                let slack_pre = bound.rhs[t] - bound.sign[t] * x[j];
                if bound.eliminated[t] {
                    rhs_init[j] += bound.sign[t] * slack_pre;
                } else {
                    rhs_init[n + m_real + kept.iter().position(|&k| k == t).unwrap()] = slack_pre;
                }
            }
            rhs_init[n..n + m_real].copy_from_slice(&r_p_real);
            let mut sol_init = vec![0.0_f64; n + m_sys];
            if fac_init
                .solve_with_deadline(&rhs_init, &mut sol_init, timeout_ctx.deadline)
                .is_ok()
            {
                let dx_inf = sol_init[..n].iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                if dx_inf.is_finite() && dx_inf < 1e15 {
                    for j in 0..n {
                        let x_new = x[j] + sol_init[j];
                        let (lb, ub) = problem.bounds[j];
                        x[j] = match (lb.is_finite(), ub.is_finite()) {
                            (true, true) => {
                                let range = ub - lb;
                                let margin = range * WARM_BOUND_REL_MARGIN;
                                if range > 2.0 * margin {
                                    x_new.clamp(lb + margin, ub - margin)
                                } else {
                                    0.5 * (lb + ub)
                                }
                            }
                            (true, false) => x_new.max(lb + warm_bound_margin(lb)),
                            (false, true) => x_new.min(ub - warm_bound_margin(ub)),
                            (false, false) => x_new,
                        };
                    }
                }
            }
        }
    }

    let mut ax_new = vec![0.0_f64; m_real];
    for col in 0..n {
        for k in a_base.col_ptr()[col]..a_base.col_ptr()[col + 1] {
            ax_new[a_base.row_ind()[k]] += a_base.values()[k] * x[col];
        }
    }
    let s_hat: Vec<f64> = b_base
        .iter()
        .zip(ax_new.iter())
        .enumerate()
        .map(|(i, (&bi, &axi))| if is_eq_base[i] { 0.0 } else { bi - axi })
        .collect();
    let y_hat_real: Vec<f64> = (0..m_real)
        .map(|i| if is_eq_base[i] { 0.0 } else { 1.0 })
        .collect();

    let s_min_ineq = s_hat
        .iter()
        .zip(is_eq_base.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
        .fold(f64::INFINITY, f64::min);
    let y_min_ineq = y_hat_real
        .iter()
        .zip(is_eq_base.iter())
        .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
        .fold(f64::INFINITY, f64::min);
    let delta_s = (-1.5 * s_min_ineq).max(0.0) + 1.0;
    let delta_y = (-1.5 * y_min_ineq).max(0.0) + 1.0;

    let s_pos: Vec<f64> = s_hat
        .iter()
        .enumerate()
        .map(|(i, &v)| if is_eq_base[i] { 0.0 } else { v + delta_s })
        .collect();
    let y_pos: Vec<f64> = y_hat_real
        .iter()
        .enumerate()
        .map(|(i, &v)| if is_eq_base[i] { 0.0 } else { v + delta_y })
        .collect();

    let sy_sum: f64 = s_pos
        .iter()
        .zip(y_pos.iter())
        .map(|(&si, &yi)| si * yi)
        .sum();
    let s_sum_pos: f64 = s_pos.iter().sum();
    let y_sum_pos: f64 = y_pos.iter().sum();
    let delta_s_corr = if y_sum_pos > UNDERFLOW_GUARD {
        sy_sum / (2.0 * y_sum_pos)
    } else {
        0.0
    };
    let delta_y_corr = if s_sum_pos > UNDERFLOW_GUARD {
        sy_sum / (2.0 * s_sum_pos)
    } else {
        0.0
    };

    for i in 0..m_real {
        s_real[i] = if is_eq_base[i] {
            0.0
        } else {
            s_pos[i] + delta_s_corr
        };
        y[i] = if is_eq_base[i] {
            0.0
        } else {
            y_pos[i] + delta_y_corr
        };
    }
    // 境界行 dual: y は独立状態なので real 行と同じ interior 補正を適用する。
    // s_t は従属量 (ループ先頭で x から導出) なのでδshift しない — 初期 x は
    // 上記 clamp で strict interior が保証され、導出 slack は BOUND_SLACK_FLOOR
    // 以上になる。
    for t in 0..n_bound {
        y[m_real + t] = 1.0 + delta_y + delta_y_corr;
    }
}
