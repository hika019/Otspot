//! Predictor-Corrector-Gondzio の共通ループ部品
//!
//! step.rs (IPM Mehrotra) と ippmm.rs (IP-PMM) の両方が使用する共通関数群。
//! アルゴリズム固有の差異（KKT因子化方式・PMM修正残差等）は呼び出し側が処理する。
//!
//! # 設計
//! - `compute_sigma_vec`: Σ = diag(s/y) 計算
//! - `predictor_step`: Mehrotra predictor（r_dual/r_primal を引数化）
//! - `corrector_step`: Mehrotra corrector（r_dual/r_primal を引数化）
//! - `gondzio_correctors`: Gondzio multiple centrality correctors
//! - `update_variables`: x, s, y 変数更新

use crate::linalg::ldl::LdlFactorizationAmd;
use super::common::fraction_to_boundary_masked;
use super::{TAU, BETA_GONDZIO, GAMMA_L, GAMMA_U, ALPHA_IMPROVE_THRESHOLD};

// ---------------------------------------------------------------------------
// データ構造
// ---------------------------------------------------------------------------

/// Predictor ステップの結果
pub(crate) struct PredictorResult {
    /// dy の predictor 解
    pub dy_pred: Vec<f64>,
    /// ds の predictor 解
    pub ds_pred: Vec<f64>,
    /// centering パラメータ σ
    pub sigma_center: f64,
}

// ---------------------------------------------------------------------------
// 共通関数群
// ---------------------------------------------------------------------------

/// Σ = diag(s_i / y_i) を計算（等式行は 0、nan/inf は sigma_max でクランプ）
pub(crate) fn compute_sigma_vec(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    sigma_max: f64,
) -> Vec<f64> {
    s.iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                let v = si / yi;
                if v.is_finite() { v } else { sigma_max }
            }
        })
        .collect()
}

/// Predictor ステップ
///
/// - `r_dual`:   r_d (IPM) または r_d_pmm (IPPMM)
/// - `r_primal`: r_p (IPM) または r_p_pmm (IPPMM)
#[allow(clippy::too_many_arguments)]
pub(crate) fn predictor_step(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    fac: &LdlFactorizationAmd,
    n: usize,
    m_ext: usize,
    mu: f64,
) -> PredictorResult {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    // r_c_pred[i] = -s[i]*y[i]（等式行は 0）
    let r_c_pred: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi })
        .collect();

    // r_p_mod_pred[i] = r_primal[i] - r_c_pred[i]/y[i]（等式行はそのまま）
    let r_p_mod_pred: Vec<f64> = r_primal
        .iter()
        .zip(r_c_pred.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    rhs[..n].copy_from_slice(r_dual);
    rhs[n..].copy_from_slice(&r_p_mod_pred);
    fac.solve(&rhs, &mut sol);

    // augmented system: sol[..n]=dx_pred（未使用）, sol[n..]=dy_pred
    let dy_pred = sol[n..].to_vec();

    let mut ds_pred = vec![0.0f64; m_ext];
    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds_pred[i] = 0.0;
        } else {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }
    }

    let alpha_s_pred = fraction_to_boundary_masked(s, &ds_pred, TAU, is_eq_ext);
    let alpha_y_pred = fraction_to_boundary_masked(y, &dy_pred, TAU, is_eq_ext);
    let alpha_pred = alpha_s_pred.min(alpha_y_pred);

    let mu_aff: f64 = if m_ineq > 0 {
        s.iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .enumerate()
            .filter(|&(i, _)| !is_eq_ext[i])
            .map(|(_, (((&si, &yi), &dsi), &dyi))| {
                (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
            })
            .sum::<f64>()
            / m_ineq as f64
    } else {
        0.0
    };

    let sigma_center = if mu > 1e-15 {
        (mu_aff / mu).powi(3).min(1.0)
    } else {
        0.0
    };

    PredictorResult {
        dy_pred,
        ds_pred,
        sigma_center,
    }
}

/// Corrector ステップ
///
/// dx, dy, ds を更新し、`(alpha, r_c_corr)` を返す。
/// r_c_corr は続く Gondzio correctors に渡す必要がある。
///
/// - `r_dual`:   r_d (IPM) または r_d_pmm (IPPMM)
/// - `r_primal`: r_p (IPM) または r_p_pmm (IPPMM)
#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector_step(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    pred: &PredictorResult,
    mu: f64,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    fac: &LdlFactorizationAmd,
    n: usize,
    m_ext: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> (f64, Vec<f64>) {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    // r_c_corr[i] = σ*μ - s[i]*y[i] - ds_pred[i]*dy_pred[i]（等式行は 0）
    let r_c_corr: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .zip(pred.ds_pred.iter())
        .zip(pred.dy_pred.iter())
        .enumerate()
        .map(|(i, (((&si, &yi), &dsi), &dyi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                pred.sigma_center * mu - si * yi - dsi * dyi
            }
        })
        .collect();

    // r_p_mod_corr[i] = r_primal[i] - r_c_corr[i]/y[i]（等式行はそのまま）
    let r_p_mod_corr: Vec<f64> = r_primal
        .iter()
        .zip(r_c_corr.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    rhs[..n].copy_from_slice(r_dual);
    rhs[n..].copy_from_slice(&r_p_mod_corr);
    fac.solve(&rhs, &mut sol);

    dx.copy_from_slice(&sol[..n]);
    dy.copy_from_slice(&sol[n..]);

    for i in 0..m_ext {
        if is_eq_ext[i] {
            ds[i] = 0.0;
        } else {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }
    }

    let alpha_s = fraction_to_boundary_masked(s, ds, TAU, is_eq_ext);
    let alpha_y = fraction_to_boundary_masked(y, dy, TAU, is_eq_ext);
    let alpha = alpha_s.min(alpha_y);

    (alpha, r_c_corr)
}

/// Gondzio multiple centrality correctors
///
/// dx, dy, ds, alpha を更新し、最終 alpha を返す。
///
/// - `r_dual`:   r_d (IPM) または r_d_pmm (IPPMM)
/// - `r_primal`: r_p (IPM) または r_p_pmm (IPPMM)
/// - `r_c_corr`: corrector_step が返した r_c_corr
#[allow(clippy::too_many_arguments)]
pub(crate) fn gondzio_correctors(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    r_c_corr: &[f64],
    sigma_vec: &[f64],
    fac: &LdlFactorizationAmd,
    n: usize,
    m_ext: usize,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> f64 {
    let total = n + m_ext;
    let mut rhs = vec![0.0f64; total];
    let mut sol = vec![0.0f64; total];

    let mut alpha_prev = alpha_init;
    for _k in 0..max_correctors {
        // (1) 目標 step size と mu（不等式行のみ）
        let alpha_target = (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
        let mu_target: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(ds.iter().zip(dy.iter()))
                .enumerate()
                .filter(|&(i, _)| !is_eq_ext[i])
                .map(|(_, ((&si, &yi), (&dsi, &dyi)))| {
                    (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                })
                .sum::<f64>()
                / m_ineq as f64
        } else {
            0.0
        };
        let mu_target = mu_target.max(0.0);

        // (2) 各 complementarity pair の目標範囲
        let target_lo = GAMMA_L * mu_target;
        let target_hi = GAMMA_U * mu_target;

        // (3) Gondzio corrector RHS 構築（eq行=0）
        let mut r_c_gondzio = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            if is_eq_ext[i] {
                r_c_gondzio[i] = 0.0;
                continue;
            }
            let si_new = s[i] + alpha_prev * ds[i];
            let yi_new = y[i] + alpha_prev * dy[i];
            let v_i = si_new * yi_new;
            let v_target = if v_i < target_lo {
                target_lo - v_i
            } else if v_i > target_hi {
                target_hi - v_i
            } else {
                0.0
            };
            r_c_gondzio[i] = r_c_corr[i] + v_target;
        }

        // (4) 修正 RHS 構築 & LDL 因子再利用 solve
        let r_p_mod_gondzio: Vec<f64> = r_primal
            .iter()
            .zip(r_c_gondzio.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        rhs[..n].copy_from_slice(r_dual);
        rhs[n..].copy_from_slice(&r_p_mod_gondzio);
        fac.solve(&rhs, &mut sol);

        let dx_new = sol[..n].to_vec();
        let dy_new = sol[n..].to_vec();
        let ds_new: Vec<f64> = (0..m_ext)
            .map(|i| {
                if is_eq_ext[i] {
                    0.0
                } else {
                    r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i]
                }
            })
            .collect();

        // (5) 新しい step size を計算（eq行スキップ）
        let alpha_s_new = fraction_to_boundary_masked(s, &ds_new, TAU, is_eq_ext);
        let alpha_y_new = fraction_to_boundary_masked(y, &dy_new, TAU, is_eq_ext);
        let alpha_new = alpha_s_new.min(alpha_y_new);

        // (6) 改善判定: 改善なしなら break
        if alpha_new < alpha_prev + ALPHA_IMPROVE_THRESHOLD {
            break;
        }

        // (7) 改善あり → 方向を更新
        dx.copy_from_slice(&dx_new);
        dy.copy_from_slice(&dy_new);
        ds.copy_from_slice(&ds_new);
        alpha_prev = alpha_new;
    }
    alpha_prev
}

/// 変数更新（x, s, y）
///
/// 等式行の s は 0 のまま維持し、y のみ更新する。
/// 不等式行は s, y 両方を更新し、正値制約 (>1e-12) を強制する。
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_variables(
    x: &mut [f64],
    s: &mut [f64],
    y: &mut [f64],
    dx: &[f64],
    ds: &[f64],
    dy: &[f64],
    alpha: f64,
    is_eq_ext: &[bool],
) {
    for i in 0..x.len() {
        x[i] += alpha * dx[i];
    }
    let m_ext = s.len();
    for i in 0..m_ext {
        if is_eq_ext[i] {
            // 等式行: s=0のまま、yは自由変数として更新
            y[i] += alpha * dy[i];
        } else {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }
    }
}
