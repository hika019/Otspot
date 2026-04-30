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
use crate::sparse::CscMatrix;
use super::common::fraction_to_boundary_masked;
use super::{TAU, BETA_GONDZIO, GAMMA_L, GAMMA_U, ALPHA_IMPROVE_THRESHOLD};

/// Iterative refinement の反復回数。各反復は内部 guard
/// (correction_inf > sol_inf, NaN/Inf, residual<threshold) で発散時 skip するため、
/// 安定問題に副作用なし。大型問題 (n+m_ext > 100k) では IR 自体が skip される。
///
/// 3 は経験値: 1 では QPILOTNO の pf=3.6e-5→2.2e-5 改善が出ず、10 にしても
/// LDL の f64 精度限界 (Σ dynamic range 1e18 × eps_f64 で相対精度 ~1e2 損失) で
/// LISWET 系の改善はない。3 で QPILOTNO 改善・他コスト軽微の balance。
pub(crate) const IR_MAX_ITERS: usize = 3;

/// Iterative refinement of LDL solve.
///
/// fac.solve produces sol approximating K * sol = rhs. With LDL precision limit
/// (rank-deficient K, large condition number), sol may have residual = rhs - K*sol
/// of magnitude proportional to ||K|| * eps_machine.
///
/// One IR step:
///   1. residual = rhs - K * sol
///   2. correction = fac.solve(residual)
///   3. sol += correction
///
/// Skipped if residual is already small (rhs_inf * 1e-13).
/// aug_mat must be the symmetric upper-triangular CSC factorized into fac.
pub(crate) fn solve_with_iterative_refinement(
    fac: &LdlFactorizationAmd,
    aug_mat: &CscMatrix,
    rhs: &[f64],
    sol: &mut [f64],
    max_iters: usize,
) {
    let n = sol.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(aug_mat.nrows, n);
    debug_assert_eq!(aug_mat.ncols, n);

    fac.solve(rhs, sol);

    if max_iters == 0 {
        return;
    }

    // 大型問題では IR の overhead が deadline を圧迫する。
    // BOYD2 (n+m_ext≈280k) で 1 反復 LDL ~30s なので IR で 2x 遅化 → 100s 予算超過。
    // 100k を境界として大型問題は IR skip（baseline 動作維持）。
    // 中小問題 (UBH1 30k / DPKLO1 0.2k 等) では IR を有効化して rank-deficient 対応。
    const IR_SKIP_LARGE_THRESHOLD: usize = 100_000;
    if n > IR_SKIP_LARGE_THRESHOLD {
        return;
    }

    let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
    let resid_skip_threshold = rhs_inf * 1e-13;

    let mut kx = vec![0.0_f64; n];
    let mut residual = vec![0.0_f64; n];
    let mut correction = vec![0.0_f64; n];

    let trace_ir = std::env::var("IR_TRACE").ok().as_deref() == Some("1");
    if trace_ir {
        let sol_inf_initial = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        eprintln!("IR_START n={} rhs_inf={:.3e} sol_inf={:.3e} thr={:.3e}", n, rhs_inf, sol_inf_initial, resid_skip_threshold);
    }
    for _ir_iter in 0..max_iters {
        // K * sol (symmetric matvec, upper-triangular CSC).
        for v in kx.iter_mut() {
            *v = 0.0;
        }
        for col in 0..aug_mat.ncols {
            for ptr in aug_mat.col_ptr[col]..aug_mat.col_ptr[col + 1] {
                let row = aug_mat.row_ind[ptr];
                let val = aug_mat.values[ptr];
                kx[row] += val * sol[col];
                if row != col {
                    kx[col] += val * sol[row];
                }
            }
        }

        let mut resid_inf = 0.0_f64;
        for i in 0..n {
            residual[i] = rhs[i] - kx[i];
            resid_inf = resid_inf.max(residual[i].abs());
        }
        if resid_inf <= resid_skip_threshold {
            if trace_ir { eprintln!("IR iter={} EXIT_resid_small resid_inf={:.3e}", _ir_iter, resid_inf); }
            return;
        }

        for v in correction.iter_mut() {
            *v = 0.0;
        }
        fac.solve(&residual, &mut correction);

        // Backtrack guard: NaN/Inf protection
        let any_bad = correction.iter().any(|v| !v.is_finite());
        if any_bad {
            if trace_ir { eprintln!("IR iter={} EXIT_nan resid_inf={:.3e}", _ir_iter, resid_inf); }
            return;
        }

        // Adaptive guard: correction が現在の sol より大きい場合は不安定（LDL 精度を
        // 超えた虚偽の補正）として skip。ill-conditioned KKT で IR が暴れる病理を防ぐ。
        // 根拠: 真の補正は元 sol より小さい高次補正のはず。同等以上の補正は LDL の precision
        // 限界を超えて誤った方向を出している証左。BOYD2/CONT-300/QSHELL 等で IR を入れて
        // PASS→TIMEOUT 退行した症状の対策。
        let correction_inf = correction.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let sol_inf = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        if trace_ir {
            eprintln!("IR iter={} resid_inf={:.3e} correction_inf={:.3e} sol_inf={:.3e}", _ir_iter, resid_inf, correction_inf, sol_inf);
        }
        if correction_inf > sol_inf {
            if trace_ir { eprintln!("IR iter={} EXIT_correction_too_large", _ir_iter); }
            return;
        }

        for i in 0..n {
            sol[i] += correction[i];
        }
    }
}

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
    aug_mat: &CscMatrix,
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
    // Iterative refinement で LDL solve 精度向上（rank-deficient Q 対応）。
    // max_iters=3: LISWET の proximal floor 突破には 1 回では足りず scaled pf~1e-6 で
    // 止まる。3 回で 1 桁改善見込み。各反復は内部 guard で発散時 skip するため安全。
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS);

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
    aug_mat: &CscMatrix,
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
    // Iterative refinement で LDL solve 精度向上 (corrector も同じ精度を得る)
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS);

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
    aug_mat: &CscMatrix,
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
        // Iterative refinement で LDL solve 精度向上 (gondzio centering corrector)
        solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS);

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

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{compute_sigma_vec, update_variables};

    /// compute_sigma_vec: 等式制約行のsigmaは0になること
    #[test]
    fn test_compute_sigma_vec_eq_row_is_zero() {
        let s = vec![2.0, 4.0];
        let y = vec![1.0, 2.0];
        // 1行目が等式（is_eq_ext[0]=true）
        let is_eq_ext = vec![true, false];
        let sigma_max = 1e6_f64;
        let result = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);
        // 等式行 → 0.0
        assert_eq!(result[0], 0.0, "等式行のsigmaは0であること");
        // 不等式行 → s/y = 4/2 = 2.0
        let expected = 4.0 / 2.0;
        assert!(
            (result[1] - expected).abs() < 1e-12,
            "不等式行のsigma = s/y = {} (expected {})",
            result[1],
            expected
        );
    }

    /// update_variables: alpha=1.0 でdx/ds/dyが完全適用・s正値制約を確認
    #[test]
    fn test_update_variables_alpha_one() {
        let mut x = vec![1.0, 2.0];
        let mut s = vec![0.5, 0.5];
        let mut y = vec![1.0, 1.0];
        let dx = vec![0.1, 0.2];
        let ds = vec![0.3, -0.6]; // 2番目の不等式行: s[1]=0.5-0.6=-0.1 → クランプされ1e-12
        let dy = vec![0.1, 0.1];
        let is_eq_ext = vec![false, false];
        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, 1.0, &is_eq_ext);
        // x の更新
        assert!((x[0] - 1.1).abs() < 1e-12);
        assert!((x[1] - 2.2).abs() < 1e-12);
        // s[0]: 0.5 + 0.3 = 0.8
        assert!((s[0] - 0.8).abs() < 1e-12);
        // s[1]: 0.5 - 0.6 = -0.1 → 正値制約で 1e-12
        assert_eq!(s[1], 1e-12, "s が負になった場合は 1e-12 にクランプされること");
        // y の更新
        assert!((y[0] - 1.1).abs() < 1e-12);
        assert!((y[1] - 1.1).abs() < 1e-12);
    }
}
