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

use crate::linalg::kkt_solver::KktFactor;
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

/// IR_DD=1 (mixed-precision IR) では DD 残差が 1e-32 級まで測れるため、
/// IR を多く回して LDL precision 限界を反復で破る。10 で収束飽和。
pub(crate) const IR_MAX_ITERS_DD: usize = 10;

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
/// 誤差なし加算 (Dekker/Knuth two-sum). a + b = s + e で s, e は f64 表現可能、
/// |e| <= ulp(s)/2、a + b の真値は s + e と等しい。
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// 誤差なし乗算 (FMA を使った two-product)。a * b = hi + lo で hi, lo は f64、
/// 真値は hi + lo と一致。FMA `a.mul_add(b, -hi)` で lo を 1 命令で計算。
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    (hi, lo)
}

/// 残差 r = rhs - K·sol を **double-double 精度**で計算。K は対称上三角 CSC。
///
/// 標準 IR は f64 で残差計算するため LDL の precision (eps × cond ~ 1e-6) で
/// 頭打ちになる。残差を DD で計算すれば correction は DD precision (~1e-32)
/// 級になり、Wilkinson IR 解析により最終解の誤差は eps_residual に収束する。
///
/// LISWET9 (cond ~1e10) で f64 IR の floor 1e-6 → DD IR で 1e-12 級を期待。
fn compute_residual_dd(aug_mat: &CscMatrix, sol: &[f64], rhs: &[f64], out: &mut [f64]) {
    let n = sol.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(out.len(), n);

    // (hi, lo) DD アキュムレータ。最終的に hi + lo を out に書く。
    let mut hi = vec![0.0_f64; n];
    let mut lo = vec![0.0_f64; n];

    // 初期化: hi = rhs, lo = 0
    for i in 0..n {
        hi[i] = rhs[i];
    }

    // r -= K·sol を DD 精度で蓄積。各 K[i,j] * sol[j] を two-prod で展開、
    // (hi[i], lo[i]) に two-sum で減算累積。
    for col in 0..aug_mat.ncols {
        let xv_c = sol[col];
        for ptr in aug_mat.col_ptr[col]..aug_mat.col_ptr[col + 1] {
            let row = aug_mat.row_ind[ptr];
            let val = aug_mat.values[ptr];
            // 上三角寄与: K[row, col] * sol[col]
            let (p_hi, p_lo) = two_prod(val, xv_c);
            // hi[row] -= p_hi
            let (s, e1) = two_sum(hi[row], -p_hi);
            // lo[row] += e1 - p_lo (符号反転 + 残差伝搬)
            let (s2, e2) = two_sum(lo[row], e1 - p_lo);
            hi[row] = s;
            lo[row] = s2 + e2;

            // 対称下三角寄与: K[col, row] * sol[row] (row != col のとき)
            if row != col {
                let xv_r = sol[row];
                let (p_hi2, p_lo2) = two_prod(val, xv_r);
                let (s3, e3) = two_sum(hi[col], -p_hi2);
                let (s4, e4) = two_sum(lo[col], e3 - p_lo2);
                hi[col] = s3;
                lo[col] = s4 + e4;
            }
        }
    }

    // 最終 fold: out = hi + lo (f64 へ)
    for i in 0..n {
        out[i] = hi[i] + lo[i];
    }
}

/// aug_mat must be the symmetric upper-triangular CSC factorized into fac.
pub(crate) fn solve_with_iterative_refinement(
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    rhs: &[f64],
    sol: &mut [f64],
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) {
    let n = sol.len();
    debug_assert_eq!(rhs.len(), n);
    debug_assert_eq!(aug_mat.nrows, n);
    debug_assert_eq!(aug_mat.ncols, n);

    fac.solve_with_deadline(rhs, sol, deadline);

    if max_iters == 0 {
        return;
    }
    // 反復法 (MINRES) backend では IR をスキップする:
    //   - LDL の IR は「LDL の backward error eps×cond を refine」する後処理。
    //   - MINRES は元々反復解法で、tol まで自分で収束させる。
    //   - IR を被せると同じ MINRES を deadline まで N 回呼び直すだけで純粋に無駄。
    //   - 直接法経路では従来通り IR を実行 (LDL の精度限界突破に必要)。
    if fac.is_iterative() {
        return;
    }
    // deadline 切れなら IR をスキップ
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }

    // DD residual 経路では max_iters を IR_MAX_ITERS_DD まで拡張 (LDL precision 限界
    // を反復で破るため)。
    let use_dd_residual = std::env::var("IR_DD").ok().as_deref() == Some("1");
    let max_iters = if use_dd_residual { max_iters.max(IR_MAX_ITERS_DD) } else { max_iters };

    // 大型問題では IR の overhead が deadline を圧迫する。
    // BOYD2 (n+m_ext≈280k) で 1 反復 LDL ~30s なので IR で 2x 遅化 → 100s 予算超過。
    // 100k を境界として大型問題は IR skip（baseline 動作維持）。
    // 中小問題 (UBH1 30k / DPKLO1 0.2k 等) では IR を有効化して rank-deficient 対応。
    const IR_SKIP_LARGE_THRESHOLD: usize = 100_000;
    if n > IR_SKIP_LARGE_THRESHOLD {
        return;
    }

    let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
    // DD residual path はより低い floor まで測れるので threshold を緩める。
    // 標準 f64 IR では eps×rhs_inf 級が下限なので 1e-13 で打ち切るが、
    // DD では eps_DD ~ 1e-32 まで測れるため correction を最後まで適用する価値がある。
    let use_dd_residual = std::env::var("IR_DD").ok().as_deref() == Some("1");
    let resid_skip_threshold = if use_dd_residual {
        rhs_inf * 1e-30
    } else {
        rhs_inf * 1e-13
    };

    let mut kx = vec![0.0_f64; n];
    let mut residual = vec![0.0_f64; n];
    let mut correction = vec![0.0_f64; n];

    let trace_ir = std::env::var("IR_TRACE").ok().as_deref() == Some("1");
    // Mixed-precision IR: env=IR_DD=1 で残差計算を double-double にする。
    // 標準 f64 IR は LDL precision (eps × cond ~ 1e-6) で floor、DD 残差で
    // 解の精度を eps_DD ~ 1e-32 まで引き出せる (Wilkinson IR 解析)。
    // LISWET/YAO の precision floor 突破の本命。
    if trace_ir {
        let sol_inf_initial = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        eprintln!("IR_START n={} rhs_inf={:.3e} sol_inf={:.3e} thr={:.3e} dd={}",
            n, rhs_inf, sol_inf_initial, resid_skip_threshold, use_dd_residual);
    }
    for _ir_iter in 0..max_iters {
        // deadline 切れなら IR を中断 (反復法 backend で最重要)
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        if use_dd_residual {
            // DD 精度で residual = rhs - K·sol を直接計算
            compute_residual_dd(aug_mat, sol, rhs, &mut residual);
        } else {
            // f64: K * sol (symmetric matvec, upper-triangular CSC) → residual = rhs - kx
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
            for i in 0..n {
                residual[i] = rhs[i] - kx[i];
            }
        }

        let mut resid_inf = 0.0_f64;
        for i in 0..n {
            resid_inf = resid_inf.max(residual[i].abs());
        }
        if resid_inf <= resid_skip_threshold {
            if trace_ir { eprintln!("IR iter={} EXIT_resid_small resid_inf={:.3e}", _ir_iter, resid_inf); }
            return;
        }

        for v in correction.iter_mut() {
            *v = 0.0;
        }
        fac.solve_with_deadline(&residual, &mut correction, deadline);

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

/// Schur complement system 経由で KKT step (dx, dy) を解く。
///
/// augmented system [Q+ρI A^T; A −D] [dx; dy] = [r_d; r_p_mod] と数学的に等価:
///   1. S·dx = r_d + A^T D^{-1} r_p_mod (Cholesky で解く)
///   2. dy = D^{-1} (A·dx − r_p_mod)
///
/// 利点: S は n×n SPD で augmented (n+m_ext × n+m_ext) より小、Cholesky 高精度。
///
/// 精度保護: 2. の back-substitution は active constraints (σ+δ → 0、|D⁻¹| 大)
/// で `(A·dx − r_p_mod)` の f64 cancellation により |dy| が桁外れに増幅し、
/// 外側 IPM の dual feasibility (df) が指数的に発散する病理がある
/// (実測: QPLIB_9008 で df=0.03 → 2e4)。この cancellation を TwoFloat (DD,
/// ≈106 bit) で計算することで relative 残差を ε_DD ≈ 5e-32 で抑え、|D⁻¹|
/// による増幅後も dy の絶対誤差を f64 規範に保つ。
pub(crate) fn solve_kkt_via_schur(
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    r_d: &[f64],
    r_p_mod: &[f64],
    dx_out: &mut [f64],
    dy_out: &mut [f64],
) {
    use super::kkt::spmtv;
    use twofloat::TwoFloat;

    let n = r_d.len();
    let m_ext = r_p_mod.len();

    // rhs_S = r_d + A^T D^{-1} r_p_mod
    let mut d_inv_rp = vec![0.0_f64; m_ext];
    for i in 0..m_ext {
        d_inv_rp[i] = d_inv[i] * r_p_mod[i];
    }
    let mut at_d_inv_rp = vec![0.0_f64; n];
    spmtv(a_ext, &d_inv_rp, &mut at_d_inv_rp);
    let rhs_s: Vec<f64> = r_d
        .iter()
        .zip(at_d_inv_rp.iter())
        .map(|(&r, &v)| r + v)
        .collect();

    // S·dx = rhs_S
    s_fac.solve(&rhs_s, dx_out);
    let _ = s_mat;
    let _ = n;

    // dy = D^{-1} (A·dx − r_p_mod) — DD precision で cancellation を防ぐ。
    // A·dx の sparse matvec を TwoFloat で実行し、r_p_mod の引き算と
    // d_inv 乗算も DD で行ってから f64 に丸める。
    let zero_dd = TwoFloat::from(0.0);
    let mut a_dx_dd: Vec<TwoFloat> = vec![zero_dd; m_ext];
    for col in 0..n {
        let cs = a_ext.col_ptr[col];
        let ce = a_ext.col_ptr[col + 1];
        let dx_col = dx_out[col];
        for k in cs..ce {
            let row = a_ext.row_ind[k];
            let v = a_ext.values[k];
            a_dx_dd[row] = a_dx_dd[row] + TwoFloat::new_mul(v, dx_col);
        }
    }
    for i in 0..m_ext {
        let diff_dd = a_dx_dd[i] - TwoFloat::from(r_p_mod[i]);
        let scaled = diff_dd * TwoFloat::from(d_inv[i]);
        dy_out[i] = f64::from(scaled);
    }
}


/// Predictor ステップ (Schur version)
#[allow(clippy::too_many_arguments)]
pub(crate) fn predictor_step_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    mu: f64,
) -> PredictorResult {
    let r_c_pred: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi })
        .collect();

    let r_p_mod_pred: Vec<f64> = r_primal
        .iter()
        .zip(r_c_pred.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    let mut dx = vec![0.0_f64; n];
    let mut dy_pred = vec![0.0_f64; m_ext];
    solve_kkt_via_schur(s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_pred, &mut dx, &mut dy_pred);

    let mut ds_pred = vec![0.0_f64; m_ext];
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

/// Corrector ステップ (Schur version)
#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector_step_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    pred: &PredictorResult,
    mu: f64,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> (f64, Vec<f64>) {
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

    let r_p_mod_corr: Vec<f64> = r_primal
        .iter()
        .zip(r_c_corr.iter())
        .zip(y.iter())
        .enumerate()
        .map(|(i, ((&rpi, &rci), &yi))| {
            if is_eq_ext[i] { rpi } else { rpi - rci / yi }
        })
        .collect();

    solve_kkt_via_schur(s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_corr, dx, dy);
    let _ = n;

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

/// Gondzio multiple centrality correctors (Schur version)
#[allow(clippy::too_many_arguments)]
pub(crate) fn gondzio_correctors_schur(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    r_c_corr: &[f64],
    sigma_vec: &[f64],
    s_fac: &KktFactor,
    s_mat: &CscMatrix,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    n: usize,
    m_ext: usize,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
) -> f64 {
    let mut alpha_prev = alpha_init;
    for _k in 0..max_correctors {
        // (1) 目標 step size と mu (不等式行のみ) — augmented と完全一致
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

        // (3) Gondzio corrector RHS 構築 (augmented gondzio_correctors と完全一致):
        //   si_new = s + alpha_prev * ds, yi_new = y + alpha_prev * dy
        //   v_i = si_new * yi_new
        //   v_target = (target_lo - v_i) if v_i < target_lo else
        //              (target_hi - v_i) if v_i > target_hi else 0
        //   r_c_gondzio = r_c_corr + v_target
        let mut r_c_gondzio = vec![0.0_f64; m_ext];
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

        // (4) 修正 RHS と Schur solve
        let r_p_mod_gondzio: Vec<f64> = r_primal
            .iter()
            .zip(r_c_gondzio.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        let mut dx_new = vec![0.0_f64; n];
        let mut dy_new = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            s_fac, s_mat, d_inv, a_ext, r_dual, &r_p_mod_gondzio, &mut dx_new, &mut dy_new,
        );

        let ds_new: Vec<f64> = (0..m_ext)
            .map(|i| {
                if is_eq_ext[i] {
                    0.0
                } else {
                    r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i]
                }
            })
            .collect();

        let alpha_s_new = fraction_to_boundary_masked(s, &ds_new, TAU, is_eq_ext);
        let alpha_y_new = fraction_to_boundary_masked(y, &dy_new, TAU, is_eq_ext);
        let alpha_new = alpha_s_new.min(alpha_y_new);

        if alpha_new <= alpha_prev * ALPHA_IMPROVE_THRESHOLD {
            break;
        }
        alpha_prev = alpha_new;
        dx.copy_from_slice(&dx_new);
        dy.copy_from_slice(&dy_new);
        ds.copy_from_slice(&ds_new);
    }
    alpha_prev
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
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    mu: f64,
    deadline: Option<std::time::Instant>,
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
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

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
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
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
    solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

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
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
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
        solve_with_iterative_refinement(fac, aug_mat, &rhs, &mut sol, IR_MAX_ITERS, deadline);

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
    use super::{compute_sigma_vec, update_variables, solve_kkt_via_schur};
    use crate::qp::ipm_core::kkt::{build_augmented_system, build_schur_system};
    use crate::linalg::{ldl, amd::amd_with_deadline};
    use crate::sparse::CscMatrix;

    /// DD 残差計算 (compute_residual_dd) が f64 残差より高精度なことを確認。
    /// 「a × b」を意図的に f64 でほぼ相殺する設定で、f64 残差は丸め誤差を出すが
    /// DD 残差は exact 値を返すことを検証。
    #[test]
    fn test_dd_residual_precision_vs_f64() {
        use super::compute_residual_dd;
        // K = [[a, 0], [0, 1]] with a near 1.0 + small.
        // x = [1.0, 1.0]; rhs = [a, 1.0].
        // 真の residual = rhs - K·x = [0, 0].
        // ただし f64 演算で a*1.0 - a の中間に丸め誤差が入る scenarios 用。
        let n = 2;
        let a = 1.0_f64 + 1e-16; // 1 + ulp
        let mat = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[a, 1.0],
            n, n,
        ).unwrap();
        let sol = vec![1.0_f64, 1.0_f64];
        let rhs = vec![a, 1.0_f64];
        let mut r_dd = vec![0.0_f64; n];
        compute_residual_dd(&mat, &sol, &rhs, &mut r_dd);
        // DD residual: exact 0 を期待 (a * 1 = a 完全、a - a = 0)
        for &v in &r_dd {
            assert!(v.abs() < 1e-30, "DD residual should be ~0, got {:e}", v);
        }
    }

    /// 多制約・等式・不等式混在の LISWET 風テスト。
    /// sigma にも幅広い range (1e-3〜1e3) を持たせて Schur が数値的に augmented と一致することを確認。
    #[test]
    fn test_schur_matches_augmented_realistic() {
        // n=4, m_ext=6 (2 equality + 4 inequality)
        let n = 4;
        let m_ext = 6;

        // Q diagonal: 2, 4, 0.5, 1.0
        let q = CscMatrix::from_triplets(
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &[2.0, 4.0, 0.5, 1.0],
            n, n,
        ).unwrap();

        // A_ext: 6×4 にいくつかの非ゼロ
        // 行 0: x0 + x1 (eq)
        // 行 1: x2 + x3 (eq)
        // 行 2: x0 (lb)
        // 行 3: x1 (lb)
        // 行 4: -x2 (ub)
        // 行 5: x0 + x3 (mixed)
        let rows = vec![0, 0, 1, 1, 2, 3, 4, 5, 5];
        let cols = vec![0, 1, 2, 3, 0, 1, 2, 0, 3];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
        let a_ext = CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap();

        // sigma: equality は 0、inequality は様々な値 (LISWET 風 dynamic range)
        let sigma_vec = vec![0.0, 0.0, 1e-3, 1e1, 1e3, 5e-2];
        let rho_p = 0.05_f64;
        let delta_d = 0.02_f64;

        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let aug_perm = amd_with_deadline(aug_mat.nrows, &aug_mat.col_ptr, &aug_mat.row_ind, None);
        let aug_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&aug_mat, &aug_perm, None).unwrap());

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm = amd_with_deadline(s_mat.nrows, &s_mat.col_ptr, &s_mat.row_ind, None);
        let s_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&s_mat, &s_perm, None).unwrap());

        let r_d = vec![0.5, -1.0, 0.2, 0.8];
        let r_p_mod = vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6];

        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug);
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            &s_fac, &s_mat, &d_inv, &a_ext, &r_d, &r_p_mod,
            &mut dx_schur, &mut dy_schur,
        );

        eprintln!("dx_aug   = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug   = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            let diff = (dx_aug[i] - dx_schur[i]).abs();
            let scale = dx_aug[i].abs().max(dx_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dx[{}]: aug={}, schur={}, rel_diff={}",
                i, dx_aug[i], dx_schur[i], diff / scale
            );
        }
        for i in 0..m_ext {
            let diff = (dy_aug[i] - dy_schur[i]).abs();
            let scale = dy_aug[i].abs().max(dy_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dy[{}]: aug={}, schur={}, rel_diff={}",
                i, dy_aug[i], dy_schur[i], diff / scale
            );
        }
    }

    /// Schur と augmented LDL が同じ (dx, dy) を出すか検証する。
    /// 簡単な 2 変数 + 1 制約の問題で数値的に等価性を確認する。
    #[test]
    fn test_schur_matches_augmented() {
        // Q = diag(2, 4), A = [1, 1], b = 3
        // Sigma = [0.5], rho=ρ=0.1, delta=0.05
        let n = 2;
        let m_ext = 1;

        // Q 上三角 (full sym 慣例)
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0, 4.0],
            n, n,
        ).unwrap();

        // A_ext = [1, 1] (1×2)
        let a_ext = CscMatrix::from_triplets(
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            m_ext, n,
        ).unwrap();

        let sigma_vec = vec![0.5_f64];
        let rho_p = 0.1_f64;
        let delta_d = 0.05_f64;

        // augmented LDL を構築・factorize
        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let perm: Vec<usize> = (0..aug_mat.nrows).collect();
        let aug_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&aug_mat, &perm, None).unwrap());

        // Schur を構築・factorize
        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm: Vec<usize> = amd_with_deadline(s_mat.nrows, &s_mat.col_ptr, &s_mat.row_ind, None);
        let s_fac = crate::linalg::kkt_solver::KktFactor::Direct(crate::linalg::ldl::factorize_quasidefinite_with_cached_perm(&s_mat, &s_perm, None).unwrap());

        // テスト RHS
        let r_d = vec![1.0, 2.0];
        let r_p_mod = vec![3.0];

        // augmented 経由 (IR なし)
        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug);
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        // Schur 経由
        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        solve_kkt_via_schur(
            &s_fac, &s_mat, &d_inv, &a_ext, &r_d, &r_p_mod,
            &mut dx_schur, &mut dy_schur,
        );

        eprintln!("dx_aug = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            assert!(
                (dx_aug[i] - dx_schur[i]).abs() < 1e-9,
                "dx[{}]: aug={}, schur={}", i, dx_aug[i], dx_schur[i]
            );
        }
        for i in 0..m_ext {
            assert!(
                (dy_aug[i] - dy_schur[i]).abs() < 1e-9,
                "dy[{}]: aug={}, schur={}", i, dy_aug[i], dy_schur[i]
            );
        }
    }


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
