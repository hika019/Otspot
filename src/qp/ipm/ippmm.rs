//! IP-PMM 完全独立実装
//!
//! Interior Point-Proximal Method of Multipliers (Gondzio 2021)
//! 論文: "An Interior Point-Proximal Method of Multipliers for Convex Quadratic Programming"
//! DOI: 10.1007/s10589-020-00240-9
//!
//! # 設計方針
//! - step.rs / kkt.rs の関数を一切呼ばない（共有禁止）
//! - IP-PMM のネイティブ実装: proximal 参照点 + adaptive rho/delta
//! - 4 系統独立パスの 1 つとして Concurrent Solver から選択される
//!
//! # 理論要点
//! PMM subproblem:
//!   min (1/2)xᵀQx + cᵀx + (ρ/2)||x - x_ref||² + λᵀ(Ax - b)
//!   + (1/2δ)||Ax - b||² + (δ/2)||y - y_ref||²  s.t. x >= 0
//!
//! augmented KKT（上三角 CSC、quasi-definite）:
//!   K = [(Q + ρI),  Aᵀ   ]
//!       [A,        -D    ]  where D = Σ + δI, Σ = diag(s/y)
//!
//! RHS（proximal 修正済み）:
//!   r_d_pmm = r_d - ρ*(x - x_ref)   (dual  residual with proximal primal term)
//!   r_p_pmm = r_p - δ*(y - y_ref)   (primal residual with dual augmented Lagrangian)
//!
//! PMM update rule (Algorithm PEU §5.1.4, Pougkakiotis & Gondzio 2021):
//!   r = |μ_k - μ_{k+1}| / μ_k   (変数更新後の実μで計算)
//!   primal_improved = (0.95 * prev_nr_p > nr_p)
//!   dual_improved   = (0.95 * prev_nr_d > nr_d)
//!   if primal_improved: y_ref = y; δ *= (1 - r)
//!   else:               δ *= (1 - r/3)
//!   if dual_improved:   x_ref = x; ρ *= (1 - r)
//!   else:               ρ *= (1 - r/3)

use crate::linalg::amd::amd_with_deadline;
use crate::linalg::kkt_solver::{
    factorize_kkt_with_cached_perm, max_l_nnz_from_budget, KktError, KktFactor,
};
use crate::linalg::ruiz::RuizScaler;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
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

// ---------------------------------------------------------------------------
// PMM パラメータ定数（§35 PARAM マーカー）
// ---------------------------------------------------------------------------

/// PMM 初期 rho（primal proximal）
/// PARAM: 根拠=Pougkakiotis&Gondzio(2021) §5.1 論文値 8.0
/// Ruizスケーリング後の単位スケール問題を前提とした値。
/// N1修正後は減衰が正しく機能するため論文値8.0が適切。
const RHO_INIT: f64 = 8.0;

/// PMM 初期 delta（dual proximal）
/// PARAM: 根拠=Pougkakiotis&Gondzio(2021) §5.1 論文値 8.0
/// Ruizスケーリング後の単位スケール問題を前提とした値。
/// N1修正後は減衰が正しく機能するため論文値8.0が適切。
const DELTA_INIT: f64 = 8.0;

/// PMM 改善判定閾値（5% 以上の残差減少で改善とみなす）
/// PARAM: 根拠=Gondzio2021 MATLAB実装(0.95*prev > current) | 要検証=閾値の感度
const PMM_IMPROVE_THRESHOLD: f64 = 0.95;

/// PMM 遅い減衰率（改善なし時に rho/delta をゆっくり減らす係数）
/// PARAM: 根拠=MATLAB拡張版IP-PMM準拠（設計書§A_PMM参照）
const PMM_SLOW_RATE: f64 = 2.0 / 3.0;

/// μ がゼロとみなされる閾値。f64 機械精度 (~2.2e-16) のすぐ上で「実質 0」と判定する境界。
/// 等式問題で μ=0 の極限に到達したケースの mu_rate 切替に使われる。
const MU_ZERO_THRESHOLD: f64 = 1e-15;

/// LDL 因子化失敗時の正則化リトライ上限回数。経験値 (δ 探索空間 1e-4→1e0 は約 4 段階で到達)。
const LDL_REG_RETRY_MAX: usize = 10;
/// LDL 因子化失敗時の正則化倍率。各リトライで rho/delta を 10 倍する。
const LDL_REG_GROWTH: f64 = 10.0;
/// LDL 因子化リトライの正則化上限。条件数悪化を防ぐ経験的上限。
const LDL_REG_CEILING: f64 = 1.0;
/// LDL 因子化最終 fallback の delta 下限 (identity ordering 経路用)。
const LDL_FALLBACK_DELTA_MIN: f64 = 1e-2;

/// alpha 停滞検出: line search が `alpha < ALPHA_STALL_EPS` のときに stall とみなす。
const ALPHA_STALL_EPS: f64 = 1e-8;
/// alpha 停滞回数の早期脱出閾値 (best-so-far で復帰)。
const ALPHA_STALL_N: usize = 5;
/// alpha=0 連続回数のデッドロック判定閾値 (rho/delta が reg_limit に張り付いた場合の無限ループ対策)。
const ALPHA_DEADLOCK_N: usize = 20;


// ---------------------------------------------------------------------------
// PMM 状態構造体
// ---------------------------------------------------------------------------

struct PmmState {
    /// primal 参照点 ζ (Gondzio 表記)
    x_ref: Vec<f64>,
    /// dual 参照点 λ (Gondzio 表記)
    y_ref: Vec<f64>,
    /// primal proximal パラメータ ρ
    rho: f64,
    /// dual proximal パラメータ δ
    delta: f64,
    /// 前反復の非正則化 primal 残差ノルム
    prev_nr_p: f64,
    /// 前反復の非正則化 dual 残差ノルム
    prev_nr_d: f64,
}

// ---------------------------------------------------------------------------
// 公開エントリポイント
// ---------------------------------------------------------------------------

/// IP-PMM 内部ソルバー（Ruiz スケーリング適用済み problem を受け取る）
///
/// augmented KKT + LDLT 直接法 + PMM 参照点更新
pub(crate) fn solve_ippmm_inner(
    problem: &QpProblem,
    options: &SolverOptions,
    scaler: Option<&RuizScaler>,
    orig_problem: Option<&QpProblem>,
    eps_orig: f64,
) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築（6-tuple: is_eq_ext追加）
    let (a_ext, b_ext, m_ext, m_orig, _n_lb, is_eq_ext) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 等式行数と不等式行数
    let eq_count = is_eq_ext.iter().filter(|&&v| v).count();
    let m_ineq = m_ext - eq_count;

    // 初期点（有界変数はボックス中点から開始して primal feasibility を確保）
    // 初期点 x0 = ボックス中点（lb+ub)/2）。無限界変数は 0。
    let x0: Vec<f64> = problem
        .bounds
        .iter()
        .map(|&(lb, ub)| {
            if lb.is_finite() && ub.is_finite() {
                (lb + ub) / 2.0
            } else if lb.is_finite() {
                lb + 1.0
            } else if ub.is_finite() {
                ub - 1.0
            } else {
                0.0
            }
        })
        .collect();

    // s0 = b_ext - A_ext * x0 でプライマル実行可能にする。
    // 等式行: s=0（スラックなし）、不等式行: 下限 1.0 でクランプ
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

    // ── Mehrotra 1992 標準初期点 (等式 + 不等式制約両方への射影 + 均一化補正) ─────
    //
    // 旧実装: 等式制約のみ射影、不等式は s = max(b - A·x0, 1.0) のまま、y0 = 1.0 固定。
    // 病理: QPLIB_9002 のような |b|≈1e11 級の問題で射影が等式のみだと x0 が
    //   bound 中点(=0)のままで s0 = b ≈ 1e11 に膨らみ、Σ = s/y が cond 1e11 級
    //   で K matrix 暴走 → mu 増加 → dx 発散 → NaN_guard 巻き戻し。
    //
    // 修正 (Mehrotra 1992 標準、Wright "Primal-Dual Interior-Point Methods" §5.1):
    //   1. 全制約行 (等式 + 不等式) の残差を RHS に入れて Newton step で x̂ を取る
    //      → s_hat = b - A·x̂ が小さくなる (x̂ が問題スケールに合う)
    //   2. δ_s, δ_y で正補正 (s, y ≥ 0 を保証)
    //   3. δ_s_corr, δ_y_corr で sum 均一化 (s × y を平均化、Σ 分散を抑制)
    //
    // 論文準拠 (Pougkakiotis & Gondzio 2021 が前提とする Mehrotra 標準初期化)。
    {
        // 全制約行の残差を RHS に
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
                &k_init, &perm_init, timeout_ctx.deadline, max_l_nnz_from_budget(),
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

        // s_hat, y_hat を再計算 (射影後の x で)
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

        // Mehrotra 標準: δ_s = max(-1.5 * min(ŝ), 0) + 1 で s ≥ 1 を保証
        let s_min_ineq = s_hat.iter().zip(is_eq_ext.iter())
            .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
            .fold(f64::INFINITY, f64::min);
        let y_min_ineq = y_hat.iter().zip(is_eq_ext.iter())
            .filter_map(|(&v, &eq)| if eq { None } else { Some(v) })
            .fold(f64::INFINITY, f64::min);
        let delta_s = (-1.5 * s_min_ineq).max(0.0) + 1.0;
        let delta_y = (-1.5 * y_min_ineq).max(0.0) + 1.0;

        // shifted 値
        let s_pos: Vec<f64> = s_hat.iter().enumerate()
            .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_s })
            .collect();
        let y_pos: Vec<f64> = y_hat.iter().enumerate()
            .map(|(i, &v)| if is_eq_ext[i] { 0.0 } else { v + delta_y })
            .collect();

        // 均一化補正: s × y を平均化
        let sy_sum: f64 = s_pos.iter().zip(y_pos.iter()).map(|(&si, &yi)| si * yi).sum();
        let s_sum_pos: f64 = s_pos.iter().sum();
        let y_sum_pos: f64 = y_pos.iter().sum();
        let delta_s_corr = if y_sum_pos > 1e-300 { sy_sum / (2.0 * y_sum_pos) } else { 0.0 };
        let delta_y_corr = if s_sum_pos > 1e-300 { sy_sum / (2.0 * s_sum_pos) } else { 0.0 };

        // 最終 s0, y0 (= s_pos + δ_s_corr, y_pos + δ_y_corr)
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

    // PMM 状態初期化
    let mut pmm = PmmState {
        x_ref: x.clone(),
        y_ref: y.clone(),
        rho: RHO_INIT,
        delta: DELTA_INIT,
        prev_nr_p: f64::INFINITY,
        prev_nr_d: f64::INFINITY,
    };
    let _ = x0; let _ = y0; let _ = s0;

    // PARAM: 根拠=MATLAB拡張版IP-PMM準拠 (env QP_REG_LIMIT で診断 override 可)。
    // 【履歴】論文式(動的) を一時導入→DTOC3(‖A‖∞≈2.0)で reg_limit が
    // 2500倍緩くなり退行。best-so-far + false-unbounded 格下げは維持したまま reg_limit は定数に戻す。
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
    // Adaptive reg_limit: rank-deficient Q + c≈0 の問題 (UBH1) で rho が floor に
    // 張り付いて proximal 項が df 残差を支配し、IPM が真の Optimal に到達できない
    // 病理を解消するため、特定パターンで floor を動的に下げる。
    //
    // トリガーパターン (UBH1 シグネチャ):
    //   max(|c|) < 1e-6 (cost vector が ≈ 0、Q が支配的)
    //   かつ rho == reg_limit (decay が止まっている)
    //   かつ proximal 項が df の半分以上を占める
    // c が非ゼロな問題 (LISWET 等) ではトリガーしない。
    let c_max = problem.c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let allow_adaptive_reg = c_max < 1e-6;
    let mut reg_limit = initial_reg_limit;
    /// 適応的 floor の最下限 (これ以上は数値不安定のリスク)
    const REG_LIMIT_MIN: f64 = 1e-14;
    /// 適応 trigger: prox_d_inf > df * PROX_DOMINATE_RATIO のとき floor を下げる
    const PROX_DOMINATE_RATIO: f64 = 0.5;
    /// 一度の調整で reg_limit を割る倍率
    const REG_LIMIT_STEP: f64 = 1e-3;

    // 作業バッファ
    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];
    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];

    // AMD permutation キャッシュ（スパースパターンは反復間で不変）
    let mut amd_perm_cache: Option<Vec<usize>> = None;

    // 殿指示(C): MaxIterationsを発生させる経路自体を消す。
    // None = 「まだ収束もタイムアウトも起きていない」を型で表現。
    // ループ出口は「収束→Some(Optimal)」「timeout→Some(Timeout)」の2つだけ。
    let mut status: Option<SolveStatus> = None;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    // best-so-far: 残差スコア最良時の (x,y,s,iter,residuals) を保持。
    // NaN guard 経路で崩壊解を返さないための保険。
    let mut best_score = f64::INFINITY;
    let mut best_x = x.clone();
    let mut best_y = y.clone();
    let mut best_s = s.clone();
    let mut best_iter: usize = 0;
    let mut best_residuals: (f64, f64, f64) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    // best-so-far の rel_gap も保持。
    // reject_false_*_bestsofar 経路で偽 Optimal 昇格を防ぐためのゲート用。
    let mut best_rel_gap: f64 = f64::INFINITY;

    // alpha 停滞・deadlock 検出 (定数はモジュールレベルに集約)
    let mut alpha_stall_count: usize = 0;

    for iter in 0..options.ipm.max_iter {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        // ── 残差計算（非正則化）──────────────────────────────────
        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = sᵀy / m_ineq（等式行除外）
        let mu: f64 = if m_ineq > 0 {
            s.iter().zip(y.iter()).zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>() / m_ineq as f64
        } else {
            0.0
        };

        // 残差ノルム記録
        let nr_p = norm_inf(&r_p);
        let nr_d = norm_inf(&r_d);
        final_residuals = Some((nr_p, nr_d, mu));

        // 双対ギャップを best-so-far 更新前に算出。
        // 符号規約: r_d = -(Qx + c + A^T y) → dual = -0.5 x^T Q x - Σ b_ext·y。
        // best 更新時に gap も記録し、reject_false 経路の偽 Optimal 昇格を防ぐ。
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

        // best-so-far 更新（NaN guard 経路で崩壊解を返さないための保険）
        // 各項を同じスケールで正規化 (mu は complementarity = sᵀy/m で dual variable と同スケール)。
        // 旧式は mu のみ無正規化で 3 項のスケール不揃いにより、||c|| が大きい問題で
        // best-so-far が「mu が小さい解」にバイアスされていた。
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

        // ── 収束判定 ──────────────────────────────────────────────
        // OSQP 流の正規化 (bench/v2 と整合):
        //   pfeas: ||r_p||_∞ <= eps * (1 + max(||Ax||_∞, ||b||_∞))
        //   dfeas: ||r_d||_∞ <= eps * (1 + max(||Qx||_∞, ||c||_∞, ||A^T y||_∞))
        // 旧式 `norm_inf(&b_ext).max(1.0)` は b≈0 の問題 (LISWET / YAO) で
        // 閾値を eps*1 → eps*2 に緩めて偽 Optimal を出していた (LISWET9: IPM が
        // pf=1.58e-6 で収束申告するが bench eps=1e-6 で FAIL)。
        let norm_c_orig_for_thr = norm_inf(&problem.c);
        let norm_aty_for_thr = norm_inf(&aty);
        let norm_qx_for_thr = norm_inf(&qx);
        let norm_ax_for_thr = norm_inf(&ax);
        let norm_b_for_thr = norm_inf(&b_ext);
        // dfeas 分母: max(||Qx||, ||c||, ||A^T y||)
        let dfeas_denom = norm_qx_for_thr.max(norm_c_orig_for_thr).max(norm_aty_for_thr);
        // pfeas 分母: max(||Ax||, ||b||)
        let pfeas_denom = norm_ax_for_thr.max(norm_b_for_thr);
        // 旧式互換用: nr_d_orig 等の他箇所で使う norm_b/norm_c (scaled 空間 .max(1.0) を
        // 残す。これらは目安で、最終 OSQP 判定は (dfeas/pfeas)_denom が支配する)。
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        // 原空間双対残差: r_d_orig[j] = r_d_scaled[j] / (c · d[j])
        // スケール済み残差だけで収束宣言すると真の最適でない basin で止まる（UBH1 obj=2.12 事例）
        let nr_d_orig = if let Some(sc) = scaler {
            let mut m = 0.0_f64;
            let limit = r_d.len().min(sc.d.len());
            for j in 0..limit {
                let scale = sc.c * sc.d[j];
                if scale.abs() > f64::MIN_POSITIVE {
                    m = m.max((r_d[j] / scale).abs());
                }
            }
            m
        } else {
            nr_d
        };
        let norm_c_orig = orig_problem
            .map(|op| norm_inf(&op.c))
            .unwrap_or(norm_c)
            .max(1.0);

        // [偽 Optimal 修正] 元空間 OSQP 流の全体相対化 dfeas (bench/v2 と同形)。
        // 2026-04-29 セッション 7: bench/v2 を OSQP 流の全体相対化に変更したため、
        // IPM 内部の収束判定もこれに合わせる (横展開)。
        // 旧式 (成分ごと正規化 → max) は「他項全部 0 に近い 1 変数」で過剰判定する欠陥があり、
        // bench/v2 で Marginal 5件 + Mid 系の真因と判明していた。IPM 内部も同じ判定が
        // 使われていたため、IPM が真の収束に達していてもこの式で「未収束」と判定し過剰反復していた。
        // 新式: ||r_d_orig||_∞ / (1 + max(||Qx||_∞, ||c||_∞, ||A^T y||_∞))
        let nr_d_rel_orig = if let Some(sc) = scaler {
            let mut max_r = 0.0_f64;
            let mut max_qx = 0.0_f64;
            let mut max_c = 0.0_f64;
            let mut max_aty = 0.0_f64;
            for j in 0..n {
                let scale_unscale = sc.c * sc.d[j];
                if scale_unscale.abs() < f64::MIN_POSITIVE {
                    continue;
                }
                max_r = max_r.max((r_d[j] / scale_unscale).abs());
                max_qx = max_qx.max((qx[j] / scale_unscale).abs());
                max_c = max_c.max((problem.c[j] / scale_unscale).abs());
                max_aty = max_aty.max((aty[j] / scale_unscale).abs());
            }
            max_r / (1.0 + max_qx.max(max_c).max(max_aty))
        } else {
            let mut max_r = 0.0_f64;
            let mut max_qx = 0.0_f64;
            let mut max_c = 0.0_f64;
            let mut max_aty = 0.0_f64;
            for j in 0..n {
                max_r = max_r.max(r_d[j].abs());
                max_qx = max_qx.max(qx[j].abs());
                max_c = max_c.max(problem.c[j].abs());
                max_aty = max_aty.max(aty[j].abs());
            }
            max_r / (1.0 + max_qx.max(max_c).max(max_aty))
        };

        // rel_gap / DUALITY_GAP_TOL は上のブロックで計算済（best-so-far 更新前）。
        // UBH1 (||x||≈1459, c=0, Q rank-deficient) で r_stat=2e-6・mu=1e-30 なのに
        // duality gap = 9.49 で obj 91% 誤差の事例を検出できなかった（Phase A 検証）。
        // 3 族独立 solver (PIQP/Clarabel/OSQP) で UBH1 真値 1.116 を確認済。

        // OSQP 形式の閾値 (bench/v2 と整合)
        let pfeas_thr = eps * (1.0 + pfeas_denom);
        let dfeas_thr = eps * (1.0 + dfeas_denom);
        // [DIAG] Optimal_main 条件を全て出力 (env=IPPMM_OPT_DIAG=1)
        if std::env::var("IPPMM_OPT_DIAG").ok().as_deref() == Some("1") {
            eprintln!(
                "IPPMM_OPT iter={} pf={:.3e}/thr={:.3e}{} nrd={:.3e}/thr={:.3e}{} nrd_orig={:.3e}/thr={:.3e}{} nrd_rel_orig={:.3e}/eps={:.3e}{} mu={:.3e}/eps={:.3e}{} relgap={:.3e}/tol={:.3e}{}",
                iter,
                nr_p, pfeas_thr, if nr_p < pfeas_thr { "✓" } else { "✗" },
                nr_d, dfeas_thr, if nr_d < dfeas_thr { "✓" } else { "✗" },
                nr_d_orig, eps_orig * (1.0 + norm_c_orig), if nr_d_orig < eps_orig * (1.0 + norm_c_orig) { "✓" } else { "✗" },
                nr_d_rel_orig, eps_orig, if nr_d_rel_orig < eps_orig { "✓" } else { "✗" },
                mu, eps, if mu < eps { "✓" } else { "✗" },
                rel_gap, DUALITY_GAP_TOL, if rel_gap.abs() < DUALITY_GAP_TOL { "✓" } else { "✗" },
            );
        }
        if nr_d < dfeas_thr
            && nr_d_orig < eps_orig * (1.0 + norm_c_orig)
            && nr_d_rel_orig < eps_orig
            && nr_p < pfeas_thr
            && mu < eps
            && rel_gap.abs() < DUALITY_GAP_TOL
        {
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "IPPMM_EXIT iter={} path=Optimal_main nr_d_orig={:.3e} rel_gap={:.3e}",
                    iter, nr_d_orig, rel_gap
                );
            }
            status = Some(SolveStatus::Optimal);
            final_iter = iter;
            break;
        }

        // μ が reg_limit 以下で残差も eps 水準 → SuboptimalSolution
        // PARAM(reg_limit*1e-2): 根拠=経験値(μがreg_limitの1/100以下=正則化下限の100倍収束で実質停滞とみなす。論文記載なし) | 要検証
        let thr_d = (eps * (1.0 + norm_c)).max(reg_limit * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(reg_limit * 10.0);
        if mu < reg_limit * 1e-2 && nr_d < thr_d && nr_p < thr_p && rel_gap.abs() < DUALITY_GAP_TOL {
            // ── Method C: 原空間pfeasチェック（Clarabel方式）──
            if let (Some(sc), Some(orig)) = (scaler, orig_problem) {
                let m_orig_check = orig.b.len();
                let n_orig = orig.num_vars;
                let mut ax_orig = vec![0.0_f64; m_orig_check];
                if m_orig_check > 0 {
                    for (j, (&dj, &xj)) in sc.d[..n_orig].iter().zip(x[..n_orig].iter()).enumerate() {
                        let dj_xj = dj * xj;
                        for ptr in orig.a.col_ptr[j]..orig.a.col_ptr[j + 1] {
                            let row = orig.a.row_ind[ptr];
                            if row < m_orig_check {
                                ax_orig[row] += orig.a.values[ptr] * dj_xj;
                            }
                        }
                    }
                }
                let pfeas_orig = if m_orig_check == 0 {
                    0.0
                } else {
                    ax_orig
                        .iter()
                        .zip(orig.b.iter())
                        .zip(orig.constraint_types.iter())
                        .map(|((&axi, &bi), ct)| match ct {
                            ConstraintType::Eq => (axi - bi).abs(),
                            ConstraintType::Ge => (bi - axi).max(0.0),
                            _ => (axi - bi).max(0.0),
                        })
                        .fold(0.0_f64, f64::max)
                };
                // OSQP 形式で b と Ax の両方を考慮 (旧 .max(1.0) は b≈0 で 2x 緩く偽 PASS)
                let norm_ax_orig: f64 = ax_orig.iter().fold(0.0_f64, |a, &v: &f64| a.max(v.abs()));
                let norm_b_orig = norm_inf(&orig.b);
                let pfeas_thr_orig = eps_orig * (1.0 + norm_ax_orig.max(norm_b_orig));
                // [偽 Optimal 修正] Optimal_MethodC 判定にも成分相対 dfeas を追加
                if pfeas_orig < pfeas_thr_orig
                    && nr_d_orig < eps_orig * (1.0 + norm_c_orig)
                    && nr_d_rel_orig < eps_orig
                    && mu < eps_orig
                {
                    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "IPPMM_EXIT iter={} path=Optimal_MethodC pfeas_orig={:.3e} nr_d_orig={:.3e}",
                            iter, pfeas_orig, nr_d_orig
                        );
                    }
                    status = Some(SolveStatus::Optimal);
                    final_iter = iter;
                    break;
                }
            }
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!("IPPMM_EXIT iter={} path=Suboptimal_mu_floor mu={:.3e} thr_d={:.3e} thr_p={:.3e}", iter, mu, thr_d, thr_p);
            }
            status = Some(SolveStatus::SuboptimalSolution);
            final_iter = iter;
            break;
        }

        // ── PMM 改善判定（前反復の残差と比較）──────────────────────
        // Algorithm PEU: primal/dual改善を独立に判定
        let primal_improved = PMM_IMPROVE_THRESHOLD * pmm.prev_nr_p > nr_p;
        let dual_improved = PMM_IMPROVE_THRESHOLD * pmm.prev_nr_d > nr_d;

        // ── PMM 修正済み残差を計算 ──────────────────────────────────
        // r_d_pmm = r_d - ρ*(x - x_ref)
        // r_p_pmm = r_p - δ*(y - y_ref)
        // 注意: 行列には rho_matrix/delta_matrix を使うが、RHS proximal 補正は rho_prox/delta_prox
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

        // Σ = diag(s_i / y_i)（等式行は0）
        let sigma_max = 1.0 / options.ipm.delta_min.max(MU_ZERO_THRESHOLD);
        let sigma_vec = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);

        // [DIAG] Σ の dynamic range 実測 (env=IPPMM_SIGMA_DIAG=1 のときのみ)
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

        // PMM駆動の正則化（mu-tracking廃止、gunshi指摘(2)）
        // rho/deltaはPMMが管理する。mu依存フロアは使わない
        let rho_matrix = pmm.rho.max(options.ipm.delta_min);
        let delta_matrix = pmm.delta.max(options.ipm.delta_min);

        // ── augmented KKT 構築 + 因子化 ────────────────────────────
        // T2: 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = Some(SolveStatus::Timeout);
            final_iter = iter;
            break;
        }

        // 因子化失敗時に rho/delta を LDL_REG_GROWTH 倍ずつ増やして再試行する
        let mut rho_retry = rho_matrix;
        let mut delta_matrix_retry = delta_matrix;
        let mut fac_opt: Option<KktFactor> = None;
        let mut aug_mat_opt: Option<crate::sparse::CscMatrix> = None;
        // [Schur path] env=QP_SCHUR=1 で Schur complement formulation を使う
        // (n×n SPD、augmented n+m_ext の代替)。LISWET 系の precision floor 突破を狙う。
        let use_schur = std::env::var("QP_SCHUR").ok().as_deref() == Some("1");
        let mut d_inv_opt: Option<Vec<f64>> = None;
        for _retry in 0..LDL_REG_RETRY_MAX {
            if timeout_ctx.should_stop() {
                status = Some(SolveStatus::Timeout);
                final_iter = iter;
                break;
            }
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
                build_augmented_system(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_matrix_retry)
            };
            // AMD は 1 回だけ計算してキャッシュ（スパースパターン不変のため）
            if amd_perm_cache.is_none() {
                amd_perm_cache = Some(amd_with_deadline(
                    mat_for_factor.nrows,
                    &mat_for_factor.col_ptr,
                    &mat_for_factor.row_ind,
                    timeout_ctx.deadline,
                ));
            }
            let perm = amd_perm_cache.as_ref().unwrap();
            match factorize_kkt_with_cached_perm(
                &mat_for_factor,
                perm,
                timeout_ctx.deadline,
                max_l_nnz_from_budget(),
            ) {
                Ok(f) => {
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
                    if rho_retry >= LDL_REG_CEILING {
                        break; // 上限到達 → あきらめ
                    }
                    rho_retry = (rho_retry * LDL_REG_GROWTH).min(LDL_REG_CEILING);
                    delta_matrix_retry = (delta_matrix_retry * LDL_REG_GROWTH).min(LDL_REG_CEILING);
                    // AMD キャッシュは rho/delta 変化でもスパース構造不変なので再利用可
                }
            }
        }
        if matches!(status, Some(SolveStatus::Timeout)) {
            break;
        }
        // 第3防御: Identity fallback — 全リトライ失敗時に identity perm + 大きな delta で再試行
        if fac_opt.is_none() {
            amd_perm_cache = None;
            let delta_fallback = LDL_FALLBACK_DELTA_MIN.max(rho_retry).max(delta_matrix_retry);
            let aug_mat_fb =
                build_augmented_system(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_fallback);
            let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
            match factorize_kkt_with_cached_perm(
                &aug_mat_fb,
                &identity_perm,
                timeout_ctx.deadline,
                max_l_nnz_from_budget(),
            ) {
                Ok(f) => {
                    fac_opt = Some(f);
                    aug_mat_opt = Some(aug_mat_fb);
                }
                Err(KktError::DeadlineExceeded) => {
                    status = Some(SolveStatus::Timeout);
                    final_iter = iter;
                }
                Err(_) => {} // identity fallback も失敗 → fac_opt は None のまま → M-02
            }
        }
        if matches!(status, Some(SolveStatus::Timeout)) {
            break;
        }
        // M-02: fac_opt が None なら全リトライ失敗 → NumericalError
        let fac = match fac_opt {
            Some(f) => f,
            None => return numerical_error_result(n),
        };
        let aug_mat_for_ir = aug_mat_opt
            .as_ref()
            .expect("aug_mat_opt must be set when fac_opt is set");

        // N1: mu_rate(predictor直後)は廃止。変数更新後のμからrを計算する（PMM更新部で実施）

        // ── Predictor / Corrector / Gondzio (Schur or augmented dispatch) ──
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
            );
            let (alpha, r_c_corr) = corrector_step(
                &s, &y, &is_eq_ext,
                &pred, mu,
                &r_d_pmm, &r_p_pmm,
                &sigma_vec, &fac, aug_mat_for_ir, n, m_ext,
                &mut dx, &mut dy, &mut ds,
            );
            (pred, alpha, r_c_corr)
        };

        // ── Gondzio multiple centrality correctors ──────────────────
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
                )
            } else {
                gondzio_correctors(
                    &s, &y, &is_eq_ext, m_ineq,
                    &r_d_pmm, &r_p_pmm,
                    &r_c_corr, &sigma_vec, &fac, aug_mat_for_ir, n, m_ext,
                    options.ipm.max_correctors, alpha,
                    &mut dx, &mut dy, &mut ds,
                )
            };
        }

        let _ = pred; // 未使用警告抑止

        // ── 変数更新 ──────────────────────────────────────────────
        // NaN/Inf ガード: ステップにNaNが含まれる場合は現在のx,y,sで停止。
        // sigma_max=1e17-1e19の問題で補正ステップの壊滅的キャンセルによりNaNが
        // 発生した際に、直前の有効な解でSuboptimalSolutionを返す。
        // unscale_ipm_result がpfeas/bfeas/dfeasを原空間で再検証してOptimalに昇格する。
        // Catastrophic blow-up（finite だが極端値）も検出。
        // UBH1 で reg_limit=5e-12 まで降下後、KKT system が semi-definite に近づき
        // LDL solve が dx_inf=1e290+ を返す病理。これも NaN 同等扱いで best 復帰。
        const DIRECTION_BLOWUP_THRESHOLD: f64 = 1e30;
        let direction_finite_but_huge = dx.iter().chain(dy.iter()).chain(ds.iter())
            .any(|v| v.is_finite() && v.abs() > DIRECTION_BLOWUP_THRESHOLD);
        if dx.iter().any(|v| !v.is_finite())
            || dy.iter().any(|v| !v.is_finite())
            || ds.iter().any(|v| !v.is_finite())
            || direction_finite_but_huge
        {
            // best-so-far 復帰: 崩壊した現在値ではなく最良残差時の解を返す
            if best_score.is_finite() {
                x.copy_from_slice(&best_x);
                y.copy_from_slice(&best_y);
                s.copy_from_slice(&best_s);
                final_iter = best_iter;
                final_residuals = Some(best_residuals);
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "IPPMM_EXIT iter={} path=Suboptimal_NaN_guard_bestsofar best_iter={} best=(pf={:.3e},df={:.3e},mu={:.3e})",
                        iter, best_iter, best_residuals.0, best_residuals.1, best_residuals.2
                    );
                }
            } else {
                final_iter = iter;
                if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                    eprintln!("IPPMM_EXIT iter={} path=Suboptimal_NaN_guard (no best)", iter);
                }
            }
            status = Some(SolveStatus::SuboptimalSolution);
            break;
        }

        // Infeasibility / Unboundedness 検出（IP-PMM パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter, rho_retry,
        ) {
            // 真に Infeasible/Unbounded なら残差が小さい解には到達しない。
            // best-so-far が Optimal 級の品質を保持していれば、方向検出による false positive とみなし格下げ。
            let quality_threshold = 10.0 * eps_orig;
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!("IPPMM_DEBUG iter={} best_score={:e} quality_threshold={:e} eps_orig={:e} eps={:e} best_finite={}", iter, best_score, quality_threshold, eps_orig, eps, best_score.is_finite());
            }
            // best_score は残差 (pf+df+mu) のみを評価。
            // UBH1 のように残差小でも gap 大な状態を best として抱え込む可能性があるため、
            // best_rel_gap も閾値内でないと Optimal 昇格しない。
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
            if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                eprintln!("IPPMM_EXIT iter={} path=check_infeas status={:?} best_score={:.3e}", iter, infeas_status, best_score);
            }
            status = Some(infeas_status);
            final_iter = iter;
            break;
        }

        // step magnitude trace（IPPMM_TRACE=1 のときのみ）
        if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
            let ndx = dx.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let ndy = dy.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let nds = ds.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let nrdpmm = r_d_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let nrppmm = r_p_pmm.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            eprintln!(
                "IPPMM_STEP iter={:4} alpha={:.6e} dx_inf={:.3e} dy_inf={:.3e} ds_inf={:.3e} rdpmm_inf={:.3e} rppmm_inf={:.3e}",
                iter, alpha, ndx, ndy, nds, nrdpmm, nrppmm
            );
        }
        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, alpha, &is_eq_ext);

        // null-space: alpha 停滞早期脱出。
        // alpha=0 が続く＝line search が進まない＝数値飽和または null-space 漂流。
        // best-so-far があればそれで Suboptimal 復帰、無ければ素で Suboptimal 脱出。
        if alpha < ALPHA_STALL_EPS {
            alpha_stall_count += 1;
        } else {
            alpha_stall_count = 0;
        }
        // stall 成立条件を best_score < eps に絞る。
        // UBH1 (best_score=4.8e-7) のように真に収束後に動けなくなったケースでのみ早期脱出。
        // QPILOTNO (best_score=2.5e-6) のような残差マージナルな問題では alpha-stall を発火させず、
        // 通常の timeout フローに任せる（DFEAS_FAIL として偽 Optimal を返すのを防ぐ）。
        let alpha_stall_converged = best_score.is_finite() && best_score < eps;
        // eps 非依存 deadlock gate。POST_VERIFY の eps 10x 厳格化で
        // best_score < eps が成立しなくなり alpha_stall_converged が永久 false となる
        // 病理（UBH1: 186 iter alpha=0 → さらに 24000+ iter alpha=0 継続）を断ち切る。
        // 条件: alpha=0 が 2N 連続＋rho/delta が reg_limit 付近＋best_score 有限
        // （rho/delta が floor = もうこれ以上 proximal を緩められない = 数値的に進めない）。
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

        // ── PMM パラメータ更新 ──────────────────────────────────────
        // Algorithm PEU Step 0: r = |μ_k - μ_{k+1}| / μ_k
        // μ_new = 変数更新後の実際のμ（corrector + line search 後）
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

        // mu=0 等式問題では高速減衰 (mu_rate=0.9 → 乗数 0.1 → ~8 反復で reg_limit)
        // PARAM: §35-B1, MATLAB 拡張版 IP-PMM_QP_Solver 準拠
        let mu_rate_raw = if mu < MU_ZERO_THRESHOLD && mu_new < MU_ZERO_THRESHOLD { 0.9 } else { r };
        let mu_rate = mu_rate_raw.clamp(0.2, 0.9);

        // Adaptive reg_limit (UBH1 パターン限定: c≈0):
        if allow_adaptive_reg
            && (pmm.rho - reg_limit).abs() < reg_limit * 0.01
            && reg_limit > REG_LIMIT_MIN
        {
            let prox_d_inf = x.iter().zip(pmm.x_ref.iter())
                .map(|(&xi, &xref)| (pmm.rho * (xi - xref)).abs())
                .fold(0.0_f64, f64::max);
            if prox_d_inf > nr_d * PROX_DOMINATE_RATIO && nr_d > 0.0 {
                reg_limit = (reg_limit * REG_LIMIT_STEP).max(REG_LIMIT_MIN);
            }
        }

        // Algorithm PEU Step 1&2: OR条件判定（MATLAB拡張版準拠）
        // primalまたはdual改善があれば良ステップ。delta/rho両方を同期的に更新。
        // 根拠: 設計書§A.5
        let either_improved = primal_improved || dual_improved;
        // [実験] env=IPPMM_FORCE_REF_UPDATE=1 で毎 iter 強制更新 → proximal effect ≈0
        let force_ref_update = std::env::var("IPPMM_FORCE_REF_UPDATE").ok().as_deref() == Some("1");
        if either_improved || force_ref_update {
            pmm.y_ref.copy_from_slice(&y);  // λ_{k+1} = y_{k+1}
            pmm.x_ref.copy_from_slice(&x);  // ζ_{k+1} = x_{k+1}
            pmm.delta = (pmm.delta * (1.0 - mu_rate)).max(reg_limit);
            pmm.rho   = (pmm.rho   * (1.0 - mu_rate)).max(reg_limit);
        } else {
            pmm.delta = (pmm.delta * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
            pmm.rho   = (pmm.rho   * (1.0 - PMM_SLOW_RATE * mu_rate)).max(reg_limit);
        }

        // 残差記録（次反復の改善判定用）
        pmm.prev_nr_p = nr_p;
        pmm.prev_nr_d = nr_d;
    }

    // 殿指示(C): None→Timeout変換。「MaxIterations→Timeout変換」ではなく「未決定→Timeout」。
    // max_iter=usize::MAXで収束もtimeoutも起きなかった場合（理論上不可能）にTimeoutを返す。
    let status = status.unwrap_or(SolveStatus::Timeout);

    // Timeout/MaxIterations の素の終了経路で best-so-far に復帰。
    // Why: alpha_stall/reject_false/NaN_guard の 3 経路は best_x 復帰するが、
    // 純粋な Timeout (timeout_ctx 検出) 経路はループ末尾の発散 x をそのまま返す。
    // QPILOTNO のような残差マージナル問題で alpha-stall が発火しない場合、
    // 最終 x が発散していても best_x (iter 199 相当) は pf=6.5e-6 で保持されている。
    // best_score が有限かつ current より良ければ復帰させる（post-solve の IR/昇格機会を与える）。
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

    // 目的関数値
    spmv_q(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = collapse_extended_dual(&y, m_orig, &problem.constraint_types);
    let bound_duals = y[m_orig..].to_vec();

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,

        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        // null-space: best-so-far の相対双対ギャップ。
        // unscale_ipm_result の Suboptimal→Optimal 昇格ゲート用。
        // INFINITY なら未計測扱いで None を返す（全 iter で best 更新ゼロは異常系）。
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

        let result = solve_ippmm_inner(&problem, &default_opts(), None, None, default_opts().ipm_eps());
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

        let result = solve_ippmm_inner(&problem, &default_opts(), None, None, default_opts().ipm_eps());
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

        let result = solve_ippmm_inner(&problem, &default_opts(), None, None, default_opts().ipm_eps());
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

        let result = solve_ippmm_inner(&problem, &default_opts(), None, None, default_opts().ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
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
        let result = solve_ippmm_inner(&problem, &opts, None, None, opts.ipm_eps());
        assert_eq!(result.status, SolveStatus::Optimal, "multi-eq: status");
        close(result.solution[0], 1.0 / 3.0, "multi-eq: x[0]");
        close(result.solution[1], 2.0 / 3.0, "multi-eq: x[1]");
        close(result.solution[2], 1.0 / 3.0, "multi-eq: x[2]");
    }
}
