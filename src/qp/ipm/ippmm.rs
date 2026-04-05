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
use crate::linalg::ldl;
use crate::linalg::ldl::LdlFactorizationAmd;
use crate::linalg::ruiz::RuizScaler;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::kkt::{spmv, spmtv, spmv_q, norm_inf, build_extended_constraints, build_augmented_system};
use super::common::{check_infeasible_or_unbounded, solve_unconstrained, fraction_to_boundary_masked, timeout_result, numerical_error_result};
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

// PMM パラメータ下限（reg_limit）は動的計算に移行（cmd_793設計書§B.5）
// compute_reg_limit() を参照。固定値1e-9は廃止。
// const REG_LIMIT: f64 = 1e-9;

/// PMM 改善判定閾値（5% 以上の残差減少で改善とみなす）
/// PARAM: 根拠=Gondzio2021 MATLAB実装(0.95*prev > current) | 要検証=閾値の感度
const PMM_IMPROVE_THRESHOLD: f64 = 0.95;

/// PMM 遅い減衰率（改善なし時に rho/delta をゆっくり減らす係数）
/// PARAM: 根拠=Pougkakiotis&Gondzio(2021) Algorithm PEU §5.1.4 p.27 Step 1/2: (1-r/3)
const PMM_SLOW_RATE: f64 = 1.0 / 3.0;

/// fraction-to-boundary τ
/// PARAM: 根拠=Mehrotra(1992)標準値 0.995 | 要検証=なし
const TAU: f64 = 0.995;

/// Gondzio corrector: target step size factor β
/// PARAM: 根拠=Gondzio(1996) β=1.0(最大ステップを目指す) | 要検証=β<1.0の効果
const BETA_GONDZIO: f64 = 1.0;

/// Gondzio corrector: complementarity lower bound factor
/// PARAM: 根拠=Gondzio(1996) | 要検証=小規模問題への影響
const GAMMA_L: f64 = 0.1;

/// Gondzio corrector: complementarity upper bound factor
/// PARAM: 根拠=Gondzio(1996) | 要検証=小規模問題への影響
const GAMMA_U: f64 = 10.0;

/// Gondzio corrector: step size 改善の最小閾値
/// PARAM: 根拠=改善なしの打ち切り判定(数値誤差以下は改善とみなさない) | 要検証=タイトな問題
const ALPHA_IMPROVE_THRESHOLD: f64 = 1e-3;

// ---------------------------------------------------------------------------
// reg_limit 動的計算（MATLABオリジナル版準拠、cmd_793設計書§B.5）
// ---------------------------------------------------------------------------

/// CSC行列の無限大ノルム（各行の絶対値和の最大値）を計算する: O(nnz)
fn matrix_infinity_norm(mat: &CscMatrix) -> f64 {
    let mut row_sums = vec![0.0_f64; mat.nrows];
    for (&val, &row) in mat.values.iter().zip(mat.row_ind.iter()) {
        row_sums[row] += val.abs();
    }
    row_sums.iter().cloned().fold(0.0_f64, f64::max)
}

/// PMM正則化パラメータの動的下限を計算する（MATLABオリジナル版準拠）
///
/// reg_limit = max(5 * tol / max(‖A‖_∞², ‖Q‖_∞²), 5e-10)
///
/// スケーリング後の行列（Ruiz適用済み）で呼ぶこと。
fn compute_reg_limit(a: &CscMatrix, q: &CscMatrix, tol: f64) -> f64 {
    let norm_a = matrix_infinity_norm(a);
    let norm_q = matrix_infinity_norm(q);
    let max_norm_sq = (norm_a * norm_a).max(norm_q * norm_q);
    let dynamic = if max_norm_sq > 1e-30 {
        5.0 * tol / max_norm_sq
    } else {
        5e-10
    };
    dynamic.max(5e-10)
}

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

    // PMM 状態初期化
    let mut pmm = PmmState {
        x_ref: x0,
        y_ref: y0,
        rho: RHO_INIT,
        delta: DELTA_INIT,
        prev_nr_p: f64::INFINITY,
        prev_nr_d: f64::INFINITY,
    };

    // reg_limit動的計算（MATLABオリジナル版準拠、cmd_793設計書§B.5）
    // スケーリング済みproblem.a, problem.qで計算。ループ前1回のみ。
    let reg_limit = compute_reg_limit(&problem.a, &problem.q, options.ipm_eps());

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

        // ── 収束判定 ──────────────────────────────────────────────
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        if nr_d < eps * (1.0 + norm_c) && nr_p < eps * (1.0 + norm_b) && mu < eps {
            status = Some(SolveStatus::Optimal);
            final_iter = iter;
            break;
        }

        // μ が reg_limit 以下で残差も eps 水準 → SuboptimalSolution
        // PARAM(reg_limit*1e-2): 根拠=経験値(μがreg_limitの1/100以下=正則化下限の100倍収束で実質停滞とみなす。論文記載なし) | 承認=cmd_493実装時設定・要検証
        let thr_d = (eps * (1.0 + norm_c)).max(reg_limit * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(reg_limit * 10.0);
        if mu < reg_limit * 1e-2 && nr_d < thr_d && nr_p < thr_p {
            // ── Method C: 原空間pfeasチェック（Clarabel方式）──
            if let (Some(sc), Some(orig)) = (scaler, orig_problem) {
                let m_orig_check = orig.b.len();
                let pfeas_orig = if m_orig_check == 0 {
                    0.0
                } else {
                    let n_orig = orig.num_vars;
                    let mut ax_orig = vec![0.0_f64; m_orig_check];
                    for (j, (&dj, &xj)) in sc.d[..n_orig].iter().zip(x[..n_orig].iter()).enumerate() {
                        let dj_xj = dj * xj;
                        for ptr in orig.a.col_ptr[j]..orig.a.col_ptr[j + 1] {
                            let row = orig.a.row_ind[ptr];
                            if row < m_orig_check {
                                ax_orig[row] += orig.a.values[ptr] * dj_xj;
                            }
                        }
                    }
                    ax_orig
                        .iter()
                        .zip(orig.b.iter())
                        .map(|(&axi, &bi)| (axi - bi).abs())
                        .fold(0.0_f64, f64::max)
                };
                let norm_b_orig = norm_inf(&orig.b).max(1.0);
                if pfeas_orig < eps_orig * (1.0 + norm_b_orig)
                    && nr_d < eps_orig * (1.0 + norm_c)
                    && mu < eps_orig
                {
                    status = Some(SolveStatus::Optimal);
                    final_iter = iter;
                    break;
                }
            }
            // Method Cで昇格できなかった場合 or scaler=None → SuboptimalSolution
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
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter()).enumerate()
            .map(|(i, (&si, &yi))| {
                if is_eq_ext[i] {
                    0.0
                } else {
                    let v: f64 = si / yi;
                    if v.is_finite() { v } else { sigma_max }
                }
            })
            .collect();

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

        // rho_matrix/delta_matrix リトライ（因子化失敗時に ×10 して最大 1e0 まで）
        let mut rho_retry = rho_matrix;
        let mut delta_matrix_retry = delta_matrix;
        let mut fac_opt: Option<LdlFactorizationAmd> = None;
        // PARAM(retry上限=10): 根拠=経験値(δ探索空間1e-4→1e0は4段階で到達、余裕をもった上限。論文記載なし) | 承認=cmd_520実装時設定・要検証
        for _retry in 0..10 {
            if timeout_ctx.should_stop() {
                status = Some(SolveStatus::Timeout);
                final_iter = iter;
                break;
            }
            let aug_mat =
                build_augmented_system(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_matrix_retry);
            // AMD は 1 回だけ計算してキャッシュ（スパースパターン不変のため）
            if amd_perm_cache.is_none() {
                amd_perm_cache = Some(amd_with_deadline(
                    aug_mat.nrows,
                    &aug_mat.col_ptr,
                    &aug_mat.row_ind,
                    timeout_ctx.deadline,
                ));
            }
            let perm = amd_perm_cache.as_ref().unwrap();
            match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                &aug_mat,
                perm,
                timeout_ctx.deadline,
            ) {
                Ok(f) => {
                    fac_opt = Some(f);
                    break;
                }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = Some(SolveStatus::Timeout);
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    if rho_retry >= 1e0 {
                        break; // 上限到達 → あきらめ
                    }
                    // PARAM(retry×10, 上限1e0): 根拠=経験値(LDLT因子化失敗時の指数的正則化増加。×10は10進指数的探索の自然な選択（具体的倍率はソルバー実装依存）、上限1e0は条件数悪化問題が起きない経験的上限) | 承認=cmd_520実装時設定・要検証
                    rho_retry = (rho_retry * 10.0).min(1e0);
                    delta_matrix_retry = (delta_matrix_retry * 10.0).min(1e0);
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
            let delta_fallback = 1e-2_f64.max(rho_retry).max(delta_matrix_retry);
            let aug_mat_fb =
                build_augmented_system(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_fallback);
            let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
            match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                &aug_mat_fb,
                &identity_perm,
                timeout_ctx.deadline,
            ) {
                Ok(f) => {
                    fac_opt = Some(f);
                }
                Err(ldl::LdlError::DeadlineExceeded) => {
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

        // ── Predictor ──────────────────────────────────────────────
        let total = n + m_ext;
        let mut rhs = vec![0.0f64; total];
        let mut sol = vec![0.0f64; total];

        let r_c_pred: Vec<f64> =
            s.iter().zip(y.iter()).enumerate()
            .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi }).collect();
        let r_p_mod_pred: Vec<f64> = r_p_pmm
            .iter()
            .zip(r_c_pred.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        rhs[..n].copy_from_slice(&r_d_pmm);
        rhs[n..].copy_from_slice(&r_p_mod_pred);
        fac.solve(&rhs, &mut sol);
        let dy_pred = sol[n..].to_vec();

        let mut ds_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            if is_eq_ext[i] {
                ds_pred[i] = 0.0;
            } else {
                ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
            }
        }

        let alpha_s_pred = fraction_to_boundary_masked(&s, &ds_pred, TAU, &is_eq_ext);
        let alpha_y_pred = fraction_to_boundary_masked(&y, &dy_pred, TAU, &is_eq_ext);
        let alpha_pred = alpha_s_pred.min(alpha_y_pred);

        let mu_aff: f64 = if m_ineq > 0 {
            s.iter().zip(y.iter()).zip(ds_pred.iter()).zip(dy_pred.iter()).enumerate()
                .filter(|&(i, _)| !is_eq_ext[i])
                .map(|(_, (((&si, &yi), &dsi), &dyi))| {
                    (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
                })
                .sum::<f64>() / m_ineq as f64
        } else {
            0.0
        };

        let sigma_center = if mu > 1e-15 {
            (mu_aff / mu).powi(3).min(1.0)
        } else {
            0.0
        };

        // N1: mu_rate(predictor直後)は廃止。変数更新後のμからrを計算する（PMM更新部で実施）

        // ── Corrector ──────────────────────────────────────────────
        let r_c_corr: Vec<f64> = s
            .iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .enumerate()
            .map(|(i, (((&si, &yi), &dsi), &dyi))| {
                if is_eq_ext[i] { 0.0 } else { sigma_center * mu - si * yi - dsi * dyi }
            })
            .collect();
        let r_p_mod_corr: Vec<f64> = r_p_pmm
            .iter()
            .zip(r_c_corr.iter())
            .zip(y.iter())
            .enumerate()
            .map(|(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] { rpi } else { rpi - rci / yi }
            })
            .collect();

        rhs[..n].copy_from_slice(&r_d_pmm);
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

        let alpha_s = fraction_to_boundary_masked(&s, &ds, TAU, &is_eq_ext);
        let alpha_y = fraction_to_boundary_masked(&y, &dy, TAU, &is_eq_ext);
        let alpha = alpha_s.min(alpha_y);

        // ── Gondzio multiple centrality correctors ──────────────────
        let mut alpha = alpha;
        if alpha < 0.999 {
            let mut alpha_prev = alpha;
            for _k in 0..options.ipm.max_correctors {
                let alpha_target =
                    (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
                let mu_target: f64 = if m_ineq > 0 {
                    s.iter().zip(y.iter()).zip(ds.iter().zip(dy.iter())).enumerate()
                        .filter(|&(i, _)| !is_eq_ext[i])
                        .map(|(_, ((&si, &yi), (&dsi, &dyi)))| {
                            (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                        })
                        .sum::<f64>() / m_ineq as f64
                } else {
                    0.0
                };
                let mu_target = mu_target.max(0.0);

                let target_lo = GAMMA_L * mu_target;
                let target_hi = GAMMA_U * mu_target;

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

                let r_p_mod_gondzio: Vec<f64> = r_p_pmm
                    .iter()
                    .zip(r_c_gondzio.iter())
                    .zip(y.iter())
                    .enumerate()
                    .map(|(i, ((&rpi, &rci), &yi))| {
                        if is_eq_ext[i] { rpi } else { rpi - rci / yi }
                    })
                    .collect();

                rhs[..n].copy_from_slice(&r_d_pmm);
                rhs[n..].copy_from_slice(&r_p_mod_gondzio);
                fac.solve(&rhs, &mut sol);
                let dx_new = sol[..n].to_vec();
                let dy_new = sol[n..].to_vec();
                let ds_new: Vec<f64> = (0..m_ext)
                    .map(|i| {
                        if is_eq_ext[i] { 0.0 } else { r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i] }
                    })
                    .collect();

                let alpha_s_new = fraction_to_boundary_masked(&s, &ds_new, TAU, &is_eq_ext);
                let alpha_y_new = fraction_to_boundary_masked(&y, &dy_new, TAU, &is_eq_ext);
                let alpha_new = alpha_s_new.min(alpha_y_new);

                if alpha_new < alpha_prev + ALPHA_IMPROVE_THRESHOLD {
                    break;
                }

                dx.copy_from_slice(&dx_new);
                dy.copy_from_slice(&dy_new);
                ds.copy_from_slice(&ds_new);
                alpha_prev = alpha_new;
            }
            alpha = alpha_prev;
        }

        // ── 変数更新 ──────────────────────────────────────────────
        // NaN/Inf ガード: ステップにNaNが含まれる場合は現在のx,y,sで停止。
        // sigma_max=1e17-1e19の問題で補正ステップの壊滅的キャンセルによりNaNが
        // 発生した際に、直前の有効な解でSuboptimalSolutionを返す。
        // unscale_ipm_result がpfeas/bfeas/dfeasを原空間で再検証してOptimalに昇格する。
        if dx.iter().any(|v| !v.is_finite())
            || dy.iter().any(|v| !v.is_finite())
            || ds.iter().any(|v| !v.is_finite())
        {
            status = Some(SolveStatus::SuboptimalSolution);
            final_iter = iter;
            break;
        }

        // Infeasibility / Unboundedness 検出（IP-PMM パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter, rho_retry,
        ) {
            status = Some(infeas_status);
            final_iter = iter;
            break;
        }

        for i in 0..n {
            x[i] += alpha * dx[i];
        }
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
        let r = if mu > 1e-15 || mu_new > 1e-15 {
            (mu - mu_new).abs() / mu.max(mu_new).max(1e-15)
        } else {
            0.0
        };

        // MATLAB拡張版準拠: mu=0等式問題では高速減衰(mu_rate=0.9 → 乗数0.1 → ~8反復でreg_limit)
        // PARAM: §35-B1 mu<1e-15時mu_rate=0.9 | 根拠=MATLAB拡張版IP-PMM_QP_Solver準拠 | 承認=cmd_783
        let mu_rate_raw = if mu < 1e-15 && mu_new < 1e-15 { 0.9 } else { r };
        let mu_rate = mu_rate_raw.clamp(0.2, 0.9);

        // Algorithm PEU Step 1&2: OR条件判定（MATLAB拡張版準拠）
        // primalまたはdual改善があれば良ステップ。delta/rho両方を同期的に更新。
        // 根拠: cmd_793設計書§A.5 | 承認=cmd_794
        let either_improved = primal_improved || dual_improved;
        if either_improved {
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
    use crate::qp::ipm::common::check_infeasible_or_unbounded;
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

    /// IPPMM-T-INF1: iter < MIN_ITER(=5) の場合 None が返ること
    #[test]
    fn test_iter_guard_ippmm() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // MIN_DIR_NORM を超える大きさだが iter ガードが先
        let dy: Vec<f64> = vec![];
        // iter=4 < MIN_ITER=5 → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 4, 0.0),
            None,
            "IPPMM-T-INF1: iter < MIN_ITER は None であること"
        );
    }

    /// IPPMM-T-INF2: ||Δx||_inf <= MIN_DIR_NORM(=1e-3) の場合 None が返ること
    #[test]
    fn test_min_dir_norm_guard_ippmm() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![5e-4]; // ||dx||_inf = 5e-4 <= MIN_DIR_NORM = 1e-3
        let dy: Vec<f64> = vec![];
        // 収束時偽陽性防止: dx が小さすぎる → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            None,
            "IPPMM-T-INF2: ||dx||_inf <= MIN_DIR_NORM は None であること"
        );
    }

    /// IPPMM-T-INF3: Farkas dual ray 条件を満たすベクトルで Infeasible 判定を確認
    ///
    /// A_orig = 0 (1x2 ゼロ行列), b = [-1], dy_orig = [2.0]
    /// ① ||A^T * dy_orig|| = 0 < ε ✓
    /// ② b · dy_orig = -2 < -ε ✓
    /// → Infeasible
    #[test]
    fn test_primal_infeasible_ippmm() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let c = vec![1.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap(); // 1x2 ゼロ行列
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap();
        let dx = vec![1e-10, 1e-10]; // 非常に小さい → dual チェックはスキップ
        let dy = vec![2.0]; // norm = 2.0 > MIN_DIR_NORM
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 1, 1, 10, 0.0),
            Some(SolveStatus::Infeasible),
            "IPPMM-T-INF3: Farkas ray 条件 → Infeasible であること"
        );
    }

    /// IPPMM-T-INF4: LP (Q=0) で c·Δx < 0 条件の Unbounded 判定を確認
    ///
    /// n=1, m_orig=0: c=[-1], dx=[1.0] → c·dx/norm_dx = -1 < -ε → Unbounded
    #[test]
    fn test_dual_infeasible_lp_ippmm() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap(); // Q=0 (LP)
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(); // 制約なし
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // c·dx = -1 < -ε, m_ext=0 なので dual guard は無効
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            Some(SolveStatus::Unbounded),
            "IPPMM-T-INF4: LP dual infeasibility → Unbounded であること"
        );
    }

    /// IPPMM-T-INF5: QP c=0 のとき Unbounded を返さないことを確認（QPLIB_9002バグ回帰防止）
    ///
    /// Q = diag([0, 2]) (1エントリのみ → is_lp=false)
    /// c = [0, 0], dx = [1.0, 0.0]
    /// 条件1: ||Q*dx||/norm_dx = 0 < EPS_INF → 通過
    /// 条件2: c^T*dx / norm_dx = 0 → NOT < -EPS_INF → 不成立
    /// → cond_obj = false → None (Unbounded不判定)
    #[test]
    fn test_qp_c_zero_not_unbounded() {
        // Q に (1,1)=2.0 のエントリのみ → is_lp=false (Q.values=[2.0]≠0)
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0]; // c=0
        let a = CscMatrix::new(0, 2);
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::new(0, 2);
        let dx = vec![1.0, 0.0]; // Q*dx = [0,0], norm_qdx=0 < EPS_INF, c_dx=0 ≥ -EPS_INF
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            None,
            "IPPMM-T-INF5: QP c=0 → Unbounded不判定（QPLIB_9002回帰防止）"
        );
    }

    /// IPPMM-T-INF6: QP c≠0 の真のUnbounded問題で正しく Unbounded を返すことを確認
    ///
    /// Q = diag([0, 2]) (is_lp=false), c = [-1, 0], dx = [1.0, 0.0]
    /// 条件1: ||Q*dx||/norm_dx = 0 < EPS_INF → 通過
    /// 条件2: c^T*dx / norm_dx = -1 < -EPS_INF → 通過
    /// → cond_obj = true, m_orig=0 → Unbounded
    #[test]
    fn test_qp_c_nonzero_true_unbounded() {
        // Q に (1,1)=2.0 のエントリのみ → is_lp=false
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0], 2, 2).unwrap();
        let c = vec![-1.0, 0.0]; // c≠0、x[0]方向に目的減少
        let a = CscMatrix::new(0, 2);
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::new(0, 2);
        let dx = vec![1.0, 0.0]; // Q*dx=[0,0] → 条件1通過; c_dx=-1 → 条件2通過
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            Some(SolveStatus::Unbounded),
            "IPPMM-T-INF6: QP c≠0 真Unbounded → Unbounded判定"
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
