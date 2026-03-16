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
//! PMM update rule (Gondzio MATLAB 参照実装より):
//!   improved = (0.95 * prev_nr_p > nr_p) || (0.95 * prev_nr_d > nr_d)
//!   if improved: x_ref = x, y_ref = y; ρ *= (1 - mu_rate); δ *= (1 - mu_rate)
//!   else:        ρ *= (1 - 0.666 * mu_rate); δ *= (1 - 0.666 * mu_rate)

use crate::linalg::amd::amd_with_deadline;
use crate::linalg::ldl;
use crate::linalg::ldl::LdlFactorizationAmd;
use crate::linalg::ruiz::RuizScaler;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

// ---------------------------------------------------------------------------
// PMM パラメータ定数（§35 PARAM マーカー）
// ---------------------------------------------------------------------------

/// PMM 初期 rho（primal proximal）
/// PARAM: 根拠=mu-tracking初期値と整合する程度の小値（8.0はGondzio2021参照実装だが
///        わしらの単一ループ実装には大きすぎてKKT解を狂わせる）
/// Ruizスケーリング後の単位スケール問題を前提とした値。
/// 非スケール問題（フォールバックパス）では条件数が増大する可能性あり。
/// augmented KKT κ≈1e8、LDLT安定範囲内
const RHO_INIT: f64 = 1e-4;

/// PMM 初期 delta（dual proximal）
/// PARAM: RHO_INITと対称に設定
/// Ruizスケーリング後の単位スケール問題を前提とした値。
/// 非スケール問題（フォールバックパス）では条件数が増大する可能性あり。
/// augmented KKT κ≈1e8、RHO_INITと対称
const DELTA_INIT: f64 = 1e-4;

/// PMM パラメータ下限（reg_limit）
/// PARAM: 根拠=数値安定性のための最小正則化値(0=完全収束) | 要検証=大規模問題での充足性
const REG_LIMIT: f64 = 1e-9;

/// PMM 改善判定閾値（5% 以上の残差減少で改善とみなす）
/// PARAM: 根拠=Gondzio2021 MATLAB実装(0.95*prev > current) | 要検証=閾値の感度
const PMM_IMPROVE_THRESHOLD: f64 = 0.95;

/// PMM 遅い減衰率（改善なし時に rho/delta をゆっくり減らす係数）
/// PARAM: 根拠=Gondzio2021 MATLAB実装(0.666 * mu_rate) | 要検証=収束速度への影響
const PMM_SLOW_RATE: f64 = 0.666;

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
        return solve_unconstrained_ippmm(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築（独自実装: kkt.rs 不使用）
    let (a_ext, b_ext, m_ext, m_orig) = build_extended_constraints_ippmm(problem);

    if m_ext == 0 {
        return solve_unconstrained_ippmm(problem, &timeout_ctx);
    }

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
    // 下限を 1.0 でクランプ（D1修正）。
    // ★ cmd_499教訓: max(1.0, |bi|+1.0) は bi依存項がQSHELL pfeasを劣化させた。
    //    max(1.0)固定（問題依存なし）が安全。Ruizスケーリング後は|bi|≈1.0のためclampは最小限。
    let mut ax0 = vec![0.0f64; m_ext];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax0[a_ext.row_ind[k]] += a_ext.values[k] * x0[col];
        }
    }
    let s0: Vec<f64> = b_ext
        .iter()
        .zip(ax0.iter())
        .map(|(&bi, &axi)| (bi - axi).max(1.0))
        .collect();
    let y0: Vec<f64> = vec![1.0; m_ext];

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

    let mut status = SolveStatus::MaxIterations;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;

    for iter in 0..options.ipm.max_iter {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // ── 残差計算（非正則化）──────────────────────────────────
        spmv_ippmm(&a_ext, &x, &mut ax);
        spmtv_ippmm(&a_ext, &y, &mut aty);
        spmv_q_ippmm(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = sᵀy / m_ext
        let mu: f64 = s.iter().zip(y.iter()).map(|(&si, &yi)| si * yi).sum::<f64>()
            / m_ext as f64;

        // 残差ノルム記録
        let nr_p = norm_inf_ippmm(&r_p);
        let nr_d = norm_inf_ippmm(&r_d);
        final_residuals = Some((nr_p, nr_d, mu));

        // ── 収束判定 ──────────────────────────────────────────────
        let norm_c = norm_inf_ippmm(&problem.c).max(1.0);
        let norm_b = norm_inf_ippmm(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        if nr_d < eps * (1.0 + norm_c) && nr_p < eps * (1.0 + norm_b) && mu < eps {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // μ が REG_LIMIT 以下で残差も eps 水準 → SuboptimalSolution
        let thr_d = (eps * (1.0 + norm_c)).max(REG_LIMIT * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(REG_LIMIT * 10.0);
        if mu < REG_LIMIT * 1e-2 && nr_d < thr_d && nr_p < thr_p {
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
                let norm_b_orig = norm_inf_ippmm(&orig.b).max(1.0);
                if pfeas_orig < eps_orig * (1.0 + norm_b_orig)
                    && nr_d < eps_orig * (1.0 + norm_c)
                    && mu < eps_orig
                {
                    status = SolveStatus::Optimal;
                    final_iter = iter;
                    break;
                }
            }
            // Method Cで昇格できなかった場合 or scaler=None → SuboptimalSolution
            status = SolveStatus::SuboptimalSolution;
            final_iter = iter;
            break;
        }

        // ── PMM 改善判定（前反復の残差と比較）──────────────────────
        let improved = (PMM_IMPROVE_THRESHOLD * pmm.prev_nr_p > nr_p)
            || (PMM_IMPROVE_THRESHOLD * pmm.prev_nr_d > nr_d);

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

        // Σ = diag(s_i / y_i)
        // sigma = s/y: Infになる場合（y→1e-12フロアかつsが大きい）は
        // 大きな有限値にクリップ。faerはInfをマトリクス値として処理できない。
        // 大きなsigmaは -(sigma+delta) >> 0 を保証し因子化はむしろ安定する。
        // SIGMA_MAX = 1/delta_min = 1e8 (delta_min=1e-8時)
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter())
            .map(|(&si, &yi)| {
                let v = si / yi;
                if v.is_finite() { v } else { sigma_max }
            })
            .collect();

        // PMM駆動の正則化（mu-tracking廃止、gunshi指摘(2)）
        // rho/deltaはPMMが管理する。mu依存フロアは使わない
        let rho_matrix = pmm.rho.max(options.ipm.delta_min);
        let delta_matrix = pmm.delta.max(options.ipm.delta_min);

        // ── augmented KKT 構築 + 因子化 ────────────────────────────
        // T2: 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // rho_matrix/delta_matrix リトライ（因子化失敗時に ×10 して最大 1e0 まで）
        let mut rho_retry = rho_matrix;
        let mut delta_matrix_retry = delta_matrix;
        let mut fac_opt: Option<LdlFactorizationAmd> = None;
        for _retry in 0..10 {
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                break;
            }
            let aug_mat =
                build_aug_ippmm(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_matrix_retry);
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
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    if rho_retry >= 1e0 {
                        break; // 上限到達 → あきらめ
                    }
                    rho_retry = (rho_retry * 10.0).min(1e0);
                    delta_matrix_retry = (delta_matrix_retry * 10.0).min(1e0);
                    // AMD キャッシュは rho/delta 変化でもスパース構造不変なので再利用可
                }
            }
        }
        if status == SolveStatus::Timeout {
            break;
        }
        // 第3防御: Identity fallback — 全リトライ失敗時に identity perm + 大きな delta で再試行
        if fac_opt.is_none() {
            amd_perm_cache = None;
            let delta_fallback = 1e-2_f64.max(rho_retry).max(delta_matrix_retry);
            let aug_mat_fb =
                build_aug_ippmm(&problem.q, &a_ext, &sigma_vec, rho_retry, delta_fallback);
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
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                }
                Err(_) => {} // identity fallback も失敗 → fac_opt は None のまま → M-02
            }
        }
        if status == SolveStatus::Timeout {
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
            s.iter().zip(y.iter()).map(|(&si, &yi)| -si * yi).collect();
        let r_p_mod_pred: Vec<f64> = r_p_pmm
            .iter()
            .zip(r_c_pred.iter())
            .zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
            .collect();

        rhs[..n].copy_from_slice(&r_d_pmm);
        rhs[n..].copy_from_slice(&r_p_mod_pred);
        fac.solve(&rhs, &mut sol);
        let dy_pred = sol[n..].to_vec();

        let mut ds_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }

        let alpha_s_pred = fraction_to_boundary_ippmm(&s, &ds_pred, TAU);
        let alpha_y_pred = fraction_to_boundary_ippmm(&y, &dy_pred, TAU);
        let alpha_pred = alpha_s_pred.min(alpha_y_pred);

        let mu_aff: f64 = s
            .iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| {
                (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
            })
            .sum::<f64>()
            / m_ext as f64;

        let sigma_center = if mu > 1e-15 {
            (mu_aff / mu).powi(3).min(1.0)
        } else {
            0.0
        };

        // PMM update で使う mu_rate（barrier 減少率の推定値）
        let mu_rate = if mu > 1e-15 {
            (mu_aff / mu).min(1.0).max(0.0)
        } else {
            0.0
        };

        // ── Corrector ──────────────────────────────────────────────
        let r_c_corr: Vec<f64> = s
            .iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| sigma_center * mu - si * yi - dsi * dyi)
            .collect();
        let r_p_mod_corr: Vec<f64> = r_p_pmm
            .iter()
            .zip(r_c_corr.iter())
            .zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
            .collect();

        rhs[..n].copy_from_slice(&r_d_pmm);
        rhs[n..].copy_from_slice(&r_p_mod_corr);
        fac.solve(&rhs, &mut sol);
        dx.copy_from_slice(&sol[..n]);
        dy.copy_from_slice(&sol[n..]);

        for i in 0..m_ext {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }

        let alpha_s = fraction_to_boundary_ippmm(&s, &ds, TAU);
        let alpha_y = fraction_to_boundary_ippmm(&y, &dy, TAU);
        let alpha = alpha_s.min(alpha_y);

        // ── Gondzio multiple centrality correctors ──────────────────
        let mut alpha = alpha;
        if alpha < 0.999 {
            let mut alpha_prev = alpha;
            for _k in 0..options.ipm.max_correctors {
                let alpha_target =
                    (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
                let mu_target: f64 = s
                    .iter()
                    .zip(y.iter())
                    .zip(ds.iter().zip(dy.iter()))
                    .map(|((&si, &yi), (&dsi, &dyi))| {
                        (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                    })
                    .sum::<f64>()
                    / m_ext as f64;
                let mu_target = mu_target.max(0.0);

                let target_lo = GAMMA_L * mu_target;
                let target_hi = GAMMA_U * mu_target;

                let mut r_c_gondzio = vec![0.0f64; m_ext];
                for i in 0..m_ext {
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
                    .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
                    .collect();

                rhs[..n].copy_from_slice(&r_d_pmm);
                rhs[n..].copy_from_slice(&r_p_mod_gondzio);
                fac.solve(&rhs, &mut sol);
                let dx_new = sol[..n].to_vec();
                let dy_new = sol[n..].to_vec();
                let ds_new: Vec<f64> = (0..m_ext)
                    .map(|i| r_c_gondzio[i] / y[i] - sigma_vec[i] * dy_new[i])
                    .collect();

                let alpha_s_new = fraction_to_boundary_ippmm(&s, &ds_new, TAU);
                let alpha_y_new = fraction_to_boundary_ippmm(&y, &dy_new, TAU);
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
            status = SolveStatus::SuboptimalSolution;
            final_iter = iter;
            break;
        }

        // Infeasibility / Unboundedness 検出（IP-PMM パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded_ippmm(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter,
        ) {
            status = infeas_status;
            final_iter = iter;
            break;
        }

        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_ext {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            // 下限: 負への転落を防ぐ（元の実装と同じ）
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }

        // ── PMM パラメータ更新 ──────────────────────────────────────
        // gunshi指摘(3): mu_rate=0時は固定倍率0.1で減衰（cycling防止）
        // mu_rate≈0の場合に rho が減らなくなる問題を防ぐ
        const PMM_MIN_DECAY: f64 = 0.1;
        let effective_rate = mu_rate.max(PMM_MIN_DECAY);

        // improved: 前反復の残差を参照（変数更新前の nr_p, nr_d と比較）
        if improved {
            // 新しい点を参照点として採用（変数更新後）
            pmm.x_ref.copy_from_slice(&x);
            pmm.y_ref.copy_from_slice(&y);
            // 積極的減衰: ρ *= (1 - effective_rate)
            pmm.rho = (pmm.rho * (1.0 - effective_rate)).max(REG_LIMIT);
            pmm.delta = (pmm.delta * (1.0 - effective_rate)).max(REG_LIMIT);
        } else {
            // 緩慢減衰: ρ *= (1 - 0.666 * effective_rate)
            pmm.rho = (pmm.rho * (1.0 - PMM_SLOW_RATE * effective_rate)).max(REG_LIMIT);
            pmm.delta = (pmm.delta * (1.0 - PMM_SLOW_RATE * effective_rate)).max(REG_LIMIT);
        }

        // 残差記録（次反復の改善判定用）
        pmm.prev_nr_p = nr_p;
        pmm.prev_nr_d = nr_d;
    }

    // 目的関数値
    spmv_q_ippmm(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = y[..m_orig].to_vec();
    let bound_duals = y[m_orig..].to_vec();

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,
        active_set: vec![],
        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 独自拡張制約構築（kkt.rs build_extended_constraints の独立実装）
// ---------------------------------------------------------------------------

/// 拡張制約行列を構築する（独立実装: kkt.rs 不使用）
///
/// 戻り値: (A_ext, b_ext, m_ext, m_orig)
fn build_extended_constraints_ippmm(
    problem: &QpProblem,
) -> (CscMatrix, Vec<f64>, usize, usize) {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let n_lb: usize = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub: usize = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    let m_ext = m + n_lb + n_ub;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut b_ext = Vec::with_capacity(m_ext);

    // 元の不等式制約
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            rows.push(problem.a.row_ind[k]);
            cols.push(col);
            vals.push(problem.a.values[k]);
        }
    }
    b_ext.extend_from_slice(&problem.b);

    // 下界制約: x_j >= lb_j → -x_j <= -lb_j
    let mut lb_row = m;
    for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            rows.push(lb_row);
            cols.push(j);
            vals.push(-1.0);
            b_ext.push(-lb);
            lb_row += 1;
        }
    }

    // 上界制約: x_j <= ub_j
    let mut ub_row = m + n_lb;
    for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
        if ub.is_finite() {
            rows.push(ub_row);
            cols.push(j);
            vals.push(1.0);
            b_ext.push(ub);
            ub_row += 1;
        }
    }

    let a_ext = if m_ext == 0 || rows.is_empty() {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap()
    };

    (a_ext, b_ext, m_ext, m)
}

// ---------------------------------------------------------------------------
// augmented KKT 構築（独立実装: kkt.rs 不使用）
// ---------------------------------------------------------------------------

/// IP-PMM augmented KKT の上三角 CSC を構築する
///
/// ```text
/// K = [(Q + ρI),  Aᵀ   ]   (ρ: primal proximal パラメータ)
///     [A,        -D    ]   (D = Σ + δI, δ: dual proximal パラメータ)
/// ```
///
/// kkt.rs の build_augmented_system に相当するが、ρ を Q 対角に加算する点が異なる。
#[allow(clippy::needless_range_loop)]
fn build_aug_ippmm(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    sigma_vec: &[f64],
    rho: f64,
    delta: f64,
) -> CscMatrix {
    let n = q.nrows;
    let m_ext = a_ext.nrows;
    let total = n + m_ext;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Part 1: Q + ρI（上三角のみ）
    let mut diag_added = vec![false; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k] + if row == col { rho } else { 0.0 };
                rows.push(row);
                cols.push(col);
                vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    // Q に対角がない変数には ρI を追加
    for i in 0..n {
        if !diag_added[i] {
            rows.push(i);
            cols.push(i);
            vals.push(rho);
        }
    }

    // Part 2: A_ext^T ブロック（右上、row < col 保証）
    for j in 0..n {
        for idx in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
            let k = a_ext.row_ind[idx];
            let v = a_ext.values[idx];
            rows.push(j);
            cols.push(n + k);
            vals.push(v);
        }
    }

    // Part 3: -(Σ + δ)I 対角ブロック（インデックス n..n+m_ext）
    for k in 0..m_ext {
        rows.push(n + k);
        cols.push(n + k);
        vals.push(-(sigma_vec[k] + delta));
    }

    if rows.is_empty() {
        CscMatrix::new(total, total)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, total, total).unwrap()
    }
}

// ---------------------------------------------------------------------------
// 制約なし QP（独立実装）
// ---------------------------------------------------------------------------

fn solve_unconstrained_ippmm(problem: &QpProblem, timeout_ctx: &TimeoutCtx) -> SolverResult {
    let n = problem.num_vars;

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if n == 0 {
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        };
    }

    // (Q + δI)x = -c を解く（δ: 数値安定性のための小さな正則化、PMM なし）
    // PARAM: 根拠=solve_unconstrained(step.rs)と同値(1e-7) | 要検証=なし
    let unc_delta: f64 = 1e-7;
    let mut triplet_rows: Vec<usize> = Vec::new();
    let mut triplet_cols: Vec<usize> = Vec::new();
    let mut triplet_vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];

    for col in 0..n {
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            let row = problem.q.row_ind[k];
            if row <= col {
                triplet_rows.push(row);
                triplet_cols.push(col);
                let v = problem.q.values[k] + if row == col { unc_delta } else { 0.0 };
                triplet_vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            triplet_rows.push(i);
            triplet_cols.push(i);
            triplet_vals.push(unc_delta);
        }
    }

    let q_reg =
        CscMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n, n).unwrap();

    match ldl::factorize(&q_reg) {
        Ok(fac) => {
            let rhs: Vec<f64> = problem.c.iter().map(|&ci| -ci).collect();
            let mut x = vec![0.0f64; n];
            fac.solve(&rhs, &mut x);

            let mut qx = vec![0.0f64; n];
            spmv_q_ippmm(&problem.q, &x, &mut qx);
            let objective = 0.5
                * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
                + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

            SolverResult {
                status: SolveStatus::Optimal,
                objective,
                solution: x,
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 1,
                ..Default::default()
            }
        }
        Err(_) => numerical_error_result(n),
    }
}

// ---------------------------------------------------------------------------
// 疎行列-ベクトル演算（独立実装: kkt.rs 不使用）
// ---------------------------------------------------------------------------

/// out = A * x（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
fn spmv_ippmm(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..a.ncols {
        let xv = x[col];
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            out[a.row_ind[k]] += a.values[k] * xv;
        }
    }
}

/// out = A^T * v（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
fn spmtv_ippmm(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    for col in 0..a.ncols {
        let mut s = 0.0;
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            s += a.values[k] * v[a.row_ind[k]];
        }
        out[col] = s;
    }
}

/// out = Q * x（全要素格納の対称 Q に対応）
#[inline]
#[allow(clippy::needless_range_loop)]
fn spmv_q_ippmm(q: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..q.ncols {
        let xv = x[col];
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            out[q.row_ind[k]] += q.values[k] * xv;
        }
    }
}

/// ||v||_∞
#[inline]
fn norm_inf_ippmm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// Infeasibility / Unboundedness 検出（IP-PMM パス）
// ---------------------------------------------------------------------------

/// Farkas 条件と双対不実行可能条件で Infeasible / Unbounded を判定する。
///
/// Farkas 条件と双対不実行可能条件を用いた Infeasible/Unbounded 判定（独立実装）。
/// ippmm.rs 内ヘルパーのみを使用（step.rs/kkt.rs 依存なし）。
///
/// # 引数
/// - `dx`, `dy`: Gondzio補正後の探索方向
/// - `m_orig`: 元の等式/不等式制約数（境界制約を除く）
/// - `m_ext`: 拡張制約数（境界スラック含む）
/// - `iter`: 現在の反復番号（MIN_ITER 未満はスキップ）
fn check_infeasible_or_unbounded_ippmm(
    dx: &[f64],
    dy: &[f64],
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_orig: usize,
    m_ext: usize,
    iter: usize,
) -> Option<SolveStatus> {
    const EPS_INF: f64 = 1e-8;
    const MIN_ITER: usize = 5;
    /// 収束時の偽陽性防止: 方向ベクトルが MIN_DIR_NORM 以下は検出スキップ。
    /// 収束時は Δx→0, Δy→0 なので norm=max(1,||Δ||)=1 となり比率が偶然ε未満になる。
    const MIN_DIR_NORM: f64 = 1e-3;

    if iter < MIN_ITER {
        return None;
    }

    let n = dx.len();

    // --- Primal Infeasibility check ---
    // ||Δy_orig|| が MIN_DIR_NORM より小さければスキップ（収束時偽陽性防止）。
    if m_orig > 0 {
        let dy_orig = &dy[..m_orig];
        let norm_dy_inf = norm_inf_ippmm(dy_orig);
        if norm_dy_inf > MIN_DIR_NORM {
            let norm_dy = norm_dy_inf;
            // A_orig^T * Δy_orig: a_ext は CSC, 行インデックス < m_orig のエントリのみ使用
            let mut at_dy = vec![0.0f64; n];
            for (j, at_dy_j) in at_dy.iter_mut().enumerate() {
                for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                    let row = a_ext.row_ind[ptr];
                    if row < m_orig {
                        *at_dy_j += a_ext.values[ptr] * dy_orig[row];
                    }
                }
            }
            let cond_a = norm_inf_ippmm(&at_dy) / norm_dy < EPS_INF;
            // b_orig · Δy_orig
            let b_dy: f64 = problem
                .b
                .iter()
                .zip(dy_orig.iter())
                .map(|(&bi, &dyi)| bi * dyi)
                .sum();
            let cond_b = b_dy / norm_dy < -EPS_INF;
            if cond_a && cond_b {
                return Some(SolveStatus::Infeasible);
            }
        }
    }

    // --- Dual Infeasibility / Unboundedness check ---
    // m_orig=0 かつ m_ext>0 の場合（境界制約のみの問題）はチェック全体をスキップ。
    // 等式制約なし問題は通常 bounded のため偽陽性回避を優先する。
    if m_orig == 0 && m_ext > 0 {
        return None;
    }
    // ||Δx|| が MIN_DIR_NORM より小さければスキップ（収束時偽陽性防止）。
    let norm_dx_inf = norm_inf_ippmm(dx);
    if norm_dx_inf <= MIN_DIR_NORM {
        return None;
    }
    let norm_dx = norm_dx_inf;

    // 目的関数方向条件: LP(Q=0) → c·Δx < -ε*norm_dx; QP(Q≠0) → ||Q*Δx+c||/norm_dx < ε
    let is_lp = problem.q.values.iter().all(|&v| v == 0.0);
    let cond_obj = if is_lp {
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        c_dx / norm_dx < -EPS_INF
    } else {
        let mut qdx = vec![0.0f64; n];
        spmv_q_ippmm(&problem.q, dx, &mut qdx);
        let qdx_plus_c_norm: f64 = qdx
            .iter()
            .zip(problem.c.iter())
            .map(|(&qi, &ci)| (qi + ci).abs())
            .fold(0.0_f64, f64::max);
        qdx_plus_c_norm / norm_dx < EPS_INF
    };
    if !cond_obj {
        return None;
    }

    // ||A_orig * Δx|| / norm_dx < ε
    if m_orig > 0 {
        let mut a_dx = vec![0.0f64; m_orig];
        for (j, &dxj) in dx.iter().enumerate() {
            for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                let row = a_ext.row_ind[ptr];
                if row < m_orig {
                    a_dx[row] += a_ext.values[ptr] * dxj;
                }
            }
        }
        if norm_inf_ippmm(&a_dx) / norm_dx >= EPS_INF {
            return None;
        }
    }

    Some(SolveStatus::Unbounded)
}

// ---------------------------------------------------------------------------
// fraction-to-boundary（独立実装）
// ---------------------------------------------------------------------------

/// α = min(1, τ · min_i { -v_i / Δv_i } for Δv_i < 0 )
fn fraction_to_boundary_ippmm(v: &[f64], dv: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for (&vi, &dvi) in v.iter().zip(dv.iter()) {
        if dvi < 0.0 {
            let step = tau * vi / (-dvi);
            if step < alpha {
                alpha = step;
            }
        }
    }
    alpha
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

pub(crate) fn timeout_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
        ..Default::default()
    }
}

pub(crate) fn numerical_error_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // MIN_DIR_NORM を超える大きさだが iter ガードが先
        let dy: Vec<f64> = vec![];
        // iter=4 < MIN_ITER=5 → None
        assert_eq!(
            check_infeasible_or_unbounded_ippmm(&dx, &dy, &problem, &a_ext, 0, 0, 4),
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![5e-4]; // ||dx||_inf = 5e-4 <= MIN_DIR_NORM = 1e-3
        let dy: Vec<f64> = vec![];
        // 収束時偽陽性防止: dx が小さすぎる → None
        assert_eq!(
            check_infeasible_or_unbounded_ippmm(&dx, &dy, &problem, &a_ext, 0, 0, 10),
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap();
        let dx = vec![1e-10, 1e-10]; // 非常に小さい → dual チェックはスキップ
        let dy = vec![2.0]; // norm = 2.0 > MIN_DIR_NORM
        assert_eq!(
            check_infeasible_or_unbounded_ippmm(&dx, &dy, &problem, &a_ext, 1, 1, 10),
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // c·dx = -1 < -ε, m_ext=0 なので dual guard は無効
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded_ippmm(&dx, &dy, &problem, &a_ext, 0, 0, 10),
            Some(SolveStatus::Unbounded),
            "IPPMM-T-INF4: LP dual infeasibility → Unbounded であること"
        );
    }
}
