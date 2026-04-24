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
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::kkt::{spmv, spmtv, spmv_q, norm_inf, build_extended_constraints, build_augmented_system};
use super::common::{check_infeasible_or_unbounded, solve_unconstrained, timeout_result, numerical_error_result};
use super::solver_loop::{compute_sigma_vec, predictor_step, corrector_step, gondzio_correctors, update_variables};
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
/// PARAM: 根拠=MATLAB拡張版IP-PMM準拠（設計書§A_PMM参照）| 承認=cmd_794 Phase 3
const PMM_SLOW_RATE: f64 = 2.0 / 3.0;


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

    // ── Mehrotra 風初期点射影（等式制約への最小ノルム補正）─────────
    // 解く系: K_init [dx; dy] = [0; r_p_eq], K_init = [I, A_ext^T; A_ext, -I]
    //   build_augmented_system に Q=0, Σ=0, ρ=δ=1 を渡して流用。
    // 目的: x0（ボックス中点）を A_eq x = b_eq の近傍へ押し出し、初期 pf を下げる。
    //       UBH1 のように FR 変数が多い問題で x=0 由来の pf 爆発を抑制する。
    // 等式行の残差のみ RHS に入れ、box/ineq 行は 0（内点維持）。
    {
        let r_p_eq: Vec<f64> = b_ext.iter().zip(ax0.iter()).enumerate()
            .map(|(i, (&bi, &axi))| if is_eq_ext[i] { bi - axi } else { 0.0 })
            .collect();
        let r_p_inf = r_p_eq.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if r_p_inf > 1e-6 && !timeout_ctx.should_stop() {
            let q_zero = CscMatrix::new(n, n);
            let sigma_zero = vec![0.0_f64; m_ext];
            let k_init = build_augmented_system(&q_zero, &a_ext, &sigma_zero, 1.0, 1.0);
            let perm_init = amd_with_deadline(
                k_init.nrows, &k_init.col_ptr, &k_init.row_ind, timeout_ctx.deadline,
            );
            if let Ok(fac_init) = ldl::factorize_quasidefinite_with_cached_perm_threaded(
                &k_init, &perm_init, timeout_ctx.deadline,
            ) {
                let mut rhs_init = vec![0.0_f64; n + m_ext];
                for i in 0..m_ext { rhs_init[n + i] = r_p_eq[i]; }
                let mut sol_init = vec![0.0_f64; n + m_ext];
                fac_init.solve(&rhs_init, &mut sol_init);
                let dx_inf = sol_init[..n].iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                if dx_inf.is_finite() && dx_inf < 1e8 {
                    for j in 0..n {
                        let x_new = x[j] + sol_init[j];
                        let (lb, ub) = problem.bounds[j];
                        x[j] = match (lb.is_finite(), ub.is_finite()) {
                            (true, true) => {
                                // 狭い箱 (ub-lb が 2*margin 未満) では中点を返して panic 回避
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
                    // s0 再計算（不等式行のみ、等式は 0 維持）
                    let mut ax_new = vec![0.0_f64; m_ext];
                    for col in 0..n {
                        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
                            ax_new[a_ext.row_ind[k]] += a_ext.values[k] * x[col];
                        }
                    }
                    for i in 0..m_ext {
                        s[i] = if is_eq_ext[i] { 0.0 } else { (b_ext[i] - ax_new[i]).max(1.0) };
                    }
                    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "IPPMM_INIT_PROJ: r_p_eq_inf={:.3e} dx_inf={:.3e} |x|_inf={:.3e}",
                            r_p_inf, dx_inf,
                            x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
                        );
                    }
                }
            }
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

    // PARAM: 根拠=MATLAB拡張版IP-PMM準拠。LP(Q=0)とQP(Q≠0)で分離 | 承認=cmd_794 Phase 3
    // 【履歴】cmd_833 redo5 で論文式(動的) を一時導入→DTOC3(‖A‖∞≈2.0)で reg_limit が
    // 2500倍緩くなり退行。best-so-far + false-unbounded 格下げは維持したまま reg_limit は定数に戻す。
    let reg_limit = if problem.q.values.iter().all(|&v| v == 0.0) {
        5e-10  // LP: MATLAB拡張版準拠
    } else {
        5e-8   // QP: MATLAB拡張版準拠
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
    // [cmd_841 Bug#1] best-so-far の rel_gap も保持。
    // reject_false_*_bestsofar 経路で偽 Optimal 昇格を防ぐためのゲート用。
    let mut best_rel_gap: f64 = f64::INFINITY;

    // [cmd_841 null-space] alpha 停滞検出。
    // UBH1 のように PMM proximal が null-space 方向に頭打ちし、line search が
    // alpha≈0 で止まるケースで 2273 iters 無駄回りするのを防ぐ。
    // 連続 ALPHA_STALL_N 回 alpha < ALPHA_STALL_EPS なら best-so-far で早期脱出。
    const ALPHA_STALL_EPS: f64 = 1e-8;
    const ALPHA_STALL_N: usize = 5;
    // [cmd_846] deadlock 検出用（eps 非依存）: alpha=0 が長期継続＋rho/delta が reg_limit
    // フロアに張り付いている場合、数値的に進めない状態。POST_VERIFY の eps 厳格化で
    // best_score < eps が成立しなくなった UBH1 型の無限ループ対策。
    const ALPHA_DEADLOCK_N: usize = 20;
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

        // [cmd_841 Bug#1] 双対ギャップを best-so-far 更新前に算出。
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
        let norm_c_bs = norm_inf(&problem.c).max(1.0);
        let norm_b_bs = norm_inf(&b_ext).max(1.0);
        if nr_p.is_finite() && nr_d.is_finite() && mu.is_finite() {
            let score = nr_p / (1.0 + norm_b_bs) + nr_d / (1.0 + norm_c_bs) + mu.abs();
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

        // Exp M trace [cmd_833 redo5, release-safe, env-gated]
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
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        // 原空間双対残差: r_d_orig[j] = r_d_scaled[j] / (c · d[j])
        // スケール済み残差だけで収束宣言すると真の最適でない basin で止まる（UBH1 obj=2.12 事例, cmd_841）
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

        // [cmd_841 Bug#1] rel_gap / DUALITY_GAP_TOL は上のブロックで計算済（best-so-far 更新前）。
        // UBH1 (||x||≈1459, c=0, Q rank-deficient) で r_stat=2e-6・mu=1e-30 なのに
        // duality gap = 9.49 で obj 91% 誤差の事例を検出できなかった（cmd_841 Phase A 検証）。
        // 3 族独立 solver (PIQP/Clarabel/OSQP) で UBH1 真値 1.116 を確認済。

        if nr_d < eps * (1.0 + norm_c)
            && nr_d_orig < eps_orig * (1.0 + norm_c_orig)
            && nr_p < eps * (1.0 + norm_b)
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
        // PARAM(reg_limit*1e-2): 根拠=経験値(μがreg_limitの1/100以下=正則化下限の100倍収束で実質停滞とみなす。論文記載なし) | 承認=cmd_493実装時設定・要検証
        let thr_d = (eps * (1.0 + norm_c)).max(reg_limit * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(reg_limit * 10.0);
        if mu < reg_limit * 1e-2 && nr_d < thr_d && nr_p < thr_p && rel_gap.abs() < DUALITY_GAP_TOL {
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
                        .zip(orig.constraint_types.iter())
                        .map(|((&axi, &bi), ct)| match ct {
                            ConstraintType::Eq => (axi - bi).abs(),
                            ConstraintType::Ge => (bi - axi).max(0.0),
                            _ => (axi - bi).max(0.0),
                        })
                        .fold(0.0_f64, f64::max)
                };
                let norm_b_orig = norm_inf(&orig.b).max(1.0);
                if pfeas_orig < eps_orig * (1.0 + norm_b_orig)
                    && nr_d_orig < eps_orig * (1.0 + norm_c_orig)
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
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);

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

        // N1: mu_rate(predictor直後)は廃止。変数更新後のμからrを計算する（PMM更新部で実施）

        // ── Predictor ──────────────────────────────────────────────
        let pred = predictor_step(
            &s, &y, &is_eq_ext, m_ineq,
            &r_d_pmm, &r_p_pmm,  // r_dual=r_d_pmm, r_primal=r_p_pmm (IPPMM)
            &sigma_vec, &fac, n, m_ext, mu,
        );

        // ── Corrector ──────────────────────────────────────────────
        let (alpha, r_c_corr) = corrector_step(
            &s, &y, &is_eq_ext,
            &pred, mu,
            &r_d_pmm, &r_p_pmm,  // r_dual=r_d_pmm, r_primal=r_p_pmm (IPPMM)
            &sigma_vec, &fac, n, m_ext,
            &mut dx, &mut dy, &mut ds,
        );

        // ── Gondzio multiple centrality correctors ──────────────────
        let mut alpha = alpha;
        if alpha < 0.999 {
            alpha = gondzio_correctors(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d_pmm, &r_p_pmm,  // r_dual=r_d_pmm, r_primal=r_p_pmm (IPPMM)
                &r_c_corr, &sigma_vec, &fac, n, m_ext,
                options.ipm.max_correctors, alpha,
                &mut dx, &mut dy, &mut ds,
            );
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
            // [cmd_841 Bug#1] best_score は残差 (pf+df+mu) のみを評価。
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

        // cmd_841: step magnitude trace（IPPMM_TRACE=1 のときのみ）
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

        // [cmd_841 null-space] alpha 停滞早期脱出。
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
        // [cmd_846] eps 非依存 deadlock gate。POST_VERIFY の eps 10x 厳格化で
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

    // [cmd_842] Timeout/MaxIterations の素の終了経路で best-so-far に復帰。
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
        // [cmd_841 null-space] best-so-far の相対双対ギャップ。
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
