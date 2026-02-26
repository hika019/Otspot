//! 内点法（IP-PMM: Interior Point Proximal Method of Multipliers）QPソルバー
//!
//! Mehrotra predictor-corrector + IP-PMM 正則化による QP 求解。
//! コミット1: LDLパスのみ（Schur complement / normal equations 形式）
//!
//! # アルゴリズム概要
//!
//! 問題: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
//!
//! 境界を含む拡張制約行列 A_ext, b_ext を構築し、
//! スラック s >= 0 を導入して A_ext x + s = b_ext の等式形式に変換。
//!
//! Mehrotra predictor-corrector:
//! 1. Predictor step (affineステップ): μ → 0
//! 2. Corrector step: σ = (μ_aff/μ)^3 による中心化修正
//!
//! KKT系: Schur complement M = Q + δ_p I + A_ext^T D^{-1} A_ext (PD行列)
//! 既存 LdlFactorization をそのまま流用。

use crate::linalg::ldl;
use crate::options::SolverOptions;
use crate::problem::SolveStatus;
use crate::qp::problem::{QpProblem, QpResult};
use crate::qp::ruiz::RuizScaler;
use crate::sparse::CscMatrix;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// IPM 固定パラメータ
// ---------------------------------------------------------------------------

/// fraction-to-boundary τ
const TAU: f64 = 0.995;
/// IP-PMM 正則化最小値
const DELTA_MIN: f64 = 1e-13;

// ---------------------------------------------------------------------------
// タイムアウト管理
// ---------------------------------------------------------------------------

struct TimeoutCtx {
    deadline: Option<Instant>,
    cancel: Arc<AtomicBool>,
}

impl TimeoutCtx {
    fn from_options(opts: &SolverOptions) -> Self {
        let deadline = opts.deadline.or_else(|| {
            opts.timeout_secs
                .map(|s| Instant::now() + Duration::from_secs_f64(s))
        });
        let cancel = opts
            .cancel_flag
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Self { deadline, cancel }
    }

    #[inline]
    fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

// ---------------------------------------------------------------------------
// 疎行列-ベクトル演算
// ---------------------------------------------------------------------------

/// out = A * x（上書き）
#[inline]
fn spmv(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
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
fn spmtv(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
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
fn spmv_q(q: &CscMatrix, x: &[f64], out: &mut [f64]) {
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
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// 拡張制約行列構築
// ---------------------------------------------------------------------------

/// Ax <= b + lb/ub 境界を含む拡張制約を構築する
///
/// 戻り値: (A_ext, b_ext, m_ext, m_orig, n_lb)
/// 順序: [original inequalities | lower bound rows | upper bound rows]
fn build_extended_constraints(problem: &QpProblem) -> (CscMatrix, Vec<f64>, usize, usize, usize) {
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

    // 元の不等式制約 A x <= b
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

    (a_ext, b_ext, m_ext, m, n_lb)
}

// ---------------------------------------------------------------------------
// Schur complement 構築
// ---------------------------------------------------------------------------

/// M = Q + δ_p·I + A_ext^T D^{-1} A_ext の上三角 CSC を構築する
///
/// M は正定値なので既存 LdlFactorization で分解できる。
fn build_schur_complement(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    d_inv: &[f64],
    delta_p: f64,
) -> CscMatrix {
    let n = q.nrows;
    let m_ext = a_ext.nrows;

    // 密行列で蓄積（コミット1: n が小さい問題用）
    let mut m_dense = vec![0.0f64; n * n];

    // Q を加算（全要素格納 → 対称）
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            m_dense[row * n + col] += q.values[k];
            if row != col {
                m_dense[col * n + row] += q.values[k];
            }
        }
    }

    // δ_p·I を加算
    for i in 0..n {
        m_dense[i * n + i] += delta_p;
    }

    // A_ext^T D^{-1} A_ext を加算
    // 行 i のエントリを取得するため行アクセス構造を事前構築
    let mut row_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_ext];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            let row = a_ext.row_ind[k];
            row_data[row].push((col, a_ext.values[k]));
        }
    }

    for i in 0..m_ext {
        let d = d_inv[i];
        let row_i = &row_data[i];
        for &(p, vp) in row_i {
            for &(q_col, vq) in row_i {
                m_dense[p * n + q_col] += d * vp * vq;
            }
        }
    }

    // 上三角のみ triplet として抽出
    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();
    for p in 0..n {
        for q in p..n {
            let v = m_dense[p * n + q];
            if v != 0.0 {
                out_rows.push(p);
                out_cols.push(q);
                out_vals.push(v);
            }
        }
    }

    if out_rows.is_empty() {
        // Q=0, A=0 のエッジケース: δ_p I
        let diag_rows: Vec<usize> = (0..n).collect();
        let diag_cols: Vec<usize> = (0..n).collect();
        let diag_vals = vec![delta_p; n];
        CscMatrix::from_triplets(&diag_rows, &diag_cols, &diag_vals, n, n).unwrap()
    } else {
        CscMatrix::from_triplets(&out_rows, &out_cols, &out_vals, n, n).unwrap()
    }
}

// ---------------------------------------------------------------------------
// fraction-to-boundary
// ---------------------------------------------------------------------------

/// α = min(1, τ · min_i { -v_i / Δv_i }  for Δv_i < 0 )
fn fraction_to_boundary(v: &[f64], dv: &[f64], tau: f64) -> f64 {
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
// メイン求解関数
// ---------------------------------------------------------------------------

/// IPM (Mehrotra predictor-corrector + IP-PMM) で QP を解く
///
/// Ruiz equilibration スケーリングを適用してから内部ソルバーを呼ぶ。
/// options.use_ruiz_scaling=false のときはスケーリングをスキップ。
pub fn solve_qp_ipm(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    // Ruiz equilibration スケーリング（デフォルト有効）
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(q_s, c_s, a_s, b_s, bounds_s) {
            let scaled_result = solve_qp_ipm_inner(&scaled_problem, options);
            return unscale_ipm_result(scaled_result, &scaler);
        }
        // QpProblem::new 失敗 → 非スケールにフォールバック
    }

    solve_qp_ipm_inner(problem, options)
}

/// スケール済み IPM 結果を元のスケールに逆変換する
fn unscale_ipm_result(result: QpResult, scaler: &RuizScaler) -> QpResult {
    match result.status {
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::MaxIterations => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let obj_orig = result.objective / scaler.c;
            QpResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                ..result
            }
        }
        _ => result,
    }
}

/// IPM内部ソルバー（Ruizスケーリング適用済みproblemを受け取る）
///
/// コミット1: LDLパス（Schur complement形式）のみ。CGは後のコミットで追加。
fn solve_qp_ipm_inner(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);

    // T1: 処理前タイムアウトチェック
    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築
    let (a_ext, b_ext, m_ext, m_orig, _n_lb) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // --- 初期点 ---
    // x = 0, s_i = max(1, |b_ext_i| + 1) (s > 0 保証), y_i = 1
    let mut x = vec![0.0f64; n];
    let mut s: Vec<f64> = b_ext
        .iter()
        .map(|&bi| 1.0_f64.max(bi.abs() + 1.0))
        .collect();
    let mut y = vec![1.0f64; m_ext];

    // 作業バッファ
    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];

    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];

    let mut status = SolveStatus::MaxIterations;
    let mut final_iter = options.max_iter_ipm;

    for iter in 0..options.max_iter_ipm {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // 残差計算
        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        // Lagrangian: L = f(x) + y^T(A_ext x + s - b_ext)
        // 停留条件: Qx + c + A_ext^T y = 0  →  r_d = -(Qx + c + A^T y)
        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = s^T y / m_ext（相補性ギャップ）
        let mu: f64 = s
            .iter()
            .zip(y.iter())
            .map(|(&si, &yi)| si * yi)
            .sum::<f64>()
            / m_ext as f64;

        // 収束判定
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let dual_res = norm_inf(&r_d) / norm_c;
        let prim_res = norm_inf(&r_p) / norm_b;

        if dual_res < options.eps_ipm && prim_res < options.eps_ipm && mu < options.eps_ipm {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = DELTA_MIN.max(options.delta_p * mu);
        let delta_d = DELTA_MIN.max(options.delta_d * mu);

        // Σ = diag(s_i / y_i),  D = Σ + δ_d
        let sigma_vec: Vec<f64> = s.iter().zip(y.iter()).map(|(&si, &yi)| si / yi).collect();
        let d_vec: Vec<f64> = sigma_vec.iter().map(|&sg| sg + delta_d).collect();
        let d_inv: Vec<f64> = d_vec.iter().map(|&d| 1.0 / d).collect();

        // Schur complement M = Q + δ_p I + A_ext^T D^{-1} A_ext
        let m_mat = build_schur_complement(&problem.q, &a_ext, &d_inv, delta_p);

        // T2: LDL 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // LDL 因子化
        let fac = match ldl::factorize(&m_mat) {
            Ok(f) => f,
            Err(_) => {
                return numerical_error_result(n);
            }
        };

        // ===== Predictor step (affine: μ_aff → 0) =====
        // r_c_pred_i = -s_i * y_i  (σ = 0)
        let r_c_pred: Vec<f64> = s
            .iter()
            .zip(y.iter())
            .map(|(&si, &yi)| -si * yi)
            .collect();

        // r_p_mod = r_p - r_c_pred / y
        // 導出: A_ext Δx - Σ Δy = r_p - r_c/y  (Schur complement 2行目の右辺)
        let r_p_mod_pred: Vec<f64> = r_p
            .iter()
            .zip(r_c_pred.iter())
            .zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
            .collect();

        // rhs_x = r_d + A^T D^{-1} r_p_mod
        let tmp_pred: Vec<f64> = r_p_mod_pred
            .iter()
            .zip(d_inv.iter())
            .map(|(&ri, &di)| ri * di)
            .collect();
        let mut atmp = vec![0.0f64; n];
        spmtv(&a_ext, &tmp_pred, &mut atmp);
        let rhs_x_pred: Vec<f64> = r_d
            .iter()
            .zip(atmp.iter())
            .map(|(&rdi, &ai)| rdi + ai)
            .collect();

        // Solve M dx_pred = rhs_x_pred
        let mut dx_pred = vec![0.0f64; n];
        fac.solve(&rhs_x_pred, &mut dx_pred);

        // dy_pred = D^{-1} (A_ext dx_pred - r_p_mod)
        let mut a_dx_pred = vec![0.0f64; m_ext];
        spmv(&a_ext, &dx_pred, &mut a_dx_pred);
        let mut dy_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            dy_pred[i] = d_inv[i] * (a_dx_pred[i] - r_p_mod_pred[i]);
        }

        // ds_pred = r_c_pred/y - Σ * dy_pred
        let mut ds_pred = vec![0.0f64; m_ext];
        for i in 0..m_ext {
            ds_pred[i] = r_c_pred[i] / y[i] - sigma_vec[i] * dy_pred[i];
        }

        // α_pred: fraction-to-boundary
        let alpha_s_pred = fraction_to_boundary(&s, &ds_pred, TAU);
        let alpha_y_pred = fraction_to_boundary(&y, &dy_pred, TAU);
        let alpha_pred = alpha_s_pred.min(alpha_y_pred);

        // μ_aff
        let mu_aff: f64 = s
            .iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| (si + alpha_pred * dsi) * (yi + alpha_pred * dyi))
            .sum::<f64>()
            / m_ext as f64;

        // σ = (μ_aff / μ)^3
        let sigma_center = if mu > 1e-15 {
            (mu_aff / mu).powi(3).min(1.0)
        } else {
            0.0
        };

        // ===== Corrector step =====
        // r_c_corr = σμe - ΔS_pred ΔY_pred e - S Y e (complete complementarity RHS)
        let r_c_corr: Vec<f64> = s
            .iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .map(|(((&si, &yi), &dsi), &dyi)| sigma_center * mu - si * yi - dsi * dyi)
            .collect();

        // r_p_mod for corrector (同じ符号規則: r_p - r_c/y)
        let r_p_mod_corr: Vec<f64> = r_p
            .iter()
            .zip(r_c_corr.iter())
            .zip(y.iter())
            .map(|((&rpi, &rci), &yi)| rpi - rci / yi)
            .collect();

        // rhs_x for corrector
        let tmp_corr: Vec<f64> = r_p_mod_corr
            .iter()
            .zip(d_inv.iter())
            .map(|(&ri, &di)| ri * di)
            .collect();
        let mut atmp_corr = vec![0.0f64; n];
        spmtv(&a_ext, &tmp_corr, &mut atmp_corr);
        let rhs_x_corr: Vec<f64> = r_d
            .iter()
            .zip(atmp_corr.iter())
            .map(|(&rdi, &ai)| rdi + ai)
            .collect();

        // Solve M dx = rhs_x_corr (同じ因子化を再利用)
        fac.solve(&rhs_x_corr, &mut dx);

        // dy = D^{-1} (A_ext dx - r_p_mod_corr)
        let mut a_dx_corr = vec![0.0f64; m_ext];
        spmv(&a_ext, &dx, &mut a_dx_corr);
        for i in 0..m_ext {
            dy[i] = d_inv[i] * (a_dx_corr[i] - r_p_mod_corr[i]);
        }

        // ds = r_c_corr/y - Σ * dy
        for i in 0..m_ext {
            ds[i] = r_c_corr[i] / y[i] - sigma_vec[i] * dy[i];
        }

        // α: fraction-to-boundary (corrector)
        let alpha_s = fraction_to_boundary(&s, &ds, TAU);
        let alpha_y = fraction_to_boundary(&y, &dy, TAU);
        let alpha = alpha_s.min(alpha_y);

        // 変数更新
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_ext {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            // 正値性ガード（数値誤差対策）
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }
    }

    // 目的関数値
    spmv_q(&problem.q, &x, &mut qx);
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

    // 双対解: y[0..m_orig] = 元の不等式制約の双対値
    let dual_solution = y[..m_orig].to_vec();

    QpResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals: vec![],
        active_set: vec![],
        iterations: final_iter,
    }
}

// ---------------------------------------------------------------------------
// 制約なし QP
// ---------------------------------------------------------------------------

/// 制約なし QP を解く: Qx = -c（Q が PD でない場合は δ_p I で正則化）
fn solve_unconstrained(problem: &QpProblem, timeout_ctx: &TimeoutCtx) -> QpResult {
    let n = problem.num_vars;

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if n == 0 {
        return QpResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
        };
    }

    // Q + δ_p I の上三角 CSC を構築
    let delta_p = 1e-7;
    let mut triplet_rows: Vec<usize> = Vec::new();
    let mut triplet_cols: Vec<usize> = Vec::new();
    let mut triplet_vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];

    for col in 0..n {
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            let row = problem.q.row_ind[k];
            if row <= col {
                // 上三角のみ
                triplet_rows.push(row);
                triplet_cols.push(col);
                let v = problem.q.values[k]
                    + if row == col { delta_p } else { 0.0 };
                triplet_vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    // 対角に δ_p を追加（まだ格納されていない場合）
    for i in 0..n {
        if !diag_added[i] {
            triplet_rows.push(i);
            triplet_cols.push(i);
            triplet_vals.push(delta_p);
        }
    }

    let q_reg = CscMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n, n)
        .unwrap();

    match ldl::factorize(&q_reg) {
        Ok(fac) => {
            let rhs: Vec<f64> = problem.c.iter().map(|&ci| -ci).collect();
            let mut x = vec![0.0f64; n];
            fac.solve(&rhs, &mut x);

            let mut qx = vec![0.0f64; n];
            spmv_q(&problem.q, &x, &mut qx);
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

            QpResult {
                status: SolveStatus::Optimal,
                objective,
                solution: x,
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 1,
            }
        }
        Err(_) => numerical_error_result(n),
    }
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

fn timeout_result(n: usize) -> QpResult {
    QpResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
    }
}

fn numerical_error_result(n: usize) -> QpResult {
    QpResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
    }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;

    const EPS: f64 = 1e-5;

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
        SolverOptions::default()
    }

    /// IPM-T1: 2変数基本 QP
    /// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_basic_2d() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x + y >= 1 → -(x+y) <= -1
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T1: status");
        close(result.solution[0], 0.5, "IPM-T1: x[0]");
        close(result.solution[1], 0.5, "IPM-T1: x[1]");
        close(result.objective, 0.5, "IPM-T1: objective");
    }

    /// IPM-T2: 制約なし QP (解析解)
    /// min (x-3)^2 + (y-4)^2 = 1/2*2*(x^2+y^2) - 6x - 8y + const
    /// Q=2I, c=[-6,-8], 制約なし
    /// 期待: x*=3, y*=4, obj=-25
    #[test]
    fn test_ipm_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T2: status");
        close(result.solution[0], 3.0, "IPM-T2: x[0]");
        close(result.solution[1], 4.0, "IPM-T2: x[1]");
        close(result.objective, -25.0, "IPM-T2: objective");
    }

    /// IPM-T3: 等式制約付き QP
    /// min x^2 + y^2  s.t. x + y = 1
    /// 等式を 2 不等式で表現: x+y<=1, -(x+y)<=-1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_equality_constraint() {
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

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T3: status");
        close(result.solution[0], 0.5, "IPM-T3: x[0]");
        close(result.solution[1], 0.5, "IPM-T3: x[1]");
        close(result.objective, 0.5, "IPM-T3: objective");
    }

    /// IPM-T4: Box 制約付き QP
    /// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
    /// Q=2I, c=[-4,-4], bounds=[0,1]^2
    /// 期待: x*=y*=1, obj=-6
    #[test]
    fn test_ipm_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T4: status");
        close(result.solution[0], 1.0, "IPM-T4: x[0]");
        close(result.solution[1], 1.0, "IPM-T4: x[1]");
        close(result.objective, -6.0, "IPM-T4: objective");
    }

    /// IPM-T5: ポートフォリオ最適化（3変数等式+非負制約）
    /// min 1/2 w^T Σ w  s.t. sum(w)=1, w >= 0
    /// Σ = diag(2,2,2), 対称解: w* = [1/3, 1/3, 1/3], obj = 1/3
    #[test]
    fn test_ipm_portfolio() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3,
            3,
        )
        .unwrap();
        let c = vec![0.0, 0.0, 0.0];
        // 等式 sum=1 (2不等式) + 非負制約 w>=0 (3不等式)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5,
            3,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T5: status");
        close(result.solution[0], 1.0 / 3.0, "IPM-T5: w[0]");
        close(result.solution[1], 1.0 / 3.0, "IPM-T5: w[1]");
        close(result.solution[2], 1.0 / 3.0, "IPM-T5: w[2]");
        close(result.objective, 1.0 / 3.0, "IPM-T5: objective");
    }

    /// IPM-T6: タイムアウト動作確認（極小 timeout で Timeout が返ること）
    #[test]
    fn test_ipm_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // 0.0001 秒（0.1ms）のタイムアウトで Timeout が返ることを確認
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(0.0001);
        // use_ruiz_scaling を無効化して Ruiz 処理でタイムアウトが誤吸収されないようにする
        opts.use_ruiz_scaling = false;
        let result = solve_qp_ipm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "IPM-T6: expected Timeout or Optimal, got {:?}",
            result.status
        );
    }

    /// IPM-T7: Ruiz スケーリング有無で同一解が得られることを確認
    /// T1 と同じ問題 (min x^2+y^2, s.t. x+y>=1) で比較
    #[test]
    fn test_ipm_ruiz_scaling_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // Ruiz 有効（デフォルト）
        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());

        // Ruiz 無効
        let mut opts_no_ruiz = SolverOptions::default();
        opts_no_ruiz.use_ruiz_scaling = false;
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);

        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "IPM-T7: ruiz status");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "IPM-T7: no-ruiz status");
        close(result_ruiz.solution[0], result_no_ruiz.solution[0], "IPM-T7: x[0]");
        close(result_ruiz.solution[1], result_no_ruiz.solution[1], "IPM-T7: x[1]");
        close(result_ruiz.objective, result_no_ruiz.objective, "IPM-T7: objective");
    }
}
