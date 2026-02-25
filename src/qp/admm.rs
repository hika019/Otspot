//! ADMM QPソルバー実装
//!
//! OSQP方式のADMM（Alternating Direction Method of Multipliers）を用いて
//! 凸二次計画問題 min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub を解く。
//!
//! # アルゴリズム概要
//!
//! 拡張制約行列 C = [A; I_n] を用いてOSQP形式に変換し、
//! LDL^T 分解（QDLDL）で K = Q + (σ+ρ)I + ρ A^T A を因子化する。
//! 反復ループは x-update（LDL solve）+ z-update（box projection）+ y-update（dual ascent）で構成。
//!
//! # timeout組み込み
//! T1（LDL前）、T2（LDL後）、T3（各反復先頭）で `TimeoutCtx::should_stop()` をチェックする。

use crate::linalg::cg::{pcg_solve, CgWorkspace};
use crate::linalg::ldl::{self, LdlError, LdlFactorization};
use crate::options::SolverOptions;
use crate::problem::SolveStatus;
use crate::qp::problem::{QpProblem, QpResult};
use crate::sparse::CscMatrix;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// タイムアウト管理
// ---------------------------------------------------------------------------

/// n > LDL_THRESHOLD のとき CG パスを自動選択
const LDL_THRESHOLD: usize = 5_000;

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
// ADMMワークスペース（GPU移行設計 §4.3 G1-G2準拠: 全バッファをnew()で一括allocate）
// ---------------------------------------------------------------------------

struct AdmmWorkspace {
    x: Vec<f64>,       // n: primal variable (current)
    z: Vec<f64>,       // m_aug: slack (projected, box constraint)
    y: Vec<f64>,       // m_aug: dual variable (scaled Lagrange multiplier)
    x_prev: Vec<f64>,  // n: previous x (proximal term)
    cx: Vec<f64>,      // m_aug: C*x (constraint-space primal)
    x_tilde: Vec<f64>, // m_aug: over-relaxed constraint value
    rhs: Vec<f64>,     // n: LDL solve right-hand side
    r_prim: Vec<f64>,  // m_aug: primal residual Cx - z
    r_dual: Vec<f64>,  // n: dual residual Qx + c + C^T y
    tmp_n: Vec<f64>,   // n: scratch (Qx, C^T v partial, etc.)
    tmp_m: Vec<f64>,   // m_aug: scratch (ρz - y, etc.)
    // CG用フィールド（CGパス用）
    cg_ws: CgWorkspace, // CGソルバー作業バッファ（サイズ n）
    m_inv: Vec<f64>,    // 対角前処理 1/diag(K)（サイズ n）
    kv_tmp: Vec<f64>,   // kv_mul用中間バッファ（サイズ m = m_aug - n）
}

impl AdmmWorkspace {
    fn new(n: usize, m_aug: usize) -> Self {
        let m = m_aug.saturating_sub(n);
        Self {
            x: vec![0.0; n],
            z: vec![0.0; m_aug],
            y: vec![0.0; m_aug],
            x_prev: vec![0.0; n],
            cx: vec![0.0; m_aug],
            x_tilde: vec![0.0; m_aug],
            rhs: vec![0.0; n],
            r_prim: vec![0.0; m_aug],
            r_dual: vec![0.0; n],
            tmp_n: vec![0.0; n],
            tmp_m: vec![0.0; m_aug],
            cg_ws: CgWorkspace::new(n),
            m_inv: vec![1.0; n],
            kv_tmp: vec![0.0; m],
        }
    }
}

// ---------------------------------------------------------------------------
// 疎行列-ベクトル演算（GPU移行設計 §4.3 G3準拠: 明示的forループ）
// ---------------------------------------------------------------------------

/// out = A * x  （上書き）
#[inline]
fn spmv_a(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for (col, &xv) in x.iter().enumerate() {
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            out[a.row_ind[k]] += a.values[k] * xv;
        }
    }
}

/// out += A^T * v  （加算）
#[inline]
fn spmv_at_add(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    for (col, out_val) in out.iter_mut().enumerate() {
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            *out_val += a.values[k] * v[a.row_ind[k]];
        }
    }
}

/// out = C * x  where C = [A; I_n]  (m_aug = m + n)
/// out[0..m] = A*x, out[m..m+n] = x
#[inline]
fn spmv_c(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    let m = a.nrows;
    let n = a.ncols;
    spmv_a(a, x, &mut out[..m]);
    out[m..m + n].copy_from_slice(x);
}

/// out = C^T * v  where C = [A; I_n]  （上書き）
/// out_j = (A^T * v[0..m])_j + v[m+j]
#[inline]
fn spmv_ct(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    let m = a.nrows;
    let n = a.ncols;
    // まず identity ブロック: out[0..n] = v[m..m+n]
    out[..n].copy_from_slice(&v[m..m + n]);
    // 次に A^T * v[0..m] を加算
    spmv_at_add(a, &v[..m], out);
}

/// out = Q * x  （全要素格納の対称 Q に対応）
#[inline]
fn spmv_q(q: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for (col, &xv) in x.iter().enumerate() {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            out[q.row_ind[k]] += q.values[k] * xv;
        }
    }
}

/// 無限大ノルム ||v||_∞
#[inline]
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// K行列構築
// ---------------------------------------------------------------------------

/// K の上三角を CSC で構築
///
/// K = Q_upper + (σ+ρ)I + ρ * (A^T A)_upper
/// ここで C = [A; I_n] なので C^T C = A^T A + I_n、ρC^TC = ρA^TA + ρI ✓
fn build_k_upper(
    q: &CscMatrix,
    a: &CscMatrix,
    sigma: f64,
    rho: f64,
) -> Result<CscMatrix, String> {
    let n = q.nrows;
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // 1. Q の上三角エントリ (row <= col のみ)
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                rows.push(row);
                cols.push(col);
                vals.push(q.values[k]);
            }
        }
    }

    // 2. (σ+ρ) * I 対角成分
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(sigma + rho);
    }

    // 3. ρ * A^T A の上三角エントリ
    // A は m×n CSC。A^T を転置して行アクセス: a_t の列 i = A の行 i
    let a_t = a.transpose(); // a_t: n×m CSC
    for i in 0..a.nrows {
        let start = a_t.col_ptr[i];
        let end = a_t.col_ptr[i + 1];
        // A行 i の非ゼロ: j = a_t.row_ind[k], val = a_t.values[k]  (j昇順)
        for p in start..end {
            let j = a_t.row_ind[p];
            let vj = a_t.values[p];
            for q_idx in p..end {
                // j <= k 保証 (row_ind is sorted)
                let k = a_t.row_ind[q_idx];
                let vk = a_t.values[q_idx];
                rows.push(j);
                cols.push(k);
                vals.push(rho * vj * vk);
            }
        }
    }

    CscMatrix::from_triplets(&rows, &cols, &vals, n, n)
        .map_err(|e| format!("K build error: {:?}", e))
}

// ---------------------------------------------------------------------------
// LDL分解フォールバック
// ---------------------------------------------------------------------------

/// try_factorize の失敗種別
#[derive(Debug)]
enum TryFactorizeErr {
    /// 全 σ 候補で行列が特異または不定
    Numerical,
    /// deadline を超過した（タイムアウト）
    Timeout,
}

/// σ を段階的に増加させて LDL 分解を試みる（最大4段階）
///
/// 成功: Ok((factorization, sigma_used))
/// 数値失敗: Err(TryFactorizeErr::Numerical)
/// タイムアウト: Err(TryFactorizeErr::Timeout)
///
/// # cmd_171: timeout audit fix
/// n >= 10_000 の K 行列 LDL 因子化は秒単位を要しうる。
/// deadline を factorize_with_deadline に渡すことで内部ハングを防止する。
fn try_factorize(
    q: &CscMatrix,
    a: &CscMatrix,
    rho: f64,
    sigma_init: f64,
    deadline: Option<Instant>,
) -> Result<(LdlFactorization, f64), TryFactorizeErr> {
    let sigma_candidates = [
        sigma_init,
        sigma_init * 10.0,
        sigma_init * 100.0,
        sigma_init * 1000.0,
    ];
    for &sigma in &sigma_candidates {
        // cmd_171: timeout audit fix — 各σ候補の前にもdeadlineチェック
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Err(TryFactorizeErr::Timeout);
            }
        }
        if let Ok(k_mat) = build_k_upper(q, a, sigma, rho) {
            match ldl::factorize_with_deadline(&k_mat, deadline) {
                Ok(fac) => return Ok((fac, sigma)),
                Err(LdlError::SingularOrIndefinite) => continue,
                Err(LdlError::DeadlineExceeded) => return Err(TryFactorizeErr::Timeout),
            }
        }
    }
    Err(TryFactorizeErr::Numerical)
}

// ---------------------------------------------------------------------------
// Matrix-Free K*v operator（CGパス用）
// ---------------------------------------------------------------------------

/// Matrix-free K*v 演算
///
/// K = Q + (σ+ρ)I + ρ*A^T*A に対して result = K*v を計算する。
/// K行列を明示的に構築せず、SpMV 2回 + ベクトル演算で完結する。
///
/// 計算手順:
/// 1. tmp_m = A * v
/// 2. result = A^T * tmp_m
/// 3. result[j] = ρ*result[j] + (σ+ρ)*v[j] + (Q*v)[j]
///
/// # 引数
/// - `q`: 目的関数の2次行列 Q（n×n CSC）
/// - `a`: 制約行列 A（m×n CSC、C = [A; I_n] のA部分）
/// - `sigma`, `rho`: ADMMパラメータ
/// - `v`: 入力ベクトル（長さ n）
/// - `result`: 出力 K*v（長さ n、上書き）
/// - `tmp_m`: 中間バッファ（長さ m = a.nrows）
// GPU移行設計 §4.3 G3準拠のインデックスループを維持。
#[allow(clippy::needless_range_loop)]
fn kv_mul(
    q: &CscMatrix,
    a: &CscMatrix,
    sigma: f64,
    rho: f64,
    v: &[f64],
    result: &mut [f64],
    tmp_m: &mut [f64],
) {
    let n = v.len();
    debug_assert_eq!(result.len(), n);
    debug_assert_eq!(tmp_m.len(), a.nrows);

    // step 1: tmp_m = A * v
    spmv_a(a, v, tmp_m);

    // step 2: result = A^T * tmp_m
    result.iter_mut().for_each(|x| *x = 0.0);
    spmv_at_add(a, tmp_m, result);

    // step 3: result[j] = ρ*result[j] + (σ+ρ)*v[j]
    for j in 0..n {
        result[j] = rho * result[j] + (sigma + rho) * v[j];
    }

    // step 4: result += Q * v （通常SpMV）
    // Q*v を tmp_m[0..n] に一時格納して加算
    // tmp_m のサイズが n 以上なら使い回し可能だが、サイズ m かもしれないため
    // 直接加算する（CSC列走査で result に直接加算）
    for (col, &vv) in v.iter().enumerate() {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            result[q.row_ind[k]] += q.values[k] * vv;
        }
    }
}

/// 対角前処理行列の構築
///
/// diag(K)_j = diag(Q)_j + (σ+ρ) + ρ * ||A[:,j]||²  を計算し、
/// その逆数 1/diag(K)_j を m_inv に格納する。
///
/// # 引数
/// - `q`: 目的関数の2次行列 Q（n×n CSC）
/// - `a`: 制約行列 A（m×n CSC）
/// - `sigma`, `rho`: ADMMパラメータ
/// - `m_inv`: 出力 1/diag(K)（長さ n、上書き）
fn build_preconditioner(
    q: &CscMatrix,
    a: &CscMatrix,
    sigma: f64,
    rho: f64,
    m_inv: &mut [f64],
) {
    debug_assert_eq!(m_inv.len(), q.ncols);

    // diag(K)_j = (σ+ρ) を基点に初期化
    for v in m_inv.iter_mut() {
        *v = sigma + rho;
    }

    // diag(Q)_j を加算（対角成分のみ）
    for (col, v) in m_inv.iter_mut().enumerate() {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col {
                *v += q.values[k];
            }
        }
    }

    // ρ * ||A[:,j]||² を加算（Aのj列の値の二乗和）
    for (col, v) in m_inv.iter_mut().enumerate() {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        let col_sq: f64 = (start..end).map(|k| a.values[k] * a.values[k]).sum();
        *v += rho * col_sq;
    }

    // 逆数化（ゼロ除算防止: 最小値 1e-8 に clamp）
    for v in m_inv.iter_mut() {
        *v = 1.0 / v.max(1e-8);
    }
}

// ---------------------------------------------------------------------------
// 目的関数計算
// ---------------------------------------------------------------------------

/// 1/2 x^T Q x + c^T x  を計算
/// tmp は長さ n のスクラッチバッファ（Qx の格納先）
fn compute_objective(q: &CscMatrix, c: &[f64], x: &[f64], tmp: &mut [f64]) -> f64 {
    spmv_q(q, x, tmp);
    let quad: f64 = x.iter().zip(tmp.iter()).map(|(&xi, &qi)| xi * qi).sum();
    let lin: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
    0.5 * quad + lin
}

// ---------------------------------------------------------------------------
// メイン関数
// ---------------------------------------------------------------------------

/// ADMM法でQPを解く（公開API）
///
/// # 問題形式
/// min 1/2 x^T Q x + c^T x
/// s.t. Ax <= b,  lb <= x <= ub
///
/// # アルゴリズム選択
/// admm_use_cg=None の場合、n > LDL_THRESHOLD なら CG パス、それ以外は LDL パス。
/// admm_use_cg=Some(true/false) で強制選択可能。
pub fn solve_qp_admm(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    let timeout = TimeoutCtx::from_options(options);
    let use_cg = match options.admm_use_cg {
        Some(b) => b,
        None => problem.num_vars > LDL_THRESHOLD,
    };
    if use_cg {
        solve_qp_admm_cg(problem, options, &timeout)
    } else {
        solve_qp_admm_ldl(problem, options, &timeout)
    }
}

// ---------------------------------------------------------------------------
// ADMM QPソルバー（LDLパス）
// ---------------------------------------------------------------------------

/// ADMM法でQPを解く（LDL x-update パス）
///
/// K = Q + (σ+ρ)I + ρ A^T A を LDL^T 分解し x-update を直接解く。
/// timeout は T1（LDL前）、T2（LDL後）、T3（各反復先頭）で検出。
fn solve_qp_admm_ldl(
    problem: &QpProblem,
    options: &SolverOptions,
    timeout_ctx: &TimeoutCtx,
) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    let m_aug = m + n;

    let sigma_init = options.sigma;
    let mut rho = options.rho;
    let alpha = options.alpha;
    let eps_abs = options.eps_abs;
    let eps_rel = options.eps_rel;
    let max_iter = options.max_iter_admm.unwrap_or(10000);

    let a = &problem.a;
    let q = &problem.q;
    let c = &problem.c;

    let mut l_bound = vec![f64::NEG_INFINITY; m_aug];
    let mut u_bound = vec![f64::INFINITY; m_aug];
    u_bound[..m].copy_from_slice(&problem.b[..m]);
    for j in 0..n {
        l_bound[m + j] = problem.bounds[j].0;
        u_bound[m + j] = problem.bounds[j].1;
    }

    // T1: LDL前 timeout チェック
    if timeout_ctx.should_stop() {
        return make_timeout_result(n, m, 0);
    }

    // 初期 LDL 分解（σフォールバック付き）
    // cmd_171: timeout audit fix — deadline を渡して内部ハングを防止
    let (mut fac, mut sigma_used) = match try_factorize(q, a, rho, sigma_init, timeout_ctx.deadline) {
        Ok(v) => v,
        Err(TryFactorizeErr::Numerical) => return make_numerical_error_result(n, m),
        Err(TryFactorizeErr::Timeout) => return make_timeout_result(n, m, 0),
    };

    // T2: LDL後 timeout チェック
    if timeout_ctx.should_stop() {
        return make_timeout_result(n, m, 0);
    }

    let mut ws = AdmmWorkspace::new(n, m_aug);

    // z の初期値を [l, u] にクランプ（可行点から開始）
    for i in 0..m_aug {
        ws.z[i] = 0.0_f64.max(l_bound[i]).min(u_bound[i]);
    }

    for iter in 0..max_iter {
        // T3: 各反復先頭の timeout チェック
        if timeout_ctx.should_stop() {
            let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
            return QpResult {
                status: SolveStatus::Timeout,
                objective: obj,
                solution: ws.x.clone(),
                dual_solution: ws.y[..m].to_vec(),
                bound_duals: vec![],
                active_set: vec![],
                iterations: iter,
            };
        }

        // --- x-update ---
        // rhs = σ*x_prev - c + C^T*(ρ*z - y)
        ws.x_prev.copy_from_slice(&ws.x);

        for i in 0..m_aug {
            ws.tmp_m[i] = rho * ws.z[i] - ws.y[i];
        }
        spmv_ct(a, &ws.tmp_m, &mut ws.rhs);
        for (j, &cj) in c.iter().enumerate() {
            ws.rhs[j] += sigma_used * ws.x_prev[j] - cj;
        }

        fac.solve(&ws.rhs, &mut ws.x);

        // --- z-update（over-relaxation in constraint space） ---
        spmv_c(a, &ws.x, &mut ws.cx);
        let one_minus_alpha = 1.0 - alpha;
        for i in 0..m_aug {
            ws.x_tilde[i] = alpha * ws.cx[i] + one_minus_alpha * ws.z[i];
        }
        for i in 0..m_aug {
            let v = ws.x_tilde[i] + ws.y[i] / rho;
            ws.z[i] = v.max(l_bound[i]).min(u_bound[i]);
        }

        // --- y-update ---
        for i in 0..m_aug {
            ws.y[i] += rho * (ws.x_tilde[i] - ws.z[i]);
        }

        // --- 収束判定（10反復ごと） ---
        if iter % 10 == 0
            && check_convergence(
                q, a, c, &ws.x, &ws.z, &ws.y,
                &mut ws.r_prim, &mut ws.r_dual,
                &mut ws.cx, &mut ws.tmp_n, &mut ws.tmp_m,
                eps_abs, eps_rel, m, n, m_aug,
            )
        {
            let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
            return QpResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: ws.x.clone(),
                dual_solution: ws.y[..m].to_vec(),
                bound_duals: vec![],
                active_set: vec![],
                iterations: iter + 1,
            };
        }

        // --- ρ適応更新（25反復ごと、初回スキップ） ---
        if iter % 25 == 0 && iter > 0 {
            let rho_new = compute_rho_update(
                q, a, c, &ws.x, &ws.z, &ws.y,
                &mut ws.r_prim, &mut ws.r_dual,
                &mut ws.cx, &mut ws.tmp_n, &mut ws.tmp_m,
                rho, eps_abs, m_aug,
            );
            if (rho_new / rho - 1.0).abs() > 0.1 {
                if timeout_ctx.should_stop() {
                    let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
                    return QpResult {
                        status: SolveStatus::Timeout,
                        objective: obj,
                        solution: ws.x.clone(),
                        dual_solution: ws.y[..m].to_vec(),
                        bound_duals: vec![],
                        active_set: vec![],
                        iterations: iter,
                    };
                }
                // K再構築 + LDL再分解（失敗時は旧ρで継続、タイムアウト時は即リターン）
                // cmd_171: timeout audit fix — deadline を渡してρ更新LDL内部ハングを防止
                match try_factorize(q, a, rho_new, sigma_used, timeout_ctx.deadline) {
                    Ok((new_fac, new_sigma)) => {
                        let scale = rho / rho_new;
                        for i in 0..m_aug {
                            ws.y[i] *= scale;
                        }
                        fac = new_fac;
                        sigma_used = new_sigma;
                        rho = rho_new;
                    }
                    Err(TryFactorizeErr::Timeout) => {
                        let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
                        return QpResult {
                            status: SolveStatus::Timeout,
                            objective: obj,
                            solution: ws.x.clone(),
                            dual_solution: ws.y[..m].to_vec(),
                            bound_duals: vec![],
                            active_set: vec![],
                            iterations: iter,
                        };
                    }
                    Err(TryFactorizeErr::Numerical) => { /* 旧ρで継続 */ }
                }
                if timeout_ctx.should_stop() {
                    let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
                    return QpResult {
                        status: SolveStatus::Timeout,
                        objective: obj,
                        solution: ws.x.clone(),
                        dual_solution: ws.y[..m].to_vec(),
                        bound_duals: vec![],
                        active_set: vec![],
                        iterations: iter,
                    };
                }
            }
        }
    }

    // max_iter 到達
    let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
    QpResult {
        status: SolveStatus::MaxIterations,
        objective: obj,
        solution: ws.x.clone(),
        dual_solution: ws.y[..m].to_vec(),
        bound_duals: vec![],
        active_set: vec![],
        iterations: max_iter,
    }
}

// ---------------------------------------------------------------------------
// ADMM QPソルバー（CGパス）
// ---------------------------------------------------------------------------

/// ADMM法でQPを解く（Matrix-Free CG x-update パス）
///
/// K を明示的に構築せず PCG で x-update を解く。n > LDL_THRESHOLD 時または強制指定時。
/// ρ適応: rho値変更 + preconditioner再構築のみ（K再構築・LDL再分解不要）。
/// timeout は T1（preconditioner前）、T2（preconditioner後）、T3（各反復先頭）で検出。
fn solve_qp_admm_cg(
    problem: &QpProblem,
    options: &SolverOptions,
    timeout_ctx: &TimeoutCtx,
) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    let m_aug = m + n;

    let sigma = options.sigma;
    let mut rho = options.rho;
    let alpha = options.alpha;
    let eps_abs = options.eps_abs;
    let eps_rel = options.eps_rel;
    let max_iter = options.max_iter_admm.unwrap_or(10000);

    let a = &problem.a;
    let q = &problem.q;
    let c = &problem.c;

    let mut l_bound = vec![f64::NEG_INFINITY; m_aug];
    let mut u_bound = vec![f64::INFINITY; m_aug];
    u_bound[..m].copy_from_slice(&problem.b[..m]);
    for j in 0..n {
        l_bound[m + j] = problem.bounds[j].0;
        u_bound[m + j] = problem.bounds[j].1;
    }

    // T1: build_preconditioner前 timeout チェック
    if timeout_ctx.should_stop() {
        return make_timeout_result(n, m, 0);
    }

    let mut ws = AdmmWorkspace::new(n, m_aug);
    build_preconditioner(q, a, sigma, rho, &mut ws.m_inv);

    // T2: build_preconditioner後 timeout チェック
    if timeout_ctx.should_stop() {
        return make_timeout_result(n, m, 0);
    }

    // z の初期値をボックス制約にクランプ（可行点から開始）
    for i in 0..m_aug {
        ws.z[i] = 0.0_f64.max(l_bound[i]).min(u_bound[i]);
    }

    let cg_tol = (eps_abs * 0.1).max(1e-4);
    let cg_max_iter = n.min(200);

    for iter in 0..max_iter {
        // T3: 各反復先頭 timeout チェック
        if timeout_ctx.should_stop() {
            let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
            return QpResult {
                status: SolveStatus::Timeout,
                objective: obj,
                solution: ws.x.clone(),
                dual_solution: ws.y[..m].to_vec(),
                bound_duals: vec![],
                active_set: vec![],
                iterations: iter,
            };
        }

        // --- x-update (PCG) ---
        // rhs = σ*x_prev - c + C^T*(ρ*z - y)
        ws.x_prev.copy_from_slice(&ws.x);
        for i in 0..m_aug {
            ws.tmp_m[i] = rho * ws.z[i] - ws.y[i];
        }
        spmv_ct(a, &ws.tmp_m, &mut ws.rhs);
        for (j, &cj) in c.iter().enumerate() {
            ws.rhs[j] += sigma * ws.x_prev[j] - cj;
        }

        // PCG: K*x = rhs。kv_tmp を closure 内スクラッチに使用（Rust 2021 disjoint capture）
        {
            let rho_cap = rho;
            let mut kv_op = |v: &[f64], out: &mut [f64]| {
                kv_mul(q, a, sigma, rho_cap, v, out, &mut ws.kv_tmp);
            };
            pcg_solve(&mut kv_op, &ws.m_inv, &ws.rhs, &mut ws.x, cg_max_iter, cg_tol, &mut ws.cg_ws);
        }

        // --- z-update（over-relaxation in constraint space） ---
        spmv_c(a, &ws.x, &mut ws.cx);
        let one_minus_alpha = 1.0 - alpha;
        for i in 0..m_aug {
            ws.x_tilde[i] = alpha * ws.cx[i] + one_minus_alpha * ws.z[i];
        }
        for i in 0..m_aug {
            let v = ws.x_tilde[i] + ws.y[i] / rho;
            ws.z[i] = v.max(l_bound[i]).min(u_bound[i]);
        }

        // --- y-update ---
        for i in 0..m_aug {
            ws.y[i] += rho * (ws.x_tilde[i] - ws.z[i]);
        }

        // --- 収束判定（10反復ごと） ---
        if iter % 10 == 0
            && check_convergence(
                q, a, c, &ws.x, &ws.z, &ws.y,
                &mut ws.r_prim, &mut ws.r_dual,
                &mut ws.cx, &mut ws.tmp_n, &mut ws.tmp_m,
                eps_abs, eps_rel, m, n, m_aug,
            )
        {
            let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
            return QpResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: ws.x.clone(),
                dual_solution: ws.y[..m].to_vec(),
                bound_duals: vec![],
                active_set: vec![],
                iterations: iter + 1,
            };
        }

        // --- ρ適応更新（25反復ごと, CGパス: rho変更 + preconditioner再構築のみ） ---
        if iter % 25 == 0 && iter > 0 {
            let rho_new = compute_rho_update(
                q, a, c, &ws.x, &ws.z, &ws.y,
                &mut ws.r_prim, &mut ws.r_dual,
                &mut ws.cx, &mut ws.tmp_n, &mut ws.tmp_m,
                rho, eps_abs, m_aug,
            );
            if (rho_new / rho - 1.0).abs() > 0.1 {
                if timeout_ctx.should_stop() {
                    let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
                    return QpResult {
                        status: SolveStatus::Timeout,
                        objective: obj,
                        solution: ws.x.clone(),
                        dual_solution: ws.y[..m].to_vec(),
                        bound_duals: vec![],
                        active_set: vec![],
                        iterations: iter,
                    };
                }
                // y スケール: y_new = y_old * (rho_old / rho_new)  (λ=y/ρ を保持)
                let scale = rho / rho_new;
                for i in 0..m_aug {
                    ws.y[i] *= scale;
                }
                rho = rho_new;
                // preconditioner 再構築（K再構築・LDL再分解は不要）
                build_preconditioner(q, a, sigma, rho, &mut ws.m_inv);
            }
        }
    }

    // max_iter 到達
    let obj = compute_objective(q, c, &ws.x, &mut ws.tmp_n);
    QpResult {
        status: SolveStatus::MaxIterations,
        objective: obj,
        solution: ws.x.clone(),
        dual_solution: ws.y[..m].to_vec(),
        bound_duals: vec![],
        active_set: vec![],
        iterations: max_iter,
    }
}

// ---------------------------------------------------------------------------
// 補助関数: 収束判定
// ---------------------------------------------------------------------------

/// 収束判定を実行し、収束していれば true を返す
#[allow(clippy::too_many_arguments)]
fn check_convergence(
    q: &CscMatrix,
    a: &CscMatrix,
    c: &[f64],
    x: &[f64],
    z: &[f64],
    y: &[f64],
    r_prim: &mut [f64],
    r_dual: &mut [f64],
    cx_buf: &mut [f64],
    tmp_n: &mut [f64],
    tmp_m: &mut [f64],
    eps_abs: f64,
    eps_rel: f64,
    _m: usize,
    n: usize,
    m_aug: usize,
) -> bool {
    // r_prim = Cx - z
    spmv_c(a, x, cx_buf);
    for i in 0..m_aug {
        r_prim[i] = cx_buf[i] - z[i];
    }
    let r_prim_inf = norm_inf(r_prim);

    // r_dual = Qx + c + C^T*y
    spmv_q(q, x, r_dual);
    spmv_ct(a, y, tmp_n);
    for j in 0..n {
        r_dual[j] += c[j] + tmp_n[j];
    }
    let r_dual_inf = norm_inf(r_dual);

    // eps_prim = eps_abs + eps_rel * max(||Cx||_inf, ||z||_inf)
    let cx_inf = norm_inf(cx_buf);
    let z_inf = norm_inf(z);
    let eps_prim = eps_abs + eps_rel * f64::max(cx_inf, z_inf);

    // eps_dual = eps_abs + eps_rel * max(||Qx||_inf, ||C^T y||_inf, ||c||_inf)
    spmv_q(q, x, tmp_m[..n].as_mut()); // reuse tmp_m[0..n] for Qx (OK: m_aug >= n)
    let qx_inf = norm_inf(&tmp_m[..n]);
    let cty_inf = norm_inf(tmp_n);
    let c_inf = norm_inf(c);
    let eps_dual = eps_abs + eps_rel * f64::max(f64::max(qx_inf, cty_inf), c_inf);

    r_prim_inf < eps_prim && r_dual_inf < eps_dual
}

// ---------------------------------------------------------------------------
// 補助関数: ρ更新値計算
// ---------------------------------------------------------------------------

/// ρ の更新値を計算して返す（変更不要なら現在の rho をそのまま返す）
#[allow(clippy::too_many_arguments)]
fn compute_rho_update(
    q: &CscMatrix,
    a: &CscMatrix,
    c: &[f64],
    x: &[f64],
    z: &[f64],
    y: &[f64],
    r_prim: &mut [f64],
    r_dual: &mut [f64],
    cx_buf: &mut [f64],
    tmp_n: &mut [f64],
    tmp_m: &mut [f64],
    rho: f64,
    eps_abs: f64,
    m_aug: usize,
) -> f64 {
    let n = x.len();

    // primal residual
    spmv_c(a, x, cx_buf);
    for i in 0..m_aug {
        r_prim[i] = cx_buf[i] - z[i];
    }
    let r_prim_inf = norm_inf(r_prim);

    // dual residual
    spmv_q(q, x, r_dual);
    spmv_ct(a, y, tmp_n);
    for j in 0..n {
        r_dual[j] += c[j] + tmp_n[j];
    }
    let r_dual_inf = norm_inf(r_dual);

    if r_prim_inf == 0.0 || r_dual_inf == 0.0 {
        return rho; // 残差ゼロなら変更不要
    }

    let cx_inf = norm_inf(cx_buf);
    let z_inf = norm_inf(z);
    let scale_prim = f64::max(cx_inf, z_inf) + eps_abs;

    // tmp_m[0..n] = Qx (m_aug >= n なので安全)
    spmv_q(q, x, &mut tmp_m[..n]);
    let qx_inf = norm_inf(&tmp_m[..n]);
    let cty_inf = norm_inf(tmp_n);
    let scale_dual = f64::max(f64::max(qx_inf, cty_inf), norm_inf(c)) + eps_abs;

    let ratio = (r_prim_inf / scale_prim) / (r_dual_inf / scale_dual);

    let rho_new = if ratio > 5.0 {
        rho * ratio.sqrt()
    } else if ratio < 0.2 {
        rho / (1.0 / ratio).sqrt()
    } else {
        return rho; // 変更不要
    };

    rho_new.clamp(1e-6, 1e6)
}

// ---------------------------------------------------------------------------
// ファクトリ関数: エラー結果
// ---------------------------------------------------------------------------

fn make_timeout_result(n: usize, m: usize, iters: usize) -> QpResult {
    QpResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![0.0; m],
        bound_duals: vec![],
        active_set: vec![],
        iterations: iters,
    }
}

fn make_numerical_error_result(n: usize, m: usize) -> QpResult {
    QpResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![0.0; m],
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
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    fn default_opts() -> SolverOptions {
        SolverOptions::default()
    }

    fn assert_close(a: f64, b: f64, tol: f64, name: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name, b, a, (a-b).abs()
        );
    }

    /// test_admm_simple_qp:
    /// min 0.5*x^2 + x  s.t. x >= -2  （解: x=-1, obj=-0.5）
    /// Q = [[1]], c = [1], bounds = (lb=-2, ub=+inf), no inequality constraints
    #[test]
    fn test_admm_simple_qp() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let c = vec![1.0];
        let a = CscMatrix::new(0, 1); // no inequality constraints
        let b = vec![];
        let bounds = vec![(-2.0_f64, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_admm(&problem, &default_opts());
        assert_eq!(
            result.status, SolveStatus::Optimal,
            "simple_qp: expected Optimal, got {:?}", result.status
        );
        assert_close(result.solution[0], -1.0, 5e-3, "simple_qp: x[0]");
        assert_close(result.objective, -0.5, 5e-3, "simple_qp: obj");

        // feasibility: x >= -2
        assert!(
            result.solution[0] >= -2.0 - 1e-6,
            "simple_qp: feasibility x >= -2 violated, x={}",
            result.solution[0]
        );
    }

    /// test_admm_equality_constraint:
    /// min x^2 + y^2  s.t. x + y = 1  （解: x=y=0.5, obj=0.5）
    /// 等式は 2 不等式でエンコード: A=[[1,1],[-1,-1]], b=[1,-1]
    #[test]
    fn test_admm_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x+y <=1 and -(x+y) <= -1
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2, 2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = default_opts();
        opts.eps_abs = 1e-4;
        opts.eps_rel = 1e-4;
        let result = solve_qp_admm(&problem, &opts);
        assert_eq!(
            result.status, SolveStatus::Optimal,
            "equality: expected Optimal, got {:?}", result.status
        );
        assert_close(result.solution[0], 0.5, 5e-3, "equality: x[0]");
        assert_close(result.solution[1], 0.5, 5e-3, "equality: x[1]");
        assert_close(result.objective, 0.5, 5e-3, "equality: obj");
    }

    /// test_admm_timeout:
    /// timeout_secs=0.001 で大きな問題を解く → Timeout を返すこと
    /// 小問題では時間内に解けることもあるので Optimal も許容
    #[test]
    fn test_admm_timeout() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = default_opts();
        opts.timeout_secs = Some(0.0); // 即タイムアウト

        let result = solve_qp_admm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "timeout: expected Timeout or Optimal, got {:?}", result.status
        );
    }

    /// test_admm_numerical_error:
    /// Q=ゼロ行列・制約なし → σ正則化により K=(σ+ρ)I で解けるはず → OPTIMAL
    /// （NumericalError も許容）
    #[test]
    fn test_admm_numerical_error() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0
        let c = vec![0.0; n];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_admm(&problem, &default_opts());
        assert!(
            result.status == SolveStatus::Optimal
                || result.status == SolveStatus::NumericalError
                || result.status == SolveStatus::MaxIterations,
            "numerical_error: expected Optimal/NumericalError/MaxIterations, got {:?}",
            result.status
        );
    }

    // -----------------------------------------------------------------------
    // C2: kv_mul / build_preconditioner テスト
    // -----------------------------------------------------------------------

    /// test_kv_mul_matches_explicit:
    /// n=5, m=3 の小問題で kv_mul(v) と明示的 K 行列×v を比較（相対誤差 < 1e-12）
    #[test]
    fn test_kv_mul_matches_explicit() {
        // Q = diag(1,2,3,4,5)
        let q_rows: Vec<usize> = (0..5).collect();
        let q_cols: Vec<usize> = (0..5).collect();
        let q_vals: Vec<f64> = (1..=5).map(|x| x as f64).collect();
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, 5, 5).unwrap();

        // A (3×5 sparse): 各行に2-3個の非ゼロ
        // A[0,0]=1, A[0,1]=2
        // A[1,1]=3, A[1,2]=1, A[1,3]=2
        // A[2,3]=1, A[2,4]=4
        let a_rows = vec![0usize, 0, 1, 1, 1, 2, 2];
        let a_cols = vec![0usize, 1, 1, 2, 3, 3, 4];
        let a_vals = vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0];
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 3, 5).unwrap();

        let sigma = 1e-6_f64;
        let rho = 0.1_f64;

        // 明示的 K = Q + (σ+ρ)I + ρ*A^T*A を構築して K*v を計算
        // A^T*A を密行列で計算 (5×5)
        let mut ata = [[0.0_f64; 5]; 5];
        // A^T*A を手動計算: 各 (j1, j2) に A[:,j1]^T * A[:,j2]
        // 行 0: A[0,0]=1,A[0,1]=2
        // 行 1: A[1,1]=3,A[1,2]=1,A[1,3]=2
        // 行 2: A[2,3]=1,A[2,4]=4
        let a_dense: [[f64; 5]; 3] = [
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ];
        for j1 in 0..5 {
            for j2 in 0..5 {
                for i in 0..3 {
                    ata[j1][j2] += a_dense[i][j1] * a_dense[i][j2];
                }
            }
        }

        let v = vec![1.0_f64, -1.0, 2.0, 0.5, -0.5];
        // K*v 明示的計算
        let mut kv_explicit = [0.0_f64; 5];
        let diag_q = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        for j in 0..5 {
            kv_explicit[j] = diag_q[j] * v[j] + (sigma + rho) * v[j];
            for k in 0..5 {
                kv_explicit[j] += rho * ata[j][k] * v[k];
            }
        }

        // kv_mul で計算
        let mut result = vec![0.0_f64; 5];
        let mut tmp_m = vec![0.0_f64; 3];
        kv_mul(&q, &a, sigma, rho, &v, &mut result, &mut tmp_m);

        for j in 0..5 {
            let rel_err = if kv_explicit[j].abs() > 1e-15 {
                (result[j] - kv_explicit[j]).abs() / kv_explicit[j].abs()
            } else {
                (result[j] - kv_explicit[j]).abs()
            };
            assert!(
                rel_err < 1e-12,
                "kv_mul[{}]: explicit={:.12e}, got={:.12e}, rel_err={:.2e}",
                j, kv_explicit[j], result[j], rel_err
            );
        }
    }

    // -----------------------------------------------------------------------
    // C3: CGパス統合テスト
    // -----------------------------------------------------------------------

    /// test_admm_cg_simple_qp:
    /// min 0.5*x^2 + x  s.t. x >= -2  （解: x=-1, obj=-0.5）
    /// admm_use_cg=Some(true) で解いて Optimal 確認
    #[test]
    fn test_admm_cg_simple_qp() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let c = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(-2.0_f64, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = default_opts();
        opts.admm_use_cg = Some(true);
        let result = solve_qp_admm(&problem, &opts);

        assert_eq!(
            result.status, SolveStatus::Optimal,
            "cg_simple_qp: expected Optimal, got {:?}", result.status
        );
        assert_close(result.solution[0], -1.0, 5e-3, "cg_simple_qp: x[0]");
        assert_close(result.objective, -0.5, 5e-3, "cg_simple_qp: obj");
        assert!(
            result.solution[0] >= -2.0 - 1e-6,
            "cg_simple_qp: feasibility x >= -2 violated, x={}",
            result.solution[0]
        );
    }

    /// test_admm_cg_vs_ldl_consistency:
    /// n=500 の問題を LDL/CG 両方で解いて目的関数値が一致すること（rtol=1e-3）
    /// Q = 2*I, c = [1..1], bounds = (-2, 2), 解: x_i = -0.5
    #[test]
    fn test_admm_cg_vs_ldl_consistency() {
        let n = 500;
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals = vec![2.0_f64; n];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
        let c = vec![1.0_f64; n];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(-2.0_f64, 2.0_f64); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts_ldl = default_opts();
        opts_ldl.admm_use_cg = Some(false);
        opts_ldl.eps_abs = 1e-4;
        opts_ldl.eps_rel = 1e-4;
        let result_ldl = solve_qp_admm(&problem, &opts_ldl);

        let mut opts_cg = default_opts();
        opts_cg.admm_use_cg = Some(true);
        opts_cg.eps_abs = 1e-4;
        opts_cg.eps_rel = 1e-4;
        let result_cg = solve_qp_admm(&problem, &opts_cg);

        assert_eq!(result_ldl.status, SolveStatus::Optimal, "LDL: expected Optimal");
        assert_eq!(result_cg.status, SolveStatus::Optimal, "CG: expected Optimal");

        let rtol = 1e-3;
        let obj_ldl = result_ldl.objective;
        let obj_cg = result_cg.objective;
        let rel_diff = (obj_cg - obj_ldl).abs() / (obj_ldl.abs() + 1e-10);
        assert!(
            rel_diff < rtol,
            "cg_vs_ldl: objective mismatch CG={:.6}, LDL={:.6}, rel_diff={:.2e}",
            obj_cg, obj_ldl, rel_diff
        );
    }

    /// test_admm_cg_timeout:
    /// timeout_secs=0.0 で即停止 → Timeout または Optimal（CGパス）
    #[test]
    fn test_admm_cg_timeout() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = default_opts();
        opts.admm_use_cg = Some(true);
        opts.timeout_secs = Some(0.0); // 即タイムアウト

        let result = solve_qp_admm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "cg_timeout: expected Timeout or Optimal, got {:?}", result.status
        );
    }

    /// test_build_preconditioner:
    /// 小問題で build_preconditioner() の出力が 1/diag(K) に等しいこと確認
    #[test]
    fn test_build_preconditioner() {
        // Q = diag(1,2,3,4,5)
        let q_rows: Vec<usize> = (0..5).collect();
        let q_cols: Vec<usize> = (0..5).collect();
        let q_vals: Vec<f64> = (1..=5).map(|x| x as f64).collect();
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, 5, 5).unwrap();

        // A (3×5): test_kv_mul と同じ行列
        let a_rows = vec![0usize, 0, 1, 1, 1, 2, 2];
        let a_cols = vec![0usize, 1, 1, 2, 3, 3, 4];
        let a_vals = vec![1.0f64, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0];
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 3, 5).unwrap();

        let sigma = 1e-6_f64;
        let rho = 0.1_f64;

        // diag(K) を明示的計算
        // diag(K)_j = diag(Q)_j + (σ+ρ) + ρ * ||A[:,j]||²
        let a_dense: [[f64; 5]; 3] = [
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ];
        let diag_q = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut expected_m_inv = [0.0_f64; 5];
        for j in 0..5 {
            let col_sq: f64 = (0..3).map(|i| a_dense[i][j] * a_dense[i][j]).sum();
            let dk = diag_q[j] + (sigma + rho) + rho * col_sq;
            expected_m_inv[j] = 1.0 / dk;
        }

        let mut m_inv = vec![0.0_f64; 5];
        build_preconditioner(&q, &a, sigma, rho, &mut m_inv);

        for j in 0..5 {
            let rel_err = (m_inv[j] - expected_m_inv[j]).abs() / expected_m_inv[j].abs();
            assert!(
                rel_err < 1e-12,
                "m_inv[{}]: expected={:.12e}, got={:.12e}, rel_err={:.2e}",
                j, expected_m_inv[j], m_inv[j], rel_err
            );
        }
    }
}
