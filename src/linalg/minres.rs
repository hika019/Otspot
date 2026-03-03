//! Preconditioned MINRES (Minimum Residual method, Paige-Saunders 1975)
//!
//! 対称不定値系 Ax = b を解くKrylov法。
//! CGと異なり、A が不定値でも（固有値に正負が混在していても）適用可能。
//! augmented KKT system K = [Q+δI, A^T; A, -D] の求解に使用する。
//!
//! ## アルゴリズム
//!
//! 前処理 M (SPD) の M^{-1}-内積で Lanczos を実行し、
//! 生成される (k+1)×k 三対角行列を Givens QR で逐次因子分解して解を更新する。
//!
//! ## 実装参考
//! Stanford MINRES.m (Paige & Saunders) / scipy.sparse.linalg.minres
//! 変数名は基本的に scipy の命名規約に従う:
//!   r2 = 物理空間の非正規化 Lanczos ベクトル
//!   y  = M^{-1} r2 (前処理済み)
//!   v  = y/beta = M^{-1} v_k (z_k)
//!   alfa = alpha_k = v' A v
//!   beta = M^{-1}-ノルム(r2) = beta_{k}

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// ワークスペース・結果型
// ---------------------------------------------------------------------------

/// MINRES ソルバーの作業バッファ（毎反復で使い回す）
pub struct MinresWorkspace {
    pub v_prev: Vec<f64>, // r1: 前ステップの非正規化 Lanczos ベクトル
    pub v_curr: Vec<f64>, // r2: 現ステップの非正規化 Lanczos ベクトル
    pub w_prev: Vec<f64>, // w_{k-2}: 解更新ベクトル (scipy w2)
    pub w_curr: Vec<f64>, // w_{k-1}: 解更新ベクトル (scipy w)
    pub av: Vec<f64>,     // 一時バッファ（A*v 計算・w_new 格納兼用）
    pub z: Vec<f64>,      // y = M^{-1} r2
    pub tmp: Vec<f64>,    // v = y/beta = M^{-1} v_k
}

impl MinresWorkspace {
    /// 長さ n のゼロ初期化バッファを確保する
    pub fn new(n: usize) -> Self {
        Self {
            v_prev: vec![0.0; n],
            v_curr: vec![0.0; n],
            w_prev: vec![0.0; n],
            w_curr: vec![0.0; n],
            av: vec![0.0; n],
            z: vec![0.0; n],
            tmp: vec![0.0; n],
        }
    }
}

/// MINRES の実行結果
pub struct MinresResult {
    pub iterations: usize,
    pub residual_norm: f64, // ||r||_∞ at termination
    pub converged: bool,
    pub timed_out: bool, // deadline/cancel により打ち切られた場合 true
}

// ---------------------------------------------------------------------------
// 補助関数
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

// ---------------------------------------------------------------------------
// MINRES ソルバー本体
// ---------------------------------------------------------------------------

/// Preconditioned MINRES: solve A*x = rhs (A: 対称, M: SPD前処理)
///
/// # 引数
/// - `kv_op`: A*v 演算 closure。`kv_op(v, out)` で `out = A*v` を上書きする。
/// - `precond_op`: M^{-1}*v 演算 closure。`precond_op(v, out)` で `out = M^{-1}*v` を上書き。
/// - `rhs`: 右辺ベクトル（長さ n）。
/// - `x`: 初期解（長さ n）。解で上書きされる。
/// - `max_iter`: 最大反復数。
/// - `tol`: 収束判定: `||r||_∞ < tol`。
/// - `ws`: 再利用可能なワークスペース（長さ n）。
/// - `deadline`: タイムアウト期限（None = 無制限）。10反復ごとにチェック。
/// - `cancel`: キャンセルフラグ（None = 無効）。10反復ごとにチェック。
///
/// # アルゴリズム概要
/// M^{-1}-内積 Lanczos + 逐次 Givens QR 因子分解 (Paige-Saunders 1975)
///
/// 変数 phibar は残差ノルムの近似値。
/// 収束判定は10反復ごとに直接 ||rhs - A*x||_∞ を計算する。
// GPU移行設計 §4.3 G3準拠: インデックスループを明示的 for で記述する
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn pminres_solve<F, P>(
    kv_op: &mut F,
    precond_op: &mut P,
    rhs: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
    ws: &mut MinresWorkspace,
    deadline: Option<Instant>,
    cancel: Option<&AtomicBool>,
) -> MinresResult
where
    F: FnMut(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    let n = rhs.len();
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(ws.v_curr.len(), n);

    // --- 初期化 ---
    // r2 = rhs - A*x0
    {
        let x_ref: &[f64] = x;
        kv_op(x_ref, &mut ws.av);
    }
    for j in 0..n {
        ws.v_curr[j] = rhs[j] - ws.av[j];
    }

    // y = M^{-1} r2
    {
        let r2: &[f64] = &ws.v_curr;
        precond_op(r2, &mut ws.z);
    }

    // beta1 = sqrt(r2 . y)  (M^{-1}-ノルム)
    let beta1 = dot(&ws.v_curr, &ws.z).abs().sqrt();

    // 初期残差確認 (= ||v_curr||_∞ = ||rhs - A*x||_∞)
    let init_res = norm_inf(&ws.v_curr);
    if init_res < tol {
        return MinresResult { iterations: 0, residual_norm: init_res, converged: true, timed_out: false };
    }
    if beta1 < f64::EPSILON {
        return MinresResult { iterations: 0, residual_norm: init_res, converged: false, timed_out: false };
    }

    // QR 状態変数 (scipy の命名規約)
    let mut beta = beta1;
    let mut oldb = 0.0_f64;
    let mut dbar = 0.0_f64;
    let mut epsln = 0.0_f64;
    let mut phibar = beta1;
    let mut cs = -1.0_f64; // 前ステップの cos
    let mut sn = 0.0_f64;  // 前ステップの sin

    // ベクトル初期化
    for j in 0..n {
        ws.v_prev[j] = 0.0;
        ws.w_prev[j] = 0.0;
        ws.w_curr[j] = 0.0;
    }

    let mut last_res = init_res;

    // --- メイン反復ループ ---
    for k in 0..max_iter {
        // =========================================================
        // Lanczos ステップ (scipy の v, y, alfa, r2, beta 計算と同じ)
        // =========================================================
        let s = 1.0 / beta;

        // tmp = v = y / beta = M^{-1} v_k  (= z_k)
        for j in 0..n {
            ws.tmp[j] = s * ws.z[j];
        }

        // av = A * tmp  (= A * M^{-1} * v_k = A z_k)
        {
            let v: &[f64] = &ws.tmp;
            kv_op(v, &mut ws.av);
        }

        // 3項漸化式: av -= (beta_k / beta_{k-1}) * r1
        if k >= 1 && oldb.abs() > f64::EPSILON {
            let ratio = beta / oldb;
            for j in 0..n {
                ws.av[j] -= ratio * ws.v_prev[j];
            }
        }

        // alfa = v' av = alpha_k
        let alfa = dot(&ws.tmp, &ws.av);

        // av -= (alfa / beta) * r2  (= alfa * v_k を引く)
        {
            let r = alfa / beta;
            for j in 0..n {
                ws.av[j] -= r * ws.v_curr[j];
            }
        }

        // r1 ← r2 (swap), r2 ← av (新 Lanczos ベクトル)
        std::mem::swap(&mut ws.v_prev, &mut ws.v_curr);
        ws.v_curr.copy_from_slice(&ws.av);

        // y = M^{-1} r2
        {
            let r2: &[f64] = &ws.v_curr;
            precond_op(r2, &mut ws.z);
        }

        // beta ← sqrt(r2 . y)
        oldb = beta;
        let rz = dot(&ws.v_curr, &ws.z);
        beta = rz.abs().sqrt();

        // =========================================================
        // QR ステップ: Givens 回転で三対角行列を上三角化
        // =========================================================
        // 前ステップの Givens (cs, sn) を新カラムに適用
        let epsln_next = sn * beta;
        let dbar_next = -(cs * beta);

        // [delta; gbar] = [cs, sn; -sn, cs] * [dbar; alfa]
        let delta = cs * dbar + sn * alfa;
        let gbar  = sn * dbar - cs * alfa;

        // 新 Givens 回転: [cs_new, sn_new; -sn_new, cs_new] * [gbar; beta] = [gamma; 0]
        let gamma = gbar.hypot(beta);
        let (cs_new, sn_new) = if gamma < f64::EPSILON {
            (-1.0_f64, 0.0_f64)
        } else {
            (gbar / gamma, beta / gamma)
        };

        // 右辺ベクトルを更新
        let phi  = cs_new * phibar;
        phibar  *= sn_new;

        // =========================================================
        // 解更新: x += phi * w_new
        // w_new = (tmp - epsln * w_{k-2} - delta * w_{k-1}) / gamma
        // =========================================================
        let denom = if gamma.abs() > f64::EPSILON { 1.0 / gamma } else { 0.0 };
        // av を w_new の一時バッファとして使用
        for j in 0..n {
            ws.av[j] = (ws.tmp[j] - epsln * ws.w_prev[j] - delta * ws.w_curr[j]) * denom;
            x[j] += phi * ws.av[j];
        }
        // w_{k-2} ← w_{k-1}, w_{k-1} ← w_new
        std::mem::swap(&mut ws.w_prev, &mut ws.w_curr);
        ws.w_curr.copy_from_slice(&ws.av);

        // QR 状態を次ステップへ
        epsln = epsln_next;
        dbar  = dbar_next;
        cs    = cs_new;
        sn    = sn_new;

        // =========================================================
        // 収束判定 (10反復ごと + Lanczos 終了時)
        // =========================================================
        if k % 10 == 0 {
            // deadline/cancel チェック
            if k > 0 {
                let timed = deadline.is_some_and(|d| Instant::now() >= d)
                    || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
                if timed {
                    return MinresResult {
                        iterations: k,
                        residual_norm: last_res,
                        converged: false,
                        timed_out: true,
                    };
                }
            }
            // 直接残差 ||rhs - A*x||_∞ を計算
            {
                let x_ref: &[f64] = x;
                kv_op(x_ref, &mut ws.av);
            }
            let res = ws.av.iter().zip(rhs.iter()).fold(0.0_f64, |acc, (&ax, &b)| acc.max((b - ax).abs()));
            last_res = res;
            if res < tol {
                return MinresResult { iterations: k + 1, residual_norm: res, converged: true, timed_out: false };
            }
        }

        // Lanczos 終了 (beta = 0 → 解は Krylov 空間内に存在)
        if beta < f64::EPSILON {
            let x_ref: &[f64] = x;
            kv_op(x_ref, &mut ws.av);
            let res = ws.av.iter().zip(rhs.iter()).fold(0.0_f64, |acc, (&ax, &b)| acc.max((b - ax).abs()));
            return MinresResult { iterations: k + 1, residual_norm: res, converged: res < tol, timed_out: false };
        }

        let _ = phibar; // suppress unused warning
    }

    // max_iter 到達
    {
        let x_ref: &[f64] = x;
        kv_op(x_ref, &mut ws.av);
    }
    let res = ws.av.iter().zip(rhs.iter()).fold(0.0_f64, |acc, (&ax, &b)| acc.max((b - ax).abs()));
    MinresResult { iterations: max_iter, residual_norm: res, converged: false, timed_out: false }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_matvec(diag: &[f64], v: &[f64], out: &mut [f64]) {
        for (j, (&dj, &vj)) in diag.iter().zip(v.iter()).enumerate() {
            out[j] = dj * vj;
        }
    }

    fn identity_precond(v: &[f64], out: &mut [f64]) {
        out.copy_from_slice(v);
    }

    /// test_minres_diagonal_spd:
    /// A = diag(1, 2, 3), rhs = [1, 2, 3], x* = [1, 1, 1]
    /// 対角 SPD 行列に対して MINRES が CG と同等の精度で収束すること
    #[test]
    fn test_minres_diagonal_spd() {
        let diag = vec![1.0_f64, 2.0, 3.0];
        let rhs  = vec![1.0_f64, 2.0, 3.0];
        let mut x  = vec![0.0_f64; 3];
        let mut ws = MinresWorkspace::new(3);

        let d = diag.clone();
        let mut kv      = |v: &[f64], out: &mut [f64]| diag_matvec(&d, v, out);
        let mut precond = identity_precond;

        let result = pminres_solve(&mut kv, &mut precond, &rhs, &mut x, 100, 1e-10, &mut ws, None, None);

        assert!(
            result.converged,
            "対角SPD: 収束しなかった (iters={}, residual={:.2e})",
            result.iterations, result.residual_norm
        );
        for (j, (&xj, &expected)) in x.iter().zip([1.0_f64, 1.0, 1.0].iter()).enumerate() {
            assert!(
                (xj - expected).abs() < 1e-6,
                "x[{}]: expected {}, got {} (diff={:.2e})",
                j, expected, xj, (xj - expected).abs()
            );
        }
    }

    /// test_minres_indefinite:
    /// A = diag(1, 1, -1, -1): 不定値対称行列（固有値に正負混在）
    /// CG では発散するが MINRES は収束すること
    /// rhs = [2, 3, -4, -5], x* = [2, 3, 4, 5]
    #[test]
    fn test_minres_indefinite() {
        let diag = vec![1.0_f64, 1.0, -1.0, -1.0];
        let rhs  = vec![2.0_f64, 3.0, -4.0, -5.0]; // A x* = [2, 3, 4, 5] で x*=[2,3,4,5]
        let mut x  = vec![0.0_f64; 4];
        let mut ws = MinresWorkspace::new(4);

        let d = diag.clone();
        let mut kv = |v: &[f64], out: &mut [f64]| diag_matvec(&d, v, out);

        // M = I (identity 前処理; 不定値A に SPD M が使える)
        let mut precond = identity_precond;

        let result = pminres_solve(&mut kv, &mut precond, &rhs, &mut x, 200, 1e-8, &mut ws, None, None);

        assert!(
            result.converged,
            "不定値対称: 収束しなかった (iters={}, residual={:.2e})",
            result.iterations, result.residual_norm
        );

        // 解の検証: A*x ≈ rhs
        let expected = vec![2.0_f64, 3.0, 4.0, 5.0];
        for (i, (&xi, &ei)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (xi - ei).abs() < 1e-6,
                "x[{}]={:.8} ≠ expected[{}]={:.8}",
                i, xi, i, ei
            );
        }
    }

    /// test_minres_preconditioner_effect:
    /// 条件数の悪い対角行列で前処理効果を確認
    /// 対角前処理あり ≤ identity 前処理なしの反復数を期待
    #[test]
    fn test_minres_preconditioner_effect() {
        // A = diag(1, 10, 100), rhs = [1, 10, 100], x* = [1, 1, 1]
        let diag = vec![1.0_f64, 10.0, 100.0];
        let rhs  = vec![1.0_f64, 10.0, 100.0];

        // 前処理なし (identity)
        let d1 = diag.clone();
        let mut kv1 = |v: &[f64], out: &mut [f64]| diag_matvec(&d1, v, out);
        let mut precond_id = identity_precond;
        let mut x_none = vec![0.0_f64; 3];
        let mut ws_none = MinresWorkspace::new(3);
        let result_none = pminres_solve(
            &mut kv1, &mut precond_id, &rhs, &mut x_none, 200, 1e-10, &mut ws_none, None, None,
        );

        // 対角前処理あり (M^{-1} = diag(1, 1/10, 1/100))
        let d2   = diag.clone();
        let minv = diag.iter().map(|&d| 1.0 / d).collect::<Vec<_>>();
        let mut kv2 = |v: &[f64], out: &mut [f64]| diag_matvec(&d2, v, out);
        let mut precond_diag = |v: &[f64], out: &mut [f64]| diag_matvec(&minv, v, out);
        let mut x_prec = vec![0.0_f64; 3];
        let mut ws_prec = MinresWorkspace::new(3);
        let result_prec = pminres_solve(
            &mut kv2, &mut precond_diag, &rhs, &mut x_prec, 200, 1e-10, &mut ws_prec, None, None,
        );

        assert!(
            result_none.converged,
            "前処理なし: 収束しなかった (iters={}, res={:.2e})",
            result_none.iterations, result_none.residual_norm
        );
        assert!(
            result_prec.converged,
            "前処理あり: 収束しなかった (iters={}, res={:.2e})",
            result_prec.iterations, result_prec.residual_norm
        );
        assert!(
            result_prec.iterations <= result_none.iterations,
            "前処理あり({}) ≤ 前処理なし({}) を期待",
            result_prec.iterations, result_none.iterations
        );
    }
}
