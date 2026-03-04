//! Preconditioned Conjugate Gradient (PCG) solver
//!
//! K を明示的に構築せず、closure で K*v 演算を提供することで
//! Matrix-Free な線形ソルバーを実現する。
//! 線形系 K*x = rhs の Matrix-Free 求解に使用する。

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// ワークスペース・結果型
// ---------------------------------------------------------------------------

/// PCG ソルバーの作業バッファ（毎反復で使い回す）
pub struct CgWorkspace {
    pub r: Vec<f64>,  // residual: rhs - K*x
    pub z: Vec<f64>,  // preconditioned residual: M^{-1} * r
    pub p: Vec<f64>,  // search direction
    pub kp: Vec<f64>, // K*p
}

impl CgWorkspace {
    /// 長さ n のゼロ初期化バッファを確保する
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            kp: vec![0.0; n],
        }
    }
}

/// PCG の実行結果
pub struct CgResult {
    pub iterations: usize,
    pub residual_norm: f64, // ||r||_∞ at termination
    pub converged: bool,
    pub timed_out: bool, // deadline/cancel により打ち切られた場合 true
}

// ---------------------------------------------------------------------------
// PCG ソルバー本体
// ---------------------------------------------------------------------------

/// Preconditioned CG: solve K*x = rhs
///
/// # 引数
/// - `kv_op`: K*v 演算 closure。`kv_op(v, out)` で `out = K*v` を上書きする。
/// - `m_inv`: 対角前処理の逆数ベクトル `1/diag(K)`（Jacobi preconditioner）。
/// - `rhs`: 右辺ベクトル（長さ n）。
/// - `x`: 初期解（長さ n）。解で上書きされる。
/// - `max_iter`: 最大反復数。
/// - `tol`: 収束判定: `||r||_∞ < tol`。
/// - `ws`: 再利用可能なワークスペース（長さ n）。
/// - `deadline`: タイムアウト期限（None = 無制限）。10反復ごとにチェック。
/// - `cancel`: キャンセルフラグ（None = 無効）。10反復ごとにチェック。
///
/// # アルゴリズム: Preconditioned CG (Polak-Ribière)
/// ```text
/// r = rhs - K*x
/// z = M^{-1} * r
/// p = z
/// for k = 0..max_iter:
///     kp = K*p
///     alpha = (r·z) / (p·kp)
///     x += alpha * p
///     r -= alpha * kp
///     if ||r||_∞ < tol: converged
///     z_new = M^{-1} * r
///     beta = (r·z_new) / (r·z)   -- Polak-Ribière: 分子を更新後 z_new で計算
///     p = z_new + beta * p
///     z = z_new
/// ```
// GPU移行設計 §4.3 G3準拠: インデックスループを明示的 for で記述する
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn pcg_solve<F>(
    kv_op: &mut F,
    m_inv: &[f64],
    rhs: &[f64],
    x: &mut [f64],
    max_iter: usize,
    tol: f64,
    ws: &mut CgWorkspace,
    deadline: Option<Instant>,
    cancel: Option<&AtomicBool>,
) -> CgResult
where
    F: FnMut(&[f64], &mut [f64]),
{
    let n = rhs.len();
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(m_inv.len(), n);
    debug_assert_eq!(ws.r.len(), n);

    // r = rhs - K*x
    kv_op(x, &mut ws.kp); // kp = K*x (temporary)
    for j in 0..n {
        ws.r[j] = rhs[j] - ws.kp[j];
    }

    // z = M^{-1} * r
    for j in 0..n {
        ws.z[j] = m_inv[j] * ws.r[j];
    }

    // p = z
    ws.p.copy_from_slice(&ws.z);

    let mut rz = dot(&ws.r, &ws.z);

    for k in 0..max_iter {
        // kp = K*p
        kv_op(&ws.p, &mut ws.kp);

        // alpha = (r·z) / (p·kp)
        // 10反復ごとにdeadline/cancelチェック
        if k % 10 == 0 && k > 0 {
            let timed = deadline.is_some_and(|d| Instant::now() >= d)
                || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
            if timed {
                let res = norm_inf(&ws.r);
                return CgResult { iterations: k, residual_norm: res, converged: false, timed_out: true };
            }
        }

        let pkp = dot(&ws.p, &ws.kp);
        if pkp.abs() < f64::EPSILON {
            // 退化: 収束とみなす
            let res = norm_inf(&ws.r);
            return CgResult { iterations: k, residual_norm: res, converged: res < tol, timed_out: false };
        }
        let alpha = rz / pkp;

        // x += alpha * p
        for j in 0..n {
            x[j] += alpha * ws.p[j];
        }
        // r -= alpha * kp
        for j in 0..n {
            ws.r[j] -= alpha * ws.kp[j];
        }

        let res = norm_inf(&ws.r);
        if res < tol {
            return CgResult { iterations: k + 1, residual_norm: res, converged: true, timed_out: false };
        }

        // z_new = M^{-1} * r  (z フィールドを z_new として再利用)
        for j in 0..n {
            ws.z[j] = m_inv[j] * ws.r[j];
        }

        // beta = (r·z_new) / rz  (Polak-Ribière)
        let rz_new = dot(&ws.r, &ws.z);
        let beta = if rz.abs() < f64::EPSILON { 0.0 } else { rz_new / rz };
        rz = rz_new;

        // p = z_new + beta * p
        for j in 0..n {
            ws.p[j] = ws.z[j] + beta * ws.p[j];
        }
    }

    let res = norm_inf(&ws.r);
    CgResult { iterations: max_iter, residual_norm: res, converged: false, timed_out: false }
}

// ---------------------------------------------------------------------------
// 内積・ノルム補助関数
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
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diag_kv(diag: Vec<f64>) -> impl Fn(&[f64], &mut [f64]) {
        move |v: &[f64], out: &mut [f64]| {
            for (j, (&dj, &vj)) in diag.iter().zip(v.iter()).enumerate() {
                out[j] = dj * vj;
            }
        }
    }

    /// test_cg_diagonal_3x3:
    /// K = diag(1.0, 2.0, 3.0), rhs = [1.0, 2.0, 3.0]
    /// x* = [1.0, 1.0, 1.0] → 3反復以内に収束すること
    #[test]
    fn test_cg_diagonal_3x3() {
        let diag = vec![1.0_f64, 2.0, 3.0];
        let m_inv: Vec<f64> = diag.iter().map(|&d| 1.0 / d).collect();
        let mut kv = make_diag_kv(diag);
        let rhs = vec![1.0_f64, 2.0, 3.0];
        let mut x = vec![0.0_f64; 3];
        let mut ws = CgWorkspace::new(3);

        let result = pcg_solve(&mut kv, &m_inv, &rhs, &mut x, 10, 1e-10, &mut ws, None, None);

        assert!(result.converged, "3x3 diagonal: not converged in {} iters", result.iterations);
        assert!(
            result.iterations <= 3,
            "3x3 diagonal: expected ≤3 iters, got {}",
            result.iterations
        );
        for (j, (&xj, &expected)) in x.iter().zip([1.0, 1.0, 1.0].iter()).enumerate() {
            assert!(
                (xj - expected).abs() < 1e-8,
                "x[{}]: expected {}, got {} (diff={:.2e})",
                j, expected, xj, (xj - expected).abs()
            );
        }
    }

    /// test_cg_sparse_spd:
    /// 5x5 対角優位 SPD 行列 K = diag(5,6,7,8,9) + off-diag terms
    /// K[i][i] = 5+i, K[i][i+1] = K[i+1][i] = 0.5 (tridiagonal)
    /// rhs = [1, 2, 3, 4, 5]
    /// 収束確認（反復数 ≤ 20）
    #[test]
    fn test_cg_sparse_spd() {
        // tridiagonal: K[i,i]=5+i, K[i,i±1]=0.5
        let mut k_mat = |v: &[f64], out: &mut [f64]| {
            let n = v.len();
            for i in 0..n {
                let diag = (5 + i) as f64;
                out[i] = diag * v[i];
                if i > 0 { out[i] += 0.5 * v[i - 1]; }
                if i < n - 1 { out[i] += 0.5 * v[i + 1]; }
            }
        };
        // diagonal preconditioner: 1/K[i,i]
        let m_inv: Vec<f64> = (0..5).map(|i| 1.0 / (5 + i) as f64).collect();
        let rhs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut x = vec![0.0_f64; 5];
        let mut ws = CgWorkspace::new(5);

        let result = pcg_solve(&mut k_mat, &m_inv, &rhs, &mut x, 20, 1e-10, &mut ws, None, None);

        assert!(result.converged, "5x5 SPD: not converged in {} iters", result.iterations);

        // 解の検証: K*x ≈ rhs
        let mut kx = vec![0.0_f64; 5];
        k_mat(&x, &mut kx);
        for (i, (&kxi, &bi)) in kx.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (kxi - bi).abs() < 1e-8,
                "(K*x)[{}]={:.8} ≠ rhs[{}]={:.8}",
                i, kxi, i, bi
            );
        }
    }

    /// test_cg_preconditioner_effect:
    /// 同じ問題を前処理あり/なしで解いて、前処理あり ≤ 前処理なしの反復数を確認
    #[test]
    fn test_cg_preconditioner_effect() {
        // ill-conditioned diagonal: diag(1, 100, 10000)
        let diag = vec![1.0_f64, 100.0, 10_000.0];
        let rhs = vec![1.0_f64, 100.0, 10_000.0];

        let mut kv = make_diag_kv(diag.clone());

        // 前処理なし: m_inv = [1, 1, 1]
        let m_inv_none = vec![1.0_f64; 3];
        let mut x_none = vec![0.0_f64; 3];
        let mut ws_none = CgWorkspace::new(3);
        let result_none = pcg_solve(&mut kv, &m_inv_none, &rhs, &mut x_none, 100, 1e-10, &mut ws_none, None, None);

        // 前処理あり: m_inv = 1/diag(K)
        let m_inv_diag: Vec<f64> = diag.iter().map(|&d| 1.0 / d).collect();
        let mut x_prec = vec![0.0_f64; 3];
        let mut ws_prec = CgWorkspace::new(3);
        let result_prec = pcg_solve(&mut kv, &m_inv_diag, &rhs, &mut x_prec, 100, 1e-10, &mut ws_prec, None, None);

        assert!(result_prec.converged, "preconditioned: not converged");
        assert!(result_none.converged, "unpreconditioned: not converged");
        assert!(
            result_prec.iterations <= result_none.iterations,
            "preconditioned({}) should require ≤ unpreconditioned({}) iterations",
            result_prec.iterations, result_none.iterations
        );
    }
}
