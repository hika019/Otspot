//! MINRES (Paige-Saunders 1975) for symmetric (possibly indefinite) `K · u = rhs`.
//!
//! IPM/IPPMM の鞍点 KKT 系 K = [Q+Σ_x, A^T; A, -Σ_y] は対称だが不定値で、CG
//! (= 正定値専用) は使えない。MINRES は対称不定値で動き、各反復は 1 回の
//! `K · v` 行列ベクトル積のみ。メモリは O(n) で fill-in を起こさない。
//!
//! 実装は SciPy `scipy.sparse.linalg.minres` (= Stanford SOL minres.m) と同じ
//! Givens-recurrence 構造を採用する。preconditioner 引数は API として用意して
//! あるが、本コミットでは恒等変換 (M = I) のみ呼ばれることを前提とする (動作する
//! 前処理は次コミットで追加)。

use crate::sparse::CscMatrix;

/// MINRES 結果統計
#[derive(Debug, Clone, Copy)]
pub struct MinresStats {
    /// 実行した反復数
    pub iters: usize,
    /// Givens recurrence による残差ノルム ||r_k||_2 推定
    pub residual_estimate: f64,
    /// 収束したか (`residual_estimate <= tol * ||b||_2`)
    pub converged: bool,
}

/// 対称行列 K を持つ系 `K · x = b` を MINRES で解く。
///
/// * `matvec`     : closure `y ← K · v` (1 反復 1 回呼ばれる)
/// * `precond`    : closure `z ← M^{-1} · r` (M は SPD 前処理)。M = I なら
///   `|r, z| z.copy_from_slice(r)` を渡す
/// * `b`          : 右辺
/// * `x`          : 初期推定 (in) / 解 (out)。0 ベクトル可
/// * `tol`        : 相対許容差 `||r||_2 / ||b||_2 ≤ tol` で停止
/// * `max_iter`   : 最大反復数
/// * `should_stop`: 反復毎にチェック、`true` を返したら break (deadline / cancel 用)
///
/// 戻り値: 反復統計 (収束したか / 反復数 / 残差推定)。
pub fn pminres<MV, PV, SS>(
    matvec: MV,
    precond: PV,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iter: usize,
    mut should_stop: SS,
) -> MinresStats
where
    MV: Fn(&[f64], &mut [f64]),
    PV: Fn(&[f64], &mut [f64]),
    SS: FnMut() -> bool,
{
    let n = b.len();
    debug_assert_eq!(x.len(), n);
    if n == 0 {
        return MinresStats {
            iters: 0,
            residual_estimate: 0.0,
            converged: true,
        };
    }

    let b_norm = norm2(b).max(f64::MIN_POSITIVE);

    // r1 = b - K x0
    let mut kx = vec![0.0f64; n];
    matvec(x, &mut kx);
    let mut r1: Vec<f64> = b.iter().zip(kx.iter()).map(|(&bi, &ki)| bi - ki).collect();
    let r1_norm = norm2(&r1);
    if r1_norm <= tol * b_norm {
        return MinresStats {
            iters: 0,
            residual_estimate: r1_norm,
            converged: true,
        };
    }

    // y = M^{-1} r1
    let mut y = vec![0.0f64; n];
    precond(&r1, &mut y);
    // beta1 = sqrt(<r1, y>)  (M^{-1}-norm)
    let beta1_sq = dot(&r1, &y);
    if beta1_sq <= 0.0 {
        // 前処理が SPD でない / 数値ゼロ → 既に解けている扱いか即時破綻
        return MinresStats {
            iters: 0,
            residual_estimate: r1_norm,
            converged: r1_norm <= tol * b_norm,
        };
    }
    let beta1 = beta1_sq.sqrt();

    // 状態変数 (SciPy minres と同じ命名)
    let mut r2 = r1.clone();
    let mut beta = beta1;
    let mut oldb = 0.0_f64;
    let mut dbar = 0.0_f64;
    let mut epsln = 0.0_f64;
    let mut phibar = beta1;
    let mut cs = -1.0_f64;
    let mut sn = 0.0_f64;
    let mut w = vec![0.0_f64; n];
    let mut w2 = vec![0.0_f64; n];

    let mut residual_estimate = beta1;
    let mut last_iter = 0usize;
    let mut converged = false;

    for itn in 1..=max_iter {
        last_iter = itn;
        if should_stop() {
            break;
        }

        // ── Lanczos step ──────────────────────────────
        // v = y / beta
        let s_inv = 1.0 / beta;
        let v: Vec<f64> = y.iter().map(|&yi| yi * s_inv).collect();

        // y = K v
        matvec(&v, &mut y);

        // y = y - (beta / oldb) * r1   (itn >= 2 のみ)
        if itn >= 2 {
            let factor = beta / oldb;
            for i in 0..n {
                y[i] -= factor * r1[i];
            }
        }

        // alfa = <v, y>
        let alfa = dot(&v, &y);

        // y = y - (alfa / beta) * r2
        let factor = alfa / beta;
        for i in 0..n {
            y[i] -= factor * r2[i];
        }

        // r1 <- r2, r2 <- y
        std::mem::swap(&mut r1, &mut r2);
        r2.copy_from_slice(&y);

        // y = M^{-1} r2
        precond(&r2, &mut y);
        oldb = beta;
        let beta_sq = dot(&r2, &y);
        if beta_sq < 0.0 {
            // 前処理が SPD でない (まれ) — 中断
            break;
        }
        beta = beta_sq.sqrt();

        // ── Apply previous Givens to [dbar; alfa] / [0; beta] ──
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        // ── New Givens [gbar; beta] → [gamma; 0] ─────
        let gamma = (gbar * gbar + beta * beta).sqrt().max(f64::MIN_POSITIVE);
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar *= sn;

        // ── Update x ─────────────────────────────────
        // w_new = (v - oldeps * w2 - delta * w) / gamma
        let inv_gamma = 1.0 / gamma;
        let mut w_new = vec![0.0_f64; n];
        for i in 0..n {
            w_new[i] = (v[i] - oldeps * w2[i] - delta * w[i]) * inv_gamma;
        }
        // x += phi * w_new
        for i in 0..n {
            x[i] += phi * w_new[i];
        }

        // 残差推定: ||r_k||_2 ≈ |phibar| (Givens recurrence の不変量)
        residual_estimate = phibar.abs();

        // 収束判定
        if residual_estimate <= tol * b_norm {
            converged = true;
            break;
        }

        // 状態回転: w2 <- w, w <- w_new
        std::mem::swap(&mut w2, &mut w);
        w = w_new;

        // beta が極小 → Krylov 部分空間枯渇 (理論上 exact solution 到達)
        if beta < f64::MIN_POSITIVE {
            converged = residual_estimate <= tol * b_norm;
            break;
        }
    }

    MinresStats {
        iters: last_iter,
        residual_estimate,
        converged,
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[inline]
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Sparse symmetric K (上三角 CSC で格納) と x の積 `y = K x`。
///
/// 上三角格納 (i ≤ j のみ非ゼロ) からフルの対称積を計算する。
pub fn matvec_sym_upper(k: &CscMatrix, x: &[f64], y: &mut [f64]) {
    let n = k.nrows;
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), n);
    for yi in y.iter_mut() {
        *yi = 0.0;
    }
    for j in 0..n {
        let xj = x[j];
        for k_idx in k.col_ptr[j]..k.col_ptr[j + 1] {
            let i = k.row_ind[k_idx];
            let v = k.values[k_idx];
            if i == j {
                y[j] += v * xj;
            } else {
                // i < j (上三角): K[i, j] = K[j, i] = v なので両方寄与
                y[i] += v * xj;
                y[j] += v * x[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// 恒等前処理 (M = I)
    fn no_precond(r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }

    /// MINRES-T1: SPD 2x2 で解析解と一致
    /// K = [[4,1],[1,3]], b = [1, 2] → x = [1/11, 7/11]
    #[test]
    fn minres_2x2_spd() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[4.0, 1.0, 3.0], 2, 2).unwrap();
        let b = vec![1.0, 2.0];
        let mut x = vec![0.0; 2];
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x,
            1e-12,
            100,
            || false,
        );
        assert!(
            stats.converged,
            "should converge for 2x2 SPD, iters={} resid={:.2e}",
            stats.iters, stats.residual_estimate
        );
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9, "x[0]≈1/11, got {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9, "x[1]≈7/11, got {}", x[1]);
    }

    /// MINRES-T2: 対称不定値 (quasidefinite) 2x2、解析解と一致
    /// K = [[2,1],[1,-1]], b = [3, 0] → x = [1, 1] (DirectLdl test と同値)
    #[test]
    fn minres_2x2_indefinite() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let b = vec![3.0, 0.0];
        let mut x = vec![0.0; 2];
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x,
            1e-12,
            100,
            || false,
        );
        assert!(
            stats.converged,
            "should converge for 2x2 indef, iters={}",
            stats.iters
        );
        assert!((x[0] - 1.0).abs() < 1e-9, "x[0]≈1, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-9, "x[1]≈1, got {}", x[1]);
    }

    /// MINRES-T3: 5x5 quasidef、LDL と数値一致
    #[test]
    fn minres_5x5_matches_ldl() {
        let entries = [
            (0, 0, 4.0),
            (0, 1, 0.5),
            (1, 1, 4.0),
            (1, 2, 0.5),
            (2, 2, 4.0),
            (0, 3, 0.3),
            (3, 3, -2.0),
            (3, 4, 0.4),
            (4, 4, -2.0),
        ];
        let rows: Vec<usize> = entries.iter().map(|(r, _, _)| *r).collect();
        let cols: Vec<usize> = entries.iter().map(|(_, c, _)| *c).collect();
        let vals: Vec<f64> = entries.iter().map(|(_, _, v)| *v).collect();
        let k = CscMatrix::from_triplets(&rows, &cols, &vals, 5, 5).unwrap();

        let b = vec![1.0, 2.0, -1.0, 0.5, -0.5];

        // Reference solution via LDL
        let factor = crate::linalg::ldl::factorize_quasidefinite_with_amd(&k, None)
            .expect("LDL should succeed");
        let mut x_ldl = vec![0.0; 5];
        factor.solve(&b, &mut x_ldl);

        // MINRES
        let mut x_minres = vec![0.0; 5];
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x_minres,
            1e-10,
            200,
            || false,
        );
        assert!(stats.converged, "should converge, iters={}", stats.iters);
        for i in 0..5 {
            assert!(
                (x_ldl[i] - x_minres[i]).abs() < 1e-7,
                "x[{}]: LDL={}, MINRES={}, diff={:.2e}",
                i,
                x_ldl[i],
                x_minres[i],
                (x_ldl[i] - x_minres[i]).abs()
            );
        }
    }

    /// MINRES-T4: should_stop で早期中断
    #[test]
    fn minres_can_be_stopped_early() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[4.0, 1.0, 3.0], 2, 2).unwrap();
        let b = vec![1.0, 2.0];
        let mut x = vec![0.0; 2];
        let mut count = 0;
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x,
            1e-12,
            100,
            || {
                count += 1;
                count > 1
            },
        );
        // count=1 で 1 反復、count>1 で stop → 1 反復で break
        assert!(stats.iters <= 1 || !stats.converged, "should stop early");
    }

    /// MINRES-T5: 0 RHS → 即時収束
    #[test]
    fn minres_zero_rhs() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[4.0, 1.0, 3.0], 2, 2).unwrap();
        let b = vec![0.0; 2];
        let mut x = vec![0.0; 2];
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x,
            1e-12,
            100,
            || false,
        );
        assert_eq!(stats.iters, 0);
        assert!(stats.converged);
        assert_eq!(x, vec![0.0; 2]);
    }

    /// MINRES-T6: 初期推定 x0 != 0 でも正しく動く
    #[test]
    fn minres_nonzero_initial_guess() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[4.0, 1.0, 3.0], 2, 2).unwrap();
        let b = vec![1.0, 2.0];
        // 真の解 x* = [1/11, 7/11]、初期 x0 = [0.1, 0.7] (近い)
        let mut x = vec![0.1, 0.7];
        let stats = pminres(
            |v, y| matvec_sym_upper(&k, v, y),
            no_precond,
            &b,
            &mut x,
            1e-12,
            100,
            || false,
        );
        assert!(stats.converged, "should converge with warm start");
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9);
    }
}
