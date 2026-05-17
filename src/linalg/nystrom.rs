//! Randomised Nyström preconditioner for SPD matrices
//!
//! Implements the algorithm of Frangella, Tropp & Udell (SIAM J. Matrix Anal. Appl. 2023).
//!
//! # Construction (per IPM iteration)
//! 1. Draw Ω ∈ ℝ^{n×ℓ} with i.i.d. N(0,1) entries
//! 2. Compute Y = MΩ  (ℓ matrix–vector products)
//! 3. Form B = ΩᵀY + ν·I  (ℓ×ℓ, symmetric)
//! 4. Compute L_B = chol(B)  (ℓ×ℓ)
//! 5. Compute Z = Y L_B^{-T}  (n×ℓ)
//! 6. Compute G = ZᵀZ + ν·I  (ℓ×ℓ), L_G = chol(G)
//!
//! # Application (Woodbury identity)
//!   P⁻¹ v = v/ν − Z (ν·I + ZᵀZ)⁻¹ Zᵀ v

use crate::linalg::rng::{fill_gaussian, Xorshift128Plus};

/// Nyström preconditioner for an n×n SPD matrix M.
pub struct NystromPrecond {
    /// Z = Y L_B^{-T}, stored col-major (n×ℓ).
    z: Vec<f64>,
    /// Cholesky L_G of (ν·I + ZᵀZ), row-major ℓ×ℓ lower-triangular.
    l_g: Vec<f64>,
    pub nu: f64,
    pub n: usize,
    pub rank: usize,
}

impl NystromPrecond {
    /// Apply P⁻¹ to `v`, writing the result to `out`.
    ///
    /// P = ν·I + Z Zᵀ,   P⁻¹ v = v/ν − Z (ν·I + ZᵀZ)⁻¹ Zᵀ v
    #[allow(clippy::needless_range_loop)]
    pub fn apply(&self, v: &[f64], out: &mut [f64]) {
        let n = self.n;
        let l = self.rank;

        // w1 = Zᵀ v  (ℓ×1)
        let mut w1 = vec![0.0f64; l];
        for j in 0..l {
            let mut s = 0.0;
            let col = &self.z[j * n..(j + 1) * n];
            for i in 0..n {
                s += col[i] * v[i];
            }
            w1[j] = s;
        }

        // Solve (ν·I + ZᵀZ) w = w1 via stored Cholesky L_G
        let w2 = chol_solve(&self.l_g, l, &w1);

        // out = v/ν − Z w2
        let inv_nu = 1.0 / self.nu;
        for i in 0..n {
            let mut zw = 0.0;
            for j in 0..l {
                zw += self.z[j * n + i] * w2[j];
            }
            out[i] = v[i] * inv_nu - zw;
        }
    }
}

/// Build a Nyström preconditioner by sampling ℓ random directions.
///
/// * `matvec` — applies M: `matvec(x, y)` sets `y = M x`
/// * `n`      — matrix dimension
/// * `rank`   — sketch rank ℓ
/// * `nu`     — regularisation (e.g. δ_p)
/// * `seed`   — RNG seed
///
/// Returns `None` if the sketch matrix is numerically singular.
pub fn build_nystrom<F>(
    matvec: F,
    n: usize,
    rank: usize,
    nu: f64,
    seed: u64,
) -> Option<NystromPrecond>
where
    F: Fn(&[f64], &mut [f64]),
{
    if n == 0 || rank == 0 {
        return None;
    }
    let l = rank.min(n);
    let mut rng = Xorshift128Plus::new(seed);

    // Ω ∈ ℝ^{n×ℓ} (col-major: column j is omega[j*n..(j+1)*n])
    let mut omega = vec![0.0f64; n * l];
    fill_gaussian(&mut rng, &mut omega);

    // Y = M Ω  (n×ℓ col-major)
    let mut y = vec![0.0f64; n * l];
    let mut tmp_in = vec![0.0f64; n];
    let mut tmp_out = vec![0.0f64; n];
    for j in 0..l {
        tmp_in.copy_from_slice(&omega[j * n..(j + 1) * n]);
        matvec(&tmp_in, &mut tmp_out);
        y[j * n..(j + 1) * n].copy_from_slice(&tmp_out);
    }

    // B = ΩᵀY + ν·I  (ℓ×ℓ row-major)
    let mut b = vec![0.0f64; l * l];
    for i in 0..l {
        for j in 0..l {
            let mut dot = 0.0;
            for k in 0..n {
                dot += omega[i * n + k] * y[j * n + k];
            }
            b[i * l + j] = dot;
        }
        b[i * l + i] += nu;
    }

    // L_B = chol(B)  (in-place, lower-triangular)
    if !chol_factorize(&mut b, l) {
        return None;
    }

    // Compute L_B^{-T} explicitly (ℓ×ℓ row-major)
    let l_b_inv_t = chol_inv_t(&b, l);

    // Z = Y · L_B^{-T}  (n×ℓ col-major)
    // Z[:,j] = Σ_k Y[:,k] * L_B^{-T}[k,j]
    let mut z = vec![0.0f64; n * l];
    for j in 0..l {
        for k in 0..l {
            let c = l_b_inv_t[k * l + j];
            if c.abs() < 1e-300 {
                continue;
            }
            let y_col = &y[k * n..(k + 1) * n];
            let z_col = &mut z[j * n..(j + 1) * n];
            for i in 0..n {
                z_col[i] += y_col[i] * c;
            }
        }
    }

    // G = ZᵀZ + ν·I  (ℓ×ℓ row-major, symmetric)
    let mut g = vec![0.0f64; l * l];
    for i in 0..l {
        for j in 0..=i {
            let mut dot = 0.0;
            let zi = &z[i * n..(i + 1) * n];
            let zj = &z[j * n..(j + 1) * n];
            for k in 0..n {
                dot += zi[k] * zj[k];
            }
            g[i * l + j] = dot;
            g[j * l + i] = dot;
        }
        g[i * l + i] += nu;
    }

    // L_G = chol(G)
    if !chol_factorize(&mut g, l) {
        // Retry with stronger regularisation
        // PARAM: 1e6 — Cholesky 失敗時の緊急正則化倍率（経験値・要検証）。G が SPD でない
        // 場合に nu を 1e6 倍に強化して再試行。nu.max(1e-8) * 1e6 = 最低 1e-2 を保証。
        // Clarabel/OSQP は Nyström 前処理を使わず比較不能。Frangella(2023) 等の論文は
        // 固定小値（1e-6〜1e-8）を推奨するが、本実装は緊急安全弁として 1e6 倍を採用。
        // ベンチ実測での妥当性検証を推奨。承認=要検証
        let nu2 = nu.max(1e-8) * 1e6;
        let mut g2 = vec![0.0f64; l * l];
        for i in 0..l {
            for j in 0..=i {
                let mut dot = 0.0;
                let zi = &z[i * n..(i + 1) * n];
                let zj = &z[j * n..(j + 1) * n];
                for k in 0..n {
                    dot += zi[k] * zj[k];
                }
                g2[i * l + j] = dot;
                g2[j * l + i] = dot;
            }
            g2[i * l + i] += nu2;
        }
        if !chol_factorize(&mut g2, l) {
            return None;
        }
        g = g2;
    }

    Some(NystromPrecond { z, l_g: g, nu, n, rank: l })
}

// ── Dense Cholesky helpers (row-major n×n) ────────────────────────────────

/// In-place lower-triangular Cholesky of sym. PD `a` (row-major n×n).
/// Zeroes the upper triangle. Returns `false` if not numerically PD.
fn chol_factorize(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if s <= 0.0 {
            return false;
        }
        let ljj = s.sqrt();
        a[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = t / ljj;
            a[j * n + i] = 0.0; // zero upper triangle
        }
    }
    true
}

/// Compute L^{-T} where L is lower-triangular (row-major n×n).
/// Returns L^{-T} in row-major format.
fn chol_inv_t(l: &[f64], n: usize) -> Vec<f64> {
    // Solve L^T X = I column by column (backward substitution)
    let mut x = vec![0.0f64; n * n];
    for j in 0..n {
        let mut xj = vec![0.0f64; n];
        xj[j] = 1.0;
        // L^T is upper-triangular: (L^T)[i,k] = L[k,i]
        for i in (0..n).rev() {
            let mut s = xj[i];
            for k in (i + 1)..n {
                s -= l[k * n + i] * xj[k];
            }
            xj[i] = s / l[i * n + i];
        }
        for i in 0..n {
            x[i * n + j] = xj[i]; // L^{-T}[i,j]
        }
    }
    x
}

/// Solve L L^T x = b using lower-triangular L (row-major n×n).
fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    // Forward: L y = b
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    // Backward: L^T x = y
    let mut x = y;
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chol_factorize_3x3() {
        // A = [[4,2,0],[2,3,1],[0,1,2]] — positive definite
        let orig = [4.0f64, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut a = orig;
        assert!(chol_factorize(&mut a, 3), "chol should succeed");
        // Verify L L^T ≈ orig
        for i in 0..3 {
            for j in 0..3 {
                let mut v = 0.0f64;
                for k in 0..3 {
                    v += a[i * 3 + k] * a[j * 3 + k];
                }
                assert!((v - orig[i * 3 + j]).abs() < 1e-10, "LLT[{i},{j}]={v:.6e}");
            }
        }
    }

    #[test]
    fn test_chol_solve() {
        // A = I_3  →  L = I  →  x = b
        let mut a = [0.0f64; 9];
        a[0] = 1.0;
        a[4] = 1.0;
        a[8] = 1.0;
        assert!(chol_factorize(&mut a, 3));
        let b = [1.0f64, 2.0, 3.0];
        let x = chol_solve(&a, 3, &b);
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-12, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn test_nystrom_diagonal_matrix() {
        // M = diag(1,2,3,...,20): build preconditioner and apply
        let n = 20usize;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let precond = build_nystrom(
            |x, y| {
                for i in 0..n {
                    y[i] = diag[i] * x[i];
                }
            },
            n,
            10,
            1e-6,
            42,
        )
        .expect("build_nystrom failed");

        let v: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut out = vec![0.0f64; n];
        precond.apply(&v, &mut out);
        // All outputs should be finite
        for &o in &out {
            assert!(o.is_finite(), "output should be finite");
        }
    }
}
