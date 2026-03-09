//! Preconditioned Conjugate Gradient (PCG) solver (cmd_295)
//!
//! Solves the SPD linear system M x = b using a caller-supplied preconditioner P.

/// Solve M x = b using PCG with preconditioner P.
///
/// * `matvec`   — closure: `y = M x`
/// * `precond`  — closure: `z = P⁻¹ r`
/// * `b`        — right-hand side (length n)
/// * `tol`      — relative residual tolerance: converge when ‖r‖₂ / ‖b‖₂ ≤ tol
/// * `max_iter` — iteration limit
/// * `x`        — initial guess (in) / solution (out)
///
/// Returns `true` when the tolerance was achieved within `max_iter` iterations.
pub fn pcg_solve<MV, PV>(
    matvec: MV,
    precond: PV,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    x: &mut [f64],
) -> bool
where
    MV: Fn(&[f64], &mut [f64]),
    PV: Fn(&[f64], &mut [f64]),
{
    let n = b.len();
    if n == 0 {
        return true;
    }

    let b_norm = norm2(b).max(1e-300);
    let tol_abs = tol * b_norm;

    // r = b − M x₀
    let mut ax = vec![0.0f64; n];
    matvec(x, &mut ax);
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();

    if norm2(&r) <= tol_abs {
        return true;
    }

    // z = P⁻¹ r,  p = z
    let mut z = vec![0.0f64; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for _ in 0..max_iter {
        // α = rᵀz / pᵀMp
        let mut ap = vec![0.0f64; n];
        matvec(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 || !pap.is_finite() {
            return false; // loss of positive-definiteness
        }
        let alpha = rz / pap;

        // x += α p,  r -= α A p
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        if norm2(&r) <= tol_abs {
            return true;
        }

        // z = P⁻¹ r
        precond(&r, &mut z);
        let rz_new = dot(&r, &z);
        if !rz_new.is_finite() {
            return false;
        }
        let beta = rz_new / rz;
        rz = rz_new;

        // p = z + β p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }

    norm2(&r) <= tol_abs
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[inline]
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcg_diagonal_system() {
        // M = diag(1,2,3,4),  b = [1,2,3,4]  →  x* = [1,1,1,1]
        let diag = [1.0f64, 2.0, 3.0, 4.0];
        let n = diag.len();
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut x = vec![0.0f64; n];
        let ok = pcg_solve(
            |xi, y| {
                for i in 0..n {
                    y[i] = diag[i] * xi[i];
                }
            },
            |r, z| {
                for i in 0..n {
                    z[i] = r[i] / diag[i];
                }
            },
            &b,
            1e-10,
            100,
            &mut x,
        );
        assert!(ok, "PCG should converge");
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-8, "x[{i}]={xi}");
        }
    }

    #[test]
    fn test_pcg_tridiagonal() {
        // 5×5 tridiagonal [2,-1,-1,2,-1,...], b=[1,0,0,0,1]
        let n = 5usize;
        let matvec = |x: &[f64], y: &mut [f64]| {
            for i in 0..n {
                y[i] = 2.0 * x[i];
                if i > 0 {
                    y[i] -= x[i - 1];
                }
                if i + 1 < n {
                    y[i] -= x[i + 1];
                }
            }
        };
        let precond = |r: &[f64], z: &mut [f64]| {
            for i in 0..n {
                z[i] = r[i] / 2.0; // Jacobi
            }
        };
        let b = [1.0f64, 0.0, 0.0, 0.0, 1.0];
        let mut x = [0.0f64; 5];
        let ok = pcg_solve(matvec, precond, &b, 1e-10, 200, &mut x);
        assert!(ok, "PCG should converge on tridiagonal");
        // Verify residual
        let mut ax = [0.0f64; 5];
        for i in 0..n {
            ax[i] = 2.0 * x[i];
            if i > 0 {
                ax[i] -= x[i - 1];
            }
            if i + 1 < n {
                ax[i] -= x[i + 1];
            }
        }
        let res: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1e-8, "residual={res:.3e}");
    }
}
