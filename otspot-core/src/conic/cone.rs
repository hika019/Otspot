//! Jordan-algebra operations and Nesterov--Todd scaling for the cone
//! `K = R_+^l x Q_{m_1} x ... x Q_{m_k}`.

use super::ConeSpec;

/// Ranges of each block in a length-`m` conic vector.
pub(super) struct Blocks {
    pub l: usize,
    pub soc: Vec<usize>,
}

impl Blocks {
    pub fn new(cone: &ConeSpec) -> Self {
        Blocks {
            l: cone.l,
            soc: cone.soc.clone(),
        }
    }

    /// Start offsets of each second-order cone within `[l..m)`.
    pub fn soc_offsets(&self) -> Vec<usize> {
        let mut offs = Vec::with_capacity(self.soc.len());
        let mut o = self.l;
        for &d in &self.soc {
            offs.push(o);
            o += d;
        }
        offs
    }

    pub fn dim(&self) -> usize {
        self.l + self.soc.iter().sum::<usize>()
    }
}

/// Cone identity `e` (ones on the orthant, `(1,0,...,0)` per SOC).
pub(super) fn identity(blk: &Blocks) -> Vec<f64> {
    let mut e = vec![0.0; blk.dim()];
    for v in e.iter_mut().take(blk.l) {
        *v = 1.0;
    }
    for &off in &blk.soc_offsets() {
        e[off] = 1.0;
    }
    e
}

/// Jordan product `x ∘ y`.
pub(super) fn jprod(blk: &Blocks, x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; blk.dim()];
    for i in 0..blk.l {
        out[i] = x[i] * y[i];
    }
    let offs = blk.soc_offsets();
    for (bi, &off) in offs.iter().enumerate() {
        let d = blk.soc[bi];
        // first entry: full dot product
        let mut dot = 0.0;
        for k in 0..d {
            dot += x[off + k] * y[off + k];
        }
        out[off] = dot;
        // tail: x0 * y1 + y0 * x1
        for k in 1..d {
            out[off + k] = x[off] * y[off + k] + y[off] * x[off + k];
        }
    }
    out
}

/// Solve `lambda ∘ u = v` for `u` (arrow-matrix inverse per block).
pub(super) fn jdiv(blk: &Blocks, lambda: &[f64], v: &[f64]) -> Vec<f64> {
    let mut u = vec![0.0; blk.dim()];
    for i in 0..blk.l {
        u[i] = v[i] / lambda[i];
    }
    let offs = blk.soc_offsets();
    for (bi, &off) in offs.iter().enumerate() {
        let d = blk.soc[bi];
        let l0 = lambda[off];
        let l1 = &lambda[off + 1..off + d];
        let v0 = v[off];
        let v1 = &v[off + 1..off + d];
        // Arrow(lambda) = [[l0, l1^T],[l1, l0 I]].
        // rho = l0^2 - ||l1||^2 (Jordan determinant).
        let nl1: f64 = l1.iter().map(|a| a * a).sum();
        let rho = l0 * l0 - nl1;
        let l1v1: f64 = l1.iter().zip(v1).map(|(a, b)| a * b).sum();
        // u0 = (l0 v0 - l1^T v1) / rho
        u[off] = (l0 * v0 - l1v1) / rho;
        // u1 = (v1 - (l1/l0)*(v0 - u0*?...)) closed form:
        // From l0*u1 + u0*l1 = v1  =>  u1 = (v1 - u0*l1)/l0
        let u0 = u[off];
        for k in 1..d {
            u[off + k] = (v[off + k] - u0 * lambda[off + k]) / l0;
        }
    }
    u
}

/// Largest `alpha in [0, cap]` such that `w + alpha*dw` stays in the cone.
/// Returns `cap` if unconstrained.
pub(super) fn max_step(blk: &Blocks, w: &[f64], dw: &[f64], cap: f64) -> f64 {
    let mut alpha = cap;
    // Orthant: w_i + alpha dw_i >= 0.
    for i in 0..blk.l {
        if dw[i] < 0.0 {
            let a = -w[i] / dw[i];
            if a < alpha {
                alpha = a;
            }
        }
    }
    let offs = blk.soc_offsets();
    for (bi, &off) in offs.iter().enumerate() {
        let d = blk.soc[bi];
        // f(a) = (w0+a dw0)^2 - ||w1+a dw1||^2 >= 0 and (w0 + a dw0) >= 0.
        let w0 = w[off];
        let dw0 = dw[off];
        let mut aa = dw0 * dw0;
        let mut bb = 2.0 * w0 * dw0;
        let mut cc = w0 * w0;
        for k in 1..d {
            aa -= dw[off + k] * dw[off + k];
            bb -= 2.0 * w[off + k] * dw[off + k];
            cc -= w[off + k] * w[off + k];
        }
        // Smallest positive root of aa*a^2 + bb*a + cc = 0 bounds alpha.
        let root = smallest_positive_root(aa, bb, cc);
        if let Some(r) = root {
            if r < alpha {
                alpha = r;
            }
        }
        // Keep w0 + a dw0 >= 0.
        if dw0 < 0.0 {
            let a = -w0 / dw0;
            if a < alpha {
                alpha = a;
            }
        }
    }
    alpha.max(0.0)
}

/// Smallest positive root of `a x^2 + b x + c` with `c >= 0` (current point
/// strictly feasible). Returns `None` when the quadratic never reaches zero for
/// positive `x`.
fn smallest_positive_root(a: f64, b: f64, c: f64) -> Option<f64> {
    let eps = 1e-14;
    if a.abs() < eps {
        // Linear: b x + c = 0.
        if b.abs() < eps {
            return None;
        }
        let x = -c / b;
        return if x > 0.0 { Some(x) } else { None };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let r1 = (-b - sq) / (2.0 * a);
    let r2 = (-b + sq) / (2.0 * a);
    let mut best: Option<f64> = None;
    for r in [r1, r2] {
        if r > 0.0 {
            best = Some(match best {
                Some(b0) => b0.min(r),
                None => r,
            });
        }
    }
    best
}

/// Nesterov--Todd scaling: block-diagonal symmetric matrices `w` and `winv`
/// (`m x m`, stored row-major) with `w z = winv s = lambda`.
pub(super) struct Scaling {
    pub w: Vec<Vec<f64>>,
    pub winv: Vec<Vec<f64>>,
}

pub(super) fn nt_scaling(blk: &Blocks, s: &[f64], z: &[f64]) -> Scaling {
    let m = blk.dim();
    let mut w = vec![vec![0.0; m]; m];
    let mut winv = vec![vec![0.0; m]; m];
    // Orthant: diagonal.
    for i in 0..blk.l {
        let wi = (s[i] / z[i]).sqrt();
        w[i][i] = wi;
        winv[i][i] = 1.0 / wi;
    }
    let offs = blk.soc_offsets();
    for (bi, &off) in offs.iter().enumerate() {
        let d = blk.soc[bi];
        let sb = &s[off..off + d];
        let zb = &z[off..off + d];
        let ss = jdet(sb);
        let zz = jdet(zb);
        let ssr = ss.sqrt();
        let zzr = zz.sqrt();
        // Normalised points.
        let sbar: Vec<f64> = sb.iter().map(|v| v / ssr).collect();
        let zbar: Vec<f64> = zb.iter().map(|v| v / zzr).collect();
        let dot: f64 = sbar.iter().zip(&zbar).map(|(a, b)| a * b).sum();
        let gamma = ((1.0 + dot) / 2.0).sqrt();
        // wbar = (sbar + J zbar) / (2 gamma), J flips tail sign.
        let mut wbar = vec![0.0; d];
        wbar[0] = (sbar[0] + zbar[0]) / (2.0 * gamma);
        for k in 1..d {
            wbar[k] = (sbar[k] - zbar[k]) / (2.0 * gamma);
        }
        let eta = (ss / zz).powf(0.25);
        // What = [[w0, w1^T],[w1, I + w1 w1^T/(1+w0)]].
        let w0 = wbar[0];
        let denom = 1.0 + w0;
        for r in 0..d {
            for col in 0..d {
                let whatv;
                if r == 0 && col == 0 {
                    whatv = w0;
                } else if r == 0 {
                    whatv = wbar[col];
                } else if col == 0 {
                    whatv = wbar[r];
                } else {
                    let id = if r == col { 1.0 } else { 0.0 };
                    whatv = id + wbar[r] * wbar[col] / denom;
                }
                w[off + r][off + col] = eta * whatv;
                // Whatinv = J What J: negate the (0,k) and (k,0) blocks.
                let mut whativ = whatv;
                if (r == 0) ^ (col == 0) {
                    whativ = -whatv;
                }
                winv[off + r][off + col] = whativ / eta;
            }
        }
    }
    Scaling { w, winv }
}

/// Jordan determinant `w0^2 - ||w1||^2`.
fn jdet(v: &[f64]) -> f64 {
    let tail: f64 = v[1..].iter().map(|a| a * a).sum();
    v[0] * v[0] - tail
}

/// Apply a block-diagonal dense matrix to a vector.
pub(super) fn mat_apply(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let m = v.len();
    let mut out = vec![0.0; m];
    for (i, row) in mat.iter().enumerate() {
        let mut acc = 0.0;
        for (j, &a) in row.iter().enumerate() {
            if a != 0.0 {
                acc += a * v[j];
            }
        }
        out[i] = acc;
    }
    out
}
