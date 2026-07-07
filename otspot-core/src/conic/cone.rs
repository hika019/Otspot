//! Jordan-algebra operations and Nesterov--Todd scaling for the cone
//! `K = R_+^l x Q_{m_1} x ... x Q_{m_k}`.

use super::ConeSpec;

/// Ranges of each block in a length-`m` conic vector.
pub(super) struct Blocks {
    pub l: usize,
    pub soc: Vec<usize>,
    offs: Vec<usize>,
}

impl Blocks {
    pub fn new(cone: &ConeSpec) -> Self {
        let mut offs = Vec::with_capacity(cone.soc.len());
        let mut o = cone.l;
        for &d in &cone.soc {
            offs.push(o);
            o += d;
        }
        Blocks {
            l: cone.l,
            soc: cone.soc.clone(),
            offs,
        }
    }

    /// Start offsets of each second-order cone within `[l..m)`.
    pub fn soc_offsets(&self) -> &[usize] {
        &self.offs
    }

    pub fn dim(&self) -> usize {
        self.l + self.soc.iter().sum::<usize>()
    }
}

/// Approximate membership `v ∈ K` (self-dual, so also `v ∈ K*`): orthant
/// components `>= -tol` and each SOC block `v0 >= ||v_rest|| - tol`.
pub(super) fn in_cone(blk: &Blocks, v: &[f64], tol: f64) -> bool {
    if v[..blk.l].iter().any(|&vi| vi < -tol) {
        return false;
    }
    let mut off = blk.l;
    for &d in &blk.soc {
        let rest = &v[off + 1..off + d];
        let nr = rest.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v[off] < nr - tol {
            return false;
        }
        off += d;
    }
    true
}

/// Cone identity `e` (ones on the orthant, `(1,0,...,0)` per SOC).
pub(super) fn identity(blk: &Blocks) -> Vec<f64> {
    let mut e = vec![0.0; blk.dim()];
    for v in e.iter_mut().take(blk.l) {
        *v = 1.0;
    }
    for &off in blk.soc_offsets() {
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

/// Nesterov--Todd scaling operators `w` and `winv` with `w z = winv s =
/// lambda`. Stored per-block: an orthant diagonal (`O(l)`) plus, per
/// second-order cone, a single normalised NT point `wbar` (length `d_i`) and
/// scale `eta` (`O(d_i)`) rather than a materialised `d_i x d_i` matrix.
///
/// `What` (the `eta=1` scaling matrix) has the closed form
/// `[[w0, w1^T],[w1, I + w1 w1^T/(1+w0)]]` (`w0 = wbar[0]`, `w1 = wbar[1..]`),
/// i.e. `Arrow'(wbar) + w1 w1^T/(1+w0)` where `Arrow'(wbar)` has corner `w0`,
/// first row/col `w1`, and plain `I` on the remaining diagonal (unlike the
/// Jordan-product arrow matrix in [`jdiv`], whose tail diagonal is `l0 I`).
/// This decomposes any mat-vec into one dot product and one rank-one update,
/// `O(d_i)` instead of `O(d_i^2)`. `What^{-1} = J What J` (`J` flips the tail
/// sign) reuses the same `wbar`/`eta`: `J` negates `Arrow'`'s off-diagonal
/// (the `w0`-linked cross terms) but leaves the rank-one term untouched
/// (`(J w1)(J w1)^T = w1 w1^T`), so both directions come from one array. This
/// is what makes a single huge SOC block (the QCQP->SOCP bridge emits one
/// block of dimension `n+2` per quadratic term) `O(d)` instead of `O(d^2)`,
/// which otherwise OOMs well before the IPM's own dense-KKT step for large
/// `n` (tracked separately).
pub(super) struct Scaling {
    l_w: Vec<f64>,
    l_winv: Vec<f64>,
    soc: Vec<SocScale>,
}

/// One second-order-cone NT factor: `eta` and the normalised point `wbar`
/// (`wbar[0]` is `w0`, `wbar[1..]` is `w1`). See [`Scaling`] for the
/// `O(d)` mat-vec this supports.
struct SocScale {
    eta: f64,
    wbar: Vec<f64>,
}

impl Scaling {
    pub(super) fn apply_w(&self, blk: &Blocks, v: &[f64]) -> Vec<f64> {
        self.apply(blk, v, false)
    }

    pub(super) fn apply_winv(&self, blk: &Blocks, v: &[f64]) -> Vec<f64> {
        self.apply(blk, v, true)
    }

    /// `O(l + sum d_i)`: diagonal orthant scaling plus one arrow + rank-one
    /// SOC mat-vec per block.
    fn apply(&self, blk: &Blocks, v: &[f64], inverse: bool) -> Vec<f64> {
        let mut out = vec![0.0; blk.dim()];
        let l_diag = if inverse { &self.l_winv } else { &self.l_w };
        for i in 0..blk.l {
            out[i] = l_diag[i] * v[i];
        }
        let offs = blk.soc_offsets();
        for (bi, &off) in offs.iter().enumerate() {
            let d = blk.soc[bi];
            let v0 = v[off];
            let v1 = &v[off + 1..off + d];
            let out1 = &mut out[off + 1..off + d];
            let out0 = apply_soc(&self.soc[bi], v0, v1, inverse, out1);
            out[off] = out0;
        }
        out
    }
}

/// Applies `What` (`inverse=false`) or `What^{-1}` (`inverse=true`), scaled by
/// `eta`/`1/eta`, to `(v0, v1)` and writes the tail into `out1`. Returns the
/// leading component. `O(d)`. See [`Scaling`] for the derivation.
fn apply_soc(soc: &SocScale, v0: f64, v1: &[f64], inverse: bool, out1: &mut [f64]) -> f64 {
    let w0 = soc.wbar[0];
    let w1 = &soc.wbar[1..];
    let denom = 1.0 + w0;
    let dot: f64 = w1.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
    let sign = if inverse { -1.0 } else { 1.0 };
    let scale = if inverse { 1.0 / soc.eta } else { soc.eta };
    let corr = dot / denom;
    for k in 0..v1.len() {
        out1[k] = (sign * w1[k] * v0 + v1[k] + w1[k] * corr) * scale;
    }
    (w0 * v0 + sign * dot) * scale
}

pub(super) fn nt_scaling(blk: &Blocks, s: &[f64], z: &[f64]) -> Scaling {
    let mut l_w = vec![0.0; blk.l];
    let mut l_winv = vec![0.0; blk.l];
    // Orthant: diagonal.
    for i in 0..blk.l {
        let wi = (s[i] / z[i]).sqrt();
        l_w[i] = wi;
        l_winv[i] = 1.0 / wi;
    }
    let offs = blk.soc_offsets();
    let mut soc = Vec::with_capacity(blk.soc.len());
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
        soc.push(SocScale { eta, wbar });
    }
    Scaling { l_w, l_winv, soc }
}

/// Jordan determinant `w0^2 - ||w1||^2`.
fn jdet(v: &[f64]) -> f64 {
    let tail: f64 = v[1..].iter().map(|a| a * a).sum();
    v[0] * v[0] - tail
}
