//! Jordan-algebra operations and Nesterov--Todd scaling for the cone
//! `K = R_+^l x Q_{m_1} x ... x Q_{m_k}`.

use super::ConeSpec;

/// Minimum second-order-cone dimension for the `O(d)` rank-1-border KKT
/// representation ([`visit_border_pattern`] / [`Scaling::border_values`]) in
/// place of the `O(d^2)` dense block ([`visit_w2_pattern`] /
/// [`Scaling::w2_values_col_major`]). Both are mathematically exact
/// (`conic_kkt_direction_matches_dense_schur_oracle`), so this is a
/// performance/memory choice only. Measured (`soc_border_threshold_crossover`
/// calibration test + the `conic_socp_route_peak_within_budget` fence
/// rebuilt at threshold `1`): a *single* cone is faster on the border path
/// at every measured `d` (`0.4ms` vs `14.7ms` full-solve at the `d~256`
/// boundary), while *many tiny* cones (10,000 x `d=3`, the CBLIB shape)
/// favor dense slightly (`27.7MB`/`0.10s` vs `33.8MB`/`0.11s` all-border).
/// `256` sits at the conservative top of the defensible `[16, 256]` range:
/// CBLIB-style suites (dims ~3-100) keep their measured-faster dense path,
/// the QCQP->SOCP bridge's single `d = n+2` block (the Phase 3b OOM case)
/// exceeds it by orders of magnitude, and the worst case is a mid-size
/// cone paying the dense path's `<=15ms`-per-solve constant.
pub(super) const SOC_BORDER_MIN_DIM: usize = 256;

/// Ranges of each block in a length-`m` conic vector, plus the auxiliary
/// (border) variable layout for second-order cones at or above
/// [`SOC_BORDER_MIN_DIM`].
///
/// Each border-enabled SOC contributes one `aux_u`/`aux_v` pair (see
/// [`visit_border_pattern`]). They live in different halves of the KKT
/// system because quasidefiniteness groups by diagonal sign: `aux_u`
/// (corner `+1`) belongs with `dx` in the positive-definite half, `aux_v`
/// (corner `-1`) with `dz` in the negative-definite half.
/// `kkt::build_skeleton` relies on this split for its column layout
/// (`dx, aux_u | dy, dz, aux_v`).
pub(super) struct Blocks {
    pub l: usize,
    pub soc: Vec<usize>,
    offs: Vec<usize>,
    /// Per-SOC auxiliary index (0-based, shared by that SOC's `aux_u` *and*
    /// `aux_v` -- each lives in its own separately-numbered region, see
    /// [`n_border`](Blocks::n_border)), `None` for SOCs below
    /// [`SOC_BORDER_MIN_DIM`] (dense path, no auxiliary variables).
    border_idx: Vec<Option<usize>>,
    n_border: usize,
}

impl Blocks {
    pub fn new(cone: &ConeSpec) -> Self {
        let mut offs = Vec::with_capacity(cone.soc.len());
        let mut o = cone.l;
        for &d in &cone.soc {
            offs.push(o);
            o += d;
        }
        let mut border_idx = Vec::with_capacity(cone.soc.len());
        let mut n_border = 0usize;
        for &d in &cone.soc {
            if d >= SOC_BORDER_MIN_DIM {
                border_idx.push(Some(n_border));
                n_border += 1;
            } else {
                border_idx.push(None);
            }
        }
        Blocks {
            l: cone.l,
            soc: cone.soc.clone(),
            offs,
            border_idx,
            n_border,
        }
    }

    /// Start offsets of each second-order cone within `[l..m)`.
    pub fn soc_offsets(&self) -> &[usize] {
        &self.offs
    }

    pub fn dim(&self) -> usize {
        self.l + self.soc.iter().sum::<usize>()
    }

    /// Number of second-order cones at or above [`SOC_BORDER_MIN_DIM`]
    /// (one `aux_u` + one `aux_v` each).
    pub(super) fn n_border(&self) -> usize {
        self.n_border
    }

    /// `Some(idx)` (0-based) if SOC block `bi` uses the border
    /// representation, `None` if it uses the dense `O(d^2)` block. `idx`
    /// indexes independently into the `aux_u` region (`n_border` columns)
    /// and the `aux_v` region (another `n_border` columns) -- the two
    /// regions are numbered separately because they live in different
    /// halves of the KKT system (see the [`Blocks`] doc comment).
    pub(super) fn border_idx(&self, bi: usize) -> Option<usize> {
        self.border_idx[bi]
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

/// Visits the upper-triangular (`row <= col`) sparsity pattern of the
/// block-diagonal `W^2` operator, in column-major order (matching CSC
/// column layout): the `l`-dimensional orthant contributes one diagonal
/// entry per row; each second-order-cone block contributes its dense
/// `d x d` upper triangle, column by column, top-to-bottom within each
/// column. `f(row, col, is_diagonal)` is called once per entry.
///
/// Depends only on the block *dimensions* (`blk`), not on any NT-scaling
/// values -- used to build the sparsity skeleton of the conic augmented KKT
/// system once, up front, independent of the per-iteration numeric values
/// produced by [`Scaling::w2_values_col_major`] (which visits the identical
/// order; the two must stay in lockstep, checked by
/// `conic_kkt_equivalence` tests).
///
/// Second-order cones at or above [`SOC_BORDER_MIN_DIM`] are skipped here --
/// they use the `O(d)` border representation instead (see
/// [`visit_border_pattern`] / [`Scaling::border_values`], Phase 3b).
pub(super) fn visit_w2_pattern(blk: &Blocks, mut f: impl FnMut(usize, usize, bool)) {
    for i in 0..blk.l {
        f(i, i, true);
    }
    for (bi, &off) in blk.soc_offsets().iter().enumerate() {
        if blk.border_idx(bi).is_some() {
            continue;
        }
        let d = blk.soc[bi];
        for c in 0..d {
            for r in 0..=c {
                f(off + r, off + c, r == c);
            }
        }
    }
}

/// Dynamic-entry kind visited by [`visit_border_pattern`] (matches the
/// values emitted by [`Scaling::border_values`] in the same order). The
/// `usize` payload on each variant is the SOC's `Blocks::border_idx`, `0`-based
/// *within that entry's own region* (`aux_u`'s `n_border` columns and
/// `aux_v`'s `n_border` columns are numbered independently -- see the
/// [`Blocks`] doc comment for why they live in different halves of the KKT
/// system).
pub(super) enum BorderEntryKind {
    /// Diagonal `dz` entry: value `-eta^2`.
    Diag,
    /// Dense coupling to the `aux_u` column `idx`: value
    /// `eta*sqrt(2)*wbar[k]`.
    CouplingU(usize),
    /// Sparse (single-row) coupling to the `aux_v` column `idx`: value
    /// `eta*sqrt(2)`.
    CouplingV(usize),
}

/// Visits the *dynamic* (per-iteration-varying) entries of the rank-1-border
/// representation for second-order cones at or above [`SOC_BORDER_MIN_DIM`]:
/// per cone (dimension `d`, row offset `off`, border index `idx`), in order,
/// `d` `Diag` entries at rows `off+k`, `d` `CouplingU(idx)` entries at rows
/// `off+k`, and `1` `CouplingV(idx)` entry at row `off`. The row is absolute
/// within the length-`m` conic vector; the column is implied by `kind`
/// (`Diag`'s is the row itself; `CouplingU`/`CouplingV`'s is `idx` within
/// their own half -- placement is `kkt::build_skeleton`'s job). Does not
/// visit the static `+1`/`-1` corners (iteration-invariant, materialized
/// once by the caller).
///
/// Depends only on block dimensions, not NT-scaling values -- mirrors
/// [`visit_w2_pattern`]'s role for the dense path; must stay in lockstep
/// with [`Scaling::border_values`], checked by
/// `conic_kkt_direction_matches_dense_schur_oracle` (`single_large_soc_border`
/// case) and `soc_border_expansion_matches_dense_w2`.
pub(super) fn visit_border_pattern(blk: &Blocks, mut f: impl FnMut(usize, BorderEntryKind)) {
    for (bi, &off) in blk.soc_offsets().iter().enumerate() {
        let Some(idx) = blk.border_idx(bi) else {
            continue;
        };
        let d = blk.soc[bi];
        for k in 0..d {
            f(off + k, BorderEntryKind::Diag);
        }
        for k in 0..d {
            f(off + k, BorderEntryKind::CouplingU(idx));
        }
        f(off, BorderEntryKind::CouplingV(idx));
    }
}

impl Scaling {
    /// Values of the block-diagonal `W^2` operator's upper triangle, in the
    /// same column-major order as [`visit_w2_pattern`].
    ///
    /// Orthant entries are `w_i^2` (`w_i` the orthant NT scaling diagonal).
    /// Each second-order-cone block uses the quadratic-representation closed
    /// form `W^2 = P(w) = 2 w w^T - jdet(w) J` (`w = eta * wbar` the
    /// *unnormalised* NT point, `J = diag(1,-1,...,-1)`; Faraut & Koranyi
    /// 1994, or Alizadeh & Goldfarb 2003 Sec. 2) -- cross-checked against
    /// `apply_soc`'s arrow+rank-one form by
    /// `nt_scaling_soc_w_squared_matches_quadratic_representation` (Phase 2).
    /// This closed form gives each `(row, col)` entry directly, `O(d^2)`
    /// total per block (dense within the block, by design -- a single huge
    /// SOC block is out of scope for this representation; see Phase 3b).
    ///
    /// Second-order cones at or above [`SOC_BORDER_MIN_DIM`] are skipped
    /// (matches [`visit_w2_pattern`]'s skip; see [`border_values`] instead).
    pub(super) fn w2_values_col_major(&self, blk: &Blocks) -> Vec<f64> {
        // Capacity over *dense-path* cones only -- a border cone's `d(d+1)/2`
        // would put the huge-SOC case right back at the O(d^2) allocation
        // this representation exists to avoid.
        let dense_cap: usize = blk
            .soc
            .iter()
            .enumerate()
            .filter(|&(bi, _)| blk.border_idx(bi).is_none())
            .map(|(_, &d)| d * (d + 1) / 2)
            .sum();
        let mut out = Vec::with_capacity(blk.l + dense_cap);
        for i in 0..blk.l {
            out.push(self.l_w[i] * self.l_w[i]);
        }
        for (bi, _) in blk.soc_offsets().iter().enumerate() {
            if blk.border_idx(bi).is_some() {
                continue;
            }
            let d = blk.soc[bi];
            let sc = &self.soc[bi];
            let w: Vec<f64> = sc.wbar.iter().map(|v| v * sc.eta).collect();
            let det_w = w[0] * w[0] - w[1..].iter().map(|v| v * v).sum::<f64>();
            for c in 0..d {
                for r in 0..=c {
                    let j_rc = if r != c {
                        0.0
                    } else if r == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    out.push(2.0 * w[r] * w[c] - det_w * j_rc);
                }
            }
        }
        out
    }

    /// Values of the rank-1-border representation's *dynamic* entries, in
    /// the same order as [`visit_border_pattern`]: exact expansion
    /// `W^2 = eta^2 (I + u u^T - v v^T)`, `u = sqrt(2) wbar`, `v = sqrt(2)
    /// e0`, from the quadratic representation `W^2 = 2 w w^T - jdet(w) J`
    /// (`w = eta*wbar`, `jdet(w) = eta^2` since `jdet(wbar) = 1` for any
    /// NT-scaling point) and the identity `J = 2 e0 e0^T - I`
    /// (`= diag(1,-1,...,-1)`, matching [`w2_values_col_major`]'s `J`).
    /// Verified by dense comparison in `soc_border_expansion_matches_dense_w2`.
    ///
    /// The `-W^2` KKT block stores `-eta^2 I` directly; the rank-1 terms are
    /// re-injected by Schur elimination of two border variables: `aux_u`
    /// (corner `+1`, dense column `eta*sqrt(2)*wbar`) contributes
    /// `-eta^2 u u^T`, `aux_v` (corner `-1`, single entry `eta*sqrt(2)` at
    /// the leading row) contributes `+eta^2 v v^T`. The `+1`/`-1` corner
    /// pair is the only sign combination whose Schur complement reproduces
    /// `-W^2` (verified numerically; the other three are off by an O(1)
    /// factor), and it is forced by `-W^2`'s negative-definiteness relative
    /// to the positive `eta^2 I` base.
    pub(super) fn border_values(&self, blk: &Blocks) -> Vec<f64> {
        const SQRT_2: f64 = std::f64::consts::SQRT_2;
        let mut out = Vec::new();
        for (bi, _) in blk.soc_offsets().iter().enumerate() {
            if blk.border_idx(bi).is_none() {
                continue;
            }
            let d = blk.soc[bi];
            let sc = &self.soc[bi];
            let eta = sc.eta;
            for _ in 0..d {
                out.push(-eta * eta);
            }
            for k in 0..d {
                out.push(eta * SQRT_2 * sc.wbar[k]);
            }
            out.push(eta * SQRT_2);
        }
        out
    }
}
