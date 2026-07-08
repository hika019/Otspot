//! Cone-block-respecting Ruiz-style equilibration for [`ConicProblem`].
//!
//! Iteratively rescales `A`/`G` rows, the shared columns, and the objective,
//! giving the scaled problem the same optimum/status (`x = D x'`, see
//! [`Equilibrator::unscale_result`]) but far better-conditioned magnitudes.
//! Cone membership is preserved because each SOC block uses a single positive
//! row scalar (`e_g` block-constant, checked by
//! `equil_soc_block_scale_is_constant`; `s in K <=> alpha s in K`); orthant
//! and equality rows scale independently.
//!
//! Motivation (#9b): CBLIB `*_w`/`sssd-strong` carry multi-order coefficient
//! ranges across SOC blocks; unequilibrated, the Mehrotra step collapses from
//! iteration 0 and `mu` diverges (`NumericalError`). Balancing the *data*
//! first fixes this with no change to `ipm.rs`. Mirrors `linalg::ruiz` (QP),
//! which can't be reused: conic has two row-blocks sharing one column scaling
//! and `G`'s rows are block-constant per SOC. Ref: Ruiz, ENSEEIHT-IRIT 2001.

use super::cone::Blocks;
use super::kkt;
use super::{ConicProblem, ConicResult};
use crate::problem::SolveStatus;

/// Ruiz sweep count: same convergence argument as `linalg::ruiz::RuizScaler`
/// (each sweep roughly halves the row/col norm deviation from 1; `f64`'s
/// mantissa bit count is enough sweeps to reach machine precision, beyond
/// which further sweeps are floating-point noise).
const EQUIL_SWEEPS: usize = f64::MANTISSA_DIGITS as usize;

/// Floor under row/column/cost inf-norms before taking `1/sqrt(.)`, so an
/// all-zero row/column (e.g. an unused variable) leaves its scale factor
/// finite instead of dividing by zero. Matches `linalg::ruiz::RuizScaler`'s
/// `EPS`.
const EQUIL_EPS: f64 = 1e-6;

/// Row/column/cost scale factors for a [`ConicProblem`]. `d` scales
/// variables/columns (shared by `A`, `G`, `c`); `e_a` scales `A`'s `p` rows
/// independently; `e_g` scales `G`'s `m` rows *block-respecting*: every row
/// within a given second-order cone carries the exact same `e_g` value
/// (orthant rows scale independently). `sigma_c` is a scalar objective
/// normalisation. Every entry is strictly positive.
pub(super) struct Equilibrator {
    pub(super) d: Vec<f64>,
    pub(super) e_a: Vec<f64>,
    pub(super) e_g: Vec<f64>,
    pub(super) sigma_c: f64,
}

impl Equilibrator {
    /// Identity scaling (no-op): `d = e_a = e_g = 1`, `sigma_c = 1`.
    fn identity(n: usize, p: usize, m: usize) -> Self {
        Equilibrator {
            d: vec![1.0; n],
            e_a: vec![1.0; p],
            e_g: vec![1.0; m],
            sigma_c: 1.0,
        }
    }

    /// Computes cone-block-respecting Ruiz equilibration for `problem`.
    /// `O(EQUIL_SWEEPS * (nnz(A) + nnz(G)))`, run once per solve (or once per
    /// `MisocpProblem::base`, never per branch-and-bound node -- see
    /// `misocp::solve_misocp`).
    pub(super) fn compute(problem: &ConicProblem) -> Self {
        let n = problem.n();
        let p = problem.p();
        let m = problem.m();
        let blk = Blocks::new(&problem.cone);
        let mut eq = Equilibrator::identity(n, p, m);
        if n == 0 {
            return eq;
        }
        for _ in 0..EQUIL_SWEEPS {
            eq.row_sweep(problem, &blk);
            eq.col_sweep(problem);
            eq.cost_sweep(problem);
        }
        eq
    }

    /// Step 1: row inf-norm update, driven *only* by `A`/`G` (see
    /// [`Self::col_sweep`]'s doc for why the RHS -- `b`/`h`, the row-scale
    /// analogue of the cost row `c` -- is deliberately excluded: a row whose
    /// matrix entries are already O(1) but whose RHS happens to be huge
    /// (e.g. a variable bound `x <= V` encoded as an orthant row with
    /// `h = V`) would otherwise lock `e_a[i]`/`e_g[i]` to `1/V`, driving the
    /// *matrix* entry -- the thing that actually needs to be well-scaled for
    /// the KKT factorization -- down to `~1/V` instead of leaving it at its
    /// already-fine `~1`. `ipm::solve` already normalises residuals against
    /// `1 + norm(b)` / `1 + norm(c)` on its own, so the RHS's raw magnitude
    /// need not (and must not) drive row/col scaling here.
    ///
    /// `A`'s `p` rows are independent; `G`'s orthant rows are independent but
    /// every second-order-cone block shares one scalar across its rows (the
    /// cone-preservation invariant).
    fn row_sweep(&mut self, problem: &ConicProblem, blk: &Blocks) {
        let n = problem.n();
        let p = problem.p();
        let m = problem.m();

        if p > 0 {
            let mut row_norms = vec![0.0f64; p];
            for col in 0..n {
                for k in problem.a.col_ptr()[col]..problem.a.col_ptr()[col + 1] {
                    let row = problem.a.row_ind()[k];
                    let val = (self.e_a[row] * problem.a.values()[k] * self.d[col]).abs();
                    if val > row_norms[row] {
                        row_norms[row] = val;
                    }
                }
            }
            for i in 0..p {
                if row_norms[i] > 0.0 {
                    self.e_a[i] /= row_norms[i].max(EQUIL_EPS).sqrt();
                }
            }
        }

        if m > 0 {
            let mut row_norms = vec![0.0f64; m];
            for col in 0..n {
                for k in problem.g.col_ptr()[col]..problem.g.col_ptr()[col + 1] {
                    let row = problem.g.row_ind()[k];
                    let val = (self.e_g[row] * problem.g.values()[k] * self.d[col]).abs();
                    if val > row_norms[row] {
                        row_norms[row] = val;
                    }
                }
            }
            for i in 0..blk.l {
                if row_norms[i] > 0.0 {
                    self.e_g[i] /= row_norms[i].max(EQUIL_EPS).sqrt();
                }
            }
            for (bi, &off) in blk.soc_offsets().iter().enumerate() {
                let d_soc = blk.soc[bi];
                let mut block_norm = 0.0f64;
                for row in off..off + d_soc {
                    if row_norms[row] > block_norm {
                        block_norm = row_norms[row];
                    }
                }
                if block_norm > 0.0 {
                    let scale = block_norm.max(EQUIL_EPS).sqrt();
                    for row in off..off + d_soc {
                        self.e_g[row] /= scale;
                    }
                }
            }
        }
    }

    /// Step 2: column inf-norm update, driven *only* by `A`/`G` (uses this
    /// sweep's just-updated row scales) -- deliberately excludes the cost
    /// row `c`. Folding a linear cost term into the same column-norm
    /// competition as the constraint data creates a degenerate fixed point:
    /// on a column where `c[j]` and, say, `G[i][j]` are comparable in
    /// magnitude, `d[j]` and `sigma_c`/`e_g[i]` become coupled only through
    /// their *product* (`d[j] * max(sigma_c, e_g[i])`), which pins nothing
    /// beyond that product -- `sigma_c` can freeze arbitrarily far from
    /// `e_g[i]`, permanently under-equilibrating that column's `G` entry by
    /// the frozen `e_g[i] / sigma_c` ratio (reproduced by
    /// `equil_recovers_optimum_under_extreme_column_scaling` before this
    /// fix). `linalg::ruiz::RuizScaler` avoids this the same way: its column
    /// step (`Step 2`) folds in `Q` (data that, like `A`/`G` here, is part of
    /// the KKT matrix being factorized) but never `q_vec` -- the linear cost
    /// term is normalised separately (`Step 3`) with no feedback into `d`.
    fn col_sweep(&mut self, problem: &ConicProblem) {
        let n = problem.n();
        let mut col_norms = vec![0.0f64; n];
        for col in 0..n {
            for k in problem.a.col_ptr()[col]..problem.a.col_ptr()[col + 1] {
                let row = problem.a.row_ind()[k];
                let val = (self.e_a[row] * problem.a.values()[k] * self.d[col]).abs();
                if val > col_norms[col] {
                    col_norms[col] = val;
                }
            }
            for k in problem.g.col_ptr()[col]..problem.g.col_ptr()[col + 1] {
                let row = problem.g.row_ind()[k];
                let val = (self.e_g[row] * problem.g.values()[k] * self.d[col]).abs();
                if val > col_norms[col] {
                    col_norms[col] = val;
                }
            }
        }
        for j in 0..n {
            if col_norms[j] > 0.0 {
                self.d[j] /= col_norms[j].max(EQUIL_EPS).sqrt();
            }
        }
    }

    /// Step 3: scalar objective normalisation -- uses this sweep's
    /// just-updated column scale but (see [`Self::col_sweep`]) never feeds
    /// back into it; a full (non-`sqrt`) division, matching
    /// `linalg::ruiz::RuizScaler`'s `q_vec`/`self.c` step, since `sigma_c` is
    /// a single global scalar with no "other side" requiring a geometric-
    /// mean split.
    fn cost_sweep(&mut self, problem: &ConicProblem) {
        let n = problem.n();
        let mut c_inf = 0.0f64;
        for j in 0..n {
            let val = (self.sigma_c * self.d[j] * problem.c[j]).abs();
            if val > c_inf {
                c_inf = val;
            }
        }
        if c_inf > 0.0 {
            self.sigma_c /= c_inf.max(EQUIL_EPS);
        }
    }

    /// Builds the scaled problem `A' = Dr_A A D`, `b' = Dr_A b`,
    /// `G' = Dr_G G D`, `h' = Dr_G h`, `c' = sigma_c D c`. Sparsity pattern
    /// is preserved exactly (only values change); `cone` is unchanged
    /// (equilibration never touches dimensions).
    pub(super) fn scale_problem(&self, problem: &ConicProblem) -> ConicProblem {
        let n = problem.n();

        let mut a = problem.a.clone();
        for col in 0..n {
            for k in problem.a.col_ptr()[col]..problem.a.col_ptr()[col + 1] {
                let row = problem.a.row_ind()[k];
                a.values[k] = self.e_a[row] * problem.a.values()[k] * self.d[col];
            }
        }
        let mut g = problem.g.clone();
        for col in 0..n {
            for k in problem.g.col_ptr()[col]..problem.g.col_ptr()[col + 1] {
                let row = problem.g.row_ind()[k];
                g.values[k] = self.e_g[row] * problem.g.values()[k] * self.d[col];
            }
        }
        let b: Vec<f64> = problem
            .b
            .iter()
            .enumerate()
            .map(|(i, &v)| self.e_a[i] * v)
            .collect();
        let h: Vec<f64> = problem
            .h
            .iter()
            .enumerate()
            .map(|(i, &v)| self.e_g[i] * v)
            .collect();
        let c: Vec<f64> = problem
            .c
            .iter()
            .enumerate()
            .map(|(j, &v)| self.sigma_c * self.d[j] * v)
            .collect();

        ConicProblem {
            c,
            a,
            b,
            g,
            h,
            cone: problem.cone.clone(),
        }
    }

    /// `x = D x'` -- also the correct transform for the primal ray
    /// (recession direction), which lives in the same length-`n`
    /// column-space as `x`.
    fn unscale_x(&self, x: &mut [f64]) {
        for (xi, &dj) in x.iter_mut().zip(&self.d) {
            *xi *= dj;
        }
    }

    /// `y = Dr_A y' / sigma_c`.
    fn unscale_y(&self, y: &mut [f64]) {
        for (yi, &ei) in y.iter_mut().zip(&self.e_a) {
            *yi *= ei / self.sigma_c;
        }
    }

    /// `z = Dr_G z' / sigma_c`.
    fn unscale_z(&self, z: &mut [f64]) {
        for (zi, &ei) in z.iter_mut().zip(&self.e_g) {
            *zi *= ei / self.sigma_c;
        }
    }

    /// `s = Dr_G^{-1} s'`.
    fn unscale_s(&self, s: &mut [f64]) {
        for (si, &ei) in s.iter_mut().zip(&self.e_g) {
            *si /= ei;
        }
    }

    /// Maps a `ConicResult` computed on the scaled problem back to
    /// `problem`'s original space: `x`, `y`, `z`, `s` are unscaled, the
    /// objective is *recomputed* from `problem.c` and the unscaled `x`
    /// (never trusted from the scaled space), and any certificate
    /// (`infeas_cert` / `primal_ray`) is independently re-verified against
    /// `problem`'s own (unscaled) data at the same `tol` -- a scaling bug
    /// can therefore never fabricate a false `Infeasible`/`Unbounded` the
    /// caller can see: if re-verification fails, the status is downgraded to
    /// `NumericalError` and the certificate dropped, exactly as an ordinary
    /// unproven inconclusive solve. See `verify_infeas_cert` /
    /// `verify_primal_ray`.
    pub(super) fn unscale_result(
        &self,
        problem: &ConicProblem,
        tol: f64,
        mut res: ConicResult,
    ) -> ConicResult {
        self.unscale_x(&mut res.x);
        self.unscale_y(&mut res.y);
        self.unscale_z(&mut res.z);
        self.unscale_s(&mut res.s);
        res.objective = problem
            .c
            .iter()
            .zip(&res.x)
            .map(|(a, b)| a * b)
            .sum::<f64>();

        if let Some((y, z)) = &mut res.infeas_cert {
            self.unscale_y(y);
            self.unscale_z(z);
            if !verify_infeas_cert(problem, y, z, tol) {
                res.status = SolveStatus::NumericalError;
                res.infeas_cert = None;
            }
        }
        if let Some(d) = &mut res.primal_ray {
            self.unscale_x(d);
            if !verify_primal_ray(problem, d, tol) {
                res.status = SolveStatus::NumericalError;
                res.primal_ray = None;
            }
        }
        res
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Independent re-verification of a Farkas infeasibility certificate
/// `(y, z)` against `problem`'s own data. Mirrors `ipm::solve`'s inline
/// check (same formulas, same scale-invariant relative tolerances) but is a
/// free-standing predicate so it can be re-run against the *original*
/// problem after [`Equilibrator::unscale_result`] maps a scaled-space
/// certificate back -- `ipm.rs` itself is untouched.
pub(super) fn verify_infeas_cert(problem: &ConicProblem, y: &[f64], z: &[f64], tol: f64) -> bool {
    let blk = Blocks::new(&problem.cone);
    let n = problem.n();
    let aty = kkt::spmtv(&problem.a, y);
    let gtz = kkt::spmtv(&problem.g, z);
    let by: f64 = problem.b.iter().zip(y).map(|(a, b)| a * b).sum();
    let hz: f64 = problem.h.iter().zip(z).map(|(a, b)| a * b).sum();
    let farkas_val = -(by + hz);
    if farkas_val <= 0.0 {
        return false;
    }
    let val_mag: f64 = problem
        .b
        .iter()
        .zip(y)
        .map(|(a, b)| (a * b).abs())
        .sum::<f64>()
        + problem
            .h
            .iter()
            .zip(z)
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>();
    let ray_res = norm2(&(0..n).map(|i| aty[i] + gtz[i]).collect::<Vec<_>>());
    let ray_mag = {
        let y_abs: Vec<f64> = y.iter().map(|v| v.abs()).collect();
        let z_abs: Vec<f64> = z.iter().map(|v| v.abs()).collect();
        let mut acc = vec![0.0; n];
        kkt::spmtv_abs_accum(&problem.a, &y_abs, &mut acc);
        kkt::spmtv_abs_accum(&problem.g, &z_abs, &mut acc);
        norm2(&acc)
    };
    let zn = norm2(z);
    if zn <= 0.0 || !(farkas_val >= tol * val_mag && ray_res <= tol * ray_mag) {
        return false;
    }
    let zs: Vec<f64> = z.iter().map(|v| v / zn).collect();
    super::cone::in_cone(&blk, &zs, tol)
}

/// Independent re-verification of an improving-ray (dual-infeasibility /
/// primal-unboundedness) certificate `d` against `problem`'s own data.
/// Mirrors `ipm::solve`'s inline check; see [`verify_infeas_cert`] for why
/// this is a free-standing predicate.
pub(super) fn verify_primal_ray(problem: &ConicProblem, d: &[f64], tol: f64) -> bool {
    let blk = Blocks::new(&problem.cone);
    let ax = kkt::spmv(&problem.a, d);
    let gx = kkt::spmv(&problem.g, d);
    let cx: f64 = problem.c.iter().zip(d).map(|(a, b)| a * b).sum();
    let descent = -cx;
    if descent <= 0.0 {
        return false;
    }
    let c_mag: f64 = problem.c.iter().zip(d).map(|(a, b)| (a * b).abs()).sum();
    let ax_res = norm2(&ax);
    let d_abs: Vec<f64> = d.iter().map(|v| v.abs()).collect();
    let ax_mag = norm2(&kkt::spmv_abs(&problem.a, &d_abs));
    let g_mag = norm2(&kkt::spmv_abs(&problem.g, &d_abs));
    let dn = norm2(d);
    if dn <= 0.0 || !(descent >= tol * c_mag && ax_res <= tol * ax_mag) {
        return false;
    }
    let recession: Vec<f64> = if g_mag > 0.0 {
        gx.iter().map(|v| -v / g_mag).collect()
    } else {
        gx.iter().map(|v| -v).collect()
    };
    super::cone::in_cone(&blk, &recession, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::{solve_misocp, BbOptions, ConeSpec, ConicOptions, MisocpProblem};
    use crate::sparse::CscMatrix;

    fn csc(rows: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut r = Vec::new();
        let mut c = Vec::new();
        let mut v = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    r.push(i);
                    c.push(j);
                    v.push(val);
                }
            }
        }
        CscMatrix::from_triplets(&r, &c, &v, nrows, ncols).unwrap()
    }

    /// Multi-block instance: `l=2` orthant rows, one dim-3 SOC, one dim-4
    /// SOC, `p=1` equality row, with a deliberately wide coefficient spread
    /// (`1e5`) so the sweeps have real work to do.
    fn multi_block_problem() -> ConicProblem {
        let n = 5usize;
        // Orthant rows (l=2): bound-like rows on x0, x1.
        // SOC block 1 (dim 3, rows 2..5): (x2, x3, x4-ish) with a `1e5` spread.
        // SOC block 2 (dim 4, rows 5..9): reuse x0..x3 with another spread.
        let g = csc(
            &[
                vec![1.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, -1e5, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, -1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, -1e-3],
                vec![-1.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, -1e4, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, -1.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, -1.0],
            ],
            9,
            n,
        );
        let h = vec![5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a = csc(&[vec![1.0, 1.0, 1.0, 1.0, 1.0]], 1, n);
        ConicProblem {
            c: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            a,
            b: vec![10.0],
            g,
            h,
            cone: ConeSpec {
                l: 2,
                soc: vec![3, 4],
            },
        }
    }

    /// Sentinel: after `compute()`, every row inside a given SOC block shares
    /// the *exact same* `e_g` scale (bit-for-bit) -- the invariant that keeps
    /// cone membership exact under scaling. Reverting to a naive per-row
    /// Ruiz sweep (independent row scaling, no block awareness) would break
    /// this.
    #[test]
    fn equil_soc_block_scale_is_constant() {
        let prob = multi_block_problem();
        let eq = Equilibrator::compute(&prob);
        let blk = Blocks::new(&prob.cone);
        for (bi, &off) in blk.soc_offsets().iter().enumerate() {
            let d = blk.soc[bi];
            let first = eq.e_g[off];
            for row in off..off + d {
                assert_eq!(
                    eq.e_g[row], first,
                    "block {bi} row {row}: e_g not block-constant"
                );
            }
        }
        // Sanity: scales are finite and strictly positive.
        for &v in eq.d.iter().chain(&eq.e_a).chain(&eq.e_g) {
            assert!(v.is_finite() && v > 0.0, "scale factor not positive: {v}");
        }
        assert!(eq.sigma_c.is_finite() && eq.sigma_c > 0.0);
    }

    /// `scale_problem` never changes the cone spec (dims are structural, not
    /// data).
    #[test]
    fn equil_preserves_cone_spec() {
        let prob = multi_block_problem();
        let eq = Equilibrator::compute(&prob);
        let scaled = eq.scale_problem(&prob);
        assert_eq!(scaled.cone, prob.cone);
        assert_eq!(scaled.a.nrows(), prob.a.nrows());
        assert_eq!(scaled.g.nrows(), prob.g.nrows());
        assert_eq!(scaled.c.len(), prob.c.len());
    }

    /// Cone membership is exactly preserved by the block-respecting row
    /// scaling: `s in K <=> (Dr_G s) in K`. Constructs `s` values strictly
    /// inside/outside `K` for each block kind and checks scaling never flips
    /// membership (up to the same `tol`).
    #[test]
    fn equil_scaling_preserves_cone_membership() {
        let prob = multi_block_problem();
        let eq = Equilibrator::compute(&prob);
        let blk = Blocks::new(&prob.cone);
        let m = prob.m();

        let mut inside = vec![0.0; m];
        inside[0] = 1.0;
        inside[1] = 2.0;
        // SOC block 1 (dim 3, off=2): (2, 1, 1) -> 2 >= sqrt(2), inside.
        inside[2] = 2.0;
        inside[3] = 1.0;
        inside[4] = 1.0;
        // SOC block 2 (dim 4, off=5): (5, 1, 1, 1) inside.
        inside[5] = 5.0;
        inside[6] = 1.0;
        inside[7] = 1.0;
        inside[8] = 1.0;
        assert!(super::super::cone::in_cone(&blk, &inside, 1e-9));

        let mut scaled = inside.clone();
        for i in 0..m {
            scaled[i] *= eq.e_g[i];
        }
        assert!(
            super::super::cone::in_cone(&blk, &scaled, 1e-9),
            "scaled-in-cone vector left the cone: {scaled:?}"
        );

        let mut outside = inside.clone();
        outside[2] = 0.5; // SOC block1 t-component too small -> outside.
        assert!(!super::super::cone::in_cone(&blk, &outside, 1e-9));
        let mut scaled_out = outside.clone();
        for i in 0..m {
            scaled_out[i] *= eq.e_g[i];
        }
        assert!(
            !super::super::cone::in_cone(&blk, &scaled_out, 1e-9),
            "scaled-outside-cone vector entered the cone: {scaled_out:?}"
        );
    }

    /// Algebraic identity (independent of IPM convergence): for *any*
    /// `(x', y', z')`, the scaled stationarity residual
    /// `r' = c' + A'^T y' + G'^T z'` and the original one
    /// `r = c + A^T y + G^T z` (with `x = D x'`, `y`, `z` unscaled per
    /// `unscale_result`'s formulas) satisfy `r[j] = r'[j] / (sigma_c * d[j])`
    /// exactly. This is the algebraic core of "equilibration cannot change
    /// the optimum" -- mirrors
    /// `linalg::ruiz::tests::dual_residual_unscale_factor_is_c_times_d`.
    #[test]
    fn equil_stationarity_residual_unscale_factor_is_sigma_c_times_d() {
        let prob = multi_block_problem();
        let eq = Equilibrator::compute(&prob);
        let scaled = eq.scale_problem(&prob);

        let mut lcg = 42u64;
        let mut next = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((lcg >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
        };
        let n = prob.n();
        let p = prob.p();
        let m = prob.m();
        let y_s: Vec<f64> = (0..p).map(|_| next()).collect();
        let z_s: Vec<f64> = (0..m).map(|_| next()).collect();

        let aty_s = kkt::spmtv(&scaled.a, &y_s);
        let gtz_s = kkt::spmtv(&scaled.g, &z_s);
        let r_s: Vec<f64> = (0..n).map(|j| scaled.c[j] + aty_s[j] + gtz_s[j]).collect();

        let mut y = y_s.clone();
        let mut z = z_s.clone();
        eq.unscale_y(&mut y);
        eq.unscale_z(&mut z);
        let aty = kkt::spmtv(&prob.a, &y);
        let gtz = kkt::spmtv(&prob.g, &z);
        let r: Vec<f64> = (0..n).map(|j| prob.c[j] + aty[j] + gtz[j]).collect();

        for j in 0..n {
            let expected = r_s[j] / (eq.sigma_c * eq.d[j]);
            assert!(
                (r[j] - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                "r[{j}]={} expected {expected} (r_s={}, sigma_c*d={})",
                r[j],
                r_s[j],
                eq.sigma_c * eq.d[j]
            );
        }
    }

    /// `Optimal` solves via `solve_socp` (equilibrated) and via raw
    /// `ipm::solve` (unscaled) on the same well-conditioned problem must
    /// agree on `x`/`objective`: equilibration must not change the answer
    /// when the unscaled path already converges cleanly.
    #[test]
    fn equil_matches_unscaled_solve_on_well_conditioned_problem() {
        let g = csc(&[vec![-1.0, 0.0], vec![0.0, -1.0]], 2, 2);
        let a = csc(&[vec![0.0, 1.0]], 1, 2);
        let prob = ConicProblem {
            c: vec![1.0, 0.0],
            a,
            b: vec![1.0],
            g,
            h: vec![0.0, 0.0],
            cone: ConeSpec { l: 0, soc: vec![2] },
        };
        let opts = ConicOptions::default();
        let raw = super::super::ipm::solve(&prob, &opts);
        let scaled_path = super::super::solve_socp(&prob, &opts);
        assert_eq!(raw.status, SolveStatus::Optimal, "{raw:?}");
        assert_eq!(scaled_path.status, SolveStatus::Optimal, "{scaled_path:?}");
        assert!(
            (raw.objective - scaled_path.objective).abs() < 1e-6,
            "raw={} scaled_path={}",
            raw.objective,
            scaled_path.objective
        );
        for j in 0..2 {
            assert!(
                (raw.x[j] - scaled_path.x[j]).abs() < 1e-5,
                "x[{j}]: raw={} scaled_path={}",
                raw.x[j],
                scaled_path.x[j]
            );
        }
    }

    /// Sentinel (root-cause reproduction, #9b): a badly-scaled *change of
    /// variables* `x = T x_tilde` (`T` diagonal, spread `1e-5..1e5`, the same
    /// "several orders of magnitude across the SOC block" pathology as the
    /// CBLIB `*_w` instances) on an otherwise trivial dim-3 SOC problem with
    /// a known closed-form optimum. The raw (unscaled) IPM must fail to
    /// reach that optimum in this budget; `solve_socp` (equilibrated) must
    /// reach it. Reverting equilibration (calling `ipm::solve` directly
    /// instead of `solve_socp`) makes this test fail -- see the `raw_*`
    /// assertions below, which are executed (not assumed).
    #[test]
    fn equil_recovers_optimum_under_extreme_column_scaling() {
        // Base: min x0 s.t. ||(x1,x2)|| <= x0, x1 = 3, x2 = 4 => x0* = 5.
        let g = csc(
            &[
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
                vec![0.0, 0.0, -1.0],
            ],
            3,
            3,
        );
        let a = csc(&[vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]], 2, 3);
        let base = ConicProblem {
            c: vec![1.0, 0.0, 0.0],
            a,
            b: vec![3.0, 4.0],
            g,
            h: vec![0.0, 0.0, 0.0],
            cone: ConeSpec { l: 0, soc: vec![3] },
        };
        let t = [1e-5, 1.0, 1e5];
        // x = T x_tilde: c1 = T c0, A1 = A0 T, G1 = G0 T (column-scale by t[j]);
        // b, h, cone unchanged. Known optimum: x_tilde*[j] = x0*[j] / t[j],
        // same objective (5.0) as the base problem -- see module doc's
        // change-of-variables derivation.
        let scale_col = |m: &CscMatrix| -> CscMatrix {
            let mut out = m.clone();
            for col in 0..m.ncols() {
                for k in m.col_ptr()[col]..m.col_ptr()[col + 1] {
                    out.values[k] = m.values()[k] * t[col];
                }
            }
            out
        };
        let badly_scaled = ConicProblem {
            c: base.c.iter().enumerate().map(|(j, &v)| v * t[j]).collect(),
            a: scale_col(&base.a),
            b: base.b.clone(),
            g: scale_col(&base.g),
            h: base.h.clone(),
            cone: base.cone.clone(),
        };
        let opts = ConicOptions::default();
        let known_obj = 5.0;

        let raw = super::super::ipm::solve(&badly_scaled, &opts);
        let raw_ok = raw.status == SolveStatus::Optimal
            && (raw.objective - known_obj).abs() < 1e-2 * known_obj;
        assert!(
            !raw_ok,
            "raw unscaled IPM unexpectedly recovered the optimum ({raw:?}); \
             sentinel needs a genuinely pathological instance -- widen `t`'s spread"
        );

        let scaled_path = super::super::solve_socp(&badly_scaled, &opts);
        assert_eq!(scaled_path.status, SolveStatus::Optimal, "{scaled_path:?}");
        assert!(
            (scaled_path.objective - known_obj).abs() < 1e-2 * known_obj,
            "obj={} want={known_obj}",
            scaled_path.objective
        );
        for j in 0..3 {
            let want = match j {
                0 => 5.0,
                1 => 3.0,
                2 => 4.0,
                _ => unreachable!(),
            } / t[j];
            assert!(
                (scaled_path.x[j] - want).abs() < 1e-2 * want.abs().max(1.0),
                "x[{j}]={} want={want}",
                scaled_path.x[j]
            );
        }
    }

    /// Sentinel for the MISOCP equilibration integration (`solve_misocp`
    /// equilibrates `base` once and feeds each B&B node's branching bounds
    /// through `build_relaxation` in *scaled* space as `lb/d[j]`, `ub/d[j]`).
    /// The integer column `x0` carries a deliberately large data coefficient
    /// (`1e4` on a redundant `x0 <= 2` orthant row), so equilibration produces
    /// `d[0] ~ 0.046`, far from `1` -- exercising the per-column bound scale.
    /// (`x1`'s column norm is anchored at `1` by its unit SOC coefficient, so
    /// it stays `d[1] = 1`; `x0` alone discriminates.) The known integer
    /// optimum is `(1, 1)` (a ball of radius `sqrt(2.5)` with integers in
    /// `[0, 2]^2`, same core as `misocp_ball_integer_optimum`). Flipping the
    /// bound-scale direction to `lb*d[j]`/`ub*d[j]` in `solve_misocp` tightens
    /// `x0`'s branching bound by `d[0]^2 ~ 2e-3` and cuts the optimum off
    /// (verified: the flip makes this assertion fail), so this discriminates
    /// the transform direction that the existing `d ~ 1` B&B tests cannot.
    #[test]
    fn misocp_equilibration_bound_scale_recovers_integer_optimum() {
        let r = 2.5_f64.sqrt();
        // Large coefficient on x0's redundant bound row forces d[0] far from 1.
        let g = csc(
            &[
                vec![1e4, 0.0],  // 1e4*x0 <= 2e4  (x0 <= 2, redundant)
                vec![0.0, 0.0],  // SOC head r
                vec![-1.0, 0.0], // SOC: x0
                vec![0.0, -1.0], // SOC: x1
            ],
            4,
            2,
        );
        let base = ConicProblem {
            c: vec![-1.0, -1.0],
            a: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            b: vec![],
            g,
            h: vec![2e4, r, 0.0, 0.0],
            cone: ConeSpec { l: 1, soc: vec![3] },
        };
        let mp = MisocpProblem {
            base,
            integers: vec![0, 1],
            int_lb: vec![0.0, 0.0],
            int_ub: vec![2.0, 2.0],
        };
        // Cross-check: equilibration really does move x0's column off 1.
        let eq = Equilibrator::compute(&mp.base);
        assert!(
            (eq.d[0] - 1.0).abs() > 0.5,
            "test precondition: d[0] must be far from 1 to discriminate bound \
             direction, got d={:?}",
            eq.d
        );
        let res = solve_misocp(&mp, &ConicOptions::default(), &BbOptions::default());
        assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
        assert!(
            (res.objective - (-2.0)).abs() < 1e-4,
            "obj={} want -2",
            res.objective
        );
        assert!((res.x[0] - 1.0).abs() < 1e-4, "x0={}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-4, "x1={}", res.x[1]);
    }
}
