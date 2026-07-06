//! Primal--dual interior-point method (Nesterov--Todd scaling, Mehrotra
//! predictor--corrector) for the standard SOCP.

use std::time::Instant;

use super::cone::{self, Blocks};
use super::{ConicOptions, ConicProblem, ConicResult};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

fn csc_to_dense(a: &CscMatrix) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0; a.ncols()]; a.nrows()];
    let cp = a.col_ptr();
    let ri = a.row_ind();
    let va = a.values();
    for j in 0..a.ncols() {
        for k in cp[j]..cp[j + 1] {
            d[ri[k]][j] = va[k];
        }
    }
    d
}

fn matvec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn matvec_t(m: &[Vec<f64>], y: &[f64], ncols: usize) -> Vec<f64> {
    let mut out = vec![0.0; ncols];
    for (i, row) in m.iter().enumerate() {
        let yi = y[i];
        if yi != 0.0 {
            for (j, &a) in row.iter().enumerate() {
                out[j] += a * yi;
            }
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Solve the (dense-represented) KKT system, preferring a sparse faer LU and
/// falling back to the dense partial-pivot LU when the sparse path fails.
fn kkt_solve(dense: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    if let Some(x) = sparse_lu_solve(dense, rhs) {
        return Some(x);
    }
    lu_solve(dense.to_vec(), rhs.to_vec())
}

/// Sparse LU (faer simplicial backend) for the KKT matrix built from `dense`.
fn sparse_lu_solve(dense: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams, NumericLu};
    use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
    use faer::{Conj, MatMut, Par};

    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }
    // Build CSC (column-major, ascending row indices per column).
    let mut col_ptr = vec![0usize; n + 1];
    let mut row_ind: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for j in 0..n {
        for (i, row) in dense.iter().enumerate() {
            let v = row[j];
            if v != 0.0 {
                row_ind.push(i);
                values.push(v);
            }
        }
        col_ptr[j + 1] = row_ind.len();
    }
    let a_sym =
        unsafe { SymbolicSparseColMatRef::<usize>::new_unchecked(n, n, &col_ptr, None, &row_ind) };
    let symbolic = factorize_symbolic_lu(a_sym, LuSymbolicParams::default()).ok()?;
    let a_num = SparseColMatRef::<'_, usize, f64>::new(a_sym, &values);
    let mut numeric = NumericLu::<usize, f64>::new();
    let req = symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default());
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);
    let lu_ref = symbolic
        .factorize_numeric_lu(&mut numeric, a_num, Par::Seq, stack, Default::default())
        .ok()?;
    let mut b = rhs.to_vec();
    let req2 = symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    let mut mem2 = MemBuffer::new(req2);
    let stack2 = MemStack::new(&mut mem2);
    let bmat = MatMut::from_column_major_slice_mut(&mut b, n, 1);
    lu_ref.solve_in_place_with_conj(Conj::No, bmat, Par::Seq, stack2);
    if b.iter().all(|v| v.is_finite()) {
        Some(b)
    } else {
        None
    }
}

/// Solve a dense linear system `M u = rhs` by LU with partial pivoting.
fn lu_solve(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = m[col][col].abs();
        for r in (col + 1)..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            m.swap(col, piv);
            rhs.swap(col, piv);
        }
        let d = m[col][col];
        for r in (col + 1)..n {
            let f = m[r][col] / d;
            if f != 0.0 {
                for c in col..n {
                    m[r][c] -= f * m[col][c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    // back-substitution
    let mut u = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = rhs[i];
        for j in (i + 1)..n {
            acc -= m[i][j] * u[j];
        }
        u[i] = acc / m[i][i];
    }
    // A pivot just above the `1e-14` threshold can still be small enough that
    // back-substitution overflows to `NaN`/`Inf` on a near-singular system.
    // Same convention as `sparse_lu_solve`: only return a solution that is
    // actually usable, so `kkt_solve`'s caller falls back to a no-progress
    // (zero) direction instead of silently propagating garbage.
    if u.iter().all(|v| v.is_finite()) {
        Some(u)
    } else {
        None
    }
}

pub(super) fn solve(problem: &ConicProblem, opts: &ConicOptions) -> ConicResult {
    if let Err(e) = problem.validate() {
        return failed(problem, SolveStatus::NotSupported(e));
    }
    let blk = Blocks::new(&problem.cone);
    let n = problem.n();
    let p = problem.p();
    let m = problem.m();
    let nu = problem.cone.degree().max(1) as f64;

    let ad = csc_to_dense(&problem.a);
    let gd = csc_to_dense(&problem.g);
    let c = &problem.c;
    let bvec = &problem.b;
    let hvec = &problem.h;

    let nb = 1.0 + norm2(bvec);
    let nc = 1.0 + norm2(c);

    let e = cone::identity(&blk);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; p];
    let mut z = e.clone();
    let mut s = e.clone();

    let mut status = SolveStatus::MaxIterations;
    let mut iterations = 0;
    let mut last = (0.0, 0.0, 0.0);
    // Machine-verified certificates; set only by the certificate branches
    // below, so downstream consumers (B&B pruning) can distinguish a proven
    // Infeasible / Unbounded from an unverified status.
    let mut verified_infeas_cert: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut verified_primal_ray: Option<Vec<f64>> = None;

    for it in 0..opts.max_iter {
        if opts.deadline.is_some_and(|d| Instant::now() >= d) {
            status = SolveStatus::Timeout;
            iterations = it;
            break;
        }
        iterations = it + 1;
        // residuals
        let aty = matvec_t(&ad, &y, n);
        let gtz = matvec_t(&gd, &z, n);
        let rx: Vec<f64> = (0..n).map(|i| c[i] + aty[i] + gtz[i]).collect();
        let ax = matvec(&ad, &x);
        let ry: Vec<f64> = (0..p).map(|i| ax[i] - bvec[i]).collect();
        let gx = matvec(&gd, &x);
        let rz: Vec<f64> = (0..m).map(|i| gx[i] + s[i] - hvec[i]).collect();

        let sz = dot(&s, &z);
        let mu = sz / nu;
        let cx = dot(c, &x);
        let by = dot(bvec, &y);
        let hz = dot(hvec, &z);

        let pres = norm2(&ry) / nb;
        let dres = norm2(&rx) / nc;
        let gap = sz / (1.0 + cx.abs());
        last = (pres, dres, gap);

        if pres < opts.tol && dres < opts.tol && gap < opts.tol {
            status = SolveStatus::Optimal;
            break;
        }
        // Certificate-based early termination. Both tests are checked every
        // iteration (degenerate relaxations, e.g. B&B children with a
        // variable fixed by branching, diverge to non-finite iterates before
        // any magnitude threshold could fire) and both are scale-invariant:
        // each residual / value is measured against the magnitude sum of its
        // own terms, so neither the data scale nor the iterate scale can
        // fake a certificate. Using the same `opts.tol` on both sides makes
        // them complementary — a near-ray on a *feasible* degenerate problem
        // (e.g. a bound pair `x <= V`, `-x <= -V`) has relative ray residual
        // equal to its relative certificate value, so it can never pass the
        // "ray holds to tol" and "value negative beyond tol" gates together.
        //
        // Primal infeasibility (Farkas): z ∈ K*, A^T y + G^T z ≈ 0 and
        // b·y + h·z < 0.
        let farkas_val = -(by + hz);
        if farkas_val > 0.0 {
            let val_mag: f64 = bvec.iter().zip(&y).map(|(a, b)| (a * b).abs()).sum::<f64>()
                + hvec.iter().zip(&z).map(|(a, b)| (a * b).abs()).sum::<f64>();
            let ray_res = (0..n)
                .map(|i| (aty[i] + gtz[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let ray_mag = {
                let mut acc = vec![0.0; n];
                for (i, row) in ad.iter().enumerate() {
                    let yi = y[i].abs();
                    for (j, &a) in row.iter().enumerate() {
                        acc[j] += a.abs() * yi;
                    }
                }
                for (k, row) in gd.iter().enumerate() {
                    let zk = z[k].abs();
                    for (j, &g) in row.iter().enumerate() {
                        acc[j] += g.abs() * zk;
                    }
                }
                norm2(&acc)
            };
            let zn = norm2(&z);
            if farkas_val >= opts.tol * val_mag && ray_res <= opts.tol * ray_mag && zn > 0.0 {
                let zs: Vec<f64> = z.iter().map(|v| v / zn).collect();
                if cone::in_cone(&blk, &zs, opts.tol) {
                    let scale = (norm2(&y) + zn).max(1.0);
                    verified_infeas_cert = Some((
                        y.iter().map(|v| v / scale).collect(),
                        z.iter().map(|v| v / scale).collect(),
                    ));
                    status = SolveStatus::Infeasible;
                    break;
                }
            }
        }
        // Dual infeasibility / primal unboundedness (improving ray): d = x
        // with A d ≈ 0, -G d ∈ K and c·d < 0 proves the objective is
        // unbounded below along d.
        let descent = -cx;
        if descent > 0.0 {
            let c_mag: f64 = c.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let ax_res = norm2(&ax);
            let ax_mag = {
                let mut acc = 0.0;
                for row in &ad {
                    let t: f64 = row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                    acc += t * t;
                }
                acc.sqrt()
            };
            let g_mag = {
                let mut acc = 0.0;
                for row in &gd {
                    let t: f64 = row.iter().zip(&x).map(|(g, b)| (g * b).abs()).sum();
                    acc += t * t;
                }
                acc.sqrt()
            };
            let xn = norm2(&x);
            if descent >= opts.tol * c_mag && ax_res <= opts.tol * ax_mag && xn > 0.0 {
                let recession: Vec<f64> = if g_mag > 0.0 {
                    gx.iter().map(|v| -v / g_mag).collect()
                } else {
                    gx.iter().map(|v| -v).collect()
                };
                if cone::in_cone(&blk, &recession, opts.tol) {
                    verified_primal_ray = Some(x.iter().map(|v| v / xn).collect());
                    status = SolveStatus::Unbounded;
                    break;
                }
            }
        }

        // NT scaling.
        let sc = cone::nt_scaling(&blk, &s, &z);
        let lambda = sc.apply_winv(&blk, &s);

        // B = winv * G (m x n), H = B^T B (n x n). Built column-by-column so
        // the block-diagonal `winv` is never materialised as an `m x m`
        // matrix.
        let mut bmat = vec![vec![0.0; n]; m];
        let mut gcol = vec![0.0; m];
        for j in 0..n {
            for (k, row) in gd.iter().enumerate() {
                gcol[k] = row[j];
            }
            let bcol = sc.apply_winv(&blk, &gcol);
            for i in 0..m {
                bmat[i][j] = bcol[i];
            }
        }
        let mut hmat = vec![vec![0.0; n]; n];
        for r in 0..m {
            let brow = &bmat[r];
            for i in 0..n {
                let bi = brow[i];
                if bi != 0.0 {
                    let hi = &mut hmat[i];
                    for j in 0..n {
                        hi[j] += bi * brow[j];
                    }
                }
            }
        }
        // Regularise (quasidefinite): H += reg, bottom-right -reg.
        let reg = 1e-10;
        for i in 0..n {
            hmat[i][i] += reg;
        }
        // Assemble KKT [[H, A^T],[A, -reg I]].
        let nn = n + p;
        let mut kkt = vec![vec![0.0; nn]; nn];
        for i in 0..n {
            for j in 0..n {
                kkt[i][j] = hmat[i][j];
            }
        }
        for q in 0..p {
            for i in 0..n {
                kkt[i][n + q] = ad[q][i];
                kkt[n + q][i] = ad[q][i];
            }
            kkt[n + q][n + q] = -reg;
        }

        let winv_rz = sc.apply_winv(&blk, &rz);

        // ---- affine direction (r_c = -lambda) ----
        let rc_aff: Vec<f64> = lambda.iter().map(|v| -v).collect();
        let (_dx_a, _dy_a, dz_a, ds_a) =
            solve_dir(&kkt, &bmat, &sc, &blk, &gd, &rx, &ry, &rz, &winv_rz, &rc_aff);

        // affine step length
        let a_s = cone::max_step(&blk, &s, &ds_a, 1e16);
        let a_z = cone::max_step(&blk, &z, &dz_a, 1e16);
        let alpha_aff = a_s.min(a_z).min(1.0);
        let mut s_aff = vec![0.0; m];
        let mut z_aff = vec![0.0; m];
        for i in 0..m {
            s_aff[i] = s[i] + alpha_aff * ds_a[i];
            z_aff[i] = z[i] + alpha_aff * dz_a[i];
        }
        let mu_aff = dot(&s_aff, &z_aff) / nu;
        let sigma = if mu > 0.0 { (mu_aff / mu).powi(3) } else { 0.0 };

        // ---- corrector ----
        let dsw = sc.apply_winv(&blk, &ds_a); // W^{-1} ds
        let dzw = sc.apply_w(&blk, &dz_a); // W dz
        let corr = cone::jprod(&blk, &dsw, &dzw);
        let ll = cone::jprod(&blk, &lambda, &lambda);
        let target: Vec<f64> = (0..m)
            .map(|i| sigma * mu * e[i] - ll[i] - corr[i])
            .collect();
        let rc = cone::jdiv(&blk, &lambda, &target);
        let (dx, dy, dz, ds) =
            solve_dir(&kkt, &bmat, &sc, &blk, &gd, &rx, &ry, &rz, &winv_rz, &rc);

        // combined step length
        let a_s = cone::max_step(&blk, &s, &ds, 1e16);
        let a_z = cone::max_step(&blk, &z, &dz, 1e16);
        let alpha = (opts.step_frac * a_s.min(a_z)).min(1.0);
        if !alpha.is_finite() || alpha <= 0.0 {
            status = SolveStatus::NumericalError;
            break;
        }
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..p {
            y[i] += alpha * dy[i];
        }
        for i in 0..m {
            z[i] += alpha * dz[i];
            s[i] += alpha * ds[i];
        }
        // A near-singular KKT solve (e.g. from a degenerate or infeasible
        // relaxation) can silently return a non-finite direction even though
        // `alpha` itself computes to a finite, positive value: `cone::max_step`
        // only inspects the cone-membership boundary of `s`/`z`, not whether
        // `dx`/`dz`/`ds` themselves are NaN/Inf. Without this guard, a single
        // corrupted iterate propagates through every subsequent residual/
        // divergence check — all of which are false on NaN — so the loop
        // silently burns through `max_iter` and reports `MaxIterations`
        // instead of the true `NumericalError`, hiding the failure from
        // callers (e.g. MISOCP branch-and-bound) that rely on the status to
        // distinguish a proven conclusion from an untrustworthy point.
        if !x.iter().all(|v| v.is_finite())
            || !z.iter().all(|v| v.is_finite())
            || !s.iter().all(|v| v.is_finite())
            || !y.iter().all(|v| v.is_finite())
        {
            status = SolveStatus::NumericalError;
            break;
        }
    }

    let objective = dot(c, &x);
    ConicResult {
        status,
        objective,
        x,
        y,
        z,
        s,
        iterations,
        residuals: last,
        primal_ray: verified_primal_ray,
        infeas_cert: verified_infeas_cert,
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_dir(
    kkt: &[Vec<f64>],
    bmat: &[Vec<f64>],
    sc: &cone::Scaling,
    blk: &Blocks,
    gd: &[Vec<f64>],
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    winv_rz: &[f64],
    rc: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, p, m) = (rx.len(), ry.len(), rz.len());
    // rhs_x = -rx - B^T (winv*rz + rc)
    let mut t: Vec<f64> = (0..m).map(|i| winv_rz[i] + rc[i]).collect();
    let bt_t = matvec_t(bmat, &t, n);
    let mut rhs = vec![0.0; n + p];
    for i in 0..n {
        rhs[i] = -rx[i] - bt_t[i];
    }
    for i in 0..p {
        rhs[n + i] = -ry[i];
    }
    let sol = kkt_solve(kkt, &rhs).unwrap_or_else(|| vec![0.0; n + p]);
    let dx = sol[0..n].to_vec();
    let dy = sol[n..n + p].to_vec();
    // dz = winv( winv(G dx + rz) + rc )
    let gdx = matvec(gd, &dx);
    for i in 0..m {
        t[i] = gdx[i] + rz[i];
    }
    let w1 = sc.apply_winv(blk, &t);
    let inner: Vec<f64> = (0..m).map(|i| w1[i] + rc[i]).collect();
    let dz = sc.apply_winv(blk, &inner);
    // ds = -rz - G dx
    let ds: Vec<f64> = (0..m).map(|i| -rz[i] - gdx[i]).collect();
    (dx, dy, dz, ds)
}

fn failed(problem: &ConicProblem, status: SolveStatus) -> ConicResult {
    ConicResult {
        status,
        objective: f64::NAN,
        x: vec![0.0; problem.n()],
        y: vec![0.0; problem.p()],
        z: vec![0.0; problem.m()],
        s: vec![0.0; problem.m()],
        iterations: 0,
        residuals: (0.0, 0.0, 0.0),
        primal_ray: None,
        infeas_cert: None,
    }
}
