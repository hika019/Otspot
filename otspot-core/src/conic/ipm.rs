//! Primal--dual interior-point method (Nesterov--Todd scaling, Mehrotra
//! predictor--corrector) for the standard SOCP.

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
    Some(u)
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

    for it in 0..opts.max_iter {
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
        // Divergence-based infeasibility / unboundedness heuristics.
        let dual_obj = -by - hz;
        if pres < 1e-7 && cx < -1e11 && norm2(&x) > 1e8 {
            status = SolveStatus::Unbounded;
            break;
        }
        if dres < 1e-7 && dual_obj > 1e11 && (norm2(&y) + norm2(&z)) > 1e8 {
            status = SolveStatus::Infeasible;
            break;
        }

        // NT scaling.
        let sc = cone::nt_scaling(&blk, &s, &z);
        let lambda = cone::mat_apply(&sc.winv, &s);

        // B = winv * G (m x n), H = B^T B (n x n).
        let mut bmat = vec![vec![0.0; n]; m];
        for (i, brow) in bmat.iter_mut().enumerate() {
            let wrow = &sc.winv[i];
            for (j, bij) in brow.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (k, &wk) in wrow.iter().enumerate() {
                    if wk != 0.0 {
                        acc += wk * gd[k][j];
                    }
                }
                *bij = acc;
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

        let winv_rz = cone::mat_apply(&sc.winv, &rz);

        // ---- affine direction (r_c = -lambda) ----
        let rc_aff: Vec<f64> = lambda.iter().map(|v| -v).collect();
        let (_dx_a, _dy_a, dz_a, ds_a) = solve_dir(
            &kkt, &bmat, &sc, &gd, &rx, &ry, &rz, &winv_rz, &rc_aff, n, p, m,
        );

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
        let dsw = cone::mat_apply(&sc.winv, &ds_a); // W^{-1} ds
        let dzw = cone::mat_apply(&sc.w, &dz_a); // W dz
        let corr = cone::jprod(&blk, &dsw, &dzw);
        let ll = cone::jprod(&blk, &lambda, &lambda);
        let target: Vec<f64> = (0..m)
            .map(|i| sigma * mu * e[i] - ll[i] - corr[i])
            .collect();
        let rc = cone::jdiv(&blk, &lambda, &target);
        let (dx, dy, dz, ds) =
            solve_dir(&kkt, &bmat, &sc, &gd, &rx, &ry, &rz, &winv_rz, &rc, n, p, m);

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
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_dir(
    kkt: &[Vec<f64>],
    bmat: &[Vec<f64>],
    sc: &cone::Scaling,
    gd: &[Vec<f64>],
    rx: &[f64],
    ry: &[f64],
    rz: &[f64],
    winv_rz: &[f64],
    rc: &[f64],
    n: usize,
    p: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let w1 = cone::mat_apply(&sc.winv, &t);
    let inner: Vec<f64> = (0..m).map(|i| w1[i] + rc[i]).collect();
    let dz = cone::mat_apply(&sc.winv, &inner);
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
    }
}
