//! Primal--dual interior-point method (Nesterov--Todd scaling, Mehrotra
//! predictor--corrector) for the standard SOCP.
//!
//! The Newton system is solved via the sparse augmented quasidefinite KKT
//! system in [`super::kkt`] (no dense `A`/`G`/KKT densification anywhere in
//! this module -- see `super::kkt`'s module doc for the system derivation).

use std::time::Instant;

use super::cone::{self, Blocks};
use super::kkt;
use super::{ConicOptions, ConicProblem, ConicResult};
use crate::linalg::kkt_solver::KktConfig;
use crate::problem::SolveStatus;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub(super) fn solve(problem: &ConicProblem, opts: &ConicOptions) -> ConicResult {
    if let Err(e) = problem.validate() {
        return failed(problem, SolveStatus::NotSupported(e));
    }
    if let Err(e) = opts.validate() {
        return failed(problem, SolveStatus::NotSupported(e));
    }
    let blk = Blocks::new(&problem.cone);
    let n = problem.n();
    let p = problem.p();
    let m = problem.m();
    let nu = problem.cone.degree().max(1) as f64;

    let c = &problem.c;
    let bvec = &problem.b;
    let hvec = &problem.h;

    let nb = 1.0 + norm2(bvec);
    let nc = 1.0 + norm2(c);
    let nh = 1.0 + norm2(hvec);

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

    let mut kkt_caches = kkt::build_kkt_caches(&problem.a, &problem.g, &blk, n, p, opts.deadline);
    let kkt_cfg = KktConfig::default();

    if let Some((sx, sy, sz, ss)) = starting_point(
        problem,
        &blk,
        n,
        p,
        m,
        &e,
        &mut kkt_caches,
        &kkt_cfg,
        opts.deadline,
    ) {
        x = sx;
        y = sy;
        z = sz;
        s = ss;
    }

    for it in 0..opts.max_iter {
        if opts.deadline.is_some_and(|d| Instant::now() >= d) {
            status = SolveStatus::Timeout;
            iterations = it;
            break;
        }
        iterations = it + 1;
        // residuals
        let aty = kkt::spmtv(&problem.a, &y);
        let gtz = kkt::spmtv(&problem.g, &z);
        let rx: Vec<f64> = (0..n).map(|i| c[i] + aty[i] + gtz[i]).collect();
        let ax = kkt::spmv(&problem.a, &x);
        let ry: Vec<f64> = (0..p).map(|i| ax[i] - bvec[i]).collect();
        let gx = kkt::spmv(&problem.g, &x);
        let rz: Vec<f64> = (0..m).map(|i| gx[i] + s[i] - hvec[i]).collect();

        let sz = dot(&s, &z);
        let mu = sz / nu;
        let cx = dot(c, &x);
        let by = dot(bvec, &y);
        let hz = dot(hvec, &z);

        // Primal feasibility spans both the equality residual (`ry = Ax - b`)
        // and the conic residual (`rz = Gx + s - h`, `s in K`). The conic term
        // is load-bearing for soundness: with `ds` recovered from scaled
        // complementarity (see `kkt::solve_dir`) `s` no longer tracks `h - Gx`
        // implicitly, so an infeasible point can reach small gap/dres while
        // `rz` stays `O(1)` -- omitting `rz` here would report such a point as
        // Optimal (false optimal on an infeasible relaxation, breaking B&B
        // pruning; cf. `socp_degenerate_fixed_var_infeasible_gets_certificate`).
        let pres = (norm2(&ry) / nb).max(norm2(&rz) / nh);
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
                let y_abs: Vec<f64> = y.iter().map(|v| v.abs()).collect();
                let z_abs: Vec<f64> = z.iter().map(|v| v.abs()).collect();
                let mut acc = vec![0.0; n];
                kkt::spmtv_abs_accum(&problem.a, &y_abs, &mut acc);
                kkt::spmtv_abs_accum(&problem.g, &z_abs, &mut acc);
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
            let x_abs: Vec<f64> = x.iter().map(|v| v.abs()).collect();
            let ax_mag = norm2(&kkt::spmv_abs(&problem.a, &x_abs));
            let g_mag = norm2(&kkt::spmv_abs(&problem.g, &x_abs));
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

        // ---- affine direction (r_c = -lambda) ----
        let rc_aff: Vec<f64> = lambda.iter().map(|v| -v).collect();
        let probe_rhs = kkt::build_rhs(&sc, &blk, n, p, m, &rx, &ry, &rz, &rc_aff);
        let factor = match kkt::factorize_with_retry(
            &mut kkt_caches,
            &sc,
            &blk,
            &probe_rhs,
            opts.deadline,
            &kkt_cfg,
        ) {
            Some(f) => f,
            None => {
                status = if opts.deadline.is_some_and(|d| Instant::now() >= d) {
                    SolveStatus::Timeout
                } else {
                    SolveStatus::NumericalError
                };
                break;
            }
        };
        let (_dx_a, _dy_a, dz_a, ds_a) = kkt::solve_dir(
            &factor, &problem.g, &sc, &blk, n, p, m, &rx, &ry, &rz, &rc_aff,
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
        let dsw = sc.apply_winv(&blk, &ds_a); // W^{-1} ds
        let dzw = sc.apply_w(&blk, &dz_a); // W dz
        let corr = cone::jprod(&blk, &dsw, &dzw);
        let ll = cone::jprod(&blk, &lambda, &lambda);
        let target: Vec<f64> = (0..m)
            .map(|i| sigma * mu * e[i] - ll[i] - corr[i])
            .collect();
        let rc = cone::jdiv(&blk, &lambda, &target);
        let (dx, dy, dz, ds) =
            kkt::solve_dir(&factor, &problem.g, &sc, &blk, n, p, m, &rx, &ry, &rz, &rc);

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

/// Signed distance of `v` from the interior of `K` (positive outside, negative
/// strictly inside): `max(-v_i)` over orthant rows, `||v_1|| - v_0` per SOC.
fn cone_distance(blk: &Blocks, v: &[f64]) -> f64 {
    let mut d = f64::NEG_INFINITY;
    for &vi in &v[..blk.l] {
        if -vi > d {
            d = -vi;
        }
    }
    for (bi, &off) in blk.soc_offsets().iter().enumerate() {
        let dim = blk.soc[bi];
        let nrm = v[off + 1..off + dim]
            .iter()
            .map(|a| a * a)
            .sum::<f64>()
            .sqrt();
        let gap = nrm - v[off];
        if gap > d {
            d = gap;
        }
    }
    d
}

/// Pushes `v` strictly into `K` along the cone identity `e` (CVXOPT/Mehrotra
/// heuristic): if `v` is on or outside the boundary (`cone_distance >= 0`),
/// shift by `(1 + distance) e`, which lands `v` exactly one unit inside. A
/// vector already interior at the data's own scale is left untouched, so the
/// shifted point inherits the problem's natural magnitude.
fn shift_into_cone(blk: &Blocks, v: &mut [f64], e: &[f64]) {
    let dist = cone_distance(blk, v);
    if dist >= 0.0 {
        let shift = 1.0 + dist;
        for (vi, &ei) in v.iter_mut().zip(e) {
            *vi += shift * ei;
        }
    }
}

/// Data-driven primal--dual starting point (Mehrotra/CVXOPT). Solves the two
/// unit-scaled (`W = I`) KKT systems
/// `K [x;y;z] = [0;b;h]` (primal, giving `s = h - G x`) and
/// `K [x;y;z] = [-c;0;0]` (dual, giving the dual slack `z`),
/// then shifts `s` and `z` into the cone interior with [`shift_into_cone`]. The
/// resulting iterate carries the problem's natural scale, so the
/// fraction-to-boundary rule no longer clamps the first steps to `~1e-3` (the
/// naive `s = z = e` start does exactly that when the equilibrated RHS is
/// `O(100)`, since equilibration deliberately does not normalise `b`/`h`; see
/// `equil`). Returns `None` (fall back to `s = z = e`) if the unit-scaled
/// factorization fails or produces a non-finite iterate.
/// Primal--dual starting iterate `(x, y, z, s)` from [`starting_point`].
type StartingPoint = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[allow(clippy::too_many_arguments)]
fn starting_point(
    problem: &ConicProblem,
    blk: &Blocks,
    n: usize,
    p: usize,
    m: usize,
    e: &[f64],
    caches: &mut kkt::KktCaches,
    kkt_cfg: &KktConfig,
    deadline: Option<Instant>,
) -> Option<StartingPoint> {
    let sc = cone::nt_scaling(blk, e, e); // s = z = e => W = I
    let n_e = n + blk.n_border();
    let total = n_e + p + m + blk.n_border();

    // Primal RHS (0, b, h); reused as the health-probe RHS for the factor.
    let mut rhs_primal = vec![0.0; total];
    rhs_primal[n_e..n_e + p].copy_from_slice(&problem.b);
    rhs_primal[n_e + p..n_e + p + m].copy_from_slice(&problem.h);

    let factor = kkt::factorize_with_retry(caches, &sc, blk, &rhs_primal, deadline, kkt_cfg)?;

    let mut sol = vec![0.0; total];
    factor.solve(&rhs_primal, &mut sol);
    let x0 = sol[0..n].to_vec();
    // Row `G x - z = h` (W = I) gives `z = G x - h`, so the primal slack that
    // zeroes `rz = G x + s - h` is `s = h - G x = -z`.
    let mut s0: Vec<f64> = sol[n_e + p..n_e + p + m].iter().map(|v| -v).collect();

    let mut rhs_dual = vec![0.0; total];
    for (dst, &ci) in rhs_dual[..n].iter_mut().zip(&problem.c) {
        *dst = -ci;
    }
    factor.solve(&rhs_dual, &mut sol);
    let y0 = sol[n_e..n_e + p].to_vec();
    // Row `c + A^T y + G^T z = 0` makes the `dz` component the dual slack `z`.
    let mut z0 = sol[n_e + p..n_e + p + m].to_vec();

    let finite = x0
        .iter()
        .chain(&s0)
        .chain(&z0)
        .chain(&y0)
        .all(|v| v.is_finite());
    if !finite {
        return None;
    }

    shift_into_cone(blk, &mut s0, e);
    shift_into_cone(blk, &mut z0, e);
    Some((x0, y0, z0, s0))
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
