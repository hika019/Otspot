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

/// Mehrotra's centering coefficient for the starting-point balancing shift
/// (Nocedal & Wright, *Numerical Optimization*, 2nd ed., eq. 14.39b).
const MEHROTRA_CENTERING: f64 = 0.5;

/// A cone block that is on or outside its boundary is pushed this far past it,
/// one unit inside along the cone identity (CVXOPT `conelp`: `a = 1 + ts`;
/// ECOS `bring2cone`: `alpha += 1`).
const CONE_INTERIOR_OFFSET: f64 = 1.0;

/// Mehrotra centering correction of the primal--dual starting iterate (Nocedal
/// & Wright, *Numerical Optimization*, eq. 14.39a/b). `s` and `z` come from two
/// independent KKT solves, so their complementarity is uneven across cones;
/// shifting each along the cone identity `e` by
/// `MEHROTRA_CENTERING * (s.z) / trace` equalises it and centers the start.
/// `e.s`, `e.z` are the cone traces of `s`, `z`.
fn balance_start(e: &[f64], s: &mut [f64], z: &mut [f64]) {
    let sz = dot(s, z);
    let tr_s = dot(e, s);
    let tr_z = dot(e, z);
    if sz > 0.0 && tr_s > 0.0 && tr_z > 0.0 {
        let ds = MEHROTRA_CENTERING * sz / tr_z;
        let dz = MEHROTRA_CENTERING * sz / tr_s;
        for (si, &ei) in s.iter_mut().zip(e) {
            *si += ds * ei;
        }
        for (zi, &ei) in z.iter_mut().zip(e) {
            *zi += dz * ei;
        }
    }
}

/// Solves the conic problem. `balance` enables the Mehrotra starting-point
/// complementarity balancing (see [`starting_point`]); root solves pass `true`,
/// branch-and-bound relaxation nodes pass `false`.
pub(super) fn solve(problem: &ConicProblem, opts: &ConicOptions, balance: bool) -> ConicResult {
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
        balance,
    ) {
        x = sx;
        y = sy;
        z = sz;
        s = ss;
    }

    for it in 0..opts.max_iter {
        if opts.stop_requested() {
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
                status = if opts.stop_requested() {
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

/// Pushes `v` into the interior of `K` along the cone identity `e`
/// (CVXOPT/Mehrotra heuristic): a block on or outside its boundary is shifted
/// one unit past it, an interior block is left untouched. Each **SOC block**
/// uses its own boundary distance rather than one global maximum, so bringing a
/// single violated block into the cone does not inject an `O(sqrt(#blocks))`
/// primal residual across the other blocks. The orthant is shifted as a single
/// block by its own `max(-v_i)`, matching ECOS/CVXOPT.
fn shift_into_cone(blk: &Blocks, v: &mut [f64]) {
    // Orthant block: one distance `max(-v_i)`; shift all rows if any is on or
    // outside the boundary.
    let mut d_orth = f64::NEG_INFINITY;
    for &vi in &v[..blk.l] {
        if -vi > d_orth {
            d_orth = -vi;
        }
    }
    if d_orth >= 0.0 {
        let shift = CONE_INTERIOR_OFFSET + d_orth;
        for vi in &mut v[..blk.l] {
            *vi += shift;
        }
    }
    // Each SOC block by its own boundary distance `||v_1|| - v_0`.
    for (bi, &off) in blk.soc_offsets().iter().enumerate() {
        let dim = blk.soc[bi];
        let nrm = v[off + 1..off + dim]
            .iter()
            .map(|a| a * a)
            .sum::<f64>()
            .sqrt();
        let dist = nrm - v[off];
        if dist >= 0.0 {
            v[off] += CONE_INTERIOR_OFFSET + dist;
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
    balance: bool,
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

    shift_into_cone(blk, &mut s0);
    shift_into_cone(blk, &mut z0);
    // Mehrotra complementarity balancing lifts the starting `mu` (`s.z`) by
    // roughly a factor of two while equalising complementarity across cones. On
    // a high-degree continuous root SOCP whose relative gap `mu*nu/(1+|c'x|)`
    // must reach `~1e-12` for `nu` in the thousands (e.g. cblib `qssp30`), that
    // extra headroom is what lets the last iterations cross `tol`. On a
    // tightly-bounded, low-degree branch-and-bound relaxation node the same
    // lift instead overshoots, so the iterate stalls one part in `~1e-9` short
    // of `tol` at the precision floor and breaks down. `balance` is therefore
    // requested only by the root solves (`solve_socp`/`solve_qcqp`), not by the
    // MISOCP/MIQCP node relaxations, which already converge from the plain
    // data-driven start.
    if balance {
        balance_start(e, &mut s0, &mut z0);
    }
    // Re-check: the shift and balancing scale by `s.z / trace`, which could in
    // principle overflow on pathological data; fall back to `s = z = e` rather
    // than seed the solve with a non-finite iterate.
    if !s0.iter().chain(&z0).all(|v| v.is_finite()) {
        return None;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::ConeSpec;

    /// Per-block shift: a violated orthant row must not perturb strictly
    /// interior SOC blocks. Under a global `max`-distance shift each interior
    /// SOC head would be inflated by `1 + 0.5 = 1.5` (to `3.5`), so the head
    /// assertions below fail.
    #[test]
    fn shift_into_cone_is_per_block_not_global() {
        let cone = ConeSpec {
            l: 1,
            soc: vec![3, 3],
        };
        let blk = Blocks::new(&cone);
        // layout: [orthant | soc0(head, tail, tail) | soc1(head, tail, tail)].
        // Orthant row violated by 0.5 (v0 = -0.5); both SOC blocks strictly
        // interior (head 2 > ||tail|| = 0).
        let mut v = vec![-0.5, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        shift_into_cone(&blk, &mut v);

        // Orthant shifted by 1 + 0.5 into +1.0 (one unit inside).
        assert!(
            (v[0] - 1.0).abs() < 1e-12,
            "orthant row must land 1.0 inside, got {}",
            v[0]
        );
        // Interior SOC heads must be untouched by the orthant's distance.
        assert!(
            (v[1] - 2.0).abs() < 1e-12,
            "interior SOC block 0 head must be untouched (global shift?), got {}",
            v[1]
        );
        assert!(
            (v[4] - 2.0).abs() < 1e-12,
            "interior SOC block 1 head must be untouched (global shift?), got {}",
            v[4]
        );
        // Tails never move.
        assert_eq!(&v[2..4], &[0.0, 0.0]);
        assert_eq!(&v[5..7], &[0.0, 0.0]);
    }

    /// A genuinely-outside SOC block is still pushed one unit inside, by its own
    /// distance, independent of the other blocks.
    #[test]
    fn shift_into_cone_pushes_each_outside_soc_by_its_own_distance() {
        let cone = ConeSpec {
            l: 0,
            soc: vec![3, 3],
        };
        let blk = Blocks::new(&cone);
        // Block 0 outside by ||(3,4)|| - 0 = 5; block 1 interior (head 10).
        let mut v = vec![0.0, 3.0, 4.0, 10.0, 1.0, 0.0];
        shift_into_cone(&blk, &mut v);
        // Block 0 head: 0 + (1 + 5) = 6, so head - ||tail|| = 6 - 5 = 1 inside.
        assert!((v[0] - 6.0).abs() < 1e-12, "block 0 head {}", v[0]);
        // Block 1 already interior (10 - 1 = 9 > 0): untouched.
        assert!((v[3] - 10.0).abs() < 1e-12, "block 1 head {}", v[3]);
    }

    /// Sentinel for the Mehrotra complementarity balancing ([`balance_start`]).
    /// An imbalanced start (per-row products `s_i z_i = [1, 100]`) must be
    /// equalised so the complementarity spread shrinks. Expected post-balance
    /// values are hand-derived (independent oracle) from the 1992 formula.
    ///
    /// Reverting `balance_start` to a no-op leaves `s = [1, 1]`, `z = [1, 100]`,
    /// so both the exact-value and the spread assertions below fail.
    #[test]
    fn balance_start_evens_out_complementarity() {
        let e = [1.0, 1.0]; // pure orthant identity
        let mut s = [1.0, 1.0];
        let mut z = [1.0, 100.0];
        // Before: products [1, 100], min/max spread = 0.01.
        balance_start(&e, &mut s, &mut z);
        // Independent hand calculation: sz = 101, e.s = 2, e.z = 101,
        //   ds = 0.5*101/101 = 0.5   => s = [1.5, 1.5]
        //   dz = 0.5*101/2  = 25.25  => z = [26.25, 125.25].
        assert!(
            (s[0] - 1.5).abs() < 1e-12 && (s[1] - 1.5).abs() < 1e-12,
            "s={s:?}"
        );
        assert!(
            (z[0] - 26.25).abs() < 1e-12 && (z[1] - 125.25).abs() < 1e-12,
            "z={z:?}"
        );
        // Products [39.375, 187.875]: spread lifted from 0.01 to ~0.21.
        let (p0, p1) = (s[0] * z[0], s[1] * z[1]);
        let spread = p0.min(p1) / p0.max(p1);
        assert!(
            spread > 0.2,
            "balanced complementarity spread {spread} not > 0.2 (unbalanced start was 0.01)"
        );
    }

    /// SOC-block balancing: the cone identity `e = (1, 0, ..., 0)` per SOC, so
    /// balancing moves only the block *heads*, leaving the tails untouched --
    /// the behaviour that is load-bearing on `qssp30`. Two dim-3 blocks with
    /// non-zero tails and imbalanced block complementarity `[10, 100]`.
    #[test]
    fn balance_start_shifts_soc_head_only_and_evens_complementarity() {
        let blk = Blocks::new(&ConeSpec {
            l: 0,
            soc: vec![3, 3],
        });
        let e = cone::identity(&blk); // (1,0,0, 1,0,0)
        let mut s = vec![5.0, 1.0, 0.0, 5.0, 1.0, 0.0];
        let mut z = vec![2.0, 0.0, 0.0, 20.0, 0.0, 0.0];
        // Per-block complementarity `s_blk . z_blk` = [10, 100], spread 0.1.
        let bdot = |s: &[f64], z: &[f64], o: usize| {
            s[o] * z[o] + s[o + 1] * z[o + 1] + s[o + 2] * z[o + 2]
        };
        let spread_before = {
            let (a, b) = (bdot(&s, &z, 0), bdot(&s, &z, 3));
            a.min(b) / a.max(b)
        };
        balance_start(&e, &mut s, &mut z);
        // First-principles: only the heads (the support of `e`) move.
        assert_eq!(&s[1..3], &[1.0, 0.0], "s block0 tail moved: {s:?}");
        assert_eq!(&s[4..6], &[1.0, 0.0], "s block1 tail moved: {s:?}");
        assert_eq!(&z[1..3], &[0.0, 0.0], "z block0 tail moved: {z:?}");
        assert_eq!(&z[4..6], &[0.0, 0.0], "z block1 tail moved: {z:?}");
        // Independent hand calculation: s.z = 110, e.s = 10, e.z = 22,
        //   ds = 0.5*110/22 = 2.5  => s heads 5 -> 7.5
        //   dz = 0.5*110/10 = 5.5  => z heads 2 -> 7.5, 20 -> 25.5.
        assert!(
            (s[0] - 7.5).abs() < 1e-12 && (s[3] - 7.5).abs() < 1e-12,
            "s heads {s:?}"
        );
        assert!(
            (z[0] - 7.5).abs() < 1e-12 && (z[3] - 25.5).abs() < 1e-12,
            "z heads {z:?}"
        );
        // First-principles: centering lifts the block complementarity spread.
        let spread_after = {
            let (a, b) = (bdot(&s, &z, 0), bdot(&s, &z, 3));
            a.min(b) / a.max(b)
        };
        assert!(
            spread_after > spread_before && spread_after > 0.25,
            "spread {spread_after} did not rise from {spread_before}"
        );
    }
}
