//! Sentinel for the data-driven primal--dual starting point in
//! `conic::ipm::solve` (`starting_point`), the Mehrotra/CVXOPT heuristic that
//! replaced the naive `s = z = e` initial iterate.
//!
//! Equilibration deliberately does not normalise the RHS `b`/`h` (only the
//! constraint matrix; see `conic::equil`), so a problem whose natural slack
//! scale is `O(V)` for large `V` leaves the equilibrated `h` at `O(V)`. From
//! the unit-scaled `s = z = e = 1`, the very first Newton direction is then
//! `O(V)` while the fraction-to-boundary rule clamps the step to `~1/V`; the
//! interior-point iteration crawls (centering parameter pinned near 1, `mu`
//! rising) and, past a large-enough `V`, never converges inside the default
//! `max_iter`. This reproduces the CBLIB `*_w` root-relaxation stall at small,
//! CI-runnable scale (no external data).
//!
//! Reverting `starting_point` (falling back to `s = z = e`) turns the assertion
//! below from `Optimal` into `MaxIterations` with a grossly wrong objective
//! (measured: iters=100, obj=+1.9e2 vs the true -1e4). This is a pure
//! starting-point discriminator: all three `ds`-recovery forms converge on it
//! identically, so it isolates the initialization from `kkt::solve_dir`.

use otspot_core::conic::{ConeSpec, ConicOptions, ConicProblem};
use otspot_core::problem::SolveStatus;
use otspot_core::sparse::CscMatrix;

/// `maximize sum(x)` over `{0 <= x_i <= U}` intersect the ball `||x|| <= R`.
/// Orthant rows give `h = U` (large RHS); the SOC head gives `h = R`. With
/// `R/sqrt(k) < U` the ball binds, so the optimum is `x_i = R/sqrt(k)`,
/// `obj = -R*sqrt(k)` (independent oracle: closed form, not the solver).
fn ball_box(k: usize, u: f64, r: f64) -> (ConicProblem, f64) {
    let m_orth = 2 * k;
    let m = m_orth + (k + 1);
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let mut h = vec![0.0; m];
    h[..k].fill(u); // x_i <= U rows
    for i in 0..k {
        rows.push(i); // x_i <= U
        cols.push(i);
        vals.push(1.0);
        rows.push(k + i); // x_i >= 0
        cols.push(i);
        vals.push(-1.0);
    }
    h[m_orth] = r; // SOC head: ||x|| <= R
    for i in 0..k {
        rows.push(m_orth + 1 + i);
        cols.push(i);
        vals.push(-1.0);
    }
    let g = CscMatrix::from_triplets(&rows, &cols, &vals, m, k).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, k).unwrap();
    let prob = ConicProblem {
        c: vec![-1.0; k],
        a,
        b: vec![],
        g,
        h,
        cone: ConeSpec {
            l: m_orth,
            soc: vec![k + 1],
        },
    };
    let xi = u.min(r / (k as f64).sqrt());
    (prob, -(k as f64) * xi)
}

#[test]
fn large_rhs_socp_converges_from_data_driven_start() {
    // U=1e5 (inactive box, |h|~1e5) so the naive unit start's first direction
    // is O(1e5); R=1e3 => optimum x_i=100, obj=-1e4.
    let (prob, known_obj) = ball_box(100, 1e5, 1e3);
    let res = otspot_core::conic::solve_socp(&prob, &ConicOptions::default());
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "large-RHS SOCP must converge from the data-driven start \
         (status={:?}, iters={}, obj={:.6e}, want {:.6e})",
        res.status,
        res.iterations,
        res.objective,
        known_obj,
    );
    let rel = (res.objective - known_obj).abs() / (1.0 + known_obj.abs());
    assert!(
        rel < 1e-6,
        "objective {:.8e} != known {:.8e} (rel {:.2e})",
        res.objective,
        known_obj,
        rel
    );
}
