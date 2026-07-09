//! Sentinel for the NT-scaled `ds` recovery (`ds = W (rc - W dz)`) in
//! `conic::kkt::solve_dir`.
//!
//! The CBLIB `*_w` instances have a permanently-active variable lower bound
//! whose orthant slack `s_i` collapses toward 0 while `z_i` stays `O(1)`, so
//! `W^2_i = s_i / z_i -> 0`. Recovering `ds` from the primal residual (`ds =
//! -rz - G dx`) then subtracts two `~1e-8` quantities whose true difference
//! (`~1e-9`) lies below the backward-stable absolute error floor of the
//! linear solve: the recovered `ds_i` takes the wrong sign, `a_s = -s_i /
//! ds_i` collapses, and the interior-point iteration stalls at
//! `MaxIterations`. The scaled recovery reads the complementarity target
//! directly through the cone algebra (no cancellation, no divide by the tiny
//! slack) and converges.
//!
//! Reverting `solve_dir` to `ds = -rz - G dx` turns this assertion from
//! `Optimal` back into `MaxIterations`. Data is gitignored; the test skips
//! gracefully when the file is absent (same convention as
//! `cbf_feasibility.rs`).

use otspot_core::conic::{solve_socp, ConicOptions};
use otspot_core::problem::SolveStatus;
use otspot_io::cbf::{parse_cbf, CbfProblem};
use std::path::Path;

#[test]
fn cblib_w_root_relaxation_reaches_optimal() {
    let path = Path::new("data/cblib_socp/20_0_1_w.cbf");
    if !path.exists() {
        eprintln!("[w-ds-sentinel] skip: data missing: {}", path.display());
        return;
    }
    let cbf = parse_cbf(path).expect("parse 20_0_1_w.cbf");
    let problem = match cbf {
        CbfProblem::Misocp { problem, .. } => problem.base,
        CbfProblem::Socp { problem, .. } => problem,
    };
    let opts = ConicOptions {
        tol: 1e-6,
        ..ConicOptions::default()
    };
    let res = solve_socp(&problem, &opts);
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "20_0_1_w root relaxation must converge (status={:?}, iters={}, \
         pres={:.3e} dres={:.3e} gap={:.3e})",
        res.status,
        res.iterations,
        res.residuals.0,
        res.residuals.1,
        res.residuals.2,
    );
}
