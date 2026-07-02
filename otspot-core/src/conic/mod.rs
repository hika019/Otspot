//! Conic optimization: second-order cone programming (SOCP) and quadratically
//! constrained programs (QCQP), plus their mixed-integer variants.
//!
//! Standard primal form solved by [`solve_socp`]:
//!
//! ```text
//! minimize    c^T x
//! subject to  A x = b            (equalities)
//!             G x + s = h,  s in K
//! ```
//!
//! with cone `K = R_+^l  x  Q_{m_1} x ... x Q_{m_k}` where each second-order
//! cone is `Q_m = { (t, u) in R x R^{m-1} : ||u||_2 <= t }`. The rows of `G`
//! (and entries of `h`, `s`) are ordered: the `l` nonnegative-orthant rows
//! first, then each second-order-cone block in `cone.soc` order.

mod cone;
mod ipm;
mod misocp;
mod nonconvex;
mod qcqp;
#[cfg(test)]
mod tests;

pub use misocp::{solve_miqcp, solve_misocp, BbOptions, MisocpProblem, MisocpResult};
pub use nonconvex::{
    solve_global_miqcp, solve_global_qcqp, GQuadConstraint, GlobalOptions, GlobalResult,
    NonconvexQcqp,
};
pub use qcqp::{
    qcqp_from_qp_problem, solve_qcqp, solve_qp_problem_as_qcqp, to_conic, QcqpProblem, QcqpResult,
    QuadConstraint,
};

use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// Cone specification: a nonnegative orthant of dimension `l` followed by
/// second-order cones with the dimensions listed in `soc`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConeSpec {
    /// Dimension of the leading nonnegative orthant `R_+^l`.
    pub l: usize,
    /// Dimensions of the second-order cones (each `>= 1`).
    pub soc: Vec<usize>,
}

impl ConeSpec {
    /// Total dimension of the cone (number of conic rows).
    pub fn dim(&self) -> usize {
        self.l + self.soc.iter().sum::<usize>()
    }

    /// Barrier degree `nu = l + (number of second-order cones)`.
    pub fn degree(&self) -> usize {
        self.l + self.soc.len()
    }
}

/// A second-order cone program in standard conic form.
#[derive(Debug, Clone)]
pub struct ConicProblem {
    /// Objective coefficients (length `n`).
    pub c: Vec<f64>,
    /// Equality matrix `A` (`p x n`); use zero rows when there are no equalities.
    pub a: CscMatrix,
    /// Equality right-hand side (length `p`).
    pub b: Vec<f64>,
    /// Conic inequality matrix `G` (`m x n`), `m == cone.dim()`.
    pub g: CscMatrix,
    /// Conic right-hand side (length `m`).
    pub h: Vec<f64>,
    /// Cone specification.
    pub cone: ConeSpec,
}

impl ConicProblem {
    /// Number of variables `n`.
    pub fn n(&self) -> usize {
        self.c.len()
    }
    /// Number of equality rows `p`.
    pub fn p(&self) -> usize {
        self.b.len()
    }
    /// Number of conic rows `m`.
    pub fn m(&self) -> usize {
        self.h.len()
    }

    /// Validate dimensional consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.a.ncols() != self.n() && self.p() > 0 {
            return Err("A has wrong column count".into());
        }
        if self.g.ncols() != self.n() {
            return Err("G column count != n".into());
        }
        if self.a.nrows() != self.p() {
            return Err("A row count != b length".into());
        }
        if self.g.nrows() != self.m() {
            return Err("G row count != h length".into());
        }
        if self.cone.dim() != self.m() {
            return Err("cone dim != m".into());
        }
        for &d in &self.cone.soc {
            if d == 0 {
                return Err("second-order cone dim must be >= 1".into());
            }
        }
        Ok(())
    }
}

/// Result of a conic solve.
#[derive(Debug, Clone)]
pub struct ConicResult {
    /// Solve status.
    pub status: SolveStatus,
    /// Primal objective `c^T x`.
    pub objective: f64,
    /// Primal solution `x` (length `n`).
    pub x: Vec<f64>,
    /// Equality dual `y` (length `p`).
    pub y: Vec<f64>,
    /// Conic dual `z` (length `m`), lies in the dual cone (self-dual here).
    pub z: Vec<f64>,
    /// Conic slack `s = h - G x` (length `m`), lies in `K`.
    pub s: Vec<f64>,
    /// Iterations performed.
    pub iterations: usize,
    /// Final `(primal_res, dual_res, duality_gap)` relative metrics.
    pub residuals: (f64, f64, f64),
}

/// Options controlling the conic interior-point solver.
#[derive(Debug, Clone)]
pub struct ConicOptions {
    /// Convergence tolerance on relative primal/dual residual and gap.
    pub tol: f64,
    /// Maximum interior-point iterations.
    pub max_iter: usize,
    /// Fraction-to-boundary step damping (`(0,1)`).
    pub step_frac: f64,
}

impl Default for ConicOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: 100,
            step_frac: 0.99,
        }
    }
}

/// Solve a second-order cone program in standard form.
pub fn solve_socp(problem: &ConicProblem, opts: &ConicOptions) -> ConicResult {
    ipm::solve(problem, opts)
}
