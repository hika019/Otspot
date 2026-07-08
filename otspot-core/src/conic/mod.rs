//! Conic optimization (SOCP/QCQP, incl. mixed-integer). [`solve_socp`] solves
//! the standard primal form `min c^T x  s.t.  A x = b,  G x + s = h, s in K`,
//! with cone `K = R_+^l x Q_{m_1} x ... x Q_{m_k}` (`Q_m = {(t,u): ||u|| <= t}`)
//! — rows ordered `l` orthant rows then each SOC block (`cone.soc`).

mod cone;
mod equil;
mod ipm;
mod kkt;
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
pub(crate) use qcqp::qcqp_matrix_to_csc;
pub use qcqp::{
    qcqp_from_qp_problem, solve_qcqp, solve_qp_problem_as_qcqp, to_conic, QcqpProblem, QcqpResult,
    QuadConstraint,
};

use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// Cone specification: nonnegative orthant of dimension `l`, then SOCs with dims in `soc`.
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

    /// Validate dimensional consistency and finiteness.
    ///
    /// Column counts are checked unconditionally even with zero rows (a
    /// mismatched `A` there previously reached a `debug_assert_eq!` panic
    /// deeper in the KKT solve; PR #25 review #36).
    pub fn validate(&self) -> Result<(), String> {
        if self.a.ncols() != self.n() {
            return Err("A column count != n".into());
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
        find_non_finite("c", &self.c)?;
        find_non_finite("b", &self.b)?;
        find_non_finite("h", &self.h)?;
        find_non_finite("A", self.a.values())?;
        find_non_finite("G", self.g.values())?;
        Ok(())
    }
}

/// Returns `Err` naming the first non-finite (NaN/±inf) entry of `xs`, if any.
fn find_non_finite(field: &str, xs: &[f64]) -> Result<(), String> {
    if let Some((i, v)) = xs.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("{field}[{i}] is not finite: {v}"));
    }
    Ok(())
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
    /// Unboundedness certificate: improving ray `d` with `A d ≈ 0`, `-G d ∈
    /// K` (asymptotically), `c^T d < 0`. `Some` only when `status == Unbounded`.
    pub primal_ray: Option<Vec<f64>>,
    /// Farkas infeasibility certificate `(y,z)`: `A^T y + G^T z ≈ 0`, `z ∈
    /// K^*`, `b^T y + h^T z < 0`. `Some` only when `status == Infeasible`.
    pub infeas_cert: Option<(Vec<f64>, Vec<f64>)>,
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
    /// Wall-clock deadline, checked once per IPM iteration or B&B node; `None` disables it.
    pub deadline: Option<std::time::Instant>,
}

impl Default for ConicOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: 100,
            step_frac: 0.99,
            deadline: None,
        }
    }
}

impl ConicOptions {
    /// Validate option ranges the IPM assumes but never checks itself.
    ///
    /// `max_iter == 0` is deliberately **not** rejected: it is a legitimate
    /// (if extreme) budget, used by tests to force an immediate
    /// `MaxIterations` without a certificate (see
    /// `misocp_unresolved_nodes_do_not_prove_infeasibility`).
    pub fn validate(&self) -> Result<(), String> {
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(format!("tol must be finite and > 0, got {}", self.tol));
        }
        if !self.step_frac.is_finite() || self.step_frac <= 0.0 || self.step_frac >= 1.0 {
            return Err(format!(
                "step_frac must be finite and in (0, 1), got {}",
                self.step_frac
            ));
        }
        Ok(())
    }
}

/// Solve a second-order cone program in standard form.
///
/// Equilibrates the data first (cone-block-respecting Ruiz scaling) so the
/// IPM sees well-conditioned magnitudes, then maps the
/// solution/duals/certificates back to `problem`'s original space. Exact up
/// to rounding — changes neither optimum nor feasible set (issue #9b).
pub fn solve_socp(problem: &ConicProblem, opts: &ConicOptions) -> ConicResult {
    // Validate before equilibrating: the Ruiz sweeps assume `validate()`'s
    // shape invariants, so reject an invalid problem here (ipm::solve keeps
    // its own guard as second line of defense).
    if let Err(e) = problem.validate() {
        return invalid_result(problem, SolveStatus::NotSupported(e));
    }
    if let Err(e) = opts.validate() {
        return invalid_result(problem, SolveStatus::NotSupported(e));
    }
    let eq = equil::Equilibrator::compute(problem);
    let scaled = eq.scale_problem(problem);
    let res = ipm::solve(&scaled, opts);
    let mut res = eq.unscale_result(problem, opts.tol, res);
    // Canonicalize only statuses whose `x` is not a usable iterate (PR #25
    // review #40): `Infeasible` -> `+inf`, `Unbounded` -> `-inf`. Inconclusive
    // statuses keep `dot(c, x)` of the real iterate in `res.x`; `NotSupported`
    // keeps its `NaN`.
    res.objective = match res.status {
        SolveStatus::Infeasible => f64::INFINITY,
        SolveStatus::Unbounded => f64::NEG_INFINITY,
        _ => res.objective,
    };
    res
}

/// `ConicResult` for a problem/options pair that failed validation, before
/// any solve is attempted. Mirrors `ipm::solve`'s own `failed()` (private to
/// that module) so `solve_socp` can reject invalid shapes up front, ahead of
/// equilibration's row/col sweeps.
fn invalid_result(problem: &ConicProblem, status: SolveStatus) -> ConicResult {
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
