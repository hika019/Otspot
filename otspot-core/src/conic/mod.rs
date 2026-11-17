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
pub mod presolve;
mod qcqp;
mod qcqp_classify;
mod qcqp_fbbt;
#[cfg(test)]
mod tests;

pub use presolve::{kkt_dim, tightened_tol, ConicReduction, ReductionStats};

pub use misocp::{solve_miqcp, solve_misocp, BbOptions, MisocpProblem, MisocpResult};
pub use nonconvex::{
    solve_global_miqcp, solve_global_qcqp, GQuadConstraint, GlobalOptions, GlobalResult,
    GlobalSeed, NonconvexQcqp,
};
pub(crate) use qcqp::qcqp_matrix_to_csc;
pub use qcqp::{
    qcqp_from_qp_problem, solve_qcqp, solve_qp_problem_as_qcqp, to_conic, QcqpProblem, QcqpResult,
    QuadConstraint,
};
pub use qcqp_classify::{
    classify_bound_requirements, effective_bound_requirement, require_finite_bounds,
    BoundRequirement,
};
pub(crate) use qcqp_classify::{
    coo_quad_row_diag_support, effective_bound_requirement_from_coo,
    require_finite_bounds_from_coo, CooQuadRow,
};
pub use qcqp_fbbt::{derive_root_box, RootFbbtBox, ROOT_FBBT_MAX_ROUNDS, ROOT_FBBT_ULP_SLACK};

use crate::problem::SolveStatus;
use otspot_num::sparse::CscMatrix;

#[cfg(test)]
thread_local! {
    /// Test-only cancel injection for `solve_socp`'s post-unscale backstop:
    /// counts calls to `test_maybe_cancel_after_ipm_solve`
    /// (invoked right after `ipm::solve` returns, before
    /// `Equilibrator::unscale_result`); once the count reaches
    /// `CANCEL_AFTER_POST_SOLVE_COMMIT`, flips the real `cancel_flag`
    /// `AtomicBool` a test threads through `ConicOptions::cancel_flag`, so
    /// the backstop's `opts.stop_requested()` observes a freshly-fired
    /// cancellation from within the equilibration/unscale window, without
    /// racing wall-clock time. Mirrors `misocp::DEADLINE_AFTER_NODE`'s
    /// fault-injection pattern.
    static POST_SOLVE_COMMIT_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CANCEL_AFTER_POST_SOLVE_COMMIT: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn test_maybe_cancel_after_ipm_solve(opts: &ConicOptions) {
    let count = POST_SOLVE_COMMIT_COUNT.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if CANCEL_AFTER_POST_SOLVE_COMMIT.with(std::cell::Cell::get) == Some(count) {
        if let Some(flag) = &opts.cancel_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

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
    /// Per-variable domain bounds `lb <= x_j <= ub` (length `n`), or empty
    /// for "all variables free". This is the *domain representation* of
    /// bound-type rows: the conic tier-1 presolve
    /// ([`presolve`]) absorbs singleton orthant identity rows (`±x_j + s_k =
    /// h_k`, the CBF `VAR L+`/`L-`/`L=` encoding) into this field so that
    /// fixed variables and domain conflicts become reductions instead of
    /// staying matrix rows. The IPM itself does not consume this field yet
    /// (consuming it natively is the tier-2 "variable cones/bounds"
    /// formulation work in the research report; out of scope here):
    /// [`solve_socp`] therefore re-materializes any non-empty `var_bounds`
    /// into orthant rows before handing the problem to the IPM, keeping the
    /// solve equivalent.
    pub var_bounds: Vec<(f64, f64)>,
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
    /// deeper in the KKT solve; PR #25 review 36).
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
        if !self.var_bounds.is_empty() && self.var_bounds.len() != self.n() {
            return Err("var_bounds length must be 0 (all free) or n".into());
        }
        for (j, &(lb, ub)) in self.var_bounds.iter().enumerate() {
            if lb.is_nan() || ub.is_nan() {
                return Err(format!("var_bounds[{j}] is NaN"));
            }
            if lb > ub {
                return Err(format!("var_bounds[{j}]: lb {lb} > ub {ub}"));
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

/// Default IPM iteration budget for a from-scratch conic solve (PR #25 review
/// #27). The conic IPM is a single Mehrotra predictor-corrector attempt with
/// no equivalent to the QP path's multi-attempt retry loop: when the caller
/// leaves `IpmOptions::max_iter` at its `usize::MAX` default,
/// `qp::ipm_solver::attempt` runs up to ~10 restart attempts (different Ruiz/
/// tightening combinations) each capped at `MAX_ITER_PER_ATTEMPT = 500`
/// iterations — a ~5,000-iteration practical ceiling, with the wall-clock
/// deadline as the actual guard for hard problems. The old default of `100`
/// was far below that ceiling: a hard QCQP/SOCP needing more than 100
/// iterations hit `MaxIterations` here while the identical problem solved
/// through the QP path kept going. This matches that practical ceiling
/// instead, so the conic IPM gets comparable room to converge while
/// `stop_requested` (checked every iteration, see
/// `ConicOptions::stop_requested`) stays the primary guard and this remains
/// only a finite safety net when no deadline is set.
pub const DEFAULT_CONIC_MAX_ITER: usize = 5_000;

/// Options controlling the conic interior-point solver.
#[derive(Debug, Clone)]
pub struct ConicOptions {
    /// Convergence tolerance on relative primal/dual residual and gap.
    pub tol: f64,
    /// Maximum interior-point iterations. Default: [`DEFAULT_CONIC_MAX_ITER`]
    /// -- the wall-clock `deadline`/`cancel_flag` (checked every iteration via
    /// `stop_requested`) is the primary guard for a hard problem, not this cap.
    pub max_iter: usize,
    /// Fraction-to-boundary step damping (`(0,1)`).
    pub step_frac: f64,
    /// Wall-clock deadline, checked once per IPM iteration or B&B node; `None` disables it.
    pub deadline: Option<std::time::Instant>,
    /// Shared cancellation flag mirroring `SolverOptions::cancel_flag`. The
    /// IPM iteration loop checks this alongside `deadline` (see
    /// `stop_requested`) so a flag fired from another thread stops a
    /// solve already in flight, not only one that starts after the flag is
    /// already set. `None` disables it (default).
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for ConicOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: DEFAULT_CONIC_MAX_ITER,
            step_frac: 0.99,
            deadline: None,
            cancel_flag: None,
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

    /// Whether an external stop condition fired: wall-clock deadline expired
    /// or the cancel flag was set. Mirrors
    /// `SolverOptions::external_stop_requested` for the conic IPM loop.
    pub(crate) fn stop_requested(&self) -> bool {
        self.deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
            || self
                .cancel_flag
                .as_ref()
                .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// Solve a second-order cone program in standard form.
///
/// Pipeline: validate → tier-1 presolve ([`presolve`]: equality
/// substitution, fixed-variable elimination, domain identity-row absorption)
/// → equilibration (cone-block-respecting Ruiz scaling) → `ipm::solve` →
/// unscale + postsolve, with the result re-verified against `problem`'s
/// *original* data before it is returned. The solve sees well-conditioned
/// (possibly smaller) data, but every returned status/certificate is backed
/// by an original-space verification, and any verification failure re-solves
/// the original problem without presolve — exact up to rounding, changes
/// neither optimum nor feasible set.
///
/// When presolve is active, `iterations`/`residuals` describe the reduced
/// problem's solve (the reduced IPM tolerance is tightened by the reduction's
/// growth factor, see [`presolve::tightened_tol`]); `x`/`y`/`z`/`s` and the
/// objective are always original-space.
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
    // Check the stop condition before paying for presolve, equilibration
    // (`EQUIL_SWEEPS * nnz` Ruiz sweeps) and `ipm::solve`'s KKT symbolic
    // factorization: an already-expired deadline or a preset cancel flag must
    // return `Timeout` immediately, not after that up-front work runs on data
    // nobody will use (PR #25 review: on a 2M-variable orthant problem this
    // up-front work alone measured multi-second wall time before this check
    // existed).
    if opts.stop_requested() {
        return invalid_result(problem, SolveStatus::Timeout);
    }
    // Input domain bounds (`var_bounds`) are a row-equivalent representation;
    // the tier-1 IPM consumes rows only, so materialize them into orthant
    // rows once up front — every stage below (presolve, equilibration, the
    // re-verification gate) then runs on that equivalent explicit form.
    // Empty `var_bounds` — every current in-repo caller — is identity.
    let materialized = if problem.var_bounds.is_empty() {
        None
    } else {
        Some(presolve::materialize_bounds(problem).0)
    };
    let problem = materialized.as_ref().unwrap_or(problem);
    let gate = (opts.tol * presolve::VERIFY_KKT_SLACK).max(opts.tol);
    match presolve::ConicReduction::reduce(problem) {
        Some(red) if red.conflict().is_some() => {
            if let Some((y, z)) = red.conflict_certificate(problem) {
                // Bound absorption proved `lb > ub`: emit the exact Farkas
                // pair, verified against original data (mirrors the
                // honesty-first certificate policy of `unscale_result`).
                if crate::conic::equil::verify_infeas_cert(problem, &y, &z, gate) {
                    let mut res = invalid_result(problem, SolveStatus::Timeout);
                    if opts.stop_requested() {
                        return res;
                    }
                    res.status = SolveStatus::Infeasible;
                    res.objective = f64::INFINITY;
                    res.y = y.clone();
                    res.z = z.clone();
                    res.infeas_cert = Some((y, z));
                    return res;
                }
            }
        }
        Some(red) if red.reduced().is_some() => {
            let reduced = red.reduced().expect("checked by guard");
            let stats = red.stats();
            let opts_red = ConicOptions {
                tol: presolve::tightened_tol(opts.tol, stats.sigma),
                ..opts.clone()
            };
            if opts_red.stop_requested() {
                return invalid_result(problem, SolveStatus::Timeout);
            }
            let eq = equil::Equilibrator::compute(reduced);
            let scaled = eq.scale_problem(reduced);
            let res = ipm::solve(&scaled, &opts_red, true);
            #[cfg(test)]
            test_maybe_cancel_after_ipm_solve(opts);
            let mut res = eq.unscale_result(reduced, opts_red.tol, res);
            if matches!(res.status, SolveStatus::Infeasible | SolveStatus::Unbounded)
                && opts.stop_requested()
            {
                res.status = SolveStatus::Timeout;
                res.primal_ray = None;
                res.infeas_cert = None;
            }
            // A tightened-tolerance `MaxIterations` must not regress the
            // status the loose tolerance would have reached: re-solve
            // unpresolvecised rather than report a spurious budget failure.
            if res.status != SolveStatus::MaxIterations {
                if let Some(mut mapped) = red.postsolve(problem, reduced, &res, gate) {
                    mapped.objective = match mapped.status {
                        SolveStatus::Infeasible => f64::INFINITY,
                        SolveStatus::Unbounded => f64::NEG_INFINITY,
                        _ => mapped.objective,
                    };
                    return mapped;
                }
            }
            // Fall through: original-space verification rejected the
            // postsolved map (or the tightened solve ran out of budget) —
            // re-solve the original problem below.
        }
        _ => {}
    }
    solve_socp_direct(problem, opts)
}

/// The presolve-free conic pipeline: equilibrate → `ipm::solve` → unscale +
/// original-space certificate re-verification. This is both the default path
/// (nothing reduced) and the fallback when a presolved result fails the
/// original-space re-verification gate.
fn solve_socp_direct(problem: &ConicProblem, opts: &ConicOptions) -> ConicResult {
    let eq = equil::Equilibrator::compute(problem);
    let scaled = eq.scale_problem(problem);
    let res = ipm::solve(&scaled, opts, true);
    #[cfg(test)]
    test_maybe_cancel_after_ipm_solve(opts);
    let mut res = eq.unscale_result(problem, opts.tol, res);
    // API 境界 backstop: equilibration/unscale/再検証
    // (`unscale_result`) は `ipm::solve` 内の受理直前チェックより後の、追加の
    // 実処理時間を持つ窓 -- ここで外部停止が観測されていれば、certificate が
    // original-space 再検証を通っていても Timeout を優先する (正直化: 検証済み
    // 証明を捨てる保守的選択)。Optimal 等それ以外の status はこの判定に含めない。
    if matches!(res.status, SolveStatus::Infeasible | SolveStatus::Unbounded)
        && opts.stop_requested()
    {
        res.status = SolveStatus::Timeout;
        res.primal_ray = None;
        res.infeas_cert = None;
    }
    // Canonicalize only statuses whose `x` is not a usable iterate
    // (PR #25 review 40): `Infeasible` -> `+inf`, `Unbounded` -> `-inf`.
    // Inconclusive statuses keep `dot(c, x)` of the real iterate in `res.x`;
    // `NotSupported` keeps its `NaN`.
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
