//! Routes continuous QCQP (`QpProblem` with non-empty `quadratic_constraints`)
//! to the conic solver stack.
//!
//! Convexity is screened by the conic bridge itself rather than reimplemented
//! here: [`conic::solve_qp_problem_as_qcqp`] rejects a problem with
//! `NotSupported` when it detects nonconvexity (indefinite `P0`/`Pi` via
//! Cholesky, or a quadratic `>=`/`=` constraint). This screen is **not a
//! convexity proof**: the bridge's Cholesky clamps pivots inside a small
//! jitter band (meant to absorb QPS 6-digit rounding), so a slightly
//! indefinite matrix can pass as "convex" and then fail numerically in the
//! SOCP solve. The route therefore accepts the convex-bridge result only when
//! it is a clean outcome ([`is_clean_convex_outcome`]); any other failure
//! falls back to the spatial (McCormick) branch-and-bound global solver,
//! which is sound for convex problems too. `Timeout` never falls back —
//! retrying after the deadline would only double the time spent.
//!
//! `QpProblem` carries no integrality information (see
//! [`crate::mip::MiqpProblem`], which layers `integer_vars` on top of a plain
//! `QpProblem`/`LpProblem` instead of adding a field to them) — so this module
//! only ever sees continuous problems.

use std::time::Instant;

use crate::conic::{
    self, qcqp_matrix_to_csc, ConicOptions, GQuadConstraint, GlobalOptions, GlobalResult,
    NonconvexQcqp, QcqpResult,
};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveRoute, SolveStatus, SolverResult};

use super::QpProblem;

pub(crate) fn solve_qcqp_via_conic(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let deadline = resolve_deadline(options);
    let c_opts = conic_options(options, deadline);

    let convex = conic::solve_qp_problem_as_qcqp(problem, &c_opts);
    if is_clean_convex_outcome(&convex) {
        return qcqp_result_to_solver_result(convex);
    }

    let nc_qp = match nonconvex_from_qp_problem(problem) {
        Ok(qp) => qp,
        Err(e) => {
            // The global fallback needs a finite box. Without one, report the
            // structural rejection for a bridge-rejected problem; for a bridge
            // numerical failure, report NumericalError rather than fabricating
            // a result from the failed convexified solve.
            return if matches!(convex.status, SolveStatus::NotSupported(_)) {
                SolverResult::not_supported(e)
            } else {
                SolverResult::numerical_error()
            };
        }
    };
    let g_opts = global_options(options);
    let global = conic::solve_global_qcqp(&nc_qp, &c_opts, &g_opts);
    global_result_to_solver_result(global)
}

/// Whether the convex-bridge result can be trusted as-is.
///
/// The bridge's Cholesky convexity screen has a jitter band, so a slightly
/// indefinite problem can be "convexified" and then fail numerically
/// (`MaxIterations` with non-finite values, `NumericalError`, …). Only a
/// clean termination is accepted; everything else (including the bridge's
/// own `NotSupported`) goes to the global solver. `Timeout` counts as clean:
/// the deadline is spent, so falling back cannot help.
fn is_clean_convex_outcome(res: &QcqpResult) -> bool {
    match res.status {
        SolveStatus::Optimal => res.objective.is_finite() && res.x.iter().all(|v| v.is_finite()),
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::Timeout => true,
        _ => false,
    }
}

/// Resolve `options.timeout_secs` / `options.deadline` into a single deadline,
/// matching the QP/LP dispatch convention (`lp_dispatch::solve_as_lp`).
fn resolve_deadline(options: &SolverOptions) -> Option<Instant> {
    if options.deadline.is_some() {
        return options.deadline;
    }
    options
        .timeout_secs
        .map(|secs| Instant::now() + std::time::Duration::from_secs_f64(secs))
}

fn conic_options(options: &SolverOptions, deadline: Option<Instant>) -> ConicOptions {
    let default = ConicOptions::default();
    // `ipm.max_iter` defaults to `usize::MAX` ("timeout is the primary guard",
    // see `IpmOptions::max_iter`); the conic IPM has no equivalent multi-attempt
    // budget, so an explicit user override is honored but the sentinel falls
    // back to the conic module's own iteration count.
    let max_iter = if options.ipm.max_iter == usize::MAX {
        default.max_iter
    } else {
        options.ipm.max_iter
    };
    ConicOptions {
        tol: options.ipm_eps(),
        max_iter,
        deadline,
        ..default
    }
}

fn global_options(options: &SolverOptions) -> GlobalOptions {
    let eps = options.ipm_eps();
    GlobalOptions {
        gap_tol: eps,
        feas_tol: eps,
        ..GlobalOptions::default()
    }
}

fn qcqp_result_to_solver_result(res: QcqpResult) -> SolverResult {
    match res.status {
        SolveStatus::Infeasible => return SolverResult::infeasible(),
        SolveStatus::Unbounded => return SolverResult::unbounded(),
        SolveStatus::NumericalError => return SolverResult::numerical_error(),
        SolveStatus::NotSupported(msg) => return SolverResult::not_supported(msg),
        _ => {}
    }
    let deadline_triggered = res.status == SolveStatus::Timeout;
    let mut result = SolverResult {
        status: res.status,
        objective: res.objective,
        solution: res.x,
        iterations: res.iterations,
        ..SolverResult::default()
    };
    result.stats.route = SolveRoute::ConicQcqpConvex;
    result.stats.deadline_triggered = deadline_triggered;
    result
}

fn global_result_to_solver_result(res: GlobalResult) -> SolverResult {
    if res.status == SolveStatus::Infeasible {
        return SolverResult::infeasible();
    }
    let deadline_triggered = res.status == SolveStatus::Timeout;
    let mut result = SolverResult {
        status: res.status,
        objective: res.objective,
        solution: res.x,
        iterations: res.nodes,
        ..SolverResult::default()
    };
    result.stats.route = SolveRoute::ConicQcqpNonconvex;
    result.stats.deadline_triggered = deadline_triggered;
    result
}

/// Convert a continuous [`QpProblem`] QCQP into a [`NonconvexQcqp`] for the
/// spatial branch-and-bound global solver.
///
/// Unlike the convex conic bridge (which rejects quadratic `>=`/`=`
/// constraints as unable to certify convexity), this accepts any constraint
/// sense and any (possibly indefinite) quadratic matrix: `>=` rows are
/// sign-flipped and `=` rows become a pair of `<=` rows.
///
/// Requires finite bounds on every variable — the McCormick relaxation needs
/// a finite box to build valid envelopes and to terminate spatial branching.
fn nonconvex_from_qp_problem(src: &QpProblem) -> Result<NonconvexQcqp, String> {
    let n = src.num_vars;
    let mut lb = Vec::with_capacity(n);
    let mut ub = Vec::with_capacity(n);
    for (j, &(l, u)) in src.bounds.iter().enumerate() {
        if !l.is_finite() || !u.is_finite() {
            return Err(format!(
                "nonconvex QCQP requires finite bounds on every variable for \
                 McCormick spatial branch-and-bound; variable {j} has bound ({l}, {u})"
            ));
        }
        lb.push(l);
        ub.push(u);
    }

    let ad = src.a.to_dense_rows();
    let mut gl_rows: Vec<Vec<f64>> = Vec::new();
    let mut hl: Vec<f64> = Vec::new();
    let mut ae_rows: Vec<Vec<f64>> = Vec::new();
    let mut be: Vec<f64> = Vec::new();
    let mut quad: Vec<GQuadConstraint> = Vec::new();

    let has_qc = !src.quadratic_constraints.is_empty();
    for k in 0..src.num_constraints {
        let qmat = has_qc
            .then(|| &src.quadratic_constraints[k])
            .filter(|qc| qc.nnz() > 0);
        let row = ad[k].clone();
        match src.constraint_types[k] {
            ConstraintType::Le => match qmat {
                Some(qc) => quad.push(GQuadConstraint {
                    p: qcqp_matrix_to_csc(qc),
                    q: row,
                    r: -src.b[k],
                }),
                None => {
                    gl_rows.push(row);
                    hl.push(src.b[k]);
                }
            },
            ConstraintType::Ge => match qmat {
                Some(qc) => quad.push(GQuadConstraint {
                    p: qcqp_matrix_to_csc(qc).scale_values(-1.0),
                    q: row.iter().map(|v| -v).collect(),
                    r: src.b[k],
                }),
                None => {
                    gl_rows.push(row.iter().map(|v| -v).collect());
                    hl.push(-src.b[k]);
                }
            },
            ConstraintType::Eq => match qmat {
                Some(qc) => {
                    let p = qcqp_matrix_to_csc(qc);
                    quad.push(GQuadConstraint {
                        p: p.clone(),
                        q: row.clone(),
                        r: -src.b[k],
                    });
                    quad.push(GQuadConstraint {
                        p: p.scale_values(-1.0),
                        q: row.iter().map(|v| -v).collect(),
                        r: src.b[k],
                    });
                }
                None => {
                    ae_rows.push(row);
                    be.push(src.b[k]);
                }
            },
        }
    }

    let g_lin = conic::csc_from_rows(&gl_rows, n);
    let a_eq = conic::csc_from_rows(&ae_rows, n);
    let p0 = (!src.is_zero_q()).then(|| src.q.clone());
    Ok(NonconvexQcqp {
        n,
        p0,
        q0: src.c.clone(),
        quad,
        g_lin,
        h_lin: hl,
        a_eq,
        b_eq: be,
        lb,
        ub,
    })
}
