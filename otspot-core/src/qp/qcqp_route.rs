//! Routes continuous QCQP (`QpProblem` with non-empty `quadratic_constraints`)
//! to the conic solver stack.
//!
//! Convexity is screened by the conic bridge itself rather than reimplemented
//! here: [`conic::solve_qp_problem_as_qcqp`] rejects a problem with
//! `NotSupported` when it detects nonconvexity (indefinite `P0`/`Pi` via
//! Cholesky, or a quadratic `>=`/`=` constraint). This screen is **not a
//! convexity proof**: the bridge's Cholesky clamps pivots inside a small
//! jitter band (meant to absorb QPS 6-digit rounding), so a slightly
//! indefinite matrix can pass as "convex" — the bridge reports this via
//! `QcqpResult::convexity_unproven`. The route accepts the convex-bridge
//! result only when the reformulation was exact **and** the outcome is clean
//! ([`is_clean_convex_outcome`]); everything else falls back to the spatial
//! (McCormick) branch-and-bound global solver, which is sound for convex
//! problems too. A clamp-free `Timeout` never falls back — retrying after
//! the deadline would only double the time spent.
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
use crate::sparse::CscMatrix;

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
/// Two conditions. First, the SOCP reformulation must be exact: when the
/// bridge's Cholesky clamped a negative jitter-band pivot
/// (`convexity_unproven`), the solved SOCP only approximates the QCQP, so
/// **no** status — not even Infeasible/Unbounded/Optimal — carries over to
/// the original problem; the global solver must decide. Second, the
/// termination must be clean: numerical failures (`MaxIterations` with
/// non-finite values, `NumericalError`, …) and the bridge's own
/// `NotSupported` go to the global solver as well. A clamp-free `Timeout`
/// counts as clean — the deadline is spent, so falling back cannot help.
fn is_clean_convex_outcome(res: &QcqpResult) -> bool {
    if res.convexity_unproven {
        return false;
    }
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

    // Row-major view of A (O(nnz)): column k of `a_rows` is row k of `src.a`,
    // giving sparse per-constraint access without densifying A.
    let a_rows = src.a.transpose();
    let mut gl_r: Vec<usize> = Vec::new();
    let mut gl_c: Vec<usize> = Vec::new();
    let mut gl_v: Vec<f64> = Vec::new();
    let mut hl: Vec<f64> = Vec::new();
    let mut ae_r: Vec<usize> = Vec::new();
    let mut ae_c: Vec<usize> = Vec::new();
    let mut ae_v: Vec<f64> = Vec::new();
    let mut be: Vec<f64> = Vec::new();
    let mut quad: Vec<GQuadConstraint> = Vec::new();
    let mut gl_count = 0usize;
    let mut ae_count = 0usize;

    let has_qc = !src.quadratic_constraints.is_empty();
    for k in 0..src.num_constraints {
        let qmat = has_qc
            .then(|| &src.quadratic_constraints[k])
            .filter(|qc| qc.nnz() > 0);
        let (row_idx, row_val) = a_rows
            .get_column(k)
            .map_err(|e| format!("A row {k}: {e:?}"))?;
        match src.constraint_types[k] {
            ConstraintType::Le => match qmat {
                Some(qc) => {
                    let mut row = vec![0.0; n];
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        row[j] = v;
                    }
                    quad.push(GQuadConstraint {
                        p: qcqp_matrix_to_csc(qc),
                        q: row,
                        r: -src.b[k],
                    });
                }
                None => {
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        gl_r.push(gl_count);
                        gl_c.push(j);
                        gl_v.push(v);
                    }
                    hl.push(src.b[k]);
                    gl_count += 1;
                }
            },
            ConstraintType::Ge => match qmat {
                Some(qc) => {
                    let mut row = vec![0.0; n];
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        row[j] = -v;
                    }
                    quad.push(GQuadConstraint {
                        p: qcqp_matrix_to_csc(qc).scale_values(-1.0),
                        q: row,
                        r: src.b[k],
                    });
                }
                None => {
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        gl_r.push(gl_count);
                        gl_c.push(j);
                        gl_v.push(-v);
                    }
                    hl.push(-src.b[k]);
                    gl_count += 1;
                }
            },
            ConstraintType::Eq => match qmat {
                Some(qc) => {
                    let mut row = vec![0.0; n];
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        row[j] = v;
                    }
                    let row_neg: Vec<f64> = row.iter().map(|v| -v).collect();
                    let p = qcqp_matrix_to_csc(qc);
                    quad.push(GQuadConstraint {
                        p: p.clone(),
                        q: row,
                        r: -src.b[k],
                    });
                    quad.push(GQuadConstraint {
                        p: p.scale_values(-1.0),
                        q: row_neg,
                        r: src.b[k],
                    });
                }
                None => {
                    for (&j, &v) in row_idx.iter().zip(row_val) {
                        ae_r.push(ae_count);
                        ae_c.push(j);
                        ae_v.push(v);
                    }
                    be.push(src.b[k]);
                    ae_count += 1;
                }
            },
        }
    }

    let g_lin = CscMatrix::from_triplets(&gl_r, &gl_c, &gl_v, gl_count, n)
        .map_err(|e| format!("G_lin build: {e:?}"))?;
    let a_eq = CscMatrix::from_triplets(&ae_r, &ae_c, &ae_v, ae_count, n)
        .map_err(|e| format!("A_eq build: {e:?}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::QcqpMatrix;
    use crate::sparse::CscMatrix;

    /// `nonconvex_from_qp_problem`'s constraint transcription must match a
    /// hand-built expectation, element-wise (independent oracle: every
    /// expected matrix/vector below is written out literally from the
    /// documented rules, not recomputed through any bridge code):
    /// - linear Le rows copied as-is into `g_lin`;
    /// - linear Ge rows sign-flipped (`-row <= -b`);
    /// - linear Eq rows into `a_eq`/`b_eq`;
    /// - quadratic Le kept as `(P, q, -b)`;
    /// - quadratic Ge sign-flipped to `(-P, -q, b)`;
    /// - quadratic Eq split into the two-sided pair `(P, q, -b)` then
    ///   `(-P, -q, b)`;
    /// - finite variable bounds copied verbatim into `lb`/`ub` (not turned
    ///   into `g_lin` rows — the McCormick box carries them).
    #[test]
    fn nonconvex_from_qp_problem_matches_hand_built_split() {
        let n = 3usize;
        let q_obj =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 4.0, 6.0], n, n).unwrap();
        let c = vec![1.0, -1.0, 0.5];
        // Row 0 (Le, linear):  x0 + x1 <= 5
        // Row 1 (Ge, linear):  x1 - x2 >= -2   => flipped: -x1 + x2 <= 2
        // Row 2 (Eq, linear):  x0 + x2 = 1
        // Row 3 (Le, quad):    (1/2)*2*x0^2 + x0 <= 3
        // Row 4 (Ge, quad):    (1/2)*2*x1^2 + x1 + x2 >= 4
        // Row 5 (Eq, quad):    x0*x1 + x0 - x2 = 0.5
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5],
            &[0, 1, 1, 2, 0, 2, 0, 1, 2, 0, 2],
            &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            6,
            n,
        )
        .unwrap();
        let b = vec![5.0, -2.0, 1.0, 3.0, 4.0, 0.5];
        let bounds = vec![(-1.0, 2.0), (0.0, 3.0), (-2.0, 5.0)];
        let ctypes = vec![
            ConstraintType::Le,
            ConstraintType::Ge,
            ConstraintType::Eq,
            ConstraintType::Le,
            ConstraintType::Ge,
            ConstraintType::Eq,
        ];
        let mut problem = QpProblem::new(q_obj, c.clone(), a, b, bounds, ctypes).unwrap();
        let mut qc3 = QcqpMatrix::new(n);
        qc3.triplets.push((0, 0, 2.0));
        let mut qc4 = QcqpMatrix::new(n);
        qc4.triplets.push((1, 1, 2.0));
        let mut qc5 = QcqpMatrix::new(n);
        qc5.triplets.push((0, 1, 1.0));
        qc5.triplets.push((1, 0, 1.0));
        problem
            .set_quadratic_constraints(vec![
                QcqpMatrix::new(n),
                QcqpMatrix::new(n),
                QcqpMatrix::new(n),
                qc3,
                qc4,
                qc5,
            ])
            .unwrap();

        let nc = nonconvex_from_qp_problem(&problem).unwrap();

        assert_eq!(nc.n, n);
        assert_eq!(nc.lb, vec![-1.0, 0.0, -2.0]);
        assert_eq!(nc.ub, vec![2.0, 3.0, 5.0]);

        // Linear split: Le row as-is, Ge row sign-flipped; bounds do NOT
        // appear as g_lin rows.
        assert_eq!(
            nc.g_lin.to_dense_rows(),
            vec![vec![1.0, 1.0, 0.0], vec![0.0, -1.0, 1.0]]
        );
        assert_eq!(nc.h_lin, vec![5.0, 2.0]);
        assert_eq!(nc.a_eq.to_dense_rows(), vec![vec![1.0, 0.0, 1.0]]);
        assert_eq!(nc.b_eq, vec![1.0]);

        // Quadratic split: Le (row 3), Ge sign-flipped (row 4), then the Eq
        // (row 5) two-sided pair in (P, q, -b), (-P, -q, b) order.
        assert_eq!(nc.quad.len(), 4);

        assert_eq!(
            nc.quad[0].p.to_dense_rows(),
            vec![
                vec![2.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(nc.quad[0].q, vec![1.0, 0.0, 0.0]);
        assert_eq!(nc.quad[0].r, -3.0);

        assert_eq!(
            nc.quad[1].p.to_dense_rows(),
            vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, -2.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(nc.quad[1].q, vec![0.0, -1.0, -1.0]);
        assert_eq!(nc.quad[1].r, 4.0);

        assert_eq!(
            nc.quad[2].p.to_dense_rows(),
            vec![
                vec![0.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(nc.quad[2].q, vec![1.0, 0.0, -1.0]);
        assert_eq!(nc.quad[2].r, -0.5);

        assert_eq!(
            nc.quad[3].p.to_dense_rows(),
            vec![
                vec![0.0, -1.0, 0.0],
                vec![-1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(nc.quad[3].q, vec![-1.0, 0.0, 1.0]);
        assert_eq!(nc.quad[3].r, 0.5);

        // Objective carries over: non-zero Q -> Some(P0), q0 = c.
        assert_eq!(
            nc.p0.unwrap().to_dense_rows(),
            vec![
                vec![2.0, 0.0, 0.0],
                vec![0.0, 4.0, 0.0],
                vec![0.0, 0.0, 6.0],
            ]
        );
        assert_eq!(nc.q0, c);
    }
}
