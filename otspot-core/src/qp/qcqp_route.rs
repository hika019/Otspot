//! Routes continuous QCQP (`QpProblem` with non-empty `quadratic_constraints`)
//! to the conic solver stack.
//!
//! Convexity is screened by the conic bridge itself, not reimplemented
//! here: [`conic::solve_qp_problem_as_qcqp`] rejects nonconvex problems
//! (`NotSupported`, via Cholesky on `P0`/`Pi` or a quadratic `>=`/`=`
//! constraint). This is **not a convexity proof**: the bridge's Cholesky
//! clamps pivots in a small jitter band, so a slightly indefinite matrix
//! can pass as "convex" (reported via `QcqpResult::convexity_unproven`).
//! The route accepts the convex-bridge result only when exact **and**
//! clean ([`is_clean_convex_outcome`]); everything else falls back to
//! the spatial (McCormick) branch-and-bound solver (sound for convex
//! too); a clamp-free `Timeout` never falls back (retrying past the
//! deadline doubles the time spent). `QpProblem` itself carries no
//! integrality info ([`crate::mip::MiqpProblem`] layers `integer_vars`
//! on top instead), so this module only ever sees continuous problems.

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

    // The convex-bridge conic IPM checks `options.cancel_flag` every
    // iteration via `ConicOptions::cancel_flag`/`stop_requested` (same
    // convention as the LP/QP routes' `external_stop_requested`), so a flag
    // already set or fired during that attempt surfaces as `Timeout` from
    // `convex` below without any extra handling here.
    let convex = conic::solve_qp_problem_as_qcqp(problem, &c_opts);
    if is_clean_convex_outcome(&convex) {
        return qcqp_result_to_solver_result(convex, problem.obj_offset);
    }

    // The McCormick spatial B&B (`nonconvex::global_core`) only checks
    // `ConicOptions::deadline`, not the cancel flag, so a flag fired here
    // (already set, or set during the convex-bridge attempt above) must stop
    // the route before launching that fallback rather than being silently
    // ignored by it.
    if options.external_stop_requested() {
        return qcqp_stop_result(SolveRoute::ConicQcqpNonconvex);
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
    global_result_to_solver_result(global, problem.obj_offset)
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
        // Same cancellation object as the LP/QP routes: the conic IPM loop
        // checks it alongside `deadline` (`ConicOptions::stop_requested`).
        cancel_flag: options.cancel_flag.clone(),
        ..default
    }
}

/// Timeout result for a QCQP solve stopped by an already-fired external stop
/// condition (deadline expired / cancel flag set), before or between the
/// convex-bridge and McCormick-fallback phases had a chance to run.
fn qcqp_stop_result(route: SolveRoute) -> SolverResult {
    let mut result = SolverResult::timeout();
    result.stats.route = route;
    result.stats.deadline_triggered = true;
    result
}

fn global_options(options: &SolverOptions) -> GlobalOptions {
    let eps = options.ipm_eps();
    let mut g = GlobalOptions {
        gap_tol: eps,
        feas_tol: eps,
        ..GlobalOptions::default()
    };
    // Honor a caller-supplied global-optimization budget (node limit / gap):
    // a smaller `max_nodes` or looser gap must bound the spatial B&B, matching
    // the QP global route (`qp::global`).
    if let Some(cfg) = options.global_optimization.as_ref() {
        g.max_nodes = cfg.max_nodes;
        // `cfg.gap_tol` is the user's *optimality* search budget (how much
        // provable sub-optimality the B&B may stop at) — a different axis
        // from `feas_tol` (how far the reported incumbent may violate the
        // original quadratic/linear constraints, `nonconvex::feasible`'s
        // accept-time check). Loosening the former must never loosen the
        // latter: `feas_tol` keeps tracking the solver's own accuracy
        // request (`eps`), with or without `cfg`, so an incumbent's reported
        // constraint violation stays bounded by the requested feasibility
        // tolerance instead of by whatever optimality gap the caller
        // accepted (PR #25 review: with `cfg` this used to alias
        // `feas_tol = cfg.gap_tol`, e.g. the default `1e-3`, letting a point
        // violating a quadratic constraint by ~1e-3 be reported `Optimal`).
        g.gap_tol = cfg.gap_tol;
    }
    g
}

fn qcqp_result_to_solver_result(res: QcqpResult, obj_offset: f64) -> SolverResult {
    match res.status {
        SolveStatus::Infeasible => return SolverResult::infeasible(),
        SolveStatus::Unbounded => return SolverResult::unbounded(),
        SolveStatus::NumericalError => return SolverResult::numerical_error(),
        SolveStatus::NotSupported(msg) => return SolverResult::not_supported(msg),
        _ => {}
    }
    // status エイリアスのまま残置: この変換層は SolverOptions を持たず実クロック
    // 判定ができない (conic 経路の Timeout は deadline ゲート mint のため実害なし。
    // options を通す場合は conic 側 API 変更が必要 — P3)。
    let deadline_triggered = res.status == SolveStatus::Timeout;
    let mut result = SolverResult {
        status: res.status,
        objective: res.objective + obj_offset,
        solution: res.x,
        iterations: res.iterations,
        ..SolverResult::default()
    };
    result.stats.route = SolveRoute::ConicQcqpConvex;
    result.stats.deadline_triggered = deadline_triggered;
    result
}

fn global_result_to_solver_result(res: GlobalResult, obj_offset: f64) -> SolverResult {
    if res.status == SolveStatus::Infeasible {
        return SolverResult::infeasible();
    }
    // The spatial B&B reports `NumericalError` only when no incumbent was
    // established (objective is a bare `+inf` sentinel, not a real `c^T x`),
    // so canonicalize it rather than adding the offset to infinity.
    if res.status == SolveStatus::NumericalError {
        return SolverResult::numerical_error();
    }
    let deadline_triggered = res.status == SolveStatus::Timeout;
    let mut result = SolverResult {
        status: res.status,
        objective: res.objective + obj_offset,
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
    src.validate().map_err(|e| e.to_string())?;
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
    use crate::options::{GlobalOptimizationConfig, Tolerance};
    use crate::qp::{solve_qp_with, QcqpMatrix};
    use crate::sparse::CscMatrix;

    /// PR #25 review ("Keep gap tolerance out of feasibility checks"): the
    /// optimality-gap search budget (`cfg.gap_tol`) and the incumbent
    /// feasibility resolution (`feas_tol`, used by `nonconvex::feasible` to
    /// accept a candidate against the *original* quadratic/linear
    /// constraints) are different axes and must not be aliased. `gap_tol`
    /// tracks `cfg.gap_tol` when a config is supplied; `feas_tol` always
    /// tracks the solver's own accuracy request (`eps`), with or without a
    /// config. Independent oracle: expected tolerances are the documented
    /// inputs (`ipm_eps()` / `cfg.gap_tol`), not recomputed through
    /// `global_options`.
    ///
    /// Replaces the prior `global_options_gap_and_feas_stay_consistent` test,
    /// which asserted the buggy aliasing (`feas_tol == cfg.gap_tol`) as
    /// intended behavior; see `global_options_feas_tol_does_not_relax_with_cfg`
    /// for the soundness demonstration (a genuinely infeasible point reported
    /// `Optimal` under the old aliasing).
    ///
    /// Sentinel: reintroducing `g.feas_tol = cfg.gap_tol` in the `cfg` branch
    /// makes the `with_cfg` assertions FAIL (feas_tol would equal
    /// `cfg.gap_tol` instead of staying at `eps`).
    #[test]
    fn global_options_feas_tol_always_tracks_eps() {
        // No cfg: both = eps (here the Fast tolerance, distinct from the
        // GlobalOptimizationConfig default gap so the two sources can't alias).
        let base = SolverOptions {
            tolerance: Some(Tolerance::Fast),
            ..SolverOptions::default()
        };
        let eps = base.ipm_eps();
        let g = global_options(&base);
        assert_eq!(g.gap_tol, eps, "no cfg: gap_tol tracks eps");
        assert_eq!(g.feas_tol, eps, "no cfg: feas_tol tracks eps");

        // With cfg (default gap_tol, deliberately != eps): gap_tol follows
        // cfg, feas_tol must NOT follow it — it stays at eps regardless.
        let cfg = GlobalOptimizationConfig::default();
        assert_ne!(
            cfg.gap_tol, eps,
            "test premise: cfg default gap must differ from eps"
        );
        let with_cfg = SolverOptions {
            tolerance: Some(Tolerance::Fast),
            global_optimization: Some(cfg.clone()),
            ..SolverOptions::default()
        };
        let gc = global_options(&with_cfg);
        assert_eq!(gc.gap_tol, cfg.gap_tol, "cfg: gap_tol tracks cfg.gap_tol");
        assert_eq!(
            gc.feas_tol, eps,
            "cfg: feas_tol must stay at eps, not cfg.gap_tol"
        );
        assert_ne!(
            gc.feas_tol, cfg.gap_tol,
            "cfg: feas_tol must not alias the optimality-gap budget"
        );
        assert_eq!(gc.max_nodes, cfg.max_nodes, "cfg: node budget honored");
    }

    /// Soundness demonstration for the same review finding, end to end
    /// through `solve_qp_with` (not just the `global_options` mapping):
    /// `min x0+x1  s.t.  x0*x1 >= 1,  x in [0.1,5]^2` has true optimum 2.0 at
    /// (1,1). With the default `global_optimization` config
    /// (`gap_tol = 1e-3`), the McCormick spatial B&B accepts the first
    /// incumbent whose relaxation vertex clears `feasible(x, feas_tol)`; if
    /// `feas_tol` aliases `cfg.gap_tol` (the pre-fix bug), that incumbent's
    /// *true* constraint residual can sit as loose as `~gap_tol` while still
    /// being reported `Optimal`.
    ///
    /// Independent oracle: the true violation is recomputed directly from
    /// the reported `x` against the original constraint (`1 - x0*x1`), not
    /// through any tolerance the route itself used.
    ///
    /// Confirmed repro before the fix: `status=Optimal`,
    /// `x=[0.99957, 0.99954]`, true violation `≈ 8.9e-4` (i.e. `x0*x1 ≈
    /// 0.9991 < 1`, violating the constraint by nearly the whole `1e-3` gap
    /// budget) with `ipm_eps()` several orders tighter.
    ///
    /// Sentinel: reintroducing `g.feas_tol = cfg.gap_tol` makes this FAIL —
    /// the true violation reverts to `~1e-3`-scale, far above the
    /// `1e-4` gate below.
    #[test]
    fn global_options_feas_tol_does_not_relax_with_cfg() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.1, 5.0), (0.1, 5.0)];
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((0, 1, 1.0));
        qc.triplets.push((1, 0, 1.0));
        let mut problem = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();
        problem.set_quadratic_constraints(vec![qc]).unwrap();

        let opts = SolverOptions {
            global_optimization: Some(GlobalOptimizationConfig::default()),
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
        assert_eq!(result.solution.len(), 2);
        let true_violation = 1.0 - result.solution[0] * result.solution[1];
        assert!(
            true_violation < 1e-4,
            "reported Optimal point must respect the requested feasibility \
             tolerance, not the (1e-3) optimality-gap budget: x={:?}, \
             true_violation={true_violation}",
            result.solution
        );
        assert!(
            (result.objective - 2.0).abs() < 1e-3,
            "objective must still be near the true optimum 2.0: {}",
            result.objective
        );
    }

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

    /// PR #25 review horizontal spread ("Validate QCQP vector length before
    /// indexing", raised on `conic::qcqp` and applied there): the same
    /// public-field hazard exists here. `QpProblem::quadratic_constraints` is
    /// public, so a caller can assign a non-empty vector shorter than
    /// `num_constraints`, bypassing `set_quadratic_constraints`. Before this
    /// guard `nonconvex_from_qp_problem` indexed `quadratic_constraints[k]`
    /// for every `k < num_constraints` and panicked with "index out of bounds"
    /// (confirmed repro: len 1, index 1 at qcqp_route.rs:256).
    ///
    /// Independent oracle: the setter's own documented invariant ("length
    /// must be 0 or num_constraints") — the guard returns `Err`, never panics.
    ///
    /// Sentinel: removing the length check makes this panic (index out of
    /// bounds) instead of returning `Err`, so the `is_err()` assertion is
    /// unreachable and the test aborts.
    #[test]
    fn nonconvex_from_qp_problem_rejects_short_quadratic_constraints() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        // 2 linear constraints.
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, n).unwrap();
        let b = vec![5.0, 5.0];
        let bounds = vec![(0.0, 3.0), (0.0, 3.0)];
        let ctypes = vec![ConstraintType::Le, ConstraintType::Le];
        let mut problem = QpProblem::new(q_obj, c, a, b, bounds, ctypes).unwrap();
        // Assign a SHORT vec (len 1 < num_constraints 2) directly to the
        // public field, bypassing set_quadratic_constraints' length check.
        problem.quadratic_constraints = vec![QcqpMatrix::new(n)];

        let r = nonconvex_from_qp_problem(&problem);
        assert!(
            r.is_err(),
            "short quadratic_constraints must return Err, not panic: {r:?}"
        );
        let msg = r.unwrap_err();
        assert!(
            msg.contains("quadratic_constraints length must be 0 or 2, got 1"),
            "unexpected error message: {msg}"
        );
    }

    /// Companion: a LONGER-than-num_constraints vector must also be rejected
    /// (the setter invariant is exact equality, not just a lower bound).
    #[test]
    fn nonconvex_from_qp_problem_rejects_long_quadratic_constraints() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        // 1 linear constraint.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
        let b = vec![5.0];
        let bounds = vec![(0.0, 3.0), (0.0, 3.0)];
        let ctypes = vec![ConstraintType::Le];
        let mut problem = QpProblem::new(q_obj, c, a, b, bounds, ctypes).unwrap();
        // len 2 > num_constraints 1.
        problem.quadratic_constraints = vec![QcqpMatrix::new(n), QcqpMatrix::new(n)];

        let r = nonconvex_from_qp_problem(&problem);
        assert!(
            r.is_err(),
            "long quadratic_constraints must return Err: {r:?}"
        );
        assert!(r.unwrap_err().contains("must be 0 or 1, got 2"));
    }
}
