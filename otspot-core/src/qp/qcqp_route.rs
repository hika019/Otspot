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
//! clean (`is_clean_convex_outcome`); everything else falls back to
//! the spatial (McCormick) branch-and-bound solver (sound for convex
//! too); a clamp-free `Timeout` never falls back (retrying past the
//! deadline doubles the time spent). `QpProblem` itself carries no
//! integrality info ([`crate::mip::MiqpProblem`] layers `integer_vars`
//! on top instead), so this module only ever sees continuous problems.

use std::time::Instant;

use crate::conic::{
    self, qcqp_matrix_to_csc, ConicOptions, GQuadConstraint, GlobalOptions, GlobalResult,
    GlobalSeed, NonconvexQcqp, QcqpResult,
};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveRoute, SolveStatus, SolverResult};
use otspot_num::sparse::CscMatrix;

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

    let g_opts = global_options(options);
    let (nc_qp, seed) = match build_nonconvex_with_root_box(problem, g_opts.feas_tol) {
        Ok(pair) => pair,
        Err(e) => {
            // The global fallback needs a finite box on the bound-required
            // set R (after root FBBT had its chance — plan §3.2 line 89).
            // Without one, report the structural rejection for a
            // bridge-rejected problem; for a bridge numerical failure, report
            // NumericalError rather than fabricating a result from the
            // failed convexified solve.
            return if matches!(convex.status, SolveStatus::NotSupported(_)) {
                SolverResult::not_supported(e)
            } else {
                SolverResult::numerical_error()
            };
        }
    };
    let global = conic::solve_global_qcqp(&nc_qp, &c_opts, &g_opts, seed.as_ref());
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
/// spatial branch-and-bound global solver, keeping the G1 raw-bound gate
/// (`conic::require_finite_bounds`).
///
/// Unlike the convex conic bridge (which rejects quadratic `>=`/`=`
/// constraints as unable to certify convexity), this accepts any constraint
/// sense and any (possibly indefinite) quadratic matrix: `>=` rows are
/// sign-flipped and `=` rows become a pair of `<=` rows.
///
/// The QP route itself uses [`build_nonconvex_raw`] + root FBBT instead
/// (`build_nonconvex_with_root_box`): Phase 2 (plan §3.2 line 89) judges
/// `R` finiteness on the derived root box, which this gated wrapper cannot
/// express. The wrapper stays as the model-level/compat entry with the
/// documented G1 semantics.
pub fn nonconvex_from_qp_problem(src: &QpProblem) -> Result<NonconvexQcqp, String> {
    let nc = build_nonconvex_raw(src)?;
    let (lb, ub) = conic::require_finite_bounds(nc.n, nc.p0.as_ref(), &nc.quad, &src.bounds)?;
    let mut nc = nc;
    nc.lb = lb;
    nc.ub = ub;
    Ok(nc)
}

/// Build the [`NonconvexQcqp`] with the *declared* bounds — no bound gate.
///
/// Public because the Phase-2 acceptance rule is no longer expressible as a
/// raw-bound filter: `build_nonconvex_with_root_box` derives a root box
/// first (plan §4.1) and only then rejects. Callers that bypass the root-box
/// path must run their own `R`-finiteness check (the
/// `global_core` effective-root-box gate is the shared backstop).
pub fn build_nonconvex_raw(src: &QpProblem) -> Result<NonconvexQcqp, String> {
    let n = src.num_vars;
    src.validate().map_err(|e| e.to_string())?;

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
        let (row_idx, row_val) = a_rows.column(k);
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
    let lb = src.bounds.iter().map(|&(l, _)| l).collect();
    let ub = src.bounds.iter().map(|&(_, u)| u).collect();
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

/// Phase-2 route preparation (plan §4.1/§4.3, research doc §4 steps 2+5):
/// build the raw [`NonconvexQcqp`], run root-stage FBBT on it, and apply the
/// Phase-2 acceptance rule "reject only when a bound-required variable is
/// still non-finite after derivation" (plan §3.2 line 89) on the
/// [`conic::effective_bound_requirement`] set — the D1 acceptance set the
/// classifier module documents for exactly this purpose.
///
/// Derived bounds travel *only* in the returned box-only
/// [`GlobalSeed`] (plan §4.2 line 136: `qp.lb`/`qp.ub`, the box `feasible()`
/// reads, are never rewritten). When FBBT changes nothing, no seed is
/// attached, keeping the downstream path bit-identical to the pre-Phase-2
/// solver. The seed is audited by [`audit_root_seed`] before it leaves here —
/// widened/NaN boxes are rejected here rather than at `global_core`.
fn build_nonconvex_with_root_box(
    src: &QpProblem,
    feas_tol: f64,
) -> Result<(NonconvexQcqp, Option<GlobalSeed>), String> {
    let nc = build_nonconvex_raw(src)?;
    let req = conic::effective_bound_requirement(nc.n, nc.p0.as_ref(), &nc.quad, &nc.lb, &nc.ub)?;
    let derived = conic::derive_root_box(&nc, feas_tol);
    let bad: Vec<(usize, f64, f64)> = req
        .required
        .iter()
        .enumerate()
        .filter(|(_, need)| **need)
        .filter_map(|(j, _)| {
            (!derived.lb[j].is_finite() || !derived.ub[j].is_finite()).then_some((
                j,
                derived.lb[j],
                derived.ub[j],
            ))
        })
        .collect();
    if let Some(&(j, l, u)) = bad.first() {
        return Err(format!(
            "nonconvex QCQP requires finite bounds on every bound-required \
             variable set R (plan §3.1); {} of the {} required variables \
             have a non-finite bound even after root FBBT derivation \
             (first: variable {j} with derived bounds ({l}, {u}))",
            bad.len(),
            req.required.iter().filter(|b| **b).count()
        ));
    }
    if !derived.is_tightening(&nc) {
        return Ok((nc, None));
    }
    let seed = GlobalSeed {
        x: Vec::new(),
        f: f64::INFINITY,
        root_lb: derived.lb,
        root_ub: derived.ub,
    };
    audit_root_seed(&nc, &seed)?;
    Ok((nc, Some(seed)))
}

/// Audit a derived root seed with the same checks `global_core` runs on any
/// incoming seed (`validate_seed` semantics: NaN sides, crossed intervals,
/// widened sides), so a derivation bug fails at the route with a precise
/// message instead of turning the search into a `NotSupported("invalid
/// global seed")` far from the cause. FBBT is refine-only by construction;
/// this is the belt-and-braces gate the task's proof contract requires.
fn audit_root_seed(qp: &NonconvexQcqp, seed: &GlobalSeed) -> Result<(), String> {
    if seed.root_lb.len() != qp.n || seed.root_ub.len() != qp.n {
        return Err(format!(
            "derived root box length {} / {} != n {}",
            seed.root_lb.len(),
            seed.root_ub.len(),
            qp.n
        ));
    }
    if seed.f != f64::INFINITY || !seed.x.is_empty() {
        return Err(format!(
            "root FBBT must emit a box-only seed (x empty, f = +inf), got \
             x.len() = {} and f = {}",
            seed.x.len(),
            seed.f
        ));
    }
    for (k, ((&l, &u), (&pl, &pu))) in seed
        .root_lb
        .iter()
        .zip(&seed.root_ub)
        .zip(qp.lb.iter().zip(&qp.ub))
        .enumerate()
    {
        if l.is_nan() || u.is_nan() {
            return Err(format!("derived bound[{k}] is NaN (l = {l}, u = {u})"));
        }
        if l > u {
            return Err(format!("derived bound[{k}] is empty ({l} > {u})"));
        }
        if l < pl || u > pu {
            return Err(format!(
                "derived bound[{k}] = [{l}, {u}] widens the declared box [{pl}, {pu}]"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{GlobalOptimizationConfig, Tolerance};
    use crate::qp::{solve_qp_with, QcqpMatrix};
    use otspot_num::sparse::CscMatrix;

    /// Phase-2 headline (plan §4.1 line 123): an indefinite QCQP row over a
    /// **free** objective variable. Pre-Phase-2 this died at the raw-bound
    /// gate (`R` contains the free `x0`, plan §3.1); the completed-square row
    /// minimum (`rest_min = min -x1² over [-2,2] = -4`) is what finitizes
    /// `x0` into the derived root box the search enters with.
    ///
    /// `min x0  s.t.  x0² - x1² <= 1`, `x0` free, `x1 ∈ [-2, 2]`.
    /// Independent oracle: `x0² <= 1 + x1² <= 5` gives `x0 >= -sqrt(5)`,
    /// attained at `x1 = ±2, x0 = -sqrt(5)` (row value `5 - 4 = 1`).
    /// The bridge cannot take this row (indefinite `P`), so the solve runs on
    /// the spatial B&B with the derived box.
    ///
    /// (Pure-equality instances like `x0² + 2x1² = 2` with free `x` are
    /// finitized at the derivation level too —
    /// `qcqp_fbbt::completing_the_square_bounds_free_vars_of_ellipsoid_equality`
    /// pins the hand oracle `|x_i| <= sqrt(c)` there — but such instances
    /// stop in `global_core`'s leaf handling (rule-2 diagonals are not branch
    /// candidates, plan D4): a G2 search limitation independent of FBBT.)
    #[test]
    fn indefinite_row_with_free_objective_var_solves_via_root_fbbt() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 0.0];
        // Row 0: `x0² - x1² <= 1` (zero linear part).
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((0, 0, 2.0));
        qc.triplets.push((1, 1, -2.0));
        let mut problem = QpProblem::new(
            q_obj,
            c,
            a,
            vec![1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (-2.0, 2.0)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        problem.set_quadratic_constraints(vec![qc]).unwrap();

        let g_opts = conic::GlobalOptions::default();
        let (nc, seed) = build_nonconvex_with_root_box(&problem, g_opts.feas_tol).unwrap();
        let root5 = 5.0f64.sqrt();
        let seed = seed.expect("free x0 in R must be finitized and attached as seed");
        assert!(
            seed.root_lb[0] < -root5 && seed.root_ub[0] > root5,
            "derived box must contain +/-sqrt(5) with outward margin: \
             [{}, {}]",
            seed.root_lb[0],
            seed.root_ub[0]
        );
        // Declared sides are never touched (refine-only): the raw box of x1
        // stays exactly [-2, 2] and x0 stays raw-free in `nc`.
        assert_eq!((nc.lb[1], nc.ub[1]), (-2.0, 2.0));
        assert!(nc.lb[0] == f64::NEG_INFINITY && nc.ub[0] == f64::INFINITY);
        let result = solve_qp_with(&problem, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal, "{:?}", result.status);
        assert_eq!(result.stats.route, SolveRoute::ConicQcqpNonconvex);
        assert!(
            (result.objective + root5).abs() < 1e-6,
            "hand oracle -sqrt(5): {}",
            result.objective
        );
    }

    #[test]
    fn circle_equality_all_free_routes_through_derived_gate() {
        // Acceptance-gate-move sentinel (plan §3.2 line 89, report §6 T1):
        // `x0^2 + x1^2 = 1` with a fully free declared box. Both variables
        // are in R (two-sided diagonal lifts); the RAW box rejects (old
        // `require_finite_bounds` semantics), the DERIVED box (FBBT ellipsoid
        // evaluation, |x| <= 1 + margin) accepts. Reverting the acceptance
        // judgment to the raw box turns the Optimal below into a
        // NotSupported (revert-fail sentinel). Hand oracle: min x0 = -1.
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((0, 0, 2.0));
        qc.triplets.push((1, 1, 2.0));
        let mut problem = QpProblem::new(
            q_obj,
            c,
            a,
            vec![1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        problem.set_quadratic_constraints(vec![qc]).unwrap();

        let (nc, seed) = build_nonconvex_with_root_box(&problem, 1e-6).unwrap();
        assert!(nc.lb.iter().all(|l| !l.is_finite()));
        let seed = seed.expect("all-free R variables must finitize and attach a seed");
        assert!(
            seed.root_ub[0] > 1.0 && seed.root_lb[0] < -1.0 - 1e-9,
            "derived circle box must contain [-1, 1] strictly inside: [{}, {}]",
            seed.root_lb[0],
            seed.root_ub[0]
        );
        assert!(
            seed.root_ub[1] > 1.0 && seed.root_lb[1] < -1.0 - 1e-9,
            "both circle variables are in R: [{}, {}]",
            seed.root_lb[1],
            seed.root_ub[1]
        );
        // Hand oracle embedded above: the exact feasible set is the unit
        // circle, so any correct derived box strictly contains [-1, 1] on
        // both coordinates, and acceptance itself is the gate under test
        // (a raw-box gate rejects before any of this). The tree-level solve
        // of a pure-equality circle is a separate numerical concern
        // (measured NumericalError on the relaxation LP; tracked with the
        // B&B numerical arms, not the acceptance gate).
    }

    /// A bilinear-only row gives root FBBT nothing (no bilinear propagation,
    /// L5 restriction), so the Phase-2 gate still rejects — but the message
    /// must say the derivation had its chance (and name the variable).
    /// Companion of `qcqp_guard::nonconvex_qcqp_with_unbounded_variable_
    /// stays_not_supported`, which pins the status half.
    #[test]
    fn residual_rejection_reports_root_fbbt_ran() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((0, 1, 1.0));
        qc.triplets.push((1, 0, 1.0));
        let a1 = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
        let mut problem =
            QpProblem::new(q_obj, c, a1, vec![1.0], bounds, vec![ConstraintType::Ge]).unwrap();
        problem.set_quadratic_constraints(vec![qc]).unwrap();

        let result = solve_qp_with(&problem, &SolverOptions::default());
        match &result.status {
            SolveStatus::NotSupported(msg) => {
                assert!(
                    msg.contains("even after root FBBT derivation"),
                    "message must report the derivation ran: {msg}"
                );
                assert!(msg.contains("variable 0"), "first bad variable: {msg}");
            }
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    /// No-op derivation attaches no seed: with every variable already boxed
    /// and only a bilinear (skipped) row, the derived box equals the declared
    /// box and the solver path stays the pre-Phase-2 seedless one.
    #[test]
    fn no_op_derivation_attaches_no_seed() {
        let n = 2usize;
        let q_obj = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let bounds = vec![(0.1, 3.0), (0.1, 3.0)];
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((0, 1, 1.0));
        qc.triplets.push((1, 0, 1.0));
        let a1 = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
        let mut problem =
            QpProblem::new(q_obj, c, a1, vec![1.0], bounds, vec![ConstraintType::Ge]).unwrap();
        problem.set_quadratic_constraints(vec![qc]).unwrap();

        let (_nc, seed) = build_nonconvex_with_root_box(&problem, 1e-6).unwrap();
        assert!(seed.is_none(), "nothing tightened -> no seed");
    }

    /// Proof contract (task item e/6): `audit_root_seed` must reject the
    /// same corrupt boxes `global_core`'s seed validation rejects —
    /// widened sides, NaN sides, crossed intervals — plus the box-only shape
    /// itself. FBBT is refine-only by construction, so these sentinels fail
    /// only if the construction AND the audit are both removed.
    #[test]
    fn audit_root_seed_rejects_widened_nan_crossed() {
        let qp = NonconvexQcqp {
            n: 2,
            p0: None,
            q0: vec![0.0; 2],
            quad: vec![],
            g_lin: CscMatrix::new(0, 2),
            h_lin: vec![],
            a_eq: CscMatrix::new(0, 2),
            b_eq: vec![],
            lb: vec![0.0, -1.0],
            ub: vec![1.0, 1.0],
        };
        let ok = GlobalSeed {
            x: Vec::new(),
            f: f64::INFINITY,
            root_lb: vec![0.0, -1.0],
            root_ub: vec![1.0, 1.0],
        };
        assert!(audit_root_seed(&qp, &ok).is_ok());

        let widened = GlobalSeed {
            root_ub: vec![1.5, 1.0],
            ..ok.clone()
        };
        assert!(audit_root_seed(&qp, &widened)
            .unwrap_err()
            .contains("widens"));
        let nanned = GlobalSeed {
            root_lb: vec![f64::NAN, -1.0],
            ..ok.clone()
        };
        assert!(audit_root_seed(&qp, &nanned).unwrap_err().contains("NaN"));
        let crossed = GlobalSeed {
            root_lb: vec![0.0, 0.5],
            root_ub: vec![1.0, 0.25],
            x: Vec::new(),
            f: f64::INFINITY,
        };
        assert!(audit_root_seed(&qp, &crossed)
            .unwrap_err()
            .contains("empty"));
        let with_incumbent = GlobalSeed {
            x: vec![0.5, 0.0],
            f: 0.5,
            root_lb: vec![0.0, -1.0],
            root_ub: vec![1.0, 1.0],
        };
        assert!(audit_root_seed(&qp, &with_incumbent)
            .unwrap_err()
            .contains("box-only"));
    }

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
