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
        Err(QcqpConversionError::DenseBudgetExceeded(_)) => {
            // Typed resource-limit refusal: the problem is structurally
            // supported, the dense expansion's demand estimate exceeds the
            // solver memory budget (the sizing gate inside `build_nonconvex_raw`
            // runs on sparse input, before any dense row). `ResourceExhausted`
            // (the `mip::parallel` refusal precedent) keeps it distinguishable
            // from the structural `NotSupported` and the algorithmic
            // `NumericalError` downstream.
            let mut result = SolverResult {
                status: SolveStatus::ResourceExhausted,
                ..SolverResult::default()
            };
            result.stats.route = SolveRoute::ConicQcqpNonconvex;
            return result;
        }
        Err(QcqpConversionError::Rejection(e)) => {
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

/// Why [`nonconvex_from_qp_problem`] refused to produce a [`NonconvexQcqp`].
/// Both payloads carry the full reason text; the variants differ in how the
/// route reports them, which is why a bare `String` is not enough.
#[derive(Debug)]
pub enum QcqpConversionError {
    /// Structural rejection: `QpProblem::validate` failed or the §3.1 bound
    /// gate found a bound-required variable without finite bounds. The
    /// payload is the public-facing `NotSupported` reason string, byte-stable
    /// across the pre-conversion gate move (it comes from the same
    /// `require_finite_bounds`-family message builder).
    Rejection(String),
    /// Accepted-class refusal: the dense-conversion demand estimate of
    /// `dense_conversion_demand_bytes` (private) exceeds the shared solver memory
    /// budget, so performing the conversion would be an unbounded OOM rather
    /// than a solve. The payload names the estimate, the demand formula's
    /// inputs and the gate/budget. This is a resource-limit refusal, not a
    /// structural statement about the problem, and surfaces as
    /// [`SolveStatus::ResourceExhausted`] (the `mip::parallel` thread-spawn
    /// refusal is the precedent for that mapping), never `NotSupported`.
    DenseBudgetExceeded(String),
}

impl std::fmt::Display for QcqpConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejection(msg) | Self::DenseBudgetExceeded(msg) => f.write_str(msg),
        }
    }
}

// The two variants differ only in reporting, never in message content, so the
// error "is" its message for inspection (`unwrap_err().contains(..)` reads
// the rejection text directly).
impl std::ops::Deref for QcqpConversionError {
    type Target = String;
    fn deref(&self) -> &String {
        match self {
            Self::Rejection(msg) | Self::DenseBudgetExceeded(msg) => msg,
        }
    }
}

/// Byte width of one stored `f64` — a dense `q: Vec<f64>` row of length `n`
/// is `DENSE_ROW_BYTES_PER_VAR * n` (`GQuadConstraint::q` is `Vec<f64>` by
/// contract; a sparse `q` is a deliberate follow-up package, not this gate's
/// concern).
const DENSE_ROW_BYTES_PER_VAR: usize = std::mem::size_of::<f64>();

/// Byte width of one `col_ptr` slot — a stored `CscMatrix` `p` keeps an
/// `(n + 1)`-entry `Vec<usize>` per constraint regardless of its own `nnz`.
const COL_PTR_BYTES_PER_COL: usize = std::mem::size_of::<usize>();

/// Retained bytes of one stored CSC non-zero (`usize` row index + `f64`
/// value; upper bound — the three `Vec` capacities are exact-sized here).
const CSC_ENTRY_BYTES: usize = std::mem::size_of::<usize>() + std::mem::size_of::<f64>();

/// Conservative byte demand of the dense row expansion
/// [`nonconvex_from_qp_problem`] performs, summed over quadratic source rows
/// (`nnz = triplets.len()` of row `k`) **before** any of it is allocated:
///
/// - `Le`/`Ge` emits one `GQuadConstraint`: dense `q`
///   (`DENSE_ROW_BYTES_PER_VAR * n`) plus one stored `p` CSC
///   (`COL_PTR_BYTES_PER_COL * (n + 1) + CSC_ENTRY_BYTES * nnz`).
/// - `Eq` emits the `(P, q, -b)` / `(-P, -q, b)` pair: **2x** the retained
///   cost above (two dense rows, two `p` CSCs), plus a per-row transient of
///   one extra `p` — the unscaled `p` stays live while `p.clone()` and
///   `scale_values(-1.0)` (which clones `col_ptr`/`row_ind` and collects a
///   fresh `values`) allocate the second stored copy.
///
/// Demand = `sum(retained) + max(single-row transient)`: transients die
/// within their row, so only the largest can coincide with the full retained
/// set. Compared against `kkt_solver::memory_budget_bytes()` — the same
/// budget the AAT builder gates on (`qp::aat::build_aat_upper_csc_scaled`),
/// so the solver refuses dense expansion exactly where its factorizations
/// would also be refused. Saturating `u128`: the estimate must never itself
/// overflow into a pass.
fn dense_conversion_demand_bytes(src: &QpProblem) -> u128 {
    let n = src.num_vars as u128;
    let dense_row = DENSE_ROW_BYTES_PER_VAR as u128 * n;
    let col_ptr = COL_PTR_BYTES_PER_COL as u128 * (n + 1);
    let has_qc = !src.quadratic_constraints.is_empty();
    let mut retained: u128 = 0;
    let mut transient_max: u128 = 0;
    for k in 0..src.num_constraints {
        // `validate` already guaranteed `len == 0 || len == num_constraints`.
        let Some(qc) = has_qc
            .then_some(&src.quadratic_constraints[k])
            .filter(|qc| !qc.triplets.is_empty())
        else {
            continue;
        };
        let p_cost = col_ptr + CSC_ENTRY_BYTES as u128 * qc.triplets.len() as u128;
        match src.constraint_types[k] {
            ConstraintType::Le | ConstraintType::Ge => {
                retained = retained.saturating_add(dense_row + p_cost);
            }
            ConstraintType::Eq => {
                retained = retained.saturating_add(2 * (dense_row + p_cost));
                transient_max = transient_max.max(p_cost);
            }
        }
    }
    retained.saturating_add(transient_max)
}

/// The normalized quadratic rows [`nonconvex_from_qp_problem`] will convert,
/// in the exact order and multiplicity it converts them (`Le`: `+P`, `Ge`:
/// `-P`, `Eq`: `+P` then `-P`; empty `QcqpMatrix` rows are linear, not quad).
/// Borrows only — feeding the sparse bound gate without materializing a
/// single dense row or per-row `CscMatrix`.
fn quad_coo_rows(src: &QpProblem) -> impl Iterator<Item = conic::CooQuadRow<'_>> + '_ {
    let has_qc = !src.quadratic_constraints.is_empty();
    (0..src.num_constraints)
        .filter_map(move |k| {
            has_qc
                .then_some(src.quadratic_constraints[k].triplets.as_slice())
                .filter(|t| !t.is_empty())
                .map(|t| (src.constraint_types[k], t))
        })
        .flat_map(|(ct, t)| {
            let pair: [Option<conic::CooQuadRow>; 2] = match ct {
                ConstraintType::Le => [
                    Some(conic::CooQuadRow {
                        triplets: t,
                        negate: false,
                    }),
                    None,
                ],
                ConstraintType::Ge => [
                    Some(conic::CooQuadRow {
                        triplets: t,
                        negate: true,
                    }),
                    None,
                ],
                ConstraintType::Eq => [
                    Some(conic::CooQuadRow {
                        triplets: t,
                        negate: false,
                    }),
                    Some(conic::CooQuadRow {
                        triplets: t,
                        negate: true,
                    }),
                ],
            };
            pair.into_iter().flatten()
        })
}

/// Convert a continuous [`QpProblem`] QCQP into a [`NonconvexQcqp`] for the
/// spatial branch-and-bound global solver, keeping the G1 raw-bound gate
/// (`conic::require_finite_bounds`). Any constraint sense and any (possibly
/// indefinite) quadratic matrix is accepted: `>=` rows are sign-flipped and
/// `=` rows become a pair of `<=` rows (the convex conic bridge instead
/// rejects quadratic `>=`/`=` as unable to certify convexity).
///
/// The QP route itself uses root FBBT instead
/// (`build_nonconvex_with_root_box`): Phase 2 (plan §3.2 line 89) judges
/// `R` finiteness on the derived root box, which this gated wrapper cannot
/// express. The wrapper stays as the model-level/compat entry with the
/// documented G1 semantics.
///
/// The bound gate runs on **sparse** input data before the first dense row
/// is allocated, so the QPLIB_8585/8683/8803/8810 class (all-equality DCQ
/// whose `R` members lack finite bounds) is rejected in `O(input)` —
/// measured pre-gate behavior was ~3.2 GiB/s RSS growth into a 12 GiB
/// self-kill on QPLIB_8585 (`n = 99_999`) for a problem the gate rejects
/// anyway. The rejection string is byte-identical to the post-conversion
/// gate's for the same input (single `qcqp_classify` message builder;
/// sentinels: `coo_gate_rejection_matches_materialized_gate`,
/// `nonconvex_gate_rejects_all_equality_dcq`) — only the timing moved. The
/// bound gate also precedes the sizing gate inside [`build_nonconvex_raw`]
/// (`sizing_gate_refuses_over_budget_before_converting`).
pub fn nonconvex_from_qp_problem(src: &QpProblem) -> Result<NonconvexQcqp, QcqpConversionError> {
    src.validate()
        .map_err(|e| QcqpConversionError::Rejection(e.to_string()))?;
    // Gate 1 (rejected-class blow-ups): the §3.1 bound requirement, evaluated
    // from the raw COO rows (`src.validate` above guarantees the parallel
    // lengths `quad_coo_rows` indexes). The all-finite fast path never
    // touches the rows; the rejection message comes from the same builder
    // the post-conversion gate used, so it is byte-identical for the same
    // input.
    let p0 = (!src.is_zero_q()).then_some(&src.q);
    let (lb, ub) =
        conic::require_finite_bounds_from_coo(src.num_vars, p0, quad_coo_rows(src), &src.bounds)
            .map_err(QcqpConversionError::Rejection)?;
    let mut nc = build_nonconvex_raw(src)?;
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
///
/// The sizing gate (Gate 2 below) runs on **sparse** input data before the
/// first dense row is allocated, and every dense path — the compat wrapper
/// *and* the root-box route — goes through here, so no caller can densify
/// beyond the solver memory budget. Problems within budget convert exactly
/// as before (no behavior change); refusal is a typed resource limit
/// (`DenseBudgetExceeded`), never a silent truncation.
pub fn build_nonconvex_raw(src: &QpProblem) -> Result<NonconvexQcqp, QcqpConversionError> {
    let n = src.num_vars;
    src.validate()
        .map_err(|e| QcqpConversionError::Rejection(e.to_string()))?;
    let p0 = (!src.is_zero_q()).then(|| src.q.clone());

    // Gate 2 (accepted-class blow-ups): refuse the O(n * m_q) dense expansion
    // up front when its demand estimate exceeds the solver memory budget.
    // Problems below budget convert exactly as before (no behavior change);
    // refusal is a typed resource limit, never a silent truncation.
    let demand = dense_conversion_demand_bytes(src);
    let budget = otspot_num::linalg::kkt_solver::memory_budget_bytes() as u128;
    if demand > budget {
        return Err(QcqpConversionError::DenseBudgetExceeded(format!(
            "qcqp_route dense-conversion sizing gate: estimated dense expansion \
             {demand} bytes (n={n}, dense row {DENSE_ROW_BYTES_PER_VAR} B/var, CSC \
             col_ptr {COL_PTR_BYTES_PER_COL} B/col, {CSC_ENTRY_BYTES} B/entry) exceeds \
             the {budget}-byte solver memory budget \
             (otspot_num::linalg::kkt_solver::memory_budget_bytes); refusing the \
             O(n*m_q) dense conversion"
        )));
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
        .map_err(|e| QcqpConversionError::Rejection(format!("G_lin build: {e:?}")))?;
    let a_eq = CscMatrix::from_triplets(&ae_r, &ae_c, &ae_v, ae_count, n)
        .map_err(|e| QcqpConversionError::Rejection(format!("A_eq build: {e:?}")))?;
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
///
/// Errors are typed like [`nonconvex_from_qp_problem`]'s: the over-budget
/// sizing refusal from [`build_nonconvex_raw`] propagates as
/// `DenseBudgetExceeded` (sparse, pre-dense), the Phase-2 acceptance
/// rejection is a `Rejection` carrying the "even after root FBBT derivation"
/// message — so the route can report the resource limit as
/// `ResourceExhausted` and the structural refusal as `NotSupported`.
///
/// Rejected-class problems never reach the dense conversion: the sparse
/// pre-gate [`sparse_root_box_rejection`] makes the identical acceptance
/// decision from the raw COO rows in `O(input)` (PR #86's
/// reject-from-sparse-data invariant, which the acceptance-gate move must
/// not silently drop).
fn build_nonconvex_with_root_box(
    src: &QpProblem,
    feas_tol: f64,
) -> Result<(NonconvexQcqp, Option<GlobalSeed>), QcqpConversionError> {
    // The bound-requirement rejection, run BEFORE `build_nonconvex_raw`'s
    // O(n * m_q) dense rows exist: same verdict, same message, from sparse
    // data (docs on `sparse_root_box_rejection`).
    if let Some(msg) = sparse_root_box_rejection(src) {
        return Err(QcqpConversionError::Rejection(msg));
    }
    let nc = build_nonconvex_raw(src)?;
    let req = conic::effective_bound_requirement(nc.n, nc.p0.as_ref(), &nc.quad, &nc.lb, &nc.ub)
        .map_err(QcqpConversionError::Rejection)?;
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
        return Err(QcqpConversionError::Rejection(root_box_rejection_message(
            (j, l, u),
            bad.len(),
            req.required.iter().filter(|b| **b).count(),
        )));
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
    audit_root_seed(&nc, &seed).map_err(QcqpConversionError::Rejection)?;
    Ok((nc, Some(seed)))
}

/// The single Phase-2 rejection-message builder, shared by the materialized
/// acceptance gate in [`build_nonconvex_with_root_box`] and the sparse
/// pre-gate [`sparse_root_box_rejection`] — byte-identity of the two paths'
/// public-facing reason string is structural, not duplicated text (same
/// pattern as `qcqp_classify::split_bounds_or_reject`).
fn root_box_rejection_message(
    first_bad: (usize, f64, f64),
    bad_count: usize,
    required_count: usize,
) -> String {
    let (j, l, u) = first_bad;
    format!(
        "nonconvex QCQP requires finite bounds on every bound-required \
         variable set R (plan §3.1); {bad_count} of the {required_count} \
         required variables have a non-finite bound even after root FBBT \
         derivation (first: variable {j} with derived bounds ({l}, {u}))"
    )
}

/// Over-approximation of the variables root FBBT can move on `src`'s rows,
/// computed from sparse data: a variable FBBT bounds must be a member of a
/// row FBBT propagates on — a linear row's `A` entry (`build_lin_rows` +
/// `propagate_row_bounds` only ever update row members), or a diagonal
/// member / linear member of a diagonal-only quadratic row (`diag_quad_pass`
/// only updates row terms; [`conic::coo_quad_row_diag_support`] reproduces
/// `diag_terms`' stored-entry skip semantics, and the non-finite `r` /
/// linear-part skips are the `src`-side checks below). Variables outside
/// this set provably keep their declared bounds in
/// `derive_root_box` — that is what makes [`sparse_root_box_rejection`]'s
/// early verdict sound. Sentinels:
/// `fbbt_derivable_support_contains_root_box_movers` (every actually-moved
/// variable is marked) and `root_box_pre_gate_agrees_with_materialized_root_box`.
fn fbbt_derivable_support(src: &QpProblem) -> Vec<bool> {
    let mut derivable = vec![false; src.num_vars];
    // Same row-major view as `build_nonconvex_raw` (O(nnz), no densification).
    let a_rows = src.a.transpose();
    let has_qc = !src.quadratic_constraints.is_empty();
    for k in 0..src.num_constraints {
        let qmat = has_qc
            .then(|| &src.quadratic_constraints[k])
            .filter(|qc| qc.nnz() > 0);
        let (row_idx, row_val) = a_rows.column(k);
        match qmat {
            None => {
                for &j in row_idx.iter() {
                    derivable[j] = true;
                }
            }
            Some(qc) => {
                // `diag_terms` skips the whole row when `r = ∓b[k]` or a
                // stored linear member is non-finite — nothing is derivable
                // through a skipped row, its members included.
                if !src.b[k].is_finite() || row_val.iter().any(|&v| !v.is_finite()) {
                    continue;
                }
                if let Some(diags) = conic::coo_quad_row_diag_support(&qc.triplets, src.num_vars) {
                    for &j in row_idx.iter() {
                        derivable[j] = true;
                    }
                    for j in diags {
                        derivable[j] = true;
                    }
                }
            }
        }
    }
    derivable
}

/// The Phase-2 bound-gate decision on sparse data, before the first dense
/// row exists (PR #86 invariant: a rejected-class all-equality DCQ — the
/// QPLIB_8585/8683/8803/8810 shape class — must be rejected in `O(input)`,
/// not after paying `O(n * m_q)` for rows the route never uses).
///
/// Returns `Some(message)` only when the full path (
/// [`build_nonconvex_raw`] → [`conic::effective_bound_requirement`] →
/// [`conic::derive_root_box`] → finiteness check) is guaranteed to reject
/// with the byte-identical [`root_box_rejection_message`]: the effective
/// set `R` is identical by shared classification code
/// ([`conic::effective_bound_requirement_from_coo`] over `quad_coo_rows`,
/// which mirrors the conversion's order/multiplicity/sign), and every bad
/// member is a variable `fbbt_derivable_support` proves FBBT cannot move,
/// so its derived side equals its declared side and the full path's `bad`
/// set equals this pre-gate's. A bad member that any propagatable row
/// touches makes the pre-gate abstain (`None`) — the materialized path then
/// judges it exactly as before (conservative: FBBT might finitize it, which
/// is Phase 2's whole point, e.g. `circle_equality_all_free_routes_through_derived_gate`).
/// Any precondition failure (invalid problem, classifier error) abstains
/// too, leaving the materialized path's error reporting byte-identical.
fn sparse_root_box_rejection(src: &QpProblem) -> Option<String> {
    // The gate's own all-finite fast path (mirrors
    // `require_finite_bounds_from_coo`): no bad candidate exists, so no
    // classification cost for the common boxed problem.
    if src
        .bounds
        .iter()
        .all(|&(l, u)| l.is_finite() && u.is_finite())
    {
        return None;
    }
    src.validate().ok()?;
    let p0 = (!src.is_zero_q()).then_some(&src.q);
    let coo_rows: Vec<conic::CooQuadRow> = quad_coo_rows(src).collect();
    let req = conic::effective_bound_requirement_from_coo(src.num_vars, p0, &coo_rows, &src.bounds)
        .ok()?;
    let bad: Vec<(usize, f64, f64)> = req
        .required
        .iter()
        .enumerate()
        .filter(|(_, need)| **need)
        .filter_map(|(j, _)| {
            let (l, u) = src.bounds[j];
            (!(l.is_finite() && u.is_finite())).then_some((j, l, u))
        })
        .collect();
    if bad.is_empty() {
        return None;
    }
    let derivable = fbbt_derivable_support(src);
    // One touchable bad variable abstains: FBBT may shrink `bad`, which
    // would change the rejection's counts — only an all-untouchable `bad`
    // guarantees the identical message.
    if bad.iter().any(|&(j, _, _)| derivable[j]) {
        return None;
    }
    Some(root_box_rejection_message(
        bad[0],
        bad.len(),
        req.required.iter().filter(|b| **b).count(),
    ))
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

    // ---------------------------------------------------------------------
    // Sparse root-box pre-gate (`sparse_root_box_rejection`) soundness:
    // corpus-wide agreement with the materialized Phase-2 gate.
    // ---------------------------------------------------------------------

    fn all_free_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); n]
    }

    fn bilinear_pair(n: usize, i: usize, j: usize) -> QcqpMatrix {
        let mut qc = QcqpMatrix::new(n);
        qc.triplets.push((i, j, 1.0));
        qc.triplets.push((j, i, 1.0));
        qc
    }

    fn empty_a(m: usize, n: usize) -> CscMatrix {
        CscMatrix::from_triplets(&[], &[], &[], m, n).unwrap()
    }

    /// Shape classes the sparse pre-gate must judge identically to the
    /// materialized root-box gate: pure bilinear rejects (`R` = rule-3
    /// endpoints, nothing derivable), DCQ equality splits, D1-demoted PSD
    /// rows over free support, DROP_TOL/exact bilinear cancellations
    /// (stored-p-empty rows carry no derivable support), and conservative
    /// abstentions where a bad member sits in a propagatable row (linear
    /// rows, diagonal-only rows — including the accepted circle/indefinite
    /// FBBT fixtures and the `coo_gate` rule-2 fixture).
    fn root_box_gate_agreement_corpus() -> Vec<(&'static str, QpProblem)> {
        let mut corpus: Vec<(&'static str, QpProblem)> = Vec::new();

        // 1. Bilinear `Ge` row, all free: the `residual_rejection_reports_
        // root_fbbt_ran` shape — rule-3 endpoints, pre-gate must reject.
        {
            let n = 2usize;
            let mut p = QpProblem::new(
                CscMatrix::new(n, n),
                vec![1.0, 1.0],
                empty_a(1, n),
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Ge],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![bilinear_pair(n, 0, 1)])
                .unwrap();
            corpus.push(("bilinear_ge_all_free", p));
        }
        // 2. DCQ equality split (8595 class at unit scale), all free.
        {
            let n = 4usize;
            let mut p = QpProblem::new(
                CscMatrix::new(n, n),
                vec![0.0; n],
                empty_a(2, n),
                vec![0.0; 2],
                all_free_bounds(n),
                vec![ConstraintType::Eq; 2],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![bilinear_pair(n, 0, 1), bilinear_pair(n, 2, 3)])
                .unwrap();
            corpus.push(("dense_dcq_eq_mini", p));
        }
        // 3. Circle equality, all free: FBBT finitizes — must ACCEPT
        // (`circle_equality_all_free_routes_through_derived_gate` shape).
        {
            let n = 2usize;
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.push((0, 0, 2.0));
            qc.triplets.push((1, 1, 2.0));
            let mut p = QpProblem::new(
                CscMatrix::new(n, n),
                vec![1.0, 0.0],
                empty_a(1, n),
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Eq],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("circle_eq_all_free", p));
        }
        // 4. Indefinite diagonal `Le` row, free x0: accepted via derivation
        // (`indefinite_row_with_free_objective_var_solves_via_root_fbbt`).
        {
            let n = 2usize;
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.push((0, 0, 2.0));
            qc.triplets.push((1, 1, -2.0));
            let mut p = QpProblem::new(
                CscMatrix::new(n, n),
                vec![1.0, 0.0],
                empty_a(1, n),
                vec![1.0],
                vec![(f64::NEG_INFINITY, f64::INFINITY), (-2.0, 2.0)],
                vec![ConstraintType::Le],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("indefinite_diag_free_x0", p));
        }
        // 5. PSD row over free support: the D1 conservative demotion puts
        // its off-diagonal endpoints in `R`; the row is FBBT-skipped (stored
        // off-diagonal), so the pre-gate must reject. `certifiably_psd`
        // holds by strict diagonal dominance (composed diag 2 > off 1).
        {
            let n = 2usize;
            let mut qc = QcqpMatrix::new(n);
            qc.triplets
                .extend([(0, 0, 4.0), (1, 1, 4.0), (0, 1, 1.0), (1, 0, 1.0)]);
            let mut p = QpProblem::new(
                CscMatrix::new(n, n),
                vec![1.0, 1.0],
                empty_a(1, n),
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Le],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("psd_row_free_support", p));
        }
        // 6. Linear equality row + nonconvex objective `p0` over free vars:
        // `R` from `p0` rule 3, but both members touch a linear row the
        // FBBT fixpoint reads — the pre-gate must ABSTAIN (the full path's
        // propagation failure, not the pre-gate's, decides). The quad rows
        // vector carries an all-zero (nnz 0) row: the row converts as
        // linear, and the corpus stays a valid route input (the raw
        // builder's per-row demand walk indexes `quadratic_constraints`).
        {
            let n = 2usize;
            let p0 = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], n, n).unwrap();
            let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
            let mut p = QpProblem::new(
                p0,
                vec![0.0; n],
                a,
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Eq],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![QcqpMatrix::new(n)])
                .unwrap();
            corpus.push(("linear_row_free_p0", p));
        }
        // 7. The `coo_gate_rejection_matches_materialized_gate` fixture
        // verbatim: rule-2 veto puts free `x3` in `R`; `x3` sits in a
        // diagonal-only flipped-`Ge` row → pre-gate abstains, the full path
        // rejects. Also exercises DROP_TOL / stored-zero / duplicate-merge
        // COO fidelity on the effective gate.
        {
            let n = 4usize;
            let q_obj = CscMatrix::from_triplets(&[0, 2], &[2, 0], &[3.0, 3.0], n, n).unwrap();
            let mut p = QpProblem::new(
                q_obj,
                vec![0.0; n],
                empty_a(6, n),
                vec![0.0; 6],
                vec![(0.0, 1.0); n],
                vec![
                    ConstraintType::Le,
                    ConstraintType::Le,
                    ConstraintType::Ge,
                    ConstraintType::Eq,
                    ConstraintType::Ge,
                    ConstraintType::Eq,
                ],
            )
            .unwrap();
            let mut q1 = QcqpMatrix::new(n);
            q1.triplets.extend([(0, 1, 2.0), (1, 0, 2.0), (0, 1, -2.0)]);
            let mut q2 = QcqpMatrix::new(n);
            q2.triplets.push((3, 3, 2.0));
            let mut q3 = QcqpMatrix::new(n);
            q3.triplets.push((0, 0, 1e-16));
            let mut q4 = QcqpMatrix::new(n);
            q4.triplets.push((1, 1, 0.0));
            let mut q5 = QcqpMatrix::new(n);
            q5.triplets
                .extend([(1, 2, 1.0), (2, 1, 1.0), (1, 2, 1e-16)]);
            p.set_quadratic_constraints(vec![QcqpMatrix::new(n), q1, q2, q3, q4, q5])
                .unwrap();
            p.bounds[3] = (f64::NEG_INFINITY, f64::INFINITY);
            corpus.push(("coo_gate_rule2_fixture", p));
        }
        // 8. Exact bilinear cancellation (merged stored `p` empty) with a
        // nonconvex `p0` over free vars: the canceled pair is NOT derivable
        // support (an empty row moves nothing), so the pre-gate must reject.
        {
            let n = 2usize;
            let p0 = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], n, n).unwrap();
            let mut qc = QcqpMatrix::new(n);
            qc.triplets
                .extend([(0, 1, 1.0), (0, 1, -1.0), (1, 0, 1.0), (1, 0, -1.0)]);
            let mut p = QpProblem::new(
                p0,
                vec![1.0, 1.0],
                empty_a(1, n),
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Le],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("exact_cancel_bilinear_p0", p));
        }
        // 9. Same shape, `DROP_TOL` cancellation instead of exact zero
        // (single 1e-16 runs merge away): identical verdict required.
        {
            let n = 2usize;
            let p0 = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], n, n).unwrap();
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.extend([(0, 1, 1e-16), (1, 0, -1e-16)]);
            let mut p = QpProblem::new(
                p0,
                vec![1.0, 1.0],
                empty_a(1, n),
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Le],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("drop_tol_cancel_bilinear_p0", p));
        }
        // 10. Diagonal-only `Le` row with a linear member (`x0 + x1² <= 1`)
        // + nonconvex `p0` on (x0, x2): `x0` is derivable (FBBT's completed
        // square bounds it — an actual mover), `x2` is not (sole member of
        // the objective only). Bad set is mixed → pre-gate abstains and the
        // full path rejects naming `x2`.
        {
            let n = 3usize;
            let p0 = CscMatrix::from_triplets(&[0, 2], &[2, 0], &[1.0, 1.0], n, n).unwrap();
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.push((1, 1, 2.0));
            let mut p = QpProblem::new(
                p0,
                vec![1.0, 0.0, 0.0],
                a,
                vec![1.0],
                all_free_bounds(n),
                vec![ConstraintType::Le],
            )
            .unwrap();
            p.set_quadratic_constraints(vec![qc]).unwrap();
            corpus.push(("diag_row_linear_member_p0", p));
        }
        corpus
    }

    /// The materialized Phase-2 gate as an independent oracle: exactly the
    /// judgment of `build_nonconvex_with_root_box` WITHOUT the sparse
    /// pre-gate (raw build → materialized `R` → derived box → finiteness
    /// check → shared message builder). Fidelity of this oracle against the
    /// shipped route is itself asserted per fixture (the pre-gate abstention
    /// branch runs the materialized path verbatim).
    fn materialized_root_box_rejection(src: &QpProblem) -> Option<String> {
        let nc = build_nonconvex_raw(src).expect("corpus problems convert raw");
        let req =
            conic::effective_bound_requirement(nc.n, nc.p0.as_ref(), &nc.quad, &nc.lb, &nc.ub)
                .expect("corpus classifies materialized");
        let derived = conic::derive_root_box(&nc, 1e-6);
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
        let &first = bad.first()?;
        Some(root_box_rejection_message(
            first,
            bad.len(),
            req.required.iter().filter(|b| **b).count(),
        ))
    }

    /// The shipped route's `Rejection` message (the materialized path, when
    /// the pre-gate abstained; the pre-gate's own, when it fired).
    fn route_root_box_rejection(src: &QpProblem) -> Option<String> {
        match build_nonconvex_with_root_box(src, 1e-6) {
            Ok(_) => None,
            Err(QcqpConversionError::Rejection(msg)) => Some(msg),
            Err(QcqpConversionError::DenseBudgetExceeded(msg)) => {
                panic!("corpus problems are unit-scale, sizing gate must not fire: {msg}")
            }
        }
    }

    /// Soundness, part 1: the COO twin of the effective set equals the
    /// materialized gate's classification field-for-field (`R`, the per-row
    /// convexity verdicts, the rule-2 diagonals) on every corpus shape —
    /// by shared fold code, pinned against drift in `lift_coeffs_from_coo`
    /// or `quad_coo_rows`' order/multiplicity/sign mirroring.
    /// sentinel: COO から導出した境界要求が materialized 版と一致する張 — quad_coo_rows の走査順序/多重度/符号が崩れると fail
    #[test]
    fn coo_effective_requirement_matches_materialized_root_box_gate() {
        for (name, p) in root_box_gate_agreement_corpus() {
            let nc = build_nonconvex_raw(&p).expect("corpus converts raw");
            let coo_rows: Vec<conic::CooQuadRow> = quad_coo_rows(&p).collect();
            let coo = conic::effective_bound_requirement_from_coo(
                p.num_vars,
                (!p.is_zero_q()).then_some(&p.q),
                &coo_rows,
                &p.bounds,
            )
            .expect("corpus classifies on COO");
            let mat =
                conic::effective_bound_requirement(nc.n, nc.p0.as_ref(), &nc.quad, &nc.lb, &nc.ub)
                    .expect("corpus classifies materialized");
            assert_eq!(coo.required, mat.required, "{name}: R membership drift");
            assert_eq!(
                coo.convex_rows, mat.convex_rows,
                "{name}: convex-row verdict drift (row stream must mirror the conversion order)"
            );
            assert_eq!(coo.rule2_diag, mat.rule2_diag, "{name}: rule-2 drift");
        }
    }

    /// Soundness, part 2: whenever the sparse pre-gate rejects, the
    /// materialized Phase-2 gate rejects with the byte-identical message;
    /// the pre-gate never rejects where the full path accepts (that would
    /// regress a Phase-2 FBBT acceptance); and on every fixture where the
    /// pre-gate abstains, the shipped route's rejection equals the oracle's
    /// (the materialized path runs verbatim there). The corpus must cover
    /// both the fire branch and the conservative-abstention-with-eventual-
    /// rejection branch, or the agreement is vacuous.
    /// sentinel: 疎入力の事前ゲートが materialized 版ゲートと意味一致する張 — 事前ゲートが oracle と分岐して受理/却下が反転すると fail
    #[test]
    fn root_box_pre_gate_agrees_with_materialized_root_box() {
        let mut pre_fires = 0usize;
        let mut abstained_full_rejections = 0usize;
        for (name, p) in root_box_gate_agreement_corpus() {
            let pre = sparse_root_box_rejection(&p);
            let oracle = materialized_root_box_rejection(&p);
            match (&pre, &oracle) {
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a, b,
                        "{name}: pre-gate message must equal the materialized gate's byte-for-byte"
                    );
                    pre_fires += 1;
                }
                (Some(a), None) => panic!(
                    "{name}: unsound pre-rejection — materialized path accepts, pre-gate says {a}"
                ),
                (None, Some(_)) => abstained_full_rejections += 1,
                (None, None) => {}
            }
            if pre.is_none() {
                let route = route_root_box_rejection(&p);
                assert_eq!(
                    route.as_ref(),
                    oracle.as_ref(),
                    "{name}: oracle must describe the shipped materialized path"
                );
            }
        }
        assert!(
            pre_fires >= 5,
            "corpus must exercise the fire branch on >= 5 shapes, got {pre_fires}"
        );
        assert!(
            abstained_full_rejections >= 3,
            "corpus must cover the conservative abstention that still rejects \
             (touchable bad members) on >= 3 shapes, got {abstained_full_rejections}"
        );
    }

    /// Soundness, part 3 (`fbbt_derivable_support` contract): every
    /// variable `derive_root_box` actually moves is marked derivable, so a
    /// variable the support builder calls untouchable provably keeps its
    /// declared bounds — the pre-gate's `derived == declared` premise.
    /// Coverage: the corpus must contain real movers (accepted fixtures and
    /// the completed-square diagonal row), not just unmoved rejections.
    #[test]
    fn fbbt_derivable_support_contains_root_box_movers() {
        let mut movers = 0usize;
        for (name, p) in root_box_gate_agreement_corpus() {
            let nc = build_nonconvex_raw(&p).expect("corpus converts raw");
            let derived = conic::derive_root_box(&nc, 1e-6);
            let derivable = fbbt_derivable_support(&p);
            for j in 0..nc.n {
                if nc.lb[j] != derived.lb[j] || nc.ub[j] != derived.ub[j] {
                    assert!(
                        derivable[j],
                        "{name}: x{j} moved by root FBBT but not marked derivable"
                    );
                    movers += 1;
                }
            }
        }
        assert!(
            movers >= 3,
            "corpus must exercise actually-moved variables, got {movers}"
        );
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

    /// Materialize the `p` matrices exactly the way the pre-gate conversion
    /// loop did (`Le`: `P`, `Ge`: `-P` via `scale_values(-1.0)`, `Eq`: the
    /// `P`/`-P` pair, empty-triplet rows skipped as linear) — the independent
    /// "old path" whose rejection string the sparse COO gate must reproduce.
    /// `q`/`r` are irrelevant to the classifier (it reads `p` only), so they
    /// stay at their defaults here.
    fn materialized_gate_input(src: &QpProblem) -> (Option<CscMatrix>, Vec<GQuadConstraint>) {
        let p0 = (!src.is_zero_q()).then(|| src.q.clone());
        let has_qc = !src.quadratic_constraints.is_empty();
        let mut quad = Vec::new();
        for k in 0..src.num_constraints {
            let Some(qc) = has_qc
                .then(|| &src.quadratic_constraints[k])
                .filter(|qc| qc.nnz() > 0)
            else {
                continue;
            };
            match src.constraint_types[k] {
                ConstraintType::Le => quad.push(GQuadConstraint {
                    p: qcqp_matrix_to_csc(qc),
                    q: vec![],
                    r: 0.0,
                }),
                ConstraintType::Ge => quad.push(GQuadConstraint {
                    p: qcqp_matrix_to_csc(qc).scale_values(-1.0),
                    q: vec![],
                    r: 0.0,
                }),
                ConstraintType::Eq => {
                    let p = qcqp_matrix_to_csc(qc);
                    quad.push(GQuadConstraint {
                        p: p.clone(),
                        q: vec![],
                        r: 0.0,
                    });
                    quad.push(GQuadConstraint {
                        p: p.scale_values(-1.0),
                        q: vec![],
                        r: 0.0,
                    });
                }
            }
        }
        (p0, quad)
    }

    /// Byte-identity contract for the pre-conversion bound gate: for one
    /// problem mixing every normalization path — linear rows, duplicate
    /// triplets that cancel under `from_triplets`' merge (`(0,1,+2) +
    /// (0,1,-2) -> 0`, dropped), a `<= 1e-15` (`DROP_TOL`) merged magnitude,
    /// stored-zero-only triplets, a sign-flipped `Ge` negative diagonal
    /// vetoing rule 2 on `x3` (whose `R` membership has no other source, so a
    /// dropped sign flip is observable), an equality `P`/`-P` split with an
    /// off-diagonal pair, and a nonconvex objective feeding rule 3 — the
    /// sparse COO gate's rejection string must equal the string the
    /// post-conversion gate produced from the materialized `p` matrices, byte
    /// for byte (`assert_eq!` on `String` is byte comparison). Both strings
    /// are read from the same run, so this pins the classification contract,
    /// not a captured literal.
    ///
    /// Sentinel: any drift in `lift_coeffs_from_coo`'s composition (filter,
    /// merge order, `DROP_TOL`, negation point, half ordering) changes `R`
    /// and makes the two strings diverge — counts, first index, or bounds
    /// text.
    #[test]
    fn coo_gate_rejection_matches_materialized_gate() {
        let n = 4usize;
        let q_obj = CscMatrix::from_triplets(&[0, 2], &[2, 0], &[3.0, 3.0], n, n).unwrap(); // objective off-diagonal (0,2): rule-3 endpoints
        let c = vec![0.0; n];
        let a = CscMatrix::from_triplets(&[], &[], &[], 6, n).unwrap();
        let b = vec![0.0; 6];
        // Row plan (see doc): 0 linear Le; 1 Le duplicate cancellation;
        // 2 Ge flipped negative diagonal on x3 (sole source of x3 in R);
        // 3 Eq DROP_TOL-only; 4 Ge stored zero only; 5 Eq bilinear (1,2).
        let ctypes = vec![
            ConstraintType::Le,
            ConstraintType::Le,
            ConstraintType::Ge,
            ConstraintType::Eq,
            ConstraintType::Ge,
            ConstraintType::Eq,
        ];
        let mut problem = QpProblem::new(q_obj, c, a, b, vec![(0.0, 1.0); n], ctypes).unwrap();
        let mut q1 = QcqpMatrix::new(n);
        q1.triplets.extend([(0, 1, 2.0), (1, 0, 2.0), (0, 1, -2.0)]);
        let mut q2 = QcqpMatrix::new(n);
        q2.triplets.push((3, 3, 2.0)); // flipped: lift coeff -1 -> rule-2 veto
        let mut q3 = QcqpMatrix::new(n);
        q3.triplets.push((0, 0, 1e-16)); // <= DROP_TOL: merged away
        let mut q4 = QcqpMatrix::new(n);
        q4.triplets.push((1, 1, 0.0)); // stored zero: filtered away
        let mut q5 = QcqpMatrix::new(n);
        q5.triplets
            .extend([(1, 2, 1.0), (2, 1, 1.0), (1, 2, 1e-16)]);
        problem
            .set_quadratic_constraints(vec![QcqpMatrix::new(n), q1, q2, q3, q4, q5])
            .unwrap();

        // x3 loses its finite bound: only the flipped-`Ge` negative diagonal
        // puts it in `R` (rule-2 veto), so a dropped sign flip on the COO
        // side would make `R` miss x3 and the gate would pass instead of
        // reject.
        problem.bounds[3] = (f64::NEG_INFINITY, f64::INFINITY);

        let sparse_err = match nonconvex_from_qp_problem(&problem) {
            Ok(_) => panic!("x3 in R via the rule-2 diagonal veto must be rejected"),
            Err(QcqpConversionError::DenseBudgetExceeded(msg)) => {
                panic!("bound rejection must precede the sizing gate: {msg}")
            }
            Err(QcqpConversionError::Rejection(msg)) => msg,
        };
        let (p0, quad) = materialized_gate_input(&problem);
        let materialized_err =
            conic::require_finite_bounds(problem.num_vars, p0.as_ref(), &quad, &problem.bounds)
                .expect_err("the old post-conversion gate rejects the same input");
        assert_eq!(
            sparse_err, materialized_err,
            "sparse pre-conversion gate must reproduce the materialized gate's \
             rejection byte for byte"
        );
        assert!(
            sparse_err.contains("required variables have a non-finite bound"),
            "sanity: this is the R-gate rejection: {sparse_err}"
        );
    }

    /// 8595-class shape pinned at unit scale (all-equality bilinear DCQ over
    /// free variables): the COMPAT WRAPPER's rejection text — the pre-Phase-2
    /// raw-bound message, which the wrapper (`nonconvex_from_qp_problem`,
    /// the `otspot-model` fallback entry) keeps byte-identical because its
    /// G1 gate stays in front of the conversion loop (PR #86 fence, intact).
    /// The QP ROUTE's Phase-2 verdict ("even after root FBBT derivation") is
    /// pinned at route scale by `residual_rejection_reports_root_fbbt_ran`
    /// and, with the peak-RSS timing sentinel, by the integration test
    /// `qcqp_route_rejects_dense_dcq_at_sparse_root_gate`
    /// (successor of `qcqp_route_rejects_dense_dcq_before_densifying`).
    /// Independent oracle: every variable is a rule-3 off-diagonal endpoint
    /// (`|R| = 4`), none has a finite bound (`bad = 4`, first index 0), so
    /// the full string is hand-derivable from the gate's documented template
    /// — the literal below was captured from the PRE-fix code (same input,
    /// dense path, `QPLIB_8585`-era message builder) and re-verified against
    /// the template.
    #[test]
    fn nonconvex_gate_rejects_all_equality_dcq() {
        let n = 4usize;
        let q_obj = CscMatrix::new(n, n);
        let a = CscMatrix::from_triplets(&[], &[], &[], 2, n).unwrap();
        let mut problem = QpProblem::new(
            q_obj,
            vec![0.0; n],
            a,
            vec![0.0; 2],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            vec![ConstraintType::Eq; 2],
        )
        .unwrap();
        let mut rows = Vec::new();
        for k in 0..2 {
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.push((2 * k, 2 * k + 1, 1.0));
            qc.triplets.push((2 * k + 1, 2 * k, 1.0));
            rows.push(qc);
        }
        problem.set_quadratic_constraints(rows).unwrap();

        let err = match nonconvex_from_qp_problem(&problem) {
            Ok(_) => panic!("all-free all-eq DCQ must be rejected"),
            Err(QcqpConversionError::DenseBudgetExceeded(msg)) => {
                panic!("bound rejection must precede the sizing gate: {msg}")
            }
            Err(QcqpConversionError::Rejection(msg)) => msg,
        };
        assert_eq!(
            err,
            "nonconvex QCQP requires finite bounds on every bound-required \
             variable set R (plan §3.1); 4 of the 4 required variables have a \
             non-finite bound (first: variable 0 with bounds (-inf, inf))"
        );
    }

    /// The sizing gate refuses the accepted class (every bound finite, so the
    /// bound gate takes its fast path) when the dense-conversion demand
    /// estimate exceeds `memory_budget_bytes()`, with the estimate and the
    /// budget named in the typed message — and it fires on the ESTIMATE,
    /// without converting: the independent oracle recomputes the documented
    /// formula by hand from the problem's parameters rather than calling the
    /// gate's own function.
    ///
    /// Ordering is also pinned here: the same over-budget shape with free
    /// bounds must come back as the structural `Rejection` (bound gate
    /// first), never as `DenseBudgetExceeded` — a rejected-class problem is
    /// never reported as a resource limit.
    ///
    /// Sentinel: deleting the sizing gate makes `matches!` see `Ok` (the
    /// conversion runs); inverting the gate order flips the free-bounds case
    /// to `DenseBudgetExceeded`.
    #[test]
    fn sizing_gate_refuses_over_budget_before_converting() {
        let n = 10_000usize;
        let m = 20_000usize;
        let mut problem = QpProblem::new(
            CscMatrix::new(n, n),
            vec![0.0; n],
            CscMatrix::from_triplets(&[], &[], &[], m, n).unwrap(),
            vec![0.0; m],
            vec![(0.0, 10.0); n],
            vec![ConstraintType::Eq; m],
        )
        .unwrap();
        let mut rows = Vec::new();
        for k in 0..m {
            let mut qc = QcqpMatrix::new(n);
            qc.triplets.extend([
                (2 * k % n, (2 * k + 1) % n, 1.0),
                ((2 * k + 1) % n, 2 * k % n, 1.0),
            ]);
            rows.push(qc);
        }
        problem.set_quadratic_constraints(rows).unwrap();

        // Independent oracle for the documented formula: 2 stored copies of
        // (dense row + p CSC) per equality row + max single-row transient
        // (one more p CSC): dense row 8n, col_ptr 8(n+1), 16 bytes/entry.
        let dense_row = 8u128 * n as u128;
        let p_cost = 8u128 * (n as u128 + 1) + 16 * 2;
        let expected_demand = m as u128 * 2 * (dense_row + p_cost) + p_cost;
        let budget = otspot_num::linalg::kkt_solver::memory_budget_bytes() as u128;
        assert!(
            expected_demand > budget,
            "test premise: estimate {expected_demand} must exceed the {budget}-byte budget"
        );

        let err = match nonconvex_from_qp_problem(&problem) {
            Ok(_) => panic!("over-budget dense conversion must be refused"),
            Err(QcqpConversionError::Rejection(msg)) => {
                panic!("finite-bound problem must not hit the bound gate: {msg}")
            }
            Err(QcqpConversionError::DenseBudgetExceeded(msg)) => msg,
        };
        assert!(
            err.contains(&expected_demand.to_string()) && err.contains(&budget.to_string()),
            "message must name the estimate and the budget: {err}"
        );

        // Ordering: free bounds on an R variable => structural rejection
        // first, even though this shape is also over budget.
        let mut free = problem.clone();
        for bound in free.bounds.iter_mut() {
            *bound = (f64::NEG_INFINITY, f64::INFINITY);
        }
        let ordered = nonconvex_from_qp_problem(&free).unwrap_err();
        assert!(
            matches!(ordered, QcqpConversionError::Rejection(_)),
            "bound gate must precede the sizing gate: {ordered}"
        );
    }
}
