//! Benchmark and test utility functions.
//!
//! Shared helpers used by integration tests, benchmark binaries, and the
//! `lp_screen` development tool.  Not part of the public solver API.

use otspot_core::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_core::qp::kkt_resid::{self, dd_impl, f64_impl};
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;
use otspot_core::tolerances::{PIVOT_TOL, ZERO_TOL};
use otspot_io::qplib::{parse_qplib, QplibError, QplibProblem};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Re-export production utilities that also live in otspot-core.
pub use otspot_core::tolerances::{obj_within_tol, OBJ_MATCH_REL_TOL};

/// Relative primal feasibility max-violation for LP / QP.
///
/// Checks `Ax {op} b` and `lb ≤ x ≤ ub` component-wise with relative scaling
/// `1 + |ax| + |b|` (or `1 + |x| + |bnd|` for bound constraints).
pub fn primal_feas_max(
    a: &CscMatrix,
    b: &[f64],
    ct: &[ConstraintType],
    bounds: &[(f64, f64)],
    x: &[f64],
) -> f64 {
    let n = bounds.len();
    if x.len() != n || x.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let ax = if a.nrows() > 0 {
        match a.mat_vec_mul(x) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        Vec::new()
    };
    let mut max_v = 0.0_f64;
    #[allow(unreachable_patterns)]
    for (i, cti) in ct.iter().enumerate() {
        if i >= ax.len() || i >= b.len() {
            continue;
        }
        let viol = match cti {
            ConstraintType::Le => (ax[i] - b[i]).max(0.0),
            ConstraintType::Ge => (b[i] - ax[i]).max(0.0),
            ConstraintType::Eq => (ax[i] - b[i]).abs(),
            _ => continue,
        };
        if !viol.is_finite() || !ax[i].is_finite() || !b[i].is_finite() {
            return f64::INFINITY;
        }
        let scale = 1.0 + ax[i].abs() + b[i].abs();
        max_v = max_v.max(viol / scale);
    }
    for (j, &(lb, ub)) in bounds.iter().enumerate() {
        if lb.is_finite() {
            max_v = max_v.max((lb - x[j]).max(0.0) / (1.0 + x[j].abs() + lb.abs()));
        }
        if ub.is_finite() {
            max_v = max_v.max((x[j] - ub).max(0.0) / (1.0 + x[j].abs() + ub.abs()));
        }
    }
    max_v
}

/// QP KKT residual max (stationarity / primal_inf / comp_ineq / comp_bound),
/// component-relative-scaled.
pub fn compute_qp_kkt_max(prob: &QpProblem, x: &[f64], y: &[f64], bd: &[f64]) -> f64 {
    let n = prob.num_vars;
    if x.len() != n {
        return f64::INFINITY;
    }
    if prob.a.nrows() > 0 && !y.is_empty() && y.len() != prob.a.nrows() {
        return f64::INFINITY;
    }

    let qx = f64_impl::qx(&prob.q, x);
    let aty = f64_impl::aty(&prob.a, y, n);
    let bound_contrib = kkt_resid::bound_contrib(&prob.bounds, bd);

    let mut max_resid = 0.0_f64;
    for j in 0..n {
        let r = qx[j] + aty[j] + bound_contrib[j] + prob.c[j];
        let scale = 1.0 + qx[j].abs() + aty[j].abs() + bound_contrib[j].abs() + prob.c[j].abs();
        max_resid = max_resid.max(r.abs() / scale);
    }

    max_resid = max_resid.max(primal_feas_max(
        &prob.a,
        &prob.b,
        &prob.constraint_types,
        &prob.bounds,
        x,
    ));

    let ax = f64_impl::ax(&prob.a, x);
    let comp_i = f64_impl::comp_ineq_products(&ax, &prob.b, &prob.constraint_types, y);
    for (i, &prod) in comp_i.iter().enumerate() {
        if prod == 0.0 {
            continue;
        }
        let scale = 1.0 + y[i].abs() * (ax[i].abs() + prob.b[i].abs());
        max_resid = max_resid.max(prod / scale);
    }
    let comp_b = kkt_resid::comp_bound_products(&prob.bounds, x, bd);
    let mut idx = 0_usize;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() && idx < bd.len() {
            let scale = 1.0 + bd[idx].abs() * (x[j].abs() + lb.abs());
            max_resid = max_resid.max(comp_b[idx] / scale);
            idx += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() && idx < bd.len() {
            let scale = 1.0 + bd[idx].abs() * (x[j].abs() + ub.abs());
            max_resid = max_resid.max(comp_b[idx] / scale);
            idx += 1;
        }
    }
    max_resid
}

/// QCQP quadratic-constraint primal feasibility max-violation.
///
/// For each constraint `k` with a non-empty `quadratic_constraints[k]`,
/// evaluates the full row `1/2 x'Q_k x + (Ax)_k {<=,=,>=} b_k`
/// (`QpProblem`'s documented QCQP row semantics) and returns the max
/// relative violation, scaled `1 + |lhs_k| + |b_k|` (matches
/// `compute_qp_kkt_max`'s `primal_feas_max` convention). Constraints with an
/// empty `QcqpMatrix` (pure-linear rows in an otherwise-QCQP problem) are
/// skipped here — `primal_feas_max`/`compute_qp_kkt_max` already cover the
/// linear-only rows via `Ax`.
///
/// Stationarity of the quadratic-constraint multipliers is out of scope:
/// this checks primal feasibility only, mirroring `primal_feas_max`'s role
/// for the linear rows.
///
/// Returns `0.0` for a pure QP/LP (`quadratic_constraints` empty), and
/// `f64::INFINITY` if `x` has the wrong length, non-finite entries, or the
/// problem's `quadratic_constraints` invariant is broken (defensive: the
/// field is public and callers can bypass `set_quadratic_constraints`).
pub fn qcqp_pfeas_max(prob: &QpProblem, x: &[f64]) -> f64 {
    if prob.quadratic_constraints.is_empty() {
        return 0.0;
    }
    if prob.validate().is_err() {
        return f64::INFINITY;
    }
    let n = prob.num_vars;
    if x.len() != n || x.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let ax = match prob.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let mut max_v = 0.0_f64;
    #[allow(unreachable_patterns)]
    for (k, qc) in prob.quadratic_constraints.iter().enumerate() {
        if qc.nnz() == 0 {
            continue;
        }
        let quad: f64 = 0.5
            * qc.triplets
                .iter()
                .map(|&(i, j, v)| v * x[i] * x[j])
                .sum::<f64>();
        let lhs = ax[k] + quad;
        let b_k = prob.b[k];
        if !lhs.is_finite() || !b_k.is_finite() {
            return f64::INFINITY;
        }
        let viol = match prob.constraint_types[k] {
            ConstraintType::Le => (lhs - b_k).max(0.0),
            ConstraintType::Ge => (b_k - lhs).max(0.0),
            ConstraintType::Eq => (lhs - b_k).abs(),
            _ => continue,
        };
        let scale = 1.0 + lhs.abs() + b_k.abs();
        max_v = max_v.max(viol / scale);
    }
    max_v
}

/// Whether a measured original-space residual disqualifies a claimed
/// solution: non-finite or `>= eps`. `None` (nothing was measurable — e.g.
/// no/short solution vector) never disqualifies; those arms carry their own
/// diagnostics instead.
pub fn is_kkt_violation(kkt_max: Option<f64>, eps: f64) -> bool {
    kkt_max.is_some_and(|v| !v.is_finite() || v >= eps)
}

/// Final bench verdict label after the KKT/QCQP-pfeas gate, shared by every
/// solution-claiming arm of `bench_qplib` (Optimal / SuboptimalSolution /
/// NonconvexLocal / NonconvexGlobal): a claimed solution whose measured
/// residual violates ([`is_kkt_violation`]) is a false positive whatever
/// label the solver put on it, so the tentative label demotes to `KKT_FAIL`.
/// Keeping the decision here (one pure function) stops the gate from being
/// re-implemented per match arm in the bench binary, where the
/// Nonconvex*/Suboptimal arms previously skipped it.
pub fn kkt_gated_label(tentative: &'static str, kkt_max: Option<f64>, eps: f64) -> &'static str {
    if is_kkt_violation(kkt_max, eps) {
        "KKT_FAIL"
    } else {
        tentative
    }
}

/// `|obj − global_ref| / (1 + |global_ref|)`. Returns `None` if either is non-finite.
pub fn compute_gap_to_global(obj: f64, global_ref: f64) -> Option<f64> {
    if !obj.is_finite() || !global_ref.is_finite() {
        return None;
    }
    Some((obj - global_ref).abs() / (1.0 + global_ref.abs()))
}

/// bench harness promotion policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchPromotionPolicy {
    QpsBenchmark,
    BenchQplib,
}

/// Promote `LocallyOptimal` with a full solution to `Optimal`.
///
/// `Timeout` is intentionally excluded: surfacing an honest Timeout is more
/// useful than silently promoting a partial incumbent.
/// `SuboptimalSolution` is also excluded because it is not a convergence proof;
/// treating it as `Optimal` creates false PASS results in benchmark harnesses.
pub fn apply_bench_status_promotion(
    result: SolverResult,
    num_vars: usize,
    policy: BenchPromotionPolicy,
) -> SolverResult {
    let eligible_status = matches!(result.status, SolveStatus::LocallyOptimal);
    let has_full_solution = !result.solution.is_empty() && result.solution.len() == num_vars;
    let obj_ok = match policy {
        BenchPromotionPolicy::QpsBenchmark => true,
        BenchPromotionPolicy::BenchQplib => result.objective.is_finite(),
    };
    if eligible_status && has_full_solution && obj_ok {
        let mut promoted = result;
        promoted.status = SolveStatus::Optimal;
        promoted
    } else {
        result
    }
}

/// Objective check result.
pub enum ObjCheckResult {
    Ok { rel_err: f64 },
    Mismatch { rel_err: f64 },
    NoRef,
}

/// Expected status for a benchmark problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

/// Load expected statuses from a baseline CSV.
///
/// Float entries → `Optimal`, `"INFEASIBLE"` → `Infeasible`, `"UNBOUNDED"` → `Unbounded`.
/// Other values (e.g., `"no_ref"`) are skipped.
pub fn load_expected_statuses(csv_path: &Path) -> HashMap<String, ExpectedStatus> {
    let mut map = HashMap::new();
    let content = match std::fs::read_to_string(csv_path) {
        Ok(c) => c,
        Err(_) => return map,
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let name = parts[0].trim().to_string();
        let status_str = parts[1].trim();
        let status = match status_str.to_uppercase().as_str() {
            "INFEASIBLE" => ExpectedStatus::Infeasible,
            "UNBOUNDED" => ExpectedStatus::Unbounded,
            _ => {
                if status_str.parse::<f64>().is_ok() {
                    ExpectedStatus::Optimal
                } else {
                    continue;
                }
            }
        };
        map.insert(name, status);
    }
    map
}

/// Determine baseline CSV path from the data directory name.
pub fn detect_csv_path(data_dir: &str, override_path: Option<&str>, root: &Path) -> PathBuf {
    if let Some(p) = override_path {
        return PathBuf::from(p);
    }
    let data_lower = data_dir.to_lowercase();
    let csv_name = if data_lower.contains("maros") {
        "maros_meszaros.csv"
    } else if data_lower.contains("qp_unbounded") || data_lower.contains("qp-unbounded") {
        "qp_unbounded.csv"
    } else if data_lower.contains("qp_infeasible") || data_lower.contains("qp-infeasible") {
        "qp_infeasible.csv"
    } else if data_lower.contains("qplib_nonconvex_official")
        || data_lower.contains("qplib-nonconvex-official")
    {
        "qplib_nonconvex_official.csv"
    } else if data_lower.contains("qplib_nonconvex") || data_lower.contains("qplib-nonconvex") {
        "qplib_nonconvex_synthetic.csv"
    } else if data_lower.contains("qplib") {
        "qplib.csv"
    } else if data_lower.contains("osqp_bench") || data_lower.contains("osqp-bench") {
        "osqp_bench.csv"
    } else if data_lower.contains("mpc_qp") || data_lower.contains("mpc-qp") {
        "mpc_qp.csv"
    } else if data_lower.contains("lp_problems_infeas") || data_lower.contains("lp-problems-infeas")
    {
        "netlib_lp_infeas.csv"
    } else if data_lower.contains("lp_problems_extra") || data_lower.contains("lp-problems-extra") {
        "netlib_lp_extra.csv"
    } else if data_lower.contains("lp_problems_unbounded")
        || data_lower.contains("lp-problems-unbounded")
    {
        "lp_problems_unbounded.csv"
    } else {
        "netlib_lp.csv"
    };
    let candidate = root.join("data/baseline_objectives").join(csv_name);
    if candidate.exists() {
        return candidate;
    }
    PathBuf::from("data/baseline_objectives").join(csv_name)
}

/// Path to the QCQP-specific QPLIB baseline (`qplib_qcqp.csv`).
///
/// `data/qplib` (and `data/qplib_unsupported`) physically mix CCQ/DCQ/QCQ
/// (quadratic-constraint) instances in among the DCL/QCL ones that
/// `detect_csv_path` resolves to `qplib.csv`; that file never lists the
/// QCQP IDs (`qplib_qcqp.csv` was added separately, PR #25), so a bare
/// `detect_csv_path` lookup leaves every QCQP-route result with no
/// reference to check against (`CHECKED[no_ref]`, an obj-regression blind
/// spot). Callers merge this file's entries into whatever `detect_csv_path`
/// resolved rather than replacing it: kept as a separate file (not merged
/// into `qplib.csv` on disk) because the two have different provenance —
/// `qplib.csv`'s own header states its values are self-measured, not
/// externally verified, while `qplib_qcqp.csv` carries officially published
/// QPLIB optimal values.
pub fn qplib_qcqp_csv_path(root: &Path) -> PathBuf {
    root.join("data/baseline_objectives/qplib_qcqp.csv")
}

/// A benchmark baseline: per-problem reference objectives paired with the
/// expected terminal status parsed from the same rows.
type Baselines = (HashMap<String, f64>, HashMap<String, ExpectedStatus>);

/// Baseline objectives/statuses for a `bench_qplib` run: `detect_csv_path`'s
/// resolution (respecting `override_path`, e.g. a `--known-optimal` CLI
/// flag) merged with `qplib_qcqp_csv_path`'s QCQP-specific entries.
///
/// The merge is unconditional -- it runs whether or not `override_path` was
/// given -- because `qplib_qcqp.csv` is a distinct file from anything
/// `detect_csv_path` could resolve to, so there is no double-counting risk,
/// and skipping the merge on an explicit override would silently reopen the
/// `CHECKED[no_ref]` gap for that invocation.
///
/// Panics (with the offending problem ID) if the two files share any problem
/// ID: the whole point of keeping `qplib_qcqp.csv` separate is that its
/// officially-published values must not be silently overwritten by — nor
/// silently overwrite — `qplib.csv`'s self-measured ones. A shared ID is a
/// data-authoring error that must fail loudly rather than resolve by
/// insertion order.
pub fn load_qplib_baselines(data_dir: &str, override_path: Option<&str>, root: &Path) -> Baselines {
    let csv = detect_csv_path(data_dir, override_path, root);
    let base_obj = load_baseline_objectives(&csv).unwrap_or_default();
    let base_stat = load_expected_statuses(&csv);
    let qcqp_csv = qplib_qcqp_csv_path(root);
    let extra_obj = load_baseline_objectives(&qcqp_csv).unwrap_or_default();
    let extra_stat = load_expected_statuses(&qcqp_csv);
    merge_qplib_baselines(base_obj, base_stat, extra_obj, extra_stat).unwrap_or_else(|dup| {
        panic!(
            "qplib.csv ({}) and qplib_qcqp.csv ({}) share problem ID(s) [{}]; \
             the QCQP baseline is kept separate precisely so its published values \
             neither overwrite nor are overwritten by qplib.csv's self-measured ones — \
             remove the duplicate from one file",
            csv.display(),
            qcqp_csv.display(),
            dup.join(", "),
        )
    })
}

/// Merge `extra` baseline objectives/statuses into `base`, erroring with the
/// sorted list of shared problem IDs if the two carry any ID in common.
///
/// The check spans both the objective map (numeric rows) and the status map
/// (`INFEASIBLE`/`UNBOUNDED` sentinel rows), so a collision on either surface
/// is caught. On success the merged `(objectives, statuses)` pair is returned;
/// on collision no merge is performed.
fn merge_qplib_baselines(
    mut base_obj: HashMap<String, f64>,
    mut base_stat: HashMap<String, ExpectedStatus>,
    extra_obj: HashMap<String, f64>,
    extra_stat: HashMap<String, ExpectedStatus>,
) -> Result<Baselines, Vec<String>> {
    let mut dups: Vec<String> = extra_obj
        .keys()
        .filter(|k| base_obj.contains_key(*k))
        .chain(extra_stat.keys().filter(|k| base_stat.contains_key(*k)))
        .cloned()
        .collect();
    if !dups.is_empty() {
        dups.sort();
        dups.dedup();
        return Err(dups);
    }
    base_obj.extend(extra_obj);
    base_stat.extend(extra_stat);
    Ok((base_obj, base_stat))
}

/// Load objective baseline CSV.
///
/// Returns an error when the file cannot be read. Callers that need a
/// best-effort empty map should use `.unwrap_or_default()`; callers that
/// require the file to exist should use `.expect(...)`.
pub fn load_baseline_objectives(csv_path: &Path) -> Result<HashMap<String, f64>, std::io::Error> {
    let mut map = HashMap::new();
    let content = std::fs::read_to_string(csv_path)?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                map.insert(parts[0].trim().to_string(), val);
            }
        }
    }
    Ok(map)
}

/// Check solver objective against a baseline CSV.
///
/// `obj_offset` is added to the CSV reference before comparison. Pass `0.0` for
/// most problem sets; pass `prob.obj_offset` for Netlib LP (which stores shifted
/// objectives in the CSV).
pub fn check_baseline_objective(
    problem_name: &str,
    solver_obj: f64,
    known: &HashMap<String, f64>,
    eps_obj: f64,
    obj_offset: f64,
) -> ObjCheckResult {
    match known.get(problem_name) {
        Some(&known_obj) => {
            if !solver_obj.is_finite() {
                return ObjCheckResult::Mismatch {
                    rel_err: f64::INFINITY,
                };
            }
            let expected = known_obj + obj_offset;
            let denom = expected.abs().max(1.0);
            let rel_err = (solver_obj - expected).abs() / denom;
            if rel_err > eps_obj {
                ObjCheckResult::Mismatch { rel_err }
            } else {
                ObjCheckResult::Ok { rel_err }
            }
        }
        None => ObjCheckResult::NoRef,
    }
}

/// Result of parsing a QPLIB file for QP benchmarking.
pub enum ParseQplibOutcome {
    /// Continuous QP problem ready to solve.
    Qp(Box<QpProblem>),
    /// Problem type not in scope (MIP / quadratic constraints).
    Unsupported(String),
    /// Parse failure.
    ParseError(String),
}

/// Parse a QPLIB file and classify the result as [`ParseQplibOutcome`].
///
/// MIP (MILP/MIQP) and quadratically-constrained problems are returned as
/// [`ParseQplibOutcome::Unsupported`]. Parse failures map to
/// [`ParseQplibOutcome::ParseError`].
///
/// External process-level timeout (e.g. `gtimeout` in `bench_parallel.sh`)
/// handles runaway parses; no Rust-level timeout is applied.
pub fn parse_qplib_outcome(path: &Path) -> ParseQplibOutcome {
    match parse_qplib(path) {
        Ok(QplibProblem::Qp(prob)) => ParseQplibOutcome::Qp(Box::new(prob)),
        Ok(QplibProblem::Milp(_)) | Ok(QplibProblem::Miqp(_)) => ParseQplibOutcome::Unsupported(
            "Variable type 'B'/'I' (binary/integer): MIP problem".to_string(),
        ),
        Err(QplibError::UnsupportedType(msg)) => ParseQplibOutcome::Unsupported(msg),
        Err(e) => ParseQplibOutcome::ParseError(e.to_string()),
    }
}

/// Per-component normalised primal feasibility.
///
/// `max_i violation_i / (1 + |Ax_i| + |b_i|)`.
/// Unlike the global-relative OSQP formula this detects a single
/// badly-violated row even when the overall scale is large.
pub fn compute_pfeas_normalized(prob: &QpProblem, solution: &[f64]) -> f64 {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    // Guard: if A.ncols ≠ num_vars the matrix is internally inconsistent; treat as NaN
    // (QpProblem construction should prevent this, but preserve old mat_vec_mul-Err semantics).
    if prob.a.ncols() != prob.num_vars {
        return f64::NAN;
    }
    f64_impl::primal_residual_rel(&prob.a, &prob.b, &prob.constraint_types, solution)
}

/// Dual feasibility in the original (unscaled) space.
///
/// Returns `(dfeas_abs, dfeas_rel_componentwise)`.
///
/// * `dfeas_abs` = `||Qx + A^T y + bound_contrib + c||_∞`
/// * `dfeas_rel` = component-wise relative version
///
/// For the LP/Simplex path (`bound_duals` empty, `reduced_costs` non-empty)
/// a complementarity-aware bound-hit check is used instead of the full
/// stationarity residual.
pub fn compute_dfeas_orig(
    prob: &QpProblem,
    solution: &[f64],
    dual_solution: &[f64],
    bound_duals: &[f64],
    reduced_costs: &[f64],
) -> (f64, f64) {
    use twofloat::TwoFloat;
    if solution.is_empty()
        || solution.len() != prob.num_vars
        || solution.iter().any(|v| !v.is_finite())
        || dual_solution.iter().any(|v| !v.is_finite())
        || bound_duals.iter().any(|v| !v.is_finite())
        || reduced_costs.iter().any(|v| !v.is_finite())
    {
        return (f64::NAN, f64::NAN);
    }
    let n_finite_bounds = prob
        .bounds
        .iter()
        .map(|&(lb, ub)| usize::from(lb.is_finite()) + usize::from(ub.is_finite()))
        .sum::<usize>();
    if !bound_duals.is_empty() && bound_duals.len() != n_finite_bounds {
        return (f64::INFINITY, f64::INFINITY);
    }
    if !reduced_costs.is_empty() && reduced_costs.len() != prob.num_vars {
        return (f64::NAN, f64::NAN);
    }
    let n = solution.len();

    // LP/Simplex path: complementarity-aware sign check on reduced costs. This
    // path is a pure function of bounds, solution, reduced_costs and c and never
    // touches `dual_solution` / A^T·y, so it is evaluated BEFORE `aty` — a short
    // or mismatched dual must yield a finite LP residual here, not an `aty` index
    // panic. The dual-length guard therefore belongs only to the KKT path below.
    if bound_duals.is_empty() && !reduced_costs.is_empty() && reduced_costs.len() == n {
        let rel_tol = PIVOT_TOL;
        let mut dfeas_abs = 0.0_f64;
        let mut dfeas_rel = 0.0_f64;
        for j in 0..n {
            let (lb_j, ub_j) = prob.bounds[j];
            if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < ZERO_TOL {
                continue;
            }
            let rc = reduced_costs[j];
            let x_j = solution[j];
            let at_lb =
                lb_j.is_finite() && (x_j - lb_j).abs() <= rel_tol * (1.0 + x_j.abs() + lb_j.abs());
            let at_ub =
                ub_j.is_finite() && (x_j - ub_j).abs() <= rel_tol * (1.0 + x_j.abs() + ub_j.abs());
            // Interior (away from any finite bound) ⇒ basic in simplex, where rc
            // is ~0 by construction; any reported magnitude is basis noise, so it
            // is not a dual-feasibility violation. Only at-bound (nonbasic) vars
            // are sign-checked above.
            let viol = if at_lb && !at_ub {
                f64::max(0.0, -rc)
            } else if at_ub && !at_lb {
                f64::max(0.0, rc)
            } else {
                0.0
            };
            dfeas_abs = dfeas_abs.max(viol);
            let scale_j = 1.0 + rc.abs() + prob.c[j].abs();
            dfeas_rel = dfeas_rel.max(viol / scale_j);
        }
        return (dfeas_abs, dfeas_rel);
    }

    // KKT path: the residual is built from A^T·y, so the dual must match
    // num_constraints — otherwise `aty` would index `y` out of bounds.
    if prob.num_constraints > 0 && dual_solution.len() != prob.num_constraints {
        return (f64::INFINITY, f64::INFINITY);
    }
    let qx: Vec<f64> = dd_impl::qx(&prob.q, solution)
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let aty: Vec<f64> = dd_impl::aty(&prob.a, dual_solution, n)
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let bound_contrib = kkt_resid::bound_contrib(&prob.bounds, bound_duals);
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel_componentwise = 0.0_f64;
    for i in 0..n {
        let (lb_i, ub_i) = prob.bounds[i];
        if lb_i.is_finite() && ub_i.is_finite() && (lb_i - ub_i).abs() < ZERO_TOL {
            continue;
        }
        if prob.a.col_ptr().len() > i + 1 && prob.a.col_ptr()[i + 1] - prob.a.col_ptr()[i] == 0 {
            continue;
        }
        let r_dd = TwoFloat::from(qx[i])
            + TwoFloat::from(aty[i])
            + TwoFloat::from(bound_contrib[i])
            + TwoFloat::from(prob.c[i]);
        let r = f64::from(r_dd).abs();
        dfeas_abs = dfeas_abs.max(r);
        let scale_i = 1.0 + qx[i].abs() + aty[i].abs() + bound_contrib[i].abs() + prob.c[i].abs();
        dfeas_rel_componentwise = dfeas_rel_componentwise.max(r / scale_i);
    }
    (dfeas_abs, dfeas_rel_componentwise)
}

/// Component-wise dual feasibility: second return value of [`compute_dfeas_orig`].
pub fn compute_dfeas_componentwise(
    prob: &QpProblem,
    solution: &[f64],
    dual_solution: &[f64],
    bound_duals: &[f64],
    reduced_costs: &[f64],
) -> f64 {
    compute_dfeas_orig(prob, solution, dual_solution, bound_duals, reduced_costs).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp_csv(content: &str) -> std::path::PathBuf {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        tmp.write_all(content.as_bytes()).unwrap();
        let path = tmp.path().to_path_buf();
        Box::leak(Box::new(tmp));
        path
    }

    #[test]
    fn test_load_expected_statuses_infeasible() {
        let csv = "problem_name,optimal_obj,source\n\
            galenet,INFEASIBLE,https://example.com\n\
            klein1,INFEASIBLE,https://example.com\n\
            unbnd_toy,UNBOUNDED,https://example.com\n\
            feasible_toy,1234.5,https://example.com\n\
            noref_toy,no_ref,https://example.com\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);

        assert_eq!(map.get("galenet"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("klein1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("unbnd_toy"), Some(&ExpectedStatus::Unbounded));
        assert_eq!(map.get("feasible_toy"), Some(&ExpectedStatus::Optimal));
        assert_eq!(map.get("noref_toy"), None);
    }

    #[test]
    fn test_load_expected_statuses_case_insensitive() {
        let csv = "problem_name,optimal_obj\n\
            p1,infeasible\n\
            p2,INFEASIBLE\n\
            p3,Infeasible\n\
            p4,unbounded\n\
            p5,UNBOUNDED\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);

        assert_eq!(map.get("p1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p2"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p3"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p4"), Some(&ExpectedStatus::Unbounded));
        assert_eq!(map.get("p5"), Some(&ExpectedStatus::Unbounded));
    }

    #[test]
    fn test_load_expected_statuses_skips_comments_and_header() {
        let csv = "# comment line\n\
            problem_name,optimal_obj\n\
            p1,INFEASIBLE\n\
            # another comment\n\
            p2,1.5\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("p1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p2"), Some(&ExpectedStatus::Optimal));
    }

    #[test]
    fn test_detect_csv_path_infeas() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_infeas", None, root);
        assert!(
            p.to_string_lossy().contains("netlib_lp_infeas.csv"),
            "Expected netlib_lp_infeas.csv, got {p:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_extra() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_extra", None, root);
        assert!(
            p.to_string_lossy().contains("netlib_lp_extra.csv"),
            "Expected netlib_lp_extra.csv, got {p:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_default_netlib() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems", None, root);
        assert!(
            p.to_string_lossy().contains("netlib_lp.csv"),
            "Expected netlib_lp.csv, got {p:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_qplib_nonconvex_synthetic() {
        let root = std::path::Path::new("/solver");
        // qplib_nonconvex (no _official suffix) → synthetic CSV
        let p = detect_csv_path("/data/qplib_nonconvex", None, root);
        assert!(
            p.to_string_lossy()
                .contains("qplib_nonconvex_synthetic.csv"),
            "Expected qplib_nonconvex_synthetic.csv, got {p:?}"
        );
        // Dash variant
        let p2 = detect_csv_path("/data/qplib-nonconvex", None, root);
        assert!(
            p2.to_string_lossy()
                .contains("qplib_nonconvex_synthetic.csv"),
            "Expected qplib_nonconvex_synthetic.csv (dash), got {p2:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_qplib_nonconvex_official_not_shadowed() {
        let root = std::path::Path::new("/solver");
        // qplib_nonconvex_official must still map to official CSV
        let p = detect_csv_path("/data/qplib_nonconvex_official", None, root);
        assert!(
            p.to_string_lossy().contains("qplib_nonconvex_official.csv"),
            "Expected qplib_nonconvex_official.csv, got {p:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_lp_problems_unbounded() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_unbounded", None, root);
        assert!(
            p.to_string_lossy().contains("lp_problems_unbounded.csv"),
            "Expected lp_problems_unbounded.csv, got {p:?}"
        );
        // Dash variant
        let p2 = detect_csv_path("/data/lp-problems-unbounded", None, root);
        assert!(
            p2.to_string_lossy().contains("lp_problems_unbounded.csv"),
            "Expected lp_problems_unbounded.csv (dash), got {p2:?}"
        );
    }

    #[test]
    fn test_detect_csv_path_lp_problems_unbounded_not_shadowed_by_infeas_extra() {
        let root = std::path::Path::new("/solver");
        // Unbounded must not fall through to infeas/extra/default
        let p_infeas = detect_csv_path("/data/lp_problems_infeas", None, root);
        assert!(
            p_infeas.to_string_lossy().contains("netlib_lp_infeas.csv"),
            "Expected netlib_lp_infeas.csv, got {p_infeas:?}"
        );
        let p_extra = detect_csv_path("/data/lp_problems_extra", None, root);
        assert!(
            p_extra.to_string_lossy().contains("netlib_lp_extra.csv"),
            "Expected netlib_lp_extra.csv, got {p_extra:?}"
        );
        let p_default = detect_csv_path("/data/lp_problems", None, root);
        assert!(
            p_default.to_string_lossy().contains("netlib_lp.csv"),
            "Expected netlib_lp.csv, got {p_default:?}"
        );
    }

    fn make_single_var_prob(bounds: Vec<(f64, f64)>, c: Vec<f64>, x_target: f64) -> QpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut p = QpProblem::new(
            CscMatrix::new(1, 1),
            c,
            a,
            vec![x_target],
            bounds,
            vec![ConstraintType::Eq],
        )
        .unwrap();
        p.obj_offset = 0.0;
        p
    }

    /// Relative bound-hit detection in `compute_dfeas_orig` is scale-invariant.
    /// x~1e6 (large) and x~1e-9 (tiny) are both correctly classified.
    #[test]
    fn test_dfeas_bound_hit_relative_structural() {
        // A: lb=0, ub=inf, x=0 (at lb), rc=+1 → feasible (z_lb=1≥0)
        let p = make_single_var_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 0.0);
        let (abs, _) = compute_dfeas_orig(&p, &[0.0], &[], &[], &[1.0]);
        assert!(abs < 1e-15, "at lb rc=+1 (ok): dfeas={}", abs);

        // B: lb=0, ub=inf, x=0 (at lb), rc=-1 → infeasible (z_lb=-1<0)
        let (abs, _) = compute_dfeas_orig(&p, &[0.0], &[], &[], &[-1.0]);
        assert!(
            (abs - 1.0).abs() < 1e-15,
            "at lb rc=-1 (bad): dfeas={}",
            abs
        );

        // C: x=100 (interior), rc=-1 → noise-tolerant (basis variable)
        let p = make_single_var_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 100.0);
        let (abs, _) = compute_dfeas_orig(&p, &[100.0], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "interior rc=-1 (basis noise): dfeas={}", abs);

        // D: x=10 at ub=10, rc=+1 → infeasible (z_ub=-1<0)
        let p = make_single_var_prob(vec![(0.0, 10.0)], vec![1.0], 10.0);
        let (abs, _) = compute_dfeas_orig(&p, &[10.0], &[], &[], &[1.0]);
        assert!(
            (abs - 1.0).abs() < 1e-15,
            "at ub rc=+1 (bad): dfeas={}",
            abs
        );

        // E: x=10 at ub=10, rc=-1 → feasible (z_ub=1≥0)
        let (abs, _) = compute_dfeas_orig(&p, &[10.0], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "at ub rc=-1 (ok): dfeas={}", abs);

        // F: free variable, rc=0.5 → Simplex basis noise tolerance
        let p = make_single_var_prob(vec![(f64::NEG_INFINITY, f64::INFINITY)], vec![1.0], 5.0);
        let (abs, _) = compute_dfeas_orig(&p, &[5.0], &[], &[], &[0.5]);
        assert!(abs < 1e-15, "free rc=0.5 (basis noise): dfeas={}", abs);

        // G: x=1e6 interior (large scale), rc=-1 → noise-tolerant
        let p = make_single_var_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 1e6);
        let (abs, _) = compute_dfeas_orig(&p, &[1e6], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "large-scale interior rc=-1: dfeas={}", abs);

        // H: x=1e-9 (tiny, relatively at lb), rc=-1 → infeasible
        let (abs, _) = compute_dfeas_orig(&p, &[1e-9], &[], &[], &[-1.0]);
        assert!(
            (abs - 1.0).abs() < 1e-15,
            "tiny x at lb, rc=-1: dfeas={}",
            abs
        );

        // I: x=1e-5 (interior), rc=-1 → noise-tolerant
        let (abs, _) = compute_dfeas_orig(&p, &[1e-5], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "x=1e-5 interior rc=-1: dfeas={}", abs);
    }

    #[test]
    fn test_pfeas_normalized_eq_both_directions() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            vec![5.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        // x=3: |3-5|=2, scale=1+3+5=9 → 2/9
        let v = compute_pfeas_normalized(&prob, &[3.0]);
        assert!((v - 2.0 / 9.0).abs() < 1e-12, "eq down: got {}", v);
        // x=5: 0
        let v = compute_pfeas_normalized(&prob, &[5.0]);
        assert!(v < 1e-12, "eq sat: got {}", v);
    }

    #[test]
    fn test_pfeas_normalized_ge() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            vec![5.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        // x=3: b-ax=2, scale=1+3+5=9 → 2/9
        let v = compute_pfeas_normalized(&prob, &[3.0]);
        assert!((v - 2.0 / 9.0).abs() < 1e-12, "ge viol: got {}", v);
        // x=7: 0
        let v = compute_pfeas_normalized(&prob, &[7.0]);
        assert!(v < 1e-12, "ge sat: got {}", v);
    }

    /// Sentinel: `compute_dfeas_componentwise` must equal `compute_dfeas_orig` second value.
    ///
    /// Three cases:
    /// - Case A (spec): interior var → 0 by spec; verifies `assert_eq!` against orig.
    ///   A `return 0.0` no-op would pass this alone (0==0 tautology).
    /// - Case B (no-op sentinel): at-lb var with rc=-0.5 → orig returns ~0.2;
    ///   a `return 0.0` no-op fails `assert_eq!(0.0, 0.2)`.
    /// - Case C (at-ub): at-ub var with rc=+0.5 → violation; verifies ub path.
    #[test]
    fn test_dfeas_componentwise_matches_orig_lp_path() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a.clone(),
            vec![5.0],
            vec![(2.0, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();

        // Case A: interior (x=5, lb=2, ub=∞). Old buggy code: false positive 0.5; new: 0.
        // Note: `return 0.0` no-op also passes here (0==0). Case B is the actual sentinel.
        let dfeas_c = compute_dfeas_componentwise(&prob, &[5.0], &[], &[], &[-0.5]);
        let (_, dfeas_rel) = compute_dfeas_orig(&prob, &[5.0], &[], &[], &[-0.5]);
        assert_eq!(dfeas_c, dfeas_rel, "case A: componentwise must equal orig");
        assert!(
            dfeas_c < 1e-15,
            "case A: interior var should be 0, got {dfeas_c}"
        );

        // Case B (no-op sentinel): x=2 at lb=2, rc=-0.5 → orig returns > 0.
        // A `return 0.0` no-op fails assert_eq! here.
        let dfeas_c = compute_dfeas_componentwise(&prob, &[2.0], &[], &[], &[-0.5]);
        let (_, dfeas_rel) = compute_dfeas_orig(&prob, &[2.0], &[], &[], &[-0.5]);
        assert_eq!(
            dfeas_c, dfeas_rel,
            "case B: at-lb componentwise must equal orig"
        );
        assert!(dfeas_c > 0.0, "case B: at lb, rc<0 → violation > 0");

        // Case C: at-ub (x=10, lb=0, ub=10, rc=+0.5 → violation at ub).
        let prob_ub = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            vec![5.0],
            vec![(0.0, 10.0)],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let dfeas_c = compute_dfeas_componentwise(&prob_ub, &[10.0], &[], &[], &[0.5]);
        let (_, dfeas_rel) = compute_dfeas_orig(&prob_ub, &[10.0], &[], &[], &[0.5]);
        assert_eq!(
            dfeas_c, dfeas_rel,
            "case C: at-ub componentwise must equal orig"
        );
        assert!(dfeas_c > 0.0, "case C: at ub, rc>0 → violation > 0");
    }

    /// The `dual_solution.len() != num_constraints` guard must stay scoped to the
    /// KKT path: it still rejects a malformed dual there (INF), but must NOT
    /// reject a valid simplex output on the reduced-cost path (empty dual).
    ///
    /// No-op sentinels: deleting the guard makes the KKT call return a finite
    /// residual (1st assert fails); dropping the `!lp_reduced_cost_path` scope
    /// makes the LP call return INF (2nd assert fails).
    #[test]
    fn test_dfeas_dual_length_guard_scoped_to_kkt_path() {
        let p = make_single_var_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 5.0);

        // KKT path (bound_duals non-empty) with dual_solution.len()=0 ≠ 1 → INF.
        let (abs, rel) = compute_dfeas_orig(&p, &[5.0], &[], &[0.5], &[]);
        assert!(
            abs.is_infinite() && rel.is_infinite(),
            "KKT path with mismatched dual must be INF, got ({abs}, {rel})"
        );

        // Reduced-cost path (reduced_costs given, bound_duals empty) with the same
        // empty dual must be accepted (finite), not rejected.
        let (abs_lp, _) = compute_dfeas_orig(&p, &[5.0], &[], &[], &[-1.0]);
        assert!(
            abs_lp.is_finite(),
            "LP reduced-cost path must accept an empty dual, got {abs_lp}"
        );
    }

    /// Regression (codex P2): the LP reduced-cost path is evaluated *before*
    /// `aty`, so a short non-empty `dual_solution` is ignored (finite result),
    /// never indexed out of bounds. Here num_constraints=2 with a length-1 dual:
    /// if `aty` ran before the LP return it would index `y[1]` and panic.
    ///
    /// No-op sentinel: moving `aty` back ahead of the LP path makes this panic.
    #[test]
    fn test_dfeas_lp_path_ignores_short_dual_no_panic() {
        // 2 vars, 2 Eq constraints (A = I₂); reduced_costs given ⇒ LP path.
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let prob = QpProblem::new(
            CscMatrix::new(2, 2),
            vec![1.0, 1.0],
            a,
            vec![3.0, 4.0],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            vec![ConstraintType::Eq, ConstraintType::Eq],
        )
        .unwrap();
        // dual_solution.len()=1 < num_constraints=2: would panic `aty` at y[1].
        let (abs, rel) = compute_dfeas_orig(&prob, &[3.0, 4.0], &[0.5], &[], &[1.0, 1.0]);
        assert!(
            abs.is_finite() && rel.is_finite(),
            "LP path must ignore a short dual and return finite, got ({abs}, {rel})"
        );
    }

    /// Real workspace root, exercising the actual checked-in
    /// `data/baseline_objectives/qplib.csv` + `qplib_qcqp.csv` (no synthetic
    /// stand-ins): `load_qplib_baselines` must return both DCL/QCL problems
    /// from `qplib.csv` *and* the CCQ/DCQ/QCQ problems that live only in
    /// `qplib_qcqp.csv` (PR #25 review "Wire QCQP baselines into benchmark
    /// selection").
    ///
    /// `QPLIB_2546` (CCQ, `data/qplib`) is the finding's own example; its
    /// expected objective is read directly from `qplib_qcqp.csv` here (an
    /// independent read, not the production loader) and compared literally.
    ///
    /// Sentinel: reverting `load_qplib_baselines` to a bare `detect_csv_path`
    /// lookup (dropping the `qplib_qcqp_csv_path` merge) makes this FAIL --
    /// `QPLIB_2546` would be absent from the returned map.
    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    #[test]
    fn test_load_qplib_baselines_merges_qcqp_csv() {
        let root = workspace_root();
        let (objectives, _statuses) = load_qplib_baselines("data/qplib", None, &root);

        // Present only in qplib_qcqp.csv (not qplib.csv, confirmed separately
        // by test_qplib_csv_and_qcqp_csv_have_disjoint_ids below).
        let qcqp_csv = qplib_qcqp_csv_path(&root);
        let qcqp_only = load_baseline_objectives(&qcqp_csv).expect("qplib_qcqp.csv must parse");
        assert!(
            qcqp_only.contains_key("QPLIB_2546"),
            "test premise: QPLIB_2546 must be in qplib_qcqp.csv"
        );
        assert_eq!(
            objectives.get("QPLIB_2546"),
            qcqp_only.get("QPLIB_2546"),
            "load_qplib_baselines must carry QPLIB_2546's objective through \
             from qplib_qcqp.csv: {objectives:?}"
        );

        // Present only in qplib.csv (DCL problem, no quadratic constraints).
        let qplib_csv = super::detect_csv_path("data/qplib", None, &root);
        let qplib_only = load_baseline_objectives(&qplib_csv).expect("qplib.csv must parse");
        assert!(
            qplib_only.contains_key("QPLIB_10034"),
            "test premise: QPLIB_10034 must be in qplib.csv"
        );
        assert_eq!(
            objectives.get("QPLIB_10034"),
            qplib_only.get("QPLIB_10034"),
            "load_qplib_baselines must still carry qplib.csv's own entries: {objectives:?}"
        );
    }

    /// Companion fact-check for the sentinel above: `qplib.csv` and
    /// `qplib_qcqp.csv` must not share *any* problem ID (not just
    /// `QPLIB_2546`), across both the objective and the status surface —
    /// a shared ID would make the merge ambiguous (whose value wins?) and
    /// is exactly what `merge_qplib_baselines` now rejects. Checked against
    /// the real checked-in CSVs so a future data edit that introduces an
    /// overlap trips here.
    #[test]
    fn test_qplib_csv_and_qcqp_csv_have_disjoint_ids() {
        let root = workspace_root();
        let qplib_csv = super::detect_csv_path("data/qplib", None, &root);
        let qcqp_csv = qplib_qcqp_csv_path(&root);

        let base_obj = load_baseline_objectives(&qplib_csv).expect("qplib.csv must parse");
        let base_stat = load_expected_statuses(&qplib_csv);
        let extra_obj = load_baseline_objectives(&qcqp_csv).expect("qplib_qcqp.csv must parse");
        let extra_stat = load_expected_statuses(&qcqp_csv);

        // Union of IDs on each side, so an overlap on either surface counts.
        let base_ids: std::collections::HashSet<&String> =
            base_obj.keys().chain(base_stat.keys()).collect();
        let shared: Vec<&String> = extra_obj
            .keys()
            .chain(extra_stat.keys())
            .filter(|k| base_ids.contains(*k))
            .collect();
        assert!(
            shared.is_empty(),
            "qplib.csv and qplib_qcqp.csv must have disjoint problem IDs, shared: {shared:?}"
        );
    }

    /// Sentinel for `merge_qplib_baselines`'s collision guard: a synthetic
    /// base + extra that share a problem ID must error with that ID, not
    /// merge by insertion order.
    ///
    /// Reverting the guard (letting `base.extend(extra)` run unconditionally)
    /// makes this FAIL — the call would return `Ok` and silently keep the
    /// extra value.
    #[test]
    fn test_merge_qplib_baselines_rejects_shared_id() {
        let mut base_obj = HashMap::new();
        base_obj.insert("QPLIB_SHARED".to_string(), 1.0);
        base_obj.insert("QPLIB_ONLY_BASE".to_string(), 2.0);
        let base_stat: HashMap<String, ExpectedStatus> = base_obj
            .keys()
            .map(|k| (k.clone(), ExpectedStatus::Optimal))
            .collect();

        let mut extra_obj = HashMap::new();
        extra_obj.insert("QPLIB_SHARED".to_string(), 99.0); // collides with base
        extra_obj.insert("QPLIB_ONLY_EXTRA".to_string(), 3.0);
        let extra_stat: HashMap<String, ExpectedStatus> = extra_obj
            .keys()
            .map(|k| (k.clone(), ExpectedStatus::Optimal))
            .collect();

        let err = super::merge_qplib_baselines(base_obj, base_stat, extra_obj, extra_stat)
            .expect_err("shared problem ID must be rejected, not merged");
        assert_eq!(
            err,
            vec!["QPLIB_SHARED".to_string()],
            "collision list must name exactly the shared ID"
        );
    }

    /// Companion: a disjoint synthetic base + extra must merge cleanly,
    /// carrying every ID from both sides through.
    #[test]
    fn test_merge_qplib_baselines_merges_disjoint() {
        let mut base_obj = HashMap::new();
        base_obj.insert("QPLIB_ONLY_BASE".to_string(), 2.0);
        let base_stat: HashMap<String, ExpectedStatus> =
            [("QPLIB_ONLY_BASE".to_string(), ExpectedStatus::Optimal)].into();

        let mut extra_obj = HashMap::new();
        extra_obj.insert("QPLIB_ONLY_EXTRA".to_string(), 3.0);
        let extra_stat: HashMap<String, ExpectedStatus> =
            [("QPLIB_ONLY_EXTRA".to_string(), ExpectedStatus::Optimal)].into();

        let (obj, stat) = super::merge_qplib_baselines(base_obj, base_stat, extra_obj, extra_stat)
            .expect("disjoint inputs must merge");
        assert_eq!(obj.get("QPLIB_ONLY_BASE"), Some(&2.0));
        assert_eq!(obj.get("QPLIB_ONLY_EXTRA"), Some(&3.0));
        assert_eq!(stat.len(), 2);
    }

    /// `--known-optimal` override path must still get the qcqp merge (not
    /// just the auto-detected default): `bench_parallel.sh` always passes an
    /// explicit override for `data/qplib`, so the merge cannot be
    /// conditional on `override_path.is_none()`.
    #[test]
    fn test_load_qplib_baselines_merges_qcqp_csv_with_override() {
        let root = workspace_root();
        let override_csv = super::detect_csv_path("data/qplib", None, &root);
        let (objectives, _) = load_qplib_baselines(
            "data/qplib",
            Some(override_csv.to_str().expect("utf8 path")),
            &root,
        );
        assert!(
            objectives.contains_key("QPLIB_2546"),
            "override path must still merge in qplib_qcqp.csv: {objectives:?}"
        );
    }

    // --- qcqp_pfeas_max ---

    use otspot_core::qp::QcqpMatrix;

    /// `1/2 x'Qx` diagonal `diag(2,2)` gives `x'Qx = x1^2 + x2^2` exactly —
    /// the independent hand-computed oracle used by every case below.
    fn make_qcqp(ct: ConstraintType, b: f64) -> QpProblem {
        let q = CscMatrix::new(2, 2);
        let a = CscMatrix::new(1, 2);
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let mut prob = QpProblem::new(q, vec![0.0, 0.0], a, vec![b], bounds, vec![ct]).unwrap();
        let mut qc = QcqpMatrix::new(2);
        qc.triplets = vec![(0, 0, 2.0), (1, 1, 2.0)];
        prob.set_quadratic_constraints(vec![qc]).unwrap();
        prob
    }

    #[test]
    fn test_qcqp_pfeas_max_le_violation_and_feasible() {
        // x1^2 + x2^2 <= 1.
        let prob = make_qcqp(ConstraintType::Le, 1.0);
        assert!(
            qcqp_pfeas_max(&prob, &[0.0, 0.0]) < 1e-9,
            "origin satisfies x1^2+x2^2<=1"
        );
        // Oracle: quad=0.5*(2*4+2*4)=8, lhs=8, viol=(8-1)=7, scale=1+8+1=10 -> 0.7.
        let v = qcqp_pfeas_max(&prob, &[2.0, 2.0]);
        assert!((v - 0.7).abs() < 1e-9, "expected 0.7, got {v}");
    }

    #[test]
    fn test_qcqp_pfeas_max_ge_violation_and_feasible() {
        // x1^2 + x2^2 >= 1.
        let prob = make_qcqp(ConstraintType::Ge, 1.0);
        assert!(
            qcqp_pfeas_max(&prob, &[2.0, 0.0]) < 1e-9,
            "quad=4 >= 1 is feasible"
        );
        // Oracle: quad=0, lhs=0, viol=(1-0)=1, scale=1+0+1=2 -> 0.5.
        let v = qcqp_pfeas_max(&prob, &[0.0, 0.0]);
        assert!((v - 0.5).abs() < 1e-9, "expected 0.5, got {v}");
    }

    #[test]
    fn test_qcqp_pfeas_max_eq_violation_and_feasible() {
        // x1^2 + x2^2 == 1.
        let prob = make_qcqp(ConstraintType::Eq, 1.0);
        assert!(
            qcqp_pfeas_max(&prob, &[1.0, 0.0]) < 1e-9,
            "quad=1 satisfies the equality exactly"
        );
        // Oracle: quad=0, lhs=0, viol=|0-1|=1, scale=1+0+1=2 -> 0.5.
        let v = qcqp_pfeas_max(&prob, &[0.0, 0.0]);
        assert!((v - 0.5).abs() < 1e-9, "expected 0.5, got {v}");
    }

    #[test]
    fn test_qcqp_pfeas_max_pure_qp_is_zero() {
        let q = CscMatrix::new(2, 2);
        let a = CscMatrix::new(1, 2);
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![1.0],
            bounds,
            vec![ConstraintType::Le],
        )
        .unwrap();
        assert_eq!(qcqp_pfeas_max(&prob, &[1e6, -1e6]), 0.0);
    }

    /// `quadratic_constraints` is a public field (bypass hazard shared with
    /// `conic::qcqp`/`qp::qcqp_route`/`mip::problem`, see `QpProblem::validate`
    /// doc): a length mismatch from direct assignment must degrade to
    /// `INFINITY`, not index-panic.
    #[test]
    fn test_qcqp_pfeas_max_rejects_bypassed_length_mismatch() {
        let mut prob = make_qcqp(ConstraintType::Le, 1.0);
        let extra = QcqpMatrix::new(2);
        prob.quadratic_constraints.push(extra); // len=2 but num_constraints=1
        assert_eq!(qcqp_pfeas_max(&prob, &[0.0, 0.0]), f64::INFINITY);
    }

    // --- KKT gate (is_kkt_violation / kkt_gated_label) ---

    #[test]
    fn test_is_kkt_violation_thresholds() {
        assert!(!is_kkt_violation(None, 1e-4));
        assert!(!is_kkt_violation(Some(9.9e-5), 1e-4));
        assert!(is_kkt_violation(Some(1e-4), 1e-4), "boundary is >= eps");
        assert!(is_kkt_violation(Some(1e-3), 1e-4));
        assert!(is_kkt_violation(Some(f64::NAN), 1e-4));
        assert!(is_kkt_violation(Some(f64::INFINITY), 1e-4));
    }

    /// P1 sentinel (review follow-up): the KKT gate must demote *every*
    /// solution-claiming label, not just Optimal/PASS — a constraint-violating
    /// solution reported as NONCONVEX_LOCAL / NONCONVEX_GLOBAL / SUBOPTIMAL is
    /// the exact false-positive path the review flagged.
    ///
    /// Reverting `kkt_gated_label` to a `tentative` pass-through (the
    /// pre-fix per-arm behaviour) makes the KKT_FAIL asserts fail.
    #[test]
    fn test_kkt_gated_label_demotes_all_solution_claiming_labels() {
        for label in ["NONCONVEX_LOCAL", "NONCONVEX_GLOBAL", "SUBOPTIMAL"] {
            assert_eq!(kkt_gated_label(label, Some(1e-3), 1e-4), "KKT_FAIL");
            assert_eq!(kkt_gated_label(label, Some(f64::NAN), 1e-4), "KKT_FAIL");
            assert_eq!(kkt_gated_label(label, Some(1e-5), 1e-4), label);
            assert_eq!(kkt_gated_label(label, None, 1e-4), label);
        }
    }

    // --- merge_qplib_baselines: Codex P2 claim (numeric vs INFEASIBLE-sentinel
    // collision on the same ID across the two baseline files) ---

    /// P2 sentinel: same problem ID numeric in `base` (qplib.csv-shaped) and
    /// `INFEASIBLE` sentinel in `extra` (qplib_qcqp.csv-shaped). Goes through
    /// the real loaders (not hand-built maps): `load_expected_statuses` maps
    /// *every* numeric row to `ExpectedStatus::Optimal` too, so `base_stat`
    /// already carries the ID and the status-map half of
    /// `merge_qplib_baselines`'s collision check (bench_utils.rs `dups`) is a
    /// superset check across both baseline surfaces, not just the objective
    /// map. Must return `Err` — if this FAILS, the Codex claim is correct and
    /// the dedup logic needs an objective/status key union fix.
    #[test]
    fn test_merge_qplib_baselines_catches_numeric_vs_infeasible_collision() {
        let base_csv = write_tmp_csv("problem_name,optimal_obj\nQPLIB_DUP,1.5\n");
        let extra_csv = write_tmp_csv("problem_name,optimal_obj\nQPLIB_DUP,INFEASIBLE\n");

        let base_obj = load_baseline_objectives(&base_csv).unwrap();
        let base_stat = load_expected_statuses(&base_csv);
        let extra_obj = load_baseline_objectives(&extra_csv).unwrap();
        let extra_stat = load_expected_statuses(&extra_csv);

        let err = super::merge_qplib_baselines(base_obj, base_stat, extra_obj, extra_stat)
            .expect_err("numeric-vs-INFEASIBLE collision on the same ID must be rejected");
        assert_eq!(err, vec!["QPLIB_DUP".to_string()]);
    }

    /// Companion, reversed roles: `INFEASIBLE` sentinel in `base`, numeric in
    /// `extra`. Same superset-check argument applies symmetrically.
    #[test]
    fn test_merge_qplib_baselines_catches_infeasible_vs_numeric_collision_reversed() {
        let base_csv = write_tmp_csv("problem_name,optimal_obj\nQPLIB_DUP2,INFEASIBLE\n");
        let extra_csv = write_tmp_csv("problem_name,optimal_obj\nQPLIB_DUP2,2.5\n");

        let base_obj = load_baseline_objectives(&base_csv).unwrap();
        let base_stat = load_expected_statuses(&base_csv);
        let extra_obj = load_baseline_objectives(&extra_csv).unwrap();
        let extra_stat = load_expected_statuses(&extra_csv);

        let err = super::merge_qplib_baselines(base_obj, base_stat, extra_obj, extra_stat)
            .expect_err("INFEASIBLE-vs-numeric collision (reversed) must be rejected");
        assert_eq!(err, vec!["QPLIB_DUP2".to_string()]);
    }
}
