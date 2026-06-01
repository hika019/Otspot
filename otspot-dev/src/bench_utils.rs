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
    if x.len() != n {
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
        SolverResult {
            status: SolveStatus::Optimal,
            ..result
        }
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
    if solution.is_empty() || solution.len() != prob.num_vars {
        return (f64::NAN, f64::NAN);
    }
    let n = solution.len();
    let qx: Vec<f64> = dd_impl::qx(&prob.q, solution)
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let aty: Vec<f64> = dd_impl::aty(&prob.a, dual_solution, n)
        .iter()
        .map(|&v| f64::from(v))
        .collect();

    // LP/Simplex path: complementarity-aware sign check on reduced costs.
    if bound_duals.is_empty() && !reduced_costs.is_empty() && reduced_costs.len() == n {
        let rel_tol = PIVOT_TOL;
        let mut dfeas_abs = 0.0_f64;
        let mut dfeas_rel = 0.0_f64;
        for j in 0..n {
            let (lb_j, ub_j) = prob.bounds[j];
            if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < ZERO_TOL {
                continue;
            }
            if prob.a.col_ptr().len() > j + 1 && prob.a.col_ptr()[j + 1] - prob.a.col_ptr()[j] == 0
            {
                continue;
            }
            let rc = reduced_costs[j];
            let x_j = solution[j];
            let at_lb =
                lb_j.is_finite() && (x_j - lb_j).abs() <= rel_tol * (1.0 + x_j.abs() + lb_j.abs());
            let at_ub =
                ub_j.is_finite() && (x_j - ub_j).abs() <= rel_tol * (1.0 + x_j.abs() + ub_j.abs());
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

    let mut bound_contrib = kkt_resid::bound_contrib(&prob.bounds, bound_duals);
    if bound_duals.is_empty() && !reduced_costs.is_empty() && reduced_costs.len() == n {
        for j in 0..n {
            bound_contrib[j] = -reduced_costs[j];
        }
    }
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
}
