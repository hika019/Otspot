//! Benchmark and test utility functions.
//!
//! Shared helpers used by integration tests, benchmark binaries, and the
//! `lp_screen` development tool.  Not part of the public solver API.

use otspot_core::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_core::qp::kkt_resid::{self, f64_impl};
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Re-export production utilities that also live in otspot-core.
pub use otspot_core::tolerances::{OBJ_MATCH_REL_TOL, obj_within_tol};
pub use otspot_core::qp::pick_best_ipm_or_simplex;

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

/// Promote `SuboptimalSolution` / `LocallyOptimal` with a full solution to `Optimal`.
///
/// `Timeout` is intentionally excluded: surfacing an honest Timeout is more
/// useful than silently promoting a partial incumbent.
pub fn apply_bench_status_promotion(
    result: SolverResult,
    num_vars: usize,
    policy: BenchPromotionPolicy,
) -> SolverResult {
    let eligible_status = matches!(
        result.status,
        SolveStatus::SuboptimalSolution | SolveStatus::LocallyOptimal
    );
    let has_full_solution = !result.solution.is_empty() && result.solution.len() == num_vars;
    let obj_ok = match policy {
        BenchPromotionPolicy::QpsBenchmark => true,
        BenchPromotionPolicy::BenchQplib => result.objective.is_finite(),
    };
    if eligible_status && has_full_solution && obj_ok {
        SolverResult { status: SolveStatus::Optimal, ..result }
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
    } else if data_lower.contains("qplib") {
        "qplib.csv"
    } else if data_lower.contains("osqp_bench") || data_lower.contains("osqp-bench") {
        "osqp_bench.csv"
    } else if data_lower.contains("mpc_qp") || data_lower.contains("mpc-qp") {
        "mpc_qp.csv"
    } else if data_lower.contains("lp_problems_infeas") || data_lower.contains("lp-problems-infeas") {
        "netlib_lp_infeas.csv"
    } else if data_lower.contains("lp_problems_extra") || data_lower.contains("lp-problems-extra") {
        "netlib_lp_extra.csv"
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
pub fn load_baseline_objectives(csv_path: &Path) -> HashMap<String, f64> {
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
        if parts.len() >= 2 {
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                map.insert(parts[0].trim().to_string(), val);
            }
        }
    }
    map
}

/// Check solver objective against a baseline CSV (1% threshold by default).
pub fn check_baseline_objective(
    problem_name: &str,
    solver_obj: f64,
    known: &HashMap<String, f64>,
    eps_obj: f64,
) -> ObjCheckResult {
    match known.get(problem_name) {
        Some(&known_obj) => {
            if !solver_obj.is_finite() {
                return ObjCheckResult::Mismatch { rel_err: f64::INFINITY };
            }
            let denom = 1.0_f64.max(known_obj.abs());
            let rel_err = (solver_obj - known_obj).abs() / denom;
            if rel_err > eps_obj {
                ObjCheckResult::Mismatch { rel_err }
            } else {
                ObjCheckResult::Ok { rel_err }
            }
        }
        None => ObjCheckResult::NoRef,
    }
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
        assert!(p.to_string_lossy().contains("netlib_lp_infeas.csv"),
            "Expected netlib_lp_infeas.csv, got {p:?}");
    }

    #[test]
    fn test_detect_csv_path_extra() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_extra", None, root);
        assert!(p.to_string_lossy().contains("netlib_lp_extra.csv"),
            "Expected netlib_lp_extra.csv, got {p:?}");
    }

    #[test]
    fn test_detect_csv_path_default_netlib() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems", None, root);
        assert!(p.to_string_lossy().contains("netlib_lp.csv"),
            "Expected netlib_lp.csv, got {p:?}");
    }
}
