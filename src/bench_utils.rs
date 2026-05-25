//! Shared benchmark utilities for development binaries.
//!
//! Not part of the published library. Included by `[[bin]]` targets via
//! `#[path = "../bench_utils.rs"] mod bench_utils;`.

use otspot_core::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_core::qp::kkt_resid::{self, f64_impl};
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use otspot_core::tolerances::{OBJ_MATCH_REL_TOL, obj_within_tol};
pub use otspot_core::qp::pick_best_ipm_or_simplex;

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
    max_resid = max_resid.max(primal_feas_max(&prob.a, &prob.b, &prob.constraint_types, &prob.bounds, x));
    let ax = f64_impl::ax(&prob.a, x);
    let comp_i = f64_impl::comp_ineq_products(&ax, &prob.b, &prob.constraint_types, y);
    for (i, &prod) in comp_i.iter().enumerate() {
        if prod == 0.0 { continue; }
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

pub fn compute_gap_to_global(obj: f64, global_ref: f64) -> Option<f64> {
    if !obj.is_finite() || !global_ref.is_finite() {
        return None;
    }
    Some((obj - global_ref).abs() / (1.0 + global_ref.abs()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchPromotionPolicy {
    QpsBenchmark,
    BenchQplib,
}

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

pub enum ObjCheckResult {
    Ok { rel_err: f64 },
    Mismatch { rel_err: f64 },
    NoRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

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
        if parts.len() < 2 { continue; }
        let name = parts[0].trim().to_string();
        let status_str = parts[1].trim();
        let status = match status_str.to_uppercase().as_str() {
            "INFEASIBLE" => ExpectedStatus::Infeasible,
            "UNBOUNDED" => ExpectedStatus::Unbounded,
            _ => {
                if status_str.parse::<f64>().is_ok() { ExpectedStatus::Optimal } else { continue; }
            }
        };
        map.insert(name, status);
    }
    map
}

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
    } else if data_lower.contains("qplib_nonconvex_official") || data_lower.contains("qplib-nonconvex-official") {
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
    if candidate.exists() { return candidate; }
    PathBuf::from("data/baseline_objectives").join(csv_name)
}

pub fn load_baseline_objectives(csv_path: &Path) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let content = match std::fs::read_to_string(csv_path) {
        Ok(c) => c,
        Err(_) => return map,
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                map.insert(parts[0].trim().to_string(), val);
            }
        }
    }
    map
}

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
            if rel_err > eps_obj { ObjCheckResult::Mismatch { rel_err } } else { ObjCheckResult::Ok { rel_err } }
        }
        None => ObjCheckResult::NoRef,
    }
}
