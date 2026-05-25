//! Regression: etamacro must reach Optimal with dfeas_rel ≤ eps at eps=1e-6.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::tolerances::{PIVOT_TOL, ZERO_TOL};
use otspot::{solve_with, QpProblem};
use std::path::Path;
use std::time::Instant;

const KNOWN_OBJ: f64 = -7.5571521774e+02;
const ETAMACRO_PATH_CANDIDATES: &[&str] = &[
    "data/lp_problems_canary/etamacro.QPS",
    "data/lp_problems/etamacro.QPS",
];

fn locate_etamacro() -> Option<&'static Path> {
    for p in ETAMACRO_PATH_CANDIDATES {
        let path = Path::new(*p);
        if path.exists() {
            return Some(Path::new(*p));
        }
    }
    None
}

fn make_lp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

/// Mirrors qps_benchmark::compute_dfeas_orig LP-path (PIVOT_TOL-relative bound-hit + sign check).
fn compute_dfeas_rel(prob: &QpProblem, solution: &[f64], reduced_costs: &[f64]) -> f64 {
    let n = prob.num_vars;
    if solution.len() != n || reduced_costs.len() != n {
        return f64::NAN;
    }
    let mut dfeas_rel = 0.0_f64;
    for j in 0..n {
        let (lb_j, ub_j) = prob.bounds[j];
        if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < ZERO_TOL {
            continue; // FX
        }
        if prob.a.col_ptr().len() > j + 1 && prob.a.col_ptr()[j + 1] - prob.a.col_ptr()[j] == 0 {
            continue; // EmptyCol
        }
        let rc = reduced_costs[j];
        let x_j = solution[j];
        let rel_tol = PIVOT_TOL;
        let at_lb = lb_j.is_finite()
            && (x_j - lb_j).abs() <= rel_tol * (1.0 + x_j.abs() + lb_j.abs());
        let at_ub = ub_j.is_finite()
            && (x_j - ub_j).abs() <= rel_tol * (1.0 + x_j.abs() + ub_j.abs());
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -rc)
        } else if at_ub && !at_lb {
            f64::max(0.0, rc)
        } else {
            0.0
        };
        let scale_j = 1.0 + rc.abs() + prob.c[j].abs();
        let rel_j = viol / scale_j;
        if rel_j > dfeas_rel {
            dfeas_rel = rel_j;
        }
    }
    dfeas_rel
}

#[test]
fn diag_etamacro_dfeas_regression() {
    let path = locate_etamacro().expect(
        "etamacro.QPS not found in canary/standard dirs — bench data 未配置。\
         scripts/download_all_bench_data.sh を実行",
    );
    let qp = parse_qps(path).expect("parse etamacro");
    let lp = make_lp(&qp);

    let eps = 1e-6;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    opts.ipm.eps = eps;

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed_s = t0.elapsed().as_secs_f64();

    let dfeas_rel = compute_dfeas_rel(&qp, &r.solution, &r.reduced_costs);
    let obj_rel_err = if KNOWN_OBJ.abs() > 0.0 {
        (r.objective - KNOWN_OBJ).abs() / (1.0 + KNOWN_OBJ.abs())
    } else {
        (r.objective - KNOWN_OBJ).abs()
    };

    eprintln!(
        "[etamacro] elapsed={:.3}s status={:?} obj={:.6e} (known {:.6e}, rel_err={:.2e}) \
         dfeas_rel={:.3e} eps={:.1e} iters={} sol_len={}/n={}",
        elapsed_s,
        r.status,
        r.objective,
        KNOWN_OBJ,
        obj_rel_err,
        dfeas_rel,
        eps,
        r.iterations,
        r.solution.len(),
        qp.num_vars,
    );

    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "etamacro: expected Optimal at eps=1e-6, got {:?}",
        r.status
    );
    assert!(
        obj_rel_err < 1e-4,
        "etamacro: obj relative error {:.2e} >= 1e-4 (got {:.6e}, known {:.6e})",
        obj_rel_err,
        r.objective,
        KNOWN_OBJ
    );
    assert!(
        dfeas_rel <= eps,
        "etamacro: dfeas_rel {:.3e} > eps {:.1e} (regression — baseline 8804da8 PASS)",
        dfeas_rel,
        eps
    );
}
