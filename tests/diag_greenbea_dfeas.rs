//! Regression: greenbea must reach Optimal with dfeas_rel ≤ eps at eps=1e-6.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{LpProblem, SolveStatus};
use otspot::tolerances::{PIVOT_TOL, ZERO_TOL};
use otspot::{solve_with, QpProblem};
use std::path::Path;
use std::time::Instant;

/// HiGHS-exact greenbea optimum (matches data/baseline_objectives/netlib_lp.csv;
/// P-D objective error 2e-16). The Netlib README value -7.2462405908e7 is low
/// precision (3 sig digits) and was the previous reference here.
const KNOWN_OBJ: f64 = -7.2555248130e+07;
const GREENBEA_PATH_CANDIDATES: &[&str] = &[
    "data/lp_problems_canary/greenbea.QPS",
    "data/lp_problems/greenbea.QPS",
];

fn locate_greenbea() -> &'static Path {
    for p in GREENBEA_PATH_CANDIDATES {
        let path = Path::new(*p);
        if path.exists() {
            return Path::new(*p);
        }
    }
    panic!(
        "greenbea.QPS not found in any of {:?} — data missing, run scripts/netlib_lp_download.sh",
        GREENBEA_PATH_CANDIDATES
    );
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
            continue;
        }
        if prob.a.col_ptr().len() > j + 1 && prob.a.col_ptr()[j + 1] - prob.a.col_ptr()[j] == 0 {
            continue;
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

fn run_once_with_timeout(label: &str, presolve_on: bool, timeout_s: f64) -> (SolveStatus, f64, f64, f64, Option<otspot::problem::TimingBreakdown>) {
    let path = locate_greenbea();
    let qp = parse_qps(path).expect("parse greenbea");
    let lp = make_lp(&qp);

    let eps = 1e-6;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_s);
    opts.ipm.eps = eps;
    opts.presolve = presolve_on;

    let t0 = Instant::now();
    let r = solve_with(&lp, &opts);
    let elapsed = t0.elapsed().as_secs_f64();

    let dfeas_rel = compute_dfeas_rel(&qp, &r.solution, &r.reduced_costs);
    let obj_rel_err = (r.objective - KNOWN_OBJ).abs() / (1.0 + KNOWN_OBJ.abs());

    eprintln!(
        "[greenbea/{label}] elapsed={elapsed:.3}s status={:?} obj={:.6e} (known {:.6e}, rel_err={obj_rel_err:.2e}) \
         dfeas_rel={dfeas_rel:.3e} eps={eps:.1e} iters={} timing={:?}",
        r.status, r.objective, KNOWN_OBJ, r.iterations, r.timing_breakdown,
    );

    (r.status, elapsed, obj_rel_err, dfeas_rel, r.timing_breakdown)
}

/// Primary regression: presolve=on (default), must converge to dfeas_rel ≤ eps.
/// Mac local ~48s で converge、CI Linux x86_64 では 120s budget 超で SuboptimalSolution。
/// env-sensitive (CI runner Mac の 2.5x 遅)、heavy profile / `--run-ignored only` で実行。
/// 系統的真因深掘りは #97 (env-sensitive test 群)。
#[test]
#[ignore = "env-sensitive: Mac ~48s / CI Linux > 120s budget。heavy profile で実行、#97 で深掘り"]
fn diag_greenbea_dfeas_full_green() {
    let (status, _elapsed, obj_rel_err, dfeas_rel, _timing) = run_once_with_timeout("presolve_on", true, 120.0);
    let eps = 1e-6;
    assert!(
        matches!(status, SolveStatus::Optimal),
        "greenbea: expected Optimal at eps=1e-6, got {:?}",
        status
    );
    // KNOWN_OBJ is now the HiGHS-exact optimum, so the solver must match it
    // tightly (was 1e-2 only to absorb the imprecise Netlib reference).
    let obj_tol = 1e-3;
    assert!(
        obj_rel_err < obj_tol,
        "greenbea: obj relative error {:.2e} >= {:.0e}",
        obj_rel_err, obj_tol,
    );
    assert!(
        dfeas_rel <= eps,
        "greenbea: dfeas_rel {:.3e} > eps {:.1e} (DFEAS_FAIL)",
        dfeas_rel, eps,
    );
}

/// Observation: confirms presolve=off claim from handoff (28s GREEN).
/// Not a strict regression — just records the fact for triage.
#[test]
#[ignore = "diagnostic: run with `--ignored` to verify presolve=off baseline"]
fn diag_greenbea_presolve_off_baseline() {
    let (status, elapsed, _obj_rel, dfeas_rel, _timing) = run_once_with_timeout("presolve_off_120", false, 120.0);
    eprintln!(
        "[greenbea/presolve_off] status={:?} elapsed={:.2}s dfeas_rel={:.3e}",
        status, elapsed, dfeas_rel
    );
    assert_ne!(
        status, SolveStatus::NumericalError,
        "presolve=off must not cause NumericalError (solver instability)"
    );
}
