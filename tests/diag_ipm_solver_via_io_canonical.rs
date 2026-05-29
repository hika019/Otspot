// IPM solver 診断テスト — otspot-io canonical parser 経由
// (元: otspot-core/src/qp/ipm_solver/mod.rs の inline test、#109 A.1 で移動)

#![allow(clippy::print_stdout, clippy::print_stderr)]

use otspot_core::options::{IpmOptions, SolverOptions};
use otspot_core::presolve::QpPresolveResult;
use otspot_core::qp::ipm_solver;
use otspot_core::sparse::CscMatrix;
use otspot_core::QpProblem;
use otspot_core::SolveStatus;
use otspot_io::qps::parse_qps;
use std::path::Path;

fn minimal_qp() -> QpProblem {
    // min 0.5 x^2  s.t. x in [0, 1]  →  x* = 0
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let a = CscMatrix::new(0, 1);
    QpProblem::new(q, vec![0.0], a, vec![], vec![(0.0, 1.0)], vec![]).unwrap()
}

#[test]
fn test_v2_hs21() {
    let path = Path::new("data/maros_meszaros/HS21.QPS");
    if !path.exists() { return; }
    let prob = parse_qps(path).expect("parse HS21");
    let opts = SolverOptions::default();
    let r = ipm_solver::solve_ipm(&prob, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
}

/// DPKLO1 が hang せず timeout/optimal で返ること。
#[test]
fn test_ipm_dpklo1() {
    let path = Path::new("data/maros_meszaros/DPKLO1.QPS");
    if !path.exists() {
        eprintln!("DPKLO1.QPS not found, skipping");
        return;
    }
    let prob = parse_qps(path).expect("parse DPKLO1");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(5.0);
    let r = ipm_solver::solve_ipm(&prob, &opts);
    eprintln!("DPKLO1 v2: status={:?} obj={} iters={}", r.status, r.objective, r.iterations);
    assert!(matches!(r.status, SolveStatus::Optimal | SolveStatus::Timeout));
}

#[test]
fn invalid_options_rejected_at_solve_ipm() {
    let prob = minimal_qp();
    let make_cases: &[(&str, fn() -> SolverOptions)] = &[
        ("neg timeout_secs", || { let mut o = SolverOptions::default(); o.timeout_secs = Some(-1.0); o }),
        ("nan primal_tol",   || { let mut o = SolverOptions::default(); o.primal_tol = f64::NAN; o }),
        ("zero primal_tol",  || { let mut o = SolverOptions::default(); o.primal_tol = 0.0; o }),
        ("neg dual_tol",     || { let mut o = SolverOptions::default(); o.dual_tol = -1e-6; o }),
        ("zero threads",     || { let mut o = SolverOptions::default(); o.threads = 0; o }),
        ("ipm eps zero",     || { let mut o = SolverOptions::default(); o.ipm = IpmOptions { eps: 0.0, ..Default::default() }; o }),
        ("inf timeout_secs", || { let mut o = SolverOptions::default(); o.timeout_secs = Some(f64::INFINITY); o }),
        ("nan timeout_secs", || { let mut o = SolverOptions::default(); o.timeout_secs = Some(f64::NAN); o }),
    ];
    for (label, make) in make_cases {
        let bad_opts = make();
        let r = ipm_solver::solve_ipm(&prob, &bad_opts);
        assert_eq!(
            r.status,
            SolveStatus::NumericalError,
            "solve_ipm with {label} must return NumericalError, got {:?}",
            r.status,
        );
    }
}

#[test]
fn invalid_options_rejected_at_run_ipm() {
    let prob = minimal_qp();
    let presolve = QpPresolveResult::no_reduction(&prob);
    let make_cases: &[(&str, fn() -> SolverOptions)] = &[
        ("neg timeout_secs", || { let mut o = SolverOptions::default(); o.timeout_secs = Some(-1.0); o }),
        ("nan primal_tol",   || { let mut o = SolverOptions::default(); o.primal_tol = f64::NAN; o }),
        ("zero primal_tol",  || { let mut o = SolverOptions::default(); o.primal_tol = 0.0; o }),
        ("neg dual_tol",     || { let mut o = SolverOptions::default(); o.dual_tol = -1e-6; o }),
        ("zero threads",     || { let mut o = SolverOptions::default(); o.threads = 0; o }),
        ("ipm eps zero",     || { let mut o = SolverOptions::default(); o.ipm = IpmOptions { eps: 0.0, ..Default::default() }; o }),
        ("neg clamp_tol",    || { let mut o = SolverOptions::default(); o.clamp_tol = -1.0; o }),
        ("inf timeout_secs", || { let mut o = SolverOptions::default(); o.timeout_secs = Some(f64::INFINITY); o }),
    ];
    for (label, make) in make_cases {
        let bad_opts = make();
        let out = ipm_solver::core::run_ipm(&prob, &presolve, &bad_opts);
        assert!(
            out.numerical_failure,
            "run_ipm with {label} must set numerical_failure=true",
        );
    }
}
