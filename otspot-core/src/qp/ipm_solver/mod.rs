//! Mehrotra IPM (IP-PMM)。
//! 1 層 retry で eps を直線厳格化、status 変換は API 境界の 1 箇所に集約、KKT は元空間判定。

pub mod outcome;
pub mod kkt;
pub mod core;
pub mod attempt;

pub use attempt::solve_ipm;

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    use super::*;
    use crate::io::qps::parse_qps;
    use crate::options::{IpmOptions, SolverOptions};
    use crate::problem::SolveStatus;
    use crate::presolve::QpPresolveResult;
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;
    use std::path::Path;

    // test_ipm_hs21_cmp_full_solver moved to otspot-io/tests/bug_repro.rs (#28 dedup).

    #[test]
    fn test_v2_hs21() {
        let path = Path::new("data/maros_meszaros/HS21.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse HS21");
        let opts = SolverOptions::default();
        let r = solve_ipm(&prob, &opts);
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
        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let r = solve_ipm(&prob, &opts);
        eprintln!("DPKLO1 v2: status={:?} obj={} iters={}", r.status, r.objective, r.iterations);
        assert!(matches!(r.status, SolveStatus::Optimal | SolveStatus::Timeout));
    }

    fn minimal_qp() -> QpProblem {
        // min 0.5 x^2  s.t. x in [0, 1]  →  x* = 0
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        QpProblem::new(q, vec![0.0], a, vec![], vec![(0.0, 1.0)], vec![]).unwrap()
    }

    /// `solve_ipm` が不正 options で panic せず NumericalError を返すことを検証する。
    ///
    /// Sentinel: `solve_ipm` の冒頭 `validate()` ガードを削除すると、負 timeout 等が
    /// `Duration::from_secs_f64` に渡り panic になるため、このテストは FAIL する。
    #[test]
    fn invalid_options_rejected_at_solve_ipm() {
        let prob = minimal_qp();
        let cases: &[(&str, SolverOptions)] = &[
            ("neg timeout_secs", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
            ("nan primal_tol",   SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
            ("zero primal_tol",  SolverOptions { primal_tol: 0.0, ..Default::default() }),
            ("neg dual_tol",     SolverOptions { dual_tol: -1e-6, ..Default::default() }),
            ("zero threads",     SolverOptions { threads: 0, ..Default::default() }),
            ("ipm eps zero",     SolverOptions {
                ipm: IpmOptions { eps: 0.0, ..Default::default() },
                ..Default::default()
            }),
            ("inf timeout_secs", SolverOptions { timeout_secs: Some(f64::INFINITY), ..Default::default() }),
            ("nan timeout_secs", SolverOptions { timeout_secs: Some(f64::NAN), ..Default::default() }),
        ];
        for (label, bad_opts) in cases {
            let r = solve_ipm(&prob, bad_opts);
            assert_eq!(
                r.status,
                SolveStatus::NumericalError,
                "solve_ipm with {label} must return NumericalError, got {:?}",
                r.status,
            );
        }
    }

    /// `run_ipm` が不正 options で panic せず numerical_failure な IpmOutcome を返すことを検証する。
    ///
    /// Sentinel: `run_ipm` の冒頭 `validate()` ガードを削除すると、負 timeout 等が
    /// 内部ソルバーに伝播して panic になるため、このテストは FAIL する。
    #[test]
    fn invalid_options_rejected_at_run_ipm() {
        let prob = minimal_qp();
        let presolve = QpPresolveResult::no_reduction(&prob);
        let cases: &[(&str, SolverOptions)] = &[
            ("neg timeout_secs", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
            ("nan primal_tol",   SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
            ("zero primal_tol",  SolverOptions { primal_tol: 0.0, ..Default::default() }),
            ("neg dual_tol",     SolverOptions { dual_tol: -1e-6, ..Default::default() }),
            ("zero threads",     SolverOptions { threads: 0, ..Default::default() }),
            ("ipm eps zero",     SolverOptions {
                ipm: IpmOptions { eps: 0.0, ..Default::default() },
                ..Default::default()
            }),
            ("neg clamp_tol",    SolverOptions { clamp_tol: -1.0, ..Default::default() }),
            ("inf timeout_secs", SolverOptions { timeout_secs: Some(f64::INFINITY), ..Default::default() }),
        ];
        for (label, bad_opts) in cases {
            let out = core::run_ipm(&prob, &presolve, bad_opts);
            assert!(
                out.numerical_failure,
                "run_ipm with {label} must set numerical_failure=true",
            );
        }
    }
}
