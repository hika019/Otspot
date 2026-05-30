//! Tests that invalid [`SolverOptions`] are rejected by solver entry points
//! before reaching the solver core.
//!
//! Sentinel: removing `validate()` from an entry causes the corresponding
//! case to return a wrong status or panic instead of `NumericalError`.

use super::super::*;
use crate::options::{IpmOptions, SolverOptions, Tolerance};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

fn make_trivial_qp() -> QpProblem {
    // min 0.5 x^2  s.t. x <= 5,  x >= 0
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![5.0];
    let bounds = vec![(0.0, f64::INFINITY)];
    QpProblem::new_all_le(q, c, a, b, bounds).unwrap()
}

fn make_trivial_lp_as_qp() -> QpProblem {
    // min x  s.t. x <= 5,  x >= 0  (Q=0 → LP path)
    let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![5.0];
    let bounds = vec![(0.0, f64::INFINITY)];
    QpProblem::new_all_le(q, c, a, b, bounds).unwrap()
}

/// `solve_qp_with` with invalid options returns NumericalError — not panic or wrong status.
///
/// Sentinel: removing `validate()` from `solve_qp_with` causes all these cases to
/// propagate invalid config into the solver and produce incorrect behaviour.
#[test]
fn invalid_options_rejected_at_qp_entry() {
    let qp = make_trivial_qp();
    let cases: &[(&str, SolverOptions)] = &[
        (
            "nan primal_tol",
            SolverOptions {
                primal_tol: f64::NAN,
                ..Default::default()
            },
        ),
        (
            "zero primal_tol",
            SolverOptions {
                primal_tol: 0.0,
                ..Default::default()
            },
        ),
        (
            "neg dual_tol",
            SolverOptions {
                dual_tol: -1e-6,
                ..Default::default()
            },
        ),
        (
            "inf timeout_secs",
            SolverOptions {
                timeout_secs: Some(f64::INFINITY),
                ..Default::default()
            },
        ),
        (
            "neg timeout_secs",
            SolverOptions {
                timeout_secs: Some(-1.0),
                ..Default::default()
            },
        ),
        (
            "zero threads",
            SolverOptions {
                threads: 0,
                ..Default::default()
            },
        ),
        (
            "custom tol nan",
            SolverOptions {
                tolerance: Some(Tolerance::Custom(f64::NAN)),
                ..Default::default()
            },
        ),
        (
            "ipm eps nan",
            SolverOptions {
                ipm: IpmOptions {
                    eps: f64::NAN,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ];
    for (label, opts) in cases {
        let result = solve_qp_with(&qp, opts);
        assert_eq!(
            result.status,
            SolveStatus::NumericalError,
            "solve_qp_with with {label} must return NumericalError"
        );
    }
}

/// `solve_qp_with` with LP problem (Q=0) and invalid options also returns NumericalError.
///
/// This ensures the validate() guard fires before the LP-forward dispatch.
#[test]
fn invalid_options_rejected_at_qp_entry_lp_path() {
    let lp_as_qp = make_trivial_lp_as_qp();
    let cases: &[(&str, SolverOptions)] = &[
        (
            "zero threads lp path",
            SolverOptions {
                threads: 0,
                ..Default::default()
            },
        ),
        (
            "nan primal_tol lp path",
            SolverOptions {
                primal_tol: f64::NAN,
                ..Default::default()
            },
        ),
        (
            "neg timeout lp path",
            SolverOptions {
                timeout_secs: Some(-0.1),
                ..Default::default()
            },
        ),
    ];
    for (label, opts) in cases {
        let result = solve_qp_with(&lp_as_qp, opts);
        assert_eq!(
            result.status,
            SolveStatus::NumericalError,
            "solve_qp_with (LP path) with {label} must return NumericalError"
        );
    }
}
