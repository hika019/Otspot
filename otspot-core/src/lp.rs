//! LP-specific entry point.
//!
//! Splits LP from the QP `Q.is_zero` dispatch so that LP-only paths
//! (simplex, future IPM-first / crash / postsolve) are owned by this
//! module. `solve_qp_with(Q=0)` keeps backward compat by forwarding
//! here; the two call sites are distinguishable via `SolverResult.stats.route`.

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveRoute, SolveStatus, SolverResult};

/// Solve an LP directly. Sets `result.stats.route = SolveRoute::LpDirect`.
///
/// Returns [`SolveStatus::NumericalError`] if `options` fails validation;
/// validation is performed by the underlying `simplex::solve_with`.
pub fn solve_lp_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let mut result = crate::simplex::solve_with(problem, options);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpDirect;
    result.stats.deadline_triggered = matches!(result.status, SolveStatus::Timeout);
    result
}

/// LP entry from `solve_qp_with(Q=0)`. Sets `result.stats.route = SolveRoute::LpForwardedFromQp`.
pub(crate) fn solve_lp_forwarded_from_qp(
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let mut result = crate::simplex::solve_with(problem, options);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpForwardedFromQp;
    result.stats.deadline_triggered = matches!(result.status, SolveStatus::Timeout);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn make_trivial_lp() -> LpProblem {
        // minimize x  s.t.  x <= 5,  x >= 0
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    /// Timeout incumbent must include `problem.obj_offset`.
    ///
    /// Sentinel: removing `SolveStatus::Timeout` from the match in `solve_lp_with`
    /// causes `result.objective == 0.0` instead of 42.5 → FAIL.
    ///
    /// `cancel_flag = true` with `deadline = None` bypasses the pre-simplex
    /// INFINITY timeout (entry.rs only checks `deadline.is_some_and(...)`).
    /// The simplex loop's first-iteration cancel check fires → Timeout with
    /// initial BFS (x_decision = 0, c^T x = 0, sf.obj_offset = 0).
    #[test]
    fn test_lp_timeout_incumbent_includes_obj_offset() {
        use std::sync::{atomic::AtomicBool, Arc};

        let mut lp = make_trivial_lp();
        lp.obj_offset = 42.5;

        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            presolve: false,
            ..Default::default()
        };

        let result = solve_lp_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "cancel_flag=true must produce Timeout"
        );
        assert!(
            result.objective.is_finite(),
            "Timeout incumbent must have finite objective (not INFINITY); got {}",
            result.objective
        );
        assert!(
            (result.objective - 42.5).abs() < 1e-9,
            "Timeout incumbent must include obj_offset 42.5; got {} \
             (sentinel: removing Timeout from match yields 0.0 ≠ 42.5)",
            result.objective
        );
    }

    /// Invalid options produce NumericalError via `solve_lp_with`.
    ///
    /// Validation is performed by `simplex::solve_with` (the load-bearing sentinel
    /// lives in `simplex::entry::invalid_options_rejected_at_simplex_entry`).
    #[test]
    fn invalid_options_rejected_at_lp_entry() {
        let lp = make_trivial_lp();
        let cases: &[(&str, SolverOptions)] = &[
            (
                "nan primal_tol",
                SolverOptions {
                    primal_tol: f64::NAN,
                    ..Default::default()
                },
            ),
            (
                "inf primal_tol",
                SolverOptions {
                    primal_tol: f64::INFINITY,
                    ..Default::default()
                },
            ),
            (
                "neg timeout_secs",
                SolverOptions {
                    timeout_secs: Some(-0.5),
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
                "nan dual_tol",
                SolverOptions {
                    dual_tol: f64::NAN,
                    ..Default::default()
                },
            ),
        ];
        for (label, opts) in cases {
            let result = solve_lp_with(&lp, opts);
            assert_eq!(
                result.status,
                SolveStatus::NumericalError,
                "solve_lp_with with {label} must return NumericalError"
            );
        }
    }
}
