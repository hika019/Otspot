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
    // Presolve-independent bound-consistency guard: an empty box (lb > ub) is
    // trivially infeasible and is reported here so correctness does not rely on
    // presolve running or on the simplex-internal empty-box check downstream.
    if crate::problem::first_infeasible_bound(&problem.bounds).is_some() {
        let mut result = SolverResult::infeasible();
        result.stats.route = SolveRoute::LpDirect;
        return result;
    }
    // Materialize timeout_secs → deadline HERE so the deadline_triggered clock
    // check below sees the same deadline the solve actually ran against
    // (a raw timeout_secs-only option set is clock-blind at this layer).
    let materialized = options.materialize_deadline();
    let options = materialized.as_ref().unwrap_or(options);
    let mut result = crate::simplex::solve_with(problem, options);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpDirect;
    result.stats.deadline_triggered =
        matches!(result.status, SolveStatus::Timeout) && options.external_stop_requested();
    result
}

/// LP entry from `solve_qp_with(Q=0)`. Sets `result.stats.route = SolveRoute::LpForwardedFromQp`.
pub(crate) fn solve_lp_forwarded_from_qp(
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let materialized = options.materialize_deadline();
    let options = materialized.as_ref().unwrap_or(options);
    let mut result = crate::simplex::solve_with(problem, options);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpForwardedFromQp;
    result.stats.deadline_triggered =
        matches!(result.status, SolveStatus::Timeout) && options.external_stop_requested();
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

    fn cx(lp: &LpProblem, sol: &[f64]) -> f64 {
        lp.c.iter().zip(sol).map(|(c, x)| c * x).sum::<f64>() + lp.obj_offset
    }

    fn solve_lp_no_presolve(lp: &LpProblem) -> SolverResult {
        let opts = SolverOptions {
            presolve: false,
            ..Default::default()
        };
        solve_lp_with(lp, &opts)
    }

    /// Sentinel: an LP with an empty variable box (lb > ub) must solve to
    /// Infeasible on the direct LP entry, with presolve either ON or OFF —
    /// never a construction Err, panic, or false Optimal.
    ///
    /// Reverting the `first_infeasible_bound` guard in `solve_lp_with` leaves
    /// the presolve-ON case still Infeasible (presolve/simplex catch it), so the
    /// guard's own contribution is exercised by asserting BOTH toggles agree.
    #[test]
    fn lp_empty_box_lb_gt_ub_is_infeasible() {
        // min x  s.t.  x <= 10 (Le),  x ∈ [5, 3]  (empty box)
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(5.0, 3.0)],
            None,
        )
        .expect("lb>ub box must be ACCEPTED at construction");
        for presolve in [true, false] {
            let opts = SolverOptions {
                presolve,
                ..Default::default()
            };
            let res = solve_lp_with(&lp, &opts);
            assert_eq!(
                res.status,
                SolveStatus::Infeasible,
                "LP empty box must be Infeasible (presolve={presolve}), got {:?}",
                res.status
            );
        }
    }

    /// Reported objective must equal `c·x` of the returned solution for an LP
    /// with a NONZERO lower bound that routes through the Big-M Phase I path.
    ///
    /// `min x  s.t.  x >= 5 (Ge),  x ∈ [3, ∞)` → optimum x=5, obj=5.
    /// The Ge constraint forces artificials ⇒ `big_m_cold_start`. The standard
    /// form shifts x = 3 + x', so `sf.obj_offset = c·lb = 3`.
    ///
    /// BUG (a4200da): `phase1.rs` recomputes `obj_orig = c·solution` from the
    /// un-shifted solution (already = c·x = 5) and then ADDS `sf.obj_offset`
    /// again ⇒ reports 8. The solution (x=5) is correct; only the scalar is wrong.
    /// This test FAILS until the Big-M path stops double-adding `sf.obj_offset`.
    /// Expected after fix: reported objective == 5.
    #[test]
    fn bigm_nonzero_lb_objective_double_count() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Ge],
            vec![(3.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.solution[0] - 5.0).abs() < 1e-6,
            "solution must be x=5; got {}",
            res.solution[0]
        );
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "reported objective {} must equal c·x {} (Big-M path double-counts \
             sf.obj_offset = c·lb = 3 ⇒ reports 8 instead of 5)",
            res.objective,
            cx(&lp, &res.solution)
        );
    }

    /// Control: Le-only LP with nonzero lower bound routes through the Le-only
    /// cold-start path, which correctly adds `sf.obj_offset` to the SHIFTED
    /// `basic_obj`. `min x s.t. x <= 10, x ∈ [3, ∞)` → x=3, obj=3. PASSES.
    /// Sentinel for the scope of the Big-M bug (this path must stay correct).
    #[test]
    fn le_only_nonzero_lb_objective_correct() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(3.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "Le-only path: reported {} must equal c·x {}",
            res.objective,
            cx(&lp, &res.solution)
        );
    }

    /// Control: bounded LP (finite ub) with nonzero lower bound routes through
    /// the BFRT bounded path, which is also correct. `min x s.t. x <= 10,
    /// x ∈ [3, 8]` → x=3, obj=3. PASSES. Sentinel for the bounded path.
    #[test]
    fn bounded_nonzero_lb_objective_correct() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(3.0, 8.0)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "bounded path: reported {} must equal c·x {}",
            res.objective,
            cx(&lp, &res.solution)
        );
    }
}
