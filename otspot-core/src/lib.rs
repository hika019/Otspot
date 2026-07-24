// Numerical solver code uses index loops over multiple arrays (a[i], b[i], c[i])
// where iterator-based rewrites hurt readability or introduce borrow conflicts.
// Solver and IPM functions legitimately accept many parameters; struct-wrapping
// would be over-engineering for hot-path internals.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

//! otspot — LP / QP / MILP / MIQP ソルバー。
//!
//! LP は改訂単体法、QP は内点法 (IPM / IP-PMM) を核とし、実行不可能・非有界判定と
//! 完全な主双対情報出力に対応する。

pub mod error;
pub use error::MpsError;
pub use error::SolverError;
/// Transitional adapters into the redesigned layered architecture.
///
/// New integrations should target `otspot-ir` directly.  This module lets the
/// legacy problem types participate while their implementations are migrated.
pub mod architecture;
pub(crate) mod basis;
pub mod options;
#[doc(hidden)]
pub mod presolve;
pub mod problem;
pub(crate) mod simplex;
pub mod sparse;
pub mod tolerances;
pub use options::{
    BranchingStrategy, DualPricing, GlobalOptimizationConfig, LpWarmStart, MipBranching, MipConfig,
    SolverOptions, Tolerance, WarmStartBasis,
};
pub mod conic;
#[doc(hidden)]
pub mod linalg;
pub mod lp;
pub mod mip;
pub mod qp;
pub mod sensitivity;

#[doc(hidden)]
pub mod diag {
    pub use crate::presolve::scaling::{
        lp_scale_profile_enabled, lp_scale_profile_snapshot, reset_lp_scale_profile,
        LpScaleProfileSnapshot,
    };

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct SimplexFallbackSnapshot {
        pub ub_violation_out_of_scope: u64,
        pub phase1_bound_violation: u64,
        pub crash_infeasible: u64,
    }

    pub fn reset_simplex_fallback_profile() {
        crate::simplex::dual_advanced::reset_fallback_profile();
    }

    pub fn simplex_fallback_profile_snapshot() -> SimplexFallbackSnapshot {
        let s = crate::simplex::dual_advanced::fallback_profile_snapshot();
        SimplexFallbackSnapshot {
            ub_violation_out_of_scope: s.ub_violation_out_of_scope,
            phase1_bound_violation: s.phase1_bound_violation,
            crash_infeasible: s.crash_infeasible,
        }
    }
}

#[cfg(test)]
pub(crate) mod test_kkt;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use conic::{
    solve_miqcp, solve_misocp, solve_qcqp, solve_socp, BbOptions, ConeSpec, ConicOptions,
    ConicProblem, ConicResult, MisocpProblem, MisocpResult, QcqpProblem, QcqpResult,
    QuadConstraint,
};
pub use lp::solve_lp_with;
pub use mip::{
    solve_milp, solve_milp_with_stats, solve_miqp, solve_miqp_with_stats, MilpProblem,
    MipProblemError, MipStats, MiqpProblem,
};
pub use problem::certificate::{BoundGapCertificate, NotProven, OptimalCertificate};
pub use problem::{SolveRoute, SolveStats, SolveStatus, SolverResult};
pub use qp::certificate::prove_optimal;
pub use qp::{solve_qp, solve_qp_global, solve_qp_with, QpProblem, QpWarmStart};
pub use sparse::CscMatrix;

/// Solve an LP with default options. Includes `problem.obj_offset` in the returned objective.
///
/// Delegates to [`solve_lp_with`].
pub fn solve(problem: &crate::problem::LpProblem) -> crate::problem::SolverResult {
    lp::solve_lp_with(problem, &SolverOptions::default())
}

pub use lp::solve_lp_with as solve_with;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, SolveStatus};
    use crate::sparse::CscMatrix;

    fn make_offset_lp(obj_offset: f64) -> crate::problem::LpProblem {
        // min x  s.t. x <= 5,  x >= 0;  optimal x* = 0, c^T x* = 0
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut lp = crate::problem::LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        lp.obj_offset = obj_offset;
        lp
    }

    /// `solve` and `solve_with` must include `obj_offset` in the returned objective.
    ///
    /// Sentinel: removing `result.objective += problem.obj_offset` from
    /// `lp::solve_lp_with` causes `result.objective == 0.0` instead of 5.0 → FAIL.
    #[test]
    fn test_legacy_lp_exports_apply_obj_offset() {
        let lp = make_offset_lp(5.0);

        let r1 = solve(&lp);
        assert_eq!(r1.status, SolveStatus::Optimal);
        assert!(
            (r1.objective - 5.0).abs() < 1e-9,
            "solve: expected 5.0 (c^Tx=0 + offset 5), got {}",
            r1.objective
        );

        let r2 = solve_with(&lp, &SolverOptions::default());
        assert_eq!(r2.status, SolveStatus::Optimal);
        assert!(
            (r2.objective - 5.0).abs() < 1e-9,
            "solve_with: expected 5.0 (c^Tx=0 + offset 5), got {}",
            r2.objective
        );
    }
}

/// Internal BFRT (Bound-Flipping Ratio Test) primitives for integration tests.
/// Deferred for removal until typed pipeline restructures the simplex tree.
#[doc(hidden)]
pub mod bound_flip {
    pub use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, bfrt_select_entering, reset_bfrt_flip_invocations, BfrtResult,
        ColBound,
    };
}

/// RAII guard that disables a production sentinel for the duration of its lifetime.
///
/// On construction: calls `enable` to disable the sentinel.
/// On drop: calls `restore` to re-enable the sentinel.
/// Panic-safe: `restore` runs even if the guarded closure panics.
#[cfg(test)]
pub(crate) struct ScopedDisable<D: Fn()> {
    restore: D,
}

#[cfg(test)]
impl<D: Fn()> ScopedDisable<D> {
    pub(crate) fn new<E: Fn()>(enable: E, restore: D) -> Self {
        enable();
        ScopedDisable { restore }
    }
}

#[cfg(test)]
impl<D: Fn()> Drop for ScopedDisable<D> {
    fn drop(&mut self) {
        (self.restore)();
    }
}

/// Apply the LP KKT optimality guard to a solver result.
///
/// Exposed for integration-test sentinel load-bearing proofs. Runs full
/// KKT+dual_sign verification via `prove_optimal_lp`; demotes false-Optimal
/// to `SuboptimalSolution`. Non-Optimal results pass through unchanged.
#[doc(hidden)]
pub fn apply_lp_primal_guard(
    result: crate::problem::SolverResult,
    problem: &crate::problem::LpProblem,
) -> crate::problem::SolverResult {
    crate::qp::certificate::guard_lp_optimal(result, problem)
}
