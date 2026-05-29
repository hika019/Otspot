// Numerical solver code uses index loops over multiple arrays (a[i], b[i], c[i])
// where iterator-based rewrites hurt readability or introduce borrow conflicts.
// Solver and IPM functions legitimately accept many parameters; struct-wrapping
// would be over-engineering for hot-path internals.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

//! # otspot — 数理最適化ソルバー
//!
//! 線形計画法（LP）・二次計画法（QP）と混合整数問題（MILP / MIQP）を解く Rust ソルバークレート。
//! LP は改訂単体法（Revised Simplex）、QP は内点法（IPM / IP-PMM）を核とし、
//! 実行不可能・非有界の判定と完全な主双対情報の出力に対応する。
//!
//! ## 主要モジュール
//!
//! | モジュール | 役割 |
//! |-----------|------|
//! | [`sparse`] | CSC 形式の疎行列・疎ベクトル演算 |
//! | [`problem`] | 問題定義（`LpProblem` / `QpProblem`、`SolveStatus`、`SolverResult`） |
//! | [`lp`] | LP 求解エントリポイント（`solve_lp_with`） |
//! | [`qp`] | 内点法ソルバー（QP、IPM / IP-PMM） |
//! | [`mip`] | 混合整数ソルバー（MILP / MIQP、branch-and-bound） |
//! | [`options`] | `SolverOptions`、`Tolerance` |
//!
//! ## 使用例
//!
//! MPS ファイルから LP 問題を読み込んで解く (via the `otspot` facade):
//!
//! ```rust,ignore
//! use std::path::Path;
//! use otspot::io::mps;
//!
//! let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS読み込み失敗");
//! let result = otspot_core::solve(&prob);
//! println!("最適値: {:?}", result);
//! ```

pub mod error;
pub use error::SolverError;
pub use error::MpsError;
#[doc(hidden)]
pub mod presolve;
pub mod sparse;
pub mod problem;
pub(crate) mod simplex;
pub(crate) mod basis;
pub mod tolerances;
pub mod options;
pub use options::{
    BranchingStrategy, DualPricing, GlobalOptimizationConfig, LpWarmStart, MipBranching, MipConfig,
    SolverOptions, Tolerance, WarmStartBasis,
};
pub mod qp;
pub mod mip;
pub mod lp;
#[doc(hidden)]
pub mod linalg;

#[cfg(test)]
pub(crate) mod test_kkt;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use sparse::CscMatrix;
pub use problem::{SolveRoute, SolveStats, SolveStatus, SolverResult};
pub use problem::certificate::{BoundGapCertificate, NotProven, OptimalCertificate};
pub use qp::certificate::prove_optimal;
pub use qp::{solve_qp, solve_qp_global, solve_qp_with, QpProblem, QpWarmStart};
pub use mip::{
    solve_milp, solve_milp_with_stats, solve_miqp, solve_miqp_with_stats, MilpProblem,
    MipProblemError, MipStats, MiqpProblem,
};
pub use lp::solve_lp_with;
pub use simplex::{solve, solve_with};

/// Internal BFRT (Bound-Flipping Ratio Test) primitives for integration tests.
/// Deferred for removal until typed pipeline restructures the simplex tree.
#[doc(hidden)]
pub mod bound_flip {
    pub use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, bfrt_select_entering, reset_bfrt_flip_invocations,
        BfrtResult, ColBound,
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
