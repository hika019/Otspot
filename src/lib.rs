//! # solver — 高性能数理最適化ソルバー
//!
//! 線形計画法（LP）を効率的に解くための Rust ソルバークレート。
//! 改訂単体法（Revised Simplex）と LU 分解を核に、
//! 高い数値精度と実用的な性能を実現する。
//!
//! ## 主要モジュール
//!
//! | モジュール | 役割 |
//! |-----------|------|
//! | [`sparse`] | CSC 形式の疎行列・疎ベクトル演算 |
//! | [`problem`] | LP 問題の定義（目的関数・制約・変数境界） |
//! | [`simplex`] | 改訂単体法ソルバー（Primal Simplex） |
//! | [`io`] | MPS 形式ファイルの読み込み |
//! | `basis` | LU 分解ベースの基底管理（内部実装・非公開） |
//!
//! ## 使用例
//!
//! MPS ファイルから LP 問題を読み込んで解く:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use solver::io::mps;
//!
//! // MPS ファイルを読み込む
//! let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS読み込み失敗");
//!
//! // 単体法で求解
//! let result = solver::solve(&prob);
//! println!("最適値: {:?}", result);
//! ```
//!
//! ## 開発ロードマップ
//!
//! - **M1 完了**: 密行列 Primal Simplex
//! - **M2 完了**: 改訂単体法 + LU 分解 + eta ファイル更新
//! - **M3 予定**: Dual Simplex
//! - **将来**: 二次計画法（QP）、逐次二次計画法（SQP）

pub mod error;
pub use error::SolverError;
#[doc(hidden)]
pub mod bench_utils;
pub(crate) mod presolve;
pub mod sparse;
pub mod problem;
pub(crate) mod simplex;
pub mod io;
pub(crate) mod basis;
pub mod model;
pub mod tolerances;
pub mod options;
pub use options::{
    BranchingStrategy, DualPricing, GlobalOptimizationConfig, LpWarmStart, MipBranching, MipConfig,
    SolverOptions, Tolerance, WarmStartBasis,
};
pub mod qp;
pub mod mip;
pub mod lp;
pub mod screening;
// linalg は ldl / kkt_solver 等にクロスモジュールテストからのみ参照される
// public helper を含むため、項目単位の dead_code 警告を抑制する。
#[allow(dead_code)]
pub(crate) mod linalg;

#[cfg(test)]
pub(crate) mod test_kkt;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use sparse::CscMatrix;
pub use problem::{SolveRoute, SolveStats, SolveStatus};
pub use model::{Model, ModelResult, ModelError, VarKind};
pub use qp::{solve_qp, solve_qp_global, solve_qp_with, QpProblem, SolverResult, QpWarmStart};
pub use mip::{
    solve_milp, solve_milp_with_stats, solve_miqp, solve_miqp_with_stats, MilpProblem,
    MipProblemError, MipStats, MiqpProblem,
};
pub use lp::solve_lp_with;
pub use simplex::{solve, solve_with};

/// Re-export of the BFRT (Bound-Flipping Ratio Test) primitive.
/// Public so integration sentinels in `tests/diag_simplex_bound_flip.rs`
/// can exercise the ratio-test step-size effect without forcing private
/// module exposure for the whole `simplex` tree.
pub mod bound_flip {
    pub use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, bfrt_select_entering, reset_bfrt_flip_invocations,
        BfrtResult, ColBound,
    };
}
pub use presolve::{
    run_presolve_with_flags, run_qp_presolve_phase1, run_qp_presolve_phase2,
    PresolveFlags, PresolveStatus,
};
pub use qp::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};

/// RAII guard that disables a production sentinel for the duration of its lifetime.
///
/// On construction: calls `enable` to disable the sentinel.
/// On drop: calls `restore` to re-enable the sentinel.
/// Panic-safe: `restore` runs even if the guarded closure panics.
///
/// Both `enable` and `restore` are `Fn()` so they may be called from `Drop`.
pub(crate) struct ScopedDisable<D: Fn()> {
    restore: D,
}

impl<D: Fn()> ScopedDisable<D> {
    pub(crate) fn new<E: Fn()>(enable: E, restore: D) -> Self {
        enable();
        ScopedDisable { restore }
    }
}

impl<D: Fn()> Drop for ScopedDisable<D> {
    fn drop(&mut self) {
        (self.restore)();
    }
}

/// Apply the LP primal guard to a solver result.
///
/// Exposed for integration-test sentinel load-bearing proofs. Production code
/// uses `simplex::entry::guard_lp_optimal` internally; this wrapper makes it
/// reachable from `tests/` without re-exporting the whole `simplex` tree.
#[doc(hidden)]
pub fn apply_lp_primal_guard(
    result: crate::problem::SolverResult,
    problem: &crate::problem::LpProblem,
) -> crate::problem::SolverResult {
    crate::simplex::guard_lp_optimal(result, problem)
}

/// Run `f` with the LP primal guard bypassed (thread-local, panic-safe).
///
/// Use in integration tests as a no-op scope guard: pass corrupt data through
/// the guard while disabled and assert it is NOT demoted to `NumericalError`.
/// The load-bearing evidence lives in the paired test that does NOT disable —
/// removing the guard body would cause that test to FAIL.
///
/// Thread-safe: affects only the current thread via `thread_local!` state.
#[doc(hidden)]
pub fn with_lp_guard_disabled<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    crate::simplex::with_lp_guard_disabled(f)
}
