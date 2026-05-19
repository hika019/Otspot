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
    BranchingStrategy, DualPricing, GlobalOptimizationConfig, LpWarmStart, SolverOptions,
    Tolerance, WarmStartBasis,
};
pub mod qp;
pub mod lp;
// linalg は ldl / kkt_solver 等にクロスモジュールテストからのみ参照される
// public helper を含むため、項目単位の dead_code 警告を抑制する。
#[allow(dead_code)]
pub(crate) mod linalg;

#[cfg(test)]
pub(crate) mod test_kkt;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use sparse::CscMatrix;
pub use problem::SolveStatus;
pub use model::{Model, ModelResult, ModelError};
pub use qp::{solve_qp, solve_qp_global, solve_qp_with, QpProblem, SolverResult, QpWarmStart};
pub use lp::solve_lp_with;
pub use simplex::{solve, solve_with};

/// Re-export of the BFRT (Bound-Flipping Ratio Test) primitive (#41).
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
