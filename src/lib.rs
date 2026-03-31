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
pub mod bench_utils;
pub(crate) mod presolve;
pub mod sparse;
pub mod problem;
pub(crate) mod simplex;
pub mod io;
pub(crate) mod basis;
pub(crate) mod backend;
pub mod model;
pub(crate) mod tolerances;
pub mod options;
pub use options::{SolverOptions, QpSolverChoice, Tolerance};
pub mod qp;
#[allow(dead_code)]
pub(crate) mod linalg;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use sparse::CscMatrix;
pub use problem::SolveStatus;
pub use model::{Model, ModelResult, ModelError};
pub use qp::{solve_qp, solve_qp_with, QpProblem, SolverResult, QpWarmStart};
pub use simplex::{solve, solve_with};
pub use presolve::{run_qp_presolve_phase1, run_qp_presolve_phase2};
pub use qp::{QpSolver, IpmSolver};
pub use qp::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};
