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
//! | [`basis`] | LU 分解ベースの基底管理（内部使用） |
//!
//! ## 使用例
//!
//! MPS ファイルから LP 問題を読み込んで解く:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use solver::io::mps;
//! use solver::simplex;
//!
//! // MPS ファイルを読み込む
//! let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS読み込み失敗");
//!
//! // 単体法で求解
//! let result = simplex::solve(&prob);
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
pub mod presolve;
pub mod sparse;
pub mod problem;
pub mod simplex;
pub mod io;
pub mod basis;
pub mod model;
pub mod tolerances;
pub mod options;
pub use options::SolverOptions;
