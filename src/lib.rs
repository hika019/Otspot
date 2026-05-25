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
//! | [`model`] | 代数モデリング API（`Model`、`constraint!` マクロ） |
//! | [`sparse`] | CSC 形式の疎行列・疎ベクトル演算 |
//! | [`problem`] | 問題定義（`LpProblem` / `QpProblem`、`SolveStatus`、`SolverResult`） |
//! | [`lp`] | LP 求解エントリポイント（`solve_lp_with`） |
//! | [`qp`] | 内点法ソルバー（QP、IPM / IP-PMM） |
//! | [`mip`] | 混合整数ソルバー（MILP / MIQP、branch-and-bound） |
//! | [`io`] | MPS / QPS / QPLIB 形式ファイルの読み込み |
//! | [`options`] | `SolverOptions`、`Tolerance` |
//!
//! ## 使用例
//!
//! MPS ファイルから LP 問題を読み込んで解く:
//!
//! ```rust,no_run
//! use std::path::Path;
//! use otspot::io::mps;
//!
//! let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS読み込み失敗");
//! let result = otspot::solve(&prob);
//! println!("最適値: {:?}", result);
//! ```

pub use otspot_core::*;
