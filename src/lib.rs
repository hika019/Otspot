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
//!
//! インラインモデリング API:
//!
//! ```rust
//! use otspot::model::{Model, constraint};
//!
//! let mut model = Model::new("example");
//! let x = model.add_var("x", 0.0, 10.0);
//! let y = model.add_var("y", 0.0, 10.0);
//! model.add_constraint(constraint!((x + y) <= 8.0));
//! model.minimize(2.0 * x + y);
//! let result = model.solve().unwrap();
//! assert!((result[x] + result[y] - 0.0).abs() < 1e-4);
//! ```
//!
//! MPS 文字列から LP をパースして解く:
//!
//! ```rust
//! use otspot::io::mps::parse_mps;
//!
//! let mps = "NAME  test\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1.0 c1 1.0\nRHS\n rhs c1 5.0\nENDATA\n";
//! let prob = parse_mps(mps).unwrap();
//! let result = otspot::solve(&prob);
//! assert_eq!(prob.num_vars, 1);
//! ```

pub use otspot_core::*;

/// Algebraic modeling API (Model, Variable, Expression, Constraint, constraint! macro).
pub use otspot_model::{
    Model, ModelError, ModelResult, SolutionProof, SolveError,
    Constraint, ConstraintSense, Expression, VarKind, Variable,
};

/// `constraint!` macro for building constraints with natural syntax.
pub use otspot_model::constraint;

/// `model` submodule — re-exports the full otspot-model API.
pub mod model {
    pub use otspot_model::*;
}

/// File I/O — MPS, QPS, and QPLIB format parsers.
pub use otspot_io as io;

/// LP coverage screening utilities.
pub use otspot_io::screening;
