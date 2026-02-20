//! LPバックエンド共通インターフェース
//!
//! 将来のGPU IPM等のバックエンド追加に備えた抽象化層。
//! 現在は CPU Revised Simplex のみ実装。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolverResult};
use crate::simplex;

/// LPソルバーの共通インターフェース
///
/// GPU IPM等の将来のバックエンドを追加する際、このtraitを実装する。
pub(crate) trait LpBackend {
    fn solve(&self, problem: &LpProblem, options: &SolverOptions) -> SolverResult;
}

/// CPU Revised Simplex実装（現行コードのラッパー）
pub(crate) struct SimplexBackend;

impl LpBackend for SimplexBackend {
    fn solve(&self, problem: &LpProblem, options: &SolverOptions) -> SolverResult {
        simplex::solve_with(problem, options)
    }
}
