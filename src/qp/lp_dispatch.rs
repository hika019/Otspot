//! LP dispatch モジュール
//!
//! Q=0 退化ケース (LP 問題) を `simplex::solve_with` (= dual_advanced 経由) に委譲する。
//! IPM dispatch / 旧 dual_only は撤廃済み。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolverResult};
use crate::simplex;

use super::QpProblem;

/// Q=0 退化ケース (LP 問題) を simplex (Harris BFRT + DSE 装備の dual_advanced) に委譲する。
pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    // deadline 確定 (timeout_secs → deadline 変換)
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

    // QpProblem → LpProblem 変換
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    let mut result = simplex::solve_with(&lp, options);

    // QP 全体の obj_offset を加味
    result.objective += problem.obj_offset;
    result
}
