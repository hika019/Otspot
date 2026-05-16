//! LP dispatch モジュール (dual-only)
//!
//! Q=0 退化ケース (LP 問題) を `simplex::dual_only::solve` に委譲する。
//! IPM dispatch / Simplex フォールバックは削除済み。

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::simplex::dual_only;

use super::QpProblem;

/// Q=0 退化ケース (LP 問題) を dual simplex に委譲して QP 結果に変換する
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

    let _ = ConstraintType::Eq; // imports kept stable

    // 単一エントリ: dual_only
    let mut result = dual_only::solve(&lp, options);

    // objective に obj_offset を加味 (LpProblem 側にも c·x + offset の慣例があるなら必要)
    // dual_only は c·x のみ返す。problem 全体の obj_offset を追加。
    result.objective += problem.obj_offset;
    result
}
