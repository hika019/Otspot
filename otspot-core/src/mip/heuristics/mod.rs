pub(crate) mod feasibility_pump;
pub(crate) mod local_branching;
pub(crate) mod rens;
pub(crate) mod rins;

use crate::mip::{branch::is_integer_feasible, integer_mask, MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};

/// sub-MIP の結果を元問題の incumbent 候補へ昇格できるか判定する品質ゲート。
///
/// status を信用せず、どの status でも元問題での整数実行可能性を独立検証し、
/// objective も元問題で再計算する。解を主張しない status (Stalled /
/// MaxIterations / NumericalError 等) の iterate は候補にしない。
pub(crate) fn usable_sub_mip_result_for_original(
    problem: &MilpProblem,
    mut result: SolverResult,
    integer_feas_tol: f64,
) -> Option<SolverResult> {
    if result.solution.is_empty() {
        return None;
    }
    if !matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        return None;
    }
    if !is_original_mip_feasible(problem, &result.solution, integer_feas_tol) {
        return None;
    }
    result.objective = original_mip_objective(problem, &result.solution)?;
    Some(result)
}

fn is_original_mip_feasible(problem: &MilpProblem, x: &[f64], tol: f64) -> bool {
    if x.len() != problem.lp.num_vars || !x.iter().all(|value| value.is_finite()) {
        return false;
    }
    for (&value, &(lb, ub)) in x.iter().zip(&problem.lp.bounds) {
        if value < lb - tol || value > ub + tol {
            return false;
        }
    }
    let activity = problem.lp.a.mat_vec_mul(x).expect(
        "x.len() == problem.lp.num_vars is checked above, and LpProblem::new_general \
         enforces problem.lp.a.ncols == problem.lp.num_vars",
    );
    for ((&lhs, &rhs), sense) in activity
        .iter()
        .zip(problem.lp.b.iter())
        .zip(problem.lp.constraint_types.iter())
    {
        match sense {
            ConstraintType::Le if lhs > rhs + tol => return false,
            ConstraintType::Ge if lhs < rhs - tol => return false,
            ConstraintType::Eq if (lhs - rhs).abs() > tol => return false,
            _ => {}
        }
    }
    let mask = integer_mask(problem.lp.num_vars, &problem.integer_vars);
    is_integer_feasible(x, &mask, tol)
}

fn original_mip_objective(problem: &MilpProblem, x: &[f64]) -> Option<f64> {
    let objective = problem
        .lp
        .c
        .iter()
        .zip(x.iter())
        .map(|(&c, &value)| c * value)
        .sum::<f64>()
        + problem.lp.obj_offset;
    objective.is_finite().then_some(objective)
}

#[cfg(not(test))]
pub(crate) fn solve_sub_milp(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> SolverResult {
    crate::mip::solve_milp(problem, options, cfg)
}

#[cfg(test)]
thread_local! {
    static SUB_MIP_CONFIGS: std::cell::RefCell<Vec<MipConfig>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static NEXT_SUB_MIP_RESULT: std::cell::RefCell<Option<SolverResult>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn solve_sub_milp(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> SolverResult {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().push(cfg.clone()));
    if let Some(result) = NEXT_SUB_MIP_RESULT.with(|result| result.borrow_mut().take()) {
        return result;
    }
    crate::mip::solve_milp(problem, options, cfg)
}

#[cfg(test)]
pub(crate) fn clear_recorded_sub_mip_configs() {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn set_next_sub_mip_result(result: SolverResult) {
    NEXT_SUB_MIP_RESULT.with(|slot| *slot.borrow_mut() = Some(result));
}

#[cfg(test)]
pub(crate) fn take_recorded_sub_mip_configs() -> Vec<MipConfig> {
    SUB_MIP_CONFIGS.with(|configs| std::mem::take(&mut *configs.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::LpProblem;
    use crate::sparse::CscMatrix;

    fn one_binary_problem() -> MilpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut lp = LpProblem::new_general(
            vec![2.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0)],
            None,
        )
        .unwrap();
        lp.obj_offset = 3.0;
        MilpProblem::new(lp, vec![0]).unwrap()
    }

    fn mixed_problem_with_continuous() -> MilpProblem {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 0.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        MilpProblem::new(lp, vec![0]).unwrap()
    }

    fn result(status: SolveStatus, solution: Vec<f64>) -> SolverResult {
        SolverResult {
            status,
            solution,
            ..SolverResult::default()
        }
    }

    #[test]
    fn timeout_sub_mip_result_is_usable_when_solution_is_original_feasible() {
        let problem = one_binary_problem();
        let accepted = usable_sub_mip_result_for_original(
            &problem,
            result(SolveStatus::Timeout, vec![1.0]),
            1e-9,
        )
        .expect("feasible timeout incumbent should be kept");

        assert_eq!(accepted.solution, vec![1.0]);
        assert_eq!(accepted.objective, 5.0);
    }

    #[test]
    fn timeout_sub_mip_result_recomputes_stale_objective() {
        let problem = one_binary_problem();
        let accepted = usable_sub_mip_result_for_original(
            &problem,
            SolverResult {
                status: SolveStatus::Timeout,
                objective: -1.0e100,
                solution: vec![1.0],
                ..SolverResult::default()
            },
            1e-9,
        )
        .expect("feasible timeout incumbent should be kept with recomputed objective");

        assert_eq!(accepted.objective, 5.0);
    }

    #[test]
    fn timeout_sub_mip_result_rejects_non_finite_continuous_value() {
        let problem = mixed_problem_with_continuous();

        assert!(usable_sub_mip_result_for_original(
            &problem,
            result(SolveStatus::Timeout, vec![1.0, f64::NAN]),
            1e-9,
        )
        .is_none());
    }

    #[test]
    fn timeout_sub_mip_result_rejects_empty_solution() {
        let problem = one_binary_problem();

        assert!(usable_sub_mip_result_for_original(
            &problem,
            result(SolveStatus::Timeout, vec![]),
            1e-9,
        )
        .is_none());
    }

    #[test]
    fn timeout_sub_mip_result_rejects_fractional_integer_solution() {
        let problem = one_binary_problem();

        assert!(usable_sub_mip_result_for_original(
            &problem,
            result(SolveStatus::Timeout, vec![0.5]),
            1e-9,
        )
        .is_none());
    }

    #[test]
    fn timeout_sub_mip_result_rejects_original_constraint_violation() {
        let problem = one_binary_problem();

        assert!(usable_sub_mip_result_for_original(
            &problem,
            result(SolveStatus::Timeout, vec![2.0]),
            1e-9,
        )
        .is_none());
    }

    /// Sentinel: status を信用しない品質ゲート。Optimal / SuboptimalSolution を
    /// 名乗っていても元問題で infeasible な解は incumbent 候補にしない。
    /// 旧実装 (Optimal | SuboptimalSolution => Some(result) の無検査通過) に
    /// revert するとこのテストが FAIL する。
    #[test]
    fn claimed_statuses_are_still_feasibility_gated() {
        let problem = one_binary_problem();
        for status in [SolveStatus::Optimal, SolveStatus::SuboptimalSolution] {
            // x=2.0 は bounds (0,1) と x<=1 の両方に違反。
            assert!(
                usable_sub_mip_result_for_original(
                    &problem,
                    result(status.clone(), vec![2.0]),
                    1e-9,
                )
                .is_none(),
                "{status:?} claiming an original-infeasible solution must be rejected"
            );
        }
    }

    /// 解を主張しない status (Stalled / MaxIterations) は feasible な iterate を
    /// 持っていても incumbent 候補にならない。
    #[test]
    fn nonclaiming_statuses_are_rejected_even_when_feasible() {
        let problem = one_binary_problem();
        for status in [SolveStatus::Stalled, SolveStatus::MaxIterations] {
            assert!(
                usable_sub_mip_result_for_original(
                    &problem,
                    result(status.clone(), vec![1.0]),
                    1e-9,
                )
                .is_none(),
                "{status:?} must not become an incumbent"
            );
        }
    }

    /// Optimal の objective も元問題で再計算される (obj_offset=3, c=[2], x=1 → 5)。
    #[test]
    fn claimed_status_objective_is_recomputed_for_original() {
        let problem = one_binary_problem();
        let accepted = usable_sub_mip_result_for_original(
            &problem,
            SolverResult {
                status: SolveStatus::Optimal,
                objective: -1.0e100,
                solution: vec![1.0],
                ..SolverResult::default()
            },
            1e-9,
        )
        .expect("feasible Optimal must be kept");
        assert_eq!(accepted.objective, 5.0);
    }
}
