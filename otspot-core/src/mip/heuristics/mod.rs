pub(crate) mod feasibility_pump;
pub(crate) mod local_branching;
pub(crate) mod rens;
pub(crate) mod rins;

use crate::mip::{branch::is_integer_feasible, integer_mask, MilpProblem, MipConfig, MipStats};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};

/// Deterministic node-relaxation LP-iteration cap shared by the RINS, RENS,
/// and local-branching sub-MIP configs (Phase 1d, P1-B).
///
/// Each heuristic's sub-MIP previously stopped via whichever of its node
/// limit (`RINS_NODE_LIMIT` / `RENS_NODE_LIMIT` / `LOCAL_BRANCHING_NODE_LIMIT`)
/// or fixed wall-clock timeout (all `10.0s`) came first, on the assumption
/// that the node limit binds first in practice and the wall-clock cap is
/// mostly a dormant safety valve. That assumption does not hold for every
/// instance: `milp_solve` on `khb05250.mps --timeout 300` (Phase 1d
/// determinism re-check) measured local branching's single call actually
/// hitting the wall-clock cap on every repeat — `local_branching_us` pegged
/// at `10_000_164`/`10_000_227`/`10_000_192` µs, essentially exactly `10.0s`,
/// with only ~1_086-1_088 total sub-MIP nodes processed across all of that
/// run's RINS+RENS+local-branching calls combined (nowhere near the 1_000-
/// 2_000 node limits) — and a wall-clock cap's cutoff point is inherently
/// timing-jitter dependent: 6 repeats of the same command gave `nodes_processed
/// ∈ {3344, 3344, 3344, 3344, 3344, 2952}` for the parent search.
///
/// This constant restores a fully deterministic stop for that case while
/// leaving genuinely-converging calls untouched. Calibration (Phase 1d,
/// same `milp_solve` runs, using each heuristic's `*_iters` counter — the
/// sub-MIP's own recursive `mip::effort::total_simplex_iters`, see
/// `run_rins`/`run_rens`/`run_local_branching`): local branching's call on
/// `khb05250` reached `local_branching_iters = 595_689` at the 10.0s
/// wall-clock cutoff; on `gt2.mps --timeout 60` (a call that legitimately
/// converges well under the wall cap, at 6.16s wall time) the same counter
/// reached `386_937`.
///
/// Note this counter is `total_simplex_iters` (`lp_iters_total +
/// strong_branch_iters`, the sub-MIP's other components being structurally
/// zero — recursive RINS/RENS/local-branching/tree-cuts are all disabled on
/// every sub-MIP config below), not the plain `lp_iters_total` that
/// [`crate::options::MipConfig::max_lp_iters`] is actually compared against
/// in `solve_mip_core`. The two measurements above are therefore upper
/// bounds on the sub-MIP's real `lp_iters_total` at each reference point, not
/// exact values. This does not weaken the derivation: `may_run_strong_branch`
/// caps `strong_branch_iters < STRONG_BRANCH_ITER_SHARE (0.10) *
/// total_simplex_iters`, so `lp_iters_total > 0.90 * total_simplex_iters`
/// always holds. That gives `khb05250`'s real `lp_iters_total` at the old
/// wall-clock cutoff a floor of `0.90 * 595_689 ≈ 536_120` — already above
/// `480_000` — and `gt2`'s real `lp_iters_total` a ceiling of exactly
/// `386_937` (the measured total is itself the ceiling) — already below
/// `480_000`. So `480_000` sits strictly between the two real-`lp_iters_total`
/// ranges regardless of exactly how each sub-MIP split its budget between
/// node relaxations and strong branching, not merely between the two
/// aggregate measurements. `480_000` itself is the geometric mean of the two
/// aggregate measurements (`sqrt(386_937 * 595_689) ≈ 480_098`, rounded) —
/// simpler to justify than picking a value from within the derived
/// real-`lp_iters_total` bounds directly, and the bounds above confirm that
/// choice still lands in the valid range. The `10.0s` wall-clock timeout
/// constants remain on each heuristic as a final safety valve for
/// pathologically slow per-iteration LP solves (e.g. a `pk1`-class instance
/// where a single simplex iteration itself takes seconds); with this cap in
/// place they are expected to stay dormant for the wide majority of sub-MIP
/// calls.
pub(crate) const SUB_MIP_MAX_LP_ITERS: u64 = 480_000;

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
         enforces problem.lp.a.ncols() == problem.lp.num_vars",
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
) -> (SolverResult, MipStats) {
    crate::mip::solve_milp_with_stats(problem, options, cfg)
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
) -> (SolverResult, MipStats) {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().push(cfg.clone()));
    if let Some(result) = NEXT_SUB_MIP_RESULT.with(|result| result.borrow_mut().take()) {
        return (result, MipStats::default());
    }
    crate::mip::solve_milp_with_stats(problem, options, cfg)
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
    use otspot_num::sparse::CscMatrix;

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
