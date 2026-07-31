pub(crate) mod feasibility_pump;
pub(crate) mod local_branching;
pub(crate) mod rens;
pub(crate) mod rins;

use crate::mip::{branch::is_integer_feasible, integer_mask, MilpProblem, MipConfig, MipStats};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use std::time::{Duration, Instant};

/// Effective wall-clock deadline for a sub-MIP call: whichever is sooner of
/// the parent's own deadline (if any) and `now + max_time_secs`, the fixed
/// per-call cap (`RINS_MAX_TIME_SECS` / `RENS_MAX_TIME_SECS` /
/// `LOCAL_BRANCHING_MAX_TIME_SECS`). Without this, setting only
/// `sub_opts.timeout_secs = Some(max_time_secs)` (with `sub_opts.deadline =
/// None`) computes a fresh deadline purely from `max_time_secs`, ignoring how
/// little of the parent's own budget remains, so the sub-MIP could run up to
/// `max_time_secs` past the user's requested overall timeout (Codex review,
/// P1). Reusing this narrows determinism only at the very end of a search —
/// when the parent deadline is the binding term, the cutoff is wall-clock
/// (hence timing-jitter) dependent again, same as before Phase 1d — but that
/// window is at most `max_time_secs` wide, at the tail of an already-timed-out
/// solve.
pub(crate) fn sub_mip_deadline(parent_deadline: &Option<Instant>, max_time_secs: f64) -> Instant {
    let capped = Instant::now() + Duration::from_secs_f64(max_time_secs);
    match parent_deadline {
        Some(d) => (*d).min(capped),
        None => capped,
    }
}

/// Deterministic node-relaxation LP-iteration cap shared by the RINS, RENS,
/// and local-branching sub-MIP configs (Phase 1d, P1-B).
///
/// Each heuristic's fixed 10.0s wall-clock sub-MIP timeout was assumed to be
/// a dormant safety valve behind its node limit, but on `khb05250.mps
/// --timeout 300` it was the actual binding stop on every repeat (~10.000s),
/// and a wall-clock cutoff is inherently timing-jitter dependent (6 repeats
/// gave `nodes_processed` of 3344 five times, 2952 once).
///
/// Calibration: `khb05250`'s local-branching call reached
/// `total_simplex_iters = 595_689` at the old 10.0s cutoff; `gt2`'s
/// (legitimately converging at 6.16s) reached `386_937`. `480_000` is their
/// geometric mean. This constant is checked against `lp_iters_total` alone,
/// not the `total_simplex_iters` aggregate above — but `may_run_strong_branch`
/// bounds `strong_branch_iters < 0.10 * total_simplex_iters`, so
/// `lp_iters_total > 0.90 * total_simplex_iters` always holds, giving
/// `khb05250` a floor of `~536_120` (above `480_000`) and `gt2` a ceiling of
/// `386_937` (below it) — `480_000` separates the two regardless of the
/// aggregation gap. The 10.0s wall-clock caps remain as a safety valve for
/// pathologically slow per-iteration LP solves.
pub(crate) const SUB_MIP_MAX_LP_ITERS: u64 = 480_000;

/// Minimum per-call iteration allowance below which a RINS/RENS/local-
/// branching sub-MIP call is skipped outright (deferred until the share
/// budget accumulates enough headroom) rather than attempted with a
/// truncated `max_lp_iters` (markshare_4_0 regression fix, follow-up to
/// Codex review P1).
///
/// `capped_sub_mip_max_lp_iters` previously ran *any* nonzero remaining
/// share as `Some(remaining)`. `effort::rins_iter_budget`'s ceiling is
/// floored at `SUB_MIP_MAX_LP_ITERS` but does not grow past it while `share *
/// total_simplex_iters` stays below that floor — so as a heuristic's own
/// cumulative usage climbs toward that static ceiling, the *remaining*
/// allowance shrinks monotonically, call by call, down through every value
/// to 1. Each call pays the same fixed per-call cost (sub-MIP setup,
/// feasibility pump, probing) regardless of its granted `max_lp_iters`. On
/// `markshare_4_0` this fragmented 71 calls of ~61.7k iterations each into
/// 3,007+ calls averaging ~1.2k iterations, whose fixed overhead regressed
/// the instance from an 841s `Optimal` to a 1000s `Timeout`.
///
/// Calibrated from the useful-call range observed before per-call share
/// capping existed: real calls that meaningfully searched a neighborhood
/// needed 26k-90k iterations. `30_000` sits just under that range's low end
/// — a smaller allowance cannot cover even the cheapest useful call, so
/// attempting it only pays overhead for a sub-MIP almost certain to be cut
/// off before finding anything. Skipping defers to a later call once the
/// growing ceiling clears this floor again.
pub(crate) const SUB_MIP_MIN_LP_ITERS: u64 = 30_000;

/// `min(SUB_MIP_MAX_LP_ITERS, remaining_share_budget)`, or `None` when
/// `remaining_share_budget` is below [`SUB_MIP_MIN_LP_ITERS`] — the sub-MIP
/// call should be skipped outright rather than attempted with a knowably
/// too-small iteration allowance (see that constant's doc).
///
/// Codex review (P1): `effort::may_run_rins`/`may_run_rens`/
/// `may_run_local_branching` only *approve* a call — approval alone did not
/// cap the size of the approved work, so `rins_sub_mip_config` and its RENS/
/// local-branching counterparts always granted the flat `SUB_MIP_MAX_LP_
/// ITERS` regardless of how little of that heuristic's own iteration share
/// (`effort::rins_iter_budget` / `rens_iter_budget` /
/// `local_branching_iter_budget`) was actually left. An "approved"
/// (`component_iters` still comfortably under its float share) call could
/// therefore still overshoot its own share by up to `SUB_MIP_MAX_LP_ITERS`
/// in a single sub-MIP solve.
pub(crate) fn capped_sub_mip_max_lp_iters(remaining_share_budget: u64) -> Option<u64> {
    if remaining_share_budget < SUB_MIP_MIN_LP_ITERS {
        None
    } else {
        Some(SUB_MIP_MAX_LP_ITERS.min(remaining_share_budget))
    }
}

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
    static SUB_MIP_DEADLINES: std::cell::RefCell<Vec<Option<Instant>>> =
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
    SUB_MIP_DEADLINES.with(|deadlines| deadlines.borrow_mut().push(options.deadline));
    if let Some(result) = NEXT_SUB_MIP_RESULT.with(|result| result.borrow_mut().take()) {
        return (result, MipStats::default());
    }
    crate::mip::solve_milp_with_stats(problem, options, cfg)
}

#[cfg(test)]
pub(crate) fn clear_recorded_sub_mip_configs() {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().clear());
    SUB_MIP_DEADLINES.with(|deadlines| deadlines.borrow_mut().clear());
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
pub(crate) fn take_recorded_sub_mip_deadlines() -> Vec<Option<Instant>> {
    SUB_MIP_DEADLINES.with(|deadlines| std::mem::take(&mut *deadlines.borrow_mut()))
}

/// SENTINEL (Codex review, P1 / markshare_4_0 follow-up): `capped_sub_mip_
/// max_lp_iters` — see its doc.
///
/// Sentinel: removing the `.min(remaining_share_budget)` cap (reverting to
/// always returning `Some(SUB_MIP_MAX_LP_ITERS)`) fails
/// `share_between_min_and_max_caps_at_the_share`; removing the `<
/// SUB_MIP_MIN_LP_ITERS` branch entirely (reverting to always `Some(...)`)
/// fails `share_below_min_lp_iters_skips_the_call`.
#[cfg(test)]
mod capped_sub_mip_max_lp_iters_tests {
    use super::{capped_sub_mip_max_lp_iters, SUB_MIP_MAX_LP_ITERS, SUB_MIP_MIN_LP_ITERS};

    #[test]
    fn share_between_min_and_max_caps_at_the_share() {
        assert_eq!(
            capped_sub_mip_max_lp_iters(50_000),
            Some(50_000),
            "a remaining share between the min and flat-constant thresholds \
             must win over the larger flat constant"
        );
    }

    #[test]
    fn share_at_exactly_min_lp_iters_runs_at_that_size() {
        assert_eq!(
            capped_sub_mip_max_lp_iters(SUB_MIP_MIN_LP_ITERS),
            Some(SUB_MIP_MIN_LP_ITERS),
            "the boundary value itself must still run, not be skipped"
        );
    }

    #[test]
    fn ample_share_caps_at_the_flat_constant() {
        assert_eq!(
            capped_sub_mip_max_lp_iters(SUB_MIP_MAX_LP_ITERS * 10),
            Some(SUB_MIP_MAX_LP_ITERS),
            "an ample remaining share must not exceed the flat constant"
        );
    }

    /// SENTINEL (markshare_4_0 regression fix): a small but *nonzero*
    /// remaining share below `SUB_MIP_MIN_LP_ITERS` must skip the call, not
    /// attempt it with a truncated `max_lp_iters` — the fragmentation this
    /// fix targets (see `SUB_MIP_MIN_LP_ITERS`'s doc).
    #[test]
    fn share_below_min_lp_iters_skips_the_call() {
        assert_eq!(
            capped_sub_mip_max_lp_iters(SUB_MIP_MIN_LP_ITERS - 1),
            None,
            "a nonzero remaining share below SUB_MIP_MIN_LP_ITERS must skip \
             the sub-MIP call, not attempt it truncated"
        );
    }

    #[test]
    fn zero_remaining_share_skips_the_call() {
        assert_eq!(
            capped_sub_mip_max_lp_iters(0),
            None,
            "a provably-zero remaining share must skip the sub-MIP call, \
             not attempt it with Some(0)"
        );
    }
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
