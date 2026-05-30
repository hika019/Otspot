//! Phase-1 QP presolve orchestrator: fixpoint loop over steps 1–12 followed by
//! the finalize pass (matrix rebuild + Ruiz / large-coeff scaling).

use super::finalize::build_result;
use super::helpers::early_infeasibility_check;
use super::state::{QpPresolveResult, Workspace};
use super::steps_basic::{step1_fix_var, step2_singleton_row, step3_singleton_col, step4_empty};
use super::steps_bounds::{
    step10_implied_bounds, step11_dual_fixing, step9_singleton_ineq_to_bound,
};
use super::steps_free::step7_free_var;
use super::steps_parallel::step8_parallel_row;
use super::steps_redundancy::{step12_redundant_final, step5_redundant};
use crate::options::SolverOptions;
use crate::qp::QpProblem;

/// Run all Phase-1 QP-presolve transforms: fixed-var / singleton / empty-row-col /
/// redundant-constraint / parallel-row / bounds-tightening, plus diagonal-Q,
/// block-structure, large-coeff rescaling, and Ruiz hookup.
pub fn run_qp_presolve_phase1(prob: &QpProblem, opts: &SolverOptions) -> QpPresolveResult {
    if let Some(status) = early_infeasibility_check(prob) {
        return QpPresolveResult {
            presolve_status: status,
            ..QpPresolveResult::no_reduction(prob)
        };
    }

    let mut ws = Workspace::from_problem(prob);
    let deadline = opts.deadline;

    let max_iter_pass = opts.presolve_max_pass;

    let mut prev_removed_count = 0usize;
    for _iter_pass in 0..max_iter_pass {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let cur_removed_count = ws.removed_cols.iter().filter(|&&b| b).count()
            + ws.removed_rows.iter().filter(|&&b| b).count();
        if _iter_pass > 0 && cur_removed_count == prev_removed_count {
            break;
        }
        prev_removed_count = cur_removed_count;

        if let Err(r) = step1_fix_var(prob, &mut ws) {
            return r;
        }
        if let Err(r) = step2_singleton_row(prob, &mut ws) {
            return r;
        }
        if let Err(r) = step9_singleton_ineq_to_bound(prob, &mut ws, deadline) {
            return r;
        }
        if let Err(r) = step3_singleton_col(prob, &mut ws, deadline) {
            return r;
        }
        if let Err(r) = step4_empty(prob, &mut ws) {
            return r;
        }
        if let Err(r) = step5_redundant(prob, &mut ws) {
            return r;
        }
        if let Err(r) = step7_free_var(prob, &mut ws, deadline) {
            return r;
        }
        if let Err(r) = step8_parallel_row(prob, &mut ws, deadline) {
            return r;
        }
        if let Err(r) = step10_implied_bounds(prob, &mut ws, deadline) {
            return r;
        }
        if let Err(r) = step11_dual_fixing(prob, &mut ws) {
            return r;
        }
        if let Err(r) = step12_redundant_final(prob, &mut ws) {
            return r;
        }
    }

    build_result(prob, opts, ws)
}
