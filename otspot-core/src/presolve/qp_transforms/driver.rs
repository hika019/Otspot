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
use otspot_num::SolveControl;
use otspot_presolve::run_fixpoint;

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

    let control = SolveControl {
        deadline,
        cancel: opts.cancel_flag.as_deref(),
    };
    let result = run_fixpoint(max_iter_pass, control, |_| {
        let before = ws.removed_cols.iter().filter(|&&b| b).count()
            + ws.removed_rows.iter().filter(|&&b| b).count();
        step1_fix_var(prob, &mut ws)?;
        step2_singleton_row(prob, &mut ws)?;
        step9_singleton_ineq_to_bound(prob, &mut ws, deadline)?;
        step3_singleton_col(prob, &mut ws, deadline)?;
        step4_empty(prob, &mut ws)?;
        step5_redundant(prob, &mut ws)?;
        step7_free_var(prob, &mut ws, deadline)?;
        step8_parallel_row(prob, &mut ws, deadline)?;
        step10_implied_bounds(prob, &mut ws, deadline)?;
        step11_dual_fixing(prob, &mut ws)?;
        step12_redundant_final(prob, &mut ws)?;
        let after = ws.removed_cols.iter().filter(|&&b| b).count()
            + ws.removed_rows.iter().filter(|&&b| b).count();
        Ok(after != before)
    });
    if let Err(result) = result {
        return result;
    }

    build_result(prob, opts, ws)
}
