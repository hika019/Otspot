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
use otspot_num::run_fixpoint;
use otspot_num::{run_step, PipelineStop, SolveControl};
#[cfg(test)]
use std::sync::atomic::AtomicBool;

// Test-only observability: counts how many transforms actually ran in the
// current pass. Purely additive bookkeeping (never gates control flow on its
// own), and entirely `#[cfg(test)]` — both the definition and every call
// site below — so it has zero footprint in production builds. Mirrors
// `transforms::driver`'s `STEPS_EXECUTED_TOTAL`.
//
// `CANCEL_AFTER_STEPS`/`CANCEL_SIGNAL` piggyback on the same counter to give
// tests a deterministic stand-in for a real expiring deadline: once the
// executed count reaches the configured target, `CANCEL_SIGNAL` flips, which
// a test feeds to `run_qp_presolve_phase1` via `SolverOptions::cancel_flag`
// (an `Arc` clone sharing the same underlying `AtomicBool`). That drives the
// exact same `run_step`/`SolveControl::check()` path a real deadline would,
// without racing wall-clock time.
#[cfg(test)]
thread_local! {
    static STEPS_EXECUTED_TOTAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CANCEL_AFTER_STEPS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static CANCEL_SIGNAL: std::sync::Arc<AtomicBool> = std::sync::Arc::new(AtomicBool::new(false));
}

#[cfg(test)]
fn test_record_step_executed() {
    let executed = STEPS_EXECUTED_TOTAL.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if CANCEL_AFTER_STEPS.with(std::cell::Cell::get) == Some(executed) {
        CANCEL_SIGNAL.with(|flag| flag.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

/// Run all Phase-1 QP-presolve transforms: fixed-var / singleton / empty-row-col /
/// redundant-constraint / parallel-row / bounds-tightening, plus diagonal-Q,
/// block-structure, large-coeff rescaling, and Ruiz hookup.
///
/// Same per-step/per-pass control contract as the LP driver
/// (`transforms::driver::run_presolve_with_flags`): `run_step` gives each
/// transform its own deadline/cancel check so a slow pass can bail before
/// running its remaining transforms, and any interruption discards the whole
/// transaction (`QpPresolveResult::no_reduction`) rather than keeping a
/// partially-applied pass.
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
    let mut interrupted = false;
    let result = run_fixpoint(max_iter_pass, control, |_| {
        let before = ws.removed_cols.iter().filter(|&&b| b).count()
            + ws.removed_rows.iter().filter(|&&b| b).count();

        macro_rules! run_or_stop {
            ($step:expr) => {{
                if !run_step(control, || $step)? {
                    interrupted = true;
                    return Ok(false);
                }
                #[cfg(test)]
                test_record_step_executed();
            }};
        }

        run_or_stop!(step1_fix_var(prob, &mut ws));
        run_or_stop!(step2_singleton_row(prob, &mut ws));
        run_or_stop!(step9_singleton_ineq_to_bound(prob, &mut ws, deadline));
        run_or_stop!(step3_singleton_col(prob, &mut ws, deadline));
        run_or_stop!(step4_empty(prob, &mut ws));
        run_or_stop!(step5_redundant(prob, &mut ws));
        run_or_stop!(step7_free_var(prob, &mut ws, deadline));
        run_or_stop!(step8_parallel_row(prob, &mut ws, deadline));
        run_or_stop!(step10_implied_bounds(prob, &mut ws, deadline));
        run_or_stop!(step11_dual_fixing(prob, &mut ws));
        run_or_stop!(step12_redundant_final(prob, &mut ws));

        let after = ws.removed_cols.iter().filter(|&&b| b).count()
            + ws.removed_rows.iter().filter(|&&b| b).count();
        Ok(after != before)
    });
    let stop = match result {
        Err(early_result) => return early_result,
        Ok(stop) => stop,
    };
    if interrupted {
        return QpPresolveResult::no_reduction(prob);
    }

    // `max_iter_pass == 0` is a deliberate "disable the iterative loop"
    // request (see `SolverOptions::presolve_max_pass` doc), not an exhausted
    // budget — `run_fixpoint` trivially reports `PassLimit` for it without
    // ever running a pass, so exclude it from the loud "ran out of passes"
    // signal.
    let pass_limit_hit = max_iter_pass > 0 && stop == PipelineStop::PassLimit;
    if pass_limit_hit {
        log::warn!(
            "QP presolve phase1 hit presolve_max_pass={max_iter_pass} without reaching a \
             fixpoint ({} vars, {} constraints); the emitted reduction may be incomplete",
            prob.num_vars,
            prob.num_constraints,
        );
    }

    build_result(prob, opts, ws, pass_limit_hit)
}

#[cfg(test)]
mod per_step_control_tests {
    //! Sentinels for the `run_step`-based per-step interruption wiring,
    //! unifying the QP driver with the LP driver's contract (`transforms::
    //! driver::per_step_control_tests::cancel_signal_stops_pass_before_all_steps_run_via_run_step`).
    use super::*;
    use crate::problem::ConstraintType;
    use otspot_num::sparse::CscMatrix;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    const STEP_COUNT: usize = 11;

    /// A wide QP built from `blocks` independent copies of a presolve-
    /// irreducible 2-variable/3-row block (same structure as `postsolve.rs`'s
    /// `lp_non_reducible`, now with `Q = diag(2)` per block so it is a
    /// genuine QP): none of steps 1-12 reduce any block. Tiled on disjoint
    /// variable/row ranges so every pass does real `O(blocks)` bookkeeping.
    fn wide_qp(blocks: usize) -> QpProblem {
        let mut q_rows = Vec::with_capacity(2 * blocks);
        let mut q_cols = Vec::with_capacity(2 * blocks);
        let mut q_vals = Vec::with_capacity(2 * blocks);
        let mut a_rows = Vec::with_capacity(6 * blocks);
        let mut a_cols = Vec::with_capacity(6 * blocks);
        let mut a_vals = Vec::with_capacity(6 * blocks);
        let mut b = Vec::with_capacity(3 * blocks);
        let mut c = Vec::with_capacity(2 * blocks);
        let mut bounds = Vec::with_capacity(2 * blocks);
        let mut cts = Vec::with_capacity(3 * blocks);
        for k in 0..blocks {
            let (x0, x1) = (2 * k, 2 * k + 1);
            let (r0, r1, r2) = (3 * k, 3 * k + 1, 3 * k + 2);
            q_rows.push(x0);
            q_cols.push(x0);
            q_vals.push(2.0);
            q_rows.push(x1);
            q_cols.push(x1);
            q_vals.push(2.0);
            // x0 + x1 <= 4
            a_rows.push(r0);
            a_cols.push(x0);
            a_vals.push(1.0);
            a_rows.push(r0);
            a_cols.push(x1);
            a_vals.push(1.0);
            // -x0 + x1 <= 2
            a_rows.push(r1);
            a_cols.push(x0);
            a_vals.push(-1.0);
            a_rows.push(r1);
            a_cols.push(x1);
            a_vals.push(1.0);
            // x0 - x1 <= 2
            a_rows.push(r2);
            a_cols.push(x0);
            a_vals.push(1.0);
            a_rows.push(r2);
            a_cols.push(x1);
            a_vals.push(-1.0);
            b.push(4.0);
            b.push(2.0);
            b.push(2.0);
            cts.push(ConstraintType::Le);
            cts.push(ConstraintType::Le);
            cts.push(ConstraintType::Le);
            c.push(-1.0);
            c.push(-2.0);
            bounds.push((0.0, f64::INFINITY));
            bounds.push((0.0, f64::INFINITY));
        }
        let q =
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, 2 * blocks, 2 * blocks).unwrap();
        let a =
            CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 3 * blocks, 2 * blocks).unwrap();
        QpProblem::new(q, c, a, b, bounds, cts).unwrap()
    }

    /// Number of steps allowed to run before `CANCEL_SIGNAL` flips, chosen
    /// strictly between 0 and `STEP_COUNT` so the test can distinguish
    /// "stopped mid-pass" from both "never started" and "ran to completion".
    const CANCEL_AFTER: usize = 5;

    /// Sentinel: a `SolveControl` whose `cancel` flag flips deterministically
    /// after `CANCEL_AFTER` steps (a mock-clock stand-in for a real expiring
    /// deadline) stops a pass partway through its transforms — proven by
    /// going through `run_step`'s actual `control.check()`, not a bypass,
    /// matching the LP driver's contract.
    ///
    /// Mutation-fail (verified manually, see commit message): replacing the
    /// `control` argument fed to `run_step` inside `run_or_stop!` with
    /// `SolveControl::default()` (severing the wiring while leaving
    /// `run_fixpoint`'s own outer per-pass control check untouched) makes
    /// `executed` jump to `STEP_COUNT` (all steps run) and the equality
    /// assertion fails, because `run_fixpoint`'s own check only fires once,
    /// before pass 0 starts (`CANCEL_SIGNAL` is not yet set at that point),
    /// and nothing else would stop the pass mid-flight.
    #[test]
    fn cancel_signal_stops_pass_before_all_steps_run_via_run_step() {
        let prob = wide_qp(1);

        STEPS_EXECUTED_TOTAL.with(|c| c.set(0));
        CANCEL_AFTER_STEPS.with(|c| c.set(Some(CANCEL_AFTER)));
        CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

        let opts = CANCEL_SIGNAL.with(|flag| SolverOptions {
            presolve_max_pass: 1,
            use_ruiz_scaling: false,
            cancel_flag: Some(Arc::clone(flag)),
            ..SolverOptions::default()
        });
        let result = run_qp_presolve_phase1(&prob, &opts);

        CANCEL_AFTER_STEPS.with(|c| c.set(None));
        let executed = STEPS_EXECUTED_TOTAL.with(|c| c.get());

        assert!(
            !result.was_reduced,
            "mid-pass cancellation must discard the transaction (no_reduction)"
        );
        assert_eq!(
            executed, CANCEL_AFTER,
            "the cancel signal flips right after step {CANCEL_AFTER} runs, so \
             run_step must let exactly that many steps through before \
             skipping the rest (out of {STEP_COUNT}), got {executed}"
        );
    }

    /// Sentinel: without any interruption, all `STEP_COUNT` steps of one
    /// pass run.
    #[test]
    fn no_interrupt_runs_all_steps_of_one_pass() {
        STEPS_EXECUTED_TOTAL.with(|c| c.set(0));
        let prob = wide_qp(10);
        let opts = SolverOptions {
            presolve_max_pass: 1,
            use_ruiz_scaling: false,
            ..SolverOptions::default()
        };
        let _ = run_qp_presolve_phase1(&prob, &opts);
        let executed = STEPS_EXECUTED_TOTAL.with(|c| c.get());
        assert_eq!(
            executed, STEP_COUNT,
            "one full pass without interruption must run all {STEP_COUNT} steps"
        );
    }
}
