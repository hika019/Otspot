//! Orchestrator: fixpoint loop over Steps 1–11 and the final reduced-problem build.

use super::bounds::step5_bounds_tightening;
use super::doubleton::step6_doubleton_equation;
use super::empty_redundant::{step3a_empty_row, step3b_empty_column, step4_redundant_constraint};
use super::fixed::step1_fixed_variable;
use super::forcing::step2b_forcing_row;
use super::free::{step7_free_var_substitution, step8_free_singleton_col};
use super::singleton::step2_singleton_row;
use super::state::{PresolveFlags, PresolveResult, PresolveState, PresolveStatus};
use crate::problem::{ConstraintType, LpProblem};
use crate::tolerances::ZERO_TOL;
use otspot_num::sparse::CscMatrix;
use otspot_num::{run_fixpoint, run_step, PipelineStop, SolveControl};
use std::sync::atomic::AtomicBool;

// Test-only observability: counts how many transforms actually ran in the
// current pass. Purely additive bookkeeping (never gates control flow), and
// entirely `#[cfg(test)]` — both the definition and every call site below —
// so it has zero footprint in production builds.
#[cfg(test)]
thread_local! {
    static STEPS_EXECUTED_TOTAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn test_record_step_executed() {
    STEPS_EXECUTED_TOTAL.with(|c| c.set(c.get() + 1));
}

/// Run LP presolve with the default per-transform flags and the default
/// fixpoint pass cap (`crate::options::DEFAULT_PRESOLVE_MAX_PASS`, currently 10).
pub fn run_presolve(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
) -> Result<PresolveResult, PresolveStatus> {
    run_presolve_with_flags(
        problem,
        deadline,
        crate::options::DEFAULT_PRESOLVE_MAX_PASS,
        None,
        PresolveFlags::default(),
    )
}

/// Variant of `run_presolve` with an explicit pass cap, cancel token, and
/// per-transform flags. Production callers thread
/// `SolverOptions::{presolve_max_pass, cancel_flag}` through `max_pass` /
/// `cancel` (same contract as `qp_transforms::driver::run_qp_presolve_phase1`);
/// sentinel / bench-gating callers vary `flags` to isolate each transform's
/// contribution.
pub fn run_presolve_with_flags(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
    max_pass: usize,
    cancel: Option<&AtomicBool>,
    flags: PresolveFlags,
) -> Result<PresolveResult, PresolveStatus> {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return Ok(PresolveResult::no_reduction(problem));
    }

    let n = problem.num_vars;
    let m = problem.num_constraints;
    let mut st = PresolveState::from_problem(problem);

    // Loop until reduction == 0. Each step removes finitely many elements.
    // `run_step` gives each transform below its own deadline/cancel check so a
    // slow pass can bail before running its remaining transforms, instead of
    // only at the next pass boundary.
    let control = SolveControl { deadline, cancel };
    let mut interrupted = false;
    let pipeline = run_fixpoint(max_pass, control, |_| {
        let prev_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let mut new_fixed_by_step5 = 0usize;
        let mut new_subst_steps = 0usize;

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

        run_or_stop!(step1_fixed_variable(&mut st, deadline));
        run_or_stop!(step2_singleton_row(&mut st, deadline));
        run_or_stop!(step2b_forcing_row(&mut st, deadline));
        run_or_stop!(step3a_empty_row(&mut st, deadline));
        run_or_stop!(step3b_empty_column(&mut st, deadline));
        run_or_stop!(step4_redundant_constraint(&mut st, deadline));
        run_or_stop!(step5_bounds_tightening(
            &mut st,
            &mut new_fixed_by_step5,
            deadline
        ));
        run_or_stop!(step6_doubleton_equation(
            &mut st,
            &mut new_subst_steps,
            deadline
        ));
        run_or_stop!(step7_free_var_substitution(
            &mut st,
            &mut new_subst_steps,
            deadline
        ));
        run_or_stop!(step8_free_singleton_col(
            &mut st,
            &mut new_subst_steps,
            deadline
        ));

        if flags.enable_parallel_row {
            run_or_stop!(crate::presolve::transforms_dup::step9_parallel_row(
                &mut st, deadline
            ));
        }
        if flags.enable_dup_dom_col {
            run_or_stop!(crate::presolve::transforms_dup::step10_dup_dom_col(
                &mut st,
                &mut new_fixed_by_step5,
                deadline
            ));
        }
        if flags.enable_dual_fixing {
            run_or_stop!(crate::presolve::transforms_dup::step11_dual_fixing(
                &mut st,
                &mut new_fixed_by_step5,
                deadline
            ));
        }

        let curr_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let reduction = curr_removed - prev_removed;
        Ok(reduction != 0 || new_fixed_by_step5 != 0 || new_subst_steps != 0)
    });
    let stop = pipeline?;
    if interrupted {
        return Ok(PresolveResult::no_reduction(problem));
    }

    // `max_pass == 0` is a deliberate "disable the iterative loop" request
    // (see `SolverOptions::presolve_max_pass` doc), not an exhausted budget —
    // `run_fixpoint` trivially reports `PassLimit` for it without ever
    // running a pass, so exclude it from the loud "ran out of passes" signal.
    let pass_limit_hit = max_pass > 0 && stop == PipelineStop::PassLimit;
    if pass_limit_hit {
        log::warn!(
            "LP presolve hit presolve_max_pass={max_pass} without reaching a fixpoint \
             ({n} vars, {m} constraints); the emitted reduction may be incomplete"
        );
    }

    // Drop bound-tightening's redundant implied bounds before emitting, so the
    // simplex standard form does not materialize a UB row for every variable a
    // retained constraint row already bounds.
    super::bounds::revert_redundant_added_bounds(&mut st);

    build_reduced_result(problem, st, n, m, pass_limit_hit)
}

fn build_reduced_result(
    problem: &LpProblem,
    st: PresolveState,
    n: usize,
    m: usize,
    pass_limit_hit: bool,
) -> Result<PresolveResult, PresolveStatus> {
    let mut col_map = vec![None; n];
    let mut new_col_idx = 0usize;
    for j in 0..n {
        if !st.removed_cols[j] {
            col_map[j] = Some(new_col_idx);
            new_col_idx += 1;
        }
    }
    let n_new = new_col_idx;

    let mut row_map = vec![None; m];
    let mut new_row_idx = 0usize;
    for i in 0..m {
        if !st.removed_rows[i] {
            row_map[i] = Some(new_row_idx);
            new_row_idx += 1;
        }
    }
    let m_new = new_row_idx;

    let was_reduced = n_new < n || m_new < m;

    let mut c_new = vec![0.0f64; n_new];
    let mut bounds_new = vec![(0.0f64, f64::INFINITY); n_new];
    for j in 0..n {
        if let Some(jj) = col_map[j] {
            c_new[jj] = st.c[j];
            bounds_new[jj] = st.bounds[j];
        }
    }

    let mut b_new = vec![0.0f64; m_new];
    let mut ct_new = vec![ConstraintType::Le; m_new];
    for i in 0..m {
        if let Some(ii) = row_map[i] {
            b_new[ii] = st.b[i];
            ct_new[ii] = st.constraint_types[i];
        }
    }

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        for &(row, val) in &st.col_entries[j] {
            if st.removed_rows[row] || val.abs() < ZERO_TOL {
                continue;
            }
            let ii = row_map[row].unwrap();
            trip_rows.push(ii);
            trip_cols.push(jj);
            trip_vals.push(val);
        }
    }

    let a_new = if trip_rows.is_empty() {
        CscMatrix::new(m_new, n_new)
    } else {
        match CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n_new) {
            Ok(a) => a,
            // Presolve is optional. If finite input overflowed during a transform,
            // discard the entire transaction and let the solver use the original LP.
            Err(_) => return Ok(PresolveResult::no_reduction(problem)),
        }
    };

    let reduced_problem = match LpProblem::new_general(
        c_new,
        a_new,
        b_new,
        ct_new,
        bounds_new,
        problem.name.clone(),
    ) {
        Ok(reduced) => reduced,
        Err(_) => return Ok(PresolveResult::no_reduction(problem)),
    };

    Ok(PresolveResult {
        reduced_problem,
        postsolve_stack: st.postsolve_stack,
        orig_num_vars: n,
        orig_num_constraints: m,
        col_map,
        row_map,
        was_reduced,
        obj_offset: st.obj_offset,
        pass_limit_hit,
    })
}

#[cfg(test)]
mod failure_tests {
    use super::*;

    #[test]
    fn invalid_rebuild_rolls_back_instead_of_substituting_zero_matrix() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut problem = LpProblem::new(vec![0.0], a, vec![1.0]).unwrap();
        problem.obj_offset = 7.25;
        let mut state = PresolveState::from_problem(&problem);
        state.col_entries[0][0].1 = f64::INFINITY;

        let result = build_reduced_result(&problem, state, 1, 1, false).unwrap();

        assert!(!result.was_reduced);
        assert_eq!(result.reduced_problem.a.nnz(), 1);
        assert_eq!(result.reduced_problem.a.values(), [1.0]);
        assert!(result.postsolve_stack.is_empty());
        assert_eq!(result.obj_offset, problem.obj_offset);
        assert_eq!(result.reduced_problem.obj_offset, problem.obj_offset);
    }

    #[test]
    fn invalid_reduced_model_rolls_back_without_leaking_postsolve_state() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut problem = LpProblem::new(vec![2.0], a, vec![1.0]).unwrap();
        problem.obj_offset = -3.5;
        let mut state = PresolveState::from_problem(&problem);
        // Matrix reconstruction remains valid; only final model validation fails.
        state.c[0] = f64::INFINITY;
        state
            .postsolve_stack
            .push(super::super::state::PostsolveStep::BoundsTightened);

        let result = build_reduced_result(&problem, state, 1, 1, false).unwrap();

        assert!(!result.was_reduced);
        assert_eq!(result.reduced_problem.c, problem.c);
        assert_eq!(result.reduced_problem.a.values(), problem.a.values());
        assert_eq!(result.reduced_problem.b, problem.b);
        assert_eq!(result.obj_offset, problem.obj_offset);
        assert_eq!(result.reduced_problem.obj_offset, problem.obj_offset);
        assert!(result.postsolve_stack.is_empty());
    }
}

#[cfg(test)]
mod per_step_control_tests {
    //! Sentinels for the `run_step`-based per-step interruption wiring
    //! (`otspot_num::run_step`), which replaced the old local
    //! interrupt-checking macro.
    use super::*;
    use crate::problem::ConstraintType;
    use otspot_num::sparse::CscMatrix;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    // Real per-step timing races real wall-clock time, so tests that depend
    // on it must not run concurrently with each other (a second thread
    // stealing CPU would blow the timing budget).
    static TIMING_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn any_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0)],
            None,
        )
        .unwrap()
    }

    /// A wide LP built from `blocks` independent copies of a known
    /// presolve-irreducible 2-variable/3-row block (same structure as
    /// `postsolve.rs`'s `lp_non_reducible`: none of steps 1–11 can fix,
    /// singleton-resolve, redundancy-eliminate, or dual-fix any block), tiled
    /// on disjoint variable/row ranges. No step reduces anything, so all 13
    /// per-pass transforms each do real `O(blocks)` bookkeeping over
    /// `PresolveState` — used to give a short real deadline enough pass
    /// duration to land mid-pass.
    fn wide_lp(blocks: usize) -> LpProblem {
        let mut rows = Vec::with_capacity(6 * blocks);
        let mut cols = Vec::with_capacity(6 * blocks);
        let mut vals = Vec::with_capacity(6 * blocks);
        let mut b = Vec::with_capacity(3 * blocks);
        let mut c = Vec::with_capacity(2 * blocks);
        let mut bounds = Vec::with_capacity(2 * blocks);
        let mut cts = Vec::with_capacity(3 * blocks);
        for k in 0..blocks {
            let (x0, x1) = (2 * k, 2 * k + 1);
            let (r0, r1, r2) = (3 * k, 3 * k + 1, 3 * k + 2);
            // x0 + x1 <= 4
            rows.push(r0);
            cols.push(x0);
            vals.push(1.0);
            rows.push(r0);
            cols.push(x1);
            vals.push(1.0);
            // -x0 + x1 <= 2
            rows.push(r1);
            cols.push(x0);
            vals.push(-1.0);
            rows.push(r1);
            cols.push(x1);
            vals.push(1.0);
            // x0 - x1 <= 2
            rows.push(r2);
            cols.push(x0);
            vals.push(1.0);
            rows.push(r2);
            cols.push(x1);
            vals.push(-1.0);
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
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3 * blocks, 2 * blocks).unwrap();
        LpProblem::new_general(c, a, b, cts, bounds, None).unwrap()
    }

    /// Block count for `wide_lp` in the real-deadline timing tests below.
    /// Empirically (both debug and `--release`), a full 13-step pass over
    /// this many blocks takes ~7ms (debug) to ~27ms (release) on this
    /// machine (see `full_pass_duration_is_comfortably_above_the_test_deadline`),
    /// comfortably above `TEST_DEADLINE`.
    const WIDE_LP_BLOCKS: usize = 8000;

    /// Deadline used to force a real, `SolveControl`-driven mid-pass stop.
    /// Must clear two margins simultaneously in both debug and release
    /// builds: long enough to outlive `PresolveState::from_problem`'s setup
    /// cost (empirically ~3-4ms for `WIDE_LP_BLOCKS`, so the deadline isn't
    /// already expired when `run_fixpoint` enters pass 0), short enough to
    /// leave a comfortable margin below the full pass duration above.
    const TEST_DEADLINE: Duration = Duration::from_micros(6000);

    /// Sentinel: a real, already-expiring `SolveControl` deadline stops a
    /// pass partway through its transforms — proven by going through
    /// `run_step`'s actual `control.check()`, not a bypass.
    ///
    /// Mutation-fail (verified manually, see commit message): replacing the
    /// `control` argument fed to `run_step` inside `run_or_stop!` with
    /// `SolveControl::default()` (severing the wiring while leaving
    /// `run_fixpoint`'s own outer per-pass control check untouched) makes
    /// `executed` jump to 13 (all steps run) and this assertion fails,
    /// because `run_fixpoint`'s own check only fires once, before pass 0
    /// starts (deadline not yet expired at that point), and nothing else
    /// would stop the pass mid-flight.
    #[test]
    fn real_deadline_stops_pass_before_all_steps_run_via_run_step() {
        let _guard = TIMING_TEST_LOCK.lock().unwrap();
        let lp = wide_lp(WIDE_LP_BLOCKS);

        STEPS_EXECUTED_TOTAL.with(|c| c.set(0));
        let deadline = Some(Instant::now() + TEST_DEADLINE);
        let result = run_presolve_with_flags(&lp, deadline, 1, None, PresolveFlags::default())
            .expect("feasible");
        let executed = STEPS_EXECUTED_TOTAL.with(|c| c.get());

        assert!(
            !result.was_reduced,
            "mid-pass deadline expiry must discard the transaction (no_reduction)"
        );
        assert!(
            executed < 13,
            "a deadline expiring mid-pass must stop before all 13 steps run, got {executed}"
        );
        assert!(
            executed > 0,
            "the deadline must not already be expired at `run_fixpoint`'s own \
             pre-pass check (that would test the outer check, not `run_step`), got {executed}"
        );
    }

    /// Calibration check (not a correctness sentinel): confirms a full
    /// 13-step pass over `wide_lp` reliably takes much longer than
    /// `TEST_DEADLINE` above, so the margin claim is a measured fact, not an
    /// assumption.
    #[test]
    fn full_pass_duration_is_comfortably_above_the_test_deadline() {
        let _guard = TIMING_TEST_LOCK.lock().unwrap();
        let lp = wide_lp(WIDE_LP_BLOCKS);
        let t0 = Instant::now();
        let result = run_presolve_with_flags(&lp, None, 1, None, PresolveFlags::default())
            .expect("feasible");
        let elapsed = t0.elapsed();
        assert!(!result.was_reduced, "wide_lp must not reduce in one pass");
        assert!(
            elapsed > TEST_DEADLINE * 3,
            "full pass took {elapsed:?}, expected > 3x TEST_DEADLINE ({:?}) \
             for the deadline test's margin to hold",
            TEST_DEADLINE * 3
        );
    }

    /// Sentinel: without any interruption, all baseline steps of one pass run
    /// (parallel-row / dup-dom-col / dual-fixing enabled by default too).
    #[test]
    fn no_interrupt_runs_all_steps_of_one_pass() {
        STEPS_EXECUTED_TOTAL.with(|c| c.set(0));

        let lp = any_lp();
        let _ = run_presolve_with_flags(&lp, None, 1, None, PresolveFlags::default()).unwrap();

        let executed = STEPS_EXECUTED_TOTAL.with(|c| c.get());
        assert_eq!(
            executed, 13,
            "one full pass with default flags runs 10 unconditional + 3 \
             flag-gated steps (parallel-row, dup-dom-col, dual-fixing)"
        );
    }

    /// Sentinel: `presolve_max_pass=1` really caps the fixpoint at one pass.
    ///
    /// `x0 + x1 = 5` is not a singleton row until Step 2 fixes `x1 = 3` from
    /// the second Eq row *within the same pass* — but Step 2 only scans rows
    /// in ascending index order once per pass, so the newly-singleton row 0
    /// (created by fixing row 1's variable) is left for the *next* pass.
    /// With `max_pass=1` that second pass never runs, so `x0` survives;
    /// with a larger cap it is fixed too.
    #[test]
    fn presolve_max_pass_1_stops_after_first_pass() {
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let capped = run_presolve_with_flags(&lp, None, 1, None, PresolveFlags::default())
            .expect("feasible");
        let uncapped = run_presolve_with_flags(
            &lp,
            None,
            crate::options::DEFAULT_PRESOLVE_MAX_PASS,
            None,
            PresolveFlags::default(),
        )
        .expect("feasible");

        assert!(
            uncapped.reduced_problem.num_vars < capped.reduced_problem.num_vars,
            "uncapped run must reduce further than a 1-pass-capped run \
             (capped={}, uncapped={})",
            capped.reduced_problem.num_vars,
            uncapped.reduced_problem.num_vars
        );
    }

    /// Sentinel: `presolve_max_pass = 0` is a documented "disable the
    /// iterative reduction loop" contract, distinct from `presolve = false`
    /// (which skips presolve's surrounding machinery entirely). `x0 = 2` (an
    /// Eq singleton row) is reducible in a single pass by Step 2, so it
    /// proves the loop never even ran once; `presolve_max_pass = 1` (any
    /// non-zero budget) must still reduce it.
    ///
    /// Revert-fails: an implementation that ran at least one pass
    /// unconditionally (ignoring `max_pass == 0`) would reduce `x0` away,
    /// flipping `was_reduced` to `true`.
    #[test]
    fn presolve_max_pass_0_disables_the_iterative_loop() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let disabled = run_presolve_with_flags(&lp, None, 0, None, PresolveFlags::default())
            .expect("feasible");
        assert!(
            !disabled.was_reduced,
            "presolve_max_pass=0 must not run even a single reduction pass"
        );
        assert_eq!(disabled.reduced_problem.num_vars, 1);
        assert!(
            !disabled.pass_limit_hit,
            "max_pass=0 is a deliberate disable, not an exhausted budget; \
             pass_limit_hit must stay false"
        );

        let enabled = run_presolve_with_flags(&lp, None, 1, None, PresolveFlags::default())
            .expect("feasible");
        assert!(
            enabled.was_reduced,
            "presolve_max_pass=1 must still reduce the Eq singleton row"
        );
    }

    /// Sentinel: `pass_limit_hit` is `true` exactly when a genuine
    /// (non-zero) pass budget is exhausted before reaching a fixpoint — the
    /// `presolve_max_pass_1_stops_after_first_pass` chain needs 2 passes to
    /// fully converge, so a 1-pass cap must report the cap was hit.
    #[test]
    fn pass_limit_hit_true_when_nonzero_budget_exhausted() {
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();

        let capped = run_presolve_with_flags(&lp, None, 1, None, PresolveFlags::default())
            .expect("feasible");
        assert!(
            capped.pass_limit_hit,
            "1-pass cap must not be enough to converge this 2-pass chain"
        );

        let uncapped = run_presolve_with_flags(
            &lp,
            None,
            crate::options::DEFAULT_PRESOLVE_MAX_PASS,
            None,
            PresolveFlags::default(),
        )
        .expect("feasible");
        assert!(
            !uncapped.pass_limit_hit,
            "a generous cap that reaches the fixpoint must not report pass_limit_hit"
        );
    }
}
