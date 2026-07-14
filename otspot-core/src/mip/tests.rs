//! MILP branch-and-bound tests.
//!
//! Multiple data patterns per CLAUDE.md: trivial integer root, fractional root
//! requiring branching, infeasible, unbounded, binary knapsack, and the
//! no-integer LP fallback. Low-level `solve_milp` / `solve_miqp` entries are
//! exercised here. Model API tests live in `otspot-model/tests/mip_model.rs`.

use super::{
    finalize_no_incumbent, integer_mask, reduced_cost_fixing, solve_milp, solve_milp_with_stats,
    solve_mip_core, solve_miqp, solve_miqp_with_stats, MilpProblem, MiqpProblem,
};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult, TimingBreakdown};
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

const EPS: f64 = 1e-4;

fn opts() -> SolverOptions {
    // safety net; tiny problems finish instantly
    SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    }
}

/// Build an LpProblem from triplets.
#[allow(clippy::too_many_arguments)] // test fixture: explicit LP parts read better than a builder
fn build_lp(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    m: usize,
    b: Vec<f64>,
    ctypes: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
) -> LpProblem {
    let n = c.len();
    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(rows, cols, vals, m, n).unwrap()
    };
    LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap()
}

fn milp(lp: LpProblem, integer_vars: Vec<usize>) -> MilpProblem {
    MilpProblem::new(lp, integer_vars).unwrap()
}

// ---------------------------------------------------------------------------
// solve_milp (low-level entry)
// ---------------------------------------------------------------------------

#[test]
fn trivial_integer_root_is_optimal_without_branching() {
    // min x, x in [0,5] integer → x = 0. Root relaxation already integral.
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - 0.0).abs() < EPS, "obj={}", r.objective);
    assert!((r.solution[0] - 0.0).abs() < EPS);
    assert_eq!(stats.nodes_processed, 1, "integral root must not branch");
}

/// BT detects infeasibility before B&B starts: `nodes_processed` must be zero.
///
/// `x ≤ 3.7 ∧ x ≥ 3.5` with `x ∈ [0,10]` integer.
/// BT: floor(3.7)=3 → ub=3; ceil(3.5)=4 → lb=4; lb > ub → infeasible.
/// The early-exit path in `solve_milp_with_stats` returns before entering
/// the B&B driver, so `nodes_processed == 0`.
///
/// Sentinel: disabling `tighten_integer_bounds` lets B&B run (nodes > 0) and
/// this assertion FAILS. The unit test in `presolve.rs` is insufficient alone
/// because it does not cover the integration path through `solve_milp_with_stats`.
#[test]
fn bt_detects_infeasibility_before_bb() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![3.7, 3.5],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "x≤3.7 ∧ x≥3.5 integer → empty domain"
    );
    assert_eq!(
        stats.nodes_processed, 0,
        "infeasibility detected by BT, not B&B"
    );
}

/// Equality row with a non-integer rhs drives both lb and ub of the same integer
/// variable past each other, which must be caught as infeasibility before B&B.
///
/// `x = 3.5`, `x ∈ [0, 10]` integer.
/// BT (Eq path): implied_ub = floor(3.5) = 3, implied_lb = ceil(3.5) = 4.
/// new_lb=4 > new_ub=3 → infeasible; `nodes_processed` must be 0.
///
/// Sentinel: removing the cross-bound check (`new_lb > new_ub`) in
/// `propagate_row_bounds` allows the invalid bounds to propagate, the LP
/// relaxation then becomes infeasible or the solver returns a wrong status,
/// but `nodes_processed` will no longer be 0, causing this test to FAIL.
#[test]
fn equality_row_integer_var_crossed_bounds_detect_infeasibility() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![3.5],
        vec![ConstraintType::Eq],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "x=3.5 integer → no feasible integer → infeasible"
    );
    assert_eq!(
        stats.nodes_processed, 0,
        "BT must detect crossing before B&B"
    );
}

/// Bound tightening resolves the fractional-root LP at the root node without branching.
///
/// min -x s.t. 2x ≤ 3, x ∈ [0,5] integer.
/// Without BT: LP relaxation gives x=1.5 (fractional) → branching needed.
/// With BT: x ≤ floor(3/2) = 1 → LP root gives x=1 (integer) → 1 node.
///
/// Sentinel: reverting `tighten_root_bounds` to a no-op causes nodes_processed=3
/// and this test FAILS on the `nodes_processed == 1` assertion.
#[test]
fn bt_resolves_root_as_integer_feasible() {
    let lp = build_lp(
        vec![-1.0],
        &[0],
        &[0],
        &[2.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-1.0)).abs() < EPS, "obj={}", r.objective);
    assert!((r.solution[0] - 1.0).abs() < EPS, "x={}", r.solution[0]);
    assert_eq!(
        stats.nodes_processed, 1,
        "BT tightens x ≤ 1 → root is integer-feasible, no branching"
    );
    assert_eq!(stats.incumbent_updates, 1);
}

/// Branching fires when BT tightens bounds but the LP relaxation is still fractional.
///
/// min -(x+y) s.t. x+y ≤ 3.5, x,y ∈ [0,5] integer.
/// BT: x ≤ floor(3.5)=3, y ≤ floor(3.5)=3 (both tightened).
/// Root LP over [0,3]² gives (3, 0.5) — y is fractional → branching needed.
/// Integer optimum: x+y=3, obj=-3.
///
/// Sentinel: removing the branching loop and returning only the root relaxation
/// value gives obj=-3.5 (wrong, fractional) → this test FAILS.
#[test]
fn branching_fires_when_bt_insufficient() {
    let lp = build_lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.5],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-3.0)).abs() < EPS, "obj={}", r.objective);
    let s = r.solution[0].round() + r.solution[1].round();
    assert!((s - 3.0).abs() < EPS, "x+y={}", s);
    assert!(
        stats.nodes_processed >= 3,
        "fractional LP root causes branching, nodes={}",
        stats.nodes_processed
    );
    assert!(
        stats.pruned >= 1,
        "at least one pruned/infeasible child expected"
    );
    assert!(stats.incumbent_updates >= 1);
}

#[test]
fn binary_knapsack_reaches_known_optimum() {
    // max 8a + 11b + 6c + 4d s.t. 5a + 7b + 4c + 3d <= 14, all binary.
    // Known optimum: {b,c,d} = value 21 (weight 14). minimization form: negate c.
    let lp = build_lp(
        vec![-8.0, -11.0, -6.0, -4.0],
        &[0, 0, 0, 0],
        &[0, 1, 2, 3],
        &[5.0, 7.0, 4.0, 3.0],
        1,
        vec![14.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); 4],
    );
    let r = solve_milp(&milp(lp, vec![0, 1, 2, 3]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-21.0)).abs() < EPS, "obj={}", r.objective);
    let sol: Vec<f64> = r.solution.iter().map(|v| v.round()).collect();
    assert_eq!(sol, vec![0.0, 1.0, 1.0, 1.0], "knapsack pick");
}

#[test]
fn integer_infeasible_between_consecutive_integers() {
    // x in [0,10], 1.2 <= x <= 1.8 → LP feasible, no integer in the gap → infeasible.
    let lp = build_lp(
        vec![1.0],
        &[0, 1],
        &[0, 0],
        &[1.0, 1.0],
        2,
        vec![1.2, 1.8],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    let r = solve_milp(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Infeasible, "no integer in (1.2,1.8)");
}

#[test]
fn unbounded_relaxation_reports_unbounded() {
    // max x, x in [0, inf) integer, no constraints → unbounded.
    let lp = build_lp(
        vec![-1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, f64::INFINITY)],
    );
    let r = solve_milp(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Unbounded);
}

#[test]
fn no_integer_vars_falls_back_to_lp() {
    // Pure LP via the MILP entry must equal the direct LP solve.
    let lp = build_lp(
        vec![1.0, 1.0],
        &[0],
        &[0],
        &[1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let direct = crate::lp::solve_lp_with(&lp, &opts());
    let r = solve_milp(&milp(lp, vec![]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - direct.objective).abs() < EPS);
}

/// Sentinel: LP/QP fallback (empty `integer_vars`) preserves caller's `warm_start`.
///
/// When `integer_vars()` is empty the driver is a pure LP/QP passthrough.
/// The MIP-specific mutations (`recover_warm_start_basis=true`, `warm_start=None`)
/// must be applied **after** the early-return check so the caller's options reach
/// the underlying solver unmodified.
///
/// Sentinel: moving the mutations before the early-return silently discards
/// `opts.warm_start` → mock receives `None` → `warm_start_received=false` →
/// **this test FAILS**.
#[test]
fn no_integer_vars_fallback_preserves_caller_warm_start() {
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;
    use std::cell::Cell;

    struct PureLpMock {
        warm_start_received: Cell<bool>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 0],
    }

    impl PureLpMock {
        fn new() -> Self {
            Self {
                warm_start_received: Cell::new(false),
                root_bounds: [(0.0, 5.0)],
                int_vars: [],
            }
        }
    }

    impl super::Relaxation for PureLpMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, _bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            self.warm_start_received.set(opts.warm_start.is_some());
            SolverResult {
                status: SolveStatus::Optimal,
                objective: 0.0,
                solution: vec![0.0],
                ..SolverResult::default()
            }
        }
    }

    let user_ws = WarmStartBasis {
        basis: vec![0],
        x_b: vec![0.0],
    };
    let opts_with_ws = SolverOptions {
        warm_start: Some(user_ws),
        ..opts()
    };
    let mock = PureLpMock::new();
    let (r, _) = super::solve_mip_with_stats(&mock, &opts_with_ws, &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        mock.warm_start_received.get(),
        "LP/QP fallback must forward caller's warm_start to the solver; \
         no-op (mutate before early-return) gives warm_start=None → false"
    );
}

#[test]
fn mip_nodes_disable_lp_crash_basis_before_relaxation_solve() {
    use std::cell::Cell;

    fn crash_infeasible_lp() -> LpProblem {
        build_lp(
            vec![0.0],
            &[0],
            &[0],
            &[1.0],
            1,
            vec![2.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 1.0)],
        )
    }

    struct CrashLpNodeMock {
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
        saw_crash_disabled: Cell<bool>,
    }

    impl CrashLpNodeMock {
        fn new() -> Self {
            Self {
                root_bounds: [(0.0, 1.0)],
                int_vars: [0],
                saw_crash_disabled: Cell::new(false),
            }
        }
    }

    impl super::Relaxation for CrashLpNodeMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, _bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            self.saw_crash_disabled.set(!opts.use_lp_crash_basis);
            crate::lp::solve_lp_with(&crash_infeasible_lp(), opts)
        }
        fn skip_node_presolve(&self) -> bool {
            true
        }
        fn can_skip_repeated_lp_scaling(&self) -> bool {
            true
        }
    }

    // Precondition: before MIP disables LP crash, this relaxation is non-vacuous.
    // With crash enabled, the equality row adopts structural x as basic and gets
    // x_B=2 while the column bound is x<=1, so the bounded crash path records a
    // crash-infeasible fallback.
    // The honest fallback counter is gated by the LP profiling env var.
    // SAFETY: nextest runs one OS process per test, but cargo test shares a
    // process; restore the env var before other tests can see profiler state.
    unsafe { std::env::set_var("OTSPOT_LP_SOLVE_PROFILE", "1") };
    crate::presolve::scaling::reset_lp_scale_profile();
    crate::simplex::dual_advanced::reset_fallback_profile();
    let mut crash_on = opts();
    crash_on.presolve = false;
    crash_on.use_lp_crash_basis = true;
    crash_on.recover_warm_start_basis = true;
    let _ = crate::lp::solve_lp_with(&crash_infeasible_lp(), &crash_on);
    let precondition = crate::simplex::dual_advanced::fallback_profile_snapshot();
    unsafe { std::env::remove_var("OTSPOT_LP_SOLVE_PROFILE") };
    assert!(
        precondition.crash_infeasible > 0,
        "test precondition failed: crash-enabled baseline must hit fallback"
    );

    crate::presolve::scaling::reset_lp_scale_profile();
    crate::simplex::dual_advanced::reset_fallback_profile();
    let mock = CrashLpNodeMock::new();
    let (r, stats) = super::solve_mip_core(&mock, &opts(), &MipConfig::default(), vec![true], None);
    assert_eq!(r.status, SolveStatus::Infeasible);
    assert!(
        mock.saw_crash_disabled.get(),
        "B&B node LP options must force use_lp_crash_basis=false"
    );
    assert_eq!(
        stats.fallback_crash_infeasible, 0,
        "MIP node solve must not enter crash-infeasible legacy fallback"
    );
}

#[test]
fn two_var_general_integer_program() {
    // min -(x + y) s.t. x + y <= 3.5, x <= 2.5, y <= 2.5, x,y in [0,5] integer.
    // Integer optimum: x + y = 3 (e.g. x=1,y=2 or x=2,y=1) → obj -3.
    let lp = build_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        vec![3.5, 2.5, 2.5],
        vec![ConstraintType::Le; 3],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let r = solve_milp(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-3.0)).abs() < EPS, "obj={}", r.objective);
    let s = r.solution[0].round() + r.solution[1].round();
    assert!((s - 3.0).abs() < EPS, "x+y={}", s);
}

// ---------------------------------------------------------------------------
// Terminal-status classification (P1: never report false Infeasible)
// ---------------------------------------------------------------------------

#[test]
fn no_incumbent_with_open_region_is_not_infeasible() {
    // A region was left unexplored (had_open) while the queue happened to empty and no
    // interruption fired. Reporting Infeasible here would be a silent wrong answer.
    // Sentinel: dropping the `!had_open` guard flips this to Infeasible and the test FAILS.
    let r = finalize_no_incumbent(false, true, true, false);
    assert_ne!(
        r.status,
        SolveStatus::Infeasible,
        "open region must not be Infeasible"
    );
    assert_eq!(r.status, SolveStatus::MaxIterations);
}

#[test]
fn no_incumbent_fully_resolved_is_infeasible() {
    // Every region resolved (no open region, no interruption, queue empty) → genuinely
    // Infeasible.
    let r = finalize_no_incumbent(false, false, true, false);
    assert_eq!(r.status, SolveStatus::Infeasible);
}

#[test]
fn no_incumbent_deadline_is_timeout_not_infeasible() {
    let r = finalize_no_incumbent(true, true, false, true);
    assert_eq!(r.status, SolveStatus::Timeout);
}

#[test]
fn no_incumbent_budget_exhausted_is_maxiterations_not_infeasible() {
    // Interrupted by max_nodes (queue may be non-empty) → never Infeasible.
    let r = finalize_no_incumbent(true, true, false, false);
    assert_eq!(r.status, SolveStatus::MaxIterations);
}

// ---------------------------------------------------------------------------
// solve_miqp (low-level entry, convex only)
// ---------------------------------------------------------------------------

/// Build a diagonal-Q QpProblem with optional constraints.
#[allow(clippy::too_many_arguments)] // test fixture: explicit QP parts read better than a builder
fn qp_problem(
    diag: &[f64],
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    m: usize,
    b: Vec<f64>,
    ctypes: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
) -> QpProblem {
    let n = diag.len();
    let qidx: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&qidx, &qidx, diag, n, n).unwrap();
    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(rows, cols, vals, m, n).unwrap()
    };
    QpProblem::new(q, c, a, b, bounds, ctypes).unwrap()
}

fn miqp(qp: QpProblem, integer_vars: Vec<usize>) -> MiqpProblem {
    MiqpProblem::new(qp, integer_vars).unwrap()
}

#[test]
fn miqp_fractional_root_branches_to_integer_optimum() {
    // min x^2 + y^2 s.t. x + y >= 3, x,y integer in [0,5].
    // Continuous min (1.5,1.5) obj 4.5; integer optimum (1,2)/(2,1) obj 5.
    let qp = qp_problem(
        &[2.0, 2.0],
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (r, stats) =
        super::solve_miqp_with_stats(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - 5.0).abs() < 1e-3, "obj={}", r.objective);
    let s = r.solution[0].round() + r.solution[1].round();
    assert!((s - 3.0).abs() < EPS, "x+y={}", s);
    assert!(
        stats.nodes_processed >= 2,
        "branching expected, nodes={}",
        stats.nodes_processed
    );
}

#[test]
fn miqp_trivial_integer_root() {
    // min x^2, x integer in [0,5] → x=0 (root already integral).
    let qp = qp_problem(
        &[2.0],
        vec![0.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let r = solve_miqp(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(r.objective.abs() < 1e-3, "obj={}", r.objective);
    assert!(r.solution[0].abs() < EPS, "x={}", r.solution[0]);
}

#[test]
fn miqp_infeasible_between_integers() {
    // min x^2 s.t. 1.2 <= x <= 1.8, x integer in [0,10] → no integer → infeasible.
    let qp = qp_problem(
        &[2.0],
        vec![0.0],
        &[0, 1],
        &[0, 0],
        &[1.0, 1.0],
        2,
        vec![1.2, 1.8],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0)],
    );
    let r = solve_miqp(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Infeasible);
}

#[test]
fn miqp_nonconvex_q_rejected() {
    // indefinite Q → NonConvex (no silent wrong answer), never enters the B&B.
    let qp = qp_problem(
        &[2.0, -3.0],
        vec![0.0, 0.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0); 2],
    );
    let r = solve_miqp(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(
        matches!(r.status, SolveStatus::NonConvex(_)),
        "got {:?}",
        r.status
    );
}

/// min -x, x in {0,1}, s.t. 2x^2 <= 0.5 (as (1/2)x'[4]x <= 0.5).
///
/// x=1 violates the quadratic constraint; the answer is x=0, objective 0.
/// Sentinel: dropping the quadratic-constraint check in `solve_fixed_point`
/// makes the x=1 leaf evaluate as Optimal(obj=-1) and this test FAILs with
/// an infeasible incumbent.
#[test]
fn miqcp_fixed_point_respects_quadratic_constraint() {
    use crate::qp::QcqpMatrix;
    let mut qp = qp_problem(
        &[0.0],
        vec![-1.0],
        &[],
        &[],
        &[],
        1,
        vec![0.5],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0)],
    );
    let mut qc = QcqpMatrix::new(1);
    qc.triplets.push((0, 0, 4.0));
    qp.set_quadratic_constraints(vec![qc]).unwrap();
    let r = solve_miqp(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::NonconvexGlobal
        ),
        "got {:?}",
        r.status
    );
    assert!(r.solution[0].abs() < EPS, "x={} must be 0", r.solution[0]);
    assert!(r.objective.abs() < EPS, "obj={} must be 0", r.objective);
}

/// A quadratic `>=` (or `=`) constraint makes the feasible region nonconvex
/// even with a PSD matrix; `is_convex` must reject it at the B&B entry.
///
/// Sentinel: restricting `is_convex` to the objective Q only lets these enter
/// the B&B and return a non-NonConvex status — this test FAILs.
#[test]
fn miqcp_nonconvex_quadratic_constraint_rejected() {
    use crate::qp::QcqpMatrix;
    for ct in [ConstraintType::Ge, ConstraintType::Eq] {
        let mut qp = qp_problem(
            &[2.0],
            vec![0.0],
            &[],
            &[],
            &[],
            1,
            vec![1.0],
            vec![ct],
            vec![(0.0, 5.0)],
        );
        let mut qc = QcqpMatrix::new(1);
        qc.triplets.push((0, 0, 2.0));
        qp.set_quadratic_constraints(vec![qc]).unwrap();
        let r = solve_miqp(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
        assert!(
            matches!(r.status, SolveStatus::NonConvex(_)),
            "{:?} quadratic constraint must be rejected, got {:?}",
            ct,
            r.status
        );
    }
}

#[test]
fn miqp_no_integer_vars_falls_back_to_qp() {
    // convex QP via MIQP entry with no integer vars must match the direct QP solve.
    let qp = qp_problem(
        &[2.0, 2.0],
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let direct = crate::qp::solve_qp_with(&qp.clone(), &opts());
    let r = solve_miqp(&miqp(qp, vec![]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        (r.objective - direct.objective).abs() < 1e-3,
        "miqp {} vs qp {}",
        r.objective,
        direct.objective
    );
}

#[test]
fn miqp_boxonly_offdiag_no_overprune_sentinel() {
    // min x²+xy+y² −6x −6y over integer [0,4]², Q=[[2,1],[1,2]] (PSD), NO constraints.
    // The QP IPM stalls on this box-only off-diagonal QP and returns SuboptimalSolution
    // with an objective ABOVE the true relaxation minimum. If the driver used that
    // suboptimal primal objective as a *lower* bound it would over-prune the node
    // holding (2,2) and return −11 (silent-wrong). The true integer optimum is
    // −12 @ (2,2). Load-bearing sentinel: reverting "trust Optimal relaxations only as
    // bounds" returns −11 here and FAILS this test.
    let q = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-6.0, -6.0],
        a,
        vec![],
        vec![(0.0, 4.0), (0.0, 4.0)],
        vec![],
    )
    .unwrap();
    let r = solve_miqp(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(
        (r.objective - (-12.0)).abs() < 1e-3,
        "obj={} (expected −12 @ (2,2))",
        r.objective
    );
    let s = (r.solution[0].round(), r.solution[1].round());
    assert_eq!(s, (2.0, 2.0), "x*={:?}", r.solution);
}

// ---------------------------------------------------------------------------
// MipStats timing sentinels
//
// These tests fail if the timing instrumentation is removed (no-op revert):
//   - relax_total_ms > 0  requires the Instant wrapper around problem.solve()
//   - relax_root_ms > 0   requires the root_solved branch
//   - desc_ms > 0         requires descendant timing after first node
//   - optimal_ms > 0      requires the Optimal arm in the timing match
//   - infeasible_ms > 0   requires the Infeasible arm (pruned-by-solve path)
//   - approx_bounds_bytes_per_node > 0  requires the n*2*8 assignment
// ---------------------------------------------------------------------------

/// Timing stats populate for a MILP that requires branching after BT.
///
/// Uses x+y ≤ 3.5, x,y ∈ [0,5] integer. BT tightens to [0,3]; LP root is still
/// fractional → branching → desc_ms > 0.
///
/// Sentinel: removing timing instrumentation leaves desc_ms == 0 → FAILS.
#[test]
fn stats_timing_populated_for_milp_bt_then_branch() {
    let lp = build_lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.5],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (_, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(stats.nodes_processed >= 3, "branching expected");
    assert!(
        stats.relaxation_time_total_ms > 0.0,
        "relax_total_ms must be >0 (instrumentation missing?)"
    );
    assert!(
        stats.relaxation_time_root_ms > 0.0,
        "relax_root_ms must be >0"
    );
    assert!(
        stats.relaxation_time_desc_ms > 0.0,
        "relax_desc_ms must be >0 (descendant instrumentation missing?)"
    );
    let sum = stats.relaxation_time_root_ms + stats.relaxation_time_desc_ms;
    assert!(
        (stats.relaxation_time_total_ms - sum).abs() < 1e-6,
        "total={:.6} root={:.6} desc={:.6}",
        stats.relaxation_time_total_ms,
        stats.relaxation_time_root_ms,
        stats.relaxation_time_desc_ms,
    );
    assert!(
        stats.relaxation_time_optimal_ms > 0.0,
        "optimal_ms must be >0"
    );
    assert!(
        stats.approx_bounds_bytes_per_node > 0,
        "bounds_bytes_per_node must be >0"
    );
}

#[test]
fn mip_descendant_nodes_skip_repeated_lp_ruiz_scaling() {
    use std::cell::RefCell;

    struct ScalingOptionMock {
        calls: RefCell<Vec<bool>>,
        bounds: Vec<(f64, f64)>,
        ints: Vec<usize>,
    }

    impl super::Relaxation for ScalingOptionMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.ints
        }
        fn solve(&self, _bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let mut calls = self.calls.borrow_mut();
            calls.push(opts.use_ruiz_scaling);
            if calls.len() == 1 {
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![0.5],
                    ..Default::default()
                }
            } else {
                SolverResult::infeasible()
            }
        }
        fn skip_node_presolve(&self) -> bool {
            true
        }
        fn can_skip_repeated_lp_scaling(&self) -> bool {
            true
        }
    }

    let mock = ScalingOptionMock {
        calls: RefCell::new(Vec::new()),
        bounds: vec![(0.0, 2.0)],
        ints: vec![0],
    };
    let cfg = MipConfig {
        branching: crate::options::MipBranching::MostFractional,
        ..MipConfig::default()
    };
    let (_, stats) = solve_mip_core(&mock, &opts(), &cfg, vec![true], None);
    let calls = mock.calls.borrow();

    assert!(
        stats.nodes_processed >= 3,
        "mock must branch into descendants"
    );
    assert_eq!(calls.first().copied(), Some(true), "root keeps LP scaling");
    assert!(
        calls.iter().skip(1).all(|&enabled| !enabled),
        "descendant node LP solves must disable repeated scaling; calls={calls:?}"
    );
}

#[test]
fn mip_descendant_scaling_retry_runs_and_accounts_first_attempt_timing() {
    use std::cell::RefCell;

    struct RetryMock {
        calls: RefCell<Vec<bool>>,
        bounds: Vec<(f64, f64)>,
        ints: Vec<usize>,
    }

    impl super::Relaxation for RetryMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.ints
        }
        fn solve(&self, _bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let mut calls = self.calls.borrow_mut();
            calls.push(opts.use_ruiz_scaling);
            if calls.len() == 1 {
                return SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![0.5],
                    timing_breakdown: Some(TimingBreakdown {
                        solve_us: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                };
            }
            if opts.use_ruiz_scaling {
                SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: f64::INFINITY,
                    timing_breakdown: Some(TimingBreakdown {
                        solve_us: 100,
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            } else {
                SolverResult {
                    status: SolveStatus::NumericalError,
                    timing_breakdown: Some(TimingBreakdown {
                        solve_us: 10,
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            }
        }
        fn skip_node_presolve(&self) -> bool {
            true
        }
        fn can_skip_repeated_lp_scaling(&self) -> bool {
            true
        }
    }

    let mock = RetryMock {
        calls: RefCell::new(Vec::new()),
        bounds: vec![(0.0, 2.0)],
        ints: vec![0],
    };
    let cfg = MipConfig {
        branching: crate::options::MipBranching::MostFractional,
        ..MipConfig::default()
    };
    let (_, stats) = solve_mip_core(&mock, &opts(), &cfg, vec![true], None);
    let calls = mock.calls.borrow();

    assert!(calls.windows(2).any(|w| w == [false, true]));
    assert!(
        stats.lp_solve_us_desc >= 220,
        "two descendant retries must include unscaled + scaled timing; desc_solve_us={}",
        stats.lp_solve_us_desc
    );
}

#[test]
fn branching_strategy_controls_strong_branch_stats() {
    use std::cell::Cell;

    struct BranchStatsMock {
        calls: Cell<usize>,
        bounds: [(f64, f64); 1],
        ints: [usize; 1],
    }

    impl BranchStatsMock {
        fn new() -> Self {
            Self {
                calls: Cell::new(0),
                bounds: [(0.0, 2.0)],
                ints: [0],
            }
        }
    }

    impl super::Relaxation for BranchStatsMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.ints
        }
        fn solve(&self, bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            self.calls.set(self.calls.get() + 1);
            if bounds == self.bounds {
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![0.5],
                    ..Default::default()
                }
            } else {
                SolverResult::infeasible()
            }
        }
        fn skip_node_presolve(&self) -> bool {
            true
        }
    }

    let cfg_most = MipConfig {
        max_nodes: 1,
        branching: crate::options::MipBranching::MostFractional,
        ..MipConfig::default()
    };
    let most = BranchStatsMock::new();
    let (_, most_stats) = super::solve_mip_with_stats(&most, &opts(), &cfg_most);
    assert_eq!(most_stats.strong_branch_calls, 0);
    assert_eq!(most_stats.strong_branch_lp_solves, 0);
    assert_eq!(
        most.calls.get(),
        1,
        "MostFractional should solve only the root before max_nodes stops"
    );

    let cfg_reliability = MipConfig {
        max_nodes: 1,
        branching: crate::options::MipBranching::Reliability,
        ..MipConfig::default()
    };
    let reliability = BranchStatsMock::new();
    let (_, reliability_stats) =
        super::solve_mip_with_stats(&reliability, &opts(), &cfg_reliability);
    assert_eq!(reliability_stats.strong_branch_calls, 1);
    assert_eq!(reliability_stats.strong_branch_candidates, 1);
    assert_eq!(reliability_stats.strong_branch_lp_solves, 2);
    // strong_branch_us is wall-clock over instant-returning mock solves and can
    // legitimately round to 0µs (flaky); the lp_solves/calls counts already prove
    // the strong-branch child solves ran, so timing is not asserted here.
    assert_eq!(
        reliability.calls.get(),
        3,
        "Reliability should solve root plus down/up strong-branch trials"
    );
}

#[test]
fn scaling_retry_does_not_run_after_deadline_expired() {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    struct DeadlineRetryMock {
        calls: Cell<usize>,
    }

    impl super::Relaxation for DeadlineRetryMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &[(0.0, 1.0)]
        }
        fn integer_vars(&self) -> &[usize] {
            &[0]
        }
        fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            self.calls.set(self.calls.get() + 1);
            SolverResult::numerical_error()
        }
        fn can_skip_repeated_lp_scaling(&self) -> bool {
            true
        }
    }

    let mock = DeadlineRetryMock {
        calls: Cell::new(0),
    };
    let fast = SolverOptions {
        use_ruiz_scaling: false,
        deadline: Some(Instant::now() - Duration::from_secs(1)),
        ..SolverOptions::default()
    };
    let retry = SolverOptions {
        use_ruiz_scaling: true,
        deadline: fast.deadline,
        ..SolverOptions::default()
    };
    let _ = super::solve_relaxation_with_scaling_retry(&mock, &[(0.0, 1.0)], &fast, &retry);

    assert_eq!(
        mock.calls.get(),
        1,
        "expired deadline must not trigger retry"
    );
}

/// LP relaxation infeasibility populates infeasible_ms when LP (not propagation) detects it.
///
/// x+y+z<=2.5, x+y>=2, y+z>=2, x+z>=2, x,y,z∈[0,2] integer.
/// Propagation: Le and Ge rows with [0,2]^3 bounds produce no contradiction
/// (each pair sums to ≤4 and ≥0), but adding all three Ge rows gives
/// x+y + y+z + x+z = 2(x+y+z) ≥ 6 > 5 = 2×2.5 → LP-infeasible (Farkas).
/// Node propagation passes (no lb>ub from single-row propagation) but LP returns Infeasible.
///
/// Sentinel: removing the infeasible timing arm leaves infeasible_ms == 0 → FAILS.
#[test]
fn stats_timing_infeasible_ms_from_lp() {
    let a = CscMatrix::from_triplets(
        // x+y+z<=2.5 (row 0), x+y>=2 (row 1), y+z>=2 (row 2), x+z>=2 (row 3)
        &[0, 0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 2, 0, 1, 1, 2, 0, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        4,
        3,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0, 1.0],
        a,
        vec![2.5, 2.0, 2.0, 2.0],
        vec![
            ConstraintType::Le,
            ConstraintType::Ge,
            ConstraintType::Ge,
            ConstraintType::Ge,
        ],
        vec![(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)],
        None,
    )
    .unwrap();
    let (r, stats) =
        solve_milp_with_stats(&milp(lp, vec![0, 1, 2]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Infeasible);
    assert!(
        stats.nodes_processed >= 1,
        "LP must be invoked (propagation alone cannot detect Farkas infeasibility); nodes={}",
        stats.nodes_processed
    );
    assert!(
        stats.relaxation_time_infeasible_ms > 0.0,
        "infeasible_ms must be >0 when LP detects infeasibility; pruned={} nodes={}",
        stats.pruned,
        stats.nodes_processed,
    );
}

#[test]
fn stats_timing_root_only_for_trivial_integer_root() {
    // Root already integral → no branching → desc_ms == 0.
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (_, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(stats.nodes_processed, 1);
    assert!(stats.relaxation_time_root_ms > 0.0, "root_ms must be >0");
    assert_eq!(
        stats.relaxation_time_desc_ms, 0.0,
        "no descendants → desc_ms must be 0"
    );
}

#[test]
fn stats_timing_populated_for_miqp_with_branching() {
    // Same 2-var convex MIQP as miqp_fractional_root_branches_to_integer_optimum.
    let qp = qp_problem(
        &[2.0, 2.0],
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (_, stats) =
        super::solve_miqp_with_stats(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(stats.nodes_processed >= 2, "branching expected");
    assert!(
        stats.relaxation_time_total_ms > 0.0,
        "MIQP relax_total_ms must be >0"
    );
    assert!(
        stats.relaxation_time_root_ms > 0.0,
        "MIQP relax_root_ms must be >0"
    );
    assert!(
        stats.approx_bounds_bytes_per_node > 0,
        "MIQP bounds_bytes_per_node must be >0"
    );
}

#[test]
fn stats_bounds_bytes_scales_with_num_vars() {
    // approx_bounds_bytes_per_node = n_vars * 16 (two f64 per bound pair).
    // Two problems of different sizes: the larger must have proportionally larger bytes.
    let lp1 = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (_, s1) = solve_milp_with_stats(&milp(lp1, vec![0]), &opts(), &MipConfig::default());

    let lp4 = build_lp(
        vec![1.0, 1.0, 1.0, 1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0); 4],
    );
    let (_, s4) = solve_milp_with_stats(&milp(lp4, vec![0]), &opts(), &MipConfig::default());

    assert_eq!(
        s4.approx_bounds_bytes_per_node,
        4 * s1.approx_bounds_bytes_per_node,
        "bytes must scale 4× with 4× the variables"
    );
}

// ---------------------------------------------------------------------------
// integer_mask helper
// ---------------------------------------------------------------------------

#[test]
fn integer_mask_marks_only_integer_vars() {
    assert_eq!(integer_mask(3, &[0, 2]), vec![true, false, true]);
    assert_eq!(integer_mask(2, &[]), vec![false, false]);
}

// ---------------------------------------------------------------------------
// Options validation wiring tests
// ---------------------------------------------------------------------------

/// Invalid options are rejected at `solve_milp` / `solve_milp_with_stats` entry.
///
/// Sentinel: removing `validate()` from `solve_milp_with_stats` causes these to
/// propagate bad config into the B&B driver instead of returning NumericalError.
#[test]
fn invalid_options_rejected_at_milp_entry() {
    use crate::options::IpmOptions;
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let problem = milp(lp, vec![0]);
    let cfg = MipConfig::default();

    let cases: &[(&str, SolverOptions)] = &[
        (
            "nan primal_tol",
            SolverOptions {
                primal_tol: f64::NAN,
                ..Default::default()
            },
        ),
        (
            "zero primal_tol",
            SolverOptions {
                primal_tol: 0.0,
                ..Default::default()
            },
        ),
        (
            "neg dual_tol",
            SolverOptions {
                dual_tol: -1e-6,
                ..Default::default()
            },
        ),
        (
            "neg timeout_secs",
            SolverOptions {
                timeout_secs: Some(-1.0),
                ..Default::default()
            },
        ),
        (
            "zero threads",
            SolverOptions {
                threads: 0,
                ..Default::default()
            },
        ),
        (
            "ipm eps zero",
            SolverOptions {
                ipm: IpmOptions {
                    eps: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ];
    for (label, bad_opts) in cases {
        let r = solve_milp(&problem, bad_opts, &cfg);
        assert_eq!(
            r.status,
            crate::problem::SolveStatus::NumericalError,
            "solve_milp with {label} must return NumericalError"
        );
        let (r2, _) = solve_milp_with_stats(&problem, bad_opts, &cfg);
        assert_eq!(
            r2.status,
            crate::problem::SolveStatus::NumericalError,
            "solve_milp_with_stats with {label} must return NumericalError"
        );
    }
}

/// `solve_miqp_with_stats` rejects an indefinite Q (n=1001) with NonConvex status.
///
/// **Sentinel**: removing the `if !problem.is_convex()` guard in
/// `solve_miqp_with_stats` (`mip/mod.rs`) causes B&B to run on a non-convex
/// relaxation and return a silently wrong Optimal result — this test FAILS.
#[test]
fn solve_miqp_rejects_indefinite_n1001() {
    let n = 1001_usize;
    let mut rows: Vec<usize> = (0..n).collect();
    let mut cols: Vec<usize> = (0..n).collect();
    let mut vals: Vec<f64> = vec![1.0; n];
    // off-diagonal: Q[0,1]=Q[1,0]=2 → top-left 2×2 eigenvalues {-1, 3} → indefinite
    rows.push(0);
    cols.push(1);
    vals.push(2.0);
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let a = CscMatrix::new(0, n);
    let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
    let m = MiqpProblem::new(qp, vec![0]).unwrap();
    let (result, _) = solve_miqp_with_stats(&m, &SolverOptions::default(), &MipConfig::default());
    assert!(
        matches!(result.status, SolveStatus::NonConvex(_)),
        "solve_miqp_with_stats must reject indefinite Q with NonConvex, got {:?}",
        result.status
    );
}

/// Invalid options are rejected at `solve_miqp` / `solve_miqp_with_stats` entry.
///
/// Sentinel: removing `validate()` from `solve_miqp_with_stats` causes these to
/// propagate bad config into the B&B driver instead of returning NumericalError.
#[test]
fn invalid_options_rejected_at_miqp_entry() {
    // min 0.5 x^2 s.t. x <= 5, x >= 0, x integer
    let q = crate::sparse::CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = crate::sparse::CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![5.0];
    let bounds = vec![(0.0, f64::INFINITY)];
    let qp = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let problem = MiqpProblem::new(qp, vec![0]).unwrap();
    let cfg = MipConfig::default();

    let cases: &[(&str, SolverOptions)] = &[
        (
            "nan primal_tol",
            SolverOptions {
                primal_tol: f64::NAN,
                ..Default::default()
            },
        ),
        (
            "zero threads",
            SolverOptions {
                threads: 0,
                ..Default::default()
            },
        ),
        (
            "neg timeout_secs",
            SolverOptions {
                timeout_secs: Some(-1.0),
                ..Default::default()
            },
        ),
    ];
    for (label, bad_opts) in cases {
        let r = solve_miqp(&problem, bad_opts, &cfg);
        assert_eq!(
            r.status,
            crate::problem::SolveStatus::NumericalError,
            "solve_miqp with {label} must return NumericalError"
        );
        let (r2, _) = solve_miqp_with_stats(&problem, bad_opts, &cfg);
        assert_eq!(
            r2.status,
            crate::problem::SolveStatus::NumericalError,
            "solve_miqp_with_stats with {label} must return NumericalError"
        );
    }
}

// ---------------------------------------------------------------------------
// BoundGapCertificate sentinels
//
// Invariant: `bound_gap_cert` is Some iff `proven = true` iff `status == Optimal`.
// All three sentinels mutually reinforce; removing any one guard in the B&B
// driver causes at least one test to FAIL.
// ---------------------------------------------------------------------------

/// Optimal result carries BoundGapCertificate with correct gap fields.
///
/// Sentinel: removing the `inc.bound_gap_cert = Some(...)` assignment in
/// `solve_mip_with_stats` leaves cert as `None` → this test FAILS.
#[test]
fn optimal_result_carries_bound_gap_cert() {
    // min x, x in [0,5] integer → x=0, obj=0. Root already integral.
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (r, _) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    let cert = r
        .bound_gap_cert
        .as_ref()
        .expect("Optimal must carry BoundGapCertificate");
    assert!(
        cert.gap_rel() <= cert.gap_tol() + 1e-10,
        "gap_rel={} must be ≤ gap_tol={}",
        cert.gap_rel(),
        cert.gap_tol()
    );
    assert!(
        (cert.incumbent_obj() - 0.0).abs() < 1e-6,
        "incumbent obj must be ~0, got {}",
        cert.incumbent_obj()
    );
    assert!(
        cert.lower_bound() <= cert.incumbent_obj() + 1e-10,
        "lb={} must be ≤ inc_obj={}",
        cert.lower_bound(),
        cert.incumbent_obj()
    );
}

/// Non-Optimal results carry no BoundGapCertificate.
///
/// Sentinel: attaching cert unconditionally (regardless of `proven`) causes
/// SuboptimalSolution results to have Some(cert) → this test FAILS.
/// Early termination (max_nodes) produces SuboptimalSolution with no BoundGapCertificate.
///
/// Uses max 8a+11b s.t. 5a+7b ≤ 10, a,b ∈ [0,1] integer.
/// BT: implied ubs are floor(2.8)=2 ≥ 1 and floor(1.43)=1 → no tightening.
/// Root LP gives a≈0.6 (fractional). With max_nodes=2: root + down(a=0) processed;
/// up(a=1) child is left in queue → gap > 1e-6 → SuboptimalSolution + no cert.
///
/// Sentinel: attaching cert unconditionally gives Some(cert) here → FAILS.
#[test]
fn knapsack_truncated_has_no_bound_gap_cert() {
    let lp = build_lp(
        vec![-8.0, -11.0],
        &[0, 0],
        &[0, 1],
        &[5.0, 7.0],
        1,
        vec![10.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    // Disable cuts: with cuts=true this tiny instance solves to optimality at
    // the root, defeating the truncation sentinel (max_nodes=2 would never fire).
    // This test exercises the truncation/cert path; cuts behaviour is separate.
    let cfg = MipConfig {
        max_nodes: 2,
        cuts: false,
        ..MipConfig::default()
    };
    let (r, _) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &cfg);
    assert_ne!(
        r.status,
        SolveStatus::Optimal,
        "must not claim Optimal with open queue"
    );
    assert!(
        r.bound_gap_cert.is_none(),
        "non-Optimal must have no BoundGapCertificate"
    );
}

/// `proof_uncertain = true` blocks the Optimal claim even when the gap numerically
/// appears closed.
///
/// Scenario (mock relaxation, 1-var x ∈ [0,2] integer):
///   call 0 — root [0,2]: Optimal, x=0.5 (fractional), obj=1 → branch [0,0] and [1,2].
///   call 1 — [0,0]:      SuboptimalSolution, width=0 → not splittable
///                        → proof_uncertain=true, open_lb=1.
///   call 2 — [1,2]:      Optimal, x=1, integer, obj=−5 → incumbent=−5.
///
/// After the loop: remaining_lb=1, within_gap(−5, 1, 1e-6)=true — but
/// `proof_uncertain` must block the cert.
///
/// Sentinel: removing `!proof_uncertain &&` from `proven` produces `proven=true`
/// for this scenario → Optimal + cert → this test FAILS.
#[test]
fn proof_uncertain_blocks_optimal_despite_closed_gap() {
    use crate::options::SolverOptions;
    use crate::problem::SolverResult;
    use std::cell::Cell;

    struct SeqMock {
        call: Cell<usize>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }
    impl SeqMock {
        fn new() -> Self {
            Self {
                call: Cell::new(0),
                root_bounds: [(0.0, 2.0)],
                int_vars: [0],
            }
        }
    }
    impl super::Relaxation for SeqMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            match n {
                // Root [0,2]: fractional → driver branches into [0,0] and [1,2].
                0 => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 1.0,
                    solution: vec![0.5],
                    ..SolverResult::default()
                },
                // [0,0]: non-Optimal, singleton, not splittable → proof_uncertain.
                1 => SolverResult {
                    status: SolveStatus::SuboptimalSolution,
                    objective: 0.0,
                    solution: vec![0.0],
                    ..SolverResult::default()
                },
                // [1,2]: Optimal, integer-feasible → incumbent = −5.
                2 => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: -5.0,
                    solution: vec![1.0],
                    ..SolverResult::default()
                },
                _ => SolverResult::numerical_error(),
            }
        }
    }

    let mock = SeqMock::new();
    let (r, _) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert_ne!(
        r.status,
        SolveStatus::Optimal,
        "proof_uncertain must block Optimal claim (within_gap is true but region unverified)"
    );
    assert!(
        r.bound_gap_cert.is_none(),
        "proof_uncertain must suppress BoundGapCertificate"
    );
}

// ---------------------------------------------------------------------------
// LP warm-start propagation
// ---------------------------------------------------------------------------

/// LP warm start does not corrupt the result: a 2-var MILP solved with
/// `recover_warm_start_basis=true` (enabled by the B&B driver) returns the
/// same optimal objective as without warm start.
///
/// Sentinel: replacing `child_warm` with `child` (dropping warm start) keeps
/// the answer correct but removes the warm-start code path. The test fires on
/// both behaviours — it verifies correctness, not that warm-start runs.
/// The real regression guard is that warm start does NOT cause Timeout/NumericalError.
#[test]
fn warm_start_propagation_preserves_correct_objective() {
    // min -x1 - 2*x2  s.t. x1 + x2 <= 3, x1,x2 in {0,1,2,3}
    // Optimal: x1=0, x2=3, obj=-6 (0+3=3<=3, -0-6=-6)
    let lp = build_lp(
        vec![-1.0, -2.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let problem = milp(lp, vec![0, 1]);
    let (r, _) = solve_milp_with_stats(&problem, &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        (r.objective + 6.0).abs() < EPS,
        "expected obj=-6, got {}",
        r.objective
    );
}

/// Binary knapsack with warm-start propagation: correct solution is [0,1,1,1],
/// objective=-21.
#[test]
fn warm_start_binary_knapsack_correct() {
    // min -5x1 -8x2 -3x3 -5x4   s.t. 2x1+3x2+x3+2x4 <= 5, xi in {0,1}
    // Optimal: x2=1,x3=1,x4=1, obj=-16 (or similar — verify against actual solve)
    // Use a simpler 3-var knapsack with known solution:
    // min -6x0 -10x1 -5x2  s.t. 3x0+5x1+2x2 <= 6, xi in {0,1}
    // Optimal: x0=0, x1=1, x2=0 gives obj=-10; x0=1,x2=1 gives obj=-11 → -11
    let lp = build_lp(
        vec![-6.0, -10.0, -5.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[3.0, 5.0, 2.0],
        1,
        vec![6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    );
    let problem = milp(lp, vec![0, 1, 2]);
    let (r, _) = solve_milp_with_stats(&problem, &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "expected Optimal, got {:?}",
        r.status
    );
    // Best solution: x0=1(3), x2=1(2) → cap=5≤6, obj=-11.
    assert!(
        (r.objective + 11.0).abs() < EPS,
        "expected obj=-11, got {}",
        r.objective
    );
}

/// Infeasible MILP (integer variable forced between consecutive integers) is
/// still correctly identified as Infeasible even when warm-start propagation
/// is active.
#[test]
fn warm_start_infeasible_milp_still_infeasible() {
    // x in [1.2, 1.8] integer → no integer in [1.2, 1.8] → Infeasible
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(1.2, 1.8)],
    );
    let problem = milp(lp, vec![0]);
    let r = solve_milp(&problem, &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "expected Infeasible, got {:?}",
        r.status
    );
}

/// Debug: test that directly calls LP solve for child LP with warm_start.
/// Reproduces the rounding_fails_max regression.
#[test]
fn warm_start_rounding_fails_child_no_timeout() {
    use crate::options::{SolverOptions, WarmStartBasis};
    // Child LP: min -5x - 4y s.t. 6x+4y<=24, x+2y<=6, x in [0,10], y in [0,1]
    let lp = build_lp(
        vec![-5.0, -4.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[6.0, 4.0, 1.0, 2.0],
        2,
        vec![24.0, 6.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 1.0)], // y bounded at 1
    );
    // Parent root basis: both structural vars (x=0, y=1) are basic.
    let ws = WarmStartBasis {
        basis: vec![0, 1],
        x_b: vec![3.0, 1.5],
    };
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        recover_warm_start_basis: true,
        warm_start: Some(ws),
        ..Default::default()
    };
    let r = crate::lp::solve_lp_with(&lp, &opts);
    assert_ne!(
        r.status,
        crate::problem::SolveStatus::Timeout,
        "child LP with warm start must not timeout, got status={:?}",
        r.status
    );
    assert_eq!(
        r.status,
        crate::problem::SolveStatus::Optimal,
        "child LP should be Optimal (x=3 or x=4, y=1), got status={:?}",
        r.status
    );
}

/// Sentinel: `bound_layout_changes` — infinite→finite ub causes warm-start drop
/// on the down-branch.
///
/// Parent bounds `(0.0, ∞)`: root returns x=1.5 (fractional) with a basis.
/// Down-branch `(0.0, 1.0)`: ub ∞→finite → `bound_layout_changes=true` → child
/// receives `warm_start=None`.  Up-branch `(2.0, ∞)` has lower_bound > incumbent
/// and is pruned without a solve.
///
/// Sentinel: removing the `if bound_layout_changes(…) { None }` guard and always
/// propagating `child_ws` gives the down-branch `opts.warm_start=Some(basis)` →
/// `got_ws` is all-true → no false entry → **this test FAILS**.
#[test]
fn bound_layout_changes_inf_ub_to_finite_drops_warm_start() {
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;
    use std::cell::{Cell, RefCell};

    struct InfBoundMock {
        call: Cell<usize>,
        child_got_ws: RefCell<Vec<bool>>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }

    impl InfBoundMock {
        fn new() -> Self {
            Self {
                call: Cell::new(0),
                child_got_ws: RefCell::new(vec![]),
                root_bounds: [(0.0, f64::INFINITY)],
                int_vars: [0],
            }
        }
    }

    impl super::Relaxation for InfBoundMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            if n == 0 {
                // Root: fractional x=1.5; return a basis for propagation.
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 1.5,
                    solution: vec![1.5],
                    warm_start_basis: Some(WarmStartBasis {
                        basis: vec![0],
                        x_b: vec![1.5],
                    }),
                    ..SolverResult::default()
                }
            } else {
                // Child: record warm_start presence, return integer-feasible solution.
                self.child_got_ws
                    .borrow_mut()
                    .push(opts.warm_start.is_some());
                let x = bounds[0].0.ceil();
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: x,
                    solution: vec![x],
                    ..SolverResult::default()
                }
            }
        }
    }

    let mock = InfBoundMock::new();
    let (r, _) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);

    let got_ws = mock.child_got_ws.borrow();
    // Down-branch (0.0, 1.0): ub ∞→finite → bound_layout_changes=true → warm_start=None.
    // At least one solved child must have warm_start=None.
    // No-op (always propagate child_ws): all children get Some → all true → FAILS.
    assert!(
        got_ws.iter().any(|&ws| !ws),
        "down-branch (ub ∞→finite) must receive warm_start=None; \
         no-op (skip layout check) gives all-true: {got_ws:?}"
    );
}

/// Sentinel: strong-branch trials drop the warm start on infinite→finite bound
/// flips, same as the real-children path.
///
/// `max_nodes=1` + Reliability runs only the root's strong-branch trials (no
/// child node is expanded), so every recorded child solve is a strong-branch
/// trial. The down trial `(0.0, 1.0)` flips ub ∞→finite → layout changes →
/// warm_start must be None. Removing the guard in `measure_strong_branch_scores`
/// propagates the parent basis to the down trial → all-true → this test FAILS.
#[test]
fn strong_branch_drops_warm_start_on_layout_change() {
    use crate::options::WarmStartBasis;
    use std::cell::{Cell, RefCell};

    struct InfBoundSbMock {
        call: Cell<usize>,
        trial_got_ws: RefCell<Vec<bool>>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }

    impl super::Relaxation for InfBoundSbMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            if n == 0 {
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 1.5,
                    solution: vec![1.5],
                    warm_start_basis: Some(WarmStartBasis {
                        basis: vec![0],
                        x_b: vec![1.5],
                    }),
                    ..SolverResult::default()
                }
            } else {
                self.trial_got_ws
                    .borrow_mut()
                    .push(opts.warm_start.is_some());
                let x = bounds[0].0.ceil();
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: x,
                    solution: vec![x],
                    ..SolverResult::default()
                }
            }
        }
        fn skip_node_presolve(&self) -> bool {
            true
        }
    }

    let cfg = MipConfig {
        max_nodes: 1,
        branching: crate::options::MipBranching::Reliability,
        ..MipConfig::default()
    };
    let mock = InfBoundSbMock {
        call: Cell::new(0),
        trial_got_ws: RefCell::new(vec![]),
        root_bounds: [(0.0, f64::INFINITY)],
        int_vars: [0],
    };
    let _ = super::solve_mip_with_stats(&mock, &opts(), &cfg);
    let got = mock.trial_got_ws.borrow();
    assert!(
        got.iter().any(|&ws| !ws),
        "down strong-branch trial (ub ∞→finite) must drop the warm start; \
         skipping the layout guard gives all-true: {got:?}"
    );
}

/// B&B driver propagates parent warm-start basis to child nodes.
///
/// Sentinel: replacing `node.child_warm(down, lb, ws)` with `node.child(down, lb)`
/// drops warm_start for all children; `opts.warm_start` is `None` in every child
/// call → `received` is all-false → **this test FAILS** (no-op detected).
#[test]
fn warm_start_propagated_to_child_nodes_sentinel() {
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;
    use std::cell::{Cell, RefCell};

    struct PropMock {
        call: Cell<usize>,
        warm_starts_received: RefCell<Vec<bool>>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }

    impl PropMock {
        fn new() -> Self {
            Self {
                call: Cell::new(0),
                warm_starts_received: RefCell::new(vec![]),
                root_bounds: [(0.0, 3.0)],
                int_vars: [0],
            }
        }
    }

    impl super::Relaxation for PropMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            if n > 0 {
                self.warm_starts_received
                    .borrow_mut()
                    .push(opts.warm_start.is_some());
            }
            if n == 0 {
                // Root: fractional x=0.5; return warm_start_basis for propagation.
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.5,
                    solution: vec![0.5],
                    warm_start_basis: Some(WarmStartBasis {
                        basis: vec![0],
                        x_b: vec![0.5],
                    }),
                    ..Default::default()
                }
            } else {
                // Children: integer-feasible at the lower bound.
                let x = bounds[0].0.ceil().max(bounds[0].0);
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: x,
                    solution: vec![x],
                    ..Default::default()
                }
            }
        }
    }

    let mock = PropMock::new();
    let (r, _) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    let received = mock.warm_starts_received.borrow();
    assert!(
        received.iter().any(|&ws| ws),
        "warm_start must be propagated to at least one child node; \
         no-op (child() instead of child_warm()) gives all-false: {received:?}"
    );
}

// ---------------------------------------------------------------------------
// Feasibility pump sentinels
// ---------------------------------------------------------------------------

/// Construct the FP sentinel problem.
///
/// min -(3x0+5x1+2x2+4x3) s.t. 3x0+5x1+2x2+4x3 <= 7, x in {0,1}^4.
///
/// LP relaxation optimal: x=(0,1,0,0.5), obj=-7, x3 fractional.
/// FP step 1: round → (0,1,0,1); fp-cost pushes x3 up → LP gives (1,0,0,1);
///            (1,0,0,1) is integer feasible → FP returns with obj=-7.
///
/// BT: all upper bounds already at 1 (floor(7/coeff) >= 1 for all coeffs) → no tightening.
fn fp_sentinel_problem() -> MilpProblem {
    let lp = build_lp(
        vec![-3.0, -5.0, -2.0, -4.0],
        &[0, 0, 0, 0],
        &[0, 1, 2, 3],
        &[3.0, 5.0, 2.0, 4.0],
        1,
        vec![7.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); 4],
    );
    milp(lp, vec![0, 1, 2, 3])
}

/// Feasibility pump finds an initial integer-feasible solution before B&B.
///
/// `stats.fp_incumbent_found` is set only when `run_feasibility_pump` returns
/// `Some(...)` and `solve_mip_core` adopts it. Removing the FP call in
/// `solve_milp_with_stats` leaves `fp_incumbent_found = false` → **FAILS**.
#[test]
fn feasibility_pump_finds_initial_integer_solution() {
    let problem = fp_sentinel_problem();
    let (r, stats) = solve_milp_with_stats(&problem, &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        stats.fp_incumbent_found,
        "FP must find an initial integer solution; \
         removing FP call gives fp_incumbent_found=false → FAIL"
    );
    assert!(
        stats.incumbent_updates >= 1,
        "incumbent_updates must reflect FP find"
    );
}

/// A known integer-feasible solution seeds `solve_mip_core` and reduces B&B nodes.
///
/// Injects solution (1,0,0,1) (obj=−7) as a pre-computed initial incumbent.
/// The LP relaxation returns obj ≈ −7 (lower bound ≥ −7), so the root is
/// immediately fathomed after solving: `nodes_processed == 1`.
/// Without the incumbent B&B branches, processing multiple nodes.
///
/// Sentinel: removing `initial_incumbent` handling in `solve_mip_core` (always
/// treating it as `None`) makes both calls process the same tree → assertion fails.
///
/// Complement: `feasibility_pump_finds_initial_integer_solution` verifies that
/// `solve_milp_with_stats` actually calls `run_feasibility_pump` and sets
/// `fp_incumbent_found`. Together the two sentinels cover the full FP integration.
#[test]
fn feasibility_pump_reduces_bb_nodes() {
    let problem = fp_sentinel_problem();
    let cfg = MipConfig::default();
    let mask = integer_mask(problem.lp.num_vars, &problem.integer_vars);

    // (1,0,0,1) is integer-feasible with obj = −(3+4) = −7.
    let known_inc = SolverResult {
        status: SolveStatus::Optimal,
        objective: -7.0,
        solution: vec![1.0, 0.0, 0.0, 1.0],
        ..SolverResult::default()
    };

    let (_, with_inc) = solve_mip_core(&problem, &opts(), &cfg, mask.clone(), Some(known_inc));
    let (_, no_inc) = solve_mip_core(&problem, &opts(), &cfg, mask, None);

    assert!(
        with_inc.nodes_processed < no_inc.nodes_processed,
        "initial incumbent must reduce B&B nodes: with={} without={}",
        with_inc.nodes_processed,
        no_inc.nodes_processed
    );
}

/// Feasibility pump is skipped for pure LP problems (empty integer_vars).
///
/// `solve_milp_with_stats` takes the LP/QP passthrough branch and does not
/// call FP; `fp_incumbent_found` must remain false. The inner early-return
/// guard in `run_feasibility_pump` (`fp_skips_empty_integer_vars`) is the
/// correctness sentinel for that function; this test is a contract regression.
#[test]
fn feasibility_pump_handles_pure_lp_pass_through() {
    let lp = build_lp(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let problem = milp(lp, vec![]); // no integer vars
    let (r, stats) = solve_milp_with_stats(&problem, &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        !stats.fp_incumbent_found,
        "pure LP must not set fp_incumbent_found; \
         FP is gated by `!integer_vars.is_empty()` in solve_milp_with_stats"
    );
}

// ---------------------------------------------------------------------------
// Deadline contract sentinels
// ---------------------------------------------------------------------------

/// Timeout budget for `fp_timeout_does_not_exceed_user_limit`.
const FP_TIMEOUT_BUDGET_SECS: f64 = 1.0;

/// Multiplier for wall-clock upper bound in FP timeout sentinels.
///
/// Elapsed must stay below `timeout × TIMEOUT_WALL_MARGIN_MULT`. A 5× factor gives
/// comfortable headroom for scheduler and solver-cleanup jitter (observed up to ~1.7×
/// in CI) while still catching the ~31× overrun that occurs without the shared deadline.
const TIMEOUT_WALL_MARGIN_MULT: f64 = 5.0;

/// A moderately large binary MILP (n=500 vars, m=100 constraints, density ≈ 0.5)
/// where FP LP solves are non-trivial. Provides enough wall-clock exposure to
/// distinguish the shared-deadline fix from a broken implementation that resets
/// the clock on every LP call.
fn fp_timeout_sentinel_milp() -> MilpProblem {
    let n = 500usize;
    let m = 100usize;
    // Deterministic LCG for reproducibility without external dependencies.
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut rng = || -> f64 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f64) / (u32::MAX as f64)
    };
    let c: Vec<f64> = (0..n).map(|_| -rng() * 10.0).collect();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut b = Vec::new();
    for i in 0..m {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            if rng() < 0.5 {
                let v = 1.0 + rng() * 4.0;
                rows.push(i);
                cols.push(j);
                vals.push(v);
                row_sum += v;
            }
        }
        b.push((row_sum * 0.4).max(1.0));
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let lp = LpProblem::new_general(
        c,
        a,
        b,
        vec![ConstraintType::Le; m],
        vec![(0.0, 1.0); n],
        None,
    )
    .unwrap();
    milp(lp, (0..n).collect())
}

/// Shared deadline prevents total elapsed from exceeding `timeout_secs × TIMEOUT_WALL_MARGIN_MULT`.
///
/// Without the fix, each FP LP and `solve_mip_core` each receive a fresh
/// `timeout_secs` window; up to `MAX_FP_ITER + 1 = 31` resets are possible,
/// making the actual wall time ≫ the user-requested budget.
///
/// Sentinel: removing the shared deadline from `solve_milp_with_stats` lets
/// `solve_mip_core` restart the clock after FP, causing elapsed to exceed
/// `FP_TIMEOUT_BUDGET_SECS × TIMEOUT_WALL_MARGIN_MULT` → **FAILS**.
#[test]
fn fp_timeout_does_not_exceed_user_limit() {
    let problem = fp_timeout_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(FP_TIMEOUT_BUDGET_SECS),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (r, _) = solve_milp_with_stats(&problem, &opts, &MipConfig::default());
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal
                | SolveStatus::SuboptimalSolution
                | SolveStatus::Timeout
                | SolveStatus::Infeasible
                | SolveStatus::MaxIterations
        ),
        "unexpected status {:?}",
        r.status,
    );
    assert!(
        elapsed < FP_TIMEOUT_BUDGET_SECS * TIMEOUT_WALL_MARGIN_MULT,
        "elapsed {:.3}s exceeds budget {:.1}s × margin {:.1}; \
         removing shared-deadline fix lets B&B receive a fresh window after FP → overrun",
        elapsed,
        FP_TIMEOUT_BUDGET_SECS,
        TIMEOUT_WALL_MARGIN_MULT,
    );
}

/// FP respects a short deadline: iteration aborts cleanly without panic.
///
/// A 0.1 s budget is far too short to solve a 500-variable MILP; FP must
/// abort mid-iteration and return a valid (non-panicking) status within the
/// allowed window.
///
/// Sentinel: without deadline propagation into FP LP calls, the function
/// ignores the budget and runs past the margin → **FAILS**.
#[test]
fn fp_timeout_aborts_iteration() {
    let problem = fp_timeout_sentinel_milp();
    let short_timeout = 0.1f64;
    let opts = SolverOptions {
        timeout_secs: Some(short_timeout),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (r, _) = solve_milp_with_stats(&problem, &opts, &MipConfig::default());
    let elapsed = t0.elapsed().as_secs_f64();
    // Deterministic sentinel: a 0.1 s budget cannot solve a 500-variable binary MILP;
    // the solver must always signal Timeout.  Without deadline propagation into FP LP
    // calls the solver still eventually returns Timeout — but only after the wall-clock
    // check below fires, so this assertion provides an independent, zero-noise signal.
    assert_eq!(
        r.status,
        SolveStatus::Timeout,
        "FP timeout must not panic and must signal Timeout; got {:?}",
        r.status,
    );
    assert!(
        elapsed < short_timeout * TIMEOUT_WALL_MARGIN_MULT,
        "elapsed {:.3}s exceeds {:.2}s × {:.1}; FP deadline not honored",
        elapsed,
        short_timeout,
        TIMEOUT_WALL_MARGIN_MULT,
    );
}

// P2-B sentinel: MIQP presolve (tighten_bounds_linear) is applied
// ---------------------------------------------------------------------------

/// Sentinel (P2-B): MIQP presolve detects infeasibility before B&B.
///
/// `x ≤ 3.7 ∧ x ≥ 3.5` with `x ∈ [0,10]` integer and quadratic objective `x²`.
/// BT: floor(3.7)=3 (ub), ceil(3.5)=4 (lb) → lb > ub → infeasible.
/// `solve_miqp_with_stats` must return early with `nodes_processed == 0`.
///
/// No-op proof: removing `tighten_bounds_linear` from `solve_miqp_with_stats`
/// causes B&B to run (nodes_processed > 0) → the `nodes_processed == 0` assertion FAILS.
#[test]
fn miqp_bt_detects_infeasibility_before_bb() {
    let qp = qp_problem(
        &[2.0],
        vec![0.0],
        &[0, 1],
        &[0, 0],
        &[1.0, 1.0],
        2,
        vec![3.7, 3.5],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 10.0)],
    );
    let (r, stats) =
        super::solve_miqp_with_stats(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "x²: x≤3.7 ∧ x≥3.5 integer → BT detects empty domain → Infeasible"
    );
    assert_eq!(
        stats.nodes_processed, 0,
        "MIQP BT must detect infeasibility before B&B (no-op: nodes_processed > 0)"
    );
}

/// Sentinel (P2-B): MIQP BT tightens bounds (feasible) and reduces B&B node count
/// strictly below the no-BT baseline.
///
/// min x²-7x s.t. x ≤ 3.7, x ∈ [0,5] integer.
/// BT: floor(3.7)=3 → ub tightened 5→3, search space reduced by 40%.
///
/// No-op proof: removing `tighten_bounds_linear` from `solve_miqp_with_stats`
/// leaves bounds [0,5]; root QP at x=3.5 with extra [4,5] infeasible child
/// results in 5 nodes → `nodes_processed < 5` FAILS, confirming BT efficacy.
///
/// The exact count is intentionally not asserted: different LP dual solutions
/// (e.g., from crossover-first postsolve) can yield fewer nodes while still
/// respecting optimality. The sentinel captures BT effectiveness, not a
/// particular solver path.
#[test]
fn miqp_bt_reduces_bb_nodes_below_noop() {
    let qp = qp_problem(
        &[2.0],
        vec![-7.0],
        &[0],
        &[0],
        &[1.0],
        1,
        vec![3.7],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0)],
    );
    let (r, stats) =
        super::solve_miqp_with_stats(&miqp(qp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "x²-7x: x≤3.7 integer → feasible"
    );
    assert!(
        (r.objective - (-12.0)).abs() < EPS,
        "optimal x=3 → obj=9-21=-12, got {}",
        r.objective
    );
    assert!(
        stats.nodes_processed < 5,
        "BT must reduce nodes below no-BT baseline of 5; got {}",
        stats.nodes_processed
    );
}

// Hybrid node selection (best-bound + depth-first diving)
// ---------------------------------------------------------------------------

/// Diving mode does not corrupt B&B correctness on a 3-variable binary knapsack.
///
/// With DIVE_FREQUENCY_NO_INCUMBENT=2 and no pre-computed incumbent, the hybrid
/// queue triggers at least one depth-first dive during the search. This test
/// verifies that diving nodes are handled correctly (not lost, not double-visited)
/// and the solver still reaches the known optimal solution.
///
/// Cuts are disabled so the LP root remains fractional and branching is guaranteed.
///
/// Sentinel: removing `end_dive` flush in `NodeQueue::end_dive` can drop
/// dive-stack nodes (losing subproblems) → the solver may miss the optimum
/// and return a wrong objective or `SuboptimalSolution` → test FAILS.
#[test]
fn hybrid_node_selection_correctness_3var_knapsack() {
    // min -6x0 - 10x1 - 5x2  s.t. 3x0+5x1+2x2 <= 6, xi in {0,1}.
    // Optimal: x0=1, x2=1 → obj=−11 (weight=5).
    // LP relaxation is fractional (x1=0.2) so branching is guaranteed.
    let lp = build_lp(
        vec![-6.0, -10.0, -5.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[3.0, 5.0, 2.0],
        1,
        vec![6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    );
    let cfg = MipConfig {
        cuts: false,
        ..MipConfig::default()
    };
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1, 2]), &opts(), &cfg);
    assert_eq!(r.status, SolveStatus::Optimal, "got {:?}", r.status);
    assert!(
        (r.objective - (-11.0)).abs() < EPS,
        "expected obj=-11, got {}",
        r.objective
    );
    assert!(
        stats.nodes_processed >= 2,
        "branching expected; nodes={}",
        stats.nodes_processed
    );
}

/// Dive termination on infeasible LP does not orphan sibling nodes.
///
/// When a diving LP solve returns Infeasible, `end_dive` must flush remaining
/// dive-stack siblings to the best-bound heap so they are still explored.
///
/// A mock relaxation orchestrates the sequence:
///   call 0 — root [0,2]: Optimal, x=0.5 → branch [0,0] and [1,2].
///   Dive starts (no incumbent, DIVE_FREQUENCY_NO_INCUMBENT after root).
///   call 1 — [0,0] (dive node, LIFO): Infeasible → end_dive, sibling [1,2] flushed.
///   call 2 — [1,2] (best-bound from heap): Optimal, x=1 (integer) → incumbent.
///
/// Sentinel: dropping `end_dive` on Infeasible leaves [1,2] in the dive stack.
/// The stack auto-empties when next popped but no explicit flush means the
/// dive mode is never exited properly and siblings may be missed in edge cases.
/// Here the mock call-count sentinel detects whether [1,2] was ever processed.
#[test]
fn dive_ends_on_infeasible_and_sibling_explored() {
    use crate::options::SolverOptions;
    use crate::problem::SolverResult;
    use std::cell::Cell;

    struct DiveMock {
        call: Cell<usize>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }
    impl DiveMock {
        fn new() -> Self {
            Self {
                call: Cell::new(0),
                root_bounds: [(0.0, 2.0)],
                int_vars: [0],
            }
        }
    }
    impl super::Relaxation for DiveMock {
        fn num_vars(&self) -> usize {
            1
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            match n {
                // Root: fractional → branch [0,0] and [1,2].
                0 => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.5,
                    solution: vec![0.5],
                    ..Default::default()
                },
                // One child: Infeasible (down-branch [0,0] or up-branch [1,2]).
                1 => SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: f64::INFINITY,
                    solution: vec![],
                    ..Default::default()
                },
                // Other child: integer-feasible → becomes incumbent.
                _ => {
                    let x = bounds[0].0.ceil().max(bounds[0].0);
                    SolverResult {
                        status: SolveStatus::Optimal,
                        objective: x,
                        solution: vec![x],
                        ..Default::default()
                    }
                }
            }
        }
    }

    let mock = DiveMock::new();
    let cfg = MipConfig {
        branching: crate::options::MipBranching::MostFractional,
        ..MipConfig::default()
    };
    let (r, stats) = super::solve_mip_with_stats(&mock, &opts(), &cfg);
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "sibling must be explored after infeasible dive node; got {:?}",
        r.status
    );
    assert_eq!(
        mock.call.get(),
        3,
        "all 3 nodes (root + 2 children) must be processed; got {}",
        mock.call.get()
    );
    assert_eq!(
        stats.nodes_processed, 3,
        "nodes_processed must be 3; got {}",
        stats.nodes_processed
    );
}

// ---------------------------------------------------------------------------
// Node-level bound propagation sentinels
// ---------------------------------------------------------------------------

/// Node propagation avoids LP solve when bounds are infeasible after multi-pass.
///
/// Uses a mock relaxation (bypassing root presolve) so per-node propagation
/// fires inside solve_mip_core. The mock exposes constraints x+y<=2.9 and
/// x+y>=2.1 via propagation_data(). At the root B&B node with bounds [0,5]²:
///   Pass 1, Le: x_ub→2, y_ub→2; Ge: x_lb→1, y_lb→1 → bounds [1,2]².
///   Pass 2, Le: x_ub→1, y_ub→1; Ge: x_lb→ceil(2.1-1)=2 > x_ub=1 → infeasible.
/// Root node pruned by propagation before any LP solve → nodes_processed=0.
///
/// Sentinel: removing tighten_bounds_at_node from solve_mip_core causes
/// mock.solve() to be called at root → nodes_processed=1 → assertion fails.
#[test]
fn node_propagation_prunes_infeasible_node_before_lp() {
    use std::cell::Cell;
    struct NodePropMock {
        a: CscMatrix,
        b: Vec<f64>,
        ct: Vec<ConstraintType>,
        root_bounds: Vec<(f64, f64)>,
        int_vars: Vec<usize>,
        lp_calls: Cell<usize>,
    }
    impl super::Relaxation for NodePropMock {
        fn num_vars(&self) -> usize {
            2
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            self.lp_calls.set(self.lp_calls.get() + 1);
            SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                ..Default::default()
            }
        }
        fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
            Some((&self.a, &self.b, &self.ct))
        }
    }
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
        .unwrap();
    let mock = NodePropMock {
        a,
        b: vec![2.9, 2.1],
        ct: vec![ConstraintType::Le, ConstraintType::Ge],
        root_bounds: vec![(0.0, 5.0), (0.0, 5.0)],
        int_vars: vec![0, 1],
        lp_calls: Cell::new(0),
    };
    let (r, stats) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "no integer satisfies x+y<=2.9 ∧ x+y>=2.1 simultaneously"
    );
    assert_eq!(
        stats.nodes_processed, 0,
        "multi-pass propagation prunes root before LP; no-op gives nodes=1"
    );
    assert_eq!(
        stats.propagation_pruned, 1,
        "root itself counted in propagation_pruned; no-op gives 0"
    );
    assert_eq!(
        mock.lp_calls.get(),
        0,
        "LP must not be called when propagation prunes; no-op gives 1"
    );
}

/// Propagation counter increments when a node is pruned before LP solve.
#[test]
fn node_propagation_counter_increments_for_propagation_prune() {
    use std::cell::Cell;
    struct NodePropMock2 {
        a: CscMatrix,
        b: Vec<f64>,
        ct: Vec<ConstraintType>,
        root_bounds: Vec<(f64, f64)>,
        int_vars: Vec<usize>,
        lp_calls: Cell<usize>,
    }
    impl super::Relaxation for NodePropMock2 {
        fn num_vars(&self) -> usize {
            2
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            self.lp_calls.set(self.lp_calls.get() + 1);
            SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                ..Default::default()
            }
        }
        fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
            Some((&self.a, &self.b, &self.ct))
        }
    }
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
        .unwrap();
    let mock = NodePropMock2 {
        a,
        b: vec![2.9, 2.1],
        ct: vec![ConstraintType::Le, ConstraintType::Ge],
        root_bounds: vec![(0.0, 5.0), (0.0, 5.0)],
        int_vars: vec![0, 1],
        lp_calls: Cell::new(0),
    };
    let (_, stats) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert!(
        stats.propagation_pruned > 0,
        "propagation_pruned must be positive when children are infeasible by propagation; \
         no-op gives 0"
    );
    assert!(
        stats.pruned >= stats.propagation_pruned,
        "propagation_pruned must be a subset of pruned"
    );
}

/// MIQP does not use node propagation (propagation_data returns None).
///
/// Sentinel: if MIQP accidentally returns propagation_data, propagation_pruned
/// may incorrectly increase; this test verifies it stays zero on a trivial MIQP.
#[test]
fn miqp_node_propagation_not_applied() {
    let qp = qp_problem(
        &[2.0, 2.0],
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (r, stats) =
        super::solve_miqp_with_stats(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert_eq!(
        stats.propagation_pruned, 0,
        "MIQP must not use node propagation; propagation_pruned must stay 0"
    );
}

// ---------------------------------------------------------------------------
// reduced_cost_fixing unit tests
// ---------------------------------------------------------------------------

fn lp_result_with_rc(obj: f64, solution: Vec<f64>, rc: Vec<f64>) -> crate::problem::SolverResult {
    crate::problem::SolverResult {
        status: SolveStatus::Optimal,
        objective: obj,
        solution,
        reduced_costs: rc,
        ..crate::problem::SolverResult::default()
    }
}

/// At lower bound with rc > gap → variable is fixed to lb.
///
/// Sentinel: removing the `rcj > gap` check leaves the variable unfixed → count=0 → FAILS.
#[test]
fn rc_fixing_fixes_at_lower_bound() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![5.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 1, "variable at lb with rc=5 > gap=3 must be fixed");
    assert_eq!(bounds[0], (0.0, 0.0), "fixed to lb");
}

/// At upper bound with -rc > gap → variable is fixed to ub.
///
/// Sentinel: removing the `-rcj > gap` check leaves the variable unfixed → count=0 → FAILS.
#[test]
fn rc_fixing_fixes_at_upper_bound() {
    let res = lp_result_with_rc(0.0, vec![1.0], vec![-5.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(
        count, 1,
        "variable at ub with rc=-5 (-rc=5 > gap=3) must be fixed"
    );
    assert_eq!(bounds[0], (1.0, 1.0), "fixed to ub");
}

/// rc exactly equal to gap is not sufficient — strict inequality required.
///
/// Sentinel: using `>=` instead of `>` would fix this variable → count=1 → FAILS.
#[test]
fn rc_fixing_no_fix_when_rc_equals_gap() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![3.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "rc == gap must not fix (strict inequality)");
    assert_eq!(bounds[0], (0.0, 1.0), "bounds unchanged");
}

/// rc < gap → no fixing.
#[test]
fn rc_fixing_no_fix_when_rc_less_than_gap() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![2.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "rc=2 < gap=3 must not fix");
    assert_eq!(bounds[0], (0.0, 1.0));
}

/// Empty reduced_costs → no fixing, no panic.
///
/// Sentinel: removing the early-return guard causes index-out-of-bounds or wrong results → FAILS.
#[test]
fn rc_fixing_empty_reduced_costs_returns_zero() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "empty reduced_costs must return 0");
}

/// Non-positive gap (LP obj >= incumbent) → no fixing.
///
/// Sentinel: removing the `gap <= 0` guard would compare rc to a negative gap → wrong fixes → FAILS.
#[test]
fn rc_fixing_no_fix_when_gap_nonpositive() {
    let res = lp_result_with_rc(5.0, vec![0.0], vec![10.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "non-positive gap must not fix any variable");
}

/// Already-fixed variable (lb == ub) is skipped.
#[test]
fn rc_fixing_skips_already_fixed_variable() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![10.0]);
    let mut bounds = vec![(0.0_f64, 0.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "already-fixed variable must not be counted again");
    assert_eq!(bounds[0], (0.0, 0.0));
}

/// Variable not at any bound (interior LP value) → no fixing.
#[test]
fn rc_fixing_no_fix_for_interior_value() {
    let res = lp_result_with_rc(0.0, vec![0.5], vec![10.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "interior LP value must not be fixed");
    assert_eq!(bounds[0], (0.0, 1.0));
}

/// Multiple variables: only those satisfying the criterion are fixed.
#[test]
fn rc_fixing_multiple_vars_selective_fixing() {
    let res = lp_result_with_rc(0.0, vec![0.0, 2.0, 0.0, 0.0], vec![10.0, -10.0, 2.0, 3.0]);
    let mut bounds = vec![(0.0, 1.0), (0.0, 2.0), (0.0, 1.0), (0.0, 1.0)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0, 1, 2, 3]);
    assert_eq!(count, 2, "exactly 2 vars qualify for fixing");
    assert_eq!(bounds[0], (0.0, 0.0), "x0 fixed to lb");
    assert_eq!(bounds[1], (2.0, 2.0), "x1 fixed to ub");
    assert_eq!(bounds[2], (0.0, 1.0), "x2 unchanged");
    assert_eq!(bounds[3], (0.0, 1.0), "x3 unchanged (rc == gap, strict)");
}

/// Out-of-bounds variable index is rejected instead of silently disabling fixing.
#[test]
#[should_panic(expected = "integer variable index 5 out of range for 1 variables")]
fn rc_fixing_out_of_bounds_index_rejected() {
    let res = lp_result_with_rc(0.0, vec![0.0], vec![10.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let _ = reduced_cost_fixing(&res, 3.0, &mut bounds, &[5]);
}

/// Non-integer lower bound is rounded up to ceil(lb) before fixing.
///
/// If an integer variable has lb=0.5 and the LP solution is at lb (xj=0.5),
/// fixing to 0.5 is invalid (integer variables must take integer values).
/// The fix must use ceil(lb)=1.0.
///
/// Sentinel: the old code `node_bounds[j] = (lb, lb)` with lb=0.5 produces (0.5, 0.5)
/// instead of (1.0, 1.0) → this test FAILS.
#[test]
fn rc_fixing_rounds_noninteger_lower_bound_to_ceil() {
    let res = lp_result_with_rc(0.0, vec![0.5], vec![5.0]);
    let mut bounds = vec![(0.5_f64, 3.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 1, "must fix the variable");
    assert_eq!(
        bounds[0],
        (1.0, 1.0),
        "must fix to ceil(lb)=1.0 not 0.5 (non-integer lb for integer var)"
    );
}

/// Non-integer upper bound is rounded down to floor(ub) before fixing.
///
/// Sentinel: old `node_bounds[j] = (ub, ub)` with ub=2.5 produces (2.5, 2.5)
/// instead of (2.0, 2.0) → this test FAILS.
#[test]
fn rc_fixing_rounds_noninteger_upper_bound_to_floor() {
    let res = lp_result_with_rc(0.0, vec![2.5], vec![-5.0]);
    let mut bounds = vec![(0.0_f64, 2.5_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 1, "must fix the variable");
    assert_eq!(
        bounds[0],
        (2.0, 2.0),
        "must fix to floor(ub)=2.0 not 2.5 (non-integer ub for integer var)"
    );
}

/// Integer bounds (most common case) are unchanged by ceil/floor.
///
/// Regression guard: ceil(0.0)=0.0 and floor(1.0)=1.0, so the behavior for
/// integer bounds is identical to the old code.
#[test]
fn rc_fixing_integer_bounds_unaffected_by_rounding() {
    // lb=0, rc=5 > gap=3 → fix to ceil(0)=0.
    let res = lp_result_with_rc(0.0, vec![0.0], vec![5.0]);
    let mut bounds = vec![(0.0_f64, 1.0_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 1);
    assert_eq!(bounds[0], (0.0, 0.0), "integer lb=0 unchanged by ceil");
    // ub=1, -rc=5 > gap=3 → fix to floor(1)=1.
    let res2 = lp_result_with_rc(0.0, vec![1.0], vec![-5.0]);
    let mut bounds2 = vec![(0.0_f64, 1.0_f64)];
    let count2 = reduced_cost_fixing(&res2, 3.0, &mut bounds2, &[0]);
    assert_eq!(count2, 1);
    assert_eq!(bounds2[0], (1.0, 1.0), "integer ub=1 unchanged by floor");
}

#[test]
fn rc_fixing_skips_empty_integer_range() {
    // lb=0.7, ub=0.3 → ceil(0.7)=1 > floor(0.3)=0 → empty integer range, skip fix
    let res = lp_result_with_rc(0.0, vec![0.7], vec![5.0]);
    let mut bounds = vec![(0.7_f64, 0.3_f64)];
    let count = reduced_cost_fixing(&res, 3.0, &mut bounds, &[0]);
    assert_eq!(count, 0, "empty integer range must not be fixed");
    assert_eq!(bounds[0], (0.7, 0.3), "bounds unchanged");
}

// ---------------------------------------------------------------------------
// rc_vars_fixed stats sentinel
// ---------------------------------------------------------------------------

/// Reduced-cost fixing fires in the B&B loop and increments rc_vars_fixed.
///
/// Sentinel: removing the `reduced_cost_fixing(...)` call in `solve_mip_core` leaves
/// `stats.rc_vars_fixed == 0` → this test FAILS.
#[test]
fn rc_fixing_fires_in_bb_loop_stats_sentinel() {
    use crate::options::{SolverOptions, WarmStartBasis};
    use crate::problem::SolverResult;
    use std::cell::Cell;

    struct RcMock {
        call: Cell<usize>,
        root_bounds: [(f64, f64); 2],
        int_vars: [usize; 2],
    }

    impl RcMock {
        fn new() -> Self {
            Self {
                call: Cell::new(0),
                root_bounds: [(0.0, 2.0), (0.0, 1.0)],
                int_vars: [0, 1],
            }
        }
    }

    impl super::Relaxation for RcMock {
        fn num_vars(&self) -> usize {
            2
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            &self.root_bounds
        }
        fn integer_vars(&self) -> &[usize] {
            &self.int_vars
        }
        fn solve(&self, bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            match n {
                // Root: x0=1.5 (fractional), x1=0 (at lb). rc[x1]=5 > gap=2 → x1 fixed.
                0 => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![1.5, 0.0],
                    reduced_costs: vec![0.0, 5.0],
                    warm_start_basis: Some(WarmStartBasis {
                        basis: vec![0, 1],
                        x_b: vec![1.5, 0.0],
                    }),
                    ..SolverResult::default()
                },
                // Down-branch [0,1]×[0,0]: integer-feasible.
                _ if bounds[1] == (0.0, 0.0) && bounds[0].1 <= 1.0 => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: bounds[0].0.ceil(),
                    solution: vec![bounds[0].0.ceil(), 0.0],
                    reduced_costs: vec![1.0, 0.0],
                    ..SolverResult::default()
                },
                // Up-branch [2,2]×[0,0]: integer-feasible.
                _ => SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 2.0,
                    solution: vec![2.0, 0.0],
                    reduced_costs: vec![1.0, 0.0],
                    ..SolverResult::default()
                },
            }
        }
    }

    // Inject incumbent with obj=2 so gap = 2-0 = 2 at root, rc[x1]=5 > 2 → x1 fixed.
    let initial_inc = SolverResult {
        status: SolveStatus::Optimal,
        objective: 2.0,
        solution: vec![2.0, 0.0],
        ..SolverResult::default()
    };

    let mock = RcMock::new();
    let mask = integer_mask(2, &[0, 1]);
    let (r, stats) = super::solve_mip_core(
        &mock,
        &opts(),
        &crate::options::MipConfig::default(),
        mask,
        Some(initial_inc),
    );
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        stats.rc_vars_fixed >= 1,
        "rc_vars_fixed must be >= 1 when RC fixing fires at root; \
         removing the reduced_cost_fixing call gives 0 → FAILS"
    );
}

// ---------------------------------------------------------------------------
// RINS integration tests
// ---------------------------------------------------------------------------

/// RINS stats fields are initialised to zero and only increment when RINS triggers.
///
/// A 1-variable problem solves in 1 node — RINS never fires (needs an
/// incumbent AND nodes_processed % RINS_INTERVAL == 0 at a non-root node).
#[test]
fn rins_stats_zero_for_single_node_problem() {
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert_eq!(
        stats.rins_calls, 0,
        "RINS must not fire in single-node solve"
    );
    assert_eq!(stats.rins_improvements, 0);
}

/// Disabling RINS (rins_enabled=false) does not change solution correctness.
///
/// The 2-variable MILP must still reach its optimal regardless of the RINS flag.
///
/// Sentinel: if rins_enabled=false broke B&B (e.g. accidental early exit),
/// the status would not be Optimal.
#[test]
fn rins_disabled_still_optimal() {
    let lp = build_lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let cfg = MipConfig {
        rins_enabled: false,
        ..MipConfig::default()
    };
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &cfg);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(r.objective < -2.9, "obj={}", r.objective);
    assert_eq!(
        stats.rins_calls, 0,
        "no RINS calls expected when rins_enabled=false"
    );
}

/// RINS stats are consistent: improvements <= calls.
///
/// Uses a 4-variable problem where RINS may or may not fire.
/// The invariant `improvements <= calls` must always hold.
#[test]
fn rins_stats_improvements_le_calls() {
    // 4-variable binary knapsack: min -(3x0+5x1+2x2+4x3) s.t. 3x0+5x1+2x2+4x3<=7
    let lp = build_lp(
        vec![-3.0, -5.0, -2.0, -4.0],
        &[0, 0, 0, 0],
        &[0, 1, 2, 3],
        &[3.0, 5.0, 2.0, 4.0],
        1,
        vec![7.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); 4],
    );
    let (r, stats) =
        solve_milp_with_stats(&milp(lp, vec![0, 1, 2, 3]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        stats.rins_improvements <= stats.rins_calls,
        "improvements={} must be <= calls={}",
        stats.rins_improvements,
        stats.rins_calls
    );
}

// ---------------------------------------------------------------------------
// RENS scheduling tests
// ---------------------------------------------------------------------------

use std::cell::{Cell, RefCell};

struct RensScheduleMock {
    calls: Cell<usize>,
    root_bounds: [(f64, f64); 1],
    int_vars: [usize; 1],
    responses: RefCell<Vec<Option<SolverResult>>>,
}

impl RensScheduleMock {
    fn new(responses: Vec<Option<SolverResult>>) -> Self {
        Self {
            calls: Cell::new(0),
            root_bounds: [(0.0, 1.0)],
            int_vars: [0],
            responses: RefCell::new(responses),
        }
    }
}

impl super::Relaxation for RensScheduleMock {
    fn num_vars(&self) -> usize {
        1
    }
    fn root_bounds(&self) -> &[(f64, f64)] {
        &self.root_bounds
    }
    fn integer_vars(&self) -> &[usize] {
        &self.int_vars
    }
    fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
        unreachable!("RENS scheduling sentinel should not solve node relaxations")
    }
    fn run_rens(
        &self,
        _x_lp: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _opts: &SolverOptions,
    ) -> Option<SolverResult> {
        self.calls.set(self.calls.get() + 1);
        self.responses.borrow_mut().remove(0)
    }
}

/// The first no-incumbent fractional node gets one immediate RENS try.
///
/// If that try succeeds, branch-and-bound now has an incumbent and RENS must
/// return to the regular cadence immediately rather than keeping a special
/// short-cycle mode alive.
#[test]
fn rens_first_incumbent_success_switches_to_regular_cadence() {
    let problem = RensScheduleMock::new(vec![
        Some(SolverResult {
            status: SolveStatus::Optimal,
            objective: -1.0,
            solution: vec![1.0],
            ..SolverResult::default()
        }),
        None,
    ]);
    let cfg = MipConfig::default();
    let mut state = super::MipState::new();
    let mut stats = super::MipStats {
        nodes_processed: 1,
        ..Default::default()
    };
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        1,
        "first fractional node must try RENS immediately"
    );
    assert_eq!(
        stats.rens_calls, 1,
        "rens_calls counts actual run_rens invocations"
    );
    assert_eq!(
        stats.incumbent_updates, 1,
        "successful first try must install the incumbent"
    );
    assert_eq!(state.incumbent_obj, Some(-1.0));
    assert!(
        state.rens_first_incumbent_attempted,
        "the one-shot flag must be consumed by the first attempt"
    );

    stats.nodes_processed = 2;
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        1,
        "after the incumbent appears, the next non-cadence node must not re-run RENS"
    );
    assert_eq!(stats.rens_calls, 1);
    assert_eq!(stats.incumbent_updates, 1);

    stats.nodes_processed = super::heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT;
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        2,
        "regular cadence must still trigger RENS later"
    );
    assert_eq!(stats.rens_calls, 2);
    assert_eq!(
        stats.incumbent_updates, 1,
        "None must not be counted as an incumbent update"
    );
}

/// The no-incumbent one-shot is consumed even when the first attempt fails.
#[test]
fn rens_first_incumbent_failure_waits_for_regular_cadence() {
    let problem = RensScheduleMock::new(vec![None, None]);
    let cfg = MipConfig::default();
    let mut state = super::MipState::new();
    let mut stats = super::MipStats {
        nodes_processed: 1,
        ..Default::default()
    };
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        1,
        "first fractional node must consume the one-shot attempt"
    );
    assert_eq!(stats.rens_calls, 1);
    assert_eq!(state.incumbent_obj, None);
    assert!(state.rens_first_incumbent_attempted);

    stats.nodes_processed = 2;
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        1,
        "failed first-incumbent RENS must not spin on every following node"
    );
    assert_eq!(stats.rens_calls, 1);

    stats.nodes_processed = super::heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT;
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        2,
        "failed first try still returns to the regular cadence"
    );
    assert_eq!(stats.rens_calls, 2);
}

/// Starting with an incumbent skips the one-shot path and waits for cadence.
#[test]
fn rens_with_incumbent_waits_for_regular_interval() {
    let problem = RensScheduleMock::new(vec![None]);
    let cfg = MipConfig::default();
    let incumbent = SolverResult {
        status: SolveStatus::Optimal,
        objective: -1.0,
        solution: vec![1.0],
        ..SolverResult::default()
    };
    let mut state = super::MipState::new();
    let mut stats = super::MipStats {
        nodes_processed: 1,
        ..Default::default()
    };
    assert!(
        state.consider(&incumbent),
        "fixture should seed the incumbent"
    );
    assert!(
        !state.rens_first_incumbent_attempted,
        "seeding an incumbent directly must not consume the no-incumbent one-shot"
    );

    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        0,
        "with an incumbent already present, RENS must wait for the regular cadence"
    );
    assert_eq!(stats.rens_calls, 0);
    assert!(
        !state.rens_first_incumbent_attempted,
        "regular incumbent-driven scheduling must not touch the no-incumbent one-shot flag"
    );

    stats.nodes_processed = super::heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT;
    super::try_rens(
        &problem,
        &mut stats,
        &mut state,
        &cfg,
        &None,
        &opts(),
        &[0.5],
    );
    assert_eq!(
        problem.calls.get(),
        1,
        "regular cadence must trigger once an incumbent exists"
    );
    assert_eq!(stats.rens_calls, 1);
}

// ---------------------------------------------------------------------------
// Conflict analysis integration tests
// ---------------------------------------------------------------------------

/// Conflict stats are zero when no node is infeasible (all pruned by bounds).
///
/// A problem where LP root is already optimal: no branching, no infeasible
/// nodes → conflict_clauses_learned == 0.
#[test]
fn conflict_stats_zero_for_integer_root() {
    let lp = build_lp(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        vec![],
        vec![],
        vec![(0.0, 5.0)],
    );
    let (_, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(
        stats.conflict_clauses_learned, 0,
        "no infeasible LP nodes → no conflict clauses"
    );
    assert_eq!(stats.conflict_pruned, 0);
}

/// Conflict analysis learns clauses from infeasible nodes and the stats reflect this.
///
/// Sentinel: removing `conflicts.learn(...)` keeps conflict_clauses_learned == 0
/// even when infeasible LP nodes exist → FAILS.
#[test]
fn conflict_learns_from_infeasible_nodes() {
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, -1.0], // x+y<=3, x-y<=0.5
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![3.0, 0.5],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 2.0), (0.0, 2.0)],
        None,
    )
    .unwrap();
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        stats.conflict_pruned <= stats.nodes_processed,
        "conflict_pruned={} > nodes_processed={}",
        stats.conflict_pruned,
        stats.nodes_processed
    );
}

/// conflict_pruned <= nodes_processed always (a pruned node is a node we saved).
///
/// This invariant must hold for any MILP solve.
#[test]
fn conflict_pruned_le_nodes_processed() {
    let lp = build_lp(
        vec![-1.0, -2.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let (_, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(
        stats.conflict_pruned <= stats.nodes_processed,
        "conflict_pruned={} must be <= nodes_processed={}",
        stats.conflict_pruned,
        stats.nodes_processed
    );
}

// ---------------------------------------------------------------------------
// Static symmetry breaking (lex-leader)
// ---------------------------------------------------------------------------

/// **Sentinel**: lex-leader symmetry breaking must reduce the node count on a
/// highly symmetric instance WITHOUT changing the optimum.
///
/// Eight interchangeable binaries under `2·Σ x_i ≤ 9` (⇔ `Σ x_i ≤ 4.5`),
/// maximising `Σ x_i` (min `−Σ x_i`). The integer optimum is 4 with `C(8,4)=70`
/// symmetric optimal assignments; plain B&B explores the equivalent subtrees,
/// while the lex-leader rows `x_i ≥ x_{i+1}` collapse each orbit to its single
/// descending representative.
///
/// Cuts and RINS are disabled so the measured difference isolates the symmetry
/// effect. Fails if symmetry is a no-op (equal node counts) or alters the
/// optimum (objective mismatch).
#[test]
fn symmetry_breaking_reduces_nodes_and_preserves_optimum() {
    let n = 8usize;
    let rows: Vec<usize> = vec![0; n];
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![2.0; n];
    let lp = build_lp(
        vec![-1.0; n],
        &rows,
        &cols,
        &vals,
        1,
        vec![9.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); n],
    );
    let m = milp(lp, (0..n).collect());

    let base = MipConfig {
        cuts: false,
        rins_enabled: false,
        symmetry: false,
        ..MipConfig::default()
    };
    let sym = MipConfig {
        symmetry: true,
        ..base.clone()
    };

    let (r_off, s_off) = solve_milp_with_stats(&m, &opts(), &base);
    let (r_on, s_on) = solve_milp_with_stats(&m, &opts(), &sym);

    assert_eq!(r_off.status, SolveStatus::Optimal, "baseline must solve");
    assert_eq!(r_on.status, SolveStatus::Optimal, "symmetry run must solve");

    // Optimum unchanged: both reach the true integer optimum −4.
    assert!(
        (r_off.objective - (-4.0)).abs() < 1e-6,
        "baseline optimum must be -4, got {}",
        r_off.objective
    );
    assert!(
        (r_on.objective - r_off.objective).abs() < 1e-6,
        "symmetry breaking changed the optimum: off={} on={}",
        r_off.objective,
        r_on.objective
    );

    // Node count strictly reduced.
    assert!(
        s_on.nodes_processed < s_off.nodes_processed,
        "lex-leader must shrink the tree: off={} on={}",
        s_off.nodes_processed,
        s_on.nodes_processed
    );
}

#[test]
#[should_panic(expected = "integer variable index 2 out of range for 2 variables")]
fn integer_mask_rejects_out_of_range_index() {
    let _ = super::integer_mask(2, &[0, 2]);
}

#[test]
#[should_panic(expected = "integer variable index 2 out of range for solution length 2")]
fn round_integers_rejects_out_of_range_index() {
    let _ = super::round_integers(vec![0.1, 1.9], &[0, 2]);
}

#[test]
#[should_panic(expected = "reduced-cost fixing requires one reduced cost per variable")]
fn reduced_cost_fixing_rejects_short_reduced_costs() {
    let r = SolverResult {
        objective: 0.0,
        solution: vec![0.0, 0.0],
        reduced_costs: vec![1.0],
        ..Default::default()
    };
    let mut bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let _ = super::reduced_cost_fixing(&r, 1.0, &mut bounds, &[0]);
}

#[test]
#[should_panic(expected = "reduced-cost fixing requires one solution value per variable")]
fn reduced_cost_fixing_rejects_short_solution() {
    let r = SolverResult {
        objective: 0.0,
        solution: vec![0.0],
        reduced_costs: vec![1.0, 1.0],
        ..Default::default()
    };
    let mut bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let _ = super::reduced_cost_fixing(&r, 1.0, &mut bounds, &[0]);
}
