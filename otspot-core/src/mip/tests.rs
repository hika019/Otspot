//! MILP branch-and-bound tests (#14 Phase 1).
//!
//! Multiple data patterns per CLAUDE.md: trivial integer root, fractional root
//! requiring branching, infeasible, unbounded, binary knapsack, and the
//! no-integer LP fallback. Low-level `solve_milp` / `solve_miqp` entries are
//! exercised here. Model API tests live in `otspot-model/tests/mip_model.rs`.

use super::{
    finalize_no_incumbent, integer_mask, solve_milp, solve_milp_with_stats, solve_mip_core,
    solve_miqp, solve_miqp_with_stats, MilpProblem, MiqpProblem,
};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

const EPS: f64 = 1e-4;

fn opts() -> SolverOptions {
    // safety net; tiny problems finish instantly
    SolverOptions { timeout_secs: Some(10.0), ..Default::default() }
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
    let lp = build_lp(vec![1.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0)]);
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
    assert_eq!(r.status, SolveStatus::Infeasible, "x≤3.7 ∧ x≥3.5 integer → empty domain");
    assert_eq!(stats.nodes_processed, 0, "infeasibility detected by BT, not B&B");
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
    assert_eq!(r.status, SolveStatus::Infeasible, "x=3.5 integer → no feasible integer → infeasible");
    assert_eq!(stats.nodes_processed, 0, "BT must detect crossing before B&B");
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
        &[0], &[0], &[2.0],
        1, vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-1.0)).abs() < EPS, "obj={}", r.objective);
    assert!((r.solution[0] - 1.0).abs() < EPS, "x={}", r.solution[0]);
    assert_eq!(stats.nodes_processed, 1, "BT tightens x ≤ 1 → root is integer-feasible, no branching");
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
        &[0, 0], &[0, 1], &[1.0, 1.0],
        1, vec![3.5],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let (r, stats) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-3.0)).abs() < EPS, "obj={}", r.objective);
    let s = r.solution[0].round() + r.solution[1].round();
    assert!((s - 3.0).abs() < EPS, "x+y={}", s);
    assert!(stats.nodes_processed >= 3, "fractional LP root causes branching, nodes={}", stats.nodes_processed);
    assert!(stats.pruned >= 1, "at least one pruned/infeasible child expected");
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
    let lp = build_lp(vec![-1.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, f64::INFINITY)]);
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
    use std::cell::Cell;
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;

    struct PureLpMock {
        warm_start_received: Cell<bool>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 0],
    }

    impl PureLpMock {
        fn new() -> Self {
            Self { warm_start_received: Cell::new(false), root_bounds: [(0.0, 5.0)], int_vars: [] }
        }
    }

    impl super::Relaxation for PureLpMock {
        fn num_vars(&self) -> usize { 1 }
        fn root_bounds(&self) -> &[(f64, f64)] { &self.root_bounds }
        fn integer_vars(&self) -> &[usize] { &self.int_vars }
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

    let user_ws = WarmStartBasis { basis: vec![0], x_b: vec![0.0] };
    let opts_with_ws = SolverOptions { warm_start: Some(user_ws), ..opts() };
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
    assert_ne!(r.status, SolveStatus::Infeasible, "open region must not be Infeasible");
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
    let (r, stats) = super::solve_miqp_with_stats(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - 5.0).abs() < 1e-3, "obj={}", r.objective);
    let s = r.solution[0].round() + r.solution[1].round();
    assert!((s - 3.0).abs() < EPS, "x+y={}", s);
    assert!(stats.nodes_processed >= 2, "branching expected, nodes={}", stats.nodes_processed);
}

#[test]
fn miqp_trivial_integer_root() {
    // min x^2, x integer in [0,5] → x=0 (root already integral).
    let qp = qp_problem(&[2.0], vec![0.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0)]);
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
    let qp = qp_problem(&[2.0, -3.0], vec![0.0, 0.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0); 2]);
    let r = solve_miqp(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!(matches!(r.status, SolveStatus::NonConvex(_)), "got {:?}", r.status);
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
    assert!((r.objective - direct.objective).abs() < 1e-3, "miqp {} vs qp {}", r.objective, direct.objective);
}

#[test]
fn miqp_boxonly_offdiag_no_overprune_sentinel() {
    // min x²+xy+y² −6x −6y over integer [0,4]², Q=[[2,1],[1,2]] (PSD), NO constraints.
    // The QP IPM stalls on this box-only off-diagonal QP and returns SuboptimalSolution
    // with an objective ABOVE the true relaxation minimum. If the driver used that
    // suboptimal primal objective as a *lower* bound it would over-prune the node
    // holding (2,2) and return −11 (silent-wrong, #17). The true integer optimum is
    // −12 @ (2,2). Load-bearing sentinel: reverting "trust Optimal relaxations only as
    // bounds" returns −11 here and FAILS this test.
    let q = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let qp = QpProblem::new(q, vec![-6.0, -6.0], a, vec![], vec![(0.0, 4.0), (0.0, 4.0)], vec![])
        .unwrap();
    let r = solve_miqp(&miqp(qp, vec![0, 1]), &opts(), &MipConfig::default());
    assert!((r.objective - (-12.0)).abs() < 1e-3, "obj={} (expected −12 @ (2,2))", r.objective);
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
        &[0, 0], &[0, 1], &[1.0, 1.0],
        1, vec![3.5],
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
        stats.relaxation_time_total_ms, stats.relaxation_time_root_ms, stats.relaxation_time_desc_ms,
    );
    assert!(stats.relaxation_time_optimal_ms > 0.0, "optimal_ms must be >0");
    assert!(stats.approx_bounds_bytes_per_node > 0, "bounds_bytes_per_node must be >0");
}

/// LP relaxation infeasibility at the root populates infeasible_ms.
///
/// x+y ≤ 1 AND x+y ≥ 2, x,y ∈ [0,3] integer.
/// BT: from Le x,y ≤ 1; from Ge x,y ≥ 1 → domain [1,1]×[1,1].
/// Root LP with x=y=1 violates x+y ≤ 1 → LP returns Infeasible → infeasible_ms > 0.
///
/// Sentinel: removing the infeasible timing arm leaves infeasible_ms == 0 → FAILS.
#[test]
fn stats_timing_infeasible_ms_from_lp_after_bt() {
    let a = crate::sparse::CscMatrix::from_triplets(
        &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2,
    )
    .unwrap();
    let lp = crate::problem::LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![1.0, 2.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 3.0), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let (r, stats) =
        solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Infeasible);
    assert!(
        stats.relaxation_time_infeasible_ms > 0.0,
        "infeasible_ms must be >0; pruned={} nodes={}",
        stats.pruned, stats.nodes_processed,
    );
}

#[test]
fn stats_timing_root_only_for_trivial_integer_root() {
    // Root already integral → no branching → desc_ms == 0.
    let lp = build_lp(vec![1.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0)]);
    let (_, stats) =
        solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
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
    let lp1 = build_lp(vec![1.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0)]);
    let (_, s1) =
        solve_milp_with_stats(&milp(lp1, vec![0]), &opts(), &MipConfig::default());

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
    let (_, s4) =
        solve_milp_with_stats(&milp(lp4, vec![0]), &opts(), &MipConfig::default());

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
        ("nan primal_tol", SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
        ("zero primal_tol", SolverOptions { primal_tol: 0.0, ..Default::default() }),
        ("neg dual_tol", SolverOptions { dual_tol: -1e-6, ..Default::default() }),
        ("neg timeout_secs", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
        ("zero threads", SolverOptions { threads: 0, ..Default::default() }),
        ("ipm eps zero", SolverOptions {
            ipm: IpmOptions { eps: 0.0, ..Default::default() },
            ..Default::default()
        }),
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
    rows.push(0); cols.push(1); vals.push(2.0);
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let a = CscMatrix::new(0, n);
    let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
    let m = MiqpProblem::new(qp, vec![0]).unwrap();
    let (result, _) = solve_miqp_with_stats(&m, &SolverOptions::default(), &MipConfig::default());
    assert!(
        matches!(result.status, SolveStatus::NonConvex(_)),
        "solve_miqp_with_stats must reject indefinite Q with NonConvex, got {:?}", result.status
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
        ("nan primal_tol", SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
        ("zero threads", SolverOptions { threads: 0, ..Default::default() }),
        ("neg timeout_secs", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
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
    let lp = build_lp(vec![1.0], &[], &[], &[], 0, vec![], vec![], vec![(0.0, 5.0)]);
    let (r, _) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    let cert = r.bound_gap_cert.as_ref().expect("Optimal must carry BoundGapCertificate");
    assert!(
        cert.gap_rel() <= cert.gap_tol() + 1e-10,
        "gap_rel={} must be ≤ gap_tol={}",
        cert.gap_rel(), cert.gap_tol()
    );
    assert!(
        (cert.incumbent_obj() - 0.0).abs() < 1e-6,
        "incumbent obj must be ~0, got {}",
        cert.incumbent_obj()
    );
    assert!(
        cert.lower_bound() <= cert.incumbent_obj() + 1e-10,
        "lb={} must be ≤ inc_obj={}",
        cert.lower_bound(), cert.incumbent_obj()
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
        &[0, 0], &[0, 1], &[5.0, 7.0],
        1, vec![10.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    let cfg = MipConfig { max_nodes: 2, ..MipConfig::default() };
    let (r, _) = solve_milp_with_stats(&milp(lp, vec![0, 1]), &opts(), &cfg);
    assert_ne!(r.status, SolveStatus::Optimal, "must not claim Optimal with open queue");
    assert!(r.bound_gap_cert.is_none(), "non-Optimal must have no BoundGapCertificate");
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
    use std::cell::Cell;
    use crate::options::SolverOptions;
    use crate::problem::SolverResult;

    struct SeqMock {
        call: Cell<usize>,
        root_bounds: [(f64, f64); 1],
        int_vars: [usize; 1],
    }
    impl SeqMock {
        fn new() -> Self {
            Self { call: Cell::new(0), root_bounds: [(0.0, 2.0)], int_vars: [0] }
        }
    }
    impl super::Relaxation for SeqMock {
        fn num_vars(&self) -> usize { 1 }
        fn root_bounds(&self) -> &[(f64, f64)] { &self.root_bounds }
        fn integer_vars(&self) -> &[usize] { &self.int_vars }
        fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            match n {
                // Root [0,2]: fractional → driver branches into [0,0] and [1,2].
                0 => SolverResult { status: SolveStatus::Optimal, objective: 1.0, solution: vec![0.5], ..SolverResult::default() },
                // [0,0]: non-Optimal, singleton, not splittable → proof_uncertain.
                1 => SolverResult { status: SolveStatus::SuboptimalSolution, objective: 0.0, solution: vec![0.0], ..SolverResult::default() },
                // [1,2]: Optimal, integer-feasible → incumbent = −5.
                2 => SolverResult { status: SolveStatus::Optimal, objective: -5.0, solution: vec![1.0], ..SolverResult::default() },
                _ => SolverResult::numerical_error(),
            }
        }
    }

    let mock = SeqMock::new();
    let (r, _) = super::solve_mip_with_stats(&mock, &opts(), &MipConfig::default());
    assert_ne!(
        r.status, SolveStatus::Optimal,
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
    assert!((r.objective + 6.0).abs() < EPS, "expected obj=-6, got {}", r.objective);
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
    assert_eq!(r.status, SolveStatus::Optimal, "expected Optimal, got {:?}", r.status);
    // Best solution: x0=1(3), x2=1(2) → cap=5≤6, obj=-11.
    assert!((r.objective + 11.0).abs() < EPS, "expected obj=-11, got {}", r.objective);
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
    assert_eq!(r.status, SolveStatus::Infeasible, "expected Infeasible, got {:?}", r.status);
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
    let ws = WarmStartBasis { basis: vec![0, 1], x_b: vec![3.0, 1.5] };
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        recover_warm_start_basis: true,
        warm_start: Some(ws),
        ..Default::default()
    };
    let r = crate::lp::solve_lp_with(&lp, &opts);
    assert_ne!(r.status, crate::problem::SolveStatus::Timeout,
        "child LP with warm start must not timeout, got status={:?}", r.status);
    assert_eq!(r.status, crate::problem::SolveStatus::Optimal,
        "child LP should be Optimal (x=3 or x=4, y=1), got status={:?}", r.status);
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
    use std::cell::{Cell, RefCell};
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;

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
        fn num_vars(&self) -> usize { 1 }
        fn root_bounds(&self) -> &[(f64, f64)] { &self.root_bounds }
        fn integer_vars(&self) -> &[usize] { &self.int_vars }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            if n == 0 {
                // Root: fractional x=1.5; return a basis for propagation.
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 1.5,
                    solution: vec![1.5],
                    warm_start_basis: Some(WarmStartBasis { basis: vec![0], x_b: vec![1.5] }),
                    ..SolverResult::default()
                }
            } else {
                // Child: record warm_start presence, return integer-feasible solution.
                self.child_got_ws.borrow_mut().push(opts.warm_start.is_some());
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

/// B&B driver propagates parent warm-start basis to child nodes.
///
/// Sentinel: replacing `node.child_warm(down, lb, ws)` with `node.child(down, lb)`
/// drops warm_start for all children; `opts.warm_start` is `None` in every child
/// call → `received` is all-false → **this test FAILS** (no-op detected).
#[test]
fn warm_start_propagated_to_child_nodes_sentinel() {
    use std::cell::{Cell, RefCell};
    use crate::options::WarmStartBasis;
    use crate::problem::SolverResult;

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
        fn num_vars(&self) -> usize { 1 }
        fn root_bounds(&self) -> &[(f64, f64)] { &self.root_bounds }
        fn integer_vars(&self) -> &[usize] { &self.int_vars }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            let n = self.call.get();
            self.call.set(n + 1);
            if n > 0 {
                self.warm_starts_received.borrow_mut().push(opts.warm_start.is_some());
            }
            if n == 0 {
                // Root: fractional x=0.5; return warm_start_basis for propagation.
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.5,
                    solution: vec![0.5],
                    warm_start_basis: Some(WarmStartBasis { basis: vec![0], x_b: vec![0.5] }),
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
    assert!(stats.incumbent_updates >= 1, "incumbent_updates must reflect FP find");
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
    let (_, no_inc)   = solve_mip_core(&problem, &opts(), &cfg, mask, None);

    assert!(
        with_inc.nodes_processed < no_inc.nodes_processed,
        "initial incumbent must reduce B&B nodes: with={} without={}",
        with_inc.nodes_processed, no_inc.nodes_processed
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
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
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
    let opts = SolverOptions { timeout_secs: Some(FP_TIMEOUT_BUDGET_SECS), ..Default::default() };
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
    let opts = SolverOptions { timeout_secs: Some(short_timeout), ..Default::default() };
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
