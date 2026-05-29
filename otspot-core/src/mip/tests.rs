//! MILP branch-and-bound tests (#14 Phase 1).
//!
//! Multiple data patterns per CLAUDE.md: trivial integer root, fractional root
//! requiring branching, infeasible, unbounded, binary knapsack, and the
//! no-integer LP fallback. Low-level `solve_milp` / `solve_miqp` entries are
//! exercised here. Model API tests live in `otspot-model/tests/mip_model.rs`.

use super::{
    finalize_no_incumbent, integer_mask, solve_milp, solve_milp_with_stats, solve_miqp,
    solve_miqp_with_stats, MilpProblem, MiqpProblem,
};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
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
fn stats_timing_populated_for_milp_with_branching() {
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
fn stats_timing_infeasible_ms_populated() {
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
fn non_optimal_result_has_no_bound_gap_cert() {
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
