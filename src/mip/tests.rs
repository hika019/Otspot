//! MILP branch-and-bound tests (#14 Phase 1).
//!
//! Multiple data patterns per CLAUDE.md: trivial integer root, fractional root
//! requiring branching, infeasible, unbounded, binary knapsack, and the
//! no-integer LP fallback. Both the low-level `solve_milp` entry and the
//! `Model` modeling API are exercised.

use super::{
    finalize_no_incumbent, integer_mask, solve_milp, solve_milp_with_stats, solve_miqp,
    MilpProblem, MiqpProblem,
};
use crate::model::{Model, ModelError, SolveError};
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

#[test]
fn fractional_root_branches_to_integer_optimum() {
    // max x s.t. 2x <= 3, x in [0,5] integer → x = 1 (LP optimum x = 1.5).
    // minimization form: min -x.
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
    assert!(stats.nodes_processed >= 3, "branching expected, nodes={}", stats.nodes_processed);
    assert!(stats.pruned >= 1, "infeasible x>=2 child must be pruned");
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
// Model modeling API
// ---------------------------------------------------------------------------

#[test]
fn model_add_int_var_maximize_branches() {
    // max x s.t. 2x <= 3, x integer in [0,5] → x = 1, obj = 1.
    let mut m = Model::new("milp_int");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.add_constraint((2.0 * x).leq(3.0));
    m.maximize(x);
    let r = m.solve().unwrap();
    assert!((r.objective() - 1.0).abs() < EPS, "obj={}", r.objective());
    assert!((r[x] - 1.0).abs() < EPS, "x={}", r[x]);
}

#[test]
fn model_binary_knapsack() {
    let mut m = Model::new("knapsack");
    let a = m.add_binary_var("a");
    let b = m.add_binary_var("b");
    let c = m.add_binary_var("c");
    let d = m.add_binary_var("d");
    m.add_constraint((5.0 * a + 7.0 * b + 4.0 * c + 3.0 * d).leq(14.0));
    m.maximize(8.0 * a + 11.0 * b + 6.0 * c + 4.0 * d);
    let r = m.solve().unwrap();
    assert!((r.objective() - 21.0).abs() < EPS, "obj={}", r.objective());
    assert_eq!(
        (r[a].round(), r[b].round(), r[c].round(), r[d].round()),
        (0.0, 1.0, 1.0, 1.0)
    );
}

#[test]
fn model_integer_infeasible_errors() {
    let mut m = Model::new("infeasible");
    let x = m.add_int_var("x", 0.0, 10.0);
    m.add_constraint((1.0 * x).geq(1.2));
    m.add_constraint((1.0 * x).leq(1.8));
    m.minimize(x);
    let err = m.solve().unwrap_err();
    assert!(matches!(err, ModelError::SolveError(SolveError::Infeasible)), "got {err:?}");
}

#[test]
fn model_integer_unbounded_errors() {
    let mut m = Model::new("unbounded");
    let x = m.add_int_var("x", 0.0, f64::INFINITY);
    m.maximize(x);
    let err = m.solve().unwrap_err();
    assert!(matches!(err, ModelError::SolveError(SolveError::Unbounded)), "got {err:?}");
}

#[test]
fn model_convex_miqp_branches_to_integer_optimum() {
    // min x^2 - 5x = 1/2·2·x^2 + (-5)x, x integer in [0,5].
    // Continuous min at x=2.5 (fractional → branch); integer optima x=2 or x=3, obj = -6.
    let mut m = Model::new("convex_miqp");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.set_diagonal_q(&[2.0]);
    m.minimize(-5.0 * x);
    let r = m.solve().unwrap();
    assert!((r.objective() - (-6.0)).abs() < EPS, "obj={}", r.objective());
    let xr = r[x].round();
    assert!(xr == 2.0 || xr == 3.0, "x must be 2 or 3, got {}", r[x]);
    assert!((r[x] - xr).abs() < EPS, "x must be integral: {}", r[x]);
}

#[test]
fn model_nonconvex_miqp_errors() {
    // indefinite Q (negative curvature) → must return ModelError::NonConvex, not silent wrong.
    // Table-driven: multiple negative-eigenvalue patterns.
    let cases: &[(&str, &[f64], &[f64])] = &[
        ("single neg", &[-2.0], &[1.0]),
        ("neg-pos-2var", &[-3.0, 2.0], &[0.0, 1.0]),
    ];
    for &(name, q_diag, c_vec) in cases {
        let n = q_diag.len();
        let mut m = Model::new(name);
        let vars: Vec<_> = (0..n).map(|i| m.add_int_var(&format!("x{i}"), 0.0, 5.0)).collect();
        m.set_diagonal_q(q_diag);
        let obj = vars.iter().zip(c_vec).fold(
            crate::model::expression::Expression::from(0.0),
            |acc, (&v, &c)| acc + c * v,
        );
        m.minimize(obj);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::NonConvex(_)),
            "[{name}] expected ModelError::NonConvex, got {err:?}"
        );
    }
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

#[test]
fn stats_timing_populated_for_milp_with_branching() {
    // fractional root → at least 3 nodes (root + 2 children); root and descendant
    // timing must both be non-zero.
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
    let (_, stats) = solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
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
    // total ≈ root + desc (floating-point tolerance)
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
fn stats_timing_infeasible_ms_populated() {
    // x in [0,10], 1.2 <= x <= 1.8 integer → LP feasible but branching produces
    // infeasible children.  infeasible_ms must be non-zero.
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
    let (r, stats) =
        solve_milp_with_stats(&milp(lp, vec![0]), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Infeasible);
    assert!(
        stats.relaxation_time_infeasible_ms > 0.0,
        "infeasible_ms must be >0; pruned={} nodes={}",
        stats.pruned,
        stats.nodes_processed,
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

#[test]
fn model_mixed_integer_continuous() {
    // min -(x + y): x integer in [0,5], y continuous in [0,5], x + y <= 3.5.
    // Optimum: x=3 (integer), y=0.5 → obj -3.5.
    let mut m = Model::new("mixed");
    let x = m.add_int_var("x", 0.0, 5.0);
    let y = m.add_var("y", 0.0, 5.0);
    m.add_constraint((x + y).leq(3.5));
    m.maximize(x + y);
    let r = m.solve().unwrap();
    assert!((r.objective() - 3.5).abs() < EPS, "obj={}", r.objective());
    assert!((r[x].round() - r[x]).abs() < EPS, "x must be integral, x={}", r[x]);
}
