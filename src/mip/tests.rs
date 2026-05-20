//! MILP branch-and-bound tests (#14 Phase 1).
//!
//! Multiple data patterns per CLAUDE.md: trivial integer root, fractional root
//! requiring branching, infeasible, unbounded, binary knapsack, and the
//! no-integer LP fallback. Both the low-level `solve_milp` entry and the
//! `Model` modeling API are exercised.

use super::{finalize_no_incumbent, solve_milp, solve_milp_with_stats, MilpProblem};
use crate::model::{Model, ModelError, SolveError};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
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
    // An untrusted/degraded relaxation left a region open (open_lb finite) while no
    // flag was set and the queue emptied. Reporting Infeasible here would be a silent
    // wrong answer. Sentinel: dropping `&& open_lb.is_infinite()` flips this to
    // Infeasible and the test FAILS.
    let r = finalize_no_incumbent(false, false, true, 1.5, false);
    assert_ne!(r.status, SolveStatus::Infeasible, "finite open_lb must not be Infeasible");
    assert_eq!(r.status, SolveStatus::MaxIterations);
}

#[test]
fn no_incumbent_fully_resolved_is_infeasible() {
    // Every region resolved as an infeasible relaxation (open_lb stays +inf, no flags,
    // queue empty) → genuinely Infeasible.
    let r = finalize_no_incumbent(false, false, true, f64::INFINITY, false);
    assert_eq!(r.status, SolveStatus::Infeasible);
}

#[test]
fn no_incumbent_deadline_is_timeout_not_infeasible() {
    let r = finalize_no_incumbent(true, false, false, f64::INFINITY, true);
    assert_eq!(r.status, SolveStatus::Timeout);
}

#[test]
fn no_incumbent_depth_limited_is_not_infeasible() {
    // depth_limited region left open even though queue happens to be empty.
    let r = finalize_no_incumbent(false, true, true, 2.0, false);
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
fn model_miqp_not_yet_supported() {
    // integer + quadratic objective → explicit error (Phase 2), never a silent wrong answer.
    let mut m = Model::new("miqp");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.set_diagonal_q(&[2.0]);
    m.minimize(x);
    let err = m.solve().unwrap_err();
    match err {
        ModelError::Internal(msg) => assert!(msg.contains("MIQP"), "msg={msg}"),
        other => panic!("expected Internal MIQP error, got {other:?}"),
    }
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
