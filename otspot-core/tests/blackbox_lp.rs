//! Black-box tests for LP solve (simplex), presolve, and postsolve/dual stages.
//!
//! Every expected value is hand-computed from the problem data (independent oracle).
//! No expected value is derived by running the solver.
//!
//! Technique labels (cited per test):
//!   EP  = Equivalence Partitioning
//!   BVA = Boundary Value Analysis
//!   DT  = Decision Table
//!   ST  = State Transition

use otspot_core::lp::solve_lp_with;
use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::sparse::CscMatrix;

const INF: f64 = f64::INFINITY;
const EPS_OBJ: f64 = 1e-6;
const EPS_X: f64 = 1e-5;
const EPS_RC: f64 = 1e-5;

fn opts() -> SolverOptions {
    SolverOptions::default()
}

fn assert_obj(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < EPS_OBJ,
        "{label}: obj={actual:.9e} expected={expected:.9e} rel={rel:.3e}"
    );
}

fn assert_x(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPS_X,
        "{label}: x={actual:.9e} expected={expected:.9e} diff={diff:.3e}"
    );
}

// ─── EQUIVALENCE PARTITIONING ──────────────────────────────────────────────

/// EP: Infeasible LP — contradictory bounds force the feasible set to ∅.
///
/// Problem: min x  s.t. x >= 5, x <= 3, 0 <= x <= 10.
/// Oracle: {x : x>=5} ∩ {x : x<=3} = ∅ → Infeasible.
#[test]
fn ep_lp_infeasible() {
    // 2 constraints: x >= 5 (Ge), x <= 3 (Le)
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![5.0, 3.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Infeasible, "ep_lp_infeasible: status");
}

/// EP: Unbounded LP — minimizing in a direction with no lower bound.
///
/// Problem: min -x  s.t. x >= 0 (lb=0, ub=+inf, no Le constraint).
/// Oracle: -x → -∞ as x → +∞ → Unbounded.
#[test]
fn ep_lp_unbounded() {
    // No constraint rows — lb=0, ub=+inf, c=-1: unbounded below.
    let a = CscMatrix::new(0, 1);
    let lp = LpProblem::new_general(
        vec![-1.0],
        a,
        vec![],
        vec![],
        vec![(0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Unbounded, "ep_lp_unbounded: status");
}

/// EP: Single-variable LP.
///
/// Problem: min x  s.t. x >= 5, 0 <= x <= 10.
/// Oracle: feasible set [5, 10], minimized at x*=5, obj=5.
#[test]
fn ep_lp_single_variable() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_lp_single_var: status");
    assert_obj(r.objective, 5.0, "ep_lp_single_var");
    assert_x(r.solution[0], 5.0, "ep_lp_single_var x*=5");
}

/// EP: All-equality constraint system (unique solution by Gaussian elimination).
///
/// Problem: min x + 2y  s.t. x + y = 3, x - y = 1, x,y >= 0.
/// Oracle (hand-solve): adding both Eq rows: 2x=4 → x=2; then y=3-2=1.
///   x*=2, y*=1, obj = 2 + 2 = 4.
/// KKT: x,y both interior → rc[0]=rc[1]=0.
#[test]
fn ep_lp_all_equality() {
    // A: row0=(1,1), row1=(1,-1)
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, -1.0],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![3.0, 1.0],
        vec![ConstraintType::Eq, ConstraintType::Eq],
        vec![(0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_lp_all_eq: status");
    assert_obj(r.objective, 4.0, "ep_lp_all_eq");
    assert_x(r.solution[0], 2.0, "ep_lp_all_eq x*=2");
    assert_x(r.solution[1], 1.0, "ep_lp_all_eq y*=1");
}

/// EP: Empty LP (0 variables, 0 constraints).
///
/// Oracle: trivially feasible (no vars, no constraints). obj=0, Status=Optimal.
#[test]
fn ep_lp_empty() {
    let a = CscMatrix::new(0, 0);
    let lp = LpProblem::new_general(vec![], a, vec![], vec![], vec![], None).unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_lp_empty: status");
    assert_obj(r.objective, 0.0, "ep_lp_empty: obj=0");
}

// ─── BOUNDARY VALUE ANALYSIS ───────────────────────────────────────────────

/// BVA: Optimal at lower bound (both variables pinned to lb=0 by cost).
///
/// Problem: min 3x + 2y  s.t. x + y <= 10, 0 <= x <= 5, 0 <= y <= 5.
/// Oracle: c=[3,2] > 0, lb=[0,0]. The Le constraint is inactive at (0,0).
///   x*=0, y*=0, obj=0.
/// KKT: dual_Le=0 (inactive), rc[0]=3≥0, rc[1]=2≥0 (both at lb ✓).
#[test]
fn bva_lp_optimal_at_lb() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![3.0, 2.0],
        a,
        vec![10.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_at_lb: status");
    assert_obj(r.objective, 0.0, "bva_at_lb");
    assert_x(r.solution[0], 0.0, "bva_at_lb x*=0");
    assert_x(r.solution[1], 0.0, "bva_at_lb y*=0");
    // Dual feasibility: reduced costs ≥ 0 at lb
    if !r.reduced_costs.is_empty() {
        assert!(
            r.reduced_costs[0] >= -EPS_RC,
            "bva_at_lb: rc[x]={} must be ≥0 (x at lb)",
            r.reduced_costs[0]
        );
        assert!(
            r.reduced_costs[1] >= -EPS_RC,
            "bva_at_lb: rc[y]={} must be ≥0 (y at lb)",
            r.reduced_costs[1]
        );
    }
}

/// BVA: Optimal at upper bound.
///
/// Problem: min -x - 2y  s.t. x + y <= 6, 0 <= x <= 4, 0 <= y <= 4.
/// Oracle: maximizing x+2y (coeff 2>1 for y → prefer y at ub).
///   y*=4 (ub), then x <= 6-4=2 → x*=2 (interior, minimize -x).
///   x*=2, y*=4, obj = -2 - 8 = -10.
/// KKT: rc[x] = c_x - A[0,0]*λ = -1 - 1*(-1) = 0 (x interior ✓).
///      rc[y] = c_y - A[0,1]*λ = -2 - 1*(-1) = -1 ≤ 0 (y at ub ✓).
///      λ = -1 ≤ 0 (Le binding ✓). Complementarity: (-1)*(6-6)=0 ✓.
#[test]
fn bva_lp_optimal_at_ub() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -2.0],
        a,
        vec![6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 4.0), (0.0, 4.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_at_ub: status");
    assert_obj(r.objective, -10.0, "bva_at_ub");
    assert_x(r.solution[0], 2.0, "bva_at_ub x*=2");
    assert_x(r.solution[1], 4.0, "bva_at_ub y*=4");
    // Dual feasibility: reduced costs ≤ 0 at ub
    if !r.reduced_costs.is_empty() {
        assert!(
            r.reduced_costs[1] <= EPS_RC,
            "bva_at_ub: rc[y]={} must be ≤0 (y at ub)",
            r.reduced_costs[1]
        );
    }
}

/// BVA: RHS = 0 (constraint boundary at origin).
///
/// Problem: min x + y  s.t. x + y >= 0, 0 <= x,y <= 5.
/// Oracle: with x,y ≥ 0, x+y ≥ 0 is already satisfied at (0,0).
///   x*=0, y*=0, obj=0.
#[test]
fn bva_lp_rhs_zero() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![0.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_rhs_zero: status");
    assert_obj(r.objective, 0.0, "bva_rhs_zero");
    assert_x(r.solution[0], 0.0, "bva_rhs_zero x*=0");
    assert_x(r.solution[1], 0.0, "bva_rhs_zero y*=0");
}

/// BVA: Fixed variable (lb == ub forces x to a unique value).
///
/// Problem: min x + y  s.t. x + y <= 10, x ∈ [3,3] (fixed), 0 <= y <= 8.
/// Oracle: x=3 forced. Minimize y → y*=0. obj=3+0=3.
#[test]
fn bva_lp_fixed_variable() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![10.0],
        vec![ConstraintType::Le],
        vec![(3.0, 3.0), (0.0, 8.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_fixed: status");
    assert_obj(r.objective, 3.0, "bva_fixed");
    assert_x(r.solution[0], 3.0, "bva_fixed x*=3");
    assert_x(r.solution[1], 0.0, "bva_fixed y*=0");
}

/// BVA: Exactly tight constraint (degenerate 1-point feasible region along 1D).
///
/// Problem: min x  s.t. x >= 2, x <= 2, 0 <= x <= 10.
/// Oracle: feasible set = {2} (unique). x*=2, obj=2.
#[test]
fn bva_lp_tight_constraint() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![2.0, 2.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_tight: status");
    assert_obj(r.objective, 2.0, "bva_tight");
    assert_x(r.solution[0], 2.0, "bva_tight x*=2");
}

/// BVA: Degenerate vertex — multiple constraints active at the optimum.
///
/// Problem: min x + y  s.t. x >= 1, y >= 1, x + y <= 5, 0 <= x,y <= 5.
/// Oracle: min x+y with x≥1, y≥1 → optimal at x=1, y=1 (both Ge active).
///   x+y=2 < 5 so Le constraint is inactive.
///   obj = 1 + 1 = 2.
/// KKT: x*=1=lb, y*=1=lb → rc[x]≥0, rc[y]≥0 (both at lb ✓).
#[test]
fn bva_lp_degenerate_vertex() {
    // Constraints: rows 0=Ge(x>=1), 1=Ge(y>=1), 2=Le(x+y<=5)
    let a = CscMatrix::from_triplets(
        &[0, 1, 2, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![1.0, 1.0, 5.0],
        vec![ConstraintType::Ge, ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_degenerate: status");
    assert_obj(r.objective, 2.0, "bva_degenerate");
    assert_x(r.solution[0], 1.0, "bva_degenerate x*=1");
    assert_x(r.solution[1], 1.0, "bva_degenerate y*=1");
}

// ─── DECISION TABLE ────────────────────────────────────────────────────────

/// DT: Le + box + minimize → optimal at lb (constraint inactive).
///
/// Problem: min x + y  s.t. x + y <= 4, 0 <= x,y <= 5.
/// Oracle: c=[1,1]>0 → both at lb=0. Constraint 0+0=0 ≤ 4 (inactive).
///   x*=0, y*=0, obj=0.
#[test]
fn dt_lp_le_box_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_le_box_min: status");
    assert_obj(r.objective, 0.0, "dt_le_box_min");
    // Both at lb
    assert_x(r.solution[0], 0.0, "dt_le_box_min x*=0");
    assert_x(r.solution[1], 0.0, "dt_le_box_min y*=0");
}

/// DT: Ge + box + minimize → constraint active, unique objective.
///
/// Problem: min x + y  s.t. x + y >= 4, 0 <= x,y <= 5.
/// Oracle: minimize with x+y≥4 → tight at x+y=4, obj=4.
///   Unique obj; solver may return any (x,y) with x+y=4 on the segment [0,4]×[0,4].
#[test]
fn dt_lp_ge_box_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![4.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_ge_box_min: status");
    assert_obj(r.objective, 4.0, "dt_ge_box_min obj=4");
    // Primal feasibility: x+y >= 4
    let sum = r.solution[0] + r.solution[1];
    assert!(
        sum >= 4.0 - 1e-5,
        "dt_ge_box_min: x+y={sum:.6} must be ≥4"
    );
}

/// DT: Eq + box + minimize → unique optimum at specific vertex.
///
/// Problem: min x + 2y  s.t. x + y = 3, 0 <= x,y <= 3.
/// Oracle: Eq forces x+y=3. c=[1,2] → prefer x (smaller cost).
///   x*=3 (at ub), y*=0 (at lb). obj=3+0=3.
#[test]
fn dt_lp_eq_box_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![3.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 3.0), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_eq_box_min: status");
    assert_obj(r.objective, 3.0, "dt_eq_box_min");
    assert_x(r.solution[0], 3.0, "dt_eq_box_min x*=3");
    assert_x(r.solution[1], 0.0, "dt_eq_box_min y*=0");
}

/// DT: Le + box + maximize (stored as min -obj).
///
/// Problem: max x + y (= min -x-y)  s.t. x + y <= 4, 0 <= x,y <= 3.
/// Oracle: max at x+y=4, each ≤ 3 → vertex (1,3) or (3,1). max obj=4, min obj=-4.
#[test]
fn dt_lp_le_box_max() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -1.0],
        a,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_le_box_max: status");
    assert_obj(r.objective, -4.0, "dt_le_box_max min obj=-4");
    // Solution on the line x+y=4 with bounds
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 4.0).abs() < 1e-5,
        "dt_le_box_max: x+y={sum:.6} must ≈4"
    );
}

/// DT: Free variable with Le constraint → Unbounded.
///
/// Problem: min x  s.t. x <= 5, x ∈ (-∞, +∞).
/// Oracle: x is free, c=-1 implicitly (no, c=1 but with lb=-inf → x can go to -inf).
///   Wait: min x, x <= 5, x free → x → -∞. Status: Unbounded.
#[test]
fn dt_lp_free_unbounded() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(
        r.status,
        SolveStatus::Unbounded,
        "dt_free_unbounded: status"
    );
}

/// DT: Eq + fixed variable + minimize.
///
/// Problem: min x + y  s.t. x + y = 5, x ∈ [2,2] (fixed), 0 <= y <= 10.
/// Oracle: x=2 (fixed by bounds). x+y=5 → y=3. obj=2+3=5.
#[test]
fn dt_lp_eq_fixed_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Eq],
        vec![(2.0, 2.0), (0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_eq_fixed: status");
    assert_obj(r.objective, 5.0, "dt_eq_fixed obj=5");
    assert_x(r.solution[0], 2.0, "dt_eq_fixed x*=2");
    assert_x(r.solution[1], 3.0, "dt_eq_fixed y*=3");
}

// ─── STATE TRANSITION ──────────────────────────────────────────────────────

/// ST: Feasible → Infeasible by tightening a Ge constraint past the ub.
///
/// P1: x ∈ [1,5], x >= 1 → Optimal (x*=1, obj=1).
/// P2: x ∈ [1,5], x >= 6 → Infeasible ({x in [1,5]} ∩ {x>=6} = ∅).
#[test]
fn st_lp_flip_to_infeasible() {
    let build = |rhs: f64| {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![rhs],
            vec![ConstraintType::Ge],
            vec![(1.0, 5.0)],
            None,
        )
        .unwrap()
    };
    // P1: feasible
    let r1 = solve_lp_with(&build(1.0), &opts());
    assert_eq!(r1.status, SolveStatus::Optimal, "st_flip: P1 status");
    assert_obj(r1.objective, 1.0, "st_flip: P1 obj=1");
    // P2: infeasible
    let r2 = solve_lp_with(&build(6.0), &opts());
    assert_eq!(r2.status, SolveStatus::Infeasible, "st_flip: P2 status");
}

/// ST: Bounded → Unbounded by removing the upper bound.
///
/// P1: min -x  s.t. x <= 4, 0 <= x <= 4 → Optimal x*=4, obj=-4.
/// P2: min -x  s.t. x <= 4, 0 <= x (ub=+inf, but still bound by Le?) — no.
///   Actually keep the Le but remove ub: bounds=(0,+inf), x<=4 via Le → still bounded.
///   Instead test with no Le: min -x, x in [0, +inf] → Unbounded.
#[test]
fn st_lp_flip_to_unbounded() {
    // P1: bounded (x has ub=4 via bounds, Le is also there)
    let a1 = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp1 = LpProblem::new_general(
        vec![-1.0],
        a1,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, 4.0)],
        None,
    )
    .unwrap();
    let r1 = solve_lp_with(&lp1, &opts());
    assert_eq!(r1.status, SolveStatus::Optimal, "st_unbounded: P1 status");
    assert_obj(r1.objective, -4.0, "st_unbounded: P1 obj=-4");

    // P2: no constraint at all (ub=+inf, c=-1 → min -x unbounded)
    let a2 = CscMatrix::new(0, 1);
    let lp2 = LpProblem::new_general(
        vec![-1.0],
        a2,
        vec![],
        vec![],
        vec![(0.0, INF)],
        None,
    )
    .unwrap();
    let r2 = solve_lp_with(&lp2, &opts());
    assert_eq!(r2.status, SolveStatus::Unbounded, "st_unbounded: P2 status");
}

// ─── PRESOLVE / POSTSOLVE ROUND-TRIP ───────────────────────────────────────

/// PRESOLVE + POSTSOLVE: Singleton Eq row — presolve eliminates x, postsolve lifts it.
///
/// Problem: min 2x + 3y + 5z  s.t.  x + y + z <= 10,  x = 4 (Eq singleton),  x,y,z >= 0.
/// Oracle (hand-solve):
///   Eq row "x=4" fixes x=4. Remaining: min 8 + 3y + 5z  s.t. y+z <= 6, y,z >= 0.
///   c_y=3>0, c_z=5>0 → y*=0, z*=0.  Full: x*=4, y*=0, z*=0, obj=8.
/// This verifies the presolve+postsolve round-trip for singleton Eq elimination.
#[test]
fn presolve_singleton_eq_postsolve_roundtrip() {
    // rows: 0=Le(x+y+z<=10), 1=Eq(x=4)
    let a = CscMatrix::from_triplets(
        &[0, 0, 0, 1],
        &[0, 1, 2, 0],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![2.0, 3.0, 5.0],
        a,
        vec![10.0, 4.0],
        vec![ConstraintType::Le, ConstraintType::Eq],
        vec![(0.0, INF), (0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "presolve_singleton_eq: status");
    assert_obj(r.objective, 8.0, "presolve_singleton_eq");
    assert_x(r.solution[0], 4.0, "presolve_singleton_eq x*=4");
    assert_x(r.solution[1], 0.0, "presolve_singleton_eq y*=0");
    assert_x(r.solution[2], 0.0, "presolve_singleton_eq z*=0");
}

/// PRESOLVE + POSTSOLVE: Forcing constraint → variable fixed to its bound.
///
/// Problem: min x + y  s.t.  x + y <= 3,  x >= 3,  0 <= x <= 5, 0 <= y <= 5.
/// Oracle: x >= 3 and x + y <= 3 → y <= 0. Combined with y >= 0 → y=0, x=3.
///   obj = 3.
#[test]
fn presolve_forcing_constraint_postsolve_roundtrip() {
    // rows: 0=Le(x+y<=3), 1=Ge(x>=3)
    let a = CscMatrix::from_triplets(
        &[0, 0, 1],
        &[0, 1, 0],
        &[1.0, 1.0, 1.0],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![3.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "presolve_forcing: status");
    assert_obj(r.objective, 3.0, "presolve_forcing");
    assert_x(r.solution[0], 3.0, "presolve_forcing x*=3");
    assert_x(r.solution[1], 0.0, "presolve_forcing y*=0");
}

/// PRESOLVE + POSTSOLVE: Empty column (variable appears only in objective).
///
/// Problem: min x + 2y + 3z  s.t.  x + y <= 5,  0 <= x,y,z <= 10.
/// Oracle: z has no constraint rows → z is free to be minimized to lb=0.
///   min x+2y with x+y<=5, x,y>=0 → x*=0, y*=0 (c>0 → lb). obj=0.
#[test]
fn presolve_empty_column_postsolve_roundtrip() {
    // z (col 2) has no entries in A
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 3).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0, 3.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "presolve_empty_col: status");
    assert_obj(r.objective, 0.0, "presolve_empty_col");
    assert_x(r.solution[2], 0.0, "presolve_empty_col z*=0");
}

// ─── POSTSOLVE / DUAL RECOVERY ─────────────────────────────────────────────

/// DUAL: Le binding constraint — verify KKT: rc sign at ub, constraint dual sign.
///
/// Problem: min -x - 2y  s.t. x + y <= 6, 0 <= x <= 4, 0 <= y <= 4.
/// Oracle: x*=2, y*=4, obj=-10 (see bva_lp_optimal_at_ub).
/// Independent KKT check (hand-derived):
///   rc[j] = c[j] - A[:,j]'*dual  (raw_rc formula)
///   Le constraint binding → dual[0] ≤ 0.
///   rc[0] = -1 - 1*dual[0] = 0 (x interior: 0<2<4) → dual[0] = -1.
///   rc[1] = -2 - 1*(-1) = -1 ≤ 0 (y at ub ✓).
///   Complementarity: dual[0]*(x+y-6) = (-1)*(6-6) = 0 ✓.
///   Primal feasibility: x+y = 2+4 = 6 ≤ 6 ✓.
#[test]
fn postsolve_lp_le_dual_kkt() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -2.0],
        a,
        vec![6.0],
        vec![ConstraintType::Le],
        vec![(0.0, 4.0), (0.0, 4.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dual_le: status");
    assert_obj(r.objective, -10.0, "dual_le");
    assert_x(r.solution[0], 2.0, "dual_le x*=2");
    assert_x(r.solution[1], 4.0, "dual_le y*=4");

    // Primal feasibility: x+y ≤ 6
    let axb = r.solution[0] + r.solution[1];
    assert!(
        axb <= 6.0 + 1e-5,
        "dual_le primal feas: x+y={axb:.6} ≤ 6"
    );

    // Dual feasibility: Le binding → dual ≤ 0
    if !r.dual_solution.is_empty() {
        assert!(
            r.dual_solution[0] <= EPS_RC,
            "dual_le: constraint dual={} must be ≤0 (Le, minimization)",
            r.dual_solution[0]
        );
        // Complementarity: dual[0]*(Ax-b) ≈ 0
        let slack = axb - 6.0; // ≈ 0 (binding)
        let comp = r.dual_solution[0] * slack;
        assert!(
            comp.abs() < 1e-5,
            "dual_le complementarity: y[0]*(Ax-b)={comp:.2e} must ≈0"
        );
    }

    // Reduced cost at ub: rc[1] ≤ 0 (y at ub)
    if !r.reduced_costs.is_empty() {
        assert!(
            r.reduced_costs[1] <= EPS_RC,
            "dual_le: rc[y]={} must be ≤0 (y at ub)",
            r.reduced_costs[1]
        );
    }
}

/// DUAL: Ge binding constraint — verify KKT: rc sign at lb, constraint dual sign.
///
/// Problem: min 2x + y  s.t. x + y >= 2, 0 <= x,y <= 10.
/// Oracle: c=[2,1]; c_y=1 < c_x=2 → prefer y. Vertex (0,2): x*=0, y*=2, obj=2.
/// Independent KKT check (hand-derived):
///   rc[j] = c[j] - A[:,j]'*dual
///   Ge binding → dual[0] ≥ 0 (Ge dual in minimization).
///   y*=2 interior (0<2<10): rc[1] = 1 - 1*dual[0] = 0 → dual[0]=1.
///   x*=0 at lb: rc[0] = 2 - 1*1 = 1 ≥ 0 ✓.
///   Complementarity: dual[0]*(x+y-2) = 1*(2-2) = 0 ✓.
///   Primal feasibility: x+y = 0+2 = 2 ≥ 2 ✓.
#[test]
fn postsolve_lp_ge_dual_kkt() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![2.0, 1.0],
        a,
        vec![2.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 10.0), (0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dual_ge: status");
    assert_obj(r.objective, 2.0, "dual_ge");
    assert_x(r.solution[0], 0.0, "dual_ge x*=0");
    assert_x(r.solution[1], 2.0, "dual_ge y*=2");

    // Primal feasibility: x+y ≥ 2
    let axb = r.solution[0] + r.solution[1];
    assert!(
        axb >= 2.0 - 1e-5,
        "dual_ge primal feas: x+y={axb:.6} ≥ 2"
    );

    // Dual feasibility: Ge binding → dual ≥ 0
    if !r.dual_solution.is_empty() {
        assert!(
            r.dual_solution[0] >= -EPS_RC,
            "dual_ge: constraint dual={} must be ≥0 (Ge, minimization)",
            r.dual_solution[0]
        );
    }

    // Reduced cost at lb: rc[0] ≥ 0 (x at lb)
    if !r.reduced_costs.is_empty() {
        assert!(
            r.reduced_costs[0] >= -EPS_RC,
            "dual_ge: rc[x]={} must be ≥0 (x at lb)",
            r.reduced_costs[0]
        );
    }
}
