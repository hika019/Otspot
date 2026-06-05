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
use otspot_model::{Expression, Model, ModelError, SolveError, Variable};

const INF: f64 = f64::INFINITY;
const EPS_OBJ: f64 = 1e-6;
const EPS_X: f64 = 1e-5;
const EPS_RC: f64 = 1e-5;
const HARD_LP_EPS_OBJ: f64 = 5e-6;
const HARD_LP_EPS_X: f64 = 2e-4;
const HARD_LP_EPS_RESID: f64 = 2e-5;
const HARD_LP_TIMEOUT_SECS: f64 = 10.0;
const HARD_LP_INF: f64 = f64::INFINITY;
const HARD_LP_LARGE_M: usize = 50;
const HARD_LP_LARGE_N: usize = 100;
const HARD_LP_LARGE_EXPECTED_OBJ: f64 = -4.406_871_388_953_238;
const HARD_LP_LARGE_EQ_GE_PAIRS: usize = 96;
const HARD_LP_LARGE_EQ_GE_K: usize = 37;
const HARD_LP_LARGE_EQ_GE_EXPECTED_OBJ: f64 = 3.404;
const HARD_LP_ILL_EXPECTED_OBJ: f64 = -6.099_999_999_814_999;
const HARD_LP_DEGENERATE_EXPECTED_OBJ: f64 = -1.0;
const HARD_LP_NEAR_TIE_EXPECTED_OBJ: f64 = -1.000_000_005_9;

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

fn hard_lp_expr(vars: &[Variable], terms: &[(usize, f64)]) -> Expression {
    let mut expr = Expression::from_constant(0.0);
    for &(var_idx, coeff) in terms {
        expr = expr + coeff * vars[var_idx];
    }
    expr
}

fn hard_lp_assert_model_obj(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < HARD_LP_EPS_OBJ,
        "{label}: obj={actual:.12e} expected={expected:.12e} rel={rel:.3e}"
    );
}

fn hard_lp_assert_model_x(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_LP_EPS_X,
        "{label}: x={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
    );
}

fn hard_lp_assert_resid(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_LP_EPS_RESID,
        "{label}: residual actual={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
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
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "ep_lp_infeasible: status"
    );
}

/// EP: Unbounded LP — minimizing in a direction with no lower bound.
///
/// Problem: min -x  s.t. x >= 0 (lb=0, ub=+inf, no Le constraint).
/// Oracle: -x → -∞ as x → +∞ → Unbounded.
#[test]
fn ep_lp_unbounded() {
    // No constraint rows — lb=0, ub=+inf, c=-1: unbounded below.
    let a = CscMatrix::new(0, 1);
    let lp = LpProblem::new_general(vec![-1.0], a, vec![], vec![], vec![(0.0, INF)], None).unwrap();
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
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
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
    let a = CscMatrix::from_triplets(&[0, 1, 2, 2], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 3, 2)
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
    assert!(sum >= 4.0 - 1e-5, "dt_ge_box_min: x+y={sum:.6} must be ≥4");
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
    let lp2 =
        LpProblem::new_general(vec![-1.0], a2, vec![], vec![], vec![(0.0, INF)], None).unwrap();
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
    let a = CscMatrix::from_triplets(&[0, 0, 0, 1], &[0, 1, 2, 0], &[1.0, 1.0, 1.0, 1.0], 2, 3)
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
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "presolve_singleton_eq: status"
    );
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
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
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
    assert!(axb <= 6.0 + 1e-5, "dual_le primal feas: x+y={axb:.6} ≤ 6");

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
    assert!(axb >= 2.0 - 1e-5, "dual_ge primal feas: x+y={axb:.6} ≥ 2");

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

// ─── EQUIVALENCE PARTITIONING (ADDITIONAL) ────────────────────────────────────

/// EP: Multiple optima — objective vector parallel to a constraint face.
///
/// Problem: min 2x + 2y  s.t. x + y >= 3,  0 <= x,y <= 5.
/// c = [2,2] is parallel to the face {x+y=3} → any (x,y) on {x+y=3, 0<=x,y<=3}
/// is optimal.  Objective value is unique even though the solution is not.
///
/// Oracle (scipy linprog highs):
///   from scipy.optimize import linprog
///   linprog([2,2], A_ub=[[-1,-1]], b_ub=[-3], bounds=[(0,5),(0,5)]) → fun=6.0
/// obj* = 2*3 = 6.
#[test]
fn ep_lp_multiple_optima() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![2.0, 2.0],
        a,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_multiple_optima: status");
    assert_obj(r.objective, 6.0, "ep_multiple_optima obj=6");
    // Any x+y=3 in [0,3] is valid — assert primal feasibility and objective only.
    let sum = r.solution[0] + r.solution[1];
    assert!(
        sum >= 3.0 - EPS_X,
        "ep_multiple_optima: x+y={sum:.6} must be ≥3"
    );
}

/// EP: Redundant constraint — x >= 1 is made redundant by x >= 2.
///
/// Problem: min x  s.t. x >= 2, x >= 1,  0 <= x <= 10.
/// Oracle: the tighter constraint x >= 2 dominates; x >= 1 is redundant.
///   x* = 2, obj = 2.
///   scipy: linprog([1], A_ub=[[-1],[-1]], b_ub=[-2,-1], bounds=[(0,10)]) → fun=2.0
#[test]
fn ep_lp_redundant_constraint() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![2.0, 1.0],
        vec![ConstraintType::Ge, ConstraintType::Ge],
        vec![(0.0, 10.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_redundant: status");
    assert_obj(r.objective, 2.0, "ep_redundant obj=2");
    assert_x(r.solution[0], 2.0, "ep_redundant x*=2");
}

/// EP: Free + bounded variable mix with Le constraint.
///
/// Problem: min -x + 2y  s.t. x + y <= 4,  x ∈ (-∞,+∞),  0 <= y <= 3.
/// Oracle: minimize -x + 2y = -(x - 2y).  Equivalently maximize x - 2y.
///   Since c_x=-1 < 0: push x as high as possible.  x+y<=4, y>=0 → x<=4.
///   Since c_y=2 > 0: push y to lb=0. At y=0: x<=4, x*=4.
///   x*=4, y*=0, obj = -4+0 = -4.
///   scipy: linprog([-1,2], A_ub=[[1,1]], b_ub=[4], bounds=[(-inf,inf),(0,3)]) → fun=-4.0
#[test]
fn ep_lp_free_bounded_mix() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, 2.0],
        a,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, INF), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_free_bounded: status");
    assert_obj(r.objective, -4.0, "ep_free_bounded obj=-4");
    assert_x(r.solution[0], 4.0, "ep_free_bounded x*=4");
    assert_x(r.solution[1], 0.0, "ep_free_bounded y*=0");
}

/// EP: Maximization with two inequality constraints, unique optimal vertex.
///
/// Problem: max 2x + 3y  (= min -2x - 3y)
///          s.t. x + y <= 6,  2x + y <= 8,  0 <= x,y <= 5.
///
/// Oracle: maximize 2x+3y on the polytope.
///   Vertex analysis: 2x+y=8 and y=5 → 2x+5=8 → x=1.5, but x+y=6.5>6 ✗.
///   y=5, x+5<=6 → x<=1; 2x+5<=8 → x<=1.5.  At x=1, y=5: 2+5=7<8 ✓, 1+5=6=6 ✓. obj=2+15=17.
///   scipy: linprog([-2,-3], A_ub=[[1,1],[2,1]], b_ub=[6,8], bounds=[(0,5),(0,5)]) → fun=-17.0
///   x*=1, y*=5, obj=-17.
#[test]
fn ep_lp_max_two_constraints_active() {
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 2.0, 1.0], 2, 2)
        .unwrap();
    let lp = LpProblem::new_general(
        vec![-2.0, -3.0],
        a,
        vec![6.0, 8.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_max_two: status");
    assert_obj(r.objective, -17.0, "ep_max_two obj=-17");
    assert_x(r.solution[0], 1.0, "ep_max_two x*=1");
    assert_x(r.solution[1], 5.0, "ep_max_two y*=5");
}

// ─── BOUNDARY VALUE ANALYSIS (ADDITIONAL) ─────────────────────────────────────

/// BVA: Large-scale RHS (1e8) — verifies numerical stability at large magnitudes.
///
/// Problem: min x  s.t. x >= 1e8,  0 <= x.
/// Oracle: x* = 1e8, obj = 1e8.
///   scipy: linprog([1], A_ub=[[-1]], b_ub=[-1e8], bounds=[(0,None)]) → fun=1e8
#[test]
fn bva_lp_rhs_large_1e8() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![1e8],
        vec![ConstraintType::Ge],
        vec![(0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_large_rhs: status");
    assert_obj(r.objective, 1e8, "bva_large_rhs obj=1e8");
    assert_x(r.solution[0], 1e8, "bva_large_rhs x*=1e8");
}

/// BVA: Small-scale RHS (1e-6) — verifies numerical stability at small magnitudes.
///
/// Problem: min x  s.t. x >= 1e-6,  0 <= x <= 1.
/// Oracle: x* = 1e-6, obj = 1e-6.
///   scipy: linprog([1], A_ub=[[-1]], b_ub=[-1e-6], bounds=[(0,1)]) → fun=1e-6
#[test]
fn bva_lp_rhs_small_1em6() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![1e-6],
        vec![ConstraintType::Ge],
        vec![(0.0, 1.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_small_rhs: status");
    assert_obj(r.objective, 1e-6, "bva_small_rhs obj=1e-6");
    let diff = (r.solution[0] - 1e-6).abs();
    assert!(
        diff < 1e-9,
        "bva_small_rhs: x={:.3e} expected=1e-6 diff={:.3e}",
        r.solution[0],
        diff
    );
}

/// BVA: Feasible boundary — the only feasible point is exactly (5, 5).
///
/// Problem: min x + y  s.t. x + y >= 10,  0 <= x,y <= 5.
/// Oracle: max(x+y) = 10 with x,y in [0,5], so the Ge is achievable only at (5,5).
///   x* = 5, y* = 5, obj = 10.
///   scipy: linprog([1,1], A_ub=[[-1,-1]], b_ub=[-10], bounds=[(0,5),(0,5)]) → fun=10.0
#[test]
fn bva_lp_feasible_boundary_exact() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![10.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_boundary: status");
    assert_obj(r.objective, 10.0, "bva_boundary obj=10");
    assert_x(r.solution[0], 5.0, "bva_boundary x*=5");
    assert_x(r.solution[1], 5.0, "bva_boundary y*=5");
}

/// BVA: RHS + ε makes previously-boundary problem infeasible.
///
/// Problem: min x + y  s.t. x + y >= 10 + 1e-4,  0 <= x,y <= 5.
/// Oracle: max(x+y) = 10 with x,y in [0,5]; RHS = 10 + 1e-4 > 10 → Infeasible.
///   scipy: linprog([1,1], A_ub=[[-1,-1]], b_ub=[-10-1e-4], bounds=[(0,5),(0,5)]) → status=2
#[test]
fn bva_lp_rhs_eps_makes_infeasible() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![10.0 + 1e-4],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "bva_rhs_eps: status must be Infeasible"
    );
}

/// BVA: Zero objective coefficient — one variable does not enter the objective.
///
/// Problem: min 0·x + y  s.t. x + y >= 3,  0 <= x,y <= 5.
/// Oracle: min y with x+y>=3, x∈[0,5], y∈[0,5].
///   c_x=0 allows x to be pushed to ub=5; then y >= 3-5 = -2 → y* = 0 (lb).
///   x* = 5 (or any x>=3), y* = 0, obj = 0.
///   scipy: linprog([0,1], A_ub=[[-1,-1]], b_ub=[-3], bounds=[(0,5),(0,5)]) → fun=0.0
#[test]
fn bva_lp_zero_obj_coefficient() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, 1.0],
        a,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_zero_obj: status");
    assert_obj(r.objective, 0.0, "bva_zero_obj obj=0");
    assert_x(r.solution[1], 0.0, "bva_zero_obj y*=0");
    // Primal feasibility: x+y >= 3
    let sum = r.solution[0] + r.solution[1];
    assert!(sum >= 3.0 - EPS_X, "bva_zero_obj: x+y={sum:.6} must be >=3");
}

// ─── DECISION TABLE (ADDITIONAL) ───────────────────────────────────────────────

/// DT: Ge + free variable + minimize.
///
/// DT cell: sense=Ge, bound=free(-∞,+∞), objective=min.
///
/// Problem: min x + 2y  s.t. x + y >= 3,  x ∈ (-∞,+∞),  0 <= y <= 8.
/// Oracle: at optimum x+y=3 (Ge tight). obj = x+2y = (3-y)+2y = 3+y.
///   Minimize over y ∈ [0,8]: y* = 0, x* = 3. obj = 3.
///   scipy: linprog([1,2], A_ub=[[-1,-1]], b_ub=[-3], bounds=[(-inf,inf),(0,8)]) → fun=3.0
#[test]
fn dt_lp_ge_free_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(f64::NEG_INFINITY, INF), (0.0, 8.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_ge_free_min: status");
    assert_obj(r.objective, 3.0, "dt_ge_free_min obj=3");
    assert_x(r.solution[0], 3.0, "dt_ge_free_min x*=3");
    assert_x(r.solution[1], 0.0, "dt_ge_free_min y*=0");
}

/// DT: Le + ub-only bounds + maximize (min -x-y).
///
/// DT cell: sense=Le, bound=ub-only(-∞, ub], objective=max (neg cost).
///
/// Problem: max x + y  (= min -x - y)  s.t. x + y <= 5,  x <= 3,  y <= 3.
/// Oracle: max x+y with x+y<=5, x<=3, y<=3.
///   On edge x+y=5: x∈[2,3] (since y=5-x<=3 → x>=2). obj_max=5. min_obj=-5.
///   scipy: linprog([-1,-1], A_ub=[[1,1]], b_ub=[5], bounds=[(-inf,3),(-inf,3)]) → fun=-5.0
#[test]
fn dt_lp_le_ub_only_max() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, 3.0), (f64::NEG_INFINITY, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_le_ub_max: status");
    assert_obj(r.objective, -5.0, "dt_le_ub_max obj=-5");
    // x+y=5 (Le binding)
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 5.0).abs() < EPS_X,
        "dt_le_ub_max: x+y={sum:.6} must=5"
    );
}

/// DT: Eq + free variable + minimize.
///
/// DT cell: sense=Eq, bound=free(-∞,+∞), objective=min.
///
/// Problem: min x + 2y  s.t. x + y = 4,  x ∈ (-∞,+∞),  0 <= y <= 8.
/// Oracle: Eq → x = 4 - y.  obj = (4-y) + 2y = 4 + y.
///   Minimize over y ∈ [0,8]: y* = 0, x* = 4. obj = 4.
///   scipy: linprog([1,2], A_eq=[[1,1]], b_eq=[4], bounds=[(-inf,inf),(0,8)]) → fun=4.0
#[test]
fn dt_lp_eq_free_min() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![4.0],
        vec![ConstraintType::Eq],
        vec![(f64::NEG_INFINITY, INF), (0.0, 8.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_eq_free_min: status");
    assert_obj(r.objective, 4.0, "dt_eq_free_min obj=4");
    assert_x(r.solution[0], 4.0, "dt_eq_free_min x*=4");
    assert_x(r.solution[1], 0.0, "dt_eq_free_min y*=0");
}

/// DT: Ge + ub-only bounds + maximize.
///
/// DT cell: sense=Ge, bound=ub-only(-∞, ub], objective=max (neg cost).
///
/// Problem: max x + 2y  (= min -x - 2y)  s.t. x + y >= 3,  x <= 4,  y <= 4.
/// Oracle: c_y=2 > c_x=1 → push y to ub=4. Then x+4>=3 → x>=-1, and lb=-∞.
///   x* = ub = 4 (maximize x), y* = ub = 4. obj_max = 4+8=12. min_obj = -12.
///   scipy: linprog([-1,-2], A_ub=[[-1,-1]], b_ub=[-3], bounds=[(-inf,4),(-inf,4)]) → fun=-12.0
#[test]
fn dt_lp_ge_ub_only_max() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, -2.0],
        a,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(f64::NEG_INFINITY, 4.0), (f64::NEG_INFINITY, 4.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_ge_ub_max: status");
    assert_obj(r.objective, -12.0, "dt_ge_ub_max obj=-12");
    assert_x(r.solution[0], 4.0, "dt_ge_ub_max x*=4");
    assert_x(r.solution[1], 4.0, "dt_ge_ub_max y*=4");
}

// ─── STATE TRANSITION (ADDITIONAL) ─────────────────────────────────────────────

/// ST: Parametric RHS sweep — status transitions Optimal→Optimal→Infeasible.
///
/// Problem: min x + y  s.t. x + y >= b,  0 <= x,y <= 5.
/// P1 (b=2): feasible interior. obj*=2.
/// P2 (b=5): feasible, constraint binding at boundary. obj*=5.
/// P3 (b=11): max feasible sum=10 < 11 → Infeasible.
///
/// Oracle: scipy confirmed b=2→2.0, b=5→5.0, b=11→infeasible.
#[test]
fn st_lp_rhs_parametric_sweep() {
    let build = |b: f64| {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![b],
            vec![ConstraintType::Ge],
            vec![(0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap()
    };
    let r1 = solve_lp_with(&build(2.0), &opts());
    assert_eq!(r1.status, SolveStatus::Optimal, "st_sweep P1: status");
    assert_obj(r1.objective, 2.0, "st_sweep P1 obj=2");

    let r2 = solve_lp_with(&build(5.0), &opts());
    assert_eq!(r2.status, SolveStatus::Optimal, "st_sweep P2: status");
    assert_obj(r2.objective, 5.0, "st_sweep P2 obj=5");

    let r3 = solve_lp_with(&build(11.0), &opts());
    assert_eq!(
        r3.status,
        SolveStatus::Infeasible,
        "st_sweep P3: Infeasible"
    );
}

/// ST: Le constraint transitions active → inactive as RHS is raised.
///
/// Problem: min -x - y  s.t. x + y <= b,  0 <= x,y <= 5.
/// P1 (b=7): Le active. x+y=7 at optimum. obj=-7.
/// P2 (b=12): Le inactive (max reachable x+y=10 < 12). x*=y*=5. obj=-10.
///
/// Oracle: scipy confirmed b=7→-7, b=12→-10.
#[test]
fn st_lp_le_active_to_inactive() {
    let build = |b: f64| {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![b],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap()
    };
    // P1: Le active — optimum is on the constraint face x+y=7
    let r1 = solve_lp_with(&build(7.0), &opts());
    assert_eq!(r1.status, SolveStatus::Optimal, "st_le_active P1: status");
    assert_obj(r1.objective, -7.0, "st_le_active P1 obj=-7");
    let sum1 = r1.solution[0] + r1.solution[1];
    assert!(
        (sum1 - 7.0).abs() < EPS_X,
        "st_le_active P1: x+y={sum1:.6} must≈7"
    );

    // P2: Le inactive — unconstrained ub pushes to (5,5)
    let r2 = solve_lp_with(&build(12.0), &opts());
    assert_eq!(r2.status, SolveStatus::Optimal, "st_le_active P2: status");
    assert_obj(r2.objective, -10.0, "st_le_active P2 obj=-10");
    assert_x(r2.solution[0], 5.0, "st_le_active P2 x*=5");
    assert_x(r2.solution[1], 5.0, "st_le_active P2 y*=5");
    // Dual of inactive Le constraint must be zero
    if !r2.dual_solution.is_empty() {
        assert!(
            r2.dual_solution[0].abs() < EPS_RC,
            "st_le_active P2: dual[0]={} must≈0 (Le inactive)",
            r2.dual_solution[0]
        );
    }
}

/// ST: Unbounded → Optimal by adding an upper bound constraint.
///
/// P1: min -x  s.t. 0 <= x < +∞. → Unbounded.
/// P2: min -x  s.t. x <= 5,  0 <= x <= 5. → Optimal, x*=5, obj=-5.
#[test]
fn st_lp_unbounded_to_optimal_by_constraint() {
    // P1: no Le constraint, unbounded
    let a1 = CscMatrix::new(0, 1);
    let lp1 =
        LpProblem::new_general(vec![-1.0], a1, vec![], vec![], vec![(0.0, INF)], None).unwrap();
    let r1 = solve_lp_with(&lp1, &opts());
    assert_eq!(r1.status, SolveStatus::Unbounded, "st_unbounded P1: status");

    // P2: Le constraint x<=5 makes it bounded
    let a2 = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp2 = LpProblem::new_general(
        vec![-1.0],
        a2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0)],
        None,
    )
    .unwrap();
    let r2 = solve_lp_with(&lp2, &opts());
    assert_eq!(r2.status, SolveStatus::Optimal, "st_unbounded P2: status");
    assert_obj(r2.objective, -5.0, "st_unbounded P2 obj=-5");
    assert_x(r2.solution[0], 5.0, "st_unbounded P2 x*=5");
}

// ─── PAIRWISE METHOD ───────────────────────────────────────────────────────────
//
// Parameters and value domains:
//   P1 sense:  {Le, Ge, Eq}
//   P2 bound:  {free(-∞,+∞), lb-only([lb,+∞)), ub-only((-∞,ub]), box([lb,ub]), fixed([a,a])}
//   P3 obj:    {min, max}
//   P4 scale:  {unit(~1), ill(~1e6 spread)}
//   P5 degen:  {non-degenerate, degenerate(≥2 constraints active at optimum)}
//
// Representative parameter combination sample (NOT full 2-way pairwise — P1×P2 alone
// has 15 pairs requiring ≥15 tests; 6 tests cannot achieve full pairwise coverage).
//
// Pairs covered per parameter combination (P1×P2, P1×P3, P2×P3 are representative):
//
// | Test | P1    | P2    | P3  | P4   | P5    |
// |------|-------|-------|-----|------|-------|
// | pw1  | Le    | free  | max | unit | non   |
// | pw2  | Ge    | box   | min | unit | degen |
// | pw3  | Eq    | lb    | min | ill  | non   |
// | pw4  | Ge    | fixed | min | unit | degen |
// | pw5  | Ge    | box   | min | ill  | degen |
// | pw6  | Le    | ub    | max | ill  | non   |
//
// Covered P1×P2 pairs: (Le,free),(Le,ub),(Ge,box),(Ge,fixed),(Eq,lb)
// Covered P1×P3 pairs: (Le,max),(Ge,min),(Eq,min)
// Covered P2×P3 pairs: (free,max),(box,min),(lb,min),(fixed,min),(ub,max)
// Covered P4×P5 pairs: (unit,non),(unit,degen),(ill,non),(ill,degen) — all 4

/// PW: Le + free + max + unit + non-degenerate.
///
/// Problem: max 2x + y  (= min -2x - y)  s.t. x + y <= 5,  x ∈ (-∞,+∞),  0 <= y <= 3.
/// Oracle: x has c=-2<0 → push x to ub. x+y<=5, y>=0 → x<=5.
///   x*=5, y*=0. obj_max=10. min_obj=-10.
///   scipy: linprog([-2,-1], A_ub=[[1,1]], b_ub=[5], bounds=[(-inf,inf),(0,3)]) → fun=-10.0
#[test]
fn pw_lp_le_free_max_unit() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-2.0, -1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, INF), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_le_free_max: status");
    assert_obj(r.objective, -10.0, "pw_le_free_max obj=-10");
    assert_x(r.solution[0], 5.0, "pw_le_free_max x*=5");
    assert_x(r.solution[1], 0.0, "pw_le_free_max y*=0");
}

/// PW: Ge + box + min + unit + degenerate (two Ge constraints active at optimum).
///
/// Problem: min x + y  s.t. x + y >= 4 (Ge),  x >= 2 (Ge),  0 <= x,y <= 5.
/// Oracle: c=[1,1] → push to lb. x>=2 and x+y>=4 → x=2, y=2 (both Ge active).
///   Degenerate vertex: 2 constraints active for 2 variables.
///   x*=2, y*=2, obj=4.
///   scipy: linprog([1,1], A_ub=[[-1,-1],[-1,0]], b_ub=[-4,-2], bounds=[(0,5),(0,5)]) → fun=4.0
#[test]
fn pw_lp_ge_box_min_unit_degen() {
    // rows: 0=Ge(x+y>=4), 1=Ge(x>=2)
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![4.0, 2.0],
        vec![ConstraintType::Ge, ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_ge_box_degen: status");
    assert_obj(r.objective, 4.0, "pw_ge_box_degen obj=4");
    assert_x(r.solution[0], 2.0, "pw_ge_box_degen x*=2");
    assert_x(r.solution[1], 2.0, "pw_ge_box_degen y*=2");
}

/// PW: Eq + lb-only + min + ill-scaled + non-degenerate.
///
/// Problem: min 1e6·x + y  s.t. x + y = 3,  x ∈ [0,+∞),  0 <= y <= 5.
/// Oracle: c_x=1e6 ≫ c_y=1 → minimize x → x*=0.  Eq: y=3. obj=0+3=3.
///   (ill-scaled: coefficient spread 1e6.)
///   scipy: linprog([1e6,1], A_eq=[[1,1]], b_eq=[3], bounds=[(0,inf),(0,5)]) → fun=3.0
#[test]
fn pw_lp_eq_lb_min_ill() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1e6, 1.0],
        a,
        vec![3.0],
        vec![ConstraintType::Eq],
        vec![(0.0, INF), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_eq_lb_ill: status");
    assert_obj(r.objective, 3.0, "pw_eq_lb_ill obj=3");
    assert_x(r.solution[0], 0.0, "pw_eq_lb_ill x*=0");
    assert_x(r.solution[1], 3.0, "pw_eq_lb_ill y*=3");
}

/// PW: Ge + fixed + min + unit + degenerate (fixed bound + Ge both active).
///
/// Problem: min x + y  s.t. x + y >= 4 (Ge),  x ∈ [2,2] (fixed),  0 <= y <= 5.
/// Oracle: x=2 (forced by fixed bounds). Ge: y >= 4-2=2. c_y=1>0 → y*=2. obj=4.
///   Degenerate: both the fixed-x bound and the Ge row are active at the optimum.
///   scipy: linprog([1,1], A_ub=[[-1,-1]], b_ub=[-4], bounds=[(2,2),(0,5)]) → fun=4.0
#[test]
fn pw_lp_ge_fixed_min_unit() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![4.0],
        vec![ConstraintType::Ge],
        vec![(2.0, 2.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_ge_fixed: status");
    assert_obj(r.objective, 4.0, "pw_ge_fixed obj=4");
    assert_x(r.solution[0], 2.0, "pw_ge_fixed x*=2");
    assert_x(r.solution[1], 2.0, "pw_ge_fixed y*=2");
}

/// PW: Ge + box + min + ill-scaled + degenerate.
///
/// Problem: min 1e-4·x + 1e-4·y  s.t. x + y >= 3 (Ge),  x >= 2 (Ge),  0 <= x,y <= 5.
/// Oracle: ill-scaled (c ≈ 1e-4). Both Ge active at x*=2, y*=1 (unique vertex).
///   obj = 1e-4·(2+1) = 3e-4.
///   scipy: linprog([1e-4,1e-4], A_ub=[[-1,-1],[-1,0]], b_ub=[-3,-2],
///           bounds=[(0,5),(0,5)]) → fun=3e-4
#[test]
fn pw_lp_ge_box_min_ill_degen() {
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1e-4, 1e-4],
        a,
        vec![3.0, 2.0],
        vec![ConstraintType::Ge, ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_ge_ill_degen: status");
    assert_obj(r.objective, 3e-4, "pw_ge_ill_degen obj=3e-4");
    assert_x(r.solution[0], 2.0, "pw_ge_ill_degen x*=2");
    assert_x(r.solution[1], 1.0, "pw_ge_ill_degen y*=1");
}

/// PW: Le + ub-only + max + ill-scaled + non-degenerate.
///
/// Problem: max 1e6·x + y  (= min -1e6·x - y)
///          s.t. x + y <= 5,  x ∈ (-∞, 4],  y ∈ (-∞, 3].
/// Oracle: c_x=-1e6 ≪ 0 → push x to ub=4. Le: y <= 5-4=1. c_y=-1 → y*=ub∩(5-x)=1.
///   x*=4, y*=1. min_obj = -1e6·4 - 1 = -4000001.
///   scipy: linprog([-1e6,-1], A_ub=[[1,1]], b_ub=[5], bounds=[(-inf,4),(-inf,3)]) → fun=-4000001
#[test]
fn pw_lp_le_ub_max_ill() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![-1e6, -1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, 4.0), (f64::NEG_INFINITY, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_le_ub_ill: status");
    assert_obj(r.objective, -4_000_001.0, "pw_le_ub_ill obj=-4000001");
    assert_x(r.solution[0], 4.0, "pw_le_ub_ill x*=4");
    assert_x(r.solution[1], 1.0, "pw_le_ub_ill y*=1");
}

// ─── CLASSIFICATION TREE METHOD ────────────────────────────────────────────────
//
// Classification tree for LP test design:
//
// Root
// ├─ Problem size
// │   ├─ Small (1-4 vars)
// │   │   ├─ Constraint mix
// │   │   │   ├─ Pure Le  → well-conditioned → ct_lp_small_le_box_max
// │   │   │   ├─ Mixed Le+Ge → ill-conditioned → ct_lp_small_mixed_le_ge_ill
// │   │   │   └─ Pure Eq → fixed vars → existing dt_lp_eq_fixed_min
// │   │   └─ Bound structure
// │   │       ├─ All-box → small tests above
// │   │       └─ Mixed (free+box+lb) → ep_lp_free_bounded_mix
// │   └─ Medium (10-30 vars)
// │       ├─ Ge sum-constraint → ct_lp_medium_10var_ge
// │       └─ (LP path exercised with larger presolve reductions)
// └─ Conditioning
//     ├─ Well-conditioned (coefficients ~1) → most tests
//     └─ Ill-conditioned (1e4+ spread) → pw tests, ct_lp_small_mixed_le_ge_ill

/// CT: Small LP — pure Le constraints, all-box bounds, well-conditioned, maximization.
///
/// Classification leaf: size=small, mix=pure-Le, bound=all-box, cond=well.
///
/// Problem: max 3x + 2y  (= min -3x - 2y)
///          s.t. x + y <= 4,  x + 2y <= 6,  0 <= x,y <= 3.
/// Oracle: vertices of polytope: (0,0), (3,0), (3,1), (2,2), (0,3).
///   obj at (3,1): 9+2=11. obj at (2,2): 6+4=10. Best: (3,1) with obj=11.
///   scipy: linprog([-3,-2], A_ub=[[1,1],[1,2]], b_ub=[4,6], bounds=[(0,3),(0,3)]) → fun=-11.0
#[test]
fn ct_lp_small_le_box_max() {
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let lp = LpProblem::new_general(
        vec![-3.0, -2.0],
        a,
        vec![4.0, 6.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ct_small_le_max: status");
    assert_obj(r.objective, -11.0, "ct_small_le_max obj=-11");
    assert_x(r.solution[0], 3.0, "ct_small_le_max x*=3");
    assert_x(r.solution[1], 1.0, "ct_small_le_max y*=1");
}

/// CT: Small LP — mixed Le+Ge constraints, 3 variables, ill-conditioned (1e4 spread).
///
/// Classification leaf: size=small, mix=Le+Ge, bound=all-box, cond=ill.
///
/// Problem: min 1e4·x + y + z  s.t. x + y + z >= 2 (Ge),  y <= 3 (Le),  0 <= x,y,z <= 5.
/// Oracle: c_x=1e4 ≫ c_y=c_z=1 → minimize x → x*=0.
///   Remaining: min y+z s.t. y+z>=2, y<=3, y,z∈[0,5]. At y=0,z=2 (or similar): obj=2.
///   scipy: linprog([1e4,1,1], A_ub=[[-1,-1,-1],[0,1,0]], b_ub=[-2,3],
///           bounds=[(0,5),(0,5),(0,5)]) → fun=2.0
#[test]
fn ct_lp_small_mixed_le_ge_ill() {
    // row 0: Ge x+y+z>=2 → stored as -x-y-z<=-2
    // row 1: Le y<=3
    let a = CscMatrix::from_triplets(&[0, 0, 0, 1], &[0, 1, 2, 1], &[1.0, 1.0, 1.0, 1.0], 2, 3)
        .unwrap();
    let lp = LpProblem::new_general(
        vec![1e4, 1.0, 1.0],
        a,
        vec![2.0, 3.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ct_small_mixed_ill: status");
    assert_obj(r.objective, 2.0, "ct_small_mixed_ill obj=2");
    assert_x(r.solution[0], 0.0, "ct_small_mixed_ill x*=0");
    // y+z=2 (Ge active): exact split doesn't matter for objective
    let sum_yz = r.solution[1] + r.solution[2];
    assert!(
        sum_yz >= 2.0 - EPS_X,
        "ct_small_mixed_ill: y+z={sum_yz:.6} must be >=2"
    );
}

/// CT: Medium LP — 10 variables, single Ge sum-constraint, all-box [0,2].
///
/// Classification leaf: size=medium(10 vars), mix=single-Ge, bound=all-box.
///
/// Problem: min sum_{i=1}^{10} i·x_i  s.t. sum(x_i) >= 10,  0 <= x_i <= 2.
/// Oracle: fill cheapest vars first (greedy is optimal for equal-weight Ge with box).
///   Sort by cost: i=1..10 in order. Fill x_1=x_2=x_3=x_4=x_5=2 → sum=10.
///   obj = 2*(1+2+3+4+5) = 2*15 = 30.
///   scipy: linprog([1..10], A_ub=[[-1]*10], b_ub=[-10], bounds=[(0,2)]*10) → fun=30.0
#[test]
fn ct_lp_medium_10var_ge() {
    // A: single row [1,1,...,1] for Ge(sum(x_i)>=10)
    // All 10 entries are in row 0 (only one constraint row), columns 0..9.
    let row_vals: Vec<usize> = vec![0; 10];
    let col_vals: Vec<usize> = (0..10).collect();
    let data: Vec<f64> = vec![1.0; 10];
    let a = CscMatrix::from_triplets(&row_vals, &col_vals, &data, 1, 10).unwrap();
    let c: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let lp = LpProblem::new_general(
        c,
        a,
        vec![10.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 2.0); 10],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ct_medium_10var: status");
    // obj = 2*(1+2+3+4+5) = 30
    assert_obj(r.objective, 30.0, "ct_medium_10var obj=30");
    // sum(x_i) >= 10  (Ge satisfied)
    let sum_x: f64 = r.solution.iter().sum();
    assert!(
        sum_x >= 10.0 - EPS_X,
        "ct_medium_10var: sum(x)={sum_x:.6} must be >=10"
    );
    // Cheapest 5 vars (i=0..4, cost 1..5) should each be at ub=2
    for i in 0..5 {
        assert_x(r.solution[i], 2.0, &format!("ct_medium_10var x[{i}]*=2"));
    }
}

// ─── HARD DATA SENTINELS (SCIPY ORACLES, MODEL EXPRESSION API) ─────────────

/// Hard LP: Eq + UB with coefficients spanning 1e-10..1e10.
///
/// SciPy oracle:
/// `linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")`
/// returned status 0, fun = -6.099999999814999, x = [4,2,4,0,0,3].
#[test]
fn hard_lp_ill_scaled_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_ill_scaled_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars = vec![
        model.add_var("x0", 0.0, 5.0),
        model.add_var("x1", 0.0, 4.0),
        model.add_var("x2", 0.0, 4.5),
        model.add_var("x3", 0.0, 3.0),
        model.add_var("x4", 0.0, 2.0),
        model.add_var("x5", 0.0, 3.0),
    ];

    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1e-10), (1, 1.0)]).eq_constraint(2.000_000_000_3),
    );
    model.add_constraint(
        hard_lp_expr(&vars, &[(2, 1e10), (3, 1.0)]).eq_constraint(40_000_000_001.0),
    );
    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (2, 1.0), (4, 1.0)]).eq_constraint(8.0));
    model.add_constraint(hard_lp_expr(&vars, &[(1, 1.0), (3, 1.0), (5, 1.0)]).eq_constraint(5.0));
    model.minimize(hard_lp_expr(
        &vars,
        &[
            (0, -1.0),
            (1, 0.25),
            (2, -0.5),
            (3, 0.75),
            (4, 0.1),
            (5, -0.2),
        ],
    ));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_ILL_EXPECTED_OBJ,
        "hard_lp_ill_scaled",
    );
    for (idx, expected) in [4.0, 2.0, 4.000_000_000_1, 0.0, 0.0, 3.0]
        .into_iter()
        .enumerate()
    {
        hard_lp_assert_model_x(
            r[vars[idx]],
            expected,
            &format!("hard_lp_ill_scaled x{idx}"),
        );
    }
}

/// Hard LP: degenerate ratio tie with multiple simultaneous leaving candidates.
///
/// SciPy oracle:
/// `linprog([-1,-1,0,0], A_ub=[[1,1,0,0],[1,0,1,0],[0,1,0,1]], ...)`
/// returned status 0 and fun = -1.0. The primal solution is intentionally
/// non-unique; the sentinel checks objective and active row feasibility.
#[test]
fn hard_lp_degenerate_ratio_tie_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_degenerate_ratio_tie_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..4)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, HARD_LP_INF))
        .collect();

    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0)]).leq(1.0));
    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (2, 1.0)]).leq(1.0));
    model.add_constraint(hard_lp_expr(&vars, &[(1, 1.0), (3, 1.0)]).leq(1.0));
    model.minimize(hard_lp_expr(&vars, &[(0, -1.0), (1, -1.0)]));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_DEGENERATE_EXPECTED_OBJ,
        "hard_lp_degenerate",
    );
    hard_lp_assert_resid(
        r[vars[0]] + r[vars[1]],
        1.0,
        "hard_lp_degenerate active row",
    );
}

/// Hard LP: finite UB becomes active under a near pivot tie.
///
/// SciPy oracle:
/// `linprog(c, A_eq=[[1,1,1],[1,1+1e-10,0]], b_eq=[1+1e-8,1+5e-9], bounds=[(0,1)]*3)`
/// returned status 0, fun = -1.0000000059, x = [0,1,0] within HiGHS feasibility tolerance.
#[test]
fn hard_lp_upper_bound_near_tie_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_upper_bound_near_tie_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..3)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0), (2, 1.0)]).eq_constraint(1.000_000_01),
    );
    model.add_constraint(
        hard_lp_expr(&vars, &[(0, 1.0), (1, 1.000_000_000_1)]).eq_constraint(1.000_000_005),
    );
    model.minimize(hard_lp_expr(
        &vars,
        &[(0, -1.0), (1, -1.000_000_001), (2, 0.05)],
    ));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_NEAR_TIE_EXPECTED_OBJ,
        "hard_lp_near_tie",
    );
    hard_lp_assert_model_x(r[vars[1]], 1.0, "hard_lp_near_tie y at ub");
}

/// Hard LP: m=50, n=100 Eq+UB synthetic instance for Phase I + Harris stress.
///
/// SciPy oracle:
/// deterministic A/c/x_ref below, then
/// `linprog(c, A_eq=A, b_eq=A@x_ref, bounds=[(0,1)]*100, method="highs")`
/// returned status 0, fun = -4.406871388953238.
#[test]
fn hard_lp_large_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_large_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..HARD_LP_LARGE_N)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    let mut x_ref = vec![0.0; HARD_LP_LARGE_N];
    for (j, xj) in x_ref.iter_mut().enumerate() {
        *xj = ((j * 37) % HARD_LP_LARGE_N) as f64 / 99.0;
        if j % 17 == 0 {
            *xj = 1.0;
        }
        if j % 19 == 0 {
            *xj = 0.0;
        }
    }

    for i in 0..HARD_LP_LARGE_M {
        let mut terms = Vec::new();
        let mut rhs = 0.0;
        for j in 0..HARD_LP_LARGE_N {
            if matches!((i * 31 + j * 17) % 7, 0 | 3 | 5) {
                let scale = 10.0_f64.powi(((i + j) % 9) as i32 - 4);
                let sign = if (i + 2 * j) % 2 == 0 { 1.0 } else { -1.0 };
                let coeff = sign * scale * (1.0 + ((i * j) % 5) as f64 * 0.1);
                terms.push((j, coeff));
                rhs += coeff * x_ref[j];
            }
        }
        model.add_constraint(hard_lp_expr(&vars, &terms).eq_constraint(rhs));
    }

    let obj_terms: Vec<_> = (0..HARD_LP_LARGE_N)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            (j, sign * (0.01 + (j % 11) as f64 * 0.03))
        })
        .collect();
    model.minimize(hard_lp_expr(&vars, &obj_terms));

    let r = model.solve().unwrap();
    hard_lp_assert_model_obj(
        r.objective(),
        HARD_LP_LARGE_EXPECTED_OBJ,
        "hard_lp_large_eq_ub",
    );
}

/// Hard LP: scaled Eq rows + one Ge row + finite UBs.
///
/// SciPy oracle:
/// `linprog(c, A_eq=Aeq, b_eq=beq, A_ub=[-ge], b_ub=[-37], bounds=[(0,1)]*192, method="highs")`
/// returned status 0, fun = 3.404.  The independent construction oracle is the
/// same: each pair satisfies `x_i + y_i = 1`, all `y_i` have zero cost, and
/// `sum x_i >= 37`, so the optimum sets the 37 cheapest `x_i` to one.
#[test]
fn hard_lp_large_scaled_eq_ge_ub_scipy_oracle() {
    let pairs = HARD_LP_LARGE_EQ_GE_PAIRS;
    let n = 2 * pairs;
    let mut rows = Vec::with_capacity(2 * pairs + pairs);
    let mut cols = Vec::with_capacity(2 * pairs + pairs);
    let mut vals = Vec::with_capacity(2 * pairs + pairs);
    let mut b = Vec::with_capacity(pairs + 1);
    let mut ctypes = Vec::with_capacity(pairs + 1);
    let mut c = vec![0.0; n];

    for i in 0..pairs {
        let scale = 10.0_f64.powi((i % 9) as i32 - 4);
        rows.push(i);
        cols.push(i);
        vals.push(scale);
        rows.push(i);
        cols.push(pairs + i);
        vals.push(scale);
        b.push(scale);
        ctypes.push(ConstraintType::Eq);

        c[i] = 0.02 + 0.004 * i as f64;
    }

    let ge_row = pairs;
    for i in 0..pairs {
        rows.push(ge_row);
        cols.push(i);
        vals.push(1.0);
    }
    b.push(HARD_LP_LARGE_EQ_GE_K as f64);
    ctypes.push(ConstraintType::Ge);

    let a = CscMatrix::from_triplets(&rows, &cols, &vals, pairs + 1, n).unwrap();
    let lp = LpProblem::new_general(c.clone(), a, b, ctypes, vec![(0.0, 1.0); n], None).unwrap();
    let mut options = opts();
    options.timeout_secs = Some(HARD_LP_TIMEOUT_SECS);
    let r = solve_lp_with(&lp, &options);

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "hard_lp_large_scaled_eq_ge_ub status"
    );
    assert_obj(
        r.objective,
        HARD_LP_LARGE_EQ_GE_EXPECTED_OBJ,
        "hard_lp_large_scaled_eq_ge_ub obj",
    );
    let sum_x: f64 = r.solution[..pairs].iter().sum();
    hard_lp_assert_resid(
        sum_x,
        HARD_LP_LARGE_EQ_GE_K as f64,
        "hard_lp_large_scaled_eq_ge_ub active Ge",
    );
    for i in 0..HARD_LP_LARGE_EQ_GE_K {
        assert_x(r.solution[i], 1.0, &format!("hard_lp_large_eq_ge x[{i}]"));
    }
    for i in HARD_LP_LARGE_EQ_GE_K..pairs {
        assert_x(r.solution[i], 0.0, &format!("hard_lp_large_eq_ge x[{i}]"));
    }
    for i in 0..pairs {
        let pair_sum = r.solution[i] + r.solution[pairs + i];
        hard_lp_assert_resid(pair_sum, 1.0, &format!("hard_lp_large_eq_ge pair[{i}]"));
    }
}

/// Hard LP: Eq and UB are contradictory.
///
/// SciPy oracle:
/// `linprog([0,0], A_eq=[[1,1]], b_eq=[3], bounds=[(0,1),(0,1)], method="highs")`
/// returned status 2 (infeasible).
#[test]
fn hard_lp_infeasible_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_infeasible_eq_ub_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 1.0);
    let y = model.add_var("y", 0.0, 1.0);
    let vars = [x, y];

    model.add_constraint(hard_lp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(3.0));
    model.minimize(Expression::from_constant(0.0));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "hard_lp_infeasible_eq_ub: expected infeasible, got {err:?}"
    );
}

/// Hard LP: Eq row plus ill-scaled cost has an unbounded improving ray.
///
/// SciPy oracle:
/// `linprog([-1e10,0], A_eq=[[0,1e-10]], b_eq=[0], bounds=[(None,None),(0,1)])`
/// returned status 3 (unbounded).
#[test]
fn hard_lp_unbounded_eq_ill_scaled_cost_expression_scipy_oracle() {
    let mut model = Model::new("hard_lp_unbounded_eq_ill_scaled_cost_expression");
    model.set_timeout(HARD_LP_TIMEOUT_SECS);
    let ray = model.add_var("ray", f64::NEG_INFINITY, HARD_LP_INF);
    let pinned = model.add_var("pinned", 0.0, 1.0);
    let vars = [ray, pinned];

    model.add_constraint(hard_lp_expr(&vars, &[(1, 1e-10)]).eq_constraint(0.0));
    model.minimize(hard_lp_expr(&vars, &[(0, -1e10)]));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Unbounded)),
        "hard_lp_unbounded_eq_ill_scaled_cost: expected unbounded, got {err:?}"
    );
}
