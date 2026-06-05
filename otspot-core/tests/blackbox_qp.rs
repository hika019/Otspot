//! Black-box tests for QP solve (IPM), presolve, and postsolve/dual stages.
//!
//! Every expected value is hand-computed from the problem data (independent oracle).
//! No expected value is derived by running the solver.
//!
//! Technique labels (cited per test):
//!   EP  = Equivalence Partitioning
//!   BVA = Boundary Value Analysis
//!   DT  = Decision Table
//!
//! QP convention: min 1/2 x'Qx + c'x  (Q is the full Hessian stored in CSC).
//! IMPORTANT: the solver returns 1/2 x'Qx + c'x WITHOUT any constant term.
//! All expected objectives are the internal QP form (no constant offset).

use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, SolveStatus};
use otspot_core::qp::solve_qp_with;
use otspot_core::qp::QpProblemError;
use otspot_core::sparse::CscMatrix;
use otspot_core::QpProblem;
use otspot_model::{Expression, Model, ModelError, QuadExpr, SolveError, Variable};

const INF: f64 = f64::INFINITY;
const EPS_OBJ: f64 = 1e-5;
const EPS_X: f64 = 1e-4;
const EPS_DUAL: f64 = 1e-4;
const HARD_QP_EPS_OBJ: f64 = 5e-5;
const HARD_QP_EPS_X: f64 = 5e-4;
const HARD_QP_TIMEOUT_SECS: f64 = 10.0;
const HARD_QP_ILL_EXPECTED_OBJ: f64 = -2.749_999_999_985_000_4;
const HARD_QP_MICRO_Q: f64 = 1e-14;
const HARD_QP_MICRO_EXPECTED_OBJ: f64 = -999_999.995;
const HARD_QP_DEGENERATE_EXPECTED_OBJ: f64 = -8.25;
const HARD_QP_DUAL_SIGN_EXPECTED_OBJ: f64 = -3.0;
const HARD_QP_LARGE_N: usize = 50;
const HARD_QP_LARGE_M: usize = 10;
const HARD_QP_LARGE_EXPECTED_OBJ: f64 = 0.113_252_558_203_024_54;

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

fn hard_qp_expr(vars: &[Variable], terms: &[(usize, f64)]) -> Expression {
    let mut expr = Expression::from_constant(0.0);
    for &(var_idx, coeff) in terms {
        expr = expr + coeff * vars[var_idx];
    }
    expr
}

fn hard_qp_obj(
    vars: &[Variable],
    linear_terms: &[(usize, f64)],
    diag_q: &[(usize, f64)],
    offdiag_q: &[(usize, usize, f64)],
) -> QuadExpr {
    let mut obj: QuadExpr = hard_qp_expr(vars, linear_terms).into();
    for &(idx, q_val) in diag_q {
        obj = obj + (0.5 * q_val) * vars[idx] * vars[idx];
    }
    for &(row, col, q_val) in offdiag_q {
        obj = obj + q_val * vars[row] * vars[col];
    }
    obj
}

fn hard_qp_assert_model_obj(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < HARD_QP_EPS_OBJ,
        "{label}: obj={actual:.12e} expected={expected:.12e} rel={rel:.3e}"
    );
}

fn hard_qp_assert_model_x(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < HARD_QP_EPS_X,
        "{label}: x={actual:.12e} expected={expected:.12e} diff={diff:.3e}"
    );
}

fn hard_qp_assert_model_route_qp_ipm(route: impl std::fmt::Debug, label: &str) {
    assert_eq!(format!("{route:?}"), "QpIpm", "{label}: route must be QpIpm");
}

// ─── EQUIVALENCE PARTITIONING ──────────────────────────────────────────────

/// EP: Optimal interior point — unconstrained convex quadratic.
///
/// Problem: min (x-2)^2 + (y-3)^2, x,y ∈ (-∞, +∞).
/// QP expansion: (x-2)^2+(y-3)^2 = 1/2*(2x^2+2y^2) + (-4x-6y) + 13.
/// QP form: Q=diag(2,2), c=[-4,-6], no A, bounds=(-inf,+inf).
/// Solver returns 1/2 x'Qx+c'x WITHOUT the constant +13.
///
/// Oracle (hand-solve): stationarity Qx+c = [2x-4, 2y-6] = 0.
///   x*=2, y*=3.
///   Internal obj = 1/2*(2*4+2*9) + (-4*2-6*3) = 13 - 26 = -13.
///
/// KKT: no constraint duals. Stationarity: Qx*+c = [4-4, 6-6] = [0,0] ✓.
#[test]
fn ep_qp_unconstrained_interior_optimal() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::new(0, 2);
    let qp = QpProblem::new(
        q,
        vec![-4.0, -6.0],
        a,
        vec![],
        vec![(f64::NEG_INFINITY, INF), (f64::NEG_INFINITY, INF)],
        vec![],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_unconstrained: status");
    // Internal obj = -13 (1/2*(2*4+2*9) + (-4*2-6*3) = 13 - 26 = -13)
    assert_obj(r.objective, -13.0, "ep_unconstrained obj=-13");
    assert_x(r.solution[0], 2.0, "ep_unconstrained x*=2");
    assert_x(r.solution[1], 3.0, "ep_unconstrained y*=3");
    // KKT stationarity: Qx*+c ≈ 0
    let grad_x = 2.0 * r.solution[0] + (-4.0);
    let grad_y = 2.0 * r.solution[1] + (-6.0);
    assert!(
        grad_x.abs() < EPS_DUAL,
        "ep_unconstrained: Qx+c[0]={grad_x:.2e} must ≈0"
    );
    assert!(
        grad_y.abs() < EPS_DUAL,
        "ep_unconstrained: Qx+c[1]={grad_y:.2e} must ≈0"
    );
}

/// EP: Optimal with equality constraint — KKT solved by hand.
///
/// Problem: min (x-2)^2+(y-3)^2  s.t. x+y=4, x,y ∈ (-∞,+∞).
/// QP expansion: same as above, constant=13 (not included in solver output).
/// QP form: Q=diag(2,2), c=[-4,-6], A=[[1,1]], b=[4], Eq.
///
/// Oracle (KKT by hand):
///   Stationarity: [2x-4+λ, 2y-6+λ]=0 → x=(4-λ)/2, y=(6-λ)/2.
///   Constraint x+y=4: (10-2λ)/2=4 → λ=1.
///   x*=1.5, y*=2.5. Check: 1.5+2.5=4 ✓.
///   Internal obj = 1/2*(2*1.5^2+2*2.5^2) + (-4*1.5-6*2.5) = 8.5 - 21 = -12.5.
///   KKT: Qx*+c+A'λ = [3-4+1, 5-6+1] = [0,0] ✓.
#[test]
fn ep_qp_constrained_equality_optimal() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-4.0, -6.0],
        a,
        vec![4.0],
        vec![(f64::NEG_INFINITY, INF), (f64::NEG_INFINITY, INF)],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_eq_constrained: status");
    // Internal obj = 1/2*(2*1.5^2+2*2.5^2) + (-4*1.5-6*2.5) = 8.5-21 = -12.5
    assert_obj(r.objective, -12.5, "ep_eq_constrained obj=-12.5");
    assert_x(r.solution[0], 1.5, "ep_eq_constrained x*=1.5");
    assert_x(r.solution[1], 2.5, "ep_eq_constrained y*=2.5");

    // KKT stationarity: Qx*+c+A'λ ≈ 0
    assert!(
        !r.dual_solution.is_empty(),
        "ep_eq_constrained: solver must return duals at Optimal"
    );
    let lambda = r.dual_solution[0];
    let kkt_x = 2.0 * r.solution[0] + (-4.0) + lambda;
    let kkt_y = 2.0 * r.solution[1] + (-6.0) + lambda;
    assert!(
        kkt_x.abs() < EPS_DUAL,
        "ep_eq_constrained KKT[x]={kkt_x:.2e}"
    );
    assert!(
        kkt_y.abs() < EPS_DUAL,
        "ep_eq_constrained KKT[y]={kkt_y:.2e}"
    );
    // Primal feasibility: x+y=4
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 4.0).abs() < 1e-4,
        "ep_eq_constrained: x+y={sum:.6} must=4"
    );
}

/// EP: Infeasible QP — contradictory inequalities in all-Le form.
///
/// Problem: min x^2+y^2  s.t. -x-y <= -5 (=x+y>=5), x+y <= 3, x,y free.
/// Oracle: {(x,y): x+y>=5 AND x+y<=3} = ∅ → Infeasible.
/// Uses all-Le form (same as the existing test_qp_infeasible) to ensure
/// the IPM detects infeasibility cleanly.
#[test]
fn ep_qp_infeasible() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    // Row 0: -x-y <= -5  (≡ x+y >= 5)
    // Row 1:  x+y <=  3
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[-1.0, -1.0, 1.0, 1.0], 2, 2)
        .unwrap();
    let qp = QpProblem::new_all_le(
        q,
        vec![0.0, 0.0],
        a,
        vec![-5.0, 3.0],
        vec![(f64::NEG_INFINITY, INF), (f64::NEG_INFINITY, INF)],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "ep_qp_infeasible: status"
    );
}

// ─── BOUNDARY VALUE ANALYSIS ───────────────────────────────────────────────

/// BVA: QP optimal at lower bound — bound active, bound dual > 0.
///
/// Problem: min (x+1)^2  s.t. 0 <= x <= 10.
/// QP expansion: x^2+2x+1 = 1/2*(2x^2) + 2x + const. Q=[2], c=[2].
/// Solver returns 1/2 x'Qx+c'x WITHOUT constant +1.
///
/// Oracle: unconstrained min at x=-1 < lb=0 → pinned to lb.
///   x*=0. Internal obj = 1/2*2*0 + 2*0 = 0.
///   Geometric: (0+1)^2=1 = internal_obj+1 ✓.
/// KKT: Qx*+c-z_lb = 0 → 0+2-z_lb=0 → z_lb=2 (lb active ✓).
#[test]
fn bva_qp_optimal_at_lb() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let a = CscMatrix::new(0, 1);
    let qp = QpProblem::new(q, vec![2.0], a, vec![], vec![(0.0, 10.0)], vec![]).unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_at_lb: status");
    // Internal obj = 0 (no constant; solver reports 1/2*Q*x*^2 + c*x* = 0+0 = 0)
    assert_obj(r.objective, 0.0, "bva_qp_at_lb internal obj=0");
    assert_x(r.solution[0], 0.0, "bva_qp_at_lb x*=0 (lb)");
    // Bound dual ≥ 0 (lb active)
    if !r.bound_duals.is_empty() {
        assert!(
            r.bound_duals[0] > -EPS_DUAL,
            "bva_qp_at_lb: bound_dual[0]={} must be ≥0 (lb active)",
            r.bound_duals[0]
        );
    }
}

/// BVA: QP optimal at upper bound — ub active.
///
/// Problem: min (x-10)^2  s.t. 0 <= x <= 5.
/// QP expansion: x^2-20x+100 = 1/2*(2x^2)-20x+const. Q=[2], c=[-20].
/// Solver returns 1/2 x'Qx+c'x WITHOUT constant +100.
///
/// Oracle: unconstrained min at x=10 > ub=5 → pinned to ub.
///   x*=5. Internal obj = 1/2*2*25 + (-20*5) = 25-100 = -75.
///   Geometric: (5-10)^2=25 = -75+100 ✓.
/// KKT: Qx*+c+z_ub = 0 → 10-20+z_ub=0 → z_ub=10 (ub active ✓).
#[test]
fn bva_qp_optimal_at_ub() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let a = CscMatrix::new(0, 1);
    let qp = QpProblem::new(q, vec![-20.0], a, vec![], vec![(0.0, 5.0)], vec![]).unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_at_ub: status");
    // Internal obj = -75 (1/2*2*25 + (-20)*5 = 25-100 = -75)
    assert_obj(r.objective, -75.0, "bva_qp_at_ub internal obj=-75");
    assert_x(r.solution[0], 5.0, "bva_qp_at_ub x*=5 (ub)");
}

/// BVA: QP optimal strictly interior with inactive constraint.
///
/// Problem: min x^2 + y^2  s.t. x + y <= 4, -5 <= x,y <= 5.
/// QP form: Q=diag(2,2), c=[0,0], A=[[1,1]], b=[4], Le.
///
/// Oracle: unconstrained min at (0,0): 0+0=0 < 4 → Le inactive.
///   x*=0, y*=0. Internal obj=0 (=geometric). Constraint dual=0.
#[test]
fn bva_qp_optimal_interior_inactive_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![4.0],
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_interior: status");
    assert_obj(r.objective, 0.0, "bva_qp_interior obj=0");
    assert_x(r.solution[0], 0.0, "bva_qp_interior x*=0");
    assert_x(r.solution[1], 0.0, "bva_qp_interior y*=0");
    // Inactive constraint → dual ≈ 0
    assert!(
        !r.dual_solution.is_empty(),
        "bva_qp_interior: solver must return duals at Optimal"
    );
    assert!(
        r.dual_solution[0].abs() < EPS_DUAL,
        "bva_qp_interior: dual[0]={} must ≈0 (inactive Le)",
        r.dual_solution[0]
    );
}

/// BVA: Degenerate QP — two constraints both active at the unique optimum.
///
/// Problem: min (x-2)^2+(y-2)^2  s.t. x+y<=4, x+y>=4, 0<=x,y<=5.
/// QP expansion: constant=8 (not in solver output). Q=diag(2,2), c=[-4,-4].
/// The two constraints force x+y=4 exactly.
///
/// Oracle: min (x-2)^2+(y-2)^2 on line x+y=4.
///   By symmetry (c_x=c_y): x*=y*=2, x+y=4 ✓.
///   Internal obj = 1/2*(2*4+2*4) + (-4*2-4*2) = 8-16 = -8.
///   Geometric: (2-2)^2+(2-2)^2=0 = -8+8 ✓.
#[test]
fn bva_qp_degenerate_two_constraints_active() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    // rows: 0=Le(x+y<=4), 1=Ge(x+y>=4)
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
        .unwrap();
    let qp = QpProblem::new(
        q,
        vec![-4.0, -4.0],
        a,
        vec![4.0, 4.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Le, ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_degenerate: status");
    // Internal obj = 1/2*(2*4+2*4) + (-4*2-4*2) = 8-16 = -8
    assert_obj(r.objective, -8.0, "bva_qp_degenerate internal obj=-8");
    assert_x(r.solution[0], 2.0, "bva_qp_degenerate x*=2");
    assert_x(r.solution[1], 2.0, "bva_qp_degenerate y*=2");
    // Primal feasibility: x+y=4
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 4.0).abs() < 1e-4,
        "bva_qp_degenerate: x+y={sum:.6} must=4"
    );
}

// ─── DECISION TABLE ────────────────────────────────────────────────────────

/// DT: Ge + box + min, constraint inactive (interior optimum).
///
/// Problem: min (x-5)^2+(y-5)^2  s.t. x+y>=8, 0<=x,y<=6.
/// QP form: Q=diag(2,2), c=[-10,-10] (expansion constant=50 not included).
///
/// Oracle: unconstrained min at (5,5): 5+5=10>8 → Ge inactive.
///   x*=5, y*=5.
///   Internal obj = 1/2*(2*25+2*25) + (-10*5-10*5) = 50-100 = -50.
#[test]
fn dt_qp_ge_box_interior() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-10.0, -10.0],
        a,
        vec![8.0],
        vec![(0.0, 6.0), (0.0, 6.0)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_qp_ge_interior: status");
    // Internal obj = 50-100 = -50
    assert_obj(r.objective, -50.0, "dt_qp_ge_interior internal obj=-50");
    assert_x(r.solution[0], 5.0, "dt_qp_ge_interior x*=5");
    assert_x(r.solution[1], 5.0, "dt_qp_ge_interior y*=5");
}

/// DT: Ge + box + min, constraint active (optimum pushed off interior).
///
/// Problem: min (x-1)^2+(y-1)^2  s.t. x+y>=6, 0<=x,y<=5.
/// QP form: Q=diag(2,2), c=[-2,-2] (constant=2 not included).
///
/// Oracle (KKT by hand):
///   Stationarity: [2x-2+λ_Ge, 2y-2+λ_Ge]=0.
///   With Ge convention (λ_Ge ≤ 0 for this solver — stored sign TBD):
///   x=y=(2-λ_Ge)/2. x+y=6: 2-λ_Ge=6 → λ_Ge=-4.
///   x*=y*=(2-(-4))/2=3. Check: 3+3=6 ✓.
///   Internal obj = 1/2*(2*9+2*9)+(-2*3-2*3) = 18-12 = 6.
///   Geometric: (3-1)^2+(3-1)^2=8 = 6+2 ✓.
#[test]
fn dt_qp_ge_box_constrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -2.0],
        a,
        vec![6.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "dt_qp_ge_constrained: status"
    );
    // Internal obj = 1/2*(2*9+2*9) + (-2*3-2*3) = 18-12 = 6
    assert_obj(r.objective, 6.0, "dt_qp_ge_constrained internal obj=6");
    assert_x(r.solution[0], 3.0, "dt_qp_ge_constrained x*=3");
    assert_x(r.solution[1], 3.0, "dt_qp_ge_constrained y*=3");
    // Primal feasibility: x+y >= 6
    let sum = r.solution[0] + r.solution[1];
    assert!(
        sum >= 6.0 - 1e-4,
        "dt_qp_ge_constrained: x+y={sum:.6} must ≥6"
    );
}

// ─── POSTSOLVE / DUAL RECOVERY ─────────────────────────────────────────────

/// POSTSOLVE: QP with equality singleton row — dual lifted correctly.
///
/// Problem: min (x-3)^2+(y-4)^2  s.t. y<=6 (Le), x=2 (Eq singleton), 0<=x,y<=10.
/// QP form: Q=diag(2,2), c=[-6,-8] (constant=25 not included).
///   A: row0 (Le, y<=6): A[0,1]=1. row1 (Eq, x=2): A[1,0]=1.
///
/// Oracle: Eq fixes x=2. Remaining: min (2-3)^2+(y-4)^2 = 1+(y-4)^2.
///   min (y-4)^2 s.t. y<=6, 0<=y<=10 → unconstrained min at y=4<6. y*=4.
///   Full: x*=2, y*=4.
///   Internal obj = 1/2*(2*4+2*16)+(-6*2-8*4) = 20-44 = -24.
///   Geometric: (2-3)^2+(4-4)^2=1 = -24+25 ✓.
///
/// KKT (hand-derived, independent):
///   Stationarity for x: 2*x*-6+λ_eq=0 → λ_eq=6-2*2=2.
///   Stationarity for y: 2*y*-8+λ_le=0 → λ_le=8-2*4=0 (Le inactive ✓).
///   Complementarity: λ_le*(y*-6)=0*(4-6)=0 ✓.
#[test]
fn postsolve_qp_dual_recovery_with_eq_singleton() {
    // A col 0 (x): only row 1 (Eq). A col 1 (y): only row 0 (Le).
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[1, 0], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-6.0, -8.0],
        a,
        vec![6.0, 2.0],
        vec![(0.0, 10.0), (0.0, 10.0)],
        vec![ConstraintType::Le, ConstraintType::Eq],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "postsolve_qp_eq: status");
    // Internal obj = 1/2*(2*4+2*16)+(-6*2-8*4) = 20-44 = -24
    assert_obj(r.objective, -24.0, "postsolve_qp_eq internal obj=-24");
    assert_x(r.solution[0], 2.0, "postsolve_qp_eq x*=2");
    assert_x(r.solution[1], 4.0, "postsolve_qp_eq y*=4");

    // Primal feasibility
    assert!(
        r.solution[1] <= 6.0 + 1e-4,
        "postsolve_qp_eq: y={} must ≤6",
        r.solution[1]
    );
    assert!(
        (r.solution[0] - 2.0).abs() < 1e-4,
        "postsolve_qp_eq: x={} must =2",
        r.solution[0]
    );

    // KKT stationarity (independent verification)
    assert!(
        !r.dual_solution.is_empty(),
        "postsolve_qp_eq: solver must return duals at Optimal"
    );
    let lambda_le = r.dual_solution[0]; // row 0: Le
    let lambda_eq = r.dual_solution[1]; // row 1: Eq
                                        // For x: Qx[0]+c[0]+A[1,0]*λ_eq = 2*x*-6+λ_eq = 0 → λ_eq=2
    let kkt_x = 2.0 * r.solution[0] + (-6.0) + lambda_eq;
    // For y: Qx[1]+c[1]+A[0,1]*λ_le = 2*y*-8+λ_le = 0 → λ_le=0
    let kkt_y = 2.0 * r.solution[1] + (-8.0) + lambda_le;
    assert!(
        kkt_x.abs() < EPS_DUAL,
        "postsolve_qp_eq KKT[x]={kkt_x:.2e} must ≈0"
    );
    assert!(
        kkt_y.abs() < EPS_DUAL,
        "postsolve_qp_eq KKT[y]={kkt_y:.2e} must ≈0"
    );
    // Le inactive → λ_le ≈ 0
    assert!(
        lambda_le.abs() < EPS_DUAL,
        "postsolve_qp_eq: λ_le={lambda_le:.2e} must ≈0 (Le inactive)"
    );
}

// ─── EQUIVALENCE PARTITIONING (ADDITIONAL) ─────────────────────────────────────

/// EP: Medium 5-variable diagonal QP — constraint active at optimum.
///
/// Problem: min 1/2*(2x1^2+...+2x5^2) + (-2x1-4x2-6x3-8x4-10x5)
///          s.t. x1+x2+x3+x4+x5 <= 10,  0 <= xi <= 4.
/// Q = diag(2,2,2,2,2), c = [-2,-4,-6,-8,-10].
/// Oracle (KKT by hand):
///   Stationarity: 2*xi* + ci + lambda = 0 → xi* = (-ci - lambda)/2.
///   Le active: sum xi* = 10.  sum(-ci-lambda)/2 = (2+4+6+8+10-5*lambda)/2 = 10
///   → 30-5*lambda=20 → lambda=2.
///   xi* = [0, 1, 2, 3, 4].  All in [0,4]: check 0,1,2,3,4<=4 ✓.
///   Internal obj = sum(xi*^2 + ci*xi*) = (0+1+4+9+16) + (0-4-12-24-40) = 30-80 = -50.
///
/// scipy.optimize.minimize SLSQP confirmed: fun=-50.0, x=[0,1,2,3,4].
#[test]
fn ep_qp_5var_diagonal() {
    // Q diagonal entries: all 2.0 (indices 0-4)
    let q_rows: Vec<usize> = (0..5).collect();
    let q_cols: Vec<usize> = (0..5).collect();
    let q_vals: Vec<f64> = vec![2.0; 5];
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, 5, 5).unwrap();
    // A: single row [1,1,1,1,1] for Le(sum<=10)
    let a_rows: Vec<usize> = vec![0; 5];
    let a_cols: Vec<usize> = (0..5).collect();
    let a_vals: Vec<f64> = vec![1.0; 5];
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 1, 5).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -4.0, -6.0, -8.0, -10.0],
        a,
        vec![10.0],
        vec![(0.0, 4.0); 5],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_qp_5var: status");
    // Internal obj = sum(xi*^2 + ci*xi*) = 30-80 = -50
    assert_obj(r.objective, -50.0, "ep_qp_5var obj=-50");
    assert_x(r.solution[0], 0.0, "ep_qp_5var x[0]*=0");
    assert_x(r.solution[1], 1.0, "ep_qp_5var x[1]*=1");
    assert_x(r.solution[2], 2.0, "ep_qp_5var x[2]*=2");
    assert_x(r.solution[3], 3.0, "ep_qp_5var x[3]*=3");
    assert_x(r.solution[4], 4.0, "ep_qp_5var x[4]*=4");
    // Le constraint binding: sum(x) = 10
    let sum_x: f64 = r.solution.iter().sum();
    assert!(
        (sum_x - 10.0).abs() < 1e-3,
        "ep_qp_5var: sum(x)={sum_x:.6} must≈10"
    );
}

/// EP: Q = 0 (pure linear) — exercises the LP dispatch path inside the QP solver.
///
/// Problem: min -x - 2y  s.t. x + y <= 6,  0 <= x <= 4,  0 <= y <= 4.
/// QP form: Q = 0₂ₓ₂, c = [-1,-2].
/// Oracle (hand-solve): maximize x+2y, c_y=2>c_x=1 → y to ub=4. Then x<=6-4=2.
///   x*=2, y*=4. Internal obj (= geometric since Q=0) = -2-8 = -10.
///   scipy.optimize.minimize SLSQP: fun=-10.0, x=[2,4].
#[test]
fn ep_qp_zero_q_linear_dispatch() {
    let q = CscMatrix::new(2, 2); // all-zero sparse matrix
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-1.0, -2.0],
        a,
        vec![6.0],
        vec![(0.0, 4.0), (0.0, 4.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ep_qp_zero_q: status");
    assert_obj(r.objective, -10.0, "ep_qp_zero_q obj=-10");
    assert_x(r.solution[0], 2.0, "ep_qp_zero_q x*=2");
    assert_x(r.solution[1], 4.0, "ep_qp_zero_q y*=4");
}

// ─── BOUNDARY VALUE ANALYSIS (ADDITIONAL) ─────────────────────────────────────

/// BVA: Ge RHS at the feasible boundary — unique feasible point (4, 2).
///
/// Problem: min (x-4)^2 + y^2  s.t. x + y >= 6,  0 <= x <= 4,  0 <= y <= 2.
/// Q = diag(2,2), c = [-8,0] (expansion of (x-4)^2+y^2 omits constant 16).
/// Oracle: max(x+y) with x∈[0,4], y∈[0,2] is 6, achieved uniquely at (4,2).
///   x* = 4, y* = 2.  Internal obj = 1/2*(2*16+2*4)+(-8*4) = 20-32 = -12.
///   Geometric: (4-4)^2+2^2=4=-12+16 ✓.
///   scipy SLSQP: fun=-12.0, x=[4,2].
#[test]
fn bva_qp_rhs_boundary_exact() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-8.0, 0.0],
        a,
        vec![6.0],
        vec![(0.0, 4.0), (0.0, 2.0)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_boundary: status");
    assert_obj(r.objective, -12.0, "bva_qp_boundary obj=-12");
    assert_x(r.solution[0], 4.0, "bva_qp_boundary x*=4");
    assert_x(r.solution[1], 2.0, "bva_qp_boundary y*=2");
}

/// BVA: Ge RHS + ε beyond the feasible boundary → Infeasible.
///
/// Problem: min x^2+y^2  s.t. x+y >= 6.01,  0 <= x,y <= 3.
/// Oracle: max(x+y)=6 with x,y in [0,3]; 6.01 > 6 → Infeasible.
///   scipy SLSQP: success=False (infeasible).
#[test]
fn bva_qp_rhs_eps_infeasible() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![6.01],
        vec![(0.0, 3.0), (0.0, 3.0)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "bva_qp_rhs_eps_infeasible: status"
    );
}

/// BVA: Variable fixed by Eq constraint — tests postsolve dual recovery.
///
/// Problem: min (1/2)(2x^2+2y^2) + (-2x-8y)  s.t. x=2 (Eq), y<=5, 0<=x,y<=5.
/// Q=diag(2,2), c=[-2,-8].
/// Oracle (KKT by hand):
///   Unconstrained min: [2x-2,2y-8]=0 → (x,y)=(1,4). Eq forces x=2.
///   With x=2: min (1/2)*2*4+(-2*2) for x (constant) + (1/2)*2*y^2+(-8*y).
///   For y: 2y-8=0 → y*=4 (unconstrained, 4<=5 ✓).
///   x*=2, y*=4.  1/2*2*x^2+c_x*x = 4-4=0; 1/2*2*y^2+c_y*y = 16-32=-16. Total=-16.
///   scipy SLSQP: fun=-16.0, x=[2,4].
///
/// KKT stationarity for x: 2*2-2+lambda_eq=0 → lambda_eq=-2.
/// Stationarity for y: 2*4-8+lambda_le=0 → lambda_le=0 (Le y<=5 inactive ✓).
#[test]
fn bva_qp_fixed_by_eq_constraint() {
    // row 0: Eq(x=2), A[0,0]=1; row 1: Le(y<=5), A[1,1]=1
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -8.0],
        a,
        vec![2.0, 5.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Eq, ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bva_qp_eq_fixed: status");
    // Internal obj = 1/2*(2*4) + (-2*2) + 1/2*(2*16) + (-8*4) = 4-4+16-32 = -16
    assert_obj(r.objective, -16.0, "bva_qp_eq_fixed obj=-16");
    assert_x(r.solution[0], 2.0, "bva_qp_eq_fixed x*=2");
    assert_x(r.solution[1], 4.0, "bva_qp_eq_fixed y*=4");

    // KKT stationarity (independent verification)
    assert!(
        !r.dual_solution.is_empty(),
        "bva_qp_eq_fixed: solver must return duals at Optimal"
    );
    let lam_eq = r.dual_solution[0]; // row 0: Eq(x=2)
    let lam_le = r.dual_solution[1]; // row 1: Le(y<=5)
                                     // x: 2*x*-2+lam_eq=0 → lam_eq should be -2
    let kkt_x = 2.0 * r.solution[0] - 2.0 + lam_eq;
    // y: 2*y*-8+lam_le=0 → lam_le should be 0
    let kkt_y = 2.0 * r.solution[1] - 8.0 + lam_le;
    assert!(kkt_x.abs() < EPS_DUAL, "bva_qp_eq_fixed KKT[x]={kkt_x:.2e}");
    assert!(kkt_y.abs() < EPS_DUAL, "bva_qp_eq_fixed KKT[y]={kkt_y:.2e}");
    // Le inactive → dual ≈ 0
    assert!(
        lam_le.abs() < EPS_DUAL,
        "bva_qp_eq_fixed: lam_le={lam_le:.2e} must≈0 (Le inactive)"
    );
}

// ─── DECISION TABLE (ADDITIONAL) ───────────────────────────────────────────────

/// DT: Le + box + min, constraint active at optimum.
///
/// DT cell: sense=Le, bound=box, objective=min; Le is ACTIVE (pushes off interior).
///
/// Problem: min (1/2)*2x^2 + (1/2)*2y^2 + (-4y)  s.t. x+y <= 1, 0<=x,y<=5.
/// Q=diag(2,2), c=[0,-4].
/// Oracle: unconstrained min at x*=0, y*=2.  0+2=2 > 1 → Le active.
///   KKT: [2x+λ, 2y-4+λ]=0, x+y=1.
///   x at lb=0 (need to check). From y: y=(4-λ)/2. With x+y=1: x=1-y=1-(4-λ)/2=(λ-2)/2.
///   x>=0 → λ>=2.  x at lb: rc[x]=2x+λ >= 0 → λ≥0.  From y: 2*y-4+λ=0 → y=(4-λ)/2.
///   If x=0: x=0=(λ-2)/2 → λ=2. y=(4-2)/2=1. Check: 0+1=1 ✓.
///   x*=0, y*=1. Internal obj = 0+(1-4) = -3.
///
/// scipy SLSQP: fun=-3.0, x=[0,1].
#[test]
fn dt_qp_le_box_min_active() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0, -4.0],
        a,
        vec![1.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_qp_le_active: status");
    // Internal obj = 1/2*2*0 + 0*0 + 1/2*2*1 + (-4)*1 = 0+1-4 = -3
    assert_obj(r.objective, -3.0, "dt_qp_le_active obj=-3");
    assert_x(r.solution[0], 0.0, "dt_qp_le_active x*=0");
    assert_x(r.solution[1], 1.0, "dt_qp_le_active y*=1");
    // Le binding
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "dt_qp_le_active: x+y={sum:.6} must≈1"
    );
}

/// DT: Le + free variable + min, Le constraint active.
///
/// DT cell: sense=Le, bound=free(-∞,+∞), objective=min; Le is ACTIVE.
///
/// Problem: min (1/2)*2x^2+(1/2)*2y^2+(-4x-2y)  s.t. x+y <= 2, x,y ∈ (-∞,+∞).
/// Q=diag(2,2), c=[-4,-2].
/// Oracle: unconstrained min at x*=2,y*=1. 2+1=3>2 → Le active.
///   KKT: [2x-4+λ, 2y-2+λ]=0, x+y=2.
///   x=(4-λ)/2, y=(2-λ)/2. x+y=3-λ=2 → λ=1.
///   x*=1.5, y*=0.5.
///   Internal obj = 1/2*(2*2.25+2*0.25)+(-4*1.5-2*0.5) = 2.5+(-6-1) = -4.5.
///
/// scipy SLSQP: fun=-4.5, x=[1.5, 0.5].
#[test]
fn dt_qp_le_free_min_active() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-4.0, -2.0],
        a,
        vec![2.0],
        vec![(f64::NEG_INFINITY, INF), (f64::NEG_INFINITY, INF)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_qp_le_free: status");
    // Internal obj = -4.5
    assert_obj(r.objective, -4.5, "dt_qp_le_free obj=-4.5");
    assert_x(r.solution[0], 1.5, "dt_qp_le_free x*=1.5");
    assert_x(r.solution[1], 0.5, "dt_qp_le_free y*=0.5");

    // KKT stationarity check (λ=1)
    assert!(
        !r.dual_solution.is_empty(),
        "dt_qp_le_free: solver must return duals at Optimal"
    );
    let lam = r.dual_solution[0];
    let kkt_x = 2.0 * r.solution[0] - 4.0 + lam;
    let kkt_y = 2.0 * r.solution[1] - 2.0 + lam;
    assert!(kkt_x.abs() < EPS_DUAL, "dt_qp_le_free KKT[x]={kkt_x:.2e}");
    assert!(kkt_y.abs() < EPS_DUAL, "dt_qp_le_free KKT[y]={kkt_y:.2e}");
}

/// DT: Eq + box + min — unique interior optimum on constraint manifold.
///
/// DT cell: sense=Eq, bound=box, objective=min.
///
/// Problem: min (1/2)*2x^2+(1/2)*2y^2+(-2x-6y)  s.t. x+y=5, 0<=x,y<=5.
/// Q=diag(2,2), c=[-2,-6].
/// Oracle (KKT by hand):
///   Stationarity: [2x-2+λ, 2y-6+λ]=0. x=(2-λ)/2, y=(6-λ)/2. x+y=4-λ=5 → λ=-1.
///   x*=(2-(-1))/2=1.5, y*=(6-(-1))/2=3.5. Both in [0,5] ✓.
///   Internal obj = 1/2*(2*2.25+2*12.25)+(-2*1.5-6*3.5) = 14.5-24 = -9.5.
///
/// scipy SLSQP: fun=-9.5, x=[1.5, 3.5].
#[test]
fn dt_qp_eq_box_min() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -6.0],
        a,
        vec![5.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "dt_qp_eq_box: status");
    assert_obj(r.objective, -9.5, "dt_qp_eq_box obj=-9.5");
    assert_x(r.solution[0], 1.5, "dt_qp_eq_box x*=1.5");
    assert_x(r.solution[1], 3.5, "dt_qp_eq_box y*=3.5");
    // Primal: x+y=5
    let sum = r.solution[0] + r.solution[1];
    assert!(
        (sum - 5.0).abs() < 1e-3,
        "dt_qp_eq_box: x+y={sum:.6} must=5"
    );
    // KKT: λ=-1 (Eq dual); stationarity
    assert!(
        !r.dual_solution.is_empty(),
        "dt_qp_eq_box: solver must return duals at Optimal"
    );
    let lam = r.dual_solution[0];
    let kkt_x = 2.0 * r.solution[0] - 2.0 + lam;
    let kkt_y = 2.0 * r.solution[1] - 6.0 + lam;
    assert!(kkt_x.abs() < EPS_DUAL, "dt_qp_eq_box KKT[x]={kkt_x:.2e}");
    assert!(kkt_y.abs() < EPS_DUAL, "dt_qp_eq_box KKT[y]={kkt_y:.2e}");
}

// ─── STATE TRANSITION (ADDITIONAL) ─────────────────────────────────────────────

/// ST: Le constraint transitions inactive → active as RHS tightens.
///
/// Problem: min (x-3)^2+(y-3)^2 = (1/2)*(2x^2+2y^2)+(-6x-6y)
///          s.t. x+y <= b,  0<=x,y<=5.
/// Q=diag(2,2), c=[-6,-6].  Unconstrained min at (3,3).
///
/// Oracle (KKT for constrained case):
///   When b >= 6 (=3+3): Le inactive. x*=3,y*=3. Internal obj=-18.
///   When b < 6: Le active. x+y=b, x=y=b/2. obj = 1/2*(2*(b/2)^2*2)+(-6*b)=b^2/2-6b.
///
/// Steps verified by scipy SLSQP:
///   b=8 (inactive): obj=-18.0, x=[3,3].
///   b=5 (active):   obj=-17.5, x=[2.5,2.5].
///   b=2 (active):   obj=-10.0, x=[1,1].
#[test]
fn st_qp_le_inactive_to_active() {
    let build = |b: f64| {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        QpProblem::new(
            q,
            vec![-6.0, -6.0],
            a,
            vec![b],
            vec![(0.0, 5.0), (0.0, 5.0)],
            vec![ConstraintType::Le],
        )
        .unwrap()
    };

    // b=8: Le inactive, unconstrained opt (3,3) satisfies 3+3=6<=8
    let r1 = solve_qp_with(&build(8.0), &opts());
    assert_eq!(r1.status, SolveStatus::Optimal, "st_qp_inactive P1: status");
    assert_obj(r1.objective, -18.0, "st_qp_inactive P1 obj=-18");
    assert_x(r1.solution[0], 3.0, "st_qp_inactive P1 x*=3");
    assert_x(r1.solution[1], 3.0, "st_qp_inactive P1 y*=3");
    // Le inactive → dual ≈ 0
    assert!(
        !r1.dual_solution.is_empty(),
        "st_qp_inactive P1: solver must return duals at Optimal"
    );
    assert!(
        r1.dual_solution[0].abs() < EPS_DUAL,
        "st_qp_inactive P1: dual={} must≈0",
        r1.dual_solution[0]
    );

    // b=5: Le active, x*=y*=2.5
    let r2 = solve_qp_with(&build(5.0), &opts());
    assert_eq!(r2.status, SolveStatus::Optimal, "st_qp_active P2: status");
    assert_obj(r2.objective, -17.5, "st_qp_active P2 obj=-17.5");
    assert_x(r2.solution[0], 2.5, "st_qp_active P2 x*=2.5");
    assert_x(r2.solution[1], 2.5, "st_qp_active P2 y*=2.5");
    let sum2 = r2.solution[0] + r2.solution[1];
    assert!(
        (sum2 - 5.0).abs() < 1e-3,
        "st_qp_active P2: x+y={sum2:.6} must≈5"
    );

    // b=2: Le active, x*=y*=1
    let r3 = solve_qp_with(&build(2.0), &opts());
    assert_eq!(r3.status, SolveStatus::Optimal, "st_qp_active P3: status");
    assert_obj(r3.objective, -10.0, "st_qp_active P3 obj=-10");
    assert_x(r3.solution[0], 1.0, "st_qp_active P3 x*=1");
    assert_x(r3.solution[1], 1.0, "st_qp_active P3 y*=1");
}

/// ST: Ge RHS → feasible at exact boundary, infeasible beyond.
///
/// Problem: min x^2+y^2  s.t. x+y >= b,  0<=x,y<=3.  Q=diag(2,2), c=[0,0].
/// P1 (b=6): unique feasible point (3,3). obj=18.
/// P2 (b=6.01): max(x+y)=6 < 6.01 → Infeasible.
#[test]
fn st_qp_ge_feasible_to_infeasible() {
    let build = |b: f64| {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![b],
            vec![(0.0, 3.0), (0.0, 3.0)],
            vec![ConstraintType::Ge],
        )
        .unwrap()
    };
    let r1 = solve_qp_with(&build(6.0), &opts());
    assert_eq!(
        r1.status,
        SolveStatus::Optimal,
        "st_qp_ge_infeas P1: status"
    );
    assert_obj(r1.objective, 18.0, "st_qp_ge_infeas P1 obj=18");

    let r2 = solve_qp_with(&build(6.01), &opts());
    assert_eq!(
        r2.status,
        SolveStatus::Infeasible,
        "st_qp_ge_infeas P2: Infeasible"
    );
}

// ─── PAIRWISE METHOD ───────────────────────────────────────────────────────────
//
// Parameters and value domains:
//   P1 sense:  {Le, Ge, Eq}
//   P2 bound:  {free(-∞,+∞), lb-only([0,+∞)), box([lb,ub]), fixed([a,a])}
//   P3 obj:    {min}  (QP is always convex, so maximization not standard)
//   P4 scale:  {unit(~1), ill(~1e6 spread)}
//   P5 degen:  {non-degenerate(interior or single active), degenerate(≥2 constraints active)}
//
// Representative parameter combination coverage (not full pairwise — see LP set):
// | Test | P1  | P2      | P4   | P5    |
// |------|-----|---------|------|-------|
// | pw1  | Le  | box     | unit | non   |
// | pw2  | Ge  | box     | unit | degen |
// | pw3  | Eq  | free    | ill  | non   |
// | pw4  | Le  | fixed   | unit | non   |
// | pw5  | Le  | lb-only | unit | non   |

/// PW: Le + box + min + unit + non-degenerate (interior optimum, Le inactive).
///
/// Problem: min (1/2)*(2x^2+2y^2)+(-2x-4y)  s.t. x+y<=10, 0<=x<=5, 0<=y<=5.
/// Q=diag(2,2), c=[-2,-4].
/// Oracle: unconstrained min at x*=1,y*=2. 1+2=3<=10 → Le inactive.
///   Internal obj = 1/2*(2+8)+(-2-8) = 5-10 = -5.
///   scipy SLSQP: fun=-5.0, x=[1,2].
#[test]
fn pw_qp_le_box_min_unit_nondeg() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -4.0],
        a,
        vec![10.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_qp_le_box: status");
    assert_obj(r.objective, -5.0, "pw_qp_le_box obj=-5");
    assert_x(r.solution[0], 1.0, "pw_qp_le_box x*=1");
    assert_x(r.solution[1], 2.0, "pw_qp_le_box y*=2");
    // Le inactive → dual ≈ 0
    assert!(
        !r.dual_solution.is_empty(),
        "pw_qp_le_box: solver must return duals at Optimal"
    );
    assert!(
        r.dual_solution[0].abs() < EPS_DUAL,
        "pw_qp_le_box: Le dual={} must≈0 (inactive)",
        r.dual_solution[0]
    );
}

/// PW: Ge + box + min + unit + degenerate (two Ge constraints both active).
///
/// Problem: min (1/2)*(2x^2+2y^2)  s.t. x+y>=4 (Ge), x>=2 (Ge), 0<=x,y<=5.
/// Q=diag(2,2), c=[0,0].
/// Oracle (KKT by hand):
///   Unconstrained min at (0,0). Both Ge push solution up.
///   Active: x+y=4, x=2 → y=2. KKT:
///     [2*2+λ1+λ2, 2*2+λ1]=0 → λ1=-4, from y: impossible if λ1>=0.
///   (Ge convention: L=f - λ'(Ax-b), λ>=0 for Ge).
///   ∇f - A'λ = 0. A'λ = [λ1+λ2, λ1]. [4-λ1-λ2, 4-λ1]=[0,0].
///   λ1=4, λ2=0. Both >=0 ✓. Degenerate: λ2=0 (Ge x>=2 active but dual=0).
///   x*=2, y*=2.  Internal obj = 1/2*(2*4+2*4) = 8.
///
/// scipy SLSQP: fun=8.0, x=[2,2].
#[test]
fn pw_qp_ge_box_min_unit_degen() {
    // row 0: Ge(x+y>=4), row 1: Ge(x>=2)
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![4.0, 2.0],
        vec![(0.0, 5.0), (0.0, 5.0)],
        vec![ConstraintType::Ge, ConstraintType::Ge],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_qp_ge_degen: status");
    assert_obj(r.objective, 8.0, "pw_qp_ge_degen obj=8");
    assert_x(r.solution[0], 2.0, "pw_qp_ge_degen x*=2");
    assert_x(r.solution[1], 2.0, "pw_qp_ge_degen y*=2");
}

/// PW: Eq + free + min + ill-scaled.
///
/// Problem: min (1/2)*(2e6·x^2+2e6·y^2)+(-4e6·x-6e6·y)  s.t. x+y=5, x,y∈(-∞,+∞).
/// Q=diag(2e6,2e6), c=[-4e6,-6e6].
/// Oracle (KKT by hand):
///   Stationarity: [2e6·x-4e6+λ, 2e6·y-6e6+λ]=0. x=(4e6-λ)/(2e6), y=(6e6-λ)/(2e6).
///   x+y=5: (10e6-2λ)/(2e6)=5 → λ=0. x*=2, y*=3.
///   (Same as unconstrained opt: 2+3=5 satisfies Eq exactly.)
///   Internal obj = 1/2*(2e6*4+2e6*9)+(-4e6*2-6e6*3) = 13e6-26e6 = -13e6.
///
/// scipy SLSQP: fun=-1.3e7, x=[2,3].
#[test]
fn pw_qp_eq_free_min_ill() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2e6, 2e6], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-4e6, -6e6],
        a,
        vec![5.0],
        vec![(f64::NEG_INFINITY, INF), (f64::NEG_INFINITY, INF)],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_qp_eq_ill: status");
    assert_obj(r.objective, -13e6, "pw_qp_eq_ill obj=-13e6");
    assert_x(r.solution[0], 2.0, "pw_qp_eq_ill x*=2");
    assert_x(r.solution[1], 3.0, "pw_qp_eq_ill y*=3");
}

/// PW: Le + fixed variable + min + unit + non-degenerate.
///
/// Problem: min (1/2)*2·x^2 + (-6·x)  s.t. x <= 5,  x ∈ [2,2] (fixed by bounds).
/// Q=[2], c=[-6].
/// Oracle: x=2 forced by bounds. Le(x<=5): 2<=5 ✓ (inactive).
///   Internal obj = 1/2*2*4+(-6*2) = 4-12 = -8.
///   Geometric: (2-3)^2=1. -8+9=1 ✓ (unconstrained min would be at x=3).
///
/// scipy SLSQP: fun=-8.0, x=[2].
#[test]
fn pw_qp_le_fixed_min_unit() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-6.0],
        a,
        vec![5.0],
        vec![(2.0, 2.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_qp_le_fixed: status");
    assert_obj(r.objective, -8.0, "pw_qp_le_fixed obj=-8");
    assert_x(r.solution[0], 2.0, "pw_qp_le_fixed x*=2");
}

/// PW: Le + lb-only + min + unit + non-degenerate (interior optimum, Le inactive).
///
/// Problem: min (x-1)^2+(y-1)^2  s.t. x+y<=3, x,y >= 0 (lb-only bounds).
/// Q=diag(2,2), c=[-2,-2] (constant=2 not in solver output).
/// Oracle: unconstrained min at x*=1,y*=1. 1+1=2<=3 → Le inactive.
///   Internal obj = 1/2*(2+2)+(-2-2) = 2-4 = -2.
///   scipy SLSQP: fun=-2.0, x=[1,1].
#[test]
fn pw_qp_le_lbonly_min_unit_nondeg() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -2.0],
        a,
        vec![3.0],
        vec![(0.0, INF), (0.0, INF)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "pw_qp_le_lbonly: status");
    assert_obj(r.objective, -2.0, "pw_qp_le_lbonly obj=-2");
    assert_x(r.solution[0], 1.0, "pw_qp_le_lbonly x*=1");
    assert_x(r.solution[1], 1.0, "pw_qp_le_lbonly y*=1");
    // Le inactive → dual ≈ 0
    assert!(
        !r.dual_solution.is_empty(),
        "pw_qp_le_lbonly: solver must return duals at Optimal"
    );
    assert!(
        r.dual_solution[0].abs() < EPS_DUAL,
        "pw_qp_le_lbonly: Le dual={} must≈0 (inactive)",
        r.dual_solution[0]
    );
}

// ─── CLASSIFICATION TREE METHOD ────────────────────────────────────────────────
//
// Classification tree for QP test design:
//
// Root
// ├─ Hessian structure
// │   ├─ Diagonal (separable) → ep_qp_5var_diagonal, pw tests
// │   └─ Dense (off-diagonal) → ct_qp_dense_hessian_3var
// ├─ Constraint type
// │   ├─ None (unconstrained) → ep_qp_unconstrained_interior_optimal
// │   ├─ Le (inequality) → dt tests, st tests, pw tests
// │   ├─ Ge (inequality) → bva tests, pw tests
// │   └─ Eq (equality) → ep_qp_constrained_equality_optimal, dt_qp_eq_box_min
// ├─ Active set at optimum
// │   ├─ Interior (no bounds/constraints active) → ep_qp_unconstrained, pw_qp_le_box_nondeg
// │   ├─ Single bound/constraint active → bva tests, dt tests
// │   └─ Multiple active → bva_qp_degenerate_two_constraints_active, pw_qp_ge_box_degen
// └─ Scale
//     ├─ Unit → most tests
//     └─ Ill-scaled → pw_qp_eq_free_min_ill

/// CT: QP with dense (off-diagonal) Hessian — unconstrained interior optimum.
///
/// Classification leaf: H=dense, constraint=Le(inactive), active-set=interior.
///
/// Problem: min (1/2)*x'Qx + c'x  s.t. x+y+z <= 6,  0 <= x,y,z <= 4.
/// Q = [[2,1,0],[1,2,1],[0,1,2]],  c = [-4,-6,-4].
/// Oracle (hand-solve unconstrained): Qx+c=0.
///   [2x+y-4, x+2y+z-6, y+2z-4]=0.
///   x=1: y=4-2=2, z=(4-2)/2=1. x+y+z=4<=6 ✓ (Le inactive). No bound violations.
///   Internal obj = (1/2)*[1,2,1]*Q*[1,2,1]' + c'*[1,2,1]
///     = (1/2)*(2+2+2+4) + (-4-12-4) = 5 + (-20) = wait let me recompute.
///   x'Qx = 1*(2*1+2) + 2*(1+4+1) + 1*(2+2) = (4)+(12)+(4) = 20. Internal = 10+(-4-12-4) = -10.
///
/// scipy SLSQP: fun=-10.0, x=[1,2,1].
#[test]
fn ct_qp_dense_hessian_3var() {
    // Q = [[2,1,0],[1,2,1],[0,1,2]] in CSC
    // Column 0: rows 0,1 → values 2,1
    // Column 1: rows 0,1,2 → values 1,2,1
    // Column 2: rows 1,2 → values 1,2
    let q = CscMatrix::from_triplets(
        &[0, 1, 0, 1, 2, 1, 2],
        &[0, 0, 1, 1, 1, 2, 2],
        &[2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        3,
        3,
    )
    .unwrap();
    let a_rows: Vec<usize> = vec![0; 3];
    let a_cols: Vec<usize> = vec![0, 1, 2];
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &[1.0, 1.0, 1.0], 1, 3).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-4.0, -6.0, -4.0],
        a,
        vec![6.0],
        vec![(0.0, 4.0); 3],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &opts());
    assert_eq!(r.status, SolveStatus::Optimal, "ct_qp_dense: status");
    // x'Qx = 20, (1/2)*20 = 10; c'x = -4-12-4 = -20. Internal obj = -10.
    assert_obj(r.objective, -10.0, "ct_qp_dense obj=-10");
    assert_x(r.solution[0], 1.0, "ct_qp_dense x*=1");
    assert_x(r.solution[1], 2.0, "ct_qp_dense y*=2");
    assert_x(r.solution[2], 1.0, "ct_qp_dense z*=1");
    // Le inactive → dual ≈ 0
    assert!(
        !r.dual_solution.is_empty(),
        "ct_qp_dense: solver must return duals at Optimal"
    );
    assert!(
        r.dual_solution[0].abs() < EPS_DUAL,
        "ct_qp_dense: Le dual={} must≈0 (inactive)",
        r.dual_solution[0]
    );
    // KKT stationarity: Qx*+c = [2+2-4, 1+4+1-6, 2+2-4] = [0,0,0]
    let kkt0 = 2.0 * r.solution[0] + r.solution[1] - 4.0;
    let kkt1 = r.solution[0] + 2.0 * r.solution[1] + r.solution[2] - 6.0;
    let kkt2 = r.solution[1] + 2.0 * r.solution[2] - 4.0;
    assert!(kkt0.abs() < EPS_DUAL, "ct_qp_dense KKT[0]={kkt0:.2e}");
    assert!(kkt1.abs() < EPS_DUAL, "ct_qp_dense KKT[1]={kkt1:.2e}");
    assert!(kkt2.abs() < EPS_DUAL, "ct_qp_dense KKT[2]={kkt2:.2e}");
}

// ─── HARD DATA SENTINELS (SCIPY ORACLES, MODEL EXPRESSION API) ─────────────

/// Hard QP: ill-scaled Q with a tiny off-diagonal value.
///
/// SciPy oracle:
/// SLSQP on `Q=[[1e10,1e-6],[1e-6,2]], c=[-1e5,-3], bounds=[(0,10)]*2`
/// returned success, fun = -2.7499999999850004, x = [9.99999985e-6, 1.5].
#[test]
fn hard_qp_ill_scaled_q_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_ill_scaled_q_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 10.0), model.add_var("y", 0.0, 10.0)];

    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -1e5), (1, -3.0)],
        &[(0, 1e10), (1, 2.0)],
        &[(0, 1, 1e-6)],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_ILL_EXPECTED_OBJ,
        "hard_qp_ill_scaled_q",
    );
    hard_qp_assert_model_x(r[vars[0]], 1.0e-5, "hard_qp_ill_scaled_q x");
    hard_qp_assert_model_x(r[vars[1]], 1.5, "hard_qp_ill_scaled_q y");
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_ill_scaled_q");
}

/// Hard QP: micro curvature just above sparse DROP_TOL must stay on the QP path.
///
/// SciPy oracle:
/// SLSQP on `min 0.5*1e-14*x^2 - x, 0<=x<=1e6`
/// returned success, fun = -999999.995, x = [1e6].
#[test]
fn hard_qp_micro_curvature_routes_to_qp_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_micro_curvature_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 1_000_000.0);
    let vars = [x];

    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -1.0)],
        &[(0, HARD_QP_MICRO_Q)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_MICRO_EXPECTED_OBJ,
        "hard_qp_micro_curvature",
    );
    hard_qp_assert_model_x(r[x], 1_000_000.0, "hard_qp_micro_curvature x at ub");
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_micro_curvature");
}

/// Hard QP: KKT degeneracy with several active bounds and an equality.
///
/// SciPy oracle:
/// SLSQP on `Q=diag(2,2,2), c=[-4,-4,-1], x+y=4, bounds=(0,2),(0,2),(0,0.5)`
/// returned success, fun = -8.25, x = [2, 2, 0.5].
#[test]
fn hard_qp_kkt_degenerate_active_bounds_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_kkt_degenerate_active_bounds_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![
        model.add_var("x", 0.0, 2.0),
        model.add_var("y", 0.0, 2.0),
        model.add_var("z", 0.0, 0.5),
    ];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(4.0));
    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -4.0), (1, -4.0), (2, -1.0)],
        &[(0, 2.0), (1, 2.0), (2, 2.0)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_DEGENERATE_EXPECTED_OBJ,
        "hard_qp_kkt_degenerate",
    );
    for (idx, expected) in [2.0, 2.0, 0.5].into_iter().enumerate() {
        hard_qp_assert_model_x(
            r[vars[idx]],
            expected,
            &format!("hard_qp_kkt_degenerate x{idx}"),
        );
    }
    assert!(
        !r.bound_duals.is_empty(),
        "hard_qp_kkt_degenerate must return bound duals"
    );
}

/// Hard QP: active Le constraint whose dual sign is a tight optimality sentinel.
///
/// SciPy oracle:
/// SLSQP on `Q=diag(2,0.2), c=[-4,-0.05], x+y<=1, x,y>=0`
/// returned success, fun = -3.0, x = [1, 0].
#[test]
fn hard_qp_dual_sign_le_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_dual_sign_le_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 10.0), model.add_var("y", 0.0, 10.0)];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).leq(1.0));
    model.minimize(hard_qp_obj(
        &vars,
        &[(0, -4.0), (1, -0.05)],
        &[(0, 2.0), (1, 0.2)],
        &[],
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_DUAL_SIGN_EXPECTED_OBJ,
        "hard_qp_dual_sign",
    );
    hard_qp_assert_model_x(r[vars[0]], 1.0, "hard_qp_dual_sign x");
    hard_qp_assert_model_x(r[vars[1]], 0.0, "hard_qp_dual_sign y");
    let duals = r.dual_solution.as_ref().expect("hard_qp_dual_sign dual");
    assert!(
        duals[0] >= -EPS_DUAL,
        "hard_qp_dual_sign: Le dual must be non-negative in the Model API convention, got {}",
        duals[0]
    );
}

/// Hard QP: n=50 sparse Q + Eq + UB synthetic stress instance.
///
/// SciPy oracle:
/// deterministic Q/c/A/x_ref below, then SLSQP with equality constraints and
/// `[0,1]` bounds returned success, fun = 0.11325255820302454.
#[test]
fn hard_qp_large_sparse_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_large_sparse_eq_ub_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|idx| model.add_var(&format!("x{idx}"), 0.0, 1.0))
        .collect();

    let x_ref: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| 0.1 + 0.8 * ((j * 23) % HARD_QP_LARGE_N) as f64 / 49.0)
        .collect();

    for i in 0..HARD_QP_LARGE_M {
        let mut terms = Vec::new();
        let mut rhs = 0.0;
        for j in 0..HARD_QP_LARGE_N {
            if matches!((i * 13 + j * 7) % 11, 0 | 2 | 5) {
                let coeff = ((i + j) % 5) as f64 * 0.1 - 0.2
                    + if i == j % HARD_QP_LARGE_M { 0.05 } else { 0.0 };
                terms.push((j, coeff));
                rhs += coeff * x_ref[j];
            }
        }
        model.add_constraint(hard_qp_expr(&vars, &terms).eq_constraint(rhs));
    }

    let linear_terms: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| (j, ((j % 9) as f64 - 4.0) * 0.02))
        .collect();
    let diag_terms: Vec<_> = (0..HARD_QP_LARGE_N)
        .map(|j| (j, 0.2 + (j % 7) as f64 * 0.05))
        .collect();
    let offdiag_terms: Vec<_> = (0..HARD_QP_LARGE_N - 1)
        .filter(|j| j % 5 == 0)
        .map(|j| (j, j + 1, 0.005))
        .collect::<Vec<_>>();
    model.minimize(hard_qp_obj(
        &vars,
        &linear_terms,
        &diag_terms,
        &offdiag_terms,
    ));

    let r = model.solve().unwrap();
    hard_qp_assert_model_obj(
        r.objective(),
        HARD_QP_LARGE_EXPECTED_OBJ,
        "hard_qp_large_sparse",
    );
    hard_qp_assert_model_route_qp_ipm(r.stats.route, "hard_qp_large_sparse");
}

/// Hard QP: API-level rejection of NaN/Inf coefficients in QpProblem::new.
///
/// This guard is direct core API coverage because `QpProblem::new` is the API
/// that must reject non-finite Q/A values before solve.
#[test]
fn hard_qp_nonfinite_coefficients_rejected_by_qpproblem_new() {
    let q_nan = CscMatrix::from_triplets(&[0], &[0], &[f64::NAN], 1, 1);
    assert!(
        matches!(
            q_nan,
            Err(otspot_core::error::SolverError::NonFiniteCoefficient { .. })
        ),
        "CscMatrix must reject NaN before QpProblem construction"
    );

    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let a_inf = CscMatrix::from_triplets(&[0], &[0], &[f64::INFINITY], 1, 1);
    assert!(
        matches!(
            a_inf,
            Err(otspot_core::error::SolverError::NonFiniteCoefficient { .. })
        ),
        "CscMatrix must reject Inf A before QpProblem construction"
    );

    let q_direct = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1)
        .unwrap()
        .scale_values(f64::NAN);
    let a = CscMatrix::new(0, 1);
    let err = QpProblem::new(q_direct, vec![0.0], a, vec![], vec![(0.0, 1.0)], vec![]).unwrap_err();
    assert!(
        matches!(err, QpProblemError::NonFiniteCoefficient { field: "Q", .. }),
        "QpProblem::new must reject non-finite Q, got {err:?}"
    );

    let q = q;
    let a_direct = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1)
        .unwrap()
        .scale_values(f64::NAN);
    let err = QpProblem::new(
        q,
        vec![0.0],
        a_direct,
        vec![0.0],
        vec![(0.0, 1.0)],
        vec![ConstraintType::Eq],
    )
    .unwrap_err();
    assert!(
        matches!(err, QpProblemError::NonFiniteCoefficient { field: "A", .. }),
        "QpProblem::new must reject non-finite A, got {err:?}"
    );
}

/// Hard QP: Eq and UB are contradictory.
///
/// SciPy oracle:
/// SLSQP on `Q=I, c=0, x+y=3, bounds=[(0,1),(0,1)]` returned failure
/// with incompatible constraints.
#[test]
fn hard_qp_infeasible_eq_ub_expression_scipy_oracle() {
    let mut model = Model::new("hard_qp_infeasible_eq_ub_expression");
    model.set_timeout(HARD_QP_TIMEOUT_SECS);
    let vars = vec![model.add_var("x", 0.0, 1.0), model.add_var("y", 0.0, 1.0)];
    model.add_constraint(hard_qp_expr(&vars, &[(0, 1.0), (1, 1.0)]).eq_constraint(3.0));
    model.minimize(hard_qp_obj(&vars, &[], &[(0, 1.0), (1, 1.0)], &[]));

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "hard_qp_infeasible_eq_ub: expected infeasible, got {err:?}"
    );
}
