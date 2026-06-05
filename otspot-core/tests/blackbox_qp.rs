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
use otspot_core::sparse::CscMatrix;
use otspot_core::QpProblem;

const INF: f64 = f64::INFINITY;
const EPS_OBJ: f64 = 1e-5;
const EPS_X: f64 = 1e-4;
const EPS_DUAL: f64 = 1e-4;

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
    if !r.dual_solution.is_empty() {
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
    }
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
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[-1.0, -1.0, 1.0, 1.0],
        2,
        2,
    )
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
    assert_eq!(r.status, SolveStatus::Infeasible, "ep_qp_infeasible: status");
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
    if !r.dual_solution.is_empty() {
        assert!(
            r.dual_solution[0].abs() < EPS_DUAL,
            "bva_qp_interior: dual[0]={} must ≈0 (inactive Le)",
            r.dual_solution[0]
        );
    }
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
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        2,
    )
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
    assert_eq!(r.status, SolveStatus::Optimal, "dt_qp_ge_constrained: status");
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
    if !r.dual_solution.is_empty() {
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
}
