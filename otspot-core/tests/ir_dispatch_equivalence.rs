//! `solve_ir` must match the legacy native entry points on every problem class.
//! This locks the canonical IR onto the real solve path (not just adapters).

use otspot_core::architecture::{conic_to_ir, lp_to_ir, milp_to_ir, qp_to_ir};
use otspot_core::conic::{solve_socp, ConeSpec, ConicOptions, ConicProblem};
use otspot_core::mip::{solve_milp, MilpProblem};
use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::qp::{solve_qp_with, QpProblem};
use otspot_core::solve_ir;
use otspot_core::sparse::CscMatrix;
use otspot_ir::{SolveContext, SolveStatus as IrStatus};

fn ctx() -> SolveContext {
    SolveContext::default()
}

#[test]
fn ir_lp_matches_native_lp() {
    // min x + y  s.t. x + y >= 1, 0 <= x,y
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![1.0],
        vec![ConstraintType::Ge],
        vec![(0.0, f64::INFINITY); 2],
        None,
    )
    .unwrap();

    let native = otspot_core::lp::solve_lp_with(&lp, &SolverOptions::default());
    let ir = solve_ir(&lp_to_ir(&lp), &SolverOptions::default(), &ctx());

    assert_eq!(ir.status, IrStatus::Optimal);
    assert_eq!(native.status, SolveStatus::Optimal);
    assert!((ir.objective.unwrap() - native.objective).abs() < 1e-9);
}

#[test]
fn ir_qp_matches_native_qp() {
    // min x^2 + y^2 s.t. x + y >= 1
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let qp = QpProblem::new_all_le(
        q,
        vec![0.0, 0.0],
        a,
        vec![-1.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
    )
    .unwrap();

    let native = solve_qp_with(&qp, &SolverOptions::default());
    let ir = solve_ir(&qp_to_ir(&qp), &SolverOptions::default(), &ctx());

    assert_eq!(native.status, SolveStatus::Optimal);
    assert_eq!(ir.status, IrStatus::Optimal);
    assert!((ir.objective.unwrap() - native.objective).abs() < 1e-6);
    for (a, b) in ir.primal.iter().zip(&native.solution) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn ir_infeasible_lp_is_reported() {
    // x <= -1 with x >= 0
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![-1.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    let ir = solve_ir(&lp_to_ir(&lp), &SolverOptions::default(), &ctx());
    assert_eq!(ir.status, IrStatus::Infeasible);
}

#[test]
fn ir_milp_matches_native_milp() {
    // min -x, x binary => x=1
    let lp = LpProblem::new_general(
        vec![-1.0],
        CscMatrix::new(0, 1),
        Vec::new(),
        Vec::new(),
        vec![(0.0, 1.0)],
        None,
    )
    .unwrap();
    let milp = MilpProblem::new(lp, vec![0]).unwrap();
    let native = solve_milp(
        &milp,
        &SolverOptions::default(),
        &otspot_core::options::MipConfig::default(),
    );
    let canonical = milp_to_ir(&milp).unwrap();
    let ir = solve_ir(&canonical, &SolverOptions::default(), &ctx());
    assert_eq!(native.status, SolveStatus::Optimal);
    assert_eq!(ir.status, IrStatus::Optimal);
    assert!((ir.objective.unwrap() - native.objective).abs() < 1e-9);
}

#[test]
fn ir_socp_matches_native_socp() {
    // min x, -x + s = -1, s >= 0 => x >= 1
    let conic = ConicProblem {
        c: vec![1.0],
        a: CscMatrix::new(0, 1),
        b: Vec::new(),
        g: CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap(),
        h: vec![-1.0],
        cone: ConeSpec {
            l: 1,
            soc: Vec::new(),
        },
    };
    let native = solve_socp(&conic, &ConicOptions::default());
    let ir = solve_ir(&conic_to_ir(&conic), &SolverOptions::default(), &ctx());
    assert_eq!(native.status, SolveStatus::Optimal);
    assert_eq!(ir.status, IrStatus::Optimal);
    assert!((ir.objective.unwrap() - native.objective).abs() < 1e-7);
}
