//! Reproduce LP simplex complementarity drift observed in `diag_kkt_proptest`
//! seed cases. Quantifies `|y_i · slack_i| / scale` for each regression LP and
//! prints comp_drift fact (#45 investigation).

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use solver::solve_lp_with;
use solver::sparse::CscMatrix;

fn comp_drift(prob: &LpProblem, res: &SolverResult) -> f64 {
    let x = res.solution.as_slice();
    let y = res.dual_solution.as_slice();
    if y.len() != prob.num_constraints {
        return f64::INFINITY;
    }
    let ax = prob.a.mat_vec_mul(x).unwrap();
    let mut comp = 0.0_f64;
    for (i, ct) in prob.constraint_types.iter().enumerate() {
        let slack = match ct {
            ConstraintType::Eq => continue,
            ConstraintType::Le => prob.b[i] - ax[i],
            ConstraintType::Ge => ax[i] - prob.b[i],
            _ => continue,
        };
        let prod = (y[i] * slack).abs();
        let scale = 1.0 + y[i].abs() * (ax[i].abs() + prob.b[i].abs());
        comp = comp.max(prod / scale);
    }
    comp
}

/// seed cc 5f9e728c... (Le, single row, 2 vars).
#[test]
fn repro_seed_5f9e_le_single_row() {
    let a = CscMatrix::from_triplets(&[0], &[1], &[334.0485457230745], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, -803.3458161579554],
        a,
        vec![0.1],
        vec![ConstraintType::Le],
        vec![(0.0, 1.5), (0.0, 1.5)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let res = solve_lp_with(&lp, &opts);
    assert_eq!(res.status, SolveStatus::Optimal);
    let x = res.solution.clone();
    let y = res.dual_solution.clone();
    let rc = res.reduced_costs.clone();
    let ax = lp.a.mat_vec_mul(&x).unwrap();
    let drift = comp_drift(&lp, &res);
    eprintln!(
        "[5f9e Le] x={:?} y={:?} rc={:?} Ax={:?} b={:?} slack={:?} drift={:.3e}",
        x, y, rc, ax, lp.b, res.slack, drift
    );
}

/// seed cc f95be4... (Ge, single row, 2 vars, lb x[0]=-1.5).
#[test]
fn repro_seed_f95b_ge_single_row() {
    let a = CscMatrix::from_triplets(
        &[0, 0],
        &[0, 1],
        &[-82.43740322950417, -973.9684889867076],
        1,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, -800.3292201645222],
        a,
        vec![-0.1],
        vec![ConstraintType::Ge],
        vec![(-1.5, 1.5), (0.0, 1.5)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let res = solve_lp_with(&lp, &opts);
    assert_eq!(res.status, SolveStatus::Optimal);
    let drift = comp_drift(&lp, &res);
    let ax = lp.a.mat_vec_mul(&res.solution).unwrap();
    eprintln!(
        "[f95b Ge] x={:?} y={:?} rc={:?} Ax={:?} b={:?} slack={:?} drift={:.3e}",
        res.solution, res.dual_solution, res.reduced_costs, ax, lp.b, res.slack, drift,
    );
}

/// seed cc a938... (mixed Ge/Le, single var basis) -- presolve OFF for path split.
#[test]
fn repro_seed_a938_ge_le_mixed_no_presolve() {
    let a = CscMatrix::from_triplets(
        &[0, 1],
        &[1, 1],
        &[4.494611664553469, -1.9339029301709725],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, 3.2567336474320614],
        a,
        vec![-0.1, 0.1],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 1.5), (f64::NEG_INFINITY, f64::INFINITY)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    opts.presolve = false;
    let res = solve_lp_with(&lp, &opts);
    assert_eq!(res.status, SolveStatus::Optimal);
    let drift = comp_drift(&lp, &res);
    let ax = lp.a.mat_vec_mul(&res.solution).unwrap();
    eprintln!(
        "[a938 GeLe NO PRESOLVE] x={:?} y={:?} rc={:?} Ax={:?} b={:?} slack={:?} drift={:.3e}",
        res.solution, res.dual_solution, res.reduced_costs, ax, lp.b, res.slack, drift,
    );
}

/// seed cc a938... (mixed Ge/Le, single var basis).
#[test]
fn repro_seed_a938_ge_le_mixed() {
    let a = CscMatrix::from_triplets(
        &[0, 1],
        &[1, 1],
        &[4.494611664553469, -1.9339029301709725],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![0.0, 3.2567336474320614],
        a,
        vec![-0.1, 0.1],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 1.5), (f64::NEG_INFINITY, f64::INFINITY)],
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let res = solve_lp_with(&lp, &opts);
    assert_eq!(res.status, SolveStatus::Optimal);
    let drift = comp_drift(&lp, &res);
    let ax = lp.a.mat_vec_mul(&res.solution).unwrap();
    eprintln!(
        "[a938 GeLe] x={:?} y={:?} rc={:?} Ax={:?} b={:?} slack={:?} drift={:.3e}",
        res.solution, res.dual_solution, res.reduced_costs, ax, lp.b, res.slack, drift,
    );
}
