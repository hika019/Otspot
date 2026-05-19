use super::*;
use crate::options::{SimplexMethod, SolverOptions};
use crate::problem::{LpProblem, SolveStatus};
use crate::sparse::CscMatrix;
use crate::test_kkt::assert_solver_invariants_lp;

fn make_lp(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
) -> LpProblem {
    let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
    LpProblem::new(c, a, b).unwrap()
}

/// LP1 (obj=-7) → reuse basis on LP2 with RHS=[5,3,3] (obj=-8).
#[test]
fn test_dual_advanced_warm_start_rhs_change() {
    let lp1 = make_lp(
        vec![-1.0, -2.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );

    // LP1 を default solver で解いて warm_start_basis を取得
    let result1 = solve_with(&lp1, &SolverOptions::default());
    assert_eq!(result1.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result1, &lp1);
    assert!(
        result1.warm_start_basis.is_some(),
        "LP1 は warm_start_basis を返すべき"
    );

    // LP2: RHS のみ変更 b=[5, 3, 3]
    let lp2 = make_lp(
        vec![-1.0, -2.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![5.0, 3.0, 3.0],
    );

    // cold-start で正解を確認
    let result2_cold = solve_with(&lp2, &SolverOptions::default());
    assert_eq!(result2_cold.status, SolveStatus::Optimal);
    assert_solver_invariants_lp(&result2_cold, &lp2);

    // DualAdvanced warm-start で解く → warm-start 経路を通す
    let opts_warm = SolverOptions {
        warm_start: result1.warm_start_basis.clone(),
        simplex_method: SimplexMethod::DualAdvanced,
        ..SolverOptions::default()
    };
    let result2_warm = solve_with(&lp2, &opts_warm);

    assert_eq!(
        result2_warm.status,
        SolveStatus::Optimal,
        "DualAdvanced warm-start は Optimal を返すべき"
    );
    assert_solver_invariants_lp(&result2_warm, &lp2);
    assert!(
        (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
        "DualAdvanced warm-start obj={}, cold-start obj={}",
        result2_warm.objective,
        result2_cold.objective
    );
}

/// LP1 (obj=-4) → reuse basis on LP2 with RHS=[6,4,4] (obj=-8).
#[test]
fn test_dual_advanced_warm_start_larger_rhs() {
    let lp1 = make_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![4.0, 3.0, 3.0],
    );

    let result1 = solve_with(&lp1, &SolverOptions::default());
    assert_eq!(result1.status, SolveStatus::Optimal);
    assert!(
        result1.warm_start_basis.is_some(),
        "LP1 は warm_start_basis を返すべき"
    );

    // LP2: RHS 拡大 b=[6, 4, 4] → 最適解 x1+x2=8, obj=-8
    let lp2 = make_lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![6.0, 4.0, 4.0],
    );

    // cold-start で正解を確認
    let result2_cold = solve_with(&lp2, &SolverOptions::default());
    assert_eq!(result2_cold.status, SolveStatus::Optimal);

    // DualAdvanced warm-start で解く
    let opts_warm = SolverOptions {
        warm_start: result1.warm_start_basis.clone(),
        simplex_method: SimplexMethod::DualAdvanced,
        ..SolverOptions::default()
    };
    let result2_warm = solve_with(&lp2, &opts_warm);

    assert_eq!(
        result2_warm.status,
        SolveStatus::Optimal,
        "DualAdvanced warm-start (larger RHS) は Optimal を返すべき"
    );
    assert!(
        (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
        "DualAdvanced warm-start obj={}, cold-start obj={}",
        result2_warm.objective,
        result2_cold.objective
    );
}

#[test]
fn test_scsd6_equality_constraints() {
    // scsd6: network flow LP with 147 all-equality constraints, 1350 vars.
    // Reported as NumericalError in 0.024s.
    let path = std::path::Path::new("data/lp_problems/scsd6.QPS");
    if !path.exists() {
        return;
    }
    let content = std::fs::read_to_string(path).unwrap();
    let lp = crate::io::mps::parse_mps(&content).unwrap();

    // Test each method independently to isolate the bug
    let methods = [
        ("Auto", SimplexMethod::Auto),
        ("Primal", SimplexMethod::Primal),
        ("Dual", SimplexMethod::Dual),
    ];
    let results: Vec<_> = methods.iter().map(|(name, method)| {
        let mut opts = SolverOptions::default();
        opts.simplex_method = *method;
        opts.presolve = false;
        let result = solve_with(&lp, &opts);
        eprintln!("scsd6 {} -> {:?} obj={:.3e}", name, result.status, result.objective);
        (*name, result.status)
    }).collect();

    for (name, status) in &results {
        assert_ne!(
            *status, SolveStatus::NumericalError,
            "scsd6 {} returned NumericalError",
            name
        );
    }
}
