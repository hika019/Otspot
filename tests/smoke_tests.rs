//! LP integration smoke tests — verify solve() returns Optimal on basic LPs.

use otspot::problem::{LpProblem, SolveStatus};
use otspot::solve;
use otspot::sparse::CscMatrix;

fn unit_diagonal_csc(n: usize) -> CscMatrix {
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0; n];
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

/// Scaling module integration: tiny LP with unit scaling must reach Optimal.
#[test]
fn scaling_module_exists() {
    let n = 3;
    let c = vec![1.0, 2.0, 3.0];
    let a = unit_diagonal_csc(n);
    let b = vec![10.0, 10.0, 10.0];
    let prob = LpProblem::new(c, a, b).unwrap();
    let result = solve(&prob);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "scaling smoke: expected Optimal, got {:?}",
        result.status
    );
}

/// Pricing module integration: LP with all-ones objective must reach Optimal.
#[test]
fn pricing_module_exists() {
    let n = 4;
    let c = vec![1.0; n];
    let a = unit_diagonal_csc(n);
    let b = vec![5.0; n];
    let prob = LpProblem::new(c, a, b).unwrap();
    let result = solve(&prob);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "pricing smoke: expected Optimal, got {:?}",
        result.status
    );
}
