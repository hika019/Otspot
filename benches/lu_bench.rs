//! Benchmark for LP solver performance
//!
//! This benchmark measures the performance of `solve()` on random LP problems
//! of varying sizes. The internal LU factorization, FTRAN, and BTRAN operations
//! are exercised indirectly through the public API.
//!
//! Note: Direct benchmarking of `LuFactorization` is not possible because it
//! is an internal implementation detail (`pub(crate)`). This benchmark uses
//! the public `solve()` API to capture end-to-end performance including LU
//! factorization and simplex iterations.
//!
//! Test problems: Random diagonally dominant LP problems at sizes
//! n=20, n=50, n=100 (variables/constraints)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use solver::problem::{ConstraintType, LpProblem};
use solver::solve;
use solver::sparse::CscMatrix;

/// Generate a random diagonally dominant LP problem with n variables and m constraints.
///
/// Problem: minimize c^T x, subject to A x <= b, x >= 0
/// The constraint matrix is sparse (10% density) with diagonal dominance
/// to ensure feasibility and bounded optimal solution.
fn generate_lp(n: usize, m: usize) -> LpProblem {
    // Simple LCG for reproducibility
    let mut rng = 12345u64;
    let next = |rng: &mut u64| -> f64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*rng >> 33) as f64) / (u32::MAX as f64)
    };

    // Objective: minimize c^T x, c_i in [0.5, 1.5]
    let c: Vec<f64> = (0..n).map(|_| 0.5 + next(&mut rng)).collect();

    // Constraint matrix A (m x n), sparse with ~10% density
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..m {
        for j in 0..n {
            if next(&mut rng) < 0.1 || i == j % m {
                // Diagonal-ish entry for feasibility
                rows.push(i);
                cols.push(j);
                let v = if i == j % m { 1.0 } else { next(&mut rng) * 0.5 };
                vals.push(v);
            }
        }
    }

    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n)
        .expect("Matrix construction should not fail");

    // RHS b_i in [n/2, n] to ensure feasibility (x=0 is feasible since b > 0)
    let b: Vec<f64> = (0..m).map(|i| (n / 2 + i % (n / 2 + 1)) as f64).collect();

    let constraint_types = vec![ConstraintType::Le; m];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];

    LpProblem::new_general(c, a, b, constraint_types, bounds, Some("bench".to_string()))
        .expect("LP construction should not fail")
}

fn bench_solve(c: &mut Criterion, n: usize, m: usize) {
    let problem = generate_lp(n, m);

    c.bench_function(&format!("solve LP {}vars {}constraints", n, m), |b| {
        b.iter(|| {
            let result = solve(black_box(&problem));
            black_box(result)
        })
    });
}

fn benchmark_solve_operations(c: &mut Criterion) {
    // Small problem
    bench_solve(c, 20, 10);
    // Medium problem
    bench_solve(c, 50, 25);
    // Larger problem
    bench_solve(c, 100, 50);
}

criterion_group!(benches, benchmark_solve_operations);
criterion_main!(benches);
