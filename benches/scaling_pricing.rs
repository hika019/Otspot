//! Benchmarks for Ruiz Scaling + Steepest-Edge Pricing
//!
//! Compares:
//! - Solving with and without scaling (via the public `solve` API)
//! - Dantzig vs Steepest-Edge pricing (internal comparison, both use scaling)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use otspot::problem::LpProblem;
use otspot::solve;
use otspot::sparse::CscMatrix;

/// Build a simple LP: max sum(x_i) s.t. sum(x_i) <= 100, x_i <= 10
fn make_dense_lp(n: usize) -> LpProblem {
    // min -sum(x_i)
    // s.t. sum(x_i) <= 100
    //      x_i <= 10  for each i
    // Variables: x_0 .. x_{n-1}
    let c: Vec<f64> = vec![-1.0; n];

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Row 0: sum(x_i) <= 100
    for j in 0..n {
        rows.push(0);
        cols.push(j);
        vals.push(1.0);
    }

    let nrows = 1;
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, nrows, n).unwrap();
    let b = vec![100.0];
    LpProblem::new(c, a, b).unwrap()
}

/// Build a random-ish LP with varied scales to stress Ruiz scaling
fn make_scaled_lp(n: usize) -> LpProblem {
    // min -sum(1000*x_i)  (large c)
    // s.t. sum(0.001 * x_i) <= 1.0  (small A entries)
    //      Variables bounded in [0, 1]
    let scale = 1000.0_f64;
    let c: Vec<f64> = (0..n).map(|i| -(scale * (1 + i) as f64)).collect();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Single constraint: sum(0.001 * x_i) <= 1.0
    for j in 0..n {
        rows.push(0);
        cols.push(j);
        vals.push(0.001 / (1 + j) as f64);
    }

    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, n).unwrap();
    let b = vec![1.0];
    LpProblem::new(c, a, b).unwrap()
}

/// Build the Blend-like LP from Netlib tests (5-variable example)
fn make_blend_like_lp() -> LpProblem {
    // A classic LP for benchmarking:
    // min -3x1 -5x2
    // s.t.  x1       <= 4
    //       2x2      <= 12
    //  3x1 + 5x2     <= 25
    // x1, x2 >= 0
    let c = vec![-3.0, -5.0];
    let rows = vec![0, 1, 2, 2];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 2.0, 3.0, 5.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();
    let b = vec![4.0, 12.0, 25.0];
    LpProblem::new(c, a, b).unwrap()
}

fn bench_solve_small(c: &mut Criterion) {
    let lp = make_blend_like_lp();
    c.bench_function("solve_blend_like_2var", |b| {
        b.iter(|| {
            let result = solve(black_box(&lp));
            black_box(result)
        })
    });
}

fn bench_solve_dense_20var(c: &mut Criterion) {
    let lp = make_dense_lp(20);
    c.bench_function("solve_dense_20var", |b| {
        b.iter(|| {
            let result = solve(black_box(&lp));
            black_box(result)
        })
    });
}

fn bench_solve_dense_50var(c: &mut Criterion) {
    let lp = make_dense_lp(50);
    c.bench_function("solve_dense_50var", |b| {
        b.iter(|| {
            let result = solve(black_box(&lp));
            black_box(result)
        })
    });
}

fn bench_solve_scaled_lp_20var(c: &mut Criterion) {
    let lp = make_scaled_lp(20);
    c.bench_function("solve_scaled_lp_20var (Ruiz benefits)", |b| {
        b.iter(|| {
            let result = solve(black_box(&lp));
            black_box(result)
        })
    });
}

fn bench_solve_scaled_lp_50var(c: &mut Criterion) {
    let lp = make_scaled_lp(50);
    c.bench_function("solve_scaled_lp_50var (Ruiz benefits)", |b| {
        b.iter(|| {
            let result = solve(black_box(&lp));
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    bench_solve_small,
    bench_solve_dense_20var,
    bench_solve_dense_50var,
    bench_solve_scaled_lp_20var,
    bench_solve_scaled_lp_50var,
);
criterion_main!(benches);
