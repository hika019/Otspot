//! Benchmark for sparse LU factorization and solve operations
//!
//! This benchmark measures the performance of:
//! - LU factorization (with Markowitz pivoting)
//! - FTRAN (forward/backward substitution: solve B*x = rhs)
//! - BTRAN (transposed solve: solve B^T*x = rhs)
//!
//! Test matrices: Random sparse matrices (5% density) at sizes 50x50, 100x100, 200x200

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use solver::basis::lu::{solve_btran, solve_ftran, LuFactorization};
use solver::sparse::CscMatrix;

/// Generate a random sparse matrix with given density (0.0 to 1.0)
/// The matrix is diagonally dominant to ensure non-singularity
fn generate_sparse_matrix(n: usize, density: f64) -> CscMatrix {
    use std::collections::HashSet;

    let target_nnz = ((n * n) as f64 * density) as usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    // Strong diagonal entries for non-singularity
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(10.0 + (i as f64) * 0.1);
    }

    // Random off-diagonal entries
    let mut rng_state = 12345u64; // Simple LCG for reproducibility
    let mut added = HashSet::new();

    while rows.len() < target_nnz {
        // Linear Congruential Generator
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (rng_state % (n as u64)) as usize;
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let c = (rng_state % (n as u64)) as usize;

        if r == c || added.contains(&(r, c)) {
            continue;
        }
        added.insert((r, c));

        rows.push(r);
        cols.push(c);
        // Generate value in range [-2.0, 2.0]
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let val = ((rng_state % 10000) as f64) / 10000.0 * 4.0 - 2.0;
        vals.push(val);
    }

    CscMatrix::from_triplets(&rows, &cols, &vals, n, n)
        .expect("Matrix construction should not fail")
}

/// Benchmark LU factorization for a given matrix size
fn bench_lu_factorize(c: &mut Criterion, n: usize) {
    let matrix = generate_sparse_matrix(n, 0.05);
    let basis: Vec<usize> = (0..n).collect();

    c.bench_function(&format!("LU factorize {}x{}", n, n), |b| {
        b.iter(|| {
            let lu = LuFactorization::factorize(black_box(&matrix), black_box(&basis));
            black_box(lu)
        })
    });
}

/// Benchmark FTRAN (solve B*x = rhs) for a given matrix size
fn bench_ftran(c: &mut Criterion, n: usize) {
    let matrix = generate_sparse_matrix(n, 0.05);
    let basis: Vec<usize> = (0..n).collect();
    let lu = LuFactorization::factorize(&matrix, &basis).expect("Factorization should succeed");

    c.bench_function(&format!("FTRAN {}x{}", n, n), |b| {
        b.iter(|| {
            let mut rhs: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
            solve_ftran(black_box(&lu), black_box(&mut rhs));
            black_box(rhs)
        })
    });
}

/// Benchmark BTRAN (solve B^T*x = rhs) for a given matrix size
fn bench_btran(c: &mut Criterion, n: usize) {
    let matrix = generate_sparse_matrix(n, 0.05);
    let basis: Vec<usize> = (0..n).collect();
    let lu = LuFactorization::factorize(&matrix, &basis).expect("Factorization should succeed");

    c.bench_function(&format!("BTRAN {}x{}", n, n), |b| {
        b.iter(|| {
            let mut rhs: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
            solve_btran(black_box(&lu), black_box(&mut rhs));
            black_box(rhs)
        })
    });
}

fn benchmark_lu_operations(c: &mut Criterion) {
    // Benchmark 50x50
    bench_lu_factorize(c, 50);
    bench_ftran(c, 50);
    bench_btran(c, 50);

    // Benchmark 100x100
    bench_lu_factorize(c, 100);
    bench_ftran(c, 100);
    bench_btran(c, 100);

    // Benchmark 200x200
    bench_lu_factorize(c, 200);
    bench_ftran(c, 200);
    bench_btran(c, 200);
}

criterion_group!(benches, benchmark_lu_operations);
criterion_main!(benches);
