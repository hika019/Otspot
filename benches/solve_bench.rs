//! Benchmark for end-to-end simplex solve on Netlib problems
//!
//! This benchmark measures the total time to solve LP problems from MPS files:
//! - Parse MPS file
//! - Initialize simplex solver
//! - Run revised simplex algorithm
//! - Return optimal solution
//!
//! Test problems: afiro, sc50a, sc50b (existing Netlib test problems)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use solver::io::mps::parse_mps_file;
use solver::solve;
use std::path::Path;

/// Benchmark solving afiro.mps (27 constraints, 32 variables)
fn bench_solve_afiro(c: &mut Criterion) {
    let path = Path::new("tests/netlib/afiro.mps");
    let problem = parse_mps_file(path).expect("Failed to parse afiro.mps");

    c.bench_function("solve afiro", |b| {
        b.iter(|| {
            let result = solve(black_box(&problem));
            black_box(result)
        })
    });
}

/// Benchmark solving sc50a.mps (50 constraints, 48 variables)
fn bench_solve_sc50a(c: &mut Criterion) {
    let path = Path::new("tests/netlib/sc50a.mps");
    let problem = parse_mps_file(path).expect("Failed to parse sc50a.mps");

    c.bench_function("solve sc50a", |b| {
        b.iter(|| {
            let result = solve(black_box(&problem));
            black_box(result)
        })
    });
}

/// Benchmark solving sc50b.mps (50 constraints, 48 variables)
fn bench_solve_sc50b(c: &mut Criterion) {
    let path = Path::new("tests/netlib/sc50b.mps");
    let problem = parse_mps_file(path).expect("Failed to parse sc50b.mps");

    c.bench_function("solve sc50b", |b| {
        b.iter(|| {
            let result = solve(black_box(&problem));
            black_box(result)
        })
    });
}

fn benchmark_netlib_problems(c: &mut Criterion) {
    bench_solve_afiro(c);
    bench_solve_sc50a(c);
    bench_solve_sc50b(c);
}

criterion_group!(benches, benchmark_netlib_problems);
criterion_main!(benches);
