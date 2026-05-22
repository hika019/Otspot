//! Benchmarks for the QP (Quadratic Programming) solver
//!
//! Compares:
//! - QP基本問題（2変数）
//! - QP中規模問題（8変数、等式制約+非負）
//! - LP問題をQP(Q=0)として解いたときの速度

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use otspot::problem::ConstraintType;
use otspot::qp::{solve_qp, QpProblem};
use otspot::sparse::CscMatrix;

/// QP基本問題（2変数）
/// min x^2+y^2  s.t. x+y >= 1
/// Q = [[2,0],[0,2]], c=[0,0], A=[[-1,-1]], b=[-1]
fn make_qp_2vars() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let constraint_types = vec![ConstraintType::Le; 1];
    QpProblem::new(q, c, a, b, bounds, constraint_types).unwrap()
}

/// QP中規模問題（8変数）
/// min 1/2 * x^T Q x, Q=2I  s.t. sum(xi)=1, xi>=0
/// 解析解: xi*=1/8, obj=1/8
fn make_qp_8vars() -> QpProblem {
    let n = 8;
    // Q = 2I
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![2.0; n];
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let c = vec![0.0; n];

    // A行列: (n+2行, n列)
    //   行0: [1,...,1] <= 1  (等式上界)
    //   行1: [-1,...,-1] <= -1  (等式下界)
    //   行2..n+1: -ei <= 0  (xi>=0)
    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();

    // 等式制約上界
    for j in 0..n {
        a_rows.push(0);
        a_cols.push(j);
        a_vals.push(1.0);
    }
    // 等式制約下界
    for j in 0..n {
        a_rows.push(1);
        a_cols.push(j);
        a_vals.push(-1.0);
    }
    // 非負制約
    for j in 0..n {
        a_rows.push(2 + j);
        a_cols.push(j);
        a_vals.push(-1.0);
    }

    let nrows = n + 2;
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, nrows, n).unwrap();
    let mut b = vec![1.0, -1.0];
    b.extend(vec![0.0; n]);
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let constraint_types = vec![ConstraintType::Le; nrows];
    QpProblem::new(q, c, a, b, bounds, constraint_types).unwrap()
}

/// LP問題をQP(Q=0)として定式化
/// min 2x+y  s.t. x+y>=1, x>=0, y>=0
fn make_qp_as_lp() -> QpProblem {
    let n = 2;
    let q = CscMatrix::new(n, n); // Q=0
    let c = vec![2.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[-1.0, -1.0, -1.0, -1.0],
        3, 2,
    ).unwrap();
    let b = vec![-1.0, 0.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let constraint_types = vec![ConstraintType::Le; 3];
    QpProblem::new(q, c, a, b, bounds, constraint_types).unwrap()
}

fn bench_qp_2vars(c: &mut Criterion) {
    let problem = make_qp_2vars();
    c.bench_function("qp 2vars basic", |b| {
        b.iter(|| {
            let result = solve_qp(black_box(&problem));
            black_box(result)
        })
    });
}

fn bench_qp_8vars(c: &mut Criterion) {
    let problem = make_qp_8vars();
    c.bench_function("qp 8vars medium", |b| {
        b.iter(|| {
            let result = solve_qp(black_box(&problem));
            black_box(result)
        })
    });
}

fn bench_qp_as_lp(c: &mut Criterion) {
    let problem = make_qp_as_lp();
    c.bench_function("qp Q=0 as lp", |b| {
        b.iter(|| {
            let result = solve_qp(black_box(&problem));
            black_box(result)
        })
    });
}

fn benchmark_qp_suite(c: &mut Criterion) {
    bench_qp_2vars(c);
    bench_qp_8vars(c);
    bench_qp_as_lp(c);
}

criterion_group!(benches, benchmark_qp_suite);
criterion_main!(benches);
