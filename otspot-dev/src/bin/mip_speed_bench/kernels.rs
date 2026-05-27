//! Shared problem generators for the MIP speed bench.
//!
//! Single source of truth for the random draws and the convex-MIQP `Q`
//! construction, consumed both by the runner (`main.rs`) and by the
//! correctness tests (`tests/mip_bench_gen_correctness.rs`, via `#[path]`).
//! Keeping them here prevents the test's view of a problem from drifting away
//! from the one the bench actually solves.

#![allow(dead_code)] // each #[path] consumer uses a subset.

use otspot_core::{
    problem::{ConstraintType, LpProblem},
    CscMatrix, MilpProblem, MiqpProblem, QpProblem,
};

/// Deterministic LCG (Knuth MMIX parameters) so runs need no external data.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// MILP generators
// ---------------------------------------------------------------------------

/// Knapsack-style MILP: min -cᵀx s.t. wᵀx <= W, first `n_int` vars integral.
/// Tight capacity (≈0.5·Σw) forces a non-trivial LP-relaxation gap.
pub fn gen_knapsack_milp(n: usize, int_ratio: f64, seed: u64) -> MilpProblem {
    let mut lcg = Lcg::new(seed);
    let n_int = ((n as f64 * int_ratio).round() as usize).max(1);

    let c: Vec<f64> = (0..n).map(|_| -lcg.next_f64() * 10.0).collect();
    let weights: Vec<f64> = (0..n).map(|_| 1.0 + lcg.next_f64() * 9.0).collect();
    let capacity = weights.iter().sum::<f64>() * 0.5;

    let rows: Vec<usize> = vec![0; n];
    let cols: Vec<usize> = (0..n).collect();
    let a = CscMatrix::from_triplets(&rows, &cols, &weights, 1, n).unwrap();
    let bounds: Vec<(f64, f64)> = (0..n).map(|_| (0.0, 1.0)).collect();
    let lp = LpProblem::new_general(c, a, vec![capacity], vec![ConstraintType::Le], bounds, None)
        .unwrap();
    MilpProblem::new(lp, (0..n_int).collect()).unwrap()
}

/// Recompute the knapsack weights / capacity for a `(n, seed)` instance without
/// rebuilding the problem (the brute-force check needs them). Must mirror the
/// draw order of [`gen_knapsack_milp`] exactly: `n` profits, then `n` weights.
pub fn knapsack_weights_capacity(n: usize, seed: u64) -> (Vec<f64>, f64) {
    let mut lcg = Lcg::new(seed);
    for _ in 0..n {
        lcg.next_f64(); // skip profits
    }
    let weights: Vec<f64> = (0..n).map(|_| 1.0 + lcg.next_f64() * 9.0).collect();
    let capacity = weights.iter().sum::<f64>() * 0.5;
    (weights, capacity)
}

/// Assignment-style MILP: `m ≈ density·n` random ≤ constraints, rhs = 0.6·Σcoeff
/// (tight → forces branching). First `n_int` vars integral.
pub fn gen_assignment_milp(n: usize, int_ratio: f64, density: f64, seed: u64) -> MilpProblem {
    let mut lcg = Lcg::new(seed ^ 0xABCD_EF01);
    let n_int = ((n as f64 * int_ratio).round() as usize).max(1);
    let m = ((n as f64 * density).round() as usize).max(1);

    let c: Vec<f64> = (0..n).map(|_| lcg.next_f64() * 10.0 - 5.0).collect();

    let mut rows = vec![];
    let mut cols = vec![];
    let mut vals = vec![];
    let mut b = vec![];
    for i in 0..m {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            if lcg.next_f64() < density {
                let v = lcg.next_f64() * 2.0 + 0.5;
                rows.push(i);
                cols.push(j);
                vals.push(v);
                row_sum += v;
            }
        }
        b.push((row_sum * 0.6).max(1.0));
    }

    let a = if rows.is_empty() {
        CscMatrix::new(m, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap()
    };
    let ctypes = vec![ConstraintType::Le; m];
    let bounds: Vec<(f64, f64)> = (0..n).map(|_| (0.0, 1.0)).collect();
    let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();
    MilpProblem::new(lp, (0..n_int).collect()).unwrap()
}

// ---------------------------------------------------------------------------
// Convex MIQP: Q = L Lᵀ + ridge·I (PSD by construction)
// ---------------------------------------------------------------------------

/// Ridge added to L Lᵀ to guarantee strict positive-definiteness.
pub const CONVEX_MIQP_RIDGE: f64 = 2.0;
/// Seed mangling so the MIQP stream differs from the MILP generators.
pub const CONVEX_MIQP_SEED_MASK: u64 = 0x1234_5678_9ABC_DEF0;

/// LCG seeded for the convex-MIQP stream.
pub fn convex_miqp_lcg(seed: u64) -> Lcg {
    Lcg::new(seed ^ CONVEX_MIQP_SEED_MASK)
}

/// Draw the dense symmetric `Q = L Lᵀ + ridge·I` and linear term `c`.
///
/// Advances `lcg` by exactly the lower-triangular `L` entries followed by the
/// `c` entries, so a caller may keep drawing from the same `lcg` afterwards
/// (e.g. for constraints) and reproduce the original stream bit-for-bit.
pub fn build_convex_qc(lcg: &mut Lcg, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut l = vec![vec![0.0_f64; n]; n];
    for (i, row) in l.iter_mut().enumerate() {
        row[i] = 1.0;
        for elem in &mut row[..i] {
            *elem = lcg.next_f64() - 0.5;
        }
    }
    let mut q = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let k_max = i.min(j) + 1;
            q[i][j] = (0..k_max).map(|k| l[i][k] * l[j][k]).sum::<f64>();
        }
        q[i][i] += CONVEX_MIQP_RIDGE;
    }
    let c: Vec<f64> = (0..n).map(|_| (lcg.next_f64() - 0.5) * 4.0).collect();
    (q, c)
}

/// Pack a dense symmetric `Q` into CSC, dropping near-zero entries.
pub fn convex_q_to_csc(q_dense: &[Vec<f64>], n: usize) -> CscMatrix {
    let mut rows = vec![];
    let mut cols = vec![];
    let mut vals = vec![];
    for (i, qi) in q_dense.iter().enumerate().take(n) {
        for (j, &v) in qi.iter().enumerate().take(n) {
            if v.abs() > 1e-14 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

/// Build a convex MIQP with `m ≈ density·n` random ≤ constraints. Integer vars
/// (first `n_int`) range `[0, 3]`; continuous vars `[0, 5]`.
pub fn gen_convex_miqp(n: usize, int_ratio: f64, density: f64, seed: u64) -> MiqpProblem {
    let n_int = ((n as f64 * int_ratio).round() as usize).max(1);
    let mut lcg = convex_miqp_lcg(seed);
    let (q_dense, c) = build_convex_qc(&mut lcg, n);
    let q = convex_q_to_csc(&q_dense, n);

    let m = ((n as f64 * density).ceil() as usize).max(1);
    let mut a_rows = vec![];
    let mut a_cols = vec![];
    let mut a_vals = vec![];
    let mut b = vec![];
    for i in 0..m {
        let mut rhs = 0.0_f64;
        for j in 0..n {
            if lcg.next_f64() < density {
                let v = lcg.next_f64() * 2.0 + 0.5;
                a_rows.push(i);
                a_cols.push(j);
                a_vals.push(v);
                rhs += v;
            }
        }
        b.push((rhs * 0.6).max(1.0));
    }
    let a = if a_rows.is_empty() {
        CscMatrix::new(m, n)
    } else {
        CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap()
    };

    let bounds: Vec<(f64, f64)> =
        (0..n).map(|i| if i < n_int { (0.0, 3.0) } else { (0.0, 5.0) }).collect();
    let qp = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    MiqpProblem::new(qp, (0..n_int).collect()).unwrap()
}
