//! Hand-built deterministic KKT shape sentinel.
//!
//! The reviewer reported a transient FAIL of
//! `prop_nonconvex_qp_kkt_invariants_constrained` shaped as
//! "3-var nonconvex Q, A row 1 only nonzero, Le×3 with symmetric bounds".
//! Extensive proptest replay (>2000 random nonconvex cases across both main
//! and the refactored branch) did not reproduce a deterministic FAIL, so we
//! instead pin a deterministic sweep across that exact geometry as a
//! permanent shape-targeted sentinel (proptest randomness can mask shape
//! coverage holes; hand-built sweeps cannot drift seed-by-seed).
//!
//! The sweep multiplies indefinite Q signatures × L offdiagonal × c sign ×
//! single nonzero A row × constraint RHS × bound magnitude, sized so the
//! whole test finishes well under the 3-minute budget.
//!
//! Sentinel proof: replacing `compute_qp_kkt_max` with a no-op (returning 0)
//! does not by itself force this test to FAIL — the assertion only fires if
//! a real KKT violation slips past the global solver — but combined with
//! `sentinel_qp_perturbed_solution_fails_kkt` in `diag_kkt_proptest.rs` the
//! no-op path is covered. This file's role is *shape* coverage, not helper
//! no-op coverage.

use solver::bench_utils::compute_qp_kkt_max;
use solver::options::{GlobalOptimizationConfig, SolverOptions};
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::{solve_qp_global, QpProblem};
use solver::sparse::CscMatrix;

const N: usize = 3;
const M: usize = 3;
const KKT_THRESHOLD_LOCAL: f64 = 1e-3;
const KKT_THRESHOLD_GLOBAL: f64 = 1.5;
const QP_TIMEOUT_SECS: f64 = 8.0;

fn dense_csc(dense: &[f64], rows: usize, cols: usize) -> CscMatrix {
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for j in 0..cols {
        for i in 0..rows {
            let x = dense[i * cols + j];
            if x.abs() > 1e-14 {
                r.push(i);
                c.push(j);
                v.push(x);
            }
        }
    }
    CscMatrix::from_triplets(&r, &c, &v, rows, cols).unwrap()
}

fn build_qp(
    d_sign: &[f64; N],
    l_off: f64,
    c: &[f64; N],
    a_row1: &[f64; N],
    b: &[f64; M],
    cts: &[ConstraintType; M],
    bnd: f64,
) -> QpProblem {
    let mut l = [0.0_f64; N * N];
    for i in 0..N {
        l[i * N + i] = 1.0;
        for j in 0..i {
            l[i * N + j] = l_off;
        }
    }
    let mut q = [0.0_f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for k in 0..=i.min(j) {
                s += l[i * N + k] * l[j * N + k] * d_sign[k];
            }
            q[i * N + j] = s;
        }
    }
    let q_csc = dense_csc(&q, N, N);
    let mut a_dense = vec![0.0_f64; M * N];
    for j in 0..N {
        a_dense[j] = a_row1[j];
    }
    let a_csc = dense_csc(&a_dense, M, N);
    let bounds = vec![(-bnd, bnd); N];
    QpProblem::new(q_csc, c.to_vec(), a_csc, b.to_vec(), bounds, cts.to_vec())
        .expect("QpProblem")
}

/// 3-var nonconvex Q × A row-1-only-nonzero × Le×3 × symmetric bound. Sweep
/// indefinite signature × offdiag × c sign × A row × RHS × bound for a
/// deterministic shape-targeted sentinel. Threshold matches
/// `diag_kkt_proptest::EPS_KKT_NONCONVEX_LOCAL` / `_GLOBAL`.
#[test]
fn repro_nonconvex_constrained_shape_sweep() {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(QP_TIMEOUT_SECS);
    let cfg = GlobalOptimizationConfig::default();
    let cts = [ConstraintType::Le, ConstraintType::Le, ConstraintType::Le];

    let signs: Vec<[f64; N]> = vec![
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ];
    let l_offs = [0.0_f64, 0.4];
    let c_vecs: Vec<[f64; N]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [-1.5, 0.8, -0.3],
    ];
    let a_rows: Vec<[f64; N]> = vec![
        [1.0, 0.0, 0.0],
        [0.5, -0.3, 0.8],
        [-1.0, 0.6, 0.0],
    ];
    let b_vecs: Vec<[f64; M]> = vec![
        [0.5, 0.5, 0.5],
        [2.0, 0.3, 0.1],
        [-0.5, 0.5, 1.0],
    ];
    let bnds = [0.5_f64, 1.0, 2.0];

    let mut fails = Vec::new();
    let mut total = 0_usize;
    for d in &signs {
        for &l in &l_offs {
            for c in &c_vecs {
                for a in &a_rows {
                    for b in &b_vecs {
                        for &bnd in &bnds {
                            total += 1;
                            let qp = build_qp(d, l, c, a, b, &cts, bnd);
                            let res = solve_qp_global(&qp, &opts, &cfg);
                            let threshold = match res.status {
                                SolveStatus::LocallyOptimal => KKT_THRESHOLD_LOCAL,
                                SolveStatus::Optimal => KKT_THRESHOLD_GLOBAL,
                                _ => continue,
                            };
                            let kkt = compute_qp_kkt_max(
                                &qp,
                                &res.solution,
                                &res.dual_solution,
                                &res.bound_duals,
                            );
                            if !(kkt.is_finite() && kkt < threshold) {
                                fails.push((*d, l, *c, *a, *b, bnd, res.status, kkt));
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "repro sweep: total={} fails={}",
        total,
        fails.len()
    );
    for f in fails.iter().take(8) {
        eprintln!(
            "FAIL d={:?} l_off={} c={:?} a={:?} b={:?} bnd={} status={:?} kkt={:.3e}",
            f.0, f.1, f.2, f.3, f.4, f.5, f.6, f.7
        );
    }
    assert!(
        fails.is_empty(),
        "{} shapes violated KKT threshold (see stderr)",
        fails.len()
    );
}
