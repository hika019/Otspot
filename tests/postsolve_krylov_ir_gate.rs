//! Integration sentinel for the postsolve saddle-point Krylov IR gate.
//!
//! The gate skips `refine_krylov_and_projection` (a full augmented-KKT factorize)
//! when the IPM solution already meets the user tolerance, observed via
//! `SolverResult.stats.postsolve_krylov_ir_skipped`.
//!
//! Load-bearing: removing the `!kkt_already_pass` gate (always refining) flips
//! `skipped` to `false` for the converged cases below, failing this test.
//! (The complementary direction — `kkt_already_passes == false` ⇒ IR runs — is
//! locked by the unit test on `kkt_already_passes` in `post_processing.rs`, and
//! by the full LP/QP regression benches showing zero status change.)

use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::sparse::CscMatrix;
use otspot::QpProblem;

/// min 0.5·xᵀ(diag d)x + cᵀx  s.t. Σx = rhs, x ≥ 0 (well-conditioned convex QP).
fn convex_qp_eq_sum(n: usize, diag: f64, c_val: f64, rhs: f64) -> QpProblem {
    let idx: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&idx, &idx, &vec![diag; n], n, n).unwrap();
    let a = CscMatrix::from_triplets(&vec![0usize; n], &idx, &vec![1.0; n], 1, n).unwrap();
    QpProblem::new(
        q,
        vec![c_val; n],
        a,
        vec![rhs],
        vec![(0.0_f64, f64::INFINITY); n],
        vec![otspot::problem::ConstraintType::Eq],
    )
    .unwrap()
}

#[test]
fn gate_skips_krylov_ir_when_already_converged() {
    // Table-driven: several well-conditioned convex QPs that converge cleanly to
    // original-space residuals far below eps, so the gate fires.
    let cases = [
        convex_qp_eq_sum(4, 2.0, -1.0, 1.0),
        convex_qp_eq_sum(8, 1.0, 0.0, 3.0),
        convex_qp_eq_sum(3, 5.0, 2.0, 2.0),
    ];
    for (i, prob) in cases.iter().enumerate() {
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;
        let res = otspot::qp::solve_qp_with(prob, &opts);
        assert_eq!(
            res.status,
            SolveStatus::Optimal,
            "case {i}: expected Optimal"
        );
        assert!(
            res.stats.postsolve_krylov_ir_skipped,
            "case {i}: a converged solution must skip the Krylov IR (gate must fire). \
             Removing the `!kkt_already_pass` gate flips this to false."
        );
    }
}
