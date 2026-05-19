//! Property-based tests for solver Optimal-exit invariants.
//!
//! Any LP that the solver reports as Optimal must satisfy:
//!   |Ax - b|_∞ / (1 + ||b||_∞) < sentinel_tol  (primal feasibility)
//!   lb ≤ x ≤ ub                                  (bound feasibility)
//!
//! These tests generate random problems across simplex paths (Primal / Dual /
//! DualAdvanced) and verify that no false-Optimal is returned. Removing the
//! production sentinel in entry.rs would cause this test to catch regressions.

use proptest::prelude::*;
use solver::options::{SimplexMethod, SolverOptions};
use solver::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use solver::solve_lp_with;
use solver::sparse::CscMatrix;

/// Normalized primal violation: max violation / (1 + ||b||_inf).
fn pfeas_normalized(a: &CscMatrix, b: &[f64], cts: &[ConstraintType], x: &[f64]) -> f64 {
    let m = b.len();
    if m == 0 || x.is_empty() {
        return 0.0;
    }
    let mut ax = vec![0.0_f64; m];
    for j in 0..x.len().min(a.ncols) {
        if let Ok((rows, vals)) = a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m {
                    ax[row] += vals[k] * x[j];
                }
            }
        }
    }
    let b_inf = b.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let viol: f64 = (0..m)
        .map(|i| match cts[i] {
            ConstraintType::Eq => (ax[i] - b[i]).abs(),
            ConstraintType::Le => (ax[i] - b[i]).max(0.0),
            ConstraintType::Ge => (b[i] - ax[i]).max(0.0),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    viol / (1.0 + b_inf)
}

/// Bound violation: max(lb - x, x - ub, 0) over all variables.
fn bound_viol(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    bounds
        .iter()
        .zip(x.iter())
        .map(|(&(lb, ub), &xi)| {
            let lo = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let hi = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lo.max(hi)
        })
        .fold(0.0_f64, f64::max)
}

/// Check Optimal invariants; panics with proptest assertion on violation.
fn assert_invariants_if_optimal(result: &SolverResult, problem: &LpProblem, label: &str) {
    if result.status != SolveStatus::Optimal {
        return;
    }
    if result.solution.is_empty() {
        return;
    }
    let pf = pfeas_normalized(
        &problem.a,
        &problem.b,
        &problem.constraint_types,
        &result.solution,
    );
    assert!(
        pf < 1e-4,
        "[{}] false-Optimal: pfeas_norm={:.3e} > 1e-4",
        label,
        pf
    );
    let bv = bound_viol(&problem.bounds, &result.solution);
    assert!(
        bv < 1e-4,
        "[{}] false-Optimal: bound_viol={:.3e} > 1e-4",
        label,
        bv
    );
}

fn make_diagonal_csc(diag: &[f64], nrows: usize, ncols: usize) -> CscMatrix {
    let k = nrows.min(ncols).min(diag.len());
    if k == 0 {
        return CscMatrix::new(nrows, ncols);
    }
    let rows: Vec<usize> = (0..k).collect();
    let cols: Vec<usize> = (0..k).collect();
    CscMatrix::from_triplets(&rows, &cols, &diag[..k].to_vec(), nrows, ncols).unwrap()
}

fn make_opts(method: SimplexMethod) -> SolverOptions {
    let mut o = SolverOptions::default();
    o.simplex_method = method;
    o.timeout_secs = Some(5.0);
    o
}

proptest! {
    /// DualAdvanced path — includes Eq constraints, exercises Big-M Phase I.
    #[test]
    fn prop_dual_advanced_optimal_invariants(
        c in prop::collection::vec(-5.0f64..5.0f64, 2usize..=6usize),
        diag in prop::collection::vec(0.1f64..3.0f64, 1usize..=5usize),
        b in prop::collection::vec(0.1f64..8.0f64, 1usize..=5usize),
        use_eq in prop::bool::ANY,
    ) {
        let n = c.len();
        let m = diag.len().min(b.len());
        if m == 0 { return Ok(()); }
        let b_m = b[..m].to_vec();
        let a = make_diagonal_csc(&diag, m, n);
        let ct: Vec<ConstraintType> = (0..m)
            .map(|i| if use_eq && i == 0 { ConstraintType::Eq } else { ConstraintType::Le })
            .collect();
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let Ok(lp) = LpProblem::new_general(c, a, b_m, ct, bounds, None) else { return Ok(()); };

        let result = solve_lp_with(&lp, &make_opts(SimplexMethod::DualAdvanced));
        assert_invariants_if_optimal(&result, &lp, "DualAdvanced");
    }

    /// Primal simplex path — two-phase primal with artificials.
    #[test]
    fn prop_primal_simplex_optimal_invariants(
        c in prop::collection::vec(-5.0f64..5.0f64, 2usize..=6usize),
        diag in prop::collection::vec(0.1f64..3.0f64, 1usize..=5usize),
        b in prop::collection::vec(0.1f64..8.0f64, 1usize..=5usize),
    ) {
        let n = c.len();
        let m = diag.len().min(b.len());
        if m == 0 { return Ok(()); }
        let b_m = b[..m].to_vec();
        let a = make_diagonal_csc(&diag, m, n);
        let ct = vec![ConstraintType::Le; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let Ok(lp) = LpProblem::new_general(c, a, b_m, ct, bounds, None) else { return Ok(()); };

        let result = solve_lp_with(&lp, &make_opts(SimplexMethod::Primal));
        assert_invariants_if_optimal(&result, &lp, "Primal");
    }

    /// Dual simplex — Le-only constraints.
    #[test]
    fn prop_dual_simplex_optimal_invariants(
        c in prop::collection::vec(-5.0f64..5.0f64, 2usize..=6usize),
        diag in prop::collection::vec(0.1f64..3.0f64, 1usize..=5usize),
        b in prop::collection::vec(0.1f64..8.0f64, 1usize..=5usize),
    ) {
        let n = c.len();
        let m = diag.len().min(b.len());
        if m == 0 { return Ok(()); }
        let b_m = b[..m].to_vec();
        let a = make_diagonal_csc(&diag, m, n);
        let ct = vec![ConstraintType::Le; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let Ok(lp) = LpProblem::new_general(c, a, b_m, ct, bounds, None) else { return Ok(()); };

        let result = solve_lp_with(&lp, &make_opts(SimplexMethod::Dual));
        assert_invariants_if_optimal(&result, &lp, "Dual");
    }

    /// Mixed Eq/Ge/Le constraints across all paths.
    #[test]
    fn prop_mixed_constraints_optimal_invariants(
        c in prop::collection::vec(-3.0f64..3.0f64, 3usize..=8usize),
        a_vals in prop::collection::vec(0.2f64..2.0f64, 3usize..=8usize),
        b_vals in prop::collection::vec(0.5f64..5.0f64, 3usize..=6usize),
        ct_bits in prop::collection::vec(0u8..3u8, 3usize..=6usize),
    ) {
        let n = c.len();
        let m = a_vals.len().min(b_vals.len()).min(ct_bits.len());
        if m == 0 { return Ok(()); }
        let b_m = b_vals[..m].to_vec();
        let a = make_diagonal_csc(&a_vals[..m], m, n);
        let ct: Vec<ConstraintType> = ct_bits[..m]
            .iter()
            .map(|&b| match b % 3 {
                0 => ConstraintType::Le,
                1 => ConstraintType::Ge,
                _ => ConstraintType::Eq,
            })
            .collect();
        let bounds = vec![(0.0_f64, 10.0_f64); n];
        let Ok(lp) = LpProblem::new_general(c, a, b_m, ct, bounds, None) else { return Ok(()); };

        let result = solve_lp_with(&lp, &make_opts(SimplexMethod::Auto));
        assert_invariants_if_optimal(&result, &lp, "Auto/Mixed");
    }
}

/// No-op proof sentinel: production sentinel disabled via direct hack detects
/// false-Optimal. This test encodes an artificially corrupt "Optimal" result
/// with |Ax-b| = 1e12 and ensures invariant check catches it.
#[test]
fn sentinel_false_optimal_detected_by_invariant_check() {
    // Build a simple 1-constraint LP: x1 <= 5, minimize x1
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0],
        a.clone(),
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    // Corrupt result: claim Optimal with x = 1e12 (violates x <= 5 massively)
    let corrupt = SolverResult {
        status: SolveStatus::Optimal,
        objective: 1e12,
        solution: vec![1e12],
        dual_solution: vec![0.0],
        reduced_costs: vec![0.0],
        slack: vec![0.0],
        ..Default::default()
    };

    let pf = pfeas_normalized(&lp.a, &lp.b, &lp.constraint_types, &corrupt.solution);
    assert!(
        pf > 1e-3,
        "Corrupt solution should have large pfeas_norm: got {:.3e}",
        pf
    );

    // Verify the real solver returns clean Optimal (no false-Optimal)
    let real_result = solve_lp_with(&lp, &SolverOptions::default());
    assert_eq!(real_result.status, SolveStatus::Optimal);
    let real_pf = pfeas_normalized(
        &lp.a,
        &lp.b,
        &lp.constraint_types,
        &real_result.solution,
    );
    assert!(
        real_pf < 1e-6,
        "Real solver should have tiny pfeas_norm: got {:.3e}",
        real_pf
    );
}
