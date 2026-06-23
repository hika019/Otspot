//! Property-based tests for solver Optimal-exit invariants.
//!
//! Any LP that the solver reports as Optimal must satisfy:
//!   |Ax - b|_∞ / (1 + ||b||_∞) < sentinel_tol  (primal feasibility)
//!   lb ≤ x ≤ ub                                  (bound feasibility)
//!
//! These tests generate random problems across simplex paths (Primal / Dual /
//! DualAdvanced) and verify that no false-Optimal is returned. Removing the
//! production sentinel in entry.rs would cause this test to catch regressions.

use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use otspot::solve_lp_with;
use otspot::sparse::CscMatrix;
use proptest::prelude::*;

/// Normalized primal violation: max violation / (1 + ||b||_inf).
fn pfeas_normalized(a: &CscMatrix, b: &[f64], cts: &[ConstraintType], x: &[f64]) -> f64 {
    let m = b.len();
    if m == 0 || x.is_empty() {
        return 0.0;
    }
    let mut ax = vec![0.0_f64; m];
    for (j, &xj) in x.iter().enumerate().take(a.ncols()) {
        if let Ok((rows, vals)) = a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m {
                    ax[row] += vals[k] * xj;
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
            let lo = if lb.is_finite() {
                (lb - xi).max(0.0)
            } else {
                0.0
            };
            let hi = if ub.is_finite() {
                (xi - ub).max(0.0)
            } else {
                0.0
            };
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
    CscMatrix::from_triplets(&rows, &cols, &diag[..k], nrows, ncols).unwrap()
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

/// Load-bearing proof for the production `guard_lp_optimal` path.
///
/// Two complementary assertions form the sentinel:
/// 1. A result with corrupt primal data (x=1e12 violates x≤5) is demoted to
///    SuboptimalSolution — proves the guard is not a no-op.
/// 2. A result with wrong-sign LP dual (Le should have y≤0 in simplex; +1 is wrong)
///    is also demoted — proves the full KKT+dual_sign check is active.
///
/// The `guard_lp_optimal_does_not_demote_clean_result` test below provides the
/// complementary pass-through evidence.
#[test]
fn guard_lp_optimal_load_bearing_production_path() {
    // LP: min x  s.t.  x ≤ 5,  x ≥ 0.
    let a_le = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp_le = LpProblem::new_general(
        vec![1.0],
        a_le,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    // Corrupt primal: x=1e12 violates x≤5 → primal feasibility fails → SuboptimalSolution.
    let mut corrupt_primal = SolverResult::default();
    corrupt_primal.status = SolveStatus::Optimal;
    corrupt_primal.objective = 1e12;
    corrupt_primal.solution = vec![1e12];
    corrupt_primal.dual_solution = vec![0.0];
    corrupt_primal.reduced_costs = vec![0.0];
    corrupt_primal.slack = vec![0.0];
    let guarded = otspot::apply_lp_primal_guard(corrupt_primal, &lp_le);
    assert_eq!(
        guarded.status,
        SolveStatus::SuboptimalSolution,
        "guard must demote corrupt Optimal (pfeas≈1e12) to SuboptimalSolution; \
         if Optimal is returned, the guard was deleted or skipped"
    );

    // LP: min −x  s.t.  x ≤ 1,  x ≥ 0. Optimal x=1, dual_solution = −1 (Le, simplex).
    let a2 = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let lp2 = LpProblem::new_general(
        vec![-1.0],
        a2,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    // Wrong-sign dual: Le should have y ≤ 0 in simplex convention; +1 is wrong.
    let mut wrong_sign = SolverResult::default();
    wrong_sign.status = SolveStatus::Optimal;
    wrong_sign.objective = -1.0;
    wrong_sign.solution = vec![1.0];
    wrong_sign.dual_solution = vec![1.0]; // should be −1 for Le
    wrong_sign.reduced_costs = vec![0.0];
    wrong_sign.slack = vec![0.0];
    let guarded2 = otspot::apply_lp_primal_guard(wrong_sign, &lp2);
    assert_eq!(
        guarded2.status,
        SolveStatus::SuboptimalSolution,
        "guard must demote wrong-sign Le dual (+1 instead of ≤0) to SuboptimalSolution; \
         if Optimal is returned, dual_sign check is not active"
    );
}

/// Production pass-through: `apply_lp_primal_guard` must not demote a clean result.
///
/// Runs a real LP through the solver and routes the result through `apply_lp_primal_guard`,
/// the same function that guards every production LP exit. A clean Optimal result must
/// not be demoted to NumericalError. This catches over-eager guard thresholds.
#[test]
fn guard_lp_optimal_does_not_demote_clean_result() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 2.0],
        a,
        vec![4.0],
        vec![ConstraintType::Le],
        vec![(0.0, f64::INFINITY); 2],
        None,
    )
    .unwrap();

    let real_result = solve_lp_with(&lp, &SolverOptions::default());
    assert_eq!(
        real_result.status,
        SolveStatus::Optimal,
        "pre-guard solve failed"
    );

    // Route through production guard: must remain Optimal.
    let guarded = otspot::apply_lp_primal_guard(real_result, &lp);
    assert_eq!(
        guarded.status,
        SolveStatus::Optimal,
        "guard must not demote a clean real Optimal result"
    );
}
