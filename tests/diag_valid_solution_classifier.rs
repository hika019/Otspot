//! Valid solution / Optimal / TIMEOUT classifier TDD sentinel.
//!
//! Contract under test:
//!
//! | State | Condition | Behavior |
//! |-------|-----------|----------|
//! | Optimal | KKT < eps AND (ref: obj within tol) | Early exit, Optimal, solution returned |
//! | Valid (Optimal) | KKT ≈ satisfied (SuboptimalSolution) in IPM + simplex Timeout | Return SuboptimalSolution, not Timeout |
//! | TIMEOUT | deadline reached + no valid solution | No solution, Timeout |
//! | Error | Unexpected | NumericalError etc. |
//!
//! Core dfl001 scenario: IPM achieves pf<eps but df>eps → SuboptimalSolution.
//! Simplex retries but times out. Current (pre-fix): TIMEOUT. Expected: SuboptimalSolution.
//!
//! TDD methodology: tests written first, FAILs confirmed, then impl, then PASSes.

use solver::bench_utils::{obj_within_tol, pick_best_ipm_or_simplex, OBJ_MATCH_REL_TOL};
use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus, SolverResult};
use solver::qp::QpProblem;
use solver::sparse::CscMatrix;
use solver::solve_qp_with;

// --- helper ---

fn make_result(status: SolveStatus, solution: Vec<f64>, objective: f64) -> SolverResult {
    SolverResult { status, solution, objective, ..Default::default() }
}

/// Simple LP: min x1+x2, s.t. x1+x2 = 1, x1,x2 >= 0
/// Optimal: obj = 1.0, any (x1,x2) with x1+x2=1, x_i>=0.
fn small_lp() -> QpProblem {
    let n = 2;
    let m = 1;
    let q = CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap();
    let c = vec![1.0, 1.0];
    let rows = vec![0usize, 0];
    let cols = vec![0usize, 1];
    let vals = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let ctypes = vec![ConstraintType::Eq; m];
    QpProblem::new(q, c, a, b, bounds, ctypes).unwrap()
}

/// Large LP: n=3200 (> LP_IPM_FIRST_N=3000), m=3200, simple diagonal system.
/// min sum(x_i), s.t. x_i = 0.5 for all i, x_i >= 0.
/// Optimal: x_i = 0.5, obj = 0.5 * n.
fn large_lp_with_known_optimal() -> (QpProblem, f64) {
    let n: usize = 3_200;
    let m: usize = n;
    let q = CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap();
    let c = vec![1.0; n];
    // A = identity n×n
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0; n];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b = vec![0.5; m];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let ctypes = vec![ConstraintType::Eq; m];
    let qp = QpProblem::new(q, c, a, b, bounds, ctypes).unwrap();
    let known_opt = 0.5 * n as f64; // = 1600.0
    (qp, known_opt)
}

// --- Test 1: small LP converges to Optimal ---

/// Existing behavior: small LP satisfies KKT → Optimal with solution.
#[test]
fn small_lp_optimal_early_exit_within_eps() {
    let qp = small_lp();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let r = solve_qp_with(&qp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "small LP must converge to Optimal");
    assert!(!r.solution.is_empty(), "solution must be non-empty");
    let obj_err = (r.objective - 1.0).abs();
    assert!(obj_err < 1e-6, "obj must be ~1.0, got {}", r.objective);
}

// --- Test 2: small LP, no reference → still Optimal ---

/// KKT < eps → Optimal regardless of whether known_optimal_obj is set.
#[test]
fn small_lp_no_ref_returns_optimal_when_kkt_within_eps() {
    let qp = small_lp();
    let opts = SolverOptions::default();
    let r = solve_qp_with(&qp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal, "KKT satisfied → Optimal (no ref needed)");
    assert!(!r.solution.is_empty());
}

// --- Test 3: large LP with known_optimal_obj → Optimal via IPM path ---

/// Large LP (triggers IPM-first dispatch). With known_optimal_obj set to the
/// analytical optimal, solver must return Optimal and not Timeout.
///
/// Uses `SolverOptions::known_optimal_obj` (new field).
/// FAILS at compile time if the field does not exist.
#[test]
fn large_lp_with_known_ref_exits_early_as_optimal() {
    let (qp, known_opt) = large_lp_with_known_optimal();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    opts.known_optimal_obj = Some(known_opt);
    let r = solve_qp_with(&qp, &opts);
    assert_eq!(
        r.status, SolveStatus::Optimal,
        "large LP with known ref must return Optimal (IPM path), got {:?}",
        r.status
    );
    assert!(r.objective.is_finite());
    let rel_err = (r.objective - known_opt).abs() / (1.0 + known_opt.abs());
    assert!(
        rel_err < 1e-4,
        "large LP obj mismatch: got {:.6e}, expected {:.6e}, rel_err={:.2e}",
        r.objective, known_opt, rel_err
    );
}

// --- Test 4: IPM SuboptimalSolution must not be silently replaced by simplex Timeout ---

/// pick_best_ipm_or_simplex: when IPM gave SuboptimalSolution + solution vector,
/// and simplex returns Timeout, prefer the IPM result.
///
/// This is the dfl001 scenario: pf<eps but df>eps → SuboptimalSolution,
/// then simplex times out. Without fix: Timeout returned. With fix: SuboptimalSolution.
///
/// FAILS before implementation because pick_best_ipm_or_simplex returns simplex (no-op stub).
#[test]
fn ipm_suboptimal_not_silently_degraded_to_timeout() {
    let ipm_subopt = make_result(
        SolveStatus::SuboptimalSolution,
        vec![0.5, 0.3, 0.2],
        1.23,
    );
    let simplex_timeout = make_result(SolveStatus::Timeout, vec![], f64::INFINITY);

    let result = pick_best_ipm_or_simplex(Some(ipm_subopt.clone()), simplex_timeout);
    assert_eq!(
        result.status,
        SolveStatus::SuboptimalSolution,
        "IPM SuboptimalSolution must be preserved over simplex Timeout"
    );
    assert_eq!(result.solution, ipm_subopt.solution, "solution must come from IPM result");
    assert_eq!(result.objective, ipm_subopt.objective, "objective must come from IPM result");
}

/// When simplex succeeds (Optimal), prefer simplex over IPM SuboptimalSolution.
#[test]
fn simplex_optimal_preferred_over_ipm_suboptimal() {
    let ipm_subopt = make_result(SolveStatus::SuboptimalSolution, vec![0.5; 3], 5.0);
    let simplex_opt = make_result(SolveStatus::Optimal, vec![0.1; 3], 1.0);

    let result = pick_best_ipm_or_simplex(Some(ipm_subopt), simplex_opt.clone());
    assert_eq!(result.status, SolveStatus::Optimal, "simplex Optimal beats IPM SuboptimalSolution");
}

/// When IPM had no candidate (e.g., IPM returned Timeout immediately), simplex result flows through.
#[test]
fn no_ipm_candidate_simplex_result_flows_through() {
    let simplex_timeout = make_result(SolveStatus::Timeout, vec![], f64::INFINITY);
    let result = pick_best_ipm_or_simplex(None, simplex_timeout.clone());
    assert_eq!(result.status, SolveStatus::Timeout, "no IPM candidate → Timeout preserved");
}

/// IPM SuboptimalSolution with empty solution must not replace simplex Timeout.
/// Empty solution = IPM never found a useful point.
#[test]
fn ipm_suboptimal_empty_solution_does_not_override_timeout() {
    let ipm_subopt_empty = make_result(SolveStatus::SuboptimalSolution, vec![], 0.0);
    let simplex_timeout = make_result(SolveStatus::Timeout, vec![], f64::INFINITY);

    let result = pick_best_ipm_or_simplex(Some(ipm_subopt_empty), simplex_timeout);
    assert_eq!(
        result.status,
        SolveStatus::Timeout,
        "SuboptimalSolution with empty solution must not override Timeout"
    );
}

// --- Test 5: no valid solution → Timeout ---

/// Large LP with 0-second timeout → solver cannot produce valid solution → Timeout.
#[test]
fn no_valid_solution_full_timeout_returns_timeout() {
    let (qp, _) = large_lp_with_known_optimal();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(0.0); // immediate timeout
    let r = solve_qp_with(&qp, &opts);
    assert_eq!(
        r.status,
        SolveStatus::Timeout,
        "0-second timeout with no prior solution must return Timeout"
    );
}

// --- Test 6: error status preserved ---

/// NumericalError from a degenerate result must not be promoted or replaced.
/// Tests that the pick_best helper respects NumericalError from IPM.
#[test]
fn numerical_error_from_ipm_preserved_when_simplex_also_fails() {
    let ipm_err = make_result(SolveStatus::NumericalError, vec![], f64::NAN);
    let simplex_timeout = make_result(SolveStatus::Timeout, vec![], f64::INFINITY);

    // NumericalError is not a "valid solution candidate" so simplex result should win
    let result = pick_best_ipm_or_simplex(Some(ipm_err), simplex_timeout.clone());
    assert_eq!(
        result.status,
        SolveStatus::Timeout,
        "IPM NumericalError is not a valid candidate; simplex Timeout should win"
    );
}

// --- Test 7: obj_within_tol utility ---

/// obj_within_tol: |obj - ref| / (1 + |ref|) < tol
#[test]
fn obj_within_tol_basic() {
    assert!(obj_within_tol(1.0, 1.0, OBJ_MATCH_REL_TOL), "identical objs match");
    assert!(obj_within_tol(1.0, 1.0 + 1e-7, OBJ_MATCH_REL_TOL), "tiny diff matches");
    assert!(!obj_within_tol(1.0, 2.0, OBJ_MATCH_REL_TOL), "far apart does not match");
    assert!(!obj_within_tol(f64::INFINITY, 1.0, OBJ_MATCH_REL_TOL), "infinite obj no match");
}
