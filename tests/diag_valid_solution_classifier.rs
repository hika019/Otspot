//! Valid solution / Optimal / TIMEOUT classifier sentinel.
//!
//! Contract under test:
//!
//! | State | Condition | Behavior |
//! |-------|-----------|----------|
//! | Optimal | KKT < eps AND (ref: obj within tol) | Early exit, Optimal, solution returned |
//! | TIMEOUT | deadline reached + no valid solution | No solution, Timeout |
//!
//! (LP は IPM を撤廃し simplex 一本化したため、旧 IPM/simplex picker の分岐契約は
//! 対象外。end-to-end の Optimal/Timeout 判定のみを pin する。)

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::qp::QpProblem;
use otspot::solve_qp_with;
use otspot::sparse::CscMatrix;
use otspot_dev::bench_utils::{obj_within_tol, OBJ_MATCH_REL_TOL};

// --- helpers ---

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

/// Large LP: n=3200, m=3200, simple diagonal system.
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
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "small LP must converge to Optimal"
    );
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
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "KKT satisfied → Optimal (no ref needed)"
    );
    assert!(!r.solution.is_empty());
}

// --- Test 3: large LP with known_optimal_obj → Optimal ---

/// Large LP with known_optimal_obj set to the analytical optimal: solver must
/// return Optimal and not Timeout.
#[test]
fn large_lp_with_known_ref_exits_early_as_optimal() {
    let (qp, known_opt) = large_lp_with_known_optimal();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    opts.known_optimal_obj = Some(known_opt);
    let r = solve_qp_with(&qp, &opts);
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "large LP with known ref must return Optimal, got {:?}",
        r.status
    );
    assert!(r.objective.is_finite());
    let rel_err = (r.objective - known_opt).abs() / (1.0 + known_opt.abs());
    assert!(
        rel_err < 1e-4,
        "large LP obj mismatch: got {:.6e}, expected {:.6e}, rel_err={:.2e}",
        r.objective,
        known_opt,
        rel_err
    );
}

// --- Test 4: no valid solution → Timeout ---

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

// --- Test 5: obj_within_tol utility ---

/// obj_within_tol: |obj - ref| / (1 + |ref|) < tol
#[test]
fn obj_within_tol_basic() {
    assert!(
        obj_within_tol(1.0, 1.0, OBJ_MATCH_REL_TOL),
        "identical objs match"
    );
    assert!(
        obj_within_tol(1.0, 1.0 + 1e-7, OBJ_MATCH_REL_TOL),
        "tiny diff matches"
    );
    assert!(
        !obj_within_tol(1.0, 2.0, OBJ_MATCH_REL_TOL),
        "far apart does not match"
    );
    assert!(
        !obj_within_tol(f64::INFINITY, 1.0, OBJ_MATCH_REL_TOL),
        "infinite obj no match"
    );
}
