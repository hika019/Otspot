//! Large feasible LP — false-Infeasible regression guard (#36/#37/#43).
//!
//! Big-M Phase I once declared Infeasible whenever an artificial stayed in the
//! basis after a Timeout/Optimal exit (`any_nonzero` short-circuit). That is
//! unsound: a slow-but-feasible LP keeps artificials simply because Phase I has
//! not finished, so the heuristic flips pilot/dfl001/ken to false-Infeasible.
//! The fix declares Infeasible ONLY via a verified Farkas certificate
//! (A^T y ≤ tol ∧ b^T y > tol).
//!
//! ## Routing note (why some tests force simplex) and what each arm covers
//!
//! `solve_qp_with` routes large LPs (n > 3000 or m > 2000) to IPM first, which
//! never touches the Big-M infeasibility arms. To exercise the simplex path we
//! set `LP_DISPATCH_NOOP=1`, forcing the Big-M Phase I to run. ken-13/ken-18
//! are too large to factorize a basis for (m ≫ 2000), so they stay on the IPM
//! path and only guard that route.
//!
//! There are two Big-M infeasibility arms; they have *different* coverage:
//! - **Timeout-arm** (`any_artificial_left && farkas`): pilot/dfl001 exhaust
//!   their budgets (12s/30s) before Phase I finishes, so they exit through this
//!   arm. These sentinels are load-bearing *for the Timeout-arm only* —
//!   flipping its `&&` to `||` (the a7b95ad band-aid) flips both to
//!   false-Infeasible (verified via no-op rewrite). This arm was already sound
//!   on `main`; the sentinels guard against re-introducing the band-aid.
//! - **Optimal-arm** (`any_artificial_in_basis && farkas`): no test reaches it
//!   with a residual nonzero artificial — Phase I never declares Optimal here
//!   on the available data. The #36 rework removed the Optimal-arm `any_nonzero`
//!   short-circuit; that removal is verified safe by (a) monotone-safety (the
//!   Farkas condition is a strict subset of `any_nonzero || farkas`, so
//!   Infeasible verdicts can only decrease) and (b) the infeasible-29 bench
//!   being bit-identical before/after — NOT by a direct sentinel here.

use otspot::io::qps::parse_qps;
use otspot::lp::solve_lp_with;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::qp::solve_qp_with;
use otspot::sparse::CscMatrix;
use std::path::Path;
use std::time::Instant;

fn solve(path_str: &str, timeout_sec: f64) -> (SolveStatus, f64, usize) {
    let path = Path::new(path_str);
    assert!(path.exists(), "data missing: {}", path_str);
    let prob = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_sec);

    let t0 = Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "{} (timeout={:.0}s) -> status={:?} obj={:.6e} wall={:.2}s iters={}",
        path_str, timeout_sec, r.status, r.objective, wall, r.iterations
    );
    (r.status, wall, r.iterations)
}

/// IPM-first routing (default for large LP).
fn assert_not_infeasible_ipm(path_str: &str, timeout_sec: f64) {
    let (status, _wall, _iters) = solve(path_str, timeout_sec);
    assert!(
        !matches!(status, SolveStatus::Infeasible),
        "{} returned Infeasible on the IPM path — feasible LP must never be \
         certified infeasible without a Farkas certificate",
        path_str
    );
}

/// Force the simplex Big-M Phase I path (`LP_DISPATCH_NOOP=1`). pilot/dfl001
/// exit via the **Timeout-arm** (budget exhausted before Phase I finishes).
/// LOAD-BEARING for that arm: changing its `any_artificial_left && farkas` to
/// `|| farkas` (the a7b95ad band-aid) flips these feasible LPs to
/// false-Infeasible and fails the assert. (The Optimal-arm is not reached here;
/// see the module docstring for its verification basis.)
fn assert_not_infeasible_forced_simplex(path_str: &str, timeout_sec: f64) {
    // SAFETY: env mutation scoped to this test; CLAUDE.md mandates nextest,
    // which isolates each test in its own process (no cross-test leak).
    std::env::set_var("LP_DISPATCH_NOOP", "1");
    let (status, _wall, _iters) = solve(path_str, timeout_sec);
    std::env::remove_var("LP_DISPATCH_NOOP");
    assert!(
        !matches!(status, SolveStatus::Infeasible),
        "{} forced through Big-M simplex returned Infeasible — the residual \
         artificial is NOT a Farkas certificate (#37/#43 false-Infeasible bug)",
        path_str
    );
}

/// pilot via the default IPM path must not be false-Infeasible.
#[test]
fn pilot_no_false_infeasible() {
    assert_not_infeasible_ipm("data/lp_problems/pilot.QPS", 120.0);
}

/// LOAD-BEARING: pilot forced through Big-M simplex. The Phase I cannot finish
/// in the budget so artificials remain in the basis at Timeout; the old
/// `any_nonzero` heuristic declared this Infeasible (verified during rework).
#[test]
fn pilot_no_false_infeasible_forced_simplex() {
    assert_not_infeasible_forced_simplex("data/lp_problems/pilot.QPS", 12.0);
}

/// LOAD-BEARING: dfl001 forced through Big-M simplex (un-ignored per #36).
/// Same mechanism as pilot, larger instance. ~30s (within per-test budget).
#[test]
fn dfl001_no_false_infeasible_forced_simplex() {
    assert_not_infeasible_forced_simplex("data/lp_problems/dfl001.QPS", 30.0);
}

/// ken-13 via IPM (m ≫ 2000 → simplex factorization impractical). Guards the
/// IPM path; resolves to Optimal quickly.
#[test]
fn ken13_no_false_infeasible() {
    assert_not_infeasible_ipm("data/lp_problems/ken-13.QPS", 60.0);
}

/// ken-18 via IPM. Heaviest instance — kept ignored for the normal suite
/// (CLAUDE.md 3-min guideline); run individually for cross-check:
/// `cargo nextest run --release ken18_no_false_infeasible --run-ignored all`.
#[test]
#[ignore = "heavy: ken-18 IPM up to 60s — individual cross-check only"]
fn ken18_no_false_infeasible() {
    assert_not_infeasible_ipm("data/lp_problems/ken-18.QPS", 60.0);
}

// ── Infeasible-arm sentinels (Farkas-cert gated routing) ─────────────────────
//
// Routing: primal Phase I Infeasible → `extract_farkas_certificate` checks the
// final basis. Valid cert (dual_solution non-empty) → trust immediately (no
// Big-M). No cert → uncertified, pilot87-class → Big-M arbiter.
//
// Two directions required:
//   (A) feasible LP (pilot87): primal Infeasible is uncertified → Big-M → !Infeasible
//   (B) infeasible LP (synthetic): primal Infeasible is Farkas-certified → Infeasible
//
// No-op proof for (A): removing `extract_farkas_certificate` (always empty dual_solution)
// makes `Infeasible if !dual_solution.is_empty()` never fire, routing all to Big-M.
// For pilot87, Big-M also times out → Timeout (still !Infeasible, test still passes).
// Stronger no-op: reverting the entire Infeasible arm to `_ => primal_result` makes
// pilot87 presolve=false return Infeasible (the original bug), which fails the assert.

/// LOAD-BEARING: pilot87 presolve=false must not be false-Infeasible (#58).
///
/// pilot87 (322 artificials: 89 Ge + 233 Eq) with presolve=false routes through
/// primal Phase I (~68s, feasible LP with cycling) → false Infeasible. The fix:
/// Farkas check on the final basis fails (LP is feasible, no dual ray exists) →
/// uncertified → Big-M arbiter → Optimal or Timeout. Both are !Infeasible.
///
/// Budget: 150s total (primal ~68s + Big-M remaining ≤ 3min guideline).
#[test]
fn pilot87_presolve_false_not_infeasible() {
    let path = Path::new("data/lp_problems/pilot87.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let prob = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(150.0);

    let t0 = Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "pilot87 presolve=false -> status={:?} obj={:.6e} wall={:.2}s iters={}",
        r.status, r.objective, wall, r.iterations
    );

    assert!(
        !matches!(r.status, SolveStatus::Infeasible),
        "pilot87 presolve=false returned Infeasible — feasible LP must never be \
         certified infeasible without a Farkas certificate (#58 false-Infeasible bug)"
    );
}

// ── Spot-check: infeasible LPs that regressed to Timeout in the naive fix ────
//
// galenet/ex72a/forest6 are Netlib infeasible LPs. main (before #58) returned
// Infeasible(iters=0) instantly via primal Phase I. The first fix attempt
// (unconditional Big-M) degraded them to Timeout. The Farkas-gated fix preserves
// them at Infeasible by trusting the primal Phase I Farkas certificate directly.

/// galenet: infeasible LP, presolve=false — must remain Infeasible (not Timeout).
#[test]
fn spot_check_galenet_no_presolve_infeasible() {
    assert_not_infeasible_regression("data/lp_problems_infeas/galenet.QPS");
}

/// ex72a: infeasible LP, presolve=false — must remain Infeasible.
#[test]
fn spot_check_ex72a_no_presolve_infeasible() {
    assert_not_infeasible_regression("data/lp_problems_infeas/ex72a.QPS");
}

/// forest6: infeasible LP, presolve=false — must remain Infeasible.
#[test]
fn spot_check_forest6_no_presolve_infeasible() {
    assert_not_infeasible_regression("data/lp_problems_infeas/forest6.QPS");
}

/// Assert that a known-infeasible LP with presolve=false still returns
/// `Infeasible` (Farkas-certified). A `Timeout` result means the fix is
/// incorrectly routing via Big-M instead of trusting the primal Farkas cert.
fn assert_not_infeasible_regression(path_str: &str) {
    let path = Path::new(path_str);
    assert!(path.exists(), "data missing: {}", path_str);
    let prob = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(30.0);

    let t0 = Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "{} presolve=false -> status={:?} wall={:.3}s iters={}",
        path_str, r.status, wall, r.iterations
    );
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "{} must be Farkas-certified Infeasible with presolve=false; \
         Timeout = regression: Farkas gate not preserving primal cert",
        path_str
    );
}

/// Bidirectional guard: trivially infeasible LP (presolve=false) must stay Infeasible.
///
/// LP: x ≤ 1 (Le), x ≥ 2 (Ge), x ≥ 0 — clearly infeasible. Primal Phase I
/// detects infeasibility and `extract_farkas_certificate` verifies the dual ray at
/// the final basis (b^T y > tol AND A^T y ≤ tol). The Farkas-certified result is
/// returned directly without Big-M. Asserts `== Infeasible` (stronger than
/// `!Optimal`) because the certificate is always valid for this trivial LP.
///
/// Over-correction guard: if the fix accidentally routed all Infeasible through
/// Big-M without Farkas gating, Big-M would cycle on this degenerate 2-constraint
/// LP and return Timeout — this test would fail.
#[test]
fn infeasible_arm_bidirectional_true_infeasible_farkas_certified() {
    // x ≤ 1 (Le) AND x ≥ 2 (Ge) — trivially infeasible, one artificial.
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1)
        .expect("CscMatrix");
    let prob = LpProblem::new_general(
        vec![0.0],
        a,
        vec![1.0, 2.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .expect("LpProblem");

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(10.0);

    let r = solve_lp_with(&prob, &opts);
    eprintln!(
        "synthetic infeasible presolve=false -> status={:?} iters={}",
        r.status, r.iterations
    );
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "trivially infeasible LP must be Farkas-certified Infeasible; \
         Timeout = over-routing to Big-M without Farkas gate"
    );
}
