//! LP/QP entry split sentinel.
//!
//! Verifies that `Model::solve` routes LP through `crate::lp::solve_lp_with`
//! and QP through `crate::qp::solve_qp_with` → IPM, that `solve_qp_with(Q=0)`
//! forwards to the LP module on a distinguishable route, and that
//! `solve_lp_with(LpProblem)` matches the forwarded result. The
//! `sentinel_proves_lp_path_regression_detectable` case is a no-op proof:
//! it routes an LP through QP entry to show the routes discriminate direct
//! vs forwarded paths, so a regression of `Model::solve` LP path onto
//! `solve_qp_with` would fail the `LpDirect` assertion.
//!
//! Per-result stats (SolverResult.stats / ModelResult.stats) replace
//! process-global AtomicU64 counters — no SENTINEL_LOCK or reset needed.

use otspot::constraint;
use otspot::model::Model;
use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveRoute};
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;

/// Pin `SimplexMethod::Primal`: dual paths regress textbook fixtures to
/// x≈1.99977 instead of vertex 2.0 on current main, which is
/// orthogonal noise to the routing assertions here.
fn lp_opts_strict() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.simplex_method = SimplexMethod::Primal;
    o
}

/// Loose enough for IPM eps=1e-6 QP solutions; LP fixtures hit vertices exactly.
const TOL_OBJ: f64 = 1e-4;
const TOL_SOL: f64 = 1e-4;

// ===========================================================================
// LP fixtures (5 pattern)
// ===========================================================================

/// (LpProblem, expected_objective, label)
fn lp_fixtures() -> Vec<(LpProblem, f64, &'static str)> {
    vec![
        lp_fix_trivial_bound(),
        lp_fix_le_two_var(),
        lp_fix_ge_two_var(),
        lp_fix_eq_three_var(),
        lp_fix_degenerate(),
    ]
}

fn lp_fix_trivial_bound() -> (LpProblem, f64, &'static str) {
    // min x s.t. x >= 1, 0 <= x <= 10
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1.0];
    let ct = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, 10.0)];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("trivial_bound".into())).unwrap();
    (lp, 1.0, "lp_trivial_bound")
}

fn lp_fix_le_two_var() -> (LpProblem, f64, &'static str) {
    // min -3x - 5y  s.t. x <= 4, 2y <= 12, 3x+2y <= 18, x,y>=0
    // Classic textbook problem; opt at (2,6) obj = -36
    let c = vec![-3.0, -5.0];
    let rows = vec![0, 1, 2, 2];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 2.0, 3.0, 2.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();
    let b = vec![4.0, 12.0, 18.0];
    let ct = vec![ConstraintType::Le; 3];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("le_two_var".into())).unwrap();
    (lp, -36.0, "lp_le_two_var")
}

fn lp_fix_ge_two_var() -> (LpProblem, f64, &'static str) {
    // min x+y  s.t. x+y >= 2, x,y >= 0  → obj=2
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![2.0];
    let ct = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("ge_two_var".into())).unwrap();
    (lp, 2.0, "lp_ge_two_var")
}

fn lp_fix_eq_three_var() -> (LpProblem, f64, &'static str) {
    // min x+2y+3z  s.t. x+y+z = 1, 0<=x,y,z<=1  → obj=1 at (1,0,0)
    let c = vec![1.0, 2.0, 3.0];
    let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
    let b = vec![1.0];
    let ct = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, 1.0); 3];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("eq_three_var".into())).unwrap();
    (lp, 1.0, "lp_eq_three_var")
}

fn lp_fix_degenerate() -> (LpProblem, f64, &'static str) {
    // min x+y s.t. x+y <= 2, x+y <= 2 (dup), x,y >= 0  → obj=0 at origin
    let c = vec![1.0, 1.0];
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();
    let b = vec![2.0, 2.0];
    let ct = vec![ConstraintType::Le; 2];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("degenerate".into())).unwrap();
    (lp, 0.0, "lp_degenerate")
}

// ===========================================================================
// QP fixtures (5 pattern)
// ===========================================================================

fn qp_fixtures() -> Vec<(QpProblem, f64, &'static str)> {
    vec![
        qp_fix_diag_psd_box(),
        qp_fix_offdiag_psd(),
        qp_fix_pure_box(),
        qp_fix_eq_inequality_mix(),
        qp_fix_large_diag_psd(50),
    ]
}

fn symmetric_q_csc(n: usize, upper_triples: &[(usize, usize, f64)]) -> CscMatrix {
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for &(r, c, v) in upper_triples {
        assert!(r <= c, "upper triple expected: ({},{})", r, c);
        rows.push(r);
        cols.push(c);
        vals.push(v);
        if r != c {
            rows.push(c);
            cols.push(r);
            vals.push(v);
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

fn qp_fix_diag_psd_box() -> (QpProblem, f64, &'static str) {
    // min 1/2 (x²+y²) s.t. x+y >= 1, x,y >= 0
    let q = symmetric_q_csc(2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let ct = vec![ConstraintType::Ge];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, 0.25, "qp_diag_psd_box")
}

fn qp_fix_offdiag_psd() -> (QpProblem, f64, &'static str) {
    let q = symmetric_q_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)]);
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(-10.0, 10.0); 2];
    let ct = vec![];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -1.0 / 3.0, "qp_offdiag_psd")
}

fn qp_fix_pure_box() -> (QpProblem, f64, &'static str) {
    let q = symmetric_q_csc(1, &[(0, 0, 1.0)]);
    let c = vec![-3.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(0.0, 2.0)];
    let ct = vec![];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -4.0, "qp_pure_box")
}

fn qp_fix_eq_inequality_mix() -> (QpProblem, f64, &'static str) {
    let q = symmetric_q_csc(3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
    let c = vec![0.0, 0.0, 0.0];
    let rows = vec![0, 0, 0, 1];
    let cols = vec![0, 1, 2, 0];
    let vals = vec![1.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
    let b = vec![3.0, 0.5];
    let bounds = vec![(-10.0, 10.0); 3];
    let ct = vec![ConstraintType::Eq, ConstraintType::Ge];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, 1.5, "qp_eq_inequality_mix")
}

fn qp_fix_large_diag_psd(n: usize) -> (QpProblem, f64, &'static str) {
    let mut q_rows = Vec::with_capacity(n);
    let mut q_cols = Vec::with_capacity(n);
    let mut q_vals = Vec::with_capacity(n);
    for i in 0..n {
        q_rows.push(i);
        q_cols.push(i);
        q_vals.push(1.0);
    }
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
    let c = vec![-1.0; n];
    let rows: Vec<usize> = vec![0; n];
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0; n];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, n).unwrap();
    let b = vec![n as f64];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let ct = vec![ConstraintType::Le];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -(n as f64) / 2.0, "qp_large_diag_psd")
}

// ===========================================================================
// Helpers
// ===========================================================================

fn assert_obj(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    let tol = TOL_OBJ * (1.0 + expected.abs());
    assert!(
        diff < tol,
        "{}: obj={:.9e} expected={:.9e} diff={:.3e} (tol={:.3e})",
        label,
        actual,
        expected,
        diff,
        tol
    );
}

// ===========================================================================
// Test cases (no SENTINEL_LOCK — per-result stats are race-free)
// ===========================================================================

#[test]
fn lp_direct_entry_solves_all_fixtures() {
    let opts = lp_opts_strict();
    for (lp, expected, label) in lp_fixtures() {
        let r = otspot::lp::solve_lp_with(&lp, &opts);
        assert_eq!(
            r.status,
            otspot::SolveStatus::Optimal,
            "{}: status={:?}",
            label,
            r.status
        );
        assert_obj(r.objective, expected, label);
        assert_eq!(
            r.stats.route,
            SolveRoute::LpDirect,
            "{}: route must be LpDirect",
            label
        );
        assert_ne!(
            r.stats.route,
            SolveRoute::LpForwardedFromQp,
            "{}: forward path must not fire on direct entry",
            label
        );
        assert_ne!(
            r.stats.route,
            SolveRoute::QpIpm,
            "{}: QP IPM must not fire for LP entry",
            label
        );
    }
}

#[test]
fn qp_direct_entry_solves_all_fixtures() {
    let opts = SolverOptions::default();
    for (qp, expected, label) in qp_fixtures() {
        let r = otspot::solve_qp_with(&qp, &opts);
        assert_eq!(
            r.status,
            otspot::SolveStatus::Optimal,
            "{}: status={:?}",
            label,
            r.status
        );
        assert_obj(r.objective, expected, label);
        assert_eq!(
            r.stats.route,
            SolveRoute::QpIpm,
            "{}: route must be QpIpm (Q!=0 → IPM)",
            label
        );
        assert_ne!(
            r.stats.route,
            SolveRoute::LpDirect,
            "{}: LP direct counter must stay 0 on QP entry",
            label
        );
        assert_ne!(
            r.stats.route,
            SolveRoute::LpForwardedFromQp,
            "{}: LP forward route must not fire (Q!=0)",
            label
        );
    }
}

#[test]
fn model_api_lp_path_uses_solve_lp_with() {
    // min x s.t. x >= 1, 0 <= x <= 10  → obj=1
    let mut m = Model::new("lp_via_model");
    let x = m.add_var("x", 0.0, 10.0);
    m.add_constraint(constraint!(x >= 1.0));
    m.minimize(x);
    let r = m.solve().expect("model lp");
    assert_obj(r.objective_value, 1.0, "model_lp_path");

    assert_eq!(
        r.stats.route,
        SolveRoute::LpDirect,
        "Model::solve LP path must use LpDirect route (got {:?})",
        r.stats.route
    );
    assert_ne!(
        r.stats.route,
        SolveRoute::QpIpm,
        "Model::solve LP path must NOT use QpIpm route"
    );
    assert_ne!(
        r.stats.route,
        SolveRoute::LpForwardedFromQp,
        "Model::solve LP path must NOT route via QP→LP forward"
    );
}

#[test]
fn model_api_qp_path_uses_solve_qp_with() {
    // min 1/2 x²  s.t. x >= 1 → obj=0.5
    let mut m = Model::new("qp_via_model");
    let x = m.add_var("x", 0.0, 10.0);
    m.add_constraint(constraint!(x >= 1.0));
    // Q[0][0]=1: (1/2)*x*x via DSL
    m.minimize(0.5 * x * x);
    let r = m.solve().expect("model qp");
    assert_obj(r.objective_value, 0.5, "model_qp_path");

    assert_eq!(
        r.stats.route,
        SolveRoute::QpIpm,
        "Model::solve QP path must use QpIpm route"
    );
    assert_ne!(
        r.stats.route,
        SolveRoute::LpDirect,
        "Model::solve QP path must NOT use LpDirect route"
    );
    assert_ne!(
        r.stats.route,
        SolveRoute::LpForwardedFromQp,
        "Q!=0 must NOT route via QP→LP forward"
    );
}

/// Q=0 through QP entry: same optimum as direct LP entry, but route = LpForwardedFromQp.
#[test]
fn qp_entry_with_zero_q_forwards_to_lp_module() {
    let opts = lp_opts_strict();
    let (lp, expected, label) = lp_fix_le_two_var();

    // Direct LP entry
    let r_direct = otspot::lp::solve_lp_with(&lp, &opts);
    assert_obj(r_direct.objective, expected, "direct_lp");
    assert_eq!(r_direct.stats.route, SolveRoute::LpDirect);

    // Wrap as Q=0 QpProblem and call QP entry
    let n = lp.num_vars;
    let q_zero = CscMatrix::new(n, n);
    let qp = QpProblem::new(
        q_zero,
        lp.c.clone(),
        (*lp.a).clone(),
        lp.b.clone(),
        lp.bounds.clone(),
        lp.constraint_types.clone(),
    )
    .unwrap();

    let r_qp = otspot::solve_qp_with(&qp, &opts);
    assert_obj(r_qp.objective, expected, "qp_zero_q_forward");
    assert_eq!(
        r_qp.stats.route,
        SolveRoute::LpForwardedFromQp,
        "{}: solve_qp_with(Q=0) must use LpForwardedFromQp route",
        label
    );
    assert_ne!(
        r_qp.stats.route,
        SolveRoute::LpDirect,
        "{}: solve_qp_with(Q=0) must NOT use LpDirect route",
        label
    );
    assert_ne!(
        r_qp.stats.route,
        SolveRoute::QpIpm,
        "{}: Q=0 must NOT trigger IPM route",
        label
    );

    // Objective equivalence
    assert!(
        (r_direct.objective - r_qp.objective).abs() < TOL_OBJ * (1.0 + expected.abs()),
        "{}: direct LP obj {:.9e} vs QP-forward obj {:.9e}",
        label,
        r_direct.objective,
        r_qp.objective
    );
}

/// No-op proof: routes discriminate direct vs forwarded paths.
/// A regression of any LP path onto `solve_qp_with` would change route to
/// LpForwardedFromQp and fail the LpDirect assertion.
#[test]
fn sentinel_proves_lp_path_regression_detectable() {
    let opts = lp_opts_strict();
    for (lp, _expected, label) in lp_fixtures() {
        let n = lp.num_vars;
        let q_zero = CscMatrix::new(n, n);
        let qp = QpProblem::new(
            q_zero,
            lp.c.clone(),
            (*lp.a).clone(),
            lp.b.clone(),
            lp.bounds.clone(),
            lp.constraint_types.clone(),
        )
        .unwrap();
        let r = otspot::solve_qp_with(&qp, &opts);
        assert_ne!(
            r.stats.route,
            SolveRoute::LpDirect,
            "{}: regression simulation (solve_qp_with on LP) must NOT use LpDirect route",
            label
        );
        assert_eq!(
            r.stats.route,
            SolveRoute::LpForwardedFromQp,
            "{}: regression simulation must use LpForwardedFromQp route (sentinel discriminator)",
            label
        );
    }
}

/// Direct LP entry and QP→LP forward must agree on objective + solution.
#[test]
fn cross_check_lp_direct_vs_qp_forward_objective() {
    let opts = lp_opts_strict();
    for (lp, expected, label) in lp_fixtures() {
        let r_direct = otspot::lp::solve_lp_with(&lp, &opts);

        let n = lp.num_vars;
        let q_zero = CscMatrix::new(n, n);
        let qp = QpProblem::new(
            q_zero,
            lp.c.clone(),
            (*lp.a).clone(),
            lp.b.clone(),
            lp.bounds.clone(),
            lp.constraint_types.clone(),
        )
        .unwrap();
        let r_fwd = otspot::solve_qp_with(&qp, &opts);

        assert_obj(r_direct.objective, expected, &format!("{}_direct", label));
        assert_obj(r_fwd.objective, expected, &format!("{}_forward", label));
        assert!(
            (r_direct.objective - r_fwd.objective).abs() < TOL_OBJ * (1.0 + expected.abs()),
            "{}: direct={:.9e} forward={:.9e}",
            label,
            r_direct.objective,
            r_fwd.objective
        );
        if !r_direct.solution.is_empty() && !r_fwd.solution.is_empty() {
            assert_eq!(r_direct.solution.len(), r_fwd.solution.len());
            for (i, (a, b)) in r_direct
                .solution
                .iter()
                .zip(r_fwd.solution.iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() < TOL_SOL * (1.0 + a.abs() + b.abs()),
                    "{}: solution[{}] direct={:.9e} forward={:.9e}",
                    label,
                    i,
                    a,
                    b
                );
            }
        }
    }
}

/// Parallel solves have independent stats — no global state leaks across results.
/// This is the primary regression test for the per-result migration.
#[test]
fn parallel_solve_stats_are_independent() {
    use std::thread;

    let opts_lp = lp_opts_strict();
    let (lp, _, _) = lp_fix_le_two_var();
    let (qp, _, _) = qp_fix_diag_psd_box();

    // Spawn LP and QP solves "simultaneously" via threads.
    let lp_clone = lp.clone();
    let opts_clone = opts_lp.clone();
    let lp_handle = thread::spawn(move || otspot::lp::solve_lp_with(&lp_clone, &opts_clone));

    let qp_opts = SolverOptions::default();
    let qp_handle = thread::spawn(move || otspot::solve_qp_with(&qp, &qp_opts));

    let r_lp = lp_handle.join().unwrap();
    let r_qp = qp_handle.join().unwrap();

    // Each result carries its own route — no contamination.
    assert_eq!(r_lp.stats.route, SolveRoute::LpDirect, "LP result route");
    assert_eq!(r_qp.stats.route, SolveRoute::QpIpm, "QP result route");

    // A third sequential solve must also see correct independent stats.
    let r_fwd = {
        let (lp2, _, _) = lp_fix_trivial_bound();
        let n = lp2.num_vars;
        let q_zero = CscMatrix::new(n, n);
        let qp2 = QpProblem::new(
            q_zero,
            lp2.c,
            (*lp2.a).clone(),
            lp2.b,
            lp2.bounds,
            lp2.constraint_types,
        )
        .unwrap();
        otspot::solve_qp_with(&qp2, &opts_lp)
    };
    assert_eq!(
        r_fwd.stats.route,
        SolveRoute::LpForwardedFromQp,
        "forwarded result route"
    );
}
