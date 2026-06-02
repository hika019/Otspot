//! Micro-Q QP dispatch sentinels.
//!
//! A QP with a tiny-but-nonzero PSD Q (e.g. Q=1e-13) is bounded — its optimum is
//! at x=1/Q with obj=-1/(2Q) — yet two magnitude-threshold checks used to
//! mis-route it as an LP and report false-Unbounded:
//!   * `QpProblem::is_zero_q` (`|v| < 1e-12`) sent it to the LP path.
//!   * QP presolve step4 (`|q| > ZERO_TOL`) treated its column as a pure-LP
//!     empty column and declared Unbounded on the cost sign.
//! Both now test *structural* zero (`v == 0.0`); stored Q values are
//! structurally non-zero (`from_triplets` drops `|v| ≤ DROP_TOL`).

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveRoute, SolveStatus};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

/// min 1/2 q x²  −  x   s.t. x ≥ 0  (no linear constraints).
/// Analytic optimum: x = 1/q, obj = −1/(2q). Bounded for any q > 0.
fn micro_q_qp(qval: f64) -> QpProblem {
    let q = CscMatrix::from_triplets(&[0], &[0], &[qval], 1, 1).unwrap();
    QpProblem::new(
        q,
        vec![-1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(0.0, f64::INFINITY)],
        vec![],
    )
    .unwrap()
}

/// Tiny PSD Q is a bounded QP: it must NOT be reported Unbounded, must route to
/// the IPM (not the LP path), and must report the analytic objective.
///
/// Sentinel: reverting `is_zero_q` to the `1e-12` threshold flips the route to
/// `LpForwardedFromQp` → Unbounded; reverting the presolve step4 `q_nnz` filter
/// to `> ZERO_TOL` declares Unbounded in presolve. Either no-op fails this test.
#[test]
fn micro_q_psd_qp_is_bounded_and_routes_to_ipm() {
    let opts = SolverOptions::default(); // presolve ON: exercises both fixes.
    // q ∈ (DROP_TOL, 1e-12]: stored, but below the old is_zero_q / ZERO_TOL cutoff.
    for qval in [1e-12, 5e-13, 1e-13, 1e-14] {
        let p = micro_q_qp(qval);
        assert!(
            !p.is_zero_q(),
            "Q={qval:e} is structurally non-zero; is_zero_q must be false"
        );
        let r = crate::qp::solve_qp_with(&p, &opts);
        assert_ne!(
            r.status,
            SolveStatus::Unbounded,
            "bounded micro-Q (Q={qval:e}) must NOT be Unbounded; got obj={}",
            r.objective
        );
        assert_eq!(
            r.stats.route,
            SolveRoute::QpIpm,
            "micro-Q must route to IPM (QpIpm), not the LP path, for Q={qval:e}"
        );
        let analytic_obj = -1.0 / (2.0 * qval);
        assert!(
            r.objective.is_finite(),
            "objective must be finite for bounded Q={qval:e}; got {}",
            r.objective
        );
        let rel = (r.objective - analytic_obj).abs() / analytic_obj.abs();
        assert!(
            rel < 1e-3,
            "Q={qval:e}: obj {} must match analytic {analytic_obj} (rel={rel:e})",
            r.objective
        );
    }
}

/// Structurally-zero Q (genuine LP forwarded from QP) must keep the LP route and
/// still report genuine unboundedness — the structural `is_zero_q` must not
/// reclassify a real LP as a QP.
///
/// Sentinel: if `is_zero_q` returned false for an all-zero Q, the route would be
/// `QpIpm` instead of `LpForwardedFromQp`.
#[test]
fn structural_zero_q_keeps_lp_route_and_unbounded() {
    let p = QpProblem::new(
        CscMatrix::new(1, 1), // structurally empty Q
        vec![-1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(0.0, f64::INFINITY)],
        vec![],
    )
    .unwrap();
    assert!(p.is_zero_q(), "all-zero Q must be is_zero_q==true");
    let r = crate::qp::solve_qp_with(&p, &SolverOptions::default());
    assert_eq!(
        r.stats.route,
        SolveRoute::LpForwardedFromQp,
        "structural-zero Q must forward to the LP path"
    );
    assert_eq!(
        r.status,
        SolveStatus::Unbounded,
        "genuine LP (Q=0, c=-1, x≥0) is unbounded and must remain so"
    );
}

/// An explicit zero stored in Q (e.g. via direct construction) is still
/// structurally zero. Guards the `v == 0.0` predicate against `is_empty()`.
#[test]
fn explicit_zero_q_value_is_structural_zero() {
    // A 2x2 Q with one stored-but-zero diagonal and one genuine entry: the
    // all-zero predicate must be false (genuine curvature present), while a
    // fully-zero stored Q must be true.
    let q_mixed = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.0_f64, 1.0], 2, 2).unwrap();
    // from_triplets drops the 0.0 entry, leaving only the genuine 1.0.
    let p_mixed = QpProblem::new(
        q_mixed,
        vec![-1.0, -1.0],
        CscMatrix::new(0, 2),
        vec![],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        vec![],
    )
    .unwrap();
    assert!(
        !p_mixed.is_zero_q(),
        "Q with a genuine 1.0 entry must not be is_zero_q"
    );

    // Linear (Q empty) with a real Ge constraint: structural zero, LP route.
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let p_lp = QpProblem::new(
        CscMatrix::new(1, 1),
        vec![1.0],
        a,
        vec![2.0],
        vec![(0.0, f64::INFINITY)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    assert!(p_lp.is_zero_q());
    let r = crate::qp::solve_qp_with(&p_lp, &SolverOptions::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - 2.0).abs() < 1e-9, "min x s.t. x≥2 → obj=2");
}
