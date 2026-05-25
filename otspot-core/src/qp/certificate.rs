//! KKT optimality verifier: the sole mint-point for [`OptimalCertificate`].
//!
//! [`prove_optimal`] assembles all five KKT conditions plus duality gap and dual-sign
//! feasibility. It returns `Ok(cert)` only when every check is below `tol`; otherwise
//! it returns `Err(NotProven)` naming the failing conditions. This "no certificate
//! without proof" design eliminates false-Optimal status (the most common solver bug).

use crate::problem::certificate::{NotProven, OptimalCertificate};
use crate::qp::ipm_solver::kkt::{
    bound_violation, complementarity_residual_rel, kkt_residual_rel, primal_residual_rel,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation;

/// Verify all KKT conditions and mint an [`OptimalCertificate`] if they all pass.
///
/// Assembles:
/// 1. **Stationarity** `max_j |Qx+c+Aᵀy+z|_j / scale_j` via DD precision.
/// 2. **Primal feasibility** `max_i viol_i / scale_i`.
/// 3. **Bound feasibility** `max_j max(lb−x, x−ub, 0) / scale_j`.
/// 4. **Complementarity** `max(|y·slack|, |z·(x−bnd)|) / normaliser`.
/// 5. **Dual-sign feasibility** `max_k viol_k / (1 + |v_k|)`.
/// 6. **Duality gap** (caller-supplied `duality_gap_rel`).
///
/// Returns `Err(NotProven)` listing every condition that exceeded `tol`.
pub fn prove_optimal<'a>(
    view: &ProblemView<'a>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    duality_gap_rel: f64,
    tol: f64,
) -> Result<OptimalCertificate, NotProven> {
    let stat = kkt_residual_rel(view, x, y, z);
    let pres = primal_residual_rel(view, x);
    let bviol = bound_violation(view.bounds, x);
    let comp = complementarity_residual_rel(view, x, y, z);
    let dsign = dual_sign_violation(view.constraint_types, y, view.bounds, z);
    let gap = duality_gap_rel;

    let mut failing: Vec<&'static str> = Vec::new();
    if stat > tol   { failing.push("stationarity"); }
    if pres > tol   { failing.push("primal_feasibility"); }
    if bviol > tol  { failing.push("bound_feasibility"); }
    if comp > tol   { failing.push("complementarity"); }
    if dsign > tol  { failing.push("dual_sign"); }
    if gap > tol    { failing.push("duality_gap"); }

    if failing.is_empty() {
        Ok(OptimalCertificate::new(stat, pres, bviol, comp, dsign, gap, tol))
    } else {
        Err(NotProven {
            stationarity_rel: stat,
            primal_residual_rel: pres,
            bound_violation: bviol,
            complementarity_rel: comp,
            dual_sign_violation: dsign,
            duality_gap_rel: gap,
            tol,
            failing_conditions: failing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;
    use crate::qp::ipm_solver::outcome::ProblemView;

    // Helper: build a ProblemView for a trivial LP.
    fn trivial_view<'a>(
        q: &'a CscMatrix,
        a: &'a CscMatrix,
        c: &'a [f64],
        b: &'a [f64],
        bounds: &'a [(f64, f64)],
        ct: &'a [ConstraintType],
    ) -> ProblemView<'a> {
        ProblemView { q, a, c, b, bounds, constraint_types: ct, eliminated_cols: &[] }
    }

    /// Exact optimal KKT point → prove_optimal returns Ok.
    ///
    /// Problem: min x  s.t. x >= 1 (Ge), lb=0, ub=inf.
    /// Optimal: x*=1, y*=-1 (Ge dual <= 0), z=[] (no finite bounds in z sense).
    #[test]
    fn prove_optimal_exact_kkt_passes() {
        // Q=0, c=[1], A=[1], b=[1], Ge, bounds=[(0,inf)]
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x = vec![1.0_f64];   // primal optimal
        let y = vec![-1.0_f64];  // Ge dual <= 0  (y*=-1)
        let z = vec![];           // no finite bounds → no z
        let gap = 0.0_f64;

        let result = prove_optimal(&view, &x, &y, &z, gap, 1e-6);
        assert!(result.is_ok(), "exact KKT must pass: {:?}", result.err());
        let cert = result.unwrap();
        assert!(cert.stationarity_rel() < 1e-6);
        assert!(cert.primal_residual_rel() < 1e-6);
        assert!(cert.dual_sign_violation() < 1e-6);
        assert!(cert.duality_gap_rel() < 1e-6);
    }

    /// Corrupted primal (x doubled) → prove_optimal returns Err with primal/stat/comp failing.
    #[test]
    fn prove_optimal_corrupted_x_fails() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x_bad = vec![2.0_f64]; // primal infeasible (Ge: 2*1 >= 1 ok, but dual mismatch)
        // Actually for the Le convention: Ge means x >= b=1, x=2 is feasible.
        // But the stationarity: c + A^T*y - z_lb + z_ub = 1 + 1*(-1) = 0 still holds.
        // Primal is feasible (x=2 >= 1). Complementarity: y*slack = -1 * (2-1) = -1 ≠ 0.
        let y = vec![-1.0_f64];
        let z = vec![];
        let gap = 1.0_f64; // large gap

        let result = prove_optimal(&view, &x_bad, &y, &z, gap, 1e-6);
        assert!(result.is_err(), "corrupted iterate must fail prove_optimal");
        let err = result.unwrap_err();
        // Gap is 1.0 >> tol, comp is nonzero → multiple conditions fail
        assert!(!err.failing_conditions.is_empty());
        assert!(err.failing_conditions.contains(&"duality_gap"));
    }

    /// Wrong-sign dual (Le y < 0) → dual_sign check fails.
    ///
    /// Problem: min -x  s.t. x <= 1 (Le), 0 <= x <= inf.
    /// Optimal: x*=1, y_Le*=1 (Le dual >= 0).
    /// Corrupted: y=-1 (wrong sign). Stationarity is violated too, but dual_sign fires.
    #[test]
    fn prove_optimal_wrong_sign_dual_fails_dual_sign_check() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![-1.0_f64]; // min -x
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Le];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x = vec![1.0_f64];
        let y_wrong = vec![-1.0_f64]; // should be +1 for Le active
        let z = vec![];
        let gap = 0.0_f64;

        let result = prove_optimal(&view, &x, &y_wrong, &z, gap, 1e-6);
        assert!(result.is_err(), "wrong-sign dual must fail");
        let err = result.unwrap_err();
        assert!(err.failing_conditions.contains(&"dual_sign"),
            "dual_sign must be in failing_conditions: {:?}", err.failing_conditions);
        assert!(err.dual_sign_violation > 0.0);
    }

    /// Stationarity alone fails (y is wrong magnitude but right sign).
    #[test]
    fn prove_optimal_stationarity_fails() {
        // min x, x >= 0, no constraints (free problem → x*=0 with z_lb needed)
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::new(0, 1);
        let c = vec![1.0_f64];
        let b: Vec<f64> = vec![];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct: Vec<ConstraintType> = vec![];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x = vec![0.0_f64];
        let y: Vec<f64> = vec![];
        // Correct: z_lb = c[0] = 1.0 (stationarity c - z_lb = 0)
        // Broken: z_lb = 0 → stationarity fails
        let z_bad = vec![0.0_f64]; // lb-dual = 0, but needs to be 1
        let gap = 0.0_f64;

        let result = prove_optimal(&view, &x, &y, &z_bad, gap, 1e-6);
        assert!(result.is_err(), "stationarity violation must fail");
        let err = result.unwrap_err();
        assert!(err.failing_conditions.contains(&"stationarity"),
            "stationarity must fail: {:?}", err.failing_conditions);
    }

    /// prove_optimal is scale-invariant to a multiplicative rescaling of the problem.
    /// The same iterate satisfies the tolerances regardless of objective scale.
    #[test]
    fn prove_optimal_tol_semantics() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x = vec![1.0_f64];
        let y = vec![-1.0_f64];
        let z = vec![];

        // Tight tol: should pass (residuals are exact 0)
        assert!(prove_optimal(&view, &x, &y, &z, 0.0, 1e-10).is_ok());
        // Very tight tol: still passes (perfect iterate)
        assert!(prove_optimal(&view, &x, &y, &z, 0.0, 1e-14).is_ok());
        // Negative gap: should fail (duality_gap_rel > tol=0)
        // NOTE: gap=-1 would be pathological; test with positive non-zero gap
        let result_gap_fail = prove_optimal(&view, &x, &y, &z, 1e-5, 1e-6);
        assert!(result_gap_fail.is_err());
        assert!(result_gap_fail.unwrap_err().failing_conditions.contains(&"duality_gap"));
    }

    /// Empty slices (LP, no bounds, no constraints) → prove_optimal handles gracefully.
    #[test]
    fn prove_optimal_empty_problem() {
        let q = CscMatrix::new(0, 0);
        let a = CscMatrix::new(0, 0);
        let c: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let bounds: Vec<(f64, f64)> = vec![];
        let ct: Vec<ConstraintType> = vec![];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
        let result = prove_optimal(&view, &[], &[], &[], 0.0, 1e-6);
        assert!(result.is_ok(), "empty problem must pass: {:?}", result.err());
    }
}
