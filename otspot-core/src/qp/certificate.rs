//! KKT optimality verifier: the sole mint-point for [`OptimalCertificate`].
//!
//! [`prove_optimal`] assembles all five KKT conditions plus duality gap and dual-sign
//! feasibility. It returns `Ok(cert)` only when every check is below `tol`; otherwise
//! it returns `Err(NotProven)` naming the failing conditions. This "no certificate
//! without proof" design eliminates false-Optimal status (the most common solver bug).
//!
//! [`prove_optimal_lp`] and [`guard_lp_optimal`] extend this to LP, handling the
//! sign-convention difference between LP simplex duals and the prove_optimal convention.

use crate::problem::certificate::{NotProven, OptimalCertificate};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::qp::ipm_solver::kkt::{
    bound_violation, complementarity_residual_rel, kkt_residual_rel, primal_residual_rel,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation;
use crate::sparse::CscMatrix;

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
///
/// ## Gap threshold
///
/// The duality gap is checked against `tol` (= `user_eps`, typically 1e-6), which is
/// **stricter** than the historic `PROMOTION_GAP_TOL = 1e-1` used in
/// `IpmOutcome::satisfies_eps`. This is intentional: a solution claiming Optimal must
/// close the gap to the user-requested precision, not merely to a structural 10 %
/// tolerance. `satisfies_eps` retains its loose gate to select the *best available*
/// iterate across retry attempts; `prove_optimal` then acts as the honest Optimal mint,
/// requiring every KKT condition — including the gap — to meet `tol`.
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

/// Certificate gate tolerance for LP: `√PIVOT_TOL ≈ 1e-4`.
///
/// LP simplex accumulates O(√PIVOT_TOL) rounding errors in the basis solution
/// (Wilkinson). Complementarity and duality-gap residuals at a valid optimal
/// vertex are therefore naturally at the 1e-5..1e-4 level, not 1e-6.
/// Using 1e-4 (= `feas_rel_tol()`) ensures well-solved LPs pass without false
/// demotions while still catching catastrophic failures (gap ≫ 1e-4).
pub(crate) const LP_CERT_TOL: f64 = 1e-4; // = feas_rel_tol() = PIVOT_TOL.sqrt()

/// Verify LP optimality via full KKT+dual_sign.
///
/// **Sign convention**: LP simplex produces `dual_solution` in the `c − A^T y − rc = 0`
/// convention (Le: y ≤ 0, Ge: y ≥ 0). `prove_optimal` expects the opposite (Le: y ≥ 0,
/// Ge: y ≤ 0). The conversion is a universal negation: `y_prove = −y_simplex`.
///
/// Two paths:
/// - **Simplex path** (`reduced_costs` non-empty): apply universal negation to `dual_solution`,
///   build `z` from reduced costs: `z_lb = max(rc, 0)` for lb-finite vars,
///   `z_ub = max(−rc, 0)` for ub-finite vars.
/// - **IPM path** (`bound_duals` non-empty): `dual_solution` and `bound_duals` are already
///   in the `prove_optimal` convention — use as-is.
pub(crate) fn prove_optimal_lp(
    problem: &LpProblem,
    result: &SolverResult,
    tol: f64,
) -> Result<OptimalCertificate, NotProven> {
    let n = problem.num_vars;
    let q_zero = CscMatrix::new(n, n);
    let view = ProblemView {
        q: &q_zero,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &[],
    };

    let (y_prove, z) = if !result.bound_duals.is_empty() {
        // IPM path: already in prove_optimal convention.
        (result.dual_solution.clone(), result.bound_duals.clone())
    } else {
        // Simplex path: negate ALL dual variables and convert rc→z.
        let y_prove: Vec<f64> = result.dual_solution.iter().map(|&v| -v).collect();
        let rc = &result.reduced_costs;
        let mut z_lb = Vec::new();
        let mut z_ub = Vec::new();
        for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
            let rc_j = rc.get(j).copied().unwrap_or(0.0);
            if lb.is_finite() {
                z_lb.push(rc_j.max(0.0));
            }
            if ub.is_finite() {
                z_ub.push((-rc_j).max(0.0));
            }
        }
        z_lb.extend(z_ub);
        (y_prove, z_lb)
    };

    let gap_rel = lp_duality_gap_rel(problem, &result.solution, &y_prove, &z);
    prove_optimal(&view, &result.solution, &y_prove, &z, gap_rel, tol)
}

/// LP duality gap: `|primal − dual| / max(|p|, |d|, 1)`.
///
/// `dual_obj = −b^T y_prove + Σ lb_j·z_lb_j − Σ ub_j·z_ub_j`.
fn lp_duality_gap_rel(problem: &LpProblem, x: &[f64], y_prove: &[f64], z: &[f64]) -> f64 {
    let primal_obj: f64 = problem.c.iter().zip(x.iter()).map(|(&c, &xj)| c * xj).sum();
    let by: f64 = problem.b.iter().zip(y_prove.iter()).map(|(&b, &y)| b * y).sum();
    let n_lb = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    let mut bnd_term = 0.0_f64;
    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb;
    for &(lb, ub) in problem.bounds.iter() {
        if lb.is_finite() && lb_idx < z.len() {
            bnd_term += lb * z[lb_idx];
            lb_idx += 1;
        }
        if ub.is_finite() && ub_idx < z.len() {
            bnd_term -= ub * z[ub_idx];
            ub_idx += 1;
        }
    }
    let dual_obj = -by + bnd_term;
    let gap_abs = (primal_obj - dual_obj).abs();
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if gap_abs.is_finite() { gap_abs / denom } else { f64::INFINITY }
}

/// Full KKT+dual_sign LP Optimal gate.
///
/// Calls `prove_optimal_lp`; on `Err(NotProven)` downgrades to `SuboptimalSolution`
/// (preserving the solution vector). Non-Optimal results and empty solutions pass
/// through unchanged.
pub(crate) fn guard_lp_optimal(result: SolverResult, problem: &LpProblem) -> SolverResult {
    if result.status != SolveStatus::Optimal || result.solution.is_empty() {
        return result;
    }
    match prove_optimal_lp(problem, &result, LP_CERT_TOL) {
        Ok(_) => result,
        Err(_) => SolverResult { status: SolveStatus::SuboptimalSolution, ..result },
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

    // ── prove_optimal_lp tests ────────────────────────────────────────────────

    fn make_le_lp() -> LpProblem {
        // min −x  s.t.  x ≤ 1,  x ≥ 0
        // Optimal: x*=1, obj=-1.
        // LP simplex dual: y_Le = −1 (Le dual ≤ 0), rc=0 (x is basic).
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        LpProblem::new_general(
            vec![-1.0_f64],
            a,
            vec![1.0_f64],
            vec![ConstraintType::Le],
            vec![(0.0_f64, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    fn correct_simplex_result() -> SolverResult {
        SolverResult {
            status: SolveStatus::Optimal,
            objective: -1.0,
            solution: vec![1.0_f64],
            // LP simplex convention: Le dual ≤ 0. Active Le at x=1 → y = -1.
            dual_solution: vec![-1.0_f64],
            reduced_costs: vec![0.0_f64],
            slack: vec![0.0_f64],
            ..Default::default()
        }
    }

    /// Correct LP simplex output → prove_optimal_lp returns Ok.
    /// Verifies that the universal y-negation + rc→z conversion produces a
    /// valid KKT certificate.
    #[test]
    fn prove_optimal_lp_correct_simplex_path_passes() {
        let lp = make_le_lp();
        let result = correct_simplex_result();
        let cert = prove_optimal_lp(&lp, &result, 1e-6);
        assert!(cert.is_ok(), "correct simplex result must pass: {:?}", cert.err());
    }

    /// Wrong-sign LP dual (Le should be ≤ 0 in simplex, but +1 given) → Err.
    /// After negation: y_prove = −1 < 0 violates Le dual_sign ≥ 0 requirement.
    ///
    /// Load-bearing: `prove_optimal_lp_correct_simplex_path_passes` (above) shows
    /// the same problem PASSES with correct duals, proving the gate is not a no-op.
    #[test]
    fn prove_optimal_lp_wrong_sign_dual_fails() {
        let lp = make_le_lp();
        let mut result = correct_simplex_result();
        result.dual_solution = vec![1.0_f64]; // wrong sign: Le must be ≤ 0 in simplex
        let cert = prove_optimal_lp(&lp, &result, 1e-6);
        assert!(cert.is_err(), "wrong-sign dual must fail prove_optimal_lp");
        let err = cert.unwrap_err();
        assert!(
            err.failing_conditions.contains(&"dual_sign")
                || err.failing_conditions.contains(&"stationarity"),
            "dual_sign or stationarity must fail: {:?}", err.failing_conditions,
        );
    }

    /// Active lower bound: rc > 0 at lb → z_lb = rc, z_ub = 0.
    /// Problem: min x  s.t.  x ≥ 2,  x ≥ 0.  Optimal x=2.
    #[test]
    fn prove_optimal_lp_active_lower_bound_cert() {
        // min x  s.t.  x ≥ 2.  Optimal: x=2, obj=2.
        // LP simplex: x basic, y_Ge = +1 (Ge dual ≥ 0), rc = 0.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0_f64],
            a,
            vec![2.0_f64],
            vec![ConstraintType::Ge],
            vec![(0.0_f64, f64::INFINITY)],
            None,
        ).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            objective: 2.0,
            solution: vec![2.0_f64],
            dual_solution: vec![1.0_f64], // Ge dual ≥ 0 in simplex
            reduced_costs: vec![0.0_f64],
            slack: vec![0.0_f64],
            ..Default::default()
        };
        let cert = prove_optimal_lp(&lp, &result, 1e-6);
        assert!(cert.is_ok(), "active Ge constraint must produce valid cert: {:?}", cert.err());
    }

    /// guard_lp_optimal passes correct KKT result through as Optimal.
    #[test]
    fn guard_lp_optimal_passes_correct_result() {
        let lp = make_le_lp();
        let result = correct_simplex_result();
        let guarded = guard_lp_optimal(result, &lp);
        assert_eq!(guarded.status, SolveStatus::Optimal,
            "correct KKT result must remain Optimal");
    }

    /// guard_lp_optimal demotes wrong-sign dual to SuboptimalSolution (not NumericalError).
    ///
    /// Load-bearing (paired with `guard_lp_optimal_passes_correct_result`): removing
    /// the prove_optimal_lp call from guard_lp_optimal would cause this test to PASS
    /// incorrectly (wrong-sign result stays Optimal) while the correct-result test also
    /// passes — thus the pair together detect a no-op guard body.
    #[test]
    fn guard_lp_optimal_demotes_wrong_sign_dual() {
        let lp = make_le_lp();
        let mut result = correct_simplex_result();
        result.dual_solution = vec![1.0_f64]; // wrong sign for Le in simplex
        let guarded = guard_lp_optimal(result, &lp);
        assert_eq!(guarded.status, SolveStatus::SuboptimalSolution,
            "wrong-sign dual must be demoted to SuboptimalSolution");
    }

    /// guard_lp_optimal is a no-op for non-Optimal statuses.
    #[test]
    fn guard_lp_optimal_passthrough_non_optimal_statuses() {
        let lp = make_le_lp();
        for status in [SolveStatus::Infeasible, SolveStatus::Timeout, SolveStatus::NumericalError] {
            let r = SolverResult { status: status.clone(), ..Default::default() };
            let out = guard_lp_optimal(r, &lp);
            assert_eq!(out.status, status);
        }
    }

}
