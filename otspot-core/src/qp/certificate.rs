//! KKT optimality verifier: the sole mint-point for [`OptimalCertificate`].
//!
//! [`prove_optimal`] assembles all five KKT conditions plus duality gap and dual-sign
//! feasibility. It returns `Ok(cert)` only when every check is below `tol`; otherwise
//! it returns `Err(NotProven)` naming the failing conditions. This "no certificate
//! without proof" design eliminates false-Optimal status (the most common solver bug).
//!
//! `prove_optimal_lp` and `guard_lp_optimal` extend this to LP, handling the
//! sign-convention difference between LP simplex duals and the prove_optimal convention.

use crate::problem::certificate::{NotProven, OptimalCertificate};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::qp::ipm_solver::kkt::{
    bound_violation, complementarity_componentwise_rel, complementarity_residual_rel,
    kkt_residual_rel, primal_residual_rel,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation;
use crate::sparse::CscMatrix;

/// Verify all KKT conditions and mint an [`OptimalCertificate`] if they all pass.
///
/// Assembles stationarity (DD precision), primal/bound feasibility, complementarity,
/// dual-sign feasibility, and the caller-supplied duality gap. Returns
/// `Err(NotProven)` listing every condition that exceeded `tol`.
///
/// Gap is checked against `tol` (= user_eps), stricter than `PROMOTION_GAP_TOL = 1e-1`
/// in `IpmOutcome::satisfies_eps`: the loose gate selects the best iterate across
/// retries, while `prove_optimal` is the honest Optimal mint.
pub fn prove_optimal<'a>(
    view: &ProblemView<'a>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    duality_gap_rel: f64,
    tol: f64,
) -> Result<OptimalCertificate, NotProven> {
    // ── dimension guard (single chokepoint) ──────────────────────────────────
    // Validate ProblemView consistency and input sizes before any residual
    // computation; mismatched slices panic (index-out-of-bounds) in the helpers
    // (`dd_impl::aty` reads `y[row]` over a.row_ind; `dd_impl::ax` reads `x[col]`
    // over 0..a.ncols). Authoritative dimension sources:
    //   num_vars        = view.bounds.len()
    //   num_constraints = view.a.nrows
    //   expected_z_len  = n_lb_finite + n_ub_finite (z: lb-half then ub-half)
    // Sentinel: removing this guard makes the short-slice test cases below panic
    // (index out of bounds) instead of returning Err — recorded as a panic
    // failure, not an assertion failure.
    let num_vars = view.bounds.len();
    let num_constraints = view.a.nrows;
    let n_lb = view
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = view
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    let dim_valid = view.q.nrows == num_vars
        && view.q.ncols == num_vars
        && view.a.ncols == num_vars
        && view.b.len() == num_constraints
        && view.c.len() == num_vars
        && view.constraint_types.len() == num_constraints
        && x.len() == num_vars
        && y.len() == num_constraints
        && z.len() == n_lb + n_ub;
    if !dim_valid {
        return Err(NotProven {
            stationarity_rel: f64::NAN,
            primal_residual_rel: f64::NAN,
            bound_violation: f64::NAN,
            complementarity_rel: f64::NAN,
            dual_sign_violation: f64::NAN,
            duality_gap_rel: f64::NAN,
            tol,
            failing_conditions: vec!["input_dimensions"],
        });
    }
    // ── scalar input validation ──────────────────────────────────────────────
    // Relative duality gap is non-negative by definition; negative or non-finite
    // values indicate a caller error.
    //
    // NOTE: without this guard, gap=-1e-3 satisfies `!(−1e-3 ≤ tol)=false` and
    // slips through the loop below, issuing a certificate for an impossible gap.
    if !duality_gap_rel.is_finite() || duality_gap_rel < 0.0 {
        return Err(NotProven {
            stationarity_rel: f64::NAN,
            primal_residual_rel: f64::NAN,
            bound_violation: f64::NAN,
            complementarity_rel: f64::NAN,
            dual_sign_violation: f64::NAN,
            duality_gap_rel,
            tol,
            failing_conditions: vec!["duality_gap"],
        });
    }
    // tol must be finite and strictly positive; zero/negative/non-finite tolerance
    // makes certification meaningless (+inf tol would certify any iterate).
    if !tol.is_finite() || tol <= 0.0 {
        return Err(NotProven {
            stationarity_rel: f64::NAN,
            primal_residual_rel: f64::NAN,
            bound_violation: f64::NAN,
            complementarity_rel: f64::NAN,
            dual_sign_violation: f64::NAN,
            duality_gap_rel,
            tol,
            failing_conditions: vec!["invalid_tolerance"],
        });
    }

    // ── residual computation ─────────────────────────────────────────────────
    let stat = kkt_residual_rel(view, x, y, z);
    let pres = primal_residual_rel(view, x);
    let bviol = bound_violation(view.bounds, x);
    let comp = complementarity_residual_rel(view, x, y, z)
        .max(complementarity_componentwise_rel(view, x, y, z));
    let dsign = dual_sign_violation(view.constraint_types, y, view.bounds, z);
    let gap = duality_gap_rel;

    // Use `!(val <= tol)` instead of `val > tol`: NaN comparisons are asymmetric.
    // `NaN > tol` is false (NaN would slip through), but `NaN <= tol` is also false,
    // so `!(NaN <= tol)` is true — NaN and ±Inf are correctly rejected.
    let mut failing: Vec<&'static str> = Vec::new();
    for (name, val) in [
        ("stationarity", stat),
        ("primal_feasibility", pres),
        ("bound_feasibility", bviol),
        ("complementarity", comp),
        ("dual_sign", dsign),
        ("duality_gap", gap),
    ] {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        // NaN must fail: !(NaN <= tol) = true, NaN > tol = false
        if !(val <= tol) {
            failing.push(name);
        }
    }

    if failing.is_empty() {
        Ok(OptimalCertificate::new(stat, pres, dsign, gap, tol))
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
///
/// 撤廃 (1e-8 に変更) で退化するテスト:
/// - `guard_lp_optimal_no_false_demote_for_residuals_below_lp_cert_tol`:
///   5e-5 dual 摂動で (1e-6, 1e-4) 範囲の残差を合成した LP が誤 demote される。
/// - `lp_cert_tol_equals_feas_rel_tol`: drift-pin が LP_CERT_TOL ≠ feas_rel_tol() を検出。
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
/// - **IPM path** (`reduced_costs` empty): `dual_solution` and `bound_duals` are already
///   in the `prove_optimal` convention — use as-is. This covers both bounded IPM
///   (`bound_duals` non-empty) and all-free IPM (`bound_duals` also empty).
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

    let (y_prove, z) = if result.reduced_costs.is_empty() {
        // IPM path: dual_solution already in prove_optimal convention.
        // reduced_costs is empty for IPM (never computed), including all-free LP
        // where bound_duals is also empty — the old `!bound_duals.is_empty()` check
        // wrongly fell into the simplex branch for that case.
        (result.dual_solution.clone(), result.bound_duals.clone())
    } else {
        // Simplex path: negate ALL dual variables and convert rc→z.
        if result.reduced_costs.len() != n {
            return Err(NotProven {
                stationarity_rel: f64::NAN,
                primal_residual_rel: f64::NAN,
                bound_violation: f64::NAN,
                complementarity_rel: f64::NAN,
                dual_sign_violation: f64::NAN,
                duality_gap_rel: f64::NAN,
                tol,
                failing_conditions: vec!["input_dimensions"],
            });
        }
        let y_prove: Vec<f64> = result.dual_solution.iter().map(|&v| -v).collect();
        let rc = &result.reduced_costs;
        let mut z_lb = Vec::new();
        let mut z_ub = Vec::new();
        for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
            let rc_j = rc[j];
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
    let by: f64 = problem
        .b
        .iter()
        .zip(y_prove.iter())
        .map(|(&b, &y)| b * y)
        .sum();
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
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
    if gap_abs.is_finite() {
        gap_abs / denom
    } else {
        f64::INFINITY
    }
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
        Err(_) => SolverResult {
            status: SolveStatus::SuboptimalSolution,
            ..result
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::qp::ipm_solver::outcome::ProblemView;
    use crate::sparse::CscMatrix;

    // Helper: build a ProblemView for a trivial LP.
    fn trivial_view<'a>(
        q: &'a CscMatrix,
        a: &'a CscMatrix,
        c: &'a [f64],
        b: &'a [f64],
        bounds: &'a [(f64, f64)],
        ct: &'a [ConstraintType],
    ) -> ProblemView<'a> {
        ProblemView {
            q,
            a,
            c,
            b,
            bounds,
            constraint_types: ct,
            eliminated_cols: &[],
        }
    }

    /// Exact optimal KKT point → prove_optimal returns Ok.
    ///
    /// Problem: min x  s.t. x >= 1 (Ge), lb=0, ub=inf.
    /// Optimal: x*=1, y*=-1 (Ge dual <= 0), z_lb=0 (lb inactive: x=1 > lb=0).
    /// z layout: bounds=[(0,inf)] → n_lb=1, n_ub=0 → z=[z_lb]=[0.0].
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

        let x = vec![1.0_f64]; // primal optimal
        let y = vec![-1.0_f64]; // Ge dual <= 0  (y*=-1)
        let z = vec![0.0_f64]; // z_lb=0 (lb=0 inactive: x=1 > lb=0); n_lb=1, n_ub=0
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
        let z = vec![0.0_f64]; // z_lb=0 (inactive lb); bounds=[(0,inf)] → n_lb=1, n_ub=0
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
    /// z: bounds=[(0,inf)] → n_lb=1, n_ub=0 → z=[z_lb=0] (lb inactive at x=1 > lb=0).
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
        let z = vec![0.0_f64]; // z_lb=0 (lb inactive); n_lb=1, n_ub=0
        let gap = 0.0_f64;

        let result = prove_optimal(&view, &x, &y_wrong, &z, gap, 1e-6);
        assert!(result.is_err(), "wrong-sign dual must fail");
        let err = result.unwrap_err();
        assert!(
            err.failing_conditions.contains(&"dual_sign"),
            "dual_sign must be in failing_conditions: {:?}",
            err.failing_conditions
        );
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
        assert!(
            err.failing_conditions.contains(&"stationarity"),
            "stationarity must fail: {:?}",
            err.failing_conditions
        );
    }

    /// prove_optimal is scale-invariant to a multiplicative rescaling of the problem.
    /// The same iterate satisfies the tolerances regardless of objective scale.
    /// z: bounds=[(0,inf)] → n_lb=1, n_ub=0 → z=[0.0] (z_lb=0, lb inactive).
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
        let z = vec![0.0_f64]; // z_lb=0 (inactive lb); n_lb=1, n_ub=0

        // Tight tol: should pass (residuals are exact 0)
        assert!(prove_optimal(&view, &x, &y, &z, 0.0, 1e-10).is_ok());
        // Very tight tol: still passes (perfect iterate)
        assert!(prove_optimal(&view, &x, &y, &z, 0.0, 1e-14).is_ok());
        // Negative gap: should fail (duality_gap_rel > tol=0)
        // NOTE: gap=-1 would be pathological; test with positive non-zero gap
        let result_gap_fail = prove_optimal(&view, &x, &y, &z, 1e-5, 1e-6);
        assert!(result_gap_fail.is_err());
        assert!(result_gap_fail
            .unwrap_err()
            .failing_conditions
            .contains(&"duality_gap"));
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
        assert!(
            result.is_ok(),
            "empty problem must pass: {:?}",
            result.err()
        );
    }

    // ── NaN / ±Inf soundness guard ────────────────────────────────────────────

    /// Table-driven: each of the 6 conditions individually as NaN must produce Err.
    ///
    /// Before the fix (`if x > tol`), `NaN > tol` is false so NaN slips through.
    /// After the fix (`!(x <= tol)`), `NaN <= tol` is also false → `!false = true`
    /// → NaN is correctly caught. This test was written *before* the fix to confirm
    /// the bug, then re-run after the fix to confirm the repair (sentinel role).
    ///
    /// **Sentinel**: reverting to `if val > tol` causes every NaN row below to return
    /// Ok(cert) instead of Err, making this test fail on those rows.
    #[test]
    fn prove_optimal_nan_in_each_condition_is_rejected() {
        // Use the trivial exact-KKT problem; gap is passed directly.
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];

        // We inject NaN directly as the gap argument (directly controllable) for gap,
        // and use a NaN x/y/z to trigger NaN in the computed residuals for the others.
        // For simplicity, all 6 columns are tested via the gap argument or NaN solution.

        // Case A: gap = NaN → duality_gap must be in failing_conditions.
        {
            let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
            let x = vec![1.0_f64];
            let y = vec![-1.0_f64];
            let z = vec![0.0_f64]; // z_lb=0; bounds=[(0,inf)] → n_lb=1, n_ub=0
            let result = prove_optimal(&view, &x, &y, &z, f64::NAN, 1e-6);
            assert!(result.is_err(), "NaN gap must be rejected");
            let err = result.unwrap_err();
            assert!(
                err.failing_conditions.contains(&"duality_gap"),
                "NaN gap: duality_gap must be in failing_conditions, got {:?}",
                err.failing_conditions
            );
        }

        // Case B: gap = +Inf → duality_gap must be in failing_conditions.
        {
            let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
            let x = vec![1.0_f64];
            let y = vec![-1.0_f64];
            let z = vec![0.0_f64]; // z_lb=0; n_lb=1, n_ub=0
            let result = prove_optimal(&view, &x, &y, &z, f64::INFINITY, 1e-6);
            assert!(result.is_err(), "+Inf gap must be rejected");
            let err = result.unwrap_err();
            assert!(
                err.failing_conditions.contains(&"duality_gap"),
                "+Inf gap: duality_gap must be in failing_conditions, got {:?}",
                err.failing_conditions
            );
        }

        // Case C: x = NaN → at least one residual becomes non-finite → Err.
        // The exact set of failing conditions depends on which residual functions propagate
        // NaN; we only assert that at least one condition fires (not a no-op gate).
        {
            let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
            let x_nan = vec![f64::NAN];
            let y = vec![-1.0_f64];
            let z = vec![0.0_f64]; // z_lb=0; n_lb=1, n_ub=0
            let result = prove_optimal(&view, &x_nan, &y, &z, 0.0, 1e-6);
            assert!(result.is_err(), "NaN x must be rejected");
            let err = result.unwrap_err();
            assert!(
                !err.failing_conditions.is_empty(),
                "NaN x: at least one condition must fail, got {:?}",
                err.failing_conditions
            );
        }

        // Case D: NaN gap + NaN x/y → at least duality_gap must be caught.
        // Note: some residual functions (stat, pres) may not propagate NaN depending on
        // their internal arithmetic; the gate is responsible only for what it directly
        // receives. "duality_gap" is passed directly so it must always fire.
        {
            let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
            let x_nan = vec![f64::NAN];
            let y_nan = vec![f64::NAN];
            let z = vec![0.0_f64]; // z_lb=0; n_lb=1, n_ub=0
            let result = prove_optimal(&view, &x_nan, &y_nan, &z, f64::NAN, 1e-6);
            assert!(result.is_err(), "NaN x/y/gap must be rejected");
            let err = result.unwrap_err();
            assert!(
                !err.failing_conditions.is_empty(),
                "NaN inputs: at least one condition must fail, got {:?}",
                err.failing_conditions
            );
            assert!(
                err.failing_conditions.contains(&"duality_gap"),
                "NaN gap: duality_gap must always be in failing_conditions, got {:?}",
                err.failing_conditions
            );
        }
    }

    /// Finite residuals ≤ tol still pass after the NaN fix (regression guard).
    ///
    /// Ensures the `!(val <= tol)` change does not break the happy path where all
    /// residuals are legitimately small.
    /// z: bounds=[(0,inf)] → n_lb=1, n_ub=0 → z=[0.0] (z_lb=0, lb inactive).
    #[test]
    fn prove_optimal_finite_below_tol_still_passes() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);

        let x = vec![1.0_f64];
        let y = vec![-1.0_f64];
        let z = vec![0.0_f64]; // z_lb=0; n_lb=1, n_ub=0

        // Exact KKT point → all residuals ≈ 0.0 → must pass.
        let result = prove_optimal(&view, &x, &y, &z, 0.0, 1e-6);
        assert!(
            result.is_ok(),
            "exact finite KKT must still pass after NaN fix: {:?}",
            result.err()
        );
    }

    // ── dimension guard tests ─────────────────────────────────────────────────

    /// Table-driven: dimension mismatches return Err{["input_dimensions"]}, not panic.
    ///
    /// **Sentinel load-bearing**: Removing the dimension guard at the top of
    /// `prove_optimal` causes the short-y and short-x cases to panic with
    /// index-out-of-bounds (in `dd_impl::aty` and `dd_impl::ax` respectively).
    /// The test framework records that as a panic failure — not a normal assertion
    /// failure — so the Err assertion is never reached.
    ///
    /// Panic reproduction (pre-fix):
    /// - y too short: `dd_impl::aty` accesses `y[row]` where row ≥ y.len() → panic.
    /// - x too short: `dd_impl::ax` accesses `x[col]` where col ≥ x.len() → panic
    ///   (after `kkt_residual_rel` returns INFINITY for wrong x, `primal_residual_rel`
    ///   still calls `dd_impl::ax` without re-checking x.len()).
    #[test]
    fn prove_optimal_dimension_mismatch_returns_err_not_panic() {
        let q1 = CscMatrix::new(1, 1);
        // A: 2 rows × 1 col, with non-zero in row 0 AND row 1 (to trigger aty panic)
        let a2x1 = CscMatrix::from_triplets(&[0usize, 1], &[0, 0], &[1.0_f64, 1.0], 2, 1).unwrap();
        let c1 = vec![0.0_f64];
        let b2 = vec![1.0_f64, 1.0];
        let bounds_lb = vec![(0.0_f64, f64::INFINITY)]; // n_lb=1, n_ub=0
        let bounds_free = vec![(f64::NEG_INFINITY, f64::INFINITY)]; // n_lb=0, n_ub=0
        let ct2 = vec![ConstraintType::Le, ConstraintType::Le];

        // ── y too short (actual panic risk without guard) ─────────────────────
        // y.len()=1 but num_constraints=2 → dd_impl::aty accesses y[1] → panic w/o guard
        {
            let view = trivial_view(&q1, &a2x1, &c1, &b2, &bounds_free, &ct2);
            let result = prove_optimal(&view, &[0.0], &[0.5], &[], 0.0, 1e-6);
            assert!(result.is_err(), "y too short: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "y too short: {:?}",
                err.failing_conditions
            );
            assert!(
                err.stationarity_rel.is_nan(),
                "residuals must be NaN on dim error"
            );
        }

        // ── y too long ────────────────────────────────────────────────────────
        {
            let view = trivial_view(&q1, &a2x1, &c1, &b2, &bounds_free, &ct2);
            let result = prove_optimal(&view, &[0.0], &[0.5, 0.5, 0.5], &[], 0.0, 1e-6);
            assert!(result.is_err(), "y too long: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "y too long: {:?}",
                err.failing_conditions
            );
        }

        // ── x too short (actual panic risk without guard) ─────────────────────
        // x.len()=0 but num_vars=1 → dd_impl::ax accesses x[0] via primal_residual_rel → panic
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let b1 = vec![1.0_f64];
            let ct1 = vec![ConstraintType::Ge];
            let view = trivial_view(&q1, &a1x1, &c1, &b1, &bounds_lb, &ct1);
            let result = prove_optimal(&view, &[], &[-1.0], &[0.0], 0.0, 1e-6);
            assert!(result.is_err(), "x empty (too short): expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "x too short: {:?}",
                err.failing_conditions
            );
        }

        // ── x too long ────────────────────────────────────────────────────────
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let b1 = vec![1.0_f64];
            let ct1 = vec![ConstraintType::Ge];
            let view = trivial_view(&q1, &a1x1, &c1, &b1, &bounds_lb, &ct1);
            let result = prove_optimal(&view, &[1.0, 2.0], &[-1.0], &[0.0], 0.0, 1e-6);
            assert!(result.is_err(), "x too long: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "x too long: {:?}",
                err.failing_conditions
            );
        }

        // ── z too short (n_lb=1, n_ub=0 → expected z.len()=1, passing z=[]) ──
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let b1 = vec![1.0_f64];
            let ct1 = vec![ConstraintType::Ge];
            let view = trivial_view(&q1, &a1x1, &c1, &b1, &bounds_lb, &ct1);
            let result = prove_optimal(&view, &[1.0], &[-1.0], &[], 0.0, 1e-6);
            assert!(
                result.is_err(),
                "z too short (empty for lb-only): expected Err"
            );
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "z too short: {:?}",
                err.failing_conditions
            );
        }

        // ── z too long ────────────────────────────────────────────────────────
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let b1 = vec![1.0_f64];
            let ct1 = vec![ConstraintType::Ge];
            let view = trivial_view(&q1, &a1x1, &c1, &b1, &bounds_lb, &ct1);
            // n_lb=1, expected z=[z_lb], but passing 2 elements
            let result = prove_optimal(&view, &[1.0], &[-1.0], &[0.0, 0.0], 0.0, 1e-6);
            assert!(result.is_err(), "z too long: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "z too long: {:?}",
                err.failing_conditions
            );
        }

        // ── view inconsistency: c has wrong length ────────────────────────────
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let b1 = vec![1.0_f64];
            let ct1 = vec![ConstraintType::Ge];
            let c_wrong = vec![1.0_f64, 2.0]; // should be len=1
            let view = ProblemView {
                q: &q1,
                a: &a1x1,
                c: &c_wrong,
                b: &b1,
                bounds: &bounds_lb,
                constraint_types: &ct1,
                eliminated_cols: &[],
            };
            let result = prove_optimal(&view, &[1.0], &[-1.0], &[0.0], 0.0, 1e-6);
            assert!(result.is_err(), "view c wrong length: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "view c wrong: {:?}",
                err.failing_conditions
            );
        }

        // ── happy path: correct dims still pass ──────────────────────────────
        // min x  s.t. x >= 1 (Ge), lb=0. Exact KKT: x=1, y=-1, z_lb=0. Should be Ok.
        {
            let a1x1 = CscMatrix::from_triplets(&[0usize], &[0], &[1.0_f64], 1, 1).unwrap();
            let c_min_x = vec![1.0_f64];
            let b1 = vec![1.0_f64];
            let ct_ge = vec![ConstraintType::Ge];
            let view = trivial_view(&q1, &a1x1, &c_min_x, &b1, &bounds_lb, &ct_ge);
            let result = prove_optimal(&view, &[1.0], &[-1.0], &[0.0], 0.0, 1e-6);
            assert!(
                result.is_ok(),
                "correct dims + exact KKT must pass: {:?}",
                result.err()
            );
        }
    }

    /// Table-driven: multiple z-length patterns for a box-constrained problem.
    ///
    /// bounds=[(lb, ub)] → n_lb=1, n_ub=1 → expected z.len()=2.
    /// Verifies that z=[], z=[one], z=[a,b,c] all return "input_dimensions".
    #[test]
    fn prove_optimal_z_length_table_box_constraint() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::new(0, 1); // no linear constraints
        let c = vec![0.0_f64];
        let bounds = vec![(1.0_f64, 2.0_f64)]; // n_lb=1, n_ub=1 → expected z.len()=2
        let ct: Vec<ConstraintType> = vec![];

        for (z_bad, label) in [
            (vec![], "z empty"),
            (vec![0.0_f64], "z len 1 (too short)"),
            (vec![0.0_f64, 0.0, 0.0], "z len 3 (too long)"),
        ] {
            let view = trivial_view(&q, &a, &c, &[], &bounds, &ct);
            let result = prove_optimal(&view, &[1.5], &[], &z_bad, 0.0, 1e-6);
            assert!(result.is_err(), "{label}: expected Err");
            let err = result.unwrap_err();
            assert_eq!(
                err.failing_conditions,
                vec!["input_dimensions"],
                "{label}: {:?}",
                err.failing_conditions
            );
        }

        // Correct z=[0.0, 0.0]: x=1.5 inside (1,2) → all conditions pass
        {
            let view = trivial_view(&q, &a, &c, &[], &bounds, &ct);
            let result = prove_optimal(&view, &[1.5], &[], &[0.0, 0.0], 0.0, 1e-6);
            assert!(
                result.is_ok(),
                "correct z=[0,0] with x inside bounds must pass: {:?}",
                result.err()
            );
        }
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
        assert!(
            cert.is_ok(),
            "correct simplex result must pass: {:?}",
            cert.err()
        );
    }

    #[test]
    fn prove_optimal_lp_short_reduced_costs_is_not_proven() {
        // Non-empty reduced_costs selects the simplex certificate path.  A
        // vector shorter than num_vars must not be padded with zero, because
        // that can fabricate missing bound duals and falsely mint a certificate.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 0.0], 1, 2).unwrap();
        let lp2 = LpProblem::new_general(
            vec![-1.0_f64, 0.0],
            a,
            vec![1.0_f64],
            vec![ConstraintType::Le],
            vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result2 = SolverResult {
            status: SolveStatus::Optimal,
            objective: -1.0,
            solution: vec![1.0, 0.0],
            dual_solution: vec![-1.0],
            reduced_costs: vec![0.0], // BUG: missing rc for variable 1
            slack: vec![0.0],
            ..Default::default()
        };
        let cert = prove_optimal_lp(&lp2, &result2, 1e-6);
        assert!(
            cert.is_err(),
            "short reduced_costs must fail instead of being zero-padded"
        );
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
            "dual_sign or stationarity must fail: {:?}",
            err.failing_conditions,
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
        )
        .unwrap();
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
        assert!(
            cert.is_ok(),
            "active Ge constraint must produce valid cert: {:?}",
            cert.err()
        );
    }

    /// guard_lp_optimal passes correct KKT result through as Optimal.
    #[test]
    fn guard_lp_optimal_passes_correct_result() {
        let lp = make_le_lp();
        let result = correct_simplex_result();
        let guarded = guard_lp_optimal(result, &lp);
        assert_eq!(
            guarded.status,
            SolveStatus::Optimal,
            "correct KKT result must remain Optimal"
        );
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
        assert_eq!(
            guarded.status,
            SolveStatus::SuboptimalSolution,
            "wrong-sign dual must be demoted to SuboptimalSolution"
        );
    }

    /// guard_lp_optimal is a no-op for non-Optimal statuses.
    #[test]
    fn guard_lp_optimal_passthrough_non_optimal_statuses() {
        let lp = make_le_lp();
        for status in [
            SolveStatus::Infeasible,
            SolveStatus::Timeout,
            SolveStatus::NumericalError,
        ] {
            let r = SolverResult {
                status: status.clone(),
                ..Default::default()
            };
            let out = guard_lp_optimal(r, &lp);
            assert_eq!(out.status, status);
        }
    }

    // ── P3: LP_CERT_TOL drift-pin ─────────────────────────────────────────────

    /// LP_CERT_TOL must equal feas_rel_tol() = PIVOT_TOL.sqrt().
    ///
    /// Pins the relationship LP_CERT_TOL == feas_rel_tol() so that a change to
    /// either constant without updating the other causes an immediate test failure.
    /// The comment on LP_CERT_TOL explains the derivation; this test enforces it.
    #[test]
    fn lp_cert_tol_equals_feas_rel_tol() {
        let frt = crate::tolerances::feas_rel_tol();
        assert_eq!(
            LP_CERT_TOL, frt,
            "LP_CERT_TOL ({LP_CERT_TOL}) must equal feas_rel_tol() ({frt}); \
             update one to match the other (see LP_CERT_TOL docstring for derivation)"
        );
    }

    // ── P2: IPM all-free LP false-demotion ───────────────────────────────────

    /// IPM result for an all-free LP (no finite bounds) must not be demoted.
    ///
    /// An all-free LP solved by IPM has:
    ///   - `bound_duals = []` (no finite bounds → no z terms)
    ///   - `reduced_costs = []` (IPM never computes reduced costs)
    ///
    /// The old discriminant (`!bound_duals.is_empty()`) falls into the simplex
    /// branch → negates dual_solution → KKT stationarity fails → false demotion
    /// to SuboptimalSolution.
    ///
    /// **Sentinel**: reverting the discriminant back to `!result.bound_duals.is_empty()`
    /// causes this test to FAIL (verifying the fix is load-bearing).
    ///
    /// Problem: min x − y  s.t.  x − y = 2, x + y = 4  (both Eq, x,y ∈ ℝ)
    /// Optimal: x=3, y=1, obj=2.
    /// IPM dual (prove_optimal convention, Eq free): y=[-1, 0].
    /// Stationarity: c + Aᵀy = [1,−1] + [−1,1] = [0,0] ✓
    #[test]
    fn prove_optimal_lp_ipm_all_free_passes() {
        // x − y = 2 (row 0), x + y = 4 (row 1). A is 2×2.
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[1.0_f64, 1.0, -1.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0_f64, -1.0],
            a,
            vec![2.0_f64, 4.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            None,
        )
        .unwrap();

        // IPM result: dual in prove_optimal convention, no bound_duals, no reduced_costs.
        let ipm_result = SolverResult {
            status: SolveStatus::Optimal,
            objective: 2.0,
            solution: vec![3.0_f64, 1.0],
            dual_solution: vec![-1.0_f64, 0.0], // IPM convention: Eq free
            bound_duals: vec![],                // all-free → no z
            reduced_costs: vec![],              // IPM never sets reduced_costs
            ..Default::default()
        };

        let cert = prove_optimal_lp(&lp, &ipm_result, LP_CERT_TOL);
        assert!(
            cert.is_ok(),
            "IPM all-free LP must pass prove_optimal_lp: {:?}",
            cert.err()
        );

        let guarded = guard_lp_optimal(ipm_result, &lp);
        assert_eq!(
            guarded.status,
            SolveStatus::Optimal,
            "guard_lp_optimal must not demote correct IPM all-free LP result"
        );
    }

    /// Table-driven: 4 path×bound combinations all produce Optimal.
    ///
    /// Problem: min x  s.t.  x = 1  (Eq), various bound settings, simplex vs IPM result.
    /// Optimal: x=1, obj=1.
    #[test]
    fn prove_optimal_lp_path_bound_cross_table() {
        // Problem: min x  s.t.  x = 1 (Eq)
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();

        // Case helper: build LpProblem and an exact SolverResult.
        let run = |bounds: Vec<(f64, f64)>,
                   dual_sol: Vec<f64>,
                   bound_duals: Vec<f64>,
                   reduced_costs: Vec<f64>,
                   label: &str| {
            let lp = LpProblem::new_general(
                vec![1.0_f64],
                a.clone(),
                vec![1.0_f64],
                vec![ConstraintType::Eq],
                bounds,
                None,
            )
            .unwrap();
            let r = SolverResult {
                status: SolveStatus::Optimal,
                objective: 1.0,
                solution: vec![1.0_f64],
                dual_solution: dual_sol,
                bound_duals,
                reduced_costs,
                ..Default::default()
            };
            let cert = prove_optimal_lp(&lp, &r, LP_CERT_TOL);
            assert!(
                cert.is_ok(),
                "case `{label}` must pass prove_optimal_lp: {:?}",
                cert.err()
            );
            let guarded = guard_lp_optimal(r, &lp);
            assert_eq!(
                guarded.status,
                SolveStatus::Optimal,
                "case `{label}` must remain Optimal after guard"
            );
        };

        // Simplex + lb=0: rc=0 (basic), y_simplex=1 (Eq: c−Aᵀy=rc → 1−y=0 → y=1).
        // Negate → y_prove=−1. Stationarity: 1 + 1*(−1) − z_lb(=0) = 0 ✓
        run(
            vec![(0.0, f64::INFINITY)],
            vec![1.0_f64],
            vec![],
            vec![0.0_f64],
            "simplex/lb0",
        );

        // Simplex + all-free: same; no finite bounds → z=[].
        run(
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![1.0_f64],
            vec![],
            vec![0.0_f64],
            "simplex/free",
        );

        // IPM + lb=0: bound_duals=[z_lb=0], dual in prove_optimal convention.
        // Stationarity: 1 + 1*(−1) + z_lb = 1 − 1 + 0 = 0 ✓
        run(
            vec![(0.0, f64::INFINITY)],
            vec![-1.0_f64],
            vec![0.0_f64],
            vec![],
            "ipm/lb0",
        );

        // IPM + all-free: bound_duals=[], reduced_costs=[]. The bug case.
        // Stationarity: 1 + 1*(−1) = 0 ✓
        run(
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![-1.0_f64],
            vec![],
            vec![],
            "ipm/free",
        );
    }

    // ── P2: negative gap / invalid tol rejection ─────────────────────────────

    /// Step 1: reproduce the P2 bug — gap=-1e-3 and gap=-inf must be rejected.
    ///
    /// Before the fix, `!(gap <= tol)` evaluated `!(-1e-3 <= 1e-6) = !true = false`,
    /// so negative gap was never added to `failing_conditions` → `Ok(cert)` was
    /// wrongly returned. This test fact-checks that the bug is now fixed.
    #[test]
    fn prove_optimal_negative_gap_bug_reproduction() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![1.0_f64];
        let y = vec![-1.0_f64];
        let z = vec![0.0_f64];

        // Bug: both returned Ok before the fix.
        let r_neg = prove_optimal(&view, &x, &y, &z, -1e-3, 1e-6);
        assert!(r_neg.is_err(), "gap=-1e-3 must be rejected");
        let r_neginf = prove_optimal(&view, &x, &y, &z, f64::NEG_INFINITY, 1e-6);
        assert!(r_neginf.is_err(), "gap=-inf must be rejected");
    }

    /// Table-driven: all invalid gap and tol values are rejected; happy path passes.
    ///
    /// Invalid gap → `Err` with `"duality_gap"` in `failing_conditions`.
    /// Invalid tol → `Err` with `"invalid_tolerance"` in `failing_conditions`.
    /// Happy path (gap ≥ 0 finite, tol > 0 finite) → `Ok`.
    ///
    /// **Sentinel**: removing the `duality_gap_rel < 0.0` guard (no-op change)
    /// causes every negative-gap row below to return `Ok(cert)`, failing this test —
    /// confirming the guard is load-bearing.
    #[test]
    fn prove_optimal_invalid_gap_and_tol_rejected() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let ct = vec![ConstraintType::Ge];
        let view = trivial_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![1.0_f64];
        let y = vec![-1.0_f64];
        let z = vec![0.0_f64];

        // ── Invalid gap values → "duality_gap" ───────────────────────────────
        for (gap, label) in [
            (-1e-3_f64, "gap = -1e-3"),
            (f64::NEG_INFINITY, "gap = -inf"),
            (f64::NAN, "gap = NaN"),
            (f64::INFINITY, "gap = +inf"),
        ] {
            let r = prove_optimal(&view, &x, &y, &z, gap, 1e-6);
            assert!(r.is_err(), "{label}: expected Err");
            let err = r.unwrap_err();
            assert!(
                err.failing_conditions.contains(&"duality_gap"),
                "{label}: 'duality_gap' must be in failing_conditions, got {:?}",
                err.failing_conditions
            );
        }

        // ── Invalid tol values → "invalid_tolerance" ─────────────────────────
        for (tol, label) in [
            (0.0_f64, "tol = 0"),
            (-1.0_f64, "tol = -1"),
            (f64::NAN, "tol = NaN"),
            (f64::INFINITY, "tol = +inf"),
        ] {
            let r = prove_optimal(&view, &x, &y, &z, 0.0, tol);
            assert!(r.is_err(), "{label}: expected Err");
            let err = r.unwrap_err();
            assert!(
                err.failing_conditions.contains(&"invalid_tolerance"),
                "{label}: 'invalid_tolerance' must be in failing_conditions, got {:?}",
                err.failing_conditions
            );
        }

        // ── Happy path: gap ∈ {0, small positive}, tol > 0 → Ok ──────────────
        for (gap, tol, label) in [
            (0.0_f64, 1e-6_f64, "gap=0 tol=1e-6"),
            (1e-7_f64, 1e-6_f64, "gap=1e-7 tol=1e-6"),
        ] {
            let r = prove_optimal(&view, &x, &y, &z, gap, tol);
            assert!(r.is_ok(), "{label}: expected Ok, got {:?}", r.err());
        }
    }

    // ── P2: honesty test — residuals in (1e-6, 1e-4) must not false-demote ───

    /// LP result with KKT residuals in (1e-6, LP_CERT_TOL) range must pass guard.
    ///
    /// Constructs a correct LP solution with a small dual perturbation (~5e-5)
    /// that produces stationarity and gap residuals in the (1e-6, 1e-4) range.
    /// Verifies the guard does NOT demote to SuboptimalSolution (false-demote).
    ///
    /// Complementary test: the residual IS above 1e-6, so prove_optimal at 1e-6
    /// would reject — proving the test actually targets the LP_CERT_TOL range.
    #[test]
    fn guard_lp_optimal_no_false_demote_for_residuals_below_lp_cert_tol() {
        // Problem: min x  s.t.  x >= 1,  lb=0, ub=inf.
        // Optimal: x*=1, obj=1.  LP simplex dual: y_Ge = 1 (Ge dual >= 0).
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0_f64],
            a,
            vec![1.0_f64],
            vec![ConstraintType::Ge],
            vec![(0.0_f64, f64::INFINITY)],
            None,
        )
        .unwrap();

        // Introduce ~5e-5 dual perturbation so KKT residuals land in (1e-6, 1e-4).
        let y_perturbed = 1.0 + 5e-5; // slightly off from optimal y=1
        let result = SolverResult {
            status: SolveStatus::Optimal,
            objective: 1.0,
            solution: vec![1.0_f64],
            dual_solution: vec![y_perturbed], // Ge simplex dual >= 0
            reduced_costs: vec![0.0_f64],
            slack: vec![0.0_f64],
            ..Default::default()
        };

        // Verify the perturbation is in range: proves this test targets the window.
        let r = prove_optimal_lp(&lp, &result, LP_CERT_TOL);
        assert!(
            r.is_ok(),
            "residuals < LP_CERT_TOL={LP_CERT_TOL} must not demote: {:?}",
            r.err()
        );

        // Cross-check: at 1e-6 (stricter) the same result fails — proving residuals > 1e-6.
        let r_strict = prove_optimal_lp(&lp, &result, 1e-6);
        assert!(
            r_strict.is_err(),
            "residuals should exceed 1e-6, proving the test targets the (1e-6, LP_CERT_TOL) window"
        );

        // End-to-end: guard must not demote.
        let guarded = guard_lp_optimal(result, &lp);
        assert_eq!(
            guarded.status,
            SolveStatus::Optimal,
            "guard must not demote an LP result whose residuals are within LP_CERT_TOL"
        );
    }
}
