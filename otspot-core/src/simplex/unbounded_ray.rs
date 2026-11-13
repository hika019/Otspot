//! Verified Unbounded-ray exposure for the LP simplex path.
//!
//! The Phase II unboundedness gates (`dual_common::lp_unbounded_ray_witness`
//! call sites in `dual.rs`, `dual_advanced/pipeline.rs`, `primal/mod.rs`,
//! `dual_advanced/phase1.rs`) re-derive a recession direction from a clean LU
//! at the exit basis before trusting an `Unbounded` exit. Until now the
//! witness was reduced to a boolean; this module maps it back to ORIGINAL
//! variable space and re-verifies it there with the eps-free independent
//! verifier shared with the QP IPM path, so the payload reaches
//! [`SolverResult::ray_certificate`] with the same contract as IPM mints.
//!
//! Consumer note (QCQP Phase 1, plan §3.2): the conic relaxation forwards the
//! payload as `RelaxResult::ray`; recession verification against the ORIGINAL
//! QCQP is consumed in G2b, not here.

use crate::problem::{LpProblem, RayCertificate};
use crate::qp::ipm_core::common::{
    base_point_primal_feasible, verify_unbounded_ray, DetectionScales,
};
use crate::qp::ipm_core::kkt::build_extended_constraints;
use crate::qp::QpProblem;
use otspot_num::sparse::CscMatrix;

use super::dual_common::LpRayWitness;
use super::primal::extract_solution;
use super::standard_form::StandardForm;

/// Map a verified ray witness at the exit basis into ORIGINAL problem
/// coordinates and re-verify it there with the QP IPM's eps-free verifier at
/// `Q = 0` (`verify_unbounded_ray`, `qp/ipm_core/common.rs:267–343` — the
/// flatness/curvature conditions are trivial for a zero `Q`, the recession and
/// descent conditions apply as-is). `Some` payload ONLY when BOTH the ray
/// check and the base-point feasibility check
/// (`base_point_primal_feasible`, `common.rs:364–387`) pass; never a
/// fabricated ray.
///
/// Payload contract (original coordinates, extended-system convention of
/// `build_extended_constraints`, `qp/ipm_core/kkt.rs:39` — Ge rows
/// sign-flipped, finite bounds expanded into rows): `‖d‖∞ = 1` exactly
/// (normalized before verification so `MIN_DIR_NORM` cannot mask a small
/// direction); recession `A_eq·d ≈ 0` and `A_ineq·d ≤ ≈0` on every
/// ≤-normalized row including bound rows (fixed `d_j ≈ 0`, ub `d_j ≤ ≈0`,
/// lb `d_j ≥ ≈0`) so `base_point + t·d` stays feasible for all `t ≥ 0`; plus
/// strict descent `cᵀd < 0` (the LP objective is linear).
///
/// `None`: mapped direction zero/non-finite; either independent check
/// rejects; an unverified `Unbounded` never reaches this function
/// (`ray_certificate` stays `None`); and presolve-reduced propagations drop
/// the payload at `entry.rs:230–252` (reduced-space coordinates ≠ original
/// space, same rule the IPM payload follows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lp_ray_payload(
    problem: &LpProblem,
    sf: &StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    witness: &LpRayWitness,
) -> Option<RayCertificate> {
    // The witness entering column is verified `q < n_enter ≤ n_total`; the
    // guard only protects against a mis-threaded witness from another space.
    if witness.entering_col >= sf.n_total {
        return None;
    }
    // Iteration-space ray from the witness (`lp_unbounded_ray_witness`,
    // `dual_common.rs:420–517` at 474a352c): `dx[q] = +1`,
    // `dx[basis[i]] = −d_bar[i]`, zero elsewhere; the verifier already
    // confirmed `rc_q < −dual_tol`, `d_bar[i] ≤ ray_floor` on structural/
    // slack basic rows and `|d_bar[i]| ≤ ray_floor` on artificial rows.
    // Basis columns beyond `sf.n_total` are Big-M artificials: noise-scale in
    // the witness and belonging to no original column, so they are dropped.
    let mut d_iter = vec![0.0_f64; sf.n_total];
    d_iter[witness.entering_col] = 1.0;
    for (row, &col) in basis.iter().enumerate() {
        if col < sf.n_total {
            if let Some(&d_bar) = witness.d_bar.get(row) {
                d_iter[col] = -d_bar;
            }
        }
    }
    // Undo Ruiz column scaling, then the shift/split variable encoding
    // (`shift_and_split_variables`, `standard_form.rs:178–225`: shifted,
    // flipped and free-split columns `x_j = offset + Σ coeff·x'`) — exactly
    // the inverse `extract_solution` (`primal/reconcile.rs:550–583`) applies
    // to primal values. The base point is that extracted vertex from the
    // SAME basis snapshot, so direction and base share one basis.
    let mut direction = vec![0.0_f64; sf.n_orig];
    for (j, d_j) in direction.iter_mut().enumerate() {
        let info = &sf.orig_var_info[j];
        let mut acc = 0.0_f64;
        for &(new_col, coeff) in &info.new_vars {
            let scale = if col_scale.is_empty() {
                1.0
            } else {
                col_scale[new_col]
            };
            acc += coeff * scale * d_iter[new_col];
        }
        *d_j = acc;
    }
    let dir_norm = direction.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if !dir_norm.is_finite() || dir_norm <= 0.0 || !direction.iter().all(|&v| v.is_finite()) {
        return None;
    }
    for v in direction.iter_mut() {
        *v /= dir_norm;
    }
    let base_point = extract_solution(sf, basis, x_b, col_scale);

    // Independent verification in original space: the LP as a zero-Q QP
    // through the IPM's eps-free verifier (Q·dx = 0, zero curvature, so its
    // flatness/curvature conditions hold trivially and only recession,
    // descent and base feasibility can reject).
    let lp_as_qp = QpProblem {
        q: CscMatrix::new(problem.num_vars, problem.num_vars),
        c: problem.c.clone(),
        a: (*problem.a).clone(),
        b: problem.b.clone(),
        bounds: problem.bounds.clone(),
        num_vars: problem.num_vars,
        num_constraints: problem.num_constraints,
        constraint_types: problem.constraint_types.clone(),
        quadratic_constraints: vec![],
        obj_offset: 0.0,
    };
    let (a_ext, b_ext, _m_ext, _m_orig, _n_lb, is_eq_ext) = build_extended_constraints(&lp_as_qp);
    let scales = DetectionScales::compute(&lp_as_qp, &a_ext, &b_ext);
    let a_dx = a_ext.mat_vec_mul(&direction).ok()?;
    // Extended-system slack motion: `s(t) = b_ext − A_ext·x(t)` decreases by
    // `A_ext·d` per unit `t` on inequality rows; equality rows carry no slack
    // (`ds = 0` forces the verifier to judge `A_eq·d ≈ 0` directly).
    let ds: Vec<f64> = (0..a_ext.nrows())
        .map(|i| if is_eq_ext[i] { 0.0 } else { -a_dx[i] })
        .collect();
    if !verify_unbounded_ray(
        &direction, &ds, &lp_as_qp, &a_ext, &b_ext, &is_eq_ext, &scales,
    ) {
        return None;
    }
    if !base_point_primal_feasible(&base_point, &a_ext, &b_ext, &is_eq_ext, &scales) {
        return None;
    }
    Some(RayCertificate {
        direction,
        base_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::solve_lp_with;
    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, SolveStatus};

    fn unbounded_opts(presolve: bool) -> SolverOptions {
        SolverOptions {
            presolve,
            ..SolverOptions::default()
        }
    }

    /// Hand oracle: `min −x s.t. x ≥ 1` (x free) → ray d = +1 (Ge recession
    /// `d ≥ 0`), objective decrease `cᵀd = −1 < 0`, base at the vertex x = 1.
    /// Routes through the artificial-variable (primal two-phase) gate.
    #[test]
    fn ge_row_unbounded_exposes_certified_ray_at_lp_entry() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_with(&lp, &unbounded_opts(false));
        assert_eq!(res.status, SolveStatus::Unbounded);
        let ray = res.ray_certificate.expect(
            "verified Unbounded must carry a ray payload (sentinel: \
                     reverting the payload wiring to None fails this assert)",
        );
        assert_eq!(ray.direction.len(), 1);
        assert_eq!(ray.base_point.len(), 1);
        assert!(
            (ray.direction[0] - 1.0).abs() <= 1e-6,
            "direction must be +e_x, got {:?}",
            ray.direction
        );
        assert!(
            (ray.base_point[0] - 1.0).abs() <= 1e-6,
            "base point must be the vertex x = 1, got {:?}",
            ray.base_point
        );
        // Independent recomputation from the hand-written problem data only.
        assert!(ray.direction[0] > 0.0, "Ge recession: d ≥ 0");
        assert!(-ray.direction[0] < 0.0, "descent: cᵀd = −d < 0");
        assert!(ray.base_point[0] >= 1.0 - 1e-6, "base feasible: x ≥ 1");
    }

    /// Hand oracle: `min x s.t. x ≤ 1` (x free) → ray d = −1, `cᵀd = −1 < 0`.
    /// Routes through the Le-only cold-start gate. The base point is the
    /// CURRENT feasible vertex at the unboundedness exit (x = 0 here — the
    /// simplex never needs to advance to the bounding vertex, because
    /// `x(t) = 0 − t` already satisfies `x ≤ 1` for every `t ≥ 0`); the
    /// contract requires feasibility of the base, not a specific vertex.
    #[test]
    fn le_row_unbounded_exposes_certified_ray_at_lp_entry() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_with(&lp, &unbounded_opts(false));
        assert_eq!(res.status, SolveStatus::Unbounded);
        let ray = res
            .ray_certificate
            .expect("verified Unbounded must carry a ray payload");
        assert!(
            (ray.direction[0] + 1.0).abs() <= 1e-6,
            "direction must be −e_x (‖d‖∞ = 1, Le recession d ≤ 0), got {:?}",
            ray.direction
        );
        assert!(
            ray.base_point[0] <= 1.0 + 1e-6,
            "base point must be feasible for x ≤ 1, got {:?}",
            ray.base_point
        );
    }

    /// Hand oracle: `min −x₁ s.t. x₁ + x₂ = 3` (both free) → recession
    /// `d₁ + d₂ = 0` with descent `−d₁ < 0` → d = (+1, −1) normalized; base
    /// any feasible point of the row. Exercises the free-variable split in
    /// the original-space mapping (two standard-form columns per variable).
    #[test]
    fn eq_row_unbounded_maps_split_free_vars_to_original_space() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, 0.0],
            a,
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let res = solve_lp_with(&lp, &unbounded_opts(false));
        assert_eq!(res.status, SolveStatus::Unbounded);
        let ray = res
            .ray_certificate
            .expect("verified Unbounded must carry a ray payload");
        assert_eq!(ray.direction.len(), 2);
        let d1 = ray.direction[0];
        let d2 = ray.direction[1];
        assert!(
            (d1 - 1.0).abs() <= 1e-6 && (d2 + 1.0).abs() <= 1e-6,
            "direction must be (+1, −1), got {:?}",
            ray.direction
        );
        // Independent recomputation: equality recession d₁ + d₂ = 0 and base
        // feasibility on the row (no bounds to check).
        assert!((d1 + d2).abs() <= 1e-6, "A_eq·d = 0");
        assert!(-d1 < 0.0, "descent cᵀd = −d₁ < 0");
        assert!(
            (ray.base_point[0] + ray.base_point[1] - 3.0).abs() <= 1e-6,
            "base must satisfy x₁ + x₂ = 3, got {:?}",
            ray.base_point
        );
    }

    /// Presolve reduces (`x₀ = 5` singleton Eq) before the reduced LP goes
    /// Unbounded in the gated core; the reduced-space payload must be
    /// discarded at the entry pass-through, while the SAME LP solved without
    /// presolve carries a hand-oracled ray: strict descent `cᵀd < 0` and
    /// recession `d_i − d_j ≤ 0` on every difference row (e.g. `d = (0,1,1,1)`;
    /// the exact ray depends on pricing, so the test asserts the contract
    /// facts, which the hand-computed cone has). LP shape (per `entry.rs`'s
    /// reduced-Unbounded fixture): presolve cannot detect the unboundedness
    /// itself (x₁..x₃ appear in active rows, no bound propagation fires), so
    /// the verdict comes from the gated simplex, not from
    /// `PresolveStatus::Unbounded`.
    ///
    /// Sentinel pair: reverting the entry.rs reduced-Unbounded pass-through to
    /// forward the raw reduced payload would expose a length-3 direction on
    /// the 4-variable problem (coordinate mismatch).
    #[test]
    fn presolve_reduced_unbounded_discards_payload() {
        // min -x1-x2-x3 s.t. x0=5 (Eq singleton), x1-x2<=1, x2-x3<=1, x3-x1<=1,
        // all vars >= 0.
        let rows = [0usize, 1, 1, 2, 2, 3, 3];
        let cols = [0usize, 1, 2, 2, 3, 3, 1];
        let vals = [1.0f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 4, 4).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, -1.0, -1.0, -1.0],
            a,
            vec![5.0, 1.0, 1.0, 1.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Le,
                ConstraintType::Le,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY); 4],
            None,
        )
        .unwrap();
        let pr = crate::presolve::run_presolve(&lp, None)
            .expect("presolve must not decide Unbounded/Infeasible itself here");
        assert!(
            pr.was_reduced,
            "test premise: the singleton Eq row must reduce the problem"
        );

        let reduced = solve_lp_with(&lp, &unbounded_opts(true));
        assert_eq!(reduced.status, SolveStatus::Unbounded);
        assert!(
            reduced.ray_certificate.is_none(),
            "reduced-space payload must not leak into original space: {:?}",
            reduced.ray_certificate
        );

        let direct = solve_lp_with(&lp, &unbounded_opts(false));
        assert_eq!(direct.status, SolveStatus::Unbounded);
        let ray = direct
            .ray_certificate
            .expect("non-reduced verified Unbounded must carry the payload");
        assert_eq!(ray.direction.len(), 4);
        assert_eq!(ray.base_point.len(), 4);
        // Hand oracle for this exit vertex: x1 enters at 0, x2..x3 stay at
        // their bases — any valid ray here is a nonneg combination pattern of
        // the difference rows; verify the two contract facts independently
        // (recession on every row + strict descent) instead of pinning the
        // pricing-dependent direction vector.
        let d = &ray.direction;
        assert!(
            d.iter().all(|&v| v.is_finite()) && (d[0]).abs() <= 1e-9,
            "x0 is fixed by the Eq row: recession d0 = 0, got {d:?}"
        );
        for &(r, s) in &[(1usize, 2usize), (2, 3), (3, 1)] {
            assert!(
                d[r] - d[s] <= 1e-6,
                "recession on difference row ({r},{s}): d{r} − d{s} ≤ 0, got {d:?}"
            );
        }
        assert!(
            -d[1] - d[2] - d[3] < -1e-6,
            "descent: cᵀd = −(d1+d2+d3) < 0, got {d:?}"
        );
        assert!(
            (ray.base_point[0] - 5.0).abs() <= 1e-6
                && ray.base_point[1..].iter().all(|&v| v >= -1e-6),
            "base must satisfy x0=5, x≥0, got {:?}",
            ray.base_point
        );
        let x = &ray.base_point;
        for &(r, s) in &[(1usize, 2usize), (2, 3), (3, 1)] {
            assert!(
                x[r] - x[s] <= 1.0 + 1e-6,
                "base must satisfy difference row {r}−{s} ≤ 1, got {x:?}"
            );
        }
    }

    /// An Optimal LP carries no ray payload (the contract is Some iff a
    /// verified Unbounded).
    #[test]
    fn optimal_lp_has_no_ray_payload() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_with(&lp, &unbounded_opts(false));
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(res.ray_certificate.is_none());
    }
}
