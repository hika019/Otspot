//! Dual Simplex. Maintains dual feasibility (r_j ≥ 0) and restores primal
//! feasibility (x_B ≥ 0). Primary use: warm-start re-optimization after RHS
//! changes (e.g. SQP).

use super::dual_common::{
    basic_obj, compute_dual_vars, compute_reduced_costs, lp_unbounded_ray_verified,
    outcome_to_result,
};
use super::pricing::{DualLeavingStrategy, MostInfeasibleLeaving, SteepestEdgePricing};
use super::trace::IterTrace;
use super::{stop_result_with_incumbent, SimplexOutcome, StandardForm};
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::presolve::LpEquilibration;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use std::sync::atomic::Ordering;

#[cfg(test)]
thread_local! {
    static ETA_UPDATE_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ETA_REJECT_ATOMIC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reject_eta_for_test(alpha: &mut SparseVec) {
    if ETA_UPDATE_DISABLE.get() {
        alpha.indices.clear();
        alpha.values.clear();
    }
}

#[cfg(not(test))]
#[inline(always)]
fn reject_eta_for_test(_: &mut SparseVec) {}

/// Two-phase dual simplex entry point. Warm-start path recomputes x_B from the
/// supplied basis; cold-start uses cost perturbation to gain dual feasibility,
/// then runs primal Phase II.
pub(crate) fn two_phase_dual_simplex(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let m = sf.m;
    let Some((a, b, c, row_scale, col_scale)) =
        LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c, options.deadline)
    else {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            ..Default::default()
        };
    };

    if let Some(warm) = &options.warm_start {
        if warm.basis.len() == m && warm.basis.iter().all(|&idx| idx < sf.n_total) {
            let mut basis = warm.basis.clone();

            // Singular basis falls through to cold start.
            if let Ok(mut basis_mgr) =
                LuBasis::new_timed(&a, &basis, options.max_etas, options.deadline)
            {
                // x_B = B^{-1} b_new
                let mut x_b_sv = SparseVec::from_dense(&b);
                basis_mgr.ftran(&mut x_b_sv);
                let mut x_b = x_b_sv.to_dense();

                // WarmStartBasis lacks at_upper, so nonbasic variables are assumed
                // at lb=0. For the "up" branch, the branched variable's lb is raised
                // above its parent value → x_b becomes negative (lb violation).
                // The legacy dual simplex has no cost-perturbation anti-cycling;
                // starting from an lb-violating point leads to cycling/Timeout.
                // Fall through to cold start when any lb violation is present.
                if !super::has_lb_violation(&x_b, options.primal_tol) {
                    let mut total_iters: usize = 0;
                    let outcome = dual_simplex_core(
                        &a,
                        &mut x_b,
                        &c,
                        &mut basis,
                        m,
                        sf.n_total,
                        options,
                        &mut total_iters,
                    );

                    let mut result = outcome_to_result(
                        outcome, sf, problem, &basis, &x_b, &col_scale, &row_scale, true, options,
                    );
                    result.iterations = total_iters;
                    return result;
                }
            }
        }
    }

    cold_start_dual(sf, problem, options, &a, &b, &c, &row_scale, &col_scale)
}

/// Cost-perturbation cold start: Dual Phase I restores primal feasibility,
/// then Primal Phase II optimizes the original objective.
#[allow(clippy::too_many_arguments)]
fn cold_start_dual(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;

    // Ge/Eq → slack basis is singular; fall back to primal simplex.
    if sf.num_artificial > 0 {
        return super::two_phase_simplex(sf, problem, options);
    }

    // Le-only: B=I, x_B = b ≥ 0 after standard-form transform.
    let mut basis = sf.initial_basis.clone();
    let mut x_b = b.to_vec();

    // Perturb costs to c̃_j = max(c_j, 0) so r̃_j = c̃_j ≥ 0 (slack basis ⇒ y=0).
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut total_iters: usize = 0;
    let phase1_outcome = dual_simplex_core(
        a,
        &mut x_b,
        &c_perturbed,
        &mut basis,
        m,
        sf.n_total,
        options,
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            // Dual-unbounded ⇒ primal-infeasible.
            return SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            };
        }
        SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
            return stop_result_with_incumbent(
                sf,
                problem,
                &basis,
                &x_b,
                col_scale,
                total_iters,
                options,
            );
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {}
    }

    let mut pricing = SteepestEdgePricing::new(sf.n_total);
    let phase2_outcome = super::revised_simplex_core(
        a,
        &mut x_b,
        c,
        b,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        &mut pricing,
        options,
        &mut total_iters,
        false,
        None,
        false,
        None,
    );

    // Gate a Phase II `Unbounded` on a re-derived recession ray (same verified
    // gate as the Big-M path). An eta-drift false-Unbounded becomes an honest
    // Stalled instead of a wrong verdict (pilot-ja reaches here via the Big-M →
    // dual-fallback handoff).
    let phase2_outcome = if matches!(phase2_outcome, SimplexOutcome::Unbounded)
        && !lp_unbounded_ray_verified(a, &basis, c, m, sf.n_total, sf.n_total, options)
    {
        SimplexOutcome::Stalled(basic_obj(c, &basis, &x_b))
    } else {
        phase2_outcome
    };

    let mut result = outcome_to_result(
        phase2_outcome,
        sf,
        problem,
        &basis,
        &x_b,
        col_scale,
        row_scale,
        false,
        options,
    );
    result.iterations = total_iters;
    result
}

/// Dual simplex core. Caller must establish dual feasibility (warm-start or
/// cost perturbation) before invocation.
pub(super) fn dual_simplex_core(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    basis: &mut [usize],
    m: usize,
    n_price: usize,
    options: &SolverOptions,
    iter_count_out: &mut usize,
) -> SimplexOutcome {
    let max_iter = usize::MAX; // timeout is the real guard

    let mut basis_mgr = match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }
    };

    let mut is_basic = vec![false; n_price];
    for &b in basis.iter() {
        if b < n_price {
            is_basic[b] = true;
        }
    }

    // r_j = c_j - y^T a_j, y = B^{-T} c_B
    let mut reduced_costs =
        compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);

    let mut leaving_strategy = MostInfeasibleLeaving;
    let mut rho_dense = vec![0.0f64; m];
    let mut trow = vec![0.0f64; n_price];
    let mut alpha_dense = vec![0.0f64; m];
    let mut trace = IterTrace::new("dual-legacy");

    for _iter in 0..max_iter {
        *iter_count_out = iter_count_out.saturating_add(1);
        let timed_out = options
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = basic_obj(c, basis, x_b);
            return SimplexOutcome::Timeout(obj);
        }

        if let Some(t) = trace.as_mut() {
            let obj = basic_obj(c, basis, x_b);
            t.log(*iter_count_out, obj, basis, false);
        }

        let leaving_row = match leaving_strategy.select_leaving(x_b, options.primal_tol, basis) {
            None => {
                let obj: f64 = basic_obj(c, basis, x_b);
                let y = compute_dual_vars(c, &mut basis_mgr, basis, m);
                return SimplexOutcome::Optimal(obj, y);
            }
            Some(p) => p,
        };

        // BTRAN: ρ = B^{-T} e_p
        let mut e_p = vec![0.0f64; m];
        e_p[leaving_row] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&e_p);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut rho_dense);

        // PRICE: trow[j] = ρ^T a_j  (non-basic columns)
        for j in 0..n_price {
            if is_basic[j] {
                trow[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut dot = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                dot += rho_dense[row] * vals[k];
            }
            trow[j] = dot;
        }

        let (entering_col, theta) =
            match dual_ratio_test(&trow, &reduced_costs, &is_basic, n_price, PIVOT_TOL) {
                None => return SimplexOutcome::Unbounded,
                Some(result) => result,
            };

        // FTRAN: α = B^{-1} a_q
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut alpha_sv);
        alpha_sv.to_dense_into(&mut alpha_dense);

        let pivot_element = alpha_dense[leaving_row];
        if pivot_element.abs() < PIVOT_TOL {
            // Unstable pivot: refactor and recompute reduced costs.
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return SimplexOutcome::SingularBasis;
                }
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
            continue;
        }

        // Commit the eta while every externally observable simplex state still
        // describes the old basis. A rejected numerical pivot must not publish
        // a new x_B, reduced-cost vector, or basis-membership map.
        #[cfg(test)]
        let atomic_snapshot = (
            x_b.to_vec(),
            reduced_costs.clone(),
            is_basic.clone(),
            basis.to_vec(),
        );
        reject_eta_for_test(&mut alpha_sv);
        match basis_mgr.update(entering_col, leaving_row, &alpha_sv) {
            Ok(()) => {}
            Err(crate::error::SolverError::SingularBasis { .. }) => {
                #[cfg(test)]
                {
                    assert_eq!(x_b, atomic_snapshot.0);
                    assert_eq!(reduced_costs, atomic_snapshot.1);
                    assert_eq!(is_basic, atomic_snapshot.2);
                    assert_eq!(basis, atomic_snapshot.3);
                    ETA_REJECT_ATOMIC_COUNT.set(ETA_REJECT_ATOMIC_COUNT.get().saturating_add(1));
                }
                return SimplexOutcome::SingularBasis;
            }
            Err(err) => panic!("internal dual-simplex eta invariant violated: {err}"),
        }

        // x_B update; step = x_B[p] / α[p] (negative).
        let step = x_b[leaving_row] / pivot_element;
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[leaving_row] = step;

        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // r_j_new = r_j - θ * trow[j] for non-basic j; r_{leaving_col} = -θ.
        let leaving_col = basis[leaving_row];
        for j in 0..n_price {
            if !is_basic[j] {
                reduced_costs[j] -= theta * trow[j];
            }
        }
        if leaving_col < n_price {
            reduced_costs[leaving_col] = -theta;
        }

        if leaving_col < n_price {
            is_basic[leaving_col] = false;
        }
        is_basic[entering_col] = true;

        basis[leaving_row] = entering_col;

        if basis_mgr_needs_refactor_approx(_iter) {
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return SimplexOutcome::SingularBasis;
                }
                let obj: f64 = basic_obj(c, basis, x_b);
                return SimplexOutcome::Timeout(obj);
            }
            reduced_costs =
                compute_reduced_costs(a, c, &mut basis_mgr, &is_basic, n_price, m, basis);
        }
    }

    let obj: f64 = basic_obj(c, basis, x_b);
    SimplexOutcome::Timeout(obj)
}

/// Throttle reduced-cost recomputation (every 50 iters) — separate from the
/// LuBasis-internal refactor check since recomputation has extra cost.
#[inline]
fn basis_mgr_needs_refactor_approx(iter: usize) -> bool {
    iter % 50 == 49
}

/// θ = min_{j: trow[j] > ε} r_j / trow[j].  None ⇒ dual unbounded.
fn dual_ratio_test(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
    pivot_tol: f64,
) -> Option<(usize, f64)> {
    let mut min_ratio = f64::INFINITY;
    let mut entering = None;

    for j in 0..n_price {
        if is_basic[j] {
            continue;
        }

        if trow[j] > pivot_tol {
            let ratio = reduced_costs[j] / trow[j];
            if ratio < min_ratio - pivot_tol {
                min_ratio = ratio;
                entering = Some(j);
            } else if (ratio - min_ratio).abs() <= pivot_tol {
                // Bland's rule for ties.
                if let Some(prev_j) = entering {
                    if j < prev_j {
                        entering = Some(j);
                    }
                }
            }
        }
    }

    entering.map(|j| (j, min_ratio))
}

#[cfg(test)]
mod tests {
    use crate::options::{SimplexMethod, SolverOptions};
    use crate::problem::{LpProblem, SolveStatus};
    use crate::simplex::solve_with;
    use crate::sparse::CscMatrix;
    use crate::test_kkt::{assert_kkt_optimal_with, dfeas_rel_bound, pfeas_abs, EPS_KKT};
    use crate::tolerances::PIVOT_TOL;

    struct EtaRejectGuard(bool);

    impl EtaRejectGuard {
        fn new() -> Self {
            super::ETA_REJECT_ATOMIC_COUNT.set(0);
            Self(super::ETA_UPDATE_DISABLE.replace(true))
        }
    }

    impl Drop for EtaRejectGuard {
        fn drop(&mut self) {
            super::ETA_UPDATE_DISABLE.set(self.0);
        }
    }

    #[test]
    fn rejected_eta_leaves_legacy_dual_state_atomic() {
        // basis {col 0}, x_B=-1; the legacy ratio convention selects col 1=e
        // and forces one otherwise-valid dual pivot.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let mut x_b = vec![-1.0];
        let mut basis = vec![0usize];
        let old_x_b = x_b.clone();
        let old_basis = basis.clone();
        let mut iters = 0;
        let _guard = EtaRejectGuard::new();

        let outcome = super::dual_simplex_core(
            &a,
            &mut x_b,
            &[0.0, 0.0],
            &mut basis,
            1,
            2,
            &SolverOptions::default(),
            &mut iters,
        );

        assert!(matches!(
            outcome,
            crate::simplex::SimplexOutcome::SingularBasis
        ));
        assert_eq!(x_b, old_x_b);
        assert_eq!(basis, old_basis);
        assert_eq!(super::ETA_REJECT_ATOMIC_COUNT.get(), 1);
    }

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    #[test]
    fn test_dual_basic_nonneg_cost() {
        // min x1 + 2*x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3 → x1=x2=0, obj=0
        let lp = make_lp(
            vec![1.0, 2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        assert_kkt_optimal_with(&lp, 0.0, "test_dual_basic_nonneg_cost", &opts);
    }

    /// Primal/Dual converge to the same KKT optimum (objective + residuals).
    #[test]
    fn test_dual_matches_primal() {
        // min -x1 - 2*x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3 → x1=1, x2=3, obj=-7
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        let primal_opts = SolverOptions {
            simplex_method: SimplexMethod::Primal,
            ..SolverOptions::default()
        };
        let dual_opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };

        assert_kkt_optimal_with(&lp, -7.0, "test_dual_matches_primal/primal", &primal_opts);
        assert_kkt_optimal_with(&lp, -7.0, "test_dual_matches_primal/dual", &dual_opts);

        let result_p = solve_with(&lp, &primal_opts);
        let result_d = solve_with(&lp, &dual_opts);
        assert!(
            (result_p.objective - result_d.objective).abs() < 1e-6,
            "Primal obj={}, Dual obj={}",
            result_p.objective,
            result_d.objective
        );
    }

    /// Warm-start with RHS-only change must satisfy full KKT (obj match alone
    /// would miss dfeas degradation on the warm-start path).
    #[test]
    fn test_dual_warm_start_rhs_change() {
        // LP1: min -x1 - 2*x2 s.t. x1+x2 ≤ 4; -x1+x2 ≤ 2; x1-x2 ≤ 2
        // x1,x2 in [0,inf). Optimal: x1=1, x2=3, obj=-7.
        let lp1 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            3,
            2,
            vec![4.0, 2.0, 2.0],
        );

        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(result1.warm_start_basis.is_some());

        // LP2: same structure, b=[5,2,2] → x1=1.5, x2=3.5, obj=-8.5
        let lp2 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            3,
            2,
            vec![5.0, 2.0, 2.0],
        );

        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        assert_kkt_optimal_with(&lp2, -8.5, "test_dual_warm_start_rhs_change", &opts_warm);
    }

    #[test]
    fn test_dual_simplex_method_option() {
        // min -x1 - x2 s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3 → obj=-4
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        assert_kkt_optimal_with(&lp, -4.0, "test_dual_simplex_method_option", &opts);
    }

    /// Warm-start asserts bound-aware dfeas_rel_bound and pfeas_abs directly;
    /// `rc ≥ 0` alone would miss pfeas / bound-aware dfeas degradation.
    #[test]
    fn test_dual_warm_start_preserves_dual_feasibility() {
        // LP1: min x1 + x2 s.t. x1+x2 ≤ 6, x1 ≤ 4, x2 ≤ 4 → x1=x2=0, obj=0
        let lp1 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![6.0, 4.0, 4.0],
        );

        // Opt-in to postsolve warm-basis recovery: lp1 is dual-fixed → presolve
        // reduces to zero vars → simplex returns warm_start_basis=None, and the
        // default path skips the postsolve synthesis for performance.
        let opts1 = SolverOptions {
            recover_warm_start_basis: true,
            ..SolverOptions::default()
        };
        let result1 = solve_with(&lp1, &opts1);
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(result1.warm_start_basis.is_some());

        // LP2: b=[5,3,3] (狭めた)。c≥0 なので最適は依然 x=0, obj=0
        let lp2 = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 3.0, 3.0],
        );

        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::Dual,
            ..SolverOptions::default()
        };
        let result2 = solve_with(&lp2, &opts_warm);
        assert_eq!(result2.status, SolveStatus::Optimal);

        let pf = pfeas_abs(&lp2.a, &lp2.b, &lp2.constraint_types, &result2.solution);
        assert!(pf < EPS_KKT, "pfeas={:.3e} > {:.3e}", pf, EPS_KKT);

        let df = dfeas_rel_bound(
            &lp2.c,
            &lp2.bounds,
            &result2.solution,
            &result2.reduced_costs,
        );
        assert!(df < EPS_KKT, "dfeas_rel_bound={:.3e} > {:.3e}", df, EPS_KKT);

        for &rc in &result2.reduced_costs {
            assert!(rc >= -PIVOT_TOL, "rc={} < -PIVOT_TOL", rc);
        }
    }

    #[test]
    fn test_dual_simplex_timeout() {
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let opts = SolverOptions {
            simplex_method: SimplexMethod::Dual,
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// Singular initial basis must not yield a spurious Optimal.
    #[test]
    fn test_dual_singular_basis_not_optimal() {
        use crate::simplex::dual::dual_simplex_core;
        use crate::simplex::SimplexOutcome;

        // Duplicate basis column → B singular → LuBasis::new fails.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let mut x_b = vec![1.0, 0.0];
        let mut basis = vec![0usize, 0];
        let opts = SolverOptions::default();
        let mut iters = 0usize;
        let outcome = dual_simplex_core(&a, &mut x_b, &c, &mut basis, 2, 2, &opts, &mut iters);
        assert!(!matches!(outcome, SimplexOutcome::Optimal(..)));
        assert!(matches!(
            outcome,
            SimplexOutcome::Timeout(..) | SimplexOutcome::SingularBasis
        ));
    }
}
