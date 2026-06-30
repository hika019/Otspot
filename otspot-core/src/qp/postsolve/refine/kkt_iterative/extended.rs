//! Extended-precision iterative refinement using TwoFloat accumulation.
//!
//! Standard Krylov IR (iterative.rs) computes residuals in DD but updates x, y
//! in f64. When corrections are at the ULP level of the current solution, the
//! f64 update `x + dx` rounds the correction away, stalling the IR.
//!
//! This module maintains the running solution (x, y) as TwoFloat vectors,
//! accumulating f64 corrections without rounding loss. The f64 LDL factorization
//! is reused as-is; only the solution accumulation and residual computation
//! benefit from extended precision.

use super::bound_refit::refit_bound_duals_kkt;
use crate::qp::kkt_resid;
use crate::qp::problem::QpProblem;
use crate::tolerances::{any_nonfinite, FX_TOL};
use twofloat::TwoFloat;

// The loop is primarily governed by EXT_REL_STALL/EXT_PROGRESS_EPS below; this
// cap prevents pathological ill-conditioned systems from monopolizing postsolve.
const MAX_EXTENDED_ITERS: usize = 200;

/// Relative stall threshold for extended IR score.
const EXT_REL_STALL: f64 = 1e-10;
/// Absolute floor for near-zero scores.
const EXT_PROGRESS_EPS: f64 = 1e-14;

fn ext_score_made_progress(score_cur: f64, score_new: f64) -> bool {
    let threshold = (EXT_REL_STALL * score_cur.abs()).max(EXT_PROGRESS_EPS);
    score_new + threshold < score_cur
}

fn optimality_worst_residual(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
    eliminated_cols: &[bool],
) -> f64 {
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    )
    .max(crate::qp::ipm_solver::kkt::primal_residual_rel(
        &view,
        &result.solution,
    ))
    .max(crate::qp::ipm_solver::kkt::bound_violation(
        problem.bounds.as_slice(),
        &result.solution,
    ))
    .max(crate::qp::ipm_solver::kkt::complementarity_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    ))
    .max(
        crate::qp::ipm_solver::kkt::complementarity_componentwise_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        ),
    )
    .max(kkt_resid::dual_sign_violation(
        &problem.constraint_types,
        &result.dual_solution,
        &problem.bounds,
        &result.bound_duals,
    ))
}

/// Q * x_ext: CSC SpMV with TwoFloat solution vector.
fn qx_ext(q: &crate::sparse::CscMatrix, x_ext: &[TwoFloat]) -> Vec<TwoFloat> {
    let n = q.nrows;
    let mut out = vec![TwoFloat::from(0.0); n];
    for col in 0..q.ncols {
        let xv = x_ext[col];
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            out[row] += TwoFloat::from(q.values[k]) * xv;
        }
    }
    out
}

/// A^T * y_ext: CSC transpose SpMV with TwoFloat dual vector.
fn aty_ext(a: &crate::sparse::CscMatrix, y_ext: &[TwoFloat], n: usize) -> Vec<TwoFloat> {
    let mut out = vec![TwoFloat::from(0.0); n];
    if a.nrows == 0 || y_ext.is_empty() {
        return out;
    }
    for col in 0..a.ncols {
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            let row = a.row_ind[k];
            out[col] += TwoFloat::from(a.values[k]) * y_ext[row];
        }
    }
    out
}

/// A * x_ext: CSC SpMV with TwoFloat solution vector.
fn ax_ext(a: &crate::sparse::CscMatrix, x_ext: &[TwoFloat]) -> Vec<TwoFloat> {
    if a.nrows == 0 {
        return Vec::new();
    }
    let mut out = vec![TwoFloat::from(0.0); a.nrows];
    for col in 0..a.ncols {
        let xv = x_ext[col];
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            out[a.row_ind[k]] += TwoFloat::from(a.values[k]) * xv;
        }
    }
    out
}

/// Extended-precision iterative refinement.
///
/// Maintains (x, y) as TwoFloat vectors and accumulates f64 LDL corrections
/// in extended precision. Returns the number of accepted refinement steps.
///
/// Safety guarantee: the original solution is preserved if extended IR doesn't
/// improve the optimality-classification score
/// `max(stationarity, primal feasibility, bound violation, complementarity, dual sign)`,
/// where complementarity is the max of problem-level and componentwise metrics.
pub(crate) fn refine_kkt_extended_precision(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::problem::ConstraintType;

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n || result.dual_solution.len() != m {
        return 0;
    }
    if n + m > crate::tolerances::LARGE_PROBLEM_THRESHOLD {
        return 0;
    }

    // Build and factorize the augmented system K = [Q+δI, A^T; A, -δI].
    const DELTA_P: f64 = 1e-10;
    const DELTA_D: f64 = 1e-10;
    const ACTIVE_TOL: f64 = 1e-8;
    const ACTIVE_PENALTY_RATIO: f64 = 1e8;
    const FACTOR_RETRY_GROWTH: f64 = 10.0;
    const FACTOR_RETRY_MAX: usize = 6;

    let sigma_zero = vec![0.0_f64; m];
    let mut k_mat = crate::qp::ipm_core::kkt::build_augmented_system(
        &problem.q,
        &problem.a,
        &sigma_zero,
        DELTA_P,
        DELTA_D,
    );

    // Apply active-set penalty on bound-active variables.
    {
        let mut k_diag_max = 0.0_f64;
        for j in 0..(n + m) {
            let cs = k_mat.col_ptr[j];
            let ce = k_mat.col_ptr[j + 1];
            for k in cs..ce {
                if k_mat.row_ind[k] == j {
                    k_diag_max = k_diag_max.max(k_mat.values[k].abs());
                    break;
                }
            }
        }
        let active_penalty = (k_diag_max * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
        for j in 0..n {
            let x = result.solution[j];
            let (lb, ub) = problem.bounds[j];
            let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
            if !is_active {
                continue;
            }
            let col_start = k_mat.col_ptr[j];
            let col_end = k_mat.col_ptr[j + 1];
            for k in col_start..col_end {
                if k_mat.row_ind[k] == j {
                    k_mat.values[k] += active_penalty;
                    break;
                }
            }
        }
    }

    let factor = {
        let mut dp = DELTA_P;
        let mut dd = DELTA_D;
        let mut cur_k = k_mat.clone();
        let mut factor_result: Option<crate::linalg::ldl::LdlFactorizationAmd> = None;
        let mut retries = 0usize;
        loop {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            match crate::linalg::ldl::factorize_quasidefinite_with_amd(&cur_k, deadline) {
                Ok(f) => {
                    factor_result = Some(f);
                    break;
                }
                Err(_) => {
                    if retries >= FACTOR_RETRY_MAX {
                        break;
                    }
                    retries += 1;
                    dp *= FACTOR_RETRY_GROWTH;
                    dd *= FACTOR_RETRY_GROWTH;
                    cur_k = crate::qp::ipm_core::kkt::build_augmented_system(
                        &problem.q,
                        &problem.a,
                        &sigma_zero,
                        dp,
                        dd,
                    );
                    let mut diag_max = 0.0_f64;
                    for j in 0..(n + m) {
                        let cs = cur_k.col_ptr[j];
                        let ce = cur_k.col_ptr[j + 1];
                        for k in cs..ce {
                            if cur_k.row_ind[k] == j {
                                diag_max = diag_max.max(cur_k.values[k].abs());
                                break;
                            }
                        }
                    }
                    let ap = (diag_max * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
                    for j in 0..n {
                        let x = result.solution[j];
                        let (lb, ub) = problem.bounds[j];
                        let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                            || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
                        if !is_active {
                            continue;
                        }
                        let cs = cur_k.col_ptr[j];
                        let ce = cur_k.col_ptr[j + 1];
                        for k in cs..ce {
                            if cur_k.row_ind[k] == j {
                                cur_k.values[k] += ap;
                                break;
                            }
                        }
                    }
                }
            }
        }
        match factor_result {
            Some(f) => f,
            None => return 0,
        }
    };

    // Exclude FX / eliminated columns from stationarity evaluation.
    let use_elim_mask = eliminated_cols.len() == n;
    let exclude_var: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            if use_elim_mask && eliminated_cols[j] {
                return true;
            }
            false
        })
        .collect();

    // Initialize extended-precision solution vectors.
    let mut x_ext: Vec<TwoFloat> = result.solution.iter().map(|&v| TwoFloat::from(v)).collect();
    let mut y_ext: Vec<TwoFloat> = result
        .dual_solution
        .iter()
        .map(|&v| TwoFloat::from(v))
        .collect();

    // Compute residuals using extended-precision x, y.
    let compute_residuals_ext =
        |x_e: &[TwoFloat], y_e: &[TwoFloat], z: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let qx_dd = qx_ext(&problem.q, x_e);
            let aty_dd = aty_ext(&problem.a, y_e, n);
            let bc_vec = kkt_resid::bound_contrib(&problem.bounds, z);

            let mut r_d = vec![0.0_f64; n];
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let r =
                    qx_dd[j] + TwoFloat::from(problem.c[j]) + aty_dd[j] + TwoFloat::from(bc_vec[j]);
                r_d[j] = f64::from(r);
            }

            let ax_dd = ax_ext(&problem.a, x_e);
            let mut r_p = vec![0.0_f64; m];
            for i in 0..m {
                let raw = f64::from(ax_dd[i] - TwoFloat::from(problem.b[i]));
                #[allow(unreachable_patterns)]
                let v = match problem.constraint_types[i] {
                    ConstraintType::Eq => raw,
                    ConstraintType::Ge if raw < 0.0 => raw,
                    ConstraintType::Le if raw > 0.0 => raw,
                    _ => 0.0,
                };
                r_p[i] = v;
            }

            (r_d, r_p)
        };

    let original_result = result.clone();
    let mut best_score = optimality_worst_residual(problem, result, eliminated_cols);
    if best_score < target_pf {
        return 0;
    }

    let mut best_result = result.clone();
    let mut accepted = 0usize;

    for _iter in 0..MAX_EXTENDED_ITERS {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }

        // Round to f64 for bound-dual refit (bound duals are f64).
        let x_f64: Vec<f64> = x_ext.iter().map(|&v| f64::from(v)).collect();
        let y_f64: Vec<f64> = y_ext.iter().map(|&v| f64::from(v)).collect();

        // Compute residuals with current extended-precision solution.
        let mut tmp_result = crate::problem::SolverResult {
            solution: x_f64,
            dual_solution: y_f64,
            bound_duals: result.bound_duals.clone(),
            ..Default::default()
        };
        refit_bound_duals_kkt(problem, &mut tmp_result, target_pf);

        let (r_d, r_p) = compute_residuals_ext(&x_ext, &y_ext, &tmp_result.bound_duals);

        let score_cur = optimality_worst_residual(problem, &tmp_result, eliminated_cols);
        if score_cur < target_pf {
            // Already below target after bound refit.
            result.solution = tmp_result.solution;
            result.dual_solution = tmp_result.dual_solution;
            result.bound_duals = tmp_result.bound_duals;
            if score_cur < best_score {
                best_result = result.clone();
            }
            accepted += 1;
            break;
        }

        // Build RHS for correction system: [−r_d; −r_p].
        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n {
            rhs[j] = -r_d[j];
        }
        for i in 0..m {
            rhs[n + i] = -r_p[i];
        }

        // Solve K * [dx; dy] = -[r_d; r_p] in f64.
        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        if any_nonfinite(&sol) {
            break;
        }

        // Accumulate corrections in TwoFloat (the key innovation).
        let mut x_new = x_ext.clone();
        let mut y_new = y_ext.clone();
        for j in 0..n {
            x_new[j] += TwoFloat::from(sol[j]);
            // Clip to bounds in TwoFloat.
            let (lb, ub) = problem.bounds[j];
            let x_f = f64::from(x_new[j]);
            if lb.is_finite() && x_f < lb {
                x_new[j] = TwoFloat::from(lb);
            }
            if ub.is_finite() && x_f > ub {
                x_new[j] = TwoFloat::from(ub);
            }
        }
        for i in 0..m {
            y_new[i] += TwoFloat::from(sol[n + i]);
        }

        // Evaluate new score with refitted bound duals.
        let x_new_f64: Vec<f64> = x_new.iter().map(|&v| f64::from(v)).collect();
        let y_new_f64: Vec<f64> = y_new.iter().map(|&v| f64::from(v)).collect();
        let mut new_result = crate::problem::SolverResult {
            solution: x_new_f64,
            dual_solution: y_new_f64,
            bound_duals: tmp_result.bound_duals.clone(),
            ..Default::default()
        };
        refit_bound_duals_kkt(problem, &mut new_result, target_pf);

        let score_new = optimality_worst_residual(problem, &new_result, eliminated_cols);

        if ext_score_made_progress(score_cur, score_new) {
            x_ext = x_new;
            y_ext = y_new;
            result.bound_duals = new_result.bound_duals;
            accepted += 1;

            if score_new < best_score {
                best_score = score_new;
                best_result = crate::problem::SolverResult {
                    solution: x_ext.iter().map(|&v| f64::from(v)).collect(),
                    dual_solution: y_ext.iter().map(|&v| f64::from(v)).collect(),
                    bound_duals: result.bound_duals.clone(),
                    ..Default::default()
                };
            }
        } else {
            break;
        }
    }

    // Restore best result if it improved over the original on the same
    // composite score used by optimality classification.
    let orig_worst = optimality_worst_residual(problem, &original_result, eliminated_cols);
    let best_worst = optimality_worst_residual(problem, &best_result, eliminated_cols);

    if best_worst < orig_worst {
        result.solution = best_result.solution;
        result.dual_solution = best_result.dual_solution;
        result.bound_duals = best_result.bound_duals;
    } else {
        // Extended IR didn't help; restore original.
        *result = original_result;
        return 0;
    }

    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, SolverResult};
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    fn complementarity(problem: &QpProblem, result: &SolverResult) -> f64 {
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };
        crate::qp::ipm_solver::kkt::complementarity_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        )
    }

    fn componentwise_complementarity(problem: &QpProblem, result: &SolverResult) -> f64 {
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };
        crate::qp::ipm_solver::kkt::complementarity_componentwise_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        )
    }

    /// Sentinel: extended-precision IR must reduce the KKT residual on an
    /// ill-conditioned QP where the initial dual has large cancellation error.
    ///
    /// Problem: min 0.5 ||x||^2, s.t. Ax = b, x free.
    ///   A = [[1, 1+d], [1-d, 1]]  with d = 1e-4
    ///   b = [2+d, 2-d]            (x*=[1,1] is primal feasible)
    ///   cond(A A^T) ~ 4/d^4 = 4e16
    ///
    /// True optimal: x=[1,1], y=[-1/d, 1/d]=[-1e4, 1e4].
    ///
    /// Starting point: x=[1,1] (exact primal), y=[-1e4+2, 1e4-2] (dual with O(1) error).
    /// The initial stationarity residual is O(1)/O(1e4) ~ O(1e-4) >> eps=1e-6.
    ///
    /// Extended IR must reduce the residual below 1e-4. The test FAILS if extended IR
    /// is a no-op (returns 0 / no improvement).
    #[test]
    fn extended_ir_reduces_residual_on_ill_conditioned_qp() {
        let d = 1e-4_f64;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0_f64, 1.0], 2, 2).unwrap();
        // A = [[1, 1+d], [1-d, 1]] in CSC (column-major)
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1],                 // row indices
            &[0, 0, 1, 1],                 // col indices
            &[1.0, 1.0 - d, 1.0 + d, 1.0], // values
            2,
            2,
        )
        .unwrap();
        let c = vec![0.0_f64, 0.0];
        let b = vec![2.0 + d, 2.0 - d];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let ct = vec![ConstraintType::Eq; 2];

        let problem = QpProblem::new(q, c, a, b, bounds, ct).unwrap();

        // Dual with controlled error: y_true = [-1/d, 1/d], perturbed by +2, -2.
        let y_true_0 = -1.0 / d;
        let y_true_1 = 1.0 / d;
        let y_init = vec![y_true_0 + 2.0, y_true_1 - 2.0];

        let mut result = SolverResult {
            solution: vec![1.0, 1.0],
            dual_solution: y_init,
            bound_duals: vec![],
            ..Default::default()
        };

        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };

        let kkt_before = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        assert!(
            kkt_before > 1e-5,
            "fixture: initial KKT residual must be large, got {kkt_before:.3e}"
        );

        let steps = refine_kkt_extended_precision(&problem, &mut result, &[], 1e-6, None);

        let kkt_after = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(
            steps > 0,
            "extended IR must accept at least one step (steps={steps})"
        );
        assert!(
            kkt_after < kkt_before,
            "extended IR must reduce KKT residual: {kkt_before:.3e} -> {kkt_after:.3e}"
        );
        assert!(
            kkt_after < 1e-5,
            "extended IR must bring residual well below initial {kkt_before:.3e}, got {kkt_after:.3e}"
        );
    }

    /// Safety sentinel: extended IR must NEVER worsen a correct result.
    ///
    /// Start from the exact optimal of a well-conditioned QP. Extended IR
    /// must return 0 steps (already below target) or preserve the solution.
    #[test]
    fn extended_ir_does_not_worsen_exact_optimal() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0_f64, 1.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0], 1, 2).unwrap();
        let c = vec![0.0_f64, 0.0];
        let b = vec![2.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let ct = vec![ConstraintType::Eq];

        let problem = QpProblem::new(q, c, a, b, bounds, ct).unwrap();

        // Exact optimal: x=[1,1], y=-1.
        let mut result = SolverResult {
            solution: vec![1.0, 1.0],
            dual_solution: vec![-1.0],
            bound_duals: vec![],
            ..Default::default()
        };

        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };

        let kkt_before = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        let _steps = refine_kkt_extended_precision(&problem, &mut result, &[], 1e-6, None);

        let kkt_after = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(
            kkt_after <= kkt_before + 1e-15,
            "extended IR must not worsen exact optimal: {kkt_before:.3e} -> {kkt_after:.3e}"
        );
    }

    /// Safety-net sentinel for P1: an inactive inequality can receive a large
    /// dual update that improves stationarity while making complementarity much
    /// worse. Such a candidate must be rejected by the same composite residual
    /// used for optimality classification.
    #[test]
    fn extended_ir_rejects_stationarity_gain_that_worsens_inequality_complementarity() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0e-12_f64, 1.0e6], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 2).unwrap();
        let c = vec![-1.0_f64, 0.0];
        let b = vec![100.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let ct = vec![ConstraintType::Le];
        let problem = QpProblem::new(q, c, a, b, bounds, ct).unwrap();

        let mut result = SolverResult {
            solution: vec![0.0, 0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..Default::default()
        };
        let original = result.clone();

        let comp_before = complementarity(&problem, &result);
        let worst_before = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(comp_before, 0.0, "fixture starts complementarity-clean");
        assert!(
            worst_before > 1e-2,
            "fixture must need refinement, got worst={worst_before:.3e}"
        );

        let steps = refine_kkt_extended_precision(&problem, &mut result, &[], 1e-6, None);

        let comp_after = complementarity(&problem, &result);
        let worst_after = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(
            steps, 0,
            "candidate that worsens complementarity must be rejected"
        );
        assert_eq!(
            result.solution, original.solution,
            "rejected candidate must restore primal solution"
        );
        assert_eq!(
            result.dual_solution, original.dual_solution,
            "rejected candidate must restore constraint duals"
        );
        assert!(
            comp_after <= comp_before + 1e-15,
            "complementarity must not worsen: {comp_before:.3e} -> {comp_after:.3e}"
        );
        assert!(
            worst_after <= worst_before + 1e-15,
            "composite optimality residual must not worsen: {worst_before:.3e} -> {worst_after:.3e}"
        );
    }

    /// Multi-row P1 sentinel: problem-level complementarity can hide a single
    /// inactive inequality dual when unrelated equality rows make the global
    /// scale large. The safety net must use the componentwise classifier metric
    /// too, so this stationarity-improving candidate is reverted.
    #[test]
    fn extended_ir_rejects_componentwise_complementarity_hidden_by_global_scale() {
        let scale_rows = 24_usize;
        let n = scale_rows + 1;
        let m = scale_rows + 1;
        let big_x = 1.0e6_f64;
        let big_y = 1.0e6_f64;

        let mut q_rows = Vec::with_capacity(n);
        let mut q_cols = Vec::with_capacity(n);
        let mut q_vals = Vec::with_capacity(n);
        q_rows.push(0);
        q_cols.push(0);
        q_vals.push(1.0e-12);
        for j in 1..n {
            q_rows.push(j);
            q_cols.push(j);
            q_vals.push(1.0);
        }
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();

        let mut a_rows = Vec::with_capacity(m);
        let mut a_cols = Vec::with_capacity(m);
        let mut a_vals = Vec::with_capacity(m);
        a_rows.push(0);
        a_cols.push(0);
        a_vals.push(1.0);
        for i in 1..m {
            a_rows.push(i);
            a_cols.push(i);
            a_vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();

        let mut c = vec![-1.0_f64; n];
        let mut b = vec![100.0_f64; m];
        let mut solution = vec![0.0_f64; n];
        let mut dual_solution = vec![0.0_f64; m];
        let mut ct = vec![ConstraintType::Le; m];
        for i in 1..m {
            c[i] = -(big_x + big_y);
            b[i] = big_x;
            solution[i] = big_x;
            dual_solution[i] = big_y;
            ct[i] = ConstraintType::Eq;
        }

        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds, ct).unwrap();

        let mut result = SolverResult {
            solution,
            dual_solution,
            bound_duals: vec![],
            ..Default::default()
        };
        let original = result.clone();

        let comp_before = complementarity(&problem, &result);
        let comp_component_before = componentwise_complementarity(&problem, &result);
        let worst_before = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(comp_before, 0.0, "fixture starts problem-level clean");
        assert_eq!(
            comp_component_before, 0.0,
            "fixture starts componentwise clean"
        );
        assert!(
            worst_before > 1e-1,
            "fixture must need stationarity refinement, got worst={worst_before:.3e}"
        );

        let steps = refine_kkt_extended_precision(&problem, &mut result, &[], 1e-6, None);

        let comp_after = complementarity(&problem, &result);
        let comp_component_after = componentwise_complementarity(&problem, &result);
        let worst_after = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(
            steps, 0,
            "candidate hidden by global complementarity scale must be rejected"
        );
        assert_eq!(
            result.solution, original.solution,
            "rejected candidate must restore primal solution"
        );
        assert_eq!(
            result.dual_solution, original.dual_solution,
            "rejected candidate must restore constraint duals"
        );
        assert!(
            comp_after <= comp_before + 1e-15,
            "problem-level complementarity must not worsen: {comp_before:.3e} -> {comp_after:.3e}"
        );
        assert!(
            comp_component_after <= comp_component_before + 1e-15,
            "componentwise complementarity must not worsen: {comp_component_before:.3e} -> {comp_component_after:.3e}"
        );
        assert!(
            worst_after <= worst_before + 1e-15,
            "composite optimality residual must not worsen: {worst_before:.3e} -> {worst_after:.3e}"
        );
    }

    /// Dual-sign sentinel: a Le row can receive a stationarity-improving
    /// negative dual update. Extended IR must reject that candidate because the
    /// adoption score matches optimality classification and includes dual sign.
    #[test]
    fn extended_ir_rejects_candidate_that_worsens_dual_sign() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![10.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let ct = vec![ConstraintType::Le];
        let problem = QpProblem::new(q, c, a, b, bounds, ct).unwrap();

        let mut result = SolverResult {
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..Default::default()
        };
        let original = result.clone();

        let sign_before = kkt_resid::dual_sign_violation(
            &problem.constraint_types,
            &result.dual_solution,
            &problem.bounds,
            &result.bound_duals,
        );
        let worst_before = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(sign_before, 0.0, "fixture starts dual-sign clean");
        assert!(
            worst_before > 1e-2,
            "fixture must need stationarity refinement, got worst={worst_before:.3e}"
        );

        let steps = refine_kkt_extended_precision(&problem, &mut result, &[], 1e-6, None);

        let sign_after = kkt_resid::dual_sign_violation(
            &problem.constraint_types,
            &result.dual_solution,
            &problem.bounds,
            &result.bound_duals,
        );
        let worst_after = optimality_worst_residual(&problem, &result, &[]);
        assert_eq!(steps, 0, "wrong-sign dual candidate must be rejected");
        assert_eq!(
            result.solution, original.solution,
            "rejected candidate must restore primal solution"
        );
        assert_eq!(
            result.dual_solution, original.dual_solution,
            "rejected candidate must restore constraint duals"
        );
        assert!(
            sign_after <= sign_before + 1e-15,
            "dual sign must not worsen: {sign_before:.3e} -> {sign_after:.3e}"
        );
        assert!(
            worst_after <= worst_before + 1e-15,
            "composite optimality residual must not worsen: {worst_before:.3e} -> {worst_after:.3e}"
        );
    }

    /// Stall detection: extended IR must terminate when corrections are
    /// at the noise floor.
    #[test]
    fn extended_ir_stall_detection() {
        assert!(
            !ext_score_made_progress(1e-1, 1e-1 - 1e-13),
            "sub-noise-floor drops must be detected as stall"
        );
        assert!(
            ext_score_made_progress(1e-1, 5e-2),
            "large drops must be recognized as progress"
        );
    }
}
