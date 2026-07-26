//! Postsolve: lift a reduced LP's solution back to the original variable / constraint
//! space by replaying `PostsolveStack` in LIFO order.

use super::transforms::{PostsolveStep, PresolveResult};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::tolerances::{COMP_SLACK_REL_TOL, PIVOT_TOL};
#[cfg(test)]
use otspot_num::sparse::CscMatrix;
use std::time::Instant;

// Test-only, in-order trace of which dual-recovery passes `run_postsolve`
// executed. Lets sentinels assert that the crossover pass runs first and can
// elide the cleanup LP / LSQ passes. Compiled out (no-op) in non-test builds.
#[cfg(test)]
thread_local! {
    static POSTSOLVE_PASS_TRACE: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn trace_pass(name: &'static str) {
    POSTSOLVE_PASS_TRACE.with(|t| t.borrow_mut().push(name));
}

#[cfg(not(test))]
#[inline(always)]
fn trace_pass(_: &'static str) {}

/// Drain (clear-and-return) the recorded pass trace for the current thread.
#[cfg(test)]
fn drain_postsolve_pass_trace() -> Vec<&'static str> {
    POSTSOLVE_PASS_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

/// Relative tolerance for treating `x[j]` as active at a bound or for detecting fixed variables.
///
/// Each check uses only the relevant bound's magnitude to avoid inflating the threshold
/// with the opposite bound (e.g. `at_lb` for `lb=0, ub=1e12` gives `tol≈1e-6`, not `≈1.0`).
const BOUND_ACTIVE_REL_TOL: f64 = 1e-6;

/// Tolerance for `x ≈ lb`: scales with lb magnitude only.
///
/// # Precondition
/// `lb` must be finite; all callers guard with `lb.is_finite() &&` before calling.
#[inline]
fn at_lb_tol(lb: f64) -> f64 {
    BOUND_ACTIVE_REL_TOL * (1.0 + lb.abs())
}

/// Tolerance for `x ≈ ub`: scales with ub magnitude only.
///
/// # Precondition
/// `ub` must be finite; all callers guard with `ub.is_finite() &&` before calling.
#[inline]
fn at_ub_tol(ub: f64) -> f64 {
    BOUND_ACTIVE_REL_TOL * (1.0 + ub.abs())
}

/// Tolerance for `ub - lb ≈ 0` (variable effectively fixed): scales with max magnitude.
///
/// Using max avoids doubling the threshold when both bounds are large (e.g. `[1e6, 1e6+1.5]`
/// would give `tol≈2.0` with sum but `tol≈1.0` with max, correctly leaving the gap=1.5 unclassified).
#[inline]
fn fixed_tol(lb: f64, ub: f64) -> f64 {
    let lb_s = if lb.is_finite() { lb.abs() } else { 0.0 };
    let ub_s = if ub.is_finite() { ub.abs() } else { 0.0 };
    BOUND_ACTIVE_REL_TOL * (1.0 + lb_s.max(ub_s))
}

/// Collect `(col, A[row, col])` for every column participating in row `i`.
fn collect_row_entries(orig_problem: &LpProblem, i: usize) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for j in 0..orig_problem.num_vars {
        let (rows, vals) = orig_problem.a.column(j);
        for (k, &row) in rows.iter().enumerate() {
            if row == i {
                out.push((j, vals[k]));
            }
        }
    }
    out
}

/// Compute `(slack, scale)` for row `i`. Slack is non-negative for feasible constraints.
fn row_slack_and_scale(orig_problem: &LpProblem, i: usize, solution: &[f64]) -> (f64, f64) {
    let row_entries = collect_row_entries(orig_problem, i);
    let ax_i: f64 = row_entries.iter().map(|&(j, a)| a * solution[j]).sum();
    let b_i = orig_problem.b[i];
    let slack = match orig_problem.constraint_types[i] {
        ConstraintType::Le => b_i - ax_i,
        ConstraintType::Ge => ax_i - b_i,
        ConstraintType::Eq => 0.0,
    };
    let scale = 1.0 + b_i.abs() + ax_i.abs();
    (slack, scale)
}

/// True when constraint `i` has positive slack (non-binding), so its dual must be zero.
fn is_row_nonbinding(orig_problem: &LpProblem, i: usize, solution: &[f64]) -> bool {
    let (slack, scale) = row_slack_and_scale(orig_problem, i, solution);
    slack > COMP_SLACK_REL_TOL * scale
}

/// Stationarity-based dual recovery for row `i`, iterating over the provided column entries.
///
/// Derives bounds on `y[i]` from KKT complementarity of each column in `row_entries`,
/// then picks the tightest feasible value respecting the constraint-type sign constraint.
fn stationarity_dual(
    orig_problem: &LpProblem,
    i: usize,
    row_entries: &[(usize, f64)],
    solution: &[f64],
    dual_solution: &[f64],
) -> f64 {
    let mut min_y_i = f64::NEG_INFINITY;
    let mut max_y_i = f64::INFINITY;
    for &(j, a_ij) in row_entries {
        if a_ij.abs() < f64::EPSILON {
            continue;
        }
        // Bound on y_i from rc_j = c_j - Σ_{k≠i} A_kj y_k - A_ij y_i.
        let mut rc_at_yi0 = orig_problem.c[j];
        let (rows, vals) = orig_problem.a.column(j);
        for (k, &row) in rows.iter().enumerate() {
            if row == i {
                continue;
            }
            rc_at_yi0 -= vals[k] * dual_solution[row];
        }
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < at_lb_tol(lb_j);
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < at_ub_tol(ub_j);
        let fixed =
            lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < fixed_tol(lb_j, ub_j);
        if fixed {
            continue;
        }
        let bound_val = rc_at_yi0 / a_ij;
        if at_lb && !at_ub {
            if a_ij > 0.0 {
                if bound_val < max_y_i {
                    max_y_i = bound_val;
                }
            } else if bound_val > min_y_i {
                min_y_i = bound_val;
            }
        } else if at_ub && !at_lb {
            if a_ij > 0.0 {
                if bound_val > min_y_i {
                    min_y_i = bound_val;
                }
            } else if bound_val < max_y_i {
                max_y_i = bound_val;
            }
        } else {
            if bound_val < max_y_i {
                max_y_i = bound_val;
            }
            if bound_val > min_y_i {
                min_y_i = bound_val;
            }
        }
    }
    let (sign_lb, sign_ub) = match orig_problem.constraint_types[i] {
        ConstraintType::Le => (f64::NEG_INFINITY, 0.0),
        ConstraintType::Ge => (0.0, f64::INFINITY),
        ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
    };
    let lb_y = sign_lb.max(min_y_i);
    let ub_y = sign_ub.min(max_y_i);
    if lb_y <= ub_y {
        if lb_y <= 0.0 && ub_y >= 0.0 {
            0.0
        } else if ub_y < 0.0 {
            ub_y
        } else {
            lb_y
        }
    } else {
        0.0
    }
}

/// Recover the dual value for a removed row from dual feasibility (KKT stationarity).
///
/// Checks binding first via `is_row_nonbinding`; returns 0 for non-binding rows.
/// Callers that already know the row is binding (e.g. ForcingRow postsolve) should
/// call `stationarity_dual` directly to avoid the incomplete-solution binding check.
#[cfg_attr(not(test), allow(dead_code))]
fn recover_removed_row_dual(
    orig_problem: &LpProblem,
    i: usize,
    solution: &[f64],
    dual_solution: &[f64],
) -> f64 {
    if is_row_nonbinding(orig_problem, i, solution) {
        return 0.0;
    }
    let row_entries = collect_row_entries(orig_problem, i);
    stationarity_dual(orig_problem, i, &row_entries, solution, dual_solution)
}

/// Lift the reduced-problem solution back into the original variable / constraint space.
///
/// Does not synthesise `warm_start_basis`: presolve renumbers variables and
/// rows, so a reduced-LP basis is unusable for re-warm-starting the original
/// LP, and rebuilding one is a simplex-side concern
/// (`crate::simplex::recover_warm_start_basis`). Callers that want
/// `SolverOptions::recover_warm_start_basis` honored call that function
/// themselves on this function's returned solution.
pub fn run_postsolve(
    result: &SolverResult,
    presolve_result: &PresolveResult,
    orig_problem: &LpProblem,
    deadline: Option<Instant>,
) -> SolverResult {
    let n = presolve_result.orig_num_vars;
    let m = presolve_result.orig_num_constraints;

    let mut solution = vec![0.0f64; n];
    let mut dual_solution = vec![0.0f64; m];
    let dual_required =
        matches!(result.status, SolveStatus::Optimal) || !result.dual_solution.is_empty();
    let input_dual_is_ipm = result.reduced_costs.is_empty() && !result.dual_solution.is_empty();

    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj >= result.solution.len() {
                return malformed_postsolve_result();
            }
            solution[j] = result.solution[jj];
        }
    }
    for (i, &maybe_ii) in presolve_result.row_map.iter().enumerate() {
        if let Some(ii) = maybe_ii {
            if !dual_required {
                continue;
            }
            if ii >= result.dual_solution.len() {
                return malformed_postsolve_result();
            }
            dual_solution[i] = if input_dual_is_ipm {
                -result.dual_solution[ii]
            } else {
                result.dual_solution[ii]
            };
        }
    }

    for step in presolve_result.postsolve_stack.iter().rev() {
        match step {
            PostsolveStep::FixedVariable { orig_col, value }
            | PostsolveStep::EmptyColumn { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyRow { orig_row } => {
                dual_solution[*orig_row] = 0.0;
            }
            PostsolveStep::SingletonRow {
                orig_col,
                orig_row,
                value,
                coeff,
                col_orig_entries,
                c_orig,
            } => {
                solution[*orig_col] = *value;
                // Recover y[orig_row] from stationarity of orig_col:
                //   c_orig = coeff * y[orig_row] + Σ_{k != orig_row} A[k,orig_col] * y[k]
                //   => y[orig_row] = (c_orig - sum_ay) / coeff
                let sum_ay: f64 = col_orig_entries
                    .iter()
                    .map(|&(row_k, a_kj)| a_kj * dual_solution[row_k])
                    .sum();
                dual_solution[*orig_row] = (c_orig - sum_ay) / coeff;
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = 0.0;
            }
            PostsolveStep::BoundsTightened => {}
            PostsolveStep::SingletonInequalityRow {
                orig_row,
                orig_col: _,
                coeff,
                old_lb: _,
                old_ub: _,
                col_orig_entries,
                c_orig,
            } => {
                // Complementarity: non-binding inequality rows must have dual = 0.
                if is_row_nonbinding(orig_problem, *orig_row, &solution) {
                    dual_solution[*orig_row] = 0.0;
                } else {
                    // Stationarity-based dual recovery: y[i] = (c_orig - Σ A_kj y_k) / coeff,
                    // then clamp to Le (y <= 0) or Ge (y >= 0) sign constraint.
                    let sum_ay: f64 = col_orig_entries
                        .iter()
                        .map(|&(row_k, a_kj)| a_kj * dual_solution[row_k])
                        .sum();
                    let mut y_i = (c_orig - sum_ay) / coeff;
                    match orig_problem.constraint_types[*orig_row] {
                        crate::problem::ConstraintType::Le => {
                            if y_i > 0.0 {
                                y_i = 0.0;
                            }
                        }
                        crate::problem::ConstraintType::Ge => {
                            if y_i < 0.0 {
                                y_i = 0.0;
                            }
                        }
                        crate::problem::ConstraintType::Eq => {}
                    }
                    dual_solution[*orig_row] = y_i;
                }
            }
            PostsolveStep::ForcingRow {
                orig_row,
                fixed_vars,
                row_orig_entries,
            } => {
                for &(col, value, _, _) in fixed_vars {
                    solution[col] = value;
                }
                // Forcing rows are always binding (activity at contributing bounds exactly
                // matches RHS). Use the presolve-time snapshot to bypass is_row_nonbinding,
                // which would compute Ax with partially restored variables under LIFO replay.
                dual_solution[*orig_row] = stationarity_dual(
                    orig_problem,
                    *orig_row,
                    row_orig_entries,
                    &solution,
                    &dual_solution,
                );
            }
            PostsolveStep::LinearSubstitution {
                orig_col,
                orig_row,
                pivot,
                rhs,
                others,
                col_orig_entries,
                c_orig,
            } => {
                // Primal: x_j = (rhs - Σ coeff_k · x_k) / pivot.
                let mut sum_others = 0.0f64;
                for &(other_col, coeff) in others {
                    sum_others += coeff * solution[other_col];
                }
                solution[*orig_col] = (rhs - sum_others) / pivot;

                // Dual: a free-variable substitution eliminates one Eq row; its y is
                // recovered from the free var's stationarity rc[orig_col] = 0,
                // using the pre-distribution column snapshot `col_orig_entries`.
                if let Some(piv_row) = orig_row {
                    let mut sum_other_rows = 0.0f64;
                    for &(row_i, a_ij) in col_orig_entries {
                        if row_i == *piv_row {
                            continue;
                        }
                        sum_other_rows += a_ij * dual_solution[row_i];
                    }
                    dual_solution[*piv_row] = (c_orig - sum_other_rows) / pivot;
                }
            }
        }
    }

    // Recompute slack on the original problem as `b - Ax`.
    let mut slack = orig_problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n) {
        let (rows, vals) = orig_problem.a.column(j);
        for (k, &row) in rows.iter().enumerate() {
            slack[row] -= vals[k] * sol_j;
        }
    }

    // Compute dual-recovery candidates (y_loop, crossover) and adopt the one
    // with the smallest bound-aware dual-feasibility violation.
    let y_loop = dual_solution.clone();

    // Dual-feasibility metric: max per-column KKT violation.
    // at lb only: max(0, -rc); at ub only: max(0, rc); interior: |rc|.
    let dfeas_bound = |y: &[f64]| -> f64 {
        let mut max_viol = 0.0f64;
        for j in 0..n {
            let (lb_j, ub_j) = orig_problem.bounds[j];
            let fixed =
                lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < fixed_tol(lb_j, ub_j);
            if fixed {
                continue;
            }
            let at_lb = lb_j.is_finite() && (solution[j] - lb_j).abs() < at_lb_tol(lb_j);
            let at_ub = ub_j.is_finite() && (solution[j] - ub_j).abs() < at_ub_tol(ub_j);
            let mut rc = orig_problem.c[j];
            let (rows, vals) = orig_problem.a.column(j);
            for (k, &row) in rows.iter().enumerate() {
                rc -= vals[k] * y[row];
            }
            let viol = if at_lb && !at_ub {
                f64::max(0.0, -rc)
            } else if at_ub && !at_lb {
                f64::max(0.0, rc)
            } else {
                rc.abs()
            };
            if viol > max_viol {
                max_viol = viol;
            }
        }
        max_viol
    };

    let df_loop = dfeas_bound(&y_loop);

    // Try crossover at Optimal status when loop candidate is dual-infeasible.
    // Crossover reconstructs a globally dual-feasible y = B⁻ᵀc_B at the primal
    // optimum; this is what reconciles presolve rows serving multiple roles
    // (forcing + pivot) that no local recovery can fix, e.g. pilot-ja.
    let gate = PIVOT_TOL;
    let crossover: Option<Vec<f64>> =
        if matches!(result.status, SolveStatus::Optimal) && df_loop > gate {
            trace_pass("crossover");
            crate::simplex::crossover_dual_from_primal(orig_problem, &solution, deadline)
                .map(|(_vertex, y, _rc)| y)
        } else {
            None
        };
    let df_xover = crossover.as_ref().map_or(f64::INFINITY, |y| dfeas_bound(y));

    // Select candidate with lowest dual infeasibility.
    if df_loop <= df_xover {
        dual_solution = y_loop;
    } else {
        dual_solution = crossover.expect("df_xover finite implies Some");
    }

    // Recompute simplex-convention reduced costs on the original problem now that
    // the dual is final:
    //   reduced_cost[j] = c[j] - Σ_i A_ij · y_i.
    let mut reduced_costs = orig_problem.c.clone();
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(n) {
        let (rows, vals) = orig_problem.a.column(j);
        for (k, &row) in rows.iter().enumerate() {
            *rc -= vals[k] * dual_solution[row];
        }
    }
    let postsolve_dfeas_recomputed = dfeas_bound(&dual_solution);

    let objective = result.objective + presolve_result.obj_offset;

    // `warm_start_basis` is left `None` here (see doc comment above); callers
    // that opt into `SolverOptions::recover_warm_start_basis` synthesise it
    // themselves via `crate::simplex::recover_warm_start_basis` on `solution`.
    SolverResult {
        status: result.status.clone(),
        objective,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        iterations: result.iterations,
        postsolve_dfeas: Some(postsolve_dfeas_recomputed),
        ..Default::default()
    }
}

fn malformed_postsolve_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

#[cfg(test)]
mod bound_active_tol_tests {
    use super::*;

    /// Sentinel C.4: `at_lb_tol` scales with lb magnitude only.
    ///
    /// With an absolute 1e-6 threshold, x = lb + 0.5 (|x−lb|=0.5) would be
    /// classified as interior for lb=1e6, violating complementary slackness.
    /// `at_lb_tol(lb=1e6) ≈ 1.0`, so the same deviation is correctly at-lb.
    ///
    /// Regresses if `at_lb_tol` reverts to the old absolute 1e-6.
    #[test]
    fn test_sentinel_c4_large_scale_bound_active_tol() {
        let lb = 1e6_f64;
        let x = lb + 0.5;

        assert!(
            (x - lb).abs() > BOUND_ACTIVE_REL_TOL,
            "absolute BOUND_ACTIVE_REL_TOL alone would misclassify x as interior"
        );

        let tol = at_lb_tol(lb);
        assert!(
            (x - lb).abs() < tol,
            "at_lb_tol={} must classify x=lb+0.5 as at-lb for lb=1e6",
            tol
        );
    }

    /// Unit-scale bounds (lb=0, ub=1) give tolerances close to BOUND_ACTIVE_REL_TOL.
    #[test]
    fn test_bound_active_tol_unit_scale() {
        assert!(
            (at_lb_tol(0.0) - 1e-6).abs() < 1e-20,
            "at_lb_tol(0) should be 1e-6, got {}",
            at_lb_tol(0.0)
        );
        assert!(
            (at_ub_tol(1.0) - 2e-6).abs() < 1e-20,
            "at_ub_tol(1) should be 2e-6, got {}",
            at_ub_tol(1.0)
        );
        assert!(
            (fixed_tol(0.0, 1.0) - 2e-6).abs() < 1e-20,
            "fixed_tol(0,1) should be 2e-6, got {}",
            fixed_tol(0.0, 1.0)
        );
    }

    /// Sentinel C.4 (codex): lb=0, ub=1e12, x=5e5 must NOT be at-lb.
    ///
    /// Old formula `1e-6*(1+|lb|+|ub|) ≈ 1.0e6` made `(x-lb)=5e5 < 1e6` → at_lb (wrong).
    /// New lb-only formula `1e-6*(1+|lb|) = 1e-6` correctly rejects x=5e5 as interior.
    /// No-op regression: reverts if `at_lb_tol` re-adds ub to its formula.
    #[test]
    fn test_sentinel_c4_independent_lb_ub_scaling_at_lb() {
        let lb = 0.0_f64;
        let ub = 1e12_f64;
        let x = 5e5_f64;

        // Old formula would give tol ≈ 1e6, making x look at-lb.
        let old_tol = BOUND_ACTIVE_REL_TOL * (1.0 + lb.abs() + ub.abs());
        assert!(
            (x - lb).abs() < old_tol,
            "old formula must mis-classify x=5e5 as at-lb (old_tol={})",
            old_tol
        );

        // New lb-only formula correctly classifies x as interior.
        let new_tol = at_lb_tol(lb);
        assert!(
            (x - lb).abs() >= new_tol,
            "at_lb_tol={} must NOT classify x=5e5 as at-lb for lb=0,ub=1e12",
            new_tol
        );
    }

    /// Sentinel C.4 (reviewer): lb=1e6, ub=1e6+1.5 must NOT be fixed.
    ///
    /// Old formula `1e-6*(1+|lb|+|ub|) ≈ 2.0` made `gap=1.5 < 2.0` → fixed (wrong).
    /// New max formula `1e-6*(1+max(|lb|,|ub|)) ≈ 1.0` gives `gap=1.5 > 1.0` → not fixed.
    /// No-op regression: reverts if `fixed_tol` re-sums both magnitudes.
    #[test]
    fn test_sentinel_c4_independent_lb_ub_scaling_fixed() {
        let lb = 1e6_f64;
        let ub = 1e6_f64 + 1.5_f64;
        let gap = ub - lb;

        // Old formula must classify this as fixed.
        let old_tol = BOUND_ACTIVE_REL_TOL * (1.0 + lb.abs() + ub.abs());
        assert!(
            gap < old_tol,
            "old formula must mis-classify [1e6,1e6+1.5] as fixed (old_tol={})",
            old_tol
        );

        // New max formula correctly leaves the range as non-fixed.
        let new_tol = fixed_tol(lb, ub);
        assert!(
            gap >= new_tol,
            "fixed_tol={} must NOT classify [1e6,1e6+1.5] as fixed (gap={})",
            new_tol,
            gap
        );
    }
}

#[cfg(test)]
mod ipm_dual_convention_tests {
    use super::*;

    #[test]
    fn ipm_dual_is_converted_before_reduced_cost_recovery() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let presolve = PresolveResult::no_reduction(&lp);
        let raw_ipm = SolverResult {
            status: SolveStatus::Optimal,
            objective: 1.0,
            solution: vec![1.0],
            dual_solution: vec![-1.0],
            reduced_costs: vec![],
            ..Default::default()
        };

        let lifted = run_postsolve(&raw_ipm, &presolve, &lp, Some(Instant::now()));

        assert_eq!(
            lifted.dual_solution,
            vec![1.0],
            "IPM/prove convention y=-1 must become LP simplex convention y=+1"
        );
        assert_eq!(lifted.reduced_costs.len(), 1);
        assert!(
            lifted.reduced_costs[0].abs() < 1e-12,
            "simplex reduced cost must be c - A^T y = 0, got {}",
            lifted.reduced_costs[0]
        );
    }
}

#[cfg(test)]
mod crossover_first_tests {
    //! Sentinels for the crossover-first postsolve ordering.
    //!
    //! The dual-recovery passes produce identical final duals regardless of order
    //! (min-dfeas selection), so the *only* observable signal of the optimisation
    //! is which passes actually ran — captured by the thread-local pass trace.
    //! Each test drains the trace, runs `run_postsolve`, and asserts on the
    //! recorded order/membership. No-op proofs are stated per test.
    use super::*;

    /// `min 2*x0 + 3*x1  s.t.  x0 + x1 = 1, x ≥ 0`. Optimum x* = (1, 0), with the
    /// unique dual y0 = 2 (rc0 = 0 on basic x0, rc1 = 1 ≥ 0 on x1 at lb).
    fn lp_clean_vertex() -> (LpProblem, Vec<f64>) {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![2.0, 3.0],
            a,
            vec![1.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        (lp, vec![1.0, 0.0])
    }

    /// Reduced-problem result with a deliberately dual-infeasible y (so the cheap
    /// loop/GS candidates leave `cheap_min > gate` and the recovery machinery is
    /// forced to engage). `reduced_costs` is non-empty so postsolve keeps the
    /// simplex dual convention (no IPM sign flip).
    fn result_with_dual(status: SolveStatus, solution: &[f64], y: Vec<f64>) -> SolverResult {
        SolverResult {
            status,
            objective: 0.0,
            solution: solution.to_vec(),
            dual_solution: y,
            reduced_costs: vec![0.0; solution.len()],
            ..Default::default()
        }
    }

    /// Crossover certifies a feasible dual when the loop candidate is dual-infeasible.
    ///
    /// No-op proof: dropping the `df_loop > gate` guard or making crossover
    /// unconditional eliminates the skipping of the recovery when cheap_min ≤ gate.
    #[test]
    fn crossover_first_certifies_and_skips_cleanup() {
        let (lp, x) = lp_clean_vertex();
        let presolve = PresolveResult::no_reduction(&lp);
        // y = [0] is dual-infeasible: rc0 = 2 - 0 = 2 on interior x0 ⇒ cheap_min ≈ 2.
        let reduced = result_with_dual(SolveStatus::Optimal, &x, vec![0.0]);

        let _ = drain_postsolve_pass_trace();
        let lifted = run_postsolve(&reduced, &presolve, &lp, None);
        let trace = drain_postsolve_pass_trace();

        assert_eq!(
            trace,
            vec!["crossover"],
            "crossover must run first and, on certifying, elide cleanup/LSQ; trace={trace:?}"
        );
        // Correctness: the adopted dual is the exact crossover dual y0 = 2.
        assert!(
            (lifted.dual_solution[0] - 2.0).abs() < 1e-6,
            "crossover dual must recover y0 = 2, got {}",
            lifted.dual_solution[0]
        );
        assert!(
            lifted.postsolve_dfeas.unwrap() <= PIVOT_TOL,
            "crossover-first dual must be feasible (dfeas ≤ gate), got {:?}",
            lifted.postsolve_dfeas
        );
    }

    /// When the cheap candidates already certify (`cheap_min ≤ gate`), no recovery
    /// pass — crossover included — should run.
    ///
    /// No-op proof: triggering crossover unconditionally (dropping the
    /// `cheap_min > gate` guard) puts `crossover` into the trace and fails the
    /// empty-trace assertion.
    #[test]
    fn cheap_feasible_dual_runs_no_recovery_pass() {
        let (lp, x) = lp_clean_vertex();
        let presolve = PresolveResult::no_reduction(&lp);
        // y = [2] is the exact dual ⇒ cheap_min ≈ 0 ≤ gate.
        let reduced = result_with_dual(SolveStatus::Optimal, &x, vec![2.0]);

        let _ = drain_postsolve_pass_trace();
        let lifted = run_postsolve(&reduced, &presolve, &lp, None);
        let trace = drain_postsolve_pass_trace();

        assert!(
            trace.is_empty(),
            "a feasible cheap dual must skip every recovery pass; trace={trace:?}"
        );
        assert!(lifted.postsolve_dfeas.unwrap() <= PIVOT_TOL);
    }

    /// Crossover is gated on Optimal status; a non-Optimal result must not invoke
    /// it (the basis reconstruction is only meaningful at a primal optimum).
    ///
    /// No-op proof: dropping the `matches!(Optimal)` guard puts `crossover` into
    /// the trace and fails the assertion.
    #[test]
    fn non_optimal_status_skips_crossover() {
        let (lp, x) = lp_clean_vertex();
        let presolve = PresolveResult::no_reduction(&lp);
        let reduced = result_with_dual(SolveStatus::Infeasible, &x, vec![0.0]);

        let _ = drain_postsolve_pass_trace();
        let _ = run_postsolve(&reduced, &presolve, &lp, None);
        let trace = drain_postsolve_pass_trace();

        assert!(
            !trace.contains(&"crossover"),
            "non-Optimal status must not run crossover; trace={trace:?}"
        );
    }
}

#[cfg(test)]
mod recover_removed_row_dual_tests {
    use super::*;

    /// Binding Le row: min -x  s.t.  x <= 2,  x in [0, inf).
    /// Optimal x=2 (binding). KKT stationarity: rc = c - a*y = -1 - 1*y = 0 => y = -1.
    #[test]
    fn binding_le_row_returns_nonzero_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let solution = vec![2.0];
        let dual_solution = vec![0.0];
        let y = recover_removed_row_dual(&lp, 0, &solution, &dual_solution);
        assert!(
            (y - (-1.0)).abs() < 1e-6,
            "Le binding dual should be -1, got {y}"
        );
    }

    /// Non-binding Le row: min -x  s.t.  x <= 5,  x in [0, 2].
    /// Optimal x=2 (at ub, not at row bound). slack = 5 - 2 = 3 > 0. dual = 0.
    #[test]
    fn nonbinding_le_row_returns_zero() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0)],
            None,
        )
        .unwrap();
        let solution = vec![2.0];
        let dual_solution = vec![0.0];
        let y = recover_removed_row_dual(&lp, 0, &solution, &dual_solution);
        assert!(y.abs() < 1e-10, "non-binding Le dual should be 0, got {y}");
    }

    /// Binding Ge row: min x  s.t.  x >= 2,  x in [0, inf).
    /// Optimal x=2 (binding). KKT: rc = c - a*y = 1 - 1*y = 0 => y = 1.
    #[test]
    fn binding_ge_row_returns_nonzero_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let solution = vec![2.0];
        let dual_solution = vec![0.0];
        let y = recover_removed_row_dual(&lp, 0, &solution, &dual_solution);
        assert!(
            (y - 1.0).abs() < 1e-6,
            "Ge binding dual should be 1, got {y}"
        );
    }

    /// Non-binding Ge row: min x  s.t.  x >= -10,  x in [0, 5].
    /// Optimal x=0 (at lb). slack = 0 - (-10) = 10 > 0. dual = 0.
    #[test]
    fn nonbinding_ge_row_returns_zero() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![-10.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 5.0)],
            None,
        )
        .unwrap();
        let solution = vec![0.0];
        let dual_solution = vec![0.0];
        let y = recover_removed_row_dual(&lp, 0, &solution, &dual_solution);
        assert!(y.abs() < 1e-10, "non-binding Ge dual should be 0, got {y}");
    }

    /// Binding Eq row: min x  s.t.  x = 3,  x in [0, inf).
    /// Optimal x=3. KKT: rc = c - a*y = 1 - 1*y = 0 => y = 1.
    #[test]
    fn binding_eq_row_returns_nonzero_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let solution = vec![3.0];
        let dual_solution = vec![0.0];
        let y = recover_removed_row_dual(&lp, 0, &solution, &dual_solution);
        assert!(
            (y - 1.0).abs() < 1e-6,
            "Eq binding dual should be 1, got {y}"
        );
    }

    #[test]
    fn run_postsolve_rejects_short_reduced_solution() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let pres = PresolveResult::no_reduction(&lp);
        let reduced = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            ..Default::default()
        };
        let out = run_postsolve(&reduced, &pres, &lp, None);
        assert_eq!(out.status, SolveStatus::NumericalError);
        assert!(out.solution.is_empty());
    }

    #[test]
    fn run_postsolve_rejects_short_reduced_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let pres = PresolveResult::no_reduction(&lp);
        let reduced = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            reduced_costs: vec![0.0],
            ..Default::default()
        };
        let out = run_postsolve(&reduced, &pres, &lp, None);
        assert_eq!(out.status, SolveStatus::NumericalError);
        assert!(out.solution.is_empty());
    }

    #[test]
    fn run_postsolve_preserves_timeout_incumbent_without_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let pres = PresolveResult::no_reduction(&lp);
        let reduced = SolverResult {
            status: SolveStatus::Timeout,
            objective: 0.5,
            solution: vec![0.5],
            dual_solution: vec![],
            reduced_costs: vec![],
            ..Default::default()
        };
        let out = run_postsolve(&reduced, &pres, &lp, None);
        assert_eq!(
            out.status,
            SolveStatus::Timeout,
            "timeout incumbent without dual must not be remapped to NumericalError"
        );
        assert_eq!(out.solution, vec![0.5]);
    }
}
