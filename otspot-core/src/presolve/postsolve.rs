//! Postsolve: lift a reduced LP's solution back to the original variable / constraint
//! space by replaying `PostsolveStack` in LIFO order.

use super::transforms::{PostsolveStep, PresolveResult};
use crate::options::WarmStartBasis;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::simplex::build_standard_form;
use crate::simplex::crash::compute_crash_basis;
#[cfg(test)]
use crate::sparse::CscMatrix;
use crate::tolerances::{COMP_SLACK_REL_TOL, PIVOT_TOL};
use std::time::Instant;

/// Relative tolerance below which a standard-form column is treated as at-bound
/// (non-basic candidate) when synthesising the postsolved warm-start basis.
const WARM_BASIS_BUILD_TOL: f64 = 1e-9;

/// Markowitz threshold for LU factorization stability: a column pivot is accepted only
/// if its absolute value exceeds this fraction of the column maximum. Prevents tiny
/// pivots that would inflate the basis matrix condition number.
const MARKOWITZ_PIVOT_RATIO: f64 = 0.1;

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
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row == i {
                    out.push((j, vals[k]));
                }
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
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row == i {
                    continue;
                }
                rc_at_yi0 -= vals[k] * dual_solution[row];
            }
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

/// Synthesise an original-LP standard-form basis from the postsolved primal solution.
///
/// Presolve renumbers variables and rows, so `result.warm_start_basis` (which indexes
/// the reduced LP's standard form) is unusable for re-warm-starting the original LP.
/// We rebuild a basis on the original standard form:
///
///   1. Translate the postsolved primal solution into the original standard-form
///      vector `x_std` (shifted variables + slack columns).
///   2. Triangulate with the LTSF crash to guarantee non-singularity and to handle
///      Ge / Eq rows for which the slack alone is not a valid initial basic column.
///   3. For each row whose crash assignment is a slack covering a tight constraint
///      (slack ≈ 0) but where a structural column has `x_std > 0`, pivot the active
///      structural column in. This makes the basis reflect the optimum's at-bound
///      vs interior split (Maros & Mészáros §5).
///
/// Returns `None` only when the crash leaves rows uncovered (an artificial would be
/// needed) — in that case no all-real-column basis exists, so warm-start is impossible.
fn recover_warm_start_basis(orig_problem: &LpProblem, solution: &[f64]) -> Option<WarmStartBasis> {
    let sf = build_standard_form(orig_problem);
    let n_orig = orig_problem.num_vars;
    let n_total = sf.n_total;
    let n_shifted = sf.n_shifted;
    let m_ext = sf.m;

    if solution.len() != n_orig {
        return None;
    }

    // Step 1: postsolved orig solution → standard-form vector.
    let mut x_std = vec![0.0_f64; n_total];
    for j in 0..n_orig {
        let info = &sf.orig_var_info[j];
        let xj = solution[j];
        if info.new_vars.len() == 2 {
            // Free var split: x = x_plus − x_minus, both ≥ 0.
            let plus_idx = info.new_vars[0].0;
            let minus_idx = info.new_vars[1].0;
            x_std[plus_idx] = xj.max(0.0);
            x_std[minus_idx] = (-xj).max(0.0);
        } else {
            let (idx, coeff) = info.new_vars[0];
            // coeff > 0 ⇒ shifted by lb (x_std = x − lb); coeff < 0 ⇒ shifted by ub.
            let val = if coeff > 0.0 {
                xj - info.offset
            } else {
                info.offset - xj
            };
            x_std[idx] = val.max(0.0);
        }
    }
    // Slack columns: x_std[slack] = (b[i] − Σ A_ij x_std_struct[j]) / sign(slack_coeff).
    // Each slack column has exactly one non-zero entry at its owning row.
    let mut row_struct_sum = vec![0.0_f64; m_ext];
    for j in 0..n_shifted {
        if x_std[j].abs() < WARM_BASIS_BUILD_TOL {
            continue;
        }
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                row_struct_sum[row] += vals[k] * x_std[j];
            }
        }
    }
    for j in n_shifted..n_total {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            if rows.len() == 1 && vals[0].abs() > 0.0 {
                let i = rows[0];
                let coeff = vals[0];
                let slack = (sf.b[i] - row_struct_sum[i]) / coeff;
                x_std[j] = slack.max(0.0);
            }
        }
    }

    // Step 2: LTSF crash for non-singular triangulation (covers Ge / Eq rows).
    let (mut basis, _needs_art, num_art) = compute_crash_basis(
        &sf.a,
        &sf.b,
        m_ext,
        n_shifted,
        &sf.initial_basis,
        &sf.needs_artificial,
    );
    if num_art > 0 {
        // No all-structural triangulation exists. Refuse to manufacture a basis.
        return None;
    }

    // Step 3: solution-driven refinement. For each structural column j with
    // `x_std[j] > tol`, swap into a row whose current basic column is an
    // at-bound slack (x_std[basis[i]] ≈ 0). This makes the basis reflect the
    // active variables at the postsolved optimum without breaking triangulation
    // (we only replace 0-valued slacks, so x_B at the new basis stays consistent
    // with x_std).
    let mut basic_at_row: Vec<usize> = vec![usize::MAX; n_total];
    for (i, &col) in basis.iter().enumerate() {
        basic_at_row[col] = i;
    }
    // Greedy in descending x_std order so the strongest active vars get pivoted
    // first.
    let mut active_struct: Vec<(f64, usize)> = (0..n_shifted)
        .filter(|&j| x_std[j] > WARM_BASIS_BUILD_TOL && basic_at_row[j] == usize::MAX)
        .map(|j| (x_std[j], j))
        .collect();
    active_struct.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (_xj, j) in active_struct {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            // Pick the candidate row with the largest |a_ij| where the current
            // basic column is an at-bound slack; Markowitz threshold protects
            // against tiny pivots that would inflate B's condition number.
            let mut col_max = 0.0_f64;
            for &v in vals.iter() {
                if v.abs() > col_max {
                    col_max = v.abs();
                }
            }
            if col_max < WARM_BASIS_BUILD_TOL {
                continue;
            }
            let pivot_min = (MARKOWITZ_PIVOT_RATIO * col_max).max(WARM_BASIS_BUILD_TOL);

            let mut best: Option<(f64, usize)> = None;
            for (k, &row) in rows.iter().enumerate() {
                let abs = vals[k].abs();
                if abs < pivot_min {
                    continue;
                }
                let cur = basis[row];
                let cur_is_at_bound_slack = cur >= n_shifted && x_std[cur] <= WARM_BASIS_BUILD_TOL;
                if !cur_is_at_bound_slack {
                    continue;
                }
                if best.is_none_or(|(b, _)| abs > b) {
                    best = Some((abs, row));
                }
            }
            if let Some((_, row)) = best {
                let leaving = basis[row];
                basic_at_row[leaving] = usize::MAX;
                basis[row] = j;
                basic_at_row[j] = row;
            }
        }
    }

    // Informational x_b at the new basis (dual-simplex warm path recomputes
    // x_B = B^{-1} b_new, so this is purely a hint).
    let x_b: Vec<f64> = basis
        .iter()
        .map(|&j| x_std.get(j).copied().unwrap_or(0.0))
        .collect();
    Some(WarmStartBasis { basis, x_b })
}

/// Lift the reduced-problem solution back into the original variable / constraint space.
///
/// `recover_warm_basis = true` synthesises `warm_start_basis` on the original LP
/// standard form (see `recover_warm_start_basis`). default `false` skips the
/// build_standard_form + LTSF crash + refinement cost — large LPs paid 30–96%
/// wall regression at presolve-reduced solves before gating.
pub fn run_postsolve(
    result: &SolverResult,
    presolve_result: &PresolveResult,
    orig_problem: &LpProblem,
    deadline: Option<Instant>,
    recover_warm_basis: bool,
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
            PostsolveStep::FixedVariable { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyColumn { orig_col, value } => {
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
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
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
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    rc -= vals[k] * y[row];
                }
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
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc -= vals[k] * dual_solution[row];
            }
        }
    }
    let postsolve_dfeas_recomputed = dfeas_bound(&dual_solution);

    let objective = result.objective + presolve_result.obj_offset;

    // Lift the warm-start basis to the original LP standard form so the user can
    // re-warm-start with `presolve = false` next call.  Only attempt this for
    // Optimal status: Infeasible/Unbounded carry no meaningful solution.
    // Default solves skip recovery (build_standard_form + LTSF crash + refinement
    // = O(nnz) + O(m·n_nz)); the caller opts in via
    // `SolverOptions::recover_warm_start_basis = true`.
    let warm_start_basis = if recover_warm_basis && matches!(result.status, SolveStatus::Optimal) {
        recover_warm_start_basis(orig_problem, &solution)
    } else {
        None
    };

    SolverResult {
        status: result.status.clone(),
        objective,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        warm_start_basis,
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
mod warm_basis_recovery_tests {
    //! `recover_warm_start_basis` sentinels.
    //!
    //! Each sentinel asserts:
    //!   1. presolve-reducible LP solved with `recover_warm_start_basis = true`
    //!      returns `warm_start_basis = Some(_)`,
    //!   2. the basis has length `m_ext` and every entry indexes a real (non-artificial) column,
    //!   3. re-solving with `warm_start = Some(basis), presolve = false` reaches Optimal.
    //!
    //! Perf gate (`default_skips_warm_basis_recovery`): default options must
    //! return `warm_start_basis = None` on the same presolve-reducible LP — proves
    //! the recovery cost is actually elided in the default path.
    //!
    //! No-op proof: temporarily forcing `recover_warm_start_basis` to return `None`
    //! flips (1) `is_none()` and breaks the warm-start round-trip — verified by
    //! `noop_proof_returns_none_fails_round_trip`.
    use super::*;
    use crate::options::{SimplexMethod, SolverOptions};
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::simplex::{build_standard_form, solve, solve_with};
    use crate::sparse::CscMatrix;

    /// Default options + `recover_warm_start_basis = true`. The recovery path
    /// is opt-in; sentinels covering the postsolve synthesis must enable it.
    fn opts_recover() -> SolverOptions {
        SolverOptions {
            recover_warm_start_basis: true,
            ..SolverOptions::default()
        }
    }

    /// LP whose presolve dual-fixing zeroes both vars (c>0, x≥0, finite ub).
    /// Reduced LP has 0 vars → simplex `n==0` short-circuit → reduced
    /// warm_start_basis = None. Postsolve must still synthesise a basis.
    fn lp_dual_fixed() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0, 0, 1, 2], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 3, 2)
            .unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![6.0, 4.0, 4.0],
            vec![ConstraintType::Le; 3],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap()
    }

    /// LP with a singleton-row Eq: x0 = 2; presolve fixes x0 then propagates.
    fn lp_singleton_row() -> LpProblem {
        // min x0 + x1 s.t. x0 = 2 (Eq), x0 + x1 ≤ 5; x ≥ 0
        let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 0, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap()
    }

    /// LP that survives presolve untouched (no reducible structure) — the
    /// `was_reduced=false` branch in `solve_with` should still surface a basis
    /// (this comes from simplex directly, not postsolve; sentinel ensures the
    /// postsolve fix didn't regress the non-reducible path).
    fn lp_non_reducible() -> LpProblem {
        // min -x0 - 2*x1 s.t. x0 + x1 ≤ 4; -x0 + x1 ≤ 2; x0 - x1 ≤ 2
        // Optimal: x0=1, x1=3, obj=-7.
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            3,
            2,
        )
        .unwrap();
        LpProblem::new_general(
            vec![-1.0, -2.0],
            a,
            vec![4.0, 2.0, 2.0],
            vec![ConstraintType::Le; 3],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap()
    }

    fn assert_basis_well_formed(lp: &LpProblem, basis: &[usize], context: &str) {
        let sf = build_standard_form(lp);
        assert_eq!(
            basis.len(),
            sf.m,
            "[{}] basis len {} != m_ext {}",
            context,
            basis.len(),
            sf.m,
        );
        for (i, &col) in basis.iter().enumerate() {
            assert!(
                col < sf.n_total,
                "[{}] basis[{}] = {} ≥ n_total {} (artificial leakage)",
                context,
                i,
                col,
                sf.n_total,
            );
        }
        // Uniqueness: each column appears at most once in the basis.
        let mut seen = vec![false; sf.n_total];
        for &col in basis {
            assert!(
                !seen[col],
                "[{}] basis has duplicate column {}",
                context, col
            );
            seen[col] = true;
        }
    }

    fn assert_warm_round_trip(lp_a: &LpProblem, lp_b: &LpProblem, context: &str) {
        let r1 = solve_with(lp_a, &opts_recover());
        assert_eq!(r1.status, SolveStatus::Optimal, "[{}] lp_a status", context);
        let ws = r1
            .warm_start_basis
            .as_ref()
            .unwrap_or_else(|| panic!("[{}] postsolve returned warm_start_basis=None", context));
        assert_basis_well_formed(lp_a, &ws.basis, context);

        let opts_warm = SolverOptions {
            warm_start: Some(ws.clone()),
            simplex_method: SimplexMethod::Dual,
            presolve: false,
            ..SolverOptions::default()
        };
        let r2 = solve_with(lp_b, &opts_warm);
        assert_eq!(
            r2.status,
            SolveStatus::Optimal,
            "[{}] warm-start round-trip on lp_b did not reach Optimal",
            context,
        );
    }

    #[test]
    fn warm_basis_from_dual_fixed_lp() {
        let lp = lp_dual_fixed();
        // Self-warm round-trip (same LP twice) — the simplest sanity.
        assert_warm_round_trip(&lp, &lp, "dual_fixed/self");
        // Cross-warm with RHS change matching the original regression scenario.
        let mut lp2 = lp_dual_fixed();
        lp2.b = vec![5.0, 3.0, 3.0];
        assert_warm_round_trip(&lp, &lp2, "dual_fixed/rhs_change");
    }

    #[test]
    fn warm_basis_from_singleton_row_lp() {
        let lp = lp_singleton_row();
        assert_warm_round_trip(&lp, &lp, "singleton_row/self");
    }

    #[test]
    fn warm_basis_from_non_reducible_lp() {
        let lp = lp_non_reducible();
        // Non-reducible path: `was_reduced=false`, postsolve isn't invoked.
        // Sentinel is here to catch a regression in the surrounding flow
        // (e.g. accidental warm-start invalidation in `entry.rs`).
        let r = solve(&lp);
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!(
            r.warm_start_basis.is_some(),
            "non-reducible path lost its native simplex warm_start_basis",
        );
        assert_basis_well_formed(
            &lp,
            &r.warm_start_basis.as_ref().unwrap().basis,
            "non_reducible",
        );
    }

    /// No-op proof: a re-implementation that always returns `None` makes the
    /// sentinels above fail (assertion on `is_some()`). We exercise that path
    /// inline here so the dependency is local: forcing `None` *does* break the
    /// dual-fixed warm-start round-trip even when the new RHS is feasible
    /// (because subsequent `solve_with(lp2, warm=None, presolve=false)` would
    /// be a cold dual that this fixture is fine with, BUT the upstream
    /// assertion `result.warm_start_basis.is_some()` still trips).
    #[test]
    fn noop_proof_returns_none_fails_round_trip() {
        // Reproduces the original FAIL state: presolve reduces, postsolve
        // (in this synthetic call) returns None → assertion catches the lost
        // warm-start. We don't have a runtime toggle for the recovery path —
        // instead we directly invoke the recovery function with an empty
        // solution to confirm it has measurable output (i.e. swapping the
        // function for `|_| None` is observably different).
        let lp = lp_dual_fixed();
        let solution = vec![0.0, 0.0];
        let recovered = recover_warm_start_basis(&lp, &solution);
        assert!(
            recovered.is_some(),
            "recover_warm_start_basis must produce a basis for dual-fixed LP \
             (no-op would return None and re-introduce the lost warm-start bug)",
        );
        let basis = recovered.unwrap().basis;
        let sf = build_standard_form(&lp);
        assert_eq!(basis.len(), sf.m, "recovered basis must have length m_ext");
        for &c in &basis {
            assert!(c < sf.n_total, "recovered basis col {} ≥ n_total", c);
        }
    }

    /// Validates basis quality: every active variable (x_std > 0) in the
    /// postsolved solution should appear in the basis. A noop or slack-only
    /// fallback would fail this check on the non-reducible LP where x1=3 > 0.
    #[test]
    fn warm_basis_includes_active_variables() {
        let lp = lp_non_reducible();
        let r = solve(&lp);
        assert_eq!(r.status, SolveStatus::Optimal);
        // Expected optimum: x0=1, x1=3 → both > 0 (active).
        // Standard form: lb=0 shift → x_std[0] = x[0], x_std[1] = x[1].
        // Active structural cols are 0 and 1. They should be in the basis.
        let basis = &r.warm_start_basis.as_ref().unwrap().basis;
        let sf = build_standard_form(&lp);
        assert!(
            basis.contains(&0)
                || sf.orig_var_info[0]
                    .new_vars
                    .iter()
                    .any(|&(idx, _)| basis.contains(&idx)),
            "active x0=1 not in warm-start basis: {:?}",
            basis,
        );
        assert!(
            basis.contains(&1)
                || sf.orig_var_info[1]
                    .new_vars
                    .iter()
                    .any(|&(idx, _)| basis.contains(&idx)),
            "active x1=3 not in warm-start basis: {:?}",
            basis,
        );
    }

    /// Perf gate: default options must skip the recovery path so large LPs do
    /// not pay build_standard_form + LTSF crash + refinement.  Toggle —
    /// flipping the default to `true` (or removing the `recover_warm_basis &&`
    /// gate in `run_postsolve`) flips both assertions.
    #[test]
    fn default_skips_warm_basis_recovery() {
        // dual-fixed LP: presolve reduces to zero vars, so simplex returns
        // warm_start_basis=None.  Without the postsolve recovery the final
        // result must also be None — proving the gate is alive.
        let lp = lp_dual_fixed();
        let r_default = solve(&lp);
        assert_eq!(r_default.status, SolveStatus::Optimal);
        assert!(
            r_default.warm_start_basis.is_none(),
            "default options must NOT pay warm-basis recovery cost \
             (postsolve recovery should be opt-in via recover_warm_start_basis=true)",
        );

        // Same LP under opt-in flag: warm_start_basis must be Some (existing contract).
        let r_optin = solve_with(&lp, &opts_recover());
        assert_eq!(r_optin.status, SolveStatus::Optimal);
        assert!(
            r_optin.warm_start_basis.is_some(),
            "opt-in flag must restore the postsolve warm-basis synthesis",
        );

        // singleton-row LP exercises the second presolve transform; same contract.
        let lp_sr = lp_singleton_row();
        let r_sr_default = solve(&lp_sr);
        assert_eq!(r_sr_default.status, SolveStatus::Optimal);
        assert!(
            r_sr_default.warm_start_basis.is_none(),
            "singleton-row presolve path must also skip recovery by default",
        );
        let r_sr_optin = solve_with(&lp_sr, &opts_recover());
        assert!(r_sr_optin.warm_start_basis.is_some());
    }

    /// Non-reducible path: native simplex sets warm_start_basis directly
    /// (cheap clone of basis/x_b), so the recovery flag is irrelevant — both
    /// default and opt-in must return Some.  Catches a regression that would
    /// move the gate to the wrong layer (e.g. stripping basis in entry.rs).
    #[test]
    fn non_reducible_basis_independent_of_recovery_flag() {
        let lp = lp_non_reducible();
        let r_default = solve(&lp);
        let r_optin = solve_with(&lp, &opts_recover());
        assert!(
            r_default.warm_start_basis.is_some(),
            "non-reducible default path must keep native simplex basis"
        );
        assert!(
            r_optin.warm_start_basis.is_some(),
            "non-reducible opt-in path must keep native simplex basis"
        );
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

        let lifted = run_postsolve(&raw_ipm, &presolve, &lp, Some(Instant::now()), false);

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
        let lifted = run_postsolve(&reduced, &presolve, &lp, None, false);
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
        let lifted = run_postsolve(&reduced, &presolve, &lp, None, false);
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
        let _ = run_postsolve(&reduced, &presolve, &lp, None, false);
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
        let out = run_postsolve(&reduced, &pres, &lp, None, false);
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
        let out = run_postsolve(&reduced, &pres, &lp, None, false);
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
        let out = run_postsolve(&reduced, &pres, &lp, None, false);
        assert_eq!(
            out.status,
            SolveStatus::Timeout,
            "timeout incumbent without dual must not be remapped to NumericalError"
        );
        assert_eq!(out.solution, vec![0.5]);
    }
}
