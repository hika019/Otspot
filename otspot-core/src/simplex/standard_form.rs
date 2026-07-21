//! Standard-form LP construction + dual recovery.
//!
//! `build_standard_form` converts an `LpProblem` into the slack-augmented
//! form consumed by the simplex cores. `extract_dual_info` reverses the
//! transformations (row sign flips, Ruiz row scaling) to recover original
//! duals, reduced costs, and slacks.
//!
//! `build_bounded_standard_form` is an alternate constructor that retains
//! explicit per-variable upper bounds instead of expanding them into UB
//! slack rows. It is the input format consumed by the BFRT dual core.
//! `wrap_to_legacy` converts a `BoundedStandardForm` into the equivalent
//! `StandardForm` (UB-row-expanded) so existing solver paths can run on
//! it unchanged — this is the equivalence the sentinel locks in.

use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::{DROP_TOL, PIVOT_TOL};

use super::primal::extract_solution;

/// Mapping from one original variable to its standard-form representation.
/// Typically 1 new var (shifted bound) or 2 (free-variable split into ±).
pub(crate) struct OrigVarInfo {
    pub(crate) offset: f64,
    pub(crate) new_vars: Vec<(usize, f64)>,
}

/// Standard-form LP: A, b, c after slack addition, variable shifts/splits,
/// and per-row sign normalization.
pub(crate) struct StandardForm {
    pub(crate) a: CscMatrix,
    pub(crate) b: Vec<f64>,
    pub(crate) c: Vec<f64>,
    pub(crate) m: usize,
    pub(crate) n_shifted: usize,
    pub(crate) n_total: usize,
    pub(crate) initial_basis: Vec<usize>,
    pub(crate) needs_artificial: Vec<bool>,
    pub(crate) num_artificial: usize,
    pub(crate) obj_offset: f64,
    pub(crate) n_orig: usize,
    pub(crate) orig_var_info: Vec<OrigVarInfo>,
    /// Per-row sign-flip flag; needed when recovering original-problem duals.
    pub(crate) row_negated: Vec<bool>,
}

#[derive(Debug)]
pub(crate) enum SimplexOutcome {
    /// Optimal objective and dual vector.
    Optimal(f64, Vec<f64>),
    Unbounded,
    /// Objective at termination. External stop: wall-clock deadline expired,
    /// cancel flag fired, or a timed sub-step (LU refactor) hit the deadline.
    Timeout(f64),
    /// Objective at termination. Iteration progress exhausted with budget
    /// remaining and no certificate: cycling/plateau bail, unverifiable
    /// unbounded ray, or a dual-feasible point whose fresh `x_B = B⁻¹b`
    /// violates primal feasibility. Callers must not report this as
    /// [`crate::problem::SolveStatus::Timeout`]; use [`stall_status`].
    Stalled(f64),
    /// Triggers IPM fallback at the caller.
    SingularBasis,
}

/// User-facing status for a [`SimplexOutcome::Stalled`] bail: an incumbent
/// solution is an unproven answer (`SuboptimalSolution`); without one the only
/// honest claim is that iteration progress was exhausted (`MaxIterations`).
pub(crate) fn stall_status(has_incumbent: bool) -> SolveStatus {
    if has_incumbent {
        SolveStatus::SuboptimalSolution
    } else {
        SolveStatus::MaxIterations
    }
}

/// Status for a non-Optimal stop classified by the actual stop condition:
/// `Timeout` only when [`external_stop_requested`], otherwise [`stall_status`].
pub(crate) fn stop_status(
    has_incumbent: bool,
    options: &crate::options::SolverOptions,
) -> SolveStatus {
    if external_stop_requested(options) {
        SolveStatus::Timeout
    } else {
        stall_status(has_incumbent)
    }
}

/// See [`crate::options::SolverOptions::external_stop_requested`].
pub(crate) fn external_stop_requested(options: &crate::options::SolverOptions) -> bool {
    options.external_stop_requested()
}

/// Incumbent-carrying result for a non-Optimal stop. Status is classified by
/// the actual stop condition: `Timeout` only when the deadline expired or the
/// cancel flag fired; otherwise the stop was an internal stall and the honest
/// status is [`stall_status`] (`SuboptimalSolution` with an incumbent,
/// `MaxIterations` without).
pub(crate) fn stop_result_with_incumbent(
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    iter: usize,
    options: &crate::options::SolverOptions,
) -> SolverResult {
    let solution = extract_solution(sf, basis, x_b, col_scale);
    // `extract_solution` already un-shifts to original variables, so `c·solution`
    // IS the complete original objective. Adding `sf.obj_offset` (= Σ c_j·lb_j)
    // would double-count the shift constant.
    let objective = problem
        .c
        .iter()
        .zip(solution.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>();
    let status = if external_stop_requested(options) {
        SolveStatus::Timeout
    } else {
        stall_status(!solution.is_empty())
    };
    SolverResult {
        status,
        objective,
        solution,
        iterations: iter,
        ..Default::default()
    }
}

/// Convert an LP into standard form: variable shifts/splits, upper-bound rows,
/// row sign normalization, slacks, initial basis with artificials.
pub(crate) fn build_standard_form(problem: &LpProblem) -> StandardForm {
    build_standard_form_with_deadline(problem, None)
        .expect("build_standard_form without deadline must not time out")
}

pub(crate) fn build_standard_form_with_deadline(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
) -> Option<StandardForm> {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();

    for j in 0..n_orig {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            obj_offset += problem.c[j] * lb;
            orig_var_info.push(OrigVarInfo {
                offset: lb,
                new_vars: vec![(idx, 1.0)],
            });
        } else if ub.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            obj_offset += problem.c[j] * ub;
            orig_var_info.push(OrigVarInfo {
                offset: ub,
                new_vars: vec![(idx, -1.0)],
            });
        } else {
            let idx_plus = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            let idx_minus = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            orig_var_info.push(OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(idx_plus, 1.0), (idx_minus, -1.0)],
            });
        }
    }

    // Upper bound rows.
    let mut ub_constraints: Vec<(usize, f64)> = Vec::new();
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() {
            let effective_ub = ub - lb;
            let new_idx = info.new_vars[0].0;
            ub_constraints.push((new_idx, effective_ub));
        }
    }
    let n_ub = ub_constraints.len();
    let m_ext = m_orig + n_ub;

    // b adjusted for variable shifts.
    let mut b = problem.b.clone();
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let offset = info.offset;
        if offset.abs() > DROP_TOL {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    b[row] -= vals[k] * offset;
                }
            }
        }
    }
    for &(_, ub_val) in &ub_constraints {
        b.push(ub_val);
    }

    let mut ctypes: Vec<ConstraintType> = problem.constraint_types.clone();
    for _ in 0..n_ub {
        ctypes.push(ConstraintType::Le);
    }

    // Row sign normalization + slack column setup.
    let mut row_negated = vec![false; m_ext];
    let mut slack_col_idx: Vec<Option<usize>> = Vec::with_capacity(m_ext);
    let mut n_slack = 0usize;
    let mut slack_coeff = vec![0.0f64; m_ext];

    for i in 0..m_ext {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        match ctypes[i] {
            ConstraintType::Le => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    // Clamp b ∈ [-PIVOT_TOL, 0) noise to 0 so the slack
                    // doesn't start at a tiny negative value.
                    if b[i] < 0.0 {
                        b[i] = 0.0;
                    }
                    slack_coeff[i] = 1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Ge => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = 1.0; // -1 negated
                } else {
                    slack_coeff[i] = -1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Eq => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                }
                slack_col_idx.push(None);
            }
        }
    }

    let n_total = n_shifted + n_slack;

    // Initial basis: pick slack where possible; flag artificials otherwise.
    let mut initial_basis = vec![0usize; m_ext];
    let mut needs_artificial = vec![false; m_ext];
    let mut num_artificial = 0usize;

    for i in 0..m_ext {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        match slack_col_idx[i] {
            Some(s_idx) => {
                let col = n_shifted + s_idx;
                if slack_coeff[i] > 0.0 {
                    // Le: slack >= 0 → no artificial needed
                    initial_basis[i] = col;
                } else if b[i].abs() <= PIVOT_TOL {
                    // Ge with b≈0: surplus at 0 is feasible, so skip the artificial
                    // (which would otherwise sit in the Phase II basis and let the
                    // constraint drift, since Phase I terminates with obj=0).
                    initial_basis[i] = col;
                } else {
                    needs_artificial[i] = true;
                    num_artificial += 1;
                    initial_basis[i] = col; // placeholder
                }
            }
            None => {
                needs_artificial[i] = true;
                num_artificial += 1;
            }
        }
    }

    let mut trip_rows = Vec::new();
    let mut trip_cols = Vec::new();
    let mut trip_vals = Vec::new();

    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        if let Ok((a_rows, a_vals)) = problem.a.get_column(j) {
            for (k, &row) in a_rows.iter().enumerate() {
                let val = a_vals[k];
                let sign = if row_negated[row] { -1.0 } else { 1.0 };
                for &(new_col, coeff) in &info.new_vars {
                    let actual_val = sign * val * coeff;
                    if actual_val.abs() > DROP_TOL {
                        trip_rows.push(row);
                        trip_cols.push(new_col);
                        trip_vals.push(actual_val);
                    }
                }
            }
        }
    }

    for (ub_idx, &(new_var_idx, _)) in ub_constraints.iter().enumerate() {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let row = m_orig + ub_idx;
        trip_rows.push(row);
        trip_cols.push(new_var_idx);
        trip_vals.push(1.0);
    }

    for i in 0..m_ext {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        if let Some(s_idx) = slack_col_idx[i] {
            let col = n_shifted + s_idx;
            trip_rows.push(i);
            trip_cols.push(col);
            trip_vals.push(slack_coeff[i]);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_ext, n_total).unwrap();

    let mut c_ext = vec![0.0; n_total];
    c_ext[..n_shifted].copy_from_slice(&new_c[..n_shifted]);

    Some(StandardForm {
        a,
        b,
        c: c_ext,
        m: m_ext,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset,
        n_orig,
        orig_var_info,
        row_negated,
    })
}

/// Standard-form LP that keeps **explicit per-variable upper bounds** instead
/// of expanding bounded variables into UB rows + slacks.
///
/// Relative to [`StandardForm`]:
/// - `m` is the original constraint count (no UB rows appended).
/// - `n_shifted` columns include only variable shifts / free-var splits.
/// - `upper_bounds[j]` is the post-shift upper bound for column `j`
///   (`f64::INFINITY` ⇒ unbounded). Slack columns are always `INFINITY`.
///
/// BFRT (`bound_flip.rs`) needs `upper_bounds[j]` finite on the candidate
/// columns to flip; the legacy `StandardForm` rewrites all bounded vars as
/// `x ≥ 0` + slack row, so BFRT would see no flip handles there.
pub(crate) struct BoundedStandardForm {
    pub(crate) a: CscMatrix,
    pub(crate) b: Vec<f64>,
    pub(crate) c: Vec<f64>,
    pub(crate) m: usize,
    pub(crate) n_shifted: usize,
    pub(crate) n_total: usize,
    pub(crate) initial_basis: Vec<usize>,
    /// Which rows require an artificial variable. Read by the Eq+UB bounded
    /// Phase I path to place artificials and gate dispatch; also used by the
    /// test helper `wrap_to_legacy`.
    pub(crate) needs_artificial: Vec<bool>,
    pub(crate) num_artificial: usize,
    pub(crate) obj_offset: f64,
    pub(crate) n_orig: usize,
    pub(crate) orig_var_info: Vec<OrigVarInfo>,
    pub(crate) row_negated: Vec<bool>,
    /// Per-column upper bound (length `n_total`). `INFINITY` ⇒ unbounded.
    pub(crate) upper_bounds: Vec<f64>,
}

/// Convert an LP into bounded standard form: variable shifts/splits + row
/// sign normalization + slacks for the *original* constraints only. Bounded
/// variables keep their upper bound in `upper_bounds[j]`.
#[cfg(test)]
pub(crate) fn build_bounded_standard_form(problem: &LpProblem) -> BoundedStandardForm {
    build_bounded_standard_form_with_deadline(problem, None)
        .expect("build_bounded_standard_form without deadline must not time out")
}

pub(crate) fn build_bounded_standard_form_with_deadline(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
) -> Option<BoundedStandardForm> {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();
    let mut var_upper: Vec<f64> = Vec::new();

    for j in 0..n_orig {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            obj_offset += problem.c[j] * lb;
            orig_var_info.push(OrigVarInfo {
                offset: lb,
                new_vars: vec![(idx, 1.0)],
            });
            var_upper.push(if ub.is_finite() {
                ub - lb
            } else {
                f64::INFINITY
            });
        } else if ub.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            obj_offset += problem.c[j] * ub;
            orig_var_info.push(OrigVarInfo {
                offset: ub,
                new_vars: vec![(idx, -1.0)],
            });
            var_upper.push(f64::INFINITY);
        } else {
            let idx_plus = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            var_upper.push(f64::INFINITY);
            let idx_minus = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            var_upper.push(f64::INFINITY);
            orig_var_info.push(OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(idx_plus, 1.0), (idx_minus, -1.0)],
            });
        }
    }

    let mut b = problem.b.clone();
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let offset = info.offset;
        if offset.abs() > DROP_TOL {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    b[row] -= vals[k] * offset;
                }
            }
        }
    }

    let mut row_negated = vec![false; m_orig];
    let mut slack_col_idx: Vec<Option<usize>> = Vec::with_capacity(m_orig);
    let mut n_slack = 0usize;
    let mut slack_coeff = vec![0.0f64; m_orig];

    for i in 0..m_orig {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        match problem.constraint_types[i] {
            ConstraintType::Le => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    if b[i] < 0.0 {
                        b[i] = 0.0;
                    }
                    slack_coeff[i] = 1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Ge => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = 1.0;
                } else {
                    slack_coeff[i] = -1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Eq => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                }
                slack_col_idx.push(None);
            }
        }
    }

    let n_total = n_shifted + n_slack;

    let mut initial_basis = vec![0usize; m_orig];
    let mut needs_artificial = vec![false; m_orig];
    let mut num_artificial = 0usize;

    for i in 0..m_orig {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        match slack_col_idx[i] {
            Some(s_idx) => {
                let col = n_shifted + s_idx;
                if slack_coeff[i] > 0.0 || b[i].abs() <= PIVOT_TOL {
                    initial_basis[i] = col;
                } else {
                    needs_artificial[i] = true;
                    num_artificial += 1;
                    initial_basis[i] = col;
                }
            }
            None => {
                needs_artificial[i] = true;
                num_artificial += 1;
            }
        }
    }

    let mut trip_rows = Vec::new();
    let mut trip_cols = Vec::new();
    let mut trip_vals = Vec::new();

    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        if let Ok((a_rows, a_vals)) = problem.a.get_column(j) {
            for (k, &row) in a_rows.iter().enumerate() {
                let val = a_vals[k];
                let sign = if row_negated[row] { -1.0 } else { 1.0 };
                for &(new_col, coeff) in &info.new_vars {
                    let actual_val = sign * val * coeff;
                    if actual_val.abs() > DROP_TOL {
                        trip_rows.push(row);
                        trip_cols.push(new_col);
                        trip_vals.push(actual_val);
                    }
                }
            }
        }
    }

    for i in 0..m_orig {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        if let Some(s_idx) = slack_col_idx[i] {
            let col = n_shifted + s_idx;
            trip_rows.push(i);
            trip_cols.push(col);
            trip_vals.push(slack_coeff[i]);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_orig, n_total).unwrap();

    let mut c_ext = vec![0.0; n_total];
    c_ext[..n_shifted].copy_from_slice(&new_c[..n_shifted]);

    let mut upper_bounds = vec![f64::INFINITY; n_total];
    upper_bounds[..n_shifted].copy_from_slice(&var_upper[..n_shifted]);

    Some(BoundedStandardForm {
        a,
        b,
        c: c_ext,
        m: m_orig,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset,
        n_orig,
        orig_var_info,
        row_negated,
        upper_bounds,
    })
}

/// Scale upper bounds by the Ruiz column-scaling vector.
///
/// During Ruiz scaling the primal variable `j` becomes `x̃_j = x_j / col_scale[j]`.
/// Upper bounds stored in `BoundedStandardForm.upper_bounds` are in the pre-scale
/// space (`u_j_orig`). The effective bound in the scaled iteration space is
/// `u_j_scaled = u_j_orig / col_scale[j]`. Pass the result as `ubs` to
/// `solve_bounded_dual` / `iterate` / `phase2_primal_bounded` so that feasibility
/// checks and BFRT weights operate in a consistent space.
///
/// `extract_solution_bounded` continues to use the **original** `bsf.upper_bounds`
/// (the col_scale factors cancel when recovering non-basic-at-upper values).
pub(crate) fn scale_upper_bounds(upper_bounds: &[f64], col_scale: &[f64]) -> Vec<f64> {
    assert!(
        col_scale.is_empty() || col_scale.len() == upper_bounds.len(),
        "col_scale must be empty (identity) or match upper_bounds"
    );
    upper_bounds
        .iter()
        .enumerate()
        .map(|(j, &u)| {
            if u.is_finite() {
                u / if col_scale.is_empty() {
                    1.0
                } else {
                    col_scale[j]
                }
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

/// Expand a `BoundedStandardForm` into the legacy `StandardForm` by adding
/// one UB row + slack per finite-upper variable. The result is bit-equivalent
/// to `build_standard_form` on the same `LpProblem` (modulo CSC canonicalization).
///
/// Test-only: used by `tests_bounded_form` to verify BSF↔SF equivalence.
#[cfg(test)]
pub(crate) fn wrap_to_legacy(bsf: &BoundedStandardForm) -> StandardForm {
    let n_shifted = bsf.n_shifted;
    let m_orig = bsf.m;
    let n_slack_orig = bsf.n_total - n_shifted;

    let ub_vars: Vec<(usize, f64)> = (0..n_shifted)
        .filter(|&j| bsf.upper_bounds[j].is_finite())
        .map(|j| (j, bsf.upper_bounds[j]))
        .collect();
    let n_ub = ub_vars.len();
    let m_ext = m_orig + n_ub;
    let n_total = bsf.n_total + n_ub;

    let mut b = bsf.b.clone();
    let mut row_negated = bsf.row_negated.clone();
    let mut initial_basis = bsf.initial_basis.clone();
    let mut needs_artificial = bsf.needs_artificial.clone();
    let mut num_artificial = bsf.num_artificial;

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();

    for col in 0..bsf.a.ncols {
        let start = bsf.a.col_ptr[col];
        let end = bsf.a.col_ptr[col + 1];
        for idx in start..end {
            trip_rows.push(bsf.a.row_ind[idx]);
            trip_cols.push(col);
            trip_vals.push(bsf.a.values[idx]);
        }
    }

    for (k, &(j, ub_val)) in ub_vars.iter().enumerate() {
        let row = m_orig + k;
        // UB row sign normalization (Le with `b = ub_val`). Mirrors the
        // build_standard_form loop so degenerate inputs (lb > ub ⇒ negative
        // shifted upper) stay bit-equivalent.
        let mut b_new = ub_val;
        let (neg, slack_coeff) = if b_new < -PIVOT_TOL {
            b_new = -b_new;
            (true, -1.0)
        } else {
            if b_new < 0.0 {
                b_new = 0.0;
            }
            (false, 1.0)
        };
        b.push(b_new);
        row_negated.push(neg);

        let var_sign = if neg { -1.0 } else { 1.0 };
        trip_rows.push(row);
        trip_cols.push(j);
        trip_vals.push(var_sign);

        let slack_col = n_shifted + n_slack_orig + k;
        trip_rows.push(row);
        trip_cols.push(slack_col);
        trip_vals.push(slack_coeff);

        if slack_coeff > 0.0 || b_new.abs() <= PIVOT_TOL {
            initial_basis.push(slack_col);
            needs_artificial.push(false);
        } else {
            needs_artificial.push(true);
            num_artificial += 1;
            initial_basis.push(slack_col);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_ext, n_total).unwrap();

    let mut c = bsf.c.clone();
    c.extend(std::iter::repeat_n(0.0, n_ub));

    let orig_var_info = bsf
        .orig_var_info
        .iter()
        .map(|info| OrigVarInfo {
            offset: info.offset,
            new_vars: info.new_vars.clone(),
        })
        .collect();

    StandardForm {
        a,
        b,
        c,
        m: m_ext,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset: bsf.obj_offset,
        n_orig: bsf.n_orig,
        orig_var_info,
        row_negated,
    }
}

/// Recover original-problem duals, reduced costs and slack from the
/// standard-form dual `y_std` and primal `solution`.
pub(crate) fn extract_dual_info(
    sf: &StandardForm,
    problem: &LpProblem,
    y_std: &[f64],
    solution: &[f64],
    row_scale: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_orig = problem.num_constraints;
    let n_orig = problem.num_vars;
    assert!(
        row_scale.is_empty() || row_scale.len() == sf.m,
        "row_scale must be empty (identity) or match the standard-form row count"
    );

    // Undo row sign flip and Ruiz row scaling on y_std.
    let mut dual_solution = vec![0.0; m_orig];
    for i in 0..m_orig {
        let sign = if sf.row_negated[i] { -1.0 } else { 1.0 };
        let rs = if row_scale.is_empty() {
            1.0
        } else {
            row_scale[i]
        };
        dual_solution[i] = sign * rs * y_std[i];
    }

    let mut slack = problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    // rc[j] = c[j] − Σ_i A_ij · y_i (Lagrangian stationarity residual on the
    // original constraints). Bound multipliers μ_lb/μ_ub are kept separate; rc
    // doubles as the bound dual via complementary slackness:
    //   at orig lb → rc =  μ_lb ≥ 0,
    //   at orig ub → rc = -μ_ub ≤ 0,
    //   interior   → rc = 0,
    //   truly fixed (lb==ub) → rc = μ_lb − μ_ub (free sign).
    // The KKT test `c − A^T y − rc = 0` (diag_afiro_y) requires this convention.
    let mut reduced_costs = problem.c.clone();
    for (j, rc_j) in reduced_costs.iter_mut().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m_orig {
                    *rc_j -= dual_solution[row] * vals[k];
                }
            }
        }
    }

    (dual_solution, reduced_costs, slack)
}
