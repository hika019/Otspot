//! Standard-form LP construction + dual recovery.
//!
//! `build_standard_form` converts an `LpProblem` into the slack-augmented
//! form consumed by the simplex cores. `extract_dual_info` reverses the
//! transformations (row sign flips, Ruiz row scaling) to recover original
//! duals, reduced costs, and slacks.

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

pub(crate) enum SimplexOutcome {
    /// Optimal objective and dual vector.
    Optimal(f64, Vec<f64>),
    Unbounded,
    /// Objective at termination.
    Timeout(f64),
    /// Triggers IPM fallback at the caller.
    SingularBasis,
}

pub(crate) fn timeout_result_with_incumbent(
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    iter: usize,
) -> SolverResult {
    let solution = extract_solution(sf, basis, x_b, col_scale);
    let objective = problem
        .c
        .iter()
        .zip(solution.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>()
        + sf.obj_offset;
    SolverResult {
        status: SolveStatus::Timeout,
        objective,
        solution,
        dual_solution: vec![],
        reduced_costs: vec![],
        slack: vec![],
        warm_start_basis: None,
        iterations: iter,
        ..Default::default()
    }
}

/// Convert an LP into standard form: variable shifts/splits, upper-bound rows,
/// row sign normalization, slacks, initial basis with artificials.
pub(crate) fn build_standard_form(problem: &LpProblem) -> StandardForm {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();

    for j in 0..n_orig {
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
        match ctypes[i] {
            ConstraintType::Le => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    // Clamp b ∈ [-PIVOT_TOL, 0) noise to 0 so the slack
                    // doesn't start at a tiny negative value.
                    if b[i] < 0.0 { b[i] = 0.0; }
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
        let row = m_orig + ub_idx;
        trip_rows.push(row);
        trip_cols.push(new_var_idx);
        trip_vals.push(1.0);
    }

    for i in 0..m_ext {
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

    StandardForm {
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

    // Undo row sign flip and Ruiz row scaling on y_std.
    let mut dual_solution = vec![0.0; m_orig];
    for i in 0..m_orig {
        let sign = if sf.row_negated[i] { -1.0 } else { 1.0 };
        let rs = row_scale.get(i).copied().unwrap_or(1.0);
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
