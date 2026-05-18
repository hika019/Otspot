//! Two-phase revised simplex with LU-based basis updates.

pub mod dual;
pub mod dual_advanced;
pub mod pricing;
pub(crate) mod primal;

use crate::options::{SimplexMethod, SolverOptions};
use crate::presolve;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::{DROP_TOL, PIVOT_TOL};

pub(crate) use primal::{two_phase_simplex, extract_solution, revised_simplex_core};
#[cfg(test)]
pub(crate) use primal::reconcile_final_basis_state;

/// Solve an LP with default options.
pub fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// Solve an LP with the supplied options. When `options.presolve` is set,
/// presolve runs before the simplex.
pub fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    // timeout_secs → deadline (mirrors qp_solve_impl).
    let mut opts_with_deadline;
    let options = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
        opts_with_deadline = options.clone();
        opts_with_deadline.deadline = Some(
            std::time::Instant::now() + std::time::Duration::from_secs_f64(secs),
        );
        &opts_with_deadline
    } else {
        options
    };

    let prof_t0 = std::time::Instant::now();

    if options.presolve {
        match presolve::run_presolve(problem, options.deadline) {
            Err(presolve::PresolveStatus::Infeasible) => {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
            Err(presolve::PresolveStatus::Unbounded) => {
                return SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
            Ok(presolve_result) if presolve_result.was_reduced => {
                // Presolve renumbers variables, so a supplied warm_start is invalidated.
                let opts_no_ws = if options.warm_start.is_some() {
                    let mut o = options.clone();
                    o.warm_start = None;
                    o.presolve = false;
                    Some(o)
                } else {
                    None
                };
                let eff_opts = opts_no_ws.as_ref().unwrap_or(options);
                let t_presolve_done = std::time::Instant::now();
                let presolve_us = t_presolve_done.duration_since(prof_t0).as_micros() as u64;
                let raw = solve_without_presolve(&presolve_result.reduced_problem, eff_opts);
                let t_solve_done = std::time::Instant::now();
                let solve_us = t_solve_done.duration_since(t_presolve_done).as_micros() as u64;
                // The reduced LP can be unsolvable while the original is fine
                // (SingularBasis on the reduced initial basis, Eq drift in Phase II).
                // Fall back to solving the original LP without presolve.
                if raw.status == SolveStatus::NumericalError {
                    return solve_without_presolve(problem, options);
                }
                let mut res = presolve::postsolve::run_postsolve(&raw, &presolve_result, problem, eff_opts.deadline);
                let postsolve_us = t_solve_done.elapsed().as_micros() as u64;
                res.timing_breakdown = Some(crate::problem::TimingBreakdown {
                    presolve_us, solve_us, postsolve_us,
                });
                // Postsolve dfeas above PIVOT_TOL means dual-recovery cannot
                // reconstruct the structure presolve removed. The original LP
                // solves cleanly, so re-attempt on the remaining deadline.
                if res.status == SolveStatus::Optimal
                    && res.postsolve_dfeas.is_some_and(|d| d > PIVOT_TOL)
                {
                    let deadline_ok = options.deadline
                        .is_none_or(|d| std::time::Instant::now() < d);
                    if deadline_ok {
                        let mut opts_off = options.clone();
                        opts_off.presolve = false;
                        // Force primal: 初回試行で feasibility 既知のため Primal で直行。
                        opts_off.simplex_method = crate::options::SimplexMethod::Primal;
                        let t_alt_start = std::time::Instant::now();
                        let mut alt = solve_without_presolve(problem, &opts_off);
                        let alt_solve_us = t_alt_start.elapsed().as_micros() as u64;
                        if alt.status == SolveStatus::Optimal
                            && alt.postsolve_dfeas.is_none()
                            && alt.objective.is_finite()
                        {
                            alt.timing_breakdown = Some(crate::problem::TimingBreakdown {
                                presolve_us: 0,
                                solve_us: alt_solve_us,
                                postsolve_us: 0,
                            });
                            return alt;
                        }
                    }
                }
                return res;
            }
            Ok(_) => {}
        }
    }

    // Catch deadline overrun before build_standard_form (presolve may have
    // returned early without reducing).
    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }

    solve_without_presolve(problem, options)
}

/// Solve without presolve.
pub(crate) fn solve_without_presolve(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    if n == 0 {
        for i in 0..m {
            if problem.b[i] < -options.primal_tol {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![0.0; m],
            reduced_costs: vec![],
            slack: problem.b.clone(),
            warm_start_basis: None,
            ..Default::default()
        };
    }

    if m == 0 {
        let mut x = vec![0.0; n];
        let mut obj = 0.0;
        for (j, x_j) in x.iter_mut().enumerate() {
            if problem.c[j] < -options.primal_tol {
                // Finite upper bound caps the maximizer; infinite ⇒ Unbounded.
                let ub = problem.bounds[j].1;
                if ub.is_infinite() {
                    return SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
            ..Default::default()
                    };
                }
                *x_j = ub;
            }
            obj += problem.c[j] * *x_j;
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: obj,
            solution: x,
            dual_solution: vec![],
            reduced_costs: problem.c.clone(),
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }

    let sf = build_standard_form(problem);

    match options.simplex_method {
        SimplexMethod::Primal => two_phase_simplex(&sf, problem, options),
        SimplexMethod::Dual => dual::two_phase_dual_simplex(&sf, problem, options),
        SimplexMethod::DualAdvanced | SimplexMethod::Auto => {
            // Auto uses dual_advanced; it falls back to two_phase_dual_simplex
            // internally for problems with Ge/Eq constraints.
            dual_advanced::solve_dual_advanced(&sf, problem, options)
        }
    }
}

/// Mapping from one original variable to its standard-form representation.
/// Typically 1 new var (shifted bound) or 2 (free-variable split into ±).
pub(crate) struct OrigVarInfo {
    offset: f64,
    new_vars: Vec<(usize, f64)>,
}

/// Standard-form LP: A, b, c after slack addition, variable shifts/splits,
/// and per-row sign normalization.
pub(crate) struct StandardForm {
    a: CscMatrix,
    b: Vec<f64>,
    c: Vec<f64>,
    m: usize,
    n_shifted: usize,
    n_total: usize,
    initial_basis: Vec<usize>,
    needs_artificial: Vec<bool>,
    num_artificial: usize,
    obj_offset: f64,
    n_orig: usize,
    orig_var_info: Vec<OrigVarInfo>,
    /// Per-row sign-flip flag; needed when recovering original-problem duals.
    row_negated: Vec<bool>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

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
    fn test_timeout_result_with_incumbent_uses_original_objective() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![3.0, 1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let basis = sf.initial_basis.clone();
        let x_b = sf.b.clone();
        let col_scale = vec![1.0; sf.n_total];

        let result = timeout_result_with_incumbent(&sf, &lp, &basis, &x_b, &col_scale, 42);

        assert_eq!(result.status, SolveStatus::Timeout);
        assert_eq!(result.iterations, 42, "iter arg は SolverResult.iterations へ反映");
        assert_eq!(result.solution.len(), 2);
        let expected_obj = lp
            .c
            .iter()
            .zip(result.solution.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>();
        assert!((result.objective - expected_obj).abs() < 1e-12, "obj={}", result.objective);
    }

    #[test]
    fn test_reconcile_final_basis_state_recomputes_xb_and_y() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 2, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
        )
        .unwrap();
        let b = vec![3.0, 5.0];
        let c = vec![4.0, 2.0, 1.0];
        let basis = vec![0usize, 2usize];
        let mut x_b = vec![0.0, 0.0];
        let mut y = vec![0.0, 0.0];

        reconcile_final_basis_state(&a, &b, &c, &basis, &mut x_b, &mut y, 50, None).unwrap();

        assert!((x_b[0] + 2.0).abs() < 1e-12, "x_b[0]={}", x_b[0]);
        assert!((x_b[1] - 5.0).abs() < 1e-12, "x_b[1]={}", x_b[1]);
        assert!((y[0] - 4.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] + 3.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn test_extract_solution_uses_dd_for_split_variable_cancellation() {
        let sf = StandardForm {
            a: CscMatrix::new(3, 3),
            b: vec![0.0, 0.0, 0.0],
            c: vec![0.0, 0.0, 0.0],
            m: 3,
            n_shifted: 3,
            n_total: 3,
            initial_basis: vec![0, 1, 2],
            needs_artificial: vec![false, false, false],
            num_artificial: 0,
            obj_offset: 0.0,
            n_orig: 1,
            orig_var_info: vec![OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(0, 1.0), (1, 1.0), (2, -1.0)],
            }],
            row_negated: vec![false, false, false],
        };
        let basis = vec![0usize, 1usize, 2usize];
        let x_b = vec![1.0_f64, 1.0e16_f64, 1.0e16_f64];
        let col_scale = vec![1.0, 1.0, 1.0];

        let solution = extract_solution(&sf, &basis, &x_b, &col_scale);

        assert_eq!(solution.len(), 1);
        assert!(
            (solution[0] - 1.0).abs() < 1e-12,
            "split-variable recomposition should preserve unit residual, got {}",
            solution[0]
        );
    }

    #[test]
    fn test_basic_2var() {
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < PIVOT_TOL,
            "Expected objective -4.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x1), "x1={}", x1);
        assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x2), "x2={}", x2);
        assert!((x1 + x2 - 4.0).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_basic_3var() {
        let lp = make_lp(
            vec![-2.0, -3.0, -1.0],
            &[0, 0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            3,
            3,
            vec![10.0, 14.0, 8.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = &result.solution;
        assert!(x[0] >= -PIVOT_TOL);
        assert!(x[1] >= -PIVOT_TOL);
        assert!(x[2] >= -PIVOT_TOL);
        assert!(x[0] + x[1] + x[2] <= 10.0 + PIVOT_TOL);
        assert!(2.0 * x[0] + x[1] <= 14.0 + PIVOT_TOL);
        assert!(x[1] + x[2] <= 8.0 + PIVOT_TOL);
        assert!(
            (result.objective - (-28.0)).abs() < PIVOT_TOL,
            "Expected objective -28.0, got {}",
            result.objective
        );
    }

    #[test]
    fn test_unbounded() {
        let lp = make_lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, -1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_infeasible() {
        let lp = make_lp(vec![1.0], &[0], &[0], &[1.0], 1, 1, vec![-1.0]);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Infeasible);
    }

    #[test]
    fn test_degenerate_zero_vars() {
        let a = CscMatrix::new(0, 0);
        let lp = LpProblem::new(vec![], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_zero_constraints_unbounded() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![-1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_zero_constraints_optimal() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_solve_with_default_options() {
        // SolverOptions::default() で solve() と同じ結果が返ること
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result_default = solve(&lp);
        let result_with = solve_with(&lp, &SolverOptions::default());
        assert_eq!(result_default.status, result_with.status);
        assert!(
            (result_default.objective - result_with.objective).abs() < PIVOT_TOL,
            "solve() and solve_with(default) should return same objective"
        );
    }

    /// min -x - y s.t. x+y ≥ 1, 0 ≤ x,y ≤ 10 ⇒ x=y=10, obj=-20.
    #[test]
    fn test_simplex_ge_defensive() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(5.0);
        let start = std::time::Instant::now();
        let result = solve_with(&lp, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "test_simplex_ge_defensive: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "Status should be Optimal");
        assert!(
            (result.objective - (-20.0)).abs() < PIVOT_TOL,
            "Expected obj=-20.0, got {}",
            result.objective
        );
        assert!(
            result.solution[0] >= -PIVOT_TOL && result.solution[0] <= 10.0 + PIVOT_TOL,
            "x should be in [0, 10], got {}",
            result.solution[0]
        );
        assert!(
            result.solution[1] >= -PIVOT_TOL && result.solution[1] <= 10.0 + PIVOT_TOL,
            "y should be in [0, 10], got {}",
            result.solution[1]
        );
        assert!(
            (result.solution[0] + result.solution[1] - 20.0).abs() < PIVOT_TOL,
            "x + y should be 20.0, got {}",
            result.solution[0] + result.solution[1]
        );
    }

    /// Le-only LP: verify dual / slack / reduced costs.
    /// min -x1-2x2 s.t. x1+x2≤4, x1≤3, x2≤3, x≥0
    ///  ⇒ x=(1,3), y=(-1,0,-1), slack=(0,2,0), rc=(0,0).
    #[test]
    fn test_dual_solution_basic_le_constraints() {
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-7.0)).abs() < PIVOT_TOL,
            "Expected obj=-7.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 3, "dual_solution should have 3 elements");
        assert!(
            (result.dual_solution[0] - (-1.0)).abs() < PIVOT_TOL,
            "y[0] should be -1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] should be 0.0 (non-binding), got {}",
            result.dual_solution[1]
        );
        assert!(
            (result.dual_solution[2] - (-1.0)).abs() < PIVOT_TOL,
            "y[2] should be -1.0, got {}",
            result.dual_solution[2]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 3, "slack should have 3 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] should be 0 (binding), got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 2.0).abs() < PIVOT_TOL,
            "slack[1] should be 2.0 (non-binding), got {}",
            result.slack[1]
        );
        assert!(
            result.slack[2].abs() < PIVOT_TOL,
            "slack[2] should be 0 (binding), got {}",
            result.slack[2]
        );

        // 被縮小費用の検証（基底変数なのでゼロ）
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] should be 0 (basic), got {}",
            result.reduced_costs[0]
        );
        assert!(
            result.reduced_costs[1].abs() < PIVOT_TOL,
            "rc[1] should be 0 (basic), got {}",
            result.reduced_costs[1]
        );
    }

    #[test]
    fn test_large_coefficient_lp() {
        // 係数に 1e12 と 1e-12 を混合した問題 → Optimal or 適切なステータス（オーバーフローしない）
        // min -1e12 * x1 + 1e-12 * x2, s.t. x1 + x2 <= 1, x1,x2 >= 0
        // 最適解: x1=1, x2=0, obj=-1e12
        let lp = make_lp(
            vec![-1e12, 1e-12],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::Timeout,
            "Expected Optimal or Timeout, got {:?}",
            result.status
        );
        assert!(!result.objective.is_nan(), "Objective should not be NaN");
        assert!(result.objective.is_finite(), "Objective should be finite for bounded LP");

        // 全係数 0.0 の目的関数 → Optimal, objective=0.0
        // min 0*x1 + 0*x2, s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        let lp_zero = make_lp(
            vec![0.0, 0.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result_zero = solve(&lp_zero);
        assert_eq!(result_zero.status, SolveStatus::Optimal, "Expected Optimal for zero-objective LP");
        assert!(
            result_zero.objective.abs() < PIVOT_TOL,
            "Expected objective=0.0, got {}",
            result_zero.objective
        );
    }

    #[test]
    fn test_highly_degenerate_lp() {
        // 高度退化 LP: 3制約が (1,1) で交わる → 基底解が退化
        // min -x1 - x2
        // s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        // 最適解: x1=1, x2=1, obj=-2（サイクリングせずに到達すること）
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal for degenerate LP");
        assert!(
            (result.objective - (-2.0)).abs() < PIVOT_TOL,
            "Expected objective=-2.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!((x1 - 1.0).abs() < PIVOT_TOL, "Expected x1=1.0, got {}", x1);
        assert!((x2 - 1.0).abs() < PIVOT_TOL, "Expected x2=1.0, got {}", x2);
    }

    /// Eq + Le mix: verify dual / slack / reduced costs.
    /// min x1+2x2 s.t. x1+x2=6, x2≤5, x≥0
    ///  ⇒ x=(6,0), y=(1,0), slack=(0,5), rc=(0,1).
    #[test]
    fn test_dual_solution_equality_constraint() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 1],
            &[1.0, 1.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![6.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - 6.0).abs() < PIVOT_TOL,
            "Expected obj=6.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 2, "dual_solution should have 2 elements");
        assert!(
            (result.dual_solution[0] - 1.0).abs() < PIVOT_TOL,
            "y[0] (Eq constraint shadow price) should be 1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] (Le constraint, non-binding) should be 0.0, got {}",
            result.dual_solution[1]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 2, "slack should have 2 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] (Eq constraint) should be 0, got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 5.0).abs() < PIVOT_TOL,
            "slack[1] (x2<=5, non-binding) should be 5.0, got {}",
            result.slack[1]
        );

        // 被縮小費用の検証
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] (x1, basic) should be 0.0, got {}",
            result.reduced_costs[0]
        );
        assert!(
            (result.reduced_costs[1] - 1.0).abs() < PIVOT_TOL,
            "rc[1] (x2, non-basic) should be 1.0, got {}",
            result.reduced_costs[1]
        );
    }

    #[test]
    fn test_free_variables_phase_i() {
        // 全変数が自由境界（-INF/INF）のLP
        // minimize x1 + x2
        // s.t. x1 + x2 = 2
        // x1, x2 in (-INF, INF)
        // → Optimal（Infeasibleを返してはならない）
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0],
            vec![crate::problem::ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "Expected Optimal for free-variable LP with Eq constraint, got {:?}",
            result.status
        );
        // 解の制約充足チェック: x1 + x2 = 2
        assert!(
            (result.solution[0] + result.solution[1] - 2.0).abs() < 1e-6,
            "Expected x1+x2=2, got x1={}, x2={}, sum={}",
            result.solution[0],
            result.solution[1],
            result.solution[0] + result.solution[1]
        );
    }

    #[test]
    fn test_hs51_feasibility_lp() {
        // HS51の実行可能性LP: find_initial_feasible_pointが構築するLPを直接テスト
        // 5変数(全自由), 6Le制約(等式制約を2不等式ペアに変換)
        // b[1]=-4.0 (負のRHS) → build_standard_formで符号反転+人工変数追加
        // 解は存在する(x=[1,1,1,1,1])のでOptimalを返すべき
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
            &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            &[1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0],
            6,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0; 5],
            a,
            vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            vec![crate::problem::ConstraintType::Le; 6],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "HS51 feasibility LP: Expected Optimal, got {:?}",
            result.status
        );
        // 解が制約を満たすか検証 (x1+3x2=4 かつ x3+x4-2x5=0 かつ x2-x5=0)
        let x = &result.solution;
        assert!(
            (x[0] + 3.0 * x[1] - 4.0).abs() < 1e-6,
            "Constraint x1+3x2=4 violated: {}",
            x[0] + 3.0 * x[1]
        );
    }

    #[test]
    fn test_finite_ub_zero_constraints() {
        // m=0 with maximize x, lb=0, ub=3 ⇒ x=3.
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new_general(
            vec![-1.0], // minimize -x (= maximize x)
            a,
            vec![],
            vec![],
            vec![(0.0, 3.0)], // lb=0, ub=3
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.solution[0] - 3.0).abs() < PIVOT_TOL,
            "Expected x=3, got {}",
            result.solution[0]
        );
        assert!(
            (result.objective - (-3.0)).abs() < PIVOT_TOL,
            "Expected obj=-3, got {}",
            result.objective
        );
    }

    #[test]
    fn test_primal_simplex_timeout() {
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
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    #[test]
    fn test_lp_timeout() {
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
            timeout_secs: Some(0.0),
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    #[test]
    fn test_lp_cancel() {
        use std::sync::Arc;
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
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// Singular initial basis (duplicate column) must not yield Optimal.
    #[test]
    fn test_singular_initial_basis_not_optimal() {
        use crate::simplex::pricing::DantzigPricing;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let mut x_b = vec![1.0, 0.0];
        let mut basis = vec![0usize, 0];
        let mut pricing = DantzigPricing;
        let opts = SolverOptions::default();
        let b = vec![1.0, 0.0];
        let mut iters = 0usize;
        let outcome = revised_simplex_core(
            &a, &mut x_b, &c, &b, &mut basis, 2, 2, 2, &mut pricing, &opts, &mut iters, false,
        );
        assert!(!matches!(outcome, SimplexOutcome::Optimal(..)));
    }

    /// `solve_with` must never surface SolveStatus::MaxIterations.
    #[test]
    fn test_solve_does_not_return_max_iterations() {
        for method in [SimplexMethod::Primal, SimplexMethod::Dual] {
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
                simplex_method: method,
                presolve: false,
                ..SolverOptions::default()
            };
            let result = solve_with(&lp, &opts);
            assert_ne!(result.status, SolveStatus::MaxIterations, "method={:?}", method);
        }
    }

    /// refactor_failed with no deadline must yield Optimal/Timeout/SingularBasis.
    #[test]
    fn test_refactor_failed_no_deadline_returns_timeout() {
        use crate::simplex::pricing::DantzigPricing;
        let a = CscMatrix::from_triplets(
            &[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3,
        ).unwrap();
        let c = vec![-1.0, -1.0, 0.0];
        let mut x_b = vec![4.0];
        let mut basis = vec![2usize];
        let mut pricing = DantzigPricing;
        // max_etas=1 forces an early refactor.
        let opts = SolverOptions {
            deadline: None,
            max_etas: 1,
            ..SolverOptions::default()
        };
        let b = vec![4.0];
        let mut iters = 0usize;
        let outcome = revised_simplex_core(
            &a, &mut x_b, &c, &b, &mut basis, 1, 3, 3, &mut pricing, &opts, &mut iters, false,
        );
        assert!(matches!(
            outcome,
            SimplexOutcome::Optimal(..) | SimplexOutcome::Timeout(_) | SimplexOutcome::SingularBasis
        ));
    }

    /// timeout_secs=0 must propagate to Timeout (small LP path).
    #[test]
    fn test_presolve_respects_deadline_small() {
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
            timeout_secs: Some(0.0),
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// At n=2000/m=1000, presolve must early-return on past deadline (no budget overrun).
    #[test]
    fn test_large_scale_presolve_respects_deadline() {
        let n = 2000usize;
        let m = 1000usize;
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
        let start = std::time::Instant::now();
        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        let elapsed = start.elapsed();
        assert_eq!(result.status, SolveStatus::Timeout);
        assert!(elapsed.as_secs_f64() < 0.5, "elapsed={:.3}s", elapsed.as_secs_f64());
    }

    /// Wall-clock must stay within K · timeout_secs.
    #[test]
    fn test_timeout_elapsed_within_budget() {
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
        let timeout_secs = 0.01f64;
        let opts = SolverOptions {
            timeout_secs: Some(timeout_secs),
            presolve: false,
            ..SolverOptions::default()
        };
        let start = std::time::Instant::now();
        let result = solve_with(&lp, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(matches!(result.status, SolveStatus::Timeout | SolveStatus::Optimal));
        assert!(elapsed < timeout_secs * 3.0 + 0.5, "elapsed={:.3}s", elapsed);
    }

    /// timeout_secs=None must still converge on a tractable LP.
    #[test]
    fn test_no_deadline_converges_finite() {
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
            timeout_secs: None,
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// extract_dual_info must subtract the upper-bound dual mu_j from rc.
    /// min -2x1-x2 s.t. x1+x2≤4, 0≤x1≤2, 0≤x2≤3  ⇒  x=(2,2).
    /// Missing mu_j would give rc[0] = -2-lambda ≠ 0 (complementarity error).
    #[test]
    fn test_extract_dual_info_ub_dual() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let problem = LpProblem::new_general(
            vec![-2.0, -1.0],
            a,
            vec![4.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 3.0)],
            None,
        )
        .unwrap();
        let opts = SolverOptions { timeout_secs: None, presolve: false, ..SolverOptions::default() };
        let result = solve_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "status should be Optimal");

        let x = &result.solution;
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0]={} should be at upper bound 2.0", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "x[1]={} should be 2.0", x[1]);

        let rc = &result.reduced_costs;

        // x[0] is at upper bound (x[0] = ub = 2) → rc[0] ≤ 0
        // If mu_j subtraction is missing, rc[0] = c[0] - lambda*a[0,0] = -1 - (-2) = 1 > 0
        assert!(rc[0] <= 1e-6, "rc[0]={} should be <= 0 (x[0] at upper bound; mu_j subtraction required)", rc[0]);

        // x[1] is strictly between bounds (0 < x[1]=2 < 3) → x[1] is basic → rc[1] ≈ 0
        assert!(rc[1].abs() < 1e-6, "rc[1]={} should be ≈ 0 (x[1] is basic)", rc[1]);

        // Upper complementarity for x[0]: (ub - x[0]) * max(-rc[0], 0) ≈ 0
        let ub0 = 2.0_f64;
        let upper_comp = (ub0 - x[0]) * (-rc[0]).max(0.0);
        assert!(upper_comp.abs() < 1e-8, "upper complementarity={} should be ≈ 0", upper_comp);
    }

    /// Degenerate Eq(b=0) artificials must not yield NumericalError.
    /// min -x4 s.t. x1+x2=0, x1+x3=0, x2+x4=1, x1+x4≤2, x≥0  ⇒ x=(0,0,0,1).
    #[test]
    fn test_degenerate_eq_zero_rhs_artificials() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 1, 3, 0, 2, 1, 2, 3],
            &[0, 0, 0, 1, 1, 2, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
            4,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 0.0, -1.0],
            a,
            vec![0.0, 0.0, 1.0, 2.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY); 4],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(result.status, SolveStatus::NumericalError);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective - (-1.0)).abs() < 1e-6);
    }

    /// Many b=0 Eq constraints (wood1p-style) must not yield NumericalError.
    /// min -x5 s.t. x1+x2=0, x2+x3=0, x3+x4=0, x1+x5=1, sum≤2, x≥0  ⇒ x5=1.
    #[test]
    fn test_multiple_zero_rhs_eq_artificials() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 3, 4, 0, 1, 4, 1, 2, 4, 2, 4, 3, 4],
            &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            5,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 0.0, 0.0, -1.0],
            a,
            vec![0.0, 0.0, 0.0, 1.0, 2.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(result.status, SolveStatus::NumericalError);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective - (-1.0)).abs() < 1e-6);
    }

    /// hs51 (free vars + Le): degenerate-artificial pivot must not singularize
    /// the basis (best_j=None fallback keeps it safe).
    #[test]
    fn test_hs51_free_var_no_singular_basis() {
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
            &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            &[
                1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0,
            ],
            6,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0; 5],
            a,
            vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            vec![crate::problem::ConstraintType::Le; 6],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(result.status, SolveStatus::NumericalError);
        assert_eq!(result.status, SolveStatus::Optimal);
    }
}

/// DualAdvanced warm-start: ensures `dual_simplex_core_advanced` is reached
/// and matches the cold-start optimum.
#[cfg(test)]
mod tests_dual_advanced {
    use super::*;
    use crate::sparse::CscMatrix;

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

    /// LP1 (obj=-7) → reuse basis on LP2 with RHS=[5,3,3] (obj=-8).
    #[test]
    fn test_dual_advanced_warm_start_rhs_change() {
        let lp1 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        // LP1 を default solver で解いて warm_start_basis を取得
        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(
            result1.warm_start_basis.is_some(),
            "LP1 は warm_start_basis を返すべき"
        );

        // LP2: RHS のみ変更 b=[5, 3, 3]
        let lp2 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 3.0, 3.0],
        );

        // cold-start で正解を確認
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // DualAdvanced warm-start で解く → warm-start 経路を通す
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::DualAdvanced,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);

        assert_eq!(
            result2_warm.status,
            SolveStatus::Optimal,
            "DualAdvanced warm-start は Optimal を返すべき"
        );
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "DualAdvanced warm-start obj={}, cold-start obj={}",
            result2_warm.objective,
            result2_cold.objective
        );
    }

    /// LP1 (obj=-4) → reuse basis on LP2 with RHS=[6,4,4] (obj=-8).
    #[test]
    fn test_dual_advanced_warm_start_larger_rhs() {
        let lp1 = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(
            result1.warm_start_basis.is_some(),
            "LP1 は warm_start_basis を返すべき"
        );

        // LP2: RHS 拡大 b=[6, 4, 4] → 最適解 x1+x2=8, obj=-8
        let lp2 = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![6.0, 4.0, 4.0],
        );

        // cold-start で正解を確認
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // DualAdvanced warm-start で解く
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::DualAdvanced,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);

        assert_eq!(
            result2_warm.status,
            SolveStatus::Optimal,
            "DualAdvanced warm-start (larger RHS) は Optimal を返すべき"
        );
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "DualAdvanced warm-start obj={}, cold-start obj={}",
            result2_warm.objective,
            result2_cold.objective
        );
    }

    #[test]
    fn test_scsd6_equality_constraints() {
        // scsd6: network flow LP with 147 all-equality constraints, 1350 vars.
        // Reported as NumericalError in 0.024s.
        let path = std::path::Path::new("data/lp_problems/scsd6.QPS");
        if !path.exists() {
            return;
        }
        let content = std::fs::read_to_string(path).unwrap();
        let lp = crate::io::mps::parse_mps(&content).unwrap();

        // Test each method independently to isolate the bug
        let methods = [
            ("Auto", SimplexMethod::Auto),
            ("Primal", SimplexMethod::Primal),
            ("Dual", SimplexMethod::Dual),
        ];
        let results: Vec<_> = methods.iter().map(|(name, method)| {
            let mut opts = SolverOptions::default();
            opts.simplex_method = *method;
            opts.presolve = false;
            let result = solve_with(&lp, &opts);
            eprintln!("scsd6 {} -> {:?} obj={:.3e}", name, result.status, result.objective);
            (*name, result.status)
        }).collect();

        for (name, status) in &results {
            assert_ne!(
                *status, SolveStatus::NumericalError,
                "scsd6 {} returned NumericalError",
                name
            );
        }
    }
}
