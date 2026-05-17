//! Postsolve: lift a reduced LP's solution back to the original variable / constraint
//! space by replaying `PostsolveStack` in LIFO order.

use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::options::SolverOptions;
use crate::tolerances::PIVOT_TOL;
use super::transforms::{PostsolveStep, PresolveResult};
use std::time::Instant;

/// Build and solve a cleanup LP that recovers `y_i` for deleted rows (and optionally a
/// perturbation on kept rows) so the full dual is KKT-consistent.
///
/// Phase 1 minimises `Σ slack` for feasibility; Phase 2 fixes the Phase-1 slack and
/// minimises `Σ|y_del| + Σ|dy|` to break ties. Kept-row perturbation is required when
/// kept↔deleted coupling is strong; it is disabled above `CLEANUP_LP_KEPT_PERT_SIZE_LIMIT`.
/// Returns an `m`-sized y vector, or `None` on construction/solve failure.
fn build_and_solve_cleanup_lp(
    orig_problem: &LpProblem,
    presolve_result: &PresolveResult,
    solution: &[f64],
    dual_solution_known: &[f64],
    deadline: Option<Instant>,
    allow_kept_perturbation: bool,
) -> Option<Vec<f64>> {
    // Bail if the parent deadline has already lapsed; a `None` deadline means
    // the caller opted into unbounded runtime (required by KKT-accuracy unit tests).
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return None;
        }
    }
    let n = orig_problem.num_vars;
    let m = orig_problem.num_constraints;
    let deleted_rows: Vec<usize> = (0..m)
        .filter(|&i| presolve_result.row_map[i].is_none())
        .collect();
    let k = deleted_rows.len();
    if k == 0 { return None; }

    let row_to_var: std::collections::HashMap<usize, usize> = deleted_rows
        .iter().enumerate().map(|(idx, &r)| (r, idx)).collect();

    let use_kept_perturbation =
        allow_kept_perturbation && n + m <= CLEANUP_LP_KEPT_PERT_SIZE_LIMIT;
    // Take the bipartite closure (deleted rows ↔ columns ↔ kept rows) so that any
    // kept row whose `y` is coupled to a deleted row gets a `dy` perturbation variable.
    // A naive 1-pass (only kept rows sharing a column with a deleted row) misses
    // indirectly-coupled violation columns and leaves Phase-1 slack non-zero.
    let coupled_kept: Vec<usize> = if use_kept_perturbation {
        // Inverted index row → cols (CSC is col-major; row traversal is otherwise slow).
        let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
        for j in 0..n {
            if let Ok((rows, _)) = orig_problem.a.get_column(j) {
                for &row in rows {
                    row_to_cols[row].push(j);
                }
            }
        }
        let mut col_affected: Vec<bool> = vec![false; n];
        let mut col_queue: Vec<usize> = Vec::new();
        for &del_row in &deleted_rows {
            for &j in &row_to_cols[del_row] {
                if !col_affected[j] {
                    col_affected[j] = true;
                    col_queue.push(j);
                }
            }
        }
        for step in &presolve_result.postsolve_stack {
            if let PostsolveStep::LinearSubstitution { orig_col, .. } = step {
                let j = *orig_col;
                if !col_affected[j] {
                    col_affected[j] = true;
                    col_queue.push(j);
                }
            }
        }
        let mut kept_in_set: Vec<bool> = vec![false; m];
        let mut coupled: Vec<usize> = Vec::new();
        let mut head = 0usize;
        while head < col_queue.len() {
            let j = col_queue[head];
            head += 1;
            if let Ok((rows, _)) = orig_problem.a.get_column(j) {
                for &row in rows {
                    if presolve_result.row_map[row].is_some() && !kept_in_set[row] {
                        kept_in_set[row] = true;
                        coupled.push(row);
                        for &j2 in &row_to_cols[row] {
                            if !col_affected[j2] {
                                col_affected[j2] = true;
                                col_queue.push(j2);
                            }
                        }
                    }
                }
            }
        }
        coupled
    } else {
        Vec::new()
    };
    let row_to_kept_var: std::collections::HashMap<usize, usize> =
        coupled_kept.iter().enumerate().map(|(idx, &r)| (r, idx)).collect();
    let m_kept = coupled_kept.len();

    // Variable layout: [y_del | dy | slack].
    let m_kept_var = if use_kept_perturbation { m_kept } else { 0 };
    let dy_offset = k;
    let slack_offset = k + m_kept_var;

    // rc_known[j] = c[j] - Σ_{i: kept} A_ij * y_kept[i]. Deleted-row y is what we solve for.
    let mut rc_known = orig_problem.c.clone();
    for j in 0..n {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if presolve_result.row_map[row].is_some() {
                    rc_known[j] -= vals[kk] * dual_solution_known[row];
                }
            }
        }
    }

    let mut tri_rows: Vec<usize> = Vec::new();
    let mut tri_cols: Vec<usize> = Vec::new();
    let mut tri_vals: Vec<f64> = Vec::new();
    let mut b_clean: Vec<f64> = Vec::new();
    let mut ct_clean: Vec<ConstraintType> = Vec::new();

    // (i) rc-sign constraints for non-fixed columns j.
    for j in 0..n {
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; }

        let mut col_terms: Vec<(usize, f64)> = Vec::new();
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if let Some(&var_idx) = row_to_var.get(&row) {
                    col_terms.push((var_idx, vals[kk]));
                } else if use_kept_perturbation {
                    if let Some(&kept_idx) = row_to_kept_var.get(&row) {
                        col_terms.push((dy_offset + kept_idx, vals[kk]));
                    }
                }
            }
        }
        if col_terms.is_empty() { continue; }

        // Complementary slackness sign on rc[j]: at lb → rc ≥ 0, at ub → rc ≤ 0,
        // interior → rc = 0. Phase-1 slack absorbs any infeasibility from degeneracy.
        let ct = if at_lb && !at_ub {
            ConstraintType::Le
        } else if at_ub && !at_lb {
            ConstraintType::Ge
        } else {
            ConstraintType::Eq
        };
        let row_idx = b_clean.len();
        for &(var_idx, a) in &col_terms {
            tri_rows.push(row_idx);
            tri_cols.push(var_idx);
            tri_vals.push(a);
        }
        b_clean.push(rc_known[j]);
        ct_clean.push(ct);
    }

    // (ii) Free-variable stationarity rc[orig_col] = 0 for each LinearSubstitution.
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::LinearSubstitution { orig_col, .. } = step {
            let j = *orig_col;
            let mut col_terms: Vec<(usize, f64)> = Vec::new();
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (kk, &row) in rows.iter().enumerate() {
                    if let Some(&var_idx) = row_to_var.get(&row) {
                        col_terms.push((var_idx, vals[kk]));
                    } else if use_kept_perturbation {
                        if let Some(&kept_idx) = row_to_kept_var.get(&row) {
                            col_terms.push((dy_offset + kept_idx, vals[kk]));
                        }
                    }
                }
            }
            if col_terms.is_empty() { continue; }
            let row_idx = b_clean.len();
            for &(var_idx, a) in &col_terms {
                tri_rows.push(row_idx);
                tri_cols.push(var_idx);
                tri_vals.push(a);
            }
            b_clean.push(rc_known[j]);
            ct_clean.push(ConstraintType::Eq);
        }
    }

    if b_clean.is_empty() { return None; }

    // Add Phase-1 slack to guarantee feasibility: Le/Ge use one slack, Eq uses ± pair.
    // Objective `min Σ slack` returns 0 iff exact rc-sign satisfaction is possible.
    let m_clean = b_clean.len();
    let mut slack_count = 0usize;
    let mut slack_cols_per_row: Vec<(usize, Option<usize>)> = Vec::with_capacity(m_clean);
    for ct in &ct_clean {
        match ct {
            ConstraintType::Eq => {
                let pos = slack_offset + slack_count;
                let neg = slack_offset + slack_count + 1;
                slack_cols_per_row.push((pos, Some(neg)));
                slack_count += 2;
            }
            _ => {
                slack_cols_per_row.push((slack_offset + slack_count, None));
                slack_count += 1;
            }
        }
    }
    for (row_idx, (s_pos, s_neg_opt)) in slack_cols_per_row.iter().enumerate() {
        let sign = match ct_clean[row_idx] {
            ConstraintType::Le => -1.0,
            ConstraintType::Ge => 1.0,
            ConstraintType::Eq => 1.0,
        };
        tri_rows.push(row_idx);
        tri_cols.push(*s_pos);
        tri_vals.push(sign);
        if let Some(s_neg) = s_neg_opt {
            tri_rows.push(row_idx);
            tri_cols.push(*s_neg);
            tri_vals.push(-1.0);
        }
    }
    let total_vars = slack_offset + slack_count;

    // Variable bounds: y_del follows the row's sign convention; dy is shifted by -y_kept[i]
    // so y_kept + dy still satisfies the sign convention; slack ∈ [0, ∞).
    let mut bounds_clean: Vec<(f64, f64)> = Vec::with_capacity(total_vars);
    for &i in &deleted_rows {
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => bounds_clean.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => bounds_clean.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => bounds_clean.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    if use_kept_perturbation {
        for &i in &coupled_kept {
            let y_kept_i = dual_solution_known[i];
            match orig_problem.constraint_types[i] {
                ConstraintType::Le => bounds_clean.push((f64::NEG_INFINITY, -y_kept_i)),
                ConstraintType::Ge => bounds_clean.push((-y_kept_i, f64::INFINITY)),
                ConstraintType::Eq => bounds_clean.push((f64::NEG_INFINITY, f64::INFINITY)),
            }
        }
    }
    for _ in 0..slack_count {
        bounds_clean.push((0.0, f64::INFINITY));
    }

    let mut c_clean = vec![0.0f64; total_vars];
    for j in slack_offset..total_vars { c_clean[j] = 1.0; }

    let a_clean = CscMatrix::from_triplets(
        &tri_rows, &tri_cols, &tri_vals, m_clean, total_vars
    ).ok()?;
    let b_clean_keep = b_clean.clone();
    let ct_clean_keep = ct_clean.clone();
    let cleanup_lp = LpProblem::new_general(
        c_clean, a_clean, b_clean, ct_clean, bounds_clean, None
    ).ok()?;

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.warm_start = None;
    // Wire the parent deadline straight through so every inner stage (parse, scale,
    // factorize, simplex iterate) checks the same clock; otherwise large cleanup
    // LPs can spend minutes in setup before any per-call budget kicks in.
    opts.deadline = deadline;
    let r1 = crate::simplex::solve_without_presolve(&cleanup_lp, &opts);
    let _ = (slack_count, m_clean);
    if r1.status != SolveStatus::Optimal || r1.solution.len() != total_vars {
        return None;
    }
    let y_del_phase1: Vec<f64> = r1.solution[..k].to_vec();
    let dy_phase1: Vec<f64> = if use_kept_perturbation {
        r1.solution[dy_offset..dy_offset + m_kept_var].to_vec()
    } else {
        Vec::new()
    };
    let slack_phase1: Vec<f64> = r1.solution[slack_offset..].to_vec();
    // Combine: deleted rows use y_del, coupled kept rows use y_kept + dy,
    // non-coupled kept rows keep their known dual.
    let assemble_full_y = |y_del: &[f64], dy: &[f64]| -> Vec<f64> {
        let mut y = dual_solution_known.to_vec();
        for (idx, &row) in deleted_rows.iter().enumerate() {
            y[row] = y_del[idx];
        }
        if use_kept_perturbation {
            for (idx, &row) in coupled_kept.iter().enumerate() {
                y[row] = dual_solution_known[row] + dy[idx];
            }
        }
        y
    };

    // Phase 2 tie-break: fix Phase-1 slack and minimise `Σ|y_del| + Σ|dy|` so
    // dual degeneracy cannot pick an arbitrary large-|y| solution. Layout:
    //   [y_del | dy | d_pos | d_neg], with Eq rows `(y_del|dy)[i] - d_pos[i] + d_neg[i] = 0`.
    let n_yvars = k + m_kept_var;
    let phase2_total_vars = 3 * n_yvars;
    let phase2_total_cons = m_clean + n_yvars;
    let mut p2_tri_rows: Vec<usize> = Vec::with_capacity(tri_rows.len() + 3 * n_yvars);
    let mut p2_tri_cols: Vec<usize> = Vec::with_capacity(tri_rows.len() + 3 * n_yvars);
    let mut p2_tri_vals: Vec<f64> = Vec::with_capacity(tri_rows.len() + 3 * n_yvars);
    let mut p2_b: Vec<f64> = Vec::with_capacity(phase2_total_cons);
    let mut p2_ct: Vec<ConstraintType> = Vec::with_capacity(phase2_total_cons);
    let p2_slack_offset = slack_offset;
    // (i) Replicate Phase-1 a*y constraints without slack, with RHS relaxed by Phase-1 slack.
    for (orig_idx, (slack_pos, slack_neg_opt)) in slack_cols_per_row.iter().enumerate() {
        for (k_t, &row) in tri_rows.iter().enumerate() {
            if row != orig_idx { continue; }
            let col = tri_cols[k_t];
            if col >= p2_slack_offset { continue; }
            p2_tri_rows.push(orig_idx);
            p2_tri_cols.push(col);
            p2_tri_vals.push(tri_vals[k_t]);
        }
        let s_p_val = slack_phase1[*slack_pos - p2_slack_offset];
        let rhs = match ct_clean_keep[orig_idx] {
            ConstraintType::Le => b_clean_keep[orig_idx] + s_p_val,
            ConstraintType::Ge => b_clean_keep[orig_idx] - s_p_val,
            ConstraintType::Eq => {
                let s_n_val = slack_phase1[slack_neg_opt.unwrap() - p2_slack_offset];
                b_clean_keep[orig_idx] - s_p_val + s_n_val
            }
        };
        p2_b.push(rhs);
        p2_ct.push(ct_clean_keep[orig_idx].clone());
    }
    // (ii) Tie-break Eq rows: (y_del|dy)[i] - d_pos[i] + d_neg[i] = 0.
    for i in 0..n_yvars {
        let row_idx = m_clean + i;
        p2_tri_rows.push(row_idx); p2_tri_cols.push(i);                  p2_tri_vals.push(1.0);
        p2_tri_rows.push(row_idx); p2_tri_cols.push(n_yvars + i);        p2_tri_vals.push(-1.0);
        p2_tri_rows.push(row_idx); p2_tri_cols.push(2 * n_yvars + i);    p2_tri_vals.push(1.0);
        p2_b.push(0.0);
        p2_ct.push(ConstraintType::Eq);
    }
    let mut p2_bounds: Vec<(f64, f64)> = Vec::with_capacity(phase2_total_vars);
    for &i in &deleted_rows {
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => p2_bounds.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => p2_bounds.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => p2_bounds.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    if use_kept_perturbation {
        for &i in &coupled_kept {
            let y_kept_i = dual_solution_known[i];
            match orig_problem.constraint_types[i] {
                ConstraintType::Le => p2_bounds.push((f64::NEG_INFINITY, -y_kept_i)),
                ConstraintType::Ge => p2_bounds.push((-y_kept_i, f64::INFINITY)),
                ConstraintType::Eq => p2_bounds.push((f64::NEG_INFINITY, f64::INFINITY)),
            }
        }
    }
    for _ in 0..(2 * n_yvars) { p2_bounds.push((0.0, f64::INFINITY)); }
    let mut p2_c = vec![0.0f64; phase2_total_vars];
    for j in n_yvars..(3 * n_yvars) { p2_c[j] = 1.0; }

    let p2_a = match CscMatrix::from_triplets(
        &p2_tri_rows, &p2_tri_cols, &p2_tri_vals, phase2_total_cons, phase2_total_vars
    ) {
        Ok(m) => m,
        Err(_) => return Some(assemble_full_y(&y_del_phase1, &dy_phase1)),
    };
    let p2_lp = match LpProblem::new_general(p2_c, p2_a, p2_b, p2_ct, p2_bounds, None) {
        Ok(l) => l,
        Err(_) => return Some(assemble_full_y(&y_del_phase1, &dy_phase1)),
    };
    let r2 = crate::simplex::solve_without_presolve(&p2_lp, &opts);
    if r2.status == SolveStatus::Optimal && r2.solution.len() == phase2_total_vars {
        let y_del_p2: Vec<f64> = r2.solution[..k].to_vec();
        let dy_p2: Vec<f64> = if use_kept_perturbation {
            r2.solution[dy_offset..dy_offset + m_kept_var].to_vec()
        } else {
            Vec::new()
        };
        Some(assemble_full_y(&y_del_p2, &dy_p2))
    } else {
        Some(assemble_full_y(&y_del_phase1, &dy_phase1))
    }
}

/// Size threshold above which kept-y perturbation is disabled (cleanup LP would
/// otherwise grow unmanageably). Mirrors `LSQ_DUAL_SIZE_LIMIT` in `compute_lsq_dual_y`.
const CLEANUP_LP_KEPT_PERT_SIZE_LIMIT: usize = 50_000;

/// Enumerate row `i`'s entries `(j, A_ij)` from a CSC matrix in O(nnz_total).
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

/// Distance below which `x[j]` is treated as active at its lb / ub.
const BOUND_ACTIVE_TOL: f64 = 1e-6;

/// Recover `y_i` of a removed row to satisfy LP dual feasibility, given the rest of `y`.
/// For each column the required rc sign yields a permissible range on `y_i`; the row's
/// constraint type (Le: y≤0, Ge: y≥0, Eq: free) intersects that range and we pick the
/// value closest to zero.
fn recover_removed_row_dual(
    orig_problem: &LpProblem,
    i: usize,
    solution: &[f64],
    dual_solution: &[f64],
) -> f64 {
    let row_entries = collect_row_entries(orig_problem, i);
    let mut min_y_i = f64::NEG_INFINITY;
    let mut max_y_i = f64::INFINITY;
    for &(j, a_ij) in &row_entries {
        if a_ij.abs() < f64::EPSILON { continue; }
        let mut rc_at_y0 = orig_problem.c[j];
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc_at_y0 -= vals[k] * dual_solution[row];
            }
        }
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; }
        let bound_val = rc_at_y0 / a_ij;
        if at_lb && !at_ub {
            if a_ij > 0.0 {
                if bound_val < max_y_i { max_y_i = bound_val; }
            } else {
                if bound_val > min_y_i { min_y_i = bound_val; }
            }
        } else if at_ub && !at_lb {
            if a_ij > 0.0 {
                if bound_val > min_y_i { min_y_i = bound_val; }
            } else {
                if bound_val < max_y_i { max_y_i = bound_val; }
            }
        } else {
            if bound_val < max_y_i { max_y_i = bound_val; }
            if bound_val > min_y_i { min_y_i = bound_val; }
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
        if lb_y <= 0.0 && ub_y >= 0.0 { 0.0 }
        else if ub_y < 0.0 { ub_y }
        else { lb_y }
    } else {
        0.0
    }
}

/// Lift the reduced-problem solution back into the original variable / constraint space.
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

    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < result.solution.len() {
                solution[j] = result.solution[jj];
            }
        }
    }
    for (i, &maybe_ii) in presolve_result.row_map.iter().enumerate() {
        if let Some(ii) = maybe_ii {
            if ii < result.dual_solution.len() {
                dual_solution[i] = result.dual_solution[ii];
            }
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
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::SingletonRow { orig_col, orig_row, value, a_ij: _, c_j: _ } => {
                solution[*orig_col] = *value;
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::BoundsTightened { .. } => {}
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

    // Compute several deleted-row y candidates and adopt whichever achieves the smallest
    // bound-aware dual-feasibility violation. Cleanup-LP alone is not guaranteed to be
    // dual-feasible under dual degeneracy, so it is compared against the Gauss-Seidel path.
    let y_loop = dual_solution.clone();

    // Gauss-Seidel: iterate `recover_removed_row_dual` and the LinearSubstitution y_piv.
    // The deadline is checked at the outer loop and every 1024 rows so very large
    // postsolves cannot ignore the parent budget.
    let y_gs = {
        let mut y = y_loop.clone();
        let mut linsub_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for step in &presolve_result.postsolve_stack {
            if let PostsolveStep::LinearSubstitution { orig_row: Some(r), .. } = step {
                linsub_rows.insert(*r);
            }
        }
        let max_iter = 50;
        let conv_tol = 1e-12;
        'gs_outer: for _ in 0..max_iter {
            if deadline.is_some_and(|d| Instant::now() >= d) { break 'gs_outer; }
            let mut max_diff = 0.0f64;
            for i in 0..m {
                if presolve_result.row_map[i].is_some() { continue; }
                if linsub_rows.contains(&i) { continue; }
                if i & 0x3ff == 0 && deadline.is_some_and(|d| Instant::now() >= d) {
                    break 'gs_outer;
                }
                let new_y = recover_removed_row_dual(orig_problem, i, &solution, &y);
                let diff = (y[i] - new_y).abs();
                if diff > max_diff { max_diff = diff; }
                y[i] = new_y;
            }
            for step in &presolve_result.postsolve_stack {
                if let PostsolveStep::LinearSubstitution {
                    orig_row: Some(piv),
                    col_orig_entries,
                    c_orig,
                    pivot,
                    ..
                } = step {
                    let mut sum = 0.0f64;
                    for &(row_i, a_ij) in col_orig_entries {
                        if row_i == *piv { continue; }
                        sum += a_ij * y[row_i];
                    }
                    let new_y = (c_orig - sum) / pivot;
                    let diff = (y[*piv] - new_y).abs();
                    if diff > max_diff { max_diff = diff; }
                    y[*piv] = new_y;
                }
            }
            if max_diff < conv_tol { break 'gs_outer; }
        }
        y
    };

    // Build dfeas_bound first so the cheap candidates (y_loop, y_gs) can gate the
    // far more expensive cleanup-LP candidates.
    let mut bound_tightened_fixed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::FixedVariable { orig_col, .. } = step {
            let (lb, ub) = orig_problem.bounds[*orig_col];
            let truly_fixed = lb.is_finite() && ub.is_finite()
                && (ub - lb).abs() < BOUND_ACTIVE_TOL;
            if !truly_fixed {
                bound_tightened_fixed.insert(*orig_col);
            }
        }
    }
    let dfeas_bound = |y: &[f64]| -> f64 {
        let mut max_viol = 0.0f64;
        for j in 0..n {
            if bound_tightened_fixed.contains(&j) { continue; }
            let (lb_j, ub_j) = orig_problem.bounds[j];
            let fixed = lb_j.is_finite() && ub_j.is_finite()
                && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
            if fixed { continue; }
            let at_lb = lb_j.is_finite() && (solution[j] - lb_j).abs() < BOUND_ACTIVE_TOL;
            let at_ub = ub_j.is_finite() && (solution[j] - ub_j).abs() < BOUND_ACTIVE_TOL;
            let mut rc = orig_problem.c[j];
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    rc -= vals[k] * y[row];
                }
            }
            let viol = if at_lb && !at_ub { f64::max(0.0, -rc) }
                else if at_ub && !at_lb { f64::max(0.0, rc) }
                else { 0.0 };
            if viol > max_viol { max_viol = viol; }
        }
        max_viol
    };

    let df_loop = dfeas_bound(&y_loop);
    let df_gs = dfeas_bound(&y_gs);
    let cheap_min = df_loop.min(df_gs);

    // Gate at the strictest LP feasibility eps used by the bench (`PIVOT_TOL`);
    // below this, cleanup LP cannot improve the verdict and only costs runtime.
    let gate = PIVOT_TOL;

    let (y_cl_nopert, y_cl_pert) = if cheap_min <= gate {
        (None, None)
    } else {
        let t0_nopert = std::time::Instant::now();
        let y_nopert = build_and_solve_cleanup_lp(
            orig_problem, presolve_result, &solution, &y_gs, deadline, false,
        );
        let t_nopert = t0_nopert.elapsed();
        let df_nopert = y_nopert.as_ref().map_or(f64::INFINITY, |y| dfeas_bound(y));
        let so_far = cheap_min.min(df_nopert);
        // The kept-y perturbation variant is much larger and often returns Inf dfeas;
        // budget it at a small multiple of the plain variant's wall time.
        let y_pert = if so_far <= gate {
            None
        } else {
            let now = std::time::Instant::now();
            let pert_budget = t_nopert.saturating_mul(4);
            let pert_deadline = match deadline {
                Some(d) => Some(d.min(now + pert_budget)),
                None => Some(now + pert_budget),
            };
            build_and_solve_cleanup_lp(
                orig_problem, presolve_result, &solution, &y_gs, pert_deadline, true,
            )
        };
        (y_nopert, y_pert)
    };

    // Compute cleanup dfeas before the LSQ gate so we can decide whether the
    // LSQ pass is worth the (often dominant, see #38) runtime.
    let df_cl_nopert = y_cl_nopert.as_ref().map_or(f64::INFINITY, |y| dfeas_bound(y));
    let df_cl_pert = y_cl_pert.as_ref().map_or(f64::INFINITY, |y| dfeas_bound(y));
    let df_cl_min = df_cl_nopert.min(df_cl_pert);

    // When both cleanup variants failed to improve the cheap candidates beyond
    // numerical drift, LSQ shares the same data path (A, c, x*) and is expected
    // to stagnate as well; running it only burns budget (dfl001: 98% of ~3s
    // postsolve). The 0.1% relative-improvement floor lets genuine cleanup
    // progress (≥0.1% of cheap_min) still trigger LSQ.
    const LSQ_CLEANUP_REL_IMPROVE: f64 = 1e-3;
    let cleanup_stagnant = df_cl_min.is_finite()
        && df_cl_min >= cheap_min * (1.0 - LSQ_CLEANUP_REL_IMPROVE);

    // LSQ projection (A^T y ≈ -c) as a fourth candidate. Cleanup LP only adjusts
    // deleted-row y; LSQ ignores the kept/deleted boundary and can rebalance the
    // full y vector when coupling is strong.
    let y_lsq: Option<Vec<f64>> = if cheap_min <= gate || cleanup_stagnant {
        #[cfg(debug_assertions)]
        if cleanup_stagnant {
            eprintln!(
                "[postsolve] LSQ skip: improvement-stagnant (cheap_min={:.3e} df_cl_min={:.3e})",
                cheap_min, df_cl_min
            );
        }
        None
    } else {
        const LSQ_DUAL_SIZE_LIMIT: usize = 50_000;
        if n + m <= LSQ_DUAL_SIZE_LIMIT && m > 0 {
            let q_empty = CscMatrix::new(n, n);
            let qp = crate::qp::QpProblem::new(
                q_empty,
                orig_problem.c.clone(),
                orig_problem.a.clone(),
                orig_problem.b.clone(),
                orig_problem.bounds.clone(),
                orig_problem.constraint_types.clone(),
            ).ok();
            qp.and_then(|qp| {
                let seed = y_cl_pert
                    .as_ref()
                    .or(y_cl_nopert.as_ref())
                    .cloned()
                    .unwrap_or_else(|| y_gs.clone());
                let tmp_result = crate::problem::SolverResult {
                    solution: solution.clone(),
                    dual_solution: seed,
                    ..Default::default()
                };
                crate::qp::compute_lsq_dual_y(&qp, &tmp_result, deadline)
            })
        } else {
            None
        }
    };

    // Adopt the candidate with smallest dfeas_bound; ties go to the cheaper computation.
    let df_lsq = y_lsq.as_ref().map_or(f64::INFINITY, |y| dfeas_bound(y));
    let min_df = df_loop
        .min(df_gs)
        .min(df_cl_nopert)
        .min(df_cl_pert)
        .min(df_lsq);
    if df_loop == min_df {
        dual_solution = y_loop;
    } else if df_gs == min_df {
        dual_solution = y_gs;
    } else if df_cl_nopert == min_df {
        dual_solution = y_cl_nopert.expect("df_cl_nopert finite implies Some");
    } else if df_cl_pert == min_df {
        dual_solution = y_cl_pert.expect("df_cl_pert finite implies Some");
    } else {
        dual_solution = y_lsq.expect("df_lsq finite implies Some");
    }

    // Recompute reduced costs on the original problem now that the dual is final:
    //   reduced_cost[j] = c[j] - Σ_i A_ij · y_i.
    let mut reduced_costs = orig_problem.c.clone();
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(n) {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc -= vals[k] * dual_solution[row];
            }
        }
    }
    // For variables fixed by bound tightening (not lb==ub originally), the rc can be
    // absorbed into the bound dual (mu_lb − mu_ub); treat it as zero to keep dual
    // feasibility consistent. True fixed variables (orig lb == ub) keep their rc.
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::FixedVariable { orig_col, .. } = step {
            let (lb, ub) = orig_problem.bounds[*orig_col];
            let truly_fixed = lb.is_finite() && ub.is_finite()
                && (ub - lb).abs() < BOUND_ACTIVE_TOL;
            if !truly_fixed && *orig_col < n {
                reduced_costs[*orig_col] = 0.0;
            }
        }
    }

    let postsolve_dfeas_recomputed = dfeas_bound(&dual_solution);

    let objective = result.objective + presolve_result.obj_offset;

    SolverResult {
        status: result.status.clone(),
        objective,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        warm_start_basis: None,
        iterations: result.iterations,
        postsolve_dfeas: Some(postsolve_dfeas_recomputed),
        ..Default::default()
    }
}
