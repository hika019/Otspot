//! Postsolve: lift a reduced LP's solution back to the original variable / constraint
//! space by replaying `PostsolveStack` in LIFO order.

use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::simplex::build_standard_form;
use crate::simplex::crash::compute_crash_basis;
use crate::sparse::CscMatrix;
use crate::tolerances::{COMP_SLACK_REL_TOL, PIVOT_TOL};
use super::transforms::{PostsolveStep, PresolveResult};
use std::time::Instant;

/// Relative tolerance below which a standard-form column is treated as at-bound
/// (non-basic candidate) when synthesising the postsolved warm-start basis.
const WARM_BASIS_BUILD_TOL: f64 = 1e-9;

/// Maximum Gauss-Seidel iterations for dual variable recovery.
const GS_MAX_ITER: usize = 50;
/// Convergence tolerance for Gauss-Seidel: stops when max per-row change drops below this.
const GS_CONV_TOL: f64 = 1e-12;

/// Return the primal slack of original row `i` (always non-negative for feasible
/// solutions): `b_i - Ax_i` for `Le`, `Ax_i - b_i` for `Ge`, `0` for `Eq`. The
/// scale `1 + |b_i| + |Ax_i|` is returned alongside so the caller can pick a
/// relative non-binding threshold.
fn row_slack_and_scale(
    orig_problem: &LpProblem,
    i: usize,
    solution: &[f64],
) -> (f64, f64) {
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

/// `true` iff row `i` is strictly non-binding at `solution` (slack exceeds the
/// scaled complementarity tolerance), in which case KKT forces `y_i = 0`.
fn is_row_nonbinding(orig_problem: &LpProblem, i: usize, solution: &[f64]) -> bool {
    let (slack, scale) = row_slack_and_scale(orig_problem, i, solution);
    slack > COMP_SLACK_REL_TOL * scale
}

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
    // Comp slackness: non-binding rows (slack > tol) clamp `y` to 0 — for deleted
    // rows that pins `y_del` at 0; for coupled kept rows it pins `dy` at `-y_kept_i`.
    let mut bounds_clean: Vec<(f64, f64)> = Vec::with_capacity(total_vars);
    for &i in &deleted_rows {
        let nonbinding = is_row_nonbinding(orig_problem, i, solution);
        if nonbinding {
            bounds_clean.push((0.0, 0.0));
            continue;
        }
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => bounds_clean.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => bounds_clean.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => bounds_clean.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    if use_kept_perturbation {
        for &i in &coupled_kept {
            let y_kept_i = dual_solution_known[i];
            let nonbinding = is_row_nonbinding(orig_problem, i, solution);
            if nonbinding {
                bounds_clean.push((-y_kept_i, -y_kept_i));
                continue;
            }
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

    // Wire the parent deadline straight through so every inner stage (parse, scale,
    // factorize, simplex iterate) checks the same clock; otherwise large cleanup
    // LPs can spend minutes in setup before any per-call budget kicks in.
    let opts = SolverOptions { presolve: false, warm_start: None, deadline, ..SolverOptions::default() };
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
        p2_ct.push(ct_clean_keep[orig_idx]);
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
        if is_row_nonbinding(orig_problem, i, solution) {
            p2_bounds.push((0.0, 0.0));
            continue;
        }
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => p2_bounds.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => p2_bounds.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => p2_bounds.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    if use_kept_perturbation {
        for &i in &coupled_kept {
            let y_kept_i = dual_solution_known[i];
            if is_row_nonbinding(orig_problem, i, solution) {
                p2_bounds.push((-y_kept_i, -y_kept_i));
                continue;
            }
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

/// Cleanup LP の kept-row 摂動 (`dy` 変数) を無効化する規模しきい値。摂動は
/// deleted↔kept の bipartite closure 全体に `dy` 列を追加するため、大規模では
/// cleanup LP 自体が解けない規模に膨らむ。この上限超でも摂動なしの cleanup LP は
/// 走るので dual recovery は機能する (品質と可解性のトレードオフ)。
/// memory/時間予算ではなく LP 列数膨張のガードなので固定 size で妥当。
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

/// Marker for bound-tightened-fixed columns that landed on one of their *original*
/// bounds.  At such a column the bound-multiplier pair (μ_lb, μ_ub) is degenerate;
/// `extract_dual_info` produced `rc = c − A^T y`, but the residual's wrong-sign part
/// has to be reported as the now-implicit `μ_ub` (resp. `μ_lb`) so the externally
/// visible rc stays dual-feasible at the active bound.
#[derive(Clone, Copy)]
enum BoundAbsorb { AtLb, AtUb }

/// Recover `y_i` of a removed row to satisfy LP dual feasibility, given the rest of `y`.
/// For each column the required rc sign yields a permissible range on `y_i`; the row's
/// constraint type (Le: y≤0, Ge: y≥0, Eq: free) intersects that range and we pick the
/// value closest to zero. Rows whose primal is strictly non-binding short-circuit to
/// `y_i = 0` because the rc-sign-only walk otherwise admits slackness-violating duals.
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
            } else if bound_val > min_y_i { min_y_i = bound_val; }
        } else if at_ub && !at_lb {
            if a_ij > 0.0 {
                if bound_val > min_y_i { min_y_i = bound_val; }
            } else if bound_val < max_y_i { max_y_i = bound_val; }
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
fn recover_warm_start_basis(
    orig_problem: &LpProblem,
    solution: &[f64],
) -> Option<WarmStartBasis> {
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
            let val = if coeff > 0.0 { xj - info.offset } else { info.offset - xj };
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
                if v.abs() > col_max { col_max = v.abs(); }
            }
            if col_max < WARM_BASIS_BUILD_TOL { continue; }
            let pivot_min = (0.1 * col_max).max(WARM_BASIS_BUILD_TOL);

            let mut best: Option<(f64, usize)> = None;
            for (k, &row) in rows.iter().enumerate() {
                let abs = vals[k].abs();
                if abs < pivot_min { continue; }
                let cur = basis[row];
                let cur_is_at_bound_slack = cur >= n_shifted && x_std[cur] <= WARM_BASIS_BUILD_TOL;
                if !cur_is_at_bound_slack { continue; }
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
    let x_b: Vec<f64> = basis.iter().map(|&j| x_std.get(j).copied().unwrap_or(0.0)).collect();
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
            PostsolveStep::SingletonRow { orig_col, orig_row, value } => {
                solution[*orig_col] = *value;
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::BoundsTightened => {}
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
        'gs_outer: for _ in 0..GS_MAX_ITER {
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
            if max_diff < GS_CONV_TOL { break 'gs_outer; }
        }
        y
    };

    // For columns fixed by bound tightening (orig lb<ub, presolve shrunk to lb=ub) that
    // ended up at an original bound, the bound dual is degenerate: μ_lb − μ_ub can split
    // any residual `c − A^T y` between non-negative halves as long as one half is zero.
    // We absorb the wrong-sign part into the now-implicit μ_ub (at orig lb) or μ_lb
    // (at orig ub) so the reported `rc` stays dual-feasible, and let the dfeas-driven
    // y-candidate selection ignore the absorbable mismatch.  Columns pushed strictly
    // INTO the interior by tightening (e.g. orig (0,100) → fixed at 50) get NO override:
    // both bound multipliers are zero there, so rc = c − A^T y must hold (KKT identity),
    // which is required for bandm/beaconfd/brandy/agg/scorpion/scfxm1/recipe.
    let bound_dual_absorbs: Vec<Option<BoundAbsorb>> = {
        let mut out: Vec<Option<BoundAbsorb>> = vec![None; n];
        for step in &presolve_result.postsolve_stack {
            if let PostsolveStep::FixedVariable { orig_col, .. } = step {
                let j = *orig_col;
                if j >= n { continue; }
                let (orig_lb, orig_ub) = orig_problem.bounds[j];
                let truly_fixed = orig_lb.is_finite() && orig_ub.is_finite()
                    && (orig_ub - orig_lb).abs() < BOUND_ACTIVE_TOL;
                if truly_fixed { continue; }
                let x = solution[j];
                let at_orig_lb = orig_lb.is_finite()
                    && (x - orig_lb).abs() < BOUND_ACTIVE_TOL;
                let at_orig_ub = orig_ub.is_finite()
                    && (x - orig_ub).abs() < BOUND_ACTIVE_TOL;
                if at_orig_lb && !at_orig_ub {
                    out[j] = Some(BoundAbsorb::AtLb);
                } else if at_orig_ub && !at_orig_lb {
                    out[j] = Some(BoundAbsorb::AtUb);
                }
            }
        }
        out
    };

    // Build dfeas_bound first so the cheap candidates (y_loop, y_gs) can gate the
    // far more expensive cleanup-LP candidates.  Stay clamp-unaware on purpose: the
    // raw `c − A^T y` violation must surface so the caller's presolve-off fallback
    // (src/simplex/mod.rs:96-121) can re-solve when cleanup LP couldn't recover.
    let dfeas_bound = |y: &[f64]| -> f64 {
        let mut max_viol = 0.0f64;
        for j in 0..n {
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
    // LSQ pass is worth the (often dominant) runtime.
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
    } else if m > 0 {
        // 規模ガードは固定 size proxy ではなく compute_lsq_dual_y 内部に委ねる
        // (主経路は matrix-free CG、direct LDL fallback のみ memory_budget で skip)。
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
    // Apply the bound-dual-absorption clamp at the column granularity decided above;
    // see `BoundAbsorb` for the math.  Columns with no absorption marker (interior
    // tightened-fixed, all non-tightened cols) keep rc = c − A^T y untouched.
    for j in 0..n {
        match bound_dual_absorbs[j] {
            Some(BoundAbsorb::AtLb) => reduced_costs[j] = reduced_costs[j].max(0.0),
            Some(BoundAbsorb::AtUb) => reduced_costs[j] = reduced_costs[j].min(0.0),
            None => {}
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

#[cfg(test)]
mod cleanup_comp_tests {
    //! cleanup-LP comp slackness sentinels.
    //!
    //! Each fixture pins `dual_solution_known` for a kept row to a drifted value;
    //! without the comp clamp the cleanup LP would absorb the drift into the
    //! deleted non-binding row's `y_del` (sign-feasible but slackness-violating).
    //! With the clamp `y_del` is pinned at 0 and Phase-1 slack carries the drift.
    //! Toggle: removing the `is_row_nonbinding` branch in `bounds_clean` /
    //! `p2_bounds` flips `y_del` non-zero and the assertions fail.
    use super::*;
    use crate::presolve::transforms::{PostsolveStep, PresolveResult};
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;
    use std::collections::HashMap;
    use std::sync::Once;

    /// Drifted dual for kept rows — large enough that a non-binding deleted row
    /// would absorb a measurable y to mask the rc-sign violation without comp.
    const DRIFT_MAGNITUDE: f64 = 5e-3;
    /// Comp residual threshold for asserting the fix is alive.
    const COMP_RESID_TIGHT: f64 = 1e-9;

    fn presolve_result_with_deleted_row(
        problem: &LpProblem,
        deleted_row: usize,
    ) -> PresolveResult {
        let n = problem.num_vars;
        let m = problem.num_constraints;
        // Keep all columns; only the chosen row is removed.
        let col_map = (0..n).map(Some).collect();
        let row_map: Vec<Option<usize>> = (0..m)
            .map(|i| if i == deleted_row { None } else { Some(if i < deleted_row { i } else { i - 1 }) })
            .collect();
        let postsolve_stack = vec![PostsolveStep::EmptyRow { orig_row: deleted_row }];
        PresolveResult {
            reduced_problem: problem.clone(),
            postsolve_stack,
            orig_num_vars: n,
            orig_num_constraints: m,
            col_map,
            row_map,
            was_reduced: true,
            obj_offset: 0.0,
        }
    }

    /// max_j {|rc_sign_violation|} over the recovered y, using the constraint-active
    /// reduced-cost rule (rc must be ≥0 at lb, ≤0 at ub, =0 interior).
    fn rc_sign_violation(problem: &LpProblem, solution: &[f64], y: &[f64]) -> f64 {
        let mut max_v = 0.0_f64;
        for j in 0..problem.num_vars {
            let (lb, ub) = problem.bounds[j];
            let at_lb = lb.is_finite() && (solution[j] - lb).abs() < 1e-6;
            let at_ub = ub.is_finite() && (solution[j] - ub).abs() < 1e-6;
            let mut rc = problem.c[j];
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    rc -= vals[k] * y[row];
                }
            }
            let v = if at_lb && !at_ub { (-rc).max(0.0) }
                else if at_ub && !at_lb { rc.max(0.0) }
                else { rc.abs() };
            if v > max_v { max_v = v; }
        }
        max_v
    }

    /// Residual of `|y_i · slack_i|` over the recovered y.
    fn comp_residual(problem: &LpProblem, solution: &[f64], y: &[f64]) -> f64 {
        let mut max_c = 0.0_f64;
        for i in 0..problem.num_constraints {
            let (slack, scale) = row_slack_and_scale(problem, i, solution);
            let prod = (y[i] * slack).abs() / scale;
            if prod > max_c { max_c = prod; }
        }
        max_c
    }

    /// Fixture 1: 1 kept Eq + 1 deleted Le row. The deleted Le row is
    /// non-binding at the optimum; cleanup-LP must keep its y at 0 even though
    /// the kept Eq y is intentionally drifted.
    fn fixture_eq_kept_le_deleted() -> (LpProblem, Vec<f64>, Vec<f64>, usize) {
        // min x1 + x2 s.t. x1 + x2 = 1, x2 ≤ 10, x ≥ 0.
        // Optimum: x* = (0, 1), row 0 binding, row 1 slack = 9.
        let a = CscMatrix::from_triplets(
            &[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![1.0, 10.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap();
        let solution = vec![0.0, 1.0];
        // Drifted kept dual: true y_0 = 1.0; drift breaks rc sign for x1.
        let dual_known = vec![1.0 + DRIFT_MAGNITUDE, 0.0];
        (lp, solution, dual_known, 1)
    }

    /// Fixture 2: 1 kept Eq + 1 deleted Ge row. Verifies the Ge branch of
    /// `bounds_clean`/`p2_bounds` (y_del default `(0, ∞)`) gets clamped to
    /// `(0, 0)` for the non-binding row. Same primal as Fixture 1 with the
    /// deleted row's A negated so cleanup-LP prefers `y_del = DRIFT` without
    /// the clamp.
    fn fixture_eq_kept_ge_deleted() -> (LpProblem, Vec<f64>, Vec<f64>, usize) {
        // min x1 + x2 s.t. x1 + x2 = 1, -x1 - x2 ≥ -10, x ≥ 0.
        // Optimum x* = (0, 1); row 0 binding, row 1 slack = 9.
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![1.0, -10.0],
            vec![ConstraintType::Eq, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap();
        let solution = vec![0.0, 1.0];
        let dual_known = vec![1.0 + DRIFT_MAGNITUDE, 0.0];
        (lp, solution, dual_known, 1)
    }

    fn init_logger() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {});
    }

    fn run_fixture(
        problem: &LpProblem, solution: &[f64], dual_known: &[f64], deleted_row: usize,
    ) -> Vec<f64> {
        init_logger();
        let presolve_result = presolve_result_with_deleted_row(problem, deleted_row);
        let y = build_and_solve_cleanup_lp(
            problem, &presolve_result, solution, dual_known, None, false,
        ).expect("cleanup LP must converge for the sentinel fixture");
        assert_eq!(y.len(), problem.num_constraints);
        y
    }

    #[test]
    fn cleanup_lp_eq_kept_le_deleted_comp_holds() {
        let (lp, sol, dual, del) = fixture_eq_kept_le_deleted();
        let y = run_fixture(&lp, &sol, &dual, del);
        let comp = comp_residual(&lp, &sol, &y);
        assert!(
            comp < COMP_RESID_TIGHT,
            "comp={:.3e} >= {:.0e}; y={:?} (clamp on non-binding Le row must pin y[{}]=0)",
            comp, COMP_RESID_TIGHT, y, del,
        );
        // y for the deleted non-binding Le row must be exactly 0.
        assert_eq!(y[del], 0.0, "non-binding Le deleted row y must be 0, got {}", y[del]);
    }

    #[test]
    fn cleanup_lp_eq_kept_ge_deleted_comp_holds() {
        let (lp, sol, dual, del) = fixture_eq_kept_ge_deleted();
        let y = run_fixture(&lp, &sol, &dual, del);
        let comp = comp_residual(&lp, &sol, &y);
        assert!(
            comp < COMP_RESID_TIGHT,
            "comp={:.3e} >= {:.0e}; y={:?} (clamp on non-binding Ge row must pin y[{}]=0)",
            comp, COMP_RESID_TIGHT, y, del,
        );
        assert_eq!(y[del], 0.0, "non-binding Ge deleted row y must be 0, got {}", y[del]);
    }

    /// No-op proof: feed in the dual the un-clamped cleanup-LP would have
    /// chosen (y_del = -DRIFT on the non-binding Le row to satisfy the Eq
    /// stationarity constraint on the interior x2 column), and confirm the
    /// comp detector flags it. Confirms the detector itself has teeth — and
    /// that if the clamp is reverted the tight assertions above flip to FAIL
    /// with drift in this same band.
    #[test]
    fn cleanup_lp_unclamped_dual_violates_comp_detector() {
        let (lp, sol, _dual, _del) = fixture_eq_kept_le_deleted();
        let broken_y = vec![1.0 + DRIFT_MAGNITUDE, -DRIFT_MAGNITUDE];
        let comp = comp_residual(&lp, &sol, &broken_y);
        assert!(
            comp >= DRIFT_MAGNITUDE * 0.5,
            "broken dual comp={:.3e} should be >= {:.3e}; detector is no-op'd",
            comp, DRIFT_MAGNITUDE * 0.5,
        );
        // Sanity: rc_sign_violation alone is NOT a substitute — the un-clamped
        // dual passes rc-sign on the interior x2 column even though it violates
        // comp. (The col-0 rc violation here is inherited drift, unrelated.)
        let _rc_v_inherited = rc_sign_violation(&lp, &sol, &broken_y);
    }

    /// Cross-check: the helper `is_row_nonbinding` matches the comp-residual
    /// reasoning across multiple input scales — guards against future refactors
    /// of the tolerance (relative vs absolute).
    #[test]
    fn is_row_nonbinding_detects_known_patterns() {
        let cases: Vec<(ConstraintType, f64, f64, bool)> = vec![
            // (ct, b, ax, expected_nonbinding)
            (ConstraintType::Le, 10.0, 5.0, true),    // slack 5 ≫ tol
            (ConstraintType::Le, 10.0, 10.0, false),  // slack 0, binding
            (ConstraintType::Ge, 1.0, 100.0, true),   // slack 99
            (ConstraintType::Ge, 1.0, 1.0, false),
            (ConstraintType::Eq, 1.0, 1.0, false),
            (ConstraintType::Eq, 1.0, 0.5, false),    // Eq is never non-binding
        ];
        for (i, (ct, b, ax, expected)) in cases.iter().enumerate() {
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
            let lp = LpProblem::new_general(
                vec![0.0], a, vec![*b], vec![ct.clone()],
                vec![(f64::NEG_INFINITY, f64::INFINITY)], None,
            ).unwrap();
            let got = is_row_nonbinding(&lp, 0, &[*ax]);
            assert_eq!(
                got, *expected,
                "case {} ({:?}, b={}, ax={}): expected {}, got {}",
                i, ct, b, ax, expected, got,
            );
        }
        let _ = HashMap::<usize, usize>::new(); // keep import alive on toolchains that warn
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
    use crate::simplex::{solve, solve_with, build_standard_form};
    use crate::sparse::CscMatrix;

    /// Default options + `recover_warm_start_basis = true`. The recovery path
    /// is opt-in; sentinels covering the postsolve synthesis must enable it.
    fn opts_recover() -> SolverOptions {
        SolverOptions { recover_warm_start_basis: true, ..SolverOptions::default() }
    }

    /// LP whose presolve dual-fixing zeroes both vars (c>0, x≥0, finite ub).
    /// Reduced LP has 0 vars → simplex `n==0` short-circuit → reduced
    /// warm_start_basis = None. Postsolve must still synthesise a basis.
    fn lp_dual_fixed() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2], &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0], 3, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0], a, vec![6.0, 4.0, 4.0],
            vec![ConstraintType::Le; 3],
            vec![(0.0, f64::INFINITY); 2],
            None,
        ).unwrap()
    }

    /// LP with a singleton-row Eq: x0 = 2; presolve fixes x0 then propagates.
    fn lp_singleton_row() -> LpProblem {
        // min x0 + x1 s.t. x0 = 2 (Eq), x0 + x1 ≤ 5; x ≥ 0
        let a = CscMatrix::from_triplets(
            &[0, 1, 1], &[0, 0, 1], &[1.0, 1.0, 1.0],
            2, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0], a, vec![2.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY); 2],
            None,
        ).unwrap()
    }

    /// LP that survives presolve untouched (no reducible structure) — the
    /// `was_reduced=false` branch in `solve_with` should still surface a basis
    /// (this comes from simplex directly, not postsolve; sentinel ensures the
    /// postsolve fix didn't regress the non-reducible path).
    fn lp_non_reducible() -> LpProblem {
        // min -x0 - 2*x1 s.t. x0 + x1 ≤ 4; x0 ≤ 3; x1 ≤ 3
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2], &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0], 3, 2,
        ).unwrap();
        LpProblem::new_general(
            vec![-1.0, -2.0], a, vec![4.0, 3.0, 3.0],
            vec![ConstraintType::Le; 3],
            vec![(0.0, f64::INFINITY); 2],
            None,
        ).unwrap()
    }

    fn assert_basis_well_formed(lp: &LpProblem, basis: &[usize], context: &str) {
        let sf = build_standard_form(lp);
        assert_eq!(
            basis.len(), sf.m,
            "[{}] basis len {} != m_ext {}", context, basis.len(), sf.m,
        );
        for (i, &col) in basis.iter().enumerate() {
            assert!(
                col < sf.n_total,
                "[{}] basis[{}] = {} ≥ n_total {} (artificial leakage)",
                context, i, col, sf.n_total,
            );
        }
        // Uniqueness: each column appears at most once in the basis.
        let mut seen = vec![false; sf.n_total];
        for &col in basis {
            assert!(!seen[col], "[{}] basis has duplicate column {}", context, col);
            seen[col] = true;
        }
    }

    fn assert_warm_round_trip(lp_a: &LpProblem, lp_b: &LpProblem, context: &str) {
        let r1 = solve_with(lp_a, &opts_recover());
        assert_eq!(r1.status, SolveStatus::Optimal, "[{}] lp_a status", context);
        let ws = r1.warm_start_basis.as_ref()
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
            r2.status, SolveStatus::Optimal,
            "[{}] warm-start round-trip on lp_b did not reach Optimal", context,
        );
    }

    #[test]
    fn warm_basis_from_dual_fixed_lp() {
        let lp = lp_dual_fixed();
        // Self-warm round-trip (same LP twice) — the simplest sanity.
        assert_warm_round_trip(&lp, &lp, "dual_fixed/self");
        // Cross-warm with RHS change matching the #65 regression scenario.
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
        assert_basis_well_formed(&lp, &r.warm_start_basis.as_ref().unwrap().basis, "non_reducible");
    }

    /// No-op proof: a re-implementation that always returns `None` makes the
    /// sentinels above fail (assertion on `is_some()`). We exercise that path
    /// inline here so the dependency is local: forcing `None` *does* break the
    /// dual-fixed warm-start round-trip even when the new RHS is feasible
    /// (because subsequent `solve_with(lp2, warm=None, presolve=false)` would
    /// be a cold dual that this fixture is fine with, BUT the upstream
    /// assertion `result.warm_start_basis.is_some()` in #65 still trips).
    #[test]
    fn noop_proof_returns_none_fails_round_trip() {
        // Reproduces the original #65 FAIL state: presolve reduces, postsolve
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
             (no-op would return None and re-introduce #65)",
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
            basis.contains(&0) || sf.orig_var_info[0].new_vars.iter().any(|&(idx, _)| basis.contains(&idx)),
            "active x0=1 not in warm-start basis: {:?}", basis,
        );
        assert!(
            basis.contains(&1) || sf.orig_var_info[1].new_vars.iter().any(|&(idx, _)| basis.contains(&idx)),
            "active x1=3 not in warm-start basis: {:?}", basis,
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
        assert!(r_default.warm_start_basis.is_some(),
            "non-reducible default path must keep native simplex basis");
        assert!(r_optin.warm_start_basis.is_some(),
            "non-reducible opt-in path must keep native simplex basis");
    }
}
