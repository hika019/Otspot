//! Post-loop construction of the reduced QP: index remapping, matrix rebuild,
//! large-coefficient rescaling, and optional Ruiz scaling.

use super::helpers::{apply_large_coeff_rescaling, count_block_components, is_diagonal_q};
use super::state::{QpPostsolveStep, QpPresolveResult, QpPresolveStatus, Workspace};
use crate::linalg::ruiz::RuizScaler;
use crate::options::SolverOptions;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

pub(super) fn build_result(
    prob: &QpProblem,
    opts: &SolverOptions,
    mut ws: Workspace,
) -> QpPresolveResult {
    let n = prob.num_vars;
    let m = prob.num_constraints;

    let mut col_map = vec![None; n];
    let mut new_col_idx = 0usize;
    for j in 0..n {
        if !ws.removed_cols[j] {
            col_map[j] = Some(new_col_idx);
            new_col_idx += 1;
        }
    }
    let n_new = new_col_idx;

    let mut col_map_inv = vec![0usize; n_new];
    for (j, &maybe_jj) in col_map.iter().enumerate().take(n) {
        if let Some(jj) = maybe_jj {
            col_map_inv[jj] = j;
        }
    }

    let mut row_map = vec![None; m];
    let mut new_row_idx = 0usize;
    for i in 0..m {
        if !ws.removed_rows[i] {
            row_map[i] = Some(new_row_idx);
            new_row_idx += 1;
        }
    }
    let m_new = new_row_idx;

    let was_reduced = n_new < n || m_new < m;

    // Fold Kahan compensation into the final c / b / obj_offset.
    for j in 0..n {
        ws.c[j] += ws.c_comp[j];
    }
    for i in 0..m {
        ws.b[i] += ws.b_comp[i];
    }
    ws.obj_offset += ws.obj_offset_comp;

    let mut c_new = vec![0.0f64; n_new];
    let mut bounds_new = vec![(f64::NEG_INFINITY, f64::INFINITY); n_new];
    for j in 0..n {
        if let Some(jj) = col_map[j] {
            c_new[jj] = ws.c[j];
            bounds_new[jj] = ws.bounds[j];
        }
    }

    let mut b_new = vec![0.0f64; m_new];
    for i in 0..m {
        if let Some(ii) = row_map[i] {
            b_new[ii] = ws.b[i];
        }
    }

    let a_new = build_reduced_a(prob, &ws, &col_map, &row_map, n, m_new, n_new);
    let q_new = build_reduced_q(prob, &ws, &col_map, n, n_new);

    let q_linear_adjust = ws.c.clone();

    let mut constraint_types_new = vec![crate::problem::ConstraintType::Le; m_new];
    for (i, &maybe_ii) in row_map.iter().enumerate().take(m) {
        if let Some(ii) = maybe_ii {
            constraint_types_new[ii] = prob.constraint_types[i];
        }
    }

    let mut reduced =
        match QpProblem::new(q_new, c_new, a_new, b_new, bounds_new, constraint_types_new) {
            Ok(p) => p,
            Err(_) => return QpPresolveResult::no_reduction(prob),
        };

    let detected_diagonal_q = is_diagonal_q(&reduced.q, n_new);
    let detected_block_components = count_block_components(&reduced.q, &reduced.a, n_new);

    // Skip large-coeff rescaling when Ruiz is enabled — chaining the two makes the
    // composite amplification uncontrollable.
    apply_large_coeff_if_needed(opts, &mut reduced, &mut ws.postsolve_stack);

    let ruiz_scaler_opt = maybe_apply_ruiz(opts, &mut reduced, n_new, m_new);

    QpPresolveResult {
        reduced,
        col_map,
        col_map_inv,
        row_map,
        obj_offset: ws.obj_offset,
        q_linear_adjust,
        postsolve_stack: ws.postsolve_stack,
        was_reduced,
        orig_num_vars: n,
        orig_num_constraints: m,
        presolve_status: QpPresolveStatus::Feasible,
        is_diagonal_q: detected_diagonal_q,
        block_components: detected_block_components,
        ruiz_scaler: ruiz_scaler_opt,
    }
}

fn build_reduced_a(
    prob: &QpProblem,
    ws: &Workspace,
    col_map: &[Option<usize>],
    row_map: &[Option<usize>],
    n: usize,
    m_new: usize,
    n_new: usize,
) -> CscMatrix {
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if ws.removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        let start = prob.a.col_ptr[j];
        let end = prob.a.col_ptr[j + 1];
        for k in start..end {
            let row = prob.a.row_ind[k];
            if ws.removed_rows[row] {
                continue;
            }
            let ii = row_map[row].unwrap();
            trip_rows.push(ii);
            trip_cols.push(jj);
            trip_vals.push(prob.a.values[k]);
        }
    }
    if trip_rows.is_empty() {
        CscMatrix::new(m_new, n_new)
    } else {
        CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n_new)
            .unwrap_or_else(|_| CscMatrix::new(m_new, n_new))
    }
}

fn build_reduced_q(
    prob: &QpProblem,
    ws: &Workspace,
    col_map: &[Option<usize>],
    n: usize,
    n_new: usize,
) -> CscMatrix {
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if ws.removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        let start = prob.q.col_ptr[j];
        let end = prob.q.col_ptr[j + 1];
        for k in start..end {
            let row = prob.q.row_ind[k];
            if ws.removed_cols[row] {
                continue;
            }
            let ii = col_map[row].unwrap();
            trip_rows.push(ii);
            trip_cols.push(jj);
            trip_vals.push(prob.q.values[k]);
        }
    }
    if trip_rows.is_empty() {
        CscMatrix::new(n_new, n_new)
    } else {
        CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, n_new, n_new)
            .unwrap_or_else(|_| CscMatrix::new(n_new, n_new))
    }
}

fn apply_large_coeff_if_needed(
    opts: &SolverOptions,
    reduced: &mut QpProblem,
    postsolve_stack: &mut super::state::QpPostsolveStack,
) {
    if opts.use_ruiz_scaling {
        return;
    }

    let n_new = reduced.num_vars;
    let mut a_mut = reduced.a.clone();
    let mut b_mut = reduced.b.clone();
    let scales = apply_large_coeff_rescaling(&mut a_mut, &mut b_mut, n_new);
    let any_scaled = scales.iter().any(|&s| (s - 1.0).abs() > 1e-12);
    if any_scaled {
        if let Ok(p) = QpProblem::new(
            reduced.q.clone(),
            reduced.c.clone(),
            a_mut,
            b_mut,
            reduced.bounds.clone(),
            reduced.constraint_types.clone(),
        ) {
            *reduced = p;
        }
        postsolve_stack.push(QpPostsolveStep::LargeCoeffRowScale { row_scales: scales });
    }
}

fn maybe_apply_ruiz(
    opts: &SolverOptions,
    reduced: &mut QpProblem,
    n_new: usize,
    m_new: usize,
) -> Option<RuizScaler> {
    if !(opts.use_ruiz_scaling && n_new > 0) {
        return None;
    }
    let mut scaler = RuizScaler::new(n_new, m_new);
    scaler.compute_with_rhs(&reduced.q, &reduced.a, &reduced.c, &[]);
    let (q_s, a_s, c_s, b_s, bounds_s) = scaler.scale_problem(
        &reduced.q,
        &reduced.a,
        &reduced.c,
        &reduced.b,
        &reduced.bounds,
    );
    match QpProblem::new(
        q_s,
        c_s,
        a_s,
        b_s,
        bounds_s,
        reduced.constraint_types.clone(),
    ) {
        Ok(p) => {
            *reduced = p;
            Some(scaler)
        }
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;

    #[test]
    fn large_coeff_rescaling_is_skipped_before_ruiz_without_touching_problem() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0e12_f64], 1, 1).unwrap();
        let mut reduced = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![2.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let original_a = reduced.a.clone();
        let original_b = reduced.b.clone();
        let opts = SolverOptions {
            use_ruiz_scaling: true,
            ..SolverOptions::default()
        };
        let mut stack = super::super::state::QpPostsolveStack::new();

        apply_large_coeff_if_needed(&opts, &mut reduced, &mut stack);

        assert_eq!(reduced.a.values, original_a.values);
        assert_eq!(reduced.a.col_ptr, original_a.col_ptr);
        assert_eq!(reduced.a.row_ind, original_a.row_ind);
        assert_eq!(reduced.b, original_b);
        assert!(stack.steps.is_empty());
    }

    #[test]
    fn large_coeff_rescaling_still_runs_without_ruiz() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0e12_f64], 1, 1).unwrap();
        let mut reduced = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![2.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let opts = SolverOptions {
            use_ruiz_scaling: false,
            ..SolverOptions::default()
        };
        let mut stack = super::super::state::QpPostsolveStack::new();

        apply_large_coeff_if_needed(&opts, &mut reduced, &mut stack);

        assert!(reduced.a.values[0] < 1.0e12);
        assert!(reduced.b[0] < 2.0);
        assert_eq!(stack.steps.len(), 1);
    }
}
