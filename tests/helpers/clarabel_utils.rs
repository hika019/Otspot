//! Shared Clarabel cross-check helpers.
//!
//! Included via `#[path = "helpers/clarabel_utils.rs"]` in integration tests
//! that need to run Clarabel as a reference solver.  Not a standalone test file.

#![allow(
    dead_code,
    clippy::field_reassign_with_default,
    clippy::type_complexity
)]

use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use otspot::problem::ConstraintType;
use otspot::QpProblem;

/// Convert a `QpProblem` into Clarabel standard form.
///
/// Clarabel solves: min 0.5 xᵀ P x + qᵀ x  s.t. A x + s = b, s ∈ K
///   Eq  → ZeroCone
///   Le  → Nonneg (s = b − Ax ≥ 0)
///   Ge  → Nonneg with A,b negated
///   bounds → extra rows (lb: −eⱼ x + s = −lb; ub: +eⱼ x + s = ub)
pub fn build_clarabel(
    prob: &QpProblem,
) -> (
    ClCsc<f64>,
    Vec<f64>,
    ClCsc<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    let n_lb = prob
        .bounds
        .iter()
        .filter(|&&(lb, _): &&(f64, f64)| lb.is_finite())
        .count();
    let n_ub = prob
        .bounds
        .iter()
        .filter(|&&(_, ub): &&(f64, f64)| ub.is_finite())
        .count();

    // Place Eq rows first so ZeroCone and NonnegativeCone are contiguous.
    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct {
        ConstraintType::Eq => 0,
        _ => 1,
    });
    let n_eq = row_ord
        .iter()
        .filter(|&&(_, ct)| ct == ConstraintType::Eq)
        .count();
    let n_le_ge = m - n_eq;

    let mut row_pos = vec![0_usize; m];
    for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() {
        row_pos[orig_row] = new_row;
    }

    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let total_rows = m + n_lb + n_ub;
    let mut b_clar = vec![0.0_f64; total_rows];

    for j in 0..n {
        for ptr in prob.a.col_ptr()[j]..prob.a.col_ptr()[j + 1] {
            let orig_row = prob.a.row_ind()[ptr];
            let val = prob.a.values()[ptr];
            let new_row = row_pos[orig_row];
            let ct = prob.constraint_types[orig_row];
            match ct {
                ConstraintType::Ge => triplets.push((new_row, j, -val)),
                _ => triplets.push((new_row, j, val)),
            }
        }
    }
    for (orig_row, ct) in prob.constraint_types.iter().enumerate() {
        let new_row = row_pos[orig_row];
        match ct {
            ConstraintType::Ge => b_clar[new_row] = -prob.b[orig_row],
            _ => b_clar[new_row] = prob.b[orig_row],
        }
    }
    let mut bound_row = m;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() {
            triplets.push((bound_row, j, -1.0));
            b_clar[bound_row] = -lb;
            bound_row += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() {
            triplets.push((bound_row, j, 1.0));
            b_clar[bound_row] = ub;
            bound_row += 1;
        }
    }

    triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &triplets {
        col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut row_ind = vec![0_usize; triplets.len()];
    let mut values = vec![0.0_f64; triplets.len()];
    let mut cursor = col_ptr.clone();
    for &(r, c, v) in &triplets {
        let pos = cursor[c];
        row_ind[pos] = r;
        values[pos] = v;
        cursor[c] += 1;
    }
    let a_clar = ClCsc::new(total_rows, n, col_ptr, row_ind, values);

    // P: upper triangular only (Clarabel convention).
    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        for ptr in prob.q.col_ptr()[j]..prob.q.col_ptr()[j + 1] {
            let i = prob.q.row_ind()[ptr];
            if i <= j {
                p_triplets.push((i, j, prob.q.values()[ptr]));
            }
        }
    }
    p_triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut p_col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &p_triplets {
        p_col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        p_col_ptr[j + 1] += p_col_ptr[j];
    }
    let mut p_row_ind = vec![0_usize; p_triplets.len()];
    let mut p_values = vec![0.0_f64; p_triplets.len()];
    let mut p_cursor = p_col_ptr.clone();
    for &(r, c, v) in &p_triplets {
        let pos = p_cursor[c];
        p_row_ind[pos] = r;
        p_values[pos] = v;
        p_cursor[c] += 1;
    }
    let p_clar = ClCsc::new(n, n, p_col_ptr, p_row_ind, p_values);

    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(n_eq));
    }
    if n_le_ge + n_lb + n_ub > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(n_le_ge + n_lb + n_ub));
    }

    (p_clar, prob.c.clone(), a_clar, b_clar, cones)
}

/// Solve with Clarabel defaults; returns `(cost_primal, x)` on success.
pub fn solve_clarabel(prob: &QpProblem) -> Option<(f64, Vec<f64>)> {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.tol_gap_abs = 1e-9;
    settings.tol_gap_rel = 1e-9;
    settings.tol_feas = 1e-9;
    settings.max_iter = 5000;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    if matches!(
        solver.info.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        Some((solver.info.cost_primal, solver.solution.x.clone()))
    } else {
        None
    }
}

/// 0.5 xᵀ Q x + cᵀ x (without obj_offset).
pub fn compute_internal_obj(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx
        .iter()
        .zip(x.iter())
        .map(|(&qi, &xi)| qi * xi)
        .sum::<f64>()
        + prob
            .c
            .iter()
            .zip(x.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>()
}
