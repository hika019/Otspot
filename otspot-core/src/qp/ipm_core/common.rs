//! IPM/IP-PMM 共通関数。

use super::kkt::{norm_inf, spmv};
use crate::linalg::ldl;
use crate::linalg::timeout::TimeoutCtx;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

/// ステップ方向 (Δx, Δy) から infeasibility / unboundedness を検出する。
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_infeasible_or_unbounded(
    dx: &[f64],
    dy: &[f64],
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_orig: usize,
    m_ext: usize,
    iter: usize,
    delta_p: f64,
) -> Option<SolveStatus> {
    const EPS_INF: f64 = 1e-8;
    const MIN_ITER: usize = 5;
    const MIN_DIR_NORM: f64 = 1e-3;

    if iter < MIN_ITER {
        return None;
    }

    let n = dx.len();

    if m_orig > 0 {
        let dy_orig = &dy[..m_orig];
        let norm_dy_inf = norm_inf(dy_orig);
        if norm_dy_inf > MIN_DIR_NORM {
            let norm_dy = norm_dy_inf;
            let mut at_dy = vec![0.0f64; n];
            for (j, at_dy_j) in at_dy.iter_mut().enumerate() {
                for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                    let row = a_ext.row_ind[ptr];
                    if row < m_orig {
                        *at_dy_j += a_ext.values[ptr] * dy_orig[row];
                    }
                }
            }
            let cond_a = norm_inf(&at_dy) / norm_dy < EPS_INF;
            let b_dy: f64 = problem
                .b
                .iter()
                .zip(dy_orig.iter())
                .map(|(&bi, &dyi)| bi * dyi)
                .sum();
            let cond_b = b_dy / norm_dy < -EPS_INF;
            if cond_a && cond_b {
                return Some(SolveStatus::Infeasible);
            }
        }
    }

    if m_orig == 0 && m_ext > 0 {
        return None;
    }
    let norm_dx_inf = norm_inf(dx);
    if norm_dx_inf <= MIN_DIR_NORM {
        return None;
    }
    let norm_dx = norm_dx_inf;

    let is_lp = problem.q.values.iter().all(|&v| v == 0.0);
    let cond_obj = if is_lp {
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        c_dx / norm_dx < -EPS_INF
    } else {
        let mut qdx = vec![0.0f64; n];
        spmv(&problem.q, dx, &mut qdx);
        for i in 0..n {
            qdx[i] += delta_p * dx[i];
        }
        let norm_qdx: f64 = qdx.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        (norm_qdx / norm_dx < EPS_INF) && (c_dx / norm_dx < -EPS_INF)
    };
    if !cond_obj {
        return None;
    }

    if m_orig > 0 {
        let mut a_dx = vec![0.0f64; m_orig];
        for (j, &dxj) in dx.iter().enumerate() {
            for ptr in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
                let row = a_ext.row_ind[ptr];
                if row < m_orig {
                    a_dx[row] += a_ext.values[ptr] * dxj;
                }
            }
        }
        if norm_inf(&a_dx) / norm_dx >= EPS_INF {
            return None;
        }
    }

    Some(SolveStatus::Unbounded)
}

/// α = min(1, τ · min_i { -v_i / Δv_i  for Δv_i < 0, skip_mask[i] == false } )
pub(crate) fn fraction_to_boundary_masked(
    v: &[f64],
    dv: &[f64],
    tau: f64,
    skip_mask: &[bool],
) -> f64 {
    let mut alpha = 1.0_f64;
    for (i, (&vi, &dvi)) in v.iter().zip(dv.iter()).enumerate() {
        if skip_mask[i] {
            continue;
        }
        if dvi < 0.0 {
            let step = tau * vi / (-dvi);
            if step < alpha {
                alpha = step;
            }
        }
    }
    alpha
}

/// 制約なし QP: Qx = -c（PD でなければ δ_p I で正則化）。
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_unconstrained(problem: &QpProblem, timeout_ctx: &TimeoutCtx) -> SolverResult {
    let n = problem.num_vars;

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if n == 0 {
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            iterations: 0,
            ..Default::default()
        };
    }

    let delta_p = 1e-7;
    let mut triplet_rows: Vec<usize> = Vec::new();
    let mut triplet_cols: Vec<usize> = Vec::new();
    let mut triplet_vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];

    for col in 0..n {
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            let row = problem.q.row_ind[k];
            if row <= col {
                triplet_rows.push(row);
                triplet_cols.push(col);
                let v = problem.q.values[k] + if row == col { delta_p } else { 0.0 };
                triplet_vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            triplet_rows.push(i);
            triplet_cols.push(i);
            triplet_vals.push(delta_p);
        }
    }

    let q_reg =
        CscMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n, n).unwrap();

    match ldl::factorize(&q_reg) {
        Ok(fac) => {
            let rhs: Vec<f64> = problem.c.iter().map(|&ci| -ci).collect();
            let mut x = vec![0.0f64; n];
            fac.solve(&rhs, &mut x);

            let mut qx = vec![0.0f64; n];
            spmv(&problem.q, &x, &mut qx);
            let objective = 0.5
                * qx.iter()
                    .zip(x.iter())
                    .map(|(&qi, &xi)| qi * xi)
                    .sum::<f64>()
                + problem
                    .c
                    .iter()
                    .zip(x.iter())
                    .map(|(&ci, &xi)| ci * xi)
                    .sum::<f64>();

            SolverResult {
                status: SolveStatus::Optimal,
                objective,
                solution: x,
                dual_solution: vec![],
                bound_duals: vec![],
                iterations: 1,
                ..Default::default()
            }
        }
        Err(_) => numerical_error_result(n),
    }
}

pub(crate) fn timeout_result(n: usize) -> SolverResult {
    SolverResult {
        solution: vec![0.0; n],
        ..SolverResult::timeout()
    }
}

pub(crate) fn numerical_error_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        iterations: 0,
        ..Default::default()
    }
}
