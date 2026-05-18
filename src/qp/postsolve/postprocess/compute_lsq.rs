//! 元問題空間で A^T y = -(Qx + c + bound_contrib) の最小二乗 y を計算 + DD-IR。

use crate::qp::linalg::{build_aat_upper_csc, compute_bound_contrib, LSQ_DUAL_SIZE_LIMIT};
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;
use crate::sparse::CscMatrix;

pub(crate) fn compute_lsq_dual_y(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) -> Option<Vec<f64>> {
    use twofloat::TwoFloat;
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return None;
    }
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return None;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let x = &result.solution;

    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target_dd: Vec<TwoFloat> = (0..n)
        .map(|j| -(qx_dd[j] + TwoFloat::from(problem.c[j]) + TwoFloat::from(bound_contrib[j])))
        .collect();

    let mut proj_lower = vec![f64::NEG_INFINITY; m];
    let mut proj_upper = vec![f64::INFINITY; m];
    for (i, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => proj_lower[i] = 0.0,
            crate::problem::ConstraintType::Ge => proj_upper[i] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    for j in 0..n {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        if ce - cs != 1 {
            continue;
        }
        let row = problem.a.row_ind[cs];
        let aij = problem.a.values[cs];
        if !aij.is_finite() || aij == 0.0 {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        if lb_finite && ub_finite && (lb - ub).abs() < FX_TOL {
            continue;
        }
        let qxj = f64::from(qx_dd[j]);
        let rhs = -(qxj + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }
        match (lb_finite, ub_finite) {
            (true, false) => {
                if aij > 0.0 {
                    proj_lower[row] = proj_lower[row].max(rhs);
                } else {
                    proj_upper[row] = proj_upper[row].min(rhs);
                }
            }
            (false, true) => {
                if aij > 0.0 {
                    proj_upper[row] = proj_upper[row].min(rhs);
                } else {
                    proj_lower[row] = proj_lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }
    let mut fixed_y: Vec<Option<f64>> = vec![None; m];
    let mut n_fixed = 0usize;
    for i in 0..m {
        let lo = proj_lower[i];
        let hi = proj_upper[i];
        if lo.is_finite() && hi.is_finite() {
            let scale = 1.0 + lo.abs().max(hi.abs());
            if (lo - hi).abs() < 1e-10 * scale {
                fixed_y[i] = Some((lo + hi) * 0.5);
                n_fixed += 1;
            }
        }
    }

    let solve_lsq_ir = |a_sub: &CscMatrix, m_sub: usize, v_dd: &[TwoFloat]| -> Option<Vec<f64>> {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let aat_sub = build_aat_upper_csc(a_sub, n, m_sub)?;
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let factor = crate::linalg::ldl::factorize(&aat_sub).ok()?;
        let build_rhs_sub = |v_dd: &[TwoFloat]| -> Vec<f64> {
            let mut acc: Vec<TwoFloat> = vec![zero_dd; m_sub];
            for col in 0..n {
                let cs = a_sub.col_ptr[col];
                let ce = a_sub.col_ptr[col + 1];
                for k in cs..ce {
                    let row = a_sub.row_ind[k];
                    let v_f64 = f64::from(v_dd[col]);
                    let lo = v_dd[col] - TwoFloat::from(v_f64);
                    acc[row] = acc[row]
                        + TwoFloat::new_mul(a_sub.values[k], v_f64)
                        + TwoFloat::new_mul(a_sub.values[k], f64::from(lo));
                }
            }
            acc.iter().map(|&v| f64::from(v)).collect()
        };
        let rhs0 = build_rhs_sub(v_dd);
        let mut y_sub = vec![0.0_f64; m_sub];
        factor.solve(&rhs0, &mut y_sub);
        if y_sub.iter().any(|v| !v.is_finite()) {
            return None;
        }
        const IR_STAGNATE_RATIO: f64 = 0.5;
        const IR_PROGRESS_EPS: f64 = 1e-18;
        let mut prev_r_inf = f64::INFINITY;
        loop {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            let mut atysub_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = a_sub.col_ptr[col];
                let ce = a_sub.col_ptr[col + 1];
                for k in cs..ce {
                    let row = a_sub.row_ind[k];
                    atysub_dd[col] =
                        atysub_dd[col] + TwoFloat::new_mul(a_sub.values[k], y_sub[row]);
                }
            }
            let r_dd: Vec<TwoFloat> = (0..n).map(|j| v_dd[j] - atysub_dd[j]).collect();
            let r_inf = r_dd.iter().fold(0.0_f64, |a, &v| a.max(f64::from(v).abs()));
            if !r_inf.is_finite() {
                break;
            }
            if prev_r_inf.is_finite() && r_inf + IR_PROGRESS_EPS >= prev_r_inf {
                break;
            }
            if prev_r_inf.is_finite() && r_inf > prev_r_inf * IR_STAGNATE_RATIO {
                break;
            }
            prev_r_inf = r_inf;
            let rhs_dy = build_rhs_sub(&r_dd);
            let mut dy = vec![0.0_f64; m_sub];
            factor.solve(&rhs_dy, &mut dy);
            if dy.iter().any(|v| !v.is_finite()) {
                break;
            }
            for i in 0..m_sub {
                y_sub[i] += dy[i];
            }
        }
        Some(y_sub)
    };

    if n_fixed == 0 {
        return solve_lsq_ir(&problem.a, m, &target_dd);
    }

    let mut free_row_local = vec![usize::MAX; m];
    let mut free_rows: Vec<usize> = Vec::with_capacity(m - n_fixed);
    for (i, fy) in fixed_y.iter().enumerate() {
        if fy.is_none() {
            free_row_local[i] = free_rows.len();
            free_rows.push(i);
        }
    }
    let m_free = free_rows.len();
    if m_free == 0 {
        return Some(fixed_y.iter().map(|fy| fy.unwrap_or(0.0)).collect());
    }

    let mut a_free_col_ptr = vec![0usize; n + 1];
    let mut a_free_row_ind: Vec<usize> = Vec::new();
    let mut a_free_values: Vec<f64> = Vec::new();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            let local_row = free_row_local[orig_row];
            if local_row != usize::MAX {
                a_free_row_ind.push(local_row);
                a_free_values.push(problem.a.values[k]);
            }
        }
        a_free_col_ptr[col + 1] = a_free_row_ind.len();
    }
    let a_free = CscMatrix {
        col_ptr: a_free_col_ptr,
        row_ind: a_free_row_ind,
        values: a_free_values,
        nrows: m_free,
        ncols: n,
    };

    let mut target_adj_dd = target_dd.clone();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            if let Some(yfi) = fixed_y[orig_row] {
                if yfi != 0.0 {
                    target_adj_dd[col] =
                        target_adj_dd[col] - TwoFloat::new_mul(problem.a.values[k], yfi);
                }
            }
        }
    }

    let y_free = match solve_lsq_ir(&a_free, m_free, &target_adj_dd) {
        Some(v) => v,
        None => return solve_lsq_ir(&problem.a, m, &target_dd),
    };

    let mut y_full = vec![0.0_f64; m];
    for (local_idx, &orig_row) in free_rows.iter().enumerate() {
        y_full[orig_row] = y_free[local_idx];
    }
    for (i, fy) in fixed_y.iter().enumerate() {
        if let Some(v) = fy {
            y_full[i] = *v;
        }
    }
    Some(y_full)
}

