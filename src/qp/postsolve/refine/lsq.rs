//! dual y を LSQ で精密化。
//!
//! - `refine_dual_lsq`: A^T y = -(Qx + c + bound_contrib) を LSQ で 1 回解く。
//!   DD 残差で改善した場合のみ採用 (退行防止)。
//! - `refine_dual_lsq_irls`: IRLS で componentwise rel を最小化 (L∞ 漸近)。

use crate::qp::linalg::{build_aat_upper_csc, compute_bound_contrib, LSQ_DUAL_SIZE_LIMIT};
use crate::qp::postsolve::postprocess::compute_lsq_dual_y;
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;

pub(crate) fn refine_dual_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let Some(y_new) = compute_lsq_dual_y(problem, result, deadline) else {
        return;
    };
    let n = problem.num_vars;
    // ill-conditioned (cond~1e12) では f64 mat_vec の cancellation noise が真残差を
    // 上回り IPM の正しい y が LSQ y に置換される。DD で比較する。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let aty_dd = |y: &[f64]| -> Vec<TwoFloat> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = problem.a.col_ptr[col];
            let ce = problem.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc
    };
    let aty_old_dd = aty_dd(&result.dual_solution);
    let aty_new_dd = aty_dd(&y_new);
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    // componentwise rel = |r_j| / (1 + |Qx_j| + |c_j| + |Aty_j| + |z_j|) で比較。
    // abs max では ill-scaled 問題で外れ残差が巨大スケールに埋もれる。
    let mut max_rel_old = 0.0_f64;
    let mut max_rel_new = 0.0_f64;
    for j in 0..n {
        let (lbj, ubj) = problem.bounds[j];
        if lbj.is_finite() && ubj.is_finite() && (lbj - ubj).abs() < FX_TOL {
            continue;
        }
        if problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0 {
            continue;
        }
        let r_old_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_old_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let r_new_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_new_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let qx_j = f64::from(qx_dd[j]).abs();
        let aty_old_j = f64::from(aty_old_dd[j]).abs();
        let aty_new_j = f64::from(aty_new_dd[j]).abs();
        let scale_old = 1.0 + qx_j + problem.c[j].abs() + aty_old_j + bound_contrib[j].abs();
        let scale_new = 1.0 + qx_j + problem.c[j].abs() + aty_new_j + bound_contrib[j].abs();
        let rel_old = f64::from(r_old_dd).abs() / scale_old;
        let rel_new = f64::from(r_new_dd).abs() / scale_new;
        if rel_old > max_rel_old {
            max_rel_old = rel_old;
        }
        if rel_new > max_rel_new {
            max_rel_new = rel_new;
        }
    }
    if max_rel_new < max_rel_old {
        result.dual_solution = y_new;
    }
}

pub(crate) fn refine_dual_lsq_irls(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eps_target: f64,
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return;
    }
    if result.dual_solution.len() != m {
        return;
    }

    let zero_dd = TwoFloat::from(0.0);

    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    let exclude: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0
        })
        .collect();

    let compute_aty = |y: &[f64]| -> Vec<f64> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    };

    let max_rel_with_aty = |aty_v: &[f64]| -> f64 {
        let mut max_rel = 0.0_f64;
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        max_rel
    };

    let mut y_curr = result.dual_solution.clone();
    let initial_aty = compute_aty(&y_curr);
    let initial_max_rel = max_rel_with_aty(&initial_aty);
    if initial_max_rel < eps_target {
        return;
    }

    let mut best_y = y_curr.clone();
    let mut best_max_rel = initial_max_rel;
    let mut prev_max_rel = initial_max_rel;

    /// 単一成分の重み上限 (rel/eps)。> 1e4 で他成分悪化との oscillation が出る。
    const MAX_WEIGHT_RATIO: f64 = 1e4;
    const STAGNATE_RATIO: f64 = 0.95;

    for irls_iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }

        // weight = (rel/eps)² ( LSQ 内部の √w 倍に対し二乗で componentwise 効果を得る )。
        let aty_v = compute_aty(&y_curr);
        let mut weights: Vec<f64> = vec![1.0; n];
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > eps_target {
                let ratio = (rel / eps_target).min(MAX_WEIGHT_RATIO);
                weights[j] = ratio * ratio;
            }
        }

        let mut a_scaled = problem.a.clone();
        for k in 0..n {
            let s = weights[k].sqrt();
            if (s - 1.0).abs() < 1e-15 {
                continue;
            }
            let cs = a_scaled.col_ptr[k];
            let ce = a_scaled.col_ptr[k + 1];
            for idx in cs..ce {
                a_scaled.values[idx] *= s;
            }
        }

        let aat_w = match build_aat_upper_csc(&a_scaled, n, m) {
            Some(mat) => mat,
            None => break,
        };
        let factor = match crate::linalg::ldl::factorize(&aat_w) {
            Ok(f) => f,
            Err(_) => break,
        };

        let mut rhs_dd: Vec<TwoFloat> = vec![zero_dd; m];
        for col in 0..n {
            let wt = weights[col] * target[col];
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                rhs_dd[row] = rhs_dd[row] + TwoFloat::new_mul(problem.a.values[k], wt);
            }
        }
        let rhs: Vec<f64> = rhs_dd.iter().map(|&v| f64::from(v)).collect();

        let mut y_new = vec![0.0_f64; m];
        factor.solve(&rhs, &mut y_new);
        if y_new.iter().any(|v| !v.is_finite()) {
            break;
        }

        let aty_new = compute_aty(&y_new);
        let new_max_rel = max_rel_with_aty(&aty_new);

        if new_max_rel < best_max_rel {
            best_y = y_new.clone();
            best_max_rel = new_max_rel;
        }

        if best_max_rel < eps_target {
            break;
        }
        if irls_iter > 0 && new_max_rel >= prev_max_rel * STAGNATE_RATIO {
            break;
        }
        prev_max_rel = new_max_rel;
        y_curr = y_new;
    }

    if best_max_rel < initial_max_rel {
        result.dual_solution = best_y;
    }
}

