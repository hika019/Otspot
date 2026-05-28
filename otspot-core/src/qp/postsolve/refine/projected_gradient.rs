//! 不等式符号 / inactive 0 制約を保ちつつ ‖A^T y - target‖² を projected gradient で下げる。

use crate::qp::linalg::compute_bound_contrib;
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;

pub(crate) fn refine_dual_projected_gradient(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] += TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    let objective = |y: &[f64]| -> Option<(f64, Vec<f64>)> {
        let aty = if problem.a.nrows > 0 {
            problem.a.transpose().mat_vec_mul(y).ok()?
        } else {
            vec![0.0_f64; n]
        };
        let mut residual = vec![0.0_f64; n];
        let mut obj = 0.0_f64;
        for j in 0..n {
            residual[j] = aty[j] - target[j];
            obj += 0.5 * residual[j] * residual[j];
        }
        Some((obj, residual))
    };

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
        let rhs = -(qx[j] + problem.c[j]) / aij;
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
    for i in 0..m {
        if proj_lower[i] > proj_upper[i] {
            let (lo, hi) = match problem.constraint_types[i] {
                crate::problem::ConstraintType::Le => (0.0, f64::INFINITY),
                crate::problem::ConstraintType::Ge => (f64::NEG_INFINITY, 0.0),
                crate::problem::ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
            };
            proj_lower[i] = lo;
            proj_upper[i] = hi;
        }
    }

    let project_feasible = |y: &mut [f64]| {
        for (i, ct) in problem.constraint_types.iter().enumerate() {
            match ct {
                crate::problem::ConstraintType::Le => y[i] = y[i].max(0.0),
                crate::problem::ConstraintType::Ge => y[i] = y[i].min(0.0),
                crate::problem::ConstraintType::Eq => {}
            }
        }
        for i in 0..m {
            y[i] = y[i].clamp(proj_lower[i], proj_upper[i]);
        }
    };

    let mut y_start = result.dual_solution.clone();
    project_feasible(&mut y_start);
    let Some((mut obj_curr, mut residual_curr)) = objective(&y_start) else {
        return;
    };
    let mut y_curr = y_start;
    let mut y_best = y_curr.clone();
    let mut obj_best = obj_curr;
    let mut prev_obj = obj_curr;

    let pg_max_iters = m.saturating_mul(2).clamp(200, 2000);
    const ACCEPT_TOL_REL: f64 = 1e-12;
    let obj_converge_thresh = 1e-16 * (n as f64).max(1.0);
    const STAGNATE_MIN_RATIO: f64 = 1e-7;

    for _ in 0..pg_max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        if obj_curr < obj_converge_thresh {
            break;
        }
        let grad = match problem.a.mat_vec_mul(&residual_curr) {
            Ok(v) => v,
            Err(_) => break,
        };
        let grad_inf = grad.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if !grad_inf.is_finite() || grad_inf < 1e-14 {
            break;
        }
        let grad_sq = grad.iter().map(|v| v * v).sum::<f64>();
        if !grad_sq.is_finite() || grad_sq < 1e-28 {
            break;
        }
        let aty_grad = match problem.a.transpose().mat_vec_mul(&grad) {
            Ok(v) => v,
            Err(_) => break,
        };
        let curvature = aty_grad.iter().map(|v| v * v).sum::<f64>();
        if !curvature.is_finite() || curvature < 1e-28 {
            break;
        }
        let base_step = (grad_sq / curvature).clamp(1e-14, 1e8);
        let mut accepted = false;
        let mut step = base_step;
        while step > 0.0 {
            let mut y_try = y_curr.clone();
            for i in 0..m {
                y_try[i] -= step * grad[i];
            }
            project_feasible(&mut y_try);
            let Some((obj_try, residual_try)) = objective(&y_try) else {
                continue;
            };
            if obj_try <= obj_curr + ACCEPT_TOL_REL * (1.0 + obj_curr) {
                y_curr = y_try;
                obj_curr = obj_try.min(obj_curr);
                residual_curr = residual_try;
                if obj_curr < obj_best {
                    y_best = y_curr.clone();
                    obj_best = obj_curr;
                }
                accepted = true;
                break;
            }
            let next_step = step * 0.5;
            if next_step == step {
                break;
            }
            step = next_step;
        }
        if !accepted {
            break;
        }
        let relative_improvement = if prev_obj > 0.0 {
            (prev_obj - obj_curr) / prev_obj
        } else {
            0.0
        };
        if relative_improvement < STAGNATE_MIN_RATIO {
            break;
        }
        prev_obj = obj_curr;
    }

    let mut tmp = result.clone();
    tmp.dual_solution = y_best;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if post < pre {
        result.dual_solution = tmp.dual_solution;
    }
}

