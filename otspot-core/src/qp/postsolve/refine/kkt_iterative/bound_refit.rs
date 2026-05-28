//! x, y を不変としたまま bound_duals を KKT stationarity から再計算 (postsolve 後の 0 埋め解消)。
//! bound_contrib = -z_lb + z_ub = -(Qx+c+A^T y) より符号で z_lb/z_ub を候補化、per-col guard 採用。

use crate::qp::linalg::compute_bound_contrib;
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;

pub(crate) fn refit_bound_duals_kkt(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return;
    }
    use twofloat::TwoFloat;
    let x = &result.solution;
    // Q*x と A^T*y は DD で積算 (f64 cancellation 防止)。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] += TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                acc[col] += TwoFloat::new_mul(
                        problem.a.values[k],
                        result.dual_solution[problem.a.row_ind[k]],
                    );
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    } else {
        vec![0.0_f64; n]
    };

    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    if n_lb + n_ub == 0 {
        return;
    }

    let mut new_bd = vec![0.0_f64; n_lb + n_ub];
    // 候補値: target = -(Qx+c+Aty) の符号で z_lb/z_ub を提示。後段 per-col guard で採用判定。
    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let target = -(qx[j] + problem.c[j] + aty[j]);
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();

        if lb_finite && ub_finite {
            // FX (lb==ub) は postsolve 慣例で 0 埋め、KKT 評価からも除外。
            if (lb - ub).abs() >= FX_TOL {
                if target > 0.0 {
                    new_bd[ub_idx] = target;
                } else {
                    new_bd[lb_idx] = -target;
                }
            }
            lb_idx += 1;
            ub_idx += 1;
        } else if lb_finite {
            new_bd[lb_idx] = (-target).max(0.0);
            lb_idx += 1;
        } else if ub_finite {
            new_bd[ub_idx] = target.max(0.0);
            ub_idx += 1;
        }
    }

    // per-col guard: col 単位で改善時のみ採用 (max ベース guard は 1 col 悪化で全 reject になる)。
    let pre_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let post_contrib = compute_bound_contrib(&problem.bounds, &new_bd, n);
    let mut accepted_bd = result.bound_duals.clone();
    if accepted_bd.len() < new_bd.len() {
        accepted_bd.resize(new_bd.len(), 0.0);
    }
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let r_pre = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + pre_contrib[j]).abs()
        };
        let r_post = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + post_contrib[j]).abs()
        };
        let take_new = !is_fx && r_post <= r_pre;
        if lb.is_finite() {
            if take_new && lb_slot < new_bd.len() {
                accepted_bd[lb_slot] = new_bd[lb_slot];
            }
            lb_slot += 1;
        }
        if ub.is_finite() {
            if take_new && ub_slot < new_bd.len() {
                accepted_bd[ub_slot] = new_bd[ub_slot];
            }
            ub_slot += 1;
        }
    }
    result.bound_duals = accepted_bd;
}

