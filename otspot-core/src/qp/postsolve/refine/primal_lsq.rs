//! borderline pf を violating 制約方向に最小ノルム射影で押し込む。
//! (A·A^T) λ = v_active を LDL + DD-IR で解き δ = A^T λ、pf 改善時のみ採用。

use crate::qp::linalg::build_aat_upper_csc;
use crate::qp::problem::QpProblem;
use crate::tolerances::any_nonfinite;

pub(crate) fn refine_primal_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    let x = &mut result.solution;

    // ill-conditioned 系で f64 sum の cancellation を防ぐため Ax を DD で積算。
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..n {
        let xv = x[col];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax_dd[problem.a.row_ind[k]] += TwoFloat::new_mul(problem.a.values[k], xv);
        }
    }
    let ax: Vec<f64> = ax_dd.iter().map(|&v| f64::from(v)).collect();
    const PRIMAL_VIOLATION_TOL: f64 = 1e-12;
    let mut v = vec![0.0_f64; m];
    let mut max_v_pre = 0.0_f64;
    for i in 0..m {
        let raw = match problem.constraint_types[i] {
            ConstraintType::Eq => ax[i] - problem.b[i],
            ConstraintType::Ge => -(ax[i] - problem.b[i]),
            ConstraintType::Le => ax[i] - problem.b[i],
        };
        if raw > PRIMAL_VIOLATION_TOL {
            v[i] = raw;
            max_v_pre = max_v_pre.max(raw);
        }
    }
    if max_v_pre <= PRIMAL_VIOLATION_TOL {
        return;
    }
    // target = ax − b で A δ = target を解く (Le/Ge/Eq とも一貫した符号)。
    let target: Vec<f64> = (0..m)
        .map(|i| {
            match problem.constraint_types[i] {
                ConstraintType::Eq => ax[i] - problem.b[i],
                ConstraintType::Ge => {
                    let r = ax[i] - problem.b[i];
                    if r < -PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
                ConstraintType::Le => {
                    let r = ax[i] - problem.b[i];
                    if r > PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect();
    let target_inf = target.iter().map(|t| t.abs()).fold(0.0_f64, f64::max);
    if target_inf <= PRIMAL_VIOLATION_TOL {
        return;
    }

    // (A A^T) λ = target を LDL + DD-IR (cond(AAT)≈1e13 の暴走を抑制)。
    let aat = match build_aat_upper_csc(&problem.a, n, m) {
        Some(mat) => mat,
        None => return,
    };
    let factor = match crate::linalg::ldl::factorize(&aat) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut lambda = vec![0.0_f64; m];
    factor.solve(&target, &mut lambda);
    if any_nonfinite(&lambda) {
        return;
    }
    const IR_STAGNATE_RATIO: f64 = 0.5;
    const IR_PROGRESS_EPS: f64 = 1e-18;
    let mut prev_r_inf = f64::INFINITY;
    loop {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let mut atl_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for j in 0..n {
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    atl_dd[j] += TwoFloat::new_mul(problem.a.values[k], lambda[i]);
                }
            }
        }
        let mut r_dd: Vec<TwoFloat> = (0..m).map(|i| TwoFloat::from(target[i])).collect();
        for j in 0..n {
            let atl_j_f64 = f64::from(atl_dd[j]);
            let atl_j_lo = atl_dd[j] - TwoFloat::from(atl_j_f64);
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    r_dd[i] = r_dd[i]
                        - TwoFloat::new_mul(problem.a.values[k], atl_j_f64)
                        - TwoFloat::new_mul(problem.a.values[k], f64::from(atl_j_lo));
                }
            }
        }
        let r_f64: Vec<f64> = r_dd.iter().map(|&v| f64::from(v)).collect();
        let r_inf = r_f64.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
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
        let mut dlambda = vec![0.0_f64; m];
        factor.solve(&r_f64, &mut dlambda);
        if any_nonfinite(&dlambda) {
            break;
        }
        for i in 0..m {
            lambda[i] += dlambda[i];
        }
    }

    let mut delta_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for j in 0..n {
        let s = problem.a.col_ptr[j];
        let e = problem.a.col_ptr[j + 1];
        for k in s..e {
            let i = problem.a.row_ind[k];
            if i < m {
                delta_dd[j] += TwoFloat::new_mul(problem.a.values[k], lambda[i]);
            }
        }
    }
    let delta: Vec<f64> = delta_dd.iter().map(|&v| f64::from(v)).collect();
    if any_nonfinite(&delta) {
        return;
    }

    let mut x_new = x.clone();
    for j in 0..n {
        x_new[j] -= delta[j];
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            x_new[j] = x_new[j].max(lb);
        }
        if ub.is_finite() {
            x_new[j] = x_new[j].min(ub);
        }
    }

    // 成分相対化での max rel violation で改善判定 (abs では ill-scaled で見逃す)。
    let ax_new = match problem.a.mat_vec_mul(&x_new) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut max_rel_pre = 0.0_f64;
    let mut max_rel_post = 0.0_f64;
    for i in 0..m {
        let raw_pre = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax[i]).max(0.0),
            ConstraintType::Le => (ax[i] - problem.b[i]).max(0.0),
        };
        let raw_post = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax_new[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax_new[i]).max(0.0),
            ConstraintType::Le => (ax_new[i] - problem.b[i]).max(0.0),
        };
        let scale_pre = 1.0 + ax[i].abs() + problem.b[i].abs();
        let scale_post = 1.0 + ax_new[i].abs() + problem.b[i].abs();
        let rel_pre = raw_pre / scale_pre;
        let rel_post = raw_post / scale_post;
        if rel_pre > max_rel_pre {
            max_rel_pre = rel_pre;
        }
        if rel_post > max_rel_post {
            max_rel_post = rel_post;
        }
    }
    if max_rel_post < max_rel_pre {
        *x = x_new;
    }
}

