//! Ruiz スケーリングラッパー・アンスケール・後検証。

use crate::linalg::ruiz::RuizScaler;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::kkt_resid;
use crate::qp::problem::QpProblem;

/// `user_eps / amplification` の machine-noise floor。
/// `IPM_EPS_NOISE_FLOOR` (ipm_core/mod.rs) と整合: core.rs の σ-tightening と
/// scaling.rs の amp-tightening は両方とも IPM convergence eps の下押しで、
/// amp > 100 が起きると 1×EPS 旧 floor が core 側 floor を defeat してしまう。
pub(crate) const EPS_FLOOR: f64 = super::IPM_EPS_NOISE_FLOOR;

/// Suboptimal → Optimal 昇格時の双対ギャップ閾値。
pub(crate) const PROMOTION_GAP_TOL: f64 = 1e-1;

/// OSQP 流 pfeas: `||v||_∞ / (1 + max(||Ax||_∞, ||b||_∞))`。A·x は cancellation 対策で DD 積算。
fn compute_pfeas_osqp(problem: &QpProblem, x: &[f64]) -> f64 {
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    if problem.num_constraints == 0 {
        return 0.0;
    }
    let m = problem.a.nrows;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..problem.a.ncols {
        let xv = x[col];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax_dd[problem.a.row_ind[k]] += TwoFloat::new_mul(problem.a.values[k], xv);
        }
    }
    let mut max_v = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for (i, (ax_dd_i, &b_i)) in ax_dd.iter().zip(problem.b.iter()).enumerate() {
        let raw_dd = *ax_dd_i - TwoFloat::from(b_i);
        let raw = f64::from(raw_dd);
        let violation = match problem.constraint_types.get(i) {
            Some(ConstraintType::Eq) => raw.abs(),
            Some(ConstraintType::Ge) => (-raw).max(0.0),
            _ => raw.max(0.0),
        };
        max_v = max_v.max(violation);
        max_ax = max_ax.max(f64::from(*ax_dd_i).abs());
        max_b = max_b.max(b_i.abs());
    }
    max_v / (1.0 + max_ax.max(max_b))
}

/// solve_qp_ippmm 用の Ruiz スケーリングラッパー。
pub(crate) fn solve_with_ruiz_scaling<F>(
    problem: &QpProblem,
    options: &SolverOptions,
    inner_solver: F,
) -> SolverResult
where
    F: Fn(&QpProblem, &SolverOptions, f64) -> SolverResult,
{
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(mut scaled_problem) = QpProblem::new(
            q_s, c_s, a_s, b_s, bounds_s, problem.constraint_types.clone(),
        ) {
            scaled_problem.obj_offset = problem.obj_offset;
            // unscale 後に元空間 eps を保証するため scaled 空間 eps を amp 倍 tighten。
            let amplification = compute_amplification(&scaler);
            let mut adjusted_opts = options.clone();
            adjusted_opts.ipm.eps =
                (options.ipm_eps() / amplification).max(EPS_FLOOR);
            // warm start: user 空間 (x, y) を scaled 空間に変換 (Ruiz: x = D·x_s, y = E·y_s/c)
            if let Some(ws) = adjusted_opts.warm_start_qp.as_mut() {
                if ws.x.len() == n && ws.y.len() == m {
                    for j in 0..n { ws.x[j] /= scaler.d[j]; }
                    for i in 0..m { ws.y[i] = scaler.c * ws.y[i] / scaler.e[i]; }
                } else {
                    eprintln!(
                        "[warm_start_qp dropped] ruiz dim mismatch: ws.x={}/{} ws.y={}/{}",
                        ws.x.len(), n, ws.y.len(), m
                    );
                    adjusted_opts.warm_start_qp = None;
                }
            }

            let scaled_result = inner_solver(
                &scaled_problem,
                &adjusted_opts,
                options.ipm_eps(),
            );
            let result = unscale_ipm_result(scaled_result, &scaler, problem, options.ipm_eps());

            if result.status == SolveStatus::MaxIterations {
                if !result.solution.is_empty() {
                    return SolverResult { status: SolveStatus::SuboptimalSolution, ..result };
                } else {
                    return SolverResult { status: SolveStatus::Timeout, ..result };
                }
            }
            return result;
        }
    }

    post_verify_solution(
        inner_solver(problem, options, options.ipm_eps()),
        problem,
        options.ipm_eps(),
    )
}

/// SuboptimalSolution を原空間で再検証し pfeas/bfeas/dfeas が eps を満たせば Optimal に昇格。
/// 非 Ruiz パス専用。
pub(crate) fn post_verify_solution(
    result: SolverResult,
    problem: &QpProblem,
    eps: f64,
) -> SolverResult {
    if result.status != SolveStatus::SuboptimalSolution || result.solution.is_empty() {
        return result;
    }
    let x = &result.solution;
    let y = &result.dual_solution;
    let bound_duals = &result.bound_duals;
    let status = if problem.num_constraints > 0 {
        let pfeas_normalized = compute_pfeas_osqp(problem, x);
        if pfeas_normalized.is_finite() && pfeas_normalized < eps {
            let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
            if bfeas_status == SolveStatus::Optimal {
                check_dfeas_status_relative(problem, x, y, bound_duals, eps)
            } else {
                bfeas_status
            }
        } else {
            SolveStatus::SuboptimalSolution
        }
    } else {
        let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
        if bfeas_status == SolveStatus::Optimal {
            check_dfeas_status_relative(problem, x, y, bound_duals, eps)
        } else {
            bfeas_status
        }
    };
    // Suboptimal→Optimal 昇格ゲート: 双対ギャップ閾値外なら Optimal に上げない。
    let status = if status == SolveStatus::Optimal {
        match result.duality_gap_rel {
            Some(g) if g.abs() >= PROMOTION_GAP_TOL => SolveStatus::SuboptimalSolution,
            _ => status,
        }
    } else {
        status
    };
    SolverResult { status, ..result }
}

/// lb <= x <= ub の違反量を検証し、超過していれば SuboptimalSolution に降格する。
pub(crate) fn check_bfeas_status(x: &[f64], bounds: &[(f64, f64)], eps: f64) -> SolveStatus {
    let bfeas: f64 = x
        .iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lb, ub))| {
            let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lb_viol.max(ub_viol)
        })
        .fold(0.0_f64, f64::max);
    if bfeas < eps {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// inf-norm 絶対基準の dfeas 検証。絶対基準の semantics を sentinel する test から呼ばれる。
#[cfg(test)]
pub(crate) fn check_dfeas_status(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
    threshold: f64,
) -> SolveStatus {
    let n = x.len();
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return SolveStatus::SuboptimalSolution,
    };
    let aty: Vec<f64> = if problem.a.nrows > 0 && !y.is_empty() {
        match problem.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return SolveStatus::SuboptimalSolution,
        }
    } else {
        vec![0.0; n]
    };
    let bound_contrib = kkt_resid::bound_contrib(&problem.bounds, bound_duals);
    let dfeas = (0..n)
        .map(|i| (qx[i] + aty[i] + bound_contrib[i] + problem.c[i]).abs())
        .fold(0.0_f64, f64::max);
    if dfeas < threshold {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// 成分相対化された stationarity + complementarity チェック。
/// stationarity だけ見ると inactive 制約の y 大 × slack 小で偽 Optimal が出るため両方判定。
pub(crate) fn check_dfeas_status_relative(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
    eps: f64,
) -> SolveStatus {
    use twofloat::TwoFloat;
    let n = x.len();
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] += TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
    if problem.a.nrows > 0 && !y.is_empty() {
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                aty_dd[col] += TwoFloat::new_mul(problem.a.values[k], y[problem.a.row_ind[k]]);
            }
        }
    }
    let bound_contrib = kkt_resid::bound_contrib(&problem.bounds, bound_duals);
    // 全体最大値スケールでは外れ残差を 1 成分でマスクするため、各成分 j を独立正規化して max。
    let mut dfeas_relative = 0.0_f64;
    for j in 0..n {
        let r_dd = qx_dd[j] + aty_dd[j] + TwoFloat::from(bound_contrib[j]) + TwoFloat::from(problem.c[j]);
        let r = f64::from(r_dd).abs();
        let scale_j = 1.0
            + f64::from(qx_dd[j]).abs()
            + f64::from(aty_dd[j]).abs()
            + bound_contrib[j].abs()
            + problem.c[j].abs();
        dfeas_relative = dfeas_relative.max(r / scale_j);
    }
    if dfeas_relative >= eps {
        return SolveStatus::SuboptimalSolution;
    }
    let comp = complementarity_relative(problem, x, y, bound_duals);
    if comp < eps {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// 元空間 complementarity 残差 (問題全体スケール正規化)。
fn complementarity_relative(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
) -> f64 {
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);

    let m = problem.a.nrows;
    let ax_dd: Vec<TwoFloat> = if m > 0 {
        let mut ax = vec![zero_dd; m];
        for col in 0..problem.a.ncols {
            let xv = x[col];
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                ax[problem.a.row_ind[k]] += TwoFloat::new_mul(problem.a.values[k], xv);
            }
        }
        ax
    } else {
        Vec::new()
    };

    let yb: f64 = y.iter().zip(problem.b.iter()).map(|(&yi, &bi)| yi * bi).sum();
    let yax: f64 = y
        .iter()
        .zip(ax_dd.iter())
        .map(|(&yi, &ax_dd_i)| yi * f64::from(ax_dd_i))
        .sum();
    let zx: f64 = {
        let mut s = 0.0_f64;
        let mut idx = 0_usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bound_duals.len() {
                s += bound_duals[idx] * x[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bound_duals.len() {
                s += bound_duals[idx] * x[j];
                idx += 1;
            }
        }
        s
    };
    let cx: f64 = problem.c.iter().zip(x.iter()).map(|(&c, &xi)| c * xi).sum();
    let xqx: f64 = {
        let qx = problem.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; x.len()]);
        qx.iter().zip(x.iter()).map(|(&q, &xi)| q * xi).sum()
    };
    let scale = 1.0
        + yb.abs()
        + yax.abs()
        + zx.abs()
        + cx.abs()
        + (0.5 * xqx).abs();

    let mut max_abs = 0.0_f64;

    if m > 0 && !y.is_empty() {
        for (i, ct) in problem.constraint_types.iter().enumerate() {
            let slack_dd = match ct {
                ConstraintType::Eq => continue,
                ConstraintType::Le => TwoFloat::from(problem.b[i]) - ax_dd[i],
                ConstraintType::Ge => ax_dd[i] - TwoFloat::from(problem.b[i]),
            };
            let prod = (f64::from(slack_dd) * y[i]).abs();
            if prod > max_abs {
                max_abs = prod;
            }
        }
    }

    if !bound_duals.is_empty() {
        let mut idx = 0_usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bound_duals.len() {
                let prod = (bound_duals[idx] * (x[j] - lb)).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bound_duals.len() {
                let prod = (bound_duals[idx] * (ub - x[j])).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
    }
    max_abs / scale
}

/// unscale 残差増幅率 = max(1/min(e), 1/(c·min(d)))。MIN_POSITIVE で div0 防護。
pub(crate) fn compute_amplification(scaler: &RuizScaler) -> f64 {
    let e_min = if scaler.e.is_empty() {
        1.0
    } else {
        scaler.e.iter().cloned().fold(f64::INFINITY, f64::min).max(f64::MIN_POSITIVE)
    };
    let d_min = if scaler.d.is_empty() {
        1.0
    } else {
        scaler.d.iter().cloned().fold(f64::INFINITY, f64::min).max(f64::MIN_POSITIVE)
    };
    (1.0 / e_min).max(1.0 / (scaler.c * d_min))
}

/// スケール済み IPM 結果を元スケールに戻し、Optimal は元空間 KKT で再検証する。
pub(crate) fn unscale_ipm_result(
    result: SolverResult,
    scaler: &RuizScaler,
    problem: &QpProblem,
    eps: f64,
) -> SolverResult {
    match result.status {
        SolveStatus::Optimal => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds);
            let obj_orig = result.objective / scaler.c;
            let (status, orig_residuals) = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(ax) => {
                        let pfeas: f64 = ax
                            .iter()
                            .zip(problem.b.iter())
                            .zip(problem.constraint_types.iter())
                            .map(|((&ax_i, &b_i), ct)| match ct {
                                crate::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                                crate::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
                                _ => (ax_i - b_i).max(0.0),
                            })
                            .fold(0.0_f64, f64::max);
                        let pfeas_normalized = compute_pfeas_osqp(problem, &x);
                        let orig_resid = result.final_residuals.map(|(_, d, g)| (pfeas, d, g));
                        let status = if pfeas_normalized.is_finite() && pfeas_normalized < eps {
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                            } else {
                                bfeas_status
                            }
                        } else {
                            SolveStatus::SuboptimalSolution
                        };
                        (status, orig_resid)
                    }
                    Err(_) => (SolveStatus::SuboptimalSolution, result.final_residuals),
                }
            } else {
                let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                let status = if bfeas_status == SolveStatus::Optimal {
                    check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                } else {
                    bfeas_status
                };
                (status, result.final_residuals)
            };
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                bound_duals,
                status,
                final_residuals: orig_residuals,
                ..result
            }
        }
        SolveStatus::Timeout => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let obj_orig = result.objective / scaler.c;
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                ..result
            }
        }
        SolveStatus::SuboptimalSolution => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds);
            let obj_orig = result.objective / scaler.c;
            let status = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(_ax) => {
                        let pfeas_normalized = compute_pfeas_osqp(problem, &x);
                        if pfeas_normalized.is_finite() && pfeas_normalized < eps {
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                            } else {
                                bfeas_status
                            }
                        } else {
                            SolveStatus::SuboptimalSolution
                        }
                    }
                    Err(_) => SolveStatus::SuboptimalSolution,
                }
            } else {
                let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                if bfeas_status == SolveStatus::Optimal {
                    check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                } else {
                    bfeas_status
                }
            };
            // null-space 漂流 (残差小・ギャップ大) の偽 Optimal を弾く最終防壁。
            let status = if status == SolveStatus::Optimal {
                match result.duality_gap_rel {
                    Some(g) if g.abs() >= PROMOTION_GAP_TOL => SolveStatus::SuboptimalSolution,
                    _ => status,
                }
            } else {
                status
            };
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                bound_duals,
                status,
                ..result
            }
        }
        _ => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::ruiz::RuizScaler;

    #[test]
    fn compute_amplification_includes_dual_side() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![0.01, 1.0];
        scaler.d = vec![0.001, 1.0];
        scaler.c = 0.1;
        let amp = compute_amplification(&scaler);
        assert!((amp - 10000.0).abs() < 1.0, "got {:.3e}", amp);
    }

    #[test]
    fn compute_amplification_primal_dominant() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![1e-5, 1.0];
        scaler.d = vec![0.5, 1.0];
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!((amp - 1e5).abs() < 10.0, "got {:.3e}", amp);
    }
}
