//! 元空間 双対ギャップ相対値計算。

use crate::problem::SolverResult;
use crate::qp::problem::QpProblem;

/// 元空間 双対ギャップ相対値: |primal_obj − dual_obj| / max(|p|, |d|, 1)。
/// QP 弱双対性 dual_obj = -1/2 x'Qx - b'y + lb'z_lb - ub'z_ub。rank-deficient Q の
/// 偽 Optimal (KKT 小だが gap 大) を弾く。FX 変数の bound 寄与は lb·停留性で解析的に置換。
pub(crate) fn compute_duality_gap_rel(problem: &QpProblem, result: &SolverResult) -> f64 {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return f64::INFINITY;
    }
    if problem.a.nrows > 0 && result.dual_solution.len() != problem.a.nrows {
        return f64::INFINITY;
    }
    let x = &result.solution;
    let qx = problem.q.mat_vec_mul(x).expect(
        "q.ncols() == x.len() == num_vars: QpProblem::new() enforces \
         q.ncols() == num_vars, and solution.len() == n is checked above",
    );
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        problem
            .a
            .transpose()
            .mat_vec_mul(&result.dual_solution)
            .expect(
                "a.transpose().ncols() == a.nrows() == num_constraints == dual_solution.len(): \
             QpProblem::new() enforces a.nrows() == num_constraints, and \
             dual_solution.len() == a.nrows is checked above",
            )
    } else {
        vec![0.0_f64; n]
    };
    let xqx: f64 = qx.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let cx: f64 = problem.c.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let primal_obj = 0.5 * xqx + cx + problem.obj_offset;

    let mut by: f64 = 0.0;
    for (&bi, &yi) in problem.b.iter().zip(result.dual_solution.iter()) {
        by += bi * yi;
    }

    // FX (lb=ub) は postsolve で z_lb,z_ub が 0 埋めされるため、
    // val * (z_lb - z_ub) = val * (qx + c + aty) で解析的に置換。
    let mut bnd_term: f64 = 0.0;
    let mut lb_idx = 0_usize;
    let mut ub_idx = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < crate::qp::FX_TOL;
        if is_fx {
            let stat_no_bnd = qx[j] + problem.c[j] + aty[j];
            bnd_term += lb * stat_no_bnd;
            if lb_finite {
                lb_idx += 1;
            }
            if ub_finite {
                ub_idx += 1;
            }
        } else {
            if lb_finite {
                if lb_idx >= result.bound_duals.len() {
                    return f64::INFINITY;
                }
                bnd_term += lb * result.bound_duals[lb_idx];
                lb_idx += 1;
            }
            if ub_finite {
                if ub_idx >= result.bound_duals.len() {
                    return f64::INFINITY;
                }
                bnd_term -= ub * result.bound_duals[ub_idx];
                ub_idx += 1;
            }
        }
    }
    let dual_obj = -0.5 * xqx - by + bnd_term + problem.obj_offset;
    let gap_abs = (primal_obj - dual_obj).abs();
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if denom > 0.0 && gap_abs.is_finite() {
        gap_abs / denom
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn one_var_box_qp() -> QpProblem {
        QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, 1.0)],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn duality_gap_rejects_short_bound_duals() {
        let qp = one_var_box_qp();
        let r = SolverResult {
            solution: vec![0.0],
            bound_duals: vec![0.0], // box variable needs lb and ub slots
            ..Default::default()
        };
        assert!(
            compute_duality_gap_rel(&qp, &r).is_infinite(),
            "missing bound-dual slot must not be ignored in duality gap"
        );
    }

    #[test]
    fn duality_gap_rejects_missing_row_dual() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let qp = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            a,
            vec![0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let r = SolverResult {
            solution: vec![0.0],
            dual_solution: vec![],
            ..Default::default()
        };
        assert!(
            compute_duality_gap_rel(&qp, &r).is_infinite(),
            "missing row dual must not be treated as zero"
        );
    }
}
