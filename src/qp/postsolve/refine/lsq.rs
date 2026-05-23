//! dual y を LSQ で精密化。
//!
//! - `refine_dual_lsq`: A^T y = -(Qx + c + bound_contrib) を LSQ で 1 回解く。
//!   DD 残差で改善した場合のみ採用 (退行防止)。
//! - `refine_dual_lsq_irls`: IRLS で componentwise rel を最小化 (L∞ 漸近)。

use crate::qp::linalg::{build_aat_upper_csc, compute_bound_contrib};
use crate::qp::postsolve::postprocess::compute_lsq_dual_y;
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;

pub(crate) fn refine_dual_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
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
    let use_elim_mask = eliminated_cols.len() == n;
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
        if use_elim_mask && eliminated_cols[j] {
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
    eliminated_cols: &[bool],
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
    if result.dual_solution.len() != m {
        return;
    }
    // 規模ガードは固定 size proxy ではなく AAT 構築の memory_budget
    // (build_aat_upper_csc) と factorize_budget の L_nnz 予算で行う
    // (build が予算超で None / factorize が WouldExceedBudget → break)。

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

    let use_elim_mask = eliminated_cols.len() == n;
    let exclude: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            use_elim_mask && eliminated_cols[j]
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
        let factor = match crate::linalg::ldl::factorize_budget(
            &aat_w,
            crate::linalg::kkt_solver::max_l_nnz_from_budget(),
        ) {
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{SolveStatus, SolverResult};
    use crate::sparse::CscMatrix;

    /// 複数 well-conditioned 問題で y=0 から LSQ refine が真の解に到達するか。
    /// no-op 化検証: refine_dual_lsq 末尾の `if max_rel_new < max_rel_old { ... }` を
    /// 削除 (= y_new を採用しない) すると全 case で初期 y=0 のまま留まり FAIL することを手動で確認済。
    #[test]
    fn refine_dual_lsq_improves_on_multi_pattern_well_conditioned() {
        struct Case {
            name: &'static str,
            a_triplets: (Vec<usize>, Vec<usize>, Vec<f64>, usize, usize),
            c: Vec<f64>,
            x: Vec<f64>,
            expected_y: Vec<f64>,
        }
        let cases = vec![
            Case {
                name: "1x1: a=2, c=6 → y=-3",
                a_triplets: (vec![0], vec![0], vec![2.0], 1, 1),
                c: vec![6.0],
                x: vec![0.0],
                expected_y: vec![-3.0],
            },
            Case {
                name: "1x1: a=4, c=-8 → y=2",
                a_triplets: (vec![0], vec![0], vec![4.0], 1, 1),
                c: vec![-8.0],
                x: vec![0.0],
                expected_y: vec![2.0],
            },
            Case {
                name: "diag 2x2: a=[3,5], c=[-9,-15] → y=[3,3]",
                a_triplets: (vec![0, 1], vec![0, 1], vec![3.0, 5.0], 2, 2),
                c: vec![-9.0, -15.0],
                x: vec![0.0, 0.0],
                expected_y: vec![3.0, 3.0],
            },
        ];

        for case in &cases {
            let (rows, cols, vals, nrows, ncols) = &case.a_triplets;
            let a = CscMatrix::from_triplets(rows, cols, vals, *nrows, *ncols).unwrap();
            let q = CscMatrix::new(*ncols, *ncols);
            let b = vec![0.0_f64; *nrows];
            let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); *ncols];
            let problem = QpProblem::new_all_le(q, case.c.clone(), a, b, bounds).unwrap();
            let mut result = SolverResult {
                status: SolveStatus::Optimal,
                solution: case.x.clone(),
                dual_solution: vec![0.0_f64; *nrows],
                bound_duals: vec![],
                ..SolverResult::default()
            };
            refine_dual_lsq(&problem, &mut result, &[], None);
            for (i, &expected) in case.expected_y.iter().enumerate() {
                assert!(
                    (result.dual_solution[i] - expected).abs() < 1e-9,
                    "{}: y[{}] expected {}, got {}",
                    case.name,
                    i,
                    expected,
                    result.dual_solution[i]
                );
            }
        }
    }

    /// #92 F2: empty A col + 未消去 var を含む問題で DD guard 評価が新旧で割れる。
    ///
    /// Fixture:
    /// - n=2, m=1。A: row 0 = [0, 1] (col 0 が空、col 1 非空)。
    /// - Q=0、c=(100, 0)、bounds [-10,10]^2、x=(0, 0.5)、初期 y=-3 (suboptimal)。
    /// - LSQ y_new = 0 (row 1 のみで y を決定、row 0 は A 空で y 無関係)。
    ///
    /// 評価値:
    /// - r[0]=Qx+c+A^Ty+bc=0+100+0+0=100 (y 独立)、scale[0]=1+0+100+0+0=101 → rel=0.99
    /// - r[1]_old=-3, scale[1]_old=4 → rel=0.75 ; r[1]_new=0, rel=0
    ///
    /// 期待挙動 (mask `[false,false]`、新 logic):
    /// - max_rel_old=max(0.99, 0.75)=0.99、max_rel_new=max(0.99, 0)=0.99
    /// - guard 不通過 → y_new 不採用 → y=-3 を保持
    ///
    /// 旧 A-only logic (col 0 skip) では row 0 を除外して max_rel_old=0.75 / max_rel_new=0
    /// → strict 改善判定で y_new 採用 → y=0 になり assert に FAIL する。
    /// no-op 化検証: skip 条件を `A.col_ptr[j+1]-col_ptr[j]==0` に手動 revert すると
    /// assert "should keep y=-3" が FAIL することを実機確認済。
    #[test]
    fn refine_dual_lsq_does_not_skip_empty_a_col_when_not_eliminated() {
        let a = CscMatrix::from_triplets(&[0], &[1], &[1.0_f64], 1, 2).unwrap();
        let q = CscMatrix::new(2, 2);
        let c = vec![100.0_f64, 0.0_f64];
        let b = vec![0.5_f64];
        let bounds = vec![(-10.0_f64, 10.0_f64), (-10.0_f64, 10.0_f64)];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap();
        let mut result_with_mask = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0_f64, 0.5_f64],
            dual_solution: vec![-3.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        let mut result_no_mask = result_with_mask.clone();

        refine_dual_lsq(&problem, &mut result_with_mask, &[false, false], None);
        refine_dual_lsq(&problem, &mut result_no_mask, &[], None);

        assert!(
            (result_with_mask.dual_solution[0] - (-3.0)).abs() < 1e-6,
            "new mask logic should keep y=-3 (row 0 dominates DD guard), got {}",
            result_with_mask.dual_solution[0]
        );
        assert!(
            (result_no_mask.dual_solution[0] - (-3.0)).abs() < 1e-6,
            "empty mask should also keep y=-3, got {}",
            result_no_mask.dual_solution[0]
        );
    }

    /// #92 F2 補完: eliminated=true を渡すと skip が復活し、y_new が採用される。
    /// 上の sentinel と同 fixture で mask だけ変えて挙動が割れることを確認する。
    #[test]
    fn refine_dual_lsq_skips_when_marked_eliminated() {
        let a = CscMatrix::from_triplets(&[0], &[1], &[1.0_f64], 1, 2).unwrap();
        let q = CscMatrix::new(2, 2);
        let c = vec![100.0_f64, 0.0_f64];
        let b = vec![0.5_f64];
        let bounds = vec![(-10.0_f64, 10.0_f64), (-10.0_f64, 10.0_f64)];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0_f64, 0.5_f64],
            dual_solution: vec![-3.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        // eliminated=true で col 0 を residual 評価から除外 → 旧 A-only 同等。
        refine_dual_lsq(&problem, &mut result, &[true, false], None);
        // col 0 skip により y_new=0 が strict 改善と判定され採用される。
        assert!(
            result.dual_solution[0].abs() < 1e-6,
            "elim=true should let y update to 0, got {}",
            result.dual_solution[0]
        );
    }

    /// refine_dual_lsq の DD guard が「LSQ y が改善しない」ケースで現状 y を保持する。
    /// 既に最適 y を与えてもう一度 refine をかけた場合に y が変わらないこと。
    #[test]
    fn refine_dual_lsq_keeps_y_when_already_optimal() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![6.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let optimal_y = -3.0_f64;
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![optimal_y],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        refine_dual_lsq(&problem, &mut result, &[], None);
        assert!(
            (result.dual_solution[0] - optimal_y).abs() < 1e-12,
            "expected y={} preserved, got {}",
            optimal_y,
            result.dual_solution[0]
        );
    }
}
