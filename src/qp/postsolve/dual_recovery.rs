//! dual recovery / refine 系で共通する小規模 helper を集約。
//!
//! - 行スラック許容、active 判定、進捗許容
//! - singleton 列由来の y 上下界 ([lower, upper] interval)
//! - row activity 計算 (A·x と |A|·|x| を一度に)
//! - candidate 列集合の cluster 拡張
//! - active な bound 変数 (z_lb / z_ub) の選択
//! - bound 制約のない自由 (interior) 列の抽出

use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;

/// 行 i が active 判定される際の slack 相対許容係数 (KKT residual と同 scale)。
pub(crate) const DUAL_RECOVERY_ACTIVE_TOL_REL: f64 = 1e-8;

/// 線形項 (a row, ax, b, |A|·|x|) スケールに比例した slack 許容を返す。
pub(crate) fn dual_recovery_row_slack_tol(
    problem: &QpProblem,
    row: usize,
    ax: f64,
    row_abs_activity: f64,
    rel: f64,
) -> f64 {
    rel * (1.0 + problem.b[row].abs() + ax.abs() + row_abs_activity)
}

/// 連続反復で改善量が f64 epsilon ノイズ floor を超えたかを判定するための許容。
pub(crate) fn dual_recovery_progress_tol(prev_kkt: f64, cur_kkt: f64, target_pf: f64) -> f64 {
    let scale = prev_kkt
        .abs()
        .max(cur_kkt.abs())
        .max(target_pf.abs())
        .max(1.0);
    64.0 * f64::EPSILON * scale
}

/// 行が active (Eq、もしくは slack < tol) か。
pub(crate) fn row_is_active_for_dual_recovery(
    problem: &QpProblem,
    row: usize,
    ax: &[f64],
    row_abs_activity: &[f64],
    slack_tol_rel: f64,
) -> bool {
    match problem.constraint_types[row] {
        crate::problem::ConstraintType::Eq => true,
        crate::problem::ConstraintType::Le => {
            let slack = problem.b[row] - ax[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
        crate::problem::ConstraintType::Ge => {
            let slack = ax[row] - problem.b[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
    }
}

/// A·x と Σ_j |A_ij|·|x_j| を一度の走査で返す。dual recovery で頻繁に必要。
pub(crate) fn compute_dual_recovery_row_activity(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ax = problem.a.mat_vec_mul(solution).ok()?;
    let mut row_abs_activity = vec![0.0_f64; problem.num_constraints];
    for j in 0..problem.num_vars {
        let xabs = solution[j].abs();
        if xabs == 0.0 {
            continue;
        }
        for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
            let row = problem.a.row_ind[k];
            row_abs_activity[row] += problem.a.values[k].abs() * xabs;
        }
    }
    Some((ax, row_abs_activity))
}

/// singleton 列 j を持つ行 i の y_i に対する feasible interval [lower, upper] を計算。
/// Le / Ge 行の sign 制約 + 明確 slack 行の 0 化 + one-sided bound 列由来の片側制約を合成する。
pub(crate) fn compute_dual_recovery_row_bounds(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if solution.len() != n {
        return None;
    }

    let qx = problem.q.mat_vec_mul(solution).ok()?;
    let (ax, row_abs_activity) = compute_dual_recovery_row_activity(problem, solution)?;

    let mut lower = vec![f64::NEG_INFINITY; m];
    let mut upper = vec![f64::INFINITY; m];

    for (row, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => lower[row] = 0.0,
            crate::problem::ConstraintType::Ge => upper[row] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..m {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            lower[i] = 0.0;
            upper[i] = 0.0;
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
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }

        let rhs = -(qx[j] + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }

        match (lb_finite, ub_finite) {
            // lb-only: qx + c + a*y = z_lb >= 0
            (true, false) => {
                if aij > 0.0 {
                    lower[row] = lower[row].max(rhs);
                } else {
                    upper[row] = upper[row].min(rhs);
                }
            }
            // ub-only: qx + c + a*y = -z_ub <= 0
            (false, true) => {
                if aij > 0.0 {
                    upper[row] = upper[row].min(rhs);
                } else {
                    lower[row] = lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }

    Some((lower, upper))
}

/// worst 候補列から start し、cluster に隣接する active 行を BFS で集める。
pub(crate) fn collect_dual_recovery_cluster_rows(
    problem: &QpProblem,
    candidate_cols: &[usize],
    candidate_rel: &[f64],
    ax: &[f64],
    row_abs_activity: &[f64],
    _target_pf: f64,
) -> Option<(usize, Vec<usize>)> {
    debug_assert_eq!(candidate_cols.len(), candidate_rel.len());
    if candidate_cols.is_empty() {
        return None;
    }

    let mut order: Vec<usize> = (0..candidate_cols.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        candidate_rel[rhs]
            .partial_cmp(&candidate_rel[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst_pos = order[0];
    let worst_j = candidate_cols[worst_pos];
    let worst_rel = candidate_rel[worst_pos];
    if !worst_rel.is_finite() || worst_rel <= 0.0 {
        return None;
    }

    const CLUSTER_REL_CUTOFF_RATIO: f64 = 0.25;
    let rel_cutoff = worst_rel * CLUSTER_REL_CUTOFF_RATIO;

    let m = problem.num_constraints;
    let mut in_cluster = vec![false; m];
    let mut rows = Vec::new();
    let push_active_rows = |col: usize, in_cluster: &mut [bool], rows: &mut Vec<usize>| {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            if !row_is_active_for_dual_recovery(
                problem,
                row,
                ax,
                row_abs_activity,
                DUAL_RECOVERY_ACTIVE_TOL_REL,
            ) {
                continue;
            }
            if !in_cluster[row] {
                in_cluster[row] = true;
                rows.push(row);
            }
        }
    };
    push_active_rows(worst_j, &mut in_cluster, &mut rows);
    if rows.is_empty() {
        return None;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &pos in &order {
            if candidate_rel[pos] < rel_cutoff {
                break;
            }
            let col = candidate_cols[pos];
            let touches_cluster = (problem.a.col_ptr[col]..problem.a.col_ptr[col + 1])
                .any(|k| in_cluster[problem.a.row_ind[k]]);
            if !touches_cluster {
                continue;
            }
            let before = rows.len();
            push_active_rows(col, &mut in_cluster, &mut rows);
            if rows.len() > before {
                changed = true;
            }
        }
    }

    rows.sort_unstable();
    Some((worst_j, rows))
}

/// active な bound 変数の slot 識別子。-1 なら lb (z_lb)、+1 なら ub (z_ub) 寄与。
#[derive(Clone, Copy)]
pub(crate) enum DualRecoveryBoundVar {
    Lower { var: usize, slot: usize },
    Upper { var: usize, slot: usize },
}

impl DualRecoveryBoundVar {
    pub(crate) fn var(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { var, .. } | DualRecoveryBoundVar::Upper { var, .. } => var,
        }
    }

    pub(crate) fn slot(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { slot, .. } | DualRecoveryBoundVar::Upper { slot, .. } => {
                slot
            }
        }
    }

    pub(crate) fn coeff(self) -> f64 {
        match self {
            DualRecoveryBoundVar::Lower { .. } => -1.0,
            DualRecoveryBoundVar::Upper { .. } => 1.0,
        }
    }
}

/// cluster 列の active bound を選別: 残差符号と既存 z>0 を見て lb/ub を unique に振る。
pub(crate) fn select_dual_recovery_local_bounds(
    problem: &QpProblem,
    solution: &[f64],
    bound_duals: &[f64],
    cols: &[usize],
    provisional_residual: &[f64],
) -> (Vec<DualRecoveryBoundVar>, Vec<usize>) {
    let n = problem.num_vars;
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let mut lb_slot_of_var = vec![None; n];
    let mut ub_slot_of_var = vec![None; n];
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            lb_slot_of_var[j] = Some(lb_slot);
            lb_slot += 1;
        }
        if ub.is_finite() {
            ub_slot_of_var[j] = Some(ub_slot);
            ub_slot += 1;
        }
    }

    let mut local_bounds = Vec::new();
    for &col in cols {
        let xj = solution[col];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[col];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }
        let lb_active = lb.is_finite()
            && ((xj - lb).abs() <= tol
                || lb_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let ub_active = ub.is_finite()
            && ((ub - xj).abs() <= tol
                || ub_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let residual_j = provisional_residual[col];
        let lb_can_help = residual_j > 0.0
            || lb_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        let ub_can_help = residual_j < 0.0
            || ub_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        match (lb_active, ub_active) {
            (true, false) => {
                if lb_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                }
            }
            (false, true) => {
                if ub_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (true, true) => {
                if lb_can_help && !ub_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                } else if ub_can_help && !lb_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                } else if lb_can_help && ub_can_help {
                    let lb_dist = (xj - lb).abs();
                    let ub_dist = (ub - xj).abs();
                    if lb_dist <= ub_dist {
                        if let Some(slot) = lb_slot_of_var[col] {
                            local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                        }
                    } else if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (false, false) => {}
        }
    }

    let mut bound_pos_of_var = vec![usize::MAX; n];
    for (pos, &bound) in local_bounds.iter().enumerate() {
        bound_pos_of_var[bound.var()] = pos;
    }
    (local_bounds, bound_pos_of_var)
}

/// bound 制約のない自由列を抽出 (どの bound にも close でない、presolve 未消去)。
/// 旧来は「A 列空 == skip」だったが kkt.rs / iterative.rs / worst_active.rs と規約を揃え、
/// `eliminated_cols` mask に統一する (linear-only var の誤 skip 防止)。
pub(crate) fn collect_dual_recovery_free_columns(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
    eliminated_cols: &[bool],
) -> Vec<usize> {
    let n = problem.num_vars;
    let use_elim_mask = eliminated_cols.len() == n;
    let mut free_idx: Vec<usize> = Vec::with_capacity(n);
    for j in 0..n {
        let xj = result.solution[j];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && (xj - lb).abs() < tol {
            continue;
        }
        if ub.is_finite() && (ub - xj).abs() < tol {
            continue;
        }
        if use_elim_mask && eliminated_cols[j] {
            continue;
        }
        free_idx.push(j);
    }
    free_idx
}

#[cfg(test)]
mod free_columns_tests {
    //! #92 F2: collect_dual_recovery_free_columns の skip 規約を
    //! 旧 A-only から `eliminated_cols` mask に揃えた sentinel。
    //! 旧 logic で A 空列が常に skip された結果、linear-only var が dual-only IR の
    //! free_idx 抽出から漏れ refine 経路で stationarity が改善されなかった。
    use super::collect_dual_recovery_free_columns;
    use crate::problem::{ConstraintType, SolverResult};
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    fn make_problem_with_aempty_col0() -> (QpProblem, SolverResult) {
        // n=2, m=1
        // A: row 0 = [0, 1] → col 0 空, col 1 非空
        // Q: diag=(0, 0) (irrelevant for free-idx filter)
        // bounds: [-10,10]² (interior 解)
        // x = (0.0, 0.5): どちらも bound 近傍ではない
        let q = CscMatrix::new(2, 2);
        let c = vec![1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[1], &[1.0_f64], 1, 2).unwrap();
        let b = vec![0.5_f64];
        let bounds = vec![(-10.0_f64, 10.0_f64), (-10.0_f64, 10.0_f64)];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();
        let result = SolverResult {
            status: crate::problem::SolveStatus::Optimal,
            solution: vec![0.0_f64, 0.5_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        (problem, result)
    }

    /// Pattern: A col 0 空 + 未消去 (eliminated_cols=[false, false])
    /// 期待: col 0 を free_idx に含む (新 logic)。旧 A-only logic では含まれず FAIL。
    #[test]
    fn empty_a_col_not_eliminated_is_in_free_idx() {
        let (problem, result) = make_problem_with_aempty_col0();
        let elim = vec![false, false];
        let free_idx = collect_dual_recovery_free_columns(&problem, &result, &elim);
        assert!(
            free_idx.contains(&0),
            "col 0 should be in free_idx (not eliminated), got {:?}",
            free_idx
        );
        assert!(
            free_idx.contains(&1),
            "col 1 should be in free_idx, got {:?}",
            free_idx
        );
    }

    /// Pattern: A col 0 空 + 消去 (eliminated_cols=[true, false])
    /// 期待: col 0 を free_idx に含まない (mask 消去)。
    #[test]
    fn empty_a_col_eliminated_is_skipped_from_free_idx() {
        let (problem, result) = make_problem_with_aempty_col0();
        let elim = vec![true, false];
        let free_idx = collect_dual_recovery_free_columns(&problem, &result, &elim);
        assert!(
            !free_idx.contains(&0),
            "col 0 should be skipped (eliminated), got {:?}",
            free_idx
        );
        assert!(
            free_idx.contains(&1),
            "col 1 should still be in free_idx, got {:?}",
            free_idx
        );
    }

    /// Mask 不在 (`&[]`) → use_elim_mask=false → 消去判定無し、純粋に bound 近傍のみで判定。
    #[test]
    fn empty_mask_disables_elimination_check() {
        let (problem, result) = make_problem_with_aempty_col0();
        let free_idx = collect_dual_recovery_free_columns(&problem, &result, &[]);
        assert!(
            free_idx.contains(&0),
            "col 0 should be in free_idx when mask absent, got {:?}",
            free_idx
        );
    }

    /// Pattern: x が bound 近傍 → eliminated_cols とは独立に skip される (退化テスト)。
    #[test]
    fn at_bound_var_skipped_regardless_of_mask() {
        let q = CscMatrix::new(2, 2);
        let c = vec![1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.5_f64];
        // x[0]=0 (lb=0 に張り付き), x[1]=1.5
        let bounds = vec![(0.0_f64, 10.0_f64), (-10.0_f64, 10.0_f64)];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();
        let result = SolverResult {
            status: crate::problem::SolveStatus::Optimal,
            solution: vec![0.0_f64, 1.5_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        // 全 mask パターンで bound 近傍判定が優先される。
        for elim in &[vec![false, false], vec![false, true], vec![]] {
            let free_idx = collect_dual_recovery_free_columns(&problem, &result, elim);
            assert!(
                !free_idx.contains(&0),
                "col 0 at lb should be skipped, mask={:?}, got {:?}",
                elim,
                free_idx
            );
        }
    }
}
