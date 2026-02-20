//! Postsolve（逆変換）モジュール
//!
//! Presolveで縮約した問題の解を元問題の解空間に復元する。
//! PostsolveStackを逆順（LIFO）に適用する。

use crate::problem::{LpProblem, SolverResult};
use super::transforms::{PostsolveStep, PresolveResult};

/// 縮約後の解を元問題の解空間に復元する。
///
/// # 引数
/// * `result` - 縮約後問題の SolverResult
/// * `presolve_result` - Presolve時に記録した変換情報
/// * `orig_problem` - 元の（縮約前の）LP問題（slack再計算に使用）
///
/// # 戻り値
/// 元問題の変数・制約数に合わせた SolverResult
pub fn run_postsolve(
    result: &SolverResult,
    presolve_result: &PresolveResult,
    orig_problem: &LpProblem,
) -> SolverResult {
    let n = presolve_result.orig_num_vars;
    let m = presolve_result.orig_num_constraints;

    // 縮約後問題の解を元変数空間に展開
    let mut solution = vec![0.0f64; n];
    let mut dual_solution = vec![0.0f64; m];

    // 縮約後のインデックスから元インデックスへ値をコピー
    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < result.solution.len() {
                solution[j] = result.solution[jj];
            }
        }
    }
    for (i, &maybe_ii) in presolve_result.row_map.iter().enumerate() {
        if let Some(ii) = maybe_ii {
            if ii < result.dual_solution.len() {
                dual_solution[i] = result.dual_solution[ii];
            }
        }
    }

    // PostsolveStack を逆順に適用して削除変数・制約を復元
    for step in presolve_result.postsolve_stack.iter().rev() {
        match step {
            PostsolveStep::FixedVariable { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyColumn { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyRow { orig_row } => {
                dual_solution[*orig_row] = 0.0;
            }
            PostsolveStep::SingletonRow { orig_col, orig_row, value } => {
                solution[*orig_col] = *value;
                dual_solution[*orig_row] = 0.0;
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = 0.0;
            }
            PostsolveStep::BoundsTightened { .. } => {
                // Bounds tightening は解の値そのものに影響しない（情報保持のみ）
            }
        }
    }

    // slack を元問題 b - Ax で再計算
    let mut slack = orig_problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n) {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    // 被縮小費用を元変数空間に展開（削除変数は 0）
    let mut reduced_costs = vec![0.0f64; n];
    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < result.reduced_costs.len() {
                reduced_costs[j] = result.reduced_costs[jj];
            }
        }
    }

    // 目的関数値 = 縮約後 objective + presolve で除いた変数の寄与
    let objective = result.objective + presolve_result.obj_offset;

    SolverResult {
        status: result.status.clone(),
        objective,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        warm_start_basis: None, // presolve と warm-start の組み合わせは未対応
    }
}
