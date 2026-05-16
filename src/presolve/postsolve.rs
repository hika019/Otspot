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
            PostsolveStep::LinearSubstitution {
                orig_col,
                orig_row,
                pivot,
                rhs,
                others,
                col_orig_entries,
                c_orig,
            } => {
                // --- Primal 復元: x_j = (rhs - Σ coeff_k * x_k) / pivot ---
                let mut sum_others = 0.0f64;
                for &(other_col, coeff) in others {
                    sum_others += coeff * solution[other_col];
                }
                solution[*orig_col] = (rhs - sum_others) / pivot;

                // --- Dual 復元: 消去された Eq 行 piv_row の y_piv ---
                // 最適性 (reduced cost = 0 for free pivot var):
                //   c_j_reduced = Σ_{i: A_ij ≠ 0} A_ij_reduced * y_i
                //   y_piv * pivot = c_j_reduced - Σ_{i ≠ piv_row} A_ij_reduced * y_i_postsolved
                // ここで A_ij_reduced は LIFO 適用時点 = 消去直前の col_entries[j]
                // (= col_orig_entries に格納済み)。
                if let Some(piv_row) = orig_row {
                    let mut sum_other_rows = 0.0f64;
                    for &(row_i, a_ij) in col_orig_entries {
                        if row_i == *piv_row {
                            continue;
                        }
                        sum_other_rows += a_ij * dual_solution[row_i];
                    }
                    dual_solution[*piv_row] = (c_orig - sum_other_rows) / pivot;
                }
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

    // 被縮小費用は dual_solution が確定した後に元問題で再計算する:
    //   reduced_cost[j] = c[j] - Σ_i A_ij * y_i
    // これは presolve で消去された変数 (FixedVar / LinearSubstitution 等) でも
    // 最適性に整合した値を与える。
    let mut reduced_costs = orig_problem.c.clone();
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(n) {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc -= vals[k] * dual_solution[row];
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
        ..Default::default()
    }
}
