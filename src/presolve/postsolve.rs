//! Postsolve（逆変換）モジュール
//!
//! Presolveで縮約した問題の解を元問題の解空間に復元する。
//! PostsolveStackを逆順（LIFO）に適用する。

use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::options::SolverOptions;
use super::transforms::{PostsolveStep, PresolveResult};

/// cleanup LP の timeout 上限 (削除行数〜数百のため小規模、5秒で十分)。
const CLEANUP_LP_TIMEOUT_SECS: f64 = 5.0;

/// Cleanup LP を構築して解き、削除行の y_i を KKT 整合に決定する。
///
/// 構造:
/// - 変数: 削除行の y_i (k 個)、constraint type の符号慣例で bound 設定
/// - 制約: 未削除列 j で rc[j] = c[j] - Σ_i A_ij y_i = 0 or ≥ 0 or ≤ 0
///   - x[j] at lb: rc ≥ 0 → Σ A_ij y_i ≤ rc_known[j]
///   - x[j] at ub: rc ≤ 0 → Σ A_ij y_i ≥ rc_known[j]
///   - interior:   rc = 0 → Σ A_ij y_i = rc_known[j]
/// - 目的: min 0 (Phase I 風実行可能性のみ)
/// - LinearSubstitution の y_piv: free 変数 rc=0 を Eq 制約で含める (orig_col の rc が 0)
///
/// 戻り値: 削除行ごとの y_i 値 (None なら cleanup LP 構築失敗 or 解けず)
fn build_and_solve_cleanup_lp(
    orig_problem: &LpProblem,
    presolve_result: &PresolveResult,
    solution: &[f64],
    dual_solution_known: &[f64],
) -> Option<Vec<f64>> {
    let n = orig_problem.num_vars;
    let m = orig_problem.num_constraints;
    let deleted_rows: Vec<usize> = (0..m)
        .filter(|&i| presolve_result.row_map[i].is_none())
        .collect();
    let k = deleted_rows.len();
    if k == 0 { return None; }

    let row_to_var: std::collections::HashMap<usize, usize> = deleted_rows
        .iter().enumerate().map(|(idx, &r)| (r, idx)).collect();

    // rc_known[j] = c[j] - Σ_{i: known} A_ij * ŷ_i
    let mut rc_known = orig_problem.c.clone();
    for j in 0..n {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if presolve_result.row_map[row].is_some() {
                    rc_known[j] -= vals[kk] * dual_solution_known[row];
                }
            }
        }
    }

    // 制約構築
    let mut tri_rows: Vec<usize> = Vec::new();
    let mut tri_cols: Vec<usize> = Vec::new();
    let mut tri_vals: Vec<f64> = Vec::new();
    let mut b_clean: Vec<f64> = Vec::new();
    let mut ct_clean: Vec<ConstraintType> = Vec::new();

    // (i) 未削除列 j の rc 符号制約
    for j in 0..n {
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; }

        // 列 j の削除行エントリのみ抽出
        let mut col_terms: Vec<(usize, f64)> = Vec::new();
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if let Some(&var_idx) = row_to_var.get(&row) {
                    col_terms.push((var_idx, vals[kk]));
                }
            }
        }
        if col_terms.is_empty() { continue; }

        // interior 列 (どちらの bound にも hit せず、fixed でもない) は LP dual の
        // 自由度で rc != 0 が自然に発生する。Eq 強制は cleanup LP を過剰制約化して
        // infeasible にするためスキップ。
        let ct = if at_lb && !at_ub {
            ConstraintType::Le
        } else if at_ub && !at_lb {
            ConstraintType::Ge
        } else {
            continue;
        };
        let row_idx = b_clean.len();
        for &(var_idx, a) in &col_terms {
            tri_rows.push(row_idx);
            tri_cols.push(var_idx);
            tri_vals.push(a);
        }
        b_clean.push(rc_known[j]);
        ct_clean.push(ct);
    }

    // (ii) LinearSubstitution の y_piv (orig_col の rc = 0 = free 変数最適性)
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::LinearSubstitution { orig_col, .. } = step {
            let j = *orig_col;
            // rc[orig_col] = c[orig_col] - Σ_i A_{i,orig_col} * y_i = 0
            //   ⇔ Σ_{i: deleted} A_{i,j} * y_i = rc_known[j] (free 変数 KKT)
            let mut col_terms: Vec<(usize, f64)> = Vec::new();
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (kk, &row) in rows.iter().enumerate() {
                    if let Some(&var_idx) = row_to_var.get(&row) {
                        col_terms.push((var_idx, vals[kk]));
                    }
                }
            }
            if col_terms.is_empty() { continue; }
            let row_idx = b_clean.len();
            for &(var_idx, a) in &col_terms {
                tri_rows.push(row_idx);
                tri_cols.push(var_idx);
                tri_vals.push(a);
            }
            b_clean.push(rc_known[j]);
            ct_clean.push(ConstraintType::Eq);
        }
    }

    if b_clean.is_empty() { return None; }

    // Phase I 風 slack 緩和で cleanup LP を常に feasible 化。
    // 各制約に slack 変数を追加し、目的を min Σ s に変更:
    //   Le: Σ a*y - s ≤ rc, s ≥ 0
    //   Ge: Σ a*y + s ≥ rc, s ≥ 0
    //   Eq: Σ a*y + s_pos - s_neg = rc, s_pos/s_neg ≥ 0
    // 最適値 0 なら厳密に rc 符号制約を満たす y が存在、0 でなければ違反量。
    let m_clean = b_clean.len();
    let mut slack_count = 0usize; // 追加 slack 変数の数 (Eq は 2、Le/Ge は 1)
    let mut slack_cols_per_row: Vec<(usize, Option<usize>)> = Vec::with_capacity(m_clean); // (s_idx, optional s_neg_idx for Eq)
    for ct in &ct_clean {
        match ct {
            ConstraintType::Eq => {
                let pos = k + slack_count;
                let neg = k + slack_count + 1;
                slack_cols_per_row.push((pos, Some(neg)));
                slack_count += 2;
            }
            _ => {
                slack_cols_per_row.push((k + slack_count, None));
                slack_count += 1;
            }
        }
    }
    // 各 slack 列を A に追加 (係数 ±1)
    for (row_idx, (s_pos, s_neg_opt)) in slack_cols_per_row.iter().enumerate() {
        let sign = match ct_clean[row_idx] {
            ConstraintType::Le => -1.0,  // a*y - s <= rc
            ConstraintType::Ge => 1.0,   // a*y + s >= rc
            ConstraintType::Eq => 1.0,   // a*y + s_pos - s_neg = rc
        };
        tri_rows.push(row_idx);
        tri_cols.push(*s_pos);
        tri_vals.push(sign);
        if let Some(s_neg) = s_neg_opt {
            tri_rows.push(row_idx);
            tri_cols.push(*s_neg);
            tri_vals.push(-1.0);
        }
    }
    let total_vars = k + slack_count;

    // 変数 bound: y は constraint type 符号慣例、slack は [0, ∞)
    let mut bounds_clean: Vec<(f64, f64)> = Vec::with_capacity(total_vars);
    for &i in &deleted_rows {
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => bounds_clean.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => bounds_clean.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => bounds_clean.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    for _ in 0..slack_count {
        bounds_clean.push((0.0, f64::INFINITY));
    }

    // 目的: min Σ slack (y の係数は 0)
    let mut c_clean = vec![0.0f64; total_vars];
    for j in k..total_vars { c_clean[j] = 1.0; }

    let a_clean = CscMatrix::from_triplets(
        &tri_rows, &tri_cols, &tri_vals, m_clean, total_vars
    ).ok()?;
    let cleanup_lp = LpProblem::new_general(
        c_clean, a_clean, b_clean, ct_clean, bounds_clean, None
    ).ok()?;

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.warm_start = None;
    opts.timeout_secs = Some(CLEANUP_LP_TIMEOUT_SECS);
    let r = crate::simplex::solve_without_presolve(&cleanup_lp, &opts);
    let _ = (slack_count, m_clean);
    if r.status == SolveStatus::Optimal && r.solution.len() == total_vars {
        // 先頭 k 個が y_i、残り slack
        Some(r.solution[..k].to_vec())
    } else {
        None
    }
}

/// CSC 形式の orig_problem.a から行 i のエントリ (j, A_ij) を列挙する。
/// O(nnz_total) の走査だが、一度だけ呼ばれる小規模問題用 (大規模では別途キャッシュ化)。
fn collect_row_entries(orig_problem: &LpProblem, i: usize) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for j in 0..orig_problem.num_vars {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row == i {
                    out.push((j, vals[k]));
                }
            }
        }
    }
    out
}

/// bound active 判定の許容差 (x[j] が lb / ub にどれだけ近ければ active と見るか)。
const BOUND_ACTIVE_TOL: f64 = 1e-6;

/// 削除行 i の y_i を LP dual feasibility に整合的に復元する。
///
/// LP KKT (bound 考慮):
///   x[j] at lb (lb finite, |x-lb|<TOL): rc[j] >= 0 が必要
///   x[j] at ub (ub finite, |x-ub|<TOL): rc[j] <= 0 が必要
///   x[j] interior: rc[j] ≈ 0 が必要
///   x[j] fixed (lb==ub): rc[j] 任意
///
/// rc[j] = rc_at_y0[j] - A_ij * y_i (y_i 以外の y は確定済み) について、
/// 各列の必要符号から y_i の許容範囲 [min_y_i, max_y_i] を導出し、
/// 制約タイプ (Le: y<=0, Ge: y>=0, Eq: free) と交差して 0 に最も近い値を選ぶ。
fn recover_removed_row_dual(
    orig_problem: &LpProblem,
    i: usize,
    solution: &[f64],
    dual_solution: &[f64],
) -> f64 {
    let row_entries = collect_row_entries(orig_problem, i);
    let mut min_y_i = f64::NEG_INFINITY;
    let mut max_y_i = f64::INFINITY;
    for &(j, a_ij) in &row_entries {
        if a_ij.abs() < f64::EPSILON { continue; }
        let mut rc_at_y0 = orig_problem.c[j];
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc_at_y0 -= vals[k] * dual_solution[row];
            }
        }
        // bound active 判定 (x[j] が lb / ub のどちらに hit しているか)
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; } // 固定変数の rc は LP 上自由
        // rc[j] = rc_at_y0 - a_ij * y_i に対する制約:
        //   at_lb && !at_ub: rc >= 0 → a_ij * y_i <= rc_at_y0
        //   at_ub && !at_lb: rc <= 0 → a_ij * y_i >= rc_at_y0
        //   interior:        rc == 0 → a_ij * y_i == rc_at_y0
        let bound_val = rc_at_y0 / a_ij;
        if at_lb && !at_ub {
            if a_ij > 0.0 {
                if bound_val < max_y_i { max_y_i = bound_val; }
            } else {
                if bound_val > min_y_i { min_y_i = bound_val; }
            }
        } else if at_ub && !at_lb {
            // 不等号が逆方向
            if a_ij > 0.0 {
                if bound_val > min_y_i { min_y_i = bound_val; }
            } else {
                if bound_val < max_y_i { max_y_i = bound_val; }
            }
        } else {
            // interior or 両端 hit: rc == 0 (等号制約)
            if bound_val < max_y_i { max_y_i = bound_val; }
            if bound_val > min_y_i { min_y_i = bound_val; }
        }
    }
    let (sign_lb, sign_ub) = match orig_problem.constraint_types[i] {
        ConstraintType::Le => (f64::NEG_INFINITY, 0.0),
        ConstraintType::Ge => (0.0, f64::INFINITY),
        ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
    };
    let lb_y = sign_lb.max(min_y_i);
    let ub_y = sign_ub.min(max_y_i);
    if lb_y <= ub_y {
        if lb_y <= 0.0 && ub_y >= 0.0 { 0.0 }
        else if ub_y < 0.0 { ub_y }
        else { lb_y }
    } else {
        0.0
    }
}

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

    // QP #14 Large Coefficient Row Scaling の逆変換:
    // A_scaled = σ A → A^T y_orig = c ⇔ (σ A)^T (y_orig/σ) = c ⇔ y_scaled = y_orig / σ
    //   → y_orig = σ * y_scaled
    // 縮約後 simplex の出力 y_scaled を元行空間に展開した後で σ_i を掛ける。
    for (i, &s) in presolve_result.row_scales.iter().enumerate() {
        if i < dual_solution.len() {
            dual_solution[i] *= s;
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
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::SingletonRow { orig_col, orig_row, value, a_ij: _, c_j: _ } => {
                solution[*orig_col] = *value;
                // y_i を KKT 整合に復元 (RedundantConstraint と同様)。
                // bound active 時も含めて rc[j]>=0 を満たす y_i を選ぶ。
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
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
                // LinearSubstitution は free 変数 (R6/R15/R5) を Eq 行から消去する変換。
                // free 変数の最適性条件 rc[j] = 0 から:
                //   c_j_orig = Σ_i A_ij * y_i
                //   → y_piv = (c_orig - Σ_{i ≠ piv_row} A_ij * y_i) / pivot
                // col_orig_entries は分配前 (= 行 i の x_j 係数を 0 化する前) の
                // active な (i, A_ij^intermediate) snapshot (piv_row 以外)。
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

    // Step 3: Cleanup LP で削除行の y を一括解決 (Plan 設計の核)。
    // 失敗時は既存 Gauss-Seidel フォールバックを実行する。
    let cleanup_solved = if let Some(y_clean) = build_and_solve_cleanup_lp(
        orig_problem, presolve_result, &solution, &dual_solution,
    ) {
        let deleted_rows: Vec<usize> = (0..m)
            .filter(|&i| presolve_result.row_map[i].is_none())
            .collect();
        for (var_idx, &i) in deleted_rows.iter().enumerate() {
            dual_solution[i] = y_clean[var_idx];
        }
        true
    } else {
        false
    };

    // Gauss-Seidel 風反復 (cleanup LP 失敗時のみフォールバック)。
    // cleanup_solved=true なら cleanup LP の解を保持 (上書きしない)。
    if !cleanup_solved {
    let mut linsub_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::LinearSubstitution { orig_row: Some(r), .. } = step {
            linsub_rows.insert(*r);
        }
    }
    let max_iter = 50;
    let conv_tol = 1e-12;
    for _ in 0..max_iter {
        let mut max_diff = 0.0f64;
        // (a) 一般削除行
        for i in 0..m {
            if presolve_result.row_map[i].is_some() { continue; }
            if linsub_rows.contains(&i) { continue; }
            let new_y = recover_removed_row_dual(orig_problem, i, &solution, &dual_solution);
            let diff = (dual_solution[i] - new_y).abs();
            if diff > max_diff { max_diff = diff; }
            dual_solution[i] = new_y;
        }
        // (b) LinearSubstitution の y_piv (free 変数 rc=0 から逆算)
        for step in &presolve_result.postsolve_stack {
            if let PostsolveStep::LinearSubstitution {
                orig_row: Some(piv),
                col_orig_entries,
                c_orig,
                pivot,
                ..
            } = step {
                let mut sum = 0.0f64;
                for &(row_i, a_ij) in col_orig_entries {
                    if row_i == *piv { continue; }
                    sum += a_ij * dual_solution[row_i];
                }
                let new_y = (c_orig - sum) / pivot;
                let diff = (dual_solution[*piv] - new_y).abs();
                if diff > max_diff { max_diff = diff; }
                dual_solution[*piv] = new_y;
            }
        }
        if max_diff < conv_tol { break; }
    }
    } // !cleanup_solved

    // 被縮小費用は dual_solution が確定した後に元問題で再計算する:
    //   reduced_cost[j] = c[j] - Σ_i A_ij * y_i
    // y が KKT 整合に復元されていれば rc も KKT 整合になる (e61f27b 設計)。
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
