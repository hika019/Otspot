//! QP Postsolve（逆変換）モジュール
//!
//! Presolve で縮約した QP 問題の解を元問題の解空間に復元する。
//! `QpPostsolveStack` を逆順（LIFO）に適用する。

use crate::problem::SolverResult;
use crate::qp::QpProblem;
use super::qp_transforms::{QpPostsolveStep, QpPresolveResult};

/// Pivot singularity threshold for dual recovery.
///
/// When |A[row, col]| < SINGULARITY_TOL, the pivot is treated as numerically
/// singular and the dual recovery for that singleton row is skipped.
/// Note: 1e-30 is much stricter than DROP_TOL (1e-15); value is retained as-is
/// pending further audit of the numerical justification.
const SINGULARITY_TOL: f64 = 1e-30;

/// 縮約後の解を元 QP 問題の解空間に復元する。
///
/// # 引数
/// * `presolve_result` - Presolve 時に記録した変換情報
/// * `reduced_sol` - 縮約後問題の SolverResult
///
/// # 戻り値
/// 元問題の変数・制約数に合わせた SolverResult
pub fn postsolve_qp(presolve_result: &QpPresolveResult, reduced_sol: &SolverResult) -> SolverResult {
    let n = presolve_result.orig_num_vars;
    let m = presolve_result.orig_num_constraints;

    // 縮約後の解を元変数空間に展開（削除変数は 0 で初期化）
    let mut solution = vec![0.0f64; n];

    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < reduced_sol.solution.len() {
                solution[j] = reduced_sol.solution[jj];
            }
        }
    }

    // 双対変数の逆変換: 縮約後空間で LargeCoeffRowScale 逆変換を適用してから
    // row_map で元制約空間に展開する。
    let mut reduced_dual = reduced_sol.dual_solution.clone();

    // PostsolveStack を逆順に適用して削除変数の値を復元
    for step in presolve_result.postsolve_stack.steps.iter().rev() {
        match step {
            QpPostsolveStep::FixedVar { idx, val, .. } => {
                solution[*idx] = *val;
            }
            QpPostsolveStep::SingletonRow { col, val, .. } => {
                solution[*col] = *val;
            }
            QpPostsolveStep::SingletonIneqToBound { .. } => {
                // Primal: x[col] is found by solving the reduced problem with tightened bounds.
                // Dual: y[row] is recovered in postsolve_qp_with_dual_recovery.
            }
            QpPostsolveStep::EmptyCol { idx, val } => {
                solution[*idx] = *val;
            }
            // LargeCoeffRowScale の双対逆変換:
            // スケール σ_i で縮約後制約を A[i]*σ_i, b[i]*σ_i と変換したため、
            // y_orig[i] = σ_i * y_scaled[i]
            QpPostsolveStep::LargeCoeffRowScale { row_scales } => {
                for (i, y) in reduced_dual.iter_mut().enumerate() {
                    if i < row_scales.len() {
                        *y *= row_scales[i];
                    }
                }
            }
        }
    }

    // row_map で縮約後双対変数を元制約空間に展開（削除制約の双対値は 0）
    let mut dual_solution = vec![0.0f64; m];
    for (i, &maybe_ii) in presolve_result.row_map.iter().enumerate() {
        if let Some(ii) = maybe_ii {
            if ii < reduced_dual.len() {
                dual_solution[i] = reduced_dual[ii];
            }
        }
    }

    // 目的関数値 = 縮約後 objective + presolve で除いた変数の定数寄与
    let objective = reduced_sol.objective + presolve_result.obj_offset;

    // reduced_costs を元変数空間に展開（削除変数は 0）
    // LP postsolve (postsolve.rs:82-89) と同方式
    // presolveで除去された変数のrc=0は実装デフォルト（数学的近似）
    let reduced_costs = if !reduced_sol.reduced_costs.is_empty() {
        // LP経路 + 縮約後問題に変数あり: col_mapで展開
        let mut rc = vec![0.0f64; n];
        for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
            if let Some(jj) = maybe_jj {
                if jj < reduced_sol.reduced_costs.len() {
                    rc[j] = reduced_sol.reduced_costs[jj];
                }
            }
        }
        rc
    } else if presolve_result.reduced.num_vars == 0 && n > 0 {
        // LP経路 + 全変数がpresolveで除去済み（singleton_col最適化等）
        // 除去変数は全て最適境界値に固定されているため rc=0
        vec![0.0f64; n]
    } else {
        // QP/IPM経路: IPMはreduced_costsを計算しない → 空を維持
        vec![]
    };

    SolverResult {
        status: reduced_sol.status.clone(),
        objective,
        solution,
        dual_solution,
        bound_duals: reduced_sol.bound_duals.clone(),

        iterations: reduced_sol.iterations,
        final_residuals: reduced_sol.final_residuals,
        duality_gap_rel: reduced_sol.duality_gap_rel,
        reduced_costs,
        slack: reduced_sol.slack.clone(),
        warm_start_basis: reduced_sol.warm_start_basis.clone(),
        timing_breakdown: reduced_sol.timing_breakdown,
        postsolve_dfeas: None,
        stats: reduced_sol.stats.clone(),
        bound_gap_cert: None,
        opt_cert: None,
    }
}

/// `postsolve_qp` の dual recovery 拡張版。
///
/// 既存 `postsolve_qp` は presolve で削除した行の `y` と固定変数の `z` を 0 で
/// 埋めるが、これは KKT を破壊する (Catastrophic 9 件 + QRECIPE の真因)。
/// 本関数は postsolve_stack を逆順に処理しながら、各 step に対応する
/// `y[row]` / `z[idx]` を解析的に復元する。
///
/// # 復元式
///
/// **SingletonRow / RedundantRowFix { row, col, val }**:
///   行 i = row は singleton (Eq) または activity-tightened Eq で削除済。
///   変数 col は val に固定。元 KKT for col:
///     Q\[col,:\]·x + c\[col\] + Σ_k A\[k,col\]·y\[k\] - z_lb\[col\] + z_ub\[col\] = 0
///   col が bound 内部 (lb < val < ub) なら z=0 で確定:
///     y\[row\] = -(Q\[col,:\]·x + c\[col\] + Σ_{k≠row} A\[k,col\]·y\[k\]) / A\[row, col\]
///   col が boundary なら z を後段で再計算 (本関数では y\[row\] のみ復元)
///
/// **FixedVar { idx, val }**:
///   変数 idx を val に固定 (lb==ub または activity から)。
///   z 復元は `core.rs::refit_bound_duals_kkt` が一括で行う (本関数は val のみ書き戻す)。
///
/// **EmptyCol { idx, val }**:
///   Q\[idx,:\]=0, A\[:,idx\]=0 で固定。KKT: c\[idx\] = z_lb\[idx\] - z_ub\[idx\]
///   c\[idx\] > 0 → val=lb, z_lb=c, c<0 → val=ub, z_ub=-c
pub fn postsolve_qp_with_dual_recovery(
    presolve_result: &QpPresolveResult,
    reduced_sol: &SolverResult,
    orig_problem: &QpProblem,
) -> SolverResult {
    // まず通常 postsolve で solution / dual_solution / bound_duals (0 埋め含む) を作成
    let mut sol = postsolve_qp(presolve_result, reduced_sol);

    if sol.solution.len() != orig_problem.num_vars {
        return sol;
    }

    let n = orig_problem.num_vars;
    let _ = n;
    // bound_duals レイアウトを正しく orig_problem の bounds に揃える。
    // postsolve_qp は reduced 空間の bound_duals をそのまま clone してくるため、
    // 元 bounds の lb/ub 数と長さが合わない場合がある。core.rs::run_ipm_with の
    // remap_bound_duals_to_orig がこの修正を行うため、ここでは長さチェックのみ。

    // postsolve_stack を **逆順 (LIFO)** で 1 pass 処理する。
    //
    // 数学的根拠: singleton 結合行列 M[l,k] = A[r_k, j_l] は上三角 (k > l のみ非零)。
    // - row r_k が step k でシングルトン化するとき、step l < k の col j_l はすでに除去済み
    //   → active 問題では A[r_k, j_l] = 0 → 逆に k > l のとき A[r_k, j_l] ≠ 0 が許される
    // 上三角系は後退代入 (逆順処理) で 1 pass 厳密に解ける。
    // forward 順は下三角を仮定した前進代入であり、この問題では発散する。
    for step in presolve_result.postsolve_stack.steps.iter().rev() {
        match step {
            QpPostsolveStep::SingletonRow { row, col, .. } => {
                recover_y_for_singleton_row(*row, *col, orig_problem, &mut sol);
            }
            QpPostsolveStep::SingletonIneqToBound { row, col, ct, .. } => {
                recover_y_for_singleton_row(*row, *col, orig_problem, &mut sol);
                // Clamp to the feasible dual sign per complementary slackness.
                let y = sol.dual_solution[*row];
                sol.dual_solution[*row] = match ct {
                    crate::problem::ConstraintType::Le => y.max(0.0),
                    crate::problem::ConstraintType::Ge => y.min(0.0),
                    _ => y,
                };
            }
            // FixedVar / EmptyCol の z 復元は core.rs::refit_bound_duals_kkt が
            // 一括で行う (bound_duals レイアウトが core.rs::remap で確定するため、
            // ここで z を計算しても上書きされる)。
            _ => {}
        }
    }

    sol
}

/// SingletonRow / RedundantRowFix で削除された行 `row` の dual `y[row]` を
/// 元 KKT 停留性から解析的に復元する。
///
/// 前提: 行 `row` は variable `col` のみが係数を持つ singleton。
///   A[row, col] * x[col] = b[row] (Eq), x[col] = val
///
/// `bound_contrib_col`: KKT for col の bound 寄与 (-z_lb + z_ub) で、
/// bound_duals が orig 空間にレイアウト確定済みなら正しい値を渡す。
/// reduced 空間レイアウトの段階では 0 を渡す (本関数の初回 pass)。
pub(crate) fn recover_y_for_singleton_row(
    row: usize,
    col: usize,
    orig: &QpProblem,
    sol: &mut SolverResult,
) {
    recover_y_for_singleton_row_with_bound(row, col, orig, sol, 0.0);
}

/// `recover_y_for_singleton_row` の bound_contrib 明示版。
pub(crate) fn recover_y_for_singleton_row_with_bound(
    row: usize,
    col: usize,
    orig: &QpProblem,
    sol: &mut SolverResult,
    bound_contrib_col: f64,
) {
    if row >= orig.num_constraints || col >= orig.num_vars {
        return;
    }
    if sol.dual_solution.len() != orig.num_constraints {
        return;
    }
    // A[row, col] を取得 (CSC: col を走査して row を探す)
    let mut a_row_col = 0.0_f64;
    let s = orig.a.col_ptr[col];
    let e = orig.a.col_ptr[col + 1];
    for k in s..e {
        if orig.a.row_ind[k] == row {
            a_row_col = orig.a.values[k];
            break;
        }
    }
    if a_row_col.abs() < SINGULARITY_TOL {
        return;
    }

    // Q[col,:]·x と (A^T y)[col] を DD で積算 (ill-conditioned 系で f64 cancellation
    // が recover_y を狂わせる)。a_row_col × y[row] 部分は target から差し引く。
    use twofloat::TwoFloat;
    let qx_col = compute_qx_at(&orig.q, &sol.solution, col);
    let mut aty_others_dd = TwoFloat::from(0.0);
    for k in s..e {
        let r = orig.a.row_ind[k];
        if r == row { continue; }
        aty_others_dd += TwoFloat::new_mul(orig.a.values[k], sol.dual_solution[r]);
    }
    let aty_col_others = f64::from(aty_others_dd);

    // KKT 停留性: Q[col,:]·x + c[col] + (A^T y)[col] + bound_contrib[col] = 0
    // → A[row, col] * y[row] = -(qx_col + c[col] + aty_col_others + bound_contrib_col)
    let target = -(qx_col + orig.c[col] + aty_col_others + bound_contrib_col);
    let y_new = target / a_row_col;
    if y_new.is_finite() {
        sol.dual_solution[row] = y_new;
    }
}

/// orig 空間の bound_duals レイアウトから 1 変数の bound_contrib (-z_lb + z_ub) を取得。
/// `bound_duals` 長 = n_lb + n_ub (orig.bounds 順)。
pub(crate) fn bound_contrib_at_var(
    bounds: &[(f64, f64)],
    bound_duals: &[f64],
    var: usize,
) -> f64 {
    if bound_duals.is_empty() {
        return 0.0;
    }
    let n_lb_total = bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    let mut contrib = 0.0_f64;
    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb_total;
    for (j, &(lb, ub)) in bounds.iter().enumerate() {
        if lb.is_finite() {
            if j == var && lb_idx < bound_duals.len() {
                contrib -= bound_duals[lb_idx];
            }
            lb_idx += 1;
        }
        if ub.is_finite() {
            if j == var && ub_idx < bound_duals.len() {
                contrib += bound_duals[ub_idx];
            }
            ub_idx += 1;
        }
    }
    contrib
}

/// 対称行列 Q (全要素格納の対称 Q、`spmv_q` と同じ慣例) で Q[col, :] · x を計算する。
///
/// (Q*x)[col] を計算。
///
/// QPS parser (`src/io/qps.rs`) は対称 Q を (i,j) と (j,i) の両方に格納する慣例なので、
/// Q.col(col) を 1 回 walk すれば Q[col,:]·x = Σ_k Q[k,col] · x[k] が得られる。
///
/// ill-conditioned 問題で f64 sum のキャンセル誤差が recover_y の精度を支配する事象を
/// 防ぐため、和は DD (TwoFloat) で行う。
fn compute_qx_at(q: &crate::sparse::CscMatrix, x: &[f64], col: usize) -> f64 {
    use twofloat::TwoFloat;
    let mut sum = TwoFloat::from(0.0);
    let s = q.col_ptr[col];
    let e = q.col_ptr[col + 1];
    for ptr in s..e {
        let k = q.row_ind[ptr];
        sum += TwoFloat::new_mul(q.values[ptr], x[k]);
    }
    f64::from(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presolve::qp_transforms::run_qp_presolve_phase1;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::qp::{solve_qp_with, QpProblem};
    use crate::sparse::CscMatrix;

    /// 固定変数の postsolve: 縮約後解に postsolve を適用し元変数空間に戻る
    #[test]
    fn test_postsolve_fixed_var() {
        // min 1/2*2*x^2  s.t. 0<=x<=1, y=0.5 (fixed)
        // 縮約後: min x^2, x のみ。解 x=0, obj=0
        // postsolve 後: x=0, y=0.5, obj = 0 + obj_offset
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64), (0.5_f64, 0.5_f64)]; // y is fixed
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let presolve_result = run_qp_presolve_phase1(&prob, &opts);

        // 縮約後は x=1 変数
        assert_eq!(presolve_result.reduced.num_vars, 1, "y fixed → 1 var");

        // 縮約後問題を解く
        let reduced_sol = solve_qp_with(&presolve_result.reduced, &opts);
        assert_eq!(reduced_sol.status, SolveStatus::Optimal, "reduced sol optimal");

        // postsolve
        let final_sol = postsolve_qp(&presolve_result, &reduced_sol);
        assert_eq!(final_sol.solution.len(), 2, "restored to 2 vars");
        assert!((final_sol.solution[1] - 0.5).abs() < 1e-8, "y=0.5 restored");
        assert_eq!(final_sol.status, SolveStatus::Optimal);
    }

    /// dual_solution の row_map 逆変換テスト:
    /// 空行（ゼロ係数制約）が削除されたとき、dual_solution が元制約空間の長さを持つことを確認。
    #[test]
    fn test_postsolve_dual_row_map() {
        // min 1/2*(x^2 + y^2)  s.t.
        //   行0: x + y <= 3    (実制約)
        //   行1: 0*x + 0*y <= 5  (空行 → presolve で削除)
        // m_orig = 2, presolve 後 m_reduced = 1
        let n = 2usize;
        let m = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0f64; n];
        // A: 行0 = [1, 1], 行1 = [0, 0]（空行）
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m, n).unwrap();
        let b = vec![3.0f64, 5.0f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let presolve_result = run_qp_presolve_phase1(&prob, &opts);

        // 空行が削除されているはず
        assert_eq!(presolve_result.orig_num_constraints, 2, "orig m=2");
        let m_reduced = presolve_result.reduced.num_constraints;
        assert_eq!(m_reduced, 1, "empty row removed → m_reduced=1");

        // フルパイプラインで解く
        let final_sol = solve_qp_with(&prob, &opts);
        assert_eq!(final_sol.status, SolveStatus::Optimal);
        // dual_solution は元制約空間（長さ 2）でなければならない
        assert_eq!(final_sol.dual_solution.len(), 2,
            "dual_solution must have orig_num_constraints length after postsolve");
    }

    /// dual_solution の値が正しく逆変換されることを確認:
    /// 制約 x + y <= 2 の QP において、KKT 条件から dual 変数の符号・値を検証。
    #[test]
    fn test_postsolve_dual_value_correctness() {
        // min 1/2*(x^2 + y^2) - 2x - 2y  s.t. x + y <= 2, x >= 0, y >= 0
        // KKT: x* = y* = 1, dual y* = 1.0 (制約はアクティブ)
        let n = 2usize;
        let m = 1usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![-2.0f64, -2.0f64]; // 線形項 → 最適解を制約上に引き寄せる
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m, n).unwrap();
        let b = vec![2.0f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&prob, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 1.0).abs() < 1e-6, "x=1");
        assert!((result.solution[1] - 1.0).abs() < 1e-6, "y=1");
        // dual >= 0 (不等式制約の KKT 条件: y >= 0)
        assert_eq!(result.dual_solution.len(), 1, "dual len=1");
        assert!(result.dual_solution[0] >= -1e-6, "dual >= 0");
        // 最適値付近では dual ≈ 1.0
        assert!((result.dual_solution[0] - 1.0).abs() < 1e-4,
            "dual ≈ 1.0, got {}", result.dual_solution[0]);
    }

    /// SingletonRow で FR var が presolve で fix されたときの dual 復元が KKT を満たすか。
    ///
    /// Setup:
    ///   3 vars (x, y, z) all FR
    ///   Eq row 0: 3*x = 6           → singleton, x = 2 (presolve で fix)
    ///   Eq row 1: x + y + z = 5     → 縮約後 y + z = 3
    ///   Q = diag(0, 1, 1), c = [0, -2, -2]
    ///   min 0.5(y² + z²) - 2y - 2z  s.t. above
    ///
    /// 解析解:
    ///   y = z = 1.5 (Lagrangian: y - 2 + λ = 0, λ = 0.5)
    ///   y_row1 = 0.5  (Eq dual)
    ///   y_row0 = -1/6 (KKT for x: 3*y_row0 + 1*y_row1 = 0)
    ///
    /// バグ: postsolve は singleton row の y[row0] を 0 fill するため、
    ///   x の KKT 残差 = 0 + 0 + 3*0 + 1*0.5 = 0.5 ≠ 0 になる。
    /// 修正後は y_row0 が再構築されて r_d ≈ 0 になるべき。
    #[test]
    fn test_postsolve_singleton_row_dual_recovery() {
        let n = 3usize;
        let m = 2usize;
        // Q = diag(0, 1, 1)。col 0 (x) は対角 Q なし、col 1,2 (y,z) は Q=1。
        let q = CscMatrix::from_triplets(&[1, 2], &[1, 2], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, -2.0, -2.0];
        // A:
        //   row 0: 3*x = 6           → A[0,0]=3
        //   row 1: 1*x + 1*y + 1*z = 5  → A[1,0]=1, A[1,1]=1, A[1,2]=1
        let a = CscMatrix::from_triplets(
            &[0, 1, 1, 1],
            &[0, 0, 1, 2],
            &[3.0, 1.0, 1.0, 1.0],
            m, n,
        ).unwrap();
        let b = vec![6.0, 5.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n]; // 全 FR
        let constraint_types = vec![
            crate::problem::ConstraintType::Eq,
            crate::problem::ConstraintType::Eq,
        ];
        let prob = QpProblem::new(q, c, a, b, bounds, constraint_types).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&prob, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "should converge");
        // x ≈ 2, y ≈ z ≈ 1.5
        assert!((result.solution[0] - 2.0).abs() < 1e-5, "x≈2 (fixed by singleton), got {}", result.solution[0]);
        assert!((result.solution[1] - 1.5).abs() < 1e-5, "y≈1.5, got {}", result.solution[1]);
        assert!((result.solution[2] - 1.5).abs() < 1e-5, "z≈1.5, got {}", result.solution[2]);

        // KKT 残差を元空間で計算: r_d[j] = (Qx)[j] + c[j] + (A^T y)[j]
        // (FR 変数なので bound_contrib = 0)
        let qx = prob.q.mat_vec_mul(&result.solution).unwrap();
        let aty = prob.a.transpose().mat_vec_mul(&result.dual_solution).unwrap();
        let mut max_rd = 0.0_f64;
        for j in 0..n {
            let r = (qx[j] + prob.c[j] + aty[j]).abs();
            max_rd = max_rd.max(r);
        }
        // x の stationarity: y_row0 が正しく復元されていれば |r_d[0]| < 1e-6
        // バグ未修正状態では r_d[0] ≈ 0.5 (= y_row1 の漏れ)
        assert!(
            max_rd < 1e-6,
            "max KKT residual should be ≈ 0, got {} (bug: postsolve dropped singleton-row dual)",
            max_rd
        );
    }

    /// Ill-scaled A での dual 復元精度: refine_dual_lsq は LDL(A·A^T) を使うため
    /// cond(A·A^T) = cond(A)² で forward error が増幅される。cond(A)≈1e6 級では
    /// LDL ε × cond² ≈ 2e-4 absolute、これが orig 空間の dual 残差に乗る。
    ///
    /// 本テストは「小規模だが ill-scaled な問題」で QPILOTNO 系の精度限界を
    /// 再現する: presolve fix + ill-scaled A → LSQ の精度限界で KKT 残差残る。
    /// QR-based LSQ への置換 / DD LDL solve で改善見込み (別タスク)。
    #[test]
    fn test_postsolve_dual_recovery_ill_scaled() {
        // 4 vars (x1, x2, x3, x4) all FR, 3 Eq constraints.
        //   row 0: 1e-3 * x1 = 1e-3       → singleton, x1 = 1 (presolve fix)
        //   row 1: 1e6 * x1 + x2 = 1e6 + 1 → well-cond だが huge entry
        //   row 2: x3 + x4 = 2
        // Q = diag(0, 1, 1, 1), c = [0, -2, -2, -2]
        //   y, z, w で 0.5y² + 0.5z² + 0.5w² - 2y - 2z - 2w を最小化。
        // 解析解: x1=1, x2=1, x3=x4=1。dual y_row0 が大きな値で復元されることが必要。
        let n = 4usize;
        let m = 3usize;
        let q = CscMatrix::from_triplets(&[1, 2, 3], &[1, 2, 3], &[1.0, 1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, -2.0, -2.0, -2.0];
        let a = CscMatrix::from_triplets(
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 2, 3],
            &[1e-3, 1e6, 1.0, 1.0, 1.0],
            m, n,
        ).unwrap();
        let b = vec![1e-3, 1e6 + 1.0, 2.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let constraint_types = vec![crate::problem::ConstraintType::Eq; 3];
        let prob = QpProblem::new(q, c, a, b, bounds, constraint_types).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&prob, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);

        // primal は OK
        assert!((result.solution[0] - 1.0).abs() < 1e-5);
        assert!((result.solution[1] - 1.0).abs() < 1e-5);
        assert!((result.solution[2] - 1.0).abs() < 1e-5);
        assert!((result.solution[3] - 1.0).abs() < 1e-5);

        // KKT 残差 (元空間) 計算
        let qx = prob.q.mat_vec_mul(&result.solution).unwrap();
        let aty = prob.a.transpose().mat_vec_mul(&result.dual_solution).unwrap();
        let mut max_rd = 0.0_f64;
        let mut max_aty = 0.0_f64;
        for j in 0..n {
            let r = (qx[j] + prob.c[j] + aty[j]).abs();
            max_rd = max_rd.max(r);
            max_aty = max_aty.max(aty[j].abs());
        }
        // 相対 KKT 残差。eps=1e-6 通過を要求。
        let denom = (1.0_f64).max(max_aty);
        let rel = max_rd / denom;
        // 現在の LDL(A·A^T) ベースの dual 復元では cond(A)≈1e6 で cond²=1e12 が
        // ε≈2e-16 に乗り、abs error ≈ 2e-4 → relative ≈ 2e-10..1e-6 に落ちる。
        // この閾値 1e-5 は「ill-scaled でも妥当な精度を維持」する目安として設定。
        // QPILOTNO (cond≈3e12) はこの閾値も超える。
        assert!(
            rel < 1e-5,
            "ill-scaled でも relative KKT 残差は 1e-5 以下を維持すべき (got {})",
            rel
        );
    }

    /// postsolve後にfinal_residualsが伝搬されることを確認
    #[test]
    fn test_postsolve_preserves_residuals() {
        // min 1/2*(x^2 + y^2) - 2x - 2y  s.t. x + y <= 2, x >= 0, y >= 0
        let n = 2usize;
        let m = 1usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![-2.0f64, -2.0f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m, n).unwrap();
        let b = vec![2.0f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let presolve_result = run_qp_presolve_phase1(&prob, &opts);
        let reduced_sol = solve_qp_with(&presolve_result.reduced, &opts);
        assert_eq!(reduced_sol.status, SolveStatus::Optimal);

        let final_sol = postsolve_qp(&presolve_result, &reduced_sol);
        assert_eq!(final_sol.final_residuals, reduced_sol.final_residuals,
            "final_residuals must be preserved after postsolve");
    }

    /// recover_y_for_singleton_row: SingletonRow 削除時の y[row] を KKT 停留性で復元する
    /// 解析公式が、y を入れ直すと col の stationarity が 0 になることを直接確認。
    ///
    /// 設計:
    ///   2 行 1 列。row 0: 2·x = 4 (singleton で削除、x = 2 に固定)、row 1: x ≤ 10。
    ///   c = 5、Q = 0、bounds = (-inf, inf)。
    ///   KKT for col 0: 0 + 5 + 2·y[0] + 1·y[1] = 0
    ///   y[1] (row 1 の dual) を 0 として recover_y → y[0] = -(5 + 0) / 2 = -2.5
    #[test]
    fn test_recover_y_for_singleton_row_zeroes_stationarity() {
        use crate::problem::ConstraintType;
        let n = 1usize;
        let m = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![5.0_f64];
        // A: row 0 = [2.0] (singleton), row 1 = [1.0]
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[2.0_f64, 1.0], m, n).unwrap();
        let b = vec![4.0_f64, 10.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq, ConstraintType::Le];
        let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

        // x = 2 (singleton で固定された値)、y = [0, 0] (初期)、bound_contrib = 0
        let mut sol = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![2.0],
            dual_solution: vec![0.0, 0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        // row=0, col=0 を SingletonRow として復元
        recover_y_for_singleton_row(0, 0, &prob, &mut sol);
        // 期待: y[0] = -(qx + c + a_others_y + bnd) / A[0, 0]
        //     = -(0 + 5 + 1*0 + 0) / 2 = -2.5
        assert!((sol.dual_solution[0] - (-2.5)).abs() < 1e-12,
            "y[0] should be -2.5, got {}", sol.dual_solution[0]);

        // 復元後 stationarity = qx + c + (A^T y)[0] = 0 + 5 + (2·(-2.5) + 1·0) = 0
        let aty0 = 2.0 * sol.dual_solution[0] + 1.0 * sol.dual_solution[1];
        let stat = 0.0 + 5.0 + aty0;
        assert!(stat.abs() < 1e-12, "stationarity zeroed after recovery, got {}", stat);
    }

    /// compute_qx_at: 対称 Q の col j に対して Σ_k Q[k,j]·x[k] を返すこと、かつ off-diag を
    /// 二重計上しないこと。QPS parser の上下三角両方格納慣例下で正しく動くか確認。
    #[test]
    fn test_compute_qx_at_symmetric_q() {
        // Q = [[2, 3], [3, 4]] (対称)、CSC は両方の (i, j) と (j, i) を格納する。
        let q = CscMatrix::from_triplets(
            &[0, 1, 0, 1], &[0, 0, 1, 1], &[2.0_f64, 3.0, 3.0, 4.0], 2, 2,
        ).unwrap();
        let x = vec![1.0_f64, 1.0];
        // (Q*x)[0] = 2·1 + 3·1 = 5; (Q*x)[1] = 3·1 + 4·1 = 7
        assert!((compute_qx_at(&q, &x, 0) - 5.0).abs() < 1e-12);
        assert!((compute_qx_at(&q, &x, 1) - 7.0).abs() < 1e-12);
    }
}
