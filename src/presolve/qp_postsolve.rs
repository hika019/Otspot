//! QP Postsolve（逆変換）モジュール
//!
//! Presolve で縮約した QP 問題の解を元問題の解空間に復元する。
//! `QpPostsolveStack` を逆順（LIFO）に適用する。

use crate::problem::SolverResult;
use super::qp_transforms::{QpPostsolveStep, QpPresolveResult};

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
            QpPostsolveStep::FixedVar { idx, val } => {
                solution[*idx] = *val;
            }
            QpPostsolveStep::SingletonRow { col, val } => {
                solution[*col] = *val;
            }
            QpPostsolveStep::EmptyCol { idx, val } => {
                solution[*idx] = *val;
            }
            // #14 LargeCoeffRowScale の双対逆変換:
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

    SolverResult {
        status: reduced_sol.status.clone(),
        objective,
        solution,
        dual_solution,
        bound_duals: reduced_sol.bound_duals.clone(),
        active_set: reduced_sol.active_set.clone(),
        iterations: reduced_sol.iterations,
        solver_used: reduced_sol.solver_used,
        final_residuals: reduced_sol.final_residuals,
        pfeas: reduced_sol.pfeas,
        dfeas: reduced_sol.dfeas,
        gap: reduced_sol.gap,
        reduced_costs: reduced_sol.reduced_costs.clone(),
        slack: reduced_sol.slack.clone(),
        warm_start_basis: reduced_sol.warm_start_basis.clone(),
    }
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

    /// postsolve後にpfeas/dfeas/gapがfinal_residualsと整合して伝搬されることを確認
    #[test]
    fn test_postsolve_preserves_residuals() {
        // min 1/2*(x^2 + y^2) - 2x - 2y  s.t. x + y <= 2, x >= 0, y >= 0
        // presolve → IPM solve → postsolve のフルパイプラインで
        // pfeas/dfeas/gap が None でないこと + final_residuals との値一致を検証
        let n = 2usize;
        let m = 1usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![-2.0f64, -2.0f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m, n).unwrap();
        let b = vec![2.0f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();

        // presolve
        let presolve_result = run_qp_presolve_phase1(&prob, &opts);

        // 縮約後問題を解く
        let reduced_sol = solve_qp_with(&presolve_result.reduced, &opts);
        assert_eq!(reduced_sol.status, SolveStatus::Optimal);

        // postsolve
        let final_sol = postsolve_qp(&presolve_result, &reduced_sol);

        // final_residuals が Some なら pfeas/dfeas/gap も Some で値一致
        if let Some((pf, df, g)) = final_sol.final_residuals {
            assert!(final_sol.pfeas.is_some(), "pfeas must be preserved after postsolve");
            assert!(final_sol.dfeas.is_some(), "dfeas must be preserved after postsolve");
            assert!(final_sol.gap.is_some(), "gap must be preserved after postsolve");
            assert_eq!(final_sol.pfeas.unwrap(), pf, "pfeas must match final_residuals");
            assert_eq!(final_sol.dfeas.unwrap(), df, "dfeas must match final_residuals");
            assert_eq!(final_sol.gap.unwrap(), g, "gap must match final_residuals");
        }
    }
}
