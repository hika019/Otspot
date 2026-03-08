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

    // 縮約後の解を元変数空間に展開（削除変数は 0 で初期化）
    let mut solution = vec![0.0f64; n];

    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < reduced_sol.solution.len() {
                solution[j] = reduced_sol.solution[jj];
            }
        }
    }

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
            // #14 LargeCoeffRowScale は双対変数の逆変換のみ。主変数 x は影響しない。
            QpPostsolveStep::LargeCoeffRowScale { .. } => {}
        }
    }

    // 目的関数値 = 縮約後 objective + presolve で除いた変数の定数寄与
    let objective = reduced_sol.objective + presolve_result.obj_offset;

    SolverResult {
        status: reduced_sol.status.clone(),
        objective,
        solution,
        dual_solution: reduced_sol.dual_solution.clone(),
        bound_duals: reduced_sol.bound_duals.clone(),
        active_set: reduced_sol.active_set.clone(),
        iterations: reduced_sol.iterations,
        solver_used: reduced_sol.solver_used,
        final_residuals: reduced_sol.final_residuals,
        ..Default::default()
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
        let prob = QpProblem::new(q, c, a, b, bounds).unwrap();

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
}
