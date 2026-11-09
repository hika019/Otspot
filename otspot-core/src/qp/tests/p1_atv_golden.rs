//! P1: postsolve refine の Aᵀy 直接 CSC 列内積化 (plan §4) の数値不変 sentinel。
//!
//! 期待値は base 429daf41 (objective 評価ごとに `transpose().mat_vec_mul` 経路)
//! の実走ビット列から採出した golden 定数。実装の確保戦略 (workspace 再利用 +
//! 受理時 swap) やカーネル置換で 1 ビットでも動いたら fail する。
//! 各 fixture は行検索の複数回不受理・best≠current・末尾 KKT ガード・
//! worst-active の dual/bound dual 更新という別々の経路を通す。

use super::super::*;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_num::sparse::CscMatrix;

/// 行検索が 13 回不受理を挟み、最終受理点が末尾 KKT ガードを通過して
/// dual_solution が実際に動く fixture (sweep idx=390)。
pub(super) fn fixture_multi_reject_moving() -> (QpProblem, SolverResult) {
    let a = CscMatrix::from_triplets(
        &[0, 1, 1, 2, 3],
        &[1, 0, 1, 0, 1],
        &[1e6_f64, 1e-6, 0.5, -4.0, 1.0],
        4,
        2,
    )
    .unwrap();
    let q = CscMatrix::from_triplets(&[1], &[1], &[4.0_f64], 2, 2).unwrap();
    let problem = QpProblem::new(
        q,
        vec![0.25_f64, 0.25],
        a,
        vec![10.0_f64, 0.5, 1.0, -3.0],
        vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)],
        vec![
            ConstraintType::Ge,
            ConstraintType::Eq,
            ConstraintType::Ge,
            ConstraintType::Ge,
        ],
    )
    .unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0_f64, 2.0],
        dual_solution: vec![0.75_f64, -0.1, 0.75, -0.1],
        bound_duals: vec![0.0_f64, 0.5],
        ..SolverResult::default()
    };
    (problem, result)
}

/// 受理が改善を伴わず best が current と異なる状態で終わる fixture (sweep idx=92)。
/// 結果は best-so-far (開始点から改善した点) で、末尾試行点ではない。
pub(super) fn fixture_best_not_current_moving() -> (QpProblem, SolverResult) {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 2], &[1e6_f64, 1e6], 2, 3).unwrap();
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0_f64, 0.001], 3, 3).unwrap();
    let problem = QpProblem::new(
        q,
        vec![5.0_f64, 5.0, 2.0],
        a,
        vec![1.0_f64, 0.5],
        vec![
            (0.0_f64, 10.0_f64),
            (f64::NEG_INFINITY, f64::INFINITY),
            (0.0_f64, f64::INFINITY),
        ],
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![2.0_f64, 0.5, 0.0],
        dual_solution: vec![-0.1_f64, 1.5],
        bound_duals: vec![3.0_f64, 0.0, 3.0],
        ..SolverResult::default()
    };
    (problem, result)
}

/// 行検索 20 回不受理 + 最終 KKT ガードが改善棄却 (PG 出力は開始点のまま)、
/// 直後の worst-active が dual/bound dual を実際に更新する fixture (sweep idx=381)。
pub(super) fn fixture_guarded_then_worst_active() -> (QpProblem, SolverResult) {
    let a = CscMatrix::from_triplets(&[0, 3, 3], &[0, 0, 1], &[7.0_f64, 1e6, 1e6], 4, 2).unwrap();
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0_f64, 4.0], 2, 2).unwrap();
    let problem = QpProblem::new(
        q,
        vec![-100000.0_f64, 2.0],
        a,
        vec![-3.0_f64, 0.5, -3.0, 0.5],
        vec![(-2.0_f64, 3.0_f64), (f64::NEG_INFINITY, f64::INFINITY)],
        vec![
            ConstraintType::Ge,
            ConstraintType::Eq,
            ConstraintType::Le,
            ConstraintType::Eq,
        ],
    )
    .unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.5_f64, 2.0],
        dual_solution: vec![-2.0_f64, 0.75, 1.5, 0.0],
        bound_duals: vec![0.0_f64, 3.0],
        ..SolverResult::default()
    };
    (problem, result)
}

fn assert_bits(actual: &[f64], expected: &[u64], field: &str) {
    let got: Vec<u64> = actual.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, expected, "{field} bit patterns drifted");
}

fn assert_sign_feasible(types: &[ConstraintType], y: &[f64]) {
    for (i, ct) in types.iter().enumerate() {
        match ct {
            ConstraintType::Le => assert!(y[i] >= 0.0, "row {i} (Le) dual {} < 0", y[i]),
            ConstraintType::Ge => assert!(y[i] <= 0.0, "row {i} (Ge) dual {} > 0", y[i]),
            ConstraintType::Eq => {}
        }
    }
}

fn kkt_rel(problem: &QpProblem, result: &SolverResult) -> f64 {
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &[],
    };
    crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    )
}

/// 行検索複数回不受理ケース: 受理判定・step 縮小・符号制約・workspace swap を
/// 通った最終 dual のビット列が base と一致すること。
#[test]
fn p1_pg_multi_reject_line_search_bitwise_golden() {
    let (problem, mut result) = fixture_multi_reject_moving();
    let pre = kkt_rel(&problem, &result);
    refine_dual_projected_gradient(&problem, &mut result, &[], None);
    let post = kkt_rel(&problem, &result);

    assert_bits(
        &result.dual_solution,
        &[
            13753958559052195057, // -7.599999999990631e-06
            13815242216922007349, // -0.10000000000380001
            0,                    // +0.0
            13815242216922281168, // -0.10000000000760001
        ],
        "dual_solution",
    );
    assert_bits(
        &result.bound_duals,
        &[0, 4602678819172646912],
        "bound_duals",
    );
    assert_sign_feasible(&problem.constraint_types, &result.dual_solution);
    assert!(
        post < pre,
        "guard accepted an update, so post ({post}) must beat pre ({pre})"
    );
}

/// best が current と異なる状態で終わるケース: 返却されるのは best-so-far
/// であり、末尾試行点 (current) ではないこと — バッファ取り違えならビットが
/// 変わる。base 実走由来の golden と一致する。
#[test]
fn p1_pg_best_not_current_bitwise_golden() {
    let (problem, mut result) = fixture_best_not_current_moving();
    let start_dual = result.dual_solution.clone();
    refine_dual_projected_gradient(&problem, &mut result, &[], None);

    assert_bits(
        &result.dual_solution,
        &[
            13757163507896795136, // -1.3000000000013001e-05
            0,                    // +0.0
        ],
        "dual_solution",
    );
    assert_bits(
        &result.bound_duals,
        &[4613937818241073152, 0, 4613937818241073152],
        "bound_duals",
    );
    assert_sign_feasible(&problem.constraint_types, &result.dual_solution);
    assert_ne!(
        result.dual_solution, start_dual,
        "fixture must move the dual"
    );
}

/// 20 回不受理の line search + 改善棄却ガード (PG 出力 = 開始点) と、続く
/// worst-active の dual/bound dual 更新のビット列が base と一致すること。
#[test]
fn p1_guarded_pg_then_worst_active_bitwise_golden() {
    let (problem, mut result) = fixture_guarded_then_worst_active();
    let start_dual = result.dual_solution.clone();

    refine_dual_projected_gradient(&problem, &mut result, &[], None);
    // KKT ガードが改善を棄却するため dual は開始点のまま (base と同一ビット)。
    assert_bits(
        &result.dual_solution,
        &[
            13835058055282163712, // -2.0
            4604930618986332160,  // 0.75
            4609434218613702656,  // 1.5
            0,                    // +0.0
        ],
        "dual_solution (after guarded PG)",
    );
    assert_eq!(result.dual_solution, start_dual);

    refine_dual_worst_active_block(&problem, &mut result, &[], None);
    assert_bits(
        &result.dual_solution,
        &[
            13835058055282163712, // -2.0
            4604930618986332160,  // 0.75
            4609434218613702656,  // 1.5
            4540198113714430976,  // 3.4377943148596046e-05
        ],
        "dual_solution (after worst-active)",
    );
    assert_bits(
        &result.bound_duals,
        &[0, 4681603773454599457], // [0.0, 99933.2441137028]
        "bound_duals (after worst-active)",
    );
    assert_sign_feasible(&problem.constraint_types, &result.dual_solution);
}
