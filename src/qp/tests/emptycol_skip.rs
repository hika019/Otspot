//! #92 真因 sentinel: EmptyCol skip 厳格化 (linear-only var を stationarity から外さない)。
//!
//! 旧 heuristic `A.col_ptr[j+1] - col_ptr[j] == 0` は presolve 用 (bd=0 慣例) だが、
//! **非凸 QP の linear-only var** (A 空 / Q 列 non-zero / c≠0) に誤発火し、stationarity が
//! 常に 0 と評価され outer guard が効かず bd が往復消滅していた (#55 audit)。
//!
//! 本 file は 4 パターン (CLAUDE.md「複数パターンのデータを用意」) で
//! 修正が真因に効くこと + 既存挙動を退化させないことを検証する。

use super::super::*;
use crate::problem::{ConstraintType, SolveStatus};
use crate::sparse::CscMatrix;

/// Pattern A (#55 audit fixture): A 空 + Q diag=(0, -2)、c=(1, 3)、box [-2, 2]^2。
/// 期待: x=(-2, -2)、bd=[1, 7, 0, 0]、status=LocallyOptimal、KKT≈0。
///
/// 旧 logic では x[0] (linear-only: A col 空 / Q col 空 / c=1) と x[1] (linear-only:
/// A col 空 / Q col non-zero / c=3) が両方 skip され bd 復元が稼働せず KKT=0.5。
#[test]
fn test_emptycol_skip_strict_linear_only_x0_and_nonconvex_x1() {
    let q = CscMatrix::from_triplets(&[1], &[1], &[-2.0_f64], 2, 2).unwrap();
    let c = vec![1.0_f64, 3.0_f64];
    let a = CscMatrix::new(0, 2);
    let b: Vec<f64> = vec![];
    let bounds = vec![(-2.0_f64, 2.0_f64), (-2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);

    assert!(
        matches!(
            result.status,
            SolveStatus::LocallyOptimal | SolveStatus::Optimal
        ),
        "expected LocallyOptimal/Optimal, got {:?}",
        result.status
    );
    // 期待値: x[0]=lb=-2 (c>0 → 下端最小), x[1]=lb=-2 (-2x² で発散方向、box 内最小は端点)
    assert!((result.solution[0] - (-2.0)).abs() < 1e-5, "x[0]: {}", result.solution[0]);
    assert!((result.solution[1] - (-2.0)).abs() < 1e-5, "x[1]: {}", result.solution[1]);

    // 元空間 KKT 残差: r[j] = (Qx)[j] + c[j] + (A^Ty)[j] + bound_contrib[j]
    // x[0]: 0 + 1 + 0 + bc[0]=0  → bd_lb[0] = 1 で r=0
    // x[1]: -2*(-2) + 3 + 0 + bc[1]=0  → 7 + bc[1] = 0 → bd_lb[1] = 7
    let qx = problem.q.mat_vec_mul(&result.solution).unwrap();
    let bd = &result.bound_duals;
    // bound_duals layout: lb 有限 2 + ub 有限 2 = 4 個
    assert_eq!(bd.len(), 4, "expected bd len=4, got {}", bd.len());
    let bc0 = -bd[0] + bd[2];
    let bc1 = -bd[1] + bd[3];
    let r0 = (qx[0] + problem.c[0] + bc0).abs();
    let r1 = (qx[1] + problem.c[1] + bc1).abs();
    assert!(r0 < 1e-4, "x[0] stationarity must be ~0, got {:.3e}", r0);
    assert!(r1 < 1e-4, "x[1] stationarity must be ~0, got {:.3e}", r1);
}

/// Pattern B: 線形 LP-like (Q=0)、c=(1, 0)、bound 0 ≤ x ≤ 1。
/// 期待: x=(0, 0/任意)、status=Optimal、KKT=0。
///
/// presolve は両 col とも eliminate (A 空 + Q 空) するため orig 空間の stationarity は
/// 厳密に 0 が要求される。EmptyCol mask が正しく適用されることを確認。
#[test]
fn test_emptycol_skip_strict_lp_like_full_elimination() {
    let q = CscMatrix::new(2, 2);
    let c = vec![1.0_f64, 0.0_f64];
    let a = CscMatrix::new(0, 2);
    let b: Vec<f64> = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    // x[0]: c=1>0 → minimise x → x=0
    assert!(result.solution[0].abs() < 1e-6, "x[0]: {}", result.solution[0]);
}

/// Pattern C: presolve で真に empty (A 空 + Q 空 + c=0) → solver が走り、解は任意の box 値で OK。
/// 修正後も skip が機能して spurious refine action が出ないことを確認する control。
#[test]
fn test_emptycol_skip_strict_truly_empty_col_preserved() {
    let q = CscMatrix::new(2, 2);
    let c = vec![0.0_f64, 0.0_f64];
    // A は 1 行 1 列 (x[1] のみ非ゼロ)、x[0] は完全に empty col。
    let a = CscMatrix::from_triplets(&[0], &[1], &[1.0_f64], 1, 2).unwrap();
    let b = vec![0.5_f64];
    let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    // x[1] = 0.5 (= b/A[0,1])
    assert!((result.solution[1] - 0.5).abs() < 1e-6, "x[1]: {}", result.solution[1]);
    // x[0] は任意 (c=0 / Q=0 / A=0)、status が Optimal で十分。
}

/// Pattern D (control): #55 fixture に Le 制約を 1 本足すと linear-only var でなくなり、
/// 旧経路でも bd[0]=1 が復元できる挙動を維持していることを確認 (退化テスト)。
#[test]
fn test_emptycol_skip_strict_with_active_constraint_unchanged() {
    let q = CscMatrix::from_triplets(&[1], &[1], &[-2.0_f64], 2, 2).unwrap();
    let c = vec![1.0_f64, 3.0_f64];
    // A: row 0 = [1, 0] (x[0] を実際に拘束)
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 2).unwrap();
    let b = vec![5.0_f64]; // Le → x[0] ≤ 5、bound 内では effective 制約なし
    let bounds = vec![(-2.0_f64, 2.0_f64), (-2.0_f64, 2.0_f64)];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);

    assert!(
        matches!(
            result.status,
            SolveStatus::LocallyOptimal | SolveStatus::Optimal
        ),
        "got {:?}",
        result.status
    );
    // 解は #55 と同じく x=(-2, -2)、bd[0]=1。
    assert!((result.solution[0] - (-2.0)).abs() < 1e-5);
    assert!((result.solution[1] - (-2.0)).abs() < 1e-5);
    let bd = &result.bound_duals;
    assert_eq!(bd.len(), 4);
    // bd_lb[0] ≈ 1 (linear stationarity from c[0]=1)
    assert!(bd[0] > 0.5, "bd_lb[0] should be ≈1, got {}", bd[0]);
}
