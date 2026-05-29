//! Sentinel: `IpmOptions.max_iter` is respected as the cumulative iteration budget
//! across all attempts. Pattern C is the load-bearing sentinel.

use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::{solve_qp_with, QpProblem};
use otspot::CscMatrix;

fn simple_convex_qp() -> QpProblem {
    // min x0^2 + x1^2  s.t. x0 + x1 = 1, x0,x1 ∈ [0, 100]
    // Optimal: x0=x1=0.5, obj=0.25
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0_f64, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0], 1, 2).unwrap();
    QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![1.0],
        vec![(0.0, 100.0), (0.0, 100.0)],
        vec![otspot::problem::ConstraintType::Eq],
    )
    .unwrap()
}

/// Pattern A: max_iter=50 は per-attempt cap として機能し、iterations は 50 以内に収まる。
///
/// Note: simple_convex_qp (n=2) は自然収束 ~10 iter なので修正 revert 時も PASS する。
/// load-bearing sentinel は Pattern C (max_iter=1) を参照。
#[test]
fn max_iter_50_honored() {
    let problem = simple_convex_qp();
    let mut opts = SolverOptions::default();
    opts.ipm.max_iter = 50;
    let result = solve_qp_with(&problem, &opts);
    assert!(
        result.iterations <= 50,
        "max_iter=50 が無視された: iterations={} (上限 50 超)",
        result.iterations
    );
}

/// Pattern B: デフォルト (max_iter=usize::MAX) では正常収束する。
#[test]
fn max_iter_default_converges() {
    let problem = simple_convex_qp();
    let result = solve_qp_with(&problem, &SolverOptions::default());
    assert_eq!(result.status, SolveStatus::Optimal, "default opts で Optimal 必須");
    // 1/2 x'Qx + c'x で Q=[2,2], x*=[0.5,0.5] → 1/2*(2*0.25+2*0.25) = 0.5
    assert!((result.objective - 0.5).abs() < 1e-5, "obj={}", result.objective);
}

/// Pattern C: max_iter=1 でも panic せず、iterations が 1 以内に収まる。
///
/// **Sentinel**: outer guard を削除して `opts.ipm.max_iter = 500` を毎 attempt に
/// 固定（修正 revert）すると、simple_convex_qp は ~10 iter で収束するため
/// `result.iterations = 10 > 1` となりこのテストが FAIL する。
#[test]
fn max_iter_1_no_panic() {
    let problem = simple_convex_qp();
    let mut opts = SolverOptions::default();
    opts.ipm.max_iter = 1;
    let result = solve_qp_with(&problem, &opts);
    assert!(
        result.iterations <= 1,
        "max_iter=1 が無視された: iterations={}",
        result.iterations
    );
}
