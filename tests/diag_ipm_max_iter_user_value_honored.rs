//! `IpmOptions.max_iter` が全 attempt の cumulative budget として機能することを verify する sentinel。
//!
//! ## 修正前の挙動
//!
//! `attempt.rs` のループ内で `opts.ipm.max_iter = MAX_ITER_PER_ATTEMPT (500)` を
//! 毎 attempt 上書きしていたため、user が設定した `ipm.max_iter` は無視されていた。
//!
//! ## 修正後の挙動
//!
//! `ipm.max_iter` は全 attempt を通じた累積 iter の上限 (outer guard) として機能する。
//! per-attempt cap は `min(MAX_ITER_PER_ATTEMPT, remaining_user_budget)` で決定される。
//!
//! ## sentinel 検出力 (no-op proof)
//!
//! - Pattern A: `max_iter=50` → `result.iterations ≤ 50` (修正前は最大 500 だった)
//! - Pattern B: `max_iter=usize::MAX` (default) → 正常収束、退化しない
//! - Pattern C: `max_iter=1` → 1 iter で打ち切り、panic しない

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

/// Pattern A: max_iter=50 が1回 attempt の iter を 50 以内に制限する。
///
/// **Sentinel**: `attempt.rs` の outer guard を削除して毎 attempt 500 固定に戻すと
/// `result.iterations` が 50 を超えうる → このテストが FAIL する。
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

/// Pattern C: max_iter=1 でも panic せず、何らかのステータスで返る。
#[test]
fn max_iter_1_no_panic() {
    let problem = simple_convex_qp();
    let mut opts = SolverOptions::default();
    opts.ipm.max_iter = 1;
    let result = solve_qp_with(&problem, &opts);
    // 1 iter で収束は期待しないが、panic / 異常終了してはならない。
    assert!(
        result.iterations <= 1,
        "max_iter=1 が無視された: iterations={}",
        result.iterations
    );
}
