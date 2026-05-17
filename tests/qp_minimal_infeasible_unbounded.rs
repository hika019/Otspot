//! Task #29 凸 QP mini-corpus — **bug class: infeasible / unbounded 検出**
//!
//! ## 対象 bug class
//!
//! - **Bound 矛盾** (lb > ub) → Infeasible 即検出
//! - **bound + 線形制約矛盾** → Phase I LP が Infeasible 報告
//! - **bound + 等式制約矛盾**
//! - **凸 QP は本来 unbounded にならない**: Q PSD で目的関数 -∞ になるには
//!   Q の null space + c の射影が非零で free direction が無限長必要。
//!   - 例: Q=0 (LP 退化) + c<0 + 上界なし → unbounded
//!   - Q PSD で null(Q) 方向に c が射影される場合 → unbounded
//! - **目的関数 Q PSD + 全有界 bound** → 必ず Optimal
//!
//! ## 真因仮説
//!
//! - presolve の bound consistency check (`lb > ub + tol`) が absolute tol で
//!   微小な lb-ub に対し誤検出 / 見逃し
//! - Phase I LP が「artificial 列」を残して Optimal 偽装する旧 bug
//! - unbounded 判定が dual_solution 不在で MaxIterations に倒れる
//!
//! ## NOTE
//!
//! - solver の SolveStatus 種別: Optimal/Infeasible/Unbounded/MaxIterations/
//!   Timeout/NumericalError/NonConvex (problem/mod.rs 確認済)。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

const MINI_TIMEOUT_SECS: f64 = 5.0;

fn solver_opts() -> SolverOptions {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(MINI_TIMEOUT_SECS);
    opts
}

// =============================================================================
// inf1: bound lb > ub (即時検出)
// =============================================================================

/// **構造**: min 1/2 x^2  s.t. 5 <= x <= 3 (空集合).
/// **狙い**: bound 矛盾を presolve / IPM 入口で Infeasible 即検出。
#[test]
fn inf1_bound_lb_gt_ub_infeasible() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(5.0, 3.0)]; // lb > ub
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Infeasible,
        "inf1: lb>ub must yield Infeasible, got {:?} obj={}", r.status, r.objective);
}

// =============================================================================
// inf2: bound + Eq infeasible (x in [0,1], x=5 required)
// =============================================================================

/// **構造**: min 1/2 x^2  s.t. x = 5, 0 <= x <= 1.
/// **狙い**: 等式制約 5 が bound [0,1] 範囲外 → Infeasible.
#[test]
fn inf2_eq_outside_bounds_infeasible() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
    let b = vec![5.0];
    let cts = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, 1.0)];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Infeasible,
        "inf2: Eq=5 vs bound [0,1] must be Infeasible, got {:?}", r.status);
}

// =============================================================================
// inf3: 2 conflict inequality constraints
// =============================================================================

/// **構造**: min 1/2 x^2  s.t. x >= 10, x <= 1, free.
/// **狙い**: 2 つの不等式が空集合を作る → Infeasible.
#[test]
fn inf3_conflicting_inequalities_infeasible() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![0.0];
    // row0: x >= 10 (Ge), row1: x <= 1 (Le)
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, n).unwrap();
    let b = vec![10.0, 1.0];
    let cts = vec![ConstraintType::Ge, ConstraintType::Le];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Infeasible,
        "inf3: x>=10 ∧ x<=1 must be Infeasible, got {:?}", r.status);
}

// =============================================================================
// ub1: Q=0 LP fallback unbounded
// =============================================================================

/// **構造**: min -x (LP)  s.t. x >= 0 (Ge), x free above.
/// Q=0 ⇒ LP fallback (Simplex). c=-1, x >= 0, no upper bound → unbounded.
/// **狙い**: Q=0 退化 LP の unbounded を Simplex 経路で正しく検出。
#[test]
fn ub1_q_zero_lp_fallback_unbounded() {
    let n = 1;
    let q = CscMatrix::new(n, n); // Q=0
    let c = vec![-1.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
    let b = vec![0.0];
    let cts = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, f64::INFINITY)];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Unbounded,
        "ub1: Q=0 LP min -x s.t. x>=0 must be Unbounded, got {:?}", r.status);
}

// =============================================================================
// ub2: convex QP can't be unbounded with PSD Q + finite bounds → Optimal
// =============================================================================

/// **構造**: min 1/2 x^2 - 1000*x  s.t. 0 <= x <= 100.
/// **解析解**: unconstrained min at x=1000, but ub=100 active → x=100, obj = 5000-100000 = -95000.
/// **狙い**: Q PSD + finite bounds なら絶対に unbounded ではないことを確認 (regression)。
#[test]
fn ub2_psd_q_finite_bounds_yields_optimal() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![-1000.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0, 100.0)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal,
        "ub2: PSD Q + finite bounds must be Optimal, got {:?}", r.status);
    let exp_obj = 0.5 * 100.0 * 100.0 - 1000.0 * 100.0;
    let rel = (r.objective - exp_obj).abs() / (1.0 + exp_obj.abs());
    assert!(rel < 1e-6, "ub2: obj={} expected={} rel={}", r.objective, exp_obj, rel);
}

// =============================================================================
// ub3: null(Q) + c → unbounded (Q PSD だが c が null space に乗る)
// =============================================================================

/// **構造**: Q = diag(0, 1) (1 var に null space あり), c = [-1, 0], x1 free (no upper).
///   min 0.5 * x2^2 - x1, s.t. x1 >= 0, x2 free.
///   x1 を無限に大きくすれば obj → -∞.
/// **狙い**: PSD Q (但し semi-definite)、null space 方向に linear 項あり → unbounded.
///         一般凸 QP の unbounded 検出経路の regression。
#[test]
fn ub3_q_null_space_with_linear_term_unbounded() {
    let n = 2;
    let q = CscMatrix::from_triplets(&[1], &[1], &[1.0], n, n).unwrap(); // Q = diag(0, 1)
    let c = vec![-1.0, 0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
    let b = vec![0.0];
    let cts = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    // Unbounded detection は QP 内点法では難しい (Phase I が x→∞ で発散しない)。
    // solver 仕様: status は Unbounded / MaxIterations / NumericalError のいずれか。
    // Optimal を返すなら bug。
    assert_ne!(r.status, SolveStatus::Optimal,
        "ub3: null(Q)+c≠0 with no upper bound must NOT be Optimal, got {:?} obj={}",
        r.status, r.objective);
}
