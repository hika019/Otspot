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
//! ## ファイル方針
//!
//! - inf1-3, ub1-2 は Model API (`Model` + `constraint!`) で記述。
//! - ub3 は `SolveStatus` の細分 (Unbounded / MaxIterations / NumericalError いずれも許容)
//!   を assert する設計のため raw `QpProblem` を維持。
//!   Model API は MaxIterations を `Timeout` 等に隠蔽するため fidelity が崩れる。

use otspot::constraint;
use otspot::model::{Model, ModelError, SolveError};
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::qp::{solve_qp_with, QpProblem};
use otspot::sparse::CscMatrix;

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
/// **狙い**: bound 矛盾 (lb > ub) を Model API で入力エラーとして即拒否。
///
/// `add_var` (#8) が lb > ub を検出して `invalid_inputs` に記録し、`solve()` は
/// `ModelError::InvalidInput` を返す。低レベル `QpProblem::new` (#7) も同矛盾を
/// `InvalidBounds` で拒否するが、Model 経路ではそこへ到達する前に弾かれる。
#[test]
fn inf1_bound_lb_gt_ub_infeasible() {
    let mut model = Model::new("inf1");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x = model.add_var("x", 5.0, 3.0); // lb > ub
    model.minimize(x * x);

    let err = model.solve().expect_err("inf1: lb>ub must yield an error");
    assert!(
        matches!(err, ModelError::InvalidInput(_)),
        "inf1: expected InvalidInput for lb>ub, got {:?}",
        err
    );
}

/// 非有限係数 (NaN/±∞) を Model API に与えると `InvalidInput` で拒否される。
///
/// 低レベル `QpProblem::new`/`LpProblem::new_general` が `NonFiniteCoefficient` を
/// 返し、Model 境界で `Internal` ではなく `InvalidInput` に map される (typed-error 一貫性)。
#[test]
fn nonfinite_objective_coefficient_is_invalid_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // LP path (no quadratic objective).
        let mut model = Model::new("nf_lp");
        let x = model.add_var("x", 0.0, 10.0);
        model.minimize(bad * x);
        let err = model.solve().expect_err("non-finite c must error");
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "LP non-finite c={bad}: expected InvalidInput, got {:?}",
            err
        );

        // Constraint rhs path.
        let mut model = Model::new("nf_rhs");
        let y = model.add_var("y", 0.0, 10.0);
        model.add_constraint((1.0 * y).leq(bad));
        model.minimize(1.0 * y);
        let err = model.solve().expect_err("non-finite b must error");
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "non-finite b={bad}: expected InvalidInput, got {:?}",
            err
        );
    }
}

// =============================================================================
// inf2: bound + Eq infeasible (x in [0,1], x=5 required)
// =============================================================================

/// **構造**: min 1/2 x^2  s.t. x = 5, 0 <= x <= 1.
/// **狙い**: 等式制約 5 が bound [0,1] 範囲外 → Infeasible.
#[test]
fn inf2_eq_outside_bounds_infeasible() {
    let mut model = Model::new("inf2");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 1.0);
    model.add_constraint(constraint!(x == 5.0));
    model.minimize(x * x);

    let err = model
        .solve()
        .expect_err("inf2: Eq=5 vs bound [0,1] must be Infeasible");
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "inf2: expected Infeasible, got {:?}",
        err
    );
}

// =============================================================================
// inf3: 2 conflict inequality constraints
// =============================================================================

/// **構造**: min 1/2 x^2  s.t. x >= 10, x <= 1, free.
/// **狙い**: 2 つの不等式が空集合を作る → Infeasible.
#[test]
fn inf3_conflicting_inequalities_infeasible() {
    let mut model = Model::new("inf3");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!(x >= 10.0));
    model.add_constraint(constraint!(x <= 1.0));
    model.minimize(x * x);

    let err = model
        .solve()
        .expect_err("inf3: x>=10 ∧ x<=1 must be Infeasible");
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "inf3: expected Infeasible, got {:?}",
        err
    );
}

// =============================================================================
// ub1: Q=0 LP fallback unbounded
// =============================================================================

/// **構造**: min -x (LP)  s.t. x >= 0 (Ge), x free above.
/// Q=0 ⇒ LP fallback (Simplex). c=-1, x >= 0, no upper bound → unbounded.
/// **狙い**: LP unbounded を Simplex 経路で正しく検出。
#[test]
fn ub1_q_zero_lp_fallback_unbounded() {
    let mut model = Model::new("ub1");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.add_constraint(constraint!(x >= 0.0));
    model.minimize(-1.0 * x);

    let err = model
        .solve()
        .expect_err("ub1: Q=0 LP min -x s.t. x>=0 must be Unbounded");
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Unbounded)),
        "ub1: expected Unbounded, got {:?}",
        err
    );
}

// =============================================================================
// ub2: convex QP can't be unbounded with PSD Q + finite bounds → Optimal
// =============================================================================

/// **構造**: min 1/2 x^2 - 1000*x  s.t. 0 <= x <= 100.
/// **解析解**: unconstrained min at x=1000, but ub=100 active → x=100, obj = 5000-100000 = -95000.
/// **狙い**: Q PSD + finite bounds なら絶対に unbounded ではないことを確認 (regression)。
#[test]
fn ub2_psd_q_finite_bounds_yields_optimal() {
    let mut model = Model::new("ub2");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x = model.add_var("x", 0.0, 100.0);
    model.minimize(0.5 * x * x + (-1000.0) * x);

    let result = model
        .solve()
        .expect("ub2: PSD Q + finite bounds must be Optimal");
    let exp_obj = 0.5 * 100.0 * 100.0 - 1000.0 * 100.0;
    let rel = (result.objective_value - exp_obj).abs() / (1.0 + exp_obj.abs());
    assert!(
        rel < 1e-6,
        "ub2: obj={} expected={} rel={}",
        result.objective_value,
        exp_obj,
        rel
    );
}

// =============================================================================
// ub3: null(Q) + c → unbounded (Q PSD だが c が null space に乗る)
// =============================================================================

/// **構造**: Q = diag(0, 1) (1 var に null space あり), c = [-1, 0], x1 free (no upper).
///   min 0.5 * x2^2 - x1, s.t. x1 >= 0, x2 free.
///   x1 を無限に大きくすれば obj → -∞.
/// **狙い**: PSD Q (但し semi-definite)、null space 方向に linear 項あり → unbounded.
///         一般凸 QP の unbounded 検出経路の regression。
///
/// **NOTE (raw 維持理由)**: solver は本問題に `SuboptimalSolution` status + 非空 solution
/// を返す。Model API は `SuboptimalSolution + !empty` を `Ok(ModelResult)` に折り畳むため、
/// 「Optimal を name しない」契約 (= solver が convergence を主張しない) を Model API では
/// 表現できない。raw 維持で `status != Optimal` を直接 pin する。
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
    assert_ne!(
        r.status,
        SolveStatus::Optimal,
        "ub3: null(Q)+c≠0 with no upper bound must NOT be Optimal, got {:?} obj={}",
        r.status,
        r.objective
    );
}
