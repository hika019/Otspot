//! Model API 拡張 単体テスト:
//! - `ModelResult.bound_duals` 露出
//! - `Model::set_obj_offset(f64)`
//! - `ModelError::SolveError::{MaxIterations, NumericalError}` 露出
//! - `Model::set_diagonal_q(&[f64])` ergonomic helper

use otspot::model::{Model, ModelError, SolutionProof, SolveError};
use otspot::SolveStatus;

const TOL: f64 = 1e-6;
const QP_TOL: f64 = 2e-3;
const INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

// ---------------------------------------------------------------------------
// bound_duals: QP path で active 境界に対し dual が報告される
// ---------------------------------------------------------------------------
#[test]
fn model_api_exposes_bound_duals_qp() {
    // min x^2 + y^2  s.t. x,y in [0,1], no row constraints
    // 解析解: x=y=0, lb 活性 → lb_dual > 0 (c の負勾配を相殺するためゼロ強度の可能性あり)
    // c=0 のため bound_dual=0 だが、長さ (n_lb+n_ub=4) は配線できていることを検証する。
    let mut model = Model::new("bound_duals_check");
    let _x = model.add_var("x", 0.0, 1.0);
    let _y = model.add_var("y", 0.0, 1.0);
    model.set_diagonal_q(&[2.0, 2.0]);
    model.minimize(0.0); // c=0
    let result = model.solve().expect("solve must succeed");

    // QP 経路では bound_duals が SolverResult から伝播される。長さ = n_lb + n_ub。
    // 両変数の lb/ub 共に有限 → 期待長 = 4。
    assert_eq!(
        result.bound_duals.len(),
        4,
        "bound_duals length expected 4 (2 lb + 2 ub), got {}",
        result.bound_duals.len()
    );
}

#[test]
fn model_result_exposes_status_and_global_proof_for_optimal_lp() {
    let mut model = Model::new("status_proof_lp");
    let x = model.add_var("x", 0.0, INF);
    model.add_constraint(otspot::model::constraint!(x >= 1.0));
    model.minimize(x);

    let result = model.solve().expect("solve must succeed");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.proof, SolutionProof::GlobalOptimal);
    assert!(result.has_global_optimality_proof());
}

#[test]
fn model_result_exposes_status_and_global_proof_for_optimal_qp() {
    let mut model = Model::new("status_proof_qp");
    let x = model.add_var("x", 0.0, INF);
    model.set_diagonal_q(&[2.0]);
    model.minimize(-2.0 * x);

    let result = model.solve().expect("solve must succeed");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.proof, SolutionProof::GlobalOptimal);
    assert!(result.has_global_optimality_proof());
}

// ---------------------------------------------------------------------------
// bound_duals: 非ゼロ Q + active lb で lb_dual > 0
// ---------------------------------------------------------------------------
#[test]
fn model_api_bound_duals_active_lb_nonzero() {
    // min 1/2 x^2 + x  s.t. x in [0, inf)
    // 無制約 min は x=-1, lb 活性 → x=0, obj=0
    // KKT: grad = x + 1 = 1 at x=0, lb_dual = 1.0
    // Q=[1] 非ゼロなので solve_as_lp dispatch を避け IPM 経路で bound_dual を計算する。
    let mut model = Model::new("bound_dual_lb_active");
    let x = model.add_var("x", 0.0, INF);
    model.set_diagonal_q(&[1.0]);
    model.minimize(1.0 * x); // c=[1.0]
    let result = model.solve().expect("solve must succeed");

    assert!(
        result[x].abs() < QP_TOL,
        "x expected 0.0, got {}",
        result[x]
    );
    assert_eq!(
        result.bound_duals.len(),
        1,
        "bound_duals length expected 1 (1 lb only), got {}",
        result.bound_duals.len()
    );
    assert!(
        result.bound_duals[0] > 0.5,
        "lb_dual expected ≈1.0 (>0.5), got {}",
        result.bound_duals[0]
    );
}

// ---------------------------------------------------------------------------
// set_obj_offset: LP 経路で offset が最終 objective に加算される
// ---------------------------------------------------------------------------
#[test]
fn model_api_set_obj_offset_lp() {
    // min x  s.t. x >= 1, x in [0, inf)
    // offset=10.0 → obj = 1 + 10 = 11
    let mut model = Model::new("lp_obj_offset");
    let x = model.add_var("x", 0.0, INF);
    model.add_constraint(otspot::model::constraint!(x >= 1.0));
    model.minimize(x);
    model.set_obj_offset(10.0);

    let result = model.solve().unwrap();
    assert!(
        (result.objective_value - 11.0).abs() < TOL,
        "offset must add to obj: got {}",
        result.objective_value
    );
}

// ---------------------------------------------------------------------------
// set_obj_offset: QP 経路でも offset が反映される
// ---------------------------------------------------------------------------
#[test]
fn model_api_set_obj_offset_qp() {
    // min x^2 + y^2  s.t. x + y == 1, x,y free
    // 解: x=y=0.5, obj=0.5. offset=-2.0 → -1.5
    let mut model = Model::new("qp_obj_offset");
    let x = model.add_var("x", NEG_INF, INF);
    let y = model.add_var("y", NEG_INF, INF);
    model.add_constraint((x + y).eq_constraint(1.0));
    model.set_diagonal_q(&[2.0, 2.0]);
    model.minimize(0.0 * x + 0.0 * y);
    model.set_obj_offset(-2.0);

    let result = model.solve().unwrap();
    assert!(
        (result.objective_value - (-1.5)).abs() < QP_TOL,
        "qp offset must add to obj: expected -1.5, got {}",
        result.objective_value
    );
}

// ---------------------------------------------------------------------------
// set_obj_offset: maximize でも offset が後段加算される (符号変換に巻き込まれない)
// ---------------------------------------------------------------------------
#[test]
fn model_api_set_obj_offset_maximize() {
    // max x  s.t. x <= 7, x >= 0; offset=3.0 → 7 + 3 = 10
    let mut model = Model::new("max_obj_offset");
    let x = model.add_var("x", 0.0, INF);
    model.add_constraint(otspot::model::constraint!(x <= 7.0));
    model.maximize(x);
    model.set_obj_offset(3.0);

    let result = model.solve().unwrap();
    assert!(
        (result.objective_value - 10.0).abs() < TOL,
        "max + offset: expected 10.0, got {}",
        result.objective_value
    );
}

// ---------------------------------------------------------------------------
// set_diagonal_q: 対角 Q ergonomic helper の正しさ
// ---------------------------------------------------------------------------
#[test]
fn model_api_set_diagonal_q() {
    // min x^2 + y^2  s.t. x + y == 1, x,y free
    // 解: x=y=0.5, obj=0.5
    let mut model = Model::new("diag_q_helper");
    let x = model.add_var("x", NEG_INF, INF);
    let y = model.add_var("y", NEG_INF, INF);
    model.set_diagonal_q(&[2.0, 2.0]);
    model.add_constraint((x + y).eq_constraint(1.0));
    model.minimize(0.0 * x + 0.0 * y);

    let result = model.solve().unwrap();
    assert!(
        (result[x] - 0.5).abs() < QP_TOL,
        "x expected 0.5, got {}",
        result[x]
    );
    assert!(
        (result[y] - 0.5).abs() < QP_TOL,
        "y expected 0.5, got {}",
        result[y]
    );
    assert!(
        (result.objective_value - 0.5).abs() < QP_TOL,
        "obj expected 0.5, got {}",
        result.objective_value
    );
}

#[test]
fn model_api_try_set_diagonal_q_dim_mismatch_returns_error() {
    let mut model = Model::new("diag_q_bad");
    let _x = model.add_var("x", 0.0, 1.0);
    let err = match model.try_set_diagonal_q(&[1.0, 1.0]) {
        Ok(_) => panic!("expected dim mismatch to fail"),
        Err(err) => err,
    };
    assert!(
        matches!(err, ModelError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn model_api_set_diagonal_q_dim_mismatch_reports_error_at_solve() {
    let mut model = Model::new("diag_q_bad_compat");
    let x = model.add_var("x", 0.0, 1.0);
    model.set_diagonal_q(&[1.0, 1.0]);
    model.minimize(x);

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn model_api_set_diagonal_q_can_recover_after_dim_mismatch() {
    let mut model = Model::new("diag_q_recover");
    let x = model.add_var("x", 0.0, 1.0);
    model.set_diagonal_q(&[1.0, 1.0]);
    model.set_diagonal_q(&[1.0]);
    model.minimize(0.0 * x);

    let result = model
        .solve()
        .expect("valid replacement should clear stale input error");
    assert!(
        result[x].abs() < QP_TOL,
        "x expected 0.0, got {}",
        result[x]
    );
}

#[test]
fn model_api_try_set_timeout_rejects_nan_and_negative() {
    let mut model = Model::new("timeout_bad");
    for timeout in [f64::NAN, -1.0] {
        let err = match model.try_set_timeout(timeout) {
            Ok(_) => panic!("expected invalid timeout to fail"),
            Err(err) => err,
        };
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "expected InvalidInput for {timeout:?}, got {err:?}"
        );
    }
}

#[test]
fn model_api_set_timeout_invalid_reports_error_at_solve() {
    let mut model = Model::new("timeout_bad_compat");
    let x = model.add_var("x", 0.0, 1.0);
    model.set_timeout(f64::NAN);
    model.minimize(x);

    let err = model.solve().unwrap_err();
    assert!(
        matches!(err, ModelError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn model_api_set_timeout_can_recover_after_invalid_value() {
    let mut model = Model::new("timeout_recover");
    let x = model.add_var("x", 0.0, 1.0);
    model.set_timeout(f64::NAN);
    model.set_timeout(1.0);
    model.minimize(x);

    let result = model
        .solve()
        .expect("valid replacement should clear stale input error");
    assert!(result[x].abs() < TOL, "x expected 0.0, got {}", result[x]);
}

// ---------------------------------------------------------------------------
// SolveError variants: MaxIterations / NumericalError が露出されている
// ---------------------------------------------------------------------------
#[test]
fn solve_error_variants_exposed() {
    // コンパイル時 + matches! で variant の存在を確認 (新 variant 追加の regression guard)
    let e_max = ModelError::SolveError(SolveError::MaxIterations);
    let e_num = ModelError::SolveError(SolveError::NumericalError);
    assert!(matches!(
        e_max,
        ModelError::SolveError(SolveError::MaxIterations)
    ));
    assert!(matches!(
        e_num,
        ModelError::SolveError(SolveError::NumericalError)
    ));
    // Display 実装も併せて確認
    assert!(format!("{}", SolveError::MaxIterations).contains("iterations"));
    assert!(format!("{}", SolveError::NumericalError).contains("Numerical"));
}
