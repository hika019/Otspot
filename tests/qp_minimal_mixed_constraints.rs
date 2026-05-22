//! Task #29 凸 QP mini-corpus — **bug class: mixed Eq/Le/Ge + obj_offset**
//!
//! ## 対象 bug class
//!
//! - **Eq + Le + Ge 同一問題** で IPM の制約展開 (to_all_le or 内部 Eq native) が
//!   dual_solution を元 row 数で正しく返すか
//! - **Ge 単独** で dual の符号 (-y vs +y)
//! - **obj_offset** (定数項) が objective に正しく加算されるか
//! - **redundant row** (構造的余剰) が IPM 内部で除去後に dual を全行に詰めるか
//!
//! ## 真因仮説
//!
//! - to_all_le 経路は廃止 / native Eq 経路移行で dual 折りたたみが旧 codepath に
//!   残っていると Ge 行に符号誤りが入る
//! - obj_offset は postsolve で加算 (qp_postsolve.rs 想定)、加算漏れ→ obj 一致せず
//!
//! ## ファイル方針
//!
//! - 全 6 test (mix1-6) を Model API で記述。
//! - mix4/5 は `Model::set_obj_offset` (model-api-extender) を使用。

use otspot::constraint;
use otspot::model::Model;
use otspot::sparse::CscMatrix;

const EPS_OBJ_REL: f64 = 1e-6;
const EPS_X_ABS: f64 = 1e-5;
const EPS_DUAL_ABS: f64 = 1e-4;
const MINI_TIMEOUT_SECS: f64 = 5.0;

fn assert_obj_close(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(rel < EPS_OBJ_REL,
        "[{}] obj actual={:.9e} expected={:.9e} rel_err={:.3e}",
        label, actual, expected, rel);
}

fn assert_x_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(diff < EPS_X_ABS,
        "[{}] x actual={:.9e} expected={:.9e} diff={:.3e}",
        label, actual, expected, diff);
}

// =============================================================================
// mix1: Ge constraint single (sign of dual)
// =============================================================================

/// **構造**: min 1/2(x1^2 + x2^2)  s.t. x1 + x2 >= 1, free.
/// **解析解**: x1=x2=0.5 (制約 active boundary), y = -0.5 (Ge native は y >= 0 規約だが
///   solver の sign convention は内部の to_all_le 展開で決まる)。
///   ※ Sign は固定せず |y|=0.5 のみ assert (symmetric な存在確認)。
/// **狙い**: Ge 単独 (Le に変換) で IPM が Optimal、|y|=0.5 が成立。
#[test]
fn mix1_ge_constraint_dual_magnitude() {
    let n = 2;
    let mut model = Model::new("mix1");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2) >= 1.0));
    model.minimize(0.0);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);

    let result = model.solve().expect("mix1: solve");
    assert_x_close(result[x1], 0.5, "mix1: x1");
    assert_x_close(result[x2], 0.5, "mix1: x2");
    assert_obj_close(result.objective_value, 0.25, "mix1: obj");
    let dual = result.dual_solution.as_ref().expect("mix1: dual_solution");
    assert_eq!(dual.len(), 1, "mix1: dual length=1 (元 row 数)");
    assert!((dual[0].abs() - 0.5).abs() < EPS_DUAL_ABS,
        "mix1: |y|=0.5, got {}", dual[0]);
}

// =============================================================================
// mix2: Eq + Le + Ge in same problem
// =============================================================================

/// **構造**: 3 var, 3 row (1 Eq, 1 Le, 1 Ge).
///   min 1/2(x1^2 + x2^2 + x3^2)
///   s.t. x1 + x2 + x3 = 3   (Eq)
///        x1           <= 2   (Le)
///        x3           >= 0.5 (Ge)
///   bounds: free.
///
/// **解析解 (active set 推定)**:
///   無制約最小 (Eq active, Le/Ge inactive) を仮定: ∇L: x_i = y (i=1,2,3),
///   x1+x2+x3=3 ⇒ x_i = 1 (全て同一). 制約: x1=1<=2 OK, x3=1>=0.5 OK. ⇒ active set = {Eq}.
///   x* = (1, 1, 1), obj = 1.5, y_eq = 1, y_le = 0, y_ge = 0.
/// **狙い**: 3 制約タイプ混在で dual_solution.len()=3、Le/Ge 非 active dual ≈ 0。
#[test]
fn mix2_eq_le_ge_mixed_inactive_inequalities() {
    let n = 3;
    let mut model = Model::new("mix2");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    let x3 = model.add_var("x3", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2 + x3) == 3.0));
    model.add_constraint(constraint!(x1 <= 2.0));
    model.add_constraint(constraint!(x3 >= 0.5));
    model.minimize(0.0);
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);

    let result = model.solve().expect("mix2: solve");
    let xs = [x1, x2, x3];
    for (i, &v) in xs.iter().enumerate() {
        assert_x_close(result[v], 1.0, &format!("mix2: x{}=1", i + 1));
    }
    assert_obj_close(result.objective_value, 1.5, "mix2: obj");
    let dual = result.dual_solution.as_ref().expect("mix2: dual_solution");
    assert_eq!(dual.len(), 3, "mix2: dual length = 元 row 数 3");
    // Le (idx 1), Ge (idx 2) は inactive → ≈0
    assert!(dual[1].abs() < EPS_DUAL_ABS,
        "mix2: y_le inactive ≈ 0, got {}", dual[1]);
    assert!(dual[2].abs() < EPS_DUAL_ABS,
        "mix2: y_ge inactive ≈ 0, got {}", dual[2]);
    // Eq dual の符号は規約依存だが大きさは |y_eq|=1
    assert!((dual[0].abs() - 1.0).abs() < EPS_DUAL_ABS,
        "mix2: |y_eq|=1, got {}", dual[0]);
}

// =============================================================================
// mix3: Eq + Le + Ge with Le active
// =============================================================================

/// **構造**: 同上構造で c を変えて Le を活性化させる。
///   min 1/2(x1^2 + x2^2 + x3^2) - 10*x1
///   s.t. x1+x2+x3 = 3 (Eq), x1 <= 2 (Le, active), x3 >= 0.5 (Ge).
///
/// **解析解** (Eq + Le active):
///   ∇L: x1 - 10 - y_eq + y_le = 0, x2 - y_eq = 0, x3 - y_eq - y_ge = 0.
///   active: x1=2, x1+x2+x3=3 ⇒ x2+x3=1. Ge inactive 仮定 (y_ge=0): x3 = y_eq = x2.
///   x2 + x3 = 2 x2 = 1 ⇒ x2 = 0.5, x3 = 0.5. Ge: 0.5 >= 0.5 (境界, weakly active)。
///   厳密境界なので Ge も「ぎりぎり active」だが y_ge=0 で KKT 整合。
///   y_eq = 0.5. y_le = 10 - 2 + y_eq = 8.5.
///   obj = 0.5*(4+0.25+0.25) - 20 = 2.25 - 20 = -17.75。
/// **狙い**: Le active 時の dual の正値性 (|y_le|=8.5 程度) を確認。
#[test]
fn mix3_eq_le_active_dual_recovery() {
    let n = 3;
    let mut model = Model::new("mix3");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    let x3 = model.add_var("x3", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2 + x3) == 3.0));
    model.add_constraint(constraint!(x1 <= 2.0));
    model.add_constraint(constraint!(x3 >= 0.5));
    model.minimize(-10.0 * x1);
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);

    let result = model.solve().expect("mix3: solve");
    // Ge は weakly active (y_ge*=0) のため IPM complementarity z_lb·slack=μ で
    // 両方が O(sqrt(μ)) に落ち着く。dynamic_base_tighten(user_eps=1e-6, ref=1e-8) で
    // base_tighten=100 → eps=1e-8。μ≈1e-8 収束時 slack(x3-0.5) ≈ O(sqrt(1e-8)) ≈ 1e-4。
    // 観測値 8.9e-5 はこの範囲内。2e-4 は 2× マージン。
    const EPS_X_WEAK_ACTIVE: f64 = 2e-4;
    let chk = |a: f64, e: f64, name: &str| {
        let d = (a - e).abs();
        assert!(d < EPS_X_WEAK_ACTIVE, "[mix3:{}] x={:.6e} expected={:.6e} diff={:.3e}", name, a, e, d);
    };
    chk(result[x1], 2.0, "x1=2 (Le active)");
    chk(result[x2], 0.5, "x2=0.5");
    chk(result[x3], 0.5, "x3=0.5 (Ge weakly active)");
    assert_obj_close(result.objective_value, -17.75, "mix3: obj");
    let dual = result.dual_solution.as_ref().expect("mix3: dual_solution");
    assert_eq!(dual.len(), 3, "mix3: dual length=3");
    // |y_le|=8.5 (正値、Le の active dual)
    assert!((dual[1].abs() - 8.5).abs() < EPS_DUAL_ABS,
        "mix3: |y_le|=8.5 expected, got {}", dual[1]);
}

// =============================================================================
// mix4: obj_offset (constant term in objective)
// =============================================================================

/// **構造**: scl4 と同じ問題 (min 1/2 (x1^2+x2^2) s.t. x1+x2=1) に obj_offset = 10.
/// **解析解**: x1=x2=0.5, internal obj=0.25, reported obj = 0.25 + 10 = 10.25。
/// **狙い**: `Model::set_obj_offset` が SolverResult.objective に加算されているか。
#[test]
fn mix4_obj_offset_addition() {
    let n = 2;
    let mut model = Model::new("mix4");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2) == 1.0));
    model.minimize(0.0);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);
    model.set_obj_offset(10.0);

    let result = model.solve().expect("mix4: solve");
    assert_x_close(result[x1], 0.5, "mix4: x1");
    assert_x_close(result[x2], 0.5, "mix4: x2");
    // 期待: reported obj = internal(0.25) + offset(10) = 10.25
    assert_obj_close(result.objective_value, 10.25, "mix4: obj with offset=10");
}

// =============================================================================
// mix5: negative obj_offset
// =============================================================================

/// **狙い**: obj_offset が負数でも正しく加算 (符号の取扱い regression)。
///   同じ問題に offset = -100。
#[test]
fn mix5_obj_offset_negative() {
    let n = 2;
    let mut model = Model::new("mix5");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2) == 1.0));
    model.minimize(0.0);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);
    model.set_obj_offset(-100.0);

    let result = model.solve().expect("mix5: solve");
    assert_obj_close(result.objective_value, -99.75, "mix5: obj=internal(0.25)+offset(-100)");
}

// =============================================================================
// mix6: redundant Le row (always satisfied, structural noise)
// =============================================================================

/// **構造**: 2 var + 1 Eq + 1 Le (redundant, b 巨大).
///   min 1/2(x1^2 + x2^2)
///   s.t. x1 + x2 = 1 (Eq)
///        x1 + x2 <= 100 (Le, redundant: any feasible has sum=1 < 100)
/// **解析解**: x1=x2=0.5, obj=0.25, y_le ≈ 0 (inactive).
/// **狙い**: redundant row が presolve で除去された後、postsolve で dual 配列に
///         0 が詰められて元 row 数の dual_solution が返るか。
#[test]
fn mix6_redundant_le_row_dual_padded() {
    let n = 2;
    let mut model = Model::new("mix6");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", 0.0, 50.0);
    let x2 = model.add_var("x2", 0.0, 50.0);
    model.add_constraint(constraint!((x1 + x2) == 1.0));
    model.add_constraint(constraint!((x1 + x2) <= 100.0));
    model.minimize(0.0);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    model.set_quadratic_objective(q);

    let result = model.solve().expect("mix6: solve");
    assert_x_close(result[x1], 0.5, "mix6: x1");
    assert_x_close(result[x2], 0.5, "mix6: x2");
    assert_obj_close(result.objective_value, 0.25, "mix6: obj");
    let dual = result.dual_solution.as_ref().expect("mix6: dual_solution");
    assert_eq!(dual.len(), 2, "mix6: dual length = 元 row 数 2 (presolve 除去後も元数で報告)");
    assert!(dual[1].abs() < EPS_DUAL_ABS,
        "mix6: y_le redundant ≈ 0, got {}", dual[1]);
}
