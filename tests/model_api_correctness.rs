//! Model API 正確性テスト
//!
//! README に示された `Model::new` / `add_var` / `constraint!` マクロ / `minimize` /
//! `maximize` / `result[x]` を直接使い、解析的に解が分かる問題で数値的正確性を検証する。
//!
//! **方針**:
//! - CSC行列・QpProblem の直叩きは使わない（それは model_api_crosscheck.rs の役割）
//! - 各テストに手計算した解析解を期待値として埋め込む
//! - 人間が読んで問題の意図と期待値の根拠が分かるようにコメントを書く

use solver::model::{Model, ModelError, SolveError};
use solver::constraint;
use solver::sparse::CscMatrix;

// ユニットテスト許容誤差。LP はシンプレックスなので 1e-6、QP は IPM なので少し緩める。
const LP_TOL: f64 = 1e-6;
const QP_TOL: f64 = 2e-3;

// ---------------------------------------------------------------------------
// 1. README の production planning 問題 — 最もシンプルな入口テスト
// ---------------------------------------------------------------------------

/// README のクイックスタート例をそのままテストする。
///
/// 問題: minimize x + 2y
///       2x + 3y <= 12
///       x  +  y >= 3
///       x in [0, +inf), y in [0, 10]
///
/// 解析解: コスト係数 c_x=1 < c_y=2 なので x を優先使用する。
///   等式 x + y = 3 (下界 active) で x=3, y=0 → obj = 3 + 0 = 3
#[test]
fn model_lp_production_planning() {
    let mut model = Model::new("production");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().expect("solve failed");

    assert!(
        (result.objective() - 3.0).abs() < LP_TOL,
        "obj={} expected 3.0", result.objective()
    );
    assert!(
        (result[x] - 3.0).abs() < LP_TOL,
        "x={} expected 3.0", result[x]
    );
    assert!(
        result[y].abs() < LP_TOL,
        "y={} expected 0.0", result[y]
    );
}

// ---------------------------------------------------------------------------
// 2. 最大化 (maximize)
// ---------------------------------------------------------------------------

/// maximize が minimize の符号反転を正しく行い、obj の符号も復元されることを検証する。
///
/// 問題: maximize 3x + 5y
///       x  + y  <= 10
///       x in [0, +inf), y in [0, +inf)
///
/// 解析解: c_y/c_x = 5/3 > 1 なので y を優先。制約 x+y<=10 で y=10, x=0 が最適。
///   max obj = 5 * 10 = 50
#[test]
fn model_lp_maximize() {
    let mut model = Model::new("revenue");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    model.add_constraint(constraint!((x + y) <= 10.0));
    model.maximize(3.0 * x + 5.0 * y);

    let result = model.solve().expect("solve failed");

    assert!(
        (result.objective() - 50.0).abs() < LP_TOL,
        "obj={} expected 50.0", result.objective()
    );
    // 最適解は y=10, x=0。LP の退化はないのでユニーク。
    assert!(
        result[x].abs() < LP_TOL,
        "x={} expected 0.0", result[x]
    );
    assert!(
        (result[y] - 10.0).abs() < LP_TOL,
        "y={} expected 10.0", result[y]
    );
}

// ---------------------------------------------------------------------------
// 3. 等式制約 (equality constraint)
// ---------------------------------------------------------------------------

/// 等式制約が正しくハンドリングされ、解が等式を満たすことを検証する。
///
/// 問題: minimize x + 2y
///       x + y == 5
///       x in [0, +inf), y in [0, +inf)
///
/// 解析解: c_x=1 < c_y=2 なので y=0 に押し付け x=5 が最適。obj = 5
#[test]
fn model_lp_equality_constraint() {
    let mut model = Model::new("eq_model");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    // constraint! マクロの等式構文
    model.add_constraint(constraint!((x + y) == 5.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().expect("solve failed");

    assert!(
        (result.objective() - 5.0).abs() < LP_TOL,
        "obj={} expected 5.0", result.objective()
    );
    // 等式制約を満たしているか
    assert!(
        (result[x] + result[y] - 5.0).abs() < LP_TOL,
        "x+y={} should equal 5.0", result[x] + result[y]
    );
}

// ---------------------------------------------------------------------------
// 4. 変数の上下限 (variable bounds)
// ---------------------------------------------------------------------------

/// 上下限が実際の最適解の位置に影響することを検証する。
///
/// 問題: minimize x + y
///       x in [2, 5], y in [1, 3]
///       (制約なし: 変数の bounds だけ)
///
/// 解析解: minimize なので各変数を下限に置く。x=2, y=1, obj=3
#[test]
fn model_lp_variable_bounds() {
    let mut model = Model::new("bounds_model");
    let x = model.add_var("x", 2.0, 5.0);
    let y = model.add_var("y", 1.0, 3.0);
    // 変数の上限を明示的な制約としても追加（simplex が bounds のみで終わらない保証）
    model.add_constraint(constraint!(x <= 5.0));
    model.add_constraint(constraint!(y <= 3.0));
    model.minimize(x + y);

    let result = model.solve().expect("solve failed");

    // 下限に張り付く
    assert!(
        (result[x] - 2.0).abs() < LP_TOL,
        "x={} expected 2.0 (lower bound)", result[x]
    );
    assert!(
        (result[y] - 1.0).abs() < LP_TOL,
        "y={} expected 1.0 (lower bound)", result[y]
    );
    assert!(
        (result.objective() - 3.0).abs() < LP_TOL,
        "obj={} expected 3.0", result.objective()
    );
}

// ---------------------------------------------------------------------------
// 5. 実行不可能 (infeasible)
// ---------------------------------------------------------------------------

/// 矛盾する制約に対して solve() が Err(Infeasible) を返すことを検証する。
///
/// 問題: x >= 5 かつ x <= 3 (解なし)
#[test]
fn model_lp_infeasible() {
    let mut model = Model::new("infeasible_model");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.add_constraint(constraint!(x >= 5.0));
    model.add_constraint(constraint!(x <= 3.0));
    model.minimize(x);

    let err = model.solve().expect_err("should be infeasible");
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "expected SolveError(Infeasible), got {:?}", err
    );
}

// ---------------------------------------------------------------------------
// 6. 非有界 (unbounded)
// ---------------------------------------------------------------------------

/// 目的関数が -∞ に向かう問題で solve() が Err を返すことを検証する。
///
/// 問題: minimize -x, x in [0, +inf) → x → +∞ で obj → -∞
#[test]
fn model_lp_unbounded() {
    let mut model = Model::new("unbounded_model");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.minimize(-1.0 * x); // x が大きいほど obj が小さい

    let err = model.solve().expect_err("should be unbounded");
    // ソルバーが Unbounded または SolveError を返せばよい
    assert!(
        matches!(err, ModelError::SolveError(_)),
        "expected SolveError (Unbounded), got {:?}", err
    );
}

// ---------------------------------------------------------------------------
// 7. QP: 二次目的 (set_quadratic_objective)
// ---------------------------------------------------------------------------

/// set_quadratic_objective を使って QP を解くことを検証する。
///
/// 問題: minimize (1/2) * 2x^2 + (1/2) * 2y^2 - 4x - 4y
///       (Q = [[2,0],[0,2]], c = [-4,-4], "1/2あり" 規約)
///       x, y in [0, +inf)
///
/// 解析解: 無制約最小化 → ∂/∂x = 2x - 4 = 0 → x=2, 同様 y=2
///   obj = (1/2)*2*(4+4) - 4*2 - 4*2 = 8 - 16 = -8
#[test]
fn model_qp_simple() {
    let mut model = Model::new("qp_simple");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    // Q = [[2, 0], [0, 2]]  ("1/2あり" 規約)
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    model.minimize(-4.0 * x + -4.0 * y);

    let result = model.solve().expect("QP solve failed");

    assert!(
        (result[x] - 2.0).abs() < QP_TOL,
        "x={} expected 2.0", result[x]
    );
    assert!(
        (result[y] - 2.0).abs() < QP_TOL,
        "y={} expected 2.0", result[y]
    );
    assert!(
        (result.objective() - (-8.0)).abs() < QP_TOL,
        "obj={} expected -8.0", result.objective()
    );
}

// ---------------------------------------------------------------------------
// 8. 双対変数 (dual variables)
// ---------------------------------------------------------------------------

/// QP の最適解での双対変数 (shadow price) が取得でき、KKT 条件から予測される値と
/// 一致することを検証する。
///
/// 問題: minimize (1/2) * x^2  (Q = [[1]], c = [0], "1/2あり" 規約)
///       x >= 1  (Ge 制約)
///       x in (-inf, +inf)
///
/// 解析解: x = 1 が最適 (Ge 制約 active)。
///   KKT: x - λ = 0 (λ は Ge 制約の Lagrange 乗数、符号規約に注意) → |λ| = 1
///
/// 符号規約の注意: この実装では Ge 制約の dual は **負** になる (Le 変換後の双対)。
///   重要なのは |dual| = 1 であること。
#[test]
fn model_dual_variables() {
    let mut model = Model::new("dual_model");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);

    // Q = [[1]] ("1/2あり" → (1/2)*1*x^2 = x^2/2)
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(constraint!(x >= 1.0));
    model.minimize(0.0 * x); // 線形項は 0

    let result = model.solve().expect("QP solve failed");

    // 主解の確認
    assert!(
        (result[x] - 1.0).abs() < QP_TOL,
        "x={} expected 1.0", result[x]
    );

    // 双対変数の取得と検証
    let dual = result.dual_solution
        .as_ref()
        .expect("dual_solution should be Some for QP");
    assert_eq!(dual.len(), 1, "should have 1 dual variable for 1 constraint");
    // |dual| ≈ 1 (符号は実装規約に依存)
    assert!(
        (dual[0].abs() - 1.0).abs() < QP_TOL,
        "|dual[0]|={} expected ≈1.0 (Ge の shadow price)", dual[0].abs()
    );
}

// ---------------------------------------------------------------------------
// 補足テスト: constraint! マクロの各構文形式を網羅する
// ---------------------------------------------------------------------------

/// constraint! マクロの単変数形式 `constraint!(x <= rhs)` を検証する。
///
/// README の「Single variable」構文の例題。
/// 問題: minimize x + y, x <= 3, y <= 4, x >= 0, y >= 0
///       解析解: x=0, y=0, obj=0
#[test]
fn model_constraint_macro_single_variable() {
    let mut model = Model::new("macro_single");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    // 単変数構文
    model.add_constraint(constraint!(x <= 3.0));
    model.add_constraint(constraint!(y <= 4.0));
    model.minimize(x + y);

    let result = model.solve().expect("solve failed");

    assert!(
        result[x].abs() < LP_TOL,
        "x={} expected 0.0", result[x]
    );
    assert!(
        result[y].abs() < LP_TOL,
        "y={} expected 0.0", result[y]
    );
    assert!(
        result.objective().abs() < LP_TOL,
        "obj={} expected 0.0", result.objective()
    );
}

/// method API (`leq`, `geq`, `eq_constraint`) でも同じ問題が解けることを検証する。
///
/// README の「メソッドAPIを使うこともできる」の例。
/// 問題: minimize x + 2y
///       (x + 2y).leq(8), (x - y).geq(0), (x + y).eq_constraint(4)
///       x, y in [0, +inf)
///
/// 解析解: Eq x+y=4 → y=4-x。目的 = x + 2(4-x) = 8 - x。
///   minimize なので x を最大化する。Ge x>=y → x>=4-x → x>=2。y>=0 → x<=4。
///   x=4, y=0 が最適。obj = 4 + 0 = 4
#[test]
fn model_method_api_constraints() {
    let mut model = Model::new("method_api");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    model.add_constraint((x + 2.0 * y).leq(8.0));
    model.add_constraint((x - y).geq(0.0));
    model.add_constraint((x + y).eq_constraint(4.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().expect("solve failed");

    assert!(
        (result.objective() - 4.0).abs() < LP_TOL,
        "obj={} expected 4.0", result.objective()
    );
    assert!(
        (result[x] - 4.0).abs() < LP_TOL,
        "x={} expected 4.0", result[x]
    );
    assert!(
        result[y].abs() < LP_TOL,
        "y={} expected 0.0", result[y]
    );
}

/// 3変数 LP で解析解を確認する。
///
/// 問題: minimize 2x + 3y + z
///       x + y + z == 6
///       x + 2y    <= 8
///       x, y, z in [0, +inf)
///
/// 解析解: c_z=1 が最小なので z を最大化する方向。
///   Eq: z = 6 - x - y。目的: 2x + 3y + (6-x-y) = x + 2y + 6
///   → x, y を下限 (=0) に。z = 6, obj = 0 + 0 + 6 = 6
#[test]
fn model_lp_three_variables() {
    let mut model = Model::new("three_var");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    let z = model.add_var("z", 0.0, f64::INFINITY);

    model.add_constraint(constraint!((x + y + z) == 6.0));
    model.add_constraint(constraint!((x + 2.0 * y) <= 8.0));
    model.minimize(2.0 * x + 3.0 * y + z);

    let result = model.solve().expect("solve failed");

    assert!(
        (result.objective() - 6.0).abs() < LP_TOL,
        "obj={} expected 6.0", result.objective()
    );
    assert!(
        result[x].abs() < LP_TOL,
        "x={} expected 0.0", result[x]
    );
    assert!(
        result[y].abs() < LP_TOL,
        "y={} expected 0.0", result[y]
    );
    assert!(
        (result[z] - 6.0).abs() < LP_TOL,
        "z={} expected 6.0", result[z]
    );
}
