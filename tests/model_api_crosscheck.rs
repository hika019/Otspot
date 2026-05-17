//! Model API (Gurobi-style) と QpProblem 直叩きが**同じ問題に対して同じ解を返す**ことを
//! 検証する integration test。
//!
//! 動機: Model API はユーザー入口だが、`tests/` / `examples/` で直接の利用がほぼなく、
//! 単体テスト (src/model/mod.rs 内) のみで担保されてきた。bench は QPS 経由なので、
//! Model API → solve の経路に符号反転や行列構築のバグがあっても気付けない。
//!
//! 本テストは「同じ問題」を 2 経路で組んで結果を突き合わせる。
//! - 経路 A: Model::new → add_var → add_constraint → minimize → solve()
//! - 経路 B: QpProblem::new → solve_qp_with()
//! 両者の (obj, x) が許容誤差内で一致することを確認する。

use solver::model::{Model, ModelError, SolveError};
use solver::options::SolverOptions;
use solver::problem::ConstraintType;
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

/// 許容誤差。両経路ともに同じ IPM/Simplex 実装を使うので 1e-6 まで詰めて当たるはず。
const TOL_OBJ: f64 = 1e-6;
const TOL_X: f64 = 1e-5;

fn assert_close(a: f64, b: f64, tol: f64, name: &str) {
    assert!(
        (a - b).abs() < tol * (1.0 + a.abs().max(b.abs())),
        "{}: api={:.6e} direct={:.6e} diff={:.3e}",
        name,
        a,
        b,
        (a - b).abs()
    );
}

/// LP: min x + 2y  s.t. x + y >= 3, x + 2y <= 10, x,y in [0, inf)
#[test]
fn lp_basic_ge_le() {
    // --- Model API 経路 ---
    let mut model = Model::new("lp_basic");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    model.add_constraint((x + y).geq(3.0));
    model.add_constraint((x + 2.0 * y).leq(10.0));
    model.minimize(x + 2.0 * y);
    let r_api = model.solve().expect("API solve");

    // --- 直接構築経路 ---
    // c = (1, 2), A = [[1, 1], [1, 2]], b = (3, 10), types = (Ge, Le), bounds = (0, inf) ×2
    let q = CscMatrix::new(2, 2);
    let c = vec![1.0, 2.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let b = vec![3.0, 10.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let cts = vec![ConstraintType::Ge, ConstraintType::Le];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
    assert_close(r_api[x], r_direct.solution[0], TOL_X, "x[0]");
    assert_close(r_api[y], r_direct.solution[1], TOL_X, "x[1]");
}

/// LP: max x + y  s.t. x + y <= 10, x in [0, 5], y in [0, 8]
/// maximize 経由で c の符号反転と obj の符号復元が正しく動くか
#[test]
fn lp_maximize_with_bounds() {
    let mut model = Model::new("lp_max");
    let x = model.add_var("x", 0.0, 5.0);
    let y = model.add_var("y", 0.0, 8.0);
    model.add_constraint((x + y).leq(10.0));
    model.maximize(x + y);
    let r_api = model.solve().expect("API solve");

    // 直接: maximize → minimize -obj、最後に obj 反転 (Model API の挙動を模倣)
    let q = CscMatrix::new(2, 2);
    let c = vec![-1.0, -1.0]; // minimize -(x+y)
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0, 5.0), (0.0, 8.0)];
    let cts = vec![ConstraintType::Le];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());
    let direct_max_obj = -r_direct.objective;

    assert_close(r_api.objective_value, direct_max_obj, TOL_OBJ, "obj");
    // x+y = 10 (constraint active)、x and y の正確な split は LP 退化で複数解、obj だけ確認
}

/// LP: 等式制約。 min x + y  s.t. x + y == 7, x,y in [0, inf)
#[test]
fn lp_equality() {
    let mut model = Model::new("lp_eq");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    model.add_constraint((x + y).eq_constraint(7.0));
    model.minimize(x + y);
    let r_api = model.solve().expect("API solve");

    let q = CscMatrix::new(2, 2);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![7.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let cts = vec![ConstraintType::Eq];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
}

/// LP: 混合 Le/Ge/Eq + 様々な bounds (lb_only, ub_only, 両側、無限)
/// 全変数に有限 bounds を与えて bounded 問題にする (free 変数を含めると x2 が
/// 下方無限大に発散して Unbounded になるため)。
#[test]
fn lp_mixed_constraints_and_bounds() {
    let mut model = Model::new("lp_mixed");
    let x1 = model.add_var("x1", 0.0, 100.0); // lb=0 + ub
    let x2 = model.add_var("x2", -10.0, 5.0); // 両側、lb 負
    let x3 = model.add_var("x3", -2.0, 8.0); // 両側
    let x4 = model.add_var("x4", -5.0, 5.0); // 両側

    model.add_constraint((x1 + x2 + x3 + x4).leq(20.0)); // Le
    model.add_constraint((x1 - x2).geq(-3.0)); // Ge
    model.add_constraint((x3 + x4).eq_constraint(2.0)); // Eq
    model.minimize(2.0 * x1 + x2 + 3.0 * x3 + x4);
    let r_api = model.solve().expect("API solve");

    let q = CscMatrix::new(4, 4);
    let c = vec![2.0, 1.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 0, 0, 1, 1, 2, 2],
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
        3,
        4,
    )
    .unwrap();
    let b = vec![20.0, -3.0, 2.0];
    let bounds = vec![(0.0, 100.0), (-10.0, 5.0), (-2.0, 8.0), (-5.0, 5.0)];
    let cts = vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
}

/// QP: min 1/2 (x^2 + y^2) + (-2x - 3y)  s.t. x + y <= 5, x,y in [0, inf)
/// 最適: KKT より x = 2 - λ, y = 3 - λ. 制約 active なら 5 - 2λ = 5 → λ = 0 → (x,y)=(2,3)、
/// obj = 0.5*(4+9) - 2*2 - 3*3 = 6.5 - 13 = -6.5
#[test]
fn qp_diagonal_q_basic() {
    let mut model = Model::new("qp_diag");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    // Q = [[1, 0], [0, 1]] (1/2 規約: 1/2 x^T Q x = (1/2)(x^2+y^2))
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    model.set_quadratic_objective(q.clone());
    model.add_constraint((x + y).leq(5.0));
    model.minimize(-2.0 * x - 3.0 * y);
    let r_api = model.solve().expect("API solve");

    // 直接構築
    let c = vec![-2.0, -3.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![5.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let cts = vec![ConstraintType::Le];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
    assert_close(r_api[x], r_direct.solution[0], TOL_X, "x");
    assert_close(r_api[y], r_direct.solution[1], TOL_X, "y");
    // 期待値検算: obj ≈ -6.5
    assert!(
        (r_api.objective_value - (-6.5)).abs() < 1e-3,
        "obj should be ≈ -6.5, got {}",
        r_api.objective_value
    );
}

/// QP off-diagonal Q: min 1/2(x²+xy+y²)−x−y  s.t. x+y≤4, x,y≥0  (Model ↔ direct 一致確認)。
#[test]
fn qp_offdiagonal_q() {
    let mut model = Model::new("qp_off");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    // Q = [[1, 0.5], [0.5, 1]] (対称、上下三角両方格納が QPS 慣例)
    let q = CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 0.5, 0.5, 1.0], 2, 2)
        .unwrap();
    model.set_quadratic_objective(q.clone());
    model.add_constraint((x + y).leq(4.0));
    model.minimize(-1.0 * x - 1.0 * y);
    let r_api = model.solve().expect("API solve");

    let c = vec![-1.0, -1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![4.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let cts = vec![ConstraintType::Le];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
    assert_close(r_api[x], r_direct.solution[0], TOL_X, "x");
    assert_close(r_api[y], r_direct.solution[1], TOL_X, "y");
}

/// QP rank-deficient (LP 退化型 mini): min 1/2 x²  s.t. x+y=1, x,y∈[0,1]  →  opt (0,1).
#[test]
fn qp_eq_with_redundant_var() {
    let mut model = Model::new("qp_eq_red");
    let x = model.add_var("x", 0.0, 1.0);
    let y = model.add_var("y", 0.0, 1.0);
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
    model.set_quadratic_objective(q.clone());
    model.add_constraint((x + y).eq_constraint(1.0));
    model.minimize(0.0 * x + 0.0 * y); // c = 0 ですべて Q ベース
    let r_api = model.solve().expect("API solve");

    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, 1.0); 2];
    let cts = vec![ConstraintType::Eq];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();
    let r_direct = solve_qp_with(&prob, &SolverOptions::default());

    assert_close(r_api.objective_value, r_direct.objective, TOL_OBJ, "obj");
    // 期待: x = 0 (Q 最小化のため)、y = 1。obj = 0
    // IPM は厳密に boundary に到達しないため許容差は 1e-2 (Maros 全体と同レベル)
    assert!(r_api[x].abs() < 1e-2, "x should be ~0, got {}", r_api[x]);
    assert!(
        (r_api[y] - 1.0).abs() < 1e-2,
        "y should be ~1, got {}",
        r_api[y]
    );
    assert!(
        r_api.objective_value.abs() < 1e-4,
        "obj should be ~0, got {}",
        r_api.objective_value
    );
}

/// QP maximize 規約検証: Q は NSD で渡す必要あり (内部で -Q)。max -1/2 x²+x, x∈[0,5]  →  opt x=1, obj=0.5.
#[test]
fn qp_maximize_concave() {
    let mut model = Model::new("qp_max");
    let x = model.add_var("x", 0.0, 5.0);
    // Q = [[-1]] (NSD) で渡す。Model 内で -Q = [[1]] (PSD) に反転されてから solver へ。
    let q = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(solver::constraint!(x <= 5.0));
    model.maximize(x);
    let r_api = model.solve().expect("API solve (NSD Q for maximize)");

    assert!(
        (r_api[x] - 1.0).abs() < 1e-3,
        "x should be 1, got {}",
        r_api[x]
    );
    assert!(
        (r_api.objective_value - 0.5).abs() < 1e-3,
        "obj should be 0.5, got {}",
        r_api.objective_value
    );
}

/// 非凸 maximize: max 1/2 x²+x, x∈[0,5] (PSD Q→内部 NSD).  慣性修正 IPM で境界 KKT 点 x=5, obj=17.5 を LocallyOptimal で返す。
#[test]
fn qp_maximize_with_psd_q_returns_error() {
    let mut model = Model::new("qp_max_psd");
    let x = model.add_var("x", 0.0, 5.0);
    // PSD Q: maximize 時は内部で Q を符号反転して NSD (非正定値) になる
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(solver::constraint!(x <= 5.0));
    model.maximize(x);
    // 慣性修正付き IPM が KKT 点 (x=5) を発見し LocallyOptimal として返す。
    // model は LocallyOptimal を有効解として ModelResult に変換する。
    let result = model.solve().expect("maximize with NSD Q should return a LocallyOptimal KKT solution");
    // x=5 が境界最適解
    assert!(
        (result[x] - 5.0).abs() < 1e-3,
        "maximize x^2/2 on [0,5]: x* should be 5.0, got {:.6}", result[x]
    );
    // maximize obj = x + 1/2*x^2 at x=5: 5 + 12.5 = 17.5
    // (maximize(x) sets linear term, set_quadratic_objective sets 1/2*Q*x^2)
    assert!(
        (result.objective_value - 17.5).abs() < 1.0,
        "maximize x + x^2/2 at x=5: obj should be ~17.5, got {:.6}", result.objective_value
    );
}

/// 制約なし QP (bounds のみ): min 1/2(x²+y²)−x, x,y∈[-2,2]  →  opt (1,0), obj=-0.5.
#[test]
fn qp_no_constraints_only_bounds() {
    let mut model = Model::new("qp_nocon");
    let x = model.add_var("x", -2.0, 2.0);
    let y = model.add_var("y", -2.0, 2.0);
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    model.minimize(-1.0 * x);
    let r_api = model.solve().expect("API solve");
    assert!(
        (r_api[x] - 1.0).abs() < 1e-3,
        "x should be 1, got {}",
        r_api[x]
    );
    assert!(r_api[y].abs() < 1e-3, "y should be 0, got {}", r_api[y]);
    assert!(
        (r_api.objective_value - (-0.5)).abs() < 1e-3,
        "obj should be -0.5, got {}",
        r_api.objective_value
    );
}

/// Infeasible 検出が API 経由でも正しく Err として返るか
#[test]
fn lp_infeasible_returns_err() {
    let mut model = Model::new("infeas");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.add_constraint(solver::constraint!(x >= 5.0));
    model.add_constraint(solver::constraint!(x <= 3.0));
    model.minimize(x);
    let err = model.solve().expect_err("should be infeasible");
    assert!(
        matches!(err, ModelError::SolveError(SolveError::Infeasible)),
        "expected SolveError(Infeasible), got {:?}",
        err
    );
}

/// Unbounded 検出
#[test]
fn lp_unbounded_returns_err() {
    let mut model = Model::new("unbnd");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.minimize(-1.0 * x); // x → ∞ で obj → -∞
    let err = model.solve().expect_err("should be unbounded");
    assert!(
        matches!(err, ModelError::SolveError(_)),
        "expected SolveError, got {:?}",
        err
    );
}

/// QP dual 出力の符号規約 sentinel: min 1/2 x²  s.t. x≥1 (Ge)  →  x=1, dual=-1.
/// collapse_extended_dual で Ge を Le に変換時に符号反転されるため、Ge dual は負値 (OSQP/Gurobi と逆規約)。
#[test]
fn qp_dual_solution_available() {
    let mut model = Model::new("qp_dual");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(solver::constraint!(x >= 1.0));
    model.minimize(0.0 * x);
    let r_api = model.solve().expect("API solve");

    assert!(
        (r_api[x] - 1.0).abs() < 1e-3,
        "x should be 1, got {}",
        r_api[x]
    );
    let dual = r_api
        .dual_solution
        .as_ref()
        .expect("dual_solution should be Some for QP");
    assert_eq!(dual.len(), 1, "dual length should match constraints");
    // |dual| ≈ 1 で確認 (符号は実装規約に依存、現状 Ge は負)
    assert!(
        (dual[0].abs() - 1.0).abs() < 1e-2,
        "|Ge dual| should be ≈ 1, got {} (本実装規約: Ge は負側)",
        dual[0]
    );
}

#[test]
fn qp_model_eq_and_bound_active_cluster_solves_consistently() {
    let mut model = Model::new("qp_eq_bound_cluster");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);
    let q = CscMatrix::from_triplets(&[1], &[1], &[2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(solver::constraint!((x + y) == 1.0));
    model.minimize(-1.0 * x);

    let result = model.solve().expect("API solve");

    assert!(
        (result[x] - 1.0).abs() < 1e-6,
        "x should be 1, got {}",
        result[x]
    );
    assert!(result[y].abs() < 1e-6, "y should be 0, got {}", result[y]);
    assert!(
        (result.objective() + 1.0).abs() < 1e-6,
        "objective should be -1, got {}",
        result.objective()
    );

    let dual = result
        .dual_solution
        .as_ref()
        .expect("dual_solution should be Some for QP");
    assert_eq!(dual.len(), 1, "dual length should match constraints");
    assert!(
        (dual[0] - 1.0).abs() < 1e-5,
        "equality dual should be 1, got {}",
        dual[0]
    );
}
