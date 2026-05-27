//! Task #29 凸 QP mini-corpus — **bug class: equality-only constraints**
//!
//! ## 対象 bug class
//!
//! - **等号制約のみ (Eq-only) で KKT 線形系を直接解くケース**
//!   - 不等式 active set が空 → IPM は最初の 1〜2 反復で収束する
//!   - 真因として疑うべき点:
//!     - presolve が Eq を fix/substitute で潰した後 postsolve が y を復元できているか
//!     - IPM 内の RNG/init phase が「不要に Phase I LP」へ落ちないか
//!     - 解析的に解ける問題で objective が一致するか
//!
//! ## このファイルのテスト方針
//!
//! - 全 4 test (eq1-4) を Model API で記述。
//! - eq4 は `ModelResult.bound_duals` (model-api-extender) で active bound 検証。

use otspot::constraint;
use otspot::model::Model;

// solver の収束判定 `ipm_eps` の default は 1e-6 (options.rs)。
// objective は relative tolerance、x は abs ≈ O(eps) で評価する。
const EPS_OBJ_REL: f64 = 1e-6;
const EPS_X_ABS: f64 = 1e-5;
const EPS_DUAL_ABS: f64 = 1e-4;

/// mini test の単一 timeout (CLAUDE.md 「test 1 つ 3 分以内」、mini は 5s 以内)。
const MINI_TIMEOUT_SECS: f64 = 5.0;

fn assert_obj_close(actual: f64, expected: f64, label: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < EPS_OBJ_REL,
        "[{}] obj actual={:.9e} expected={:.9e} rel_err={:.3e}",
        label, actual, expected, rel
    );
}

fn assert_x_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPS_X_ABS,
        "[{}] x actual={:.9e} expected={:.9e} diff={:.3e}",
        label, actual, expected, diff
    );
}

// =============================================================================
// eq1: single Eq, 2 free vars, diagonal Q
// =============================================================================

/// **構造**: min 1/2 (x1^2 + x2^2)  s.t. x1 + x2 = 1, x free.
/// **KKT 直解**: x1 = x2 = 0.5, y = 0.5, obj = 0.25.
/// **狙い**: 最小の Eq-only QP で IPM が「Phase II 直行」収束することを確認。
#[test]
fn eq1_two_free_vars_single_equality() {
    let mut model = Model::new("eq1");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2) == 1.0));
    // Q=diag(1,1): 1/2*x^2 per var → DSL 0.5*(x*x)
    model.minimize(0.5 * x1 * x1 + 0.5 * x2 * x2);

    let result = model.solve().expect("eq1: solve");
    assert_x_close(result[x1], 0.5, "eq1: x1");
    assert_x_close(result[x2], 0.5, "eq1: x2");
    assert_obj_close(result.objective_value, 0.25, "eq1: obj");
    let dual = result.dual_solution.as_ref().expect("eq1: dual_solution");
    assert_eq!(dual.len(), 1, "eq1: dual length");
    assert!((dual[0].abs() - 0.5).abs() < EPS_DUAL_ABS, "eq1: |y|=0.5, got {}", dual[0]);
}

// =============================================================================
// eq2: 3 Eq + 4 free vars (overdetermined KKT)
// =============================================================================

/// **構造**: HS51 と同じ問題だが warm-start 無し。
/// **狙い**: maros_meszaros_qp::test_hs51 は warm-start で Phase I を回避していたが、
///         本 test は default cold-start で同じ最適解に到達するか検証する。
/// **真因仮説**: もし FAIL すれば Phase I LP の自由変数初期化に bug 残存。
#[test]
fn eq2_hs51_cold_start_regression() {
    let mut model = Model::new("eq2_hs51");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    let x3 = model.add_var("x3", f64::NEG_INFINITY, f64::INFINITY);
    let x4 = model.add_var("x4", f64::NEG_INFINITY, f64::INFINITY);
    let x5 = model.add_var("x5", f64::NEG_INFINITY, f64::INFINITY);
    // 等式 3 本: x1+3*x2=4, x3+x4-2*x5=0, x2-x5=0
    model.add_constraint(constraint!((x1 + 3.0 * x2) == 4.0));
    model.add_constraint(constraint!((x3 + x4 - 2.0 * x5) == 0.0));
    model.add_constraint(constraint!((x2 - x5) == 0.0));
    // Q=[[2,-2,0,0,0],[-2,4,2,0,0],[0,2,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]
    // Q[i][i]=v → (v/2)*xi*xi; Q[i][j]=v (i≠j) → v*(xi*xj)
    model.minimize(
        x1 * x1 + 2.0 * x2 * x2 + x3 * x3 + x4 * x4 + x5 * x5
        + (-2.0) * (x1 * x2) + 2.0 * (x2 * x3)
        + (-4.0) * x2 + (-4.0) * x3 + (-2.0) * x4 + (-2.0) * x5,
    );

    let result = model.solve().expect("eq2: cold-start solve (HS51 真因確認)");
    let xs = [x1, x2, x3, x4, x5];
    for (i, &v) in xs.iter().enumerate() {
        assert_x_close(result[v], 1.0, &format!("eq2: x{}=1", i + 1));
    }
    assert_obj_close(result.objective_value, -6.0, "eq2: obj=-6");
}

// =============================================================================
// eq3: pure equality with non-PD positive-semidefinite Q (one zero eigenvalue)
// =============================================================================

/// **構造**: min 1/2 (x1+x2-x3)^2  s.t. x1+x2+x3 = 3, all free.
/// Q = v v^T で v=[1,1,-1]^T → rank-1 (singular)。
/// **解析解**: u = x1+x2-x3 を最小化 (u^2/2)、s.t. x1+x2+x3=3.
///   u=0 ⇔ x1+x2 = x3. 同時に x1+x2+x3=3 ⇒ 2x3=3 ⇒ x3=1.5, x1+x2=1.5。
///   x1, x2 は自由 (任意分配)。obj=0。
/// **狙い**: rank-deficient Q (PSD だが PD でない) で IPM が NumericalError を
///         起こさず Optimal に到達するか。Mehrotra 系では regularization 必須。
#[test]
fn eq3_rank_deficient_psd_q_equality() {
    let mut model = Model::new("eq3_rank1");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", f64::NEG_INFINITY, f64::INFINITY);
    let x2 = model.add_var("x2", f64::NEG_INFINITY, f64::INFINITY);
    let x3 = model.add_var("x3", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x1 + x2 + x3) == 3.0));
    // Q = v v^T, v=[1,1,-1]: 1/2*(x1+x2-x3)^2
    // = 0.5*x1^2+0.5*x2^2+0.5*x3^2 + x1*x2 - x1*x3 - x2*x3
    model.minimize(
        0.5 * x1 * x1 + 0.5 * x2 * x2 + 0.5 * x3 * x3
        + 1.0 * (x1 * x2) + (-1.0) * (x1 * x3) + (-1.0) * (x2 * x3),
    );

    let result = model.solve().expect("eq3: rank-1 Q + Eq の Optimal 到達");
    assert_obj_close(result.objective_value, 0.0, "eq3: obj=0");
    assert_x_close(result[x3], 1.5, "eq3: x3=1.5");
    let sum12 = result[x1] + result[x2];
    assert_x_close(sum12, 1.5, "eq3: x1+x2=1.5");
    let sum_all = result[x1] + result[x2] + result[x3];
    assert_x_close(sum_all, 3.0, "eq3: sum=3");
}

// =============================================================================
// eq4: Eq + bounds, no Le/Ge (bound active at optimum)
// =============================================================================

/// **構造**: min 1/2 (x1^2 + x2^2) - x1 - 2*x2  s.t. x1 + x2 = 2, 0 <= x_i <= 10
/// **KKT 直解 (bound 非 active と仮定)**: x1 = 1 - y, x2 = 2 - y, x1+x2=2 ⇒ y = 0.5.
///   x1 = 0.5, x2 = 1.5。両方 (0,10) 内点なので bound 非 active OK。
///   obj = 0.5*(0.25+2.25) - 0.5 - 3 = 1.25 - 3.5 = -2.25。
/// **狙い**: bound あり Eq で IPM が bound shift しないか (bound 非 active なら dual=0)。
///         `ModelResult.bound_duals` (model-api-extender) で検証。
#[test]
fn eq4_equality_with_inactive_bounds() {
    let mut model = Model::new("eq4");
    model.set_timeout(MINI_TIMEOUT_SECS);
    let x1 = model.add_var("x1", 0.0, 10.0);
    let x2 = model.add_var("x2", 0.0, 10.0);
    model.add_constraint(constraint!((x1 + x2) == 2.0));
    model.minimize(0.5 * x1 * x1 + 0.5 * x2 * x2 + (-1.0) * x1 + (-2.0) * x2);

    let result = model.solve().expect("eq4: solve");
    assert_x_close(result[x1], 0.5, "eq4: x1");
    assert_x_close(result[x2], 1.5, "eq4: x2");
    assert_obj_close(result.objective_value, -2.25, "eq4: obj");
    // bound_duals: 全 bound inactive → 全要素 ≈ 0 (CLAUDE.md L20「実装を正とするな」 — 期待値で assert)
    for (k, &bd) in result.bound_duals.iter().enumerate() {
        assert!(bd.abs() < EPS_DUAL_ABS, "eq4: bound_dual[{}]={} expected 0 (bound inactive)", k, bd);
    }
}
