//! `check_eq_feasibility` 過剰発火 regression guard。
//! 退化 Eq 制約を持つ小規模 LP で Optimal が返ることを assert する
//! (相対閾値 `feas_rel_tol() * (1 + |b| + |Ax|)` の scale 非依存性検証)。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem, SolveStatus};
use solver::solve_with;
use solver::sparse::CscMatrix;

const EPS_KKT: f64 = 1e-6;

/// 退化 Eq 制約を持つ小規模 LP の構築。
///
/// 構造:
/// - n_eq 個の Eq 制約が「同一頂点」で active になる (退化頂点 = basis に
///   redundant な等式が残る LP optimum)。
/// - 各 Eq 行は単一の決定変数 + slack で構成し、`x = b` 形式で退化を誘発。
///
/// e.g. n_eq=3, n_dec=4 で:
///   x0 + 0 = 0     (Eq, x0=0)
///   x1 + 0 = 0     (Eq, x1=0)
///   x2 + 0 = 0     (Eq, x2=0)
///   x3       = 1   (Eq, x3=1)
///   min x3
/// → 全 4 行が Eq、3 個が退化 (x0=x1=x2=0 が basis の縮退)。
fn build_degenerate_eq_lp(n_eq_zero: usize, value_last: f64, scale_mix: bool) -> LpProblem {
    let n_total = n_eq_zero + 1;
    let m = n_eq_zero + 1;

    let mut tri_rows = Vec::new();
    let mut tri_cols = Vec::new();
    let mut tri_vals = Vec::new();

    // 退化 Eq: x_i = 0 (i < n_eq_zero)
    for i in 0..n_eq_zero {
        tri_rows.push(i);
        tri_cols.push(i);
        // scale_mix 時に i=0 だけ 1e3、i=1 だけ 1e-3 で数値条件数を悪化
        let v = if scale_mix && i == 0 { 1e3 }
                else if scale_mix && i == 1 { 1e-3 }
                else { 1.0 };
        tri_vals.push(v);
    }
    // 最後の Eq: x_{n_eq_zero} = value_last
    tri_rows.push(n_eq_zero);
    tri_cols.push(n_eq_zero);
    tri_vals.push(1.0);

    let a = CscMatrix::from_triplets(&tri_rows, &tri_cols, &tri_vals, m, n_total).unwrap();

    // b: 退化行 = 0、最終行 = value_last (×係数 1)
    let mut b = vec![0.0_f64; m];
    b[n_eq_zero] = value_last;
    let cts = vec![ConstraintType::Eq; m];

    // c: 最終列のみコスト 1、それ以外 0
    let mut c = vec![0.0_f64; n_total];
    c[n_eq_zero] = 1.0;

    let bounds = vec![(0.0_f64, f64::INFINITY); n_total];

    LpProblem::new_general(
        c, a, b, cts, bounds,
        Some(format!("degen_eq_n{}_v{}", n_eq_zero, value_last)),
    ).unwrap()
}

fn assert_optimal_with_value(lp: &LpProblem, expected: f64, label: &str) {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(2.0);
    let r = solve_with(lp, &opts);
    eprintln!("[{}] status={:?} obj={:.6e} expected={:.6e}", label, r.status, r.objective, expected);
    assert_eq!(
        r.status, SolveStatus::Optimal,
        "[{}] expected Optimal, got {:?}. (check_eq_feasibility 過剰発火の疑い)",
        label, r.status
    );
    let obj_err = (r.objective - expected).abs() / (1.0 + expected.abs());
    assert!(obj_err < EPS_KKT, "[{}] obj err {:.3e}", label, obj_err);
}

/// 3 個の退化 Eq 制約 (x0=x1=x2=0) + 1 個の active Eq (x3=5)。
/// 退化頂点での Eq feasibility が完璧に満たせるシンプル case。
#[test]
fn bug5a_degenerate_eq_simple() {
    let lp = build_degenerate_eq_lp(3, 5.0, false);
    assert_optimal_with_value(&lp, 5.0, "bug5a_degen_eq_simple");
}

/// 5 個の退化 Eq + scale mix (1e3 / 1e-3) で数値条件数を悪化。
/// 数値誤差が `FEASIBILITY_TOL = 1e-4` に近づきうる border case。
#[test]
fn bug5b_degenerate_eq_scale_mix() {
    let lp = build_degenerate_eq_lp(5, 1e-3, true);
    assert_optimal_with_value(&lp, 1e-3, "bug5b_degen_eq_scale_mix");
}

/// 8 個の退化 Eq + 単純値。退化数の上限ストレステスト。
#[test]
fn bug5c_degenerate_eq_many() {
    let lp = build_degenerate_eq_lp(8, 1.0, false);
    assert_optimal_with_value(&lp, 1.0, "bug5c_degen_eq_many");
}

/// **境界条件**: Eq 制約 1 行で `|Ax - b|` が exactly 0 (理想 case)。
/// `check_eq_feasibility` のロジック自体の sanity test。
#[test]
fn bug5d_single_eq_constraint_clean() {
    // min x + y, s.t. x + y = 3, x,y >= 0
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![3.0];
    let cts = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug5d_eq_clean".into())).unwrap();
    assert_optimal_with_value(&lp, 3.0, "bug5d_single_eq_clean");
}

/// 大きい b スケール (b=1e6) で退化系の数値ノイズが false NumericalError
/// 化されないこと。相対閾値 `feas_rel_tol() * (1 + |b| + |Ax|)` の scale
/// 非依存性 regression guard。
#[test]
fn bug5e_large_b_scale_degenerate() {
    // 5 退化 Eq (x_i = 0) + 1 active Eq (x_5 = 1e6)
    let n = 6;
    let m = 6;
    let mut tri_rows = Vec::new();
    let mut tri_cols = Vec::new();
    let mut tri_vals = Vec::new();
    for i in 0..5 {
        tri_rows.push(i);
        tri_cols.push(i);
        tri_vals.push(1.0);
    }
    tri_rows.push(5);
    tri_cols.push(5);
    tri_vals.push(1.0);
    let a = CscMatrix::from_triplets(&tri_rows, &tri_cols, &tri_vals, m, n).unwrap();

    let mut b = vec![0.0; m];
    b[5] = 1e6;
    let cts = vec![ConstraintType::Eq; m];

    let mut c = vec![0.0; n];
    c[5] = 1.0;
    let bounds = vec![(0.0, f64::INFINITY); n];

    let lp = LpProblem::new_general(
        c, a, b, cts, bounds, Some("bug5e_large_b_scale".into())
    ).unwrap();
    assert_optimal_with_value(&lp, 1e6, "bug5e_large_b_scale_degenerate");
}

/// 大スケール解 (|x|≈1e6) で `|Ax|` も大きいときの LU 残差を相対閾値で許容。
#[test]
fn bug5f_large_solution_scale() {
    // min -x s.t. x = 1e6, x >= 0
    let c = vec![-1.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1e6];
    let cts = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, f64::INFINITY)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug5f_large_x".into())).unwrap();
    assert_optimal_with_value(&lp, -1e6, "bug5f_large_solution_scale");
}

// ge/eq cold start infeasible 検出 sanity (klein-style row 矛盾)。

/// 単純 infeasible: x >= 3 と x <= 1 の矛盾 (mini smoke)。
#[test]
fn bug6a_simple_infeasible() {
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
    let b = vec![3.0, 1.0];
    let cts = vec![ConstraintType::Ge, ConstraintType::Le];
    let bounds = vec![(0.0, 10.0)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug6a_simple_inf".into())).unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(1.0);
    let r = solve_with(&lp, &opts);
    eprintln!("[bug6a] status={:?}", r.status);
    assert_eq!(r.status, SolveStatus::Infeasible);
}

/// ge / eq 混在 infeasible: x + y >= 5, x + y = 2 (klein-style row 矛盾)。
/// presolve でも検出されうる; primal fallback 経路の sanity。
#[test]
fn bug6b_ge_eq_mix_infeasible() {
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1], &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0], 2, 2,
    ).unwrap();
    let b = vec![5.0, 2.0];
    let cts = vec![ConstraintType::Ge, ConstraintType::Eq];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug6b_ge_eq_inf".into())).unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(1.0);
    let r = solve_with(&lp, &opts);
    eprintln!("[bug6b] status={:?}", r.status);
    assert_eq!(r.status, SolveStatus::Infeasible);
}

/// 3 var / 3 row klein-style infeasible: 等式系で over-determined。
#[test]
fn bug6c_overdetermined_eq_infeasible() {
    // x + y + z = 5
    // x + y     = 3
    //         z = 3   (この 3 つから z=2 と z=3 が矛盾 → infeasible)
    let c = vec![1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 0, 1, 1, 2], &[0, 1, 2, 0, 1, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3, 3,
    ).unwrap();
    let b = vec![5.0, 3.0, 3.0];
    let cts = vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Eq];
    let bounds = vec![(0.0, f64::INFINITY); 3];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug6c_overdet_eq".into())).unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(1.0);
    let r = solve_with(&lp, &opts);
    eprintln!("[bug6c] status={:?}", r.status);
    assert_eq!(r.status, SolveStatus::Infeasible);
}
