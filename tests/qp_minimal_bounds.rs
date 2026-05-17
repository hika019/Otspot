//! Task #29 凸 QP mini-corpus — **bug class: bounds / bound dual recovery**
//!
//! ## 対象 bug class
//!
//! - **bound-only QP** (制約行 m=0、bounds のみ) で最適性に到達するか。
//! - **bound active 時の bound_duals** が正しく復元されるか。
//!   - lb active: bound_dual > 0 (=Q_j x + c_j の符号反転、reduced cost 相当)
//!   - ub active: bound_dual > 0 (上側 dual の絶対値)
//!   - 配列順は `[lb_dual; n_lb] ++ [ub_dual; n_ub]` (SolverResult docs §2.5)
//! - **片側 bound (lb only / ub only)** で IPM が誤った dual を返さないか。
//! - **fixed 変数 (lb==ub)** の bound dual 復元 (presolve fix 経路)。
//!
//! ## 真因仮説
//!
//! - bound_duals の符号規約 (lb は -reduced_cost、ub は +reduced_cost) が
//!   postsolve で混同されると dual feasibility は KKT 満たすが報告値が逆。
//! - 片側 bound の n_lb/n_ub カウントは bounds.iter().filter(...) で計算する
//!   ため、INF/NEG_INF 判定が誤ると配列長ずれ。

use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

const EPS_OBJ_REL: f64 = 1e-6;
const EPS_X_ABS: f64 = 1e-5;
const EPS_DUAL_ABS: f64 = 1e-4;
const MINI_TIMEOUT_SECS: f64 = 5.0;

fn solver_opts() -> SolverOptions {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(MINI_TIMEOUT_SECS);
    opts
}

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
// bnd1: bound-only QP, no A (interior optimum)
// =============================================================================

/// **構造**: min 1/2 (x1^2 + x2^2) - 3*x1 - 4*x2, s.t. 0 <= x_i <= 10. No A.
/// **解析解**: ∇f=0 ⇒ x1=3, x2=4 (両方 interior)。obj = 1/2*(9+16) - 9 - 16 = -12.5。
/// **狙い**: m=0 で IPM が空行列を正しくスキップするか。bound_duals=0 報告。
#[test]
fn bnd1_no_constraints_interior_optimum() {
    let n = 2;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![-3.0, -4.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0, 10.0), (0.0, 10.0)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd1: status");
    assert_x_close(r.solution[0], 3.0, "bnd1: x1=3");
    assert_x_close(r.solution[1], 4.0, "bnd1: x2=4");
    assert_obj_close(r.objective, -12.5, "bnd1: obj");
    assert_eq!(r.dual_solution.len(), 0, "bnd1: dual_solution empty");
    // bound_duals: 全 inactive → 各 ≈ 0
    for (k, &bd) in r.bound_duals.iter().enumerate() {
        assert!(bd.abs() < EPS_DUAL_ABS, "bnd1: bound_dual[{}]={} expected 0", k, bd);
    }
}

// =============================================================================
// bnd2: bound-only QP, lower bound active
// =============================================================================

/// **構造**: min 1/2 x1^2 + 5*x1 + 1/2 x2^2 - 2*x2, s.t. 0 <= x_i <= 10. No A.
/// **解析解**: 内点最小は x1=-5 (lb 違反) ⇒ x1=0 (lb active), x2=2 (interior).
///   obj = 0 + 0 + 0.5*4 - 4 = -2.0。
///   KKT: Q x + c - z_l = 0 → x1=0: 1*0 + 5 - z_l1 = 0 ⇒ z_l1 = 5。
/// **狙い**: lb active → bound_dual[lb idx of x1] ≈ 5 を確認。
#[test]
fn bnd2_lower_bound_active_dual_recovery() {
    let n = 2;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![5.0, -2.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0, 10.0), (0.0, 10.0)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd2: status");
    assert_x_close(r.solution[0], 0.0, "bnd2: x1=0 (lb active)");
    assert_x_close(r.solution[1], 2.0, "bnd2: x2=2 (interior)");
    assert_obj_close(r.objective, -2.0, "bnd2: obj");
    // bound_duals = [lb_dual(x0), lb_dual(x1), ub_dual(x0), ub_dual(x1)]
    // 期待: lb_dual(x0)=5, lb_dual(x1)≈0, ub_dual=0 全部
    assert_eq!(r.bound_duals.len(), 4, "bnd2: bound_duals length n_lb+n_ub = 2+2");
    assert!((r.bound_duals[0] - 5.0).abs() < EPS_DUAL_ABS,
        "bnd2: lb_dual(x0)=5 expected, got {}", r.bound_duals[0]);
    assert!(r.bound_duals[1].abs() < EPS_DUAL_ABS,
        "bnd2: lb_dual(x1)≈0 expected, got {}", r.bound_duals[1]);
    assert!(r.bound_duals[2].abs() < EPS_DUAL_ABS,
        "bnd2: ub_dual(x0)≈0 expected, got {}", r.bound_duals[2]);
    assert!(r.bound_duals[3].abs() < EPS_DUAL_ABS,
        "bnd2: ub_dual(x1)≈0 expected, got {}", r.bound_duals[3]);
}

// =============================================================================
// bnd3: upper bound active
// =============================================================================

/// **構造**: min 1/2 x^2 - 20*x, s.t. 0 <= x <= 5. No A.
/// **解析解**: 内点最小は x=20 (ub 違反) ⇒ x=5 (ub active).
///   KKT: x + c + z_u = 0 → 5 - 20 + z_u = 0 ⇒ z_u = 15。
///   obj = 0.5*25 - 100 = -87.5。
/// **狙い**: ub active → bound_dual[ub idx of x] ≈ 15 を確認。
///         **片側 (lb のみ / ub のみ) を混在させる効果**: x には両 bound あり。
#[test]
fn bnd3_upper_bound_active_dual_recovery() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![-20.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0, 5.0)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd3: status");
    assert_x_close(r.solution[0], 5.0, "bnd3: x=5 (ub active)");
    assert_obj_close(r.objective, -87.5, "bnd3: obj");
    // bound_duals = [lb_dual(x0), ub_dual(x0)] (n_lb=1, n_ub=1)
    assert_eq!(r.bound_duals.len(), 2, "bnd3: bound_duals length");
    assert!(r.bound_duals[0].abs() < EPS_DUAL_ABS,
        "bnd3: lb_dual≈0 (x=5 not at lb), got {}", r.bound_duals[0]);
    assert!((r.bound_duals[1] - 15.0).abs() < EPS_DUAL_ABS,
        "bnd3: ub_dual=15 expected, got {}", r.bound_duals[1]);
}

// =============================================================================
// bnd4: one-sided bounds (lb only, ub only, fully free mixed)
// =============================================================================

/// **構造**: 3 var. x1 in [0, INF), x2 in (-INF, 5], x3 free.
///   min 1/2(x1^2 + x2^2 + x3^2) - x1 - 6*x2 - 2*x3. No A.
/// **解析解**: 内点最小 → x1=1, x2=6 (ub 違反) ⇒ x2=5 (ub active), x3=2 (free interior).
///   bound_duals 配列: n_lb=1 (x1のみ), n_ub=1 (x2のみ) → 長さ 2。
///   [lb_dual(x1), ub_dual(x2)] = [0 (interior), 1 (active, KKT)]。
///   x3 は free なので bound_duals に含まれない。
///   obj = 0.5*(1+25+4) - 1 - 30 - 4 = 15 - 35 = -20。
/// **狙い**: bound_duals 配列長と並びが「有限 bound のみカウント」になっているか。
#[test]
fn bnd4_one_sided_bounds_array_layout() {
    let n = 3;
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], n, n).unwrap();
    let c = vec![-1.0, -6.0, -2.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![
        (0.0, f64::INFINITY),           // x1: lb only
        (f64::NEG_INFINITY, 5.0),       // x2: ub only
        (f64::NEG_INFINITY, f64::INFINITY), // x3: free
    ];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd4: status");
    assert_x_close(r.solution[0], 1.0, "bnd4: x1=1 (interior)");
    assert_x_close(r.solution[1], 5.0, "bnd4: x2=5 (ub active)");
    assert_x_close(r.solution[2], 2.0, "bnd4: x3=2 (free interior)");
    assert_obj_close(r.objective, -20.0, "bnd4: obj");
    // bound_duals = [lb_dual(x1), ub_dual(x2)] (length 2)
    assert_eq!(r.bound_duals.len(), 2,
        "bnd4: bound_duals length must equal n_lb_finite + n_ub_finite = 1+1");
    assert!(r.bound_duals[0].abs() < EPS_DUAL_ABS,
        "bnd4: lb_dual(x1)≈0 expected, got {}", r.bound_duals[0]);
    assert!((r.bound_duals[1] - 1.0).abs() < EPS_DUAL_ABS,
        "bnd4: ub_dual(x2)=1 expected, got {}", r.bound_duals[1]);
}

// =============================================================================
// bnd5: fixed variable (lb==ub) via presolve fix
// =============================================================================

/// **構造**: min 1/2(x1^2 + x2^2) - x1 - x2, s.t. x1 in [2, 2] (fixed), x2 in [0, 10].
/// **解析解**: x1=2 (fixed), x2=1 (interior min)。
///   obj = 0.5*(4+1) - 2 - 1 = 2.5 - 3 = -0.5。
/// **狙い**: lb==ub の fixed 変数 (presolve で除去) で objective が正しく合計されるか。
///         bound_duals は fixed 変数 idx で 0 が入る (SolverResult docs)。
#[test]
fn bnd5_fixed_variable_objective_offset() {
    let n = 2;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(2.0, 2.0), (0.0, 10.0)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd5: status");
    assert_x_close(r.solution[0], 2.0, "bnd5: x1=2 (fixed)");
    assert_x_close(r.solution[1], 1.0, "bnd5: x2=1 (interior)");
    assert_obj_close(r.objective, -0.5, "bnd5: obj=-0.5");
}

// =============================================================================
// bnd6: ill-scaled bounds (lb=1e-8, ub=1e8)
// =============================================================================

/// **構造**: min 1/2 x^2 - 1000*x, s.t. 1e-8 <= x <= 1e8. No A.
/// **解析解**: 内点 x=1000 (両 bound から離れた interior)。
///   obj = 0.5*1e6 - 1e6 = -5e5。
/// **狙い**: 極端な bound scale で IPM の barrier path が x=1000 に収束するか。
///         BOUND_TOL absolute なら 1e-8 lb を「active」と誤判定する。
#[test]
fn bnd6_ill_scaled_bounds_interior_optimum() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![-1000.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(1e-8, 1e8)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "bnd6: status");
    // x=1000 ± O(eps*scale)。1000 を中心に rel err < 1e-4 で OK。
    let rel = (r.solution[0] - 1000.0).abs() / 1000.0;
    assert!(rel < 1e-4, "bnd6: x≈1000, got {} (rel={:.3e})", r.solution[0], rel);
    assert_obj_close(r.objective, -500_000.0, "bnd6: obj=-5e5");
    // bound_duals は両方 inactive → ≈0
    for (k, &bd) in r.bound_duals.iter().enumerate() {
        // 大スケール bound → dual も多少ノイズ。EPS_DUAL_ABS * scale 許容。
        let tol = EPS_DUAL_ABS * (1.0 + r.solution[0].abs());
        assert!(bd.abs() < tol, "bnd6: bound_dual[{}]={} (tol={:.1e})", k, bd, tol);
    }
}
