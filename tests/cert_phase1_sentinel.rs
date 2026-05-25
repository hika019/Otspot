//! Phase 1 cert sentinel: prove_optimal は実データで no-op 改竄に FAIL する。
//!
//! ## 設計
//!
//! `prove_optimal` は KKT 条件を全て検証し、OK のみ `OptimalCertificate` を返す。
//! no-op 化 (常に Ok を返す) するとこのファイルの sentinel テストが FAIL する。
//!
//! **使用データ**: Maros-Meszaros HS21.QPS (QP, 2 vars 1 constraint)
//! LP simplex は dual sign 規約が IPM と異なるため QP (IPM path) を使用。
//!
//! 2 つの sentinel:
//! 1. **primal sentinel**: x を 2 倍に改竄 → prove_optimal が Err を返す
//! 2. **dual_sign sentinel**: active 制約の y を符号反転 → complementarity が 0 でも
//!    dual_sign が reject する (dual_sign を外すと FAIL)
//!
//! ## CLAUDE.md 準拠
//! - 実データ (Maros-Meszaros) 使用: no-skip
//! - data 欠落時は assert で panic

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::prove_optimal;
use otspot::qp::{ipm_solver::outcome::ProblemView, solve_qp_with};
use std::path::Path;

const HS21_PATH: &str = "data/data/maros_meszaros/HS21.QPS";
const QADLITTL_PATH: &str = "data/data/maros_meszaros/QADLITTL.QPS";
const TOL: f64 = 1e-5;

fn load_qp_and_solve(path_str: &str) -> (otspot::QpProblem, otspot::SolverResult) {
    let path = Path::new(path_str);
    assert!(
        path.exists(),
        "data missing: {} — Maros-Meszaros QP data 必須",
        path_str
    );
    let qp = parse_qps(path).expect("parse QPS");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let result = solve_qp_with(&qp, &opts);
    assert_eq!(
        result.status, SolveStatus::Optimal,
        "{} must solve to Optimal, got {:?}",
        path_str, result.status
    );
    (qp, result)
}

fn make_view<'a>(qp: &'a otspot::QpProblem) -> ProblemView<'a> {
    ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    }
}

/// 正常解: prove_optimal は Ok を返す (sentinel baseline)。
#[test]
fn prove_optimal_accepts_true_optimal_hs21() {
    let (qp, result) = load_qp_and_solve(HS21_PATH);
    let view = make_view(&qp);
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(&view, &result.solution, &result.dual_solution, &result.bound_duals, gap, TOL);
    assert!(
        cert.is_ok(),
        "HS21 最適解は prove_optimal が Ok を返すべき: {:?}",
        cert.err()
    );
    let c = cert.unwrap();
    assert!(c.stationarity_rel() < TOL, "stat={:.3e}", c.stationarity_rel());
    assert!(c.primal_residual_rel() < TOL, "pres={:.3e}", c.primal_residual_rel());
    assert!(c.dual_sign_violation() < TOL, "dsign={:.3e}", c.dual_sign_violation());
}

/// 主変数を 2 倍に改竄 → prove_optimal は Err を返す。
///
/// **sentinel load-bearing**: prove_optimal を no-op にするとこのテストが FAIL。
#[test]
fn prove_optimal_rejects_scaled_primal_hs21() {
    let (qp, result) = load_qp_and_solve(HS21_PATH);
    let view = make_view(&qp);
    let x_bad: Vec<f64> = result.solution.iter().map(|&v| v * 2.0 + 1.0).collect();
    let gap = 1.0;
    let cert = prove_optimal(&view, &x_bad, &result.dual_solution, &result.bound_duals, gap, TOL);
    assert!(
        cert.is_err(),
        "改竄された主変数 (2x+1) は prove_optimal が Err を返すべき"
    );
}

/// dual 符号反転 → dual_sign_violation が detect する。
///
/// complementarity は active 制約 (slack≈0) で y*slack≈0 となるが、
/// dual_sign は y の符号規約違反を直接検出する。
///
/// **dual_sign sentinel load-bearing**: prove_optimal から dual_sign チェックを
/// 外すと failing_conditions に "dual_sign" が入らなくなりこのテストが FAIL。
///
/// QADLITTL (QP, Le 制約多数) を使用。active 制約 idx を実データから選択。
#[test]
fn prove_optimal_dual_sign_sentinel_active_constraint_y_negated_qadlittl() {
    let (qp, result) = load_qp_and_solve(QADLITTL_PATH);
    let view = make_view(&qp);
    let m = qp.num_constraints;

    let ax = qp.a.mat_vec_mul(&result.solution).unwrap_or_else(|_| vec![0.0_f64; m]);

    let tol_slack = 1e-3;
    let tol_y = 1e-5;

    // active な不等式制約を探す (|y| > tol_y かつ slack ≈ 0)
    let active_indices: Vec<usize> = (0..m.min(result.dual_solution.len()))
        .filter(|&i| {
            let slack = match qp.constraint_types[i] {
                ConstraintType::Le => qp.b[i] - ax[i],
                ConstraintType::Ge => ax[i] - qp.b[i],
                _ => return false,
            };
            slack.abs() < tol_slack && result.dual_solution[i].abs() > tol_y
        })
        .collect();

    assert!(
        !active_indices.is_empty(),
        "QADLITTL に active な不等式制約 (|y|>{tol_y}, slack<{tol_slack}) が見つからない:\n\
        num_constraints={m}, max|y|={:.3e}",
        result.dual_solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    );

    // 最初の active 制約の y を符号反転 (complementarity はゼロになる可能性があるが dual_sign は fail)
    let idx = active_indices[0];
    let mut y_bad = result.dual_solution.clone();
    y_bad[idx] = -y_bad[idx];

    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(&view, &result.solution, &y_bad, &result.bound_duals, gap, TOL);

    assert!(
        cert.is_err(),
        "符号反転した dual (constraint #{idx}) は prove_optimal が Err を返すべき"
    );
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"dual_sign"),
        "dual_sign が failing_conditions に含まれるべき: {:?}\n\
        dual_sign_violation={:.3e}, idx={idx}, y_orig={:.3e}, y_bad={:.3e}, ct={:?}",
        err.failing_conditions,
        err.dual_sign_violation,
        result.dual_solution[idx],
        y_bad[idx],
        qp.constraint_types[idx],
    );
}

/// 実測: QADLITTL の QP IPM 最適解において Le≥0, Ge≤0 を確認。
/// prove_optimal の dual_sign チェックが正規最適解を誤拒否しないことの実証。
#[test]
fn dual_sign_convention_observation_qadlittl() {
    let (qp, result) = load_qp_and_solve(QADLITTL_PATH);
    let m = qp.num_constraints;
    let eps = 1e-4;

    let mut le_max_neg = 0.0_f64;
    let mut ge_max_pos = 0.0_f64;

    for i in 0..m.min(result.dual_solution.len()) {
        match qp.constraint_types[i] {
            ConstraintType::Le => {
                if result.dual_solution[i] < -eps {
                    le_max_neg = le_max_neg.max(-result.dual_solution[i]);
                }
            }
            ConstraintType::Ge => {
                if result.dual_solution[i] > eps {
                    ge_max_pos = ge_max_pos.max(result.dual_solution[i]);
                }
            }
            _ => {}
        }
    }

    eprintln!(
        "[dual_sign observation] QADLITTL: Le max-neg-y={:.3e}, Ge max-pos-y={:.3e}",
        le_max_neg, ge_max_pos
    );

    // HS21 Le/Ge 符号確認 (dual_sign convention: Le>=0, Ge<=0)
    assert!(
        le_max_neg < TOL,
        "Le 双対に負の値: max_neg={:.3e}. 規約 (Le>=0) が実データで成立しない",
        le_max_neg
    );
    assert!(
        ge_max_pos < TOL,
        "Ge 双対に正の値: max_pos={:.3e}. 規約 (Ge<=0) が実データで成立しない",
        ge_max_pos
    );

    // prove_optimal も QADLITTL を受理する
    let view = make_view(&qp);
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(&view, &result.solution, &result.dual_solution, &result.bound_duals, gap, TOL);
    assert!(
        cert.is_ok(),
        "QADLITTL 最適解は prove_optimal が Ok を返すべき: {:?}",
        cert.err()
    );
}
