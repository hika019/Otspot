//! Phase 1 cert sentinel: prove_optimal は実データで no-op 改竄に FAIL する。
//!
//! ## 設計
//!
//! `prove_optimal` は KKT 条件を全て検証し、OK のみ `OptimalCertificate` を返す。
//! no-op 化 (常に Ok を返す) するとこのファイルの sentinel テストが FAIL する。
//!
//! **使用データ**: Maros-Meszaros HS21.QPS / QADLITTL.QPS (QP, IPM path)
//! 合成 QP fixture: box 制約 QP (active upper bound) + cancelling Le 制約 QP
//!
//! ## CLAUDE.md 準拠
//! - 実データ (Maros-Meszaros) 使用: no-skip
//! - data 欠落時は assert で panic

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::prove_optimal;
use otspot::qp::{ipm_solver::outcome::ProblemView, solve_qp, solve_qp_with, QpProblem};
use otspot::CscMatrix;
use std::path::Path;

const HS21_PATH: &str = "data/maros_meszaros/HS21.QPS";
const QADLITTL_PATH: &str = "data/maros_meszaros/QADLITTL.QPS";
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
        result.status,
        SolveStatus::Optimal,
        "{} must solve to Optimal, got {:?}",
        path_str,
        result.status
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

// ── real-data sentinels ──────────────────────────────────────────────────────

/// 正常解: prove_optimal は Ok を返す (sentinel baseline)。
#[test]
fn prove_optimal_accepts_true_optimal_hs21() {
    let (qp, result) = load_qp_and_solve(HS21_PATH);
    let view = make_view(&qp);
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
        gap,
        TOL,
    );
    assert!(
        cert.is_ok(),
        "HS21 最適解は prove_optimal が Ok を返すべき: {:?}",
        cert.err()
    );
    let c = cert.unwrap();
    assert!(
        c.stationarity_rel() < TOL,
        "stat={:.3e}",
        c.stationarity_rel()
    );
    assert!(
        c.primal_residual_rel() < TOL,
        "pres={:.3e}",
        c.primal_residual_rel()
    );
    assert!(
        c.dual_sign_violation() < TOL,
        "dsign={:.3e}",
        c.dual_sign_violation()
    );
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
    let cert = prove_optimal(
        &view,
        &x_bad,
        &result.dual_solution,
        &result.bound_duals,
        gap,
        TOL,
    );
    assert!(
        cert.is_err(),
        "改竄された主変数 (2x+1) は prove_optimal が Err を返すべき"
    );
}

/// dual 符号反転 → dual_sign_violation が detect する。
///
/// QADLITTL の active 制約 (|y|>1e-5, slack<1e-3) の y を符号反転。
/// complementarity は active 制約 (slack≈0) で y*slack≈0 となるが、
/// dual_sign は y の符号規約違反を直接検出する。
///
/// **dual_sign sentinel load-bearing**: prove_optimal から dual_sign チェックを
/// 外すと failing_conditions に "dual_sign" が入らなくなりこのテストが FAIL。
#[test]
fn prove_optimal_dual_sign_sentinel_active_constraint_y_negated_qadlittl() {
    let (qp, result) = load_qp_and_solve(QADLITTL_PATH);
    let view = make_view(&qp);
    let m = qp.num_constraints;

    let ax =
        qp.a.mat_vec_mul(&result.solution)
            .unwrap_or_else(|_| vec![0.0_f64; m]);

    let tol_slack = 1e-3;
    let tol_y = 1e-5;

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
        result
            .dual_solution
            .iter()
            .fold(0.0_f64, |a, &v| a.max(v.abs()))
    );

    let idx = active_indices[0];
    let mut y_bad = result.dual_solution.clone();
    y_bad[idx] = -y_bad[idx];

    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(
        &view,
        &result.solution,
        &y_bad,
        &result.bound_duals,
        gap,
        TOL,
    );

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
#[test]
fn dual_sign_convention_observation_qadlittl() {
    let (qp, result) = load_qp_and_solve(QADLITTL_PATH);
    let m = qp.num_constraints;
    let eps = 1e-4;

    let mut le_max_neg = 0.0_f64;
    let mut ge_max_pos = 0.0_f64;

    for i in 0..m.min(result.dual_solution.len()) {
        match qp.constraint_types[i] {
            ConstraintType::Le if result.dual_solution[i] < -eps => {
                le_max_neg = le_max_neg.max(-result.dual_solution[i]);
            }
            ConstraintType::Ge if result.dual_solution[i] > eps => {
                ge_max_pos = ge_max_pos.max(result.dual_solution[i]);
            }
            _ => {}
        }
    }

    eprintln!(
        "[dual_sign observation] QADLITTL: Le max-neg-y={:.3e}, Ge max-pos-y={:.3e}",
        le_max_neg, ge_max_pos
    );

    assert!(
        le_max_neg < TOL,
        "Le 双対に負の値: max_neg={:.3e}",
        le_max_neg
    );
    assert!(
        ge_max_pos < TOL,
        "Ge 双対に正の値: max_pos={:.3e}",
        ge_max_pos
    );

    let view = make_view(&qp);
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
        gap,
        TOL,
    );
    assert!(
        cert.is_ok(),
        "QADLITTL 最適解は prove_optimal が Ok を返すべき: {:?}",
        cert.err()
    );
}

// ── active upper bound sentinel ──────────────────────────────────────────────

/// box QP を解いて最適解を返す。
/// bound_duals=[z_lb, z_ub], z_ub>0 (active upper bound 実証済み)。
fn solve_box_qp() -> otspot::SolverResult {
    let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[2.0_f64], 1, 1).unwrap();
    let a = CscMatrix::new(0, 1);
    let prob = QpProblem::new(q, vec![-20.0], a, vec![], vec![(0.0, 5.0)], vec![]).unwrap();
    let result = solve_qp(&prob);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "box QP must solve to Optimal"
    );
    result
}

fn box_qp_problem() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[2.0_f64], 1, 1).unwrap();
    let a = CscMatrix::new(0, 1);
    QpProblem::new(q, vec![-20.0], a, vec![], vec![(0.0, 5.0)], vec![]).unwrap()
}

/// box QP 真の最適解: prove_optimal が Ok を返す (active ub sentinel baseline)。
#[test]
fn prove_optimal_accepts_active_ub_box_qp() {
    let qp = box_qp_problem();
    let result = solve_box_qp();
    // z_ub > 0 の経路を通る実証
    assert!(result.bound_duals.len() >= 2, "bound_duals len must be >=2");
    let z_ub = result.bound_duals[1];
    assert!(z_ub > 1.0, "z_ub at active ub must be >0, got {z_ub}");

    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
        gap,
        TOL,
    );
    assert!(
        cert.is_ok(),
        "box QP active-ub 最適解は Ok を返すべき: {:?}",
        cert.err()
    );
}

/// z_ub を符号反転 → prove_optimal は Err(dual_sign) を返す。
///
/// **active ub sentinel load-bearing**: dual_sign の z_ub ≥ 0 チェックを外すと FAIL。
/// 現 P1 バグ修正 (z_ub <= 0 → >= 0) で z_ub > 0 経路が正しく検証されることを実証。
#[test]
fn prove_optimal_rejects_negated_z_ub_box_qp() {
    let qp = box_qp_problem();
    let result = solve_box_qp();
    assert!(result.bound_duals.len() >= 2, "bound_duals len must be >=2");

    // z_ub を符号反転 (ub active で z_ub > 0 が正、負にすると violation)
    let mut z_bad = result.bound_duals.clone();
    let z_ub_orig = z_bad[1];
    z_bad[1] = -z_ub_orig;
    assert!(z_ub_orig > 0.0, "前提: z_ub_orig > 0 (実測確認済み)");

    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    let gap = result.duality_gap_rel.unwrap_or(0.0);
    let cert = prove_optimal(
        &view,
        &result.solution,
        &result.dual_solution,
        &z_bad,
        gap,
        TOL,
    );
    assert!(cert.is_err(), "z_ub 符号反転は Err を返すべき");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"dual_sign"),
        "dual_sign が failing_conditions に含まれるべき (z_ub < 0 violation): {:?}\n\
        dual_sign_violation={:.3e}, z_ub_orig={z_ub_orig:.3e}",
        err.failing_conditions,
        err.dual_sign_violation,
    );
}

// ── isolated dual_sign sentinel ──────────────────────────────────────────────

/// dual_sign 単独 sentinel: stationarity≈0 を保ったまま dual 符号だけ違反させる。
///
/// ## 構成
/// `min 0` (c=0, Q=0), constraints: x≤1 (Le), −x≤−1 (Le), bounds=(−∞,∞)
///
/// 真の双対: y=[v, v] (stationarity: 0 + v·1 + v·(−1) = 0 ✓)
/// 偽の双対: y=[−v, −v] (Le の符号規約違反)
///
/// stationarity: 0 + (−v)·1 + (−v)·(−1) = −v + v = 0 ✓
/// complementarity: |y·slack| = |(−v)·0| = 0 ✓ (両制約 active, slack=0)
/// dual_sign: (−v)/(1+v) > tol ✗
///
/// これは "prove_optimal から dual_sign 項を外すと誤って Ok" を実証する:
/// stat=0, pres=0, bviol=0, comp=0, gap=0 → dual_sign のみが棄却の根拠。
#[test]
fn prove_optimal_dual_sign_isolated_stationarity_zero_cancelling_le() {
    // A = [[1], [-1]] (2 Le constraints), b = [1, -1], c = [0], Q = 0
    let q = CscMatrix::new(1, 1);
    let a = CscMatrix::from_triplets(&[0usize, 1], &[0, 0], &[1.0_f64, -1.0], 2, 1).unwrap();
    let c = vec![0.0_f64];
    let b = vec![1.0_f64, -1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let ct = vec![ConstraintType::Le, ConstraintType::Le];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };

    let x = vec![1.0_f64]; // 両制約 active (slack = 0)
    let v = 0.1_f64; // dual magnitude (well above TOL=1e-5)

    // 正当な双対: stat=0, dsign=0
    let y_good = vec![v, v];
    let cert_good = prove_optimal(&view, &x, &y_good, &[], 0.0, TOL);
    assert!(
        cert_good.is_ok(),
        "正当な双対は Ok を返すべき: {:?}",
        cert_good.err()
    );

    // 符号反転双対: stat=0 (キャンセル), comp=0 (active), dsign>tol
    let y_bad = vec![-v, -v];
    let cert_bad = prove_optimal(&view, &x, &y_bad, &[], 0.0, TOL);
    assert!(
        cert_bad.is_err(),
        "符号反転双対 (stationarity=0 保持) は Err を返すべき"
    );
    let err = cert_bad.unwrap_err();
    assert_eq!(
        err.failing_conditions,
        vec!["dual_sign"],
        "dual_sign のみが failing_conditions に含まれるべき (stationarity は 0 で pass): {:?}\n\
        stat={:.3e}, pres={:.3e}, bviol={:.3e}, comp={:.3e}, dsign={:.3e}, gap={:.3e}",
        err.failing_conditions,
        err.stationarity_rel,
        err.primal_residual_rel,
        err.bound_violation,
        err.complementarity_rel,
        err.dual_sign_violation,
        err.duality_gap_rel,
    );
}

// ── per-condition mutation sentinels ────────────────────────────────────────
//
// 各 KKT 条件を単独で tol 超過させ、他の 5 条件は pass する点を構成する。
// prove_optimal が各条件を独立に検出することを実証 (no-op 相当の partial 実装 sentinel)。

/// stationarity のみ fail: `min x^2, x≤−1 (Le)`, x=−1.5, y=0 (勾配とのミスマッチ)
#[test]
fn mutation_only_stationarity_fails() {
    let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[2.0_f64], 1, 1).unwrap();
    let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0_f64], 1, 1).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0],
        a,
        vec![-1.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    // x=-1.5 (Le: -1.5 ≤ -1 ✓), y=0 (stationarity: 2*(-1.5)+0+0 = -3 ≠ 0)
    // pres=0, bviol=0, comp=|0*(-1-(-1.5))|=0, dsign=0, gap=0
    let cert = prove_optimal(&view, &[-1.5], &[0.0], &[], 0.0, TOL);
    assert!(cert.is_err(), "stationarity violation must fail");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"stationarity"),
        "stationarity must be in failing_conditions: {:?}",
        err.failing_conditions
    );
    // primal, bviol, comp, dsign, gap は pass を確認
    assert!(
        err.primal_residual_rel < TOL,
        "pres should pass, got {:.3e}",
        err.primal_residual_rel
    );
    assert!(err.bound_violation < TOL, "bviol should pass");
    assert!(
        err.complementarity_rel < TOL,
        "comp should pass, got {:.3e}",
        err.complementarity_rel
    );
    assert!(err.dual_sign_violation < TOL, "dsign should pass");
    assert!(err.duality_gap_rel < TOL, "gap should pass");
}

/// primal_feasibility のみ fail: `min x^2, x≤−1 (Le)`, x=0, y=0 (制約違反)
#[test]
fn mutation_only_primal_feas_fails() {
    let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[2.0_f64], 1, 1).unwrap();
    let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0_f64], 1, 1).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0],
        a,
        vec![-1.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    // x=0 (Le: 0 ≤ -1 違反!), y=0 (stat: Q*0+0+0=0 ✓), z=[]
    // comp=|0*(-1-0)|=0 ✓, dsign=0 ✓, gap=0 ✓
    let cert = prove_optimal(&view, &[0.0], &[0.0], &[], 0.0, TOL);
    assert!(cert.is_err(), "primal feasibility violation must fail");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"primal_feasibility"),
        "primal_feasibility must fail: {:?}",
        err.failing_conditions
    );
    assert!(
        err.stationarity_rel < TOL,
        "stat should pass, got {:.3e}",
        err.stationarity_rel
    );
    assert!(err.bound_violation < TOL, "bviol should pass");
    assert!(err.complementarity_rel < TOL, "comp should pass");
    assert!(err.dual_sign_violation < TOL, "dsign should pass");
    assert!(err.duality_gap_rel < TOL, "gap should pass");
}

/// bound_feasibility のみ fail: `min 0, bounds [1, 2]`, x=3 (ub 超過)
#[test]
fn mutation_only_bound_feas_fails() {
    let q = CscMatrix::new(1, 1);
    let a = CscMatrix::new(0, 1);
    let qp = QpProblem::new(q, vec![0.0], a, vec![], vec![(1.0, 2.0)], vec![]).unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    // bounds=[(1,2)]: n_lb=1, n_ub=1 → z=[z_lb, z_ub].
    // Q=0, c=0, no constraints → stationarity = -z_lb + z_ub. Use z=[0,0] (stat=0 ✓).
    // x=3 (ub=2 超過: bviol=1), pres=0 ✓, comp=|0*(3-1)|+|0*(2-3)|=0 ✓, dsign=0 ✓
    let cert = prove_optimal(&view, &[3.0], &[], &[0.0, 0.0], 0.0, TOL);
    assert!(cert.is_err(), "bound feasibility violation must fail");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"bound_feasibility"),
        "bound_feasibility must fail: {:?}",
        err.failing_conditions
    );
    assert!(err.stationarity_rel < TOL, "stat should pass");
    assert!(err.primal_residual_rel < TOL, "pres should pass");
    assert!(err.complementarity_rel < TOL, "comp should pass");
    assert!(err.dual_sign_violation < TOL, "dsign should pass");
    assert!(err.duality_gap_rel < TOL, "gap should pass");
}

/// complementarity のみ fail: QP where inactive constraint has non-zero y.
///
/// `min (x1−1)^2 + (x2−1)^2, x1+x2 ≤ 3 (Le)`, no bounds (free).
/// 制約 inactive (Ax = 1.5 < 3, slack = 1.5 > 0) だが y=0.5 ≠ 0 → comp fail。
/// stationarity: Q*x + c + A^T*y = 0 を満たす x を解析的に選択。
///   stat_j = 0 ⟺ 2(x_j − 1) + y = 0 ⟺ x_j = 1 − y/2 = 0.75
#[test]
fn mutation_only_complementarity_fails() {
    // Q = diag(2, 2), c = [-2, -2], A = [1, 1] (Le ≤ 3)
    let q = CscMatrix::from_triplets(&[0usize, 1], &[0, 1], &[2.0_f64, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0usize, 0], &[0, 1], &[1.0_f64, 1.0], 1, 2).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-2.0, -2.0],
        a,
        vec![3.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    // x = [0.75, 0.75]: stat = 2*0.75−2+0.5 = 0 for each component ✓
    // Ax = 1.5 ≤ 3 (inactive, slack = 1.5)
    // comp = |y*slack| = |0.5 * 1.5| = 0.75 > TOL ✗
    // dsign: y=0.5 for Le → y ≥ 0 ✓
    let cert = prove_optimal(&view, &[0.75, 0.75], &[0.5], &[], 0.0, TOL);
    assert!(cert.is_err(), "complementarity violation must fail");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"complementarity"),
        "complementarity must fail: {:?}",
        err.failing_conditions
    );
    assert!(
        err.stationarity_rel < TOL,
        "stat should pass, got {:.3e}",
        err.stationarity_rel
    );
    assert!(err.primal_residual_rel < TOL, "pres should pass");
    assert!(err.bound_violation < TOL, "bviol should pass");
    assert!(
        err.dual_sign_violation < TOL,
        "dsign should pass, got {:.3e}",
        err.dual_sign_violation
    );
    assert!(err.duality_gap_rel < TOL, "gap should pass");
}

/// dual_sign のみ fail (per-condition mutation): isolated sentinel と同一構成を利用。
///
/// `prove_optimal_dual_sign_isolated_stationarity_zero_cancelling_le` が
/// dual_sign 単独で fail することを mutation sentinel として再確認。
#[test]
fn mutation_only_dual_sign_fails() {
    let q = CscMatrix::new(1, 1);
    let a = CscMatrix::from_triplets(&[0usize, 1], &[0, 0], &[1.0_f64, -1.0], 2, 1).unwrap();
    let qp = QpProblem::new(
        q,
        vec![0.0],
        a,
        vec![1.0, -1.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY)],
        vec![ConstraintType::Le, ConstraintType::Le],
    )
    .unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    let x = vec![1.0_f64];
    let v = 0.1_f64;
    let y_bad = vec![-v, -v]; // stat cancels: (-v)*1 + (-v)*(-1) = 0
    let cert = prove_optimal(&view, &x, &y_bad, &[], 0.0, TOL);
    assert!(cert.is_err());
    let err = cert.unwrap_err();
    assert_eq!(
        err.failing_conditions,
        vec!["dual_sign"],
        "dual_sign のみ fail: {:?}",
        err.failing_conditions
    );
}

/// duality_gap のみ fail: 完璧な KKT 点に gap > tol を渡す。
#[test]
fn mutation_only_duality_gap_fails() {
    // min -x, s.t. x ≤ 1 (Le), bounds [0, ∞) — optimal x=1, y=1
    let q = CscMatrix::new(1, 1);
    let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0_f64], 1, 1).unwrap();
    let qp = QpProblem::new(
        q,
        vec![-1.0],
        a,
        vec![1.0],
        vec![(0.0, f64::INFINITY)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let view = ProblemView {
        q: &qp.q,
        a: &qp.a,
        c: &qp.c,
        b: &qp.b,
        bounds: &qp.bounds,
        constraint_types: &qp.constraint_types,
        eliminated_cols: &[],
    };
    // bounds=[(0,inf)]: n_lb=1, n_ub=0 → z=[z_lb].
    // x=1, y=1 (Le active, y≥0 ✓), z_lb=0 (lb inactive: x=1 > lb=0).
    // stat: Q*1 + (-1) + 1*1 - z_lb = -1+1-0 = 0 ✓
    // pres: Ax - b = 1-1 = 0 ✓, comp: |1*0| + |0*(1-0)| = 0 ✓, dsign: y=1≥0, z_lb=0≥0 ✓
    let large_gap = 1.0; // gap >> TOL
    let cert = prove_optimal(&view, &[1.0], &[1.0], &[0.0], large_gap, TOL);
    assert!(cert.is_err(), "duality_gap violation must fail");
    let err = cert.unwrap_err();
    assert!(
        err.failing_conditions.contains(&"duality_gap"),
        "duality_gap must fail: {:?}",
        err.failing_conditions
    );
    assert!(
        err.stationarity_rel < TOL,
        "stat should pass, got {:.3e}",
        err.stationarity_rel
    );
    assert!(err.primal_residual_rel < TOL, "pres should pass");
    assert!(err.bound_violation < TOL, "bviol should pass");
    assert!(
        err.complementarity_rel < TOL,
        "comp should pass, got {:.3e}",
        err.complementarity_rel
    );
    assert!(err.dual_sign_violation < TOL, "dsign should pass");
}
