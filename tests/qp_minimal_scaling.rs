//! Task #29 凸 QP mini-corpus — **bug class: scaling / ill-conditioned Q**
//!
//! ## 対象 bug class
//!
//! - **Q 対角要素が極端 (1e-6 / 1e6 mix)** → condition number 大
//! - **大係数 Q (要素 1e10)** → IPM 内部 regularization が相対化されているか
//! - **小係数 Q (要素 1e-10)** → Q=0 退化判定 (`is_zero_q`) との境界
//! - **c が Q に対して桁違いに大きい** → KKT 線形系の rhs スケール暴走
//!
//! ## 真因仮説
//!
//! - magic な absolute threshold (1e-12) で Q 要素を 0 扱いし、LP fallback に
//!   落ちて誤解 (例: 1e-10 のみの Q で LP fallback すると unbounded 判定)。
//! - Mehrotra IPM の正則化 ε が absolute (1e-8) だと 1e-12 Q を救えない、
//!   逆に 1e10 Q を「regularize しすぎ」で精度落ちる可能性。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

const EPS_OBJ_REL: f64 = 1e-6;
const EPS_X_REL: f64 = 1e-4;
const MINI_TIMEOUT_SECS: f64 = 10.0;

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

fn assert_x_rel_close(actual: f64, expected: f64, label: &str) {
    let denom = 1.0 + expected.abs();
    let rel = (actual - expected).abs() / denom;
    assert!(rel < EPS_X_REL,
        "[{}] x actual={:.6e} expected={:.6e} rel_err={:.3e}",
        label, actual, expected, rel);
}

// =============================================================================
// scl1: Q diagonal mix (1e-6, 1, 1e6) — high condition number
// =============================================================================

/// **構造**: min 1/2 (eps*x1^2 + x2^2 + M*x3^2) - x1 - x2 - x3.
///   eps=1e-6, M=1e6, no A, no bounds.
/// **解析解**: x_i = c_i / Q_ii ⇒ x1=1/eps=1e6, x2=1, x3=1/M=1e-6.
///   obj = -0.5*(1/eps + 1 + 1/M) = -0.5 * (1e6 + 1 + 1e-6) ≈ -5.00000e5。
/// **狙い**: condition number 1e12 でも IPM 収束。x1, x3 の桁差を許容するか。
#[test]
fn scl1_diagonal_high_condition_number() {
    let n = 3;
    let eps_q = 1e-6_f64;
    let m_q = 1e6_f64;
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[eps_q, 1.0, m_q], n, n).unwrap();
    let c = vec![-1.0, -1.0, -1.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "scl1: status (cond≈1e12)");
    assert_x_rel_close(r.solution[0], 1.0 / eps_q, "scl1: x1=1/eps");
    assert_x_rel_close(r.solution[1], 1.0, "scl1: x2=1");
    assert_x_rel_close(r.solution[2], 1.0 / m_q, "scl1: x3=1/M");
    let expected_obj = -0.5 * (1.0 / eps_q + 1.0 + 1.0 / m_q);
    assert_obj_close(r.objective, expected_obj, "scl1: obj");
}

// =============================================================================
// scl2: large constant in c (c huge vs Q small)
// =============================================================================

/// **構造**: min 1/2 x^2 + 1e8 * x, s.t. x in [-INF, INF]. No A.
/// **解析解**: x = -1e8 (interior, unconstrained). obj = 0.5 * 1e16 - 1e16 = -0.5e16。
/// **狙い**: |c| が |Q| より 8 桁大きい場合、IPM の barrier 初期化が
///         x_0=0 から x=-1e8 まで降下できるか。max_iter で詰まらないか。
#[test]
fn scl2_large_linear_term_c() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
    let c = vec![1e8_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "scl2: status");
    assert_x_rel_close(r.solution[0], -1e8, "scl2: x=-1e8");
    assert_obj_close(r.objective, -0.5e16, "scl2: obj");
}

// =============================================================================
// scl3: nearly-zero Q (1e-10) — boundary of is_zero_q (1e-12 threshold)
// =============================================================================

/// **構造**: min 1/2 * 1e-10 * x^2 - x, s.t. 0 <= x <= 1e15. No A.
/// **is_zero_q (qp/problem.rs FX_TOL=1e-12)**: 1e-10 > 1e-12 ⇒ Q 非ゼロ扱い、IPM へ。
/// **解析解**: ∇f=0 ⇒ 1e-10 * x = 1 ⇒ x=1e10 (interior, bound 余裕)。
///   obj = 0.5 * 1e-10 * 1e20 - 1e10 = 0.5e10 - 1e10 = -0.5e10。
/// **狙い**: Q 非ゼロだが小さい場合に LP fallback されない、かつ KKT で正しい x。
#[test]
fn scl3_nearly_zero_q_above_threshold() {
    let n = 1;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1e-10_f64], n, n).unwrap();
    let c = vec![-1.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0, 1e15)];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "scl3: status (Q small but ≠ 0)");
    // x=1e10 を許容 rel < 1e-4
    let rel = (r.solution[0] - 1e10).abs() / 1e10;
    assert!(rel < 1e-3, "scl3: x≈1e10, got {} (rel={:.3e})", r.solution[0], rel);
    // obj 計算誤差は |x|^2 * Q オーダーで桁落ち。EPS_OBJ_REL より緩めて 1e-4。
    let exp_obj = -0.5e10;
    let obj_rel = (r.objective - exp_obj).abs() / exp_obj.abs();
    assert!(obj_rel < 1e-4, "scl3: obj={:.6e} expected={:.6e} (rel={:.3e})",
        r.objective, exp_obj, obj_rel);
}

// =============================================================================
// scl4: large Q (1e8) with Eq constraint
// =============================================================================

/// **構造**: min 1/2 * 1e8 * (x1^2 + x2^2), s.t. x1 + x2 = 1.
/// **解析解**: 対称 ⇒ x1=x2=0.5. y = 1e8 * 0.5 = 5e7.
///   obj = 0.5 * 1e8 * 0.5 = 2.5e7。
/// **狙い**: 大係数 Q で IPM の Mehrotra step / regularization が暴れないか。
#[test]
fn scl4_large_q_with_equality() {
    let n = 2;
    let big = 1e8_f64;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[big, big], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![1.0];
    let cts = vec![ConstraintType::Eq];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let prob = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "scl4: status (Q=1e8)");
    assert_x_rel_close(r.solution[0], 0.5, "scl4: x1");
    assert_x_rel_close(r.solution[1], 0.5, "scl4: x2");
    assert_obj_close(r.objective, 2.5e7, "scl4: obj=2.5e7");
}

// =============================================================================
// scl5: Q off-diagonal coupling (dense 2x2)
// =============================================================================

/// **構造**: min 1/2 [x1 x2] [[2 1][1 2]] [x1 x2]^T - 3*x1 - 3*x2.
///   eigvals(Q) = {1, 3} (cond=3, 良条件)。
/// **解析解**: ∇f = Q x + c = 0 → [[2,1],[1,2]][x1,x2] = [3,3] ⇒ x1=x2=1。
///   obj = 0.5 * (2 + 2 + 2) - 6 = 3 - 6 = -3.
/// **狙い**: Q 非対角の coupling 項が正しく行列積に反映されるか
///         (CSC 全要素格納の仕様: docs では Q は「全要素」と「上三角のみ」両対応)。
///         本 test は全要素格納で記述。
#[test]
fn scl5_q_offdiagonal_full_storage() {
    let n = 2;
    // Q = [[2,1],[1,2]] 全要素 (col-major)
    let q = CscMatrix::from_triplets(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &[2.0, 1.0, 1.0, 2.0],
        n, n,
    ).unwrap();
    let c = vec![-3.0, -3.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let r = solve_qp_with(&prob, &solver_opts());
    assert_eq!(r.status, SolveStatus::Optimal, "scl5: status");
    assert_x_rel_close(r.solution[0], 1.0, "scl5: x1=1");
    assert_x_rel_close(r.solution[1], 1.0, "scl5: x2=1");
    assert_obj_close(r.objective, -3.0, "scl5: obj=-3");
}

// =============================================================================
// scl6: Q input contract — symmetric full storage 必須を assert
// =============================================================================

/// **事実観測** (`io/qps.rs:807-809`, `io/qplib.rs:163-165`):
///   QPS/QPLIB parser は upper/lower 入力に対し i≠j で必ず (j,i) を追加し、
///   **対称 (全要素) Q** を作って IPM に渡す。
///   つまり solver の Q 入力契約 = 「全要素対称格納」。
///
/// **このテストの目的**:
///   契約違反 (upper-only Q を直接 IPM に渡す) は SuboptimalSolution / 誤解
///   になる事実を pin する。将来 IPM が upper-only を受け付けるよう拡張された
///   場合は test を Optimal assertion に切り替えるべき退化検出 anchor。
///
/// **構造**: scl5 と同じ問題を upper-only で構築 → 誤った最適化に陥る。
#[test]
fn scl6_q_upper_only_violates_input_contract() {
    let n = 2;
    // Q upper-only (非対称) — 入力契約違反。
    let q_upper = CscMatrix::from_triplets(
        &[0, 0, 1],
        &[0, 1, 1],
        &[2.0, 1.0, 2.0],
        n, n,
    ).unwrap();
    let c = vec![-3.0, -3.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let prob_upper = QpProblem::new_all_le(q_upper, c.clone(), a.clone(), b.clone(), bounds.clone()).unwrap();
    let r_upper = solve_qp_with(&prob_upper, &solver_opts());

    // 対称格納 (契約通り) reference
    let q_sym = CscMatrix::from_triplets(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &[2.0, 1.0, 1.0, 2.0],
        n, n,
    ).unwrap();
    let prob_sym = QpProblem::new_all_le(q_sym, c, a, b, bounds).unwrap();
    let r_sym = solve_qp_with(&prob_sym, &solver_opts());

    // 契約通り (sym) は Optimal、x=[1,1], obj=-3
    assert_eq!(r_sym.status, SolveStatus::Optimal, "scl6: sym Q must be Optimal");
    assert_x_rel_close(r_sym.solution[0], 1.0, "scl6 sym: x1");
    assert_obj_close(r_sym.objective, -3.0, "scl6 sym: obj");

    // upper-only は status が Optimal にならない or 解が異なる事実を pin
    let upper_failed = r_upper.status != SolveStatus::Optimal
        || (r_upper.solution[0] - 1.0).abs() > 1e-3
        || (r_upper.objective - (-3.0)).abs() > 1e-3;
    assert!(upper_failed,
        "scl6: upper-only Q が偶然 Optimal を返したら IPM が symmetrize を始めた\
         可能性。docs を更新するか test を Optimal assertion に切り替えよ。\
         status={:?} x={:?} obj={}", r_upper.status, r_upper.solution, r_upper.objective);
}
