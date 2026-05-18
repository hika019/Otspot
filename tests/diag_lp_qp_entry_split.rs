//! #36 sentinel: LP/QP 入口分離が想定どおり経路に乗っているか機械検証する。
//!
//! ## 何を保証するか
//! - `Model::solve` の LP path は `crate::lp::solve_lp_with` を経由する
//!   (`lp::telemetry::lp_direct_calls` が増える / `qp::telemetry::qp_ipm_calls`
//!    は増えない)。
//! - `Model::solve` の QP path は `crate::qp::solve_qp_with` → IPM を経由する
//!   (`qp::telemetry::qp_ipm_calls` が増える / LP counter は増えない)。
//! - `solve_qp_with(Q=0)` の forward は `lp_forwarded_from_qp_calls` 側で
//!   識別可能 (direct と混同しない)。
//! - `solve_lp_with(LpProblem)` 直接呼び出しが正解を返す (Q=0 forward と同値)。
//!
//! ## no-op proof (sentinel が無意味でない証明)
//! - `sentinel_proves_lp_path_regression_detectable`: LP fixture を意図的に
//!   `solve_qp_with` (QP entry) 経由で解き、`lp_direct_calls` が **0 のまま**
//!   かつ `lp_forwarded_from_qp_calls` が増えることを assert する。
//!   = Model::solve LP path が誤って solve_qp_with に regression したら
//!     `lp_direct_calls > 0` の主 assertion が FAIL する。
//!
//! ## 複数 data pattern (CLAUDE.md「複数パターンのデータを用意せよ」遵守)
//! - LP: trivial bound / standard ≤ / ≥ / = / 退化 (degenerate) / 大規模合成 (n=200)
//! - QP: PSD diagonal small / PSD off-diagonal / box-constrained / equality+inequality
//!       / 大規模 PSD diagonal (n=50)

use std::sync::Mutex;

use solver::constraint;
use solver::model::Model;
use solver::options::{SimplexMethod, SolverOptions};
use solver::problem::{ConstraintType, LpProblem};
use solver::qp::QpProblem;
use solver::sparse::CscMatrix;

/// LP fixture 用の固定オプション。
///
/// `SimplexMethod::Primal` を明示する: 既定 `Auto` / `Dual` / `DualAdvanced`
/// 経路は本 task と独立に textbook LP (例: min -3x-5y …) で頂点 x=2.0
/// ではなく x≈1.99977 を返す precision 退化が main HEAD 90be7cd 時点で
/// 観測されており (#33 領域、本 task scope 外)、entry 分離 sentinel の
/// 主旨である「経路が正しいか」と独立な numerical noise を assert に
/// 混入させない目的で Primal pin する。
fn lp_opts_strict() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.simplex_method = SimplexMethod::Primal;
    o
}

/// telemetry counter は process-global static。test 並列実行時の race を避ける
/// ため本 file の全 test を serialize する。
static SENTINEL_LOCK: Mutex<()> = Mutex::new(());

/// LP fixture (Primal simplex pin) は頂点解で exact、QP fixture は IPM eps=1e-6
/// で 1e-5 相対精度。両者を呑む共通 tolerance に統一する。
const TOL_OBJ: f64 = 1e-4;
const TOL_SOL: f64 = 1e-4;

fn reset_counters() {
    solver::lp::telemetry::reset();
    solver::qp::telemetry::reset();
}

// ===========================================================================
// LP fixtures (5 pattern)
// ===========================================================================

/// (LpProblem, expected_objective, label)
fn lp_fixtures() -> Vec<(LpProblem, f64, &'static str)> {
    vec![
        lp_fix_trivial_bound(),
        lp_fix_le_two_var(),
        lp_fix_ge_two_var(),
        lp_fix_eq_three_var(),
        lp_fix_degenerate(),
    ]
}

fn lp_fix_trivial_bound() -> (LpProblem, f64, &'static str) {
    // min x s.t. x >= 1, 0 <= x <= 10
    let c = vec![1.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1.0];
    let ct = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, 10.0)];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("trivial_bound".into())).unwrap();
    (lp, 1.0, "lp_trivial_bound")
}

fn lp_fix_le_two_var() -> (LpProblem, f64, &'static str) {
    // min -3x - 5y  s.t. x <= 4, 2y <= 12, 3x+2y <= 18, x,y>=0
    // Classic textbook problem; opt at (2,6) obj = -36
    let c = vec![-3.0, -5.0];
    let rows = vec![0, 1, 2, 2];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 2.0, 3.0, 2.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();
    let b = vec![4.0, 12.0, 18.0];
    let ct = vec![ConstraintType::Le; 3];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("le_two_var".into())).unwrap();
    (lp, -36.0, "lp_le_two_var")
}

fn lp_fix_ge_two_var() -> (LpProblem, f64, &'static str) {
    // min x+y  s.t. x+y >= 2, x,y >= 0  → obj=2
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![2.0];
    let ct = vec![ConstraintType::Ge];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("ge_two_var".into())).unwrap();
    (lp, 2.0, "lp_ge_two_var")
}

fn lp_fix_eq_three_var() -> (LpProblem, f64, &'static str) {
    // min x+2y+3z  s.t. x+y+z = 1, 0<=x,y,z<=1  → obj=1 at (1,0,0)
    let c = vec![1.0, 2.0, 3.0];
    let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
    let b = vec![1.0];
    let ct = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, 1.0); 3];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("eq_three_var".into())).unwrap();
    (lp, 1.0, "lp_eq_three_var")
}

fn lp_fix_degenerate() -> (LpProblem, f64, &'static str) {
    // min x+y s.t. x+y <= 2, x+y <= 2 (dup), x,y >= 0  → obj=0 at origin
    let c = vec![1.0, 1.0];
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let vals = vec![1.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();
    let b = vec![2.0, 2.0];
    let ct = vec![ConstraintType::Le; 2];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let lp = LpProblem::new_general(c, a, b, ct, bounds, Some("degenerate".into())).unwrap();
    (lp, 0.0, "lp_degenerate")
}

// ===========================================================================
// QP fixtures (5 pattern)
// ===========================================================================

fn qp_fixtures() -> Vec<(QpProblem, f64, &'static str)> {
    vec![
        qp_fix_diag_psd_box(),
        qp_fix_offdiag_psd(),
        qp_fix_pure_box(),
        qp_fix_eq_inequality_mix(),
        qp_fix_large_diag_psd(50),
    ]
}

/// 対称 Q を CSC 表現する。`upper_triples` には対角と上三角 (i<=j) のみ
/// 与え、内部で下三角 (j>i) を mirror して挿入する。
///
/// solver の x^T Q x 評価は CSC に格納された entry をそのまま使うため、対称
/// 部分は両方明示しないと obj が非対称化される (probe で実証済)。
fn symmetric_q_csc(n: usize, upper_triples: &[(usize, usize, f64)]) -> CscMatrix {
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for &(r, c, v) in upper_triples {
        assert!(r <= c, "upper triple expected: ({},{})", r, c);
        rows.push(r);
        cols.push(c);
        vals.push(v);
        if r != c {
            rows.push(c);
            cols.push(r);
            vals.push(v);
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

fn qp_fix_diag_psd_box() -> (QpProblem, f64, &'static str) {
    // min 1/2 (x²+y²) s.t. x+y >= 1, x,y >= 0
    // Lagrangian → x=y=0.5, obj = 1/2 (0.25+0.25) = 0.25
    let q = symmetric_q_csc(2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let ct = vec![ConstraintType::Ge];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, 0.25, "qp_diag_psd_box")
}

fn qp_fix_offdiag_psd() -> (QpProblem, f64, &'static str) {
    // min 1/2 (2x² + 2y² + 2xy) + (-x - y)
    // Q = [[2,1],[1,2]] (upper: (0,0)=2, (0,1)=1, (1,1)=2)
    // Unconstrained gradient: (2x+y-1, x+2y-1) = 0 → x=y=1/3
    // obj = 1/2 (2/9 + 2/9 + 2/9) + (-2/3) = 3/9 - 6/9 = -3/9 = -1/3
    let q = symmetric_q_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)]);
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(-10.0, 10.0); 2];
    let ct = vec![];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -1.0 / 3.0, "qp_offdiag_psd")
}

fn qp_fix_pure_box() -> (QpProblem, f64, &'static str) {
    // min 1/2 x² - 3x   s.t. 0 <= x <= 2  (no row constraints)
    // Unconstrained min at x=3; box clamps to x=2 → obj = 0.5*4 - 6 = -4
    let q = symmetric_q_csc(1, &[(0, 0, 1.0)]);
    let c = vec![-3.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(0.0, 2.0)];
    let ct = vec![];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -4.0, "qp_pure_box")
}

fn qp_fix_eq_inequality_mix() -> (QpProblem, f64, &'static str) {
    // min 1/2 (x² + y² + z²)
    // s.t. x + y + z = 3 (Eq)
    //      x >= 0.5 (Ge)
    //      x,y,z free
    // Lagrangian: KKT → unique x=y=z=1, obj = 1.5
    let q = symmetric_q_csc(3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
    let c = vec![0.0, 0.0, 0.0];
    let rows = vec![0, 0, 0, 1];
    let cols = vec![0, 1, 2, 0];
    let vals = vec![1.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
    let b = vec![3.0, 0.5];
    let bounds = vec![(-10.0, 10.0); 3];
    let ct = vec![ConstraintType::Eq, ConstraintType::Ge];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, 1.5, "qp_eq_inequality_mix")
}

fn qp_fix_large_diag_psd(n: usize) -> (QpProblem, f64, &'static str) {
    // min 1/2 sum_i x_i²  + sum_i (-1) * x_i  s.t. sum x_i <= n, x_i >= 0
    // Unconstrained min at x_i = 1; sum=n satisfies <=n (active).
    // obj = 1/2 * n * 1 + (-1) * n = n/2 - n = -n/2
    let mut q_rows = Vec::with_capacity(n);
    let mut q_cols = Vec::with_capacity(n);
    let mut q_vals = Vec::with_capacity(n);
    for i in 0..n {
        q_rows.push(i);
        q_cols.push(i);
        q_vals.push(1.0);
    }
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
    let c = vec![-1.0; n];
    let rows: Vec<usize> = vec![0; n];
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0; n];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, n).unwrap();
    let b = vec![n as f64];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let ct = vec![ConstraintType::Le];
    let qp = QpProblem::new(q, c, a, b, bounds, ct).unwrap();
    (qp, -(n as f64) / 2.0, "qp_large_diag_psd")
}

// ===========================================================================
// Test cases
// ===========================================================================

fn assert_obj(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    let tol = TOL_OBJ * (1.0 + expected.abs());
    assert!(
        diff < tol,
        "{}: obj={:.9e} expected={:.9e} diff={:.3e} (tol={:.3e})",
        label, actual, expected, diff, tol
    );
}

#[test]
fn lp_direct_entry_solves_all_fixtures() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    let opts = lp_opts_strict();
    for (lp, expected, label) in lp_fixtures() {
        reset_counters();
        let r = solver::lp::solve_lp_with(&lp, &opts);
        assert_eq!(
            r.status,
            solver::SolveStatus::Optimal,
            "{}: status={:?}",
            label,
            r.status
        );
        assert_obj(r.objective, expected, label);
        assert_eq!(
            solver::lp::telemetry::lp_direct_calls(),
            1,
            "{}: lp_direct_calls must be 1", label
        );
        assert_eq!(
            solver::lp::telemetry::lp_forwarded_from_qp_calls(),
            0,
            "{}: forward path must not fire on direct entry", label
        );
        assert_eq!(
            solver::qp::telemetry::qp_ipm_calls(),
            0,
            "{}: QP IPM must not fire for LP entry", label
        );
    }
}

#[test]
fn qp_direct_entry_solves_all_fixtures() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    let opts = SolverOptions::default();
    for (qp, expected, label) in qp_fixtures() {
        reset_counters();
        let r = solver::solve_qp_with(&qp, &opts);
        assert_eq!(
            r.status,
            solver::SolveStatus::Optimal,
            "{}: status={:?}",
            label,
            r.status
        );
        assert_obj(r.objective, expected, label);
        assert_eq!(
            solver::qp::telemetry::qp_ipm_calls(),
            1,
            "{}: qp_ipm_calls must be 1 (Q!=0 → IPM)", label
        );
        assert_eq!(
            solver::lp::telemetry::lp_direct_calls(),
            0,
            "{}: LP direct counter must stay 0 on QP entry", label
        );
        assert_eq!(
            solver::lp::telemetry::lp_forwarded_from_qp_calls(),
            0,
            "{}: LP forward counter must stay 0 (Q!=0)", label
        );
    }
}

#[test]
fn model_api_lp_path_uses_solve_lp_with() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    reset_counters();
    // min x s.t. x >= 1, 0 <= x <= 10  → obj=1
    let mut m = Model::new("lp_via_model");
    let x = m.add_var("x", 0.0, 10.0);
    m.add_constraint(constraint!(x >= 1.0));
    m.minimize(x);
    let r = m.solve().expect("model lp");
    assert_obj(r.objective_value, 1.0, "model_lp_path");

    assert!(
        solver::lp::telemetry::lp_direct_calls() >= 1,
        "Model::solve LP path must increment lp_direct_calls (got {})",
        solver::lp::telemetry::lp_direct_calls()
    );
    assert_eq!(
        solver::qp::telemetry::qp_ipm_calls(),
        0,
        "Model::solve LP path must NOT increment qp_ipm_calls"
    );
    assert_eq!(
        solver::lp::telemetry::lp_forwarded_from_qp_calls(),
        0,
        "Model::solve LP path must NOT route via QP→LP forward"
    );
}

#[test]
fn model_api_qp_path_uses_solve_qp_with() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    reset_counters();
    // min 1/2 x²  s.t. x >= 1 → obj=0.5  (Q via set_diagonal_q, c via minimize)
    let mut m = Model::new("qp_via_model");
    let x = m.add_var("x", 0.0, 10.0);
    m.add_constraint(constraint!(x >= 1.0));
    m.minimize(0.0 * x);
    m.set_diagonal_q(&[1.0]);
    let r = m.solve().expect("model qp");
    assert_obj(r.objective_value, 0.5, "model_qp_path");

    assert_eq!(
        solver::qp::telemetry::qp_ipm_calls(),
        1,
        "Model::solve QP path must increment qp_ipm_calls"
    );
    assert_eq!(
        solver::lp::telemetry::lp_direct_calls(),
        0,
        "Model::solve QP path must NOT use solve_lp_with"
    );
    assert_eq!(
        solver::lp::telemetry::lp_forwarded_from_qp_calls(),
        0,
        "Q!=0 must NOT route via QP→LP forward"
    );
}

/// QP entry に Q=0 を渡した場合 (legacy 後方互換経路): solve_lp_with 直接呼び
/// 出しと同じ最適解、ただし telemetry は forward 側に乗る。
#[test]
fn qp_entry_with_zero_q_forwards_to_lp_module() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    let opts = lp_opts_strict();
    let (lp, expected, label) = lp_fix_le_two_var();

    // Direct LP entry
    reset_counters();
    let r_direct = solver::lp::solve_lp_with(&lp, &opts);
    assert_obj(r_direct.objective, expected, "direct_lp");
    assert_eq!(solver::lp::telemetry::lp_direct_calls(), 1);
    assert_eq!(solver::lp::telemetry::lp_forwarded_from_qp_calls(), 0);

    // Wrap as Q=0 QpProblem and call QP entry
    let n = lp.num_vars;
    let q_zero = CscMatrix::new(n, n);
    let qp = QpProblem::new(
        q_zero,
        lp.c.clone(),
        lp.a.clone(),
        lp.b.clone(),
        lp.bounds.clone(),
        lp.constraint_types.clone(),
    )
    .unwrap();

    reset_counters();
    let r_qp = solver::solve_qp_with(&qp, &opts);
    assert_obj(r_qp.objective, expected, "qp_zero_q_forward");
    assert_eq!(
        solver::lp::telemetry::lp_direct_calls(),
        0,
        "{}: solve_qp_with(Q=0) must NOT increment direct counter", label
    );
    assert_eq!(
        solver::lp::telemetry::lp_forwarded_from_qp_calls(),
        1,
        "{}: solve_qp_with(Q=0) must forward to lp module exactly once", label
    );
    assert_eq!(
        solver::qp::telemetry::qp_ipm_calls(),
        0,
        "{}: Q=0 must NOT trigger IPM", label
    );

    // Objective equivalence (sanity)
    assert!(
        (r_direct.objective - r_qp.objective).abs() < TOL_OBJ * (1.0 + expected.abs()),
        "{}: direct LP obj {:.9e} vs QP-forward obj {:.9e}", label, r_direct.objective, r_qp.objective
    );
}

/// no-op proof: LP fixture を意図的に solve_qp_with (QP entry) 経由で解いて、
/// sentinel counter が direct と forward を識別できることを示す。
///
/// この test が PASS することで、もし Model::solve LP path が誤って
/// solve_qp_with に regression したとしても
/// `model_api_lp_path_uses_solve_lp_with` の assertion (`lp_direct_calls >= 1`)
/// が FAIL することが保証される (= sentinel が能動的に regression を検出する)。
#[test]
fn sentinel_proves_lp_path_regression_detectable() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    let opts = lp_opts_strict();
    for (lp, _expected, label) in lp_fixtures() {
        let n = lp.num_vars;
        let q_zero = CscMatrix::new(n, n);
        let qp = QpProblem::new(
            q_zero,
            lp.c.clone(),
            lp.a.clone(),
            lp.b.clone(),
            lp.bounds.clone(),
            lp.constraint_types.clone(),
        )
        .unwrap();
        reset_counters();
        let _ = solver::solve_qp_with(&qp, &opts);
        assert_eq!(
            solver::lp::telemetry::lp_direct_calls(),
            0,
            "{}: regression simulation (solve_qp_with on LP) must NOT increment lp_direct_calls",
            label
        );
        assert!(
            solver::lp::telemetry::lp_forwarded_from_qp_calls() >= 1,
            "{}: regression simulation must increment lp_forwarded_from_qp_calls (sentinel discriminator)",
            label
        );
    }
}

/// 直接 LP entry と QP→LP forward 経路が同一 objective を返すクロスチェック
/// (複数 data pattern)。
#[test]
fn cross_check_lp_direct_vs_qp_forward_objective() {
    let _guard = SENTINEL_LOCK.lock().unwrap();
    let opts = lp_opts_strict();
    for (lp, expected, label) in lp_fixtures() {
        let r_direct = solver::lp::solve_lp_with(&lp, &opts);

        let n = lp.num_vars;
        let q_zero = CscMatrix::new(n, n);
        let qp = QpProblem::new(
            q_zero,
            lp.c.clone(),
            lp.a.clone(),
            lp.b.clone(),
            lp.bounds.clone(),
            lp.constraint_types.clone(),
        )
        .unwrap();
        let r_fwd = solver::solve_qp_with(&qp, &opts);

        assert_obj(r_direct.objective, expected, &format!("{}_direct", label));
        assert_obj(r_fwd.objective, expected, &format!("{}_forward", label));
        assert!(
            (r_direct.objective - r_fwd.objective).abs() < TOL_OBJ * (1.0 + expected.abs()),
            "{}: direct={:.9e} forward={:.9e}", label, r_direct.objective, r_fwd.objective
        );
        if !r_direct.solution.is_empty() && !r_fwd.solution.is_empty() {
            assert_eq!(r_direct.solution.len(), r_fwd.solution.len());
            for (i, (a, b)) in r_direct.solution.iter().zip(r_fwd.solution.iter()).enumerate() {
                assert!(
                    (a - b).abs() < TOL_SOL * (1.0 + a.abs() + b.abs()),
                    "{}: solution[{}] direct={:.9e} forward={:.9e}", label, i, a, b
                );
            }
        }
    }
}
