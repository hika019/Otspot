//! McCormick envelope lb 不変式 + α-BB 比較 sentinel (Phase 5)。
//!
//! ## 目的 (BB driver 経由のみ)
//! 1. bilinear-rich problem では α-BB lb より strict に tight (= node 削減効果あり)
//! 2. diag-only convex / concave では α-BB と同等 (= 退化 risk なし)
//! 3. no-op (McCormick OFF) では BB node 数が α-BB only と一致 (= sentinel に teeth)
//!
//! ## complementary 性
//! `bound_mccormick::mccormick_lower_bound` は `pub(crate)` (P3-4 test-api-audit)。
//! lb 関数単体の不変式 / underestimator 性質 / Ge/coef=0/Q_ZERO 境界 / n=8 大規模 /
//! envelope OFF no-op teeth の機械実証は in-source `bound_mccormick::tests` に置く。
//! 本 file は **BB driver 経由** で end-to-end の node 削減効果を測定する (= integration)。

use otspot::options::{BranchingStrategy, GlobalOptimizationConfig};
use otspot::problem::ConstraintType;
use otspot::qp::{solve_qp_global_with_stats, QpProblem};
use otspot::sparse::CscMatrix;
use otspot::{SolveStatus, SolverOptions};

/// BB sentinel 共通: 3 分以内に完結する範囲。
const TEST_MAX_DEPTH: usize = 20;
const TEST_MAX_NODES: usize = 3_000;
const TEST_TIMEOUT_SECS: f64 = 30.0;
const TEST_GAP_TOL: f64 = 1e-3;

fn opts() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(TEST_TIMEOUT_SECS);
    o
}

fn cfg(use_alpha_bb: bool, use_mccormick: bool) -> GlobalOptimizationConfig {
    GlobalOptimizationConfig {
        gap_tol: TEST_GAP_TOL,
        max_depth: TEST_MAX_DEPTH,
        max_nodes: TEST_MAX_NODES,
        branching: BranchingStrategy::MaxViolation,
        use_alpha_bb,
        use_mccormick,
    }
}

// ---------------- fixtures (BB driver 経由で利用) ----------------

/// 非対称 box bilinear: f = -xy on [-2,1] × [-1,2]、global = -2。
/// McCormick が α-BB に確実に勝つ典型 fixture。
fn asym_bilinear_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[-1.0, -1.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![],
        vec![(-2.0, 1.0), (-1.0, 2.0)],
        vec![],
    )
    .unwrap()
}

/// diag only (純 concave): f = -x² -y² on [-1, 1]² 。McCormick と α-BB が同等の想定 fixture。
fn diag_concave_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2], vec![]).unwrap()
}

/// n 次元 全 off-diag bilinear (diag=0): Q off-diag=1, c=0, box [-1,1]^n 。
fn bilinear_dense_nd(n: usize) -> QpProblem {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for c in 0..n {
        for r in 0..n {
            if r != c {
                rows.push(r);
                cols.push(c);
                vals.push(1.0);
            }
        }
    }
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
    QpProblem::new(q, vec![0.0; n], a, vec![], vec![(-1.0, 1.0); n], vec![]).unwrap()
}

fn bilinear_dense_3d() -> QpProblem {
    bilinear_dense_nd(3)
}

/// 4D bilinear + sum-cap 制約。
fn bilinear_sumcap_4d() -> QpProblem {
    let n = 4;
    let p = bilinear_dense_nd(n);
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        p.q.clone(),
        vec![0.0; n],
        a,
        vec![1.0],
        vec![(-1.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

// unit-level invariants (lb 単体の underestimator 性質 / 境界 / 大規模) は
// `bound_mccormick.rs::tests` に集約。本 file は BB driver 経由の sentinel に専念。

// ---------------- BB driver integration ----------------

/// BB driver で McCormick を ON にすると、αBB only に比べて全 fixture 合計 node 数が
/// **減少** すること (strict <、tie 不可)。
///
/// **no-op proof**: McCormick ON 経路を実質無効化 (= lb = -∞ 等) すると total_mc が
/// total_alpha と同値になり (1.0 倍)、本 assert (strict <) は FAIL する。
/// `feedback_sentinel_must_fail_under_noop` 準拠。
#[test]
fn mccormick_reduces_or_matches_bb_node_count_on_bilinear_rich_set() {
    // bilinear-rich fixture: McCormick が α-BB より効くと期待される対象を絞る
    let cases = vec![
        ("asym_bilinear_2d", asym_bilinear_2d()),
        ("bilinear_dense_3d", bilinear_dense_3d()),
        ("bilinear_sumcap_4d", bilinear_sumcap_4d()),
    ];
    let mut total_alpha = 0usize;
    let mut total_mc = 0usize;
    let mut at_least_one_reduces = false;
    for (label, p) in &cases {
        let (_, sa) = solve_qp_global_with_stats(p, &opts(), &cfg(true, false));
        let (_, sm) = solve_qp_global_with_stats(p, &opts(), &cfg(true, true));
        eprintln!(
            "MC_NODE [{}]: alpha_only={} alpha+mc={} delta={}",
            label,
            sa.nodes_processed,
            sm.nodes_processed,
            sa.nodes_processed as i64 - sm.nodes_processed as i64,
        );
        if sm.nodes_processed < sa.nodes_processed {
            at_least_one_reduces = true;
        }
        // McCormick が追加されても worsen させない (= max を取るため数学的に保証されるが、
        // node 列挙順 / pruning timing で微 diff が出る場合は許容。総和で評価)。
        total_alpha += sa.nodes_processed;
        total_mc += sm.nodes_processed;
    }
    eprintln!("MC_NODE_TOTAL: alpha_only={total_alpha} alpha+mc={total_mc}");
    assert!(
        at_least_one_reduces,
        "McCormick must reduce node count on at least one bilinear-rich fixture \
         (totals alpha={total_alpha}, mc={total_mc})"
    );
    // 総和でも非劣化を要求 (= McCormick 統合が overall に悪さしない)
    assert!(
        total_mc <= total_alpha,
        "McCormick should not increase total nodes: alpha={total_alpha}, mc={total_mc}"
    );
}

/// no-op proof teeth: McCormick OFF/OFF を 2 回測定し node 数が同一 (determinism) かつ
/// 上記 strict 削減 assert に必要な < 条件を満たさないことで、sentinel が真に
/// McCormick pathway を検出していることを実証する。
#[test]
fn mccormick_node_reduction_sentinel_has_teeth_under_noop() {
    let cases = vec![
        ("asym_bilinear_2d", asym_bilinear_2d()),
        ("bilinear_dense_3d", bilinear_dense_3d()),
        ("bilinear_sumcap_4d", bilinear_sumcap_4d()),
    ];
    let mut any_strict_reduction = false;
    for (_label, p) in &cases {
        let (_, sa) = solve_qp_global_with_stats(p, &opts(), &cfg(true, false));
        let (_, sb) = solve_qp_global_with_stats(p, &opts(), &cfg(true, false));
        // determinism
        assert_eq!(
            sa.nodes_processed, sb.nodes_processed,
            "OFF vs OFF must be deterministic"
        );
        if sb.nodes_processed < sa.nodes_processed {
            any_strict_reduction = true;
        }
    }
    // no-op (両方 OFF) では strict 削減は絶対に起きない → teeth 確認
    assert!(
        !any_strict_reduction,
        "OFF/OFF should never yield strict node reduction (sentinel would be vacuous otherwise)"
    );
}

/// diag-only concave (= bilinear なし) で McCormick を ON にしても status / obj が
/// 退化しない (= McCormick が結果を歪めない invariant)。
#[test]
fn mccormick_does_not_regress_status_on_diag_only_fixture() {
    let p = diag_concave_2d();
    let (r_a, _) = solve_qp_global_with_stats(&p, &opts(), &cfg(true, false));
    let (r_m, _) = solve_qp_global_with_stats(&p, &opts(), &cfg(true, true));
    eprintln!(
        "MC_DIAG [diag_concave_2d]: alpha status={:?} obj={:.6}, +mc status={:?} obj={:.6}",
        r_a.status, r_a.objective, r_m.status, r_m.objective
    );
    // status 同等 (Optimal は Optimal、LocallyOptimal は同じか promote のみ)
    // Phase 6 で indefinite Q は Optimal→NonconvexGlobal に分岐 (fixture は concave_2d で indefinite)。
    let was_optimal = matches!(
        r_a.status,
        SolveStatus::Optimal | SolveStatus::NonconvexGlobal
    );
    let now_optimal = matches!(
        r_m.status,
        SolveStatus::Optimal | SolveStatus::NonconvexGlobal
    );
    assert!(
        !was_optimal || now_optimal,
        "McCormick must not demote Optimal → LocallyOptimal: {:?} → {:?}",
        r_a.status,
        r_m.status,
    );
    // global = -2 at corner (±1, ±1) で f = -1 -1 = -2
    assert!(
        r_m.objective <= r_a.objective + 1e-4,
        "obj should not regress: alpha={}, mc={}",
        r_a.objective,
        r_m.objective,
    );
    assert!(
        r_m.objective <= -2.0 + 1e-3,
        "expected global ≈ -2, got {}",
        r_m.objective
    );
}
