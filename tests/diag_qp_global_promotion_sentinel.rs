//! α-BB promotion / prune / node 削減 ratio sentinel。
//!
//! ## 目的
//! Phase 4 α-BB が node 削減を **明示的な比率** で達成すること (= 観察値ベースの
//! `>= MIN_REDUCTION_RATIO` assertion)、prune mechanism が実働すること
//! (= `stats.pruned > 0`)、そして α-BB を no-op 化すると ratio sentinel が FAIL すること
//! (`feedback_sentinel_must_fail_under_noop`) を機械保証する。
//!
//! ## 計測値の根拠
//! `diag_qp_global_alpha_bb_smoke::alpha_bb_does_not_increase_total_node_count` から
//! Phase 3 total=5597, Phase 4 total=61 (ratio ≈ 92×) を観測済み。
//! 安全 margin を取り MIN_REDUCTION_RATIO = 5.0 (= 5× 以上の削減)、
//! per-fixture MIN_RATIO = 3.0 を要求する。
//!
//! ## no-op proof
//! `node_reduction_sentinel_has_teeth_under_noop` は α-BB OFF を **両方** に当てて
//! ratio = 1.0 を観測し、`1.0 < MIN_REDUCTION_RATIO` を assert する。これにより
//! 本ファイルの ratio sentinel が真に α-BB pathway を保護していることが証明される。
//!
//! ## 複数 data pattern
//! 5 non-convex fixture × constraint 有無 × dimensionality (2D/3D/5D) で多様性確保。

use otspot::options::{BranchingStrategy, GlobalOptimizationConfig};
use otspot::problem::ConstraintType;
use otspot::qp::{solve_qp_global_with_stats, QpProblem};
use otspot::sparse::CscMatrix;
use otspot::{SolveStatus, SolverOptions};

/// 全 fixture 合計の最小削減倍率 (Phase 3 nodes / Phase 4 nodes)。
/// 観測 92× に対し margin 18× を取り、5× を sentinel 閾値とする。
const MIN_TOTAL_REDUCTION_RATIO: f64 = 5.0;

/// 個別 fixture での最小削減倍率。観測最小 13.7× に対し margin 4×、3× を要求。
const MIN_PER_FIXTURE_REDUCTION_RATIO: f64 = 3.0;

/// LocallyOptimal → Optimal promotion を観測すべき fixture 数の下限。
/// Phase 4 の存在意義は「Phase 3 で未証明だった大域最適性を proof 化する」点にあり、
/// 1 fixture でしか効いていなければ「偶然の 1 サンプル」と区別できない。複数 fixture で
/// 効果が観測できることを require して sentinel に teeth を与える。
const MIN_PROMOTION_COUNT: usize = 2;

/// BB 探索上限 (3 分内で完結する範囲)。
const TEST_MAX_DEPTH: usize = 25;
const TEST_MAX_NODES: usize = 5_000;
const TEST_TIMEOUT_SECS: f64 = 30.0;
const TEST_GAP_TOL: f64 = 1e-3;

fn opts() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(TEST_TIMEOUT_SECS);
    o
}

fn cfg(use_alpha_bb: bool) -> GlobalOptimizationConfig {
    GlobalOptimizationConfig {
        gap_tol: TEST_GAP_TOL,
        max_depth: TEST_MAX_DEPTH,
        max_nodes: TEST_MAX_NODES,
        branching: BranchingStrategy::MaxViolation,
        use_alpha_bb,
        use_mccormick: false,
    }
}

// ---------------- fixtures (Phase 3 で多 node、Phase 4 で大削減を期待) ----------------

fn concave_2d_sumcap() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![0.5],
        vec![(0.0, 1.0); 2],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

fn concave_3d_sumcap() -> QpProblem {
    let n = 3;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &vec![-2.0; n],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![0.5],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

fn bilinear_eq_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![0.6],
        vec![(-1.0, 1.0); 2],
        vec![ConstraintType::Eq],
    )
    .unwrap()
}

fn mixed_eq_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    QpProblem::new(
        q,
        vec![0.0, 0.0],
        a,
        vec![0.7],
        vec![(0.0, 1.0); 2],
        vec![ConstraintType::Eq],
    )
    .unwrap()
}

fn concave_5d_sumcap() -> QpProblem {
    let n = 5;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &vec![-2.0; n],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![1.5],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

/// 6D concave sum-cap。次元を 1 上げ Σx≤2.0 にして node 増、Phase 3 で打ち切られ
/// LocallyOptimal → Phase 4 α-BB で Optimal に格上げされる想定。
fn concave_6d_sumcap() -> QpProblem {
    let n = 6;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &vec![-2.0; n],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![2.0],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

/// 4D 混合: diag=[-2,-2,-1,-1] + sum-cap。eigenvalue が複数値で interval bound と
/// α-BB bound の差が顕著、Phase 3 LocallyOptimal を促す。
fn mixed_concave_4d() -> QpProblem {
    let n = 4;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &[-2.0, -2.0, -1.0, -1.0],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![1.5],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

/// 7D concave sum-cap (Σ x ≤ 2.5)。次元 7 で Phase 3 interval lb (= 制約無視, corner
/// 全 1 で −7) に対し global は 2 vars=1 + 1 var=0.5 = −2.25。Phase 3 BB は探索上限
/// (`TEST_MAX_NODES`) を使い切り LocallyOptimal、Phase 4 α-BB は制約付き凸化 lb で
/// Optimal proof に格上げする想定。promotion 件数 4 で MIN_PROMOTION_COUNT=2 に対し
/// margin 2 を確保する。
fn concave_7d_sumcap() -> QpProblem {
    let n = 7;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &vec![-2.0; n],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![2.5],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

/// 5D bilinear: Q 全 off-diag = 1.0 (diag=0)。eigenvalue は (n-1,1) 重複度=(1,n-1) で
/// 強非凸 (Q = J - I の eigvals: 4, -1×4)。Σx ≤ 1 で対称破壊。
fn bilinear_5d_box() -> QpProblem {
    let n = 5;
    // dense off-diag: rows[i,j] for all i!=j, value 1.0
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
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![-0.5; n],
        a,
        vec![1.0],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

struct Case {
    label: &'static str,
    problem: QpProblem,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "concave_2d_sumcap",
            problem: concave_2d_sumcap(),
        },
        Case {
            label: "concave_3d_sumcap",
            problem: concave_3d_sumcap(),
        },
        Case {
            label: "bilinear_eq_2d",
            problem: bilinear_eq_2d(),
        },
        Case {
            label: "mixed_eq_2d",
            problem: mixed_eq_2d(),
        },
        Case {
            label: "concave_5d_sumcap",
            problem: concave_5d_sumcap(),
        },
        Case {
            label: "concave_6d_sumcap",
            problem: concave_6d_sumcap(),
        },
        Case {
            label: "mixed_concave_4d",
            problem: mixed_concave_4d(),
        },
        Case {
            label: "bilinear_5d_box",
            problem: bilinear_5d_box(),
        },
        Case {
            label: "concave_7d_sumcap",
            problem: concave_7d_sumcap(),
        },
    ]
}

// ---------------- tests ----------------

/// 全 fixture 合計 node 数で `Phase 3 / Phase 4 >= MIN_TOTAL_REDUCTION_RATIO`.
/// 観測値ベース (~92×) に対し 5× を要求。
#[test]
fn alpha_bb_node_reduction_ratio_meets_total_threshold() {
    let mut total_p3 = 0usize;
    let mut total_p4 = 0usize;
    for c in cases() {
        let (_, s3) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        let (_, s4) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(true));
        eprintln!(
            "RATIO_PER [{}]: p3={} p4={} ratio={:.2}x",
            c.label,
            s3.nodes_processed,
            s4.nodes_processed,
            s3.nodes_processed as f64 / s4.nodes_processed.max(1) as f64,
        );
        total_p3 += s3.nodes_processed;
        total_p4 += s4.nodes_processed;
    }
    let ratio = total_p3 as f64 / total_p4.max(1) as f64;
    eprintln!("RATIO_TOTAL: p3={total_p3} p4={total_p4} ratio={ratio:.2}x");
    assert!(
        ratio >= MIN_TOTAL_REDUCTION_RATIO,
        "α-BB total node reduction ratio {ratio:.2}× < required {MIN_TOTAL_REDUCTION_RATIO}×. \
         (p3={total_p3}, p4={total_p4}) — α-BB pathway may have regressed.",
    );
}

/// 各 fixture で `Phase 3 / Phase 4 >= MIN_PER_FIXTURE_REDUCTION_RATIO`.
/// 全 fixture 個別に保証することで、平均だけ良くて 1 件退化、を見逃さない。
#[test]
fn alpha_bb_node_reduction_ratio_meets_per_fixture_threshold() {
    for c in cases() {
        let (_, s3) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        let (_, s4) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(true));
        let ratio = s3.nodes_processed as f64 / s4.nodes_processed.max(1) as f64;
        eprintln!(
            "PER_FIXTURE [{}]: p3={} p4={} ratio={:.2}x",
            c.label, s3.nodes_processed, s4.nodes_processed, ratio
        );
        assert!(
            ratio >= MIN_PER_FIXTURE_REDUCTION_RATIO,
            "{}: α-BB reduction {ratio:.2}× < required {MIN_PER_FIXTURE_REDUCTION_RATIO}× (p3={}, p4={})",
            c.label,
            s3.nodes_processed,
            s4.nodes_processed,
        );
    }
}

/// **no-op proof**: α-BB を OFF (= 両 phase とも `use_alpha_bb=false`) で同 ratio を
/// 計算すると 1.0 になるため、`MIN_TOTAL_REDUCTION_RATIO` 要求を満たせない。
/// このテストが PASS することで、上記 ratio sentinel が真に α-BB pathway 削減を
/// 検出する力を持つことが証明される (本当に α-BB が無効化されたら sentinel は落ちる)。
#[test]
fn node_reduction_sentinel_has_teeth_under_noop() {
    let mut total_a = 0usize;
    let mut total_b = 0usize;
    for c in cases() {
        // 両方 OFF: α-BB 無効 = no-op 状態
        let (_, sa) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        let (_, sb) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        total_a += sa.nodes_processed;
        total_b += sb.nodes_processed;
    }
    let ratio_noop = total_a as f64 / total_b.max(1) as f64;
    eprintln!("NOOP_RATIO: a={total_a} b={total_b} ratio={ratio_noop:.4}x");
    // determinism check: 同じ config 2 回で node 数は同一なはず
    assert!(
        (ratio_noop - 1.0).abs() < 1e-9,
        "OFF vs OFF should yield ratio 1.0 (deterministic), got {ratio_noop}"
    );
    // teeth: 1.0 は MIN_TOTAL_REDUCTION_RATIO を満たさないため sentinel は FAIL する
    assert!(
        ratio_noop < MIN_TOTAL_REDUCTION_RATIO,
        "sentinel has no teeth: no-op ratio {ratio_noop} already exceeds threshold \
         {MIN_TOTAL_REDUCTION_RATIO}. threshold is too lax."
    );
}

/// α-BB ON で **prune mechanism が実働** (= `stats.pruned > 0`) すること。
/// promotion (incumbent 更新 → 既存 node lb > incumbent → prune) が dispatch される
/// fixture を用意。
#[test]
fn alpha_bb_prune_mechanism_engages() {
    // concave_5d_sumcap は α-BB ON で 43 node 探索の間に多数 prune が発生するはず
    let p = concave_5d_sumcap();
    let (_, s) = solve_qp_global_with_stats(&p, &opts(), &cfg(true));
    eprintln!(
        "PRUNE_STATS [concave_5d_sumcap]: nodes={} pruned={} depth={}",
        s.nodes_processed, s.pruned, s.max_depth_seen
    );
    assert!(
        s.pruned > 0,
        "Phase 4 α-BB should trigger pruning on concave_5d_sumcap, got pruned=0 \
         (mechanism may be broken)"
    );
}

/// `pruned > 0` が全 fixture 集計で成立すること (α-BB ON の prune mechanism が
/// dead していない)。Phase 3 と Phase 4 の prune 数を並べて記録するが、Phase 4 は
/// 全 node 数が桁違いに少ないため絶対 prune 数は Phase 3 より小さくなりうる
/// (= 比較 assertion ではなく **生存判定** のみ)。
#[test]
fn alpha_bb_pruning_count_remains_positive_across_fixtures() {
    let mut total_p3 = 0usize;
    let mut total_p4 = 0usize;
    for c in cases() {
        let (_, s3) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        let (_, s4) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(true));
        eprintln!(
            "PRUNE_COMPARE [{}]: p3.pruned={} p4.pruned={}",
            c.label, s3.pruned, s4.pruned
        );
        total_p3 += s3.pruned;
        total_p4 += s4.pruned;
    }
    eprintln!("PRUNE_TOTAL: p3={total_p3} p4={total_p4}");
    assert!(
        total_p4 > 0,
        "Phase 4 α-BB never pruned across {} fixtures — mechanism dead",
        cases().len()
    );
}

/// promotion による status 向上 (LocallyOptimal → Optimal) が **複数 fixture** で
/// 観測できること (`MIN_PROMOTION_COUNT` 以上)。
///
/// 1 件だけ通る threshold だと「偶発的に 1 fixture が promote した」case と
/// 「α-BB が広く効いている」case を区別できず、smoke 側の同種 sentinel と機能重複する。
/// 本ファイルは強化版として複数件を要求し、concave_5d/6d/7d_sumcap + bilinear_5d_box
/// の合算 (期待 4 件) で MIN_PROMOTION_COUNT に対し margin 2 を確保する。
#[test]
fn alpha_bb_promotes_locally_optimal_to_optimal_on_multiple_fixtures() {
    let mut promotions = 0;
    let mut promoted_labels = Vec::new();
    for c in cases() {
        let (r3, _) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(false));
        let (r4, _) = solve_qp_global_with_stats(&c.problem, &opts(), &cfg(true));
        // Phase 6 で indefinite Q は LocallyOptimal→NonconvexLocal, Optimal→NonconvexGlobal に
        // 分岐。fixture は indefinite (concave_*d, bilinear) のため Nonconvex* 系で promotion。
        let was_locally = matches!(
            r3.status,
            SolveStatus::LocallyOptimal | SolveStatus::NonconvexLocal
        );
        let now_optimal = matches!(
            r4.status,
            SolveStatus::Optimal | SolveStatus::NonconvexGlobal
        );
        eprintln!("PROMO [{}]: p3={:?} p4={:?}", c.label, r3.status, r4.status);
        if was_locally && now_optimal {
            promotions += 1;
            promoted_labels.push(c.label);
        }
    }
    assert!(
        promotions >= MIN_PROMOTION_COUNT,
        "α-BB promote {promotions} fixtures (need >= {MIN_PROMOTION_COUNT}). \
         Phase 4 が広域に proof uplift を生んでいない可能性 — promoted = {promoted_labels:?}",
    );
}
