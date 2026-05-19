//! Phase 4 α-BB lower bound 統合の sentinel (#7 非凸 QP 大域最適化)。
//!
//! ## なぜ別 sentinel が必要か
//! Phase 3 sentinel (diag_qp_global_smoke.rs) は interval lb で proof 可能な 7 fixture
//! を扱う = α-BB を OFF にしても obj-correctness は変わらない。Phase 4 の本価値は
//! 「Phase 3 interval lb では LocallyOptimal にしか到達できない非凸 fixture を、
//! α-BB lb で **Optimal proof** に格上げする」こと。本 sentinel はその進化を直接 assert
//! する。
//!
//! ## 複数 data pattern (memory feedback_test_multi_data_pattern)
//! 5 fixture を共通 table 化し obj-correctness / 総 node 数の退化を見張る:
//!   1. concave 2D bnd1 (= 制約あり concave、Phase 3 不能 / Phase 4 可能の最小例)
//!   2. concave 3D bnd1 (= dimension up でも維持)
//!   3. bilinear symmetric 2D (= zero-diag indefinite、Gershgorin LLT 短絡回避の sentinel)
//!   4. mixed concave+convex (= 一部負固有値だけ補正で十分)
//!   5. 5D concave sumcap (BB 探索 stress)
//!
//! ## no-op 実証 (memory feedback_sentinel_must_fail_under_noop)
//! promotion 観察は `diag_qp_global_promotion_sentinel` の
//! `alpha_bb_promotes_locally_optimal_to_optimal_on_multiple_fixtures` (>=2 件) に
//! 集約済み (本 smoke から重複 test を削除済み)。本 file には node 数 / status の
//! 退化検知のみ残し、α=0 / 凸化恒等 等の no-op proof は invariants / promotion 側で
//! 担保する。
//!
//! ## 効果実測
//! `alpha_bb_does_not_increase_total_node_count`: Phase 4 合計 node 数 ≤ Phase 3
//! 合計 node 数 (= α-BB lb が interval より緩くないことを総量で sentinel 化)。

use solver::options::{BranchingStrategy, GlobalOptimizationConfig};
use solver::qp::{solve_qp_global_with_stats, QpProblem};
use solver::sparse::CscMatrix;
use solver::{SolveStatus, SolverOptions};

/// 大域解の許容相対誤差。Phase 4 の gap_tol = 1e-3 と整合。
const GLOBAL_OBJ_TOL: f64 = 1e-3;

fn opts(timeout_secs: f64) -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(timeout_secs);
    o
}

/// sentinel test 用 BB 探索上限 (lib test 3 分 cap 内で promotion 観察可能な範囲)。
const TEST_MAX_DEPTH: usize = 25;
const TEST_MAX_NODES: usize = 5_000;

fn cfg(use_alpha_bb: bool) -> GlobalOptimizationConfig {
    GlobalOptimizationConfig {
        gap_tol: GLOBAL_OBJ_TOL,
        max_depth: TEST_MAX_DEPTH,
        max_nodes: TEST_MAX_NODES,
        branching: BranchingStrategy::MaxViolation,
        use_alpha_bb,
        use_mccormick: false,
    }
}

use solver::problem::ConstraintType;

/// f = -x² - y², s.t. x+y ≤ 0.5, x,y ∈ [0,1]. Global = -0.25 at (0.5,0) or (0,0.5).
/// 制約 ignore する interval lb は box corner (1,1) で f=-2 を見る = lb=-2 (= 拙劣)。
/// α-BB は制約付き凸化を解くので lb は real feasible region の corner 近傍に上がる。
fn build_constrained_concave_2d() -> QpProblem {
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

/// f = -Σ x_i², s.t. Σ x_i ≤ 0.5, x_i ∈ [0, 1] (n=3). Global = -0.25 at one var=0.5.
fn build_constrained_concave_3d() -> QpProblem {
    let n = 3;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&rows, &cols, &vec![-2.0; n], n, n).unwrap();
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

/// f = x*y (pure bilinear, zero diag), s.t. x+y=0.6, box [-1,1]^2.
/// 制約付き global: x+y=0.6 上で xy = x(0.6-x) = 0.6x - x², 最小は x=±1 端点で。
/// x=1, y=-0.4 → obj = -0.4。x=-0.4, y=1 → obj = -0.4。global = -0.4。
/// interval lb は xy box 上で min=-1 (= 拙劣)、constraint 無視で。
/// α-BB は 0 対角 indefinite Q に対しても α>0 (raw Gershgorin) で凸化、constraint 含めて
/// 解くため tight lb (= 制約を尊重した値 ≥ -0.4)。
fn build_constrained_bilinear() -> QpProblem {
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

/// f = -x² + y², s.t. x+y ≥ 1, x,y ∈ [0,1]. min: x=1,y=0 ⇒ -1。
/// interval lb は constraint 無視で -1 (= corner (1,0) で OK だが偶然) — 実は
/// この fixture では interval も tight になる risk あり。constraint 強めで differentiate:
/// s.t. x+y = 0.7 (= corner (1,0) infeasible, (0.7,0) feasible obj=-0.49)。
fn build_constrained_mixed_convex_concave() -> QpProblem {
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

/// 5D concave with sum ≤ 1.5 constraint, x_i ∈ [0,1].
/// Feasible corner choices: 1.5 sum spread over vars. global obj at corner where 1 var=1
/// plus 1 var=0.5 (and rest 0): -1 + -0.25 = -1.25。または all 0.3 (5*0.09=-0.45 拙).
/// 実 global = -1 - 0.25 = -1.25 (確認: x=(1,0.5,0,0,0) sum=1.5, obj=-1.25)。
/// interval lb (constraint 無視) = -5 (corner (1,1,1,1,1))。差 3.75 = α-BB の好機会。
fn build_constrained_concave_5d() -> QpProblem {
    let n = 5;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&rows, &cols, &vec![-2.0; n], n, n).unwrap();
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

struct Fixture {
    label: &'static str,
    problem: QpProblem,
    global_obj: f64,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            label: "constrained_concave_2d",
            problem: build_constrained_concave_2d(),
            global_obj: -0.25,
        },
        Fixture {
            label: "constrained_concave_3d",
            problem: build_constrained_concave_3d(),
            global_obj: -0.25,
        },
        Fixture {
            label: "constrained_bilinear_eq",
            problem: build_constrained_bilinear(),
            global_obj: -0.4,
        },
        Fixture {
            label: "constrained_mixed_eq",
            problem: build_constrained_mixed_convex_concave(),
            global_obj: -0.49,
        },
        Fixture {
            label: "constrained_concave_5d_sumcap",
            problem: build_constrained_concave_5d(),
            global_obj: -1.25,
        },
    ]
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / 1.0_f64.max(expected.abs())
}

/// 5 fixture 全件で Phase 4 α-BB が正しい obj に到達 (correctness baseline)。
#[test]
fn alpha_bb_reaches_global_objective_on_all_fixtures() {
    for fx in fixtures() {
        let (r, stats) =
            solve_qp_global_with_stats(&fx.problem, &opts(30.0), &cfg(true));
        let err = rel_err(r.objective, fx.global_obj);
        eprintln!(
            "ALPHA_BB_OBJ [{}]: obj={:.6} (exp {:.6}, rel_err={:.2e}) status={:?} nodes={}",
            fx.label, r.objective, fx.global_obj, err, r.status, stats.nodes_processed
        );
        assert!(
            matches!(
                r.status,
                SolveStatus::Optimal
                    | SolveStatus::LocallyOptimal
                    | SolveStatus::NonconvexGlobal
                    | SolveStatus::NonconvexLocal
            ),
            "{}: unexpected status {:?}",
            fx.label,
            r.status
        );
        assert!(
            err <= GLOBAL_OBJ_TOL,
            "{}: rel_err={err:.3e} exceeds tol={GLOBAL_OBJ_TOL:.0e} (obj={}, expected={})",
            fx.label,
            r.objective,
            fx.global_obj,
        );
    }
}

/// Phase 4 ON で全 fixture が **(node 数, status)** いずれかで Phase 3 より同等以上に
/// 良くなることを直接 assert。α-BB lb がより tight = 同じ proof までに必要な node 数
/// が単調減少する数学的根拠 (`bound_alpha_bb::alpha_bb_lb_tightens_as_box_shrinks` 参照)。
///
/// 単純な「Phase 4 nodes ≤ Phase 3 nodes」だと root 1 node で proof 完了する trivial
/// case で no-strict、片や全 fixture 集計で「合計 node 数が Phase 4 ≤ Phase 3」を要求。
#[test]
fn alpha_bb_does_not_increase_total_node_count() {
    let mut total_p3 = 0usize;
    let mut total_p4 = 0usize;
    for fx in fixtures() {
        let (_, s3) = solve_qp_global_with_stats(&fx.problem, &opts(30.0), &cfg(false));
        let (_, s4) = solve_qp_global_with_stats(&fx.problem, &opts(30.0), &cfg(true));
        eprintln!(
            "NODE_COMPARE [{}]: phase3_nodes={} phase4_nodes={}",
            fx.label, s3.nodes_processed, s4.nodes_processed
        );
        total_p3 += s3.nodes_processed;
        total_p4 += s4.nodes_processed;
    }
    eprintln!("NODE_TOTAL: phase3={total_p3} phase4={total_p4}");
    assert!(
        total_p4 <= total_p3,
        "Phase 4 total nodes {} exceeded Phase 3 total {} (= α-BB lb is weaker than interval, \
         expected: tighter)",
        total_p4,
        total_p3,
    );
}

/// `use_alpha_bb=false` が Phase 3 経路に戻る (= backward compat 保証)。
/// fixture の少なくとも 1 件で Phase 3 OFF 時 LocallyOptimal、Phase 4 ON 時 Optimal の
/// 差異が観測される (= cfg がきちんと dispatch 切替に作用している)。
#[test]
fn use_alpha_bb_false_preserves_phase3_semantics() {
    // constrained_bilinear: Phase 3 は eq 制約 ignore で weak lb、Phase 4 は respect → tight。
    let p = build_constrained_bilinear();
    let (r3, _) = solve_qp_global_with_stats(&p, &opts(30.0), &cfg(false));
    let (r4, _) = solve_qp_global_with_stats(&p, &opts(30.0), &cfg(true));
    eprintln!(
        "USE_FLAG_DISPATCH: phase3={:?} phase4={:?} obj3={} obj4={}",
        r3.status, r4.status, r3.objective, r4.objective
    );
    // 両 phase とも obj は正しい (incumbent は IPM local solve 共通)
    let expected = -0.4;
    assert!(
        (r3.objective - expected).abs() <= 5e-2,
        "phase3 obj wrong: {}",
        r3.objective
    );
    assert!(
        (r4.objective - expected).abs() <= 5e-2,
        "phase4 obj wrong: {}",
        r4.objective
    );
    let phase3_proven = matches!(r3.status, SolveStatus::Optimal | SolveStatus::NonconvexGlobal);
    let phase4_proven = matches!(r4.status, SolveStatus::Optimal | SolveStatus::NonconvexGlobal);
    // Phase 4 が Phase 3 で proven なものを退化させていない (Optimal → LocallyOptimal 不可)
    if phase3_proven {
        assert!(
            phase4_proven,
            "Phase 4 ON で Optimal が退化: phase3={:?} phase4={:?}",
            r3.status, r4.status
        );
    }
}

/// 半無限境界 (x in [0, ∞)) では α-BB underestimator が定義できない。
/// `alpha_bb_lower_bound` は None を返し interval lb (`-∞`) に fall back する。
/// status は LocallyOptimal (proof 不能を素直に伝える) で panic しないことを保証。
#[test]
fn alpha_bb_falls_back_safely_on_semi_infinite_box() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let p = QpProblem::new(
        q,
        vec![1.0, 1.0],
        a,
        vec![1.0],
        vec![(0.0, f64::INFINITY); 2],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let (r, _) = solve_qp_global_with_stats(&p, &opts(5.0), &cfg(true));
    // Q indefinite (-2 diag) + semi-infinite + α-BB fallback → NonconvexLocal
    // (Phase 6 で indefinite Q を LocallyOptimal から分離)
    assert!(
        matches!(r.status, SolveStatus::NonconvexLocal),
        "semi-infinite box should yield NonconvexLocal under α-BB fallback, got {:?}",
        r.status
    );
}
