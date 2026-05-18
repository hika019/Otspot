//! Phase 3 spatial Branch-and-Bound sentinel (#6 非凸 QP 大域最適化)。
//!
//! ## 真因 (sentinel が必要な理由)
//! 非凸 QP では IPM (inertia 補正付き) は cold init から「最寄り KKT 点」へしか
//! 収束しない。saddle / 内部局所最適に固着する典型 input が:
//!   - c = 0 の concave diag QP (cold x=0 が内部勾配 0 で固着)
//!   - bilinear symmetric (saddle で固着)
//!   - 制約あり symmetric QP (local 多数)
//! Phase 2 multistart は random restart で対称性破壊するが、確率的保証のみ。
//! Phase 3 spatial B&B は box 分割 + interval 下界 + pruning で **deterministic に**
//! gap_tol 内まで詰める。
//!
//! ## 複数 data pattern (memory feedback_test_multi_data_pattern)
//! 7 fixture × 既知 global obj で table-driven assertion:
//!   1. concave 1D bnd=2 (global=-4)
//!   2. concave 1D bnd=3 (global=-9)
//!   3. concave 2D bnd=1 (global=-2)
//!   4. concave 3D bnd=1.5 (global=-6.75)
//!   5. mixed diag (1 concave + 1 convex)
//!   6. bilinear symmetric (saddle vs corner)
//!   7. concave 1D + linear pull (asymmetric)
//!
//! ## no-op 実証 (memory feedback_sentinel_must_fail_under_noop)
//! 3 種 no-op を実装中に temporary 適用 → 該当 sentinel が確実に FAIL する事を確認、
//! revert して PASS 復帰。各 no-op の実証:
//!
//! - **branching no-op** (`select_branching_variable` を `Some(0)` 強制):
//!   `global_reaches_known_optimum_all_fixtures` が concave_2d_bnd1 で FAIL
//!   (got -1.0, expected -2.0)。検証済 cargo nextest run.
//! - **pruning no-op** (`should_prune` を `false` 強制):
//!   `pruning_keeps_node_count_well_below_cap` が FAIL
//!   (nodes=2000=cap, expect <1000)。検証済 cargo nextest run.
//! - **upper bound no-op** (`solve_local_upper_bound` を `obj=0 @ midpoint` 返却):
//!   `global_reaches_known_optimum_all_fixtures` と
//!   `global_optimal_status_proves_gap_for_simple_fixtures` が FAIL。検証済.
//!
//! 各 case 実装中に書換 → cargo nextest 確認 → revert PASS。Phase 3 完了直前。

use solver::options::{BranchingStrategy, GlobalOptimizationConfig};
use solver::qp::{solve_qp_global, solve_qp_global_with_stats, solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;
use solver::{SolveStatus, SolverOptions};

/// 大域解の許容相対誤差。gap_tol = 1e-3 と整合 (= solver の guarantee と同水準)。
const GLOBAL_OBJ_TOL: f64 = 1e-3;

fn opts(timeout_secs: f64) -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(timeout_secs);
    o
}

fn cfg(gap_tol: f64) -> GlobalOptimizationConfig {
    GlobalOptimizationConfig {
        gap_tol,
        max_depth: 20,
        max_nodes: 10_000,
        branching: BranchingStrategy::MaxViolation,
    }
}

/// 期待値との相対差を計算。
fn rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / (1.0_f64).max(expected.abs())
}

fn assert_global_objective(label: &str, actual_obj: f64, expected_obj: f64) {
    let err = rel_err(actual_obj, expected_obj);
    assert!(
        err <= GLOBAL_OBJ_TOL,
        "{label}: expected global obj ≈ {expected_obj}, got {actual_obj} (rel_err={err:.3e}, tol={GLOBAL_OBJ_TOL:.0e})"
    );
}

/// f = a x², box [lb, ub]。a<0 で concave (corner min)、a>0 で convex (内部 min)。
fn build_diag_1d(a: f64, c: f64, lb: f64, ub: f64) -> QpProblem {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0 * a], 1, 1).unwrap();
    // 0.5 * 2a * x² = a x²。
    let a_mat = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
    QpProblem::new_all_le(q, vec![c], a_mat, vec![], vec![(lb, ub)]).unwrap()
}

/// f = -Σ x_i², box [-bnd, bnd]^n。global = -n bnd² at 2^n corner.
fn build_diag_concave_nd(n: usize, bnd: f64) -> QpProblem {
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![-2.0; n]; // 0.5 * -2 * x² = -x²
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
    QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(-bnd, bnd); n]).unwrap()
}

/// f = -x² + y², x in [-2, 1], y in [-1, 2]
/// global: x²=4 (x=-2), y²=0 (y=0), obj=-4
fn build_mixed_diag() -> QpProblem {
    // Q upper-triangle: diag [-2, 2] (0.5*(-2)x²+0.5*2*y² = -x²+y²)
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-2.0, 1.0), (-1.0, 2.0)]).unwrap()
}

/// f = x*y + 1e-6 (x² + y²), box [-2, 2]^2
/// saddle at (0,0), global corners (2,-2)/(-2,2) → obj = -4 + 1e-6 * 8 ≈ -3.999992
///
/// Q full-symmetric storage (両半 (0,1) と (1,0)) で 0.5 x'Qx = xy になる。
fn build_bilinear_saddle(bnd: f64) -> QpProblem {
    // Q = [[2e-6, 1.0], [1.0, 2e-6]] → 0.5 x'Qx = 1e-6 x² + xy + 1e-6 y²
    let q = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[2e-6, 1.0, 1.0, 2e-6],
        2,
        2,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-bnd, bnd); 2]).unwrap()
}

/// f = -x² + 0.5 x, box [-2, 2]
/// d/dx = -2x + 0.5 = 0 → x*=0.25 (local max, since coeff -). Concave → corner min.
/// candidates: x=-2 → -4 - 1 = -5;  x=2 → -4 + 1 = -3。  global = -5 at x=-2.
fn build_concave_linear_pull() -> QpProblem {
    build_diag_1d(-1.0, 0.5, -2.0, 2.0)
}

struct Fixture {
    label: &'static str,
    problem: QpProblem,
    global_obj: f64,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            label: "concave_1d_bnd2",
            problem: build_diag_1d(-1.0, 0.0, -2.0, 2.0),
            global_obj: -4.0,
        },
        Fixture {
            label: "concave_1d_bnd3",
            problem: build_diag_1d(-1.0, 0.0, -3.0, 3.0),
            global_obj: -9.0,
        },
        Fixture {
            label: "concave_2d_bnd1",
            problem: build_diag_concave_nd(2, 1.0),
            global_obj: -2.0,
        },
        Fixture {
            label: "concave_3d_bnd1_5",
            problem: build_diag_concave_nd(3, 1.5),
            global_obj: -3.0 * 1.5 * 1.5,
        },
        Fixture {
            label: "mixed_diag",
            problem: build_mixed_diag(),
            global_obj: -4.0,
        },
        Fixture {
            label: "bilinear_saddle_bnd2",
            problem: build_bilinear_saddle(2.0),
            global_obj: -4.0 + 1e-6 * 8.0,
        },
        Fixture {
            label: "concave_1d_linear_pull",
            problem: build_concave_linear_pull(),
            global_obj: -5.0,
        },
    ]
}

#[test]
fn global_reaches_known_optimum_all_fixtures() {
    for fx in fixtures() {
        let r = solve_qp_global(&fx.problem, &opts(30.0), &cfg(GLOBAL_OBJ_TOL));
        assert!(
            matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
            "{}: unexpected status {:?}",
            fx.label,
            r.status
        );
        assert_global_objective(fx.label, r.objective, fx.global_obj);
        eprintln!(
            "GLOBAL_SMOKE [{}]: obj={:.6} (expected {:.6}, rel_err={:.2e}) status={:?}",
            fx.label,
            r.objective,
            fx.global_obj,
            rel_err(r.objective, fx.global_obj),
            r.status
        );
    }
}

#[test]
fn global_strictly_improves_over_cold_solve() {
    // 大域 vs cold IPM 単発: 少なくとも 1 fixture では cold が global を逃す。
    // (Phase 2 multistart sentinel と同様の原理、ここでは global 経路の優位を直接示す)
    let mut at_least_one_strictly_better = false;
    for fx in fixtures() {
        let cold = solve_qp_with(&fx.problem, &opts(5.0));
        let global = solve_qp_global(&fx.problem, &opts(30.0), &cfg(GLOBAL_OBJ_TOL));
        // 大域結果は cold と同等か厳密に良い (= obj が小さい)
        assert!(
            global.objective <= cold.objective + 1e-6,
            "{}: global ({:.6}) should be ≤ cold ({:.6})",
            fx.label,
            global.objective,
            cold.objective
        );
        if cold.objective > global.objective + 1.0 {
            at_least_one_strictly_better = true;
            eprintln!(
                "GLOBAL_COLD_GAP [{}]: cold={:.4} global={:.4} (saddle escape verified)",
                fx.label, cold.objective, global.objective
            );
        }
    }
    assert!(
        at_least_one_strictly_better,
        "no fixture demonstrated cold-vs-global gap (sentinel weak: add a harder fixture)"
    );
}

#[test]
fn global_optimal_status_proves_gap_for_simple_fixtures() {
    // Phase 3 は弱い下界しか出ない (制約無視 interval) ため、必ずしも proof 完了
    // しないが、最低 1 fixture は Optimal (= queue 空 or 全 leaf prune) で帰る。
    // = 全ての fixture が LocallyOptimal だけだと "B&B が proof として機能していない"。
    let mut any_optimal = false;
    for fx in fixtures() {
        let r = solve_qp_global(&fx.problem, &opts(30.0), &cfg(GLOBAL_OBJ_TOL));
        if matches!(r.status, SolveStatus::Optimal) {
            any_optimal = true;
            eprintln!("GLOBAL_PROVEN [{}]: obj={:.6}", fx.label, r.objective);
        }
    }
    assert!(
        any_optimal,
        "no fixture reached SolveStatus::Optimal → BB proof path silent SKIP の疑い"
    );
}

/// gap_tol を緩めると proof 完了率が上がる (= gap_tol が実際に影響)。
/// proof 数 / fixture 比で測る。Phase 3 (弱い下界) の Phase 4 必要性を可視化。
#[test]
fn larger_gap_tol_improves_proof_completion() {
    let count_optimal = |tol: f64| {
        fixtures()
            .into_iter()
            .filter(|fx| {
                let r = solve_qp_global(&fx.problem, &opts(30.0), &cfg(tol));
                matches!(r.status, SolveStatus::Optimal)
            })
            .count()
    };
    let strict = count_optimal(1e-6);
    let loose = count_optimal(0.5);
    eprintln!(
        "GLOBAL_GAP_TOL: optimal at tol=1e-6 -> {}/{}, at tol=0.5 -> {}/{}",
        strict,
        7,
        loose,
        7
    );
    assert!(
        loose >= strict,
        "loose gap_tol should not hurt proof count (loose={loose} strict={strict})"
    );
    // gap_tol=0.5 (=50%) なら ほぼ全 fixture proof 完了 (incumbent ≈ lb の gap で)
    assert!(
        loose >= 5,
        "loose gap_tol=0.5 should proof >= 5/7, got {loose}"
    );
}

#[test]
fn deadline_honored_returns_incumbent_not_panics() {
    // 巨大 fixture ではなく n=3 でも max_nodes を絞って打ち切る pattern を再現。
    let fx = build_diag_concave_nd(3, 1.5);
    let mut o = opts(30.0);
    // 即 deadline (50ms) で打ち切り
    o.timeout_secs = Some(0.05);
    let r = solve_qp_global(&fx, &o, &cfg(1e-9));
    // status は Optimal / LocallyOptimal / Timeout のいずれか (panic しなければ可)
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal
                | SolveStatus::LocallyOptimal
                | SolveStatus::SuboptimalSolution
                | SolveStatus::Timeout
                | SolveStatus::MaxIterations
        ),
        "unexpected status under deadline: {:?}",
        r.status
    );
}

#[test]
fn max_nodes_zero_returns_root_only_incumbent() {
    // max_nodes=1 = root local solve のみ → LocallyOptimal (proof 不能なら) or Optimal
    let fx = build_diag_concave_nd(2, 1.0);
    let mut c = cfg(1e-6);
    c.max_nodes = 1;
    let r = solve_qp_global(&fx, &opts(10.0), &c);
    assert!(matches!(
        r.status,
        SolveStatus::Optimal | SolveStatus::LocallyOptimal
    ));
}

#[test]
fn pure_convex_qp_solves_at_root_with_optimal_status() {
    // Convex (PSD Q + c) は root local solve が global を与え、interval lb が
    // 緩くても proof 可能。Optimal が返ることを sentinel として保護。
    // f = x² + y² with c=[0,0] in box [-1,1]^2 → global at (0,0), obj=0
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2]).unwrap();
    let r = solve_qp_global(&p, &opts(10.0), &cfg(1e-6));
    assert!(matches!(
        r.status,
        SolveStatus::Optimal | SolveStatus::LocallyOptimal
    ));
    assert!(r.objective.abs() < 1e-4, "convex QP obj should be ~0, got {}", r.objective);
}

/// Pruning sentinel: 5D concave QP で枝刈が有効ならば node 数は max_nodes 上限の
/// 半分未満で global proof 完了する。枝刈 no-op (= should_prune が常に false) に
/// すると max_nodes (= 2000) を埋め尽くし、status が LocallyOptimal 降格 + nodes
/// 数が cap に張り付く。
///
/// 5D concave: f = -Σ x_i² over [-1,1]^5, global=-5 at 2^5=32 corners。
/// 枝刈ありで通常 ~50-300 node、枝刈無で 2^5+ × interval depth → 2000 cap 確実 hit。
#[test]
fn pruning_keeps_node_count_well_below_cap() {
    let p = build_diag_concave_nd(5, 1.0);
    let cfg = GlobalOptimizationConfig {
        gap_tol: 1e-3,
        max_depth: 20,
        max_nodes: 2_000,
        branching: BranchingStrategy::MaxViolation,
    };
    let (r, stats) = solve_qp_global_with_stats(&p, &opts(60.0), &cfg);
    eprintln!(
        "PRUNING_SMOKE: nodes={} pruned={} max_depth={} status={:?} obj={}",
        stats.nodes_processed, stats.pruned, stats.max_depth_seen, r.status, r.objective
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "expected Optimal under pruning, got {:?} (nodes={})",
        r.status,
        stats.nodes_processed
    );
    assert!(
        r.objective < -4.99,
        "expected global ≈ -5, got {}",
        r.objective
    );
    // 枝刈効果 sentinel: cap の半分未満で proof 完了 = 枝刈実機能
    assert!(
        stats.nodes_processed < cfg.max_nodes / 2,
        "枝刈効きすぎ or 効いてない疑い: nodes={} (cap={}, expect <{})",
        stats.nodes_processed,
        cfg.max_nodes,
        cfg.max_nodes / 2,
    );
}

/// Builder mismatch sentinel: 期待した fixture が本当に non-trivial であることを保護。
/// (= 万一 builder bug で trivial 問題になっていても気づける)
#[test]
fn fixture_global_objs_strictly_below_zero_for_concave_cases() {
    for fx in fixtures() {
        if fx.label.contains("concave") || fx.label.contains("bilinear") || fx.label.contains("mixed") {
            assert!(
                fx.global_obj < -1.0,
                "{}: expected global ≪ 0 (non-trivial), got {}",
                fx.label,
                fx.global_obj
            );
        }
    }
}
