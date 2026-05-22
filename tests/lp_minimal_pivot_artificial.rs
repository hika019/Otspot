//! `pivot_out_degenerate_artificials` の BTRAN candidate scan 計算量 regression guard。
//! K artificial × N 非基底列の合成 LP を build し、解時間が wall budget 内に
//! 収まることで O(n_artificial × n_total) 退行を検知する。

use std::time::Instant;

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::sparse::CscMatrix;

/// Timeout budget for the O(n_artificial × n_total) regression guard.
///
/// Pre-fix: K=3,N=50 ≈ 0.5–1.0 s; K=5,N=80 ≈ 2–3 s; K=11,N=200 >> 3 s.
/// The solver's timeout converts a per-iteration cost explosion into a
/// `SolveStatus::Timeout`, which the `assert_eq!(status, Optimal)` catches.
const MAX_SOLVE_WALL_SECS: f64 = 3.0;

/// Iteration limit for cycling / algorithmic regression detection.
///
/// These toy problems (K artificial × N decision cols) should solve in
/// O(K + N) simplex pivots. Any correct implementation finishes well
/// below this ceiling; a cycling regression would exceed it.
const MAX_PIVOT_ITERS: usize = 2_000;

/// `K` 個の artificial-prone Eq 行 × `N` 非基底列の合成 LP を build。
///
/// 構造:
/// - n = N + K (decision + K dummy var, dummy は Eq 行で fix)
/// - m = K (Eq 行のみ)
/// - 各 Eq 行 i: dummy_i = 0, plus 全 decision col への単位係数 (退化を誘発)
///   - これにより Phase I で dummy_i が basis を 0 で占有 → artificial 退化
/// - decision col の c = 1.0 (min)、bound [0, 1]
/// - dummy col の c = 0.0, bound [0, 0] (固定; ただし presolve OFF で明示的に
///   simplex に解かせるので artificial 経路を発火させる)
///
/// `presolve = false` で simplex 単体に解かせ、pivot_out 退行を直接観測する。
fn build_pivot_workload(k_artificial: usize, n_decision: usize) -> LpProblem {
    let n_dummy = k_artificial;
    let n_total = n_decision + n_dummy;
    let m = k_artificial;

    let mut tri_rows = Vec::new();
    let mut tri_cols = Vec::new();
    let mut tri_vals = Vec::new();

    // 各 Eq 行 i: a_{i, dummy_i} = 1 (dummy 列 = N + i 番目)、b_i = 0
    // + 全 decision 列に小さな係数 0.1 (Phase I で artificial pivot 候補を多発)
    for i in 0..m {
        // dummy entry (= row i identity for dummy var)
        tri_rows.push(i);
        tri_cols.push(n_decision + i);
        tri_vals.push(1.0);
        // decision entries
        for j in 0..n_decision {
            tri_rows.push(i);
            tri_cols.push(j);
            tri_vals.push(0.1);
        }
    }

    let a = CscMatrix::from_triplets(&tri_rows, &tri_cols, &tri_vals, m, n_total).unwrap();

    // 目的: min Σ c[j] x[j], decision に正コスト、dummy は 0
    let mut c = vec![1.0_f64; n_total];
    for j in n_decision..n_total {
        c[j] = 0.0;
    }

    // bounds: decision in [0, 1], dummy 固定 = 0 (lb=ub=0)
    let mut bounds = vec![(0.0_f64, 1.0_f64); n_total];
    for j in n_decision..n_total {
        bounds[j] = (0.0, 0.0); // dummy 固定
    }

    // 右辺: 全 Eq 行 = 0 (dummy が 0 で吸収、decision 全 0 が一意 feasible)
    let b = vec![0.0_f64; m];
    let cts = vec![ConstraintType::Eq; m];

    LpProblem::new_general(
        c, a, b, cts, bounds,
        Some(format!("pivot_workload_K{}_N{}", k_artificial, n_decision)),
    ).unwrap()
}

/// 共通 assert: 解時間が `MAX_SOLVE_WALL_SECS` を超えたら退行とみなす。
fn assert_solve_under_budget(lp: &LpProblem, expected_obj: f64, label: &str) {
    let mut opts = SolverOptions::default();
    // pivot_out 経路を確実に発火させるため presolve OFF (presolve は dummy 列を
    // EmptyColumn / FixedVar で吸収して artificial 経路を bypass しうる)。
    opts.presolve = false;
    // Timeout == budget: per-iteration cost explosion → Timeout status → fails the
    // Optimal assert below (deterministic sentinel replacing the wall check).
    opts.timeout_secs = Some(MAX_SOLVE_WALL_SECS);

    let t0 = Instant::now();
    let r = solve_with(lp, &opts);
    let elapsed = t0.elapsed().as_secs_f64();

    eprintln!(
        "[{}] elapsed={:.3}s status={:?} obj={:.3e} iters={}",
        label, elapsed, r.status, r.objective, r.iterations
    );

    // Primary sentinel: per-iteration cost explosion → Timeout → fails here.
    assert_eq!(r.status, SolveStatus::Optimal, "[{}] status={:?}", label, r.status);
    let obj_err = (r.objective - expected_obj).abs() / (1.0 + expected_obj.abs());
    assert!(obj_err < 1e-6, "[{}] obj={:.6e} expected={:.6e}", label, r.objective, expected_obj);
    // Secondary sentinel: cycling → iteration explosion → fails here.
    assert!(
        r.iterations < MAX_PIVOT_ITERS,
        "[{}] iterations {} ≥ {} — possible cycling regression",
        label, r.iterations, MAX_PIVOT_ITERS
    );
}

/// K=3 artificial × N=50 nonbasic。osa-60 (K=11, N=243k) の構造縮約版。
#[test]
fn bug3a_pivot_out_small_k3_n50() {
    let lp = build_pivot_workload(3, 50);
    assert_solve_under_budget(&lp, 0.0, "bug3a_k3_n50");
}

/// K=5 × N=80。BTRAN 経路が無いと per-artificial cost が線形に膨らむ。
#[test]
fn bug3b_pivot_out_medium_k5_n80() {
    let lp = build_pivot_workload(5, 80);
    assert_solve_under_budget(&lp, 0.0, "bug3b_k5_n80");
}

/// K=11 × N=200。osa-60 の K=11 を一致させ N を桁スケールダウン。
#[test]
fn bug3c_pivot_out_osa60_proxy_k11_n200() {
    let lp = build_pivot_workload(11, 200);
    assert_solve_under_budget(&lp, 0.0, "bug3c_k11_n200_osa60_proxy");
}
