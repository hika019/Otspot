//! Correctness tests for the MIP speed-bench synthetic generators.
//!
//! The generators live in the bench's `kernels` module and are pulled in here
//! via `#[path]`, so there is a single source of truth — the tested problems
//! cannot drift from the benchmarked ones. The convex MIQP built here reuses
//! the SAME shared `Q = LLᵀ + ridge` construction but, unlike the bench, drops
//! the side constraints and fixes an all-integer `[0,3]` box so the optimum is
//! brute-forceable.
//!
//! Strategy:
//!  - All-integer knapsack (n ≤ 10): brute-force verifies the solver to 1e-3.
//!  - Convex MIQP: PSD assertion + brute-force objective over the integer box.
//!  - Assignment MILP: status valid (Optimal/Infeasible) across seeds/densities.
//!  - Multiple data patterns (3 seeds per case) guard against single-seed luck.

use otspot::{
    options::{MipConfig, SolverOptions},
    problem::SolveStatus,
    solve_milp_with_stats, solve_miqp_with_stats, CscMatrix, MiqpProblem, QpProblem,
};

#[path = "../otspot-dev/src/bin/mip_speed_bench/kernels.rs"]
mod kernels;
use kernels::{
    build_convex_qc, convex_miqp_lcg, convex_q_to_csc, gen_assignment_milp, gen_knapsack_milp,
    knapsack_weights_capacity,
};

/// Integer-box upper bound of the brute-forceable MIQP built below.
const MIQP_INT_UB: i64 = 3;

// ---------------------------------------------------------------------------
// Test-only brute-force references and the small (brute-forceable) MIQP build
// ---------------------------------------------------------------------------

fn brute_force_all_int_knapsack(n: usize, c: &[f64], weights: &[f64], cap: f64) -> Option<f64> {
    let mut best = None::<f64>;
    for mask in 0u32..(1u32 << n) {
        let w: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| weights[j]).sum();
        if w <= cap + 1e-9 {
            let obj: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| c[j]).sum();
            best = Some(best.map_or(obj, |b: f64| b.min(obj)));
        }
    }
    best
}

/// Unconstrained, all-integer convex MIQP over the `[0,3]` box, built from the
/// shared `Q = LLᵀ + ridge` construction. Returns the problem plus the dense `Q`
/// and linear `c` so the brute-force can score the very same instance.
fn build_test_miqp(n: usize, seed: u64) -> (MiqpProblem, Vec<Vec<f64>>, Vec<f64>) {
    let mut lcg = convex_miqp_lcg(seed);
    let (q_dense, c) = build_convex_qc(&mut lcg, n);
    let q = convex_q_to_csc(&q_dense, n);
    let qp = QpProblem::new_all_le(
        q,
        c.clone(),
        CscMatrix::new(0, n),
        vec![],
        vec![(0.0, MIQP_INT_UB as f64); n],
    )
    .unwrap();
    let prob = MiqpProblem::new(qp, (0..n).collect()).unwrap();
    (prob, q_dense, c)
}

/// Exact integer optimum of min 0.5 xᵀQx + cᵀx over x ∈ {0..=ub}^n (the
/// generator imposes no other constraints). Full integer-box enumeration; n ≤ 5.
fn brute_force_miqp(q_dense: &[Vec<f64>], c: &[f64], n: usize, ub: i64) -> f64 {
    let pts = (ub + 1) as usize;
    let total = pts.pow(n as u32);
    let mut best = f64::INFINITY;
    let mut x = vec![0.0_f64; n];
    for code in 0..total {
        let mut rem = code;
        for xi in x.iter_mut() {
            *xi = (rem % pts) as f64;
            rem /= pts;
        }
        let mut xqx = 0.0_f64;
        for (i, qi) in q_dense.iter().enumerate().take(n) {
            for (j, &qij) in qi.iter().enumerate().take(n) {
                xqx += qij * x[i] * x[j];
            }
        }
        let lin: f64 = (0..n).map(|i| c[i] * x[i]).sum();
        let obj = 0.5 * xqx + lin;
        if obj < best {
            best = obj;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_opts_20s() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(20.0);
    o
}

fn default_cfg() -> MipConfig {
    MipConfig::default()
}

// ---------------------------------------------------------------------------
// Tests: All-integer knapsack — brute-force vs solver (multiple seeds)
// ---------------------------------------------------------------------------

/// For n=6 (2^6=64 points), 3 seeds: solver must match brute-force to 1e-3.
#[test]
fn knapsack_all_int_n6_matches_brute_force_3seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[42u64, 137, 999] {
        let prob = gen_knapsack_milp(6, 1.0, seed);
        let c = prob.lp.c.clone();
        let (weights, cap) = knapsack_weights_capacity(6, seed);
        let bf = brute_force_all_int_knapsack(6, &c, &weights, cap)
            .expect("6-var knapsack feasible");
        let (res, _stats) = solve_milp_with_stats(&prob, &opts, &cfg);
        assert_eq!(res.status, SolveStatus::Optimal, "seed={} must solve to Optimal", seed);
        let solver_obj = res.objective;
        assert!(
            (solver_obj - bf).abs() < 1e-3,
            "seed={}: solver={:.4} bf={:.4} diff={:.4}",
            seed, solver_obj, bf, (solver_obj - bf).abs()
        );
    }
}

/// n=8 (2^8=256): tighter test, still fast.
#[test]
fn knapsack_all_int_n8_matches_brute_force_3seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[1u64, 2, 3] {
        let prob = gen_knapsack_milp(8, 1.0, seed);
        let c = prob.lp.c.clone();
        let (weights, cap) = knapsack_weights_capacity(8, seed);
        let bf = brute_force_all_int_knapsack(8, &c, &weights, cap)
            .expect("8-var knapsack feasible");
        let (res, _stats) = solve_milp_with_stats(&prob, &opts, &cfg);
        assert_eq!(res.status, SolveStatus::Optimal, "seed={} must solve to Optimal", seed);
        assert!(
            (res.objective - bf).abs() < 1e-3,
            "seed={}: solver={:.4} bf={:.4}",
            seed, res.objective, bf
        );
    }
}

/// n=10 (2^10=1024): 3 seeds, validates B&B search depth.
#[test]
fn knapsack_all_int_n10_matches_brute_force_3seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[100u64, 200, 300] {
        let prob = gen_knapsack_milp(10, 1.0, seed);
        let c = prob.lp.c.clone();
        let (weights, cap) = knapsack_weights_capacity(10, seed);
        let bf = brute_force_all_int_knapsack(10, &c, &weights, cap)
            .expect("10-var knapsack feasible");
        let (res, _) = solve_milp_with_stats(&prob, &opts, &cfg);
        assert_eq!(res.status, SolveStatus::Optimal, "seed={}", seed);
        assert!(
            (res.objective - bf).abs() < 1e-3,
            "seed={}: solver={:.4} bf={:.4}",
            seed, res.objective, bf
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Convex MIQP — PSD verification + brute-force objective
// ---------------------------------------------------------------------------

/// Q = LLᵀ + ridge·I must be PSD for all n and seeds.
#[test]
fn convex_miqp_generator_is_always_psd_multiple_seeds() {
    for &n in &[4usize, 8, 12] {
        for &seed in &[0u64, 42, 999] {
            let (prob, _, _) = build_test_miqp(n, seed);
            assert!(
                prob.is_convex(),
                "n={} seed={}: Q = LLᵀ + ridge must be PSD (gen bug)",
                n, seed
            );
        }
    }
}

/// Convex MIQP small (n=4) solves to Optimal with finite objective.
#[test]
fn convex_miqp_n4_solves_optimal_3seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[42u64, 137, 999] {
        let (prob, _, _) = build_test_miqp(4, seed);
        let (res, stats) = solve_miqp_with_stats(&prob, &opts, &cfg);
        assert_eq!(res.status, SolveStatus::Optimal,
            "n=4 seed={} should be Optimal, got {:?}", seed, res.status);
        assert!(res.objective.is_finite(), "objective finite n=4 seed={}", seed);
        assert!(stats.nodes_processed > 0, "must have explored at least root");
    }
}

/// MIQP solver objective must equal the exact integer-box optimum (brute-force
/// over {0..=3}^n). The only ground-truth objective check for convex MIQP — the
/// other MIQP tests assert status/finiteness only, so a silently-wrong objective
/// would pass them. Multiple n and seeds.
#[test]
fn convex_miqp_objective_matches_brute_force_multiple_seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &n in &[3usize, 4] {
        for &seed in &[42u64, 137, 999] {
            let (prob, q_dense, c) = build_test_miqp(n, seed);
            let bf = brute_force_miqp(&q_dense, &c, n, MIQP_INT_UB);
            let (res, _) = solve_miqp_with_stats(&prob, &opts, &cfg);
            assert_eq!(res.status, SolveStatus::Optimal,
                "n={} seed={} must be Optimal, got {:?}", n, seed, res.status);
            assert!(
                (res.objective - bf).abs() < 1e-3,
                "n={} seed={}: solver={:.6} brute_force={:.6} diff={:.6}",
                n, seed, res.objective, bf, (res.objective - bf).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: Assignment MILP — structural checks (multiple seeds, densities)
// ---------------------------------------------------------------------------

/// Assignment MILP n=8 solves quickly with a valid status.
#[test]
fn assignment_milp_n8_optimal_multiple_seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[10u64, 20, 30] {
        for &density in &[0.3_f64, 0.6] {
            let prob = gen_assignment_milp(8, 1.0, density, seed);
            let (res, _) = solve_milp_with_stats(&prob, &opts, &cfg);
            // Optimal or Infeasible (tight rhs may make it infeasible).
            assert!(
                matches!(res.status, SolveStatus::Optimal | SolveStatus::Infeasible),
                "n=8 seed={} density={}: unexpected status {:?}",
                seed, density, res.status
            );
        }
    }
}

/// Mixed-integer (50% int, 50% continuous) assignment MILP n=10 converges.
#[test]
fn assignment_milp_mixed_n10_converges_3seeds() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[7u64, 77, 777] {
        let prob = gen_assignment_milp(10, 0.5, 0.4, seed);
        let (res, _) = solve_milp_with_stats(&prob, &opts, &cfg);
        assert!(
            matches!(res.status, SolveStatus::Optimal | SolveStatus::Infeasible),
            "seed={}: unexpected {:?}", seed, res.status
        );
    }
}

/// Knapsack with continuous variables (int_ratio=0.5) relaxes half the integer
/// constraints, so the mixed optimum is ≤ the all-integer optimum.
#[test]
fn knapsack_mixed_int_cont_objective_le_all_int_bound() {
    let opts = default_opts_20s();
    let cfg = default_cfg();
    for &seed in &[42u64, 137, 999] {
        let mixed = gen_knapsack_milp(8, 0.5, seed);
        let all_int = gen_knapsack_milp(8, 1.0, seed);
        let (r_mixed, _) = solve_milp_with_stats(&mixed, &opts, &cfg);
        let (r_allint, _) = solve_milp_with_stats(&all_int, &opts, &cfg);
        if r_mixed.status == SolveStatus::Optimal && r_allint.status == SolveStatus::Optimal {
            assert!(
                r_mixed.objective <= r_allint.objective + 1e-4,
                "seed={}: mixed_obj={:.4} should be ≤ allint_obj={:.4}",
                seed, r_mixed.objective, r_allint.objective
            );
        }
    }
}
