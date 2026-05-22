//! Simplex crash basis sentinel: 構造列で artificial を被覆して Phase I の
//! 反復数を削減することを検証する。
//!
//! 検出原理:
//!  - cold start (crash 無効化): 全 Eq/Ge 行で artificial が必要、Phase I が
//!    すべての artificial を駆出する。
//!  - crash 適用: 構造列 (符号一致 pivot) で行を被覆 → Phase I の駆出対象を削減。
//!  - sentinel 強度: 2 種類の構造 LP で TOLERANCE 単一の偽 PASS を排除:
//!    (a) network-style: 各 Eq 行に 1 つの singleton 列 + 共通 hub 列
//!    (b) ill-scaled network: (a) 同型 + flow 列 scale [1e-4, 1e4] (pivot 安定性)
//!
//! 期待: iter ratio < CRASH_ITER_RATIO_UPPER (= 0.9)。
//! no-op 化 (try_apply_crash を常に None に倒す) で sentinel が FAIL することを実証済。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::sparse::CscMatrix;

/// 最低有効効果 10% (= cold の 0.9 倍以下 iter)。これより緩いと no-op fallback が
/// sentinel から漏れる。
const CRASH_ITER_REDUCTION_MARGIN: f64 = 0.1;
const CRASH_ITER_RATIO_UPPER: f64 = 1.0 - CRASH_ITER_REDUCTION_MARGIN;

/// network-flow 風 LP: 各 Eq 行に singleton 構造列 (cap 変数) + 共通の少数 hub 列。
/// network LP (ken, pds 系) と同じ構造で singleton crash の効きが大きい。
///
/// 構造:
/// - n_flow flow 変数 (各々 1 行のみに登場、係数=1.0 or -1.0)、b 行は 1.0
/// - n_hub hub 変数 (全行に登場、係数小、cost のみで結合)
/// - 制約: 各 i に対し x_flow[i] + Σ_h hub_coeff[h,i] * x_hub[h] = b[i] (Eq)
fn build_network_lp(n_flow: usize, n_hub: usize, seed_init: u64) -> LpProblem {
    let mut seed: u64 = seed_init;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
    };

    let n = n_flow + n_hub;
    let m_eq = n_flow;
    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();

    // singleton flow 列: 行 i に col i のみ、係数 1.0
    for i in 0..n_flow {
        a_rows.push(i);
        a_cols.push(i);
        a_vals.push(1.0);
    }
    // hub 列: 全行に小さい係数
    for h in 0..n_hub {
        for i in 0..n_flow {
            let v = 0.01 + 0.02 * (next(&mut seed) + 1.0) * 0.5;
            a_rows.push(i);
            a_cols.push(n_flow + h);
            a_vals.push(v);
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_eq, n).unwrap();

    // b は正 + epsilon noise (実行可能性確保)
    let b: Vec<f64> = (0..m_eq).map(|_| 1.0 + (next(&mut seed) + 1.0) * 0.25).collect();
    let c: Vec<f64> = (0..n).map(|_| next(&mut seed)).collect();
    let bounds = vec![(0.0_f64, 10.0_f64); n];

    LpProblem::new_general(
        c, a, b, vec![ConstraintType::Eq; m_eq], bounds, None,
    ).unwrap()
}

/// ill-scaled network LP: build_network_lp 同型構造に列 scale 8 桁 dynamic range を
/// 載せた variant。crash basis が pivot 数値安定性 (CRASH_PIVOT_REL) を維持できるかの
/// sentinel。
fn build_ill_scaled_network_lp(n_flow: usize, n_hub: usize, seed_init: u64) -> LpProblem {
    let mut seed: u64 = seed_init;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
    };

    let n = n_flow + n_hub;
    let m_eq = n_flow;
    // log-uniform [1e-4, 1e4] scaling per flow col (singleton pivot もこれで scaled)
    let flow_scale: Vec<f64> = (0..n_flow).map(|j| {
        let exp = -4.0 + 8.0 * (j as f64 / (n_flow - 1) as f64);
        10.0_f64.powf(exp)
    }).collect();

    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();

    for i in 0..n_flow {
        a_rows.push(i);
        a_cols.push(i);
        a_vals.push(flow_scale[i]);
    }
    for h in 0..n_hub {
        for i in 0..n_flow {
            let v = (0.01 + 0.02 * (next(&mut seed) + 1.0) * 0.5) * flow_scale[i];
            a_rows.push(i);
            a_cols.push(n_flow + h);
            a_vals.push(v);
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_eq, n).unwrap();

    // b は flow_scale を反映して符号一致が自明な構造に。
    let b: Vec<f64> = (0..m_eq).map(|i| flow_scale[i] * (1.0 + (next(&mut seed) + 1.0) * 0.25)).collect();
    let c: Vec<f64> = (0..n).map(|_| next(&mut seed)).collect();
    let bounds = vec![(0.0_f64, 100.0_f64); n];

    LpProblem::new_general(
        c, a, b, vec![ConstraintType::Eq; m_eq], bounds, None,
    ).unwrap()
}

/// network-style LP で crash が iter 数を削減すること。
#[test]
fn crash_basis_reduces_iters_network_lp() {
    let problem = build_network_lp(1200, 8, 0x_F1_F2_F3_F4_F5_F6_F7_F8);

    let mut cold_opts = SolverOptions::default();
    cold_opts.use_lp_crash_basis = false;
    cold_opts.timeout_secs = Some(60.0);
    let cold = solve_with(&problem, &cold_opts);
    assert_eq!(cold.status, SolveStatus::Optimal,
        "cold must be Optimal; got {:?}", cold.status);

    let mut crash_opts = SolverOptions::default();
    crash_opts.use_lp_crash_basis = true;
    crash_opts.timeout_secs = Some(60.0);
    let crashed = solve_with(&problem, &crash_opts);
    assert_eq!(crashed.status, SolveStatus::Optimal,
        "crash must be Optimal; got {:?}", crashed.status);

    let obj_diff = (crashed.objective - cold.objective).abs()
        / (1.0 + cold.objective.abs());
    assert!(obj_diff < 1e-6, "crash obj drift: {:.3e}", obj_diff);

    let iter_ratio = crashed.iterations as f64 / cold.iterations.max(1) as f64;
    eprintln!(
        "LP_CRASH_SMOKE_NETWORK: cold_iters={} crash_iters={} ratio={:.3}",
        cold.iterations, crashed.iterations, iter_ratio
    );

    // silent SKIP 検出: crash が no-op 化されれば cold と一致。
    assert!(
        crashed.iterations < cold.iterations,
        "crash appears silently dropped: cold={} crash={} (expected crash < cold)",
        cold.iterations, crashed.iterations
    );

    assert!(
        iter_ratio < CRASH_ITER_RATIO_UPPER,
        "crash iter reduction below margin: ratio={:.3} ≥ {:.3} (cold={} crash={})",
        iter_ratio, CRASH_ITER_RATIO_UPPER, cold.iterations, crashed.iterations
    );
}

/// ill-scaled network LP で crash が pivot 安定性 (CRASH_PIVOT_REL) を守りつつ
/// iter 削減する。列 scale 8 桁 dynamic range に対し markowitz threshold が機能。
#[test]
fn crash_basis_handles_ill_scaled_network_lp() {
    let problem = build_ill_scaled_network_lp(800, 6, 0x_A1_A2_A3_A4_A5_A6_A7_A8);

    let mut cold_opts = SolverOptions::default();
    cold_opts.use_lp_crash_basis = false;
    cold_opts.timeout_secs = Some(60.0);
    let cold = solve_with(&problem, &cold_opts);
    assert_eq!(cold.status, SolveStatus::Optimal,
        "ill-scaled cold must Optimal; got {:?}", cold.status);

    let mut crash_opts = SolverOptions::default();
    crash_opts.use_lp_crash_basis = true;
    crash_opts.timeout_secs = Some(60.0);
    let crashed = solve_with(&problem, &crash_opts);
    assert_eq!(crashed.status, SolveStatus::Optimal,
        "ill-scaled crash must Optimal; got {:?}", crashed.status);

    let obj_diff = (crashed.objective - cold.objective).abs()
        / (1.0 + cold.objective.abs());
    assert!(obj_diff < 1e-4, "ill-scaled crash obj drift: {:.3e}", obj_diff);

    let iter_ratio = crashed.iterations as f64 / cold.iterations.max(1) as f64;
    eprintln!(
        "LP_CRASH_SMOKE_NETWORK_ILL: cold_iters={} crash_iters={} ratio={:.3}",
        cold.iterations, crashed.iterations, iter_ratio
    );

    assert!(
        crashed.iterations < cold.iterations,
        "ill-scaled crash silently dropped: cold={} crash={}",
        cold.iterations, crashed.iterations
    );

    assert!(
        iter_ratio < CRASH_ITER_RATIO_UPPER,
        "ill-scaled crash iter reduction insufficient: ratio={:.3} ≥ {:.3} (cold={} crash={})",
        iter_ratio, CRASH_ITER_RATIO_UPPER, cold.iterations, crashed.iterations
    );
}
