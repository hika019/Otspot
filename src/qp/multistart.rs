//! Multi-start local search (#5 Phase 2)。
//!
//! 非凸 QP では IPM が出発点ごとに異なる局所最適に収束する。
//! `solve_qp_multistart` は cold + (n_starts-1) random initial を順次解き、
//! 最良 objective を持つ結果を採用する。
//!
//! Phase 3 spatial Branch-and-Bound の incumbent (上界) 供給を主用途とする。
//! 単独では大域最適保証は無く、α-BB / McCormick (Phase 4-5) と組み合わせて
//! 完全な大域最適化器となる。

use crate::options::{MultiStartConfig, QpWarmStart, SolverOptions, StartStrategy};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use std::time::{Duration, Instant};

/// Numerical Recipes LCG 定数 (Park-Miller-Carta 系列)。full-period 2^32。
/// xorshift より弱いが multistart 用途では再現性とポータビリティを優先。
const LCG_A: u64 = 1_664_525;
const LCG_C: u64 = 1_013_904_223;
const LCG_M_MASK: u64 = 0xFFFF_FFFF;

/// LCG state を 1 step 進めて [0, 2^32) を返す。state=0 は LCG 固着なので
/// caller 側で 1 にクランプする (MultiStartConfig::default も同様)。
fn lcg_next(state: &mut u64) -> u64 {
    *state = (state.wrapping_mul(LCG_A).wrapping_add(LCG_C)) & LCG_M_MASK;
    *state
}

fn lcg_uniform_01(state: &mut u64) -> f64 {
    // [0, 2^32) を [0, 1) に正規化。+1.0 は半開区間化用。
    (lcg_next(state) as f64) / (u32::MAX as f64 + 1.0)
}

/// 1 出発点を box bounds 内で一様サンプリング。無限境界は ±unbounded_range にクランプ。
fn sample_random_box(state: &mut u64, bounds: &[(f64, f64)], unbounded_range: f64) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lb, ub)| {
            let lo = if lb.is_finite() { lb.max(-unbounded_range) } else { -unbounded_range };
            let hi = if ub.is_finite() { ub.min(unbounded_range) } else { unbounded_range };
            if hi <= lo {
                // 退化 (lb=ub の FX 変数等): lo を返す。
                return lo;
            }
            let u = lcg_uniform_01(state);
            lo + u * (hi - lo)
        })
        .collect()
}

/// LHS: 各次元を n_starts 個の strata に分割、列ごとに Fisher-Yates 順列、
/// 各 start は stratum 内で一様サンプリング。box 全域被覆性は pure random より高い。
/// 返り値は n_starts 個の初期点 (cold #0 も含むため caller が #0 を破棄する)。
fn latin_hypercube(
    seed: u64,
    n_starts: usize,
    bounds: &[(f64, f64)],
    unbounded_range: f64,
) -> Vec<Vec<f64>> {
    let n = bounds.len();
    if n == 0 || n_starts == 0 {
        return Vec::new();
    }
    // LHS 専用 stream (seed を bias して衝突回避)。
    let mut state = seed.wrapping_add(0xA5A5_5A5A_5A5A_A5A5);
    if state == 0 {
        state = 1;
    }

    let perms: Vec<Vec<usize>> = (0..n)
        .map(|_| {
            let mut p: Vec<usize> = (0..n_starts).collect();
            for i in (1..n_starts).rev() {
                let j = (lcg_next(&mut state) as usize) % (i + 1);
                p.swap(i, j);
            }
            p
        })
        .collect();

    (0..n_starts)
        .map(|s| {
            (0..n)
                .map(|j| {
                    let (lb, ub) = bounds[j];
                    let lo = if lb.is_finite() { lb.max(-unbounded_range) } else { -unbounded_range };
                    let hi = if ub.is_finite() { ub.min(unbounded_range) } else { unbounded_range };
                    if hi <= lo {
                        return lo;
                    }
                    let stratum = perms[j][s];
                    let u = lcg_uniform_01(&mut state);
                    let frac = (stratum as f64 + u) / n_starts as f64;
                    lo + frac * (hi - lo)
                })
                .collect()
        })
        .collect()
}

/// status を「より好ましい順」に rank 付け (小さいほど良い)。
fn status_rank(s: &SolveStatus) -> u8 {
    use SolveStatus::*;
    match s {
        Optimal => 0,
        LocallyOptimal => 1,
        SuboptimalSolution => 2,
        MaxIterations => 3,
        Timeout => 4,
        NumericalError => 5,
        NonConvex(_) => 6,
        Unbounded => 7,
        Infeasible => 8,
    }
}

/// 2 結果のうち「良い方」を返す。status 優先 → 同 status は finite obj 小優先。
fn pick_better(a: SolverResult, b: SolverResult) -> SolverResult {
    let ra = status_rank(&a.status);
    let rb = status_rank(&b.status);
    match ra.cmp(&rb) {
        std::cmp::Ordering::Less => a,
        std::cmp::Ordering::Greater => b,
        std::cmp::Ordering::Equal => match (a.objective.is_finite(), b.objective.is_finite()) {
            (true, true) => {
                if a.objective <= b.objective {
                    a
                } else {
                    b
                }
            }
            (true, false) => a,
            (false, true) => b,
            (false, false) => a,
        },
    }
}

/// 内部: warm_start_qp = warm を注入して solve_qp_with を呼ぶ。
fn solve_one(problem: &QpProblem, base_opts: &SolverOptions, warm: Option<QpWarmStart>) -> SolverResult {
    let mut opts = base_opts.clone();
    opts.warm_start_qp = warm;
    // 再入防止: 内部呼び出しでは multistart を剥がす。
    opts.multistart = None;
    crate::qp::solve_qp_with(problem, &opts)
}

/// random initial を生成 (cold #0 を除く #1..#n_starts 用)。
fn build_random_starts(config: &MultiStartConfig, bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let extra = config.n_starts.saturating_sub(1);
    if extra == 0 {
        return Vec::new();
    }
    let seed = if config.seed == 0 { 1 } else { config.seed };
    match config.strategy {
        StartStrategy::RandomBox => {
            let mut state = seed;
            (0..extra)
                .map(|_| sample_random_box(&mut state, bounds, config.unbounded_range))
                .collect()
        }
        StartStrategy::LatinHypercube => {
            // n_starts strata 全体を生成、cold #0 用の先頭 1 件を破棄。
            let all = latin_hypercube(seed, config.n_starts, bounds, config.unbounded_range);
            all.into_iter().skip(1).collect()
        }
    }
}

/// Multi-start QP solver。`config.n_starts == 1` は cold solve 1 回 (= 既存挙動)。
///
/// `options.warm_start_qp` は無視される (multistart が cold/random を全て生成するため)。
/// `options.timeout_secs` / `options.deadline` は全 start で共有 (deadline は入口で固定)。
pub fn solve_qp_multistart(
    problem: &QpProblem,
    options: &SolverOptions,
    config: &MultiStartConfig,
) -> SolverResult {
    if config.n_starts <= 1 {
        let mut opts = options.clone();
        opts.multistart = None;
        return crate::qp::solve_qp_with(problem, &opts);
    }

    // 全 start で共有する deadline を固定。timeout_secs は内部で deadline に変換し
    // 二重カウント (各 start で timeout_secs 分使う) を防ぐ。
    let mut shared_opts = options.clone();
    if shared_opts.deadline.is_none() {
        if let Some(secs) = shared_opts.timeout_secs {
            shared_opts.deadline = Some(Instant::now() + Duration::from_secs_f64(secs));
        }
    }
    shared_opts.timeout_secs = None;

    // start #0 = cold (warm_start_qp = None)。これが「first only no-op」時の返却値。
    let mut best = solve_one(problem, &shared_opts, None);

    let m_orig = problem.num_constraints;
    let randoms = build_random_starts(config, &problem.bounds);
    for x_init in randoms {
        if let Some(d) = shared_opts.deadline {
            if Instant::now() >= d {
                break;
            }
        }
        // y/μ は cold で初期化 (random initial は primal x のみ既知)。
        let warm = QpWarmStart {
            x: x_init,
            y: vec![0.0; m_orig],
            mu: 1.0,
        };
        let r = solve_one(problem, &shared_opts, Some(warm));
        best = pick_better(best, r);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_deterministic_and_in_unit_interval() {
        let mut a = 42u64;
        let mut b = 42u64;
        for _ in 0..1000 {
            let va = lcg_uniform_01(&mut a);
            let vb = lcg_uniform_01(&mut b);
            assert_eq!(va, vb, "LCG must be deterministic for same seed");
            assert!((0.0..1.0).contains(&va), "u = {va} out of [0, 1)");
        }
    }

    #[test]
    fn sample_random_box_respects_finite_bounds() {
        let bounds = vec![(-1.0, 1.0), (0.0, 5.0), (-100.0, -10.0)];
        let mut state = 12345u64;
        for _ in 0..100 {
            let x = sample_random_box(&mut state, &bounds, 1000.0);
            assert_eq!(x.len(), 3);
            for (xi, &(lb, ub)) in x.iter().zip(bounds.iter()) {
                assert!(*xi >= lb && *xi <= ub, "x={xi} out of [{lb}, {ub}]");
            }
        }
    }

    #[test]
    fn sample_random_box_clamps_infinite_bounds_to_unbounded_range() {
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)];
        let mut state = 7u64;
        for _ in 0..100 {
            let x = sample_random_box(&mut state, &bounds, 3.0);
            assert!(x[0] >= -3.0 && x[0] <= 3.0);
            assert!(x[1] >= 0.0 && x[1] <= 3.0);
        }
    }

    #[test]
    fn latin_hypercube_covers_each_stratum_once_per_dim() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
        let n_starts = 8;
        let pts = latin_hypercube(99, n_starts, &bounds, 100.0);
        assert_eq!(pts.len(), n_starts);
        for dim in 0..2 {
            let (lo, hi) = bounds[dim];
            let width = (hi - lo) / n_starts as f64;
            let mut hit = vec![false; n_starts];
            for p in pts.iter() {
                let stratum = ((p[dim] - lo) / width) as usize;
                let stratum = stratum.min(n_starts - 1);
                hit[stratum] = true;
            }
            assert!(
                hit.iter().all(|&b| b),
                "dim {dim}: not all strata covered: {hit:?}"
            );
        }
    }

    #[test]
    fn pick_better_prefers_lower_obj_when_status_ties() {
        let a = SolverResult {
            status: SolveStatus::LocallyOptimal,
            objective: -1.0,
            ..Default::default()
        };
        let b = SolverResult {
            status: SolveStatus::LocallyOptimal,
            objective: -5.0,
            ..Default::default()
        };
        let r = pick_better(a.clone(), b.clone());
        assert_eq!(r.objective, -5.0);
        let r = pick_better(b, a);
        assert_eq!(r.objective, -5.0);
    }

    #[test]
    fn pick_better_prefers_optimal_over_suboptimal_even_if_obj_worse() {
        let opt = SolverResult {
            status: SolveStatus::Optimal,
            objective: 100.0,
            ..Default::default()
        };
        let sub = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -100.0,
            ..Default::default()
        };
        let r = pick_better(opt.clone(), sub);
        assert_eq!(r.status, SolveStatus::Optimal);
    }
}
