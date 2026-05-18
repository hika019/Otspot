//! Multi-start local search (#5 Phase 2)。
//!
//! 非凸 QP では IPM が出発点ごとに異なる局所最適に収束する。
//! `solve_qp_multistart` は cold + (n_starts-1) random initial を解き、
//! 最良 objective を持つ結果を採用する。
//!
//! Phase 3 spatial Branch-and-Bound の incumbent (上界) 供給を主用途とする。
//! 単独では大域最適保証は無く、α-BB / McCormick (Phase 4-5) と組み合わせて
//! 完全な大域最適化器となる。
//!
//! ## 並列化
//!
//! `SolverOptions::threads` (user 指定) を使い、`min(n_starts, threads)` の
//! ローカル `rayon::ThreadPool` を構築し並列実行する。各 inner solve は
//! `threads = 1` 強制 (二重並列化抑止)。
//!
//! 全 thread から書き込まれる共有 state は無い (各 start は独立に SolverOptions を
//! clone)。結果は `Vec<SolverResult>` に index 順で collect → 順次 `pick_better`
//! で reduce。同 seed なら thread 数によらず結果は決定論的。
//!
//! ## test-only hooks
//!
//! `MultiStartHooks` は thread sentinel (peak parallelism 検証) 用の hook 注入
//! 機構。`pub(crate)` で外部に公開しない。`solve_qp_multistart` は内部で hooks
//! = None を渡し production code は影響なし。

use crate::options::{MultiStartConfig, QpWarmStart, SolverOptions, StartStrategy};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

/// Numerical Recipes LCG 定数 (Park-Miller-Carta 系列)。full-period 2^32。
/// xorshift より弱いが multistart 用途では再現性とポータビリティを優先。
const LCG_A: u64 = 1_664_525;
const LCG_C: u64 = 1_013_904_223;
const LCG_M_MASK: u64 = 0xFFFF_FFFF;

/// 無限境界変数のサンプリング半径 |x| <= range。
/// IPPMM `solve_ippmm_inner` cold-init 規約 (zero in bounds → 0 / lb_fin → lb+1 /
/// ub_fin → ub-1) と整合する origin 近傍半径。
pub(crate) const MULTISTART_UNBOUNDED_RANGE: f64 = 10.0;

/// LCG state を 1 step 進めて [0, 2^32) を返す。state=0 は LCG 固着なので
/// caller 側で 1 にクランプする (`build_random_starts` も同様)。
fn lcg_next(state: &mut u64) -> u64 {
    *state = (state.wrapping_mul(LCG_A).wrapping_add(LCG_C)) & LCG_M_MASK;
    *state
}

fn lcg_uniform_01(state: &mut u64) -> f64 {
    (lcg_next(state) as f64) / (u32::MAX as f64 + 1.0)
}

/// 1 出発点を box bounds 内で一様サンプリング。無限境界は ±MULTISTART_UNBOUNDED_RANGE
/// にクランプ。退化 (lb=ub の FX 変数) は lo を返す。
fn sample_random_box(state: &mut u64, bounds: &[(f64, f64)]) -> Vec<f64> {
    bounds
        .iter()
        .map(|&(lb, ub)| {
            let lo = if lb.is_finite() {
                lb.max(-MULTISTART_UNBOUNDED_RANGE)
            } else {
                -MULTISTART_UNBOUNDED_RANGE
            };
            let hi = if ub.is_finite() {
                ub.min(MULTISTART_UNBOUNDED_RANGE)
            } else {
                MULTISTART_UNBOUNDED_RANGE
            };
            if hi <= lo {
                return lo;
            }
            let u = lcg_uniform_01(state);
            lo + u * (hi - lo)
        })
        .collect()
}

/// LHS: 各次元を n_starts strata に分割、列ごとに Fisher-Yates 順列、各 start は
/// stratum 内で一様サンプリング。box 全域被覆性は pure random より高い。
/// 返り値は n_starts 個の初期点 (cold #0 も含むため caller が #0 を破棄する)。
fn latin_hypercube(seed: u64, n_starts: usize, bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let n = bounds.len();
    if n == 0 || n_starts == 0 {
        return Vec::new();
    }
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
                    let lo = if lb.is_finite() {
                        lb.max(-MULTISTART_UNBOUNDED_RANGE)
                    } else {
                        -MULTISTART_UNBOUNDED_RANGE
                    };
                    let hi = if ub.is_finite() {
                        ub.min(MULTISTART_UNBOUNDED_RANGE)
                    } else {
                        MULTISTART_UNBOUNDED_RANGE
                    };
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
/// 必ず `threads=1` & `multistart=None` に剥がし、二重並列 / 再入を断つ。
fn solve_one(
    problem: &QpProblem,
    base_opts: &SolverOptions,
    warm: Option<QpWarmStart>,
) -> SolverResult {
    let mut opts = base_opts.clone();
    opts.warm_start_qp = warm;
    opts.multistart = None;
    opts.threads = 1;
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
                .map(|_| sample_random_box(&mut state, bounds))
                .collect()
        }
        StartStrategy::LatinHypercube => {
            let all = latin_hypercube(seed, config.n_starts, bounds);
            all.into_iter().skip(1).collect()
        }
    }
}

/// thread sentinel 用 hook (peak parallelism 計測)。`pub(crate)` で外部非公開。
/// production code は `solve_qp_multistart_with_hooks(.., None)` を経由するため
/// 実行時 overhead はゼロ。
pub(crate) struct MultiStartHooks {
    pub on_solve_enter: Arc<dyn Fn() + Send + Sync>,
    pub on_solve_exit: Arc<dyn Fn() + Send + Sync>,
}

/// Multi-start QP solver。`config.n_starts == 1` は cold solve 1 回 (= 既存挙動)。
///
/// `options.warm_start_qp` は無視される (multistart が cold/random を全て生成するため)。
/// `options.timeout_secs` / `options.deadline` は全 start で共有 (deadline は入口で固定)。
/// `options.threads` で並列度 = `min(n_starts, threads)` を自動分配。
pub fn solve_qp_multistart(
    problem: &QpProblem,
    options: &SolverOptions,
    config: &MultiStartConfig,
) -> SolverResult {
    solve_qp_multistart_with_hooks(problem, options, config, None)
}

/// 実体 (内部 + テスト用 hooks)。production は `solve_qp_multistart` 経由 = hooks None。
pub(crate) fn solve_qp_multistart_with_hooks(
    problem: &QpProblem,
    options: &SolverOptions,
    config: &MultiStartConfig,
    hooks: Option<&MultiStartHooks>,
) -> SolverResult {
    if config.n_starts <= 1 {
        return solve_one(problem, options, None);
    }

    // 全 start で共有する deadline を固定。
    let mut shared_opts = options.clone();
    if shared_opts.deadline.is_none() {
        if let Some(secs) = shared_opts.timeout_secs {
            shared_opts.deadline = Some(Instant::now() + Duration::from_secs_f64(secs));
        }
    }
    shared_opts.timeout_secs = None;

    // 並列度: user 指定 threads と n_starts の min、1 未満は serial。
    let parallel = options.threads.max(1).min(config.n_starts);

    // 各 start (index 0..n_starts) の (warm option) を事前確定。index 0 = cold。
    let randoms = build_random_starts(config, &problem.bounds);
    let m_orig = problem.num_constraints;
    let warms: Vec<Option<QpWarmStart>> = std::iter::once(None)
        .chain(randoms.into_iter().map(|x| {
            Some(QpWarmStart {
                x,
                y: vec![0.0; m_orig],
                mu: 1.0,
            })
        }))
        .collect();

    let worker = |warm: Option<QpWarmStart>| -> SolverResult {
        if let Some(h) = hooks {
            (h.on_solve_enter)();
        }
        let r = solve_one(problem, &shared_opts, warm);
        if let Some(h) = hooks {
            (h.on_solve_exit)();
        }
        r
    };

    // 並列度 1 はオーバーヘッド回避のため直接 sequential。
    let results: Vec<SolverResult> = if parallel <= 1 {
        warms
            .into_iter()
            .take_while(|_| !shared_opts.deadline.is_some_and(|d| Instant::now() >= d))
            .map(worker)
            .collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build()
            .expect("rayon ThreadPool build failed");
        pool.install(|| {
            warms
                .into_par_iter()
                .map(worker)
                .collect::<Vec<SolverResult>>()
        })
    };

    // index 順 reduce で並列実行下でも決定論的。
    results
        .into_iter()
        .reduce(pick_better)
        .unwrap_or_else(|| solve_one(problem, &shared_opts, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn lcg_deterministic_and_in_unit_interval() {
        let mut a = 42u64;
        let mut b = 42u64;
        for _ in 0..1000 {
            let va = lcg_uniform_01(&mut a);
            let vb = lcg_uniform_01(&mut b);
            assert_eq!(va, vb);
            assert!((0.0..1.0).contains(&va));
        }
    }

    #[test]
    fn sample_random_box_respects_finite_bounds() {
        let bounds = vec![(-1.0, 1.0), (0.0, 5.0), (-100.0, -10.0)];
        let mut state = 12345u64;
        for _ in 0..100 {
            let x = sample_random_box(&mut state, &bounds);
            assert_eq!(x.len(), 3);
            for (xi, &(lb, ub)) in x.iter().zip(bounds.iter()) {
                assert!(*xi >= lb && *xi <= ub);
            }
        }
    }

    #[test]
    fn sample_random_box_clamps_infinite_bounds() {
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)];
        let mut state = 7u64;
        for _ in 0..100 {
            let x = sample_random_box(&mut state, &bounds);
            assert!(x[0].abs() <= MULTISTART_UNBOUNDED_RANGE);
            assert!(x[1] >= 0.0 && x[1] <= MULTISTART_UNBOUNDED_RANGE);
        }
    }

    #[test]
    fn latin_hypercube_covers_each_stratum_once_per_dim() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
        let n_starts = 8;
        let pts = latin_hypercube(99, n_starts, &bounds);
        assert_eq!(pts.len(), n_starts);
        for dim in 0..2 {
            let (lo, hi) = bounds[dim];
            let width = (hi - lo) / n_starts as f64;
            let mut hit = vec![false; n_starts];
            for p in pts.iter() {
                let stratum = (((p[dim] - lo) / width) as usize).min(n_starts - 1);
                hit[stratum] = true;
            }
            assert!(hit.iter().all(|&b| b), "dim {dim}: {hit:?}");
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

    /// 同 seed / 異 threads で結果完全一致 (race-free + index-ordered reduce)。
    #[test]
    fn multistart_deterministic_across_threads_count() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let c = vec![0.0_f64; 2];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let bounds = vec![(-3.0, 3.0); 2];
        let prob = QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap();

        let cfg = MultiStartConfig {
            n_starts: 8,
            seed: 0xABCD,
            strategy: StartStrategy::RandomBox,
        };
        let mut o1 = SolverOptions::default();
        o1.timeout_secs = Some(20.0);
        o1.threads = 1;
        let mut o4 = o1.clone();
        o4.threads = 4;
        let r1 = solve_qp_multistart(&prob, &o1, &cfg);
        let r4 = solve_qp_multistart(&prob, &o4, &cfg);
        assert!(
            (r1.objective - r4.objective).abs() < 1e-9,
            "thread=1 vs 4 must match: r1={} r4={}",
            r1.objective,
            r4.objective
        );
    }

    /// thread sentinel: hook で実観測した peak parallelism が
    /// `min(threads, n_starts)` 以下 (上限) かつ `>= 2` (実並列稼働) であることを assert。
    /// table-driven、複数 (threads, n_starts) パターンを cover。
    #[test]
    fn threads_actually_parallel_and_within_limit() {
        // (threads, n_starts, expected_peak_lower, expected_peak_upper)
        let cases = [
            (2_usize, 10_usize, 2_usize, 2_usize),
            (4, 10, 2, 4),
            (8, 16, 2, 8),
            (4, 2, 1, 2), // n_starts < threads → peak <= n_starts
        ];

        // 軽量 toy QP: small bilinear。多回 solve でも < 1s。
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let c = vec![0.0_f64; 2];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let bounds = vec![(-3.0, 3.0); 2];
        let prob = QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap();

        for (threads, n_starts, lo, hi) in cases.iter().copied() {
            let active = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let a_enter = active.clone();
            let p_enter = peak.clone();
            let a_exit = active.clone();
            let hooks = MultiStartHooks {
                on_solve_enter: Arc::new(move || {
                    let n = a_enter.fetch_add(1, Ordering::SeqCst) + 1;
                    // 既存 peak より大きければ書き戻し
                    let mut prev = p_enter.load(Ordering::SeqCst);
                    while n > prev {
                        match p_enter.compare_exchange(
                            prev,
                            n,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        ) {
                            Ok(_) => break,
                            Err(actual) => prev = actual,
                        }
                    }
                    // 並列タスクが重なる時間窓を作る (50 ms)。
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }),
                on_solve_exit: Arc::new(move || {
                    a_exit.fetch_sub(1, Ordering::SeqCst);
                }),
            };
            let cfg = MultiStartConfig {
                n_starts,
                seed: 1,
                strategy: StartStrategy::RandomBox,
            };
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(30.0);
            opts.threads = threads;
            let _ = solve_qp_multistart_with_hooks(&prob, &opts, &cfg, Some(&hooks));

            let observed = peak.load(Ordering::SeqCst);
            assert!(
                observed >= lo,
                "threads={threads} n_starts={n_starts}: peak={observed} expected >= {lo} (並列稼働不足)"
            );
            assert!(
                observed <= hi,
                "threads={threads} n_starts={n_starts}: peak={observed} exceeds upper {hi} (上限超過)"
            );
        }
    }

    /// `threads=1` は完全 serial (peak=1) を保証 (deterministic test 環境保護)。
    #[test]
    fn threads_eq_1_is_serial() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let c = vec![0.0_f64; 2];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let bounds = vec![(-3.0, 3.0); 2];
        let prob = QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap();

        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let a_enter = active.clone();
        let p_enter = peak.clone();
        let a_exit = active.clone();
        let hooks = MultiStartHooks {
            on_solve_enter: Arc::new(move || {
                let n = a_enter.fetch_add(1, Ordering::SeqCst) + 1;
                p_enter.fetch_max(n, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(10));
            }),
            on_solve_exit: Arc::new(move || {
                a_exit.fetch_sub(1, Ordering::SeqCst);
            }),
        };
        let cfg = MultiStartConfig {
            n_starts: 6,
            seed: 1,
            strategy: StartStrategy::RandomBox,
        };
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(20.0);
        opts.threads = 1;
        let _ = solve_qp_multistart_with_hooks(&prob, &opts, &cfg, Some(&hooks));
        assert_eq!(peak.load(Ordering::SeqCst), 1, "threads=1 must be serial");
    }
}
