//! Multistart 並列 path の deadline 短絡 sentinel。
//!
//! parallel branch (`into_par_iter().map(worker).collect()`) は rayon が queued task を
//! mid-flight cancel できず、deadline 超過後も残 worker が presolve/attempt setup を
//! 貫通して wall-clock が伸びていた。worker 入口で deadline を確認し超過済は Timeout
//! stub を返すよう修正。
//!
//! no-op 実証は src/qp/multistart.rs の unit test
//! `deadline_shortcut_skips_post_deadline_workers` で hook 経由 disable し wall-clock 2x
//! 膨張を fact 化。当 integration test は public API 経由の wall-clock 観測 sentinel。
//!
//! 複数 data pattern: (n_starts, threads, deadline) 4 組 × indef shape 2 種 = 8 case。

use otspot::options::{MultiStartConfig, StartStrategy};
use otspot::qp::multistart::solve_qp_multistart;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;
use otspot::SolverOptions;
use std::time::Instant;

/// 中規模 indefinite QP: n=20 で per-solve が deadline (300ms) より長くなりやすい。
/// shortcut 無効化下では wall-clock が顕著に伸びる工夫。
fn build_indef_n(n: usize, bnd: f64) -> QpProblem {
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    // 半数を負 (concave) 半数を正 (convex) で indefinite 構成
    let vals: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { -2.0 } else { 2.0 })
        .collect();
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let c = vec![0.0; n];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
    let bounds = vec![(-bnd, bnd); n];
    QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap()
}

/// deadline + TOLERANCE_MS 以内に wall-clock が収まることを assert。
/// TOLERANCE_MS は (per-solve overhead × parallel) + rayon dispatch を吸収する保守値。
const TOLERANCE_MS: u128 = 2_000;

#[test]
#[allow(clippy::type_complexity)]
fn parallel_path_wallclock_bounded_by_deadline_table_driven() {
    // 4 case × 2 shape = 8 ケース、全 case で wall_clock <= deadline + TOLERANCE。
    let cases: &[(usize, usize, u64)] = &[
        // (n_starts, threads, deadline_ms)
        (8, 2, 100),
        (8, 4, 150),
        (16, 4, 200),
        (16, 8, 250),
    ];
    let problems: &[(&str, fn() -> QpProblem)] = &[
        ("indef_n20", || build_indef_n(20, 5.0)),
        ("indef_n10", || build_indef_n(10, 3.0)),
    ];

    for (label, build) in problems.iter() {
        let prob = build();
        for &(n_starts, threads, deadline_ms) in cases.iter() {
            let mut cfg = MultiStartConfig::default();
            cfg.n_starts = n_starts;
            cfg.seed = 0xC0FFEE;
            cfg.strategy = StartStrategy::RandomBox;
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(deadline_ms as f64 / 1000.0);
            opts.threads = threads;

            let t0 = Instant::now();
            let _ = solve_qp_multistart(&prob, &opts, &cfg);
            let dur_ms = t0.elapsed().as_millis();

            let budget = deadline_ms as u128 + TOLERANCE_MS;
            assert!(
                dur_ms <= budget,
                "{label} (n_starts={n_starts}, threads={threads}, deadline={deadline_ms}ms): \
                 wall_clock={dur_ms}ms exceeded budget={budget}ms (= deadline + {TOLERANCE_MS}ms tol)"
            );
        }
    }
}

#[test]
fn parallel_path_wallclock_bounded_at_near_zero_deadline() {
    // deadline 1ms (= 実質直ぐ越え) で大量 starts → shortcut で wall-clock 短く
    // (timeout_secs 経由 = public API、SolverOptions::deadline は pub(crate) のため)。
    let prob = build_indef_n(20, 5.0);
    let mut cfg = MultiStartConfig::default();
    cfg.n_starts = 16;
    cfg.seed = 1;
    cfg.strategy = StartStrategy::RandomBox;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(0.001); // 1ms
    opts.threads = 4;

    let t0 = Instant::now();
    let _r = solve_qp_multistart(&prob, &opts, &cfg);
    let dur_ms = t0.elapsed().as_millis();
    // 全 worker shortcut → 1秒以内 (rayon thread pool 構築 + 最初 batch slip 込)
    assert!(
        dur_ms <= 1_000,
        "near-zero deadline multistart should finish quickly, got {dur_ms}ms"
    );
}

#[test]
fn serial_path_also_short_circuits() {
    // threads=1 の serial path でも take_while + worker check で deadline 超過後の
    // worker が起動しないこと (= serial path の既存挙動を保護)。
    let prob = build_indef_n(20, 5.0);
    let mut cfg = MultiStartConfig::default();
    cfg.n_starts = 16;
    cfg.seed = 1;
    cfg.strategy = StartStrategy::RandomBox;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(0.2);
    opts.threads = 1;

    let t0 = Instant::now();
    let _ = solve_qp_multistart(&prob, &opts, &cfg);
    let dur_ms = t0.elapsed().as_millis();
    assert!(
        dur_ms <= 200 + TOLERANCE_MS,
        "serial path wall_clock={dur_ms}ms exceeds deadline=200ms + {TOLERANCE_MS}ms tol"
    );
}
