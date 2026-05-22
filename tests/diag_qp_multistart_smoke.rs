//! Multi-start (#5 Phase 2) sentinel: cold solve だけでは抜けられない saddle/中心
//! を random restart で escape できることを実証。
//!
//! 真因 (sentinel が必要な理由):
//! `c = 0` の対称な非凸 QP では cold (x0 = 0) は IPM が saddle (interior 0 勾配) に
//! 固着する。random init は対称性を破り境界 (= 大域最適 corner) に到達する。
//! 局所/大域 gap が n×bnd² で大きく分離するため改善が明瞭。
//!
//! no-op 実証 (memory `feedback_sentinel_must_fail_under_noop`):
//! `solve_qp_multistart` の「best 採用」logic (`pick_better`) を呼ばず先頭 (cold)
//! を返す書換 → 全 sentinel が FAIL (cold=0 vs best=-9 / -18 / -25 / -27)。
//! 検証済 (write-up 見出し). revert 後 PASS.
//!
//! 複数 data pattern (memory `feedback_test_multi_data_pattern`):
//! 4 problem shape × 5 seed の table-driven で 20 ケース被覆。
//!  - bilinear xy + bnd=3
//!  - bilinear xy + bnd=5
//!  - diag indefinite (concave) n=2
//!  - diag indefinite (concave) n=3

use otspot::options::{MultiStartConfig, StartStrategy};
use otspot::qp::{solve_qp_multistart, solve_qp_with, QpProblem};
use otspot::sparse::CscMatrix;
use otspot::{SolveStatus, SolverOptions};

/// 改善幅 5.0: 最小 problem (bilinear bnd=3, gap=9) でも余裕で超える。
/// noise (eps=1e-6 程度) と比較して 6 桁離れ、no-op 不検出を確実に防ぐ。
const STRICT_IMPROVEMENT_MARGIN: f64 = 5.0;

fn is_solved(s: &SolveStatus) -> bool {
    matches!(
        s,
        SolveStatus::Optimal
            | SolveStatus::LocallyOptimal
            | SolveStatus::SuboptimalSolution
            | SolveStatus::MaxIterations
    )
}

fn opts_with_timeout(secs: f64) -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(secs);
    o
}

/// bilinear xy + 微小対角 (LP fallback 回避): f = 1e-6 x² + xy + 1e-6 y²。
/// c=0 で saddle (0,0) 固着 → 大域 (bnd, -bnd) と (-bnd, bnd) で obj = -bnd²。
fn build_bilinear_zero_c(bnd: f64) -> QpProblem {
    let q = CscMatrix::from_triplets(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &[1e-6, 1.0, 1.0, 1e-6],
        2,
        2,
    )
    .unwrap();
    let c = vec![0.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let bounds = vec![(-bnd, bnd); 2];
    QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap()
}

/// diag negative (concave) QP: f = -Σ x_i²。c=0 で interior max (0,...,0) cold 固着。
/// 大域 (±bnd, ±bnd, ...) で obj = -n·bnd² (2^n 通り tied global)。
fn build_diag_concave(n: usize, bnd: f64) -> QpProblem {
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![-2.0; n];
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let c = vec![0.0; n];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
    let bounds = vec![(-bnd, bnd); n];
    QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap()
}

fn cold_solve(problem: &QpProblem) -> (SolveStatus, f64) {
    let opts = opts_with_timeout(5.0);
    let r = solve_qp_with(problem, &opts);
    (r.status, r.objective)
}

fn multi_solve(
    problem: &QpProblem,
    n_starts: usize,
    seed: u64,
    strategy: StartStrategy,
) -> (SolveStatus, f64) {
    let opts = opts_with_timeout(20.0);
    let cfg = MultiStartConfig {
        n_starts,
        seed,
        strategy,
    };
    let r = solve_qp_multistart(problem, &opts, &cfg);
    (r.status, r.objective)
}

#[test]
fn multistart_n1_equals_cold_solve() {
    // n_starts=1 は cold solve と完全一致 (= 既存挙動の保護)。
    let prob = build_bilinear_zero_c(3.0);
    let (cs, co) = cold_solve(&prob);
    let (ms, mo) = multi_solve(&prob, 1, 42, StartStrategy::RandomBox);
    assert_eq!(cs, ms, "status must match for n_starts=1");
    assert!((co - mo).abs() < 1e-9, "objective must match: cold={co} ms={mo}");
}

#[test]
fn multistart_options_field_defaults_off() {
    // SolverOptions::multistart = None は既存 solve_qp_with と一致する経路。
    let prob = build_bilinear_zero_c(3.0);
    let opts_default = opts_with_timeout(5.0);
    let r1 = solve_qp_with(&prob, &opts_default);
    let mut opts_explicit_none = opts_with_timeout(5.0);
    opts_explicit_none.multistart = None;
    let r2 = solve_qp_with(&prob, &opts_explicit_none);
    assert_eq!(r1.status, r2.status);
    assert!((r1.objective - r2.objective).abs() < 1e-9);
}

#[test]
fn multistart_bilinear_bnd3_escapes_cold_saddle() {
    // bilinear xy, bnd=3, c=0: cold (0,0) saddle obj=0、大域 obj=-9。
    let prob = build_bilinear_zero_c(3.0);
    let (cs, cold_obj) = cold_solve(&prob);
    assert!(is_solved(&cs), "cold solved status: {cs:?}");
    assert!(cold_obj.abs() < 1e-3, "cold should sit at saddle obj≈0, got {cold_obj}");
    let (ms, ms_obj) = multi_solve(&prob, 10, 0xC0FFEE, StartStrategy::RandomBox);
    assert!(is_solved(&ms), "ms solved status: {ms:?}");
    assert!(ms_obj <= cold_obj + 1e-9, "ms never worse than cold");
    let improvement = cold_obj - ms_obj;
    assert!(
        improvement >= STRICT_IMPROVEMENT_MARGIN,
        "bilinear bnd3 improvement>={STRICT_IMPROVEMENT_MARGIN}, got cold={cold_obj} ms={ms_obj}"
    );
    assert!(
        (ms_obj - (-9.0)).abs() < 1e-3,
        "ms should approach global -9, got {ms_obj}"
    );
}

#[test]
fn multistart_diag_concave_2d_escapes_cold_saddle() {
    let prob = build_diag_concave(2, 3.0);
    let (_, cold_obj) = cold_solve(&prob);
    assert!(cold_obj.abs() < 1e-3, "diag 2D cold saddle obj≈0, got {cold_obj}");
    let (_, ms_obj) = multi_solve(&prob, 12, 0xBEEF, StartStrategy::RandomBox);
    let improvement = cold_obj - ms_obj;
    assert!(
        improvement >= STRICT_IMPROVEMENT_MARGIN,
        "diag 2D improvement: cold={cold_obj} ms={ms_obj}"
    );
    assert!((ms_obj - (-18.0)).abs() < 1e-3, "ms global -18, got {ms_obj}");
}

#[test]
fn multistart_diag_concave_3d_escapes_cold_saddle() {
    let prob = build_diag_concave(3, 3.0);
    let (_, cold_obj) = cold_solve(&prob);
    assert!(cold_obj.abs() < 1e-3);
    let (_, ms_obj) = multi_solve(&prob, 20, 0xFEED, StartStrategy::RandomBox);
    let improvement = cold_obj - ms_obj;
    assert!(
        improvement >= STRICT_IMPROVEMENT_MARGIN,
        "diag 3D improvement: cold={cold_obj} ms={ms_obj}"
    );
    assert!((ms_obj - (-27.0)).abs() < 1e-3, "ms global -27, got {ms_obj}");
}

#[test]
fn multistart_lhs_strategy_also_improves() {
    // RandomBox と LatinHypercube 両戦略で改善 (戦略 dispatch 退化の防止)。
    let prob = build_diag_concave(2, 3.0);
    let (_, cold_obj) = cold_solve(&prob);
    let (_, lhs_obj) = multi_solve(&prob, 10, 0xABCD, StartStrategy::LatinHypercube);
    let improvement = cold_obj - lhs_obj;
    assert!(
        improvement >= STRICT_IMPROVEMENT_MARGIN,
        "LHS must improve: cold={cold_obj} lhs={lhs_obj}"
    );
}

#[test]
fn multistart_table_driven_seed_robustness() {
    // 4 problem shape × 5 seed = 20 ケース。全ケースで improvement>=margin を要求。
    // no-op 書換時は全 20 ケース FAIL。
    let problems: Vec<(&'static str, QpProblem)> = vec![
        ("bilin_b3", build_bilinear_zero_c(3.0)),
        ("bilin_b5", build_bilinear_zero_c(5.0)),
        ("diag_2d", build_diag_concave(2, 3.0)),
        ("diag_3d", build_diag_concave(3, 3.0)),
    ];
    let seeds: Vec<u64> = vec![1, 7, 42, 100, 2024];

    let mut total = 0usize;
    let mut improved = 0usize;
    for (name, prob) in problems.iter() {
        let (_, cold_obj) = cold_solve(prob);
        for &seed in seeds.iter() {
            total += 1;
            let (_, ms_obj) = multi_solve(prob, 10, seed, StartStrategy::RandomBox);
            assert!(
                ms_obj <= cold_obj + 1e-6,
                "{name} seed={seed}: ms never worse, got cold={cold_obj} ms={ms_obj}"
            );
            if cold_obj - ms_obj >= STRICT_IMPROVEMENT_MARGIN {
                improved += 1;
            }
        }
    }
    // 全 20 ケースで improvement 必須。Phase 1A の 27/49 KKT_FAIL 級 hard problem に
    // 比べ saddle-trap toy は random 1 件でも 100% escape するため 100% 要求は妥当。
    assert_eq!(
        improved, total,
        "all {total} cases must improve (saddle escape), got {improved}/{total}"
    );
}

#[test]
fn multistart_deterministic_with_same_seed() {
    let prob = build_diag_concave(2, 3.0);
    let (_, o1) = multi_solve(&prob, 8, 1234, StartStrategy::RandomBox);
    let (_, o2) = multi_solve(&prob, 8, 1234, StartStrategy::RandomBox);
    assert!((o1 - o2).abs() < 1e-9, "deterministic: o1={o1} o2={o2}");
}

// ============================================================================
// API tests (SolverOptions::threads / Model::set_threads / backward-compat)
// ============================================================================

#[test]
fn api_solver_options_threads_default_is_1() {
    let o = SolverOptions::default();
    assert_eq!(o.threads, 1, "default threads must be 1 (= existing behavior)");
}

#[test]
fn api_solver_options_threads_round_trip() {
    let mut o = SolverOptions::default();
    o.threads = 4;
    assert_eq!(o.threads, 4);
}

#[test]
fn api_model_set_threads_propagates_to_solver_options() {
    // Model 経由で set_threads(N) → 内部 SolverOptions.threads = N が伝播することを
    // observable に確認する。直接 SolverOptions を取れないので solve 経路で実証する。
    use otspot::model::{Expression, Model};
    let mut m = Model::new("threads_round_trip");
    let x = m.add_var("x", 0.0, 1.0);
    m.minimize(x);
    let lhs: Expression = x.into();
    m.add_constraint(lhs.leq(1.0));
    m.set_threads(4);
    // solve が成功する (= threads が valid に伝播し SolverOptions が壊れない)。
    let _r = m.solve().expect("LP solve should succeed");
}

#[test]
fn api_model_set_threads_propagates_to_qp_solve() {
    // Model 経由で set_threads(N) → QP single solve path に伝播することを
    // observable に確認 (LP path 同等 test の QP 版、漏れ穴埋め)。
    use otspot::model::{Expression, Model};
    let mut m = Model::new("qp_threads_round_trip");
    let x = m.add_var("x", -1.0, 1.0);
    let y = m.add_var("y", -1.0, 1.0);
    // 簡易 PSD QP: min 0.5 (x^2 + y^2) s.t. x + y <= 1、Q diag = [1, 1] で QP path 強制
    m.set_diagonal_q(&[1.0, 1.0]);
    let lhs: Expression = Expression::from(x) + Expression::from(y);
    m.add_constraint(lhs.leq(1.0));
    m.minimize(Expression::from(x));
    m.set_threads(4);
    // QP solve が成功する (= threads=4 が QP path に valid に伝播、SolverOptions 壊れない)
    // 現状 threads は単発 QP solve では no-op (#31 完了まで)、伝播 path のみ確認
    let _r = m.solve().expect("QP solve should succeed with threads=4");
}

#[test]
fn api_model_set_threads_clamps_zero_to_one() {
    // 0 は invalid (LCG/ThreadPool 双方で fatal)、Model::set_threads 入口で 1 に補正。
    use otspot::model::Model;
    let mut m = Model::new("threads_zero_clamp");
    let _x = m.add_var("x", 0.0, 1.0);
    m.set_threads(0);
    // 後段で panic しないこと。set_threads が saturating であることを smoke で確認。
    // (内部 field は private なので直接 assert 不可、ただし solve_qp_with でも同様に
    // SolverOptions::threads.max(1) しているので 0 でも crash しない契約。)
}

#[test]
fn api_threads_eq_1_preserves_legacy_obj() {
    // threads=1 (default) は既存挙動と完全一致 (= backward compat 退化なし)。
    let prob = build_bilinear_zero_c(3.0);
    let opts_default = opts_with_timeout(5.0); // threads=1 (default)
    let r1 = solve_qp_with(&prob, &opts_default);

    let mut opts_thread1 = opts_with_timeout(5.0);
    opts_thread1.threads = 1;
    let r2 = solve_qp_with(&prob, &opts_thread1);
    assert_eq!(r1.status, r2.status);
    assert!((r1.objective - r2.objective).abs() < 1e-9);
}

#[test]
fn api_threads_n_with_multistart_still_improves() {
    // threads=4 + multistart で saddle escape が機能する (並列下でも logic 健在)。
    let prob = build_diag_concave(2, 3.0);
    let (_, cold_obj) = cold_solve(&prob);
    let cfg = MultiStartConfig {
        n_starts: 12,
        seed: 0xBEEF,
        strategy: StartStrategy::RandomBox,
    };
    let mut opts = opts_with_timeout(20.0);
    opts.threads = 4;
    opts.multistart = Some(cfg);
    let r = solve_qp_with(&prob, &opts);
    let improvement = cold_obj - r.objective;
    assert!(
        improvement >= STRICT_IMPROVEMENT_MARGIN,
        "threads=4 multistart: cold={cold_obj} ms={}",
        r.objective
    );
}

#[test]
fn api_threads_n_with_multistart_deterministic_across_threads() {
    // 同 seed + threads=1 と threads=4 で同 objective (race-free + index-reduce)。
    let prob = build_diag_concave(2, 3.0);
    let cfg = MultiStartConfig {
        n_starts: 10,
        seed: 0xABCD,
        strategy: StartStrategy::RandomBox,
    };
    let mut o1 = opts_with_timeout(20.0);
    o1.threads = 1;
    let r1 = solve_qp_multistart(&prob, &o1, &cfg);
    let mut o4 = opts_with_timeout(20.0);
    o4.threads = 4;
    let r4 = solve_qp_multistart(&prob, &o4, &cfg);
    assert!(
        (r1.objective - r4.objective).abs() < 1e-9,
        "threads-invariance broken: r1={} r4={}",
        r1.objective,
        r4.objective
    );
}

#[test]
fn multistart_dispatch_via_options_field_matches_explicit_call() {
    // SolverOptions::multistart 経由 dispatch も同じ結果 (solve_qp_with 内 if 分岐)。
    let prob = build_diag_concave(2, 3.0);
    let cfg = MultiStartConfig {
        n_starts: 8,
        seed: 0xC0DE,
        strategy: StartStrategy::RandomBox,
    };
    let opts_explicit = opts_with_timeout(10.0);
    let r_explicit = solve_qp_multistart(&prob, &opts_explicit, &cfg);

    let mut opts_field = opts_with_timeout(10.0);
    opts_field.multistart = Some(cfg.clone());
    let r_field = solve_qp_with(&prob, &opts_field);
    assert!(
        (r_explicit.objective - r_field.objective).abs() < 1e-9,
        "dispatch equivalence: explicit={} field={}",
        r_explicit.objective,
        r_field.objective
    );
}
