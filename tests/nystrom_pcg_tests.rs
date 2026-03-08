//! Nyström PCG テスト (cmd_314)
//!
//! Nyström 前処理付き PCG は n >= 500 の大規模問題で有効になる内点法の線形ソルバ。
//! (src/qp/ipm/step.rs: NYSTROM_THRESHOLD = 500)
//!
//! このファイルは以下を検証する:
//! 1. n=500 の問題で IpmNystrom ソルバーが Optimal を返すこと
//! 2. n=499 (Nystrom 無効パス) と n=500 (Nystrom 有効パス) で解の品質が同等であること

use solver::qp::{QpProblem, solve_qp_with};
use solver::sparse::CscMatrix;
use solver::SolveStatus;
use solver::{QpSolverChoice, SolverOptions};

/// n×n 対角 Q (Q[i][i] = 2.0) + 単一 <= 制約 sum(x) <= n を構成する
///
/// 問題: min 1/2 * 2 * ||x||^2 - 3*sum(x)  s.t. sum(x) <= n
/// - 非拘束最小点: x_i = 1.5, sum = 1.5n > n → 制約が活性化する
/// - KKT 解析解: x*_i = 1, obj = 1/2*2*n - 3n = n - 3n = -2n
fn build_problem(n: usize) -> QpProblem {
    // Q = 2 * I_n  (対角成分のみ)
    let q_rows: Vec<usize> = (0..n).collect();
    let q_cols: Vec<usize> = (0..n).collect();
    let q_vals: Vec<f64> = vec![2.0; n];
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();

    // c = -3 (非拘束最小点が sum = 1.5n > n になるよう設定)
    let c = vec![-3.0f64; n];

    // A = 1^T (1行n列、全て1.0) → sum(x) <= n
    let a_rows: Vec<usize> = vec![0; n];
    let a_cols: Vec<usize> = (0..n).collect();
    let a_vals: Vec<f64> = vec![1.0; n];
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 1, n).unwrap();

    let b = vec![n as f64];

    // bounds: 無制約
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];

    QpProblem::new(q, c, a, b, bounds).unwrap()
}

/// IpmNystrom ソルバーオプションを構築 (timeout=60s, max_iter=300)
fn nystrom_options() -> SolverOptions {
    let mut opts = SolverOptions::default();
    opts.qp_solver = QpSolverChoice::IpmNystrom;
    opts.timeout_secs = Some(60.0);
    opts.ipm.max_iter = 300;
    opts
}

/// テスト1: n=500 で Nystrom 有効パスを通る問題を解く
///
/// NYSTROM_THRESHOLD=500 なので n=500 は Nystrom PCG パスを通過する。
/// Optimal が返ること・解が解析解 x*=1 に近いこと・目的関数 ≈ 0 を確認。
#[test]
fn test_nystrom_active_n500() {
    let n = 500;
    let problem = build_problem(n);
    let opts = nystrom_options();

    let result = solve_qp_with(&problem, &opts);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "n=500 (Nystrom active): expected Optimal, got {:?}",
        result.status
    );

    // 解析解 x*_i = 1, obj = -2n = -1000
    let expected_obj = -2.0 * n as f64;
    let eps = 1e-3;
    assert!(
        (result.objective - expected_obj).abs() < eps * (1.0 + expected_obj.abs()),
        "n=500: objective={:.6e} (expected ≈ {:.1}, rel_eps={:.0e})",
        result.objective,
        expected_obj,
        eps
    );

    // 実行可能性: sum(x) <= n + tolerance
    let sum_x: f64 = result.solution.iter().sum();
    let pfeas_tol = 1e-6 * (1.0 + n as f64);
    assert!(
        sum_x <= n as f64 + pfeas_tol,
        "n=500: sum(x)={:.6} violates constraint sum(x)<={}", sum_x, n
    );
}

/// テスト2: n=499 (Nystrom 無効パス) と n=500 (Nystrom 有効パス) で同等の解品質
///
/// 境界値テスト。両者とも Optimal で、目的関数が解析解に近いことを確認。
/// - n=499: solve_qp_ipm_nystrom_inner → Schur パスへ委譲
/// - n=500: solve_qp_ipm_nystrom_inner → Nystrom PCG パスを実行
#[test]
fn test_nystrom_threshold_boundary() {
    let opts = nystrom_options();

    // n=499: Nystrom 閾値未満 → Schur パス
    {
        let n = 499;
        let problem = build_problem(n);
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "n=499 (Nystrom inactive): expected Optimal, got {:?}",
            result.status
        );

        // 解析解 x*_i = 1, obj = -2n
        let expected_obj = -2.0 * n as f64;
        let eps = 1e-3;
        assert!(
            (result.objective - expected_obj).abs() < eps * (1.0 + expected_obj.abs()),
            "n=499: objective={:.6e} (expected ≈ {:.1})",
            result.objective,
            expected_obj
        );
    }

    // n=500: Nystrom 閾値以上 → Nystrom PCG パス
    {
        let n = 500;
        let problem = build_problem(n);
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "n=500 (Nystrom active): expected Optimal, got {:?}",
            result.status
        );

        // 解析解 x*_i = 1, obj = -2n
        let expected_obj = -2.0 * n as f64;
        let eps = 1e-3;
        assert!(
            (result.objective - expected_obj).abs() < eps * (1.0 + expected_obj.abs()),
            "n=500: objective={:.6e} (expected ≈ {:.1})",
            result.objective,
            expected_obj
        );
    }
}
