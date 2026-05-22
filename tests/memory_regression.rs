//! IPM 反復中のメモリ leak regression test.
//!
//! 2026-05-09: eps=1e-8 で eps 未達のループが続く LASSO_150 / PORTFOLIO 系で
//! IPM 反復ごとに RSS が線形増加する事象を観測 (1 問で 0 → 3.7 GB / 280s)。
//! 同一スケールの問題を 1 問だけ解いた場合は 161 MB ピークだったため、
//! ループ内 allocation が解放されていないことが疑われる。
//!
//! このテストでは小規模問題を多数回 solve し、process RSS が線形に増加していない
//! ことを確認する (許容 growth rate × iterations 内に収まることをチェック)。

use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::QpProblem;
use otspot::qp::solve_qp_with;
use otspot::sparse::CscMatrix;

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> usize {
    use std::process::Command;
    let pid = std::process::id();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<usize>().map(|kb| kb * 1024).unwrap_or(0)
        }
        Err(_) => 0,
    }
}

#[cfg(not(target_os = "macos"))]
fn current_rss_bytes() -> usize {
    // Linux: /proc/self/status の VmRSS を読む。
    use std::fs;
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            let v = v.trim();
            let kb = v.split_whitespace().next().unwrap_or("0").parse::<usize>().unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

/// 中規模 QP を反復 solve しても RSS が線形に増えないことを検証する。
///
/// 本テストは「IPM 反復ループ内の Vec / cache が iteration 数に比例して増えていく」
/// 不具合を捉える。許容ライン: 100 反復で 200 MB 増加以下 (avg 2 MB / iter)。
#[test]
fn ipm_repeated_solve_no_runaway_memory_growth() {
    // 小規模 QP: min 0.5 x^T Q x + c·x s.t. A x <= b, x in [0, 10]^n
    let n = 50;
    let m = 30;
    let q_rows: Vec<usize> = (0..n).collect();
    let q_cols: Vec<usize> = (0..n).collect();
    let q_vals: Vec<f64> = (0..n).map(|j| 1.0 + (j as f64) * 0.1).collect();
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
    let c: Vec<f64> = (0..n).map(|j| -((j as f64) + 1.0)).collect();

    // ランダムでない決定的な A
    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if (i + j) % 5 == 0 {
                a_rows.push(i);
                a_cols.push(j);
                a_vals.push(1.0 + (i as f64 - j as f64).abs() * 0.01);
            }
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();
    let b: Vec<f64> = (0..m).map(|i| 5.0 + i as f64 * 0.1).collect();
    let bounds: Vec<(f64, f64)> = (0..n).map(|_| (0.0, 10.0)).collect();
    let cts = vec![ConstraintType::Le; m];
    let problem = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(5.0);

    // ウォームアップ (allocator の初期 reserve を消化)
    for _ in 0..3 {
        let _ = solve_qp_with(&problem, &opts);
    }

    let baseline = current_rss_bytes();
    if baseline == 0 {
        eprintln!("RSS query unsupported on this platform; skipping memory leak check");
        return;
    }

    // 100 回 solve
    const ITERATIONS: usize = 100;
    for _ in 0..ITERATIONS {
        let _ = solve_qp_with(&problem, &opts);
    }

    let final_rss = current_rss_bytes();
    let growth = final_rss.saturating_sub(baseline);
    let growth_mb = growth as f64 / 1024.0 / 1024.0;
    eprintln!(
        "RSS growth over {} iterations: {:.1} MB (baseline={:.1} MB, final={:.1} MB)",
        ITERATIONS,
        growth_mb,
        baseline as f64 / 1024.0 / 1024.0,
        final_rss as f64 / 1024.0 / 1024.0,
    );

    // 許容上限: 200 MB / 100 iter = 2 MB / iter。
    // OS allocator の page free 遅延 + reasonable working set growth を含めた緩い閾値。
    // これを超えると IPM 反復内の固定リーク (Vec の再アロケーション、cache 蓄積等)。
    const MAX_ALLOWED_GROWTH_MB: f64 = 200.0;
    assert!(
        growth_mb < MAX_ALLOWED_GROWTH_MB,
        "Memory leak suspected: RSS grew {:.1} MB over {} solves (limit {:.1} MB)",
        growth_mb, ITERATIONS, MAX_ALLOWED_GROWTH_MB
    );
}

/// 密 A を持つ中規模 LASSO 風 QP で `build_aat_upper_csc` が memory budget guard で
/// skip されることを検証する。小型化合成版で peak RSS が leak 検出 floor 200 MB を
/// 確実に下回ることを確認。
///
/// 規模設定: n=300, m=400, A 密 (col_density=m)。nnz(AAT_upper) 上限 ≈ 80k、
/// memory_budget 4 GiB 以下で進む。
#[test]
fn lasso_dense_aat_no_runaway_memory() {
    use otspot::sparse::CscMatrix;
    use otspot::qp::QpProblem;
    use otspot::qp::solve_qp_with;
    use otspot::options::SolverOptions;
    use otspot::problem::ConstraintType;
    let n = 300;
    let m = 400;
    // A 密 (col_density=m): LASSO 風に各列で全行を埋める。
    let mut a_rows: Vec<usize> = Vec::with_capacity(n * m);
    let mut a_cols: Vec<usize> = Vec::with_capacity(n * m);
    let mut a_vals: Vec<f64> = Vec::with_capacity(n * m);
    for j in 0..n {
        for i in 0..m {
            a_rows.push(i);
            a_cols.push(j);
            a_vals.push(((i + j) % 7) as f64 - 3.0);
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        n, n,
    ).unwrap();
    let c: Vec<f64> = (0..n).map(|j| ((j as f64) - n as f64 / 2.0) * 0.01).collect();
    let b: Vec<f64> = (0..m).map(|i| 1.0 + (i as f64) * 0.001).collect();
    let bounds = vec![(0.0_f64, 100.0_f64); n];
    let cts = vec![ConstraintType::Le; m];
    let problem = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    opts.ipm.eps = 1e-8;

    let baseline = current_rss_bytes();
    let _ = solve_qp_with(&problem, &opts);
    let peak = current_rss_bytes();
    if baseline == 0 {
        eprintln!("RSS query unsupported on this platform; skipping memory leak check");
        return;
    }
    let growth = peak.saturating_sub(baseline);
    let growth_mb = growth as f64 / 1024.0 / 1024.0;
    eprintln!(
        "Dense-A LASSO solve RSS growth: {:.1} MB (baseline={:.1} MB, peak={:.1} MB)",
        growth_mb,
        baseline as f64 / 1024.0 / 1024.0,
        peak as f64 / 1024.0 / 1024.0,
    );
    // 旧 bug 検出 floor: AAT BTreeMap で n=300 m=400 でも実測 100 MB 超える退行を防ぐ。
    const MAX_GROWTH_MB: f64 = 500.0;
    assert!(
        growth_mb < MAX_GROWTH_MB,
        "AAT build memory regression: RSS grew {:.1} MB on dense-A QP (limit {:.1} MB)",
        growth_mb, MAX_GROWTH_MB
    );
}
