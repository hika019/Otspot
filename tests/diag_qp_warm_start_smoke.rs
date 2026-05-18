//! IP-PMM warm start (#12) sentinel: cold solve の (x*, y*, μ*) を warm として
//! 再求解 → iter / wall ともに有意な短縮を要求。
//!
//! - iter 比 < 0.7: 決定論的、Phase 3 B&B の主要 ROI 検証
//! - wall 比 < 0.8: noise 軽減のため iter より緩い (並列実行下の variance 配慮)

use solver::qp::{solve_qp_with, QpProblem, QpWarmStart};
use solver::sparse::CscMatrix;
use solver::{SolveStatus, SolverOptions};

/// 決定論的 diagonal PSD Q + random sparse 不等式 + box bound の合成 QP。
/// diag Q は確実に PSD、IPM が安定に Optimal を返す。
fn build_medium_convex_qp(n: usize, m: usize, density: f64) -> QpProblem {
    let mut seed: u64 = 0x_DEAD_BEEF_CAFE_BABE;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
    };

    let q_diag: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * (next(&mut seed) + 1.0)).collect();
    let q_rows: Vec<usize> = (0..n).collect();
    let q_cols: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_diag, n, n).unwrap();

    let c: Vec<f64> = (0..n).map(|_| next(&mut seed)).collect();

    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        for i in 0..m {
            if (next(&mut seed) + 1.0) * 0.5 < density {
                a_rows.push(i);
                a_cols.push(j);
                a_vals.push(next(&mut seed));
            }
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();

    let b_rhs: Vec<f64> = (0..m).map(|_| 0.5 + (next(&mut seed) + 1.0) * 0.5).collect();
    let bounds = vec![(-2.0_f64, 2.0_f64); n];
    QpProblem::new_all_le(q, c, a, b_rhs, bounds).unwrap()
}

#[test]
fn warm_start_30pct_speedup_smoke() {
    let problem = build_medium_convex_qp(600, 350, 0.12);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let cold_result = solve_qp_with(&problem, &opts);
    assert_eq!(cold_result.status, SolveStatus::Optimal, "cold must be Optimal");
    let cold_iters = cold_result.iterations;

    let ws = QpWarmStart {
        x: cold_result.solution.clone(),
        y: cold_result.dual_solution.clone(),
        mu: cold_result.gap.unwrap_or(1e-6).max(1e-10),
    };
    let mut warm_opts = SolverOptions::default();
    warm_opts.timeout_secs = Some(60.0);
    warm_opts.warm_start_qp = Some(ws.clone());
    let warm_result = solve_qp_with(&problem, &warm_opts);
    assert_eq!(warm_result.status, SolveStatus::Optimal, "warm must be Optimal");
    let warm_iters = warm_result.iterations;

    let obj_diff = (warm_result.objective - cold_result.objective).abs()
        / (1.0 + cold_result.objective.abs());
    assert!(obj_diff < 1e-4, "warm obj drift: {:.3e}", obj_diff);

    const N: usize = 5;
    let measure = |with_ws: bool| -> f64 {
        let walls: Vec<f64> = (0..N).map(|_| {
            let mut o = SolverOptions::default();
            o.timeout_secs = Some(60.0);
            if with_ws { o.warm_start_qp = Some(ws.clone()); }
            let t = std::time::Instant::now();
            let r = solve_qp_with(&problem, &o);
            assert_eq!(r.status, SolveStatus::Optimal);
            t.elapsed().as_secs_f64()
        }).collect();
        walls.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let cold_wall = measure(false);
    let warm_wall = measure(true);
    let wall_ratio = warm_wall / cold_wall;
    let iter_ratio = warm_iters as f64 / cold_iters as f64;
    eprintln!(
        "WARM_START_SMOKE: cold_iters={} warm_iters={} iter_ratio={:.3} | cold={:.3}ms warm={:.3}ms wall_ratio={:.3}",
        cold_iters, warm_iters, iter_ratio,
        cold_wall * 1000.0, warm_wall * 1000.0, wall_ratio
    );
    assert!(
        iter_ratio < 0.7,
        "warm/cold iter ratio expected < 0.7, got {:.3} (cold={} warm={})",
        iter_ratio, cold_iters, warm_iters
    );
    assert!(
        wall_ratio < 0.75,
        "warm/cold wall ratio expected < 0.75, got {:.3} (cold={:.3}ms warm={:.3}ms)",
        wall_ratio, cold_wall * 1000.0, warm_wall * 1000.0
    );
}

/// 退化 warm (x が境界上、μ=0、y=0) でも interior 補正で IPM が起動して Optimal。
#[test]
fn warm_start_degenerate_inputs_handled() {
    let problem = build_medium_convex_qp(40, 20, 0.3);
    let ws = QpWarmStart {
        x: vec![-2.0; 40],
        y: vec![0.0; 20],
        mu: 0.0,
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    opts.warm_start_qp = Some(ws);
    let r = solve_qp_with(&problem, &opts);
    assert_eq!(r.status, SolveStatus::Optimal,
        "degenerate warm must still converge; got {:?}", r.status);
}
