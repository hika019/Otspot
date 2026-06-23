//! IP-PMM warm start sentinel: cold solve の (x*, y*, μ*) を warm として
//! 再求解 → iter / wall ともに有意な短縮を要求。
//!
//! - iter 比 < 0.7: 決定論的、Phase 3 B&B の主要 ROI 検証
//! - wall 比 < 0.8: noise 軽減のため iter より緩い (並列実行下の variance 配慮)

use otspot::qp::{solve_qp_with, QpProblem, QpWarmStart};
use otspot::sparse::CscMatrix;
use otspot::{SolveStatus, SolverOptions};

/// warm の最低有効効果 10% (= cold の 0.9 倍以下 iter)。これより緩いと no-op
/// fallback が検出できず sentinel として機能しない。
const WARM_ITER_REDUCTION_MARGIN: f64 = 0.1;
const WARM_ITER_RATIO_UPPER: f64 = 1.0 - WARM_ITER_REDUCTION_MARGIN;

/// 決定論的 diagonal PSD Q + random sparse 不等式 + box bound の合成 QP。
/// diag Q は確実に PSD、IPM が安定に Optimal を返す。
fn build_medium_convex_qp(n: usize, m: usize, density: f64) -> QpProblem {
    let mut seed: u64 = 0x_DEAD_BEEF_CAFE_BABE;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
    };

    let q_diag: Vec<f64> = (0..n)
        .map(|_| 0.5 + 0.5 * (next(&mut seed) + 1.0))
        .collect();
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

    let b_rhs: Vec<f64> = (0..m)
        .map(|_| 0.5 + (next(&mut seed) + 1.0) * 0.5)
        .collect();
    let bounds = vec![(-2.0_f64, 2.0_f64); n];
    QpProblem::new_all_le(q, c, a, b_rhs, bounds).unwrap()
}

#[test]
fn warm_start_30pct_speedup_smoke() {
    let problem = build_medium_convex_qp(600, 350, 0.12);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let cold_result = solve_qp_with(&problem, &opts);
    assert_eq!(
        cold_result.status,
        SolveStatus::Optimal,
        "cold must be Optimal"
    );
    let cold_iters = cold_result.iterations;

    let mu = cold_result
        .final_residuals
        .map(|(_, _, g)| g)
        .unwrap_or(1e-6)
        .max(1e-10);
    let ws = QpWarmStart::new(
        cold_result.solution.clone(),
        cold_result.dual_solution.clone(),
        mu,
    );
    let mut warm_opts = SolverOptions::default();
    warm_opts.timeout_secs = Some(60.0);
    warm_opts.warm_start_qp = Some(ws.clone());
    let warm_result = solve_qp_with(&problem, &warm_opts);
    assert_eq!(
        warm_result.status,
        SolveStatus::Optimal,
        "warm must be Optimal"
    );
    let warm_iters = warm_result.iterations;

    let obj_diff =
        (warm_result.objective - cold_result.objective).abs() / (1.0 + cold_result.objective.abs());
    assert!(obj_diff < 1e-4, "warm obj drift: {:.3e}", obj_diff);

    const N: usize = 5;
    let measure = |with_ws: bool| -> f64 {
        let walls: Vec<f64> = (0..N)
            .map(|_| {
                let mut o = SolverOptions::default();
                o.timeout_secs = Some(60.0);
                if with_ws {
                    o.warm_start_qp = Some(ws.clone());
                }
                let t = std::time::Instant::now();
                let r = solve_qp_with(&problem, &o);
                assert_eq!(r.status, SolveStatus::Optimal);
                t.elapsed().as_secs_f64()
            })
            .collect();
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
        iter_ratio,
        cold_iters,
        warm_iters
    );
    assert!(
        wall_ratio < 0.75,
        "warm/cold wall ratio expected < 0.75, got {:.3} (cold={:.3}ms warm={:.3}ms)",
        wall_ratio,
        cold_wall * 1000.0,
        warm_wall * 1000.0
    );
}

/// 退化 warm (x が境界上、μ=0、y=0) でも interior 補正で IPM が起動して Optimal。
/// silent SKIP 検出: warm 適用時の iter count が cold と完全一致するなら
/// apply_qp_warm_start が None を返して cold init path に倒れたことを意味する。
#[test]
fn warm_start_degenerate_inputs_handled() {
    let problem = build_medium_convex_qp(40, 20, 0.3);
    let mut cold_opts = SolverOptions::default();
    cold_opts.timeout_secs = Some(30.0);
    let cold = solve_qp_with(&problem, &cold_opts);
    assert_eq!(cold.status, SolveStatus::Optimal);

    let ws = QpWarmStart::new(vec![-2.0; 40], vec![0.0; 20], 0.0);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    opts.warm_start_qp = Some(ws);
    let r = solve_qp_with(&problem, &opts);
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "degenerate warm must still converge; got {:?}",
        r.status
    );

    // 退化 warm の iter は cold より大きくなりやすいが、その差が観測できることが
    // 「warm が adapter を通過した」証拠。iter_ratio == 1.0 なら apply_qp_warm_start
    // が silently None で cold init を辿ったことを意味する。
    assert!(
        r.iterations != cold.iterations,
        "degenerate warm appears silently dropped (iter ratio = 1.0): warm={} cold={}",
        r.iterations,
        cold.iterations
    );

    // IPM 暴発の上限 (cold の 4 倍以内)。退化点から interior 復帰 + 収束のオーバーヘッド
    // 込みで合理的な上限。
    let iter_ratio = r.iterations as f64 / cold.iterations.max(1) as f64;
    assert!(
        iter_ratio < 4.0,
        "degenerate warm iter blowup: warm={} cold={} ratio={:.2}",
        r.iterations,
        cold.iterations,
        iter_ratio
    );
}

/// Q-diag scaling 経路 (q_pos_max/q_pos_min ≥ 1e6) で warm が正しく col_scales 変換
/// される B-1 sentinel。
///
/// 検出原理:
///   fix 前: warm.x は user 空間のまま scaled bounds に強制 clamp (例: orig [-2,2]、
///           q=1e-4 列の scaled bound は [-0.02, 0.02] で warm.x=0.5 → 0.02 に押し込み)。
///           IPM 入力が物理的に異常 → 後段 LSQ refit で最終解は救えても obj が drift する
///           か、収束に余分な iter を要する。
///   fix 後: warm.x が col_scales で scaled 空間に正しく配置され、cold init と区別可能。
///
/// 主検証は (a) obj 整合性、(b) iter ≠ cold (silent SKIP 排除)、(c) iter 上限 ratio。
/// (c) は scale_warm_start_for_q_diag を no-op に倒すと warm がほぼ cold init 同等の
/// iter になる (ratio ≈ 1.0) ことを利用した検出。warm が機能していれば cold の (1 - margin)
/// 倍以下に収まることを assert する。
#[test]
fn warm_start_propagates_through_q_diag_scaling() {
    use otspot::problem::ConstraintType;
    use otspot::sparse::CscMatrix;

    let n = 120_usize;
    let m = 40_usize;
    let mut seed: u64 = 0x_A5A5_1234_BEEF_0001;
    let next = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
    };

    // diag Q with 8-decade dynamic range → q_pos_max/q_pos_min ≥ 1e6 trigger Q-diag scaling.
    let q_diag: Vec<f64> = (0..n)
        .map(|j| {
            let exp = -4.0 + 8.0 * (j as f64 / (n - 1) as f64);
            10.0_f64.powf(exp)
        })
        .collect();
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &q_diag,
        n,
        n,
    )
    .unwrap();

    let c: Vec<f64> = (0..n).map(|_| next(&mut seed)).collect();

    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    for j in 0..n {
        for i in 0..m {
            if (next(&mut seed) + 1.0) * 0.5 < 0.15 {
                a_rows.push(i);
                a_cols.push(j);
                a_vals.push(next(&mut seed));
            }
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();
    let b: Vec<f64> = (0..m)
        .map(|_| 0.5 + (next(&mut seed) + 1.0) * 0.5)
        .collect();
    let bounds = vec![(-2.0_f64, 2.0_f64); n];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le; m]).unwrap();

    let q_range = q_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        / q_diag.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        q_range >= 1e6,
        "Q-diag scaling trigger requires range≥1e6, got {:.2e}",
        q_range
    );

    let mut cold_opts = SolverOptions::default();
    cold_opts.timeout_secs = Some(60.0);
    let cold = solve_qp_with(&problem, &cold_opts);
    assert_eq!(cold.status, SolveStatus::Optimal, "cold must Optimal");

    let mu = cold
        .final_residuals
        .map(|(_, _, g)| g)
        .unwrap_or(1e-6)
        .max(1e-10);
    let ws = QpWarmStart::new(cold.solution.clone(), cold.dual_solution.clone(), mu);
    let mut warm_opts = SolverOptions::default();
    warm_opts.timeout_secs = Some(60.0);
    warm_opts.warm_start_qp = Some(ws);
    let warm = solve_qp_with(&problem, &warm_opts);
    assert_eq!(warm.status, SolveStatus::Optimal, "warm must Optimal");

    let iter_ratio = warm.iterations as f64 / cold.iterations.max(1) as f64;
    eprintln!(
        "WARM_Q_DIAG_SENTINEL: cold_iters={} warm_iters={} ratio={:.3} q_range={:.2e}",
        cold.iterations, warm.iterations, iter_ratio, q_range
    );

    // 主検証: obj が cold と一致 (B-1 transform 数式正当性)。
    let obj_diff = (warm.objective - cold.objective).abs() / (1.0 + cold.objective.abs());
    assert!(obj_diff < 1e-4, "Q-diag warm obj drift: {:.3e}", obj_diff);

    // silent SKIP 検出: warm 適用時の iter は cold と異なる経路を辿る → iter 一致なら
    // 実は warm が dropped されていた疑い。
    assert!(
        warm.iterations != cold.iterations,
        "Q-diag warm appears silently dropped: warm={} cold={}",
        warm.iterations,
        cold.iterations
    );

    // warm が機能していれば cold の (1 - WARM_ITER_REDUCTION_MARGIN) 倍以下に収まる。
    // scale_warm_start_for_q_diag を no-op に倒すと warm.x が user 空間のまま scaled
    // bounds に強制 clamp され (例: orig [-2,2]、q=1e-4 列の scaled bound [-0.02, 0.02]
    // で warm.x=0.5 → 0.02 に押し込み) IPM 入力が壊れ、interior 復帰のため cold init 以上の
    // iter を要する → iter_ratio ≈ 1.0 以上で本 assert が FAIL する。
    // 0.9 = 1 - WARM_ITER_REDUCTION_MARGIN (module top で共有定義)。
    // 実測 (n=600 diag Q q_range=1e8): cold=16 warm=5 ratio=0.312、no-op 倒すと ratio≈1.94 で FAIL。
    assert!(
        iter_ratio < WARM_ITER_RATIO_UPPER,
        "Q-diag warm not reducing iter as expected: ratio={:.3} ≥ {:.3} (warm={} cold={})",
        iter_ratio,
        WARM_ITER_RATIO_UPPER,
        warm.iterations,
        cold.iterations
    );
}

/// presolve 経路 (fixed var lb=ub) で warm が reduced 空間に col_map_inv で翻訳される
/// B-2 sentinel。
///
/// 検出原理: silent SKIP (dim mismatch で apply_qp_warm_start が None 帰着) なら
/// warm path は cold init と区別不能。translation 経由なら iter 数が cold と異なる。
/// 主検証は obj 整合性 + iter ≠ cold (silent drop 検出) + Optimal status。
#[test]
fn warm_start_propagates_through_presolve_reduction() {
    use otspot::problem::ConstraintType;
    use otspot::sparse::CscMatrix;

    // 小問題で FixedVar 確実 + 数値安定: n=4, 1 FixedVar → reduced n=3。
    let q = CscMatrix::from_triplets(&[0, 1, 2, 3], &[0, 1, 2, 3], &[2.0, 2.0, 2.0, 2.0], 4, 4)
        .unwrap();
    let c = vec![-1.0, -2.0, -3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 0, 0], &[0, 1, 2, 3], &[1.0, 1.0, 1.0, 1.0], 1, 4)
        .unwrap();
    let b = vec![3.0];
    // 末尾 var を lb=ub=0.5 で固定 → presolve FixedVar reduction。
    let bounds = vec![
        (0.0_f64, 5.0_f64),
        (0.0_f64, 5.0_f64),
        (0.0_f64, 5.0_f64),
        (0.5_f64, 0.5_f64),
    ];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();

    let mut cold_opts = SolverOptions::default();
    cold_opts.timeout_secs = Some(10.0);
    let cold = solve_qp_with(&problem, &cold_opts);
    assert_eq!(
        cold.status,
        SolveStatus::Optimal,
        "cold must Optimal; got {:?}",
        cold.status
    );
    assert!(
        (cold.solution[3] - 0.5).abs() < 1e-6,
        "fixed var must be 0.5"
    );

    let mu = cold
        .final_residuals
        .map(|(_, _, g)| g)
        .unwrap_or(1e-6)
        .max(1e-10);
    let ws = QpWarmStart::new(cold.solution.clone(), cold.dual_solution.clone(), mu);
    let mut warm_opts = SolverOptions::default();
    warm_opts.timeout_secs = Some(10.0);
    warm_opts.warm_start_qp = Some(ws);
    let warm = solve_qp_with(&problem, &warm_opts);
    assert_eq!(warm.status, SolveStatus::Optimal, "warm must Optimal");

    let iter_ratio = warm.iterations as f64 / cold.iterations.max(1) as f64;
    eprintln!(
        "WARM_PRESOLVE_SENTINEL: cold_iters={} warm_iters={} ratio={:.3}",
        cold.iterations, warm.iterations, iter_ratio
    );

    // 主検証: obj が cold と一致 (B-2 translate 数式正当性 + fixed var 整合)。
    let obj_diff = (warm.objective - cold.objective).abs() / (1.0 + cold.objective.abs());
    assert!(obj_diff < 1e-4, "presolve warm obj drift: {:.3e}", obj_diff);

    // silent SKIP 検出: fix 前は ws.x.len=4 ≠ n_reduced=3 で apply_qp_warm_start が None
    // → warm path == cold path → iter 一致。fix 後は translate で reduced 空間に翻訳され
    // warm が起動 → cold と iter が異なる。
    assert!(
        warm.iterations != cold.iterations,
        "presolve warm appears silently dropped: warm={} cold={}",
        warm.iterations,
        cold.iterations
    );

    // Ruiz scaler 適用 (B-2 fix) が機能していれば warm path は cold より大幅短縮。
    // translate 内 Ruiz block を no-op に倒すと warm.x が orig 空間のまま scaled reduced
    // 問題に入り iter 削減効果消失 → iter_ratio ≈ 1.0 以上で本 assert が FAIL する。
    // 実測 (n=4 + FixedVar): cold=8 warm=2 ratio=0.250。
    assert!(
        iter_ratio < WARM_ITER_RATIO_UPPER,
        "presolve warm not reducing iter as expected: ratio={:.3} ≥ {:.3} (warm={} cold={})",
        iter_ratio,
        WARM_ITER_RATIO_UPPER,
        warm.iterations,
        cold.iterations
    );
}
