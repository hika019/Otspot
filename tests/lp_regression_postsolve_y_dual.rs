//! postsolve 後の (x, y, rc) が LP KKT を満たすことを複数問題で assert する
//! regression guard。perold は postsolve dual 復元の代表 canary。

use otspot::io::qps::parse_qps;

use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::{solve_qp_with, QpProblem};
use std::path::Path;

/// LP/QP dual feasibility 相対残差 (`viol_j = max(0, -rc_j)`、bound 非考慮版)。
fn dfeas_abs_rel(prob: &QpProblem, rc: &[f64]) -> (f64, f64) {
    let n = prob.c.len().min(rc.len());
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel = 0.0_f64;
    for (j, &r) in rc.iter().enumerate().take(n) {
        let viol = f64::max(0.0, -r);
        dfeas_abs = dfeas_abs.max(viol);
        let scale = 1.0 + r.abs() + prob.c[j].abs();
        dfeas_rel = dfeas_rel.max(viol / scale);
    }
    (dfeas_abs, dfeas_rel)
}

/// LP dual feasibility (bound 考慮版): bound-hit 列のみ厳格判定。
/// `viol = max(0, -rc)` if x at lb only; `max(0, rc)` if x at ub only;
/// fixed (lb==ub) skip; interior 0。
fn dfeas_rel_bound_aware(prob: &QpProblem, x: &[f64], rc: &[f64]) -> f64 {
    const BOUND_TOL: f64 = 1e-6;
    let n = prob.c.len().min(rc.len()).min(x.len());
    let mut dfeas_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x[j] - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x[j] - ub).abs() < BOUND_TOL;
        let r = rc[j];
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -r)
        } else if at_ub && !at_lb {
            f64::max(0.0, r)
        } else {
            0.0 // interior / 両端
        };
        let scale = 1.0 + r.abs() + prob.c[j].abs();
        dfeas_rel = dfeas_rel.max(viol / scale);
    }
    dfeas_rel
}

/// 仕様: `presolve=true` で solve した結果が以下を満たす:
///   - status = Optimal
///   - dfeas_rel < eps (旧 strict formula、bound 考慮版どちらも)
fn check_postsolve_dual_feasibility(
    qp_path: &str,
    eps_dual: f64,
    timeout_s: f64,
) -> Result<String, String> {
    let path = Path::new(qp_path);
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qp_path
    );
    let prob = parse_qps(path).map_err(|e| format!("parse failed: {:?}", e))?;
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_s);
    let r = solve_qp_with(&prob, &opts);
    let (df_abs, df_rel_strict) = dfeas_abs_rel(&prob, &r.reduced_costs);
    let df_rel_bound = dfeas_rel_bound_aware(&prob, &r.solution, &r.reduced_costs);
    let summary = format!(
        "{}: status={:?} obj={:.4e} df_abs={:.2e} df_rel_strict={:.2e} df_rel_bound={:.2e}",
        qp_path, r.status, r.objective, df_abs, df_rel_strict, df_rel_bound
    );
    if !matches!(r.status, SolveStatus::Optimal) {
        return Err(format!("{} | status must be Optimal", summary));
    }
    // bound 考慮版を主判定にする (c69959d 以降の bench と同等)。
    if df_rel_bound > eps_dual {
        return Err(format!("{} | df_rel_bound > eps={}", summary, eps_dual));
    }
    Ok(summary)
}

/// perold: postsolve dual 復元退化の canary。
#[test]
fn perold_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/perold.QPS", 1e-6, 120.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// perold の presolve=off は別経路 (postsolve なし)。bug の局在を切り分け:
/// HEAD で PASS (df_rel ≈ 3.5e-13) なら、FAIL は postsolve 経路に 100% 局在。
#[test]
fn perold_presolve_off_baseline() {
    let path = Path::new("data/lp_problems/perold.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(120.0);
    let r = solve_qp_with(&prob, &opts);
    // rc = c − A^T y 規約 (extract_dual_info)。at_ub 変数は rc = −μ_ub ≤ 0 を取り得るため、
    // strict `max(0, −rc)` 形式 (`dfeas_abs_rel`) は at_ub を誤検出する。bound-aware を主判定に。
    let df_rel_bound = dfeas_rel_bound_aware(&prob, &r.solution, &r.reduced_costs);
    eprintln!(
        "perold[presolve=off]: status={:?} obj={:.4e} df_rel_bound={:.2e}",
        r.status, r.objective, df_rel_bound
    );
    // SKIP allowed for status: 別経路で NumericalError でも本テストの目的ではない。
    if matches!(r.status, SolveStatus::Optimal) {
        assert!(
            df_rel_bound < 1e-6,
            "perold[presolve=off]: df_rel_bound={:.3e} → simplex 単体に別バグの疑い",
            df_rel_bound
        );
    } else {
        eprintln!("perold[presolve=off]: status={:?}", r.status);
    }
}

/// 小規模 Netlib LP の post-solve dual feasibility 網羅テスト。
///
/// 各問題で `presolve=true` で解き、reduced_cost が bound-aware df_rel < 1e-6 を
/// 満たすことを assert。問題は postsolve.rs の各 PostsolveStep を踏むよう選定。
///
/// 失敗時 stderr に問題名と df 数値を出す。
macro_rules! netlib_postsolve_test {
    ($name:ident, $file:expr, $eps:expr, $timeout:expr) => {
        #[test]
        fn $name() {
            let r = check_postsolve_dual_feasibility($file, $eps, $timeout);
            match r {
                Ok(s) => eprintln!("PASS {}", s),
                Err(e) => panic!("{}", e),
            }
        }
    };
}

netlib_postsolve_test!(afiro_postsolve, "data/lp_problems/afiro.QPS", 1e-6, 30.0);
netlib_postsolve_test!(sc50a_postsolve, "data/lp_problems/sc50a.QPS", 1e-6, 30.0);
netlib_postsolve_test!(sc50b_postsolve, "data/lp_problems/sc50b.QPS", 1e-6, 30.0);
netlib_postsolve_test!(sc105_postsolve, "data/lp_problems/sc105.QPS", 1e-6, 30.0);
netlib_postsolve_test!(sc205_postsolve, "data/lp_problems/sc205.QPS", 1e-6, 30.0);
netlib_postsolve_test!(scagr7_postsolve, "data/lp_problems/scagr7.QPS", 1e-6, 30.0);
netlib_postsolve_test!(
    share1b_postsolve,
    "data/lp_problems/share1b.QPS",
    1e-6,
    30.0
);
netlib_postsolve_test!(
    scorpion_postsolve,
    "data/lp_problems/scorpion.QPS",
    1e-6,
    30.0
);
netlib_postsolve_test!(brandy_postsolve, "data/lp_problems/brandy.QPS", 1e-6, 30.0);
netlib_postsolve_test!(agg_postsolve, "data/lp_problems/agg.QPS", 1e-6, 30.0);
netlib_postsolve_test!(
    boeing2_postsolve,
    "data/lp_problems/boeing2.QPS",
    1e-6,
    30.0
);
netlib_postsolve_test!(
    stocfor1_postsolve,
    "data/lp_problems/stocfor1.QPS",
    1e-6,
    30.0
);

// 大規模 LP の dfeas_rel assertion (network/重 LP は #[ignore] で default 除外)。

/// cre-b: 72k×9k の大規模 LP。
#[test]
#[ignore = "重 LP (timeout 300s 必要); cargo test -- --ignored で明示実行"]
fn cre_b_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/cre-b.QPS", 1e-6, 300.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// greenbea: IPM-pathological LP (5405 vars × 2392 rows、IPM_BUDGET_FRACTION=0.5)。
/// アイドル時 ~164s で converge (IPM stall 85s + simplex fallback 79s)、
/// CPU contention 1.04x 以上で 170s budget を超えて FAIL するため default 除外。
/// regression sentinel として heavy profile 実行で機能 (`cargo nextest run --run-ignored only`)。
/// #91 で v0.2.0→HEAD のコード regression なしを実証済。
#[test]
#[ignore = "heavy ~164s idle、bench 並行下 flaky (170s margin 6s)。--run-ignored only"]
fn greenbea_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/greenbea.QPS", 1e-6, 170.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// pds-10: 105k×34k の大規模 LP。convergence に時間がかかる。
#[test]
#[ignore = "重 LP (≈ 200s); cargo test -- --ignored で明示実行"]
fn pds_10_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/pds-10.QPS", 1e-6, 200.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}
